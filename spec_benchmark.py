#!/usr/bin/env python3
"""
Unified spec-writing benchmark generator.

End-to-end pipeline:
  1. Scan source files → find open spec fns with non-trivial bodies
  2. For each candidate: erase spec body → run Verus
       - If Verus FAILS after erasure → good task (proofs depend on this spec)
       - If Verus still passes → skip (spec is unused / redundant)
  3. Emit task file with %%SPEC_START / %%SPEC_END markers
  4. Optionally sample a diverse subset (difficulty × project × return type)
  5. Write benchmark directory: tasks/, ground-truth/, tasks.jsonl

Marker format (inside task files):
    pub open spec fn foo(x: int) -> bool {
        // %%SPEC_START
        arbitrary()
        // %%SPEC_END
    }

Ground-truth files keep the same markers around the real body:
    pub open spec fn foo(x: int) -> bool {
        // %%SPEC_START
        x > 0 && x < 100
        // %%SPEC_END
    }

This makes patching trivial: replace everything between the markers.

Usage:
    # Generate full candidate pool (with verification filter)
    python spec_benchmark.py generate \\
        --source-dir benchmarks/VeruSAGE-Bench/source-projects \\
        --output-dir benchmarks/VeruSAGE-Bench-Spec-v2

    # Sample 100 diverse tasks from the pool
    python spec_benchmark.py sample \\
        --pool-dir benchmarks/VeruSAGE-Bench-Spec-v2 \\
        --output-dir benchmarks/VeruSAGE-Bench-Spec-100-v2 \\
        --target 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERUS_DEFAULT = os.environ.get(
    "VERUS_BIN",
    str(Path(__file__).resolve().parent.parent.parent
        / "verus-binary" / "verus" / "verus"),
)

SPEC_START_MARKER = "// %%SPEC_START"
SPEC_END_MARKER = "// %%SPEC_END"

# Regex matching the beginning of a spec fn declaration.
SPEC_FN_RE = re.compile(
    r'^(\s*)'                          # leading whitespace (group 1)
    r'(pub\s+)?'                       # optional pub (group 2)
    r'(open\s+|closed\s+)?'           # optional open/closed (group 3)
    r'spec\s*(?:\(checked\)\s*)?'      # spec or spec(checked)
    r'fn\s+'                           # fn keyword
    r'([A-Za-z_]\w*)',                 # function name (group 4)
    re.MULTILINE,
)

# Files with known issues that prevent verification regardless of spec changes
KNOWN_BAD_PATTERNS = [
    # nrkernel: repr(transparent) + Ghost<nat> denied by newer Rust
    (lambda src: 'repr(transparent)' in src and 'Ghost<nat>' in src,
     "repr(transparent) + Ghost<nat>"),
    # storage: unresolved deps_hack crate
    (lambda src: 'use deps_hack::' in src or 'use deps_hack :' in src,
     "deps_hack crate"),
]

PROJECT_PREFIX = {
    'anvil-controller': 'AC',
    'anvil-library': 'AL',
    'atmosphere': 'OS',
    'ironkv': 'IR',
    'memory-allocator': 'MA',
    'node-replication': 'NO',
    'nrkernel': 'NR',
    'storage': 'ST',
    'vest': 'VE',
}

DIFFICULTY_TIERS = {
    "easy": (1, 2),
    "medium-easy": (3, 5),
    "medium": (6, 10),
    "hard": (11, 20),
    "very-hard": (21, 999),
}

TIER_TARGETS = {
    "easy": 15,
    "medium-easy": 20,
    "medium": 25,
    "hard": 25,
    "very-hard": 15,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SpecFnInfo:
    """Parsed info about a spec function."""
    name: str
    fn_line: int           # 0-indexed line of fn keyword
    brace_line: int        # 0-indexed line of opening `{`
    end_line: int          # 0-indexed line of closing `}`
    body: str              # text between { and } (excluding braces)
    body_line_count: int
    signature: str         # full text from fn keyword through opening `{`
    indent: str            # leading whitespace on fn line
    is_open: bool
    return_type: str


@dataclass
class TaskRecord:
    """Metadata for one benchmark task."""
    task_id: str
    source_project: str
    source_file: str       # relative to source-projects/
    target_fn: str
    target_return_type: str
    body_lines: int
    file_lines: int
    is_open: bool
    difficulty: str
    ground_truth_body: str  # the original spec body (for reference)


# ---------------------------------------------------------------------------
# Brace matching
# ---------------------------------------------------------------------------

def find_matching_brace(lines: list[str], start_line: int) -> int | None:
    """Find the line of the closing `}` matching the first `{` on start_line.

    Returns 0-indexed line number, or None.
    """
    depth = 0
    in_block_comment = False
    in_string = False
    escape_next = False

    for li in range(start_line, len(lines)):
        line = lines[li]
        ci_start = 0
        if li == start_line:
            idx = line.find("{")
            if idx == -1:
                continue
            ci_start = idx
        ci = ci_start
        while ci < len(line):
            ch = line[ci]
            if escape_next:
                escape_next = False
                ci += 1
                continue
            if in_block_comment:
                if ch == "*" and ci + 1 < len(line) and line[ci + 1] == "/":
                    in_block_comment = False
                    ci += 2
                    continue
                ci += 1
                continue
            if in_string:
                if ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
                ci += 1
                continue
            if ch == "/" and ci + 1 < len(line):
                if line[ci + 1] == "/":
                    break  # line comment → skip rest
                if line[ci + 1] == "*":
                    in_block_comment = True
                    ci += 2
                    continue
            if ch == '"':
                in_string = True
                ci += 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return li
            ci += 1
    return None


# ---------------------------------------------------------------------------
# Spec function parser
# ---------------------------------------------------------------------------

def parse_return_type(sig: str) -> str:
    """Extract return type from a spec fn signature string."""
    # Find the last `->` not inside brackets
    dp, da = 0, 0
    last_arrow = -1
    for i in range(len(sig) - 1):
        ch = sig[i]
        if ch == '(':
            dp += 1
        elif ch == ')':
            dp -= 1
        elif ch == '<':
            da += 1
        elif ch == '>':
            da -= 1
        elif ch == '-' and sig[i + 1] == '>' and dp == 0 and da == 0:
            last_arrow = i
    if last_arrow == -1:
        return "()"

    ret = sig[last_arrow + 2:].strip()

    # Track brackets to find where type ends
    dp, da = 0, 0
    for i, ch in enumerate(ret):
        if ch == '(':
            dp += 1
        elif ch == ')':
            dp -= 1
        elif ch == '<':
            da += 1
        elif ch == '>':
            da -= 1
        elif dp == 0 and da == 0 and ch == '\n':
            ret = ret[:i]
            break

    # Strip trailing Verus clauses
    for kw in ('requires', 'ensures', 'recommends', 'decreases', 'when', '{'):
        idx = ret.find(kw)
        if idx > 0:
            ret = ret[:idx]

    return ret.strip().rstrip(',').rstrip(';').strip() or "()"


def parse_spec_functions(source: str) -> list[SpecFnInfo]:
    """Find all spec functions in a Verus source file."""
    lines = source.split('\n')
    results = []

    for m in SPEC_FN_RE.finditer(source):
        fn_name = m.group(4)
        is_open = m.group(3) is not None and 'open' in m.group(3)
        fn_line = source[:m.start()].count('\n')
        indent = m.group(1)

        # Find opening `{`
        brace_line = None
        for li in range(fn_line, min(fn_line + 50, len(lines))):
            col = lines[li].find('{')
            if col != -1:
                brace_line = li
                break
        if brace_line is None:
            continue

        # Find closing `}`
        end_line = find_matching_brace(lines, brace_line)
        if end_line is None:
            continue

        # Extract signature (fn line → opening brace, inclusive)
        sig = '\n'.join(lines[fn_line:brace_line + 1])

        # Extract body (lines between { and })
        body_lines_list = lines[brace_line + 1:end_line]
        body = '\n'.join(body_lines_list)

        # Parse return type from multi-line signature
        sig_flat = ' '.join(l.strip() for l in lines[fn_line:brace_line + 1])
        ret_type = parse_return_type(sig_flat)

        body_lc = len([l for l in body_lines_list if l.strip()])

        results.append(SpecFnInfo(
            name=fn_name,
            fn_line=fn_line,
            brace_line=brace_line,
            end_line=end_line,
            body=body,
            body_line_count=body_lc,
            signature=sig,
            indent=indent,
            is_open=is_open,
            return_type=ret_type,
        ))

    return results


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def body_is_trivial(body: str) -> bool:
    """Check if spec body is too trivial to be a good benchmark task."""
    s = body.strip()
    if not s or len(s) < 5:
        return True
    if s in ('true', 'false', 'arbitrary()'):
        return True
    if re.match(r'^self\.\w+$', s):
        return True
    # Single unimplemented!() or todo!()
    if s in ('unimplemented!()', 'todo!()'):
        return True
    return False


def has_uninterp_or_external(source: str, fn_line: int) -> bool:
    """Check if the spec fn has #[verifier(external_body)] or is uninterp."""
    lines = source.split('\n')
    # Check preceding attributes and the fn line itself
    for al in range(max(0, fn_line - 5), fn_line + 1):
        line = lines[al]
        if 'external_body' in line or 'uninterp' in line:
            return True
    return False


def source_has_known_issues(source: str) -> str | None:
    """Return a reason if this source file has known unfixable issues."""
    for check_fn, reason in KNOWN_BAD_PATTERNS:
        if check_fn(source):
            return reason
    return None


def body_hash(body: str) -> str:
    """Normalized hash for deduplication."""
    normalized = re.sub(r'\s+', ' ', body.strip())
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Marker insertion & erasure
# ---------------------------------------------------------------------------

def insert_markers_and_erase(source: str, spec_fn: SpecFnInfo) -> str:
    """Replace the spec body with markers + arbitrary().

    Returns modified source with:
        fn foo(...) -> T {
            // %%SPEC_START
            arbitrary()
            // %%SPEC_END
        }
    """
    lines = source.split('\n')
    brace_line = spec_fn.brace_line
    end_line = spec_fn.end_line

    # Compute body indent (one level deeper than fn indent)
    body_indent = spec_fn.indent + "    "

    # Build replacement
    replacement = [
        f"{body_indent}{SPEC_START_MARKER}",
        f"{body_indent}arbitrary()",
        f"{body_indent}{SPEC_END_MARKER}",
    ]

    new_lines = (
        lines[:brace_line + 1]       # up to and including `{` line
        + replacement                  # marker + arbitrary()
        + lines[end_line:]            # from closing `}` onward
    )
    return '\n'.join(new_lines)


def insert_markers_keep_body(source: str, spec_fn: SpecFnInfo) -> str:
    """Insert markers around the spec body, keeping the original body.

    Returns modified source with:
        fn foo(...) -> T {
            // %%SPEC_START
            <original body>
            // %%SPEC_END
        }
    """
    lines = source.split('\n')
    brace_line = spec_fn.brace_line
    end_line = spec_fn.end_line
    body_indent = spec_fn.indent + "    "

    body_lines = lines[brace_line + 1:end_line]

    new_lines = (
        lines[:brace_line + 1]
        + [f"{body_indent}{SPEC_START_MARKER}"]
        + body_lines
        + [f"{body_indent}{SPEC_END_MARKER}"]
        + lines[end_line:]
    )
    return '\n'.join(new_lines)


# ---------------------------------------------------------------------------
# Verus runner
# ---------------------------------------------------------------------------

def run_verus(code: str, verus_bin: str, timeout: int = 120) -> tuple[bool, str]:
    """Write code to a temp file, run Verus.

    Returns (passed, output_snippet).
    """
    with tempfile.NamedTemporaryFile(suffix=".rs", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [verus_bin, tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout + result.stderr
        passed = "0 errors" in output
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        snippet = "\n".join(lines[-3:]) if lines else "(no output)"
        return passed, snippet
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except OSError as e:
        return False, f"os_error: {e}"
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Task ID generation
# ---------------------------------------------------------------------------

def make_task_id(project: str, source_file: str, fn_name: str) -> str:
    """Generate a unique task ID: PREFIX_spec__FILESTEM__FNNAME.

    Includes the immediate parent dir (e.g. 'verified' or 'unverified')
    to disambiguate.  Truncates with a hash suffix if the result would
    exceed filesystem limits.
    """
    import hashlib
    prefix = PROJECT_PREFIX.get(project, project[:2].upper())
    parts = Path(source_file).parts  # ('project', 'subdir', ..., 'file.rs')
    stem = Path(parts[-1]).stem
    # Include parent dir to disambiguate verified/unverified copies
    parent = parts[-2] if len(parts) > 2 else ""
    if parent:
        raw = f"{prefix}_spec__{parent}__{stem}__{fn_name}"
    else:
        raw = f"{prefix}_spec__{stem}__{fn_name}"
    # Linux filename limit is 255 bytes; keep .rs extension (3) + safety
    MAX_LEN = 240
    if len(raw) > MAX_LEN:
        h = hashlib.md5(raw.encode()).hexdigest()[:8]
        raw = raw[:MAX_LEN - 9] + "_" + h
    return raw


def difficulty_tier(body_lines: int) -> str:
    for tier, (lo, hi) in DIFFICULTY_TIERS.items():
        if lo <= body_lines <= hi:
            return tier
    return "very-hard"


# ---------------------------------------------------------------------------
# Generate subcommand  — process one source file
# ---------------------------------------------------------------------------

def process_file(
    filepath: Path,
    project: str,
    source_dir: Path,
    verus_bin: str,
    timeout: int,
    dedup: set[str],
    max_file_lines: int = 2000,
) -> list[tuple[TaskRecord, str, str]]:
    """Process a single verified source file.

    Returns list of (record, task_code, gt_code) for accepted tasks.
    """
    source = filepath.read_text()

    # Skip files with known issues
    issue = source_has_known_issues(source)
    if issue:
        return []

    # Parse all spec fns
    spec_fns = parse_spec_functions(source)
    if not spec_fns:
        return []

    file_lines = source.count('\n') + 1

    # Skip very large files — they almost always timeout on Verus
    if file_lines > max_file_lines:
        return []

    rel_path = str(filepath.relative_to(source_dir))

    # First: verify the original source compiles
    orig_ok, _ = run_verus(source, verus_bin, timeout)
    if not orig_ok:
        return []

    accepted = []

    for sf in spec_fns:
        # --- Filter: skip trivial / uninterp / external ---
        if body_is_trivial(sf.body):
            continue
        if not sf.is_open:
            continue
        if has_uninterp_or_external(source, sf.fn_line):
            continue
        if sf.body_line_count < 1:
            continue

        # --- Dedup ---
        bh = body_hash(sf.body)
        dedup_key = f"{project}:{sf.name}:{bh}"
        if dedup_key in dedup:
            continue
        dedup.add(dedup_key)

        # --- Erasure test: does removing the body cause Verus to fail? ---
        erased_code = insert_markers_and_erase(source, sf)
        erased_ok, _ = run_verus(erased_code, verus_bin, timeout)
        if erased_ok:
            # Verus still passes after erasure → spec is not load-bearing
            continue

        # Good task! Erasing it breaks proofs.
        task_id = make_task_id(project, rel_path, sf.name)
        task_code = erased_code                           # markers + arbitrary()
        gt_code = insert_markers_keep_body(source, sf)    # markers + original body

        record = TaskRecord(
            task_id=task_id,
            source_project=project,
            source_file=rel_path,
            target_fn=sf.name,
            target_return_type=sf.return_type,
            body_lines=sf.body_line_count,
            file_lines=file_lines,
            is_open=sf.is_open,
            difficulty=difficulty_tier(sf.body_line_count),
            ground_truth_body=sf.body.strip(),
        )
        accepted.append((record, task_code, gt_code))

    return accepted


# ---------------------------------------------------------------------------
# Generate subcommand
# ---------------------------------------------------------------------------

def cmd_generate(args):
    source_dir = args.source_dir
    output_dir = args.output_dir
    verus_bin = args.verus_bin
    timeout = args.timeout

    tasks_dir = output_dir / "tasks"
    gt_dir = output_dir / "ground-truth"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Collect all .rs files to process
    all_files: list[tuple[Path, str]] = []
    for proj_dir in sorted(source_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        project = proj_dir.name
        if args.only_project and project != args.only_project:
            continue
        for rs in sorted(proj_dir.rglob("*.rs")):
            all_files.append((rs, project))

    print(f"Found {len(all_files)} source files across "
          f"{len(set(p for _, p in all_files))} projects")

    all_records: list[TaskRecord] = []
    dedup: set[str] = set()
    used_ids: dict[str, int] = {}       # task_id -> count (for collision handling)
    t0 = time.time()

    for i, (filepath, project) in enumerate(all_files, 1):
        if i % 10 == 0 or i == len(all_files):
            elapsed = time.time() - t0
            print(f"\r  [{i}/{len(all_files)}] "
                  f"{len(all_records)} tasks accepted "
                  f"({elapsed:.0f}s)", end="", flush=True)

        results = process_file(
            filepath, project, source_dir,
            verus_bin, timeout, dedup,
            max_file_lines=args.max_file_lines,
        )
        for record, task_code, gt_code in results:
            tid = record.task_id
            # Handle task ID collisions (same fn name in different impls)
            if tid in used_ids:
                used_ids[tid] += 1
                tid = f"{tid}_{used_ids[tid]}"
                record = TaskRecord(**{**asdict(record), "task_id": tid})
            else:
                used_ids[tid] = 1
            (tasks_dir / f"{tid}.rs").write_text(task_code)
            (gt_dir / f"{tid}.rs").write_text(gt_code)
            all_records.append(record)

    print()
    elapsed = time.time() - t0
    print(f"\nGenerated {len(all_records)} verified tasks in {elapsed:.0f}s")

    # Write metadata
    jsonl = output_dir / "tasks.jsonl"
    with open(jsonl, "w") as f:
        for r in all_records:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"Metadata: {jsonl}")

    _print_summary(all_records)


# ---------------------------------------------------------------------------
# Sample subcommand
# ---------------------------------------------------------------------------

def cmd_sample(args):
    pool_dir = args.pool_dir
    jsonl = pool_dir / "tasks.jsonl"
    if not jsonl.exists():
        print(f"Error: {jsonl} not found", file=sys.stderr)
        sys.exit(1)

    records = [json.loads(line) for line in open(jsonl)]
    print(f"Loaded {len(records)} candidate tasks from pool")

    selected = stratified_sample(records, args.target, args.seed)
    print(f"Selected {len(selected)} tasks")

    # Write output
    out_dir = args.output_dir
    out_tasks = out_dir / "tasks"
    out_gt = out_dir / "ground-truth"
    out_tasks.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    pool_tasks = pool_dir / "tasks"
    pool_gt = pool_dir / "ground-truth"

    for rec in selected:
        tid = rec["task_id"]
        src_t = pool_tasks / f"{tid}.rs"
        src_g = pool_gt / f"{tid}.rs"
        if src_t.exists():
            shutil.copy2(src_t, out_tasks / f"{tid}.rs")
        if src_g.exists():
            shutil.copy2(src_g, out_gt / f"{tid}.rs")

    with open(out_dir / "tasks.jsonl", "w") as f:
        for rec in selected:
            f.write(json.dumps(rec) + "\n")

    print(f"\nBenchmark written to {out_dir}")
    _print_summary_from_dicts(selected)


def stratified_sample(
    records: list[dict],
    target: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Stratified sample across difficulty tiers and projects."""
    rng = random.Random(seed)

    # Ensure difficulty is set
    for r in records:
        r.setdefault("difficulty", difficulty_tier(r["body_lines"]))

    # Group by (difficulty, project)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["difficulty"], r["source_project"])].append(r)

    selected: list[dict] = []

    for tier, tier_target in TIER_TARGETS.items():
        proj_pools: dict[str, list[dict]] = {}
        for (d, p), rs in groups.items():
            if d == tier:
                proj_pools[p] = list(rs)

        if not proj_pools:
            continue

        total_pool = sum(len(v) for v in proj_pools.values())

        # 1 per project minimum, rest proportional
        alloc: dict[str, int] = {p: 1 for p in proj_pools}
        remaining = tier_target - len(proj_pools)
        if remaining > 0:
            for p in proj_pools:
                extra = round(remaining * len(proj_pools[p]) / total_pool)
                alloc[p] += extra

        for p, n in alloc.items():
            pool = proj_pools[p]
            n = min(n, len(pool))
            sampled = _diverse_sample(pool, n, rng)
            selected.extend(sampled)

    # Fill if short
    if len(selected) < target:
        ids = {r["task_id"] for r in selected}
        rest = [r for r in records if r["task_id"] not in ids]
        proj_counts = Counter(r["source_project"] for r in selected)
        rest.sort(key=lambda r: (proj_counts.get(r["source_project"], 0), rng.random()))
        selected.extend(rest[:target - len(selected)])

    # Trim if over
    while len(selected) > target:
        tier_counts = Counter(r["difficulty"] for r in selected)
        most = tier_counts.most_common(1)[0][0]
        tier_items = [i for i, r in enumerate(selected) if r["difficulty"] == most]
        selected.pop(rng.choice(tier_items))

    return selected


def _diverse_sample(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n items preferring return-type diversity."""
    if n >= len(pool):
        return list(pool)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_type[r["target_return_type"]].append(r)

    selected = []
    types = list(by_type.keys())
    rng.shuffle(types)
    for t in types:
        if len(selected) >= n:
            break
        item = rng.choice(by_type[t])
        selected.append(item)
        by_type[t].remove(item)

    if len(selected) < n:
        remaining = [r for rs in by_type.values() for r in rs]
        rng.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    return selected


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_summary(records: list[TaskRecord]):
    dicts = [asdict(r) for r in records]
    _print_summary_from_dicts(dicts)


def _print_summary_from_dicts(records: list[dict]):
    print(f"\n{'='*60}")
    print(f"Tasks: {len(records)}")

    print("\nDifficulty:")
    for tier, cnt in sorted(Counter(r["difficulty"] for r in records).items()):
        print(f"  {tier}: {cnt}")

    print("\nProjects:")
    for p, cnt in sorted(Counter(r["source_project"] for r in records).items()):
        print(f"  {p}: {cnt}")

    rt = Counter(r["target_return_type"] for r in records)
    print(f"\nReturn types: {len(rt)} distinct")
    for t, cnt in rt.most_common(10):
        print(f"  {t}: {cnt}")

    sizes = sorted(r["body_lines"] for r in records)
    if sizes:
        mid = sizes[len(sizes) // 2]
        avg = sum(sizes) / len(sizes)
        print(f"\nBody size: {sizes[0]}-{sizes[-1]} lines "
              f"(median={mid}, mean={avg:.1f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verus-bin", default=VERUS_DEFAULT)

    sub = parser.add_subparsers(dest="command")

    # --- generate ---
    gen = sub.add_parser("generate",
                         help="Scan sources, filter via Verus, emit tasks")
    gen.add_argument("--source-dir", type=Path,
                     default=Path("benchmarks/VeruSAGE-Bench/source-projects"))
    gen.add_argument("--output-dir", type=Path,
                     default=Path("benchmarks/VeruSAGE-Bench-Spec-v2"))
    gen.add_argument("--only-project", type=str, default=None,
                     help="Process only this project (e.g. 'ironkv')")
    gen.add_argument("--timeout", type=int, default=120,
                     help="Verus timeout per invocation (seconds)")
    gen.add_argument("--max-file-lines", type=int, default=2000,
                     help="Skip files above this line count (default: 2000)")

    # --- sample ---
    samp = sub.add_parser("sample",
                          help="Sample diverse subset from candidate pool")
    samp.add_argument("--pool-dir", type=Path, required=True,
                      help="Pool directory (output of 'generate')")
    samp.add_argument("--output-dir", type=Path, required=True)
    samp.add_argument("--target", type=int, default=100)
    samp.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "sample":
        cmd_sample(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
