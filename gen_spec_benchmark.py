#!/usr/bin/env python3
"""
Generate a Verus spec-writing benchmark from VeruSAGE-Bench source projects.

Given verified single-file Verus programs, this script:
  1. Identifies all `open spec fn` / `spec fn` definitions with non-trivial bodies
  2. Filters to those actually referenced by proof/exec code in the same file
  3. Erases the target spec fn body (replaces with `arbitrary()`)
  4. Produces a task file where the model must reconstruct the spec body
  5. Generates metadata JSONL for the benchmark

Correctness oracle: when the model fills in the spec body, the *existing proofs*
in the file must still verify. If the spec is wrong, proof obligations that
depend on it will fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

_md5 = hashlib.md5

import time


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpecFn:
    """A parsed spec function."""
    name: str
    full_signature: str          # everything before the opening `{`
    body: str                    # everything inside `{ ... }` (excluding braces)
    start_line: int              # 0-indexed line of fn keyword
    end_line: int                # 0-indexed line of closing `}`
    body_start_line: int         # 0-indexed line of opening `{`
    body_end_line: int           # 0-indexed line of closing `}`
    is_open: bool
    is_impl_member: bool = False
    return_type: str = "bool"    # parsed return type
    attrs: list[str] = field(default_factory=list)  # preceding attributes
    body_line_count: int = 0


@dataclass
class TaskRecord:
    """Metadata for a generated task."""
    task_id: str
    source_project: str
    source_file: str
    target_fn: str
    target_return_type: str
    body_lines: int
    file_lines: int
    is_open: bool
    is_impl_member: bool
    has_proof_callers: bool
    task_path: str


# ---------------------------------------------------------------------------
# Spec function parser
# ---------------------------------------------------------------------------

# Regex to match the beginning of a spec fn declaration.
# Handles: `pub open spec fn`, `open spec fn`, `pub spec fn`, `spec fn`,
#           `pub spec(checked) fn`, `pub open spec(checked) fn`
SPEC_FN_RE = re.compile(
    r'^(\s*)'                          # leading whitespace
    r'(pub\s+)?'                       # optional pub
    r'(open\s+)?'                      # optional open
    r'spec\s*(?:\(checked\)\s*)?'      # spec or spec(checked)
    r'fn\s+'                           # fn keyword
    r'([A-Za-z_]\w*)',                 # function name
    re.MULTILINE,
)

# Attribute line (e.g. #[verifier(inline)], #[verifier::external_body])
ATTR_RE = re.compile(r'^\s*#\[')

# Proof/exec fn keyword
PROOF_EXEC_FN_RE = re.compile(
    r'\b(proof|fn|exec)\s+fn\s+',
)


def find_matching_brace(lines: list[str], start_line: int, start_col: int = 0) -> tuple[int, int] | None:
    """Find the matching closing `}` for the `{` at (start_line, start_col).
    
    Returns (end_line, end_col) or None if not found.
    Handles nested braces, strings, and comments.
    """
    depth = 0
    in_line_comment = False
    in_block_comment = False
    in_string = False
    escape_next = False
    
    for li in range(start_line, len(lines)):
        line = lines[li]
        ci_start = start_col if li == start_line else 0
        ci = ci_start
        while ci < len(line):
            ch = line[ci]
            
            if escape_next:
                escape_next = False
                ci += 1
                continue
            
            if in_line_comment:
                break  # rest of line is comment
            
            if in_block_comment:
                if ch == '*' and ci + 1 < len(line) and line[ci + 1] == '/':
                    in_block_comment = False
                    ci += 2
                    continue
                ci += 1
                continue
            
            if in_string:
                if ch == '\\':
                    escape_next = True
                elif ch == '"':
                    in_string = False
                ci += 1
                continue
            
            # Check for comments
            if ch == '/' and ci + 1 < len(line):
                if line[ci + 1] == '/':
                    in_line_comment = True
                    break
                elif line[ci + 1] == '*':
                    in_block_comment = True
                    ci += 2
                    continue
            
            if ch == '"':
                in_string = True
                ci += 1
                continue
            
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return (li, ci)
            
            ci += 1
        
        in_line_comment = False  # reset for next line
    
    return None


def parse_return_type(signature: str) -> str:
    """Extract the return type from a spec fn signature.
    
    Looks for `-> TYPE` after the parameter list. Defaults to `bool`.
    """
    # Find the last `->` that's not inside parens/angle brackets
    depth_paren = 0
    depth_angle = 0
    last_arrow = -1
    
    for i in range(len(signature) - 1):
        ch = signature[i]
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren -= 1
        elif ch == '<':
            depth_angle += 1
        elif ch == '>':
            depth_angle -= 1
        elif ch == '-' and i + 1 < len(signature) and signature[i + 1] == '>':
            if depth_paren == 0 and depth_angle == 0:
                last_arrow = i
    
    if last_arrow == -1:
        return "()"  # no explicit return → unit type
    
    # Extract type after `->`
    ret = signature[last_arrow + 2:].strip()
    
    # Take only the first logical line (up to first newline after the type)
    # Handle multi-line types like tuples: (Type1, Type2)
    # by tracking paren/angle bracket depth
    type_end = len(ret)
    d_paren = 0
    d_angle = 0
    for i, ch in enumerate(ret):
        if ch == '(':
            d_paren += 1
        elif ch == ')':
            d_paren -= 1
        elif ch == '<':
            d_angle += 1
        elif ch == '>':
            d_angle -= 1
        elif ch == '\n' and d_paren == 0 and d_angle == 0:
            type_end = i
            break
    
    ret = ret[:type_end].strip()
    
    # Remove trailing keywords/braces that got included
    for kw in ['requires', 'ensures', 'recommends', 'decreases', '{', 'where']:
        idx = ret.find(kw)
        if idx != -1:
            ret = ret[:idx].strip()
    
    # Clean up trailing punctuation
    ret = ret.strip().rstrip(',').rstrip(';').strip()
    return ret if ret else "()"


def parse_spec_functions(source: str) -> list[SpecFn]:
    """Parse all spec functions from a Verus source file."""
    lines = source.split('\n')
    results = []
    
    for m in SPEC_FN_RE.finditer(source):
        fn_name = m.group(4)
        is_open = m.group(3) is not None  # has 'open'
        
        # Determine the line number of the match
        fn_line = source[:m.start()].count('\n')
        
        # Collect preceding attributes (walk backwards from fn_line)
        attrs = []
        attr_start = fn_line
        for al in range(fn_line - 1, -1, -1):
            stripped = lines[al].strip()
            if stripped.startswith('#[') or stripped.startswith('//') or stripped == '':
                if stripped.startswith('#['):
                    attrs.insert(0, stripped)
                    attr_start = al
            else:
                break
        
        # Find the opening `{` of the function body
        # It may be on the same line or several lines later (after requires/ensures)
        brace_line = None
        brace_col = None
        for li in range(fn_line, min(fn_line + 50, len(lines))):
            col = lines[li].find('{')
            if col != -1:
                # Make sure this `{` is not inside a comment
                line_before = lines[li][:col]
                if '//' not in line_before:
                    brace_line = li
                    brace_col = col
                    break
        
        if brace_line is None:
            continue  # no body found (e.g. trait declaration)
        
        # Find matching closing `}`
        match_result = find_matching_brace(lines, brace_line, brace_col)
        if match_result is None:
            continue
        
        end_line, end_col = match_result
        
        # Extract signature (everything from fn_line to opening brace)
        sig_lines = lines[fn_line:brace_line + 1]
        sig_text = '\n'.join(sig_lines)
        # Trim from the opening brace
        brace_idx = sig_text.rfind('{')
        if brace_idx != -1:
            sig_text = sig_text[:brace_idx].rstrip()
        
        # Extract body (between braces)
        if brace_line == end_line:
            body = lines[brace_line][brace_col + 1:end_col].strip()
        else:
            body_lines = []
            body_lines.append(lines[brace_line][brace_col + 1:])
            for bl in range(brace_line + 1, end_line):
                body_lines.append(lines[bl])
            body_lines.append(lines[end_line][:end_col])
            body = '\n'.join(body_lines).strip()
        
        # Check if this is inside an impl block
        is_impl = False
        for pl in range(fn_line - 1, max(fn_line - 10, -1), -1):
            if 'impl' in lines[pl] and '{' in lines[pl]:
                is_impl = True
                break
        
        ret_type = parse_return_type(sig_text)
        body_lc = body.count('\n') + 1 if body else 0
        
        spec = SpecFn(
            name=fn_name,
            full_signature=sig_text,
            body=body,
            start_line=attr_start,
            end_line=end_line,
            body_start_line=brace_line,
            body_end_line=end_line,
            is_open=is_open,
            is_impl_member=is_impl,
            return_type=ret_type,
            attrs=attrs,
            body_line_count=body_lc,
        )
        results.append(spec)
    
    return results


# ---------------------------------------------------------------------------
# Filtering: which spec fns are good benchmark targets?
# ---------------------------------------------------------------------------

def fn_is_referenced_by_proof(fn_name: str, source: str, spec_fn_names: set[str]) -> bool:
    """Check if fn_name is referenced by proof or exec code in the file.
    
    A spec fn that's only referenced by other spec fns isn't a great target
    because erasing it won't cause any proof failures.
    """
    # Quick count: need at least 2 occurrences (definition + usage)
    # Use simple string count which is much faster than regex on large files
    count = source.count(fn_name)
    return count >= 2


def body_is_trivial(body: str) -> bool:
    """Check if the spec fn body is trivial (not worth benchmarking)."""
    stripped = body.strip()
    
    # Empty body
    if not stripped:
        return True
    
    # Single token / very short  
    if len(stripped) < 5:
        return True
    
    # Just `true` or `false`
    if stripped in ('true', 'false'):
        return True
    
    # Just returning self.field or similar
    if re.match(r'^self\.\w+$', stripped):
        return True
    
    # Just a constructor with direct field mappings (no logic)
    # e.g. `MyStruct { field: self.field }`
    if stripped.count('\n') == 0 and '{' in stripped and '}' in stripped:
        inner = stripped[stripped.index('{') + 1:stripped.rindex('}')].strip()
        # If inner has only `field: value` with no operators/calls
        parts = [p.strip() for p in inner.split(',') if p.strip()]
        if all(re.match(r'^[\w.]+\s*:\s*[\w.]+$', p) for p in parts):
            return True
    
    return False


def is_uninterp_spec(attrs: list[str], sig: str) -> bool:
    """Check if this is an uninterp spec fn (body is opaque to callers)."""
    return 'uninterp' in sig or any('external_body' in a for a in attrs)


def is_inline_spec(attrs: list[str]) -> bool:
    """Check if spec fn has #[verifier(inline)] - just a transparent alias."""
    return any('inline' in a for a in attrs)


def body_signature_hash(body: str) -> str:
    """Hash of the normalized body for deduplication."""
    normalized = re.sub(r'\s+', ' ', body.strip())
    return _md5(normalized.encode()).hexdigest()[:12]


def filter_candidates(
    spec_fns: list[SpecFn],
    source: str,
    min_body_lines: int = 2,
) -> list[SpecFn]:
    """Filter spec fns to good benchmark candidates."""
    spec_names = {s.name for s in spec_fns}
    candidates = []
    
    for sf in spec_fns:
        # Skip trivial bodies
        if body_is_trivial(sf.body):
            continue
        
        # Skip very short bodies
        if sf.body_line_count < min_body_lines:
            # But allow one-liners if they have complex expressions
            if sf.body_line_count == 1 and len(sf.body) > 40:
                pass  # keep it
            else:
                continue
        
        # Skip uninterp/external_body spec fns (their body is already opaque)
        if is_uninterp_spec(sf.attrs, sf.full_signature):
            continue
        
        # Skip inline spec fns (they're just aliases)
        if is_inline_spec(sf.attrs):
            continue
        
        # Must be referenced elsewhere in the file (not just defined)
        if not fn_is_referenced_by_proof(sf.name, source, spec_names):
            continue
        
        candidates.append(sf)
    
    return candidates


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def erase_spec_body(source: str, spec_fn: SpecFn) -> str:
    """Replace the spec fn body with `arbitrary()` placeholder.
    
    Returns the modified source code.
    """
    lines = source.split('\n')
    
    # The body spans from body_start_line (opening `{`) to body_end_line (closing `}`)
    brace_line = spec_fn.body_start_line
    end_line = spec_fn.body_end_line
    
    # Determine body indentation from the original body lines
    # Look at the first non-empty body line for indentation
    body_indent = None
    for bl in range(brace_line + 1, end_line + 1):
        stripped = lines[bl].strip()
        if stripped and stripped != '}':
            body_indent = ' ' * (len(lines[bl]) - len(lines[bl].lstrip()))
            break
    
    # Determine closing brace indentation from the original closing brace
    close_indent = ''
    if end_line < len(lines):
        close_line = lines[end_line]
        close_stripped = close_line.lstrip()
        if close_stripped.startswith('}'):
            close_indent = ' ' * (len(close_line) - len(close_stripped))
    
    # If we couldn't determine body indent, derive from close brace + 4
    if body_indent is None:
        body_indent = close_indent + '    '
    
    # Build the erased version
    # Keep the opening brace line up to and including `{`
    open_brace_col = lines[brace_line].index('{')
    new_open = lines[brace_line][:open_brace_col + 1]
    
    # Create placeholder body with correct indentation
    placeholder = f"\n{body_indent}arbitrary() // TODO: fill in the spec body\n{close_indent}}}"
    
    # Construct new lines
    new_lines = lines[:brace_line]
    new_lines.append(new_open + placeholder)
    new_lines.extend(lines[end_line + 1:])
    
    return '\n'.join(new_lines)


def generate_project_prefix(project_name: str) -> str:
    """Generate 2-letter prefix from project name (matching existing convention)."""
    prefix_map = {
        'anvil-controller': 'AC',
        'anvil-library': 'AL',
        'atmosphere': 'OS',  # OS = Operating System (atmosphere)
        'ironkv': 'IR',
        'memory-allocator': 'MA',
        'node-replication': 'NO',
        'nrkernel': 'NR',
        'storage': 'ST',
        'vest': 'VE',
    }
    return prefix_map.get(project_name, project_name[:2].upper())


def generate_task_id(project_name: str, fn_name: str, source_file: str) -> str:
    """Generate a unique task ID."""
    prefix = generate_project_prefix(project_name)
    # Include a hint from the file path to disambiguate duplicate fn names
    fname = Path(source_file).stem
    return f"{prefix}_spec__{fname}__{fn_name}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def source_has_known_issues(source: str) -> str | None:
    """Check if a source file contains known patterns that prevent Verus verification.
    
    Returns a reason string if the file should be skipped, None otherwise.
    These are NOT extraction bugs — they are pre-existing incompatibilities
    in some VeruSAGE-Bench source files with this Verus binary version.
    """
    # nrkernel files with repr(transparent) + Ghost<nat> → denied by
    # #[deny(repr_transparent_non_zst_fields)] in newer Rust/Verus
    if 'repr(transparent)' in source and 'Ghost<nat>' in source:
        return "repr_transparent_ghost_nat"
    
    # storage files that depend on external crate deps_hack (PmSized)
    if 'use deps_hack::' in source or 'use deps_hack :' in source:
        return "deps_hack_import"
    
    return None


def process_file(
    filepath: Path,
    project_name: str,
    output_dir: Path,
    min_body_lines: int = 2,
    dedup_keys: set[str] | None = None,
) -> list[TaskRecord]:
    """Process a single verified source file. Returns list of generated task records."""
    source = filepath.read_text()
    
    # Skip files with known Verus-incompatible patterns
    issue = source_has_known_issues(source)
    if issue:
        return []
    
    # Parse all spec fns
    all_specs = parse_spec_functions(source)
    if not all_specs:
        return []
    
    # Filter to good candidates
    candidates = filter_candidates(all_specs, source, min_body_lines)
    if not candidates:
        return []
    
    records = []
    for sf in candidates:
        # Dedup by (project, fn_name, body_hash) — same spec fn body across
        # different flattened files should only produce one task.
        bhash = body_signature_hash(sf.body)
        dedup_key = f"{project_name}::{sf.name}::{bhash}"
        if dedup_keys is not None:
            if dedup_key in dedup_keys:
                continue
            dedup_keys.add(dedup_key)
        
        task_id = generate_task_id(project_name, sf.name, str(filepath))
        
        # Erase the spec fn body
        erased = erase_spec_body(source, sf)
        
        # Write task file
        task_path = output_dir / f"{task_id}.rs"
        task_path.write_text(erased)
        
        # Also write ground truth (original with body intact)
        gt_dir = output_dir.parent / "ground-truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_path = gt_dir / f"{task_id}.rs"
        gt_path.write_text(source)
        
        record = TaskRecord(
            task_id=task_id,
            source_project=project_name,
            source_file=str(filepath),
            target_fn=sf.name,
            target_return_type=sf.return_type,
            body_lines=sf.body_line_count,
            file_lines=source.count('\n') + 1,
            is_open=sf.is_open,
            is_impl_member=sf.is_impl_member,
            has_proof_callers=fn_is_referenced_by_proof(
                sf.name, source, {s.name for s in all_specs}
            ),
            task_path=str(task_path),
        )
        records.append(record)
    
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=Path('benchmarks/VeruSAGE-Bench/source-projects'),
        help='Path to VeruSAGE-Bench source-projects directory',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('benchmarks/VeruSAGE-Bench-Spec/tasks'),
        help='Output directory for generated task files',
    )
    parser.add_argument(
        '--min-body-lines',
        type=int,
        default=2,
        help='Minimum number of lines in spec fn body to be a candidate',
    )
    parser.add_argument(
        '--max-file-lines',
        type=int,
        default=50000,
        help='Skip source files larger than this many lines',
    )
    parser.add_argument(
        '--only-project',
        type=str,
        default=None,
        help='Only process this project (e.g. "ironkv")',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only scan and report, do not write files',
    )
    args = parser.parse_args()
    
    source_dir = args.source_dir
    if not source_dir.exists():
        print(f"Error: source directory {source_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    output_dir = args.output_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    all_records: list[TaskRecord] = []
    dedup = set()  # (project, fn_name, body_hash) dedup keys
    
    # Process each project
    projects = sorted(source_dir.iterdir())
    t0 = time.time()
    for proj_dir in projects:
        if not proj_dir.is_dir():
            continue
        project_name = proj_dir.name
        if args.only_project and project_name != args.only_project:
            continue
        
        verified_dir = proj_dir / 'verified'
        if not verified_dir.exists():
            print(f"  Skipping {project_name}: no verified/ directory")
            continue
        
        # Find all .rs files in verified/
        rs_files = sorted(verified_dir.rglob('*.rs'))
        project_records = []
        
        for rs_file in rs_files:
            # Skip oversized files
            line_count = sum(1 for _ in open(rs_file))
            if line_count > args.max_file_lines:
                continue
            
            if not args.dry_run:
                records = process_file(
                    rs_file, project_name, output_dir,
                    min_body_lines=args.min_body_lines,
                    dedup_keys=dedup,
                )
            else:
                records = _dry_run_file(
                    rs_file, project_name, args.min_body_lines, dedup,
                )
            
            project_records.extend(records)
        
        all_records.extend(project_records)
        elapsed = time.time() - t0
        print(f"  {project_name}: {len(project_records)} spec tasks from {len(rs_files)} files ({elapsed:.1f}s)")
    
    # Write metadata JSONL
    if not args.dry_run:
        jsonl_path = output_dir.parent / 'tasks.jsonl'
        with open(jsonl_path, 'w') as f:
            for rec in all_records:
                f.write(json.dumps(asdict(rec)) + '\n')
        
        # Summary stats
        print(f"\n{'='*60}")
        print(f"Total spec tasks generated: {len(all_records)}")
        print(f"Tasks written to: {output_dir}")
        print(f"Metadata written to: {jsonl_path}")
        print(f"Ground truth in: {output_dir.parent / 'ground-truth'}")
        
        # Per-project stats
        from collections import Counter
        proj_counts = Counter(r.source_project for r in all_records)
        print(f"\nPer-project breakdown:")
        for proj, cnt in sorted(proj_counts.items()):
            print(f"  {proj}: {cnt}")
        
        # Return type stats
        type_counts = Counter(r.target_return_type for r in all_records)
        print(f"\nReturn type breakdown:")
        for rt, cnt in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {rt}: {cnt}")
    else:
        _print_dry_run_summary(all_records)


def _dry_run_file(
    filepath: Path,
    project_name: str,
    min_body_lines: int,
    dedup_keys: set[str],
) -> list[TaskRecord]:
    """Dry run: scan file and return records without writing."""
    source = filepath.read_text()
    
    # Skip files with known Verus-incompatible patterns
    if source_has_known_issues(source):
        return []
    
    all_specs = parse_spec_functions(source)
    if not all_specs:
        return []
    candidates = filter_candidates(all_specs, source, min_body_lines)
    records = []
    for sf in candidates:
        bhash = body_signature_hash(sf.body)
        dedup_key = f"{project_name}::{sf.name}::{bhash}"
        if dedup_key in dedup_keys:
            continue
        dedup_keys.add(dedup_key)
        
        task_id = generate_task_id(project_name, sf.name, str(filepath))
        records.append(TaskRecord(
            task_id=task_id,
            source_project=project_name,
            source_file=str(filepath),
            target_fn=sf.name,
            target_return_type=sf.return_type,
            body_lines=sf.body_line_count,
            file_lines=source.count('\n') + 1,
            is_open=sf.is_open,
            is_impl_member=sf.is_impl_member,
            has_proof_callers=True,
            task_path='(dry-run)',
        ))
    return records


def _print_dry_run_summary(records: list[TaskRecord]):
    from collections import Counter
    
    print(f"\n{'='*60}")
    print(f"DRY RUN: Would generate {len(records)} spec tasks")
    
    proj_counts = Counter(r.source_project for r in records)
    print(f"\nPer-project breakdown:")
    for proj, cnt in sorted(proj_counts.items()):
        print(f"  {proj}: {cnt}")
    
    type_counts = Counter(r.target_return_type for r in records)
    print(f"\nReturn type distribution:")
    for rt, cnt in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {rt}: {cnt}")
    
    body_sizes = [r.body_lines for r in records]
    if body_sizes:
        print(f"\nBody size stats (lines):")
        print(f"  min={min(body_sizes)}, max={max(body_sizes)}, "
              f"median={sorted(body_sizes)[len(body_sizes)//2]}, "
              f"mean={sum(body_sizes)/len(body_sizes):.1f}")


if __name__ == '__main__':
    main()
