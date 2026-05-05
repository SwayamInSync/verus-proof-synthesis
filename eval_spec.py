#!/usr/bin/env python3
"""
Two-stage evaluation for spec-writing benchmark tasks.

Stage 1 – Patch & Verify:
    Replace the erased body (between ``// %%SPEC_START`` and
    ``// %%SPEC_END`` markers) with the model's generated spec body,
    then run Verus.  If Verus reports 0 errors → Stage 1 passes.

Stage 2 – Equivalence Check:
    Starting from the ground truth file (which already verifies),
    inject the model's spec as a `__model` variant and add a proof lemma
    that asserts extensional equality between GT and model spec.
    If Verus can prove the equivalence lemma → Stage 2 passes.
    Skipped when the erased region is not a single spec fn body
    (e.g. trait members, impl bodies, macro content).

Usage:
    python eval_spec.py single --task-id AC_spec__...__allocate \\
        --model-body body.rs \\
        --benchmark-dir benchmarks/VeruSAGE-Bench-Spec-v2

    python eval_spec.py batch --model-dir results/qwen3/ \\
        --benchmark-dir benchmarks/VeruSAGE-Bench-Spec-v2

    python eval_spec.py selftest \\
        --benchmark-dir benchmarks/VeruSAGE-Bench-Spec-v2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# New marker format (spec_benchmark.py v2)
SPEC_START_MARKER = "// %%SPEC_START"
SPEC_END_MARKER   = "// %%SPEC_END"

# Legacy marker (for backward compatibility with old benchmarks)
ERASED_MARKER_LEGACY = "arbitrary() // TODO: fill in the spec body"

VERUS_DEFAULT = os.environ.get(
    "VERUS_BIN",
    str(Path(__file__).resolve().parent.parent.parent
        / "verus-binary" / "verus" / "verus"),
)

# Matches:  [pub] [open|closed] [uninterp] spec[(checked)] fn <name>
SPEC_FN_RE = re.compile(
    r"^\s*(pub\s+)?(open\s+|closed\s+)?(uninterp\s+)?"
    r"spec\s*(?:\(checked\)\s*)?fn\s+([A-Za-z_]\w*)",
)


@dataclass
class EvalResult:
    task_id: str
    stage1_pass: bool
    stage1_detail: str
    stage2_pass: bool | None        # None = skipped
    stage2_detail: str


# ---------------------------------------------------------------------------
# Verus runner
# ---------------------------------------------------------------------------

def run_verus(code: str, verus_bin: str, timeout: int = 120) -> tuple[bool, str]:
    """Write *code* to a temp file, invoke Verus, return ``(passed, snippet)``."""
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
        lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
        snippet = "\n".join(lines[-3:]) if lines else "(no output)"
        return passed, snippet
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except OSError as e:
        return False, f"os_error: {e}"
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Brace matching
# ---------------------------------------------------------------------------

def find_matching_brace(lines: list[str], start_line: int) -> int | None:
    """Return the line of the closing ``}`` matching the first ``{`` on
    *start_line*.  Handles strings, ``//`` and ``/* */`` comments."""
    depth = 0
    in_block_comment = False
    in_string = False
    escape_next = False

    for li in range(start_line, len(lines)):
        line = lines[li]
        ci = line.find("{") if li == start_line else 0
        if li == start_line and ci == -1:
            continue
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
                    break
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
# Marker & structural-location helpers
# ---------------------------------------------------------------------------

def find_marker_line(lines: list[str]) -> int | None:
    """Return 0-indexed line of the SPEC_START marker (or legacy marker)."""
    for i, line in enumerate(lines):
        if SPEC_START_MARKER in line:
            return i
    # Fallback: legacy format
    for i, line in enumerate(lines):
        if ERASED_MARKER_LEGACY in line:
            return i
    return None


def find_marker_end_line(lines: list[str], start: int) -> int | None:
    """Return 0-indexed line of ``// %%SPEC_END`` after *start*, or None."""
    for i in range(start + 1, len(lines)):
        if SPEC_END_MARKER in lines[i]:
            return i
    return None


def _uses_new_markers(lines: list[str]) -> bool:
    """Check whether the file uses the new SPEC_START/SPEC_END markers."""
    return any(SPEC_START_MARKER in l for l in lines)


def find_enclosing_spec_fn(lines: list[str], target_line: int) -> int | None:
    """Walk backward from *target_line* to find a ``spec fn`` whose brace-
    delimited body actually **contains** *target_line*.

    Returns the fn declaration line index, or None (marker is not inside
    a spec fn – e.g. it is in a trait body, impl body, or macro).
    """
    for i in range(target_line, -1, -1):
        m = SPEC_FN_RE.match(lines[i])
        if m:
            brace_line = _find_opening_brace(lines, i)
            if brace_line is None:
                continue                     # abstract (trait) method
            fn_end = find_matching_brace(lines, brace_line)
            if fn_end is not None and brace_line <= target_line <= fn_end:
                return i
            continue                         # fn doesn't contain target_line
        # Stop at structural boundaries that cannot nest a spec fn.
        stripped = lines[i].strip()
        if re.match(
            r"^(pub\s+)?(trait|impl|struct|enum|mod|macro_rules!)\b",
            stripped,
        ):
            break
    return None


def _find_opening_brace(lines: list[str], start: int) -> int | None:
    """Find the ``{`` that opens the body of a fn declared at *start*.

    Returns None for bodyless declarations (trait methods ending with ``;``).
    """
    for j in range(start, min(start + 30, len(lines))):
        line = lines[j]
        # A `;` before any `{` means this is a bodyless declaration
        semi_pos = line.find(";")
        brace_pos = line.find("{")
        if semi_pos != -1 and (brace_pos == -1 or semi_pos < brace_pos):
            return None
        if brace_pos != -1:
            return j
    return None


def _is_inside_impl(lines: list[str], fn_line: int) -> bool:
    """True when *fn_line* is inside an ``impl`` block (not a trait)."""
    fn_indent = len(lines[fn_line]) - len(lines[fn_line].lstrip())
    for i in range(fn_line - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        # Skip lines strictly deeper than the target fn
        if indent > fn_indent:
            continue
        if re.match(r"^(pub\s+)?trait\b", stripped):
            # Only a definitive boundary when at lower indent
            if indent < fn_indent:
                return False
            continue
        if stripped.startswith("impl") and "{" in lines[i]:
            end = find_matching_brace(lines, i)
            if end is not None and end > fn_line:
                return True
            continue
        if fn_indent > 0 and indent < fn_indent:
            if re.match(r"^(pub\s+)?(struct|enum|mod|fn|macro_rules!)\b", stripped):
                return False
    return False


def _is_inside_trait_impl(lines: list[str], fn_line: int) -> bool:
    """True when *fn_line* is inside an ``impl Trait for Type`` block.

    We cannot inject new methods into such blocks (E0407), so S2
    must be skipped.
    """
    fn_indent = len(lines[fn_line]) - len(lines[fn_line].lstrip())
    for i in range(fn_line - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent > fn_indent:
            continue
        if stripped.startswith("impl") and "{" in lines[i]:
            end = find_matching_brace(lines, i)
            if end is not None and end > fn_line:
                return bool(re.search(r"\bfor\b", stripped))
            continue
        if fn_indent > 0 and indent < fn_indent:
            if re.match(
                r"^(pub\s+)?(trait|struct|enum|mod|fn|macro_rules!)\b", stripped
            ):
                return False
    return False


def _is_inside_trait(lines: list[str], fn_line: int) -> bool:
    """True when *fn_line* is inside a ``trait`` definition."""
    fn_indent = len(lines[fn_line]) - len(lines[fn_line].lstrip())
    for i in range(fn_line - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent > fn_indent:
            continue
        if re.match(r"^(pub\s+)?trait\b", stripped) and "{" in lines[i]:
            end = find_matching_brace(lines, i)
            return end is not None and end > fn_line
        if stripped.startswith("impl") and "{" in lines[i]:
            end = find_matching_brace(lines, i)
            if end is not None and end > fn_line:
                return False  # inside impl, not trait
            continue
        if fn_indent > 0 and indent < fn_indent:
            if re.match(r"^(pub\s+)?(struct|enum|mod|fn|macro_rules!)\b",
                         stripped):
                return False
    return False


def _find_impl_end(lines: list[str], fn_line: int) -> int | None:
    """Closing ``}`` line of the impl block that encloses *fn_line*."""
    fn_indent = len(lines[fn_line]) - len(lines[fn_line].lstrip())
    for i in range(fn_line - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped or not stripped.startswith("impl") or "{" not in lines[i]:
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent > fn_indent:
            continue
        end = find_matching_brace(lines, i)
        if end is not None and end > fn_line:
            return end
    return None


# ---------------------------------------------------------------------------
# Lightweight signature helpers
# ---------------------------------------------------------------------------

def _collect_header(lines: list[str], fn_line: int, brace_line: int) -> str:
    return " ".join(lines[li].strip() for li in range(fn_line, brace_line + 1))


def _extract_fn_name(header: str) -> str | None:
    m = re.search(r"fn\s+([A-Za-z_]\w*)", header)
    return m.group(1) if m else None


def _extract_params_str(header: str) -> str:
    """Return the text between the balanced ``(…)`` after ``fn name[<…>]``.

    Skips ``spec(checked)`` so we find the *parameter* parens, not the
    mode modifier.
    """
    m = re.search(r"fn\s+([A-Za-z_]\w*)", header)
    if not m:
        return ""
    after = m.end()
    # Skip optional generics <…>
    if after < len(header) and header[after] == "<":
        depth = 0
        for i in range(after, len(header)):
            if header[i] == "<":
                depth += 1
            elif header[i] == ">":
                depth -= 1
                if depth == 0:
                    after = i + 1
                    break
    start = header.find("(", after)
    if start == -1:
        return ""
    depth, end = 0, -1
    for i in range(start, len(header)):
        if header[i] == "(":
            depth += 1
        elif header[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    return header[start + 1 : end].strip() if end != -1 else ""


def _split_params(s: str) -> list[str]:
    parts: list[str] = []
    depth, cur = 0, []
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _param_name(param: str) -> str:
    if param in ("self", "&self", "&mut self"):
        return param
    depth = 0
    for i, ch in enumerate(param):
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif ch == ":" and depth == 0:
            return param[:i].strip()
    return param.strip()


def _extract_return_type(header: str) -> str:
    # Find the params paren (after fn name + generics, skip spec(checked))
    m = re.search(r"fn\s+([A-Za-z_]\w*)", header)
    if not m:
        return "()"
    after = m.end()
    if after < len(header) and header[after] == "<":
        depth = 0
        for i in range(after, len(header)):
            if header[i] == "<":
                depth += 1
            elif header[i] == ">":
                depth -= 1
                if depth == 0:
                    after = i + 1
                    break
    start = header.find("(", after)
    if start == -1:
        return "()"
    depth, paren_end = 0, -1
    for i in range(start, len(header)):
        if header[i] == "(":
            depth += 1
        elif header[i] == ")":
            depth -= 1
            if depth == 0:
                paren_end = i
                break
    arrow = header.find("->", paren_end)
    if arrow == -1:
        return "()"
    rt = header[arrow + 2:].strip()
    if rt.endswith("{"):
        rt = rt[:-1].strip()
    for kw in ("recommends", "requires", "ensures", "decreases", "when"):
        idx = rt.find(kw)
        if idx > 0:
            rt = rt[:idx].strip().rstrip(",")
    return rt.strip() or "()"


def _extract_generics(header: str, fn_name: str) -> str:
    fn_kw = header.find("fn " + fn_name)
    after = fn_kw + len("fn " + fn_name)
    if after >= len(header) or header[after] != "<":
        return ""
    depth = 0
    for gi in range(after, len(header)):
        if header[gi] == "<":
            depth += 1
        elif header[gi] == ">":
            depth -= 1
            if depth == 0:
                return header[after : gi + 1]
    return ""


# ---------------------------------------------------------------------------
# Stage 1: Patch & Verify
# ---------------------------------------------------------------------------

def patch_task(task_code: str, model_body: str) -> str | None:
    """Replace the erased region with *model_body*.  Returns patched code.

    Supports both the new ``// %%SPEC_START`` / ``// %%SPEC_END`` markers
    and the legacy single-line ``arbitrary() // TODO`` marker.
    """
    lines = task_code.split("\n")
    start = find_marker_line(lines)
    if start is None:
        return None

    if _uses_new_markers(lines):
        end = find_marker_end_line(lines, start)
        if end is None:
            return None
        # Indent model body to match the marker's indentation
        marker_indent = " " * (len(lines[start]) - len(lines[start].lstrip()))
        body_lines = model_body.strip().splitlines()
        indented = []
        for bl in body_lines:
            indented.append((marker_indent + bl) if bl.strip() else bl)
        # Replace everything between SPEC_START and SPEC_END (exclusive)
        new_lines = lines[: start + 1] + indented + lines[end:]
        return "\n".join(new_lines)
    else:
        # Legacy: single-line marker replacement
        for line in lines:
            if ERASED_MARKER_LEGACY in line:
                indent = " " * (len(line) - len(line.lstrip()))
                break
        else:
            return None
        body_lines = model_body.strip().splitlines()
        indented = "\n".join(
            (indent + bl if bl.strip() else bl) for bl in body_lines
        )
        return task_code.replace(indent + ERASED_MARKER_LEGACY, indented, 1)


def stage1_verify(
    task_code: str, model_body: str, verus_bin: str, timeout: int = 120,
) -> tuple[bool, str, str]:
    """Patch & run Verus.  Returns ``(passed, detail, patched_code)``."""
    patched = patch_task(task_code, model_body)
    if patched is None:
        return False, "marker not found in task", ""
    passed, detail = run_verus(patched, verus_bin, timeout)
    return passed, detail, patched


# ---------------------------------------------------------------------------
# Stage 2: Equivalence Harness
# ---------------------------------------------------------------------------

def _strip_via_clause(header_lines: list[str]) -> list[str]:
    """Remove ``via <ident>`` from header (used with ``decreases``).

    Handles qualified paths such as ``via Self::helper_decreases``.
    """
    return [re.sub(r"\bvia\s+[A-Za-z_][\w:]*", "", l) for l in header_lines]


def build_equivalence_harness(
    gt_code: str,
    model_body: str,
    marker_line: int,           # 0-indexed line of marker in *task* file
    task_code: str | None = None,
) -> str | None:
    """Build an equivalence harness from ground-truth + model body.

    Uses position-based fn detection (via *marker_line*) to find the
    correct spec fn – avoids name-collision issues.

    Returns the harness source, or None when S2 cannot be applied
    (e.g. the target is inside a trait or is not a spec fn).
    """
    # Cross-validate: marker must be inside a spec fn in the *task* code.
    # This catches cases where the marker replaces an entire impl body
    # (multiple fns) rather than a single fn body.
    if task_code is not None:
        task_lines = task_code.split("\n")
        if find_enclosing_spec_fn(task_lines, marker_line) is None:
            return None

    gt_lines = gt_code.split("\n")

    # Lines before marker_line are identical in task and GT, so we can
    # locate the enclosing spec fn at marker_line in the GT.
    fn_line = find_enclosing_spec_fn(gt_lines, marker_line)
    if fn_line is None:
        return None                          # not inside a spec fn → skip S2

    if _is_inside_trait(gt_lines, fn_line):
        return None                          # can't inject into a trait

    if _is_inside_trait_impl(gt_lines, fn_line):
        return None                          # can't add non-trait methods to impl Trait for Type

    brace_line = _find_opening_brace(gt_lines, fn_line)
    if brace_line is None:
        return None
    fn_end = find_matching_brace(gt_lines, brace_line)
    if fn_end is None:
        return None

    in_impl = _is_inside_impl(gt_lines, fn_line)

    # ---- Parse original fn ----
    header = _collect_header(gt_lines, fn_line, brace_line)
    fn_name = _extract_fn_name(header)
    if fn_name is None:
        return None
    params_str = _extract_params_str(header)
    param_names = [_param_name(p) for p in _split_params(params_str)]
    has_self = any(p in ("self", "&self", "&mut self") for p in param_names)
    ret_type = _extract_return_type(header)
    generics = _extract_generics(header, fn_name)

    # ---- Build __model fn ----
    model_hdr = []
    for li in range(fn_line, brace_line + 1):
        line = gt_lines[li]
        if li == fn_line:
            line = re.sub(
                rf"fn\s+{re.escape(fn_name)}\b",
                f"fn {fn_name}__model", line, count=1,
            )
        model_hdr.append(line)
    model_hdr = _strip_via_clause(model_hdr)

    indent = " " * (len(gt_lines[fn_line]) - len(gt_lines[fn_line].lstrip()))
    body_indent = indent + "    "
    indented_model = "\n".join(
        (body_indent + bl if bl.strip() else bl)
        for bl in model_body.strip().splitlines()
    )

    # ---- Equivalence proof ----
    equiv_op = "<==>" if ret_type.strip() == "bool" else "=~="
    non_self = [p for p in param_names if p not in ("self", "&self", "&mut self")]
    args = ", ".join(non_self)

    if has_self:
        gt_call = f"self.{fn_name}({args})"
        md_call = f"self.{fn_name}__model({args})"
    elif in_impl:
        gt_call = f"Self::{fn_name}({args})"
        md_call = f"Self::{fn_name}__model({args})"
    else:
        gt_call = f"{fn_name}({args})"
        md_call = f"{fn_name}__model({args})"

    block = "\n".join(model_hdr) + "\n"
    block += f"{indented_model}\n{indent}}}\n\n"
    block += (
        f"{indent}proof fn {fn_name}__equiv{generics}({params_str})\n"
        f"{indent}    ensures\n"
        f"{indent}        {gt_call} {equiv_op} {md_call},\n"
        f"{indent}{{\n{indent}}}"
    )

    # ---- Insertion point ----
    if in_impl:
        impl_end = _find_impl_end(gt_lines, fn_line)
        insert_at = impl_end if impl_end is not None else fn_end + 1
    else:
        insert_at = fn_end + 1

    new = gt_lines[:insert_at] + ["", block, ""] + gt_lines[insert_at:]
    return "\n".join(new)


# ---------------------------------------------------------------------------
# Debug dump
# ---------------------------------------------------------------------------

def _dump(debug_dir: Path, name: str, content: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / name).write_text(content)


# ---------------------------------------------------------------------------
# Stage 2 runner
# ---------------------------------------------------------------------------

def stage2_equivalence(
    gt_code: str,
    model_body: str,
    marker_line: int,
    verus_bin: str,
    timeout: int = 120,
    debug_dir: Path | None = None,
    task_id: str = "",
    task_code: str | None = None,
) -> tuple[bool, str]:
    harness = build_equivalence_harness(gt_code, model_body, marker_line, task_code)
    if harness is None:
        return (None, "skipped (target not a single spec fn body)")  # type: ignore[return-value]

    passed, detail = run_verus(harness, verus_bin, timeout)

    if debug_dir is not None and not passed:
        _dump(debug_dir, f"{task_id}_s2_harness.rs", harness)
        _dump(debug_dir, f"{task_id}_s2_harness_output.txt", detail)
        print(f"    [debug] S2 harness → {debug_dir / f'{task_id}_s2_harness.rs'}")

    return passed, detail


# ---------------------------------------------------------------------------
# Single-task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    task_id: str,
    model_body: str,
    benchmark_dir: Path,
    metadata: dict,
    verus_bin: str = VERUS_DEFAULT,
    timeout: int = 120,
    debug_dir: Path | None = None,
) -> EvalResult:
    task_path = benchmark_dir / "tasks" / f"{task_id}.rs"
    gt_path = benchmark_dir / "ground-truth" / f"{task_id}.rs"

    if not task_path.exists() or not gt_path.exists():
        return EvalResult(task_id, False, "file not found", None, "skipped")

    task_code = task_path.read_text()
    gt_code = gt_path.read_text()

    task_lines = task_code.split("\n")
    marker_line = find_marker_line(task_lines)
    if marker_line is None:
        return EvalResult(task_id, False, "marker not found", None, "skipped")

    # ---- Stage 1 ----
    s1_pass, s1_detail, patched = stage1_verify(
        task_code, model_body, verus_bin, timeout,
    )
    if not s1_pass and debug_dir is not None:
        _dump(debug_dir, f"{task_id}_s1_patched.rs",
              patched if patched else "(patch failed)")
        _dump(debug_dir, f"{task_id}_s1_verus_output.txt", s1_detail)
        print(f"    [debug] S1 patched → {debug_dir / f'{task_id}_s1_patched.rs'}")

    if not s1_pass:
        return EvalResult(task_id, False, s1_detail, None, "skipped (stage1 failed)")

    # ---- Stage 2 ----
    s2_pass, s2_detail = stage2_equivalence(
        gt_code, model_body, marker_line,
        verus_bin, timeout,
        debug_dir=debug_dir, task_id=task_id,
        task_code=task_code,
    )
    return EvalResult(task_id, s1_pass, s1_detail, s2_pass, s2_detail)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def load_metadata(benchmark_dir: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for line in open(benchmark_dir / "tasks.jsonl"):
        rec = json.loads(line)
        meta[rec["task_id"]] = rec
    return meta


def evaluate_batch(
    model_dir: Path,
    benchmark_dir: Path,
    verus_bin: str = VERUS_DEFAULT,
    timeout: int = 120,
    debug_dir: Path | None = None,
) -> list[EvalResult]:
    meta = load_metadata(benchmark_dir)
    results: list[EvalResult] = []

    files = sorted(model_dir.glob("*.rs")) + sorted(model_dir.glob("*.body"))
    for f in files:
        task_id = f.stem
        if task_id not in meta:
            print(f"  SKIP {task_id}: not in benchmark metadata")
            continue
        model_body = f.read_text().strip()
        result = evaluate_task(
            task_id, model_body, benchmark_dir, meta[task_id],
            verus_bin, timeout, debug_dir=debug_dir,
        )
        status = "S1=" + ("PASS" if result.stage1_pass else "FAIL")
        if result.stage2_pass is not None:
            status += " S2=" + ("PASS" if result.stage2_pass else "FAIL")
        else:
            status += " S2=SKIP"
        print(f"  {task_id} {status}")
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Selftest helpers
# ---------------------------------------------------------------------------

def extract_gt_body(task_code: str, gt_code: str) -> tuple[str, int] | None:
    """Extract the ground-truth body from the GT file.

    With new markers: extracts text between ``// %%SPEC_START``
    and ``// %%SPEC_END`` in the GT file (trivial, no end-matching).

    With legacy markers: uses end-matching to handle cases where the
    replacement region spans more lines than just the marker.

    Returns ``(body_text, marker_idx)`` or *None* on failure.
    """
    gt_lines = gt_code.split("\n")

    # --- New marker format: extract from GT directly ---
    if _uses_new_markers(gt_lines):
        start = find_marker_line(gt_lines)
        if start is None:
            return None
        end = find_marker_end_line(gt_lines, start)
        if end is None:
            return None
        body_lines = gt_lines[start + 1 : end]
        # Strip leading/trailing blank lines
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        # Strip common indentation so the body is "raw"
        # (patch_task will re-indent to the marker level)
        if body_lines:
            min_indent = min(
                len(l) - len(l.lstrip()) for l in body_lines if l.strip()
            )
            body_lines = [l[min_indent:] if l.strip() else l for l in body_lines]
        return "\n".join(body_lines), start

    # --- Legacy format: end-matching ---
    task_lines = task_code.split("\n")
    marker_idx = find_marker_line(task_lines)
    if marker_idx is None:
        return None

    tail_match = 0
    max_tail = min(len(task_lines) - marker_idx - 1,
                   len(gt_lines) - marker_idx - 1)
    while tail_match < max_tail:
        if task_lines[-(1 + tail_match)] == gt_lines[-(1 + tail_match)]:
            tail_match += 1
        else:
            break

    gt_body_end = len(gt_lines) - tail_match
    if gt_body_end <= marker_idx:
        return None

    return "\n".join(gt_lines[marker_idx:gt_body_end]), marker_idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-dir", type=Path,
        default=Path("benchmarks/VeruSAGE-Bench-Spec-v2"),
    )
    parser.add_argument("--verus-bin", type=str, default=VERUS_DEFAULT)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--debug-dir", type=Path, default=None)

    sub = parser.add_subparsers(dest="command")

    single = sub.add_parser("single", help="Evaluate one task")
    single.add_argument("--task-id", required=True)
    single.add_argument("--model-body", type=Path, required=True)

    batch = sub.add_parser("batch", help="Evaluate a directory of model outputs")
    batch.add_argument("--model-dir", type=Path, required=True)
    batch.add_argument("--output", type=Path, default=None)

    selftest = sub.add_parser("selftest", help="Self-test with ground truth bodies")
    selftest.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.command == "single":
        meta = load_metadata(args.benchmark_dir)
        if args.task_id not in meta:
            print(f"Error: {args.task_id} not in benchmark", file=sys.stderr)
            sys.exit(1)
        body = args.model_body.read_text().strip()
        r = evaluate_task(
            args.task_id, body, args.benchmark_dir, meta[args.task_id],
            args.verus_bin, args.timeout, debug_dir=args.debug_dir,
        )
        print(f"Task:   {r.task_id}")
        print(f"Stage1: {'PASS' if r.stage1_pass else 'FAIL'} — {r.stage1_detail}")
        if r.stage2_pass is not None:
            print(f"Stage2: {'PASS' if r.stage2_pass else 'FAIL'} — {r.stage2_detail}")
        else:
            print(f"Stage2: SKIPPED — {r.stage2_detail}")

    elif args.command == "batch":
        results = evaluate_batch(
            args.model_dir, args.benchmark_dir,
            args.verus_bin, args.timeout, debug_dir=args.debug_dir,
        )
        s1_pass = sum(1 for r in results if r.stage1_pass)
        s2_tested = [r for r in results if r.stage2_pass is not None]
        s2_pass = sum(1 for r in s2_tested if r.stage2_pass)
        print(f"\n{'=' * 60}")
        print(f"Results: {len(results)} tasks evaluated")
        print(f"  Stage 1 (verify):      {s1_pass}/{len(results)} pass")
        print(f"  Stage 2 (equivalence): {s2_pass}/{len(s2_tested)} pass"
              f"  ({len(results) - len(s2_tested)} skipped)")
        if args.output:
            with open(args.output, "w") as f:
                for r in results:
                    f.write(json.dumps({
                        "task_id": r.task_id,
                        "stage1_pass": r.stage1_pass,
                        "stage1_detail": r.stage1_detail,
                        "stage2_pass": r.stage2_pass,
                        "stage2_detail": r.stage2_detail,
                    }) + "\n")
            print(f"  Results → {args.output}")

    elif args.command == "selftest":
        meta = load_metadata(args.benchmark_dir)
        task_ids = list(meta.keys())
        if args.limit:
            task_ids = task_ids[: args.limit]

        print(f"Self-test: {len(task_ids)} tasks with ground truth bodies\n")
        n_s1_pass = n_s1_fail = n_s2_pass = n_s2_fail = n_s2_skip = 0

        for tid in task_ids:
            task_path = args.benchmark_dir / "tasks" / f"{tid}.rs"
            gt_path = args.benchmark_dir / "ground-truth" / f"{tid}.rs"
            if not task_path.exists() or not gt_path.exists():
                print(f"  {tid[:70]:70s} SKIP (files missing)")
                continue

            task_code = task_path.read_text()
            gt_code = gt_path.read_text()

            extraction = extract_gt_body(task_code, gt_code)
            if extraction is None:
                print(f"  {tid[:70]:70s} SKIP (body extraction failed)")
                continue

            gt_body, marker_idx = extraction

            # Stage 1: verify GT directly (bypasses patch_task which only
            # handles single-line replacement; the GT is the correct result)
            s1_pass, s1_detail = run_verus(gt_code, args.verus_bin, args.timeout)

            if not s1_pass and args.debug_dir is not None:
                _dump(args.debug_dir, f"{tid}_s1_gt.rs", gt_code)
                _dump(args.debug_dir, f"{tid}_s1_verus_output.txt", s1_detail)
                print(f"    [debug] S1 GT → {args.debug_dir / f'{tid}_s1_gt.rs'}")

            # Stage 2: equivalence harness (only if S1 passes)
            s2_pass: bool | None = None
            s2_detail = "skipped (stage1 failed)"
            if s1_pass:
                s2_pass, s2_detail = stage2_equivalence(
                    gt_code, gt_body, marker_idx,
                    args.verus_bin, args.timeout,
                    debug_dir=args.debug_dir, task_id=tid,
                    task_code=task_code,
                )

            s1 = "PASS" if s1_pass else "FAIL"
            if s2_pass is True:
                s2 = "PASS"
            elif s2_pass is False:
                s2 = "FAIL"
            else:
                s2 = "SKIP"

            if s1_pass:
                n_s1_pass += 1
            else:
                n_s1_fail += 1
            if s2_pass is True:
                n_s2_pass += 1
            elif s2_pass is False:
                n_s2_fail += 1
            else:
                n_s2_skip += 1

            print(f"  {tid[:70]:70s} S1={s1} S2={s2}")
            if not s1_pass:
                print(f"    S1: {s1_detail[:120]}")
            if s2_pass is False:
                print(f"    S2: {s2_detail[:120]}")

        print(f"\n{'=' * 60}")
        total = n_s1_pass + n_s1_fail
        print(f"Stage 1: {n_s1_pass}/{total} pass, {n_s1_fail} fail")
        print(f"Stage 2: {n_s2_pass} pass, {n_s2_fail} fail, {n_s2_skip} skip")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
