#!/usr/bin/env python3
"""
Plain / vanilla harness for benchmarking LLM-based Verus proof repair.

No agentic framework, no action decomposition — just:
  1. Read the .rs task file
  2. Ask the LLM to fill in the proof
  3. Verify with Verus
  4. On failure, feed the Verus error back and retry up to N times

Usage:
  python plain-harness.py --config config.json                    # fresh run, all tasks
  python plain-harness.py --config config.json --workers 3        # 3 parallel workers
  python plain-harness.py --config config.json --continue-from DIR  # resume
"""

import argparse
import csv
import difflib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import openai

shutdown_event = threading.Event()
_io_lock = threading.Lock()

CSV_FIELDS = [
    "file", "status", "exit_code", "attempts_used", "time_seconds",
    "input_tokens", "output_tokens", "total_tokens",
]

_output_code_dir: Path | None = None
_logs_dir: Path | None = None

SYSTEM_PROMPT = """\
You are a Verus verification expert. Your task is to fix proof bodies \
so that the code verifies successfully with Verus.

Rules:
- Only add/modify ghost proof code: assertions, lemma calls, loop invariants, \
decreases clauses, and reveal/trigger annotations.
- Do NOT change executable code, requires/ensures clauses, or function signatures.
- Do NOT use assume(...).
- Express your changes using the SEARCH/REPLACE format described below.

Response format — every edit must use this exact structure:

```rust
<<<<<<< SEARCH
    // exact lines from the original code
=======
    // replacement lines
>>>>>>> REPLACE
```

Rules for SEARCH/REPLACE blocks:
1. The SEARCH section must be an exact, uniquely-identifiable snippet from the code.
2. Preserve indentation exactly.
3. Multiple SEARCH/REPLACE blocks are allowed.
4. To insert code, include surrounding context in SEARCH.
5. Before the blocks, briefly explain your reasoning.
"""

INITIAL_PROMPT_TEMPLATE = """\
The following Verus code has proof bodies that need to be filled in or fixed \
so that it verifies. Provide SEARCH/REPLACE edits.

```verus
{code}
```
"""

REPAIR_PROMPT_TEMPLATE = """\
The following Verus code has proof bodies that need to be filled in or fixed \
so that it verifies. Provide SEARCH/REPLACE edits.

Original code:
```verus
{original_code}
```

{history}

The most recent code failed Verus verification with the following errors:

```
{errors}
```

Provide SEARCH/REPLACE edits to fix the proof. Your edits will be applied to \
the **Original code**.
**Important**: Analyze the previous attempts carefully before proposing a new repair. \
Avoid repeating failed approaches. Consider why each attempt failed and ensure your \
solution addresses those issues.
"""


# ---------------------------------------------------------------------------
#  Attempt history tracking
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    """Record of a single repair attempt for history accumulation."""
    attempt_id: int
    status: str  # "REJECTED_EDITS", "REJECTED_UNSAFE", "VERUS_FAIL", "LLM_ERROR"
    diff: str  # SEARCH/REPLACE-style diff
    error_text: str  # Verus errors or rejection reason
    code_before: str = ""  # code state before this attempt
    code_after: str | None = None  # code state after (None if edits failed)


def generate_search_replace_diff(old: str, new: str, context: int = 3) -> str:
    """
    Generate a compact SEARCH/REPLACE style diff between old and new code,
    matching the format verusage uses in its history prompts.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines, fromfile="before", tofile="after", n=context
    ))
    if not diff_lines:
        return "(No changes detected)"
    return "".join(diff_lines)


def format_attempt_history(history: list[AttemptRecord], max_attempts: int = 8) -> str:
    """
    Format accumulated attempt history for inclusion in the LLM prompt.
    Mirrors the verusage `format_action_history` style.
    """
    if not history:
        return ""

    entries = history[-max_attempts:]  # keep last N

    lines = ["## Previous Repair Attempts"]
    for rec in entries:
        if rec.status == "VERUS_FAIL":
            status_label = "FAILED (verification errors)"
        elif rec.status == "REJECTED_EDITS":
            status_label = "REJECTED (could not apply edits)"
        elif rec.status == "REJECTED_UNSAFE":
            status_label = "REJECTED (unsafe change)"
        elif rec.status == "LLM_ERROR":
            status_label = "ERROR (LLM call failed)"
        else:
            status_label = rec.status

        lines.append(f"\n### Attempt {rec.attempt_id}: {status_label}")

        #Instead of the following just log the LLM's response on SEARCH/REPLACE section
        # if rec.code_after and rec.code_before:
        #     diff = generate_search_replace_diff(rec.code_before, rec.code_after)
        #     lines.append("- **Changes applied**:")
        #     lines.append("  ```diff")
        #     lines.append(f"  {diff.rstrip()}")
        #     lines.append("  ```")
        # elif rec.diff:
        #     lines.append(f"- **Note**: {rec.diff}")

        lines.append("- **Changes applied**:")
        lines.append("  ```diff")
        lines.append(f"  {rec.diff.rstrip()}")
        lines.append("  ```")

        # Show the resulting error
        if rec.error_text:
            # Truncate very long error output to avoid blowing up context
            err = rec.error_text
            if len(err) > 2000:
                err = err[:2000] + "\n... (truncated)"
            lines.append("- **Verus output**:")
            lines.append("  ```")
            lines.append(f"  {err.rstrip()}")
            lines.append("  ```")

    return "\n".join(lines)


def insert_loop_isolation(code: str) -> str:
    """Insert #[verifier::loop_isolation(false)] after verus! macro if missing."""
    if "loop_isolation(false)" in code:
        return code
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if "verus!" in line.strip():
            lines.insert(i + 1, "#[verifier::loop_isolation(false)]")
            return "\n".join(lines)
    return code  # no verus! macro found


def is_safe_change(original: str, modified: str) -> tuple[bool, str]:
    """
    Two-layer safety check matching verusage:
      1. String-level: no added assume/admit/external_body
      2. AST-level: Lynette 'additions' check (ensures only ghost code changed)
    """
    # Layer 1: string-level checks (fast)
    checks = [
        ("admit()", "admit() calls"),
        ("assume(", "assume(...) calls"),
        ("#[verifier::external_body]", "external_body annotations"),
        ("#[verifier::admit]", "verifier::admit annotations"),
    ]
    for pattern, desc in checks:
        if original.count(pattern) != modified.count(pattern):
            return False, f"{desc} changed ({original.count(pattern)} -> {modified.count(pattern)})"

    # Layer 2: Lynette AST-level check
    lynette_ok, lynette_msg = _lynette_additions_check(original, modified)
    if not lynette_ok:
        return False, lynette_msg

    return True, ""


_LYNETTE_BIN: Path | None = None


def _find_lynette() -> Path | None:
    """Locate the Lynette binary relative to this script."""
    global _LYNETTE_BIN
    if _LYNETTE_BIN is not None:
        return _LYNETTE_BIN

    # Same resolution as verusage/utils.py: <repo>/utils/lynette/source/target/debug/lynette[.exe]
    base = Path(__file__).parent / "utils" / "lynette" / "source" / "target" / "debug"
    for name in ("lynette.exe", "lynette"):
        candidate = base / name
        if candidate.exists():
            _LYNETTE_BIN = candidate
            return _LYNETTE_BIN
    return None


def _lynette_additions_check(original: str, modified: str) -> tuple[bool, str]:
    """
    Run Lynette 'additions' subcommand to verify only ghost/proof code was changed.
    Returns (ok, reason). If Lynette is not found, passes with a warning.
    """
    lynette = _find_lynette()
    if lynette is None:
        return True, "(Lynette not found, skipping AST check)"

    import tempfile as _tf

    orig_f = _tf.NamedTemporaryFile(
        mode="w", delete=False, prefix="plain_orig_", suffix=".rs", encoding="utf-8"
    )
    orig_f.write(original)
    orig_f.close()
    changed_f = _tf.NamedTemporaryFile(
        mode="w", delete=False, prefix="plain_chg_", suffix=".rs", encoding="utf-8"
    )
    changed_f.write(modified)
    changed_f.close()

    try:
        result = subprocess.run(
            [str(lynette), "additions", orig_f.name, changed_f.name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, ""
        stdout = result.stdout.strip()
        if "Disallowed changes detected" in stdout:
            return False, f"Lynette: disallowed changes to executable code ({stdout})"
        return False, f"Lynette check failed (exit {result.returncode}): {stdout or result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return True, "(Lynette timed out, skipping)"
    except Exception as e:
        return True, f"(Lynette error: {e}, skipping)"
    finally:
        try:
            os.unlink(orig_f.name)
        except OSError:
            pass
        try:
            os.unlink(changed_f.name)
        except OSError:
            pass


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def make_client(cfg: dict) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=cfg.get("aoai_api_key", ["dummy"])[0],
        base_url=cfg.get("aoai_api_base", ["http://localhost:8000/v1"])[0],
        max_retries=cfg.get("aoai_max_retries", 3),
    )


def extract_code(response: str) -> str | None:
    """Fallback: pull the first ```verus/rust``` block if no SEARCH/REPLACE found."""
    for lang in ("verus", "rust", ""):
        pattern = rf"```{lang}\s*\n(.*?)```"
        m = re.search(pattern, response, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def parse_search_replace(response: str) -> list[tuple[str, str]]:
    """
    Parse SEARCH/REPLACE blocks from LLM response.
    Returns list of (search_text, replace_text) tuples.
    """
    # Strip outer code fences that wrap the whole response
    cleaned = re.sub(r"```(?:rust|verus)?\s*\n?|```\s*$", "", response, flags=re.MULTILINE)
    pattern = r"<{7,}\s*SEARCH\s*\n(.*?)\n={7,}\s*\n(.*?)\n>{7,}\s*REPLACE"
    ops = []
    for m in re.finditer(pattern, cleaned, re.DOTALL):
        ops.append((m.group(1), m.group(2)))
    return ops

def _normalize_whitespace(text: str) -> str:
    """Strip all leading/trailing whitespace per line for indentation-agnostic matching."""
    return "\n".join(l.strip() for l in text.split("\n"))


def _apply_via_line_range(code: str, search_norm: str, replace: str, code_norm: str) -> str | None:
    """
    Given that *search_norm* is known to be a substring of *code_norm*,
    map back to line numbers in the original *code* and splice in *replace*.
    Returns the updated code, or None if the mapping fails.
    """
    idx = code_norm.find(search_norm)
    if idx < 0:
        return None
    start_line = code_norm[:idx].count("\n")
    end_line = start_line + search_norm.count("\n")
    lines = code.split("\n")
    lines[start_line:end_line + 1] = replace.split("\n")
    return "\n".join(lines)


def apply_edits(original: str, response: str) -> tuple[str | None, int, int]:
    """
    Apply SEARCH/REPLACE edits from *response* to *original*.
    Applies each block sequentially in order; skips blocks whose SEARCH
    text cannot be found in the current code.

    Matching strategy (tried in order):
      1. Exact substring match
      2. Whitespace-normalized match (tabs→spaces, trailing whitespace stripped)

    Returns (result_code, applied_count, total_count).
    result_code is None only when zero edits could be applied AND
    fallback extraction also failed.
    """
    ops = parse_search_replace(response)
    if ops:
        code = original
        applied = 0
        for search, replace in ops:
            # Strategy 1: exact match
            if search in code:
                code = code.replace(search, replace, 1)
                applied += 1
                continue

            # Strategy 2: whitespace-normalized match
            # Handles tab↔space differences and trailing whitespace
            search_norm = _normalize_whitespace(search)
            code_norm = _normalize_whitespace(code)
            if search_norm in code_norm:
                result = _apply_via_line_range(code, search_norm, replace, code_norm)
                if result is not None:
                    code = result
                    applied += 1
                    continue

            # No match — skip this block
        if applied > 0:
            return code, applied, len(ops)
        return None, 0, len(ops)
    else:
        # Fallback: maybe LLM returned full code in a fenced block
        extracted = extract_code(response)
        return (extracted, 1, 1) if extracted else (None, 0, 0)


def run_verus(code: str, verus_path: str, timeout: int = 90) -> tuple[bool, str, int]:
    """
    Write *code* to a temp file, invoke Verus, return (verified, error_text, exit_code).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".rs", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        cmd = [verus_path, "--multiple-errors", "5", "--rlimit", "250", tmp]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        combined = (result.stderr or "") + "\n" + (result.stdout or "")
        verified = result.returncode == 0
        return verified, combined.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return False, "Verus timed out", -1
    except Exception as e:
        return False, f"Verus invocation error: {e}", -2
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def call_llm(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> tuple[str, int, int]:
    """
    Single LLM call. Returns (content, prompt_tokens, completion_tokens).
    Handles thinking-model None content.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    content = resp.choices[0].message.content
    if content is None:
        content = ""
    prompt_tok = resp.usage.prompt_tokens if resp.usage else 0
    compl_tok = resp.usage.completion_tokens if resp.usage else 0
    return content, prompt_tok, compl_tok


def repair_task(
    rs_path: Path,
    client: openai.OpenAI,
    cfg: dict,
    max_repairs: int,
) -> dict:
    """
    Run up to *max_repairs* LLM iterations on a single .rs task.
    Returns a result dict suitable for CSV, or None if shutdown.
    """
    if shutdown_event.is_set():
        return None

    model = cfg["aoai_debug_model"]
    max_tokens = cfg.get("action_max_tokens", 32768)
    temperature = cfg.get("debug_temp", 1.0)
    api_timeout = cfg.get("api_timeout", 300)
    verus_path = cfg["verus_path"]

    original_code = rs_path.read_text(encoding="utf-8")
    current_code = insert_loop_isolation(original_code)
    # Safety baseline = code after loop_isolation (matches verusage behavior)
    safety_baseline = current_code
    total_in = total_out = 0
    start = time.time()
    last_errors = ""
    attempt_history: list[AttemptRecord] = []

    print(f"    [{rs_path.name}] starting ({len(original_code)} chars)...", flush=True)

    for attempt in range(1, max_repairs + 1):
        if shutdown_event.is_set():
            return None

        print(f"    [{rs_path.name}] attempt {attempt}/{max_repairs} — calling LLM...", flush=True)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if attempt == 1:
            messages.append({
                "role": "user",
                "content": INITIAL_PROMPT_TEMPLATE.format(code=current_code),
            })
        else:
            history_text = format_attempt_history(attempt_history)
            messages.append({
                "role": "user",
                "content": REPAIR_PROMPT_TEMPLATE.format(
                    original_code=original_code,
                    history=history_text, errors=last_errors
                ),
            })

        content = None
        p_tok = c_tok = 0
        try:
            content, p_tok, c_tok = call_llm(
                client, model, messages, max_tokens, temperature, api_timeout
            )
            total_in += p_tok
            total_out += c_tok
            print(f"    [{rs_path.name}] attempt {attempt} — LLM done ({p_tok}+{c_tok} tokens), applying edits...", flush=True)
        except openai.BadRequestError as e:
            print(f"    [{rs_path.name}] attempt {attempt} — BadRequest: {e}", flush=True)
            _log_attempt(rs_path.stem, attempt, messages, None, None,
                         None, "", None, f"BadRequest: {e}", 0, 0)
            last_errors = f"LLM BadRequest: {e}"
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="LLM_ERROR",
                diff="LLM BadRequest error, no response is generated", error_text=str(e),
                code_before=current_code,
            ))
            continue
        except Exception as e:
            print(f"    [{rs_path.name}] attempt {attempt} — LLM error: {e}", flush=True)
            _log_attempt(rs_path.stem, attempt, messages, None, None,
                         None, "", None, f"LLM error: {e}", 0, 0)
            last_errors = f"LLM error: {e}"
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="LLM_ERROR",
                diff="LLM call failed, no response is generated", error_text=str(e),
                code_before=current_code,
            ))
            continue
        
        # Extract the raw LLM suggestion for logging (even if we can't apply it)
        _log_llm_response(rs_path.stem, attempt, content)

        # Build suggested_change from all parsed SEARCH/REPLACE blocks for history logging
        sr_blocks = parse_search_replace(content)
        if sr_blocks:
            parts = []
            for search, replace in sr_blocks:
                parts.append(f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE")
            suggested_change = "\n\n".join(parts)
        else:
            # No SEARCH/REPLACE found — use raw content
            suggested_change = "LLM did not provide valid SEARCH/REPLACE blocks. Raw response:\n" + content

        new_code, edits_applied, edits_total = apply_edits(current_code, content)
        if not new_code:
            fail_msg = f"Failed to apply SEARCH/REPLACE edits (0/{edits_total} blocks matched)"
            _log_attempt(rs_path.stem, attempt, messages, content, None,
                         None, "", None, fail_msg, p_tok, c_tok)
            last_errors = fail_msg
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="REJECTED_EDITS",
                diff=suggested_change,
                error_text=fail_msg,
                code_before=current_code,
            ))
            continue
        if edits_applied < edits_total:
            print(f"    [{rs_path.name}] attempt {attempt} — applied {edits_applied}/{edits_total} SEARCH/REPLACE blocks (rest skipped: no match)", flush=True)

        safe, reason = is_safe_change(safety_baseline, new_code)
        if not safe:
            _log_attempt(rs_path.stem, attempt, messages, content, new_code,
                         False, reason, None, "", p_tok, c_tok)
            last_errors = f"Rejected: {reason}"
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="REJECTED_UNSAFE",
                diff=suggested_change, error_text=f"Unsafe change rejected: {reason}",
                code_before=current_code, code_after=new_code,
            ))
            continue  # don't update current_code with unsafe changes

        code_before_verus = current_code  # snapshot for history
        current_code = new_code

        verified, error_text, verus_exit = run_verus(current_code, verus_path)

        # Log the full attempt trace
        _log_attempt(rs_path.stem, attempt, messages, content, current_code,
                     True, "", verified, error_text, p_tok, c_tok)

        if not verified:
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="VERUS_FAIL",
                diff=suggested_change, error_text=error_text,
                code_before=code_before_verus, code_after=current_code,
            ))

        if verified:
            _save_output_code(rs_path.stem, current_code, "verified")
            elapsed = time.time() - start
            return {
                "file": rs_path.name,
                "status": "VERIFIED",
                "exit_code": 0,
                "attempts_used": attempt,
                "time_seconds": round(elapsed, 2),
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            }
        last_errors = error_text

    _save_output_code(rs_path.stem, current_code, "failed")
    elapsed = time.time() - start
    return {
        "file": rs_path.name,
        "status": "FAILED",
        "exit_code": 233,
        "attempts_used": max_repairs,
        "time_seconds": round(elapsed, 2),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
    }


def _save_output_code(stem: str, code: str, status: str):
    """Save the final code for a task to the output directory."""
    if _output_code_dir is None:
        return
    try:
        out = _output_code_dir / f"{stem}_{status}.rs"
        out.write_text(code, encoding="utf-8")
    except Exception:
        pass


def _get_task_log_dir(stem: str) -> Path | None:
    """Get/create a per-task log directory: <output>/logs/<task_stem>/"""
    if _logs_dir is None:
        return None
    d = _logs_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log_attempt(
    stem: str,
    attempt: int,
    messages: list[dict],
    llm_response: str | None,
    edit_result: str | None,
    safety_ok: bool | None,
    safety_reason: str,
    verus_verified: bool | None,
    verus_errors: str,
    tokens_in: int,
    tokens_out: int,
):
    """
    Write one attempt's log to <output>/logs/<task>/attempt-N.md

    Three clean sections:
      <Input Prompt>   — complete prompt sent to the LLM
      <LLM Response>   — raw LLM output
      <Result>         — what happened (edits / safety / verus)
    """
    d = _get_task_log_dir(stem)
    if d is None:
        return
    try:
        parts: list[str] = []

        # ---- <Input Prompt> ----
        parts.append("<Input Prompt>")
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}]")
            parts.append(content)
            parts.append("")
        parts.append("</Input Prompt>")
        parts.append("")

        # ---- <LLM Response> ----
        parts.append("<LLM Response>")
        parts.append(llm_response if llm_response is not None else "(no response / error)")
        parts.append("</LLM Response>")
        parts.append("")

        # ---- <Result> ----
        parts.append("<Result>")
        parts.append(f"Tokens: in={tokens_in}, out={tokens_out}")

        if edit_result is None:
            parts.append("Edits: FAILED to apply")
        else:
            parts.append(f"Edits: applied ({len(edit_result)} chars)")

        if safety_ok is not None:
            if safety_ok:
                parts.append("Safety: PASSED")
            else:
                parts.append(f"Safety: FAILED — {safety_reason}")

        if verus_verified is not None:
            if verus_verified:
                parts.append("Verus: VERIFIED")
            else:
                parts.append("Verus: FAILED")
                if verus_errors:
                    parts.append(verus_errors)
        elif verus_errors:
            parts.append(f"Note: {verus_errors}")

        parts.append("</Result>")

        (d / f"attempt-{attempt}.md").write_text("\n".join(parts), encoding="utf-8")

    except Exception:
        pass

def _log_llm_response(stem: str, attempt: int, llm_response: str | None):
    """
    Save the raw LLM response to <output>/logs/<task>/response-N.txt
    for cross-checking / debugging.
    """
    d = _get_task_log_dir(stem)
    if d is None:
        return
    try:
        text = llm_response if llm_response is not None else "(no response)"
        (d / f"response-{attempt}.txt").write_text(text, encoding="utf-8")
    except Exception:
        pass

def run_batch(
    tasks_dir: Path,
    cfg: dict,
    output_dir: Path,
    max_repairs: int,
    num_workers: int,
    continue_from: Path | None,
):
    global _output_code_dir
    all_files = sorted(tasks_dir.glob("*.rs"))
    if not all_files:
        print(f"No .rs files in {tasks_dir}")
        return

    done_files: set[str] = set()
    results: list[dict] = []

    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    code_dir = output_dir / "outputs"
    code_dir.mkdir(parents=True, exist_ok=True)
    _output_code_dir = code_dir

    global _logs_dir
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _logs_dir = logs_dir

    if continue_from and csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done_files.add(row["file"])
                results.append(row)
        print(f"Loaded {len(done_files)} previous results")

    pending = [f for f in all_files if f.name not in done_files]
    print(f"Tasks: {len(pending)} pending / {len(all_files)} total")
    print(f"Workers: {num_workers}, Max repairs: {max_repairs}")
    print(f"Output: {output_dir}\n")

    write_header = not csv_path.exists() or not continue_from
    if write_header:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    with open(summary_path, "a" if continue_from else "w") as sf:
        sf.write(f"{'Resuming' if continue_from else 'Starting'} plain harness "
                 f"at {time.strftime('%Y%m%d-%H%M%S')}\n")
        sf.write(f"Model: {cfg['aoai_debug_model']}\n")
        sf.write(f"Workers: {num_workers}, Repairs: {max_repairs}\n")
        sf.write("=" * 80 + "\n\n")

    client = make_client(cfg)

    orig_handler = signal.getsignal(signal.SIGINT)

    def _sigint(signum, frame):
        if not shutdown_event.is_set():
            print("\n>>> Ctrl+C — finishing in-flight tasks, then stopping...", flush=True)
            shutdown_event.set()

    signal.signal(signal.SIGINT, _sigint)

    done_count = [0]
    verified_count = [0]
    batch_t0 = time.time()

    def record(result: dict):
        with _io_lock:
            results.append(result)
            done_count[0] += 1
            if result["status"] == "VERIFIED":
                verified_count[0] += 1
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                w.writerow({k: result.get(k, "") for k in CSV_FIELDS})
            d = done_count[0]
            t = len(pending)
            elapsed = time.time() - batch_t0
            rate = d / elapsed if elapsed > 0 else 0
            eta = (t - d) / rate if rate > 0 else 0
            print(
                f"  [{d}/{t}] {result['file']}: {result['status']} "
                f"(attempts={result['attempts_used']}, "
                f"time={result['time_seconds']}s, "
                f"tokens={result['total_tokens']}, "
                f"verified={verified_count[0]}, "
                f"ETA={eta/3600:.1f}h)",
                flush=True,
            )

    try:
        if num_workers <= 1:
            for rs in pending:
                if shutdown_event.is_set():
                    break
                r = repair_task(rs, client, cfg, max_repairs)
                if r is None:
                    break
                record(r)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futs = {}
                for rs in pending:
                    if shutdown_event.is_set():
                        break
                    fut = pool.submit(repair_task, rs, client, cfg, max_repairs)
                    futs[fut] = rs
                for fut in as_completed(futs):
                    if shutdown_event.is_set():
                        for f in futs:
                            f.cancel()
                        break
                    r = fut.result()
                    if r is None:
                        continue
                    record(r)
    finally:
        signal.signal(signal.SIGINT, orig_handler)

    wall = time.time() - batch_t0
    v = sum(1 for r in results if r.get("status") == "VERIFIED")
    total = len(results)
    tot_in = sum(int(r.get("input_tokens", 0)) for r in results)
    tot_out = sum(int(r.get("output_tokens", 0)) for r in results)

    with open(summary_path, "a") as sf:
        sf.write(f"\nFINAL: {v}/{total} verified "
                 f"({100*v/total:.1f}% | wall {wall:.0f}s)\n")
        sf.write(f"Tokens: {tot_in + tot_out:,} "
                 f"(in={tot_in:,}, out={tot_out:,})\n")

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "model": cfg["aoai_debug_model"],
            "max_repairs": max_repairs,
            "workers": num_workers,
            "wall_seconds": round(wall, 2),
            "total": len(all_files),
            "verified": v,
            "failed": total - v,
            "tokens_in": tot_in,
            "tokens_out": tot_out,
            "tasks": [{k: r.get(k, "") for k in CSV_FIELDS} for r in results],
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done: {v}/{total} verified ({100*v/total:.1f}%)")
    print(f"Wall: {wall:.0f}s ({wall/3600:.1f}h)")
    print(f"Tokens: {tot_in+tot_out:,}")
    print(f"Results: {csv_path}")
    print(f"{'='*60}")

    if shutdown_event.is_set():
        print(f"\n>>> Interrupted. Re-run with --continue-from {output_dir} to resume.")


def main():
    p = argparse.ArgumentParser(
        description="Plain vanilla LLM harness for VeruSAGE-Bench"
    )
    p.add_argument("--config", required=True, help="Path to JSON config file")
    p.add_argument(
        "--tasks-dir",
        default="../benchmarks/VeruSAGE-Bench/tasks",
        help="Directory containing .rs task files",
    )
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--repairs", type=int, default=5, help="Max repair attempts per task")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p.add_argument("--continue-from", default=None, help="Resume from this output dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    tasks_dir = Path(args.tasks_dir)

    if not tasks_dir.is_dir():
        print(f"Error: tasks dir not found: {tasks_dir}")
        sys.exit(1)

    if args.continue_from:
        output_dir = Path(args.continue_from)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path(f"plain-results-{ts}")

    run_batch(
        tasks_dir=tasks_dir,
        cfg=cfg,
        output_dir=output_dir,
        max_repairs=args.repairs,
        num_workers=args.workers,
        continue_from=Path(args.continue_from) if args.continue_from else None,
    )


if __name__ == "__main__":
    main()
