"""
Stage 1: LLM-based spec body generation for Verus verification tasks.

Workflow:
  1. Read a task file (spec body replaced with `arbitrary() // TODO: fill in the spec body`)
  2. Prompt the LLM to provide the spec body
  3. Safety-check the response (string-level + Lynette compare)
  4. Patch the spec body into the task file
  5. Run Verus to verify
  6. If verified → return the model spec for stage 2
  7. If not verified → return failure
"""

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

from utility import (
    load_config, make_client, call_llm, run_verus,
    is_safe_content, lynette_compare,
)


# ---------------------------------------------------------------------------
#  Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Rust based Verus verification expert. Your task is to fill in the body of a \
spec function so that the code verifies successfully with Verus.

Rules:
- Provide ONLY the spec function body — the expression(s) that go inside the function braces.
- Do NOT include the function signature, curly braces, or any other surrounding code.
- Do NOT use assume(...), admit(), #[verifier::external_body], or #[verifier::admit].
- The body must be a valid Verus spec expression.

Respond with exactly ONE fenced block labeled `spec_body`:

```spec_body
// the spec function body here
```
"""

INITIAL_PROMPT_TEMPLATE = """\
The following Verus code has a spec function whose body is replaced with \
`arbitrary()` and needs to be filled in so the code verifies.

```verus
{code}
```

Provide the spec function body.
"""

REPAIR_PROMPT_TEMPLATE = """\
The following Verus code has a spec function whose body is replaced with \
`arbitrary()` and needs to be filled in so the code verifies.

```verus
{code}
```

{history}

The most recent attempt failed Verus verification with the following errors:

```
{errors}
```

Provide a corrected spec function body.
**Important**: Analyze the previous attempts carefully before proposing a fix. \
Avoid repeating failed approaches.
"""


# ---------------------------------------------------------------------------
#  Data types
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    attempt_id: int
    status: str  # "REJECTED_PARSE", "REJECTED_UNSAFE", "VERUS_FAIL", "LLM_ERROR"
    response: str
    error_text: str


# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

def _log_attempt(
    attempt: int,
    log_dir: Path | None,
    *,
    messages: list[dict] | None = None,
    llm_response: str | None = None,
    parsed_spec_body: str | None = None,
    safety_ok: bool | None = None,
    safety_reason: str = "",
    patched_code: str | None = None,
    verus_verified: bool | None = None,
    verus_errors: str = "",
    verus_exit_code: int | None = None,
    status: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
):
    if log_dir is None:
        return
    path = log_dir / f"attempt-{attempt}.md"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Stage 1 — Attempt {attempt}\n\n")
            f.write(f"**Status**: {status}\n")
            f.write(f"**Tokens**: prompt={tokens_in}, completion={tokens_out}\n\n")

            f.write("---\n## 1. Input Prompt\n\n")
            if messages:
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    f.write(f"### [{role}]\n```\n{content}\n```\n\n")
            else:
                f.write("(no prompt recorded)\n\n")

            f.write("---\n## 2. LLM Response\n\n")
            if llm_response is not None:
                f.write(f"```\n{llm_response}\n```\n\n")
            else:
                f.write("(no response — LLM call failed)\n\n")

            f.write("---\n## 3. Parsed Spec Body\n\n")
            if parsed_spec_body is not None:
                f.write(f"```rust\n{parsed_spec_body}\n```\n\n")
            else:
                f.write("(parsing failed)\n\n")

            f.write("---\n## 4. Safety Check\n\n")
            if safety_ok is None:
                f.write("(not reached)\n\n")
            elif safety_ok:
                f.write("PASSED\n\n")
            else:
                f.write(f"FAILED: {safety_reason}\n\n")

            f.write("---\n## 5. Patched Code\n\n")
            if patched_code is not None:
                f.write(f"```rust\n{patched_code}\n```\n\n")
            else:
                f.write("(not reached)\n\n")

            f.write("---\n## 6. Verus Result\n\n")
            if verus_verified is None:
                f.write("(not reached)\n\n")
            elif verus_verified:
                f.write(f"VERIFIED (exit code {verus_exit_code})\n\n")
            else:
                f.write(f"FAILED (exit code {verus_exit_code})\n\n")
                f.write(f"```\n{verus_errors}\n```\n\n")

            f.write("---\n## 7. Summary\n\n")
            f.write(f"{status}\n")
    except Exception as e:
        print(f"  [logging] Failed to write {path}: {e}", flush=True)


# ---------------------------------------------------------------------------
#  Spec body parsing
# ---------------------------------------------------------------------------

# Regex to find the target spec function whose body is `arbitrary() // TODO: ...`
TODO_PATTERN = re.compile(
    r"(arbitrary\(\)\s*//\s*TODO:\s*fill in the spec body)",
    re.DOTALL,
)


def parse_spec_body(response: str) -> str | None:
    """Extract spec_body from a fenced block in the LLM response."""
    m = re.search(r"```spec_body\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: try generic verus/rust fences
    for lang in ("verus", "rust", ""):
        m = re.search(rf"```{lang}\s*\n(.*?)```", response, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def patch_spec_body(task_code: str, spec_body: str) -> str | None:
    """Replace the `arbitrary() // TODO: ...` marker with the spec body."""
    if not TODO_PATTERN.search(task_code):
        return None
    # Use a lambda to avoid re.sub interpreting backslashes in spec_body
    # as backreferences (e.g. \1 would silently corrupt the output).
    return TODO_PATTERN.sub(lambda _m: spec_body, task_code, count=1)


# ---------------------------------------------------------------------------
#  Safety checks (spec-specific)
# ---------------------------------------------------------------------------

def is_safe_spec_change(original_code: str, patched_code: str, spec_body: str) -> tuple[bool, str]:
    """
    Two-layer safety check for spec body changes:
      1. String-level: no disallowed constructs in the spec body
      2. Lynette compare (no flags): strips all ghost/spec code and compares
         only exec code, ensuring the model didn't tamper with executable code.

    NOTE: We intentionally do NOT pass --spec to lynette compare, because the
    spec body IS expected to change (that's the whole point of stage 1).
    We also do NOT use `lynette additions` here because it only supports the
    proof-filling pipeline (unimplemented!() placeholders), not spec-body filling.
    """
    # Layer 1: string-level
    safe, reason = is_safe_content(spec_body)
    if not safe:
        return False, reason

    # Layer 2: Lynette compare (no flags = compare exec code only)
    # Strips all ghost/spec code; if only the spec body changed, this passes.
    # If the model also changed exec code, this catches it.
    ok, msg = lynette_compare(original_code, patched_code)
    if not ok:
        return False, msg

    return True, ""


# ---------------------------------------------------------------------------
#  History formatting
# ---------------------------------------------------------------------------

def format_attempt_history(history: list[AttemptRecord], max_attempts: int = 8) -> str:
    if not history:
        return ""
    entries = history[-max_attempts:]
    lines = ["## Previous Attempts"]
    for rec in entries:
        if rec.status == "VERUS_FAIL":
            label = "FAILED (verification errors)"
        elif rec.status == "REJECTED_PARSE":
            label = "REJECTED (could not parse response)"
        elif rec.status == "REJECTED_UNSAFE":
            label = "REJECTED (unsafe change)"
        elif rec.status == "LLM_ERROR":
            label = "ERROR (LLM call failed)"
        else:
            label = rec.status
        lines.append(f"\n### Attempt {rec.attempt_id}: {label}")
        lines.append("- **LLM response**:")
        lines.append("  ```")
        lines.append(f"  {rec.response.rstrip()}")
        lines.append("  ```")
        if rec.error_text:
            err = rec.error_text[:2000] + ("\n... (truncated)" if len(rec.error_text) > 2000 else "")
            lines.append("- **Error output**:")
            lines.append("  ```")
            lines.append(f"  {err.rstrip()}")
            lines.append("  ```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Stage 1 core
# ---------------------------------------------------------------------------

def run_stage_1(
    task_code: str,
    verus_path: str,
    client,
    cfg: dict,
    max_attempts: int = 5,
    log_dir: Path | None = None,
) -> tuple[bool, str | None]:
    """
    Run up to max_attempts LLM iterations to fill in the spec body.
    Returns (success, model_spec_body_or_None).
    """
    model = cfg["aoai_debug_model"]
    max_tokens = cfg.get("action_max_tokens", 32768)
    temperature = cfg.get("debug_temp", 1.0)
    api_timeout = cfg.get("api_timeout", 300)

    attempt_history: list[AttemptRecord] = []
    total_in = total_out = 0
    start = time.time()
    last_errors = ""

    for attempt in range(1, max_attempts + 1):
        print(f"  [Stage 1] Attempt {attempt}/{max_attempts} — calling LLM...", flush=True)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if attempt == 1:
            messages.append({
                "role": "user",
                "content": INITIAL_PROMPT_TEMPLATE.format(code=task_code),
            })
        else:
            history_text = format_attempt_history(attempt_history)
            messages.append({
                "role": "user",
                "content": REPAIR_PROMPT_TEMPLATE.format(
                    code=task_code,
                    history=history_text,
                    errors=last_errors,
                ),
            })

        # --- Call LLM ---
        try:
            content, p_tok, c_tok = call_llm(
                client, model, messages, max_tokens, temperature, api_timeout
            )
            total_in += p_tok
            total_out += c_tok
            print(f"  [Stage 1] Attempt {attempt} — LLM done ({p_tok}+{c_tok} tokens)", flush=True)
        except Exception as e:
            print(f"  [Stage 1] Attempt {attempt} — LLM error: {e}", flush=True)
            last_errors = f"LLM error: {e}"
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="LLM_ERROR",
                response="(LLM call failed)", error_text=str(e),
            ))
            _log_attempt(attempt, log_dir, messages=messages, llm_response=None,
                         status="LLM_ERROR", verus_errors=str(e))
            continue

        # --- Parse response ---
        spec_body = parse_spec_body(content)
        if spec_body is None:
            fail_msg = "Could not extract spec_body from LLM response"
            print(f"  [Stage 1] Attempt {attempt} — {fail_msg}", flush=True)
            last_errors = fail_msg
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="REJECTED_PARSE",
                response=content, error_text=fail_msg,
            ))
            _log_attempt(attempt, log_dir, messages=messages, llm_response=content,
                         status="REJECTED_PARSE — " + fail_msg,
                         tokens_in=p_tok, tokens_out=c_tok)
            continue

        # --- Patch ---
        patched = patch_spec_body(task_code, spec_body)
        if patched is None:
            fail_msg = "Could not find TODO marker in task code to patch"
            print(f"  [Stage 1] Attempt {attempt} — {fail_msg}", flush=True)
            last_errors = fail_msg
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="REJECTED_PARSE",
                response=content, error_text=fail_msg,
            ))
            _log_attempt(attempt, log_dir, messages=messages, llm_response=content,
                         parsed_spec_body=spec_body,
                         status="REJECTED_PARSE — " + fail_msg,
                         tokens_in=p_tok, tokens_out=c_tok)
            continue

        # --- Safety check ---
        safe, reason = is_safe_spec_change(task_code, patched, spec_body)
        if not safe:
            print(f"  [Stage 1] Attempt {attempt} — Rejected (unsafe): {reason}", flush=True)
            last_errors = f"Rejected: {reason}"
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="REJECTED_UNSAFE",
                response=content, error_text=f"Unsafe: {reason}",
            ))
            _log_attempt(attempt, log_dir, messages=messages, llm_response=content,
                         parsed_spec_body=spec_body,
                         safety_ok=False, safety_reason=reason,
                         patched_code=patched,
                         status="REJECTED_UNSAFE — " + reason,
                         tokens_in=p_tok, tokens_out=c_tok)
            continue

        # --- Verify ---
        verified, error_text, exit_code = run_verus(patched, verus_path)
        if verified:
            elapsed = time.time() - start
            print(f"  [Stage 1] Attempt {attempt} — VERIFIED ({elapsed:.1f}s, {total_in+total_out} tokens)", flush=True)
            _log_attempt(attempt, log_dir, messages=messages, llm_response=content,
                         parsed_spec_body=spec_body,
                         safety_ok=True, patched_code=patched,
                         verus_verified=True, verus_exit_code=exit_code,
                         status="VERIFIED",
                         tokens_in=p_tok, tokens_out=c_tok)
            return True, spec_body

        print(f"  [Stage 1] Attempt {attempt} — Verus failed (exit {exit_code})", flush=True)
        last_errors = error_text
        attempt_history.append(AttemptRecord(
            attempt_id=attempt, status="VERUS_FAIL",
            response=content, error_text=error_text,
        ))
        _log_attempt(attempt, log_dir, messages=messages, llm_response=content,
                     parsed_spec_body=spec_body,
                     safety_ok=True, patched_code=patched,
                     verus_verified=False, verus_errors=error_text, verus_exit_code=exit_code,
                     status="VERUS_FAIL",
                     tokens_in=p_tok, tokens_out=c_tok)

    elapsed = time.time() - start
    print(f"  [Stage 1] All {max_attempts} attempts exhausted ({elapsed:.1f}s, {total_in+total_out} tokens)", flush=True)
    return False, None


# ---------------------------------------------------------------------------
#  Main (standalone mode)
# ---------------------------------------------------------------------------

def main(args):
    cfg = load_config(args.config)
    verus_path = args.verus_binary_path or cfg.get("verus_path", "verus")
    max_attempts = args.max_attempts

    task_code = Path(args.task_file).read_text(encoding="utf-8")

    # Set up logging
    task_name = Path(args.task_file).stem
    output_dir = Path(args.output_dir) if args.output_dir else Path("output")
    log_dir = output_dir / "logs" / task_name / "stage-1"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {log_dir}")

    client = make_client(cfg)
    success, spec_body = run_stage_1(
        task_code=task_code,
        verus_path=verus_path,
        client=client,
        cfg=cfg,
        max_attempts=max_attempts,
        log_dir=log_dir,
    )

    if success:
        print(f"Stage 1 PASSED — spec body:\n{spec_body}")
        # Write the spec body to a file for stage 2
        spec_out = output_dir / task_name / "model_spec.txt"
        spec_out.parent.mkdir(parents=True, exist_ok=True)
        spec_out.write_text(spec_body, encoding="utf-8")
        print(f"Spec body written to: {spec_out}")
    else:
        print("Stage 1 FAILED — could not generate a verified spec body.")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: LLM spec body generation + Verus verification")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--task-file", type=str, required=True, help="Path to the .rs task file")
    parser.add_argument("--verus-binary-path", type=str, default=None, help="Path to Verus binary (overrides config)")
    parser.add_argument("--max-attempts", type=int, default=5, help="Max LLM attempts")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for logs and outputs")
    args = parser.parse_args()
    main(args)
