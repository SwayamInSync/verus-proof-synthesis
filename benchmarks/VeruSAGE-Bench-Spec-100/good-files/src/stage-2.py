import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

from utility import load_config, make_client, call_llm, run_verus, is_safe_content


# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------


def init_logging(output_dir: Path, task_name: str) -> Path:
    """Create and return a per-task log directory: <output_dir>/logs/<task_name>/"""
    d = output_dir / "logs" / task_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log_attempt(
    attempt: int,
    log_dir: Path | None,
    *,
    messages: list[dict] | None = None,
    llm_response: str | None = None,
    parsed_helper_lemmas: str | None = None,
    parsed_proof_body: str | None = None,
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
    """
    Write one attempt's full log to <logs_dir>/attempt-<N>.md

    Sections:
      1. Input Prompt   — complete prompt sent to the LLM
      2. LLM Response   — raw LLM output
      3. Parsing        — extracted helper_lemmas and proof_body
      4. Safety Check   — pass/fail + reason
      5. Patched Code   — the full code sent to Verus (if applicable)
      6. Verus Result   — verified / errors
      7. Status Summary — final outcome of this attempt
    """
    if log_dir is None:
        return
    path = log_dir / f"attempt-{attempt}.md"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Attempt {attempt}\n\n")
            f.write(f"**Status**: {status}\n")
            f.write(f"**Tokens**: prompt={tokens_in}, completion={tokens_out}\n\n")

            # Section 1: Input prompt
            f.write("---\n## 1. Input Prompt\n\n")
            if messages:
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    f.write(f"### [{role}]\n```\n{content}\n```\n\n")
            else:
                f.write("(no prompt recorded)\n\n")

            # Section 2: Raw LLM response
            f.write("---\n## 2. LLM Response\n\n")
            if llm_response is not None:
                f.write(f"```\n{llm_response}\n```\n\n")
            else:
                f.write("(no response — LLM call failed)\n\n")

            # Section 3: Parsing results
            f.write("---\n## 3. Parsed Content\n\n")
            if parsed_proof_body is not None:
                f.write("### Helper Lemmas\n```rust\n")
                f.write(parsed_helper_lemmas or "(empty)")
                f.write("\n```\n\n### Proof Body\n```rust\n")
                f.write(parsed_proof_body)
                f.write("\n```\n\n")
            else:
                f.write("(parsing failed — could not extract proof_body)\n\n")

            # Section 4: Safety check
            f.write("---\n## 4. Safety Check\n\n")
            if safety_ok is None:
                f.write("(not reached)\n\n")
            elif safety_ok:
                f.write("PASSED\n\n")
            else:
                f.write(f"FAILED: {safety_reason}\n\n")

            # Section 5: Patched code
            f.write("---\n## 5. Patched Code\n\n")
            if patched_code is not None:
                f.write(f"```rust\n{patched_code}\n```\n\n")
            else:
                f.write("(not reached — patching not attempted)\n\n")

            # Section 6: Verus result
            f.write("---\n## 6. Verus Result\n\n")
            if verus_verified is None:
                f.write("(not reached — Verus not invoked)\n\n")
            elif verus_verified:
                f.write(f"VERIFIED (exit code {verus_exit_code})\n\n")
            else:
                f.write(f"FAILED (exit code {verus_exit_code})\n\n")
                f.write(f"```\n{verus_errors}\n```\n\n")

            # Section 7: Summary
            f.write("---\n## 7. Summary\n\n")
            f.write(f"{status}\n")
    except Exception as e:
        print(f"  [logging] Failed to write {path}: {e}", flush=True)


# ---------------------------------------------------------------------------
#  Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Verus verification expert. Your task is to complete the proof body \
of an equivalence lemma and optionally write helper lemmas so that the code \
verifies successfully with Verus.

Rules:
- Only provide ghost proof code: assertions, lemma calls, loop invariants, \
decreases clauses, and reveal/trigger annotations.
- Do NOT change executable code, requires/ensures clauses, or function signatures.
- Do NOT use assume(...), admit(), #[verifier::external_body], or #[verifier::admit].
- Respond with exactly TWO fenced blocks labeled `helper_lemmas` and `proof_body`.

Response format — use this exact structure:

```helper_lemmas
// any helper lemmas needed (leave empty if none)
```

```proof_body
// the body of the equivalence_lemma proof fn
```
"""

INITIAL_PROMPT_TEMPLATE = """\
The following Verus code contains an equivalence lemma whose proof body \
and helper lemmas need to be filled in so the code verifies.

```verus
{code}
```

Provide the content for the `helper_lemmas` and `proof_body` sections.
"""

REPAIR_PROMPT_TEMPLATE = """\
The following Verus code contains an equivalence lemma whose proof body \
and helper lemmas need to be filled in so the code verifies.

```verus
{code}
```

{history}

The most recent attempt failed Verus verification with the following errors:

```
{errors}
```

Provide corrected content for the `helper_lemmas` and `proof_body` sections.
**Important**: Analyze the previous attempts carefully before proposing a fix. \
Avoid repeating failed approaches.
"""


# ---------------------------------------------------------------------------
#  Data types
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    """Record of a single repair attempt for history accumulation."""
    attempt_id: int
    status: str  # "REJECTED_EDITS", "REJECTED_UNSAFE", "VERUS_FAIL", "LLM_ERROR"
    response: str  # LLM response
    error_text: str  # Verus errors or rejection reason


# ---------------------------------------------------------------------------
#  Marker patching
# ---------------------------------------------------------------------------

def _patch_between_markers(content: str, start_tag: str, end_tag: str, replacement: str) -> str:
    """Replace everything between start_tag and end_tag markers with replacement."""
    pattern = rf"(//%%\s*{re.escape(start_tag)}\n).*?(//%%\s*{re.escape(end_tag)})"
    patched = re.sub(pattern, r"\1" + replacement.rstrip("\n") + "\n" + r"\2", content, flags=re.DOTALL)
    return patched


def patch_spec(model_spec: str, eval_harness_content: str) -> str:
    return _patch_between_markers(eval_harness_content, "MODEL_SPEC_START", "MODEL_SPEC_END", model_spec)


def patch_helper_lemmas(helper_lemmas: str, eval_harness_content: str) -> str:
    return _patch_between_markers(eval_harness_content, "HELPER_LEMMA_START", "HELPER_LEMMA_END", helper_lemmas)


def patch_proof_body(proof_body: str, eval_harness_content: str) -> str:
    return _patch_between_markers(eval_harness_content, "PROOF_BODY_START", "PROOF_BODY_END", proof_body)


# ---------------------------------------------------------------------------
#  LLM response parsing
# ---------------------------------------------------------------------------

def parse_llm_response(response: str) -> tuple[str | None, str | None]:
    """
    Extract helper_lemmas and proof_body from fenced blocks in the LLM response.
    Returns (helper_lemmas, proof_body). Either may be None if not found.
    """
    helper_lemmas = None
    proof_body = None

    helper_match = re.search(r"```helper_lemmas\s*\n(.*?)```", response, re.DOTALL)
    if helper_match:
        helper_lemmas = helper_match.group(1).strip()

    proof_match = re.search(r"```proof_body\s*\n(.*?)```", response, re.DOTALL)
    if proof_match:
        proof_body = proof_match.group(1).strip()

    # Fallback: try generic verus/rust fences if structured blocks not found
    if proof_body is None and helper_lemmas is None:
        for lang in ("verus", "rust", ""):
            m = re.search(rf"```{lang}\s*\n(.*?)```", response, re.DOTALL)
            if m:
                proof_body = m.group(1).strip()
                helper_lemmas = ""
                break

    return helper_lemmas, proof_body


# ---------------------------------------------------------------------------
#  History formatting
# ---------------------------------------------------------------------------

def format_attempt_history(history: list[AttemptRecord], max_attempts: int = 8) -> str:
    if not history:
        return ""

    entries = history[-max_attempts:]
    lines = ["## Previous Repair Attempts"]

    for rec in entries:
        if rec.status == "VERUS_FAIL":
            status_label = "FAILED (verification errors)"
        elif rec.status == "REJECTED_EDITS":
            status_label = "REJECTED (could not parse response)"
        elif rec.status == "REJECTED_UNSAFE":
            status_label = "REJECTED (unsafe change)"
        elif rec.status == "LLM_ERROR":
            status_label = "ERROR (LLM call failed)"
        else:
            status_label = rec.status

        lines.append(f"\n### Attempt {rec.attempt_id}: {status_label}")
        lines.append("- **LLM response**:")
        lines.append("  ```")
        lines.append(f"  {rec.response.rstrip()}")
        lines.append("  ```")

        if rec.error_text:
            err = rec.error_text
            if len(err) > 2000:
                err = err[:2000] + "\n... (truncated)"
            lines.append("- **Error output**:")
            lines.append("  ```")
            lines.append(f"  {err.rstrip()}")
            lines.append("  ```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Repair loop
# ---------------------------------------------------------------------------

def run_equivalence_repair_loop(
    patched_harness: str,
    verus_path: str,
    client,
    cfg: dict,
    last_errors: str,
    max_repairs: int = 5,
    log_dir: Path | None = None,
) -> bool:
    """
    Run up to max_repairs LLM iterations to fill in HELPER_LEMMA and PROOF_BODY
    markers so the equivalence lemma verifies.
    Returns True if verification succeeds.
    """
    model = cfg["aoai_debug_model"]
    max_tokens = cfg.get("action_max_tokens", 32768)
    temperature = cfg.get("debug_temp", 1.0)
    api_timeout = cfg.get("api_timeout", 300)

    attempt_history: list[AttemptRecord] = []
    total_in = total_out = 0
    start = time.time()

    for attempt in range(1, max_repairs + 1):
        print(f"  Repair attempt {attempt}/{max_repairs} — calling LLM...", flush=True)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if attempt == 1:
            messages.append({
                "role": "user",
                "content": INITIAL_PROMPT_TEMPLATE.format(code=patched_harness),
            })
        else:
            history_text = format_attempt_history(attempt_history)
            messages.append({
                "role": "user",
                "content": REPAIR_PROMPT_TEMPLATE.format(
                    code=patched_harness,
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
            print(f"  Attempt {attempt} — LLM done ({p_tok}+{c_tok} tokens)", flush=True)
        except Exception as e:
            print(f"  Attempt {attempt} — LLM error: {e}", flush=True)
            last_errors = f"LLM error: {e}"
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="LLM_ERROR",
                response="(LLM call failed)", error_text=str(e),
            ))
            _log_attempt(
                attempt, log_dir, messages=messages, llm_response=None,
                status="LLM_ERROR",
                verus_errors=str(e),
            )
            continue

        # --- Parse response ---
        helper_lemmas, proof_body = parse_llm_response(content)
        if proof_body is None:
            fail_msg = "Could not extract proof_body from LLM response"
            print(f"  Attempt {attempt} — {fail_msg}", flush=True)
            last_errors = fail_msg
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="REJECTED_EDITS",
                response=content, error_text=fail_msg,
            ))
            _log_attempt(
                attempt, log_dir, messages=messages, llm_response=content,
                status="REJECTED_EDITS — " + fail_msg,
                tokens_in=p_tok, tokens_out=c_tok,
            )
            continue

        if helper_lemmas is None:
            helper_lemmas = ""

        # --- Safety check ---
        combined_content = helper_lemmas + "\n" + proof_body
        safe, reason = is_safe_content(combined_content)
        if not safe:
            print(f"  Attempt {attempt} — Rejected (unsafe): {reason}", flush=True)
            last_errors = f"Rejected: {reason}"
            attempt_history.append(AttemptRecord(
                attempt_id=attempt, status="REJECTED_UNSAFE",
                response=content, error_text=f"Unsafe change rejected: {reason}",
            ))
            _log_attempt(
                attempt, log_dir, messages=messages, llm_response=content,
                parsed_helper_lemmas=helper_lemmas, parsed_proof_body=proof_body,
                safety_ok=False, safety_reason=reason,
                status="REJECTED_UNSAFE — " + reason,
                tokens_in=p_tok, tokens_out=c_tok,
            )
            continue

        # --- Patch into harness ---
        candidate = patch_helper_lemmas(helper_lemmas, patched_harness)
        candidate = patch_proof_body(proof_body, candidate)

        # --- Verify ---
        verified, error_text, exit_code = run_verus(candidate, verus_path)
        if verified:
            elapsed = time.time() - start
            print(f"  Attempt {attempt} — VERIFIED ({elapsed:.1f}s, {total_in+total_out} tokens)", flush=True)
            _log_attempt(
                attempt, log_dir, messages=messages, llm_response=content,
                parsed_helper_lemmas=helper_lemmas, parsed_proof_body=proof_body,
                safety_ok=True, patched_code=candidate,
                verus_verified=True, verus_exit_code=exit_code,
                status="VERIFIED",
                tokens_in=p_tok, tokens_out=c_tok,
            )
            return True

        print(f"  Attempt {attempt} — Verus failed (exit {exit_code})", flush=True)
        last_errors = error_text
        attempt_history.append(AttemptRecord(
            attempt_id=attempt, status="VERUS_FAIL",
            response=content, error_text=error_text,
        ))
        _log_attempt(
            attempt, log_dir, messages=messages, llm_response=content,
            parsed_helper_lemmas=helper_lemmas, parsed_proof_body=proof_body,
            safety_ok=True, patched_code=candidate,
            verus_verified=False, verus_errors=error_text, verus_exit_code=exit_code,
            status="VERUS_FAIL",
            tokens_in=p_tok, tokens_out=c_tok,
        )

    elapsed = time.time() - start
    print(f"  All {max_repairs} repair attempts exhausted ({elapsed:.1f}s, {total_in+total_out} tokens)", flush=True)
    return False


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = load_config(args.config)
    verus_path = args.verus_binary_path or cfg.get("verus_path", "verus")
    max_repairs = args.max_repairs

    with open(args.model_spec_file, "r") as f:
        model_spec = f.read()
    with open(args.eval_harness_task_file, "r") as f:
        eval_harness_content = f.read()

    # Derive a task name from the harness filename for logging
    task_name = Path(args.eval_harness_task_file).stem
    output_dir = Path(args.output_dir) if args.output_dir else Path("output")
    log_dir = init_logging(output_dir, task_name)
    print(f"Logs will be written to: {log_dir}")

    print("Patching model spec into evaluation harness...")
    patched_content = patch_spec(model_spec, eval_harness_content)

    print("Running initial Verus verification...")
    verified, error_text, exit_code = run_verus(patched_content, verus_path)

    # Log the initial verification as attempt 0
    _log_attempt(
        0, log_dir, patched_code=patched_content,
        verus_verified=verified, verus_errors=error_text, verus_exit_code=exit_code,
        status="VERIFIED (initial)" if verified else "FAILED (initial)",
    )

    if verified:
        print("Proof verified successfully with patched model spec.")
        return True

    print(f"Initial verification failed (exit {exit_code}), entering repair loop...")
    client = make_client(cfg)
    success = run_equivalence_repair_loop(
        patched_harness=patched_content,
        verus_path=verus_path,
        client=client,
        cfg=cfg,
        last_errors=error_text,
        max_repairs=max_repairs,
        log_dir=log_dir,
    )

    if success:
        print("Equivalence proof verified successfully.")
    else:
        print("Failed to verify equivalence proof after all attempts.")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Insert model spec, verify equivalence with repair loop")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--model-spec-file", type=str, required=True, help="Path to the LLM response file containing model specs")
    parser.add_argument("--eval-harness-task-file", type=str, required=True, help="Path to the evaluation harness task file")
    parser.add_argument("--verus-binary-path", type=str, default=None, help="Path to the Verus binary (overrides config)")
    parser.add_argument("--max-repairs", type=int, default=5, help="Max repair attempts")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for logs and outputs")
    args = parser.parse_args()
    main(args)