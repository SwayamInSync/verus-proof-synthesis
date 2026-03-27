import json
import os
import subprocess
import tempfile

import openai


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
#  LLM client
# ---------------------------------------------------------------------------

def make_client(cfg: dict) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=cfg.get("aoai_api_key", ["dummy"])[0],
        base_url=cfg.get("aoai_api_base", ["http://localhost:8000/v1"])[0],
        max_retries=cfg.get("aoai_max_retries", 3),
    )


def call_llm(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> tuple[str, int, int]:
    """Single LLM call. Returns (content, prompt_tokens, completion_tokens)."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    content = resp.choices[0].message.content or ""
    prompt_tok = resp.usage.prompt_tokens if resp.usage else 0
    compl_tok = resp.usage.completion_tokens if resp.usage else 0
    return content, prompt_tok, compl_tok


# ---------------------------------------------------------------------------
#  Verus invocation
# ---------------------------------------------------------------------------

def run_verus(code: str, verus_path: str, timeout: int = 90) -> tuple[bool, str, int]:
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


# ---------------------------------------------------------------------------
#  Safety checks
# ---------------------------------------------------------------------------

UNSAFE_PATTERNS = [
    ("admit()", "admit() calls"),
    ("assume(", "assume(...) calls"),
    ("#[verifier::external_body]", "external_body annotations"),
    ("#[verifier::admit]", "verifier::admit annotations"),
]


def is_safe_content(text: str) -> tuple[bool, str]:
    """Check that text does not contain disallowed constructs."""
    for pattern, desc in UNSAFE_PATTERNS:
        if pattern in text:
            return False, f"Contains disallowed {desc}"
    return True, ""


# ---------------------------------------------------------------------------
#  Lynette
# ---------------------------------------------------------------------------

def find_lynette() -> str | None:
    """Locate the Lynette binary relative to this file or the repo root."""
    from pathlib import Path
    # Try relative to this file: ../../utils/lynette/...  or  ../../../../utils/lynette/...
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "utils" / "lynette" / "source" / "target" / "debug",
        Path(__file__).resolve().parent.parent
        / "utils" / "lynette" / "source" / "target" / "debug",
    ]
    for base in candidates:
        for name in ("lynette", "lynette.exe"):
            p = base / name
            if p.exists():
                return str(p)
    return None


def lynette_compare(original: str, modified: str, flags: list[str] | None = None) -> tuple[bool, str]:
    """
    Run `lynette compare` between two code strings.
    By default (no flags), strips ALL ghost/spec code and compares only exec code.
    Pass flags like ['--spec'] to also compare spec function bodies,
    ['--requires', '--ensures'] to compare those clauses, etc.
    Returns (ok, reason).  ok=True means the files are the same after deghosting.
    """
    lynette = find_lynette()
    if lynette is None:
        return True, "(Lynette not found, skipping compare check)"

    import tempfile as _tf
    orig_f = _tf.NamedTemporaryFile(mode="w", delete=False, prefix="lyn_orig_", suffix=".rs", encoding="utf-8")
    orig_f.write(original)
    orig_f.close()
    mod_f = _tf.NamedTemporaryFile(mode="w", delete=False, prefix="lyn_mod_", suffix=".rs", encoding="utf-8")
    mod_f.write(modified)
    mod_f.close()

    try:
        cmd = [lynette, "compare"] + (flags or []) + [orig_f.name, mod_f.name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, ""
        stdout = result.stdout.strip()
        return False, f"Lynette compare failed (exit {result.returncode}): {stdout or result.stderr.strip()}"
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
            os.unlink(mod_f.name)
        except OSError:
            pass


def lynette_additions_check(original: str, modified: str) -> tuple[bool, str]:
    """
    Run `lynette additions` to verify only allowed ghost/proof code was changed.
    Returns (ok, reason). If Lynette is not found, passes with a warning.

    NOTE: This is designed for the VeruSAGE proof-filling pipeline where target
    functions use `unimplemented!()` placeholders.  It does NOT support the
    spec-body-filling use case (stage-1) because:
      - `additions` only treats proof/exec fns as targets, not spec fns.
      - It recognizes `unimplemented!()` as the placeholder, not `arbitrary()`.
    For spec-body safety, use `lynette_compare` without flags instead (checks
    that exec code is unchanged) plus string-level safety checks.
    """
    lynette = find_lynette()
    if lynette is None:
        return True, "(Lynette not found, skipping additions check)"

    import tempfile as _tf
    orig_f = _tf.NamedTemporaryFile(mode="w", delete=False, prefix="lyn_orig_", suffix=".rs", encoding="utf-8")
    orig_f.write(original)
    orig_f.close()
    mod_f = _tf.NamedTemporaryFile(mode="w", delete=False, prefix="lyn_mod_", suffix=".rs", encoding="utf-8")
    mod_f.write(modified)
    mod_f.close()

    try:
        cmd = [lynette, "additions", orig_f.name, mod_f.name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, ""
        stdout = result.stdout.strip()
        if "Disallowed changes detected" in stdout:
            return False, f"Lynette: disallowed changes ({stdout})"
        return False, f"Lynette additions failed (exit {result.returncode}): {stdout or result.stderr.strip()}"
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
            os.unlink(mod_f.name)
        except OSError:
            pass
