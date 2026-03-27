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
