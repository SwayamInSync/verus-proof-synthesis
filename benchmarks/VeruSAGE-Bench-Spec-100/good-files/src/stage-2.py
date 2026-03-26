import argparse
import re
import tempfile
import subprocess
import os


def patch_spec(model_spec, eval_harness_content):
    pattern = r"(//%%\s*MODEL_SPEC_START\n).*?(//%%\s*MODEL_SPEC_END)"
    replacement = r"\1" + model_spec.rstrip("\n") + "\n    " + r"\2"
    return re.sub(pattern, replacement, eval_harness_content, flags=re.DOTALL)


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

def run_equivalence_repair_loop(N=5):
    print(f"Running equivalence repair loop with N={N}...")
    # TODO: Implement the equivalence repair loop logic here

def main(args):
    model_spec = None
    eval_harness_content = None
    with open(args.model_spec_file, "r") as f:
        model_spec = f.read()
    with open(args.eval_harness_task_file, "r") as f:
        eval_harness_content = f.read()
    
    print("Patching model spec into evaluation harness...")
    patched_content = patch_spec(model_spec, eval_harness_content)
    print(patched_content)
    verified, error_text, exit_code = run_verus(patched_content, args.verus_binary_path)
    if not verified:
        run_equivalence_repair_loop(N=5)
    
    else:
        print("Proof verified successfully with patched model spec.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Insert loop isolation attribute and model specs")
    parser.add_argument("--model-spec-file", type=str, required=True, help="Path to the LLM response file containing model specs")
    parser.add_argument("--eval-harness-task-file", type=str, required=True, help="Path to the evaluation harness task file")
    parser.add_argument("--verus-binary-path", type=str, required=True, help="Path to the Verus binary for equivalence checking")
    args = parser.parse_args()
    main(args)