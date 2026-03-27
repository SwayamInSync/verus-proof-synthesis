#!/usr/bin/env python3
"""
Runner: Orchestrates Stage 1 + Stage 2 evaluation over the task manifest.

For each task in the manifest (that has an equivalence harness):
  1. Run Stage 1: LLM fills spec body → Verus verifies
  2. If Stage 1 passes, run Stage 2: equivalence proof via harness
  3. Score: 0 (nothing), 0.5 (stage-1 only), 1.0 (both stages)

Usage:
  python runner.py --config config.json [--manifest task_manifest.json] [--output-dir output]
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# Stage modules (same directory)
from stage_1 import run_stage_1
from stage_2 import run_equivalence_repair_loop, patch_spec
from utility import load_config, make_client, run_verus


# ---------------------------------------------------------------------------
#  Data types
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_name: str
    score: float = 0.0          # 0, 0.5, or 1.0
    stage_1_passed: bool = False
    stage_1_attempts: int = 0
    stage_2_passed: bool = False
    stage_2_attempts: int = 0
    model_spec_body: str | None = None
    error: str | None = None     # for unexpected failures
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
#  Single-task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    task_name: str,
    task_file: Path,
    harness_file: Path,
    verus_path: str,
    client,
    cfg: dict,
    max_attempts_s1: int,
    max_attempts_s2: int,
    output_dir: Path,
) -> TaskResult:
    """Run Stage 1 + Stage 2 for one task. Returns a TaskResult."""
    result = TaskResult(task_name=task_name)
    t0 = time.time()

    print(f"\n{'='*70}", flush=True)
    print(f"  Task: {task_name}", flush=True)
    print(f"{'='*70}", flush=True)

    # Read files
    try:
        task_code = task_file.read_text(encoding="utf-8")
        harness_code = harness_file.read_text(encoding="utf-8")
    except Exception as e:
        result.error = f"Failed to read files: {e}"
        result.elapsed_s = time.time() - t0
        print(f"  ERROR: {result.error}", flush=True)
        return result

    # --- Stage 1 ---
    print(f"\n  --- Stage 1: Spec Body Generation ---", flush=True)
    log_dir_s1 = output_dir / "logs" / task_name / "stage-1"
    log_dir_s1.mkdir(parents=True, exist_ok=True)

    try:
        s1_passed, spec_body = run_stage_1(
            task_code=task_code,
            verus_path=verus_path,
            client=client,
            cfg=cfg,
            max_attempts=max_attempts_s1,
            log_dir=log_dir_s1,
        )
    except Exception as e:
        result.error = f"Stage 1 exception: {e}"
        result.elapsed_s = time.time() - t0
        print(f"  Stage 1 EXCEPTION: {e}", flush=True)
        return result

    result.stage_1_passed = s1_passed
    # Count attempts from log files
    result.stage_1_attempts = len(list(log_dir_s1.glob("attempt-*.md")))

    if not s1_passed:
        result.score = 0.0
        result.elapsed_s = time.time() - t0
        print(f"\n  Result: Stage 1 FAILED → score = 0.0", flush=True)
        return result

    result.model_spec_body = spec_body
    result.score = 0.5
    print(f"\n  Stage 1 PASSED (score so far: 0.5)", flush=True)

    # --- Stage 2 ---
    print(f"\n  --- Stage 2: Equivalence Proof ---", flush=True)
    log_dir_s2 = output_dir / "logs" / task_name / "stage-2"
    log_dir_s2.mkdir(parents=True, exist_ok=True)

    # Patch model spec into harness
    patched_harness = patch_spec(spec_body, harness_code)

    # Try initial verification (maybe equivalence holds trivially)
    print(f"  Running initial Verus verification on harness...", flush=True)
    verified, error_text, exit_code = run_verus(patched_harness, verus_path)

    if verified:
        result.stage_2_passed = True
        result.stage_2_attempts = 0
        result.score = 1.0
        result.elapsed_s = time.time() - t0
        print(f"  Equivalence verified trivially (no LLM needed)!", flush=True)
        print(f"\n  Result: Both stages PASSED → score = 1.0", flush=True)
        return result

    # Need LLM repair loop
    print(f"  Initial verification failed, entering repair loop...", flush=True)
    try:
        s2_passed = run_equivalence_repair_loop(
            patched_harness=patched_harness,
            verus_path=verus_path,
            client=client,
            cfg=cfg,
            last_errors=error_text,
            max_repairs=max_attempts_s2,
            log_dir=log_dir_s2,
        )
    except Exception as e:
        result.error = f"Stage 2 exception: {e}"
        result.elapsed_s = time.time() - t0
        print(f"  Stage 2 EXCEPTION: {e}", flush=True)
        return result

    result.stage_2_passed = s2_passed
    result.stage_2_attempts = len(list(log_dir_s2.glob("attempt-*.md")))

    if s2_passed:
        result.score = 1.0
        print(f"\n  Result: Both stages PASSED → score = 1.0", flush=True)
    else:
        result.score = 0.5
        print(f"\n  Result: Stage 1 passed, Stage 2 FAILED → score = 0.5", flush=True)

    result.elapsed_s = time.time() - t0
    return result


# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[TaskResult]) -> dict:
    """Print and return a summary dict."""
    total = len(results)
    scores = [r.score for r in results]
    total_score = sum(scores)

    s1_pass = sum(1 for r in results if r.stage_1_passed)
    s2_pass = sum(1 for r in results if r.stage_2_passed)
    errors = sum(1 for r in results if r.error)

    print(f"\n{'='*70}", flush=True)
    print(f"  EVALUATION SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Tasks evaluated:    {total}", flush=True)
    print(f"  Stage 1 passed:     {s1_pass}/{total}", flush=True)
    print(f"  Stage 2 passed:     {s2_pass}/{total}", flush=True)
    print(f"  Errors:             {errors}/{total}", flush=True)
    print(f"  Total score:        {total_score:.1f}/{total:.1f}", flush=True)
    print(f"  Average score:      {total_score/total:.3f}" if total else "", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n  Per-task breakdown:", flush=True)
    print(f"  {'Task':<75} {'S1':>3} {'S2':>3} {'Score':>6}", flush=True)
    print(f"  {'-'*75} {'---':>3} {'---':>3} {'------':>6}", flush=True)
    for r in results:
        s1 = "Y" if r.stage_1_passed else ("E" if r.error else "N")
        s2 = "Y" if r.stage_2_passed else ("-" if not r.stage_1_passed else "N")
        name = r.task_name[:75]
        print(f"  {name:<75} {s1:>3} {s2:>3} {r.score:>6.1f}", flush=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": total,
        "stage_1_passed": s1_pass,
        "stage_2_passed": s2_pass,
        "errors": errors,
        "total_score": total_score,
        "max_possible_score": float(total),
        "average_score": total_score / total if total else 0,
        "results": [asdict(r) for r in results],
    }
    return summary


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spec-100 Benchmark Runner: Stage 1 (spec gen) + Stage 2 (equivalence proof)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to JSON config file (LLM endpoint, model, etc.)",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to task_manifest.json (default: auto-detect in benchmark dir)",
    )
    parser.add_argument(
        "--benchmark-dir", type=str, default=None,
        help="Path to the good-files/ benchmark directory (default: auto-detect relative to this script)",
    )
    parser.add_argument(
        "--verus-binary-path", type=str, default=None,
        help="Path to Verus binary (overrides config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for logs, results, and outputs (default: output/<timestamp>)",
    )
    parser.add_argument(
        "--max-attempts-s1", type=int, default=5,
        help="Max LLM attempts for Stage 1 (default: 5)",
    )
    parser.add_argument(
        "--max-attempts-s2", type=int, default=5,
        help="Max LLM attempts for Stage 2 (default: 5)",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help="Run only the specified task name(s). Omit to run all.",
    )
    args = parser.parse_args()

    # --- Resolve paths ---
    cfg = load_config(args.config)
    verus_path = args.verus_binary_path or cfg.get("verus_path", "verus")

    # Benchmark dir
    if args.benchmark_dir:
        bench_dir = Path(args.benchmark_dir)
    else:
        bench_dir = Path(__file__).resolve().parent.parent  # src/ -> good-files/
    assert bench_dir.is_dir(), f"Benchmark dir not found: {bench_dir}"

    # Manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = bench_dir / "task_manifest.json"
    assert manifest_path.is_file(), f"Manifest not found: {manifest_path}"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter tasks
    if args.tasks:
        missing = [t for t in args.tasks if t not in manifest]
        if missing:
            print(f"WARNING: tasks not in manifest: {missing}", flush=True)
        task_names = [t for t in args.tasks if t in manifest]
    else:
        task_names = list(manifest.keys())

    if not task_names:
        print("No tasks to run.", flush=True)
        sys.exit(1)

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("output") / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Print run info ---
    print(f"{'='*70}", flush=True)
    print(f"  Spec-100 Benchmark Runner", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Config:       {args.config}", flush=True)
    print(f"  Verus:        {verus_path}", flush=True)
    print(f"  Benchmark:    {bench_dir}", flush=True)
    print(f"  Output:       {output_dir}", flush=True)
    print(f"  Tasks:        {len(task_names)}", flush=True)
    print(f"  Max attempts: S1={args.max_attempts_s1}, S2={args.max_attempts_s2}", flush=True)
    print(f"  Model:        {cfg.get('aoai_debug_model', '?')}", flush=True)
    print(f"{'='*70}", flush=True)

    # --- Create LLM client ---
    client = make_client(cfg)

    # --- Run tasks ---
    results: list[TaskResult] = []
    run_start = time.time()

    for i, task_name in enumerate(task_names, 1):
        entry = manifest[task_name]
        task_file = bench_dir / entry["task_file"]
        harness_file = bench_dir / entry["equivalence_harness_file"]

        print(f"\n[{i}/{len(task_names)}] {task_name}", flush=True)

        if not task_file.is_file():
            print(f"  SKIP: task file not found: {task_file}", flush=True)
            results.append(TaskResult(task_name=task_name, error="task file not found"))
            continue
        if not harness_file.is_file():
            print(f"  SKIP: harness file not found: {harness_file}", flush=True)
            results.append(TaskResult(task_name=task_name, error="harness file not found"))
            continue

        r = evaluate_task(
            task_name=task_name,
            task_file=task_file,
            harness_file=harness_file,
            verus_path=verus_path,
            client=client,
            cfg=cfg,
            max_attempts_s1=args.max_attempts_s1,
            max_attempts_s2=args.max_attempts_s2,
            output_dir=output_dir,
        )
        results.append(r)

        # Save incremental results after each task
        summary = print_summary(results)
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)

    # --- Final summary ---
    run_elapsed = time.time() - run_start
    summary = print_summary(results)
    summary["total_elapsed_s"] = run_elapsed

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total time: {run_elapsed:.1f}s", flush=True)
    print(f"  Results saved to: {results_path}", flush=True)

    return summary["average_score"]


if __name__ == "__main__":
    main()
