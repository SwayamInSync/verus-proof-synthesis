#!/usr/bin/env python3
"""
Select ~100 diverse spec-writing tasks from the full 1309 candidates.

Strategy:
  1. Assign difficulty tiers based on body line count
  2. Stratified sampling across (difficulty × project)
  3. Maximize return-type diversity
  4. Ensure every project is represented
  5. Optionally validate ground truth verifies with Verus

Target distribution:
  easy (1-2 lines):        15 tasks  (warm-up, buildable)
  medium-easy (3-5 lines): 20 tasks
  medium (6-10 lines):     25 tasks
  hard (11-20 lines):      25 tasks
  very-hard (21+ lines):   15 tasks  (stretch goals)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def difficulty_tier(body_lines: int) -> str:
    if body_lines <= 2:
        return "easy"
    if body_lines <= 5:
        return "medium-easy"
    if body_lines <= 10:
        return "medium"
    if body_lines <= 20:
        return "hard"
    return "very-hard"


TIER_TARGETS = {
    "easy": 15,
    "medium-easy": 20,
    "medium": 25,
    "hard": 25,
    "very-hard": 15,
}


def stratified_sample(
    records: list[dict],
    target_total: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Stratified sample across difficulty tiers and projects."""
    rng = random.Random(seed)

    for r in records:
        r["difficulty"] = difficulty_tier(r["body_lines"])

    # Group by (difficulty, project)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["difficulty"], r["source_project"])].append(r)

    selected: list[dict] = []

    for tier, tier_target in TIER_TARGETS.items():
        # Pools per project for this tier
        proj_pools: dict[str, list[dict]] = {}
        for (d, p), rs in groups.items():
            if d == tier:
                proj_pools[p] = rs

        if not proj_pools:
            continue

        total_pool = sum(len(v) for v in proj_pools.values())

        # Allocate: 1 per project minimum, rest proportional to pool size
        alloc: dict[str, int] = {}
        for p in proj_pools:
            alloc[p] = 1

        remaining = tier_target - len(proj_pools)
        if remaining > 0:
            for p in proj_pools:
                extra = round(remaining * len(proj_pools[p]) / total_pool)
                alloc[p] += extra

        # Cap and sample
        for p, n in alloc.items():
            pool = proj_pools[p]
            n = min(n, len(pool))
            # Prefer diverse return types within each pool
            sampled = _diverse_sample(pool, n, rng)
            selected.extend(sampled)

    # If we're short, fill from under-represented projects/tiers
    if len(selected) < target_total:
        selected_ids = {r["task_id"] for r in selected}
        remaining_pool = [r for r in records if r["task_id"] not in selected_ids]
        # Prefer under-represented projects
        proj_counts = Counter(r["source_project"] for r in selected)
        remaining_pool.sort(key=lambda r: (proj_counts.get(r["source_project"], 0), rng.random()))
        needed = target_total - len(selected)
        selected.extend(remaining_pool[:needed])

    # If we're over, trim from the most over-represented tier
    while len(selected) > target_total:
        tier_counts = Counter(r["difficulty"] for r in selected)
        most_common_tier = tier_counts.most_common(1)[0][0]
        # Remove a random task from that tier
        tier_tasks = [i for i, r in enumerate(selected) if r["difficulty"] == most_common_tier]
        to_remove = rng.choice(tier_tasks)
        selected.pop(to_remove)

    return selected


def _diverse_sample(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n items from pool, preferring diverse return types."""
    if n >= len(pool):
        return list(pool)

    # Group by return type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_type[r["target_return_type"]].append(r)

    selected = []
    # First pass: one from each return type
    types = list(by_type.keys())
    rng.shuffle(types)
    for t in types:
        if len(selected) >= n:
            break
        item = rng.choice(by_type[t])
        selected.append(item)
        by_type[t].remove(item)

    # Fill remaining randomly
    if len(selected) < n:
        remaining = [r for t in by_type.values() for r in t]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected


def verify_ground_truth(task: dict, verus_bin: str, gt_dir: Path, timeout: int = 120) -> tuple[bool, str]:
    """Check that the ground truth file verifies with Verus.
    
    Returns (passed, detail_string).
    """
    gt_path = gt_dir / f"{task['task_id']}.rs"
    if not gt_path.exists():
        return False, "file not found"
    try:
        result = subprocess.run(
            [verus_bin, str(gt_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        # Check for "0 errors" in the verification results line
        if "0 errors" in output:
            return True, "verified"
        else:
            # Extract error count
            import re
            m = re.search(r'(\d+) errors', output)
            err_count = m.group(1) if m else "unknown"
            return False, f"{err_count} errors"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except OSError as e:
        return False, f"os_error: {e}"


def batch_verify(tasks: list[dict], verus_bin: str, gt_dir: Path,
                 timeout: int = 120, workers: int = 4) -> dict[str, bool]:
    """Verify multiple tasks in parallel. Returns {task_id: passed}."""
    results = {}
    total = len(tasks)
    
    def verify_one(task):
        passed, detail = verify_ground_truth(task, verus_bin, gt_dir, timeout)
        return task["task_id"], passed, detail
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(verify_one, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            tid, passed, detail = future.result()
            results[tid] = passed
            done += 1
            status = "PASS" if passed else f"FAIL ({detail})"
            passed_count = sum(1 for v in results.values() if v)
            failed_count = done - passed_count
            print(f"\r  [{done}/{total}] pass={passed_count} fail={failed_count} | {tid}: {status}" + " " * 20, end="", flush=True)
    
    print()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("benchmarks/VeruSAGE-Bench-Spec"),
        help="Directory with tasks.jsonl, tasks/, ground-truth/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/VeruSAGE-Bench-Spec-100"),
        help="Output directory for curated benchmark",
    )
    parser.add_argument("--target", type=int, default=100, help="Number of tasks to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ground truth files with Verus before including",
    )
    parser.add_argument(
        "--verus-bin",
        type=str,
        default=os.environ.get(
            "VERUS_BIN",
            "/home/t-swsingh/proof-model/ours/verified-code-gen/verus-binary/verus/verus",
        ),
        help="Path to Verus binary",
    )
    parser.add_argument("--verify-timeout", type=int, default=120, help="Verus timeout per task (seconds)")
    parser.add_argument("--verify-workers", type=int, default=4, help="Parallel verification workers")
    args = parser.parse_args()

    input_dir = args.input_dir
    jsonl_path = input_dir / "tasks.jsonl"
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    records = [json.loads(line) for line in open(jsonl_path)]
    print(f"Loaded {len(records)} candidate tasks")

    # Select first, then verify only the sampled tasks
    selected = stratified_sample(records, target_total=args.target, seed=args.seed)
    print(f"Selected {len(selected)} tasks (before verification)")

    if args.verify:
        gt_dir = input_dir / "ground-truth"
        selected_ids = {r["task_id"] for r in selected}
        remaining_pool = [r for r in records if r["task_id"] not in selected_ids]
        already_tried = set(selected_ids)

        # Round 1: verify all selected tasks
        to_verify = list(selected)
        verified_ids: set[str] = set()  # task_ids known to pass

        max_rounds = 10
        for round_num in range(1, max_rounds + 1):
            print(f"\n--- Verification round {round_num}: checking {len(to_verify)} tasks ---")
            print(f"  Workers: {args.verify_workers}, Timeout: {args.verify_timeout}s")

            results = batch_verify(
                to_verify, args.verus_bin, gt_dir,
                timeout=args.verify_timeout,
                workers=args.verify_workers,
            )

            # Update verified set with newly passed tasks
            for tid, passed in results.items():
                if passed:
                    verified_ids.add(tid)

            # Partition selected into passed and failed
            passed = [r for r in selected if r["task_id"] in verified_ids]
            failed = [r for r in selected if r["task_id"] not in verified_ids]
            print(f"  Passed: {len(passed)}, Failed: {len(failed)}")

            if not failed:
                break

            # Backfill: for each failed task, find a replacement from the same
            # (difficulty, project) bucket, falling back to same difficulty tier
            rng = random.Random(args.seed + round_num)
            replacements: list[dict] = []

            for ftask in failed:
                tier = ftask["difficulty"]
                proj = ftask["source_project"]
                # Best: same tier + project
                candidates = [r for r in remaining_pool
                              if r["task_id"] not in already_tried
                              and difficulty_tier(r["body_lines"]) == tier
                              and r["source_project"] == proj]
                if not candidates:
                    # Fallback: same tier, any project
                    candidates = [r for r in remaining_pool
                                  if r["task_id"] not in already_tried
                                  and difficulty_tier(r["body_lines"]) == tier]
                if not candidates:
                    # Last resort: any remaining candidate
                    candidates = [r for r in remaining_pool
                                  if r["task_id"] not in already_tried]
                if candidates:
                    pick = rng.choice(candidates)
                    pick["difficulty"] = difficulty_tier(pick["body_lines"])
                    replacements.append(pick)
                    already_tried.add(pick["task_id"])

            selected = passed + replacements
            # Update remaining pool
            selected_ids = {r["task_id"] for r in selected}
            remaining_pool = [r for r in records if r["task_id"] not in selected_ids
                              and r["task_id"] not in already_tried]

            if not replacements:
                print(f"  No more replacements available, keeping {len(selected)} tasks")
                break

            # Next round: only verify the NEW replacements
            to_verify = replacements
            print(f"  Backfilled {len(replacements)} replacements, verifying only those")

        print(f"\nFinal verified selection: {len(selected)} tasks")

    # Write output
    out_dir = args.output_dir
    tasks_dir = out_dir / "tasks"
    gt_out_dir = out_dir / "ground-truth"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    gt_out_dir.mkdir(parents=True, exist_ok=True)

    src_tasks = input_dir / "tasks"
    src_gt = input_dir / "ground-truth"

    for task in selected:
        tid = task["task_id"]
        # Copy task file
        src = src_tasks / f"{tid}.rs"
        if src.exists():
            shutil.copy2(src, tasks_dir / f"{tid}.rs")
        # Copy ground truth
        src = src_gt / f"{tid}.rs"
        if src.exists():
            shutil.copy2(src, gt_out_dir / f"{tid}.rs")

    # Write metadata
    with open(out_dir / "tasks.jsonl", "w") as f:
        for task in selected:
            f.write(json.dumps(task) + "\n")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Curated benchmark: {len(selected)} tasks")
    print(f"Output: {out_dir}")

    print(f"\nDifficulty distribution:")
    for tier, cnt in sorted(Counter(r["difficulty"] for r in selected).items()):
        print(f"  {tier}: {cnt}")

    print(f"\nProject distribution:")
    for proj, cnt in sorted(Counter(r["source_project"] for r in selected).items()):
        print(f"  {proj}: {cnt}")

    print(f"\nReturn type diversity:")
    rt_counts = Counter(r["target_return_type"] for r in selected)
    print(f"  {len(rt_counts)} distinct return types")
    for rt, cnt in rt_counts.most_common(10):
        print(f"  {rt}: {cnt}")

    body_sizes = sorted(r["body_lines"] for r in selected)
    print(f"\nBody size range: {body_sizes[0]}-{body_sizes[-1]} lines")
    print(f"  median={body_sizes[len(body_sizes)//2]}, mean={sum(body_sizes)/len(body_sizes):.1f}")


if __name__ == "__main__":
    main()
