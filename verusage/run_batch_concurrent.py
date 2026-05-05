#!/usr/bin/env python3
"""
Batch runner for Verus repair tasks.
Runs repair on all files in a directory and logs results.
Supports parallel execution via --workers N for faster throughput.
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Global shutdown state (shared across threads) ──────────────────────────
shutdown_event = threading.Event()            # set on Ctrl+C
_active_procs: dict[int, subprocess.Popen] = {}  # pid -> Popen
_active_procs_lock = threading.Lock()


def _register_proc(proc: subprocess.Popen):
    with _active_procs_lock:
        _active_procs[proc.pid] = proc


def _unregister_proc(proc: subprocess.Popen):
    with _active_procs_lock:
        _active_procs.pop(proc.pid, None)


def _kill_all_children():
    """Kill every tracked child process group."""
    with _active_procs_lock:
        for pid, proc in list(_active_procs.items()):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        _active_procs.clear()


def _sigint_handler(signum, frame):
    """Handle Ctrl+C gracefully: kill children, prevent EXCEPTION results."""
    if shutdown_event.is_set():
        return  # already shutting down
    print("\n\n>>> Ctrl+C received — shutting down, killing child processes...", flush=True)
    shutdown_event.set()
    _kill_all_children()

# CSV field names used throughout
CSV_FIELDNAMES = [
    "file",
    "exit_code",
    "status",
    "time_seconds",
    "input_tokens",
    "output_tokens",
    "total_tokens",
]


def parse_tokens_from_log(log_file):
    """Parse input and output tokens from the log file"""
    total_input = 0
    total_output = 0

    if not log_file or not Path(log_file).exists():
        return total_input, total_output

    try:
        with open(log_file) as f:
            content = f.read()
            # Find all token counts (format: "Input tokens: 1234")
            input_tokens = re.findall(r"Input tokens:\s*(\d+)", content)
            output_tokens = re.findall(r"Output tokens:\s*(\d+)", content)

            total_input = sum(int(t) for t in input_tokens)
            total_output = sum(int(t) for t in output_tokens)
    except Exception as e:
        print(f"Warning: Failed to parse tokens from {log_file}: {e}")

    return total_input, total_output


def run_single_task(rs_file, config_file, output_dir, repair_steps, verus_args, timeout=1200):
    """
    Run repair on a single .rs file. This is a standalone function suitable for
    parallel execution — no shared mutable state.

    Returns:
        dict with keys: file, exit_code, status, time_seconds, input_tokens, output_tokens, total_tokens
        Returns None if shutdown was requested (task should not be recorded).
    """
    # Check for shutdown before even starting
    if shutdown_event.is_set():
        return None

    rs_file = Path(rs_file)
    output_dir = Path(output_dir)
    file_name = rs_file.name

    # Output file path
    output_file = output_dir / f"{rs_file.stem}_output.rs"

    # Build command
    cmd = [
        "python3",
        "main.py",
        "--config",
        config_file,
        "--mode",
        "agent",
        "--input",
        str(rs_file),
        "--output",
        str(output_file),
        "--outdir",
        str(output_dir),
        "--repair",
        str(repair_steps),
    ]

    if verus_args:
        cmd.extend(["--verus-args", verus_args])

    # Run main.py
    start_time = time.time()
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        _register_proc(proc)

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            _unregister_proc(proc)
            elapsed = time.time() - start_time
            return {
                "file": file_name,
                "exit_code": -1,
                "status": "TIMEOUT",
                "time_seconds": round(elapsed, 2),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        _unregister_proc(proc)

        # If shutdown happened while we were waiting, discard result
        if shutdown_event.is_set():
            return None

        elapsed = time.time() - start_time

        if exit_code == 0:
            status = "VERIFIED"
        elif exit_code == 233:
            status = "FAILED"
        else:
            # Negative exit codes = killed by signal (e.g. -15=SIGTERM, -9=SIGKILL)
            # Don't record these as EXCEPTION if shutdown was requested
            if exit_code < 0:
                return None  # killed by our signal handler or system
            status = "EXCEPTION"
            try:
                with open(output_dir / f"{rs_file.stem}_error.log", "w") as ef:
                    ef.write("STDOUT:\n")
                    ef.write((stdout or "") + "\n")
                    ef.write("STDERR:\n")
                    ef.write((stderr or "") + "\n")
            except Exception:
                pass

        # Find and parse log file for token counts
        log_file = None
        candidates = list(output_dir.glob(f"o-{rs_file.stem}-*"))
        if candidates:
            newest_subdir = max(candidates, key=lambda p: p.stat().st_mtime)
            log_file = newest_subdir / "verus-repair.log"

        input_tokens, output_tokens = parse_tokens_from_log(log_file)

        return {
            "file": file_name,
            "exit_code": exit_code,
            "status": status,
            "time_seconds": round(elapsed, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    except Exception as e:
        if proc is not None:
            _unregister_proc(proc)
        if shutdown_event.is_set():
            return None
        elapsed = time.time() - start_time
        return {
            "file": file_name,
            "exit_code": -2,
            "status": "ERROR",
            "time_seconds": round(elapsed, 2),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "_error": str(e),
        }


def run_batch(input_dir, config_file, output_dir=None, verus_args="", continue_from=None,
              repair_steps=None, num_workers=1, skip_projects=None):
    """
    Run repair on all .rs files in the input directory.

    Args:
        input_dir: Path to directory containing .rs files
        config_file: Path to config JSON file
        output_dir: Output directory for batch results (optional)
        verus_args: Additional arguments to pass to Verus
        continue_from: Directory to continue execution from (optional)
        repair_steps: Override number of repair steps (reads from config if None)
        num_workers: Number of parallel workers (default 1 = sequential)
    """
    input_dir_path = Path(input_dir)

    # Resolve repair steps: CLI override > config > default 20
    if repair_steps is None:
        try:
            with open(config_file) as cf:
                cfg = json.load(cf)
                repair_steps = cfg.get("repair_steps", 20)
        except Exception:
            repair_steps = 20
    print(f"Repair steps per task: {repair_steps}")
    print(f"Parallel workers: {num_workers}")

    # Results tracking
    results = []
    processed_files = set()

    if continue_from:
        output_dir = Path(continue_from)
        if not output_dir.exists():
            print(f"Error: Continue directory not found: {continue_from}")
            return
        print(f"Continuing from: {output_dir}")

        # Load existing results from CSV (preferred as it's incremental)
        csv_file = output_dir / "results.csv"
        if csv_file.exists():
            try:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert types back from string
                        try:
                            task = {
                                "file": row["file"],
                                "exit_code": int(row["exit_code"]),
                                "status": row["status"],
                                "time_seconds": float(row["time_seconds"]),
                                "input_tokens": int(row["input_tokens"]),
                                "output_tokens": int(row["output_tokens"]),
                                "total_tokens": int(row["total_tokens"]),
                            }
                            results.append(task)
                            processed_files.add(task["file"])
                        except (ValueError, KeyError) as e:
                            print(f"Warning: Skipping malformed row in CSV: {row} ({e})")

                print(f"Loaded {len(results)} previous results from CSV")
            except Exception as e:
                print(f"Warning: Failed to load existing results from CSV: {e}")

        # Fallback to JSON if no results loaded from CSV
        if not results:
            json_file = output_dir / "results.json"
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if "tasks" in data:
                            results = data["tasks"]
                            processed_files = {task["file"] for task in results}
                    print(f"Loaded {len(results)} previous results from JSON")
                except Exception as e:
                    print(f"Warning: Failed to load existing results from JSON: {e}")
    else:
        # Create output directory
        benchmark_name = input_dir_path.parent.name + "_" + input_dir_path.name
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(f"batch-results-{benchmark_name}-{timestamp}")

        output_dir.mkdir(parents=True, exist_ok=True)

    # Get all .rs files
    all_rs_files = sorted(input_dir_path.glob("*.rs"))

    # Filter out files from skipped projects (matched by prefix before first '__')
    if skip_projects:
        skip_prefixes = tuple(p + "__" for p in skip_projects)
        skipped = [f for f in all_rs_files if f.name.startswith(skip_prefixes)]
        all_rs_files = [f for f in all_rs_files if not f.name.startswith(skip_prefixes)]
        if skipped:
            print(f"Skipping {len(skipped)} files from projects: {', '.join(skip_projects)}")

    # Filter out processed files
    rs_files = [f for f in all_rs_files if f.name not in processed_files]

    if not rs_files:
        print(f"No new .rs files to process in {input_dir}")
        if not results:
            return
    else:
        print(f"Found {len(rs_files)} new files to process (total {len(all_rs_files)})")

    print(f"Output directory: {output_dir}")

    summary_file = output_dir / "summary.txt"
    csv_file = output_dir / "results.csv"

    # Create CSV file with headers if it doesn't exist or we are starting new
    write_header = not continue_from or not csv_file.exists()
    if write_header:
        with open(csv_file, "w", newline="") as cf:
            csv_writer = csv.DictWriter(cf, fieldnames=CSV_FIELDNAMES)
            csv_writer.writeheader()

    # Write summary header
    mode = "a" if continue_from else "w"
    with open(summary_file, mode) as sf:
        if not continue_from:
            sf.write("Batch Repair Run\n")
            sf.write(f"Benchmark: {input_dir_path.parent.name}_{input_dir_path.name}\n")
            sf.write(f"Timestamp: {time.strftime('%Y%m%d-%H%M%S')}\n")
            sf.write(f"Config: {config_file}\n")
            sf.write(f"Total files: {len(all_rs_files)}\n")
            sf.write(f"Workers: {num_workers}\n")
            sf.write("=" * 80 + "\n\n")
        else:
            sf.write(f"\nResuming run at {time.strftime('%Y%m%d-%H%M%S')} with {num_workers} workers\n")
            sf.write("=" * 80 + "\n\n")

    # Thread lock for file I/O (CSV, summary, results list)
    io_lock = threading.Lock()
    # Progress counters
    completed_count = [0]  # mutable container for closure
    verified_so_far = [0]
    batch_start_time = time.time()

    def record_result(task_result, task_idx):
        """Thread-safe recording of a task result."""
        with io_lock:
            results.append(task_result)
            completed_count[0] += 1
            if task_result["status"] == "VERIFIED":
                verified_so_far[0] += 1

            # Append to CSV
            with open(csv_file, "a", newline="") as cf:
                csv_writer = csv.DictWriter(cf, fieldnames=CSV_FIELDNAMES)
                csv_writer.writerow({k: v for k, v in task_result.items() if k in CSV_FIELDNAMES})

            # Append to summary
            with open(summary_file, "a") as sf:
                sf.write(f"[{task_idx:3d}] {task_result['file']}\n")
                sf.write(f"Status: {task_result['status']}\n")
                if task_result["status"] == "TIMEOUT":
                    sf.write(f"Timeout after {task_result['time_seconds']:.2f} seconds\n")
                elif task_result["status"] == "ERROR":
                    sf.write(f"Error: {task_result.get('_error', 'unknown')}\n")
                else:
                    sf.write(f"Exit code: {task_result['exit_code']}\n")
                    sf.write(f"Time: {task_result['time_seconds']:.2f} seconds\n")
                    sf.write(f"Input tokens: {task_result['input_tokens']}\n")
                    sf.write(f"Output tokens: {task_result['output_tokens']}\n")
                    sf.write(f"Total tokens: {task_result['total_tokens']}\n")
                sf.write("-" * 80 + "\n\n")

            done = completed_count[0]
            total = len(rs_files)
            elapsed_batch = time.time() - batch_start_time
            rate = done / elapsed_batch if elapsed_batch > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  [{done}/{total}] {task_result['file']}: {task_result['status']} "
                f"(time={task_result['time_seconds']:.0f}s, "
                f"tokens={task_result.get('total_tokens', 0)}, "
                f"verified={verified_so_far[0]}, "
                f"ETA={eta/3600:.1f}h)",
                flush=True,
            )

    start_idx = len(processed_files) + 1

    # Install signal handler for graceful shutdown
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    try:
      if num_workers <= 1:
        # Sequential execution (original behavior)
        for idx, rs_file in enumerate(rs_files, start_idx):
            if shutdown_event.is_set():
                break
            print(f"\n[{idx}/{len(all_rs_files)}] Processing: {rs_file.name}", flush=True)
            task_result = run_single_task(
                rs_file, config_file, output_dir, repair_steps, verus_args
            )
            if task_result is None:  # shutdown or killed
                break
            record_result(task_result, idx)
      else:
        # Parallel execution
        print(f"\nStarting parallel execution with {num_workers} workers...")
        print(f"Queuing {len(rs_files)} tasks (idx {start_idx}-{start_idx + len(rs_files) - 1} of {len(all_rs_files)})...", flush=True)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_info = {}
            for idx, rs_file in enumerate(rs_files, start_idx):
                if shutdown_event.is_set():
                    break
                future = executor.submit(
                    run_single_task,
                    rs_file, config_file, output_dir, repair_steps, verus_args
                )
                future_to_info[future] = (idx, rs_file)

            print(f"All {len(future_to_info)} tasks queued. First results in ~2-5 min...\n", flush=True)

            for future in as_completed(future_to_info):
                if shutdown_event.is_set():
                    # Cancel remaining futures
                    for f in future_to_info:
                        f.cancel()
                    break
                idx, rs_file = future_to_info[future]
                try:
                    task_result = future.result()
                except Exception as e:
                    if shutdown_event.is_set():
                        break
                    task_result = {
                        "file": rs_file.name,
                        "exit_code": -2,
                        "status": "ERROR",
                        "time_seconds": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "_error": str(e),
                    }
                if task_result is None:  # shutdown or killed
                    continue
                record_result(task_result, idx)
    finally:
      # Restore original handler
      signal.signal(signal.SIGINT, original_sigint)
      # Make sure all children are dead
      _kill_all_children()

    if shutdown_event.is_set():
        new_done = completed_count[0]
        print(f"\n>>> Shutdown complete. {new_done} tasks finished cleanly in this session.", flush=True)
        print(f">>> Total results in CSV: {len(results)}. Re-run with --continue-from to resume.", flush=True)

    # Write final summary
    verified_count = sum(1 for r in results if r["status"] == "VERIFIED")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    timeout_count = sum(1 for r in results if r["status"] == "TIMEOUT")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    exception_count = sum(1 for r in results if r["status"] == "EXCEPTION")
    total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
    total_tokens = total_input_tokens + total_output_tokens

    total_processed = len(results)
    total_wall_time = time.time() - batch_start_time

    with open(summary_file, "a") as sf:
        sf.write("\n" + "=" * 80 + "\n")
        sf.write("FINAL SUMMARY\n")
        sf.write("=" * 80 + "\n")
        sf.write(f"Total files:  {len(all_rs_files)}\n")
        sf.write(f"Processed:    {total_processed}\n")
        sf.write(f"Workers:      {num_workers}\n")
        sf.write(f"Wall time:    {total_wall_time:.0f}s ({total_wall_time/3600:.1f}h)\n")
        if total_processed > 0:
            sf.write(
                f"Verified:     {verified_count} ({100*verified_count/total_processed:.1f}%)\n"
            )
            sf.write(f"Failed:       {failed_count} ({100*failed_count/total_processed:.1f}%)\n")
            sf.write(f"Timeout:      {timeout_count} ({100*timeout_count/total_processed:.1f}%)\n")
            sf.write(
                f"Exception:    {exception_count} ({100*exception_count/total_processed:.1f}%)\n"
            )
            sf.write(f"Error:        {error_count} ({100*error_count/total_processed:.1f}%)\n")
        sf.write("\nToken Usage:\n")
        sf.write(f"Input tokens:  {total_input_tokens:,}\n")
        sf.write(f"Output tokens: {total_output_tokens:,}\n")
        sf.write(f"Total tokens:  {total_tokens:,}\n")

    # Save JSON results
    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "benchmark": input_dir_path.parent.name + "_" + input_dir_path.name,
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "config": config_file,
                "workers": num_workers,
                "wall_time_seconds": round(total_wall_time, 2),
                "total": len(all_rs_files),
                "verified": verified_count,
                "failed": failed_count,
                "timeout": timeout_count,
                "exception": exception_count,
                "error": error_count,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "tasks": [{k: v for k, v in r.items() if k in CSV_FIELDNAMES} for r in results],
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 80}")
    print("Batch run complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Wall time: {total_wall_time:.0f}s ({total_wall_time/3600:.1f}h) with {num_workers} workers")
    if total_processed > 0:
        print(
            f"Verified: {verified_count}/{total_processed} ({100*verified_count/total_processed:.1f}%)"
        )
    print(
        f"Total tokens: {total_tokens:,} (input: {total_input_tokens:,}, output: {total_output_tokens:,})"
    )
    print("\nOutput files:")
    print(f"  - CSV:     {csv_file}")
    print(f"  - JSON:    {output_dir / 'results.json'}")
    print(f"  - Summary: {summary_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch runner for Verus repair tasks")
    parser.add_argument("input_dir", help="Directory containing .rs files to process")
    parser.add_argument("config_file", help="Path to config JSON file")
    parser.add_argument("--output-dir", default=None, help="Output directory for batch results")
    parser.add_argument(
        "--verus-args", type=str, default="", help="Additional arguments to pass to Verus"
    )
    parser.add_argument(
        "--continue-from", type=str, default=None, help="Directory to continue execution from"
    )
    parser.add_argument(
        "--repair", type=int, default=None,
        help="Override repair steps per task (default: read from config, fallback 20)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1 = sequential). "
             "Higher values send concurrent requests to vLLM for better throughput."
    )    
    parser.add_argument(
        '--skip-projects', type=str, default=None, metavar='PREFIX',
        help='Skip task files from these projects (comma-separated prefixes, e.g. "AC,AL,NR"). '
             "Use when a model was trained on those repos and they should be excluded."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        sys.exit(1)

    if args.workers < 1:
        print(f"Error: --workers must be >= 1, got {args.workers}")
        sys.exit(1)

    run_batch(
        args.input_dir, args.config_file, args.output_dir, args.verus_args, args.continue_from,
        args.repair, args.workers,
        [p.strip() for p in args.skip_projects.split(',')] if args.skip_projects else None
    )
