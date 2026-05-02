#!/usr/bin/env python3
"""
Automated multi-model runner for the verus-proof-synthesis plain harness.

Given a JSON config listing local models with their download sources, vllm
launch args, and shared experiment params, this script sequentially per model:

  1. Downloads weights (HuggingFace snapshot_download or azcopy).
  2. Patches verusage/vllm_config.json into the per-run dir with model name,
     port, max_token / max_model_context (= --max-model-len), and
     action_max_tokens (8192 for max_model_len == 32768, else 81920).
  3. Launches vllm on the local machine.
  4. Waits for /health readiness (with log-tail watchdog for fatal errors).
  5. Runs plain-harness.py against the patched config + VeruSAGE-Bench tasks,
     capturing combined stdout/stderr to <run_dir>/harness.log.
  6. Tears down the vllm server and deletes the weights.
  7. Moves on to the next model.

After all models complete, prints a sanity-check summary aggregated from each
run's results.csv (status counts, attempt distribution, wall-clock totals) and
writes <log-dir>/_summary_<batch_ts>.{json,md}.

Usage:
  python run_models.py --config configs/full_run.json
  python run_models.py --config configs/full_run.json --only Qwen/Qwen3-Coder-30B-A3B-Instruct
  python run_models.py --config configs/full_run.json --force
  python run_models.py --config configs/full_run.json --dry-run
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Paths & constants ─────────────────────────────────────────────────────────

REPO_DIR = Path(__file__).resolve().parent           # benchmarks/verus-proof-synthesis/
DEFAULT_LOG_DIR = REPO_DIR / "logs"
HARNESS_SCRIPT = REPO_DIR / "plain-harness.py"
VLLM_CONFIG_TEMPLATE = REPO_DIR / "verusage" / "vllm_config.json"

# Patterns in vllm.log that indicate the server is dead-on-arrival.
VLLM_FATAL_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"address already in use", re.IGNORECASE),
    re.compile(r"\[Errno 98\]"),
    re.compile(r"RuntimeError:", re.IGNORECASE),
    re.compile(r"ValueError:", re.IGNORECASE),
    re.compile(r"AssertionError:", re.IGNORECASE),
    re.compile(r"OSError:", re.IGNORECASE),
    re.compile(r"ImportError:", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError:", re.IGNORECASE),
    re.compile(r"FATAL", re.IGNORECASE),
]

# Required experiment params (must come from defaults or per-model override).
REQUIRED_EXPERIMENT_KEYS = [
    "port", "repairs", "harness_workers",
    "verus_path", "tasks_dir",
]


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Config handling ───────────────────────────────────────────────────────────

def derive_model_label(name: str) -> str:
    return name.replace("/", "_").replace(":", "-")


def load_and_validate_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a JSON object")
    if "models" not in cfg or not isinstance(cfg["models"], list) or not cfg["models"]:
        raise ValueError(f"Config at {path} must contain a non-empty 'models' list")
    cfg.setdefault("defaults", {})
    if not isinstance(cfg["defaults"], dict):
        raise ValueError("Config 'defaults' must be an object")
    return cfg


def _extract_max_model_len(vllm_extra_args: list[str]) -> int:
    """Find --max-model-len <N> in vllm_extra_args; raise if absent or malformed."""
    for i, a in enumerate(vllm_extra_args):
        if a == "--max-model-len":
            if i + 1 >= len(vllm_extra_args):
                raise ValueError("--max-model-len requires a value")
            try:
                return int(vllm_extra_args[i + 1])
            except ValueError as e:
                raise ValueError(f"--max-model-len value must be int, got {vllm_extra_args[i+1]!r}") from e
        if a.startswith("--max-model-len="):
            return int(a.split("=", 1)[1])
    raise ValueError("vllm_extra_args must contain --max-model-len <N>")


def _action_max_tokens_for(max_model_len: int) -> int:
    """8192 for 32k context, 81920 for any larger context."""
    if max_model_len == 32768:
        return 8192
    if max_model_len > 32768:
        return 81920
    raise ValueError(
        f"max_model_len={max_model_len} is below the supported 32768 minimum"
    )


def resolve_model_config(defaults: dict[str, Any], model_entry: dict[str, Any]) -> dict[str, Any]:
    if "name" not in model_entry:
        raise ValueError(f"Each model entry must have a 'name' field: {model_entry}")
    if "source" not in model_entry or not isinstance(model_entry["source"], dict):
        raise ValueError(f"Model {model_entry['name']!r} missing 'source' object")
    src = model_entry["source"]
    if src.get("type") not in ("hf", "azcopy"):
        raise ValueError(
            f"Model {model_entry['name']!r}: source.type must be 'hf' or 'azcopy', got {src.get('type')!r}"
        )
    if src["type"] == "hf" and "repo_id" not in src:
        raise ValueError(f"Model {model_entry['name']!r}: hf source requires 'repo_id'")
    if src["type"] == "azcopy" and "url" not in src:
        raise ValueError(f"Model {model_entry['name']!r}: azcopy source requires 'url'")

    resolved = copy.deepcopy(defaults)
    for k, v in model_entry.items():
        if k == "vllm_extra_args" and v is not None:
            resolved["vllm_extra_args"] = list(v)
        elif k == "env" and v is not None:
            merged = dict(resolved.get("env") or {})
            merged.update(v)
            resolved["env"] = merged
        else:
            resolved[k] = v
    resolved.setdefault("vllm_extra_args", [])
    resolved.setdefault("env", {})
    resolved.setdefault("weights_dir", "/workdir/models")
    resolved.setdefault("vllm_startup_timeout_sec", 1800)

    missing = [k for k in REQUIRED_EXPERIMENT_KEYS if k not in resolved]
    if missing:
        raise ValueError(
            f"Model {model_entry['name']!r} is missing required experiment params: {missing}"
        )

    # Auto-derive token budgets from --max-model-len.
    max_model_len = _extract_max_model_len(resolved["vllm_extra_args"])
    resolved["max_model_len"] = max_model_len
    resolved["max_token"] = max_model_len
    resolved["max_model_context"] = max_model_len
    resolved["action_max_tokens"] = _action_max_tokens_for(max_model_len)
    return resolved


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


# ── Resumability ──────────────────────────────────────────────────────────────

def find_completed_run(model_log_dir: Path) -> Path | None:
    if not model_log_dir.is_dir():
        return None
    for ts_dir in sorted(model_log_dir.iterdir(), reverse=True):
        meta = ts_dir / "run_meta.json"
        if not meta.is_file():
            continue
        try:
            d = json.loads(meta.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("completed") is True:
            return ts_dir
    return None


# ── vllm config patching ──────────────────────────────────────────────────────

def build_vllm_config(model_name: str, resolved: dict[str, Any], dest: Path) -> Path:
    """Copy verusage/vllm_config.json to <dest>/vllm_config.json with patches.

    Patches: aoai_api_base port, aoai_generation_model, aoai_debug_model,
    max_token, max_model_context, action_max_tokens, repair_steps, verus_path,
    api_timeout, aoai_max_retries, debug_max_attempt, debug_temp.
    """
    if not VLLM_CONFIG_TEMPLATE.is_file():
        raise RuntimeError(f"vllm_config template not found: {VLLM_CONFIG_TEMPLATE}")
    cfg = json.loads(VLLM_CONFIG_TEMPLATE.read_text())
    port = int(resolved["port"])
    cfg["aoai_api_base"] = [f"http://localhost:{port}/v1"]
    cfg["aoai_api_key"] = cfg.get("aoai_api_key") or ["dummy"]
    cfg["aoai_generation_model"] = model_name
    cfg["aoai_debug_model"] = model_name
    cfg["max_token"] = int(resolved["max_token"])
    cfg["max_model_context"] = int(resolved["max_model_context"])
    cfg["action_max_tokens"] = int(resolved["action_max_tokens"])
    cfg["repair_steps"] = int(resolved["repairs"])
    cfg["verus_path"] = str(resolved["verus_path"])
    cfg["api_timeout"] = int(resolved.get("api_timeout", 300))
    cfg["aoai_max_retries"] = int(resolved.get("aoai_max_retries", 3))
    cfg["debug_max_attempt"] = int(resolved.get("debug_max_attempt", 3))
    cfg["debug_temp"] = float(resolved.get("debug_temp", 1.0))
    cfg["use_openai"] = True

    dest.parent.mkdir(parents=True, exist_ok=True)
    out_path = dest / "vllm_config.json"
    out_path.write_text(json.dumps(cfg, indent=4) + "\n")
    return out_path


# ── Weights download ──────────────────────────────────────────────────────────

def download_weights_hf(repo_id: str, dest: Path, revision: str | None,
                        log_path: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is not installed; install it or use azcopy source"
        ) from e

    token = os.environ.get("HF_TOKEN")
    log(f"  HF snapshot_download repo_id={repo_id} revision={revision} -> {dest}")
    if token is None:
        log("  (HF_TOKEN not set — gated models will fail)")

    dest.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n=== HF snapshot_download {repo_id} -> {dest} ===\n")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        revision=revision,
        token=token,
    )


def download_weights_azcopy(url: str, dest: Path, log_path: Path,
                            extra_args: list[str] | None = None) -> None:
    if shutil.which("azcopy") is None:
        raise RuntimeError("azcopy binary not found on PATH")
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["azcopy", "copy", url, str(dest), "--recursive=true"]
    if extra_args:
        cmd.extend(extra_args)
    log(f"  azcopy: {' '.join(cmd[:3])} <url> {' '.join(cmd[3:])}")
    with open(log_path, "a") as logf:
        logf.write(f"\n=== azcopy copy <url> -> {dest} ===\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"azcopy failed with returncode={proc.returncode}; see {log_path}")


def verify_weights(dest: Path) -> Path:
    if not dest.is_dir():
        raise RuntimeError(f"Weights dir does not exist after download: {dest}")
    if (dest / "config.json").is_file():
        return dest
    nested = [p for p in dest.iterdir() if p.is_dir() and (p / "config.json").is_file()]
    if len(nested) == 1:
        log(f"  weights nested one level deep: using {nested[0]}")
        return nested[0]
    if len(nested) > 1:
        raise RuntimeError(
            f"Weights dir {dest} contains multiple subdirs with config.json: {nested}"
        )
    raise RuntimeError(
        f"Weights dir {dest} is missing HF config.json — download incomplete or wrong url"
    )


# ── vllm lifecycle ────────────────────────────────────────────────────────────

def is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


def http_get_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionResetError, socket.timeout, OSError):
        return False


def launch_vllm(weights_dir: Path, extra_args: list[str], log_path: Path,
                port: int, env_overrides: dict[str, str] | None = None) -> subprocess.Popen:
    if is_port_open(port):
        raise RuntimeError(
            f"Port {port} is already in use — refusing to launch vllm. "
            f"Free the port or pick a different one in the config."
        )
    cmd = ["vllm", "serve", str(weights_dir), *extra_args]
    log(f"  vllm cmd: {' '.join(cmd)}")
    env = os.environ.copy()
    if env_overrides:
        for k, v in env_overrides.items():
            env[str(k)] = str(v)
        log(f"  vllm env overrides: {env_overrides}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "wb", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    return proc


def _scan_log_for_fatal(log_path: Path, last_pos: int) -> tuple[int, str | None]:
    if not log_path.is_file():
        return last_pos, None
    try:
        size = log_path.stat().st_size
        if size <= last_pos:
            return last_pos, None
        with open(log_path, "rb") as f:
            f.seek(last_pos)
            chunk = f.read(size - last_pos).decode("utf-8", errors="replace")
        for line in chunk.splitlines():
            for pat in VLLM_FATAL_PATTERNS:
                if pat.search(line):
                    return size, line.strip()[:300]
        return size, None
    except OSError:
        return last_pos, None


def wait_for_vllm(proc: subprocess.Popen, port: int, log_path: Path,
                  timeout_sec: int) -> float:
    health_url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    last_pos = 0
    last_status_log = 0.0

    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vllm process exited early with returncode={proc.returncode}; see {log_path}"
            )
        last_pos, fatal = _scan_log_for_fatal(log_path, last_pos)
        if fatal is not None:
            raise RuntimeError(f"vllm log contains fatal error: {fatal!r}; see {log_path}")

        if http_get_ok(health_url, timeout=3.0):
            return time.time() - start

        elapsed = time.time() - start
        if elapsed > timeout_sec:
            raise RuntimeError(
                f"vllm did not become ready within {timeout_sec}s; see {log_path}"
            )
        if elapsed - last_status_log > 30:
            log(f"  ... still waiting for vllm /health ({int(elapsed)}s elapsed)")
            last_status_log = elapsed
        time.sleep(5)


def _vllm_related_pids(main_pid: int) -> list[int]:
    if shutil.which("pgrep") is None:
        return []
    pats = [
        r"vllm serve",
        r"vllm\.entrypoints",
        r"VLLM_DP_Coordinator",
        r"EngineCore",
        r"APIServer",
        r"multiproc_executor",
        r"DPCoordinator",
    ]
    pids: set[int] = set()
    for pat in pats:
        try:
            out = subprocess.run(
                ["pgrep", "-f", pat], capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        pids.add(int(line))
        except Exception:
            pass
    pids.discard(os.getpid())
    pids.discard(main_pid)
    return sorted(pids)


def _gpu_used_mib() -> int:
    if shutil.which("nvidia-smi") is None:
        return -1
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return -1
        vals = []
        for ln in out.stdout.strip().splitlines():
            ln = ln.strip()
            if ln.isdigit():
                vals.append(int(ln))
        return max(vals) if vals else -1
    except Exception:
        return -1


def teardown_vllm(proc: subprocess.Popen, port: int) -> None:
    main_pid = proc.pid
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(main_pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log("  vllm didn't exit on SIGTERM; sending SIGKILL")
            try:
                os.killpg(os.getpgid(main_pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log("  WARNING: vllm did not die after SIGKILL")

    for sig in (signal.SIGTERM, signal.SIGKILL):
        stragglers = _vllm_related_pids(main_pid)
        if not stragglers:
            break
        log(f"  killing {len(stragglers)} vllm straggler(s) with "
            f"{signal.Signals(sig).name}: {stragglers}")
        for pid in stragglers:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError as e:
                log(f"    WARNING: cannot signal pid {pid}: {e}")
        time.sleep(5 if sig == signal.SIGTERM else 2)
    remaining = _vllm_related_pids(main_pid)
    if remaining:
        log(f"  WARNING: vllm stragglers still alive after SIGKILL: {remaining}")

    deadline = time.time() + 30
    while time.time() < deadline and is_port_open(port):
        time.sleep(1)
    if is_port_open(port):
        log(f"  WARNING: port {port} still in use after teardown")

    deadline = time.time() + 120
    while time.time() < deadline:
        used = _gpu_used_mib()
        if used < 0 or used <= 512:
            break
        time.sleep(2)
    _log_gpu_memory("post-teardown")


def _log_gpu_memory(tag: str) -> None:
    if shutil.which("nvidia-smi") is None:
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
            log(f"  GPU memory ({tag}):")
            leaked = False
            for ln in lines:
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) == 3:
                    idx, used, total = parts
                    log(f"    GPU {idx}: {used} / {total} MiB used")
                    try:
                        if int(used) > 1024:
                            leaked = True
                    except ValueError:
                        pass
                else:
                    log(f"    {ln}")
            if leaked and tag.startswith("post-"):
                log("  WARNING: GPU memory still allocated after teardown — possible leak")
    except Exception as e:
        log(f"  (nvidia-smi snapshot failed: {e})")


# ── Plain-harness subprocess ──────────────────────────────────────────────────

def run_plain_harness(vllm_config_path: Path, output_dir: Path, log_file: Path,
                      resolved: dict[str, Any]) -> tuple[int, float]:
    cmd = [
        sys.executable, "-u", str(HARNESS_SCRIPT),
        "--config", str(vllm_config_path),
        "--tasks-dir", str(resolved["tasks_dir"]),
        "--output-dir", str(output_dir),
        "--repairs", str(resolved["repairs"]),
        "--workers", str(resolved["harness_workers"]),
    ]
    log(f"  harness cmd: {' '.join(cmd)}")
    log(f"  harness log: {log_file}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log_file, "wb") as out:
        proc = subprocess.run(
            cmd, stdout=out, stderr=subprocess.STDOUT, cwd=REPO_DIR,
        )
    return proc.returncode, time.time() - start


# ── Per-model orchestration ───────────────────────────────────────────────────

class ModelRunResult:
    def __init__(self, name: str, model_label: str):
        self.name = name
        self.model_label = model_label
        self.run_dir: Path | None = None
        self.completed = False
        self.skipped = False
        self.error: str | None = None
        self.timings: dict[str, float] = {}
        self.exit_codes: dict[str, int] = {}


def run_one_model(model_entry: dict[str, Any], defaults: dict[str, Any],
                  log_dir: Path, force: bool, dry_run: bool) -> ModelRunResult:
    name = model_entry["name"]
    model_label = derive_model_label(name)
    result = ModelRunResult(name, model_label)

    log("")
    log(f"=== Model: {name} (label={model_label}) ===")

    resolved = resolve_model_config(defaults, model_entry)
    weights_dir = Path(resolved["weights_dir"]) / model_label

    model_log_dir = log_dir / model_label
    if not force:
        completed = find_completed_run(model_log_dir)
        if completed is not None:
            log(f"  skipping (already completed at {completed})")
            result.skipped = True
            result.completed = True
            result.run_dir = completed
            return result

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = model_log_dir / timestamp
    result.run_dir = run_dir
    vllm_log = run_dir / "vllm.log"
    download_log = run_dir / "download.log"
    harness_log = run_dir / "harness.log"
    harness_output = run_dir / "plain_harness_output"
    config_path = run_dir / "config.json"
    meta_path = run_dir / "run_meta.json"

    config_dump = {
        "model_name": name,
        "model_label": model_label,
        "weights_dir": str(weights_dir),
        "resolved": resolved,
        "source": model_entry["source"],
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
    }

    if dry_run:
        log("  [dry-run] would create run dir: " + str(run_dir))
        log(f"  [dry-run] max_model_len={resolved['max_model_len']} "
            f"max_token={resolved['max_token']} "
            f"max_model_context={resolved['max_model_context']} "
            f"action_max_tokens={resolved['action_max_tokens']}")
        log("  [dry-run] resolved config:")
        log("  " + json.dumps(config_dump, indent=2).replace("\n", "\n  "))
        src = model_entry["source"]
        if src["type"] == "hf":
            log(f"  [dry-run] would: huggingface_hub.snapshot_download({src['repo_id']!r}, "
                f"local_dir={str(weights_dir)!r}, revision={src.get('revision')!r})")
        else:
            log(f"  [dry-run] would: azcopy copy <url> {weights_dir} --recursive=true")
        if resolved.get("env"):
            log(f"  [dry-run] would set env: {resolved['env']}")
        log(f"  [dry-run] would: vllm serve {weights_dir} {' '.join(resolved['vllm_extra_args'])} "
            f"--served-model-name {name} --port {resolved['port']}")
        log(f"  [dry-run] would write patched vllm_config.json to {run_dir / 'vllm_config.json'}")
        log(f"  [dry-run] would: python plain-harness.py --config <patched> "
            f"--tasks-dir {resolved['tasks_dir']} --output-dir {harness_output} "
            f"--repairs {resolved['repairs']} --workers {resolved['harness_workers']}")
        log(f"  [dry-run] would: rm -rf {weights_dir}")
        result.completed = True
        return result

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_dump, indent=2))

    # Build patched vllm_config.json up-front so it's available even if vllm
    # never starts (useful for debugging).
    vllm_config_path = build_vllm_config(name, resolved, run_dir)
    log(f"  patched vllm_config: {vllm_config_path}")

    meta: dict[str, Any] = {
        "model_name": name,
        "model_label": model_label,
        "completed": False,
        "weights_dir": str(weights_dir),
        "vllm_pid": None,
        "exit_codes": {},
        "timings": {},
        "error": None,
        "max_model_len": resolved["max_model_len"],
        "max_token": resolved["max_token"],
        "max_model_context": resolved["max_model_context"],
        "action_max_tokens": resolved["action_max_tokens"],
    }

    def _save_meta() -> None:
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    _save_meta()

    proc: subprocess.Popen | None = None
    overall_start = time.time()
    try:
        # 1. Download weights.
        t0 = time.time()
        src = model_entry["source"]
        if src["type"] == "hf":
            download_weights_hf(src["repo_id"], weights_dir, src.get("revision"), download_log)
        else:
            download_weights_azcopy(src["url"], weights_dir, download_log,
                                    extra_args=resolved.get("azcopy_extra_args"))
        model_dir = verify_weights(weights_dir)
        meta["timings"]["download_sec"] = time.time() - t0
        log(f"  weights ready in {meta['timings']['download_sec']:.1f}s")
        _save_meta()

        # 2. Launch vllm. Inject --served-model-name (so the OpenAI client sees
        #    the same model id the harness sends) and --port, if not set.
        t0 = time.time()
        vllm_args = list(resolved["vllm_extra_args"])
        if "--served-model-name" not in vllm_args:
            vllm_args += ["--served-model-name", name]
        if "--port" not in vllm_args:
            vllm_args += ["--port", str(resolved["port"])]
        proc = launch_vllm(model_dir, vllm_args, vllm_log,
                           resolved["port"], env_overrides=resolved.get("env") or {})
        meta["vllm_pid"] = proc.pid
        _save_meta()
        startup_sec = wait_for_vllm(proc, resolved["port"], vllm_log,
                                    resolved["vllm_startup_timeout_sec"])
        meta["timings"]["vllm_startup_sec"] = startup_sec
        log(f"  vllm ready in {startup_sec:.1f}s (pid={proc.pid})")
        _save_meta()

        # 3. Run plain-harness.py.
        rc, dur = run_plain_harness(vllm_config_path, harness_output, harness_log, resolved)
        meta["timings"]["harness_sec"] = dur
        meta["exit_codes"]["harness"] = rc
        _save_meta()
        log(f"  plain-harness done in {dur:.1f}s (rc={rc})")

        if rc == 0:
            meta["completed"] = True
            result.completed = True
        else:
            meta["error"] = f"plain-harness exited with rc={rc}"
            result.error = meta["error"]
            log(f"  WARNING: plain-harness exited non-zero (rc={rc}); not marking run as completed")
        result.exit_codes = dict(meta["exit_codes"])
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        result.error = meta["error"]
        log(f"  ERROR: {meta['error']}")
        log("  " + traceback.format_exc().replace("\n", "\n  "))
    finally:
        if proc is not None:
            try:
                teardown_vllm(proc, resolved["port"])
            except Exception as e:
                log(f"  WARNING: vllm teardown failed: {e}")
        if weights_dir.exists():
            try:
                shutil.rmtree(weights_dir)
                log(f"  weights deleted: {weights_dir}")
            except Exception as e:
                log(f"  WARNING: failed to delete {weights_dir}: {e}")
        meta["timings"]["total_sec"] = time.time() - overall_start
        _save_meta()
        result.timings = dict(meta["timings"])

    return result


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise_run(run_dir: Path) -> dict[str, Any]:
    """Aggregate plain-harness results.csv from <run_dir>/plain_harness_output/."""
    out: dict[str, Any] = {}
    csv_path = run_dir / "plain_harness_output" / "results.csv"
    if csv_path.is_file():
        status_counts: Counter[str] = Counter()
        exit_codes: Counter[str] = Counter()
        attempts_dist: Counter[str] = Counter()
        total = 0
        verified = 0
        total_time = 0.0
        total_in_tok = 0
        total_out_tok = 0
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    total += 1
                    status = (row.get("status") or "").strip() or "unknown"
                    status_counts[status] += 1
                    # plain-harness writes "VERIFIED" / "FAILED" (uppercase).
                    if status.upper() == "VERIFIED":
                        verified += 1
                    rc = (row.get("exit_code") or "").strip()
                    if rc:
                        exit_codes[rc] += 1
                    a = (row.get("attempts_used") or "").strip()
                    if a:
                        attempts_dist[a] += 1
                    try:
                        total_time += float(row.get("time_seconds") or 0)
                    except ValueError:
                        pass
                    try:
                        total_in_tok += int(row.get("input_tokens") or 0)
                    except ValueError:
                        pass
                    try:
                        total_out_tok += int(row.get("output_tokens") or 0)
                    except ValueError:
                        pass
            out["plain_harness"] = {
                "total": total,
                "verified": verified,
                "pass_rate": (verified / total) if total else 0.0,
                "status_counts": dict(status_counts),
                "exit_code_histogram": dict(sorted(exit_codes.items())),
                "attempts_used_histogram": dict(sorted(attempts_dist.items())),
                "total_harness_time_sec": total_time,
                "total_input_tokens": total_in_tok,
                "total_output_tokens": total_out_tok,
            }
        except Exception as e:
            log(f"  WARNING: failed to parse {csv_path}: {e}")
            out["plain_harness"] = None
    else:
        out["plain_harness"] = None

    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            out["timings"] = meta.get("timings", {})
            out["exit_codes"] = meta.get("exit_codes", {})
            out["completed"] = bool(meta.get("completed"))
            out["error"] = meta.get("error")
            out["max_model_len"] = meta.get("max_model_len")
            out["action_max_tokens"] = meta.get("action_max_tokens")
        except json.JSONDecodeError:
            pass
    return out


def _fmt_pct(num: int, den: int) -> str:
    if not den:
        return "    -"
    return f"{100*num/den:6.2f}%"


def render_summary(per_model: list[tuple[str, Path, dict]]) -> str:
    out = []
    out.append("")
    out.append("=" * 78)
    out.append("  Multi-model plain-harness run summary")
    out.append("=" * 78)
    for name, run_dir, s in per_model:
        out.append("")
        out.append(f"  Model: {name}")
        out.append(f"  Run dir: {run_dir}")
        if s.get("completed") is False:
            out.append(f"  STATUS: FAILED — {s.get('error')}")
        elif s.get("completed") is True:
            out.append("  STATUS: completed")
        if s.get("max_model_len") is not None:
            out.append(f"  max_model_len={s['max_model_len']} "
                       f"action_max_tokens={s.get('action_max_tokens')}")
        timings = s.get("timings") or {}
        if timings:
            t_parts = [f"{k}={v:.1f}s" for k, v in timings.items() if isinstance(v, (int, float))]
            out.append("  Timings: " + ", ".join(t_parts))
        ec = s.get("exit_codes") or {}
        if ec:
            out.append(f"  Exit codes: {ec}")

        ph = s.get("plain_harness")
        if ph:
            out.append(f"  -- plain-harness: {ph['verified']}/{ph['total']} verified  "
                       f"({_fmt_pct(ph['verified'], ph['total']).strip()})")
            for k, c in sorted(ph["status_counts"].items(), key=lambda kv: -kv[1]):
                out.append(f"      status={k:<32}: {c:4d}  ({_fmt_pct(c, ph['total']).strip()})")
            if ph["exit_code_histogram"]:
                hist = ", ".join(f"{rc}:{c}" for rc, c in ph["exit_code_histogram"].items())
                out.append(f"      verus exit codes: {hist}")
            if ph["attempts_used_histogram"]:
                hist = ", ".join(f"{a}:{c}" for a, c in ph["attempts_used_histogram"].items())
                out.append(f"      attempts used:    {hist}")
            out.append(f"      total harness time: {ph['total_harness_time_sec']:.1f}s, "
                       f"in_tok={ph['total_input_tokens']}, out_tok={ph['total_output_tokens']}")
    out.append("")
    return "\n".join(out)


def render_summary_md(per_model: list[tuple[str, Path, dict]]) -> str:
    lines = ["# Multi-model plain-harness run summary", ""]
    lines += [
        "| Model | Pass | max_model_len | action_max_tokens | Status |",
        "|---|---|---|---|---|",
    ]
    for name, _run_dir, s in per_model:
        ph = s.get("plain_harness") or {}
        cell = (f"{ph['verified']}/{ph['total']} ({_fmt_pct(ph['verified'], ph['total']).strip()})"
                if ph else "—")
        mml = s.get("max_model_len") or "—"
        amt = s.get("action_max_tokens") or "—"
        status = "✓ completed" if s.get("completed") else f"✗ {s.get('error') or 'incomplete'}"
        lines.append(f"| {name} | {cell} | {mml} | {amt} | {status} |")

    for name, run_dir, s in per_model:
        lines += ["", f"## {name}", "", f"Run dir: `{run_dir}`", ""]
        ph = s.get("plain_harness")
        if ph:
            lines += [f"### plain-harness — {ph['verified']}/{ph['total']} verified", ""]
            lines += ["| Status | Count | % |", "|---|---|---|"]
            total = ph["total"]
            for k, c in sorted(ph["status_counts"].items(), key=lambda kv: -kv[1]):
                lines.append(f"| {k} | {c} | {_fmt_pct(c, total).strip()} |")
            if ph["exit_code_histogram"]:
                lines += ["", "Verus exit codes:", "", "| exit_code | count |", "|---|---|"]
                for rc, c in ph["exit_code_histogram"].items():
                    lines.append(f"| {rc} | {c} |")
            if ph["attempts_used_histogram"]:
                lines += ["", "Attempts used:", "", "| attempts_used | count |", "|---|---|"]
                for a, c in ph["attempts_used_histogram"].items():
                    lines.append(f"| {a} | {c} |")
            lines += [
                "",
                f"- total harness time: {ph['total_harness_time_sec']:.1f}s",
                f"- input tokens: {ph['total_input_tokens']}",
                f"- output tokens: {ph['total_output_tokens']}",
            ]
        timings = s.get("timings") or {}
        if timings:
            lines += ["", "Timings:", ""]
            for k, val in timings.items():
                if isinstance(val, (int, float)):
                    lines.append(f"- {k}: {val:.1f}s")
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)
        log(f"Received {sig_name}; initiating graceful shutdown (cleaning up vllm + weights)")
        raise SystemExit(128 + signum)
    for s in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(s, _handler)
        except (ValueError, OSError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to JSON run config")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help=f"Base log directory (default: {DEFAULT_LOG_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-run models even if a prior completed run exists")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of model names to run (others skipped)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved configs and subprocess commands without executing")
    args = parser.parse_args()

    _install_signal_handlers()

    cfg = load_and_validate_config(args.config)
    defaults = cfg.get("defaults", {})
    models = cfg["models"]

    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        models = [m for m in models if m["name"] in wanted]
        if not models:
            print(f"--only filter '{args.only}' matched no models", file=sys.stderr)
            return 2

    args.log_dir.mkdir(parents=True, exist_ok=True)
    batch_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log(f"Batch timestamp: {batch_ts}")
    log(f"Log dir: {args.log_dir}")
    log(f"Models: {[m['name'] for m in models]}")
    if args.dry_run:
        log("DRY RUN — no subprocess will be executed")

    results: list[ModelRunResult] = []
    for entry in models:
        try:
            r = run_one_model(entry, defaults, args.log_dir, args.force, args.dry_run)
        except Exception as e:
            log(f"  FATAL during model {entry.get('name')!r}: {e}")
            log("  " + traceback.format_exc().replace("\n", "\n  "))
            r = ModelRunResult(entry.get("name", "?"), derive_model_label(entry.get("name", "?")))
            r.error = str(e)
        results.append(r)

    if args.dry_run:
        log("Dry run complete.")
        return 0

    log("")
    log("Building summary...")
    per_model: list[tuple[str, Path, dict]] = []
    for r in results:
        if r.run_dir is None:
            per_model.append((r.name, Path("?"), {"completed": False, "error": r.error}))
            continue
        s = summarise_run(r.run_dir)
        if "completed" not in s:
            s["completed"] = r.completed
        if r.error and not s.get("error"):
            s["error"] = r.error
        per_model.append((r.name, r.run_dir, s))

    text = render_summary(per_model)
    print(text)

    summary_json_path = args.log_dir / f"_summary_{batch_ts}.json"
    summary_md_path = args.log_dir / f"_summary_{batch_ts}.md"
    summary_json_path.write_text(json.dumps(
        [{"name": n, "run_dir": str(d), **s} for n, d, s in per_model],
        indent=2, default=str,
    ))
    summary_md_path.write_text(render_summary_md(per_model))
    log(f"Summary written: {summary_json_path}")
    log(f"Summary written: {summary_md_path}")

    any_failed = any(not s.get("completed") for _n, _d, s in per_model)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
