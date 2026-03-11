#!/usr/bin/env python3
"""
Run Experiment 1 commands from documentation/RUN_COMMANDS.md (EXPERIMENT 1 section).

Runs: baseline; stabilizing firm sweep (dlc=1) with 1, 2, 4 firms × seeds 8,16,64;
      stabilizing firm sweep (dlc=3) with 1, 2, 4 firms × seeds 8,16,64;
      stabilizing firm sweep (dlc=5) with 1, 2, 4 firms × seeds 8,16,64.
All with Gemini 2.5 Flash, 365 timesteps. --wtp-algo defaults to none; use --wtp-algo ewtp for eWTP. --overhead-costs 14.

Usage: From project root:
  python scripts/exp1_flash.py            # sequential (default)
  python scripts/exp1_flash.py --workers 3  # 3 parallel runs
"""
import argparse
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Project root: parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "ai_bazaar").exists():
    PROJECT_ROOT = Path.cwd()
LOGS_DIR = PROJECT_ROOT / "logs" / "exp1_flash"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
SUMMARY_LOG = LOGS_DIR / f"exp1_flash_{TIMESTAMP}.log"

# Base argv shared by all runs (no --name, no num-stabilizing-firms, no discovery-limit-consumers, no seed)
# --wtp-algo defaults to none (infinite WTP); add --wtp-algo ewtp for eWTP.
_BASE = [
    "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
    "--firm-type", "LLM", "--num-goods", "1", "--num-firms", "5",
    "--consumer-type", "CES", "--num-consumers", "50",
    "--max-timesteps", "365", "--firm-initial-cash", "500",
    "--overhead-costs", "14",
    "--consumer-scenario", "THE_CRASH",
    "--llm", "gemini-2.5-flash", "--max-tokens", "2000",
    "--prompt-algo", "cot", "--no-diaries",
]

# (log_label, argv) — log_label used for log filename; argv = full list for subprocess
EXP1_RUNS: list[tuple[str, list[str]]] = []

# ---- Baseline (no stabilizing firm, dlc=3) ----
EXP1_RUNS.append((
    "exp1_baseline",
    ["--name", "exp1_baseline", "--discovery-limit-consumers", "3", "--seed", "8"] + _BASE,
))

# ---- Stabilizing firm sweep: dlc=1, default dlf. 1, 2, 4 stabilizing firms × seeds 8, 16, 64 ----
for n_stab in (1, 2, 4):
    for seed in (8, 16, 64):
        log_label = f"exp1_stab_{n_stab}_dlc1_seed{seed}"
        EXP1_RUNS.append((
            log_label,
            ["--name", log_label, "--discovery-limit-consumers", "1", "--num-stabilizing-firms", str(n_stab), "--seed", str(seed)] + _BASE,
        ))

# ---- Stabilizing firm sweep: dlc=3, default dlf. 1, 2, 4 stabilizing firms × seeds 8, 16, 64 ----
for n_stab in (1, 2, 4):
    for seed in (8, 16, 64):
        log_label = f"exp1_stab_{n_stab}_dlc3_seed{seed}"
        EXP1_RUNS.append((
            log_label,
            ["--name", log_label, "--discovery-limit-consumers", "3", "--num-stabilizing-firms", str(n_stab), "--seed", str(seed)] + _BASE,
        ))

# ---- Stabilizing firm sweep: dlc=5 (num_firms), default dlf. 1, 2, 4 stabilizing firms × seeds 8, 16, 64 ----
for n_stab in (1, 2, 4):
    for seed in (8, 16, 64):
        log_label = f"exp1_stab_{n_stab}_dlc5_seed{seed}"
        EXP1_RUNS.append((
            log_label,
            ["--name", log_label, "--discovery-limit-consumers", "5", "--num-stabilizing-firms", str(n_stab), "--seed", str(seed)] + _BASE,
        ))


_print_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _print_lock:
        print(line, flush=True)
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(log_label: str, args: list[str]) -> int:
    """Run a single simulation; returns the process exit code."""
    cmd = [sys.executable, "-m", "ai_bazaar.main"] + args
    log_path = LOGS_DIR / f"{log_label}_{TIMESTAMP}.log"
    log(f"Starting: {log_label}")
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                # Prefix parallel output so it's attributable in the terminal.
                with _print_lock:
                    print(f"[{log_label}] {line}", end="", flush=True)
            proc.wait()
        if proc.returncode != 0:
            log(f"WARNING: {log_label} exited with code {proc.returncode}")
    except Exception as e:
        log(f"ERROR in {log_label}: {e}")
        return -1
    log(f"Finished: {log_label}")
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 (Flash)")
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel simulation runs (default: 1 = sequential). "
            "Keep low (2-4) to avoid Gemini API rate limits — each run calls the "
            "LLM for every firm at every timestep."
        ),
    )
    cli = parser.parse_args()

    log(f"Experiment 1 (flash) started. Project root: {PROJECT_ROOT}")
    log(f"Total runs: {len(EXP1_RUNS)}  |  workers: {cli.workers}")

    if cli.workers <= 1:
        # Sequential: stream output live as before.
        for log_label, args in EXP1_RUNS:
            run_one(log_label, args)
    else:
        # Parallel: cap workers to number of runs.
        workers = min(cli.workers, len(EXP1_RUNS))
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for log_label, args in EXP1_RUNS:
                fut = pool.submit(run_one, log_label, args)
                futures[fut] = log_label
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    rc = fut.result()
                    if rc != 0:
                        log(f"Run {label} finished with non-zero exit code {rc}")
                except Exception as e:
                    log(f"Run {label} raised an exception: {e}")

    log("Experiment 1 (flash) completed.")


if __name__ == "__main__":
    main()
