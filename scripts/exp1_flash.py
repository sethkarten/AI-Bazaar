#!/usr/bin/env python3
"""
Run Experiment 1 commands from documentation/RUN_COMMANDS.md (EXPERIMENT 1 section).

Runs: baseline, stabilizing firm sweep (dlc=3 default dlf) 1–5, stabilizing firm sweep (dlc=3 dlf=3) 1–5.
All with seed 8, Gemini 2.5 Flash, 365 timesteps.

Usage: From project root: python scripts/exp1_flash.py
"""
import subprocess
import sys
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

# Base argv shared by all runs (no --name, no num-stabilizing-firms, no discovery-limit-firms)
_BASE = [
    "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
    "--firm-type", "LLM", "--num-goods", "1", "--num-firms", "5",
    "--consumer-type", "CES", "--num-consumers", "50",
    "--max-timesteps", "365", "--firm-initial-cash", "500",
    "--consumer-scenario", "THE_CRASH",
    "--llm", "gemini-2.5-flash", "--max-tokens", "2000",
    "--prompt-algo", "cot", "--no-diaries", "--seed", "8",
]

# (run_name, argv) — argv = ["--name", name, ...] + _BASE + optional extras
EXP1_RUNS = [
    # Baseline (no stabilizing firm, dlc=3, dlf=3)
    (
        "exp1_baseline",
        ["--name", "exp1_baseline", "--discovery-limit-consumers", "3", "--discovery-limit-firms", "3"] + _BASE,
    ),
]

# Stabilizing sweep dlc=3, default dlf (no --discovery-limit-firms)
for n in range(1, 6):
    EXP1_RUNS.append((
        f"exp1_stab_{n}",
        ["--name", f"exp1_stab_{n}", "--discovery-limit-consumers", "3", "--num-stabilizing-firms", str(n)] + _BASE,
    ))

# Stabilizing sweep dlc=3, dlf=3
for n in range(1, 6):
    EXP1_RUNS.append((
        f"exp1_stab_dlf3_{n}",
        ["--name", f"exp1_stab_dlf3_{n}", "--discovery-limit-consumers", "3", "--discovery-limit-firms", "3", "--num-stabilizing-firms", str(n)] + _BASE,
    ))


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    log(f"Experiment 1 (flash) started. Project root: {PROJECT_ROOT}")
    for name, args in EXP1_RUNS:
        log(f"Starting: {name}")
        cmd = [sys.executable, "-m", "ai_bazaar.main"] + args
        log_path = LOGS_DIR / f"{name}_{TIMESTAMP}.log"
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                p = subprocess.run(
                    cmd,
                    cwd=PROJECT_ROOT,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if p.returncode != 0:
                log(f"WARNING: {name} exited with code {p.returncode}")
        except Exception as e:
            log(f"ERROR in {name}: {e}")
        log(f"Finished: {name}")
    log("Experiment 1 (flash) completed.")


if __name__ == "__main__":
    main()
