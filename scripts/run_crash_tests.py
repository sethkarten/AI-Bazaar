#!/usr/bin/env python3
"""
Run the Crash Tests from documentation/RUN_COMMANDS.md (THE_CRASH section).

Includes:
  - Large (discovery limit variation): crash_test_large_1..4
  - Hetero: crash_test_hetero_1
  - Cheap: crash_test_cheap_1
  - Single: crash_test_single_1

Usage: From project root: python scripts/run_crash_tests.py
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Project root: parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "ai_bazaar").exists():
    PROJECT_ROOT = Path.cwd()
LOGS_DIR = PROJECT_ROOT / "logs" / "crash_tests"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
SUMMARY_LOG = LOGS_DIR / f"crash_tests_{TIMESTAMP}.log"

# Each entry: (run_name, full argv for ai_bazaar.main, excluding "python -m ai_bazaar.main")
CRASH_TESTS = [
    # Large: discovery limit variation
    (
        "crash_test_large_1",
        [
            "--name", "crash_test_large_1",
            "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
            "--use-env", "--firm-type", "LLM", "--num-goods", "1",
            "--num-firms", "8", "--consumer-type", "CES", "--num-consumers", "40",
            "--max-timesteps", "50", "--firm-initial-cash", "1000",
            "--consumer-scenario", "THE_CRASH", "--llm", "gemini-2.5-flash",
            "--discovery-limit-consumers", "1", "--max-tokens", "2000", "--prompt-algo", "cot",
            "--no-diaries", "--seed", "8",
        ],
    ),
    (
        "crash_test_large_2",
        [
            "--name", "crash_test_large_2",
            "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
            "--use-env", "--firm-type", "LLM", "--num-goods", "1",
            "--num-firms", "8", "--consumer-type", "CES", "--num-consumers", "40",
            "--max-timesteps", "50", "--firm-initial-cash", "1000",
            "--consumer-scenario", "THE_CRASH", "--llm", "gemini-2.5-flash",
            "--discovery-limit-consumers", "2", "--max-tokens", "2000", "--prompt-algo", "cot",
            "--no-diaries", "--seed", "8",
        ],
    ),
    (
        "crash_test_large_3",
        [
            "--name", "crash_test_large_3",
            "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
            "--use-env", "--firm-type", "LLM", "--num-goods", "1",
            "--num-firms", "8", "--consumer-type", "CES", "--num-consumers", "40",
            "--max-timesteps", "50", "--firm-initial-cash", "1000",
            "--consumer-scenario", "THE_CRASH", "--llm", "gemini-2.5-flash",
            "--discovery-limit-consumers", "5", "--max-tokens", "2000", "--prompt-algo", "cot",
            "--no-diaries", "--seed", "8",
        ],
    ),
    (
        "crash_test_large_4",
        [
            "--name", "crash_test_large_4",
            "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
            "--use-env", "--firm-type", "LLM", "--num-goods", "1",
            "--num-firms", "8", "--consumer-type", "CES", "--num-consumers", "40",
            "--max-timesteps", "50", "--firm-initial-cash", "1000",
            "--consumer-scenario", "THE_CRASH", "--llm", "gemini-2.5-flash",
            "--discovery-limit-consumers", "8", "--max-tokens", "2000", "--prompt-algo", "cot",
            "--no-diaries", "--seed", "8",
        ],
    ),
    # Hetero: heterogeneous supply costs
    (
        "crash_test_hetero_1",
        [
            "--name", "crash_test_hetero_1",
            "--use-cost-pref-gen", "--max-supply-unit-cost", "5",
            "--use-env", "--firm-type", "LLM", "--num-goods", "1",
            "--num-firms", "5", "--consumer-type", "CES", "--num-consumers", "40",
            "--max-timesteps", "30", "--firm-initial-cash", "1000",
            "--consumer-scenario", "THE_CRASH", "--llm", "gemini-2.5-flash",
            "--discovery-limit-consumers", "2", "--max-tokens", "2000", "--prompt-algo", "cot",
            "--no-diaries", "--seed", "8",
        ],
    ),
    # Cheap: low firm initial cash
    (
        "crash_test_cheap_1",
        [
            "--name", "crash_test_cheap_1",
            "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
            "--use-env", "--firm-type", "LLM", "--num-goods", "1",
            "--num-firms", "5", "--consumer-type", "CES", "--num-consumers", "40",
            "--max-timesteps", "30", "--firm-initial-cash", "250",
            "--consumer-scenario", "THE_CRASH", "--llm", "gemini-2.5-flash",
            "--discovery-limit-consumers", "2", "--max-tokens", "2000", "--prompt-algo", "cot",
            "--no-diaries", "--seed", "8",
        ],
    ),
    # Single: 1 firm
    (
        "crash_test_single_1",
        [
            "--name", "crash_test_single_1",
            "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
            "--use-env", "--firm-type", "LLM", "--num-goods", "1",
            "--num-firms", "1", "--consumer-type", "CES", "--num-consumers", "40",
            "--max-timesteps", "30", "--firm-initial-cash", "1000",
            "--consumer-scenario", "THE_CRASH", "--llm", "gemini-2.5-flash",
            "--discovery-limit-consumers", "2", "--max-tokens", "2000", "--prompt-algo", "cot",
            "--no-diaries", "--seed", "8",
        ],
    ),
]


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    log(f"Crash tests started. Project root: {PROJECT_ROOT}")
    for name, args in CRASH_TESTS:
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
    log("Crash tests completed.")


if __name__ == "__main__":
    main()
