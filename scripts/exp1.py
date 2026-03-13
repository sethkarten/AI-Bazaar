#!/usr/bin/env python3
"""
Run Experiment 1 commands from documentation/RUN_COMMANDS.md (EXPERIMENT 1 section).

Runs: baseline; stabilizing firm sweep (dlc=1) with 1, 2, 4, 5 firms × seeds 8,16,64;
      stabilizing firm sweep (dlc=3) with 1, 2, 4, 5 firms × seeds 8,16,64;
      stabilizing firm sweep (dlc=5) with 1, 2, 4, 5 firms × seeds 8,16,64.
All with Gemini 2.5 Flash, 365 timesteps.
Settings: wtp-algo=none; all non-stabilizing firms use competitive persona (default);
          THE_CRASH consumer scoring is price-only (no --crash-rep-scoring). --overhead-costs 14.

Usage: From project root:
  python scripts/exp1.py                        # all runs, sequential
  python scripts/exp1.py --workers 3            # 3 parallel runs
  python scripts/exp1.py --dlc 1 3              # only dlc=1 and dlc=3 cells
  python scripts/exp1.py --n-stab 4 5           # only n_stab=4 and n_stab=5
  python scripts/exp1.py --seeds 8              # only seed=8
  python scripts/exp1.py --run exp1_baseline exp1_stab_2_dlc3_seed8  # exact labels
  python scripts/exp1.py --skip-existing        # skip runs whose log dir already exists
  python scripts/exp1.py --list                 # print matching runs, don't execute
  python scripts/exp1.py --llm gemma3:4b --service ollama --port 11434
  Filters combine with AND: --dlc 3 --n-stab 1 2 --seeds 8 16
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
LOGS_DIR = PROJECT_ROOT / "logs" / "exp1"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
SUMMARY_LOG = LOGS_DIR / f"exp1_{TIMESTAMP}.log"

# Fixed base args shared by all runs — LLM/service/port are set at runtime via CLI args.
# wtp-algo=none; all non-stab firms competitive (default); no --crash-rep-scoring (price-only).
_BASE_FIXED = [
    "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
    "--firm-type", "LLM", "--num-goods", "1", "--num-firms", "5",
    "--consumer-type", "CES", "--num-consumers", "50",
    "--max-timesteps", "365", "--firm-initial-cash", "500",
    "--overhead-costs", "14",
    "--consumer-scenario", "THE_CRASH",
    "--wtp-algo", "none",
    "--max-tokens", "2000",
    "--prompt-algo", "cot", "--no-diaries",
]


def build_runs(base: list[str]) -> list[tuple[str, list[str], dict]]:
    """Construct the full run list using the supplied base argv."""
    runs: list[tuple[str, list[str], dict]] = []

    # ---- Baseline (no stabilizing firm, dlc=3) ----
    runs.append((
        "exp1_baseline",
        ["--name", "exp1_baseline", "--discovery-limit-consumers", "3", "--seed", "8"] + base,
        {"dlc": 3, "n_stab": 0, "seed": 8},
    ))

    # ---- Stabilizing firm sweep: dlc=1, 3, 5 × n_stab=1,2,4,5 × seeds 8,16,64 ----
    for dlc in (1, 3, 5):
        for n_stab in (1, 2, 4, 5):
            for seed in (8, 16, 64):
                log_label = f"exp1_stab_{n_stab}_dlc{dlc}_seed{seed}"
                runs.append((
                    log_label,
                    ["--name", log_label,
                     "--discovery-limit-consumers", str(dlc),
                     "--num-stabilizing-firms", str(n_stab),
                     "--seed", str(seed)] + base,
                    {"dlc": dlc, "n_stab": n_stab, "seed": seed},
                ))

    return runs


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


def filter_runs(
    runs: list[tuple[str, list[str], dict]],
    dlc_filter: list[int] | None,
    n_stab_filter: list[int] | None,
    seed_filter: list[int] | None,
    run_filter: list[str] | None,
    skip_existing: bool,
) -> list[tuple[str, list[str], dict]]:
    """Return the subset of runs matching all supplied filters (AND logic)."""
    selected = []
    for label, argv, meta in runs:
        if run_filter is not None and label not in run_filter:
            continue
        if dlc_filter is not None and meta["dlc"] not in dlc_filter:
            continue
        if n_stab_filter is not None and meta["n_stab"] not in n_stab_filter:
            continue
        if seed_filter is not None and meta["seed"] not in seed_filter:
            continue
        if skip_existing and (PROJECT_ROOT / "logs" / label).is_dir():
            continue
        selected.append((label, argv, meta))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1 (Flash)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel simulation runs (default: 1 = sequential). "
            "Keep low (2-4) to avoid Gemini API rate limits."
        ),
    )
    parser.add_argument(
        "--llm", type=str, default="gemini-2.5-flash",
        help="LLM model to use (default: gemini-2.5-flash). E.g. --llm gemini-2.0-flash gemma3:4b llama3.1:8b",
    )
    parser.add_argument(
        "--service", type=str, default=None,
        help="Model service backend (default: none / Gemini API). E.g. --service ollama --service vllm",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for local model server (default: none). E.g. --port 11434 for Ollama.",
    )
    parser.add_argument(
        "--dlc", type=int, nargs="+", metavar="N",
        help="Only run cells with these discovery-limit-consumers values (e.g. --dlc 1 3).",
    )
    parser.add_argument(
        "--n-stab", type=int, nargs="+", metavar="N", dest="n_stab",
        help="Only run cells with these n_stab values (e.g. --n-stab 1 4 5).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", metavar="N",
        help="Only run these seeds (e.g. --seeds 8 64).",
    )
    parser.add_argument(
        "--run", type=str, nargs="+", metavar="LABEL", dest="runs",
        help="Only run these exact run labels (e.g. --run exp1_baseline exp1_stab_2_dlc3_seed8).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip any run whose log directory already exists under logs/.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print matching runs and exit without executing them.",
    )
    cli = parser.parse_args()

    # Build the full base argv for this invocation.
    llm_args = ["--llm", cli.llm]
    if cli.service:
        llm_args += ["--service", cli.service]
    if cli.port:
        llm_args += ["--port", str(cli.port)]
    base = _BASE_FIXED + llm_args

    all_runs = build_runs(base)

    selected = filter_runs(
        all_runs,
        dlc_filter=cli.dlc,
        n_stab_filter=cli.n_stab,
        seed_filter=cli.seeds,
        run_filter=cli.runs,
        skip_existing=cli.skip_existing,
    )

    if cli.list:
        print(f"Matching runs ({len(selected)} / {len(all_runs)} total):")
        for label, _, meta in selected:
            print(f"  {label}  [dlc={meta['dlc']} n_stab={meta['n_stab']} seed={meta['seed']}]")
        return

    if not selected:
        print("No runs matched the supplied filters. Use --list to inspect.")
        return

    log(f"Experiment 1 (flash) started. Project root: {PROJECT_ROOT}")
    log(f"Selected runs: {len(selected)} / {len(all_runs)}  |  workers: {cli.workers}  |  llm: {cli.llm}")

    if cli.workers <= 1:
        for log_label, args, _ in selected:
            run_one(log_label, args)
    else:
        workers = min(cli.workers, len(selected))
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for log_label, args, _ in selected:
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
