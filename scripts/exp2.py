#!/usr/bin/env python3
"""
Run Experiment 2 (LEMON_MARKET) — baseline and Sybil-only conditions (no Guardian agent).

Run matrix:
  - Baseline (no sybil): 3 seeds {8, 16, 64}
  - Sybil sweep:  n_sybil ∈ {3, 6, 9, 12} × rho_min ∈ {0.3} × seeds {8, 16, 64}
  Total: 15 runs

Fixed settings: 12 total seller slots (honest = 12 - sybil_cluster_size), 12 LLM buyers,
  100 timesteps, LEMON_MARKET, seller-type=LLM, reputation-alpha=0.9,
  reputation-initial=0.8, discovery-limit-consumers=5, max-tokens=2000,
  prompt-algo=cot, no-diaries, heterogeneous seller personas (standard/detailed/terse/optimistic).

Usage: From project root:
  python scripts/exp2.py                        # all runs, sequential
  python scripts/exp2.py --workers 3            # 3 parallel runs
  python scripts/exp2.py --n-sybil 0            # only baseline (no sybil)
  python scripts/exp2.py --n-sybil 3 6          # only those sybil cells
  python scripts/exp2.py --rho-min 0.3          # only rho_min=0.3 cells
  python scripts/exp2.py --seeds 8              # only seed=8
  python scripts/exp2.py --run exp2_baseline_seed8 exp2_sybil_6_rho0.3_seed16
  python scripts/exp2.py --skip-existing        # skip runs whose log dir already exists
  python scripts/exp2.py --list                 # print matching runs, don't execute
  python scripts/exp2.py --llm gemma3:4b --service ollama --port 11434
  Filters combine with AND: --n-sybil 6 --seeds 8 16
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
LOGS_DIR = PROJECT_ROOT / "logs" / "exp2"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
SUMMARY_LOG = LOGS_DIR / f"exp2_{TIMESTAMP}.log"

# Fixed base args shared by all runs.
# 12 total seller slots (honest = 12 - sybil_cluster_size); 12 LLM buyers; 100 timesteps.
# Heterogeneous seller personas: 3 each of standard, detailed, terse, optimistic.
_BASE_FIXED = [
    "--consumer-scenario", "LEMON_MARKET",
    "--firm-type", "LLM",
    "--num-sellers", "12",
    "--num-buyers", "12",
    "--max-timesteps", "40",
    "--seller-type", "LLM",
    "--reputation-alpha", "0.9",
    "--reputation-initial", "0.8",
    "--discovery-limit-consumers", "5",
    "--max-tokens", "2000",
    "--prompt-algo", "cot",
    "--no-diaries",
    "--seller-personas", "standard:3,detailed:3,terse:3,optimistic:3",
]


def build_runs(base: list[str]) -> list[tuple[str, list[str], dict]]:
    """Construct the full run list using the supplied base argv."""
    runs: list[tuple[str, list[str], dict]] = []

    # ---- Baseline (no sybil) ----
    for seed in (8, 16, 64):
        label = f"exp2_baseline_seed{seed}"
        runs.append((
            label,
            ["--name", label,
             "--sybil-cluster-size", "0",
             "--seed", str(seed)] + base,
            {"n_sybil": 0, "rho_min": None, "seed": seed},
        ))

    # ---- Sybil sweep (no guardian): n_sybil × rho_min × seeds ----
    for n_sybil in (3, 6, 9, 12):
        for rho_min in (0.3,):
            for seed in (8, 16, 64):
                label = f"exp2_sybil_{n_sybil}_rho{rho_min}_seed{seed}"
                runs.append((
                    label,
                    ["--name", label,
                     "--sybil-cluster-size", str(n_sybil),
                     "--sybil-rho-min", str(rho_min),
                     "--seed", str(seed)] + base,
                    {"n_sybil": n_sybil, "rho_min": rho_min, "seed": seed},
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
    n_sybil_filter: list[int] | None,
    rho_min_filter: list[float] | None,
    seed_filter: list[int] | None,
    run_filter: list[str] | None,
    skip_existing: bool,
) -> list[tuple[str, list[str], dict]]:
    """Return the subset of runs matching all supplied filters (AND logic)."""
    selected = []
    for label, argv, meta in runs:
        if run_filter is not None and label not in run_filter:
            continue
        if n_sybil_filter is not None and meta["n_sybil"] not in n_sybil_filter:
            continue
        if rho_min_filter is not None:
            # baseline runs have rho_min=None; include them only if 0 is in n_sybil_filter
            if meta["rho_min"] not in rho_min_filter:
                continue
        if seed_filter is not None and meta["seed"] not in seed_filter:
            continue
        if skip_existing and (PROJECT_ROOT / "logs" / label).is_dir():
            continue
        selected.append((label, argv, meta))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 2 — LEMON_MARKET baseline and Sybil sweep (no Guardian)",
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
        help="LLM model to use (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--service", type=str, default=None,
        help="Model service backend (default: none / Gemini API). E.g. --service ollama",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for local model server (default: none). E.g. --port 11434 for Ollama.",
    )
    parser.add_argument(
        "--n-sybil", type=int, nargs="+", metavar="N", dest="n_sybil",
        help="Only run cells with these sybil cluster sizes (e.g. --n-sybil 0 3 6 9 12). 0 = baseline.",
    )
    parser.add_argument(
        "--rho-min", type=float, nargs="+", metavar="F", dest="rho_min",
        help="Only run cells with these rho_min values (e.g. --rho-min 0.3).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", metavar="N",
        help="Only run these seeds (e.g. --seeds 8 64).",
    )
    parser.add_argument(
        "--run", type=str, nargs="+", metavar="LABEL", dest="runs",
        help="Only run these exact run labels.",
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

    llm_args = ["--llm", cli.llm]
    if cli.service:
        llm_args += ["--service", cli.service]
    if cli.port:
        llm_args += ["--port", str(cli.port)]
    base = _BASE_FIXED + llm_args

    all_runs = build_runs(base)

    selected = filter_runs(
        all_runs,
        n_sybil_filter=cli.n_sybil,
        rho_min_filter=cli.rho_min,
        seed_filter=cli.seeds,
        run_filter=cli.runs,
        skip_existing=cli.skip_existing,
    )

    if cli.list:
        print(f"Matching runs ({len(selected)} / {len(all_runs)} total):")
        for label, _, meta in selected:
            rho_str = f"rho_min={meta['rho_min']}" if meta["rho_min"] is not None else "no-sybil"
            print(f"  {label}  [n_sybil={meta['n_sybil']} {rho_str} seed={meta['seed']}]")
        return

    if not selected:
        print("No runs matched the supplied filters. Use --list to inspect.")
        return

    log(f"Experiment 2 started. Project root: {PROJECT_ROOT}")
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

    log("Experiment 2 completed.")


if __name__ == "__main__":
    main()
