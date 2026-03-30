#!/usr/bin/env python3
"""
Run Experiment 1 commands from documentation/RUN_COMMANDS.md (EXPERIMENT 1 section).

Runs: baseline (dlc=3, n_stab=0) × seeds 8,16,64 as exp1_*_stab_0_dlc3_seed*;
      stabilizing firm sweep (dlc=1) with 1, 2, 3, 4, 5 firms × seeds 8,16,64;
      stabilizing firm sweep (dlc=3) with 1, 2, 3, 4, 5 firms × seeds 8,16,64;
      stabilizing firm sweep (dlc=5) with 1, 2, 3, 4, 5 firms × seeds 8,16,64.
All with Gemini 2.5 Flash, 365 timesteps by default (override with --max-timesteps).
Settings: wtp-algo=none; all non-stabilizing firms use competitive persona (default);
          THE_CRASH consumer scoring is price-only (no --crash-rep-scoring). --overhead-costs 14.

Usage: From project root:
  python scripts/exp1.py                        # all runs, sequential
  python scripts/exp1.py --workers 3            # 3 parallel runs
  python scripts/exp1.py --log-prompts          # enable THE_CRASH firm prompt logging
  python scripts/exp1.py --dlc 1 3              # only dlc=1 and dlc=3 cells
  python scripts/exp1.py --n-stab 4 5           # only n_stab=4 and n_stab=5
  python scripts/exp1.py --seeds 8              # only seed=8
  python scripts/exp1.py --run exp1_gemini-2.5-flash_stab_0_dlc3_seed8 exp1_gemini-2.5-flash_stab_2_dlc3_seed8
  python scripts/exp1.py --skip-existing        # skip runs whose log dir already exists
  python scripts/exp1.py --list                 # print matching runs, don't execute
  python scripts/exp1.py --llm gemma3:4b --service ollama --port 11434
  python scripts/exp1.py --max-tokens 4096         # LLM completion cap (default: 2000)
  Filters combine with AND: --dlc 3 --n-stab 1 2 --seeds 8 16
"""
import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Project root: parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "ai_bazaar").exists():
    PROJECT_ROOT = Path.cwd()

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
# Set in main() from --llm (logs/exp1_{model_slug}/).
LOGS_DIR: Path | None = None
SUMMARY_LOG: Path | None = None

# Fixed base args shared by all runs — LLM/service/port are set at runtime via CLI args.
# wtp-algo=none; all non-stab firms competitive (default); no --crash-rep-scoring (price-only).
_BASE_FIXED = [
    "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
    "--firm-type", "LLM", "--num-goods", "1", "--num-firms", "5",
    "--consumer-type", "CES", "--num-consumers", "50",
    "--firm-initial-cash", "500",
    "--overhead-costs", "14",
    "--consumer-scenario", "THE_CRASH",
    "--wtp-algo", "none",
    "--prompt-algo", "cot", "--no-diaries",
]


def llm_filesystem_slug(llm: str) -> str:
    """Make model id safe as a single path segment (Windows + POSIX)."""
    s = llm.strip()
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = s.replace(":", "_")
    return s or "model"


def build_runs(base: list[str], name_prefix: str) -> list[tuple[str, list[str], dict]]:
    """Construct the full run list using the supplied base argv.

    Run names are ``{name_prefix}_{suffix}``; ``--log-dir`` is ``logs/{name_prefix}``
    so each simulation's run directory is under ``logs/{name_prefix}/``.
    """
    runs: list[tuple[str, list[str], dict]] = []
    log_dir_arg = f"logs/{name_prefix}"

    # ---- Baseline (no stabilizing firm, dlc=3) — same seeds as sweep ----
    for seed in (8, 16, 64):
        label = f"{name_prefix}_stab_0_dlc3_seed{seed}"
        runs.append((
            label,
            ["--name", label, "--log-dir", log_dir_arg,
             "--discovery-limit-consumers", "3",
             "--num-stabilizing-firms", "0",
             "--seed", str(seed)] + base,
            {"dlc": 3, "n_stab": 0, "seed": seed},
        ))

    # ---- Stabilizing firm sweep: dlc=1, 3, 5 × n_stab=1,2,3,4,5 × seeds 8,16,64 ----
    n_stab_values = (1, 2, 3, 4, 5)
    for dlc in (1, 3, 5):
        for n_stab in n_stab_values:
            for seed in (8, 16, 64):
                log_label = f"{name_prefix}_stab_{n_stab}_dlc{dlc}_seed{seed}"
                runs.append((
                    log_label,
                    ["--name", log_label, "--log-dir", log_dir_arg,
                     "--discovery-limit-consumers", str(dlc),
                     "--num-stabilizing-firms", str(n_stab),
                     "--seed", str(seed)] + base,
                    {"dlc": dlc, "n_stab": n_stab, "seed": seed},
                ))

    return runs


_print_lock = threading.Lock()
_progress_lock = threading.Lock()


def format_duration(seconds: float) -> str:
    """Human-readable duration for logs."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m}m {s:.0f}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def log(msg: str) -> None:
    assert SUMMARY_LOG is not None
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _print_lock:
        print(line, flush=True)
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_run_finished(
    log_label: str,
    elapsed_sec: float,
    completed_durations: list[float],
    total_runs: int,
    worker_count: int,
    batch_wall_start: float,
) -> None:
    """Log per-run duration, ETA for remaining jobs, and % of est. total time left."""
    with _progress_lock:
        completed_durations.append(elapsed_sec)
        n_done = len(completed_durations)
        n_rem = total_runs - n_done
        dur_fmt = format_duration(elapsed_sec)
        if n_rem == 0:
            log(f"Finished: {log_label} ({dur_fmt}) — all {total_runs} run(s) complete.")
            return
        avg = sum(completed_durations) / n_done
        eta_remaining = (n_rem * avg) / worker_count
        wall_elapsed = time.monotonic() - batch_wall_start
        total_est = wall_elapsed + eta_remaining
        pct_left = (100.0 * eta_remaining / total_est) if total_est > 0 else 0.0
        log(
            f"Finished: {log_label} ({dur_fmt}) | "
            f"est. {format_duration(eta_remaining)} left for {n_rem} run(s) "
            f"({pct_left:.1f}% of est. total time remaining)"
        )


def run_one(log_label: str, args: list[str]) -> tuple[int, float]:
    """Run a single simulation; returns (exit code, elapsed seconds)."""
    cmd = [sys.executable, "-m", "ai_bazaar.main"] + args
    assert LOGS_DIR is not None
    log_path = LOGS_DIR / f"{log_label}_{TIMESTAMP}.log"
    log(f"Starting: {log_label}")
    t0 = time.monotonic()
    proc: subprocess.Popen[str] | None = None
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
            proc.wait()
        elapsed = time.monotonic() - t0
        assert proc is not None
        if proc.returncode != 0:
            log(f"WARNING: {log_label} exited with code {proc.returncode}")
        return proc.returncode, elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        log(f"ERROR in {log_label}: {e}")
        return -1, elapsed


def filter_runs(
    runs: list[tuple[str, list[str], dict]],
    dlc_filter: list[int] | None,
    n_stab_filter: list[int] | None,
    seed_filter: list[int] | None,
    run_filter: list[str] | None,
    skip_existing: bool,
    name_prefix: str,
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
        if skip_existing and (PROJECT_ROOT / "logs" / name_prefix / label).is_dir():
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
        "--openrouter-provider", type=str, nargs="+", default=None, metavar="PROVIDER",
        help=(
            "Preferred OpenRouter provider order for provider/model slugs "
            "(e.g. --openrouter-provider anthropic). If omitted, OpenRouter auto-selects."
        ),
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
        help="Only run these exact run labels (e.g. --run exp1_gemini-2.5-flash_stab_0_dlc3_seed8).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip any run whose log directory already exists under logs/exp1_<model>/.",
    )
    parser.add_argument(
        "--max-timesteps", type=int, default=365, metavar="T",
        help="Episode length in timesteps (default: 365).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2000, metavar="N",
        help="Maximum LLM completion tokens per call (default: 2000).",
    )
    parser.add_argument(
        "--log-prompts", action="store_true",
        help="Enable --log-crash-firm-prompts for each run.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print matching runs and exit without executing them.",
    )
    cli = parser.parse_args()

    global LOGS_DIR, SUMMARY_LOG
    name_prefix = f"exp1_{llm_filesystem_slug(cli.llm)}"
    LOGS_DIR = PROJECT_ROOT / "logs" / name_prefix
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_LOG = LOGS_DIR / f"exp1_{TIMESTAMP}.log"

    # Build the full base argv for this invocation.
    llm_args = ["--llm", cli.llm]
    if cli.service:
        llm_args += ["--service", cli.service]
    if cli.port:
        llm_args += ["--port", str(cli.port)]
    if cli.openrouter_provider:
        llm_args += ["--openrouter-provider", *cli.openrouter_provider]
    if cli.log_prompts:
        llm_args += ["--log-crash-firm-prompts"]
    base = (
        _BASE_FIXED
        + ["--max-tokens", str(cli.max_tokens), "--max-timesteps", str(cli.max_timesteps)]
        + llm_args
    )

    all_runs = build_runs(base, name_prefix)

    selected = filter_runs(
        all_runs,
        dlc_filter=cli.dlc,
        n_stab_filter=cli.n_stab,
        seed_filter=cli.seeds,
        run_filter=cli.runs,
        skip_existing=cli.skip_existing,
        name_prefix=name_prefix,
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

    total = len(selected)
    worker_count = 1 if cli.workers <= 1 else min(cli.workers, total)
    completed_durations: list[float] = []
    batch_start = time.monotonic()

    if cli.workers <= 1:
        for log_label, args, _ in selected:
            rc, elapsed = run_one(log_label, args)
            log_run_finished(log_label, elapsed, completed_durations, total, worker_count, batch_start)
            if rc != 0:
                log(f"Run {log_label} finished with non-zero exit code {rc}")
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for log_label, args, _ in selected:
                fut = pool.submit(run_one, log_label, args)
                futures[fut] = log_label
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    rc, elapsed = fut.result()
                    log_run_finished(label, elapsed, completed_durations, total, worker_count, batch_start)
                    if rc != 0:
                        log(f"Run {label} finished with non-zero exit code {rc}")
                except Exception as e:
                    log(f"Run {label} raised an exception: {e}")

    total_wall = time.monotonic() - batch_start
    log(
        f"Experiment 1 completed. Total wall time: {format_duration(total_wall)}. "
        f"Runs directory: logs/{name_prefix}/"
    )


if __name__ == "__main__":
    main()
