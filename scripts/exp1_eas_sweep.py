#!/usr/bin/env python3
"""
Run Experiment 1 across all dense open-weight models (include=1) from
documentation/EAS_vs_MODEL_SIZE.md, via OpenRouter.

Each model gets its own logs/ subdirectory (logs/exp1_{model_slug}/) and a
dlc=3-only sweep: baseline (k=0) + k={3,5} x seeds={8,16,64}.

A single ThreadPoolExecutor dispatches all runs across all models, so --workers
controls the total number of parallel simulations (not per-model parallelism).

Usage:
  python scripts/exp1_eas_sweep.py                         # all models, sequential
  python scripts/exp1_eas_sweep.py --workers 4             # 4 parallel runs
  python scripts/exp1_eas_sweep.py --skip-existing         # skip already-done runs
  python scripts/exp1_eas_sweep.py --list                  # print matching runs, no exec
  python scripts/exp1_eas_sweep.py --models llama-3.2-3b mistral-7b  # model substring filter
  python scripts/exp1_eas_sweep.py --dlc 3 --n-stab 3     # cell filters (same as exp1.py)
  python scripts/exp1_eas_sweep.py --seeds 8               # seed filter
  python scripts/exp1_eas_sweep.py --max-tokens 2000 --max-timesteps 365
  python scripts/exp1_eas_sweep.py --openrouter-provider Together
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "ai_bazaar").exists():
    PROJECT_ROOT = Path.cwd()

_DEBUG_AGENT_LOG = PROJECT_ROOT / "debug-90a41f.log"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        rec = {
            "sessionId": "90a41f",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "timestamp": int(time.time() * 1000),
            "data": data,
        }
        with open(_DEBUG_AGENT_LOG, "a", encoding="utf-8") as df:
            df.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass
    # #endregion

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# ── Dense open-weight models with include=1 (EAS_vs_MODEL_SIZE.md) ─────────
# (display_name, params_B, openrouter_model_id)
DENSE_MODELS = [
    ("Llama 3.2 3B",        3.0,   "meta-llama/llama-3.2-3b-instruct"),
    ("Gemma 3 4B",          4.0,   "google/gemma-3-4b-it"),
    ("Mistral 7B",          7.3,   "mistralai/mistral-7b-instruct"),
    ("Llama 3.1 8B",        8.0,   "meta-llama/llama-3.1-8b-instruct"),
    ("Qwen3 8B",            8.2,   "qwen/qwen3-8b"),
    ("Gemma 3 12B",         12.0,  "google/gemma-3-12b-it"),
    ("Phi-4",               14.0,  "microsoft/phi-4"),
    ("DS-R1-D 14B",         14.0,  "deepseek/deepseek-r1-distill-qwen-14b"),
    ("Mistral Small 24B",   24.0,  "mistralai/mistral-small-3.1-24b-instruct"),
    ("Gemma 3 27B",         27.0,  "google/gemma-3-27b-it"),
    ("OLMo 2 32B",          32.0,  "allenai/olmo-2-32b-instruct"),
    ("OLMo 3.1 32B Think",  32.0,  "allenai/olmo-3.1-32b-think"),
    ("DS-R1-D 32B",         32.0,  "deepseek/deepseek-r1-distill-qwen-32b"),
    ("Llama 3.3 70B",       70.0,  "meta-llama/llama-3.3-70b-instruct"),
    ("Llama 3.1 70B",       70.0,  "meta-llama/llama-3.1-70b-instruct"),
    ("DS-R1-D 70B",         70.0,  "deepseek/deepseek-r1-distill-llama-70b"),
    ("Nemotron 70B",        70.0,  "nvidia/llama-3.1-nemotron-70b-instruct"),
    ("Qwen2.5 72B",         72.0,  "qwen/qwen-2.5-72b-instruct"),
    ("Llama 3.1 405B",      405.0, "meta-llama/llama-3.1-405b-instruct"),
    ("Hermes 3 405B",       405.0, "nousresearch/hermes-3-llama-3.1-405b"),
    ("Hermes 4 405B",       405.0, "nousresearch/hermes-4-405b"),
]

# Fixed simulation args (matches exp1.py _BASE_FIXED)
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

# ── Shared state ────────────────────────────────────────────────────────────
_print_lock    = threading.Lock()
_progress_lock = threading.Lock()
SUMMARY_LOG: Path | None = None


# ── Helpers (mirrors exp1.py) ───────────────────────────────────────────────

def llm_filesystem_slug(llm: str) -> str:
    s = llm.strip()
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = s.replace(":", "_")
    return s or "model"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = int(seconds // 60), seconds % 60
        return f"{m}m {s:.0f}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _print_lock:
        print(line, flush=True)
    if SUMMARY_LOG is not None:
        with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_run_finished(
    label: str, elapsed: float,
    completed: list[float], total: int,
    workers: int, wall_start: float,
) -> None:
    with _progress_lock:
        completed.append(elapsed)
        n_done = len(completed)
        n_rem  = total - n_done
        if n_rem == 0:
            log(f"Finished: {label} ({format_duration(elapsed)}) — all {total} run(s) done.")
            return
        avg        = sum(completed) / n_done
        eta        = (n_rem * avg) / max(workers, 1)
        wall_el    = time.monotonic() - wall_start
        total_est  = wall_el + eta
        pct_left   = 100.0 * eta / total_est if total_est > 0 else 0.0
        log(
            f"Finished: {label} ({format_duration(elapsed)}) | "
            f"est. {format_duration(eta)} left for {n_rem} run(s) "
            f"({pct_left:.1f}% remaining)"
        )


def run_one(label: str, argv: list[str], model_logs_dir: Path) -> tuple[int, float]:
    cmd      = [sys.executable, "-m", "ai_bazaar.main"] + argv
    log_path = model_logs_dir / f"{label}_{TIMESTAMP}.log"
    log(f"Starting: {label}")
    t0 = time.monotonic()
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
            proc.wait()
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            log(f"WARNING: {label} exited with code {proc.returncode}")
            # #region agent log
            _agent_debug_log(
                "H3",
                "exp1_eas_sweep.py:run_one:nonzero",
                "child ai_bazaar.main exited non-zero",
                {"label": label, "returncode": proc.returncode, "log_path": str(log_path)},
            )
            # #endregion
        return proc.returncode, elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        log(f"ERROR in {label}: {e}")
        # #region agent log
        _agent_debug_log(
            "H3",
            "exp1_eas_sweep.py:run_one:exception",
            "Popen/streaming failed",
            {"label": label, "exc_type": type(e).__name__, "exc_msg": str(e)},
        )
        # #endregion
        return -1, elapsed


# ── Run construction ────────────────────────────────────────────────────────

def build_runs_for_model(or_id: str, base: list[str]) -> list[tuple[str, list[str], dict]]:
    """Build the full run list for one model (same sweep as exp1.py)."""
    name_prefix  = f"exp1_{llm_filesystem_slug(or_id)}"
    log_dir_arg  = f"logs/{name_prefix}"
    runs = []

    # Baseline
    bl = f"{name_prefix}_baseline"
    runs.append((bl, [
        "--name", bl, "--log-dir", log_dir_arg,
        "--discovery-limit-consumers", "3", "--seed", "8",
    ] + base, {"or_id": or_id, "name_prefix": name_prefix, "dlc": 3, "n_stab": 0, "seed": 8}))

    # Sweep
    for dlc in (3,):
        for n_stab in (3, 5):
            for seed in (8, 16, 64):
                label = f"{name_prefix}_stab_{n_stab}_dlc{dlc}_seed{seed}"
                runs.append((label, [
                    "--name", label, "--log-dir", log_dir_arg,
                    "--discovery-limit-consumers", str(dlc),
                    "--num-stabilizing-firms", str(n_stab),
                    "--seed", str(seed),
                ] + base, {"or_id": or_id, "name_prefix": name_prefix,
                           "dlc": dlc, "n_stab": n_stab, "seed": seed}))
    return runs


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    global SUMMARY_LOG

    ap = argparse.ArgumentParser(
        description="Exp1 sweep over all dense open-weight models (OpenRouter).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--workers",      type=int, default=1,
                    help="Total parallel simulation runs across all models (default: 1).")
    ap.add_argument("--openrouter-provider", type=str, nargs="+", default=None, metavar="P",
                    help="Preferred OpenRouter provider(s) (e.g. --openrouter-provider Together).")
    ap.add_argument("--models",       type=str, nargs="+", default=None, metavar="SUBSTR",
                    help="Only run models whose OR ID or display name contains any of these substrings.")
    ap.add_argument("--dlc",          type=int, nargs="+", metavar="N",
                    help="Only run cells with these dlc values.")
    ap.add_argument("--n-stab",       type=int, nargs="+", metavar="N", dest="n_stab",
                    help="Only run cells with these n_stab values.")
    ap.add_argument("--seeds",        type=int, nargs="+", metavar="N",
                    help="Only run these seeds.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs whose log directory already exists.")
    ap.add_argument("--max-timesteps", type=int, default=365, metavar="T")
    ap.add_argument("--max-tokens",    type=int, default=2000, metavar="N")
    ap.add_argument("--log-prompts",   action="store_true",
                    help="Enable --log-crash-firm-prompts for each run.")
    ap.add_argument("--list",          action="store_true",
                    help="Print matching runs and exit.")
    cli = ap.parse_args()

    # ── Set up summary log ───────────────────────────────────────────────
    sweep_logs_dir = PROJECT_ROOT / "logs"
    sweep_logs_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_LOG = sweep_logs_dir / f"exp1_eas_sweep_{TIMESTAMP}.log"

    # ── Filter model list ────────────────────────────────────────────────
    active_models = DENSE_MODELS
    if cli.models:
        filters = [f.lower() for f in cli.models]
        active_models = [
            (name, params, or_id)
            for name, params, or_id in DENSE_MODELS
            if any(f in or_id.lower() or f in name.lower() for f in filters)
        ]
        if not active_models:
            print(f"No models matched --models filter: {cli.models}")
            print("Available models:")
            for name, _, or_id in DENSE_MODELS:
                print(f"  {or_id}  ({name})")
            return

    # ── Build base argv ──────────────────────────────────────────────────
    extra = [
        "--max-tokens", str(cli.max_tokens),
        "--max-timesteps", str(cli.max_timesteps),
    ]
    if cli.openrouter_provider:
        extra += ["--openrouter-provider", *cli.openrouter_provider]
    if cli.log_prompts:
        extra += ["--log-crash-firm-prompts"]

    # ── Build all runs across all models ─────────────────────────────────
    all_runs: list[tuple[str, list[str], dict]] = []
    for name, params, or_id in active_models:
        base = _BASE_FIXED + extra + ["--llm", or_id]
        all_runs.extend(build_runs_for_model(or_id, base))

    # ── Apply cell / seed / skip filters ─────────────────────────────────
    selected = []
    for label, argv, meta in all_runs:
        if cli.dlc   is not None and meta["dlc"]    not in cli.dlc:
            continue
        if cli.n_stab is not None and meta["n_stab"] not in cli.n_stab:
            continue
        if cli.seeds  is not None and meta["seed"]   not in cli.seeds:
            continue
        if cli.skip_existing:
            run_dir = PROJECT_ROOT / "logs" / meta["name_prefix"] / label
            if run_dir.is_dir():
                continue
        selected.append((label, argv, meta))

    if cli.list:
        print(f"Matching runs: {len(selected)} / {len(all_runs)} total "
              f"across {len(active_models)} model(s)")
        current_prefix = None
        for label, _, meta in selected:
            if meta["name_prefix"] != current_prefix:
                current_prefix = meta["name_prefix"]
                print(f"\n  [{current_prefix}]")
            print(f"    {label}  [dlc={meta['dlc']} n_stab={meta['n_stab']} seed={meta['seed']}]")
        return

    if not selected:
        print("No runs matched the supplied filters. Use --list to inspect.")
        return

    log(f"EAS sweep started. Project root: {PROJECT_ROOT}")
    log(f"Models: {len(active_models)}  |  Total runs: {len(selected)}  |  Workers: {cli.workers}")
    for name, _, or_id in active_models:
        log(f"  {or_id}  ({name})")

    # Ensure per-model log dirs exist
    model_log_dirs: dict[str, Path] = {}
    for _, _, meta in selected:
        np_ = meta["name_prefix"]
        if np_ not in model_log_dirs:
            d = PROJECT_ROOT / "logs" / np_
            d.mkdir(parents=True, exist_ok=True)
            model_log_dirs[np_] = d

    # ── Execute ───────────────────────────────────────────────────────────
    total        = len(selected)
    workers      = max(1, min(cli.workers, total))
    completed: list[float] = []
    wall_start   = time.monotonic()

    if workers == 1:
        for label, argv, meta in selected:
            rc, elapsed = run_one(label, argv, model_log_dirs[meta["name_prefix"]])
            log_run_finished(label, elapsed, completed, total, 1, wall_start)
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for label, argv, meta in selected:
                fut = pool.submit(run_one, label, argv, model_log_dirs[meta["name_prefix"]])
                futures[fut] = (label, meta)
            for fut in as_completed(futures):
                label, meta = futures[fut]
                try:
                    rc, elapsed = fut.result()
                    log_run_finished(label, elapsed, completed, total, workers, wall_start)
                    if rc != 0:
                        log(f"Run {label} exited with code {rc}")
                except Exception as e:
                    log(f"Run {label} raised an exception: {e}")

    log(
        f"EAS sweep complete. Wall time: {format_duration(time.monotonic() - wall_start)}. "
        f"Summary log: {SUMMARY_LOG}"
    )


if __name__ == "__main__":
    main()
