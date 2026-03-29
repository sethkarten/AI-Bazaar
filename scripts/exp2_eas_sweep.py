#!/usr/bin/env python3
"""
Run Experiment 2 (LEMON_MARKET) across all dense open-weight models from
documentation/EAS_vs_MODEL_SIZE.md, via OpenRouter.

Buyer model is swept over DENSE_MODELS; seller model is fixed via --seller-llm
(required). Each buyer model gets its own logs/ subdirectory
(logs/exp2_{buyer_slug}/) and the full 3×2×3 sybil sweep + K=0 baseline:
  K ∈ {0, 3, 6, 9}  ×  rep_visible ∈ {True, False}  ×  seeds {8, 16, 64}
  → 24 runs per buyer model

Health score (composite, perfect = 1.0):
  score = (1 − sybil_rev_share) / 3
        + consumer_surplus_norm / 3
        + detection_premium_norm / 3
  where detection_premium is the buyer preference for honest over sybil sellers
  (sybil detection rate proxy), consumer_surplus is normalised by the K=0
  baseline, and sybil_rev_share ∈ [0, 1].

A single ThreadPoolExecutor dispatches all runs across all models, so --workers
controls the total number of parallel simulations (not per-model parallelism).

Usage:
  python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it
  python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --workers 4
  python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --skip-existing
  python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --list
  python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --models llama-3.2-3b
  python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --k 0 3
  python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --rep-visible 1
  python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --seeds 8
  python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --max-timesteps 50
"""

import argparse
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

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# ── Dense open-weight models with include=1 (EAS_vs_MODEL_SIZE.md) ─────────
# (display_name, params_B, openrouter_model_id)
DENSE_MODELS = [
    ("Llama 3.2 3B",        3.0,   "meta-llama/llama-3.2-3b-instruct"),
    ("Gemma 3 4B",          4.0,   "google/gemma-3-4b-it"),
    ("Mistral 7B",          7.3,   "mistralai/mistral-7b-instruct-v0.1"),
    ("Llama 3.1 8B",        8.0,   "meta-llama/llama-3.1-8b-instruct"),
    ("Qwen3 8B",            8.2,   "qwen/qwen3-8b"),
    ("Gemma 3 12B",         12.0,  "google/gemma-3-12b-it"),
    ("Phi-4",               14.0,  "microsoft/phi-4"),
    # ("DS-R1-D 14B",       14.0,  "deepseek/deepseek-r1-distill-qwen-14b"),  # removed from OpenRouter
    ("Mistral Small 24B",   24.0,  "mistralai/mistral-small-3.1-24b-instruct"),
    ("Gemma 3 27B",         27.0,  "google/gemma-3-27b-it"),
    ("DS-R1-D 32B",         32.0,  "deepseek/deepseek-r1-distill-qwen-32b"),
    ("Llama 3.3 70B",       70.0,  "meta-llama/llama-3.3-70b-instruct"),
    ("Llama 3.1 70B",       70.0,  "meta-llama/llama-3.1-70b-instruct"),
    ("DS-R1-D 70B",         70.0,  "deepseek/deepseek-r1-distill-llama-70b"),
    ("Nemotron 70B",        70.0,  "nvidia/llama-3.1-nemotron-70b-instruct"),
    ("Qwen2.5 72B",         72.0,  "qwen/qwen-2.5-72b-instruct"),
    # ("Llama 3.1 405B",    405.0, "meta-llama/llama-3.1-405b-instruct"),     # removed from OpenRouter
    ("Hermes 3 405B",       405.0, "nousresearch/hermes-3-llama-3.1-405b"),
    ("Hermes 4 405B",       405.0, "nousresearch/hermes-4-405b"),
]

NUM_TOTAL_SELLERS = 12   # total seller slots (honest + sybil) — constant
SEEDS     = (8, 16, 64)
K_VALUES  = (3, 6, 9)    # sybil cluster sizes; K=0 → baseline (no sybil)
RHO_MIN   = 0.3

# Fixed simulation args (matches exp2.py _BASE_FIXED)
_BASE_FIXED = [
    "--consumer-scenario", "LEMON_MARKET",
    "--firm-type", "LLM",
    "--num-buyers", "12",
    "--seller-type", "LLM",
    "--reputation-initial", "0.8",
    "--sybil-rho-min", str(RHO_MIN),
    "--discovery-limit-consumers", "3",
    "--no-diaries",
]

# ── Shared state ─────────────────────────────────────────────────────────────
_print_lock    = threading.Lock()
_progress_lock = threading.Lock()
SUMMARY_LOG: Path | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def llm_filesystem_slug(llm: str) -> str:
    s = llm.strip()
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = s.replace(":", "_")
    return s or "model"


def seller_personas_spec(num_honest: int) -> str:
    """Distribute honest sellers evenly across four persona styles."""
    styles = ["standard", "detailed", "terse", "optimistic"]
    base, remainder = divmod(num_honest, len(styles))
    counts = [base + (1 if i < remainder else 0) for i in range(len(styles))]
    return ",".join(f"{style}:{count}" for style, count in zip(styles, counts) if count > 0)


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
        avg       = sum(completed) / n_done
        eta       = (n_rem * avg) / max(workers, 1)
        wall_el   = time.monotonic() - wall_start
        total_est = wall_el + eta
        pct_left  = 100.0 * eta / total_est if total_est > 0 else 0.0
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
        return proc.returncode, elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        log(f"ERROR in {label}: {e}")
        return -1, elapsed


# ── Run construction ──────────────────────────────────────────────────────────

def build_runs_for_model(or_id: str, base: list[str]) -> list[tuple[str, list[str], dict]]:
    """Build the full 24-run list for one model (K × rep_visible × seed)."""
    name_prefix = f"exp2_{llm_filesystem_slug(or_id)}"
    log_dir_arg = f"logs/{name_prefix}"
    runs = []

    # ---- Baseline: K=0 × rep_visible ∈ {True, False} ----
    personas_k0 = seller_personas_spec(NUM_TOTAL_SELLERS)
    for rep_visible in (True, False):
        rep_tag = "rep1" if rep_visible else "rep0"
        extra   = [] if rep_visible else ["--no-buyer-rep"]
        for seed in SEEDS:
            label = f"{name_prefix}_k0_{rep_tag}_seed{seed}"
            runs.append((label, [
                "--name", label, "--log-dir", log_dir_arg,
                "--num-sellers", str(NUM_TOTAL_SELLERS),
                "--sybil-cluster-size", "0",
                "--seller-personas", personas_k0,
                "--seed", str(seed),
            ] + extra + base, {
                "or_id": or_id, "name_prefix": name_prefix,
                "k": 0, "rep_visible": rep_visible, "seed": seed,
            }))

    # ---- Sybil grid: K × rep_visible × seed ----
    for k in K_VALUES:
        num_honest = NUM_TOTAL_SELLERS - k
        personas   = seller_personas_spec(num_honest)
        saturation = k / NUM_TOTAL_SELLERS
        for rep_visible in (True, False):
            rep_tag = "rep1" if rep_visible else "rep0"
            extra   = [] if rep_visible else ["--no-buyer-rep"]
            for seed in SEEDS:
                label = f"{name_prefix}_k{k}_{rep_tag}_seed{seed}"
                runs.append((label, [
                    "--name", label, "--log-dir", log_dir_arg,
                    "--num-sellers", str(NUM_TOTAL_SELLERS),
                    "--sybil-cluster-size", str(k),
                    "--seller-personas", personas,
                    "--seed", str(seed),
                ] + extra + base, {
                    "or_id": or_id, "name_prefix": name_prefix,
                    "k": k, "rep_visible": rep_visible, "seed": seed,
                    "saturation": saturation,
                }))

    return runs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global SUMMARY_LOG

    ap = argparse.ArgumentParser(
        description="Exp2 LEMON_MARKET sweep over all dense open-weight models (OpenRouter).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--workers",      type=int, default=1,
                    help="Total parallel simulation runs across all models (default: 1).")
    ap.add_argument("--openrouter-provider", type=str, nargs="+", default=None, metavar="P",
                    help="Preferred OpenRouter provider(s) for all agents.")
    ap.add_argument("--buyer-openrouter-provider", type=str, nargs="+", default=None, metavar="P",
                    dest="buyer_openrouter_provider",
                    help="Preferred OpenRouter provider(s) for buyer agents. Falls back to --openrouter-provider.")
    ap.add_argument("--seller-openrouter-provider", type=str, nargs="+", default=None, metavar="P",
                    dest="seller_openrouter_provider",
                    help="Preferred OpenRouter provider(s) for seller/sybil agents. Falls back to --openrouter-provider.")
    ap.add_argument("--models",       type=str, nargs="+", default=None, metavar="SUBSTR",
                    help="Only run models whose OR ID or display name contains any of these substrings.")
    ap.add_argument("--k",            type=int, nargs="+", metavar="K",
                    help="Only run cells with these K values (0 = baseline). E.g. --k 0 3 6")
    ap.add_argument("--rep-visible",  type=int, nargs="+", metavar="0|1", dest="rep_visible",
                    help="Only run cells with this rep_visible setting (1=visible, 0=hidden).")
    ap.add_argument("--seeds",        type=int, nargs="+", metavar="N",
                    help="Only run these seeds.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs whose log directory already exists.")
    ap.add_argument("--max-timesteps", type=int, default=50, metavar="T",
                    help="Episode length in timesteps (default: 50).")
    ap.add_argument("--max-tokens",    type=int, default=2000, metavar="N")
    ap.add_argument("--prompt-algo",   type=str, default="cot",
                    choices=["io", "cot", "sc"],
                    help="Prompt algorithm for all runs (default: cot).")
    ap.add_argument("--seller-llm",     type=str, required=True, metavar="MODEL",
                    help="Fixed LLM for honest sellers and sybil principal (required).")
    ap.add_argument("--buyer-service",  type=str, default=None, metavar="SVC", dest="buyer_service",
                    help="Service backend for buyer agents (vllm|ollama). Falls back to --service.")
    ap.add_argument("--seller-service", type=str, default=None, metavar="SVC", dest="seller_service",
                    help="Service backend for seller agents (vllm|ollama). Falls back to --service.")
    ap.add_argument("--service",        type=str, default=None, metavar="SVC",
                    help="Service backend for all agents (vllm|ollama).")
    ap.add_argument("--buyer-port",     type=int, default=None, metavar="N", dest="buyer_port",
                    help="Port for buyer LLM service. Falls back to --port.")
    ap.add_argument("--seller-port",    type=int, default=None, metavar="N", dest="seller_port",
                    help="Port for seller LLM service. Falls back to --port.")
    ap.add_argument("--port",           type=int, default=None, metavar="N",
                    help="Port for local model server.")
    ap.add_argument("--log-buyer-prompts",  action="store_true",
                    help="Log buyer LLM prompts/responses per run.")
    ap.add_argument("--log-seller-prompts", action="store_true",
                    help="Log seller/sybil LLM prompts/responses per run.")
    ap.add_argument("--list",          action="store_true",
                    help="Print matching runs and exit.")
    cli = ap.parse_args()

    # ── Set up summary log ───────────────────────────────────────────────
    sweep_logs_dir = PROJECT_ROOT / "logs"
    sweep_logs_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_LOG = sweep_logs_dir / f"exp2_eas_sweep_{TIMESTAMP}.log"

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
        "--max-tokens",    str(cli.max_tokens),
        "--max-timesteps", str(cli.max_timesteps),
        "--prompt-algo",   cli.prompt_algo,
    ]
    if cli.openrouter_provider:
        extra += ["--openrouter-provider", *cli.openrouter_provider]
    if cli.buyer_openrouter_provider:
        extra += ["--buyer-openrouter-provider", *cli.buyer_openrouter_provider]
    if cli.seller_openrouter_provider:
        extra += ["--seller-openrouter-provider", *cli.seller_openrouter_provider]
    if cli.service:
        extra += ["--service", cli.service]
    if cli.buyer_service:
        extra += ["--buyer-service", cli.buyer_service]
    if cli.seller_service:
        extra += ["--seller-service", cli.seller_service]
    if cli.port:
        extra += ["--port", str(cli.port)]
    if cli.buyer_port:
        extra += ["--buyer-port", str(cli.buyer_port)]
    if cli.seller_port:
        extra += ["--seller-port", str(cli.seller_port)]
    if cli.log_buyer_prompts:
        extra += ["--log-buyer-prompts"]
    if cli.log_seller_prompts:
        extra += ["--log-seller-prompts"]

    # ── Build all runs across all models ─────────────────────────────────
    # or_id is the buyer model; seller is fixed via --seller-llm.
    all_runs: list[tuple[str, list[str], dict]] = []
    for _name, _params, or_id in active_models:
        base = _BASE_FIXED + extra + [
            "--llm",        or_id,            # fallback; also labels the run
            "--buyer-llm",  or_id,
            "--seller-llm", cli.seller_llm,
        ]
        all_runs.extend(build_runs_for_model(or_id, base))

    # ── Apply filters ─────────────────────────────────────────────────────
    rep_visible_filter = None
    if cli.rep_visible is not None:
        rep_visible_filter = [bool(v) for v in cli.rep_visible]

    selected = []
    for label, argv, meta in all_runs:
        if cli.k              is not None and meta["k"]           not in cli.k:
            continue
        if rep_visible_filter is not None and meta["rep_visible"] not in rep_visible_filter:
            continue
        if cli.seeds          is not None and meta["seed"]        not in cli.seeds:
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
            k   = meta["k"]
            sat = f"sat={meta.get('saturation', 0):.0%}" if k > 0 else "no-sybil"
            rep = "rep=visible" if meta["rep_visible"] else "rep=hidden"
            print(f"    {label:60s}  [K={k:2d}  {sat:10s}  {rep}  seed={meta['seed']}]")
        return

    if not selected:
        print("No runs matched the supplied filters. Use --list to inspect.")
        return

    log(f"EAS sweep (exp2) started. Project root: {PROJECT_ROOT}")
    log(f"Buyer models: {len(active_models)}  |  Total runs: {len(selected)}  |  Workers: {cli.workers}")
    log(f"Fixed seller LLM: {cli.seller_llm}")
    log(f"Grid: K∈{{0,{K_VALUES}}}  rep_visible∈{{True,False}}  seeds={SEEDS}  rho_min={RHO_MIN}")
    log(f"Total sellers fixed at {NUM_TOTAL_SELLERS}; honest = {NUM_TOTAL_SELLERS} - K")
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
    total      = len(selected)
    workers    = max(1, min(cli.workers, total))
    completed: list[float] = []
    wall_start = time.monotonic()

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
        f"EAS sweep (exp2) complete. Wall time: {format_duration(time.monotonic() - wall_start)}. "
        f"Summary log: {SUMMARY_LOG}"
    )


if __name__ == "__main__":
    main()
