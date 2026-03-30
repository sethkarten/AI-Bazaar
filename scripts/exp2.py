#!/usr/bin/env python3
"""
Run Experiment 2 (LEMON_MARKET) — 3×3×2 sybil sweep + baseline.

Run matrix
----------
Baseline (no sybil):
  K=0 × rep_visible ∈ {True, False} × seeds {8, 16, 64} → 6 runs

Sybil grid:
  K ∈ {3, 6, 9} × rep_visible ∈ {True, False} × seeds {8, 16, 64} → 18 runs

Total: 24 runs

Design notes
------------
- Total sellers always = 12; honest = 12 - K.
  Sybil saturation: K=3 → 25%, K=6 → 50%, K=9 → 75% of listings.
- rep_visible=False passes --no-buyer-rep (buyers see only description and price).
- Seller personas fixed: evenly distributed across standard/detailed/terse/optimistic.
- rho_min fixed at 0.3 (sybil identity retired when rolling window rep drops below 0.3).
- Run names are {name_prefix}_{suffix}; --log-dir logs/{name_prefix} per run.

Usage (from project root)
--------------------------
  python scripts/exp2.py                             # all 24 runs, sequential
  python scripts/exp2.py --workers 3                 # 3 parallel runs
  python scripts/exp2.py --k 0                       # baseline only (both rep conditions)
  python scripts/exp2.py --k 3 6                     # only K=3 and K=6 cells
  python scripts/exp2.py --rep-visible 1             # only rep-visible cells
  python scripts/exp2.py --rep-visible 0             # only no-rep cells
  python scripts/exp2.py --seeds 8                   # only seed=8
  python scripts/exp2.py --k 9 --seeds 8 16 --rep-visible 0
  python scripts/exp2.py --run exp2_gemini-2.5-flash_k0_rep1_seed8
  python scripts/exp2.py --skip-existing             # skip runs whose log dir exists
  python scripts/exp2.py --list                      # print matching runs, don't execute
  python scripts/exp2.py --prompt-algo cot           # override prompt algorithm
  python scripts/exp2.py --log-buyer-prompts --log-seller-prompts
  python scripts/exp2.py --llm gemma3:4b --service ollama --port 11434
  Filters combine with AND.
"""
import argparse
import re
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
# Set in main() from --llm (logs/exp2_{model_slug}/).
LOGS_DIR: Path | None = None
SUMMARY_LOG: Path | None = None

NUM_TOTAL_SELLERS = 12  # total seller slots (honest + sybil) — constant across all conditions
SEEDS = (8, 16, 64)
K_VALUES = (3, 6, 9)   # sybil cluster sizes; honest = NUM_TOTAL_SELLERS - K
RHO_MIN = 0.3

# Fixed args shared by every run (model, prompt-algo, and logging flags added at runtime).
_BASE_FIXED = [
    "--consumer-scenario", "LEMON_MARKET",
    "--firm-type", "LLM",
    "--num-buyers", "12",
    "--seller-type", "LLM",
    "--reputation-initial", "0.8",
    "--sybil-rho-min", str(RHO_MIN),
    "--discovery-limit-consumers", "3",
    "--max-tokens", "2000",
    "--no-diaries",
    # --max-timesteps, --prompt-algo, seller-personas, llm args added at runtime
]


def llm_filesystem_slug(llm: str) -> str:
    """Make model id safe as a single path segment (Windows + POSIX)."""
    s = llm.strip()
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = s.replace(":", "_")
    return s or "model"


def seller_personas_spec(num_honest: int) -> str:
    """Distribute num_honest sellers evenly across the four persona styles."""
    styles = ["standard", "detailed", "terse", "optimistic"]
    base, remainder = divmod(num_honest, len(styles))
    counts = [base + (1 if i < remainder else 0) for i in range(len(styles))]
    return ",".join(f"{style}:{count}" for style, count in zip(styles, counts) if count > 0)


def build_runs(base: list[str], name_prefix: str) -> list[tuple[str, list[str], dict]]:
    """Construct the full run list.

    Run names are ``{name_prefix}_{suffix}``; ``--log-dir`` is ``logs/{name_prefix}``
    so each simulation's run directory lands under ``logs/{name_prefix}/``.
    """
    runs: list[tuple[str, list[str], dict]] = []
    log_dir_arg = f"logs/{name_prefix}"

    # ---- Baseline: K=0 × rep_visible ∈ {True, False} ----
    personas_k0 = seller_personas_spec(NUM_TOTAL_SELLERS)
    for rep_visible in (True, False):
        rep_tag = "rep1" if rep_visible else "rep0"
        extra = [] if rep_visible else ["--no-buyer-rep"]
        for seed in SEEDS:
            label = f"{name_prefix}_k0_{rep_tag}_seed{seed}"
            runs.append((
                label,
                ["--name", label, "--log-dir", log_dir_arg,
                 "--num-sellers", str(NUM_TOTAL_SELLERS),
                 "--sybil-cluster-size", "0",
                 "--seller-personas", personas_k0,
                 "--seed", str(seed)] + extra + base,
                {"k": 0, "rep_visible": rep_visible, "seed": seed},
            ))

    # ---- Sybil grid: K × rep_visible × seed ----
    for k in K_VALUES:
        num_honest = NUM_TOTAL_SELLERS - k
        personas = seller_personas_spec(num_honest)
        saturation = k / NUM_TOTAL_SELLERS
        for rep_visible in (True, False):
            for seed in SEEDS:
                rep_tag = "rep1" if rep_visible else "rep0"
                label = f"{name_prefix}_k{k}_{rep_tag}_seed{seed}"
                extra = [] if rep_visible else ["--no-buyer-rep"]
                runs.append((
                    label,
                    ["--name", label, "--log-dir", log_dir_arg,
                     "--num-sellers", str(NUM_TOTAL_SELLERS),
                     "--sybil-cluster-size", str(k),
                     "--seller-personas", personas,
                     "--seed", str(seed)] + extra + base,
                    {"k": k, "rep_visible": rep_visible, "seed": seed,
                     "saturation": saturation},
                ))

    return runs


_print_lock = threading.Lock()
_progress_lock = threading.Lock()

# ── Live status board (active only when run directly, not when imported) ──────
_C_RESET  = "\033[0m"
_C_BOLD   = "\033[1m"
_C_DIM    = "\033[2m"
_C_YELLOW = "\033[33m"
_C_GREEN  = "\033[32m"
_C_RED    = "\033[31m"

_run_status:   dict[str, str]            = {}  # label → "pending"/"running"/"done"/"failed"
_run_progress: dict[str, tuple[int,int]] = {}  # label → (current_step, total_steps)
_status_lock  = threading.Lock()
_board_lines  = 0      # lines currently on screen belonging to the board
_board_active = False  # enabled only when script is run directly
_workers      = 1      # set in main()
_use_color    = True   # False with --no-color or non-TTY
_called_as_main = False  # set True in __main__ block


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_C_RESET}" if _use_color else text


def _erase_board() -> None:
    """Erase the status board. Caller must hold _print_lock."""
    global _board_lines
    if _board_lines > 0 and _use_color and sys.stdout.isatty():
        sys.stdout.write(f"\033[{_board_lines}A\033[J")
        sys.stdout.flush()
        _board_lines = 0


def _draw_board() -> None:
    """Draw the status board below current output. Caller must hold _print_lock."""
    global _board_lines
    if not _board_active or not _use_color or not sys.stdout.isatty():
        return

    with _status_lock:
        snap = dict(_run_status)

    running = [l for l, s in snap.items() if s == "running"]
    n_done  = sum(1 for s in snap.values() if s in ("done", "failed"))
    n_fail  = sum(1 for s in snap.values() if s == "failed")
    n_total = len(snap)

    sep = _c(_C_DIM, "─" * 62)
    lines = [sep]

    status_str = (
        f"{_c(_C_BOLD, 'ACTIVE')}  "
        f"{_c(_C_YELLOW, str(len(running)))}/{_workers} running  |  "
        f"{_c(_C_GREEN, str(n_done))}/{n_total} done"
    )
    if n_fail:
        status_str += f"  |  {_c(_C_RED, str(n_fail) + ' failed')}"
    lines.append(status_str)

    if running:
        for label in running:
            prog = _run_progress.get(label)
            prog_str = f" {_c(_C_DIM, f'[{prog[0]}/{prog[1]}]')}" if prog else ""
            lines.append(f"  {_c(_C_YELLOW, '▶')} {label}{prog_str}")
    else:
        lines.append(f"  {_c(_C_DIM, '(idle — waiting for next run)')}")

    lines.append(sep)

    output = "\n".join(lines) + "\n"
    sys.stdout.write(output)
    sys.stdout.flush()
    _board_lines = len(lines)


def _set_status(label: str, status: str) -> None:
    """Update run status and redraw the board."""
    with _status_lock:
        _run_status[label] = status
    with _print_lock:
        _erase_board()
        _draw_board()


def _set_progress(label: str, current: int, total: int) -> None:
    """Update timestep progress for a running label and redraw the board."""
    _run_progress[label] = (current, total)
    with _print_lock:
        _erase_board()
        _draw_board()


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
        _erase_board()
        print(line, flush=True)
        _draw_board()
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
    _set_status(log_label, "running")
    log(f"Starting: {log_label}")
    t0 = time.monotonic()
    proc: subprocess.Popen[str] | None = None
    try:
        output_lines: list[str] = []
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
            _PROGRESS_RE = re.compile(r'Completed (\d+)/(\d+) timesteps')
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                output_lines.append(line)
                m = _PROGRESS_RE.search(line)
                if m:
                    _set_progress(log_label, int(m.group(1)), int(m.group(2)))
            proc.wait()
        elapsed = time.monotonic() - t0
        assert proc is not None
        if proc.returncode != 0:
            _set_status(log_label, "failed")
            with _print_lock:
                _erase_board()
                for line in output_lines:
                    print(f"[{log_label}] {line}", end="", flush=True)
                _draw_board()
            log(f"WARNING: {log_label} exited with code {proc.returncode}")
        else:
            _set_status(log_label, "done")
        return proc.returncode, elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        _set_status(log_label, "failed")
        log(f"ERROR in {log_label}: {e}")
        return -1, elapsed


def filter_runs(
    runs: list[tuple[str, list[str], dict]],
    k_filter: list[int] | None,
    rep_visible_filter: list[bool] | None,
    seed_filter: list[int] | None,
    run_filter: list[str] | None,
    skip_existing: bool,
    name_prefix: str,
) -> list[tuple[str, list[str], dict]]:
    selected = []
    for label, argv, meta in runs:
        if run_filter is not None and label not in run_filter:
            continue
        if k_filter is not None and meta["k"] not in k_filter:
            continue
        if rep_visible_filter is not None and meta["rep_visible"] not in rep_visible_filter:
            continue
        if seed_filter is not None and meta["seed"] not in seed_filter:
            continue
        if skip_existing and (PROJECT_ROOT / "logs" / name_prefix / label).is_dir():
            continue
        selected.append((label, argv, meta))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 2 — LEMON_MARKET 3×3×2 sybil sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel simulation workers (default: 1). Keep ≤3 to stay within API rate limits.",
    )
    parser.add_argument(
        "--llm", type=str, default="gemini-2.5-flash",
        help="LLM model used as fallback for both buyers and sellers (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--buyer-llm", type=str, default=None, dest="buyer_llm",
        help="LLM for buyer agents. Falls back to --llm if unset.",
    )
    parser.add_argument(
        "--seller-llm", type=str, default=None, dest="seller_llm",
        help="LLM for honest sellers and sybil principal. Falls back to --llm if unset.",
    )
    parser.add_argument(
        "--service", type=str, default=None,
        help="Model service backend for all agents (e.g. ollama).",
    )
    parser.add_argument(
        "--buyer-service", type=str, default=None, dest="buyer_service",
        help="Service backend for buyer agents. Falls back to --service if unset.",
    )
    parser.add_argument(
        "--seller-service", type=str, default=None, dest="seller_service",
        help="Service backend for seller/sybil agents. Falls back to --service if unset.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for local model server.",
    )
    parser.add_argument(
        "--buyer-port", type=int, default=None, dest="buyer_port",
        help="Port for buyer LLM service. Falls back to --port if unset.",
    )
    parser.add_argument(
        "--seller-port", type=int, default=None, dest="seller_port",
        help="Port for seller LLM service. Falls back to --port if unset.",
    )
    parser.add_argument(
        "--openrouter-provider", type=str, nargs="+", default=None, metavar="PROVIDER",
        help="Preferred OpenRouter provider order for all agents (e.g. --openrouter-provider anthropic).",
    )
    parser.add_argument(
        "--buyer-openrouter-provider", type=str, nargs="+", default=None, metavar="PROVIDER",
        dest="buyer_openrouter_provider",
        help="Preferred OpenRouter provider(s) for buyer agents. Falls back to --openrouter-provider.",
    )
    parser.add_argument(
        "--seller-openrouter-provider", type=str, nargs="+", default=None, metavar="PROVIDER",
        dest="seller_openrouter_provider",
        help="Preferred OpenRouter provider(s) for seller/sybil agents. Falls back to --openrouter-provider.",
    )
    parser.add_argument(
        "--max-timesteps", type=int, default=50,
        help="Episode length passed to all runs (default: 50).",
    )
    parser.add_argument(
        "--prompt-algo", type=str, default="cot", choices=["io", "cot", "sc"],
        help="Prompt algorithm passed to all runs (default: cot).",
    )
    parser.add_argument(
        "--log-buyer-prompts", action="store_true",
        help="Append buyer LLM prompts/responses to lemon_agent_prompts.jsonl for each run.",
    )
    parser.add_argument(
        "--log-seller-prompts", action="store_true",
        help="Append seller/sybil LLM prompts/responses to lemon_agent_prompts.jsonl for each run.",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", metavar="K",
        help="Only run cells with these K values (0 = baseline). E.g. --k 0 3 6",
    )
    parser.add_argument(
        "--rep-visible", type=int, nargs="+", metavar="0|1", dest="rep_visible",
        help="Only run cells with this rep_visible setting (1=visible, 0=hidden). E.g. --rep-visible 1",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", metavar="N",
        help="Only run these seeds. E.g. --seeds 8 64",
    )
    parser.add_argument(
        "--run", type=str, nargs="+", metavar="LABEL", dest="runs",
        help="Only run these exact run labels.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip any run whose log directory already exists under logs/exp2_<model>/.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print matching runs and exit without executing.",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color in terminal output.",
    )
    cli = parser.parse_args()

    global LOGS_DIR, SUMMARY_LOG, _board_active, _workers, _use_color
    if cli.no_color or not sys.stdout.isatty():
        _use_color = False

    name_prefix = f"exp2_{llm_filesystem_slug(cli.buyer_llm or cli.llm)}"
    LOGS_DIR = PROJECT_ROOT / "logs" / name_prefix
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_LOG = LOGS_DIR / f"exp2_{TIMESTAMP}.log"

    llm_args = ["--llm", cli.llm]
    if cli.buyer_llm:
        llm_args += ["--buyer-llm", cli.buyer_llm]
    if cli.seller_llm:
        llm_args += ["--seller-llm", cli.seller_llm]
    if cli.service:
        llm_args += ["--service", cli.service]
    if cli.buyer_service:
        llm_args += ["--buyer-service", cli.buyer_service]
    if cli.seller_service:
        llm_args += ["--seller-service", cli.seller_service]
    if cli.port:
        llm_args += ["--port", str(cli.port)]
    if cli.buyer_port:
        llm_args += ["--buyer-port", str(cli.buyer_port)]
    if cli.seller_port:
        llm_args += ["--seller-port", str(cli.seller_port)]
    if cli.openrouter_provider:
        llm_args += ["--openrouter-provider", *cli.openrouter_provider]
    if cli.buyer_openrouter_provider:
        llm_args += ["--buyer-openrouter-provider", *cli.buyer_openrouter_provider]
    if cli.seller_openrouter_provider:
        llm_args += ["--seller-openrouter-provider", *cli.seller_openrouter_provider]

    prompt_args = ["--max-timesteps", str(cli.max_timesteps), "--prompt-algo", cli.prompt_algo]

    log_args = []
    if cli.log_buyer_prompts:
        log_args += ["--log-buyer-prompts"]
    if cli.log_seller_prompts:
        log_args += ["--log-seller-prompts"]

    base = _BASE_FIXED + llm_args + prompt_args + log_args

    all_runs = build_runs(base, name_prefix)

    rep_visible_filter = None
    if cli.rep_visible is not None:
        rep_visible_filter = [bool(v) for v in cli.rep_visible]

    selected = filter_runs(
        all_runs,
        k_filter=cli.k,
        rep_visible_filter=rep_visible_filter,
        seed_filter=cli.seeds,
        run_filter=cli.runs,
        skip_existing=cli.skip_existing,
        name_prefix=name_prefix,
    )

    if cli.list:
        print(f"Matching runs ({len(selected)} / {len(all_runs)} total):")
        for label, _, meta in selected:
            k = meta["k"]
            sat = f"sat={meta.get('saturation', 0):.0%}" if k > 0 else "no-sybil"
            rep = "rep=visible" if meta["rep_visible"] else "rep=hidden"
            print(f"  {label:55s}  [K={k:2d}  {sat:10s}  {rep}  seed={meta['seed']}]")
        return

    if not selected:
        print("No runs matched the supplied filters. Use --list to inspect.")
        return

    log(f"Experiment 2 started. Project root: {PROJECT_ROOT}")
    log(f"Selected: {len(selected)}/{len(all_runs)} runs  |  workers: {cli.workers}  |  llm: {cli.llm}  |  prompt-algo: {cli.prompt_algo}")
    log(f"Grid: K∈{{0,*{K_VALUES}}}  rep_visible∈{{True,False}}  seeds={SEEDS}  rho_min={RHO_MIN}")
    log(f"Total sellers fixed at {NUM_TOTAL_SELLERS}; honest = {NUM_TOTAL_SELLERS} - K")

    total = len(selected)
    worker_count = 1 if cli.workers <= 1 else min(cli.workers, total)
    completed_durations: list[float] = []
    batch_start = time.monotonic()

    # Activate live board only when run directly from CLI
    if _called_as_main:
        with _status_lock:
            for label, _, _ in selected:
                _run_status[label] = "pending"
        _workers = worker_count
        _board_active = True
        _draw_board()

    if cli.workers <= 1:
        for label, args, _ in selected:
            rc, elapsed = run_one(label, args)
            log_run_finished(label, elapsed, completed_durations, total, worker_count, batch_start)
            if rc != 0:
                log(f"Run {label} finished with non-zero exit code {rc}")
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for label, args, _ in selected:
                fut = pool.submit(run_one, label, args)
                futures[fut] = label
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    rc, elapsed = fut.result()
                    log_run_finished(label, elapsed, completed_durations, total, worker_count, batch_start)
                    if rc != 0:
                        log(f"Run {label} finished with non-zero exit code {rc}")
                except Exception as e:
                    log(f"Run {label} raised an exception: {e}")

    with _print_lock:
        _erase_board()

    total_wall = time.monotonic() - batch_start
    log(
        f"Experiment 2 completed. Total wall time: {format_duration(total_wall)}. "
        f"Runs directory: logs/{name_prefix}/"
    )


if __name__ == "__main__":
    _called_as_main = True
    main()
