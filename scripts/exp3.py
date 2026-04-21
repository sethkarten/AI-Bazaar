#!/usr/bin/env python3
"""
Run Experiment 3 (Supply Shock & Sybil Flood) — recovery benchmark.

Sub-experiments
---------------
exp3a  THE_CRASH + supply cost shock at t=25
  Grid: n_stab ∈ {3,5} × dlc=3 (fixed) × seeds {8,16,64} → 6 runs

exp3b  LEMON_MARKET + sybil flood at t=15
  Grid: k_initial=6 (fixed) × rep_visible ∈ {True,False} × seeds {8,16,64} → 6 runs
  Flood target: 4 × (12 − 6) = 24 sybils → 80% saturation post-shock.

Total: 12 runs per model

Usage (from project root)
--------------------------
  python scripts/exp3.py                              # all 12 runs, sequential (gemini-3-flash-preview)
  python scripts/exp3.py --workers 3                  # 3 parallel workers
  python scripts/exp3.py --test-llm my-finetuned-model  # swap the model under test
  python scripts/exp3.py --seller-llm other-model     # override fixed seller (lemon only)
  python scripts/exp3.py --experiment crash            # only exp3a (6 runs)
  python scripts/exp3.py --experiment lemon            # only exp3b (6 runs)
  python scripts/exp3.py --n-stab 3 5                 # crash: only n_stab=3 or 5
  python scripts/exp3.py --rep-visible 1               # lemon: rep-visible only
  python scripts/exp3.py --seeds 8                     # seed=8 only
  python scripts/exp3.py --skip-existing
  python scripts/exp3.py --list
  python scripts/exp3.py --no-color
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
# Set in main() from --llm
LOGS_DIR: Path | None = None
SUMMARY_LOG: Path | None = None

# ── Exp3a constants ────────────────────────────────────────────────────────────
CRASH_SEEDS = (8, 16, 64)
CRASH_N_STAB_VALUES = (3, 5)
CRASH_DLC_VALUES = (3,)

_BASE_CRASH = [
    "--use-cost-pref-gen", "--max-supply-unit-cost", "1",
    "--firm-type", "LLM", "--num-goods", "1", "--num-firms", "5",
    "--consumer-type", "CES", "--num-consumers", "50",
    "--firm-initial-cash", "500",
    "--overhead-costs", "14",
    "--consumer-scenario", "THE_CRASH",
    "--wtp-algo", "none",
    "--prompt-algo", "io", "--no-diaries",
    "--shock-timestep", "25",
    "--post-shock-unit-cost", "2.0",
]

# ── Exp3b constants ────────────────────────────────────────────────────────────
LEMON_SEEDS = (8, 16, 64)
LEMON_K_VALUES = (6,)
NUM_TOTAL_SELLERS = 12
RHO_MIN = 0.3

_BASE_LEMON = [
    "--consumer-scenario", "LEMON_MARKET",
    "--firm-type", "LLM",
    "--num-buyers", "12",
    "--seller-type", "LLM",
    "--reputation-initial", "0.8",
    "--sybil-rho-min", str(RHO_MIN),
    "--discovery-limit-consumers", "3",
    "--max-tokens", "1000",
    "--no-diaries",
    "--prompt-algo", "io",
    "--shock-timestep", "15",
    # --post-shock-sybil-cluster-size is set per-cell in build_lemon_runs
    # to achieve exactly 80% saturation: flood_k = 4 × (12 − k_initial)
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


def build_crash_runs(
    base: list[str],
    label_prefix: str,
    log_dir_name: str,
    max_timesteps: int = 100,
) -> list[tuple[str, list[str], dict]]:
    """Build exp3a run list: n_stab × dlc × seed."""
    runs: list[tuple[str, list[str], dict]] = []
    log_dir_arg = f"logs/{log_dir_name}"

    for n_stab in CRASH_N_STAB_VALUES:
        for dlc in CRASH_DLC_VALUES:
            for seed in CRASH_SEEDS:
                label = f"{label_prefix}_stab{n_stab}_dlc{dlc}_seed{seed}"
                stab_args = ["--num-stabilizing-firms", str(n_stab)] if n_stab > 0 else []
                argv = (
                    ["--name", label, "--log-dir", log_dir_arg,
                     "--max-timesteps", str(max_timesteps),
                     "--discovery-limit-consumers", str(dlc),
                     "--seed", str(seed)]
                    + stab_args
                    + base
                )
                runs.append((
                    label,
                    argv,
                    {"n_stab": n_stab, "dlc": dlc, "seed": seed},
                ))

    return runs


def build_lemon_runs(
    base: list[str],
    label_prefix: str,
    log_dir_name: str,
    max_timesteps: int = 45,
) -> list[tuple[str, list[str], dict]]:
    """Build exp3b run list: k_initial × rep_visible × seed."""
    runs: list[tuple[str, list[str], dict]] = []
    log_dir_arg = f"logs/{log_dir_name}"

    for k_initial in LEMON_K_VALUES:
        num_honest = NUM_TOTAL_SELLERS - k_initial
        personas = seller_personas_spec(num_honest)
        # Flood to 80% saturation: flood_k / (num_honest + flood_k) = 0.8
        # → flood_k = 4 × num_honest
        flood_k = 4 * num_honest
        for rep_visible in (True, False):
            for seed in LEMON_SEEDS:
                rep_tag = "rep1" if rep_visible else "rep0"
                label = f"{label_prefix}_k{k_initial}_{rep_tag}_seed{seed}"
                extra = [] if rep_visible else ["--no-buyer-rep"]
                argv = (
                    ["--name", label, "--log-dir", log_dir_arg,
                     "--max-timesteps", str(max_timesteps),
                     "--num-sellers", str(NUM_TOTAL_SELLERS),
                     "--sybil-cluster-size", str(k_initial),
                     "--seller-personas", personas,
                     "--post-shock-sybil-cluster-size", str(flood_k),
                     "--seed", str(seed)]
                    + extra
                    + base
                )
                runs.append((
                    label,
                    argv,
                    {"k_initial": k_initial, "flood_k": flood_k,
                     "rep_visible": rep_visible, "seed": seed},
                ))

    return runs


# ── Progress board (mirrors exp2.py) ──────────────────────────────────────────
_print_lock = threading.Lock()
_progress_lock = threading.Lock()

_C_RESET  = "\033[0m"
_C_BOLD   = "\033[1m"
_C_DIM    = "\033[2m"
_C_YELLOW = "\033[33m"
_C_GREEN  = "\033[32m"
_C_RED    = "\033[31m"

_run_status:   dict[str, str]            = {}
_run_progress: dict[str, tuple[int,int]] = {}
_status_lock  = threading.Lock()
_board_lines  = 0
_board_active = False
_workers      = 1
_use_color    = True
_called_as_main = False


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_C_RESET}" if _use_color else text


def _erase_board() -> None:
    global _board_lines
    if _board_lines > 0 and _use_color and sys.stdout.isatty():
        sys.stdout.write(f"\033[{_board_lines}A\033[J")
        sys.stdout.flush()
        _board_lines = 0


def _draw_board() -> None:
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
    with _status_lock:
        _run_status[label] = status
    with _print_lock:
        _erase_board()
        _draw_board()


def _set_progress(label: str, current: int, total: int) -> None:
    _run_progress[label] = (current, total)
    with _print_lock:
        _erase_board()
        _draw_board()


def format_duration(seconds: float) -> str:
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


def filter_crash_runs(
    runs: list[tuple[str, list[str], dict]],
    n_stab_filter: list[int] | None,
    dlc_filter: list[int] | None,
    seed_filter: list[int] | None,
    run_filter: list[str] | None,
    skip_existing: bool,
    log_dir_name: str,
) -> list[tuple[str, list[str], dict]]:
    selected = []
    for label, argv, meta in runs:
        if run_filter is not None and label not in run_filter:
            continue
        if n_stab_filter is not None and meta["n_stab"] not in n_stab_filter:
            continue
        if dlc_filter is not None and meta["dlc"] not in dlc_filter:
            continue
        if seed_filter is not None and meta["seed"] not in seed_filter:
            continue
        if skip_existing and (PROJECT_ROOT / "logs" / log_dir_name / label).is_dir():
            continue
        selected.append((label, argv, meta))
    return selected


def filter_lemon_runs(
    runs: list[tuple[str, list[str], dict]],
    k_filter: list[int] | None,
    rep_visible_filter: list[bool] | None,
    seed_filter: list[int] | None,
    run_filter: list[str] | None,
    skip_existing: bool,
    log_dir_name: str,
) -> list[tuple[str, list[str], dict]]:
    selected = []
    for label, argv, meta in runs:
        if run_filter is not None and label not in run_filter:
            continue
        if k_filter is not None and meta["k_initial"] not in k_filter:
            continue
        if rep_visible_filter is not None and meta["rep_visible"] not in rep_visible_filter:
            continue
        if seed_filter is not None and meta["seed"] not in seed_filter:
            continue
        if skip_existing and (PROJECT_ROOT / "logs" / log_dir_name / label).is_dir():
            continue
        selected.append((label, argv, meta))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 3 — Supply Shock & Sybil Flood recovery benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", choices=["crash", "lemon", "both"], default="both",
        help="Which sub-experiment to run (default: both).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel simulation workers (default: 1).",
    )
    parser.add_argument(
        "--test-llm", type=str, default="gemini-3-flash-preview", dest="test_llm",
        help="Model under test: used as the firm LLM (crash) and buyer LLM (lemon). "
             "Default: gemini-3-flash-preview.",
    )
    parser.add_argument(
        "--seller-llm", type=str, default="gemini-3-flash-preview", dest="seller_llm",
        help="Fixed seller/sybil LLM for lemon sub-experiment. "
             "Default: gemini-3-flash-preview.",
    )
    parser.add_argument(
        "--service", type=str, default=None,
        help="Model service backend for all agents (e.g. ollama).",
    )
    parser.add_argument(
        "--buyer-service", type=str, default=None, dest="buyer_service",
        help="Service backend for buyer agents. Falls back to --service.",
    )
    parser.add_argument(
        "--seller-service", type=str, default=None, dest="seller_service",
        help="Service backend for seller agents. Falls back to --service.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for local model server.",
    )
    parser.add_argument(
        "--buyer-port", type=int, default=None, dest="buyer_port",
        help="Port for buyer LLM service. Falls back to --port.",
    )
    parser.add_argument(
        "--seller-port", type=int, default=None, dest="seller_port",
        help="Port for seller LLM service. Falls back to --port.",
    )
    parser.add_argument(
        "--openrouter-provider", type=str, nargs="+", default=None, metavar="PROVIDER",
        help="Preferred OpenRouter provider order for all agents.",
    )
    parser.add_argument(
        "--buyer-openrouter-provider", type=str, nargs="+", default=None, metavar="PROVIDER",
        dest="buyer_openrouter_provider",
        help="Preferred OpenRouter provider(s) for buyer agents.",
    )
    parser.add_argument(
        "--seller-openrouter-provider", type=str, nargs="+", default=None, metavar="PROVIDER",
        dest="seller_openrouter_provider",
        help="Preferred OpenRouter provider(s) for seller/sybil agents.",
    )
    parser.add_argument(
        "--max-timesteps", type=int, default=None,
        help="Override episode length for all runs (crash default=100, lemon default=50).",
    )
    parser.add_argument(
        "--prompt-algo", type=str, default="io", choices=["io", "cot", "sc"],
        help="Prompt algorithm passed to all runs (default: io).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1000,
        help="Max tokens per LLM call (default: 1000).",
    )
    parser.add_argument(
        "--stab-llm", type=str, default=None, dest="stab_llm",
        help="(crash) Model name for stabilizing firm requests (vLLM LoRA alias, e.g. 'stab'). "
             "Uses same --service and --port as non-stabilizing firms.",
    )
    parser.add_argument(
        "--listing-corpus", type=str, default=None, dest="listing_corpus",
        help="(lemon) Path to pre-compiled listing corpus JSON. Replaces seller/sybil LLM calls.",
    )
    # Crash-specific filters
    parser.add_argument(
        "--n-stab", type=int, nargs="+", metavar="N", dest="n_stab",
        help="(crash) Only run cells with these n_stab values. E.g. --n-stab 0 3",
    )
    parser.add_argument(
        "--dlc", type=int, nargs="+", metavar="N",
        help="(crash) Only run cells with these dlc values. E.g. --dlc 1 5",
    )
    # Lemon-specific filters
    parser.add_argument(
        "--k", type=int, nargs="+", metavar="K",
        help="(lemon) Only run cells with these k_initial values. E.g. --k 3 6",
    )
    parser.add_argument(
        "--rep-visible", type=int, nargs="+", metavar="0|1", dest="rep_visible",
        help="(lemon) Only run cells with this rep_visible setting (1=visible, 0=hidden).",
    )
    # Shared filters
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
        help="Skip any run whose log directory already exists.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print matching runs and exit without executing.",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color in terminal output.",
    )
    # Prompt logging
    parser.add_argument(
        "--log-buyer-prompts", action="store_true",
        help="(lemon) Append buyer prompts/responses to lemon_agent_prompts.jsonl.",
    )
    parser.add_argument(
        "--log-seller-prompts", action="store_true",
        help="(lemon) Append seller/sybil prompts/responses to lemon_agent_prompts.jsonl.",
    )
    parser.add_argument(
        "--log-prompts", action="store_true",
        help="(crash) Append firm prompts/responses to crash_agent_prompts.jsonl.",
    )
    cli = parser.parse_args()

    global LOGS_DIR, SUMMARY_LOG, _board_active, _workers, _use_color
    if cli.no_color or not sys.stdout.isatty():
        _use_color = False

    # test-llm drives the model slug (used for both crash firm and lemon buyer)
    model_slug = llm_filesystem_slug(cli.test_llm)

    # ── Build LLM arg blocks ──────────────────────────────────────────────────
    # Crash: test-llm is the firm LLM
    llm_args_crash = ["--llm", cli.test_llm, "--max-tokens", str(cli.max_tokens)]
    if cli.service:
        llm_args_crash += ["--service", cli.service]
    if cli.port:
        llm_args_crash += ["--port", str(cli.port)]
    if cli.openrouter_provider:
        llm_args_crash += ["--openrouter-provider", *cli.openrouter_provider]
    if cli.stab_llm:
        llm_args_crash += ["--stab-llm", cli.stab_llm]

    # Lemon: test-llm → buyer; seller-llm → seller/sybil
    llm_args_lemon = [
        "--buyer-llm", cli.test_llm,
        "--seller-llm", cli.seller_llm,
        "--max-tokens", str(cli.max_tokens),
    ]
    if cli.service:
        llm_args_lemon += ["--service", cli.service]
    if cli.port:
        llm_args_lemon += ["--port", str(cli.port)]
    if cli.openrouter_provider:
        llm_args_lemon += ["--openrouter-provider", *cli.openrouter_provider]
    if cli.buyer_service:
        llm_args_lemon += ["--buyer-service", cli.buyer_service]
    if cli.buyer_port:
        llm_args_lemon += ["--buyer-port", str(cli.buyer_port)]
    if cli.buyer_openrouter_provider:
        llm_args_lemon += ["--buyer-openrouter-provider", *cli.buyer_openrouter_provider]
    if cli.seller_service:
        llm_args_lemon += ["--seller-service", cli.seller_service]
    if cli.seller_port:
        llm_args_lemon += ["--seller-port", str(cli.seller_port)]
    if cli.seller_openrouter_provider:
        llm_args_lemon += ["--seller-openrouter-provider", *cli.seller_openrouter_provider]
    if cli.listing_corpus:
        llm_args_lemon += ["--listing-corpus", cli.listing_corpus]

    prompt_args = ["--prompt-algo", cli.prompt_algo]

    # All runs for a given test-llm share one folder: logs/exp3_{model_slug}/
    shared_dir = f"exp3_{model_slug}"

    # ── Build crash runs ──────────────────────────────────────────────────────
    crash_runs: list[tuple[str, list[str], dict]] = []
    if cli.experiment in ("crash", "both"):
        crash_ts = cli.max_timesteps if cli.max_timesteps is not None else 100
        crash_log_args = ["--log-crash-firm-prompts"] if cli.log_prompts else []
        crash_base = _BASE_CRASH + llm_args_crash + prompt_args + crash_log_args
        crash_runs = build_crash_runs(crash_base, "exp3a", shared_dir, max_timesteps=crash_ts)

        crash_runs = filter_crash_runs(
            crash_runs,
            n_stab_filter=cli.n_stab,
            dlc_filter=cli.dlc,
            seed_filter=cli.seeds,
            run_filter=cli.runs,
            skip_existing=cli.skip_existing,
            log_dir_name=shared_dir,
        )

    # ── Build lemon runs ──────────────────────────────────────────────────────
    lemon_runs: list[tuple[str, list[str], dict]] = []
    if cli.experiment in ("lemon", "both"):
        lemon_ts = cli.max_timesteps if cli.max_timesteps is not None else 50
        lemon_log_args = []
        if cli.log_buyer_prompts:
            lemon_log_args += ["--log-buyer-prompts"]
        if cli.log_seller_prompts:
            lemon_log_args += ["--log-seller-prompts"]
        lemon_base = _BASE_LEMON + llm_args_lemon + prompt_args + lemon_log_args

        rep_visible_filter = None
        if cli.rep_visible is not None:
            rep_visible_filter = [bool(v) for v in cli.rep_visible]

        all_lemon = build_lemon_runs(lemon_base, "exp3b", shared_dir, max_timesteps=lemon_ts)
        lemon_runs = filter_lemon_runs(
            all_lemon,
            k_filter=cli.k,
            rep_visible_filter=rep_visible_filter,
            seed_filter=cli.seeds,
            run_filter=cli.runs,
            skip_existing=cli.skip_existing,
            log_dir_name=shared_dir,
        )

    selected = crash_runs + lemon_runs

    # ── --list mode ───────────────────────────────────────────────────────────
    if cli.list:
        total_crash = len(build_crash_runs([], "exp3a", shared_dir)) if cli.experiment in ("crash","both") else 0
        total_lemon = len(build_lemon_runs([], "exp3b", shared_dir)) if cli.experiment in ("lemon","both") else 0
        print(f"Matching runs ({len(selected)} / {total_crash + total_lemon} total)  →  logs/{shared_dir}/")
        for label, _, meta in selected:
            if "n_stab" in meta:
                print(f"  {label:55s}  [crash  n_stab={meta['n_stab']}  dlc={meta['dlc']}  seed={meta['seed']}]")
            else:
                flood_k = meta["flood_k"]
                num_honest = NUM_TOTAL_SELLERS - meta["k_initial"]
                post_sat = flood_k / (num_honest + flood_k)
                rep = "rep=visible" if meta["rep_visible"] else "rep=hidden"
                print(f"  {label:55s}  [lemon  K={meta['k_initial']}→{flood_k}  sat={post_sat:.0%}  {rep}  seed={meta['seed']}]")
        return

    if not selected:
        print("No runs matched the supplied filters. Use --list to inspect.")
        return

    # ── Setup logging dirs ────────────────────────────────────────────────────
    LOGS_DIR = PROJECT_ROOT / "logs" / shared_dir
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_LOG = LOGS_DIR / f"exp3_{TIMESTAMP}.log"

    total = len(selected)
    worker_count = 1 if cli.workers <= 1 else min(cli.workers, total)
    completed_durations: list[float] = []
    batch_start = time.monotonic()

    log(f"Experiment 3 started. Project root: {PROJECT_ROOT}")
    log(f"Selected: {len(selected)} runs  |  workers: {worker_count}  |  test-llm: {cli.test_llm}  |  seller-llm: {cli.seller_llm}  |  prompt-algo: {cli.prompt_algo}")
    if crash_runs:
        log(f"Crash (exp3a): {len(crash_runs)} runs  n_stab∈{CRASH_N_STAB_VALUES}  dlc∈{CRASH_DLC_VALUES}  seeds={CRASH_SEEDS}")
    if lemon_runs:
        log(f"Lemon (exp3b): {len(lemon_runs)} runs  K∈{LEMON_K_VALUES}  flood→80% sat  rep_visible=True  seeds={LEMON_SEEDS}")

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
        f"Experiment 3 completed. Total wall time: {format_duration(total_wall)}. "
        f"All logs: logs/{shared_dir}/"
    )


if __name__ == "__main__":
    _called_as_main = True
    main()
