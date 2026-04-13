"""
exp3_lemon_recovery.py — consumer surplus recovery after sybil flood shock.

Recovery metric: 5-step trailing rolling mean of average buyer consumer
surplus (``lemon_market_avg_consumer_surplus`` from state files).  Recovery
is declared when the smoothed surplus returns to within ±rel_threshold of
the smoothed pre-shock baseline.

Single-panel figure:
  For each k_initial group, runs are aligned by timestep and aggregated:
    - Bright mean line: rolling_mean applied to the per-timestep mean across seeds.
    - Shaded ±1 std band: rolling_mean applied to the per-timestep std across seeds.
  Vertical dashed line at shock_t.  Horizontal band ±10% of pre-shock baseline.

Usage
-----
  python exp3_lemon_recovery.py
  python exp3_lemon_recovery.py --logs-dir logs/exp3_gemini-3-flash-preview/
  python exp3_lemon_recovery.py --rolling-k 3   # 3-step rolling window
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from exp3_common import (
    find_shock_timestep,
    compute_consumer_surplus_series,
    rolling_mean,
    compute_recovery_time,
)

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "lines.linewidth":   1.5,
    "lines.markersize":  5,
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "grid.color":        "gray",
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.8",
    "figure.dpi":        100,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":       False,
})

_DEFAULT_SHOCK_T = 15


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _parse_run_label(name: str) -> dict:
    """Extract k_initial, rep, seed from exp3b_*_k{K}_rep{R}_seed{S} labels."""
    import re
    result = {"k_initial": None, "rep": None, "seed": None}
    m = re.search(r"_k(\d+)", name)
    if m:
        result["k_initial"] = int(m.group(1))
    m = re.search(r"rep(\d+)", name)
    if m:
        result["rep"] = int(m.group(1))
    m = re.search(r"seed(\d+)", name)
    if m:
        result["seed"] = int(m.group(1))
    return result


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_runs(
    logs_dir: Path,
    window: int,
    rel_threshold: float,
    rolling_k: int = 5,
) -> list[dict]:
    """Return a list of run records.

    Each record has keys: name, k_initial, rep, seed, ts, cs_raw, cs_smooth,
    shock_t, recovery.

    The recovery check is performed on the *smoothed* (rolling-mean) consumer
    surplus series.
    """
    run_dirs = sorted(d for d in logs_dir.iterdir() if d.is_dir())
    records = []
    for run_dir in run_dirs:
        if "_k" not in run_dir.name:
            continue
        if not (run_dir / "states.json").is_file() and not list(run_dir.glob("state_t*.json")):
            continue
        label = _parse_run_label(run_dir.name)
        shock_t = find_shock_timestep(str(run_dir))
        if shock_t is None:
            shock_t = _DEFAULT_SHOCK_T
        ts, cs_raw = compute_consumer_surplus_series(str(run_dir))
        if len(ts) == 0:
            continue
        cs_smooth = rolling_mean(cs_raw, rolling_k)
        recovery = compute_recovery_time(
            ts, cs_smooth, shock_t,
            window=window,
            rel_threshold=rel_threshold,
        )
        records.append({
            "name":      run_dir.name,
            "k_initial": label["k_initial"],
            "rep":       label["rep"],
            "seed":      label["seed"],
            "ts":        ts,
            "cs_raw":    cs_raw,
            "cs_smooth": cs_smooth,
            "shock_t":   shock_t,
            "recovery":  recovery,
        })
    return records


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _align_series(records_k: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align cs_smooth series from a group of runs onto a common timestep grid.

    Returns (common_ts, mean_arr, std_arr) where mean and std are computed
    across runs at each shared timestep.  Timesteps present in only some runs
    are excluded (intersection).
    """
    # Build intersection of timesteps
    ts_sets = [set(r["ts"].tolist()) for r in records_k]
    common_ts_set = ts_sets[0]
    for s in ts_sets[1:]:
        common_ts_set = common_ts_set & s
    common_ts = np.array(sorted(common_ts_set), dtype=float)

    if len(common_ts) == 0:
        return np.array([]), np.array([]), np.array([])

    # Stack aligned series
    matrix = np.full((len(records_k), len(common_ts)), np.nan)
    for i, r in enumerate(records_k):
        ts = r["ts"]
        cs = r["cs_smooth"]
        for j, t in enumerate(common_ts):
            idx = np.where(ts == t)[0]
            if len(idx) > 0:
                matrix[i, j] = cs[idx[0]]

    mean_arr = np.nanmean(matrix, axis=0)
    std_arr  = np.nanstd(matrix, axis=0, ddof=0)
    return common_ts, mean_arr, std_arr


def make_figure(
    records: list[dict],
    window: int,
    rel_threshold: float,
    rolling_k: int,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.75, 3.2), constrained_layout=True)

    # -------------------------------------------------------------------
    # Grouping: k_initial values
    # -------------------------------------------------------------------
    k_values = sorted(set(
        r["k_initial"] for r in records if r["k_initial"] is not None
    ))
    if not k_values:
        k_values = [None]

    # Okabe-Ito for ≤8 groups; fall back to viridis for more
    OKABE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442"]
    if len(k_values) <= len(OKABE):
        color_map = {v: OKABE[i] for i, v in enumerate(k_values)}
    else:
        cmap = matplotlib.colormaps["viridis"].resampled(len(k_values))
        color_map = {v: cmap(i / max(len(k_values) - 1, 1)) for i, v in enumerate(k_values)}

    # -------------------------------------------------------------------
    # Pre-shock baseline (for reference band)
    # -------------------------------------------------------------------
    shock_ts_present = [r["shock_t"] for r in records if r["shock_t"] is not None]
    shock_t_ref = int(np.median(shock_ts_present)) if shock_ts_present else _DEFAULT_SHOCK_T

    all_baseline_vals = []
    for r in records:
        shock_t = r["shock_t"]
        if shock_t is not None and len(r["ts"]) > 0:
            pre_mask = (r["ts"] >= shock_t - window) & (r["ts"] < shock_t)
            if pre_mask.any():
                all_baseline_vals.append(float(r["cs_smooth"][pre_mask].mean()))
    global_baseline = float(np.mean(all_baseline_vals)) if all_baseline_vals else None

    # -------------------------------------------------------------------
    # Plot mean ± 1 std per k_initial group
    # -------------------------------------------------------------------
    groups: dict = defaultdict(list)
    for r in records:
        groups[r["k_initial"]].append(r)

    for kv in k_values:
        color = color_map.get(kv, "#555555")
        label_str = f"k={kv}" if kv is not None else "runs"

        recs_k = groups[kv]
        common_ts, mean_arr, std_arr = _align_series(recs_k)
        if len(common_ts) == 0:
            continue

        # Smooth mean and std independently
        sm_mean = rolling_mean(mean_arr, rolling_k)
        sm_std  = rolling_mean(std_arr,  rolling_k)

        ax.fill_between(
            common_ts,
            sm_mean - sm_std,
            sm_mean + sm_std,
            color=color, alpha=0.18, zorder=1,
        )
        ax.plot(
            common_ts, sm_mean,
            color=color, lw=2.0, alpha=1.0, zorder=3,
            label=label_str,
        )

    # -------------------------------------------------------------------
    # Shock line and baseline band
    # -------------------------------------------------------------------
    ax.axvline(
        shock_t_ref, color="#555555", linestyle="--", lw=1.2, alpha=0.8,
        label="Shock", zorder=4,
    )

    if global_baseline is not None:
        thr = max(rel_threshold * abs(global_baseline), 0.05)
        ax.axhspan(
            global_baseline - thr, global_baseline + thr,
            alpha=0.10, color="#0072B2", zorder=1,
            label=f"Pre-shock baseline \u00b1{int(rel_threshold*100)}%",
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Avg. consumer surplus")
    ax.set_title(
        f"Consumer surplus recovery after sybil flood "
        f"({rolling_k}-step rolling mean, \u00b11 std)",
        fontsize=9,
    )
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(records: list[dict], window: int) -> None:
    col_w = 55
    header = (
        f"{'Run':<{col_w}}  {'shock_t':>7}  {'recovery':>8}  "
        f"{'base_cs':>8}  {'post_cs':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in records:
        ts, cs, shock_t = r["ts"], r["cs_smooth"], r["shock_t"]
        pre_mask = (ts >= shock_t - window) & (ts < shock_t)
        baseline_cs = float(cs[pre_mask].mean()) if pre_mask.any() else float("nan")
        post_mask = (ts > shock_t) & (ts <= shock_t + 5)
        post_cs = float(cs[post_mask].mean()) if post_mask.any() else float("nan")
        rec = r["recovery"] if r["recovery"] is not None else "\u2014"
        print(
            f"  {r['name']:<{col_w-2}}"
            f"  {shock_t:>7}"
            f"  {str(rec):>8}"
            f"  {baseline_cs:>8.1f}"
            f"  {post_cs:>8.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp3b lemon recovery — consumer surplus after sybil flood"
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/exp3_gemini-3-flash-preview/",
        help="Top-level log directory containing per-run subdirectories.",
    )
    parser.add_argument(
        "--fig-dir",
        default=str(Path(__file__).resolve().parents[3] / "fig" / "exp3"),
        help="Root output directory for figures.",
    )
    parser.add_argument(
        "--window", type=int, default=5,
        help="Pre-shock window size for baseline estimation (default: 5).",
    )
    parser.add_argument(
        "--rel-threshold", type=float, default=0.10,
        help="Relative recovery threshold (default: 0.10 = 10%%).",
    )
    parser.add_argument(
        "--rolling-k", type=int, default=5, dest="rolling_k",
        help="Rolling mean window size for smoothing consumer surplus (default: 5).",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        print(f"Logs directory not found: {logs_dir}")
        return

    records = collect_runs(
        logs_dir, args.window, args.rel_threshold,
        rolling_k=args.rolling_k,
    )
    if not records:
        print(f"No run data found under {logs_dir}")
        return

    print_summary(records, args.window)

    fig = make_figure(
        records, args.window, args.rel_threshold,
        rolling_k=args.rolling_k,
    )

    src_name = logs_dir.resolve().name
    out_dir = Path(args.fig_dir) / src_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_lemon_recovery.pdf"
    fig.savefig(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
