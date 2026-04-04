"""
exp3_crash_recovery.py — markup ratio recovery after supply cost shock.

Recovery is gated by two conditions (both must hold):
  1. Markup ratio returns to within ±rel_threshold of the pre-shock baseline.
  2. Additional bankruptcies post-shock do not exceed --max-additional-bankruptcies
     (default 1). Runs that breach this limit are marked as "never recovered".

Two-panel figure:
  Left:  mu_t time series, one line per run, colored by n_stab.
         Dashed lines indicate runs blocked by bankruptcy.
         Vertical dashed line at shock_t. Horizontal band ±10% of pre-shock baseline.
  Right: Bar chart of mean recovery time (steps after shock) by n_stab.
         Error bars = std across seeds/dlc.

Usage
-----
  python exp3_crash_recovery.py
  python exp3_crash_recovery.py --logs-dir logs/exp3a_gemini-2.5-flash/
  python exp3_crash_recovery.py --max-additional-bankruptcies 0   # strict: no deaths allowed
  python exp3_crash_recovery.py --max-additional-bankruptcies 2   # lenient
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Allow running from any working directory by resolving the scripts dir
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from exp3_common import (
    find_shock_timestep,
    compute_markup_ratio_series,
    compute_bankruptcy_series,
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


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _parse_run_label(name: str) -> dict:
    """Extract n_stab, dlc, seed from exp3a_*_stab{N}_dlc{D}_seed{S} labels.

    Returns a dict with keys 'n_stab', 'dlc', 'seed' (all int).
    Missing fields default to None.
    """
    import re
    result = {"n_stab": None, "dlc": None, "seed": None}
    m = re.search(r"stab(\d+)", name)
    if m:
        result["n_stab"] = int(m.group(1))
    m = re.search(r"dlc(\d+)", name)
    if m:
        result["dlc"] = int(m.group(1))
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
    max_additional_bankruptcies: int = 1,
) -> list[dict]:
    """Return a list of run records.

    Each record has keys: name, n_stab, dlc, seed, ts, mu, dead, shock_t,
    recovery, bankruptcy_blocked, pre_dead, post_dead_max.

    Recovery is gated by both markup ratio *and* bankruptcy: if the number
    of additional bankruptcies after the shock exceeds
    ``max_additional_bankruptcies``, recovery is set to max_T regardless
    of whether the markup ratio returned to the baseline band.
    """
    run_dirs = sorted(d for d in logs_dir.iterdir() if d.is_dir())
    records = []
    for run_dir in run_dirs:
        if "_stab" not in run_dir.name:
            continue
        if not (run_dir / "states.json").is_file() and not list(run_dir.glob("state_t*.json")):
            continue
        label = _parse_run_label(run_dir.name)
        shock_t = find_shock_timestep(str(run_dir))
        ts, mu = compute_markup_ratio_series(str(run_dir))
        _, dead = compute_bankruptcy_series(str(run_dir))
        if len(ts) == 0:
            continue

        bankruptcy_blocked = False
        pre_dead = 0
        post_dead_max = 0

        if shock_t is None:
            recovery = None
        else:
            pre_mask = (ts >= shock_t - window) & (ts < shock_t)
            pre_dead = int(dead[pre_mask].max()) if pre_mask.any() else 0
            post_mask = ts > shock_t
            post_dead_max = int(dead[post_mask].max()) if post_mask.any() else pre_dead
            additional = post_dead_max - pre_dead

            if additional > max_additional_bankruptcies:
                bankruptcy_blocked = True
                recovery = int(ts.max() - shock_t)
            else:
                recovery = compute_recovery_time(
                    ts, mu, shock_t,
                    window=window,
                    rel_threshold=rel_threshold,
                )

        records.append({
            "name":   run_dir.name,
            "n_stab": label["n_stab"],
            "dlc":    label["dlc"],
            "seed":   label["seed"],
            "ts":     ts,
            "mu":     mu,
            "dead":   dead,
            "shock_t": shock_t,
            "recovery": recovery,
            "bankruptcy_blocked": bankruptcy_blocked,
            "pre_dead": pre_dead,
            "post_dead_max": post_dead_max,
        })
    return records


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    records: list[dict],
    window: int,
    rel_threshold: float,
    max_additional_bankruptcies: int,
) -> plt.Figure:
    fig, (ax_ts, ax_bar) = plt.subplots(
        1, 2,
        figsize=(10, 4),
        constrained_layout=True,
    )

    # -------------------------------------------------------------------
    # Determine grouping variable: n_stab values present
    # -------------------------------------------------------------------
    n_stab_values = sorted(set(
        r["n_stab"] for r in records if r["n_stab"] is not None
    ))
    if not n_stab_values:
        n_stab_values = [None]

    cmap = matplotlib.colormaps["viridis"].resampled(max(len(n_stab_values), 1))
    color_map = {v: cmap(i / max(len(n_stab_values) - 1, 1)) for i, v in enumerate(n_stab_values)}

    # -------------------------------------------------------------------
    # Left panel: time series
    # -------------------------------------------------------------------

    all_baseline_vals = []
    for r in records:
        if r["shock_t"] is not None and len(r["ts"]) > 0:
            pre_mask = (r["ts"] >= r["shock_t"] - window) & (r["ts"] < r["shock_t"])
            if pre_mask.any():
                all_baseline_vals.append(float(r["mu"][pre_mask].mean()))

    global_baseline = float(np.mean(all_baseline_vals)) if all_baseline_vals else None

    for r in records:
        ns = r["n_stab"]
        color = color_map.get(ns, color_map.get(None, "#555555"))
        linestyle = "--" if r.get("bankruptcy_blocked") else "-"
        ax_ts.plot(
            r["ts"], r["mu"],
            color=color, alpha=0.7, lw=1.2, linestyle=linestyle, zorder=2,
        )

    shock_ts_present = [r["shock_t"] for r in records if r["shock_t"] is not None]
    if shock_ts_present:
        shock_t_ref = int(np.median(shock_ts_present))
        ax_ts.axvline(
            shock_t_ref, color="#555555", linestyle="--", lw=1.2, alpha=0.7,
            label="Shock", zorder=3,
        )

    if global_baseline is not None:
        band_lo = global_baseline * (1 - rel_threshold)
        band_hi = global_baseline * (1 + rel_threshold)
        ax_ts.axhspan(
            band_lo, band_hi,
            alpha=0.12, color="#0072B2", zorder=1,
            label=f"Baseline \u00b1{int(rel_threshold*100)}%",
        )

    for ns in n_stab_values:
        label_str = f"n_stab={ns}" if ns is not None else "runs"
        ax_ts.plot([], [], color=color_map[ns], lw=1.8, label=label_str)

    n_blocked = sum(1 for r in records if r.get("bankruptcy_blocked"))
    if n_blocked:
        ax_ts.plot([], [], color="gray", lw=1.2, linestyle="--",
                   label=f"Bankruptcy blocked ({n_blocked})")

    ax_ts.set_xlabel("Timestep")
    ax_ts.set_ylabel("Markup ratio \u03bc")
    ax_ts.set_title("Markup ratio over time")
    ax_ts.legend(loc="best")

    # -------------------------------------------------------------------
    # Right panel: bar chart of mean recovery time by n_stab
    # -------------------------------------------------------------------
    recovery_by_ns: dict = defaultdict(list)
    for r in records:
        if r["recovery"] is not None:
            key = r["n_stab"] if r["n_stab"] is not None else "?"
            recovery_by_ns[key].append(r["recovery"])

    bar_keys = sorted(recovery_by_ns.keys(), key=lambda x: (x is None, x))
    bar_means = [np.mean(recovery_by_ns[k]) for k in bar_keys]
    bar_stds  = [np.std(recovery_by_ns[k]) if len(recovery_by_ns[k]) > 1 else 0.0
                 for k in bar_keys]
    bar_colors = [color_map.get(k, "#888888") for k in bar_keys]
    x_pos = np.arange(len(bar_keys))

    if bar_keys:
        ax_bar.bar(
            x_pos, bar_means, yerr=bar_stds,
            color=bar_colors, edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"elinewidth": 1.0},
            zorder=2,
        )
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(
            [str(k) for k in bar_keys],
            rotation=30, ha="right",
        )
        for xi, (mean, std) in enumerate(zip(bar_means, bar_stds)):
            ax_bar.text(
                xi, mean + std + max(bar_means) * 0.02,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=7,
            )
    else:
        ax_bar.text(0.5, 0.5, "No recovery data", ha="center", va="center",
                    transform=ax_bar.transAxes, fontsize=9)

    ax_bar.set_xlabel("n_stab")
    ax_bar.set_ylabel("Recovery time (steps)")
    title = "Mean recovery time by n_stab"
    if max_additional_bankruptcies < 999:
        title += f"\n(>{max_additional_bankruptcies} extra bankruptcy \u2192 no recovery)"
    ax_bar.set_title(title)

    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(records: list[dict], window: int) -> None:
    col_w = 55
    header = (
        f"{'Run':<{col_w}}  {'shock_t':>7}  {'recovery':>8}  "
        f"{'base_mu':>7}  {'post_mu':>7}  "
        f"{'pre_dead':>8}  {'post_dead':>9}  {'blocked':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in records:
        ts, mu, shock_t = r["ts"], r["mu"], r["shock_t"]
        if shock_t is None:
            print(f"  {r['name']:<{col_w-2}}  {'—':>7}  {'no shock':>8}")
            continue
        pre_mask = (ts >= shock_t - window) & (ts < shock_t)
        baseline_mu = float(mu[pre_mask].mean()) if pre_mask.any() else float("nan")
        post_mask = (ts > shock_t) & (ts <= shock_t + 5)
        post_mu = float(mu[post_mask].mean()) if post_mask.any() else float("nan")
        rec = r["recovery"] if r["recovery"] is not None else "—"
        blocked = "YES" if r.get("bankruptcy_blocked") else ""
        print(
            f"  {r['name']:<{col_w-2}}"
            f"  {shock_t:>7}"
            f"  {str(rec):>8}"
            f"  {baseline_mu:>7.3f}"
            f"  {post_mu:>7.3f}"
            f"  {r.get('pre_dead', 0):>8}"
            f"  {r.get('post_dead_max', 0):>9}"
            f"  {blocked:>7}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp3a crash recovery — markup ratio after supply shock"
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/exp3a_gemini-3-flash-preview/",
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
        "--max-additional-bankruptcies", type=int, default=1,
        dest="max_additional_bankruptcies",
        help="Max additional firm deaths post-shock before recovery is "
             "declared impossible (default: 1).",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        print(f"Logs directory not found: {logs_dir}")
        return

    records = collect_runs(
        logs_dir, args.window, args.rel_threshold,
        max_additional_bankruptcies=args.max_additional_bankruptcies,
    )
    if not records:
        print(f"No run data found under {logs_dir}")
        return

    print_summary(records, args.window)

    fig = make_figure(
        records, args.window, args.rel_threshold,
        max_additional_bankruptcies=args.max_additional_bankruptcies,
    )

    src_name = logs_dir.resolve().name
    out_dir = Path(args.fig_dir) / src_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_crash_recovery.pdf"
    fig.savefig(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
