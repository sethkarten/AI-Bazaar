"""
exp7_blinded.py — Fig D: Detection Mechanism — Identity Memory vs. Quality Reasoning.

Three-panel grouped bar chart:
  Panel 1: Sybil detection rate
  Panel 2: Sybil revenue share
  Panel 3: Consumer surplus

Four bar groups per panel:
  1. Base buyer,        ID-visible  (exp2 baseline, k=K)  — sky blue,    solid
  2. Base buyer,        blinded     (exp7,          k=K)  — sky blue,    hatched
  3. Skeptical Guardian, ID-visible (exp2,          k=K)  — vermillion,  solid
  4. Skeptical Guardian, blinded    (exp7,          k=K)  — vermillion,  hatched

Expected run naming convention (exp7):
  logs/exp7_{slug}/exp7_{slug}_{buyer_type}_blinded_k{k}_rep{rep}_seed{seed}/
    buyer_type ∈ {base, guardian}
    rep ∈ {0, 1}  (rep0=reputation_hidden, rep1=reputation_visible)

Exp2 run dirs resolved via exp2_common.resolve_run_dir().

Usage:
    python paper/fig/scripts/exp7/exp7_blinded.py \\
        [--slug anthropic_claude-sonnet-4-6] [--k 6] \\
        [--logs-dir logs/] [--output ...]
"""

import argparse
import glob
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "exp2"))
try:
    from exp2_common import resolve_run_dir as _exp2_resolve
    _HAS_EXP2_COMMON = True
except ImportError:
    _HAS_EXP2_COMMON = False

SEEDS = [8, 16, 64]

# Okabe-Ito
COLOR_BASE     = "#56B4E9"   # sky blue
COLOR_GUARDIAN = "#D55E00"   # vermillion

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8.5,
    "lines.linewidth":    1.5,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex":        False,
    "pdf.fonttype":       42,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_states(run_dir):
    p = os.path.join(run_dir, "states.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    files = sorted(glob.glob(os.path.join(run_dir, "state_t*.json")))
    states = []
    for fp in files:
        with open(fp) as f:
            states.append(json.load(f))
    return states


def compute_metrics(run_dir):
    """
    Return dict with keys:
      detection_rate  — mean(1 - sybil_passed/sybil_seen) across timesteps with sybil traffic
      sybil_rev_share — mean lemon_market_sybil_revenue_share
      consumer_surplus — mean lemon_market_avg_consumer_surplus
    Returns None if run_dir missing or no data.
    """
    if not os.path.isdir(run_dir):
        return None
    states = load_states(run_dir)
    if not states:
        return None

    det_vals    = []
    rev_vals    = []
    surplus_vals = []

    for s in states:
        # Detection rate from consumer aggregates
        for c in s.get("consumers", []):
            sybil_seen   = c.get("sybil_seen_total", 0)
            sybil_passed = c.get("sybil_passed_total", 0)
            if sybil_seen and sybil_seen > 0:
                det_vals.append(1.0 - sybil_passed / sybil_seen)

        rev = s.get("lemon_market_sybil_revenue_share")
        if rev is not None:
            rev_vals.append(float(rev))

        surplus = s.get("lemon_market_avg_consumer_surplus")
        if surplus is not None:
            surplus_vals.append(float(surplus))

    if not rev_vals and not surplus_vals:
        return None

    return {
        "detection_rate":   float(np.mean(det_vals))     if det_vals     else float("nan"),
        "sybil_rev_share":  float(np.mean(rev_vals))     if rev_vals     else float("nan"),
        "consumer_surplus": float(np.mean(surplus_vals)) if surplus_vals else float("nan"),
    }


def load_condition(logs_dir, slug, source, buyer_type, k, rep_visible=True):
    """
    Load metrics for one (source, buyer_type) condition across SEEDS.
    source: "exp2" | "exp7"
    buyer_type: "base" | "guardian"
    Returns list of metric dicts.
    """
    records = []
    for seed in SEEDS:
        if source == "exp2":
            if _HAS_EXP2_COMMON:
                d = _exp2_resolve(
                    os.path.join(logs_dir, f"exp2_{slug}"),
                    name_prefix=f"exp2_{slug}",
                    k=k,
                    rep_visible=rep_visible,
                    seed=seed,
                )
            else:
                rep_tag = "rep1" if rep_visible else "rep0"
                name = f"exp2_{slug}_k{k}_{rep_tag}_seed{seed}"
                d = os.path.join(logs_dir, f"exp2_{slug}", name)
                if not os.path.isdir(d):
                    d = None
        else:  # exp7
            rep_tag = "rep1" if rep_visible else "rep0"
            name = f"exp7_{slug}_{buyer_type}_blinded_k{k}_{rep_tag}_seed{seed}"
            d = os.path.join(logs_dir, f"exp7_{slug}", name)
            if not os.path.isdir(d):
                d = None

        if d is None:
            continue
        m = compute_metrics(d)
        if m is not None:
            records.append(m)
    return records


def agg(records, key):
    """Return (mean, min, max) or (nan, nan, nan) if no data."""
    vals = [r[key] for r in records if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


# ── Plotting ──────────────────────────────────────────────────────────────────

METRICS = [
    ("detection_rate",   "Sybil Detection Rate",  "Higher = better",  [0, 1]),
    ("sybil_rev_share",  "Sybil Revenue Share",   "Lower = better",   [0, 1]),
    ("consumer_surplus", "Consumer Surplus",       "Higher = better",  None),
]

BAR_GROUPS = [
    # (label,                color,         hatch, source, buyer_type)
    ("Base — visible",       COLOR_BASE,     "",    "exp2", "base"),
    ("Base — blinded",       COLOR_BASE,     "//",  "exp7", "base"),
    ("Guardian — visible",   COLOR_GUARDIAN, "",    "exp2", "guardian"),
    ("Guardian — blinded",   COLOR_GUARDIAN, "//",  "exp7", "guardian"),
]


def draw_metric_panel(ax, metric_key, all_records, ylim, ylabel, subtitle):
    """Draw one grouped bar panel for a single metric."""
    bar_width = 0.6
    xs = np.arange(len(BAR_GROUPS))

    for xi, (label, color, hatch, source, btype) in enumerate(BAR_GROUPS):
        records = all_records[(source, btype)]
        mean, mn, mx = agg(records, metric_key)
        bar_h = mean if not np.isnan(mean) else 0.0
        ax.bar(xi, bar_h, width=bar_width, color=color, hatch=hatch,
               edgecolor="white" if not hatch else color,
               linewidth=0.5, zorder=3, alpha=0.9)
        if not np.isnan(mean):
            ax.errorbar(xi, mean, yerr=[[mean - mn], [mx - mean]],
                        fmt="none", color="0.3", capsize=3, lw=1.0, zorder=4)

    # Annotate gap between bar 0 (base visible) and bar 1 (base blinded)
    rec_bv = all_records[("exp2", "base")]
    rec_bb = all_records[("exp7", "base")]
    mean_bv, *_ = agg(rec_bv, metric_key)
    mean_bb, *_ = agg(rec_bb, metric_key)
    if not np.isnan(mean_bv) and not np.isnan(mean_bb):
        delta = mean_bb - mean_bv
        if abs(delta) > 0.02:
            ax.annotate(f"{delta:+.2f}", xy=(0.5, max(mean_bv, mean_bb) + 0.03),
                        ha="center", va="bottom", fontsize=7, color="0.4")

    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in BAR_GROUPS], rotation=15, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(subtitle, fontsize=10, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(ylim[0] - 0.05, ylim[1] + 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fig D: exp7 blinded detection.")
    ap.add_argument("--slug",     default="anthropic_claude-sonnet-4-6")
    ap.add_argument("--k",        type=int, default=6)
    ap.add_argument("--rep",      type=int, default=1, choices=[0, 1],
                    help="Rep visibility (1=visible, 0=hidden)")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--output",   default=None)
    args = ap.parse_args()

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp7"))
        args.output = os.path.join(fig_dir, "exp7_blinded.pdf")

    slug     = args.slug
    k        = args.k
    rep_vis  = bool(args.rep)
    logs_dir = args.logs_dir

    # Load all four conditions
    all_records = {}
    for label, color, hatch, source, btype in BAR_GROUPS:
        all_records[(source, btype)] = load_condition(
            logs_dir, slug, source, btype, k, rep_visible=rep_vis)
        n = len(all_records[(source, btype)])
        print(f"  {source}/{btype}: {n} seeds loaded", flush=True)

    total = sum(len(v) for v in all_records.values())
    if total == 0:
        warnings.warn(f"No data found for slug='{slug}' k={k}. Saving empty figure.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.8), constrained_layout=True)

    for ax, (metric_key, metric_name, direction, ylim) in zip(axes, METRICS):
        draw_metric_panel(ax, metric_key, all_records, ylim,
                          ylabel=metric_name, subtitle=f"{metric_name}\n({direction})")

    # Legend (shared, in first panel)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_BASE,     edgecolor="white",       label="Base buyer — visible"),
        Patch(facecolor=COLOR_BASE,     edgecolor=COLOR_BASE,    hatch="//", label="Base buyer — blinded"),
        Patch(facecolor=COLOR_GUARDIAN, edgecolor="white",       label="Guardian — visible"),
        Patch(facecolor=COLOR_GUARDIAN, edgecolor=COLOR_GUARDIAN, hatch="//", label="Guardian — blinded"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=7)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
