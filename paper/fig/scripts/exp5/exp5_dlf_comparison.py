"""
exp5_dlf_comparison.py — Fig: Firm Visibility (dlf) vs Baseline (dlc=3).

1×3 grouped bar chart — one panel per k (stabilizing firm count): k ∈ {1, 3, 5}.
Each panel shows the bankruptcy rate for three conditions, all at dlc=3 / gemini-3-flash-preview:

  (A) exp1,  dlc=3        — baseline consumer-visibility only
  (B) exp5,  dlf=1        — firm-visibility added, dlf=1
  (C) exp5,  dlf=3        — firm-visibility added, dlf=3

Error bars = [min, max] across seeds {8, 16, 64}.
Reference dashed line at b_r = 0.5.

Usage:
    python paper/fig/scripts/exp5/exp5_dlf_comparison.py [--logs-dir logs/] [--output ...]
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "exp1"))
from exp1_paths import resolve_run_dir as _exp1_resolve

# ── Constants ─────────────────────────────────────────────────────────────────

SLUG  = "gemini-3-flash-preview"
SEEDS = [8, 16, 64]
K_VALUES = [1, 3, 5]   # n_stab values
DLC_BASELINE = 3
DLF_VALUES   = [1, 3]  # dlf=5 excluded (redundant)

# Okabe-Ito
COLOR_EXP1  = "#56B4E9"   # sky blue      — exp1 baseline
COLOR_DLF1  = "#D55E00"   # vermillion    — exp5 dlf=1
COLOR_DLF3  = "#009E73"   # bluish-green  — exp5 dlf=3
COLOR_DLF5  = "#E69F00"   # orange        — exp5 dlf=5

CONDITIONS = [
    ("exp1\ndlc=3", COLOR_EXP1,  "exp1", 3),
    ("exp5\ndlf=1", COLOR_DLF1,  "exp5", 1),
    ("exp5\ndlf=3", COLOR_DLF3,  "exp5", 3),
    ("exp1\ndlc,dlf=5", COLOR_DLF5,  "exp1", 5),
]

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.5,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.alpha":         0.3,
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


def compute_br(run_dir):
    """Bankruptcy rate = 1 - (firms_active_final / firms_active_initial)."""
    if run_dir is None or not os.path.isdir(run_dir):
        return None
    states = load_states(run_dir)
    if not states:
        return None
    first = sum(1 for f in states[0].get("firms", []) if f.get("in_business"))
    last  = sum(1 for f in states[-1].get("firms", []) if f.get("in_business"))
    if first == 0:
        return None
    return 1.0 - last / first


def exp1_dir(logs_dir, k, seed):
    model_logs = os.path.join(logs_dir, f"exp1_{SLUG}")
    return _exp1_resolve(model_logs, dlc=DLC_BASELINE, n_stab=k, seed=seed, model=SLUG)


def exp5_dir(logs_dir, k, dlf, seed):
    name = f"exp5_{SLUG}_stab_{k}_dlf{dlf}_seed{seed}"
    d = os.path.join(logs_dir, f"exp5_{SLUG}", name)
    return d if os.path.isdir(d) else None


def load_condition_brs(logs_dir, k, exp, dl):
    """Return list of b_r values across seeds for one (k, exp, dl) condition.

    For exp1: dl is the dlc value.
    For exp5: dl is the dlf value.
    """
    brs = []
    for seed in SEEDS:
        if exp == "exp1":
            model_logs = os.path.join(logs_dir, f"exp1_{SLUG}")
            d = _exp1_resolve(model_logs, dlc=dl, n_stab=k, seed=seed, model=SLUG)
        else:
            d = exp5_dir(logs_dir, k, dl, seed)
        br = compute_br(d)
        if br is not None:
            brs.append(br)
    return brs


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_panel(ax, k, data, title):
    """
    Draw one bar-group panel for a given k value.
    data: list of (label, color, brs) per condition.
    """
    xs = np.arange(len(data))
    bar_w = 0.55

    for xi, (label, color, brs) in enumerate(data):
        if not brs:
            continue
        srs  = [1.0 - b for b in brs]
        mean = float(np.mean(srs))
        lo   = float(np.min(srs))
        hi   = float(np.max(srs))
        ax.bar(xi, mean, width=bar_w, color=color, edgecolor="white",
               linewidth=0.5, zorder=3, alpha=0.9)
        ax.errorbar(xi, mean, yerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]],
                    fmt="none", color="0.3", capsize=3, lw=1.0, zorder=4)

    ax.axhline(0.5, color="#555555", lw=0.9, ls="--", zorder=2, alpha=0.8)

    ax.set_xticks(xs)
    ax.set_xticklabels([d[0] for d in data], fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Success Rate $1 - b_r$", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fig: dlf comparison bankruptcy rate.")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--output",   default=None)
    args = ap.parse_args()

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp5"))
        args.output = os.path.join(fig_dir, "exp5_dlf_comparison.pdf")

    logs_dir = args.logs_dir

    # Load all conditions
    # all_data[k] = list of (label, color, brs) for each condition
    all_data = {}
    for k in K_VALUES:
        panel_data = []
        for label, color, exp, dl in CONDITIONS:
            brs = load_condition_brs(logs_dir, k, exp, dl)
            n = len(brs)
            cond_str = f"exp1 dlc={dl}" if exp == "exp1" else f"exp5 dlf={dl}"
            print(f"  k={k}  {cond_str}: {n} seeds  br={[f'{b:.2f}' for b in brs]}", flush=True)
            panel_data.append((label, color, brs))
        all_data[k] = panel_data

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.8), constrained_layout=True)

    for ax, k in zip(axes, K_VALUES):
        draw_panel(ax, k, all_data[k], title=f"$k = {k}$ stabilizing firms")

    # Only first panel gets y-label; remove from others for space
    for ax in axes[1:]:
        ax.set_ylabel("")

    # Shared legend below panels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_EXP1, edgecolor="white", label=f"exp1, dlc={DLC_BASELINE} (baseline)"),
        Patch(facecolor=COLOR_DLF1, edgecolor="white", label="exp5, dlf=1"),
        Patch(facecolor=COLOR_DLF3, edgecolor="white", label="exp5, dlf=3"),
        Patch(facecolor=COLOR_DLF5, edgecolor="white", label="exp1, dlc=dlf=5"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8,
               frameon=True, framealpha=0.9, edgecolor="0.8")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
