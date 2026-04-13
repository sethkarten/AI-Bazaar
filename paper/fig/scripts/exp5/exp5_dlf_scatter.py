"""
exp5_dlf_scatter.py — Fig: DLF Ablation — Bankruptcy Rate vs k.

1×3 line chart, one panel per dlf value (1, 3, 5).
Each panel plots bankruptcy rate vs stabilizing firms k ∈ {0, 1, 3, 5}
for gemini-3-flash-preview, with shaded min/max band across seeds {8, 16, 64}.

Run naming: logs/exp5_{slug}/exp5_{slug}_stab_{k}_dlf{dlf}_seed{seed}/

Usage:
    python paper/fig/scripts/exp5/exp5_dlf_scatter.py [--logs-dir logs/] [--output ...]
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

# ── Constants ─────────────────────────────────────────────────────────────────

SLUG      = "gemini-3-flash-preview"
SEEDS     = [8, 16, 64]
K_VALUES  = [0, 1, 3, 5]
DLF_VALUES = [1, 3, 5]

# Okabe-Ito — one model, use blue
MODELS = [
    ("Gemini 3 Flash", SLUG, "#0072B2", "o"),
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
    "lines.markersize":   5,
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


def resolve_run_dir(logs_dir, slug, k, dlf, seed):
    name = f"exp5_{slug}_stab_{k}_dlf{dlf}_seed{seed}"
    d = os.path.join(logs_dir, f"exp5_{slug}", name)
    return d if os.path.isdir(d) else None


def load_model_sweep(logs_dir, slug, dlf):
    """Return {k: [br, ...]} across seeds for one (slug, dlf) combination."""
    result = {}
    for k in K_VALUES:
        brs = []
        for seed in SEEDS:
            d = resolve_run_dir(logs_dir, slug, k, dlf, seed)
            br = compute_br(d)
            if br is not None:
                brs.append(br)
        if brs:
            result[k] = brs
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_panel(ax, sweep_data_list, dlf, show_ylabel, show_legend):
    """
    Draw one panel for a given dlf value.
    sweep_data_list: list of (label, color, marker, {k: [brs]})
    """
    for label, color, marker, sweep in sweep_data_list:
        xs, means, mins, maxs = [], [], [], []
        for k in K_VALUES:
            brs = sweep.get(k)
            if not brs:
                continue
            srs = [1.0 - b for b in brs]
            xs.append(k)
            means.append(float(np.mean(srs)))
            mins.append(float(np.min(srs)))
            maxs.append(float(np.max(srs)))
        if not xs:
            continue
        xs = np.array(xs)
        ax.plot(xs, means, color=color, marker=marker, markersize=5,
                lw=1.5, label=label, zorder=4)
        ax.fill_between(xs, mins, maxs, color=color, alpha=0.15, zorder=2)

    ax.axhline(0.5, color="#555555", lw=1.0, ls="--", alpha=0.8, zorder=3)
    ax.set_xticks(K_VALUES)
    ax.set_xlim(-0.4, K_VALUES[-1] + 0.4)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Stabilizing Firms ($k$)")
    if show_ylabel:
        ax.set_ylabel("Success Rate $1 - b_r$")
    ax.set_title(f"$dlf = {dlf}$", fontsize=10, fontweight="bold")
    if show_legend:
        ax.legend(loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fig: DLF ablation bankruptcy rate.")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--output",   default=None)
    args = ap.parse_args()

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp5"))
        args.output = os.path.join(fig_dir, "exp5_dlf_scatter.pdf")

    logs_dir = args.logs_dir

    # Load all data
    # all_sweeps[dlf] = list of (label, color, marker, {k: [brs]})
    all_sweeps = {}
    for dlf in DLF_VALUES:
        panel_data = []
        for label, slug, color, marker in MODELS:
            sweep = load_model_sweep(logs_dir, slug, dlf)
            n = sum(len(v) for v in sweep.values())
            print(f"  dlf={dlf}  {label}: {n} seed-cells", flush=True)
            panel_data.append((label, color, marker, sweep))
        all_sweeps[dlf] = panel_data

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.8), constrained_layout=True,
                             sharey=True)

    for ax, dlf in zip(axes, DLF_VALUES):
        show_ylabel  = (dlf == DLF_VALUES[0])
        show_legend  = (dlf == DLF_VALUES[-1])
        draw_panel(ax, all_sweeps[dlf], dlf,
                   show_ylabel=show_ylabel, show_legend=show_legend)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
