"""
exp5_dlf_heatmap.py — Fig B: Harness Stability Threshold Under Firm Visibility.

Heatmap of bankruptcy rate over the k × dlf grid for exp5.
Analogous to the exp1 heatmap over k × dlc.

Grid:
  Rows:    k ∈ {0, 1, 3, 5}   (stabilizing firms)
  Columns: dlf ∈ {1, 3, 5}    (firm discovery limit)
  Color:   bankruptcy rate b_r  (YlOrRd, [0, 1])
  Text:    mean b_r per cell

Missing cells rendered as hatched gray.

Usage:
    python paper/fig/scripts/exp5/exp5_dlf_heatmap.py [--slug gemini-3-flash-preview] \\
        [--logs-dir logs/] [--output ...]
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

K_VALUES   = [0, 1, 3, 5]
DLF_VALUES = [1, 3, 5]
SEEDS      = [8, 16, 64]

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "lines.linewidth":    1.5,
    "axes.linewidth":     0.8,
    "axes.grid":          False,
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


def build_grid(logs_dir, slug):
    """Return (grid_mean, grid_available) each shape (len(K_VALUES), len(DLF_VALUES))."""
    n_rows = len(K_VALUES)
    n_cols = len(DLF_VALUES)
    grid_mean  = np.full((n_rows, n_cols), np.nan)
    grid_avail = np.zeros((n_rows, n_cols), dtype=bool)

    for i, k in enumerate(K_VALUES):
        for j, dlf in enumerate(DLF_VALUES):
            brs = []
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, slug, k, dlf, seed)
                if d is None:
                    continue
                br = compute_br(d)
                if br is not None:
                    brs.append(br)
            if brs:
                grid_mean[i, j]  = float(np.mean(brs))
                grid_avail[i, j] = True

    return grid_mean, grid_avail


# ── Drawing utilities ─────────────────────────────────────────────────────────

def draw_hatch_cell(ax, col_idx, row_idx):
    """Draw a hatched rectangle over a missing cell."""
    rect = mpatches.FancyBboxPatch(
        (col_idx - 0.5, row_idx - 0.5), 1.0, 1.0,
        boxstyle="square,pad=0",
        linewidth=0,
        facecolor="#cccccc",
        hatch="///",
        edgecolor="#888888",
        zorder=5,
    )
    ax.add_patch(rect)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fig B: exp5 dlf heatmap.")
    ap.add_argument("--slug",     default="gemini-3-flash-preview")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--output",   default=None)
    args = ap.parse_args()

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp5"))
        args.output = os.path.join(fig_dir, "exp5_dlf_heatmap.pdf")

    print(f"Loading exp5 data for slug='{args.slug}' ...", flush=True)
    grid_mean, grid_avail = build_grid(args.logs_dir, args.slug)
    n_avail = int(np.sum(grid_avail))
    print(f"  {n_avail} / {len(K_VALUES)*len(DLF_VALUES)} cells with data", flush=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5), constrained_layout=True)

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Render heatmap
    im = ax.imshow(grid_mean, cmap=cmap, norm=norm,
                   origin="upper", aspect="auto",
                   extent=[-0.5, len(DLF_VALUES) - 0.5,
                            len(K_VALUES) - 0.5, -0.5])

    # Hatch missing cells; annotate present cells
    for i in range(len(K_VALUES)):
        for j in range(len(DLF_VALUES)):
            if not grid_avail[i, j]:
                draw_hatch_cell(ax, j, i)
            else:
                val = grid_mean[i, j]
                norm_val = norm(val)
                rgba = cmap(norm_val)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if lum < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=txt_color, fontweight="bold", zorder=6)

    ax.set_xticks(range(len(DLF_VALUES)))
    ax.set_xticklabels([f"$dlf={d}$" for d in DLF_VALUES])
    ax.set_yticks(range(len(K_VALUES)))
    ax.set_yticklabels([f"$k={k}$" for k in K_VALUES])
    ax.set_xlabel("Firm Discovery Limit ($dlf$)")
    ax.set_ylabel("Stabilizing Firms ($k$)")
    ax.set_title(f"Bankruptcy Rate — {args.slug}", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Bankruptcy rate $b_r$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
