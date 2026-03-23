"""
Fig: Experiment 1 Token Usage — input and output tokens across (n_stab, dlc) configurations.

Two-panel figure:
  (A) Input tokens  — grouped bars, n_stab on x-axis, dlc encoded by color,
                      mean bar height with per-seed dots
  (B) Output tokens — same structure, independent y-scale

n_stab=0 (baseline): only dlc=3, seed=8 (single bar, no seed-variance dots).
All other cells: seeds 8, 16, 64.  Missing run dirs silently skipped.

Usage:
    python paper/fig/scripts/exp1/exp1_tokens.py [--logs-dir logs/] [--output ...]
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
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from exp1_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data

# ── Experiment matrix ──────────────────────────────────────────────────────
N_STAB_VALUES = [0, 1, 2, 4, 5]
DLC_VALUES    = [1, 3, 5]
SEEDS         = [8, 16, 64]

# ── Okabe-Ito palette — dlc encoded by color ───────────────────────────────
DLC_COLORS = {1: "#0072B2", 3: "#E69F00", 5: "#009E73"}
DLC_LABELS = {1: "dlc = 1", 3: "dlc = 3", 5: "dlc = 5"}

# ── Bar layout constants ───────────────────────────────────────────────────
BAR_W       = 0.20
DLC_OFFSETS = {1: -0.22, 3: 0.0, 5: 0.22}
GROUP_STEP  = 1.0

# ── rcParams (FigureMakerAgent standard) ──────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.4,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":        False,
    "pdf.fonttype":       42,
})

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "exp1", "exp1_tokens.pdf"
)


# ── Directory resolution (mirrors other exp1 figure scripts exactly) ───────

def resolve_run_dir(logs_dir, dlc, n_stab, seed, model=""):
    """Return run directory path for given config; None if it doesn't exist."""
    if model:
        if n_stab == 0:
            if dlc == 3 and seed == 8:
                path = os.path.join(logs_dir, f"exp1_{model}_baseline")
                return path if os.path.isdir(path) else None
            return None
        path = os.path.join(logs_dir, f"exp1_{model}_stab_{n_stab}_dlc{dlc}_seed{seed}")
        return path if os.path.isdir(path) else None
    if n_stab == 0:
        if dlc == 3 and seed == 8:
            path = os.path.join(logs_dir, "exp1_baseline")
            return path if os.path.isdir(path) else None
        return None
    if n_stab == 5:
        path = os.path.join(logs_dir, f"exp1_stab_5_dlc{dlc}_seed{seed}")
        return path if os.path.isdir(path) else None
    path = os.path.join(logs_dir, f"exp1_stab_{n_stab}_dlc{dlc}_seed{seed}")
    return path if os.path.isdir(path) else None


def collect_run_dirs(logs_dir, model=""):
    dirs = []
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if d:
                    dirs.append(d)
    return dirs


# ── Data loading ───────────────────────────────────────────────────────────

def load_token_file(run_dir):
    """Return (input_tokens, output_tokens) from the run's token_usage.json, or None."""
    matches = glob.glob(os.path.join(run_dir, "*_token_usage.json"))
    if not matches:
        return None
    try:
        with open(matches[0]) as fh:
            j = json.load(fh)
        return {
            "input":  int(j.get("input_tokens", 0)),
            "output": int(j.get("output_tokens", 0)),
        }
    except Exception:
        return None


def load_all_tokens(logs_dir, model=""):
    """
    Returns nested dict:
        data[n_stab][dlc] = list of {"input": int, "output": int} per seed
    Cells with no data are absent from the inner dict.
    """
    data = {n_stab: {} for n_stab in N_STAB_VALUES}
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            records = []
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if run_dir is None:
                    continue
                rec = load_token_file(run_dir)
                if rec is not None:
                    records.append(rec)
            if records:
                data[n_stab][dlc] = records
    return data


# ── Cache serialization ────────────────────────────────────────────────────

def _serialize(data):
    """Flatten nested {n_stab: {dlc: [records]}} to {"n_stab,dlc": [records]}."""
    out = {}
    for n_stab, dlc_map in data.items():
        for dlc, records in dlc_map.items():
            out[f"{n_stab},{dlc}"] = records
    return out


def _deserialize(raw):
    data = {n_stab: {} for n_stab in N_STAB_VALUES}
    for key, records in raw.items():
        n_stab_s, dlc_s = key.split(",")
        n_stab, dlc = int(n_stab_s), int(dlc_s)
        if n_stab in data:
            data[n_stab][dlc] = records
    return data


# ── Drawing ────────────────────────────────────────────────────────────────

def _group_center(g_idx):
    return g_idx * GROUP_STEP


def draw_panel(ax, data, key, ylabel, panel_label, show_legend=False):
    """Draw one panel (key = "input" or "output")."""
    for g_idx, n_stab in enumerate(N_STAB_VALUES):
        cx = _group_center(g_idx)
        for dlc in DLC_VALUES:
            if dlc not in data[n_stab]:
                continue
            records = data[n_stab][dlc]
            vals = [r[key] for r in records]
            mean_val = float(np.mean(vals))
            color = DLC_COLORS[dlc]
            x = cx + DLC_OFFSETS[dlc]

            ax.bar(
                x, mean_val,
                width=BAR_W * 0.88,
                color=color,
                alpha=0.82,
                zorder=3,
                label=DLC_LABELS[dlc] if g_idx == 0 else "_nolegend_",
            )

            for v in vals:
                ax.scatter(
                    x, v,
                    s=14,
                    color=color,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=5,
                )

    tick_xs = [_group_center(i) for i in range(len(N_STAB_VALUES))]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels([f"$k={k}$" for k in N_STAB_VALUES])
    ax.set_xlabel("Stabilizing firms $k$")
    ax.set_xlim(tick_xs[0] - 0.55, tick_xs[-1] + 0.55)

    if key == "input":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M")
        )
    else:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k")
        )

    ax.set_ylabel(ylabel)
    ax.set_title(panel_label, loc="left", fontweight="normal")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linewidth=0.5, color="0.85", zorder=0)
    ax.set_axisbelow(True)

    if show_legend:
        handles = [
            mpatches.Patch(color=DLC_COLORS[d], label=DLC_LABELS[d], alpha=0.82)
            for d in DLC_VALUES
        ]
        ax.legend(handles=handles, loc="upper left", framealpha=0.9)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Exp1 token usage figure.")
    ap.add_argument("--logs-dir", default="logs/",
                    help="Directory containing run folders (default: logs/)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Output PDF path")
    # Accepted for compatibility with exp1_run_all.py; not used by this script.
    ap.add_argument("--good",    default="food")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model",   default="")
    args = ap.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp1_tokens", "tokens")
    run_dirs   = collect_run_dirs(args.logs_dir, args.model)

    if is_cache_fresh(cache_path, run_dirs, args.logs_dir, "tokens"):
        print(f"Using cached data: {cache_path}", flush=True)
        data = _deserialize(load_cache_data(cache_path))
    else:
        print(f"Loading runs from: {args.logs_dir}", flush=True)
        data = load_all_tokens(args.logs_dir, model=args.model)
        save_cache(cache_path, _serialize(data), args.logs_dir, "tokens")
        print(f"Cached data: {cache_path}", flush=True)

    n_cells = sum(len(v) for v in data.values())
    print(f"Cells with token data: {n_cells}", flush=True)

    fig, (ax_in, ax_out) = plt.subplots(
        1, 2,
        figsize=(7.0, 3.2),
        constrained_layout=True,
    )

    draw_panel(ax_in,  data, key="input",  ylabel="Input tokens",
               panel_label="(A) Input Tokens",  show_legend=True)
    draw_panel(ax_out, data, key="output", ylabel="Output tokens",
               panel_label="(B) Output Tokens", show_legend=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
