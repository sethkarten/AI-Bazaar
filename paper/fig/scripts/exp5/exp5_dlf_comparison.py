"""
exp5_dlf_comparison.py — Fig A: Symmetric vs. Asymmetric Transparency.

Two-panel line plot comparing:
  Left:  Consumer visibility — dlc sweep at k=0 (data from exp1)
  Right: Firm visibility     — dlf sweep at k=0 (data from exp5)

X-axis: discovery limit ∈ {1, 3, 5}
Y-axis: bankruptcy rate b_r ∈ [0, 1]
Lines:  one per model; shaded band = min/max across 3 seeds.
Reference: dashed horizontal at b_r = 0.5.

Usage:
    python paper/fig/scripts/exp5/exp5_dlf_comparison.py [--logs-dir logs/] [--output ...]
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", ".."))

# ── Resolve exp1_paths ────────────────────────────────────────────────────────
_EXP1_SCRIPTS = os.path.join(_SCRIPT_DIR, "..", "exp1")
sys.path.insert(0, _EXP1_SCRIPTS)
from exp1_paths import resolve_run_dir as _exp1_resolve

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS = [
    ("gemini-3-flash-preview",      "Gemini 3 Flash", "#0072B2", "o"),
    ("anthropic_claude-sonnet-4.6", "Sonnet 4.6",     "#009E73", "^"),
    ("openai_gpt-5.4",              "GPT 5.4",        "#E69F00", "s"),
]
DL_VALUES = [1, 3, 5]
SEEDS     = [8, 16, 64]

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8.5,
    "lines.linewidth":    1.6,
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
    """Load states.json (preferred) or sorted state_t*.json fallback."""
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
    """Return bankruptcy rate = 1 - (active_firms_final / active_firms_initial)."""
    states = load_states(run_dir)
    if not states:
        return None
    first = sum(1 for f in states[0].get("firms", []) if f.get("in_business"))
    last  = sum(1 for f in states[-1].get("firms", []) if f.get("in_business"))
    if first == 0:
        return None
    return 1.0 - last / first


def exp1_run_dir(logs_dir, slug, dlc, seed):
    """Resolve exp1 k=0 run dir for a given (slug, dlc, seed)."""
    model_logs = os.path.join(logs_dir, f"exp1_{slug}")
    return _exp1_resolve(model_logs, dlc=dlc, n_stab=0, seed=seed, model=slug)


def exp5_run_dir(logs_dir, slug, dlf, seed):
    """Resolve exp5 k=0 run dir for a given (slug, dlf, seed)."""
    name = f"exp5_{slug}_stab_0_dlf{dlf}_seed{seed}"
    d = os.path.join(logs_dir, f"exp5_{slug}", name)
    return d if os.path.isdir(d) else None


def load_model_panel(logs_dir, slug, dl_values, resolve_fn):
    """
    Return {dl: [br_seed1, br_seed2, ...]} for a model across discovery levels.
    resolve_fn(logs_dir, slug, dl, seed) -> run_dir or None
    """
    result = {}
    for dl in dl_values:
        brs = []
        for seed in SEEDS:
            d = resolve_fn(logs_dir, slug, dl, seed)
            if d is None:
                continue
            br = compute_br(d)
            if br is not None:
                brs.append(br)
        if brs:
            result[dl] = brs
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_panel(ax, all_data, models, title, xlabel):
    """Draw one panel (dlc or dlf sweep) on ax."""
    slope_first = None

    for slug, label, color, marker in models:
        data = all_data.get(slug, {})
        xs, means, mins, maxs = [], [], [], []
        for dl in DL_VALUES:
            brs = data.get(dl)
            if not brs:
                continue
            xs.append(dl)
            srs = [1.0 - b for b in brs]  # success rate = 1 - b_r
            means.append(float(np.mean(srs)))
            mins.append(float(np.min(srs)))
            maxs.append(float(np.max(srs)))
        if not xs:
            continue

        xs_arr    = np.array(xs)
        means_arr = np.array(means)
        mins_arr  = np.array(mins)
        maxs_arr  = np.array(maxs)

        ax.plot(xs_arr, means_arr, color=color, marker=marker,
                markersize=5, label=label, zorder=4)
        ax.fill_between(xs_arr, mins_arr, maxs_arr,
                        color=color, alpha=0.15, zorder=2)

        # Capture slope for first non-empty model (for callout)
        if slope_first is None and len(xs) >= 2:
            slope_first = float(np.polyfit(xs_arr, means_arr, 1)[0])

    # Reference line at success rate = 0.5 (stability threshold)
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--", zorder=1)

    ax.set_xticks(DL_VALUES)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Success Rate $1 - b_r$")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return slope_first


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fig A: dlc vs dlf comparison.")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good",     default="food")
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--output",   default=None)
    args = ap.parse_args()

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp5"))
        args.output = os.path.join(fig_dir, "exp5_dlf_comparison.pdf")

    logs_dir = args.logs_dir

    # Load data in parallel
    dlc_data = {}  # slug -> {dl: [br, ...]}
    dlf_data = {}

    def _load_exp1(entry):
        slug = entry[0]
        return slug, load_model_panel(logs_dir, slug, DL_VALUES, exp1_run_dir)

    def _load_exp5(entry):
        slug = entry[0]
        return slug, load_model_panel(logs_dir, slug, DL_VALUES, exp5_run_dir)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs1 = {pool.submit(_load_exp1, m): m[0] for m in MODELS}
        futs5 = {pool.submit(_load_exp5, m): m[0] for m in MODELS}
        for fut in as_completed(futs1):
            slug, d = fut.result()
            n = sum(len(v) for v in d.values())
            print(f"  exp1 {slug}: {n} seed-cells loaded", flush=True)
            dlc_data[slug] = d
        for fut in as_completed(futs5):
            slug, d = fut.result()
            n = sum(len(v) for v in d.values())
            print(f"  exp5 {slug}: {n} seed-cells loaded", flush=True)
            dlf_data[slug] = d

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.75, 2.8),
                                             constrained_layout=True)

    slope_dlc = draw_panel(ax_left,  dlc_data, MODELS,
                           "Consumer visibility ($dlc$ sweep)",
                           "Consumer discovery limit $dlc$")
    slope_dlf = draw_panel(ax_right, dlf_data, MODELS,
                           "Firm visibility ($dlf$ sweep)",
                           "Firm discovery limit $dlf$")

    # Shared legend on left panel
    handles, labels = ax_left.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_right.get_legend_handles_labels()
    if handles:
        ax_left.legend(handles, labels, loc="upper left", fontsize=8)


    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
