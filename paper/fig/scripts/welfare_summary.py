"""
Fig S1: Welfare Summary — Grouped Bar Chart

Final-timestep mean consumer surplus (± std) across four conditions:
  1. Crash — Baseline
  2. Crash — Stabilizing firm
  3. Lemon Market — Baseline
  4. Lemon Market — Skeptical guardian

Intended as a single-panel overview figure showing that both interventions
improve welfare. Suitable for abstract, intro, or conclusion sections.

Usage:
    python welfare_summary.py \
        --crash-baseline-dirs    logs/crash_seed42 logs/crash_seed1 \
        --crash-stabilizing-dirs logs/crash_stab_seed42 logs/crash_stab_seed1 \
        --lemon-baseline-dirs    logs/lemon_seed42 logs/lemon_seed1 \
        --lemon-guardian-dirs    logs/lemon_guardian_seed42 logs/lemon_guardian_seed1
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
})


def load_run(run_dir):
    states_path = os.path.join(run_dir, "states.json")
    if os.path.isfile(states_path):
        with open(states_path) as f:
            return json.load(f)
    files = glob.glob(os.path.join(run_dir, "state_t*.json"))
    files.sort(key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or "0"))
    states = []
    for p in files:
        with open(p) as f:
            states.append(json.load(f))
    return states


def final_consumer_surplus(run_dirs):
    """Return list of final-timestep mean consumer surplus, one value per run."""
    vals = []
    for run_dir in run_dirs:
        files = load_run(run_dir)
        if not files:
            continue
        db = DataFrameBuilder(states=files)
        cs = db.consumer_surplus_per_consumer_over_time()
        if cs.empty:
            continue
        last_t = cs["timestep"].max()
        final_mean = cs[cs["timestep"] == last_t]["value"].mean()
        vals.append(final_mean)
    return vals


def main():
    parser = argparse.ArgumentParser(description="Fig S1: Welfare summary bar chart")
    parser.add_argument("--crash-baseline-dirs",    nargs="+", default=[])
    parser.add_argument("--crash-stabilizing-dirs", nargs="+", default=[])
    parser.add_argument("--lemon-baseline-dirs",    nargs="+", default=[])
    parser.add_argument("--lemon-guardian-dirs",    nargs="+", default=[])
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "welfare_summary.pdf"))
    args = parser.parse_args()

    conditions = [
        ("Crash\nBaseline",         args.crash_baseline_dirs,    "#d62728"),
        ("Crash\nStabilizing firm", args.crash_stabilizing_dirs, "#2ca02c"),
        ("Lemon\nBaseline",         args.lemon_baseline_dirs,    "#ff7f0e"),
        ("Lemon\nGuardian",         args.lemon_guardian_dirs,    "#1f77b4"),
    ]

    labels, means, stds, colors, ns = [], [], [], [], []
    for label, dirs, color in conditions:
        vals = final_consumer_surplus(dirs)
        labels.append(label)
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        colors.append(color)
        ns.append(len(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=6, edgecolor="black", linewidth=0.8)

    # Annotate n
    for bar, n, mean in zip(bars, ns, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
                f"n={n}", ha="center", va="bottom", fontsize=9, color="gray")

    # Bracket showing intervention effect for crash
    if means[0] > 0 and means[1] > 0:
        y_top = max(means[0], means[1]) + max(stds) * 1.5
        ax.annotate("", xy=(x[1], y_top), xytext=(x[0], y_top),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
        ax.text((x[0] + x[1]) / 2, y_top * 1.02, "intervention\neffect",
                ha="center", va="bottom", fontsize=8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Final mean consumer surplus", fontsize=12)
    ax.set_title("Welfare Summary: Baseline vs Intervention", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
