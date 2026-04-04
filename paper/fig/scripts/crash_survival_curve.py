"""
Fig C2: Firm Survival Curve — The Crash (Kaplan-Meier style)

Y-axis: fraction of initial firms still in business.
One curve per condition (baseline vs stabilizing firm), with shaded
confidence bands across seeds.

Usage:
    python crash_survival_curve.py \
        --baseline-dirs logs/crash_seed42 logs/crash_seed1 \
        --stabilizing-dirs logs/crash_stab_seed42 logs/crash_stab_seed1
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
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
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


def survival_series(run_dirs):
    """
    Returns (timesteps, mean_fraction, std_fraction) across runs.
    Fraction = active_firms / initial_firm_count.
    """
    all_series = []
    all_timesteps = None
    for run_dir in run_dirs:
        files = load_run(run_dir)
        if not files:
            continue
        db = DataFrameBuilder(states=files)
        df = db.firms_in_business_over_time().sort_values("timestep")
        initial = df["value"].iloc[0] if not df.empty else 1
        if initial == 0:
            initial = 1
        frac = df["value"] / initial
        ts = df["timestep"].values
        if all_timesteps is None:
            all_timesteps = ts
        all_series.append(frac.values)

    if not all_series:
        return np.array([]), np.array([]), np.array([])

    # Align to shortest run
    min_len = min(len(s) for s in all_series)
    trimmed = np.array([s[:min_len] for s in all_series])
    return all_timesteps[:min_len], trimmed.mean(axis=0), trimmed.std(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Fig C2: Crash survival curve")
    parser.add_argument("--baseline-dirs", nargs="+", default=[])
    parser.add_argument("--stabilizing-dirs", nargs="+", default=[])
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "crash_survival_curve.pdf"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4))

    conditions = []
    if args.baseline_dirs:
        conditions.append(("Baseline", args.baseline_dirs, "#d62728", "-"))
    if args.stabilizing_dirs:
        conditions.append(("Stabilizing firm", args.stabilizing_dirs, "#2ca02c", "--"))

    for label, dirs, color, ls in conditions:
        ts, mean, std = survival_series(dirs)
        if len(ts) == 0:
            continue
        ax.plot(ts, mean, color=color, linestyle=ls, label=f"{label} (n={len(dirs)})")
        ax.fill_between(ts, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                        color=color, alpha=0.15)

    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel("Fraction of firms in business", fontsize=12)
    ax.set_title("The Crash: Firm Survival Curve", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
