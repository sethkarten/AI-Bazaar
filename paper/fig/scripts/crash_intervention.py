"""
Fig C3: Stabilizing Firm Intervention — The Crash

3-row panel comparing baseline vs stabilizing-firm condition:
  (a) Average market price over time
  (b) Active firm count over time
  (c) Mean consumer surplus over time

Multiple runs per condition are aggregated (mean ± std band).

Usage:
    python crash_intervention.py \
        --baseline-dirs logs/crash_seed42 logs/crash_seed1 \
        --stabilizing-dirs logs/crash_stab_seed42 logs/crash_stab_seed1 \
        --good food
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
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CONDITIONS = {
    "Baseline":          ("#d62728", "-"),
    "Stabilizing firm":  ("#2ca02c", "--"),
}


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


def aggregate_series(run_dirs, extract_fn):
    """Run extract_fn(DataFrameBuilder) -> (timesteps_array, values_array) for each run, aggregate."""
    all_ts, all_vals = None, []
    for run_dir in run_dirs:
        files = load_run(run_dir)
        if not files:
            continue
        db = DataFrameBuilder(states=files)
        ts, vals = extract_fn(db)
        if all_ts is None:
            all_ts = ts
        all_vals.append(vals)
    if not all_vals:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(v) for v in all_vals)
    arr = np.array([v[:min_len] for v in all_vals])
    return all_ts[:min_len], arr.mean(axis=0), arr.std(axis=0)


def extract_avg_price(good):
    def _fn(db):
        df = db.price_per_firm_over_time(good)
        avg = df.groupby("timestep")["value"].mean().reset_index().sort_values("timestep")
        return avg["timestep"].values, avg["value"].values
    return _fn


def extract_active_firms(db):
    df = db.firms_in_business_over_time().sort_values("timestep")
    return df["timestep"].values, df["value"].values.astype(float)


def extract_consumer_surplus(db):
    df = db.consumer_surplus_per_consumer_over_time()
    agg = df.groupby("timestep")["value"].mean().reset_index().sort_values("timestep")
    return agg["timestep"].values, agg["value"].values


def main():
    parser = argparse.ArgumentParser(description="Fig C3: Crash intervention comparison")
    parser.add_argument("--baseline-dirs", nargs="+", default=[])
    parser.add_argument("--stabilizing-dirs", nargs="+", default=[])
    parser.add_argument("--good", default=None)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "crash_intervention.pdf"))
    args = parser.parse_args()

    # Auto-detect good name
    good = args.good
    if good is None:
        all_dirs = (args.baseline_dirs or []) + (args.stabilizing_dirs or [])
        for d in all_dirs:
            files = load_run(d)
            if files:
                db = DataFrameBuilder(states=files)
                goods = db._all_good_names()
                if goods:
                    good = goods[0]
                    break
        good = good or "food"

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax_price, ax_firms, ax_surplus = axes

    condition_dirs = []
    if args.baseline_dirs:
        condition_dirs.append(("Baseline", args.baseline_dirs))
    if args.stabilizing_dirs:
        condition_dirs.append(("Stabilizing firm", args.stabilizing_dirs))

    for label, dirs in condition_dirs:
        color, ls = CONDITIONS.get(label, ("#7f7f7f", "-"))

        # (a) Average price
        ts, mean, std = aggregate_series(dirs, extract_avg_price(good))
        if len(ts):
            ax_price.plot(ts, mean, color=color, linestyle=ls, label=f"{label} (n={len(dirs)})")
            ax_price.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)

        # (b) Active firms
        ts, mean, std = aggregate_series(dirs, extract_active_firms)
        if len(ts):
            ax_firms.plot(ts, mean, color=color, linestyle=ls)
            ax_firms.fill_between(ts, np.clip(mean - std, 0, None), mean + std, color=color, alpha=0.15)

        # (c) Consumer surplus
        ts, mean, std = aggregate_series(dirs, extract_consumer_surplus)
        if len(ts):
            ax_surplus.plot(ts, mean, color=color, linestyle=ls)
            ax_surplus.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)

    ax_price.set_ylabel("Avg. market price ($\\$$)", fontsize=11)
    ax_price.set_title("The Crash: Intervention Comparison", fontsize=12)
    ax_price.legend(loc="upper right")

    ax_firms.set_ylabel("Active firms", fontsize=11)
    ax_firms.yaxis.get_major_locator().set_params(integer=True)

    ax_surplus.set_ylabel("Mean consumer surplus", fontsize=11)
    ax_surplus.set_xlabel("Timestep $t$", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
