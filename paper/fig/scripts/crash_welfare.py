"""
Fig C4: Welfare Cost of the Crash

Top panel: aggregate consumer surplus over time (mean ± std across runs).
Bottom panel: Gini coefficient of firm cash — rising inequality as firms go bankrupt.

Contrasts with VendingBench's individual-agent net worth framing by showing
distributional / welfare consequences of the crash at the market level.

Usage:
    python crash_welfare.py --run-dirs logs/crash_seed42 logs/crash_seed1 logs/crash_seed7
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


def main():
    parser = argparse.ArgumentParser(description="Fig C4: Crash welfare cost")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "crash_welfare.pdf"))
    args = parser.parse_args()

    surplus_series, gini_series = [], []
    all_ts = None

    for run_dir in args.run_dirs:
        files = load_run(run_dir)
        if not files:
            print(f"Warning: no state files in {run_dir}", file=sys.stderr)
            continue
        db = DataFrameBuilder(states=files)

        # Consumer surplus (mean across consumers per timestep)
        cs_df = db.consumer_surplus_per_consumer_over_time()
        cs_agg = cs_df.groupby("timestep")["value"].mean().reset_index().sort_values("timestep")

        # Gini of firm cash
        gini_df = db.metrics_over_time(metrics=["gini"])

        # Only use firm cash for gini (filter ledger to firm_* keys only)
        # metrics_over_time uses all agents; compute firm-only gini manually
        from ai_bazaar.utils.dataframe_builder import _gini
        firm_gini_rows = []
        for s in db.states:
            t = s["timestep"]
            firm_cash = [v for k, v in s["ledger"]["money"].items() if k.startswith("firm_")]
            firm_gini_rows.append({"timestep": t, "gini": _gini(firm_cash)})

        import pandas as pd
        gini_firm = pd.DataFrame(firm_gini_rows).sort_values("timestep")

        ts = cs_agg["timestep"].values
        if all_ts is None:
            all_ts = ts
        surplus_series.append(cs_agg["value"].values)
        gini_series.append(gini_firm["gini"].values)

    if not surplus_series:
        print("No data loaded.", file=sys.stderr)
        sys.exit(1)

    min_len = min(len(s) for s in surplus_series + gini_series)
    ts = all_ts[:min_len]
    surplus_arr = np.array([s[:min_len] for s in surplus_series])
    gini_arr = np.array([s[:min_len] for s in gini_series])

    surplus_mean, surplus_std = surplus_arr.mean(0), surplus_arr.std(0)
    gini_mean, gini_std = gini_arr.mean(0), gini_arr.std(0)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    color_surplus = "#1f77b4"
    ax_top.plot(ts, surplus_mean, color=color_surplus, label=f"Mean consumer surplus (n={len(args.run_dirs)})")
    ax_top.fill_between(ts, surplus_mean - surplus_std, surplus_mean + surplus_std,
                        color=color_surplus, alpha=0.2)
    ax_top.set_ylabel("Mean consumer surplus", fontsize=12)
    ax_top.set_title("The Crash: Welfare Cost", fontsize=13)
    ax_top.legend(loc="upper right")

    color_gini = "#d62728"
    ax_bot.plot(ts, gini_mean, color=color_gini, label="Firm cash Gini")
    ax_bot.fill_between(ts, np.clip(gini_mean - gini_std, 0, 1), np.clip(gini_mean + gini_std, 0, 1),
                        color=color_gini, alpha=0.2)
    ax_bot.set_ylabel("Firm cash Gini coefficient", fontsize=12)
    ax_bot.set_xlabel("Timestep $t$", fontsize=12)
    ax_bot.set_ylim(0, 1.05)
    ax_bot.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
