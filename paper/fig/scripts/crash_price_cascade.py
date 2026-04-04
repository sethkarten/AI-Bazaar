"""
Fig C1: Price Cascade — The Crash

Top panel: per-firm price trajectories over time. Vertical dashed lines mark the
timestep when each firm goes bankrupt (in_business flips to False). Multiple runs
are overlaid with low alpha to show cross-seed consistency.
Bottom panel: count of active firms over time.

Usage:
    python crash_price_cascade.py --run-dirs logs/crash_seed42 logs/crash_seed1 logs/crash_seed7
    python crash_price_cascade.py --run-dirs logs/crash_seed42 --output paper/fig/crash_price_cascade.pdf
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

# Allow import from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FIRM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f"]


def load_run(run_dir):
    """Return sorted list of state dicts for a run directory."""
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


def bankruptcy_timesteps(db):
    """Return dict {firm_name: timestep} of when each firm first goes bankrupt."""
    bankruptcies = {}
    prev_status = {}
    for s in db.states:
        t = s["timestep"]
        for f in s.get("firms", []):
            name = f.get("name")
            active = f.get("in_business", True)
            if name and prev_status.get(name, True) and not active:
                bankruptcies[name] = t
            if name:
                prev_status[name] = active
    return bankruptcies


def main():
    parser = argparse.ArgumentParser(description="Fig C1: Crash price cascade")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "crash_price_cascade.pdf"))
    parser.add_argument("--good", default=None, help="Good name for price axis (auto-detected if omitted)")
    args = parser.parse_args()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
    alpha = max(0.25, 1.0 / len(args.run_dirs))

    for run_dir in args.run_dirs:
        files = load_run(run_dir)
        if not files:
            print(f"Warning: no state files in {run_dir}", file=sys.stderr)
            continue
        db = DataFrameBuilder(states=files)

        # Auto-detect good name
        good = args.good
        if good is None:
            goods = db._all_good_names()
            good = goods[0] if goods else "food"

        price_df = db.price_per_firm_over_time(good)
        active_df = db.firms_in_business_over_time()
        bankrupt = bankruptcy_timesteps(db)
        all_firms = sorted(price_df["firm"].unique())

        for i, firm in enumerate(all_firms):
            color = FIRM_COLORS[i % len(FIRM_COLORS)]
            firm_prices = price_df[price_df["firm"] == firm].sort_values("timestep")
            ax_top.plot(firm_prices["timestep"], firm_prices["value"],
                        color=color, alpha=alpha, label=firm if len(args.run_dirs) == 1 else None)
            if firm in bankrupt:
                ax_top.axvline(bankrupt[firm], color=color, linestyle=":", alpha=alpha * 1.5, linewidth=1)

        ax_bot.plot(active_df["timestep"], active_df["value"],
                    color="black", alpha=alpha, linewidth=1.5)

    ax_top.set_ylabel(f"Price ($\\$$/unit)", fontsize=12)
    ax_top.set_title("The Crash: Price Cascade", fontsize=13)
    if len(args.run_dirs) == 1:
        ax_top.legend(loc="upper right", ncol=2)

    ax_bot.set_ylabel("Active firms", fontsize=11)
    ax_bot.set_xlabel("Timestep $t$", fontsize=12)
    ax_bot.yaxis.get_major_locator().set_params(integer=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
