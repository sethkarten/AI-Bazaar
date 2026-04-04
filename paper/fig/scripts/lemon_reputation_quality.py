"""
Fig L2: Reputation Trajectories — Honest vs Sybil Firms

Top panel: reputation over time, colored by firm type (honest=blue, sybil=red).
Bottom panel: sales-weighted average price per firm type (proxy for quality signaling
premium; actual quality signal vs delivered quality fields not always in state).

Firm types are read from firm_attributes.json in the run directory (sybil field).
Falls back to treating all firms as honest if the file is absent.

Usage:
    python lemon_reputation_quality.py --run-dirs logs/lemon_seed42 logs/lemon_seed1
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
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLOR_HONEST = "#2ca02c"
COLOR_SYBIL  = "#d62728"


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


def load_firm_types(run_dir):
    """Returns dict {firm_name: is_sybil (bool)} from firm_attributes.json."""
    attr_path = os.path.join(run_dir, "firm_attributes.json")
    if not os.path.exists(attr_path):
        return {}
    with open(attr_path) as f:
        attrs = json.load(f)
    if isinstance(attrs, list):
        return {a.get("name", f"firm_{i}"): bool(a.get("sybil", False))
                for i, a in enumerate(attrs)}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Fig L2: Lemon market reputation by firm type")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "lemon_reputation_quality.pdf"))
    args = parser.parse_args()

    honest_rep_series, sybil_rep_series = [], []
    honest_price_series, sybil_price_series = [], []
    all_ts = None

    for run_dir in args.run_dirs:
        files = load_run(run_dir)
        if not files:
            continue
        db = DataFrameBuilder(states=files)
        firm_types = load_firm_types(run_dir)

        rep_df = db.reputation_per_firm_over_time()
        ts_vals = sorted(rep_df["timestep"].unique())
        if all_ts is None:
            all_ts = np.array(ts_vals)

        all_firms = rep_df["firm"].unique()
        sybil_firms  = [f for f in all_firms if firm_types.get(f, False)]
        honest_firms = [f for f in all_firms if not firm_types.get(f, False)]

        def mean_series(firms, df, col="value"):
            if not firms:
                return None
            sub = df[df["firm"].isin(firms)].groupby("timestep")[col].mean().sort_index()
            return sub.values

        h_rep = mean_series(honest_firms, rep_df)
        s_rep = mean_series(sybil_firms, rep_df)
        if h_rep is not None:
            honest_rep_series.append(h_rep)
        if s_rep is not None:
            sybil_rep_series.append(s_rep)

        # Price as proxy for quality signaling premium
        goods = db._all_good_names()
        if goods:
            price_df = db.price_per_firm_over_time(goods[0])
            h_price = mean_series(honest_firms, price_df)
            s_price = mean_series(sybil_firms, price_df)
            if h_price is not None:
                honest_price_series.append(h_price)
            if s_price is not None:
                sybil_price_series.append(s_price)

    if not honest_rep_series and not sybil_rep_series:
        print("No reputation data found.", file=sys.stderr)
        sys.exit(1)

    def agg(series_list, ts):
        if not series_list:
            return None, None, None
        min_len = min(len(s) for s in series_list)
        arr = np.array([s[:min_len] for s in series_list])
        return ts[:min_len], arr.mean(0), arr.std(0)

    fig, (ax_rep, ax_price) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for label, series_list, color in [
        ("Honest firms", honest_rep_series, COLOR_HONEST),
        ("Sybil firms",  sybil_rep_series,  COLOR_SYBIL),
    ]:
        ts, mean, std = agg(series_list, all_ts)
        if ts is None:
            continue
        ax_rep.plot(ts, mean, color=color, label=label)
        ax_rep.fill_between(ts, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=color, alpha=0.15)

    ax_rep.set_ylabel("Reputation", fontsize=12)
    ax_rep.set_ylim(0, 1.05)
    ax_rep.set_title("The Lemon Market: Reputation by Firm Type", fontsize=13)
    ax_rep.legend(loc="upper right")

    for label, series_list, color in [
        ("Honest firms", honest_price_series, COLOR_HONEST),
        ("Sybil firms",  sybil_price_series,  COLOR_SYBIL),
    ]:
        ts, mean, std = agg(series_list, all_ts)
        if ts is None:
            continue
        ax_price.plot(ts, mean, color=color, label=label)
        ax_price.fill_between(ts, np.clip(mean - std, 0, None), mean + std,
                              color=color, alpha=0.15)

    ax_price.set_ylabel("Avg. price ($\\$$)", fontsize=12)
    ax_price.set_xlabel("Timestep $t$", fontsize=12)
    ax_price.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
