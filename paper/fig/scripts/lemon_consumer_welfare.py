"""
Fig L4: Consumer Welfare Harm — The Lemon Market

Stacked area chart of mean consumer utility components over time:
  - Goods utility (positive)
  - Cash utility (positive)
  - Labor disutility (negative, shown as absolute value)

Vertical tick marks along the x-axis show timesteps where Sybil firms
recorded sales (proxy for "consumer bought a lemon").

Usage:
    python lemon_consumer_welfare.py --run-dirs logs/lemon_seed42 logs/lemon_seed1
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

COMPONENT_COLORS = {
    "Goods utility (avg)":   "#2ca02c",
    "Cash utility (avg)":    "#1f77b4",
    "Labor disutility (avg)": "#d62728",
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


def load_firm_types(run_dir):
    attr_path = os.path.join(run_dir, "firm_attributes.json")
    if not os.path.exists(attr_path):
        return {}
    with open(attr_path) as f:
        attrs = json.load(f)
    if isinstance(attrs, list):
        return {a.get("name", f"firm_{i}"): bool(a.get("sybil", False))
                for i, a in enumerate(attrs)}
    return {}


def sybil_sale_timesteps(run_dir, db):
    """Return set of timesteps where any sybil firm made at least one sale."""
    firm_types = load_firm_types(run_dir)
    sybil_firms = {name for name, is_sybil in firm_types.items() if is_sybil}
    if not sybil_firms:
        return set()
    sale_ts = set()
    for s in db.states:
        t = s["timestep"]
        for f in s.get("firms", []):
            if f.get("name") in sybil_firms:
                sales = f.get("sales_this_step") or {}
                if any(v > 0 for v in sales.values() if isinstance(v, (int, float))):
                    sale_ts.add(t)
    return sale_ts


def main():
    parser = argparse.ArgumentParser(description="Fig L4: Lemon market consumer welfare")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "lemon_consumer_welfare.pdf"))
    args = parser.parse_args()

    component_series = {k: [] for k in COMPONENT_COLORS}
    all_ts = None
    all_sybil_ts = set()

    for run_dir in args.run_dirs:
        files = load_run(run_dir)
        if not files:
            continue
        db = DataFrameBuilder(states=files)
        util_df = db.consumer_utility_components_over_time()
        ts_vals = sorted(util_df["timestep"].unique())
        if all_ts is None:
            all_ts = np.array(ts_vals)

        for component in COMPONENT_COLORS:
            sub = util_df[util_df["metric"] == component].sort_values("timestep")
            component_series[component].append(sub["value"].values)

        all_sybil_ts |= sybil_sale_timesteps(run_dir, db)

    if all_ts is None:
        print("No utility data found.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 5))

    goods_mean = labor_mean = cash_mean = None
    for component, series_list in component_series.items():
        if not series_list:
            continue
        min_len = min(len(s) for s in series_list)
        arr = np.array([s[:min_len] for s in series_list])
        mean = arr.mean(0)
        ts = all_ts[:min_len]
        if component == "Goods utility (avg)":
            goods_mean = (ts, mean)
        elif component == "Cash utility (avg)":
            cash_mean = (ts, mean)
        elif component == "Labor disutility (avg)":
            labor_mean = (ts, np.abs(mean))

    # Stacked area: goods + cash (positive), labor (negative offset)
    if goods_mean and cash_mean:
        ts = goods_mean[0]
        g = goods_mean[1]
        c = cash_mean[1][:len(g)]
        ax.stackplot(ts, g, c,
                     labels=["Goods utility", "Cash utility"],
                     colors=[COMPONENT_COLORS["Goods utility (avg)"],
                             COMPONENT_COLORS["Cash utility (avg)"]],
                     alpha=0.6)
    if labor_mean:
        ts_l, l = labor_mean
        ax.plot(ts_l, -l, color=COMPONENT_COLORS["Labor disutility (avg)"],
                linestyle="--", linewidth=1.5, label="Labor disutility (neg.)")

    # Mark sybil sale timesteps as ticks along x-axis
    y_tick = ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else -0.1
    sybil_ts_sorted = sorted(all_sybil_ts)
    if sybil_ts_sorted:
        ax.vlines(sybil_ts_sorted, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[0] * 0.85,
                  color="#ff7f0e", linewidth=0.8, alpha=0.6, label="Sybil sale event")

    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel("Mean consumer utility component", fontsize=12)
    ax.set_title("The Lemon Market: Consumer Welfare Harm", fontsize=13)
    ax.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
