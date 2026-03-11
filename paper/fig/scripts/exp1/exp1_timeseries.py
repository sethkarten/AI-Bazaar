"""
Fig 3: Experiment 1 Representative Time-Series

2×2 panels for four representative (dlc, n_stab) cells:
  top-left     Worst-case:      dlc=5, n_stab=1
  top-right    Discovery only:  dlc=1, n_stab=1
  bottom-left  Stabilizers:     dlc=5, n_stab=4
  bottom-right Best-case:       dlc=1, n_stab=4

Per panel:
  - Faint thin lines per seed (alpha=0.35)
  - Thick mean line
  - Dashed unit-cost reference line
  - Light red axhspan below unit cost

Usage:
    python exp1_timeseries.py [--logs-dir logs/] [--good food] [--output ...]
      [--cells "5,1 1,1 5,4 1,4"]
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
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

SEEDS = [8, 16, 64]

DEFAULT_CELLS = [
    (5, 1, "Worst-case"),
    (1, 1, "Discovery only"),
    (5, 4, "Stabilizers only"),
    (1, 4, "Best-case"),
]


def resolve_run_dir(logs_dir, dlc, n_stab, seed):
    if n_stab == 0 and dlc == 3 and seed == 8:
        path = os.path.join(logs_dir, "exp1_baseline")
    elif n_stab == 0:
        return None
    else:
        path = os.path.join(logs_dir, f"exp1_stab_{n_stab}_dlc{dlc}_seed{seed}")
    return path if os.path.isdir(path) else None


def load_states(run_dir):
    files = glob.glob(os.path.join(run_dir, "state_t*.json"))
    files.sort(key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or "0"))
    return files


def get_unit_cost(run_dir):
    attr_path = os.path.join(run_dir, "firm_attributes.json")
    if not os.path.isfile(attr_path):
        return 1.0
    try:
        with open(attr_path) as f:
            attrs = json.load(f)
        costs = []
        for firm in attrs:
            uc = firm.get("supply_unit_costs", {})
            costs.extend(v for v in uc.values() if isinstance(v, (int, float)))
        return float(np.mean(costs)) if costs else 1.0
    except Exception:
        return 1.0


def get_timeseries(run_dir, good):
    """Returns dict with arrays: timesteps, avg_price, active_firms."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)

    # Average price per timestep (in-business firms only)
    price_df = db.price_per_firm_over_time(good)
    per_ts_price = (
        price_df[price_df["value"] > 0]
        .groupby("timestep")["value"]
        .mean()
        .reset_index()
        .sort_values("timestep")
    )

    # Active firms per timestep
    firms_df = db.firms_in_business_over_time().sort_values("timestep")

    timesteps = per_ts_price["timestep"].values
    avg_price = per_ts_price["value"].values
    return {
        "timesteps": timesteps,
        "avg_price": avg_price,
    }


def parse_cells(cells_str):
    """Parse "5,1 1,1 5,4 1,4" into [(5,1), (1,1), (5,4), (1,4)]."""
    result = []
    for token in cells_str.strip().split():
        parts = token.split(",")
        if len(parts) == 2:
            result.append((int(parts[0]), int(parts[1]), f"dlc={parts[0]},stab={parts[1]}"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Fig 3: Exp1 Time-series")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "exp1", "exp1_timeseries.pdf"),
    )
    parser.add_argument(
        "--cells", default=None,
        help='Override representative cells as "dlc,nstab" pairs, e.g. "5,1 1,1 5,4 1,4"',
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir
    good = args.good
    output = args.output

    cells = parse_cells(args.cells) if args.cells else DEFAULT_CELLS

    if len(cells) != 4:
        print(f"Warning: expected 4 cells, got {len(cells)}. Padding/truncating.")
        while len(cells) < 4:
            cells.append(cells[-1])
        cells = cells[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes_flat = axes.flatten()

    for ax_idx, (dlc, n_stab, label) in enumerate(cells):
        ax = axes_flat[ax_idx]
        ax_row = ax_idx // 2

        # Gather unit cost from first available seed
        unit_cost = 1.0
        for seed in SEEDS:
            run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed)
            if run_dir:
                unit_cost = get_unit_cost(run_dir)
                break

        all_ts = []
        all_prices = []

        for seed in SEEDS:
            run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed)
            if run_dir is None:
                print(f"  Missing: dlc={dlc}, n_stab={n_stab}, seed={seed}")
                continue
            ts_data = get_timeseries(run_dir, good)
            if ts_data is None:
                continue
            ts = ts_data["timesteps"]
            prices = ts_data["avg_price"]
            all_ts.append(ts)
            all_prices.append(prices)
            # Faint seed line
            ax.plot(ts, prices, color="#2166ac", linewidth=0.8, alpha=0.35, zorder=2)

        # Mean line
        if all_prices:
            min_len = min(len(p) for p in all_prices)
            ts_common = all_ts[0][:min_len]
            price_arr = np.array([p[:min_len] for p in all_prices])
            mean_prices = price_arr.mean(axis=0)
            ax.plot(ts_common, mean_prices, color="#2166ac", linewidth=2.2, zorder=4, label="Mean")

        # Unit cost reference
        ax.axhline(unit_cost, color="#d62728", linestyle="--", linewidth=1.4,
                   alpha=0.8, label=f"Unit cost c={unit_cost:.2f}", zorder=3)

        # Light red shading below unit cost
        y_min = ax.get_ylim()[0] if all_prices else 0
        ax.axhspan(0, unit_cost, color="#d62728", alpha=0.06, zorder=1)

        ax.set_title(f"{label}\n(dlc={dlc}, stab={n_stab})", fontsize=12)
        ax.legend(loc="upper right", fontsize=8)

        if ax_idx % 2 == 0:
            ax.set_ylabel("Avg. market price", fontsize=11)
        ax.set_xlabel("Timestep $t$", fontsize=10)

    fig.suptitle("Experiment 1: Representative Market Trajectories", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
