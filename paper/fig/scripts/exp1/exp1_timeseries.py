"""
Fig 3: Experiment 1 Representative Time-Series

2×2 panels for four representative (dlc, n_stab) cells:
  top-left     Worst-case:      dlc=5, n_stab=1
  top-right    Discovery only:  dlc=1, n_stab=1
  bottom-left  Stabilizers:     dlc=5, n_stab=4
  bottom-right Best-case:       dlc=1, n_stab=4

Per panel:
  - Faint thin lines per seed (alpha=0.3, lw=0.8)
  - Thick mean line (lw=1.8) with ±1σ shaded band
  - Dashed unit-cost reference line
  - Light shading below unit cost

Usage:
    python exp1_timeseries.py [--logs-dir logs/] [--good food] [--output ...]
      [--cells "5,1 1,1 5,4 1,4"]
"""

import argparse
import concurrent.futures
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
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.5,
    "lines.markersize":   5,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "grid.color":         "gray",
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":        False,
})

# Okabe-Ito: blue for price series, vermillion for unit-cost reference
COLOR_PRICE    = "#0072B2"  # Okabe Blue
COLOR_COST_REF = "#D55E00"  # Okabe Vermillion

SEEDS = [8, 16, 64]

DEFAULT_CELLS = [
    (5, 1, "Worst-case"),
    (1, 1, "Discovery only"),
    (5, 4, "Stabilizers only"),
    (1, 4, "Best-case"),
]


def resolve_run_dir(logs_dir, dlc, n_stab, seed):
    if n_stab == 0:
        # Baseline (no stabilizing firm): only exists for dlc=3, seed=8
        if dlc == 3 and seed == 8:
            path = os.path.join(logs_dir, "exp1_baseline")
            return path if os.path.isdir(path) else None
        return None
    if n_stab == 5:
        path = os.path.join(logs_dir, f"exp1_stab_5_dlc{dlc}_seed{seed}")
        return path if os.path.isdir(path) else None
    path = os.path.join(logs_dir, f"exp1_stab_{n_stab}_dlc{dlc}_seed{seed}")
    return path if os.path.isdir(path) else None


def load_states(run_dir):
    files = glob.glob(os.path.join(run_dir, "state_t*.json"))
    files.sort(key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or "0"))
    valid = []
    for p in files:
        if os.path.getsize(p) == 0:
            continue
        try:
            with open(p) as f:
                json.load(f)
            valid.append(p)
        except (json.JSONDecodeError, OSError):
            pass
    return valid


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
    """Returns dict with arrays: timesteps, avg_price."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)

    price_df = db.price_per_firm_over_time(good)
    per_ts_price = (
        price_df[price_df["value"] > 0]
        .groupby("timestep")["value"]
        .mean()
        .reset_index()
        .sort_values("timestep")
    )
    if per_ts_price.empty:
        return None
    return {
        "timesteps": per_ts_price["timestep"].values,
        "avg_price": per_ts_price["value"].values,
    }


def parse_cells(cells_str):
    """Parse "5,1 1,1 5,4 1,4" into [(dlc, n_stab, label), ...]."""
    result = []
    for token in cells_str.strip().split():
        parts = token.split(",")
        if len(parts) == 2:
            result.append((int(parts[0]), int(parts[1]), f"dlc={parts[0]}, $k$={parts[1]}"))
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
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cells = parse_cells(args.cells) if args.cells else DEFAULT_CELLS
    if len(cells) != 4:
        print(f"Warning: expected 4 cells, got {len(cells)}. Padding/truncating.")
        while len(cells) < 4:
            cells.append(cells[-1])
        cells = cells[:4]

    # Load all timeseries in parallel
    jobs = [
        (ax_idx, dlc, n_stab, label, seed)
        for ax_idx, (dlc, n_stab, label) in enumerate(cells)
        for seed in SEEDS
        if resolve_run_dir(args.logs_dir, dlc, n_stab, seed) is not None
    ]
    total = len(jobs)
    print(f"Loading {total} runs...", flush=True)

    ts_results = {}  # (ax_idx, seed) -> ts_data or None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_job = {
            ex.submit(get_timeseries, resolve_run_dir(args.logs_dir, dlc, n_stab, seed), args.good):
                (ax_idx, dlc, n_stab, label, seed)
            for ax_idx, dlc, n_stab, label, seed in jobs
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_job):
            ax_idx, dlc, n_stab, label, seed = future_to_job[future]
            done += 1
            ts_data = future.result()
            print(f"  [{done}/{total}] {label} seed={seed} — {'ok' if ts_data else 'empty'}", flush=True)
            if ts_data:
                ts_results[(ax_idx, seed)] = ts_data

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax_idx, (dlc, n_stab, label) in enumerate(cells):
        ax = axes_flat[ax_idx]

        # Unit cost from first available seed
        unit_cost = 1.0
        for seed in SEEDS:
            run_dir = resolve_run_dir(args.logs_dir, dlc, n_stab, seed)
            if run_dir:
                unit_cost = get_unit_cost(run_dir)
                break

        all_ts     = []
        all_prices = []

        for seed in SEEDS:
            if (ax_idx, seed) not in ts_results:
                if resolve_run_dir(args.logs_dir, dlc, n_stab, seed) is None:
                    print(f"  Missing: dlc={dlc}, n_stab={n_stab}, seed={seed}")
                continue
            ts_data = ts_results[(ax_idx, seed)]
            ts     = ts_data["timesteps"]
            prices = ts_data["avg_price"]
            all_ts.append(ts)
            all_prices.append(prices)
            ax.plot(ts, prices, color=COLOR_PRICE, linewidth=0.8, alpha=0.3, zorder=2)

        # Mean line + ±1σ band
        if all_prices:
            min_len   = min(len(p) for p in all_prices)
            ts_common = all_ts[0][:min_len]
            price_arr = np.array([p[:min_len] for p in all_prices])
            mean_p    = price_arr.mean(axis=0)
            std_p     = price_arr.std(axis=0)

            ax.fill_between(
                ts_common, mean_p - std_p, mean_p + std_p,
                color=COLOR_PRICE, alpha=0.15, zorder=3, label="±1σ",
            )
            ax.plot(ts_common, mean_p, color=COLOR_PRICE, linewidth=1.8, zorder=4, label="Mean")

        # Unit cost reference line + shading
        ax.axhline(unit_cost, color=COLOR_COST_REF, linestyle="--", linewidth=1.2,
                   alpha=0.8, label=f"Unit cost $c$={unit_cost:.2f}", zorder=5)
        ax.axhspan(0, unit_cost, color=COLOR_COST_REF, alpha=0.06, zorder=1)

        ax.set_title(f"{label}")
        ax.legend(loc="upper right")
        ax.set_xlabel("Timestep $t$")
        if ax_idx % 2 == 0:
            ax.set_ylabel("Avg. market price")

    fig.suptitle("Experiment 1: Representative Market Trajectories", fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
