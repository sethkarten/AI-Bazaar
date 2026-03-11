"""
Fig 2: Experiment 1 Interaction Line Plots

Three panels: Bankruptcy rate | Price std | Final price / unit_cost ratio
x-axis: dlc ∈ {1, 3, 5}
Lines: n_stab ∈ {1, 2, 4} (n_stab=0 omitted — only 1 seed at dlc=3)
Faint jittered scatter for raw seeds; error bars show min/max.

Usage:
    python exp1_interaction.py [--logs-dir logs/] [--good food] [--output ...]
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

DLC_VALUES = [1, 3, 5]
N_STAB_VALUES = [1, 2, 4]
SEEDS = [8, 16, 64]

# Blues palette: light to dark
COLORS = [plt.cm.Blues(v) for v in [0.4, 0.6, 0.85]]
JITTER = 0.05


def resolve_run_dir(logs_dir, dlc, n_stab, seed):
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


def compute_metrics(run_dir, good):
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    states = db.states

    first_firms = len(states[0].get("firms", []))
    if first_firms == 0:
        return None

    # Bankruptcy rate
    firms_df = db.firms_in_business_over_time().sort_values("timestep")
    if firms_df.empty:
        return None
    last_active = int(firms_df.iloc[-1]["value"])
    bankruptcy_rate = 1.0 - last_active / first_firms

    # Final avg price
    last_state = states[-1]
    prices_at_last = []
    for f in last_state.get("firms", []):
        if f.get("in_business", False):
            prices = f.get("prices") or {}
            p = prices.get(good)
            if isinstance(p, (int, float)) and p > 0:
                prices_at_last.append(p)
    final_avg_price = float(np.mean(prices_at_last)) if prices_at_last else 0.0

    # Price std
    price_df = db.price_per_firm_over_time(good)
    if not price_df.empty:
        per_ts_mean = price_df[price_df["value"] > 0].groupby("timestep")["value"].mean()
        price_std = float(per_ts_mean.std()) if len(per_ts_mean) > 1 else 0.0
    else:
        price_std = 0.0

    return {
        "bankruptcy_rate": bankruptcy_rate,
        "final_avg_price": final_avg_price,
        "price_std": price_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Fig 2: Exp1 Interaction plots")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "exp1", "exp1_interaction.pdf"),
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir
    good = args.good
    output = args.output

    # Collect per-seed unit costs for ratio computation
    all_unit_costs = []
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed)
                if run_dir:
                    all_unit_costs.append(get_unit_cost(run_dir))
    unit_cost = float(np.mean(all_unit_costs)) if all_unit_costs else 1.0
    print(f"Unit cost (global mean): {unit_cost:.3f}")

    # data[n_stab][dlc] = list of per-seed metric dicts
    data = {ns: {dlc: [] for dlc in DLC_VALUES} for ns in N_STAB_VALUES}

    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed)
                if run_dir is None:
                    continue
                m = compute_metrics(run_dir, good)
                if m is not None:
                    data[n_stab][dlc].append(m)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_br, ax_pstd, ax_ratio = axes

    panel_configs = [
        (ax_br,    "bankruptcy_rate",  "Bankruptcy Rate",          lambda v, uc: v),
        (ax_pstd,  "price_std",        "Price Volatility σ",       lambda v, uc: v),
        (ax_ratio, "final_avg_price",  "Final Price / Unit Cost",  lambda v, uc: v / uc if uc > 0 else v),
    ]

    rng = np.random.default_rng(42)
    x_positions = np.array(range(len(DLC_VALUES)), dtype=float)

    for ax, metric_key, title, transform in panel_configs:
        for idx, n_stab in enumerate(N_STAB_VALUES):
            color = COLORS[idx]
            means, lo_errs, hi_errs = [], [], []
            has_data = False

            for j, dlc in enumerate(DLC_VALUES):
                seed_vals = [transform(m[metric_key], unit_cost) for m in data[n_stab][dlc]]
                if not seed_vals:
                    means.append(np.nan)
                    lo_errs.append(0)
                    hi_errs.append(0)
                    continue
                has_data = True
                mean_v = float(np.mean(seed_vals))
                lo_v = float(np.min(seed_vals))
                hi_v = float(np.max(seed_vals))
                means.append(mean_v)
                lo_errs.append(max(0.0, mean_v - lo_v))
                hi_errs.append(max(0.0, hi_v - mean_v))

                # Raw seed scatter with jitter
                jitter = rng.uniform(-JITTER, JITTER, size=len(seed_vals))
                ax.scatter(
                    [x_positions[j] + jitter_i for jitter_i in jitter],
                    seed_vals,
                    color=color, alpha=0.3, s=25, zorder=3,
                )

            valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
            if len(valid) < 2:
                continue

            x_vals = [x_positions[i] for i, _ in valid]
            y_vals = [m for _, m in valid]
            valid_lo = [lo_errs[i] for i, _ in valid]
            valid_hi = [hi_errs[i] for i, _ in valid]

            ax.errorbar(
                x_vals, y_vals,
                yerr=[valid_lo, valid_hi],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=6,
                capsize=4,
                label=f"n_stab={n_stab}",
                zorder=5,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"dlc={d}" for d in DLC_VALUES])
        ax.set_xlabel("Discovery limit (consumers)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc="best", fontsize=9)

    ax_br.set_ylabel("Bankruptcy rate", fontsize=11)
    ax_pstd.set_ylabel("Price std σ", fontsize=11)
    ax_ratio.set_ylabel("Final price / unit cost", fontsize=11)

    # Reference line at ratio=1 for final price panel
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Break-even")
    ax_ratio.legend(loc="best", fontsize=9)

    fig.suptitle("Experiment 1: DLC × Stabilizing Firm Interaction", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
