"""
Fig L1: Market Volume Decay — The Lemon Market (Akerlof Effect)

Three lines over time: Listings, Bids, Passes.
Annotates the crossover point where Passes > Bids (market distrust threshold).
Multiple runs are aggregated with mean ± std bands.

Empirically demonstrates the Akerlof (1970) prediction: quality uncertainty
causes market volume collapse, a phenomenon invisible in single-agent benchmarks.

Usage:
    python lemon_market_freeze.py --run-dirs logs/lemon_seed42 logs/lemon_seed1
"""

import argparse
import glob
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

METRIC_STYLE = {
    "Listings": ("#7f7f7f", "-"),
    "Bids":     ("#2ca02c", "-"),
    "Passes":   ("#d62728", "--"),
}


def load_run(run_dir):
    files = glob.glob(os.path.join(run_dir, "state_t*.json"))
    files.sort(key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or "0"))
    return files


def main():
    parser = argparse.ArgumentParser(description="Fig L1: Lemon market freeze")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "lemon_market_freeze.pdf"))
    args = parser.parse_args()

    series_by_metric = {m: [] for m in METRIC_STYLE}
    all_ts = None

    for run_dir in args.run_dirs:
        files = load_run(run_dir)
        if not files:
            print(f"Warning: no state files in {run_dir}", file=sys.stderr)
            continue
        db = DataFrameBuilder(state_files=files)
        df = db.lemon_market_metrics_over_time()
        if df.empty:
            print(f"Warning: no lemon market data in {run_dir}", file=sys.stderr)
            continue
        ts = sorted(df["timestep"].unique())
        if all_ts is None:
            all_ts = np.array(ts)
        for metric in METRIC_STYLE:
            sub = df[df["metric"] == metric].sort_values("timestep")
            series_by_metric[metric].append(sub["value"].values)

    if all_ts is None or not any(series_by_metric.values()):
        print("No lemon market data found.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    crossover_ts = None
    plot_data = {}
    for metric, series_list in series_by_metric.items():
        if not series_list:
            continue
        min_len = min(len(s) for s in series_list)
        arr = np.array([s[:min_len] for s in series_list])
        mean, std = arr.mean(0), arr.std(0)
        plot_data[metric] = (all_ts[:min_len], mean, std)

    # Find bids/passes crossover (where passes first exceed bids)
    if "Bids" in plot_data and "Passes" in plot_data:
        ts_b, bids_mean, _ = plot_data["Bids"]
        ts_p, passes_mean, _ = plot_data["Passes"]
        min_len = min(len(ts_b), len(ts_p))
        diff = passes_mean[:min_len] - bids_mean[:min_len]
        cross_idx = np.argmax(diff > 0) if np.any(diff > 0) else None
        if cross_idx is not None and cross_idx > 0:
            crossover_ts = ts_b[cross_idx]

    for metric, (ts, mean, std) in plot_data.items():
        color, ls = METRIC_STYLE[metric]
        ax.plot(ts, mean, color=color, linestyle=ls, label=f"{metric} (n={len(args.run_dirs)})")
        ax.fill_between(ts, np.clip(mean - std, 0, None), mean + std, color=color, alpha=0.12)

    if crossover_ts is not None:
        ax.axvline(crossover_ts, color="#ff7f0e", linestyle=":", linewidth=1.5)
        ax.text(crossover_ts + 0.3, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
                "Passes > Bids", fontsize=9, color="#ff7f0e", va="top")

    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel("Count per timestep", fontsize=12)
    ax.set_title("The Lemon Market: Market Volume Decay (Akerlof Effect)", fontsize=13)
    ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
