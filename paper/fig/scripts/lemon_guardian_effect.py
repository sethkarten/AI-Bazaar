"""
Fig L3: Skeptical Guardian Intervention — The Lemon Market

3-row panel comparing baseline vs guardian condition:
  (a) Bids / (Bids + Passes) ratio — market trust level
  (b) Active honest firm count over time
  (c) Mean consumer surplus over time

Multiple runs per condition aggregated with mean ± std bands.

Usage:
    python lemon_guardian_effect.py \
        --baseline-dirs logs/lemon_seed42 logs/lemon_seed1 \
        --guardian-dirs logs/lemon_guardian_seed42 logs/lemon_guardian_seed1
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
    "Skeptical guardian": ("#2ca02c", "--"),
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


def extract_market_trust(db):
    """Returns (ts, bids_ratio) where ratio = bids/(bids+passes)."""
    df = db.lemon_market_metrics_over_time()
    if df.empty:
        return np.array([]), np.array([])
    ts_vals = sorted(df["timestep"].unique())
    ratios = []
    for t in ts_vals:
        sub = df[df["timestep"] == t]
        bids = sub[sub["metric"] == "Bids"]["value"].sum()
        passes = sub[sub["metric"] == "Passes"]["value"].sum()
        total = bids + passes
        ratios.append(bids / total if total > 0 else np.nan)
    return np.array(ts_vals), np.array(ratios)


def extract_honest_firms(run_dir, db):
    firm_types = load_firm_types(run_dir)
    df = db.firms_in_business_over_time()
    # If no type info, count all active firms
    if not firm_types:
        return df["timestep"].values, df["value"].values.astype(float)
    # Count only honest firms per timestep
    rows = []
    for s in db.states:
        t = s["timestep"]
        honest_active = sum(
            1 for f in s.get("firms", [])
            if f.get("in_business", False) and not firm_types.get(f.get("name", ""), False)
        )
        rows.append(honest_active)
    return df["timestep"].values, np.array(rows, dtype=float)


def extract_consumer_surplus(db):
    cs = db.consumer_surplus_per_consumer_over_time()
    agg = cs.groupby("timestep")["value"].mean().reset_index().sort_values("timestep")
    return agg["timestep"].values, agg["value"].values


def aggregate_series(run_dirs, extract_fn):
    all_ts, all_vals = None, []
    for run_dir in run_dirs:
        files = load_run(run_dir)
        if not files:
            continue
        db = DataFrameBuilder(states=files)
        ts, vals = extract_fn(run_dir, db)
        if all_ts is None:
            all_ts = ts
        all_vals.append(vals)
    if not all_vals:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(v) for v in all_vals)
    arr = np.array([v[:min_len] for v in all_vals])
    return all_ts[:min_len], arr.mean(0), arr.std(0)


def main():
    parser = argparse.ArgumentParser(description="Fig L3: Lemon guardian effect")
    parser.add_argument("--baseline-dirs", nargs="+", default=[])
    parser.add_argument("--guardian-dirs", nargs="+", default=[])
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "lemon_guardian_effect.pdf"))
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax_trust, ax_firms, ax_surplus = axes

    condition_dirs = []
    if args.baseline_dirs:
        condition_dirs.append(("Baseline", args.baseline_dirs))
    if args.guardian_dirs:
        condition_dirs.append(("Skeptical guardian", args.guardian_dirs))

    for label, dirs in condition_dirs:
        color, ls = CONDITIONS.get(label, ("#7f7f7f", "-"))
        n = len(dirs)

        # (a) Market trust ratio
        ts, mean, std = aggregate_series(dirs, lambda d, db: extract_market_trust(db))
        if len(ts):
            ax_trust.plot(ts, mean, color=color, linestyle=ls, label=f"{label} (n={n})")
            ax_trust.fill_between(ts, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                                  color=color, alpha=0.15)

        # (b) Honest firm count
        ts, mean, std = aggregate_series(dirs, extract_honest_firms)
        if len(ts):
            ax_firms.plot(ts, mean, color=color, linestyle=ls)
            ax_firms.fill_between(ts, np.clip(mean - std, 0, None), mean + std,
                                  color=color, alpha=0.15)

        # (c) Consumer surplus
        ts, mean, std = aggregate_series(dirs, lambda d, db: extract_consumer_surplus(db))
        if len(ts):
            ax_surplus.plot(ts, mean, color=color, linestyle=ls)
            ax_surplus.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)

    ax_trust.set_ylabel("Market trust\n(Bids / (Bids+Passes))", fontsize=10)
    ax_trust.set_ylim(0, 1.05)
    ax_trust.set_title("The Lemon Market: Guardian Intervention Effect", fontsize=12)
    ax_trust.legend(loc="upper right")

    ax_firms.set_ylabel("Active honest firms", fontsize=10)
    ax_firms.yaxis.get_major_locator().set_params(integer=True)

    ax_surplus.set_ylabel("Mean consumer surplus", fontsize=10)
    ax_surplus.set_xlabel("Timestep $t$", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
