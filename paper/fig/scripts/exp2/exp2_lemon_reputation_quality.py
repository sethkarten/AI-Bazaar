"""
Fig Exp2-C: Reputation Trajectories — Honest vs Sybil firms, per condition.

Two panels stacked:
  Top:    Mean reputation of honest firms over time, lines per (K, rep_visible).
  Bottom: Mean reputation of sybil firms over time, lines per (K, rep_visible).

Lines coloured by K; solid = rep visible, dashed = rep hidden.
Mean ± 1σ bands across seeds.

Usage:
    python exp2_lemon_reputation_quality.py [--logs-dir logs/] [--good car] [--output ...]
"""

import argparse
import concurrent.futures
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder
from exp2_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data, infer_name_prefix
from exp2_common import (
    SEEDS, K_VALUES, COLORS_K, LS_REP,
    resolve_run_dir, collect_all_run_dirs,
    load_state_files, load_firm_types,
    build_aggregate, serialize_agg, deserialize_agg, plot_band,
)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.5, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5, "grid.color": "gray",
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "text.usetex": False, "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def get_rep_series(run_dir: str):
    """Return {"honest": (ts, vals), "sybil": (ts, vals)} or None."""
    files = load_state_files(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    firm_types = load_firm_types(run_dir)
    rep_df = db.reputation_per_firm_over_time()
    if rep_df.empty:
        return None

    ts_vals = sorted(rep_df["timestep"].unique())
    ts = np.array(ts_vals)

    all_firms    = rep_df["firm"].unique()
    sybil_firms  = [f for f in all_firms if firm_types.get(f, False)]
    honest_firms = [f for f in all_firms if not firm_types.get(f, False)]

    def mean_ts(firms):
        if not firms:
            return None
        sub = rep_df[rep_df["firm"].isin(firms)].groupby("timestep")["value"].mean().sort_index()
        return ts, sub.values

    h = mean_ts(honest_firms)
    s = mean_ts(sybil_firms)
    if h is None and s is None:
        return None
    return {"honest": h, "sybil": s}


# ---------------------------------------------------------------------------
# Aggregation (per firm type)
# ---------------------------------------------------------------------------

def aggregate_rep(results: dict, firm_type: str) -> dict:
    sub = {}
    for (k, rv, seed), data in results.items():
        if data is None or data.get(firm_type) is None:
            sub[(k, rv, seed)] = None
        else:
            sub[(k, rv, seed)] = data[firm_type]
    return build_aggregate(sub)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Exp2 Fig C: Reputation by firm type")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good", default="car")
    ap.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "exp2", "exp2_lemon_reputation_quality.pdf"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name_prefix = infer_name_prefix(args.logs_dir)
    print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_all_run_dirs(args.logs_dir, name_prefix, include_baseline=False)
    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_lemon_reputation_quality", args.good)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        raw = load_cache_data(cache_path)
        honest_agg = deserialize_agg(raw["honest"])
        sybil_agg  = deserialize_agg(raw["sybil"])
    else:
        jobs = []
        for k in K_VALUES:
            for rv in [True, False]:
                for seed in SEEDS:
                    d = resolve_run_dir(args.logs_dir, name_prefix, k, rv, seed)
                    if d:
                        jobs.append((k, rv, seed, d))
                    else:
                        print(f"  Missing: K={k} rep={int(rv)} seed={seed}", flush=True)

        print(f"Loading {len(jobs)} runs ...", flush=True)
        results: dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {ex.submit(get_rep_series, d): (k, rv, seed) for k, rv, seed, d in jobs}
            done, total = 0, len(jobs)
            for future in concurrent.futures.as_completed(future_map):
                k, rv, seed = future_map[future]
                done += 1
                results[(k, rv, seed)] = future.result()
                print(f"  [{done}/{total}] K={k} rep={int(rv)} seed={seed} — "
                      f"{'ok' if results[(k, rv, seed)] else 'empty'}", flush=True)

        honest_agg = aggregate_rep(results, "honest")
        sybil_agg  = aggregate_rep(results, "sybil")
        cache_data = {"honest": serialize_agg(honest_agg), "sybil": serialize_agg(sybil_agg)}
        save_cache(cache_path, cache_data, args.logs_dir, args.good)
        print(f"Cached: {cache_path}", flush=True)
        honest_agg = deserialize_agg(cache_data["honest"])
        sybil_agg  = deserialize_agg(cache_data["sybil"])

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, (ax_h, ax_s) = plt.subplots(2, 1, figsize=(7, 5.5), constrained_layout=True, sharex=True)
    fig.suptitle("Firm Reputation over Time", fontsize=10, fontweight="bold")

    for k in K_VALUES:
        color = COLORS_K[k]
        sat   = k / 12
        for rv in [True, False]:
            rep_tag = "rep" if rv else "no-rep"
            lbl = f"K={k} ({sat:.0%}), {rep_tag}"
            plot_band(ax_h, honest_agg.get((k, rv)), color, lbl, ls=LS_REP[rv])
            plot_band(ax_s, sybil_agg.get((k, rv)),  color, lbl, ls=LS_REP[rv])

    ax_h.set_ylabel("Mean reputation score")
    ax_h.set_ylim(0, 1.05)
    ax_h.set_title("(A) Honest sellers", loc="left")
    ax_h.legend(loc="best", fontsize=7.5, ncol=2)

    ax_s.set_ylabel("Mean reputation score")
    ax_s.set_ylim(0, 1.05)
    ax_s.set_title("(B) Sybil sellers", loc="left")
    ax_s.set_xlabel("Timestep")
    ax_s.legend(loc="best", fontsize=7.5, ncol=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
