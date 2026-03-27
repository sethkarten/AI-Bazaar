"""
Fig Exp2-D: Consumer Welfare — average consumer surplus over time per condition.

Single panel: mean consumer surplus per timestep, lines per (K, rep_visible).
Lines coloured by K; solid = rep visible, dashed = rep hidden.
Mean ± 1σ bands across seeds.

Uses lemon_market_avg_consumer_surplus from state files (the per-step average
surplus among buyers who transacted that timestep).

Usage:
    python exp2_lemon_consumer_welfare.py [--logs-dir logs/] [--good car] [--output ...]
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
from exp2_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data, infer_name_prefix
from exp2_common import (
    SEEDS, K_VALUES, COLORS_K, LS_REP,
    resolve_run_dir, collect_all_run_dirs,
    load_state_files, build_aggregate, serialize_agg, deserialize_agg, plot_band,
)
K_ALL = [0] + K_VALUES

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

def get_welfare_series(run_dir: str):
    """Return (ts_array, surplus_array) or None."""
    import json
    files = load_state_files(run_dir)
    if not files:
        return None
    pts = []
    for p in files:
        with open(p) as f:
            s = json.load(f)
        t = s.get("timestep")
        v = s.get("lemon_market_avg_consumer_surplus")
        if t is not None and v is not None:
            pts.append((t, float(v)))
    if not pts:
        return None
    pts.sort()
    return np.array([x[0] for x in pts]), np.array([x[1] for x in pts])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Exp2 Fig D: Consumer Welfare")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good", default="car")
    ap.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "exp2", "exp2_lemon_consumer_welfare.pdf"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name_prefix = infer_name_prefix(args.logs_dir)
    print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_all_run_dirs(args.logs_dir, name_prefix, include_baseline=True)
    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_lemon_consumer_welfare", args.good)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        agg = deserialize_agg(load_cache_data(cache_path)["agg"])
    else:
        jobs = []
        for k in K_ALL:
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
            future_map = {ex.submit(get_welfare_series, d): (k, rv, seed) for k, rv, seed, d in jobs}
            done, total = 0, len(jobs)
            for future in concurrent.futures.as_completed(future_map):
                k, rv, seed = future_map[future]
                done += 1
                results[(k, rv, seed)] = future.result()
                print(f"  [{done}/{total}] K={k} rep={int(rv)} seed={seed} — "
                      f"{'ok' if results[(k, rv, seed)] else 'empty'}", flush=True)

        agg = build_aggregate(results)
        cache_data = {"agg": serialize_agg(agg)}
        save_cache(cache_path, cache_data, args.logs_dir, args.good)
        print(f"Cached: {cache_path}", flush=True)
        agg = deserialize_agg(cache_data["agg"])

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5), constrained_layout=True)
    fig.suptitle("Consumer Surplus over Time", fontsize=10, fontweight="bold")

    for k in K_ALL:
        color = COLORS_K[k]
        sat   = k / 12
        for rv in [True, False]:
            entry = agg.get((k, rv))
            if entry is None:
                continue
            rep_tag = "rep" if rv else "no-rep"
            lbl = f"K={k} ({sat:.0%}), {rep_tag}"
            plot_band(ax, entry, color, lbl, ls=LS_REP[rv])

    ax.axhline(0.0, color="#555555", lw=1.2, ls=":", alpha=0.7, zorder=2, label="Break-even ($=0$)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Avg. consumer surplus (\\$)")
    ax.legend(loc="best", fontsize=7.5)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
