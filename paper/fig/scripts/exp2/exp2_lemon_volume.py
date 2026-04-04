"""
Fig Exp2-B: Market Volume — Listings, Bids, Passes over time.

Two panels side-by-side:
  Left:  reputation visible (rep1) — bid count and pass count per condition
  Right: reputation hidden  (rep0) — same

Lines coloured by K; bids = solid, passes = dashed within each condition.
Mean ± 1σ bands across seeds.

Usage:
    python exp2_lemon_volume.py [--logs-dir logs/] [--good car] [--output ...]
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

def get_volume_series(run_dir: str):
    """Return {"bids": (ts, vals), "passes": (ts, vals)} or None."""
    files = load_state_files(run_dir)
    if not files:
        return None
    import json
    bids_pts, passes_pts = [], []
    for s in files:
        t = s.get("timestep")
        b = s.get("lemon_market_bids_count")
        pa = s.get("lemon_market_passes_count")
        if t is not None and b is not None:
            bids_pts.append((t, float(b)))
        if t is not None and pa is not None:
            passes_pts.append((t, float(pa)))
    if not bids_pts:
        return None
    bids_pts.sort();  passes_pts.sort()
    import numpy as np
    return {
        "bids":   (np.array([x[0] for x in bids_pts]),   np.array([x[1] for x in bids_pts])),
        "passes": (np.array([x[0] for x in passes_pts]), np.array([x[1] for x in passes_pts])),
    }


# ---------------------------------------------------------------------------
# Aggregation (wraps build_aggregate per metric)
# ---------------------------------------------------------------------------

def aggregate_volume(results: dict, metric: str) -> dict:
    """Extract one metric from volume results and aggregate."""
    sub = {}
    for (k, rv, seed), data in results.items():
        sub[(k, rv, seed)] = data[metric] if data is not None and metric in data else None
    return build_aggregate(sub)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Exp2 Fig B: Market Volume")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good", default="car")
    ap.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "exp2", "exp2_lemon_volume.pdf"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name_prefix = infer_name_prefix(args.logs_dir)
    print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_all_run_dirs(args.logs_dir, name_prefix, include_baseline=True)
    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_lemon_volume", args.good)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        raw = load_cache_data(cache_path)
        bids_agg   = deserialize_agg(raw["bids"])
        passes_agg = deserialize_agg(raw["passes"])
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
            future_map = {ex.submit(get_volume_series, d): (k, rv, seed) for k, rv, seed, d in jobs}
            done, total = 0, len(jobs)
            for future in concurrent.futures.as_completed(future_map):
                k, rv, seed = future_map[future]
                done += 1
                results[(k, rv, seed)] = future.result()
                print(f"  [{done}/{total}] K={k} rep={int(rv)} seed={seed} — "
                      f"{'ok' if results[(k, rv, seed)] else 'empty'}", flush=True)

        bids_agg   = aggregate_volume(results, "bids")
        passes_agg = aggregate_volume(results, "passes")
        cache_data = {"bids": serialize_agg(bids_agg), "passes": serialize_agg(passes_agg)}
        save_cache(cache_path, cache_data, args.logs_dir, args.good)
        print(f"Cached: {cache_path}", flush=True)
        bids_agg   = deserialize_agg(cache_data["bids"])
        passes_agg = deserialize_agg(cache_data["passes"])

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), constrained_layout=True, sharey=True)
    fig.suptitle("Market Activity over Time", fontsize=10, fontweight="bold")

    for rep_visible, ax, title in [(True, axes[0], "(A) Reputation visible"), (False, axes[1], "(B) Reputation hidden")]:
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Timestep")
        for k in K_ALL:
            color = COLORS_K[k]
            sat   = k / 12
            b_entry = bids_agg.get((k, rep_visible))
            p_entry = passes_agg.get((k, rep_visible))
            if b_entry is not None:
                plot_band(ax, b_entry, color, f"K={k} ({sat:.0%}) bids", ls="-")
            if p_entry is not None:
                plot_band(ax, p_entry, color, f"K={k} ({sat:.0%}) passes", ls="--", alpha_band=0.08)
        ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=7.5, ncol=2)

    axes[0].set_ylabel("Count per timestep\n(— bids, - - passes)")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
