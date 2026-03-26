"""
Fig Exp2-F: Market Distrust — buyer pass rate on ALL listings over time.

Distrust ratio = passes / (bids + passes). When this approaches 1, buyers
are refusing to engage with the market (Akerlof freeze). Higher K pushes
distrust higher; hiding reputation from buyers should amplify this.

Two panels side-by-side:
  Left:  reputation visible (rep1)
  Right: reputation hidden  (rep0)

Lines coloured by K; mean ± 1σ bands across seeds.

Usage:
    python exp2_market_collapse.py [--logs-dir logs/] [--good car] [--output ...]
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

def get_distrust_series(run_dir: str):
    """Return (ts, distrust_ratio) where distrust = passes / (bids + passes)."""
    import json
    files = load_state_files(run_dir)
    if not files:
        return None
    pts = []
    for p in files:
        with open(p) as f:
            s = json.load(f)
        t = s.get("timestep")
        bids   = s.get("lemon_market_bids_count", 0) or 0
        passes = s.get("lemon_market_passes_count", 0) or 0
        total  = bids + passes
        if t is not None and total > 0:
            pts.append((t, passes / total))
    if not pts:
        return None
    pts.sort()
    return np.array([x[0] for x in pts]), np.array([x[1] for x in pts])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Exp2 Fig F: Market Distrust Ratio")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good", default="car")
    ap.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "exp2", "exp2_market_collapse.pdf"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name_prefix = infer_name_prefix(args.logs_dir)
    print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_all_run_dirs(args.logs_dir, name_prefix, include_baseline=False)
    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_market_collapse", args.good)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        agg = deserialize_agg(load_cache_data(cache_path)["agg"])
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
            future_map = {ex.submit(get_distrust_series, d): (k, rv, seed) for k, rv, seed, d in jobs}
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
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), constrained_layout=True, sharey=True)
    fig.suptitle("Market Distrust over Time (Buyer Pass Rate)", fontsize=10, fontweight="bold")

    for rep_visible, ax, title in [(True, axes[0], "(A) Reputation visible"), (False, axes[1], "(B) Reputation hidden")]:
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Timestep")
        ax.axhline(0.5, color="#555555", lw=1.2, ls="--", alpha=0.8, zorder=2, label="50% pass rate (Akerlof threshold)")
        for k in K_VALUES:
            entry = agg.get((k, rep_visible))
            if entry is None:
                continue
            sat = k / 12
            lbl = f"K={k} ({sat:.0%} sybil)"
            plot_band(ax, entry, COLORS_K[k], lbl)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=7.5)

    axes[0].set_ylabel("Pass rate (passes / total activity)")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
