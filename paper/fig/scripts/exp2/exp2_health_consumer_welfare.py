"""
Exp2 Consumer Welfare Health Score — K × rep_visible heatmap.

Health score Hᵥ ∈ [0, 1] per grid cell, derived from the mean consumer
surplus across seeds with global min-max normalisation:

    Hᵥ = (w̄ − w_min) / (w_max − w_min)

    w̄ = mean consumer surplus for the cell (averaged over timesteps, then seeds)

A value of 1 indicates the highest welfare observed across the entire grid;
0 indicates the lowest.  Mean ± SE annotations in each cell.

Usage:
    python exp2_health_consumer_welfare.py [--logs-dir logs/] [--good car] [--output ...]
"""

import argparse
import concurrent.futures
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from exp2_cache import (
    get_data_dir, get_cache_path, is_cache_fresh,
    save_cache, load_cache_data, infer_name_prefix,
)
from exp2_common import (
    SEEDS, K_VALUES, load_state_files, resolve_run_dir, collect_all_run_dirs,
)

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.5,
    "axes.linewidth":     0.8,
    "axes.grid":          False,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":        False,
    "pdf.fonttype":       42,
})

K_ALL   = [0] + K_VALUES       # [0, 3, 6, 9]
REP_ALL = [True, False]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def compute_welfare(run_dir: str) -> float | None:
    """Return mean consumer surplus across all timesteps, or None."""
    files = load_state_files(run_dir)
    if not files:
        return None
    vals = []
    for p in files:
        with open(p) as f:
            s = json.load(f)
        w = s.get("lemon_market_avg_consumer_surplus")
        if w is not None:
            vals.append(float(w))
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_per_seed_welfare(logs_dir: str, name_prefix: str, workers: int = 8):
    """Return per_seed: dict[(i, j) -> [welfare_per_seed]]."""
    jobs = []
    for i, k in enumerate(K_ALL):
        for j, rv in enumerate(REP_ALL):
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, name_prefix, k, rv, seed)
                if d:
                    jobs.append((i, j, k, rv, seed, d))
                else:
                    print(f"  Missing: K={k} rep={int(rv)} seed={seed}", flush=True)

    print(f"Loading {len(jobs)} runs ...", flush=True)
    cell_vals: dict[tuple[int, int], list[float]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(compute_welfare, d): (i, j, k, rv, seed)
            for i, j, k, rv, seed, d in jobs
        }
        done, total = 0, len(jobs)
        for future in concurrent.futures.as_completed(future_map):
            i, j, k, rv, seed = future_map[future]
            done += 1
            val = future.result()
            status = "ok" if val is not None else "empty"
            print(f"  [{done}/{total}] K={k} rep={int(rv)} seed={seed} — {status}", flush=True)
            if val is not None:
                cell_vals.setdefault((i, j), []).append(val)

    return cell_vals


def try_reuse_heatmap_cache(data_dir: str, good: str):
    """Try to extract per-seed welfare from the exp2_heatmap cache."""
    heatmap_cache = get_cache_path(data_dir, "exp2_heatmap", good)
    if not os.path.isfile(heatmap_cache):
        return None
    try:
        hm_data = load_cache_data(heatmap_cache)
        psd_raw = hm_data.get("per_seed_data", {})
        if not psd_raw:
            return None
        cell_vals: dict[tuple[int, int], list[float]] = {}
        for key_str, cell in psd_raw.items():
            i, j = (int(x) for x in key_str.split(","))
            welfares = cell.get("consumer_welfare", [])
            if welfares:
                cell_vals[(i, j)] = list(welfares)
        return cell_vals if cell_vals else None
    except Exception as e:
        print(f"Heatmap cache reuse failed ({e}), computing from scratch.", flush=True)
        return None


# ---------------------------------------------------------------------------
# Health score computation
# ---------------------------------------------------------------------------

def compute_health(cell_vals: dict) -> dict:
    """Min-max normalise welfare into Hᵥ ∈ [0, 1] per cell.

    Returns dict[(i, j) -> {"health_mean", "health_se", "welfare_mean",
                             "welfare_se", "n", "seeds"}]
    """
    all_welfare = []
    for vals in cell_vals.values():
        all_welfare.extend(vals)

    w_min = min(all_welfare) if all_welfare else 0.0
    w_max = max(all_welfare) if all_welfare else 1.0
    w_range = w_max - w_min if w_max > w_min else 1.0

    results = {}
    for (i, j), vals in cell_vals.items():
        n = len(vals)
        seed_health = [(v - w_min) / w_range for v in vals]
        mean_h = float(np.mean(seed_health))
        se_h = (float(np.std(seed_health, ddof=1) / np.sqrt(n))
                if n > 1 else 0.0)
        mean_w = float(np.mean(vals))
        se_w = (float(np.std(vals, ddof=1) / np.sqrt(n))
                if n > 1 else 0.0)
        results[(i, j)] = {
            "health_mean":  mean_h,
            "health_se":    se_h,
            "welfare_mean": mean_w,
            "welfare_se":   se_w,
            "n":            n,
            "seeds":        seed_health,
            "w_min":        w_min,
            "w_range":      w_range,
        }
    return results


# ---------------------------------------------------------------------------
# Cache serialisation
# ---------------------------------------------------------------------------

def _serialize(cell_vals, health_cells):
    cv_ser = {f"{i},{j}": vals for (i, j), vals in cell_vals.items()}
    hc_ser = {}
    for (i, j), cell in health_cells.items():
        c = dict(cell)
        c.pop("seeds", None)
        hc_ser[f"{i},{j}"] = c
    return {"cell_vals": cv_ser, "health_cells": hc_ser}


def _deserialize(data):
    cv_raw = data.get("cell_vals", {})
    cell_vals = {
        (int(k.split(",")[0]), int(k.split(",")[1])): vals
        for k, vals in cv_raw.items()
    }
    hc_raw = data.get("health_cells", {})
    health_cells = {
        (int(k.split(",")[0]), int(k.split(",")[1])): cell
        for k, cell in hc_raw.items()
    }
    return cell_vals, health_cells


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(ax, fig, health_cells):
    """Welfare health-score heatmap over K × rep_visible grid."""
    n_row, n_col = len(K_ALL), len(REP_ALL)
    grid  = np.full((n_row, n_col), np.nan)
    avail = np.zeros((n_row, n_col), dtype=bool)

    for (i, j), cell in health_cells.items():
        grid[i, j]  = cell["health_mean"]
        avail[i, j] = True

    cmap    = plt.get_cmap("RdYlGn")
    display = np.ma.masked_invalid(grid)
    im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                        label="$H_W$ (welfare health)")

    for i in range(n_row):
        for j in range(n_col):
            if not avail[i, j]:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    boxstyle="square,pad=0", linewidth=0,
                    facecolor="#cccccc", hatch="///",
                    edgecolor="#888888", zorder=5,
                )
                ax.add_patch(rect)

    for (i, j), cell in health_cells.items():
        val = cell["health_mean"]
        se  = cell["health_se"]
        w_mean = cell["welfare_mean"]
        rgba = cmap(val)
        lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        txt_col = "white" if lum < 0.5 else "black"
        label = f"{val:.2f}\n±{se:.2f}\n(\\${w_mean:.1f})"
        ax.text(j, i, label, ha="center", va="center",
                fontsize=8, color=txt_col, zorder=10)

    col_labels = ["Rep. visible", "Rep. hidden"]
    row_labels = [f"K={k}" for k in K_ALL]
    ax.set_xticks(range(n_col))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n_row))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Reputation visibility")
    ax.set_ylabel("Sybil count ($K$)")
    ax.set_title("Consumer Welfare Health Score")

    hatch_patch = mpatches.Patch(
        facecolor="#cccccc", hatch="///", edgecolor="#888888", label="No data",
    )
    cbar.ax.legend(
        handles=[hatch_patch], loc="upper center",
        bbox_to_anchor=(0.5, -0.08), bbox_transform=cbar.ax.transAxes,
        fontsize=7, borderpad=0.5, handlelength=1.2,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Exp2 Consumer Welfare Health Score")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good", default="car")
    ap.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "exp2",
        "exp2_health_consumer_welfare.pdf"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name_prefix = infer_name_prefix(args.logs_dir)
    print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_all_run_dirs(args.logs_dir, name_prefix, include_baseline=True)
    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_health_consumer_welfare", args.good)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        cell_vals, health_cells = _deserialize(load_cache_data(cache_path))
    else:
        cell_vals = None
        if not args.force:
            cell_vals = try_reuse_heatmap_cache(data_dir, args.good)
            if cell_vals:
                print("Reused welfare data from exp2_heatmap cache.", flush=True)

        if cell_vals is None:
            cell_vals = build_per_seed_welfare(
                args.logs_dir, name_prefix, workers=args.workers)

        health_cells = compute_health(cell_vals)
        save_cache(cache_path,
                   _serialize(cell_vals, health_cells),
                   args.logs_dir, args.good)
        print(f"Cached: {cache_path}", flush=True)

    # Summary
    print("\nPer-cell welfare health:")
    for (i, j), cell in sorted(health_cells.items()):
        k  = K_ALL[i] if i < len(K_ALL) else "?"
        rv = REP_ALL[j] if j < len(REP_ALL) else "?"
        print(f"  K={k} rep={int(rv)}: H_W={cell['health_mean']:.3f} "
              f"±{cell['health_se']:.3f}  "
              f"(welfare=${cell['welfare_mean']:.1f})")

    all_seeds = []
    for cell in health_cells.values():
        all_seeds.extend(cell.get("seeds", [cell["health_mean"]]))
    if all_seeds:
        agg_mean = float(np.mean(all_seeds))
        agg_se = (float(np.std(all_seeds, ddof=1) / np.sqrt(len(all_seeds)))
                  if len(all_seeds) > 1 else 0.0)
        print(f"\nAggregate H_W: {agg_mean:.3f} ± {agg_se:.3f}  (n={len(all_seeds)})")

    # Plot
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    fig.subplots_adjust(left=0.18, right=0.88, bottom=0.18, top=0.82)

    plot_heatmap(ax, fig, health_cells)

    eq = r"$H_W = \frac{\bar{w} - w_{\min}}{w_{\max} - w_{\min}}$"
    fig.suptitle("Experiment 2: Consumer Welfare Health",
                 fontweight="bold", y=0.97)
    fig.text(0.5, 0.03, eq, ha="center", va="bottom", fontsize=8.5)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
