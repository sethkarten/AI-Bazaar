"""
Fig Exp2: Experiment 2 Heatmap — 2×2 metric heatmap over K × rep_visible grid.

Metrics:
  A) Sybil detection premium   (PuOr diverging, higher = buyers prefer honest)
  B) Avg consumer surplus      (YlGn, higher = better for buyers)
  C) Sybil revenue share       (RdPu, higher = sybils capturing more revenue)
  D) Market volume (total bids)(YlGn log-norm, relative to baseline)

Grid: K ∈ {0, 3, 6, 9}  ×  rep_visible ∈ {True, False}
  K=0 baseline: only rep_visible=True (rep0 cell hatched)
  Sybil detection: K=0 has no sybil → hatched

Per-seed dots overlaid on each cell.
Mean ± SE annotations in each cell.

Usage:
    python exp2_heatmap.py [--logs-dir logs/] [--good car] [--output ...]
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
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from exp2_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data, infer_name_prefix
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

K_ALL   = [0] + K_VALUES    # [0, 3, 6, 9]
REP_ALL = [True, False]      # rep_visible: True=rep1, False=rep0


# ---------------------------------------------------------------------------
# Metric extraction (scalar per run)
# ---------------------------------------------------------------------------

def compute_metrics(run_dir: str, k: int) -> dict | None:
    """Compute scalar metrics for one run directory."""
    files = load_state_files(run_dir)
    if not files:
        return None

    welfare_vals   = []
    sybil_rev_vals = []
    bids_total     = 0
    detection_vals = []  # per-timestep detection premium

    for p in files:
        with open(p) as f:
            s = json.load(f)

        # Consumer welfare: avg consumer surplus per step
        w = s.get("lemon_market_avg_consumer_surplus")
        if w is not None:
            welfare_vals.append(float(w))

        # Sybil revenue share
        sr = s.get("lemon_market_sybil_revenue_share")
        if sr is not None:
            sybil_rev_vals.append(float(sr))

        # Market volume: cumulative bids
        b = s.get("lemon_market_bids_count")
        if b is not None:
            bids_total += float(b)

        # Detection premium (K=0 has no sybil → skip)
        if k > 0:
            consumers = s.get("consumers", [])
            honest_hits, sybil_hits = [], []
            for cdata in consumers:
                if not isinstance(cdata, dict):
                    continue
                h_seen   = cdata.get("honest_seen_total",   0) or 0
                h_passed = cdata.get("honest_passed_total", 0) or 0
                s_seen   = cdata.get("sybil_seen_total",    0) or 0
                s_passed = cdata.get("sybil_passed_total",  0) or 0
                if h_seen > 0:
                    honest_hits.append((h_seen - h_passed) / h_seen)
                if s_seen > 0:
                    sybil_hits.append((s_seen - s_passed) / s_seen)
            if honest_hits and sybil_hits:
                detection_vals.append(np.mean(honest_hits) - np.mean(sybil_hits))

    return {
        "detection_premium": float(np.mean(detection_vals)) if detection_vals else np.nan,
        "consumer_welfare":  float(np.mean(welfare_vals))   if welfare_vals   else np.nan,
        "sybil_rev_share":   float(np.mean(sybil_rev_vals)) if sybil_rev_vals else np.nan,
        "market_volume":     float(bids_total),
    }


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

METRIC_NAMES = ["detection_premium", "consumer_welfare", "sybil_rev_share", "market_volume"]


def build_grid(logs_dir: str, name_prefix: str, workers: int = 8):
    n_row = len(K_ALL)
    n_col = len(REP_ALL)

    grids       = {m: np.full((n_row, n_col), np.nan) for m in METRIC_NAMES}
    annotations = {m: [[None] * n_col for _ in range(n_row)] for m in METRIC_NAMES}
    available   = np.zeros((n_row, n_col), dtype=bool)

    # Build job list
    jobs = []
    for i, k in enumerate(K_ALL):
        rep_opts = [True] if k == 0 else REP_ALL
        for j, rv in enumerate(REP_ALL):
            if rv not in rep_opts:
                continue
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, name_prefix, k, rv, seed)
                if d:
                    jobs.append((i, j, k, rv, seed, d))
                else:
                    print(f"  Missing: K={k} rep={int(rv)} seed={seed}", flush=True)

    print(f"Loading {len(jobs)} runs ...", flush=True)
    cell_vals = {}  # (i, j) -> {metric: [seed_val, ...]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(compute_metrics, d, k): (i, j, k, rv, seed)
            for i, j, k, rv, seed, d in jobs
        }
        done, total = 0, len(jobs)
        for future in concurrent.futures.as_completed(future_map):
            i, j, k, rv, seed = future_map[future]
            done += 1
            metrics = future.result()
            status = "ok" if metrics else "empty"
            print(f"  [{done}/{total}] K={k} rep={int(rv)} seed={seed} — {status}", flush=True)
            if metrics:
                key = (i, j)
                if key not in cell_vals:
                    cell_vals[key] = {m: [] for m in METRIC_NAMES}
                for m in METRIC_NAMES:
                    v = metrics[m]
                    if not np.isnan(v):
                        cell_vals[key][m].append(v)

    per_seed_data = {}
    for (i, j), mdict in cell_vals.items():
        primary_vals = mdict.get("consumer_welfare", [])
        if not primary_vals:
            primary_vals = mdict.get("market_volume", [])
        if not primary_vals:
            continue
        available[i, j] = True
        per_seed_data[(i, j)] = {m: list(v) for m, v in mdict.items()}
        for m in METRIC_NAMES:
            vals = mdict[m]
            if not vals:
                continue
            mean_v = float(np.mean(vals))
            grids[m][i, j] = mean_v
            if len(vals) > 1:
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                annotations[m][i][j] = f"{mean_v:.2f}\n±{se:.2f}"
            else:
                annotations[m][i][j] = f"{mean_v:.2f}"

    return grids, annotations, available, per_seed_data


# ---------------------------------------------------------------------------
# Cache serialisation
# ---------------------------------------------------------------------------

def _serialize(grids, annotations, available, per_seed_data):
    psd_ser = {
        f"{i},{j}": {m: vals for m, vals in cell.items()}
        for (i, j), cell in per_seed_data.items()
    }
    return {
        "grids":         {k: v.tolist() for k, v in grids.items()},
        "annotations":   annotations,
        "available":     available.tolist(),
        "per_seed_data": psd_ser,
    }


def _deserialize(data):
    grids     = {k: np.array(v) for k, v in data["grids"].items()}
    available = np.array(data["available"], dtype=bool)
    psd_raw   = data.get("per_seed_data", {})
    per_seed_data = {
        (int(k.split(",")[0]), int(k.split(",")[1])): cell
        for k, cell in psd_raw.items()
    }
    return grids, data["annotations"], available, per_seed_data


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_hatch_cell(ax, col_idx, row_idx):
    rect = mpatches.FancyBboxPatch(
        (col_idx - 0.5, row_idx - 0.5), 1.0, 1.0,
        boxstyle="square,pad=0",
        linewidth=0,
        facecolor="#cccccc",
        hatch="///",
        edgecolor="#888888",
        zorder=5,
    )
    ax.add_patch(rect)


def draw_seed_dots(ax, col, row, seed_vals):
    n = len(seed_vals)
    if n == 0:
        return
    if n == 1:
        xs = [col]
    elif n == 2:
        xs = [col - 0.18, col + 0.18]
    else:
        xs = [col - 0.25, col, col + 0.25]
    y = row + 0.30
    for x in xs:
        ax.scatter([x], [y], s=10, c=["#0072B2"],
                   edgecolors="white", linewidths=0.5, zorder=8, clip_on=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Exp2 Heatmap")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good", default="car")
    ap.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "exp2", "exp2_heatmap.pdf"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name_prefix = infer_name_prefix(args.logs_dir)
    print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_all_run_dirs(args.logs_dir, name_prefix, include_baseline=True)
    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_heatmap", args.good)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        cached = load_cache_data(cache_path)
        if "per_seed_data" in cached:
            print(f"Using cached data: {cache_path}", flush=True)
            grids, annotations, available, per_seed_data = _deserialize(cached)
        else:
            print("Cache missing per_seed_data, rebuilding ...", flush=True)
            grids, annotations, available, per_seed_data = build_grid(
                args.logs_dir, name_prefix, workers=args.workers)
            save_cache(cache_path, _serialize(grids, annotations, available, per_seed_data),
                       args.logs_dir, args.good)
    else:
        grids, annotations, available, per_seed_data = build_grid(
            args.logs_dir, name_prefix, workers=args.workers)
        save_cache(cache_path, _serialize(grids, annotations, available, per_seed_data),
                   args.logs_dir, args.good)
        print(f"Cached: {cache_path}", flush=True)

    # Volume normalisation relative to K=0 baseline
    bl_i = K_ALL.index(0)
    bl_j = REP_ALL.index(True)
    baseline_vol = grids["market_volume"][bl_i, bl_j]
    valid_vols   = grids["market_volume"][~np.isnan(grids["market_volume"])]
    if np.isnan(baseline_vol) or baseline_vol == 0:
        baseline_vol = float(np.mean(valid_vols)) if len(valid_vols) > 0 else 1.0

    vol_norm_grid  = np.full_like(grids["market_volume"], np.nan)
    vol_annots     = [[None] * len(REP_ALL) for _ in range(len(K_ALL))]
    for i in range(len(K_ALL)):
        for j in range(len(REP_ALL)):
            if not np.isnan(grids["market_volume"][i, j]):
                ratio = grids["market_volume"][i, j] / baseline_vol
                vol_norm_grid[i, j] = ratio
                vol_annots[i][j] = f"{ratio:.2f}x"

    # Panel config: (title, metric_key, colormap, vmin, vmax_override)
    panels = [
        ("(A) Sybil detection premium",  "detection_premium", "PuOr",  None, None),
        ("(B) Avg consumer surplus",      "consumer_welfare",  "YlGn",  None, None),
        ("(C) Sybil revenue share",       "sybil_rev_share",   "RdPu",  0.0,  1.0),
        ("(D) Total market volume",       "vol_norm",          "YlGn",  None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 9.0), constrained_layout=True)
    fig.suptitle("Experiment 2: Lemon Market — Summary Heatmap", fontweight="bold")

    col_labels = ["Rep. visible", "Rep. hidden"]
    row_labels  = [f"K={k}" for k in K_ALL]

    for ax, (title, metric_key, cmap_name, vmin_override, vmax_override) in zip(
            axes.flatten(), panels):

        if metric_key == "vol_norm":
            grid   = vol_norm_grid
            annots = vol_annots
        else:
            grid   = grids[metric_key]
            annots = annotations[metric_key]

        # K=0 has no sybil → detection_premium undefined
        if metric_key == "detection_premium":
            k0_i = K_ALL.index(0)
            grid = grid.copy()
            grid[k0_i, :] = np.nan

        valid_vals = grid[~np.isnan(grid)]

        if len(valid_vals) == 0:
            ax.set_title(title)
            ax.axis("off")
            continue

        cmap = plt.get_cmap(cmap_name)

        if metric_key == "detection_premium":
            # Diverging around 0
            abs_max = max(abs(float(np.nanmin(valid_vals))),
                          abs(float(np.nanmax(valid_vals)))) or 0.5
            norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        elif metric_key == "vol_norm":
            v_min_raw = max(float(np.nanmin(valid_vals)), 0.05)
            v_max_raw = float(np.nanmax(valid_vals))
            if v_max_raw <= v_min_raw:
                v_max_raw = v_min_raw * 2
            norm = mcolors.LogNorm(vmin=v_min_raw, vmax=v_max_raw)
        else:
            vmin_plot = vmin_override if vmin_override is not None else float(np.nanmin(valid_vals))
            vmax_plot = vmax_override if vmax_override is not None else float(np.nanmax(valid_vals))
            if vmax_plot <= vmin_plot:
                vmax_plot = vmin_plot + 1
            norm = mcolors.Normalize(vmin=vmin_plot, vmax=vmax_plot)

        display = np.ma.masked_invalid(grid)
        im = ax.imshow(display, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=9)
        if metric_key == "vol_norm":
            fmt = mticker.ScalarFormatter()
            fmt.set_scientific(False)
            cb.ax.yaxis.set_major_formatter(fmt)
            cb.ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # Hatch unavailable cells
        for i in range(len(K_ALL)):
            for j in range(len(REP_ALL)):
                # K=0, rep_hidden is always unavailable
                k0_unavail = (K_ALL[i] == 0 and not REP_ALL[j])
                # detection_premium for K=0 is undefined
                det_unavail = (metric_key == "detection_premium" and K_ALL[i] == 0)
                if k0_unavail or det_unavail or not available[i, j]:
                    draw_hatch_cell(ax, j, i)

        # Cell text annotations
        for i in range(len(K_ALL)):
            for j in range(len(REP_ALL)):
                txt = annots[i][j] if annots[i][j] else None
                if metric_key == "detection_premium" and K_ALL[i] == 0:
                    txt = None
                if txt is None:
                    continue
                val = grid[i, j]
                if np.isnan(val):
                    continue
                try:
                    norm_val = norm(val)
                    rgba     = cmap(float(np.clip(norm_val, 0.0, 1.0)))
                    lum      = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    txt_color = "white" if lum < 0.5 else "black"
                except Exception:
                    txt_color = "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.5, color=txt_color, zorder=10)

        # Seed dots
        for i in range(len(K_ALL)):
            for j in range(len(REP_ALL)):
                if not available[i, j]:
                    continue
                cell_data = per_seed_data.get((i, j), {})
                seed_vals = cell_data.get(
                    "detection_premium" if metric_key == "detection_premium" else
                    "consumer_welfare"  if metric_key == "consumer_welfare"  else
                    "sybil_rev_share"   if metric_key == "sybil_rev_share"   else
                    "market_volume", []
                )
                draw_seed_dots(ax, j, i, seed_vals)

        ax.set_xticks(range(len(REP_ALL)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(K_ALL)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Reputation visibility")
        ax.set_ylabel("Sybil count (K)")
        ax.set_title(title)

    hatch_patch = mpatches.Patch(
        facecolor="#cccccc", hatch="///", edgecolor="#888888", label="No data / N/A")
    fig.legend(handles=[hatch_patch], loc="lower center", ncol=1,
               bbox_to_anchor=(0.5, -0.03), fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
