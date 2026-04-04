"""
exp1_model_comparison.py -- Multi-model performance comparison heatmaps.

Compares M models across 5 metrics:
  (A) Bankruptcy Rate         -- RdPu,    [0, 1] fixed
  (B) Final Avg Price / c     -- coolwarm, centered at unit cost
  (C) Total Market Volume     -- YlGn,    log-normalized, globally consistent
  (D) Price Volatility        -- YlOrRd,  [0, global_max]
  (E) Composite Health Score  -- RdYlGn,  [0, 1] fixed

Layout: 5 rows x M columns of heatmaps (dlc x k grid per cell).
Colormaps for rows B-D normalized globally across all models for fair comparison.
One colorbar per row.

Settings: k in {0, 1, 3, 5},  dlc in {1, 3, 5}

Data loading (per model, in priority order):
  1. Comparison cache:  paper/fig/exp1/comparisons/{name}/data/
  2. Heatmap cache:     paper/fig/exp1/{src}/data/exp1_heatmap_{good}.json
                        (written by exp1_run_all.py — reused here to avoid recomputation)
  3. Fallback:          compute from raw state files via build_grid()

Output: paper/fig/exp1/comparisons/{name}/{name}.pdf

Usage:
    python paper/fig/scripts/exp1/exp1_model_comparison.py \\
        --name claude_vs_gpt \\
        --src exp1_anthropic_claude-sonnet-4.6 \\
        --src exp1_openai_gpt-5.4 \\
        [--logs-dir logs/] [--good food] [--workers 8]
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
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder
from exp1_cache import get_cache_path, is_cache_fresh, save_cache, load_cache_data
from exp1_paths import SEEDS, resolve_run_dir

# ── Experiment matrices ────────────────────────────────────────────────────
# Full heatmap sweep (used to validate heatmap cache freshness)
HEATMAP_N_STAB_ALL = [0, 1, 2, 3, 4, 5]

# Subset shown in the comparison figure
N_STAB_VALUES = [0, 1, 3, 5]
DLC_VALUES    = [1, 3, 5]

# Row indices in the full heatmap grid that correspond to N_STAB_VALUES
COMP_ROW_IDX = [HEATMAP_N_STAB_ALL.index(k) for k in N_STAB_VALUES]

METRIC_NAMES = ["bankruptcy_rate", "final_avg_price", "total_volume", "price_std"]

# Base directory for comparison outputs, relative to this script
_SCRIPT_DIR   = os.path.dirname(__file__)
_COMPARISONS_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "exp1", "comparisons")

# ── rcParams (FigureMakerAgent standard) ───────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.4,
    "axes.linewidth":     0.8,
    "axes.grid":          False,
    "axes.axisbelow":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":        False,
    "pdf.fonttype":       42,
})


# ── Directory resolution ───────────────────────────────────────────────────

def collect_run_dirs(logs_dir, model="", n_stab_list=None):
    if n_stab_list is None:
        n_stab_list = N_STAB_VALUES
    dirs = []
    for n_stab in n_stab_list:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if d:
                    dirs.append(d)
    return dirs


# ── Serialization (comparison subset format) ──────────────────────────────

def _serialize(grids, annotations, available, unit_cost, per_seed_data):
    psd_ser = {
        f"{i},{j}": {m: vals for m, vals in cell.items()}
        for (i, j), cell in per_seed_data.items()
    }
    return {
        "grids":         {k: v.tolist() for k, v in grids.items()},
        "annotations":   annotations,
        "available":     available.tolist(),
        "unit_cost":     unit_cost,
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
    return grids, data["annotations"], available, data["unit_cost"], per_seed_data


def _deserialize_heatmap(data):
    """Deserialize the full 6-row heatmap cache (same format as exp1_heatmap.py)."""
    grids     = {k: np.array(v) for k, v in data["grids"].items()}
    available = np.array(data["available"], dtype=bool)
    psd_raw   = data.get("per_seed_data", {})
    per_seed_data = {
        (int(k.split(",")[0]), int(k.split(",")[1])): cell
        for k, cell in psd_raw.items()
    }
    return grids, data["annotations"], available, data["unit_cost"], per_seed_data


def _subset_to_comparison(grids, annotations, available, per_seed_data):
    """Subset from full 6-row heatmap data down to comparison N_STAB_VALUES rows."""
    rows = np.array(COMP_ROW_IDX)
    sub_grids     = {m: g[rows, :] for m, g in grids.items()}
    sub_available = available[rows, :]
    sub_ann       = {m: [ann[i] for i in COMP_ROW_IDX] for m, ann in annotations.items()}
    sub_psd = {}
    for new_i, old_i in enumerate(COMP_ROW_IDX):
        for (oi, j), cell in per_seed_data.items():
            if oi == old_i:
                sub_psd[(new_i, j)] = cell
    return sub_grids, sub_ann, sub_available, sub_psd


# ── Data loading ───────────────────────────────────────────────────────────

def load_states(run_dir):
    states_path = os.path.join(run_dir, "states.json")
    if os.path.isfile(states_path):
        with open(states_path) as f:
            return json.load(f)
    files = glob.glob(os.path.join(run_dir, "state_t*.json"))
    files.sort(key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or "0"))
    states = []
    for p in files:
        if os.path.getsize(p) == 0:
            continue
        try:
            with open(p) as f:
                states.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return states


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
    db = DataFrameBuilder(states=files)
    firms_df = db.firms_in_business_over_time().sort_values("timestep")
    if firms_df.empty:
        return None
    states = db.states
    first_firms = len(states[0].get("firms", []))
    if first_firms == 0:
        return None
    last_active = int(firms_df.iloc[-1]["value"])
    bankruptcy_rate = 1.0 - last_active / first_firms

    last_state = states[-1]
    prices_at_last = [
        f["prices"].get(good)
        for f in last_state.get("firms", [])
        if f.get("in_business") and isinstance(f.get("prices", {}).get(good), (int, float))
        and f["prices"][good] > 0
    ]
    final_avg_price = float(np.mean(prices_at_last)) if prices_at_last else 0.0

    vol_df = db.filled_orders_count_over_time()
    total_volume = int(vol_df["value"].sum()) if not vol_df.empty else 0

    price_df = db.price_per_firm_over_time(good)
    if not price_df.empty:
        per_ts_mean = price_df[price_df["value"] > 0].groupby("timestep")["value"].mean()
        price_std = float(per_ts_mean.std()) if len(per_ts_mean) > 1 else 0.0
    else:
        price_std = 0.0

    return {
        "bankruptcy_rate": bankruptcy_rate,
        "final_avg_price": final_avg_price,
        "total_volume":    total_volume,
        "price_std":       price_std,
    }


def build_grid(logs_dir, good, workers=8, model=""):
    """Compute metrics from raw state files for the comparison N_STAB subset."""
    n_row = len(N_STAB_VALUES)
    n_col = len(DLC_VALUES)
    grids       = {m: np.full((n_row, n_col), np.nan) for m in METRIC_NAMES}
    annotations = {m: [[None] * n_col for _ in range(n_row)] for m in METRIC_NAMES}
    available   = np.zeros((n_row, n_col), dtype=bool)
    unit_costs  = []

    jobs = []
    for i, n_stab in enumerate(N_STAB_VALUES):
        for j, dlc in enumerate(DLC_VALUES):
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if run_dir is not None:
                    jobs.append((i, j, n_stab, dlc, seed, run_dir))

    print(f"  Computing from scratch: {len(jobs)} runs for model={model or '(default)'}...", flush=True)

    cell_seed_vals  = {}
    cell_unit_costs = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_job = {
            ex.submit(compute_metrics, run_dir, good): (i, j, n_stab, dlc, seed, run_dir)
            for i, j, n_stab, dlc, seed, run_dir in jobs
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_job):
            i, j, n_stab, dlc, seed, run_dir = future_to_job[future]
            done += 1
            metrics = future.result()
            if metrics:
                if (i, j) not in cell_seed_vals:
                    cell_seed_vals[(i, j)]  = {m: [] for m in METRIC_NAMES}
                    cell_unit_costs[(i, j)] = []
                for m in METRIC_NAMES:
                    cell_seed_vals[(i, j)][m].append(metrics[m])
                cell_unit_costs[(i, j)].append(get_unit_cost(run_dir))
                unit_costs.append(cell_unit_costs[(i, j)][-1])

    per_seed_data = {}
    for (i, j), seed_vals in cell_seed_vals.items():
        if seed_vals["bankruptcy_rate"]:
            available[i, j] = True
            per_seed_data[(i, j)] = {m: list(seed_vals[m]) for m in METRIC_NAMES}
            for m in METRIC_NAMES:
                vals   = seed_vals[m]
                mean_v = float(np.mean(vals))
                grids[m][i, j] = mean_v
                if len(vals) > 1:
                    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    annotations[m][i][j] = f"{mean_v:.2f}\n+-{se:.2f}"
                else:
                    annotations[m][i][j] = f"{mean_v:.2f}"

    unit_cost = float(np.mean(unit_costs)) if unit_costs else 1.0
    return grids, annotations, available, unit_cost, per_seed_data


def load_model_data(src, logs_dir, good, workers, comp_data_dir, fig_exp1_dir):
    """
    Load grid data for one model. Priority:
      1. Comparison-specific cache in comp_data_dir
      2. Heatmap cache written by exp1_run_all (paper/fig/exp1/{src}/data/)
      3. Full compute via build_grid()
    Returns (grids, annotations, available, unit_cost, per_seed_data).
    """
    model    = src[len("exp1_"):] if src.startswith("exp1_") else src
    logs_dir = os.path.join(logs_dir, src)

    # ── 1. Comparison cache ──────────────────────────────────────────────
    comp_cache = get_cache_path(comp_data_dir, f"exp1_model_comparison_{model}", good)
    comp_run_dirs = collect_run_dirs(logs_dir, model, n_stab_list=N_STAB_VALUES)

    if is_cache_fresh(comp_cache, comp_run_dirs, logs_dir, good):
        cached = load_cache_data(comp_cache)
        if "per_seed_data" in cached:
            print(f"  [{model}] Using comparison cache.", flush=True)
            return _deserialize(cached)

    # ── 2. Heatmap cache (written by exp1_run_all) ───────────────────────
    heatmap_data_dir = os.path.join(fig_exp1_dir, src, "data")
    heatmap_cache    = get_cache_path(heatmap_data_dir, "exp1_heatmap", good)
    heatmap_run_dirs = collect_run_dirs(logs_dir, model, n_stab_list=HEATMAP_N_STAB_ALL)

    if is_cache_fresh(heatmap_cache, heatmap_run_dirs, logs_dir, good):
        raw = load_cache_data(heatmap_cache)
        if "per_seed_data" in raw:
            print(f"  [{model}] Loading from heatmap cache, subsetting to k={N_STAB_VALUES}.", flush=True)
            grids_full, ann_full, avail_full, unit_cost, psd_full = _deserialize_heatmap(raw)
            grids, annotations, available, per_seed_data = _subset_to_comparison(
                grids_full, ann_full, avail_full, psd_full)
            save_cache(comp_cache,
                       _serialize(grids, annotations, available, unit_cost, per_seed_data),
                       logs_dir, good)
            return grids, annotations, available, unit_cost, per_seed_data

    # ── 3. Fallback: compute from raw state files ────────────────────────
    print(f"  [{model}] No cache found, computing from raw state files.", flush=True)
    grids, annotations, available, unit_cost, per_seed_data = build_grid(
        logs_dir, good, workers, model)
    save_cache(comp_cache,
               _serialize(grids, annotations, available, unit_cost, per_seed_data),
               logs_dir, good)
    print(f"  [{model}] Cached -> {comp_cache}", flush=True)
    return grids, annotations, available, unit_cost, per_seed_data


# ── Health score ───────────────────────────────────────────────────────────

def compute_health_grids(all_data):
    """Compute health score grids for all models with global normalization."""
    all_pstd, all_pdev = [], []
    for d in all_data.values():
        g  = d["grids"]
        uc = d["unit_cost"]
        pstd = g["price_std"][~np.isnan(g["price_std"])]
        pdev = np.abs(g["final_avg_price"] / uc - 1.0)
        pdev = pdev[~np.isnan(pdev)]
        all_pstd.extend(pstd.tolist())
        all_pdev.extend(pdev.tolist())

    global_max_pstd = max(all_pstd) if all_pstd else 1.0
    global_max_pdev = max(all_pdev) if all_pdev else 1.0

    health_grids = {}
    for name, d in all_data.items():
        g  = d["grids"]
        uc = d["unit_cost"]
        n_row, n_col = g["bankruptcy_rate"].shape
        health = np.full((n_row, n_col), np.nan)
        for i in range(n_row):
            for j in range(n_col):
                br = g["bankruptcy_rate"][i, j]
                p  = g["final_avg_price"][i, j]
                ps = g["price_std"][i, j]
                if np.isnan(br):
                    continue
                if br >= 1.0:
                    health[i, j] = 0.0
                else:
                    s_surv  = 1.0 - br
                    s_stab  = 1.0 - ps / global_max_pstd if global_max_pstd > 0 else 1.0
                    s_price = 1.0 - min(abs(p / uc - 1.0) / global_max_pdev, 1.0) \
                              if global_max_pdev > 0 else 1.0
                    health[i, j] = (s_surv + s_stab + s_price) / 3.0
        health_grids[name] = health

    return health_grids


# ── Cell drawing helpers ───────────────────────────────────────────────────

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


def draw_seed_dots(ax, col, row, seed_br_vals):
    n = len(seed_br_vals)
    if n == 0:
        return
    if n == 1:
        xs = [col]
    elif n == 2:
        xs = [col - 0.18, col + 0.18]
    else:
        xs = [col - 0.25, col, col + 0.25]
    y = row + 0.30
    for x, br in zip(xs, seed_br_vals):
        dot_color = "#CC0000" if br > 0 else "#009E73"
        ax.scatter([x], [y], s=10, c=[dot_color],
                   edgecolors="white", linewidths=0.5, zorder=8, clip_on=True)


class _BelowCostNorm(mcolors.Normalize):
    """Map [vmin, vmax] -> [0, 1]; values below vmin map to 1.0 (max red)."""
    def __call__(self, value, clip=None):
        val    = np.ma.asarray(value, dtype=float)
        scaled = (val - self.vmin) / (self.vmax - self.vmin)
        scaled = np.ma.where(val < self.vmin, 1.0, scaled)
        scaled = np.ma.clip(scaled, 0.0, 1.0)
        if np.ndim(value) == 0:
            return float(scaled)
        return scaled


# ── Panel drawing ──────────────────────────────────────────────────────────

def draw_panel(ax, grid, annots, available, per_seed_data,
               grids_all, unit_cost, norm, cmap,
               show_ylabel, show_xlabel, show_colname, colname, rowname):
    display = np.ma.masked_invalid(grid)
    im = ax.imshow(display, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if not available[i, j]:
                draw_hatch_cell(ax, j, i)

    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            lbl = annots[i][j]
            if lbl is None:
                continue
            val      = grid[i, j]
            norm_val = norm(val)
            rgba     = cmap(float(np.clip(norm_val, 0.0, 1.0)))
            lum      = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, lbl, ha="center", va="center",
                    fontsize=6.5, color="white" if lum < 0.5 else "black", zorder=10)
            if (i, j) in per_seed_data:
                draw_seed_dots(ax, j, i, per_seed_data[(i, j)]["bankruptcy_rate"])

    br_grid = grids_all["bankruptcy_rate"]
    p_grid  = grids_all["final_avg_price"]
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if available[i, j] and br_grid[i, j] < 0.5 and p_grid[i, j] >= unit_cost:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    linewidth=2.0, edgecolor="black", facecolor="none", zorder=15))

    ax.set_xticks(range(len(DLC_VALUES)))
    ax.set_xticklabels([str(d) for d in DLC_VALUES])
    ax.set_yticks(range(len(N_STAB_VALUES)))
    if show_ylabel:
        ax.set_yticklabels([f"$k={n}$" for n in N_STAB_VALUES])
        ax.set_ylabel(rowname)
    else:
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xlabel("dlc")
    if show_colname:
        ax.set_title(colname, fontsize=9, fontweight="bold")

    return im


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Multi-model exp1 comparison figure.")
    ap.add_argument("--name", required=True,
                    help="Name for this comparison (used as folder and file name).")
    ap.add_argument("--src", action="append", metavar="SRC",
                    help="Subdirectory within logs/ per model (e.g. exp1_anthropic_claude-sonnet-4.6). Repeat for each model.")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good",    default="food")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.src:
        ap.error("At least one --src is required.")

    # ── Output and cache paths ───────────────────────────────────────────
    comp_dir      = os.path.abspath(os.path.join(_COMPARISONS_DIR, args.name))
    comp_data_dir = os.path.join(comp_dir, "data")
    output_path   = os.path.join(comp_dir, f"{args.name}.pdf")
    os.makedirs(comp_data_dir, exist_ok=True)

    # paper/fig/exp1/ — where per-model heatmap caches live
    fig_exp1_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp1"))

    # ── Load data per model ──────────────────────────────────────────────
    all_data    = {}
    model_order = []

    for src in args.src:
        model = src[len("exp1_"):] if src.startswith("exp1_") else src
        model_order.append(model)
        grids, annotations, available, unit_cost, per_seed_data = load_model_data(
            src, args.logs_dir, args.good, args.workers, comp_data_dir, fig_exp1_dir)
        all_data[model] = {
            "grids":         grids,
            "annotations":   annotations,
            "available":     available,
            "unit_cost":     unit_cost,
            "per_seed_data": per_seed_data,
        }

    M = len(model_order)

    # ── Health grids (global normalization across all models) ─────────────
    health_grids = compute_health_grids(all_data)
    for name in model_order:
        all_data[name]["health_grid"] = health_grids[name]

    # ── Global norms for rows B-D ─────────────────────────────────────────
    all_prices_ablated = []
    for d in all_data.values():
        g = d["grids"]["final_avg_price"]
        ablated = g[1:, :][~np.isnan(g[1:, :])]
        all_prices_ablated.extend(ablated.tolist())
    global_uc  = float(np.mean([d["unit_cost"] for d in all_data.values()]))
    price_vmax = float(np.nanmax(all_prices_ablated)) if all_prices_ablated else global_uc + 1.0
    if price_vmax <= global_uc:
        price_vmax = global_uc + 1.0

    all_vols = []
    for d in all_data.values():
        g = d["grids"]["total_volume"]
        all_vols.extend(g[~np.isnan(g)].tolist())
    vol_vmin = max(float(np.min(all_vols)), 0.05) if all_vols else 0.05
    vol_vmax = float(np.max(all_vols)) if all_vols else 1.0
    if vol_vmax <= vol_vmin:
        vol_vmax = vol_vmin * 2

    all_pstd = []
    for d in all_data.values():
        g = d["grids"]["price_std"]
        ablated = g[1:, :][~np.isnan(g[1:, :])]
        all_pstd.extend(ablated.tolist())
    pstd_vmax = float(np.max(all_pstd)) if all_pstd else 1.0

    # ── Row config ────────────────────────────────────────────────────────
    ROWS = [
        ("(A) Bankruptcy Rate",     "RdPu",    mcolors.Normalize(0.0, 1.0)),
        ("(B) Final Price / $c$",   "coolwarm", _BelowCostNorm(vmin=global_uc, vmax=price_vmax)),
        ("(C) Total Market Volume", "YlGn",    mcolors.LogNorm(vmin=vol_vmin, vmax=vol_vmax)),
        ("(D) Price Volatility",    "YlOrRd",  mcolors.Normalize(0.0, pstd_vmax)),
        ("(E) Health Score",        "RdYlGn",  mcolors.Normalize(0.0, 1.0)),
    ]
    N_ROWS = len(ROWS)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(3.2 * M + 0.8, 3.6 * N_ROWS), constrained_layout=True)
    gs  = fig.add_gridspec(N_ROWS, M + 1, width_ratios=[1.0] * M + [0.07])

    for r_idx, (row_label, cmap_name, norm) in enumerate(ROWS):
        cmap    = plt.get_cmap(cmap_name)
        last_im = None
        cbar_ax = fig.add_subplot(gs[r_idx, M])

        for c_idx, name in enumerate(model_order):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            d  = all_data[name]

            if r_idx == 4:  # health score
                grid       = d["health_grid"]
                annots_raw = [
                    [f"{grid[i, j]:.2f}" if not np.isnan(grid[i, j]) else None
                     for j in range(len(DLC_VALUES))]
                    for i in range(len(N_STAB_VALUES))
                ]
            else:
                metric_key = METRIC_NAMES[r_idx]
                grid       = d["grids"][metric_key]
                annots_raw = d["annotations"][metric_key]

            im = draw_panel(
                ax=ax,
                grid=grid,
                annots=annots_raw,
                available=d["available"],
                per_seed_data=d["per_seed_data"],
                grids_all=d["grids"],
                unit_cost=d["unit_cost"],
                norm=norm,
                cmap=cmap,
                show_ylabel=(c_idx == 0),
                show_xlabel=(r_idx == N_ROWS - 1),
                show_colname=(r_idx == 0),
                colname=name,
                rowname=row_label,
            )
            last_im = im

        cb = fig.colorbar(last_im, cax=cbar_ax)
        cb.ax.tick_params(labelsize=7)
        if cmap_name == "YlGn":
            fmt = mticker.ScalarFormatter()
            fmt.set_scientific(False)
            cbar_ax.yaxis.set_major_formatter(fmt)
            cbar_ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        cbar_ax.set_ylabel(row_label, fontsize=7, rotation=270, labelpad=14, va="bottom")

    hatch_patch  = mpatches.Patch(facecolor="#cccccc", hatch="///", edgecolor="#888888",
                                   label="No data")
    border_patch = mpatches.Rectangle(
        (0, 0), 1, 1, linewidth=2.0, edgecolor="black", facecolor="none",
        label="Stable zone (BR < 50%, price >= c)")
    green_dot = mpatches.Patch(color="#009E73", label="Seed: survived")
    red_dot   = mpatches.Patch(color="#CC0000", label="Seed: collapsed")
    fig.legend(handles=[hatch_patch, border_patch, green_dot, red_dot],
               loc="lower center", ncol=4, bbox_to_anchor=(0.45, -0.02), fontsize=8)

    fig.suptitle(f"Experiment 1: Model Comparison — {args.name}", fontweight="bold", fontsize=11)

    fig.savefig(output_path)
    print(f"\nSaved -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
