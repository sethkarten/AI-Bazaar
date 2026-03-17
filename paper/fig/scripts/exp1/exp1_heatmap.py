"""
Fig 1: Experiment 1 Heatmap — 2×2 metric heatmap over dlc × n_stab grid.

Metrics:
  A) Bankruptcy rate  (RdPu, higher = worse)
  B) Final avg price  (RdBu diverging, centered at unit cost c)
  C) Total volume     (YlGn log-normalized, relative to baseline)
  D) Price volatility (YlOrRd, higher = worse)

Grid: dlc ∈ {1, 3, 5}  ×  n_stab ∈ {0, 1, 2, 4, 5}
  n_stab=0: baseline (no stabilizing firm), exists only for dlc=3 seed=8 → "exp1_baseline"
  All other cells: "exp1_stab_{n_stab}_dlc{dlc}_seed{seed}", averaged over seeds 8, 16, 64.
Missing cells rendered as hatched NaN.
Per-seed dots overlaid on each cell (green=survived, red=collapsed).
Stability borders: black outline when bankruptcy_rate < 0.5 AND final_avg_price >= unit_cost.

Usage:
    python exp1_heatmap.py [--logs-dir logs/] [--good food] [--output ...]
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
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder
from exp1_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "lines.linewidth":    1.5,
    "axes.linewidth":     0.8,
    "axes.grid":          False,   # heatmaps don't use grid
    "axes.axisbelow":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":        False,
})

DLC_VALUES    = [1, 3, 5]
N_STAB_VALUES = [0, 1, 2, 4, 5]
SEEDS         = [8, 16, 64]


def collect_run_dirs(logs_dir):
    dirs = []
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, dlc, n_stab, seed)
                if d:
                    dirs.append(d)
    return dirs


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


def resolve_run_dir(logs_dir, dlc, n_stab, seed):
    """Return run directory path for given config; None if doesn't exist."""
    if n_stab == 0:
        # Baseline (no stabilizing firm): only exists for dlc=3, seed=8
        if dlc == 3 and seed == 8:
            path = os.path.join(logs_dir, "exp1_baseline")
            return path if os.path.isdir(path) else None
        return None
    if n_stab == 5:
        path = os.path.join(logs_dir, f"exp1_stab_5_dlc{dlc}_seed{seed}")
        return path if os.path.isdir(path) else None
    # Standard sweep runs
    path = os.path.join(logs_dir, f"exp1_stab_{n_stab}_dlc{dlc}_seed{seed}")
    return path if os.path.isdir(path) else None


def load_states(run_dir):
    """Sorted list of valid (non-empty, parseable) state_t*.json paths in run_dir."""
    files = glob.glob(os.path.join(run_dir, "state_t*.json"))
    files.sort(key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or "0"))
    valid = []
    for p in files:
        if os.path.getsize(p) == 0:
            continue
        try:
            with open(p) as f:
                json.load(f)
            valid.append(p)
        except (json.JSONDecodeError, OSError):
            pass
    return valid


def get_unit_cost(run_dir):
    """Mean supply_unit_cost from firm_attributes.json; fallback 1.0."""
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
    """Returns scalar dict: bankruptcy_rate, final_avg_price, total_volume, price_std."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)

    # Bankruptcy rate: 1 - (active firms at last step) / num_firms
    firms_df = db.firms_in_business_over_time().sort_values("timestep")
    if firms_df.empty:
        return None
    states = db.states
    first_firms = len(states[0].get("firms", []))
    if first_firms == 0:
        return None
    last_active = int(firms_df.iloc[-1]["value"])
    bankruptcy_rate = 1.0 - last_active / first_firms

    # Final avg price: mean price among in-business firms at last timestep
    last_state = states[-1]
    active_firm_names = {f["name"] for f in last_state.get("firms", []) if f.get("in_business", False)}
    prices_at_last = []
    for f in last_state.get("firms", []):
        if f.get("in_business", False) and f.get("name") in active_firm_names:
            prices = f.get("prices") or {}
            p = prices.get(good)
            if isinstance(p, (int, float)) and p > 0:
                prices_at_last.append(p)
    final_avg_price = float(np.mean(prices_at_last)) if prices_at_last else 0.0

    # Total volume: sum of filled_orders_count across all timesteps
    vol_df = db.filled_orders_count_over_time()
    total_volume = int(vol_df["value"].sum()) if not vol_df.empty else 0

    # Price std: std of per-timestep mean price across all timesteps
    price_df = db.price_per_firm_over_time(good)
    if not price_df.empty:
        per_ts_mean = price_df[price_df["value"] > 0].groupby("timestep")["value"].mean()
        price_std = float(per_ts_mean.std()) if len(per_ts_mean) > 1 else 0.0
    else:
        price_std = 0.0

    return {
        "bankruptcy_rate": bankruptcy_rate,
        "final_avg_price": final_avg_price,
        "total_volume": total_volume,
        "price_std": price_std,
    }


def build_grid(logs_dir, good, workers=8):
    """
    Returns dict[metric_name] -> 2D array shape (len(N_STAB_VALUES), len(DLC_VALUES)).
    NaN where data missing. Also returns global unit_cost, boolean available mask,
    and per_seed_data: {(i, j): {metric: [seed_val1, seed_val2, ...]}}
    """
    n_row = len(N_STAB_VALUES)
    n_col = len(DLC_VALUES)
    metric_names = ["bankruptcy_rate", "final_avg_price", "total_volume", "price_std"]
    grids       = {m: np.full((n_row, n_col), np.nan) for m in metric_names}
    annotations = {m: [[None] * n_col for _ in range(n_row)] for m in metric_names}
    available   = np.zeros((n_row, n_col), dtype=bool)
    unit_costs  = []

    jobs = []
    for i, n_stab in enumerate(N_STAB_VALUES):
        for j, dlc in enumerate(DLC_VALUES):
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed)
                if run_dir is not None:
                    jobs.append((i, j, n_stab, dlc, seed, run_dir))

    total = len(jobs)
    print(f"Loading {total} runs...", flush=True)

    cell_seed_vals  = {}  # (i, j) -> {metric: [values]}
    cell_unit_costs = {}  # (i, j) -> [unit costs]

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
            label  = f"stab={n_stab} dlc={dlc} seed={seed}"
            status = "ok" if metrics else "empty"
            print(f"  [{done}/{total}] {label} — {status}", flush=True)
            if metrics:
                if (i, j) not in cell_seed_vals:
                    cell_seed_vals[(i, j)]  = {m: [] for m in metric_names}
                    cell_unit_costs[(i, j)] = []
                for m in metric_names:
                    cell_seed_vals[(i, j)][m].append(metrics[m])
                cell_unit_costs[(i, j)].append(get_unit_cost(run_dir))
                unit_costs.append(cell_unit_costs[(i, j)][-1])

    per_seed_data = {}
    for (i, j), seed_vals in cell_seed_vals.items():
        if seed_vals["bankruptcy_rate"]:
            available[i, j] = True
            per_seed_data[(i, j)] = {m: list(seed_vals[m]) for m in metric_names}
            for m in metric_names:
                vals   = seed_vals[m]
                mean_v = float(np.mean(vals))
                grids[m][i, j]       = mean_v
                annotations[m][i][j] = f"{mean_v:.2f}"

    unit_cost = float(np.mean(unit_costs)) if unit_costs else 1.0
    return grids, annotations, available, unit_cost, per_seed_data


class _BelowCostNorm(mcolors.Normalize):
    """
    Linear [vmin, vmax] → [0, 1] (blue→red via coolwarm).
    Values below vmin (unit cost) are mapped to 1.0 (max red) instead of 0.0 (blue).
    """
    def __call__(self, value, clip=None):
        val = np.ma.asarray(value, dtype=float)
        scaled = (val - self.vmin) / (self.vmax - self.vmin)
        scaled = np.ma.where(val < self.vmin, 1.0, scaled)
        scaled = np.ma.clip(scaled, 0.0, 1.0)
        if np.ndim(value) == 0:
            return float(scaled)
        return scaled


def draw_hatch_cell(ax, col_idx, row_idx):
    """Draw a hatched rectangle over cell (col_idx, row_idx) in imshow coordinates."""
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
    """Overlay per-seed dots at bottom of cell. Red=collapsed (br>0), Green=survived."""
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
        ax.scatter(
            [x], [y], s=10, c=[dot_color],
            edgecolors="white", linewidths=0.5,
            zorder=8, clip_on=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Fig 1: Exp1 Heatmap")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "exp1", "exp1_heatmap.pdf"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp1_heatmap", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir)

    if is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        cached = load_cache_data(cache_path)
        if "per_seed_data" in cached:
            print(f"Using cached data: {cache_path}", flush=True)
            grids, annotations, available, unit_cost, per_seed_data = _deserialize(cached)
        else:
            print("Cache missing per_seed_data, rebuilding...", flush=True)
            grids, annotations, available, unit_cost, per_seed_data = build_grid(
                args.logs_dir, args.good, workers=args.workers)
            save_cache(cache_path, _serialize(grids, annotations, available, unit_cost, per_seed_data),
                       args.logs_dir, args.good)
    else:
        print(f"Loading runs from: {args.logs_dir}")
        grids, annotations, available, unit_cost, per_seed_data = build_grid(
            args.logs_dir, args.good, workers=args.workers)
        save_cache(cache_path, _serialize(grids, annotations, available, unit_cost, per_seed_data),
                   args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)
    print(f"Unit cost: {unit_cost:.3f}")

    # Volume normalization (baseline = dlc=3, n_stab=0)
    bl_i = N_STAB_VALUES.index(0)
    bl_j = DLC_VALUES.index(3)
    baseline_vol = grids["total_volume"][bl_i, bl_j]
    valid_vols = grids["total_volume"][~np.isnan(grids["total_volume"])]
    if np.isnan(baseline_vol) or baseline_vol == 0:
        baseline_vol = float(np.mean(valid_vols)) if len(valid_vols) > 0 else 1.0

    vol_norm_grid = np.full_like(grids["total_volume"], np.nan)
    vol_annotations = [[None] * len(DLC_VALUES) for _ in range(len(N_STAB_VALUES))]
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if not np.isnan(grids["total_volume"][i, j]):
                ratio = grids["total_volume"][i, j] / baseline_vol
                vol_norm_grid[i, j] = ratio
                vol_annotations[i][j] = f"{ratio:.2f}x"

    # Panel config: (title, metric_key, colormap, mode)
    panels = [
        ("(A) Bankruptcy Rate",        "bankruptcy_rate", "RdPu",   "regular"),
        ("(B) Final Avg Price / $c$",  "final_avg_price", "coolwarm", "range"),
        ("(C) Total Market Volume",    "vol_norm",        "YlGn",   "lognorm"),
        ("(D) Price Volatility $σ$",   "price_std",       "coolwarm", "regular"),
    ]

    # 5 n_stab rows × 3 dlc cols — use full two-column width
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (title, metric_key, cmap_name, mode) in zip(axes_flat, panels):
        # Select grid and annotations
        if metric_key == "vol_norm":
            grid   = vol_norm_grid
            annots = vol_annotations
        else:
            grid   = grids[metric_key]
            annots = annotations[metric_key]

        valid_vals = grid[~np.isnan(grid)]

        if len(valid_vals) == 0:
            ax.set_title(title)
            ax.axis("off")
            continue

        cmap = plt.get_cmap(cmap_name)

        # Compute norm per mode
        if mode == "range":
            # vmax from ablated runs only; vmin anchored at unit cost c
            ablated_vals = grid[1:, :][~np.isnan(grid[1:, :])]
            vmax_plot = float(np.nanmax(ablated_vals)) if len(ablated_vals) > 0 else float(np.nanmax(valid_vals))
            if vmax_plot <= unit_cost:
                vmax_plot = unit_cost + 1.0
            norm = _BelowCostNorm(vmin=unit_cost, vmax=vmax_plot)
        elif mode == "diverging":
            center  = unit_cost
            abs_max = max(abs(float(np.nanmin(grid)) - center),
                          abs(float(np.nanmax(grid)) - center)) or 1.0
            norm = mcolors.Normalize(vmin=center - abs_max, vmax=center + abs_max)
        elif mode == "lognorm":
            v_min_raw = max(float(np.nanmin(valid_vals)), 0.05)
            v_max_raw = float(np.nanmax(valid_vals))
            if v_max_raw <= v_min_raw:
                v_max_raw = v_min_raw * 2
            norm = mcolors.LogNorm(vmin=v_min_raw, vmax=v_max_raw)
        else:  # regular
            vmin_plot = 0.0 if metric_key == "bankruptcy_rate" else float(np.nanmin(valid_vals))
            if metric_key == "bankruptcy_rate":
                vmax_plot = 1.0
            elif metric_key == "price_std":
                # Cap at ablated-run max to prevent baseline outlier from washing out variation
                ablated_vals = grid[1:, :][~np.isnan(grid[1:, :])]
                vmax_plot = float(np.nanmax(ablated_vals)) if len(ablated_vals) > 0 else float(np.nanmax(valid_vals))
            else:
                vmax_plot = float(np.nanmax(valid_vals))
            if vmax_plot <= vmin_plot:
                vmax_plot = vmin_plot + 1
            norm = mcolors.Normalize(vmin=vmin_plot, vmax=vmax_plot)

        display = np.ma.masked_invalid(grid)
        im = ax.imshow(
            display,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        cb_ax = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb_ax.ax.tick_params(labelsize=9)

        # Hatch missing cells
        for i in range(len(N_STAB_VALUES)):
            for j in range(len(DLC_VALUES)):
                if not available[i, j]:
                    draw_hatch_cell(ax, j, i)

        # Cell annotations (shifted up to make room for dots)
        for i in range(len(N_STAB_VALUES)):
            for j in range(len(DLC_VALUES)):
                if annots[i][j] is None:
                    continue
                val      = grid[i, j]
                norm_val = norm(val)
                rgba     = cmap(float(np.clip(norm_val, 0.0, 1.0)))
                lum      = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if lum < 0.5 else "black"
                ax.text(
                    j, i, annots[i][j],
                    ha="center", va="center",
                    fontsize=9, color=txt_color,
                    zorder=10,
                )

        # Stability borders: black outline when bankrupt < 50% AND price >= unit cost
        for i in range(len(N_STAB_VALUES)):
            for j in range(len(DLC_VALUES)):
                if not available[i, j]:
                    continue
                mb = grids["bankruptcy_rate"][i, j]
                mp = grids["final_avg_price"][i, j]
                if mb < 0.5 and mp >= unit_cost:
                    rect = mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1.0, 1.0,
                        linewidth=2.5, edgecolor="black", facecolor="none", zorder=15,
                    )
                    ax.add_patch(rect)

        ax.set_xticks(range(len(DLC_VALUES)))
        ax.set_xticklabels([f"{d}" for d in DLC_VALUES])
        ax.set_yticks(range(len(N_STAB_VALUES)))
        ax.set_yticklabels([f"$k$={n}" for n in N_STAB_VALUES])
        ax.set_xlabel("Consumer discovery limit (dlc)")
        ax.set_ylabel("Stabilizing firms ($k$)")
        ax.set_title(title)

    # Shared legend
    hatch_patch = mpatches.Patch(
        facecolor="#cccccc", hatch="///", edgecolor="#888888", label="No data")
    border_patch = mpatches.Rectangle(
        (0, 0), 1, 1, linewidth=2.5, edgecolor="black", facecolor="none",
        label="Stable zone (bankrupt $<$50\\%, price $\\geq c$)")
    fig.legend(handles=[hatch_patch, border_patch], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    fig.suptitle("Experiment 1: Stabilizing Firm Ablation", fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
