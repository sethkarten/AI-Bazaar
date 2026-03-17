"""
Fig 3: Experiment 1 — Crash Dynamics (3×3 timeseries)

Columns (left → right): increasing stabilization at dlc=3
  A: n_stab=0  (baseline crash, seed=8 only)
  B: n_stab=2  (partial recovery, seeds 8,16,64)
  C: n_stab=4  (stable market, seeds 8,16,64)

Rows (top → bottom):
  1: Mean price ± 1σ band + per-seed faint lines + unit-cost reference
  2: Active firm count (step-down survival curve)
  3: Filled orders per step

Usage:
    python exp1_timeseries.py [--logs-dir logs/] [--good food] [--output ...]
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
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder
from exp1_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data

plt.rcParams.update({
    "font.family":                    "serif",
    "font.size":                      11,
    "axes.titlesize":                 12,
    "axes.labelsize":                 11,
    "xtick.labelsize":                10,
    "ytick.labelsize":                10,
    "legend.fontsize":                10,
    "axes.axisbelow":                 True,
    "axes.grid":                      True,
    "grid.alpha":                     0.3,
    "grid.linewidth":                 0.5,
    "grid.color":                     "gray",
    "figure.constrained_layout.use":  True,
    "savefig.pad_inches":             0.01,
    "legend.frameon":                 True,
    "legend.framealpha":              0.9,
    "legend.edgecolor":               "0.8",
    "pdf.fonttype":                   42,
})

COLOR_PRICE    = "#0072B2"   # Okabe Blue
COLOR_FIRMS    = "#009E73"   # Okabe Green
COLOR_ORDERS   = "#E69F00"   # Okabe Orange
COLOR_COST_REF = "#D55E00"   # Okabe Vermillion

COLUMNS = [
    {"n_stab": 0, "dlc": 3, "seeds": [8],        "label": "No Stabilizer\n(Baseline)"},
    {"n_stab": 2, "dlc": 3, "seeds": [8, 16, 64], "label": "2 Stabilizing\nFirms"},
    {"n_stab": 4, "dlc": 3, "seeds": [8, 16, 64], "label": "4 Stabilizing\nFirms"},
]

ROW_LABELS = ["Mean price", "Active firms", "Filled orders/step"]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def collect_run_dirs(logs_dir):
    dirs = []
    for col in COLUMNS:
        for seed in col["seeds"]:
            d = resolve_run_dir(logs_dir, col["n_stab"], col["dlc"], seed)
            if d:
                dirs.append(d)
    return dirs


def _serialize(results, unit_costs):
    """Convert results dict (tuple keys, numpy arrays) to JSON-safe dict."""
    ser = {}
    for (col_idx, seed), run_data in results.items():
        key = f"{col_idx},{seed}"
        ser[key] = {}
        for metric, val in run_data.items():
            if val is None:
                ser[key][metric] = None
            else:
                ts, vals = val
                ser[key][metric] = [ts.tolist(), vals.tolist()]
    return {"results": ser, "unit_costs": unit_costs}


def _deserialize(data):
    results = {}
    for k, run_data in data["results"].items():
        col_idx, seed = (int(x) for x in k.split(","))
        deser = {}
        for metric, val in run_data.items():
            if val is None:
                deser[metric] = None
            else:
                deser[metric] = (np.array(val[0]), np.array(val[1]))
        results[(col_idx, seed)] = deser
    return results, data["unit_costs"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def resolve_run_dir(logs_dir, n_stab, dlc, seed):
    if n_stab == 0:
        if dlc == 3 and seed == 8:
            path = os.path.join(logs_dir, "exp1_baseline")
            return path if os.path.isdir(path) else None
        return None
    path = os.path.join(logs_dir, f"exp1_stab_{n_stab}_dlc{dlc}_seed{seed}")
    return path if os.path.isdir(path) else None


def load_states(run_dir):
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


# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------

def get_price_series(run_dir, good):
    """Returns (timesteps, mean_price_per_step) or None."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    price_df = db.price_per_firm_over_time(good)
    per_ts = (
        price_df[price_df["value"] > 0]
        .groupby("timestep")["value"]
        .mean()
        .reset_index()
        .sort_values("timestep")
    )
    if per_ts.empty:
        return None
    return per_ts["timestep"].values, per_ts["value"].values


def get_active_firms_series(run_dir):
    """Returns (timesteps, count_per_step) or None."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    df = db.firms_in_business_over_time().sort_values("timestep")
    if df.empty:
        return None
    return df["timestep"].values, df["value"].values


def get_volume_series(run_dir):
    """Returns (timesteps, filled_orders_per_step) or None."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    df = db.filled_orders_count_over_time().sort_values("timestep")
    if df.empty:
        return None
    return df["timestep"].values, df["value"].values


def load_one_run(run_dir, good):
    """Load all three metrics for a single run directory."""
    price  = get_price_series(run_dir, good)
    firms  = get_active_firms_series(run_dir)
    volume = get_volume_series(run_dir)
    return {"price": price, "firms": firms, "volume": volume}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

class _BelowCostNorm(mcolors.Normalize):
    """Linear [vmin=cost, vmax] → [0,1]; values below cost → 1.0 (max red)."""
    def __call__(self, value, clip=None):
        val = np.ma.asarray(value, dtype=float)
        scaled = (val - self.vmin) / (self.vmax - self.vmin)
        scaled = np.ma.where(val < self.vmin, 1.0, scaled)
        scaled = np.ma.clip(scaled, 0.0, 1.0)
        if np.ndim(value) == 0:
            return float(scaled)
        return scaled


def _gradient_line(ax, ts, vals, cmap, norm, lw=1.8, alpha=1.0, zorder=4):
    """Draw a line whose color varies per segment according to cmap(norm(value))."""
    if len(ts) < 2:
        return
    points   = np.array([ts, vals], dtype=float).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_vals = (vals[:-1] + vals[1:]) / 2.0
    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(seg_vals)
    ax.add_collection(lc)


def plot_price_column(ax, seeds_data, cmap, norm, is_baseline):
    """Price row: gradient line whose color tracks the heatmap avg-price scale."""
    valid = [(ts, v) for ts, v in seeds_data if ts is not None]
    if not valid:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=8)
        return

    if is_baseline or len(valid) == 1:
        ts, vals = valid[0]
        _gradient_line(ax, ts, vals, cmap, norm, lw=1.8, zorder=4)
    else:
        # Thin gradient seed lines
        for ts, vals in valid:
            _gradient_line(ax, ts, vals, cmap, norm, lw=0.7, alpha=0.35, zorder=2)

        # Bold gradient mean ± 1σ band
        ts_list  = [ts   for ts, _ in valid]
        val_list = [vals for _, vals in valid]
        common, arr = _interp_common(ts_list, val_list)
        mean_v = arr.mean(axis=0)
        std_v  = arr.std(axis=0)
        # Band: single color sampled at the overall mean price
        band_color = cmap(float(norm(float(np.mean(mean_v)))))
        ax.fill_between(common, mean_v - std_v, mean_v + std_v,
                        color=band_color, alpha=0.15, zorder=3)
        _gradient_line(ax, common, mean_v, cmap, norm, lw=1.8, zorder=4)


def _interp_common(ts_list, val_list):
    """Interpolate all series to a common integer timestep grid, return (ts, array)."""
    if not ts_list:
        return None, None
    t_min = min(ts[0]  for ts in ts_list)
    t_max = max(ts[-1] for ts in ts_list)
    common = np.arange(t_min, t_max + 1, dtype=float)
    interped = np.array([np.interp(common, ts.astype(float), v.astype(float))
                         for ts, v in zip(ts_list, val_list)])
    return common, interped


def plot_metric_column(ax, seeds_data, metric_key, color, is_baseline,
                       drawstyle="default", y_min=None):
    """
    Plot one (row, column) cell.

    seeds_data: list of (ts, values) tuples (or None entries for missing seeds)
    is_baseline: if True, single bold line only (no band, no faint seeds)
    """
    valid = [(ts, v) for ts, v in seeds_data if ts is not None]
    if not valid:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=8)
        return

    ds_kw = {"drawstyle": drawstyle} if drawstyle != "default" else {}

    if is_baseline or len(valid) == 1:
        ts, vals = valid[0]
        ax.plot(ts, vals, color=color, linewidth=1.8, zorder=4, **ds_kw)
    else:
        # Thin individual seed lines
        for ts, vals in valid:
            ax.plot(ts, vals, color=color, linewidth=0.7, alpha=0.35, zorder=2, **ds_kw)

        # Bold mean ± 1σ band on common grid
        ts_list  = [ts   for ts, _ in valid]
        val_list = [vals for _, vals in valid]
        common, arr = _interp_common(ts_list, val_list)
        mean_v = arr.mean(axis=0)
        std_v  = arr.std(axis=0)
        lo = mean_v - std_v
        if y_min is not None:
            lo = np.maximum(lo, y_min)
            mean_v = np.maximum(mean_v, y_min)
        ax.fill_between(common, lo, mean_v + std_v,
                        color=color, alpha=0.15, zorder=3)
        ax.plot(common, mean_v, color=color, linewidth=1.8, zorder=4, **ds_kw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fig 3: Crash Dynamics timeseries (3×3)")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good",     default="food")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "exp1", "exp1_timeseries.pdf"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp1_timeseries", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir)

    if is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        results, unit_costs = _deserialize(load_cache_data(cache_path))
    else:
        # Collect all (col_idx, seed) → run_dir jobs
        jobs = []
        for col_idx, col in enumerate(COLUMNS):
            for seed in col["seeds"]:
                run_dir = resolve_run_dir(args.logs_dir, col["n_stab"], col["dlc"], seed)
                if run_dir:
                    jobs.append((col_idx, seed, run_dir))
                else:
                    print(f"  Missing: n_stab={col['n_stab']}, dlc={col['dlc']}, seed={seed}")

        print(f"Loading {len(jobs)} runs...", flush=True)
        results = {}  # (col_idx, seed) -> data dict
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {
                ex.submit(load_one_run, run_dir, args.good): (col_idx, seed)
                for col_idx, seed, run_dir in jobs
            }
            done = 0
            for future in concurrent.futures.as_completed(future_map):
                col_idx, seed = future_map[future]
                done += 1
                data = future.result()
                has = {k: (v is not None) for k, v in data.items()}
                col = COLUMNS[col_idx]
                print(f"  [{done}/{len(jobs)}] n_stab={col['n_stab']} seed={seed} — {has}", flush=True)
                results[(col_idx, seed)] = data

        # Load unit costs (sequential, fast)
        unit_costs = []
        for col in COLUMNS:
            uc = 1.0
            for seed in col["seeds"]:
                run_dir = resolve_run_dir(args.logs_dir, col["n_stab"], col["dlc"], seed)
                if run_dir:
                    uc = get_unit_cost(run_dir)
                    break
            unit_costs.append(uc)

        save_cache(cache_path, _serialize(results, unit_costs), args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

    # Price gradient colormap: coolwarm with _BelowCostNorm matching heatmap panel B.
    # vmin = unit cost c (blue anchor); vmax = max price seen in ablated runs (n_stab > 0).
    uc_ref = float(np.mean(unit_costs)) if unit_costs else 1.0
    ablated_price_max = 0.0
    for col_idx, col in enumerate(COLUMNS):
        if col["n_stab"] > 0:
            for seed in col["seeds"]:
                d = results.get((col_idx, seed))
                if d and d["price"] is not None:
                    ts, vals = d["price"]
                    if len(vals) > 0:
                        ablated_price_max = max(ablated_price_max, float(np.max(vals)))
    if ablated_price_max <= uc_ref:
        ablated_price_max = uc_ref + 1.0
    _cmap_price = plt.get_cmap("coolwarm")
    _price_norm = _BelowCostNorm(vmin=uc_ref, vmax=ablated_price_max)

    # Build figure
    n_rows, n_cols = 3, len(COLUMNS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(9.0, 7.5),
        sharex="col",
        constrained_layout=True,
    )

    metric_keys   = ["price", "firms", "volume"]
    colors        = [COLOR_PRICE, COLOR_FIRMS, COLOR_ORDERS]
    drawstyles    = ["default", "steps-post", "default"]
    y_mins        = [None, 0, 0]

    for col_idx, col in enumerate(COLUMNS):
        is_baseline = (col["n_stab"] == 0)

        for row_idx in range(n_rows):
            ax = axes[row_idx][col_idx]
            metric = metric_keys[row_idx]
            ds     = drawstyles[row_idx]
            ym     = y_mins[row_idx]

            seeds_data = []
            for seed in col["seeds"]:
                key  = (col_idx, seed)
                data = results.get(key)
                if data and data[metric] is not None:
                    seeds_data.append(data[metric])
                else:
                    seeds_data.append((None, None))

            if row_idx == 0:
                plot_price_column(ax, seeds_data, _cmap_price, _price_norm, is_baseline)
            else:
                color = colors[row_idx]
                plot_metric_column(ax, seeds_data, metric, color, is_baseline,
                                   drawstyle=ds, y_min=ym)

            # Collapse markers: vertical dashed red line at first firm-drop timestep
            if row_idx == 1 and not is_baseline:
                for (ts_f, firms_f) in [sd for sd in seeds_data if sd[0] is not None]:
                    first_drop = next(
                        (t for t, f in zip(ts_f, firms_f) if f < 5), None)
                    if first_drop is not None:
                        ax.axvline(first_drop, color='#D55E00', lw=0.8,
                                   ls='--', alpha=0.6, zorder=3)

            # Price row extras
            if row_idx == 0:
                uc = unit_costs[col_idx]
                ax.axhline(uc, color=COLOR_COST_REF, linestyle="--", linewidth=1.2,
                           alpha=0.8, zorder=5, label=f"Unit cost {uc:.1f}")
                ax.axhspan(0, uc, color=COLOR_COST_REF, alpha=0.05, zorder=1)
                ax.legend(loc="upper right", handlelength=1.5, fontsize=10)
                ax.set_title(col["label"])

            # Active firms row: integer y-axis, capped at 5
            if row_idx == 1:
                ax.set_ylim(0, 5.5)
                ax.set_yticks([0, 1, 2, 3, 4, 5])

            # Orders row: non-negative
            if row_idx == 2:
                ax.set_ylim(bottom=0)
                ax.set_xlabel("Day")

            # Row y-labels on leftmost column
            if col_idx == 0:
                ax.set_ylabel(ROW_LABELS[row_idx])

    for row_i in range(n_rows):
        for col_i in range(n_cols):
            axes[row_i][col_i].set_xlim(0, 365)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
