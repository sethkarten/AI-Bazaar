"""
Fig: Experiment 3a — Crash Dynamics (3×3 timeseries)

Columns (left -> right): increasing stabilization at dlc=3
  A: n_stab=1  (seeds 8, 16, 64)
  B: n_stab=3  (seeds 8, 16, 64)
  C: n_stab=5  (seeds 8, 16, 64)

Rows (top -> bottom):
  1: Mean price +/- 1-sigma band + per-seed faint lines + unit-cost reference
     Vertical shock line at t=25 labelled "Shock (c x10)".
  2: Active firm count (step-down survival curve)
  3: Filled orders per step
     Vertical shock line at t=25 labelled "Shock (c x10)".

Run directory naming: exp3a_{model}_stab{n_stab}_dlc{dlc}_seed{seed}
  (no underscore after "stab", no baseline run)

Usage:
    python exp3_crash_timeseries.py [--logs-dir logs/exp3a_gemini-3-flash-preview/] [--good food] [--output ...]
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
COLOR_COST_REF = "#D55E00"   # Okabe Vermillion (also used for shock line)
COLOR_SHOCK    = "#D55E00"   # Okabe Vermillion

SHOCK_T = 25  # cost-shock timestep; fixed for all exp3a runs

DLC_FIXED = 3  # all columns use dlc=3

COLUMNS = [
    {"n_stab": 3, "dlc": DLC_FIXED, "seeds": [8, 16, 64], "label": "n_stab = 3"},
    {"n_stab": 5, "dlc": DLC_FIXED, "seeds": [8, 16, 64], "label": "n_stab = 5"},
]

ROW_LABELS = ["Mean price", "Active firms", "Filled orders/step"]


# ---------------------------------------------------------------------------
# Cache helpers (inlined)
# ---------------------------------------------------------------------------

def _get_cache_path(output_path, script_stem, good):
    fig_dir = os.path.dirname(os.path.abspath(output_path))
    data_dir = os.path.join(fig_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"{script_stem}_{good}.json")


def _newest_run_mtime(run_dirs):
    newest = 0.0
    for d in run_dirs:
        if not d or not os.path.isdir(d):
            continue
        states_path = os.path.join(d, "states.json")
        candidates = (
            [states_path] if os.path.isfile(states_path)
            else glob.glob(os.path.join(d, "state_t*.json"))
        )
        for f in candidates:
            try:
                mtime = os.path.getmtime(f)
                if mtime > newest:
                    newest = mtime
            except OSError:
                pass
    return newest


def _is_cache_fresh(cache_path, run_dirs, logs_dir, good):
    if not os.path.isfile(cache_path):
        return False
    try:
        with open(cache_path) as f:
            cached = json.load(f)
        meta = cached.get("_meta", {})
        if meta.get("logs_dir") != os.path.abspath(logs_dir):
            return False
        if meta.get("good") != good:
            return False
    except Exception:
        return False
    cache_mtime = os.path.getmtime(cache_path)
    return cache_mtime >= _newest_run_mtime(run_dirs)


def _save_cache(cache_path, data_dict, logs_dir, good):
    import time as _time
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "_meta": {
            "logs_dir": os.path.abspath(logs_dir),
            "good":     good,
            "created":  _time.time(),
        },
        "data": data_dict,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f)


def _load_cache_data(cache_path):
    with open(cache_path) as f:
        return json.load(f)["data"]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

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

def resolve_run_dir(logs_dir, n_stab, dlc, seed, model=""):
    """Return run directory path for given config; None if doesn't exist."""
    suffix = f"_stab{n_stab}_dlc{dlc}_seed{seed}"
    if model:
        path = os.path.join(logs_dir, f"exp3a_{model}{suffix}")
        if os.path.isdir(path):
            return path
    path = os.path.join(logs_dir, f"exp3a{suffix}")
    return path if os.path.isdir(path) else None


def collect_run_dirs(logs_dir, model=""):
    dirs = []
    for col in COLUMNS:
        for seed in col["seeds"]:
            d = resolve_run_dir(logs_dir, col["n_stab"], col["dlc"], seed, model=model)
            if d:
                dirs.append(d)
    return dirs


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


# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------

def get_price_series(run_dir, good):
    """Returns (timesteps, mean_price_per_step) or None."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(states=files)
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
    db = DataFrameBuilder(states=files)
    df = db.firms_in_business_over_time().sort_values("timestep")
    if df.empty:
        return None
    return df["timestep"].values, df["value"].values


def get_volume_series(run_dir):
    """Returns (timesteps, filled_orders_per_step) or None."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(states=files)
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
    """Linear [vmin=cost, vmax] -> [0,1]; values below cost -> 1.0 (max red)."""
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


def plot_price_column(ax, seeds_data, cmap, norm):
    """Price row: gradient line whose color tracks the heatmap avg-price scale."""
    valid = [(ts, v) for ts, v in seeds_data if ts is not None]
    if not valid:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=8)
        return

    if len(valid) == 1:
        ts, vals = valid[0]
        _gradient_line(ax, ts, vals, cmap, norm, lw=1.8, zorder=4)
    else:
        for ts, vals in valid:
            _gradient_line(ax, ts, vals, cmap, norm, lw=0.7, alpha=0.35, zorder=2)

        ts_list  = [ts   for ts, _ in valid]
        val_list = [vals for _, vals in valid]
        common, arr = _interp_common(ts_list, val_list)
        mean_v = arr.mean(axis=0)
        std_v  = arr.std(axis=0)
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


def plot_metric_column(ax, seeds_data, metric_key, color,
                       drawstyle="default", y_min=None):
    """
    Plot one (row, column) cell.

    seeds_data: list of (ts, values) tuples (or None entries for missing seeds)
    """
    valid = [(ts, v) for ts, v in seeds_data if ts is not None]
    if not valid:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=8)
        return

    ds_kw = {"drawstyle": drawstyle} if drawstyle != "default" else {}

    if len(valid) == 1:
        ts, vals = valid[0]
        ax.plot(ts, vals, color=color, linewidth=1.8, zorder=4, **ds_kw)
    else:
        for ts, vals in valid:
            ax.plot(ts, vals, color=color, linewidth=0.7, alpha=0.35, zorder=2, **ds_kw)

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
    parser = argparse.ArgumentParser(description="Exp3a Crash Dynamics timeseries (3x3)")
    parser.add_argument("--logs-dir", default="logs/exp3a_gemini-3-flash-preview/")
    parser.add_argument("--good",     default="food")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "exp3", "exp3_crash_timeseries.pdf"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    if not args.model:
        stem = os.path.basename(os.path.normpath(args.logs_dir))
        if stem.startswith("exp3a_"):
            args.model = stem[len("exp3a_"):]
        elif stem.startswith("exp3_"):
            args.model = stem[len("exp3_"):]

    cache_path = _get_cache_path(args.output, "exp3_crash_timeseries", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir, args.model)

    if _is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        results, unit_costs = _deserialize(_load_cache_data(cache_path))
    else:
        jobs = []
        for col_idx, col in enumerate(COLUMNS):
            for seed in col["seeds"]:
                run_dir = resolve_run_dir(args.logs_dir, col["n_stab"], col["dlc"], seed, model=args.model)
                if run_dir:
                    jobs.append((col_idx, seed, run_dir))
                else:
                    print(f"  Missing: n_stab={col['n_stab']}, dlc={col['dlc']}, seed={seed}")

        print(f"Loading {len(jobs)} runs...", flush=True)
        results = {}
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

        unit_costs = []
        for col in COLUMNS:
            uc = 1.0
            for seed in col["seeds"]:
                run_dir = resolve_run_dir(args.logs_dir, col["n_stab"], col["dlc"], seed, model=args.model)
                if run_dir:
                    uc = get_unit_cost(run_dir)
                    break
            unit_costs.append(uc)

        _save_cache(cache_path, _serialize(results, unit_costs), args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

    # Price gradient colormap: coolwarm with _BelowCostNorm
    uc_ref = float(np.mean(unit_costs)) if unit_costs else 1.0
    ablated_price_max = 0.0
    for col_idx, col in enumerate(COLUMNS):
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

    n_rows, n_cols = 3, len(COLUMNS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(9.0, 7.5),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    metric_keys = ["price", "firms", "volume"]
    colors      = [COLOR_PRICE, COLOR_FIRMS, COLOR_ORDERS]
    drawstyles  = ["default", "steps-post", "default"]
    y_mins      = [None, 0, 0]

    for col_idx, col in enumerate(COLUMNS):
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
                plot_price_column(ax, seeds_data, _cmap_price, _price_norm)
            else:
                plot_metric_column(ax, seeds_data, metric, colors[row_idx],
                                   drawstyle=ds, y_min=ym)

            # Collapse markers on firm-count row: vertical dashed line at first firm-drop
            if row_idx == 1:
                for (ts_f, firms_f) in [sd for sd in seeds_data if sd[0] is not None]:
                    first_drop = next(
                        (t for t, f in zip(ts_f, firms_f) if f < 5), None)
                    if first_drop is not None:
                        ax.axvline(first_drop, color=COLOR_SHOCK, lw=0.8,
                                   ls='--', alpha=0.6, zorder=3)

            # Shock line at t=25 (price row and orders row)
            if row_idx in (0, 2):
                ax.axvline(
                    SHOCK_T,
                    color=COLOR_SHOCK,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.8,
                    zorder=6,
                    label="Shock (c\u00d710)",
                )
                if row_idx == 0:
                    # Legend for shock line placed in price row only (avoid duplication)
                    ax.legend(
                        loc="upper right",
                        handlelength=1.5,
                        fontsize=9,
                        handles=[
                            plt.Line2D([0], [0], color=COLOR_SHOCK, linestyle="--",
                                       linewidth=1.4, label="Shock (c\u00d710)"),
                        ],
                    )

            # Price row extras
            if row_idx == 0:
                uc = unit_costs[col_idx]
                ax.axhline(uc, color=COLOR_COST_REF, linestyle=":", linewidth=1.2,
                           alpha=0.8, zorder=5, label=f"Unit cost {uc:.1f}")
                ax.axhspan(0, uc, color=COLOR_COST_REF, alpha=0.05, zorder=1)
                ax.set_title(col["label"])

            # Active firms row
            if row_idx == 1:
                ax.set_ylim(0, 5.5)
                ax.set_yticks([0, 1, 2, 3, 4, 5])

            # Orders row
            if row_idx == 2:
                ax.set_ylim(bottom=0)
                ax.set_xlabel("Day")

            # Row y-labels on leftmost column
            if col_idx == 0:
                ax.set_ylabel(ROW_LABELS[row_idx])

    for row_i in range(n_rows):
        for col_i in range(n_cols):
            axes[row_i][col_i].set_xlim(0, 100)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
