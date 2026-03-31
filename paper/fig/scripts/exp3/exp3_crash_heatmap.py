"""
Fig: Experiment 3a Heatmap — 1×4 metric heatmap over dlc × n_stab grid.

Metrics:
  A) Bankruptcy rate  (RdPu, higher = worse)
  B) Final avg price  (coolwarm diverging, centered at unit cost c)
  C) Total volume     (YlGn log-normalized)
  D) Price volatility (coolwarm, higher = worse)

Grid: dlc ∈ {3, 5}  ×  n_stab ∈ {1, 3, 5}
  All cells: "exp3a_{model}_stab{n_stab}_dlc{dlc}_seed{seed}", averaged over seeds 8, 16, 64.
  Note: exp3a uses stab{N} (no underscore), no baseline run (n_stab=0 does not exist).
Missing cells rendered as hatched NaN.
Per-seed dots overlaid on each cell (green=survived, red=collapsed).
Stability borders: black outline when bankruptcy_rate < 0.5 AND final_avg_price >= unit_cost.
Column headers include "shock @ t=25" annotation.

Usage:
    python exp3_crash_heatmap.py [--logs-dir logs/exp3a_gemini-3-flash-preview/] [--good food] [--output ...]
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
    "axes.grid":          False,
    "axes.axisbelow":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":        False,
})

DLC_VALUES    = [3]
N_STAB_VALUES = [3, 5]
SEEDS         = [8, 16, 64]

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# Cache helpers (inlined — no shared exp3 cache module yet)
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
        for f in glob.glob(os.path.join(d, "state_t*.json")):
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
        import time as _time
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
# Run directory resolution
# ---------------------------------------------------------------------------

def resolve_run_dir(logs_dir, dlc, n_stab, seed, model=""):
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
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if d:
                    dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
    active_firm_names = {f["name"] for f in last_state.get("firms", []) if f.get("in_business", False)}
    prices_at_last = []
    for f in last_state.get("firms", []):
        if f.get("in_business", False) and f.get("name") in active_firm_names:
            prices = f.get("prices") or {}
            p = prices.get(good)
            if isinstance(p, (int, float)) and p > 0:
                prices_at_last.append(p)
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
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if run_dir is not None:
                    jobs.append((i, j, n_stab, dlc, seed, run_dir))

    total = len(jobs)
    print(f"Loading {total} runs...", flush=True)

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
                grids[m][i, j] = mean_v
                if len(vals) > 1:
                    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    annotations[m][i][j] = f"{mean_v:.2f}\n+-{se:.2f}"
                else:
                    annotations[m][i][j] = f"{mean_v:.2f}"

    unit_cost = float(np.mean(unit_costs)) if unit_costs else 1.0
    return grids, annotations, available, unit_cost, per_seed_data


# ---------------------------------------------------------------------------
# Colormap / drawing helpers
# ---------------------------------------------------------------------------

class _BelowCostNorm(mcolors.Normalize):
    """
    Linear [vmin, vmax] -> [0, 1] (blue->red via coolwarm).
    Values below vmin (unit cost) are mapped to 1.0 (max red).
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp3a Crash Heatmap")
    parser.add_argument("--logs-dir", default="logs/exp3a_gemini-3-flash-preview/")
    parser.add_argument("--good", default="food")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "exp3", "exp3_crash_heatmap.pdf"
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

    cache_path = _get_cache_path(args.output, "exp3_crash_heatmap", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir, args.model)

    if _is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        cached = _load_cache_data(cache_path)
        if "per_seed_data" in cached:
            print(f"Using cached data: {cache_path}", flush=True)
            grids, annotations, available, unit_cost, per_seed_data = _deserialize(cached)
        else:
            print("Cache missing per_seed_data, rebuilding...", flush=True)
            grids, annotations, available, unit_cost, per_seed_data = build_grid(
                args.logs_dir, args.good, workers=args.workers, model=args.model)
            _save_cache(cache_path, _serialize(grids, annotations, available, unit_cost, per_seed_data),
                        args.logs_dir, args.good)
    else:
        print(f"Loading runs from: {args.logs_dir}")
        grids, annotations, available, unit_cost, per_seed_data = build_grid(
            args.logs_dir, args.good, workers=args.workers, model=args.model)
        _save_cache(cache_path, _serialize(grids, annotations, available, unit_cost, per_seed_data),
                    args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)
    print(f"Unit cost: {unit_cost:.3f}")

    # Volume normalization: relative to mean across all available cells (no baseline in exp3a)
    valid_vols = grids["total_volume"][~np.isnan(grids["total_volume"])]
    baseline_vol = float(np.mean(valid_vols)) if len(valid_vols) > 0 else 1.0
    if baseline_vol == 0:
        baseline_vol = 1.0

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
        ("(A) Bankruptcy Rate",       "bankruptcy_rate", "RdPu",    "regular"),
        ("(B) Final Avg Price / $c$", "final_avg_price", "coolwarm", "range"),
        ("(C) Total Market Volume",   "vol_norm",        "YlGn",    "lognorm"),
        ("(D) Price Volatility $s$",  "price_std",       "coolwarm", "regular"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.2), constrained_layout=True)
    axes_flat = axes

    for panel_idx, (ax, (title, metric_key, cmap_name, mode)) in enumerate(zip(axes_flat, panels)):
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

        if mode == "range":
            ablated_vals = grid[~np.isnan(grid)]
            vmax_plot = float(np.nanmax(ablated_vals)) if len(ablated_vals) > 0 else float(np.nanmax(valid_vals))
            if vmax_plot <= unit_cost:
                vmax_plot = unit_cost + 1.0
            norm = _BelowCostNorm(vmin=unit_cost, vmax=vmax_plot)
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
                vmax_plot = float(np.nanmax(valid_vals))
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
        if mode == "lognorm":
            fmt = mticker.ScalarFormatter()
            fmt.set_scientific(False)
            cb_ax.ax.yaxis.set_major_formatter(fmt)
            cb_ax.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        elif mode == "range":
            cb_ax.ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="upper"))

        # Hatch missing cells
        for i in range(len(N_STAB_VALUES)):
            for j in range(len(DLC_VALUES)):
                if not available[i, j]:
                    draw_hatch_cell(ax, j, i)

        # Cell annotations
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
                    j, i - 0.07, annots[i][j],
                    ha="center", va="center",
                    fontsize=9.0, linespacing=0.95, color=txt_color,
                    zorder=10,
                )

        # Per-seed dots
        for i in range(len(N_STAB_VALUES)):
            for j in range(len(DLC_VALUES)):
                if (i, j) in per_seed_data:
                    draw_seed_dots(ax, j, i, per_seed_data[(i, j)]["bankruptcy_rate"])

        # Stability borders
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
        # Column labels with shock annotation below
        ax.set_xticklabels([f"dlc={d}\nshock @ t=25" for d in DLC_VALUES], fontsize=9)
        ax.set_yticks(range(len(N_STAB_VALUES)))
        ax.set_yticklabels([f"$k$={n}" for n in N_STAB_VALUES])
        ax.set_xlabel("Consumer discovery limit (dlc)")
        if panel_idx == 0:
            ax.set_ylabel("Stabilizing firms ($k$)")
        ax.set_title(title)

    # Shared legend
    hatch_patch = mpatches.Patch(
        facecolor="#cccccc", hatch="///", edgecolor="#888888", label="No data")
    border_patch = mpatches.Rectangle(
        (0, 0), 1, 1, linewidth=2.5, edgecolor="black", facecolor="none",
        label="Stable zone (bankrupt <50%, price >= c)")
    fig.legend(handles=[hatch_patch, border_patch], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), fontsize=9)

    fig.suptitle("Experiment 3a: Cost Shock — Stabilizing Firm Ablation", fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
