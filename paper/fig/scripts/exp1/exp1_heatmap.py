"""
Fig 1: Experiment 1 Heatmap — 2×2 metric heatmap over dlc × n_stab grid.

Metrics:
  A) Bankruptcy rate  (YlOrRd, higher = worse)
  B) Final avg price  (RdBu diverging, centered at unit cost c)
  C) Total volume     (YlGn, higher = better)
  D) Price volatility (YlOrRd, higher = worse)

Grid: dlc ∈ {1, 3, 5}  ×  n_stab ∈ {0, 1, 2, 4, 5}
  n_stab=0: baseline (no stabilizing firm), exists only for dlc=3 seed=8 → "exp1_baseline"
  All other cells: "exp1_stab_{n_stab}_dlc{dlc}_seed{seed}", averaged over seeds 8, 16, 64.
Missing cells rendered as hatched NaN.

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
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder

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
    NaN where data missing. Also returns global unit_cost and boolean available mask.
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

    for (i, j), seed_vals in cell_seed_vals.items():
        if seed_vals["bankruptcy_rate"]:
            available[i, j] = True
            for m in metric_names:
                vals   = seed_vals[m]
                mean_v = float(np.mean(vals))
                lo, hi = float(np.min(vals)), float(np.max(vals))
                grids[m][i, j]       = mean_v
                annotations[m][i][j] = f"{mean_v:.2f}\n[{lo:.2f}–{hi:.2f}]"

    unit_cost = float(np.mean(unit_costs)) if unit_costs else 1.0
    return grids, annotations, available, unit_cost


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

    print(f"Loading runs from: {args.logs_dir}")
    grids, annotations, available, unit_cost = build_grid(args.logs_dir, args.good, workers=args.workers)
    print(f"Unit cost: {unit_cost:.3f}")

    # Panel config: (title, metric_key, colormap, diverging)
    panels = [
        ("(A) Bankruptcy Rate",        "bankruptcy_rate", "YlOrRd", False),
        ("(B) Final Avg Price / $c$",  "final_avg_price", "RdBu",   True),
        ("(C) Total Market Volume",    "total_volume",    "YlGn",   False),
        ("(D) Price Volatility $σ$",   "price_std",       "YlOrRd", False),
    ]

    # 5 n_stab rows × 3 dlc cols — use full two-column width, taller to fit extra row
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (title, metric, cmap_name, diverging) in zip(axes_flat, panels):
        grid   = grids[metric]
        annots = annotations[metric]
        valid_vals = grid[~np.isnan(grid)]

        if len(valid_vals) == 0:
            ax.set_title(title)
            ax.axis("off")
            continue

        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
        cmap = plt.get_cmap(cmap_name)

        if diverging:
            center  = unit_cost
            abs_max = max(abs(vmin - center), abs(vmax - center)) or 1.0
            vmin_plot, vmax_plot = center - abs_max, center + abs_max
        else:
            vmin_plot = vmin
            vmax_plot = vmax if vmax > vmin else vmin + 1

        display = np.ma.masked_invalid(grid)
        im = ax.imshow(
            display,
            cmap=cmap,
            vmin=vmin_plot,
            vmax=vmax_plot,
            aspect="auto",
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

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
                norm_val = (val - vmin_plot) / (vmax_plot - vmin_plot) if vmax_plot != vmin_plot else 0.5
                rgba     = cmap(norm_val)
                lum      = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if lum < 0.5 else "black"
                if metric == "final_avg_price" and val < unit_cost:
                    txt_color = "#cc0000"
                ax.text(
                    j, i, annots[i][j],
                    ha="center", va="center",
                    fontsize=7, color=txt_color,
                    zorder=10,
                )

        # Bold outlines on best-case cells: (dlc=1, n_stab=4) and (dlc=5, n_stab=4)
        for dlc_k, nstab_k in [(1, 4), (5, 4)]:
            if dlc_k in DLC_VALUES and nstab_k in N_STAB_VALUES:
                j_k = DLC_VALUES.index(dlc_k)
                i_k = N_STAB_VALUES.index(nstab_k)
                rect = mpatches.Rectangle(
                    (j_k - 0.5, i_k - 0.5), 1.0, 1.0,
                    linewidth=2.0, edgecolor="black", facecolor="none", zorder=15,
                )
                ax.add_patch(rect)

        ax.set_xticks(range(len(DLC_VALUES)))
        ax.set_xticklabels([f"dlc={d}" for d in DLC_VALUES])
        ax.set_yticks(range(len(N_STAB_VALUES)))
        ax.set_yticklabels([f"$k$={n}" for n in N_STAB_VALUES])
        ax.set_xlabel("Consumer discovery limit (dlc)")
        ax.set_ylabel("Stabilizing firms ($k$)")
        ax.set_title(title)

    # Shared legend for hatch pattern
    hatch_patch = mpatches.Patch(
        facecolor="#cccccc", hatch="///", edgecolor="#888888",
        label="No data (single-seed special runs only)",
    )
    fig.legend(
        handles=[hatch_patch],
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle("Experiment 1: Stabilizing Firm Ablation", fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
