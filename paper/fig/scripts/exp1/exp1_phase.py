"""
Fig: Experiment 1 Phase Diagram — stable vs. collapse regime.

Three panels:
  (A) Binary phase map: stable (green) vs. collapse (red).
      Stability criterion: bankruptcy_rate < 0.5 AND final_avg_price >= unit_cost.
      Contour line at stability boundary overlaid.
  (B) Stability margin heatmap: distance to boundary
      = min(0.5 - bankruptcy_rate, final_avg_price/unit_cost - 1).
      Positive = inside stable zone; negative = outside.
      Diverging colormap (RdYlGn) centered at 0.
  (C) k*(dlc) bar chart: minimum k (stabilizing firms) achieving stability per dlc value.

Grid: dlc ∈ {1, 3, 5}  ×  n_stab ∈ {0, 1, 2, 4, 5}.

Usage:
    python exp1_phase.py [--logs-dir logs/] [--good food] [--output ...]
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

DLC_VALUES    = [1, 3, 5]
N_STAB_VALUES = [0, 1, 2, 4, 5]
SEEDS         = [8, 16, 64]


def collect_run_dirs(logs_dir, model=""):
    dirs = []
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if d:
                    dirs.append(d)
    return dirs


def resolve_run_dir(logs_dir, dlc, n_stab, seed, model=""):
    if model:
        if n_stab == 0:
            if dlc == 3 and seed == 8:
                path = os.path.join(logs_dir, f"exp1_{model}_baseline")
                return path if os.path.isdir(path) else None
            return None
        path = os.path.join(logs_dir, f"exp1_{model}_stab_{n_stab}_dlc{dlc}_seed{seed}")
        return path if os.path.isdir(path) else None
    if n_stab == 0:
        if dlc == 3 and seed == 8:
            path = os.path.join(logs_dir, "exp1_baseline")
            return path if os.path.isdir(path) else None
        return None
    if n_stab == 5:
        path = os.path.join(logs_dir, f"exp1_stab_5_dlc{dlc}_seed{seed}")
        return path if os.path.isdir(path) else None
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


def compute_metrics(run_dir, good):
    """Returns dict with bankruptcy_rate, final_avg_price, unit_cost."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    states = db.states

    first_firms = len(states[0].get("firms", []))
    if first_firms == 0:
        return None

    firms_df = db.firms_in_business_over_time().sort_values("timestep")
    if firms_df.empty:
        return None
    bankruptcy_rate = 1.0 - int(firms_df.iloc[-1]["value"]) / first_firms

    last_state = states[-1]
    prices_at_last = [
        f["prices"].get(good)
        for f in last_state.get("firms", [])
        if f.get("in_business") and isinstance(f.get("prices", {}).get(good), (int, float))
        and f["prices"][good] > 0
    ]
    final_avg_price = float(np.mean(prices_at_last)) if prices_at_last else 0.0

    return {
        "bankruptcy_rate": bankruptcy_rate,
        "final_avg_price": final_avg_price,
        "unit_cost":       get_unit_cost(run_dir),
    }


def build_grid(logs_dir, good, workers=8, model=""):
    """Returns averaged grids for bankruptcy_rate and final_avg_price, plus available mask."""
    n_row = len(N_STAB_VALUES)
    n_col = len(DLC_VALUES)
    grid_br    = np.full((n_row, n_col), np.nan)
    grid_price = np.full((n_row, n_col), np.nan)
    available  = np.zeros((n_row, n_col), dtype=bool)
    unit_costs = []

    jobs = []
    for i, n_stab in enumerate(N_STAB_VALUES):
        for j, dlc in enumerate(DLC_VALUES):
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if run_dir:
                    jobs.append((i, j, n_stab, dlc, seed, run_dir))

    total = len(jobs)
    print(f"Loading {total} runs...", flush=True)

    cell_vals = {}  # (i, j) -> {br: [], price: [], uc: []}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_job = {
            ex.submit(compute_metrics, run_dir, good): (i, j, n_stab, dlc, seed, run_dir)
            for i, j, n_stab, dlc, seed, run_dir in jobs
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_job):
            i, j, n_stab, dlc, seed, run_dir = future_to_job[future]
            done += 1
            m = future.result()
            status = "ok" if m else "empty"
            print(f"  [{done}/{total}] stab={n_stab} dlc={dlc} seed={seed} — {status}", flush=True)
            if m:
                key = (i, j)
                if key not in cell_vals:
                    cell_vals[key] = {"br": [], "price": [], "uc": []}
                cell_vals[key]["br"].append(m["bankruptcy_rate"])
                cell_vals[key]["price"].append(m["final_avg_price"])
                cell_vals[key]["uc"].append(m["unit_cost"])
                unit_costs.append(m["unit_cost"])

    for (i, j), vals in cell_vals.items():
        available[i, j]  = True
        grid_br[i, j]    = float(np.mean(vals["br"]))
        grid_price[i, j] = float(np.mean(vals["price"]))

    unit_cost = float(np.mean(unit_costs)) if unit_costs else 1.0
    return grid_br, grid_price, available, unit_cost


def _serialize(grid_br, grid_price, available, unit_cost):
    return {
        "grid_br":    grid_br.tolist(),
        "grid_price": grid_price.tolist(),
        "available":  available.tolist(),
        "unit_cost":  unit_cost,
    }


def _deserialize(data):
    return (
        np.array(data["grid_br"]),
        np.array(data["grid_price"]),
        np.array(data["available"], dtype=bool),
        float(data["unit_cost"]),
    )


def draw_hatch_cell(ax, col_idx, row_idx):
    rect = mpatches.FancyBboxPatch(
        (col_idx - 0.5, row_idx - 0.5), 1.0, 1.0,
        boxstyle="square,pad=0", linewidth=0,
        facecolor="#cccccc", hatch="///", edgecolor="#888888", zorder=5,
    )
    ax.add_patch(rect)


def _setup_heatmap_axes(ax):
    ax.set_xticks(range(len(DLC_VALUES)))
    ax.set_xticklabels([str(d) for d in DLC_VALUES])
    ax.set_yticks(range(len(N_STAB_VALUES)))
    ax.set_yticklabels([f"$k$={n}" for n in N_STAB_VALUES])
    ax.set_xlabel("Discovery limit (dlc)")
    ax.set_ylabel("Stabilizing firms ($k$)")


def make_figure(grid_br, grid_price, available, unit_cost):
    # ── Stability classification ─────────────────────────────────────────────
    stable = np.zeros_like(grid_br, dtype=bool)
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if available[i, j]:
                stable[i, j] = (grid_br[i, j] < 0.5) and (grid_price[i, j] >= unit_cost)

    # ── Stability margin: min(0.5 - br, price/c - 1) ────────────────────────
    margin = np.full_like(grid_br, np.nan)
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if available[i, j]:
                br_dist    = 0.5 - grid_br[i, j]
                price_dist = grid_price[i, j] / unit_cost - 1.0
                margin[i, j] = min(br_dist, price_dist)

    # ── k*(dlc): minimum k achieving stability ───────────────────────────────
    k_star = []
    for j in range(len(DLC_VALUES)):
        found = None
        for i, k in enumerate(N_STAB_VALUES):
            if available[i, j] and stable[i, j]:
                found = k
                break
        k_star.append(found)

    # ── Figure layout: 3 panels ──────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3, figsize=(14.0, 5.5),
        gridspec_kw={"width_ratios": [1, 1, 0.9]},
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c = axes

    # ── Panel A: Binary phase map ────────────────────────────────────────────
    colors_a = np.full(grid_br.shape, np.nan)
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if available[i, j]:
                colors_a[i, j] = 1.0 if stable[i, j] else 0.0

    cmap_a = mcolors.ListedColormap(["#D73027", "#1A9641"])  # red=collapse, green=stable
    norm_a = mcolors.BoundaryNorm([0.0, 0.5, 1.0], cmap_a.N)
    display_a = np.ma.masked_invalid(colors_a)
    ax_a.imshow(display_a, cmap=cmap_a, norm=norm_a, aspect="auto", interpolation="nearest")

    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if not available[i, j]:
                draw_hatch_cell(ax_a, j, i)

    # Contour line at stability boundary
    stable_for_contour = np.where(np.isnan(colors_a), 0.0, colors_a)
    x_c = np.arange(len(DLC_VALUES))
    y_c = np.arange(len(N_STAB_VALUES))
    if stable_for_contour.min() < stable_for_contour.max():
        ax_a.contour(x_c, y_c, stable_for_contour, levels=[0.5],
                     colors="black", linewidths=2.5, zorder=10)

    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if available[i, j]:
                label = "stable" if stable[i, j] else "collapse"
                ax_a.text(j, i, label, ha="center", va="center",
                          fontsize=9, color="white", fontweight="bold", zorder=6)

    _setup_heatmap_axes(ax_a)
    ax_a.set_title("(A) Phase Map")
    stable_patch   = mpatches.Patch(facecolor="#1A9641", label="Stable")
    collapse_patch = mpatches.Patch(facecolor="#D73027", label="Collapse")
    ax_a.legend(handles=[stable_patch, collapse_patch], loc="upper right",
                fontsize=9, handlelength=1.2)

    # ── Panel B: Stability margin ────────────────────────────────────────────
    valid_margins = margin[~np.isnan(margin)]
    if len(valid_margins) > 0:
        abs_max = max(abs(float(np.nanmin(margin))), abs(float(np.nanmax(margin))), 0.1)
    else:
        abs_max = 1.0

    norm_b   = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
    cmap_b   = plt.get_cmap("RdYlGn")
    display_b = np.ma.masked_invalid(margin)
    im_b = ax_b.imshow(display_b, cmap=cmap_b, norm=norm_b, aspect="auto", interpolation="nearest")
    cb_b = fig.colorbar(im_b, ax=ax_b, shrink=0.85, pad=0.03)
    cb_b.set_label("Stability margin", fontsize=10)
    cb_b.ax.tick_params(labelsize=9)

    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if not available[i, j]:
                draw_hatch_cell(ax_b, j, i)

    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if available[i, j] and not np.isnan(margin[i, j]):
                val      = margin[i, j]
                norm_val = float(np.clip(norm_b(val), 0.0, 1.0))
                rgba     = cmap_b(norm_val)
                lum      = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_col  = "white" if lum < 0.5 else "black"
                ax_b.text(j, i, f"{val:+.2f}", ha="center", va="center",
                          fontsize=9, color=txt_col, zorder=10)

    _setup_heatmap_axes(ax_b)
    ax_b.set_title("(B) Stability Margin")

    # ── Panel C: k*(dlc) bar chart ───────────────────────────────────────────
    xs_c = np.arange(len(DLC_VALUES))
    ys_c = [k if k is not None else max(N_STAB_VALUES) + 1 for k in k_star]
    bar_colors = ["#1A9641" if k is not None else "#D73027" for k in k_star]

    ax_c.bar(xs_c, ys_c, color=bar_colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax_c.set_xticks(xs_c)
    ax_c.set_xticklabels([str(d) for d in DLC_VALUES])
    ax_c.set_yticks(N_STAB_VALUES)
    ax_c.set_yticklabels([str(k) for k in N_STAB_VALUES])
    ax_c.set_xlabel("Discovery limit (dlc)")
    ax_c.set_ylabel(r"Min stabilizing firms $k^*$")
    ax_c.set_title(r"(C) $k^*$(dlc)")
    ax_c.set_ylim(0, max(N_STAB_VALUES) + 0.8)
    ax_c.grid(True, axis="y", alpha=0.3)

    for xi, k in enumerate(k_star):
        label = str(k) if k is not None else "none"
        height = k if k is not None else max(N_STAB_VALUES) + 1
        ax_c.text(xi, height + 0.1, label,
                  ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotation explaining the criterion
    ax_c.text(0.5, -0.18,
              r"Stable: $b_r < 0.5$ and $\bar{p} \geq c$",
              transform=ax_c.transAxes, ha="center", va="top",
              fontsize=9, color="0.4", style="italic")

    fig.suptitle("Experiment 1: Stability Phase Diagram", fontweight="bold")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Fig: Exp1 Phase Diagram")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "exp1", "exp1_phase.pdf"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp1_phase", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir, args.model)

    if is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        cached = load_cache_data(cache_path)
        if "grid_br" in cached:
            print(f"Using cached data: {cache_path}", flush=True)
            grid_br, grid_price, available, unit_cost = _deserialize(cached)
        else:
            print("Cache missing grid_br, rebuilding...", flush=True)
            grid_br, grid_price, available, unit_cost = build_grid(
                args.logs_dir, args.good, workers=args.workers, model=args.model)
            save_cache(cache_path,
                       _serialize(grid_br, grid_price, available, unit_cost),
                       args.logs_dir, args.good)
    else:
        print(f"Loading runs from: {args.logs_dir}")
        grid_br, grid_price, available, unit_cost = build_grid(
            args.logs_dir, args.good, workers=args.workers, model=args.model)
        save_cache(cache_path,
                   _serialize(grid_br, grid_price, available, unit_cost),
                   args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

    print(f"Unit cost: {unit_cost:.3f}")

    fig = make_figure(grid_br, grid_price, available, unit_cost)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
