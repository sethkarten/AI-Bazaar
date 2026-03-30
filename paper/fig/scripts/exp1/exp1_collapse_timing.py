"""
Fig: Experiment 1 Collapse Timing.

Single heatmap showing median first-collapse timestep across the (n_stab × dlc) grid.
  - Stable cells (no collapse in any seed): shown in solid green.
  - Collapsed cells: colored by median first-collapse timestep (early = darker red).
  - Per-seed dots overlaid inside each cell (red = collapsed, green = survived).

first-collapse timestep = first t where any firm has in_business = False.

Usage:
    python exp1_collapse_timing.py [--logs-dir logs/] [--good food] [--output ...]
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
from exp1_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data
from exp1_paths import DLC_VALUES, N_STAB_VALUES, SEEDS, resolve_run_dir

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

MAX_TIMESTEPS = 365


def collect_run_dirs(logs_dir, model=""):
    dirs = []
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if d:
                    dirs.append(d)
    return dirs


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


def find_first_collapse(run_dir):
    """
    Returns the timestep of the first bankruptcy event, or None if no collapse.
    Scans state files in order; returns t when active firms < initial firms.
    """
    files = load_states(run_dir)
    if not files:
        return None

    # Get initial firm count from first state
    try:
        with open(files[0]) as f:
            first_state = json.load(f)
        initial_firms = len(first_state.get("firms", []))
    except Exception:
        return None

    if initial_firms == 0:
        return None

    # Scan subsequent states for first bankruptcy
    for p in files[1:]:
        try:
            with open(p) as f:
                state = json.load(f)
            firms = state.get("firms", [])
            active = sum(1 for firm in firms if firm.get("in_business", False))
            if active < initial_firms:
                ts = state.get("timestep")
                if ts is None:
                    digits = "".join(filter(str.isdigit, os.path.basename(p)))
                    ts = int(digits) if digits else None
                return ts
        except Exception:
            pass
    return None


def build_grid(logs_dir, workers=8, model=""):
    """
    Returns:
      grid_median  : 2D array of median first-collapse timestep (NaN if no collapses)
      grid_se      : 2D array of SE of collapse timestep across seeds (NaN if <2 collapses)
      per_seed     : dict {(i, j): [collapse_ts or None per seed]}
      available    : bool array, True where at least one seed has data
    """
    n_row = len(N_STAB_VALUES)
    n_col = len(DLC_VALUES)
    grid_median = np.full((n_row, n_col), np.nan)
    grid_se     = np.full((n_row, n_col), np.nan)
    per_seed    = {}
    available   = np.zeros((n_row, n_col), dtype=bool)

    jobs = []
    for i, n_stab in enumerate(N_STAB_VALUES):
        for j, dlc in enumerate(DLC_VALUES):
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if run_dir:
                    jobs.append((i, j, n_stab, dlc, seed, run_dir))

    total = len(jobs)
    print(f"Loading {total} runs...", flush=True)

    cell_ts = {}  # (i, j) -> [ts or None per seed]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_job = {
            ex.submit(find_first_collapse, run_dir): (i, j, n_stab, dlc, seed)
            for i, j, n_stab, dlc, seed, run_dir in jobs
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_job):
            i, j, n_stab, dlc, seed = future_to_job[future]
            done += 1
            ts = future.result()
            status = f"collapse@t={ts}" if ts is not None else "no collapse"
            print(f"  [{done}/{total}] stab={n_stab} dlc={dlc} seed={seed} — {status}", flush=True)
            cell_ts.setdefault((i, j), []).append(ts)

    for (i, j), ts_list in cell_ts.items():
        available[i, j] = True
        per_seed[(i, j)] = ts_list
        collapses = [t for t in ts_list if t is not None]
        if collapses:
            grid_median[i, j] = float(np.median(collapses))
            if len(collapses) > 1:
                grid_se[i, j] = float(np.std(collapses, ddof=1) / np.sqrt(len(collapses)))

    return grid_median, grid_se, per_seed, available


def _serialize(grid_median, grid_se, per_seed, available):
    ps_ser = {
        f"{i},{j}": [t if t is not None else -1 for t in ts]
        for (i, j), ts in per_seed.items()
    }
    return {
        "grid_median": grid_median.tolist(),
        "grid_se":     grid_se.tolist(),
        "per_seed":    ps_ser,
        "available":   available.tolist(),
    }


def _deserialize(data):
    ps_raw = data.get("per_seed", {})
    per_seed = {}
    for k, ts_list in ps_raw.items():
        i, j = int(k.split(",")[0]), int(k.split(",")[1])
        per_seed[(i, j)] = [t if t != -1 else None for t in ts_list]
    grid_se_raw = data.get("grid_se")
    grid_se = np.array(grid_se_raw) if grid_se_raw is not None else np.full_like(
        np.array(data["grid_median"]), np.nan)
    return (
        np.array(data["grid_median"]),
        grid_se,
        per_seed,
        np.array(data["available"], dtype=bool),
    )


def draw_hatch_cell(ax, col_idx, row_idx):
    rect = mpatches.FancyBboxPatch(
        (col_idx - 0.5, row_idx - 0.5), 1.0, 1.0,
        boxstyle="square,pad=0", linewidth=0,
        facecolor="#cccccc", hatch="///", edgecolor="#888888", zorder=5,
    )
    ax.add_patch(rect)


def draw_stable_cell(ax, col_idx, row_idx):
    """Solid green cell for no-collapse runs."""
    rect = mpatches.FancyBboxPatch(
        (col_idx - 0.5, row_idx - 0.5), 1.0, 1.0,
        boxstyle="square,pad=0", linewidth=0,
        facecolor="#1A9641", zorder=4,
    )
    ax.add_patch(rect)


def make_figure(grid_median, grid_se, per_seed, available):
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 6.5), constrained_layout=True)

    vmin = 0
    vmax = MAX_TIMESTEPS
    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap.set_bad(color="#e8e8e8")  # masked cells show as light gray

    # Draw heatmap (only collapsed cells will have values; stable = NaN = masked)
    display_plot = np.ma.masked_invalid(grid_median)
    im = ax.imshow(
        display_plot, cmap=cmap, norm=norm,
        aspect="auto", interpolation="nearest", zorder=2,
    )

    # Overlay stable (green) cells on top of masked areas
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if available[i, j] and np.isnan(grid_median[i, j]):
                draw_stable_cell(ax, j, i)

    # Hatch missing cells
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if not available[i, j]:
                draw_hatch_cell(ax, j, i)

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label("Median first-collapse timestep", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    # Annotate cells: median time or "stable"
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if not available[i, j]:
                continue
            if np.isnan(grid_median[i, j]):
                ax.text(j, i - 0.08, "stable", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white", zorder=10)
            else:
                val      = grid_median[i, j]
                norm_val = float(np.clip(norm(val), 0.0, 1.0))
                rgba     = cmap(norm_val)
                lum      = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_col  = "white" if lum < 0.5 else "black"
                se       = grid_se[i, j]
                label    = f"t={int(val)}\n±{se:.0f}" if not np.isnan(se) else f"t={int(val)}"
                ax.text(j, i - 0.08, label, ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=txt_col, zorder=10)

    # Per-seed dots showing individual seed outcomes
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if (i, j) not in per_seed:
                continue
            ts_list = per_seed[(i, j)]
            n = len(ts_list)
            if n == 1:
                xs = [j]
            elif n == 2:
                xs = [j - 0.20, j + 0.20]
            else:
                xs = [j - 0.22, j, j + 0.22]
            for s_idx, ts in enumerate(ts_list):
                dot_x = xs[s_idx] if s_idx < len(xs) else j
                dot_color = "#CC0000" if ts is not None else "#1A9641"
                ax.scatter([dot_x], [i + 0.30], s=13, c=[dot_color],
                           edgecolors="white", linewidths=0.4, zorder=12, clip_on=True)

    ax.set_xticks(range(len(DLC_VALUES)))
    ax.set_xticklabels([str(d) for d in DLC_VALUES])
    ax.set_yticks(range(len(N_STAB_VALUES)))
    ax.set_yticklabels([f"$k$={n}" for n in N_STAB_VALUES])
    ax.set_xlabel("Consumer discovery limit (dlc)")
    ax.set_ylabel("Stabilizing firms ($k$)")
    ax.set_title("Experiment 1: First Collapse Timing", fontweight="bold", fontsize=13)

    # Legend
    stable_patch = mpatches.Patch(facecolor="#1A9641", label="No collapse (stable)")
    hatch_patch  = mpatches.Patch(facecolor="#cccccc", hatch="///", edgecolor="#888888",
                                   label="No data")
    red_dot   = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#CC0000",
                            markersize=7, label="Seed collapsed")
    green_dot = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1A9641",
                            markersize=7, label="Seed survived")
    fig.legend(
        handles=[stable_patch, hatch_patch, red_dot, green_dot],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
        fontsize=9, frameon=True, framealpha=0.9, edgecolor="0.8",
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Fig: Exp1 Collapse Timing")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food",
                        help="Good name (unused; accepted for exp1_run_all.py compatibility)")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "exp1",
                             "exp1_collapse_timing.pdf"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp1_collapse_timing", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir, args.model)

    if is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        cached = load_cache_data(cache_path)
        if "grid_median" in cached:
            print(f"Using cached data: {cache_path}", flush=True)
            grid_median, grid_se, per_seed, available = _deserialize(cached)
        else:
            print("Cache missing grid_median, rebuilding...", flush=True)
            grid_median, grid_se, per_seed, available = build_grid(args.logs_dir, workers=args.workers, model=args.model)
            save_cache(cache_path,
                       _serialize(grid_median, grid_se, per_seed, available),
                       args.logs_dir, args.good)
    else:
        print(f"Loading runs from: {args.logs_dir}")
        grid_median, grid_se, per_seed, available = build_grid(args.logs_dir, workers=args.workers, model=args.model)
        save_cache(cache_path,
                   _serialize(grid_median, grid_se, per_seed, available),
                   args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

    fig = make_figure(grid_median, grid_se, per_seed, available)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
