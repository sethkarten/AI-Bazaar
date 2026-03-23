"""
Fig: Experiment 1 Survival — side-by-side heatmaps of:
  (A) Collapse probability (fraction of seeds where any firm went bankrupt)
  (B) mean number of firms in business at end of run

Grid: dlc ∈ {1, 3, 5}  ×  n_stab ∈ {0, 1, 2, 4, 5}
  n_stab=0: baseline (no stabilizing firm), exists only for dlc=3 seed=8 → "exp1_baseline"
  All others: "exp1_stab_{n_stab}_dlc{dlc}_seed{seed}", averaged over seeds 8, 16, 64.

Panel A: annotated as "n_collapsed/n_seeds"; red=high collapse probability.
Panel B: per-seed dots overlaid (green=all firms survive, red=any bankrupt).
Missing cells rendered as hatched gray.
Single-seed special cases annotated with "(1 seed)".

Usage:
    python exp1_survival.py [--logs-dir logs/] [--output ...]
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
MAX_TIMESTEPS = 365
N_FIRMS_TOTAL = 5


def collect_run_dirs(logs_dir, model=""):
    dirs = []
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if d:
                    dirs.append(d)
    return dirs


def _serialize(grid_collapse, annotations_collapse, grid_firms, annotations_firms,
               per_seed_firms, available, single_seed):
    psd_ser = {f"{i},{j}": vals for (i, j), vals in per_seed_firms.items()}
    return {
        "grid_collapse":        grid_collapse.tolist(),
        "annotations_collapse": annotations_collapse,
        "grid_firms":           grid_firms.tolist(),
        "annotations_firms":    annotations_firms,
        "per_seed_firms":       psd_ser,
        "available":            available.tolist(),
        "single_seed":          single_seed.tolist(),
    }


def _deserialize(data):
    psd_raw = data.get("per_seed_firms", {})
    per_seed_firms = {
        (int(k.split(",")[0]), int(k.split(",")[1])): vals
        for k, vals in psd_raw.items()
    }
    return (
        np.array(data["grid_collapse"]),
        data["annotations_collapse"],
        np.array(data["grid_firms"]),
        data["annotations_firms"],
        per_seed_firms,
        np.array(data["available"], dtype=bool),
        np.array(data["single_seed"], dtype=bool),
    )


def resolve_run_dir(logs_dir, dlc, n_stab, seed, model=""):
    """Return run directory path for given config; None if doesn't exist."""
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


def _load_both(run_dir):
    """Load survival and firms-at-end for a run directory in a single pass."""
    files = load_states(run_dir)
    if not files:
        return None, None
    last_file = files[-1]
    survival = None
    firms_count = None
    try:
        with open(last_file) as f:
            state = json.load(f)
        ts = state.get("timestep")
        if isinstance(ts, (int, float)):
            survival = int(ts)
        else:
            digits = "".join(filter(str.isdigit, os.path.basename(last_file)))
            survival = int(digits) if digits else None
        firms = state.get("firms", [])
        firms_count = sum(1 for firm in firms if firm.get("in_business", False))
    except Exception:
        pass
    return survival, firms_count


def build_grid(logs_dir, workers=8, model=""):
    """
    Returns:
      grid_collapse        : 2D np.ndarray, collapse probability per cell; NaN where missing
      annotations_collapse : 2D list of "n_collapsed/n_seeds" strings (or None)
      grid_firms           : 2D np.ndarray, mean firms in business at end; NaN where missing
      annotations_firms    : 2D list of annotation strings (or None)
      per_seed_firms       : dict {(i, j): [firms_count_seed0, firms_count_seed1, ...]}
      available            : 2D bool array, True where at least one seed has data
      single_seed          : 2D bool array, True where exactly one seed contributed
    """
    n_row = len(N_STAB_VALUES)
    n_col = len(DLC_VALUES)
    grid_collapse        = np.full((n_row, n_col), np.nan)
    annotations_collapse = [[None] * n_col for _ in range(n_row)]
    grid_firms           = np.full((n_row, n_col), np.nan)
    annotations_firms    = [[None] * n_col for _ in range(n_row)]
    per_seed_firms       = {}
    available            = np.zeros((n_row, n_col), dtype=bool)
    single_seed          = np.zeros((n_row, n_col), dtype=bool)

    jobs = []
    for i, n_stab in enumerate(N_STAB_VALUES):
        for j, dlc in enumerate(DLC_VALUES):
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if run_dir is not None:
                    jobs.append((i, j, n_stab, dlc, seed, run_dir))

    total = len(jobs)
    print(f"Loading {total} runs...", flush=True)

    cell_survival = {}   # (i, j) -> [survival_timestep]
    cell_firms    = {}   # (i, j) -> [firms_count or None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_job = {
            ex.submit(_load_both, run_dir): (i, j, n_stab, dlc, seed, run_dir)
            for i, j, n_stab, dlc, seed, run_dir in jobs
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_job):
            i, j, n_stab, dlc, seed, run_dir = future_to_job[future]
            done += 1
            ts, firms_count = future.result()
            label  = f"stab={n_stab} dlc={dlc} seed={seed}"
            status = f"t={ts} firms={firms_count}" if ts is not None else "empty"
            print(f"  [{done}/{total}] {label} — {status}", flush=True)
            if ts is not None:
                cell_survival.setdefault((i, j), []).append(ts)
                cell_firms.setdefault((i, j), []).append(firms_count)

    for (i, j), ts_vals in cell_survival.items():
        available[i, j] = True
        if len(ts_vals) == 1:
            single_seed[i, j] = True

    for (i, j), firms_vals in cell_firms.items():
        n_seeds     = len(firms_vals)
        n_collapsed = sum(1 for f in firms_vals if f is not None and f < N_FIRMS_TOTAL)
        prob        = n_collapsed / n_seeds if n_seeds > 0 else np.nan
        grid_collapse[i, j] = prob
        if n_seeds > 1 and not np.isnan(prob):
            se_collapse = float(np.sqrt(max(prob * (1 - prob) / n_seeds, 0)))
            annotations_collapse[i][j] = f"{round(prob * 100):.0f}%\n±{se_collapse:.2f}"
        else:
            annotations_collapse[i][j] = f"{round(prob * 100):.0f}%"

        valid_firms = [f for f in firms_vals if f is not None]
        if valid_firms:
            grid_firms[i, j] = float(np.mean(valid_firms))
            if len(valid_firms) > 1:
                se_firms = float(np.std(valid_firms, ddof=1) / np.sqrt(len(valid_firms)))
                annotations_firms[i][j] = f"{grid_firms[i, j]:.1f}\n±{se_firms:.1f}"
            else:
                annotations_firms[i][j] = f"{grid_firms[i, j]:.1f}"
        per_seed_firms[(i, j)] = firms_vals

    return (grid_collapse, annotations_collapse, grid_firms, annotations_firms,
            per_seed_firms, available, single_seed)


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


def draw_seed_dots_firms(ax, col, row, seed_firms_vals):
    """Overlay dots: green if all firms survive, red if any bankrupt."""
    n = len(seed_firms_vals)
    if n == 0:
        return
    if n == 1:
        xs = [col]
    elif n == 2:
        xs = [col - 0.18, col + 0.18]
    else:
        xs = [col - 0.25, col, col + 0.25]
    y = row + 0.28
    for x, fv in zip(xs, seed_firms_vals):
        if fv is None:
            continue
        dot_color = "#009E73" if fv >= N_FIRMS_TOTAL else "#CC0000"
        ax.scatter([x], [y], s=10, c=[dot_color],
                   edgecolors="white", linewidths=0.5, zorder=8, clip_on=True)


def _draw_heatmap(ax, fig, data, annotations, available, cmap, vmin, vmax, cbar_label,
                  per_seed_dot_data=None, dot_fn=None):
    """
    Draw a single heatmap panel onto ax.  Returns the AxesImage for the colorbar.
    """
    display = np.ma.masked_invalid(data)
    im = ax.imshow(
        display,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # Hatch missing cells
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            if not available[i, j]:
                draw_hatch_cell(ax, j, i)

    # Cell annotations
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            ann = annotations[i][j]
            if ann is None:
                continue
            val      = data[i, j]
            norm_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            norm_val = float(np.clip(norm_val, 0.0, 1.0))
            rgba     = plt.get_cmap(cmap)(norm_val)
            lum      = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "white" if lum < 0.5 else "black"

            ax.text(
                j, i, ann,
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=txt_color,
                zorder=10,
            )

    # Per-seed dots
    if per_seed_dot_data is not None and dot_fn is not None:
        for i in range(len(N_STAB_VALUES)):
            for j in range(len(DLC_VALUES)):
                if (i, j) in per_seed_dot_data:
                    dot_fn(ax, j, i, per_seed_dot_data[(i, j)])

    ax.set_xticks(range(len(DLC_VALUES)))
    ax.set_xticklabels([str(d) for d in DLC_VALUES])
    ax.set_yticks(range(len(N_STAB_VALUES)))
    ax.set_yticklabels([f"$k$={n}" for n in N_STAB_VALUES])
    ax.set_xlabel("Consumer discovery limit (dlc)")
    ax.set_ylabel("Stabilizing firms ($k$)")

    return im


def make_figure(grid_collapse, annotations_collapse, grid_firms, annotations_firms,
                per_seed_firms, available, single_seed):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.5), constrained_layout=True)

    ax_left, ax_right = axes

    # ── Left panel: collapse probability ──────────────────────────────────────
    _draw_heatmap(
        ax_left, fig, grid_collapse, annotations_collapse, available,
        cmap="Reds", vmin=0, vmax=1,
        cbar_label="Collapse probability",
    )
    ax_left.set_title("(A) Collapse Probability")

    # ── Right panel: firms in business at end ──────────────────────────────────
    _draw_heatmap(
        ax_right, fig, grid_firms, annotations_firms, available,
        cmap="Blues", vmin=0, vmax=N_FIRMS_TOTAL,
        cbar_label="Firms in business at end",
    )
    ax_right.set_title("(B) Firms Surviving at End")

    # ── Shared suptitle ────────────────────────────────────────────────────────
    fig.suptitle("Experiment 1: Market Survival", fontweight="bold", fontsize=13)

    # ── Shared legend (placed below both panels) ───────────────────────────────
    hatch_patch = mpatches.Patch(
        facecolor="#cccccc", hatch="///", edgecolor="#888888",
        label="No data",
    )
    fig.legend(
        handles=[hatch_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=1,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Fig: Exp1 Market Survival Heatmaps (1x2)")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food", help="Good name (unused; accepted for compatibility with exp1_run_all.py)")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "exp1", "exp1_survival.pdf"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp1_survival", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir, args.model)

    if is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        cached = load_cache_data(cache_path)
        if "grid_collapse" in cached:
            print(f"Using cached data: {cache_path}", flush=True)
            (grid_collapse, annotations_collapse, grid_firms, annotations_firms,
             per_seed_firms, available, single_seed) = _deserialize(cached)
        else:
            print("Cache missing grid_collapse, rebuilding...", flush=True)
            (grid_collapse, annotations_collapse, grid_firms, annotations_firms,
             per_seed_firms, available, single_seed) = build_grid(
                args.logs_dir, workers=args.workers, model=args.model)
            save_cache(cache_path,
                       _serialize(grid_collapse, annotations_collapse, grid_firms,
                                  annotations_firms, per_seed_firms, available, single_seed),
                       args.logs_dir, args.good)
    else:
        print(f"Loading runs from: {args.logs_dir}")
        (grid_collapse, annotations_collapse, grid_firms, annotations_firms,
         per_seed_firms, available, single_seed) = build_grid(
            args.logs_dir, workers=args.workers, model=args.model)
        save_cache(cache_path,
                   _serialize(grid_collapse, annotations_collapse, grid_firms,
                               annotations_firms, per_seed_firms, available, single_seed),
                   args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

    # Always recompute collapse annotations from the grid so stale caches don't show old format
    for i in range(len(N_STAB_VALUES)):
        for j in range(len(DLC_VALUES)):
            v = grid_collapse[i, j]
            if not np.isnan(v):
                annotations_collapse[i][j] = f"{round(v * 100):.0f}%"

    n_available = int(available.sum())
    print(f"Cells with data: {n_available} / {len(N_STAB_VALUES) * len(DLC_VALUES)}")
    if n_available > 0:
        if not np.all(np.isnan(grid_collapse)):
            print(f"Collapse probability range: {np.nanmin(grid_collapse):.2f} – {np.nanmax(grid_collapse):.2f}")
        if not np.all(np.isnan(grid_firms)):
            print(f"Firms-at-end range: {np.nanmin(grid_firms):.1f} – {np.nanmax(grid_firms):.1f}")

    fig = make_figure(grid_collapse, annotations_collapse, grid_firms, annotations_firms,
                      per_seed_firms, available, single_seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
