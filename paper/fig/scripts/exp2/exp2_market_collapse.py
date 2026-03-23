"""
Fig Exp2-F: Experiment 2 — Market Collapse Timestep

Heatmap: first timestep where filled_orders_count < num_honest_sellers,
for each (n_sybil, seed) pair. Hatched cells indicate no collapse observed.

Grid rows = n_sybil in {0,3,6,9,12} (y-axis)
Grid cols = seed in {8,16,64} (x-axis)

Usage:
    python exp2_market_collapse.py [--logs-dir logs/] [--good car] [--output ...] [--workers 8]
"""

import argparse
import concurrent.futures
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from exp2_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "lines.linewidth":   1.5,
    "lines.markersize":  5,
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "grid.color":        "gray",
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.8",
    "figure.dpi":        100,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":       False,
    "pdf.fonttype":      42,
})

SEEDS = [8, 16, 64]
N_SYBIL_VALUES = [0, 3, 6, 9, 12]
RHO_MIN = 0.3

COLORS_N_SYBIL = {
    0:  "#999999",
    3:  "#56B4E9",
    6:  "#E69F00",
    9:  "#009E73",
    12: "#D55E00",
}


# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------

def resolve_run_dir(logs_dir, n_sybil, seed):
    if n_sybil == 0:
        path = os.path.join(logs_dir, f"exp2_baseline_seed{seed}")
    else:
        path = os.path.join(logs_dir, f"exp2_sybil_{n_sybil}_rho{RHO_MIN}_seed{seed}")
    return path if os.path.isdir(path) else None


def collect_run_dirs(logs_dir):
    dirs = []
    for n in N_SYBIL_VALUES:
        for s in SEEDS:
            d = resolve_run_dir(logs_dir, n, s)
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


# ---------------------------------------------------------------------------
# Metric extractor
# ---------------------------------------------------------------------------

def get_collapse_timestep(run_dir, n_sybil):
    """Return first timestep where filled_orders_count < (12 - n_sybil), or None."""
    num_honest = 12 - n_sybil
    if num_honest <= 0:
        return None
    files = load_states(run_dir)
    if not files:
        return None
    for p in files:
        with open(p) as f:
            s = json.load(f)
        t = s.get("timestep", 0)
        filled = s.get("filled_orders_count", 0)
        if not isinstance(filled, (int, float)):
            filled = 0
        if filled < num_honest:
            return t
    return None  # never collapsed


def load_one_job(run_dir, n_sybil, seed):
    t = get_collapse_timestep(run_dir, n_sybil)
    return n_sybil, seed, t


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _luminance(hex_color):
    """Perceived luminance from a hex color string, range [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def make_figure(collapse_matrix):
    """
    collapse_matrix: np.ndarray shape (5, 3), rows=N_SYBIL_VALUES, cols=SEEDS.
    NaN where no collapse / missing data.
    """
    n_rows = len(N_SYBIL_VALUES)
    n_cols = len(SEEDS)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    fig.suptitle("Market Collapse Timestep", fontweight="bold", fontsize=10)

    # Disable the global grid for the heatmap axes
    ax.grid(False)

    valid_vals = collapse_matrix[~np.isnan(collapse_matrix)]
    vmax = float(np.nanmax(valid_vals)) if valid_vals.size > 0 else 40.0
    vmin = 0.0

    masked = np.ma.masked_invalid(collapse_matrix)
    cmap = plt.get_cmap("RdYlGn_r")
    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )

    # Hatch NaN cells and annotate all cells
    for row_i, n_sybil in enumerate(N_SYBIL_VALUES):
        for col_j, seed in enumerate(SEEDS):
            val = collapse_matrix[row_i, col_j]
            if np.isnan(val):
                rect = mpatches.FancyBboxPatch(
                    (col_j - 0.5, row_i - 0.5), 1.0, 1.0,
                    boxstyle="square,pad=0",
                    linewidth=0,
                    facecolor="#cccccc",
                    hatch="///",
                    edgecolor="#888888",
                    zorder=5,
                )
                ax.add_patch(rect)
                ax.text(
                    col_j, row_i, "\u2014",
                    ha="center", va="center",
                    fontsize=9, color="#555555",
                    zorder=6,
                )
            else:
                # Choose text color by luminance of the cell's colormap color
                norm_val = (val - vmin) / max(vmax - vmin, 1e-9)
                cell_rgba = cmap(norm_val)
                cell_hex = "#{:02x}{:02x}{:02x}".format(
                    int(cell_rgba[0] * 255),
                    int(cell_rgba[1] * 255),
                    int(cell_rgba[2] * 255),
                )
                text_color = "white" if _luminance(cell_hex) < 0.5 else "black"
                ax.text(
                    col_j, row_i, f"t={int(val)}",
                    ha="center", va="center",
                    fontsize=9, color=text_color,
                    fontweight="bold", zorder=6,
                )

    # Reference lines between rows
    for y in [0.5, 1.5, 2.5, 3.5]:
        ax.axhline(y, color="#555555", lw=0.8, ls="--", alpha=0.6, zorder=7)

    # Axes labels and ticks
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(n) for n in N_SYBIL_VALUES])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Sybil cluster size")
    ax.set_title("First step where filled orders < honest sellers")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Collapse timestep", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp2 Fig F: Market Collapse Timestep")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="car")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "exp2", "exp2_market_collapse.pdf"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Ignore cache and rebuild from scratch")
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_market_collapse", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        raw = load_cache_data(cache_path)
        # Deserialize: list-of-lists with None -> np.nan
        raw_matrix = raw["collapse"]
        collapse_matrix = np.array(
            [[np.nan if v is None else float(v) for v in row] for row in raw_matrix]
        )
    else:
        # Build jobs: all (n_sybil, seed) pairs
        jobs = []
        for n in N_SYBIL_VALUES:
            for seed in SEEDS:
                run_dir = resolve_run_dir(args.logs_dir, n, seed)
                if run_dir:
                    jobs.append((n, seed, run_dir))
                else:
                    print(f"  Missing: n_sybil={n}, seed={seed}", flush=True)

        print(f"Loading {len(jobs)} runs ...", flush=True)

        # collapse_results[(n_sybil, seed)] = timestep or None
        collapse_results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {
                ex.submit(load_one_job, run_dir, n, seed): (n, seed)
                for n, seed, run_dir in jobs
            }
            done = 0
            total = len(jobs)
            for future in concurrent.futures.as_completed(future_map):
                n, seed = future_map[future]
                done += 1
                _n, _seed, t = future.result()
                collapse_results[(n, seed)] = t
                if t is not None:
                    print(
                        f"  [{done}/{total}] n_sybil={n} seed={seed} — t={t}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{done}/{total}] n_sybil={n} seed={seed} — no collapse",
                        flush=True,
                    )

        # Build 5×3 matrix
        collapse_matrix = np.full((len(N_SYBIL_VALUES), len(SEEDS)), np.nan)
        for row_i, n in enumerate(N_SYBIL_VALUES):
            for col_j, seed in enumerate(SEEDS):
                t = collapse_results.get((n, seed))
                if t is not None:
                    collapse_matrix[row_i, col_j] = float(t)

        # Serialize: NaN -> None for JSON
        raw_matrix = [
            [None if np.isnan(v) else v for v in row]
            for row in collapse_matrix.tolist()
        ]
        cache_data = {"collapse": raw_matrix}
        save_cache(cache_path, cache_data, args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

    fig = make_figure(collapse_matrix)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
