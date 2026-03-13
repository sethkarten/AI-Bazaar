"""
Fig: Experiment 1 Survival — side-by-side heatmaps of:
  (A) mean timesteps until simulation ended
  (B) mean number of firms in business at end of run

Grid: dlc ∈ {1, 3, 5}  ×  n_stab ∈ {0, 1, 2, 4, 5}
  n_stab=0: baseline (no stabilizing firm), exists only for dlc=3 seed=8 → "exp1_baseline"
  n_stab=5: stab baseline (all 5 firms stabilizing), exists only for dlc=3 seed=8 → "exp1_stab_baseline"
  All others: "exp1_stab_{n_stab}_dlc{dlc}_seed{seed}", averaged over seeds 8, 16, 64.

Survival = last timestep recorded.  365 → survived full horizon.
Firms at end = count of firms with in_business == True in final state file.
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

# Whether a (n_stab, dlc) cell is a "special case" with only one seed available
_SPECIAL_SINGLE_SEED = {(0, 3), (5, 3)}


def resolve_run_dir(logs_dir, dlc, n_stab, seed):
    """Return run directory path for given config; None if doesn't exist."""
    if n_stab == 0:
        if dlc == 3 and seed == 8:
            path = os.path.join(logs_dir, "exp1_baseline")
            return path if os.path.isdir(path) else None
        return None
    if n_stab == 5:
        if dlc == 3 and seed == 8:
            path = os.path.join(logs_dir, "exp1_stab_baseline")
            return path if os.path.isdir(path) else None
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


def compute_survival(run_dir):
    """Return the last timestep recorded in the run (int), or None if no data."""
    files = load_states(run_dir)
    if not files:
        return None
    last_file = files[-1]
    try:
        with open(last_file) as f:
            state = json.load(f)
        ts = state.get("timestep")
        if isinstance(ts, (int, float)):
            return int(ts)
        # Fallback: infer from filename digit
        digits = "".join(filter(str.isdigit, os.path.basename(last_file)))
        return int(digits) if digits else None
    except Exception:
        return None


def compute_firms_at_end(run_dir):
    """
    Return the count of firms with in_business == True in the last valid state
    file, or None if no data.
    """
    files = load_states(run_dir)
    if not files:
        return None
    last_file = files[-1]
    try:
        with open(last_file) as f:
            state = json.load(f)
        firms = state.get("firms", [])
        count = sum(1 for firm in firms if firm.get("in_business", False))
        return count
    except Exception:
        return None


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


def _make_annotation(vals):
    """Return annotation string for a list of numeric values."""
    vals_arr = np.array(vals, dtype=float)
    mean_v = float(np.mean(vals_arr))
    lo, hi = float(np.min(vals_arr)), float(np.max(vals_arr))
    n_seeds = len(vals)
    if n_seeds == 1:
        return mean_v, f"{mean_v:.1f}\n(1 seed)"
    range_str = f"[{lo:.0f}–{hi:.0f}]" if lo != hi else f"[{lo:.0f}]"
    return mean_v, f"{mean_v:.1f}\n{range_str}"


def build_grid(logs_dir, workers=8):
    """
    Returns:
      grid             : 2D np.ndarray shape (n_stab, dlc), mean survival timestep; NaN where missing
      annotations      : 2D list of annotation strings (or None)
      grid_firms       : 2D np.ndarray shape (n_stab, dlc), mean firms in business at end; NaN where missing
      annotations_firms: 2D list of annotation strings (or None)
      available        : 2D bool array, True where at least one seed has data
      single_seed      : 2D bool array, True where exactly one seed contributed
    """
    n_row = len(N_STAB_VALUES)
    n_col = len(DLC_VALUES)
    grid             = np.full((n_row, n_col), np.nan)
    annotations      = [[None] * n_col for _ in range(n_row)]
    grid_firms       = np.full((n_row, n_col), np.nan)
    annotations_firms = [[None] * n_col for _ in range(n_row)]
    available        = np.zeros((n_row, n_col), dtype=bool)
    single_seed      = np.zeros((n_row, n_col), dtype=bool)

    jobs = []
    for i, n_stab in enumerate(N_STAB_VALUES):
        for j, dlc in enumerate(DLC_VALUES):
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed)
                if run_dir is not None:
                    jobs.append((i, j, n_stab, dlc, seed, run_dir))

    total = len(jobs)
    print(f"Loading {total} runs...", flush=True)

    # Accumulate per-cell seed values for both metrics
    cell_survival = {}   # (i, j) -> [survival_timestep]
    cell_firms    = {}   # (i, j) -> [firms_count]

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
            if firms_count is not None:
                cell_firms.setdefault((i, j), []).append(firms_count)

    # Build survival grid
    for (i, j), vals in cell_survival.items():
        mean_v, ann = _make_annotation(vals)
        grid[i, j]        = mean_v
        annotations[i][j] = ann
        available[i, j]   = True
        if len(vals) == 1:
            single_seed[i, j] = True

    # Build firms-at-end grid (available/single_seed already set from survival pass)
    for (i, j), vals in cell_firms.items():
        mean_v, ann = _make_annotation(vals)
        grid_firms[i, j]         = mean_v
        annotations_firms[i][j]  = ann

    return grid, annotations, grid_firms, annotations_firms, available, single_seed


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


def _draw_heatmap(ax, fig, data, annotations, available, cmap, vmin, vmax, cbar_label):
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

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

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

            lines     = ann.split("\n")
            main_line = lines[0]
            sub_line  = lines[1] if len(lines) > 1 else ""

            ax.text(
                j, i - 0.12, main_line,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=txt_color,
                zorder=10,
            )
            if sub_line:
                ax.text(
                    j, i + 0.22, sub_line,
                    ha="center", va="center",
                    fontsize=7, color=txt_color,
                    zorder=10,
                )

    ax.set_xticks(range(len(DLC_VALUES)))
    ax.set_xticklabels([str(d) for d in DLC_VALUES])
    ax.set_yticks(range(len(N_STAB_VALUES)))
    ax.set_yticklabels([f"$k$={n}" for n in N_STAB_VALUES])
    ax.set_xlabel("Consumer discovery limit (dlc)")
    ax.set_ylabel("Stabilizing firms ($k$)")

    return im


def make_figure(grid, annotations, grid_firms, annotations_firms, available, single_seed):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.5), constrained_layout=True)

    ax_left, ax_right = axes

    # ── Left panel: survival duration ─────────────────────────────────────────
    _draw_heatmap(
        ax_left, fig, grid, annotations, available,
        cmap="YlGn", vmin=0, vmax=MAX_TIMESTEPS,
        cbar_label="Timesteps until collapse (max 365)",
    )
    ax_left.set_title("(A) Survival Duration")

    # ── Right panel: firms in business at end ──────────────────────────────────
    _draw_heatmap(
        ax_right, fig, grid_firms, annotations_firms, available,
        cmap="Blues", vmin=0, vmax=N_FIRMS_TOTAL,
        cbar_label="Firms in business at end",
    )
    ax_right.set_title("(B) Firms Surviving at End")

    # ── Shared suptitle ────────────────────────────────────────────────────────
    fig.suptitle("Experiment 1: Market Survival", fontweight="bold", fontsize=11)

    # ── Shared legend (placed below both panels) ───────────────────────────────
    hatch_patch = mpatches.Patch(
        facecolor="#cccccc", hatch="///", edgecolor="#888888",
        label="No data",
    )
    single_patch = mpatches.Patch(
        facecolor="white", edgecolor="#555555", linewidth=0.8,
        label="(1 seed) = single-seed run",
    )
    fig.legend(
        handles=[hatch_patch, single_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        fontsize=7,
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
    args = parser.parse_args()

    print(f"Loading runs from: {args.logs_dir}")
    grid, annotations, grid_firms, annotations_firms, available, single_seed = build_grid(
        args.logs_dir, workers=args.workers
    )

    n_available = int(available.sum())
    print(f"Cells with data: {n_available} / {len(N_STAB_VALUES) * len(DLC_VALUES)}")
    if n_available > 0:
        print(f"Survival range: {np.nanmin(grid):.0f} – {np.nanmax(grid):.0f} timesteps")
        if not np.all(np.isnan(grid_firms)):
            print(f"Firms-at-end range: {np.nanmin(grid_firms):.1f} – {np.nanmax(grid_firms):.1f}")

    fig = make_figure(grid, annotations, grid_firms, annotations_firms, available, single_seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
