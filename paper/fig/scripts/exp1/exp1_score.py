"""
Fig 4: Experiment 1 Multi-Dimensional Performance Score

Combines bankruptcy rate, final price deviation from unit cost, and price volatility
into a composite market health score and a multi-metric trade-off visualization.

Grid: n_stab ∈ {0, 1, 2, 4, 5} × dlc ∈ {1, 3, 5} × seeds {8, 16, 64}.
Includes all n_stab=5 runs (exp1_stab_5_dlc{dlc}_seed{seed}).

Panels A, B, C — Per-dlc scatter (one panel per dlc value):
  x: Price volatility σ (lower = more stable)
  y: |Final price / unit cost − 1| (deviation from break-even; lower = better)
  color: bankruptcy rate (RdYlGn_r colormap)
  error bars: min/max range across seeds
  Pareto frontier overlaid as dashed line

Panel D — Composite health score heatmap:
  score = mean(survival_score, price_level_score, stability_score)
  all components normalized to [0, 1]; higher = healthier market
  YlGn colormap; missing cells hatched

Usage:
    python exp1_score.py [--logs-dir logs/] [--good food] [--output ...]
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
import matplotlib.gridspec as gridspec
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
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "grid.color":         "gray",
    "legend.frameon":     True,
    "legend.framealpha":  0.92,
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


# ── Data loading (mirrors exp1_heatmap.py) ────────────────────────────────────

def collect_run_dirs(logs_dir):
    dirs = []
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, dlc, n_stab, seed)
                if d:
                    dirs.append(d)
    return dirs


def _serialize(cell_data):
    # tuple keys → "n_stab,dlc" strings; values are lists of plain dicts (already JSON-safe)
    return {f"{ns},{dlc}": runs for (ns, dlc), runs in cell_data.items()}


def _deserialize(data):
    return {
        (int(k.split(",")[0]), int(k.split(",")[1])): runs
        for k, runs in data.items()
    }


def resolve_run_dir(logs_dir, dlc, n_stab, seed):
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

    price_df = db.price_per_firm_over_time(good)
    if not price_df.empty:
        per_ts_mean = price_df[price_df["value"] > 0].groupby("timestep")["value"].mean()
        price_std = float(per_ts_mean.std()) if len(per_ts_mean) > 1 else 0.0
    else:
        price_std = 0.0

    return {
        "bankruptcy_rate": bankruptcy_rate,
        "final_avg_price": final_avg_price,
        "price_std":       price_std,
        "unit_cost":       get_unit_cost(run_dir),
    }


def load_all_metrics(logs_dir, good, workers=8):
    """
    Returns cell_data: Dict[(n_stab, dlc), List[metric_dict]]
    Each list contains one dict per available seed.
    """
    jobs = []
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed)
                if run_dir:
                    jobs.append((n_stab, dlc, seed, run_dir))

    print(f"Loading {len(jobs)} runs...", flush=True)
    cell_data = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_job = {
            ex.submit(compute_metrics, run_dir, good): (n_stab, dlc, seed)
            for n_stab, dlc, seed, run_dir in jobs
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_job):
            n_stab, dlc, seed = future_to_job[future]
            done += 1
            m = future.result()
            status = "ok" if m else "empty"
            print(f"  [{done}/{len(jobs)}] stab={n_stab} dlc={dlc} seed={seed} — {status}", flush=True)
            if m:
                key = (n_stab, dlc)
                cell_data.setdefault(key, []).append(m)

    return cell_data


# ── Score computation ─────────────────────────────────────────────────────────

def compute_composite(cell_data):
    """
    Returns per-cell aggregates (mean ± range) and composite scores.
    Normalizes all components globally to [0, 1].
    """
    # Collect global ranges for normalization
    all_br, all_pstd, all_pdev = [], [], []
    cell_means = {}

    for (n_stab, dlc), runs in cell_data.items():
        uc    = float(np.mean([r["unit_cost"] for r in runs]))
        br    = [r["bankruptcy_rate"]                    for r in runs]
        pstd  = [r["price_std"]                          for r in runs]
        pdev  = [abs(r["final_avg_price"] / uc - 1.0)   for r in runs]
        cell_means[(n_stab, dlc)] = {
            "br_mean":   float(np.mean(br)),
            "br_lo":     float(np.min(br)),
            "br_hi":     float(np.max(br)),
            "pstd_mean": float(np.mean(pstd)),
            "pstd_lo":   float(np.min(pstd)),
            "pstd_hi":   float(np.max(pstd)),
            "pdev_mean": float(np.mean(pdev)),
            "pdev_lo":   float(np.min(pdev)),
            "pdev_hi":   float(np.max(pdev)),
            "unit_cost": uc,
            "n_seeds":   len(runs),
        }
        all_br.extend(br);  all_pstd.extend(pstd);  all_pdev.extend(pdev)

    max_pstd = max(all_pstd) if all_pstd else 1.0
    max_pdev = max(all_pdev) if all_pdev else 1.0

    # Add composite score to each cell
    for key, agg in cell_means.items():
        survival   = 1.0 - agg["br_mean"]
        stability  = 1.0 - agg["pstd_mean"] / max_pstd
        price_lvl  = 1.0 - min(agg["pdev_mean"] / max_pdev, 1.0)
        agg["composite"] = (survival + stability + price_lvl) / 3.0
        agg["max_pstd"]  = max_pstd
        agg["max_pdev"]  = max_pdev

    return cell_means


# ── Plotting ──────────────────────────────────────────────────────────────────

def pareto_front(pts):
    """Non-dominated points (lower x AND lower y is better), sorted by x."""
    pts_sorted = sorted(pts, key=lambda p: p[0])
    front, min_y = [], float('inf')
    for p in pts_sorted:
        if p[1] < min_y:
            front.append(p)
            min_y = p[1]
    return front


def plot_scatter_panel(ax, cell_means, dlc_val, cmap, norm, panel_label):
    """One scatter sub-panel for a given dlc value."""
    points = sorted(
        [(ns, agg) for (ns, dlc), agg in cell_means.items() if dlc == dlc_val],
        key=lambda x: x[0]
    )
    if not points:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=8)
        ax.set_title(panel_label)
        return

    xs, ys = [], []
    for ns, agg in points:
        x  = agg["pstd_mean"]
        y  = agg["pdev_mean"]
        br = agg["br_mean"]
        color = cmap(norm(br))

        if agg["n_seeds"] > 1:
            xerr = [[agg["pstd_mean"] - agg["pstd_lo"]], [agg["pstd_hi"] - agg["pstd_mean"]]]
            yerr = [[agg["pdev_mean"] - agg["pdev_lo"]], [agg["pdev_hi"] - agg["pdev_mean"]]]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                        fmt="none", color=color, alpha=0.6,
                        elinewidth=0.5, capsize=0, zorder=2)

        ax.scatter(x, y, s=55, c=[color],
                   edgecolors="white", linewidths=0.5, zorder=4)
        ax.annotate(f"$k$={ns}", (x, y),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=9, color="0.3", zorder=5)
        xs.append(x); ys.append(y)

    # Pareto frontier
    front = pareto_front(list(zip(xs, ys)))
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.plot(fx, fy, '--', color="0.45", linewidth=0.9, alpha=0.7, zorder=3,
                label="Pareto front")

    ax.annotate("← ideal", xy=(0.04, 0.05), xycoords="axes fraction",
                fontsize=9, color="0.5", style="italic")
    ax.set_xlabel("Price volatility $σ$")
    ax.set_ylabel("Price distortion")
    ax.set_title(panel_label)
    ax.set_xlim(left=max(0, min(xs) * 0.85) if xs else 0)
    ax.set_ylim(bottom=max(0, min(ys) * 0.85) if ys else 0)


def plot_heatmap(ax, fig, cell_means):
    """Panel D: composite health score heatmap over dlc × n_stab."""
    n_row = len(N_STAB_VALUES)
    n_col = len(DLC_VALUES)
    grid  = np.full((n_row, n_col), np.nan)
    avail = np.zeros((n_row, n_col), dtype=bool)

    for (n_stab, dlc), agg in cell_means.items():
        if n_stab in N_STAB_VALUES and dlc in DLC_VALUES:
            i = N_STAB_VALUES.index(n_stab)
            j = DLC_VALUES.index(dlc)
            grid[i, j]  = agg["composite"]
            avail[i, j] = True

    cmap    = plt.get_cmap("YlGn")
    display = np.ma.masked_invalid(grid)
    im = ax.imshow(
        display, cmap=cmap, vmin=0.0, vmax=1.0,
        aspect="auto", interpolation="nearest",
    )
    cb_hm = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="Health score")
    cb_hm.set_label("Health score", fontsize=11)
    cb_hm.ax.tick_params(labelsize=9)

    # Hatch missing cells
    for i in range(n_row):
        for j in range(n_col):
            if not avail[i, j]:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    boxstyle="square,pad=0", linewidth=0,
                    facecolor="#cccccc", hatch="///", edgecolor="#888888", zorder=5,
                )
                ax.add_patch(rect)

    # Annotate cells
    for i in range(n_row):
        for j in range(n_col):
            if not avail[i, j]:
                continue
            val      = grid[i, j]
            rgba     = cmap(val)
            lum      = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_col  = "white" if lum < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=txt_col, zorder=10)

    ax.set_xticks(range(n_col))
    ax.set_xticklabels([f"dlc={d}" for d in DLC_VALUES])
    ax.set_yticks(range(n_row))
    ax.set_yticklabels([f"$k$={n}" for n in N_STAB_VALUES])
    ax.set_xlabel("Consumer discovery limit")
    ax.set_ylabel("Stabilizing firms ($k$)")
    ax.set_title("(D) Composite\nHealth Score")
    ax.grid(False)

    # Hatch legend
    hatch_patch = mpatches.Patch(
        facecolor="#cccccc", hatch="///", edgecolor="#888888", label="No data",
    )
    ax.legend(handles=[hatch_patch], loc="lower right", fontsize=9, borderpad=0.5)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fig 4: Exp1 Multi-Dimensional Score")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "exp1", "exp1_score.pdf"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp1_score", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir)

    if is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        cell_data = _deserialize(load_cache_data(cache_path))
    else:
        cell_data = load_all_metrics(args.logs_dir, args.good, workers=args.workers)
        save_cache(cache_path, _serialize(cell_data), args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

    cell_means = compute_composite(cell_data)

    print("\nComposite scores:")
    for (ns, dlc), agg in sorted(cell_means.items()):
        print(f"  k={ns} dlc={dlc}: composite={agg['composite']:.3f}  "
              f"(br={agg['br_mean']:.2f}, pstd={agg['pstd_mean']:.3f}, "
              f"pdev={agg['pdev_mean']:.3f})")

    # ── Layout: 3 scatter panels + colorbar + heatmap ───────────────────────
    fig = plt.figure(figsize=(13.0, 5.5))
    gs  = gridspec.GridSpec(
        1, 5, figure=fig,
        width_ratios=[1, 1, 1, 0.07, 1.8],
        left=0.07, right=0.97, bottom=0.18, top=0.90,
        wspace=0.45,
    )
    ax_dlc = {
        1: fig.add_subplot(gs[0, 0]),
        3: fig.add_subplot(gs[0, 1]),
        5: fig.add_subplot(gs[0, 2]),
    }
    ax_cbar_sc = fig.add_subplot(gs[0, 3])
    ax_heatmap = fig.add_subplot(gs[0, 4])

    # Shared bankruptcy colorbar for scatter panels
    cmap_br = plt.get_cmap("RdYlGn_r")
    norm_br = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap_br, norm=norm_br)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar_sc)
    cb.set_label("Bankruptcy rate", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    for dlc_val, panel_lbl in [(1, "(A) dlc=1"), (3, "(B) dlc=3"), (5, "(C) dlc=5")]:
        plot_scatter_panel(ax_dlc[dlc_val], cell_means, dlc_val, cmap_br, norm_br, panel_lbl)

    plot_heatmap(ax_heatmap, fig, cell_means)

    fig.suptitle(
        "Experiment 1: Multi-Dimensional Market Health",
        fontweight="bold", y=0.97,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
