"""
Fig 4: Experiment 1 Multi-Dimensional Performance Score

Combines bankruptcy rate, final price deviation from unit cost, and price volatility
into a composite market health score and a multi-metric trade-off visualization.

Panel A — Trade-off scatter:
  x: Price volatility σ (lower = more stable)
  y: |Final price / unit cost − 1| (deviation from break-even; lower = better)
  bubble size: bankruptcy rate (larger = worse)
  color: n_stab (Okabe-Ito)
  marker: dlc (●=1, ■=3, ▲=5)
  error bars: min/max range across seeds

Panel B — Composite health score heatmap:
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

# Okabe-Ito palette, one color per n_stab level
STAB_COLORS = {
    0: "#000000",  # Black   — baseline (no stab)
    1: "#E69F00",  # Orange
    2: "#0072B2",  # Blue
    4: "#009E73",  # Bluish Green
    5: "#CC79A7",  # Reddish Purple — stab baseline (all stab)
}
DLC_MARKERS = {1: "o", 3: "s", 5: "^"}
DLC_LABELS  = {1: "dlc=1", 3: "dlc=3", 5: "dlc=5"}


# ── Data loading (mirrors exp1_heatmap.py) ────────────────────────────────────

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

def plot_scatter(ax, cell_means):
    """Panel A: multi-dimensional trade-off scatter."""
    ax.set_axisbelow(True)

    # Bubble size: s = area proportional to bankruptcy rate
    # br=0 → s=25 (still visible); br=1 → s=350
    SIZE_MIN, SIZE_MAX = 25, 350

    plotted_stab = set()
    plotted_dlc  = set()

    for (n_stab, dlc), agg in cell_means.items():
        x     = agg["pstd_mean"]
        y     = agg["pdev_mean"]
        xerr  = [[agg["pstd_mean"] - agg["pstd_lo"]], [agg["pstd_hi"] - agg["pstd_mean"]]]
        yerr  = [[agg["pdev_mean"] - agg["pdev_lo"]], [agg["pdev_hi"] - agg["pdev_mean"]]]
        size  = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * agg["br_mean"]
        color = STAB_COLORS[n_stab]
        marker = DLC_MARKERS[dlc]
        alpha  = 0.80

        # Error bars (only when multi-seed)
        if agg["n_seeds"] > 1:
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                fmt="none", color=color, alpha=0.45,
                capsize=2.5, elinewidth=0.8, zorder=2,
            )

        sc = ax.scatter(
            x, y, s=size,
            color=color, marker=marker,
            alpha=alpha, linewidths=0.6, edgecolors="white",
            zorder=4,
        )

        # Point label: "(dlc, k)" offset slightly
        label_txt = f"({dlc},{n_stab})"
        ax.annotate(
            label_txt, (x, y),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6.5, color=color,
            zorder=5,
        )

        plotted_stab.add(n_stab)
        plotted_dlc.add(dlc)

    # ── Legends ──────────────────────────────────────────────────────────
    # Color legend: n_stab
    stab_handles = [
        mpatches.Patch(color=STAB_COLORS[ns], label=f"$k$={ns}")
        for ns in sorted(plotted_stab)
    ]
    leg1 = ax.legend(
        handles=stab_handles, title="Stab. firms",
        loc="upper left", fontsize=7, title_fontsize=7.5,
        handlelength=1.0, borderpad=0.6,
    )
    ax.add_artist(leg1)

    # Marker legend: dlc
    dlc_handles = [
        plt.scatter([], [], marker=DLC_MARKERS[d], color="#555555", s=35, label=DLC_LABELS[d])
        for d in sorted(plotted_dlc)
    ]
    leg2 = ax.legend(
        handles=dlc_handles, title="Discovery",
        loc="upper right", fontsize=7, title_fontsize=7.5,
        handlelength=0.8, borderpad=0.6,
    )
    ax.add_artist(leg2)

    # Size legend: bankruptcy rate
    for br_val, label in [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]:
        s = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * br_val
        ax.scatter([], [], s=s, color="#aaaaaa", alpha=0.7,
                   label=f"{label}", linewidths=0.5, edgecolors="white")
    ax.legend(
        title="Bankruptcy", loc="lower right",
        fontsize=7, title_fontsize=7.5,
        handlelength=0.8, borderpad=0.6,
        labelspacing=0.8,
    )

    # "Ideal" annotation
    ax.annotate(
        "← ideal", xy=(0.03, 0.04), xycoords="axes fraction",
        fontsize=7, color="#555555", style="italic",
    )

    ax.set_xlabel("Price volatility $σ$")
    ax.set_ylabel("Price deviation $|p/c - 1|$")
    ax.set_title("(A) Trade-off Landscape")
    ax.set_xlim(left=max(0, ax.get_xlim()[0] - 0.01))
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.005))


def plot_heatmap(ax, fig, cell_means):
    """Panel B: composite health score heatmap over dlc × n_stab."""
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
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="Health score")

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
                    fontsize=8, color=txt_col, zorder=10)

    ax.set_xticks(range(n_col))
    ax.set_xticklabels([f"dlc={d}" for d in DLC_VALUES])
    ax.set_yticks(range(n_row))
    ax.set_yticklabels([f"$k$={n}" for n in N_STAB_VALUES])
    ax.set_xlabel("Consumer discovery limit")
    ax.set_ylabel("Stabilizing firms ($k$)")
    ax.set_title("(B) Composite Health Score")
    ax.grid(False)

    # Hatch legend
    hatch_patch = mpatches.Patch(
        facecolor="#cccccc", hatch="///", edgecolor="#888888", label="No data",
    )
    ax.legend(handles=[hatch_patch], loc="lower right", fontsize=7, borderpad=0.5)


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

    cell_data  = load_all_metrics(args.logs_dir, args.good, workers=args.workers)
    cell_means = compute_composite(cell_data)

    print("\nComposite scores:")
    for (ns, dlc), agg in sorted(cell_means.items()):
        print(f"  k={ns} dlc={dlc}: composite={agg['composite']:.3f}  "
              f"(br={agg['br_mean']:.2f}, pstd={agg['pstd_mean']:.3f}, "
              f"pdev={agg['pdev_mean']:.3f})")

    # ── Layout: left scatter (wider) + right heatmap ─────────────────────
    fig = plt.figure(figsize=(7.0, 4.2))
    gs  = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[5, 3],
        left=0.08, right=0.97, bottom=0.12, top=0.90,
        wspace=0.40,
    )
    ax_scatter = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1])

    plot_scatter(ax_scatter, cell_means)
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
