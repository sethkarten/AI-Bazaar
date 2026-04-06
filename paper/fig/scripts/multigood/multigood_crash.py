"""
multigood_crash.py — Fig F: Multi-Good Crash Generalization Check.

Two-panel timeseries figure showing crash dynamics in a 2-good market:
  Left:  Price trajectories per good (mean price / unit cost over time)
  Right: Firm survival count over time

Expected run naming convention:
  logs/multigood_{slug}/multigood_{slug}_baseline_seed{seed}/

Usage:
    python paper/fig/scripts/multigood/multigood_crash.py \\
        [--slug gemini-3-flash-preview] [--goods food clothing] \\
        [--logs-dir logs/] [--output ...]
"""

import argparse
import glob
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEEDS        = [8, 16, 64]
# Okabe-Ito colors for goods
GOOD_COLORS  = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8.5,
    "lines.linewidth":    1.6,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex":        False,
    "pdf.fonttype":       42,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_states(run_dir):
    p = os.path.join(run_dir, "states.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    files = sorted(glob.glob(os.path.join(run_dir, "state_t*.json")))
    states = []
    for fp in files:
        with open(fp) as f:
            states.append(json.load(f))
    return states


def load_unit_cost(run_dir, good):
    attr_path = os.path.join(run_dir, "firm_attributes.json")
    if not os.path.isfile(attr_path):
        return 1.0
    try:
        with open(attr_path) as f:
            attrs = json.load(f)
        costs = [
            a["supply_unit_costs"][good]
            for a in attrs
            if isinstance(a.get("supply_unit_costs"), dict)
            and good in a["supply_unit_costs"]
        ]
        return float(np.mean(costs)) if costs else 1.0
    except Exception:
        return 1.0


def detect_goods(run_dir):
    """Detect goods from first timestep of states."""
    states = load_states(run_dir)
    if not states:
        return []
    for f in states[0].get("firms", []):
        prices = f.get("prices", {})
        if prices:
            return list(prices.keys())
    return []


def compute_timeseries(run_dir, goods):
    """
    Return dict per good:
      {good: {"ts": array, "price_ratio": array, "survival": array}}
    Also returns "survival" (firm count over all timesteps).
    """
    states = load_states(run_dir)
    if not states:
        return None

    unit_costs = {g: load_unit_cost(run_dir, g) for g in goods}
    result = {g: {"ts": [], "price_ratio": []} for g in goods}
    survival = []

    for i, s in enumerate(states):
        firms = s.get("firms", [])
        n_active = sum(1 for f in firms if f.get("in_business"))
        survival.append(n_active)

        for g in goods:
            uc = unit_costs[g]
            prices = [
                f["prices"][g]
                for f in firms
                if f.get("in_business")
                and isinstance(f.get("prices", {}).get(g), (int, float))
                and f["prices"][g] > 0
            ]
            if prices:
                result[g]["ts"].append(i)
                result[g]["price_ratio"].append(float(np.mean(prices)) / uc)

    for g in goods:
        result[g]["ts"] = np.array(result[g]["ts"])
        result[g]["price_ratio"] = np.array(result[g]["price_ratio"])
    result["_survival"] = np.array(survival)
    result["_n_ts"]     = len(states)

    return result


def resolve_run_dir(logs_dir, slug, seed):
    name = f"multigood_{slug}_baseline_seed{seed}"
    d = os.path.join(logs_dir, f"multigood_{slug}", name)
    return d if os.path.isdir(d) else None


def load_all_seeds(logs_dir, slug, goods):
    """Return list of per-seed timeseries dicts."""
    results = []
    for seed in SEEDS:
        d = resolve_run_dir(logs_dir, slug, seed)
        if d is None:
            continue
        ts_data = compute_timeseries(d, goods)
        if ts_data is not None:
            results.append(ts_data)
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_price_panel(ax, all_seed_data, goods):
    """Left panel: price ratio per good, per-seed traces + mean."""
    for gi, good in enumerate(goods):
        color = GOOD_COLORS[gi % len(GOOD_COLORS)]

        # Find common length
        min_len = min(len(d[good]["ts"]) for d in all_seed_data
                      if good in d and len(d[good]["ts"]) > 0)
        if min_len == 0:
            continue

        traces = np.array([d[good]["price_ratio"][:min_len]
                           for d in all_seed_data
                           if good in d and len(d[good]["ts"]) >= min_len])
        if traces.shape[0] == 0:
            continue
        ts = all_seed_data[0][good]["ts"][:min_len]

        # Per-seed faint traces
        for trace in traces:
            ax.plot(ts, trace, color=color, lw=0.8, alpha=0.3, zorder=2)

        # Mean line
        ax.plot(ts, np.mean(traces, axis=0), color=color, lw=2.0,
                label=good.capitalize(), zorder=4)

    ax.axhline(1.0, color="0.5", lw=0.8, ls="--", zorder=1, label="$p/c = 1$ (break-even)")
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Price / Unit Cost ($p/c$)")
    ax.set_title("Price Trajectories per Good", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_survival_panel(ax, all_seed_data, goods):
    """Right panel: firm survival count + first bankruptcy annotations."""
    if not all_seed_data:
        ax.set_title("Firm Survival", fontsize=10, fontweight="bold")
        return

    min_len = min(d["_n_ts"] for d in all_seed_data)
    traces  = np.array([d["_survival"][:min_len] for d in all_seed_data])
    ts_all  = np.arange(min_len)

    for trace in traces:
        ax.step(ts_all, trace, color="#0072B2", lw=0.8, alpha=0.3, where="post", zorder=2)
    ax.step(ts_all, np.mean(traces, axis=0), color="#0072B2", lw=2.0,
            where="post", label="Active firms (mean)", zorder=4)

    # Annotate first bankruptcy per good (when mean price_ratio drops to 0)
    first_crash = {}
    for gi, good in enumerate(goods):
        min_pr_len = min(len(d[good]["ts"]) for d in all_seed_data
                         if good in d and len(d[good]["ts"]) > 0)
        if min_pr_len == 0:
            continue
        traces_pr = np.array([d[good]["price_ratio"][:min_pr_len]
                               for d in all_seed_data
                               if good in d and len(d[good]["ts"]) >= min_pr_len])
        mean_pr = np.mean(traces_pr, axis=0)
        # First t where mean_pr ≈ 0 (below 0.05 threshold)
        crash_ts = np.where(mean_pr < 0.05)[0]
        if len(crash_ts) > 0:
            first_crash[good] = int(crash_ts[0])

    color = GOOD_COLORS
    for gi, (good, t_crash) in enumerate(first_crash.items()):
        ax.axvline(t_crash, color=color[gi % len(color)], ls=":", lw=0.8, alpha=0.7)
        ax.text(t_crash + 2, 0.5 + gi * 0.3,
                f"{good.capitalize()} crash $t≈{t_crash}$",
                fontsize=7, color=color[gi % len(color)], va="bottom")

    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Active Firms")
    ax.set_title("Firm Survival Over Time", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fig F: multi-good crash figure.")
    ap.add_argument("--slug",     default="gemini-3-flash-preview")
    ap.add_argument("--goods",    nargs="+", default=None,
                    help="Good names (default: auto-detect from first run)")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--output",   default=None)
    args = ap.parse_args()

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "multigood"))
        args.output = os.path.join(fig_dir, "multigood_crash.pdf")

    slug     = args.slug
    logs_dir = args.logs_dir

    # Auto-detect goods from first available run
    goods = args.goods
    if goods is None:
        for seed in SEEDS:
            d = resolve_run_dir(logs_dir, slug, seed)
            if d is not None:
                goods = detect_goods(d)
                if goods:
                    print(f"  Auto-detected goods: {goods}", flush=True)
                    break
    if not goods:
        goods = ["food", "clothing"]
        warnings.warn(f"No runs found for slug='{slug}'. Using default goods={goods}.")

    all_seed_data = load_all_seeds(logs_dir, slug, goods)
    print(f"  Loaded {len(all_seed_data)} seeds", flush=True)

    if not all_seed_data:
        warnings.warn(f"No data found for slug='{slug}' in {logs_dir}. Saving empty figure.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.75, 2.8),
                                             constrained_layout=True)

    if all_seed_data:
        draw_price_panel(ax_left, all_seed_data, goods)
        draw_survival_panel(ax_right, all_seed_data, goods)
    else:
        ax_left.set_title("Price Trajectories per Good", fontsize=10, fontweight="bold")
        ax_right.set_title("Firm Survival Over Time", fontsize=10, fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
