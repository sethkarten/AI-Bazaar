"""
exp4_shocks.py — Fig E: Market Response to Supply and Demand Shocks.

Two-panel timeseries figure with shock applied at t=182 (run midpoint):
  Left:  Supply shock — unit cost step change (c: 1.0 → 2.0 or 0.5)
  Right: Demand shock — Poisson lambda step change (×2 or ×0.5)

Each panel:
  X-axis: timestep t ∈ [0, T]
  Y-axis (primary):   mean price / current unit cost (p/c)
  Y-axis (secondary): active firm count (step function, lighter color)
  Lines: baseline, condition_up, condition_down
  Shaded bands: min/max across seeds
  Vertical dashed line at t=182 labeled "Shock at t=182"

Expected run naming convention:
  logs/exp4_{slug}/exp4_{slug}_{condition}_seed{seed}/
    condition ∈ {baseline, supply_up, supply_down, demand_up, demand_down}

Usage:
    python paper/fig/scripts/exp4/exp4_shocks.py \\
        [--slug gemini-3-flash-preview] [--logs-dir logs/] [--output ...]
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

SEEDS     = [8, 16, 64]
SHOCK_T   = 182

# Colors for the three lines per panel
LINE_STYLES = [
    ("baseline",    "Baseline (unshocked)", "#0072B2", "-"),
    (None,          None,                   None,      None),  # filled by caller
    (None,          None,                   None,      None),
]

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
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


def load_unit_cost(run_dir, good="food"):
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


def price_timeseries(run_dir, good="food"):
    """
    Return (timesteps, price_ratio, active_counts) arrays.
    price_ratio[t] = mean_price(t) / unit_cost (pre-shock unit cost).
    active_counts[t] = number of in-business firms.
    """
    states = load_states(run_dir)
    if not states:
        return None, None, None
    unit_cost = load_unit_cost(run_dir, good)

    ts, prices, active = [], [], []
    for i, s in enumerate(states):
        firms = s.get("firms", [])
        in_biz_prices = [
            f["prices"][good]
            for f in firms
            if f.get("in_business")
            and isinstance(f.get("prices", {}).get(good), (int, float))
            and f["prices"][good] > 0
        ]
        n_active = sum(1 for f in firms if f.get("in_business"))
        if in_biz_prices:
            ts.append(i)
            prices.append(float(np.mean(in_biz_prices)) / unit_cost)
        active.append(n_active)

    return (np.array(ts) if ts else None,
            np.array(prices) if prices else None,
            np.array(active))


def resolve_run_dir(logs_dir, slug, condition, seed):
    name = f"exp4_{slug}_{condition}_seed{seed}"
    d = os.path.join(logs_dir, f"exp4_{slug}", name)
    return d if os.path.isdir(d) else None


def load_condition_ts(logs_dir, slug, condition, good="food"):
    """
    Return mean (and min/max envelope) timeseries across seeds.
    Returns (ts_common, mean_price, min_price, max_price, mean_active) or None.
    """
    all_ts, all_prices, all_active = [], [], []
    for seed in SEEDS:
        d = resolve_run_dir(logs_dir, slug, condition, seed)
        if d is None:
            continue
        ts, pr, ac = price_timeseries(d, good)
        if ts is None:
            continue
        all_ts.append(ts)
        all_prices.append(pr)
        all_active.append(ac)

    if not all_ts:
        return None

    # Align on common timestep range (shortest run)
    min_len = min(len(t) for t in all_ts)
    prices_stack = np.array([p[:min_len] for p in all_prices])
    active_stack = np.array([a[:min_len] for a in all_active])
    ts_common    = all_ts[0][:min_len]

    return {
        "ts":           ts_common,
        "mean_price":   np.mean(prices_stack, axis=0),
        "min_price":    np.min(prices_stack, axis=0),
        "max_price":    np.max(prices_stack, axis=0),
        "mean_active":  np.mean(active_stack, axis=0),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_shock_panel(ax, ax2, conditions_data, panel_title):
    """
    conditions_data: list of (condition_label, color, linestyle, data_dict)
    data_dict keys: ts, mean_price, min_price, max_price, mean_active
    """
    for label, color, ls, data in conditions_data:
        if data is None:
            continue
        ts  = data["ts"]
        ax.plot(ts, data["mean_price"], color=color, ls=ls,
                lw=1.6, label=label, zorder=4)
        ax.fill_between(ts, data["min_price"], data["max_price"],
                        color=color, alpha=0.15, zorder=2)

    # Secondary axis: active firm count (from first available condition)
    for _, color, ls, data in conditions_data:
        if data is not None:
            ts_all = np.arange(len(data["mean_active"]))
            ax2.step(ts_all, data["mean_active"], color="0.7", lw=0.8,
                     where="post", zorder=1)
            break

    # Shock line
    ax.axvline(SHOCK_T, color="0.4", ls="--", lw=0.8, zorder=5)
    ax.text(SHOCK_T + 2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1.0,
            f"Shock at $t={SHOCK_T}$", fontsize=7, color="0.4", va="top")

    # Reference line at p/c = 1
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":", zorder=1)

    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Price / Unit Cost ($p/c$)")
    ax.set_title(panel_title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax2.set_ylabel("Active firms", fontsize=7, color="0.6")
    ax2.tick_params(axis="y", labelcolor="0.6", labelsize=7)
    ax2.spines["top"].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fig E: exp4 supply/demand shocks.")
    ap.add_argument("--slug",     default="gemini-3-flash-preview")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good",     default="food")
    ap.add_argument("--output",   default=None)
    args = ap.parse_args()

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp4"))
        args.output = os.path.join(fig_dir, "exp4_shocks.pdf")

    slug     = args.slug
    logs_dir = args.logs_dir
    good     = args.good

    # Load all conditions
    conditions = ["baseline", "supply_up", "supply_down", "demand_up", "demand_down"]
    data = {}
    for cond in conditions:
        d = load_condition_ts(logs_dir, slug, cond, good)
        n_seeds = sum(1 for seed in SEEDS
                      if resolve_run_dir(logs_dir, slug, cond, seed))
        print(f"  {cond}: {n_seeds} seeds", flush=True)
        data[cond] = d

    total = sum(1 for v in data.values() if v is not None)
    if total == 0:
        warnings.warn(f"No data found for slug='{slug}' in {logs_dir}. Saving empty figure.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0), constrained_layout=True)
    ax_l, ax_r = axes
    ax2_l = ax_l.twinx()
    ax2_r = ax_r.twinx()

    supply_conditions = [
        ("Baseline",       "#0072B2", "-",  data.get("baseline")),
        ("Cost $×2$",      "#D55E00", "--", data.get("supply_up")),
        ("Cost $×0.5$",    "#009E73", "-.", data.get("supply_down")),
    ]
    demand_conditions = [
        ("Baseline",       "#0072B2", "-",  data.get("baseline")),
        ("Demand $×2$",    "#D55E00", "--", data.get("demand_up")),
        ("Demand $×0.5$",  "#009E73", "-.", data.get("demand_down")),
    ]

    draw_shock_panel(ax_l, ax2_l, supply_conditions, "Supply Shock (unit cost step change)")
    draw_shock_panel(ax_r, ax2_r, demand_conditions, "Demand Shock ($\\lambda$ step change)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
