"""
exp_crash_appendix.py -- Multi-model crash appendix figure (Exp1 or Exp5).

3 rows × 3 cols line-plot figure:
  Rows:    Price / Unit Cost  |  Market Volume  |  Price Volatility
  Cols:    dlc/dlf = 1        |  dlc/dlf = 3    |  dlc/dlf = 5
  Lines:   one per model, mean across seeds ± 1σ shaded band
  X-axis:  k ∈ {0, 1, 3, 5}  (stabilizing firms)

Metrics per run (from states.json):
  Price / Unit Cost  : mean over timesteps of mean(firm_price / unit_cost) across
                       active firms with a valid positive price
  Market Volume      : mean filled_orders_count per timestep
  Price Volatility   : std of per-timestep mean price across active firms

Exp1  run pattern: logs/exp1_{slug}/exp1_{slug}_stab_{k}_dlc{dlc}_seed{seed}/states.json
Exp5  run pattern: logs/exp5_{slug}/exp5_{slug}_stab_{k}_dlf{dlf}_seed{seed}/states.json

Usage:
    # Exp1 (default models: Gemini 3 Flash, GPT 5.4, Sonnet 4.6)
    python paper/fig/scripts/exp_crash_appendix.py --exp exp1

    # Exp5 (only models with data; missing cells silently skipped)
    python paper/fig/scripts/exp_crash_appendix.py --exp exp5

    # Custom model list (slug:label pairs)
    python paper/fig/scripts/exp_crash_appendix.py --exp exp1 \\
        --models gemini-3-flash-preview:"Gemini 3 Flash" \\
                 openai_gpt-5.4:"GPT 5.4"

Output:
    paper/fig/exp1/exp1_crash_appendix.pdf  (--exp exp1)
    paper/fig/exp5/exp5_crash_appendix.pdf  (--exp exp5)
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Default model registries ─────────────────────────────────────────────────
# (slug, display_label, color, marker)
EXP1_MODELS = [
    ("gemini-3-flash-preview",      "Gemini 3 Flash", "#0072B2", "o"),
    ("openai_gpt-5.4",              "GPT 5.4",        "#E69F00", "s"),
    ("anthropic_claude-sonnet-4.6", "Sonnet 4.6",     "#009E73", "^"),
]

EXP5_MODELS = [
    ("gemini-3-flash-preview", "Gemini 3 Flash", "#0072B2", "o"),
]

K_VALUES   = [0, 1, 3, 5]
DLC_VALUES = [1, 3, 5]
SEEDS      = [8, 16, 64]

_SCRIPT_DIR = os.path.dirname(__file__)

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
    "grid.alpha":         0.3,
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

def run_dir(logs_dir: str, exp: str, slug: str, k: int, dlc: int, seed: int) -> str:
    """Return run directory path."""
    dim = "dlf" if exp == "exp5" else "dlc"
    name = f"{exp}_{slug}_stab_{k}_{dim}{dlc}_seed{seed}"
    return os.path.join(logs_dir, f"{exp}_{slug}", name)


def load_unit_cost(run_d: str, good: str = "food") -> float:
    attr_path = os.path.join(run_d, "firm_attributes.json")
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


def compute_metrics(run_d: str, good: str = "food") -> dict | None:
    states_path = os.path.join(run_d, "states.json")
    if not os.path.isfile(states_path):
        return None
    try:
        with open(states_path) as f:
            states = json.load(f)
    except Exception:
        return None
    if not states:
        return None

    unit_cost = load_unit_cost(run_d, good)

    price_per_ts   = []   # mean(price/unit_cost) per timestep
    volume_per_ts  = []   # filled_orders_count per timestep
    raw_price_per_ts = [] # mean raw price per timestep (for volatility)

    for s in states:
        firms = s.get("firms", [])
        active_prices = [
            f["prices"][good]
            for f in firms
            if f.get("in_business")
            and isinstance(f.get("prices", {}).get(good), (int, float))
            and f["prices"][good] > 0
        ]
        if active_prices:
            mean_price = float(np.mean(active_prices))
            price_per_ts.append(mean_price / unit_cost)
            raw_price_per_ts.append(mean_price)

        vol = s.get("filled_orders_count")
        if vol is not None:
            volume_per_ts.append(float(vol))

    if not price_per_ts:
        return None

    return {
        "price_ratio": float(np.mean(price_per_ts)),
        "volume":      float(np.mean(volume_per_ts)) if volume_per_ts else 0.0,
        "volatility":  float(np.std(raw_price_per_ts, ddof=1))
                       if len(raw_price_per_ts) > 1 else 0.0,
    }


def load_cell(logs_dir, exp, slug, k, dlc, good="food"):
    """Return list of metric dicts (one per seed) for one (k, dlc) cell."""
    records = []
    for seed in SEEDS:
        d = run_dir(logs_dir, exp, slug, k, dlc, seed)
        m = compute_metrics(d, good)
        if m is not None:
            records.append(m)
    return records


def load_model_data(logs_dir, exp, slug, good="food"):
    """Return {(k, dlc): [metric_dict, ...]} for all cells."""
    data = {}
    for dlc in DLC_VALUES:
        for k in K_VALUES:
            records = load_cell(logs_dir, exp, slug, k, dlc, good)
            if records:
                data[(k, dlc)] = records
    return data


# ── Aggregation ───────────────────────────────────────────────────────────────

def agg_metric(records: list[dict], key: str):
    """Return (mean, std) for a metric key across records."""
    vals = [r[key] for r in records if key in r]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


# ── Plotting ──────────────────────────────────────────────────────────────────

METRICS = [
    ("price_ratio", "Price / Unit Cost"),
    ("volume",      "Market Volume"),
    ("volatility",  "Price Volatility"),
]


def plot_panel(ax, all_model_data, models, metric_key, dlc, dim_label,
               show_xlabel=False, show_ylabel=False, show_legend=False):
    for slug, label, color, marker in models:
        model_data = all_model_data.get(slug, {})
        xs, means, lows, highs = [], [], [], []
        for k in K_VALUES:
            records = model_data.get((k, dlc))
            if records is None:
                continue
            mean, std = agg_metric(records, metric_key)
            if mean is None:
                continue
            xs.append(k)
            means.append(mean)
            lows.append(mean - std)
            highs.append(mean + std)

        if not xs:
            continue
        xs     = np.array(xs)
        means  = np.array(means)
        lows   = np.array(lows)
        highs  = np.array(highs)

        ax.plot(xs, means, color=color, marker=marker, markersize=5,
                label=label, zorder=4)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.18, zorder=2)

    # Reference line at y=1 for price ratio
    if metric_key == "price_ratio":
        ax.axhline(1.0, color="0.6", lw=0.8, ls="--", zorder=1)

    ax.set_xticks(K_VALUES)
    if show_xlabel:
        ax.set_xlabel(f"Stabilizing Firms ($k$)")
    if show_ylabel:
        # ylabel set per-row in main
        pass
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlim(-0.3, K_VALUES[-1] + 0.3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Crash appendix figure (Exp1 or Exp5).")
    ap.add_argument("--exp",      default="exp1", choices=["exp1", "exp5"],
                    help="Which experiment to plot (default: exp1).")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good",     default="food")
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--output",   default=None,
                    help="Output PDF path. Default: paper/fig/{exp}/{exp}_crash_appendix.pdf")
    args = ap.parse_args()

    exp = args.exp
    models = EXP1_MODELS if exp == "exp1" else EXP5_MODELS
    dim_name = "dlf" if exp == "exp5" else "dlc"

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", exp))
        args.output = os.path.join(fig_dir, f"{exp}_crash_appendix.pdf")

    # Load all model data in parallel
    all_model_data = {}

    def _load(entry):
        slug = entry[0]
        return slug, load_model_data(args.logs_dir, exp, slug, args.good)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_load, m): m[0] for m in models}
        for fut in as_completed(futures):
            slug, data = fut.result()
            n_cells = sum(1 for v in data.values() if v)
            print(f"  {slug}: {n_cells} cells with data", flush=True)
            all_model_data[slug] = data

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows, n_cols = len(METRICS), len(DLC_VALUES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.2 * n_rows),
                             constrained_layout=True)

    col_titles = [f"${dim_name} = {d}$" for d in DLC_VALUES]

    for col_idx, (dlc, col_title) in enumerate(zip(DLC_VALUES, col_titles)):
        axes[0, col_idx].set_title(col_title, fontsize=11, fontweight="bold")

    for row_idx, (metric_key, metric_label) in enumerate(METRICS):
        for col_idx, dlc in enumerate(DLC_VALUES):
            ax = axes[row_idx, col_idx]
            is_bottom = (row_idx == n_rows - 1)
            is_left   = (col_idx == 0)
            is_top_right = (row_idx == 0 and col_idx == n_cols - 1)

            plot_panel(ax, all_model_data, models, metric_key, dlc, dim_name,
                       show_xlabel=is_bottom,
                       show_ylabel=is_left,
                       show_legend=is_top_right)

            if is_left:
                ax.set_ylabel(metric_label)
            if is_bottom:
                ax.set_xlabel(f"Stabilizing Firms ($k$)")

    # Single shared legend in top-right panel
    handles, labels = [], []
    for slug, label, color, marker in models:
        handles.append(plt.Line2D([0], [0], color=color, marker=marker,
                                  markersize=5, linewidth=1.6, label=label))
        labels.append(label)
    axes[0, -1].legend(handles=handles, labels=labels,
                       loc="upper right", fontsize=8.5, framealpha=0.9)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
