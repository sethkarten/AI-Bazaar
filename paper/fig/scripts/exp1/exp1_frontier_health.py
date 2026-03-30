"""
exp1_frontier_health.py -- Frontier model health profile comparison.

Three-panel line plot comparing composite market health score across
frontier LLMs (Gemini 3 Flash, GPT 5.4, Sonnet 4.6).

Layout: 1x3 panels (one per dlc in {1, 3, 5}).
  x-axis:  stabilizing firms  k
  y-axis:  composite health score  S in [0, 1]
  lines:   one per model, with +/- SE shaded bands

  S = (S_surv + S_price + S_stab) / 3
  normalised globally across all three models.

Only k in {0, 1, 3, 5} is used (same grid as GPT/Sonnet); Gemini's k=2 and k=4
cells are excluded so comparisons are aligned across models.

Data loading:
  1. Heatmap cache: paper/fig/exp1/{src}/data/exp1_heatmap_{good}.json
  2. Fallback:      compute from raw state files in logs/{src}/

Output: paper/fig/exp1/exp1_frontier_health.pdf

Usage:
    python exp1_frontier_health.py [--good food] [--output ...] [--logs-dir logs/]
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
from exp1_cache import load_cache_data
from exp1_paths import N_STAB_VALUES, DLC_VALUES, SEEDS, resolve_run_dir

# ── Constants ─────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_FIG_EXP1_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp1"))
DEFAULT_OUTPUT = os.path.join(_FIG_EXP1_DIR, "exp1_frontier_health.pdf")

FRONTIER_MODELS = [
    # (display_name, cache_dir_candidates, color, marker)
    ("Gemini 3 Flash", ["exp1_gemini-3-flash-preview"],
     "#0072B2", "o"),
    ("GPT 5.4",        ["exp1_openai_gpt-5.4", "openai_gpt-5.4"],
     "#D55E00", "s"),
    ("Sonnet 4.6",     ["exp1_anthropic_claude-sonnet-4.6"],
     "#009E73", "^"),
]

# Shared k grid across frontier models (GPT/Sonnet omit k=2, k=4; drop Gemini there too)
FRONTIER_K_COMPARE = frozenset({0, 1, 3, 5})

# ── rcParams (consistent with other exp1 figure scripts) ─────────────────
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
    "savefig.pad_inches": 0.02,
    "text.usetex":        False,
    "pdf.fonttype":       42,
})


# ── Data loading ──────────────────────────────────────────────────────────

def _find_cache(dir_candidates, good, fig_exp1_dir):
    """Return path to the first existing heatmap cache among candidates."""
    for dirname in dir_candidates:
        path = os.path.join(fig_exp1_dir, dirname, "data", f"exp1_heatmap_{good}.json")
        if os.path.isfile(path):
            return path
    return None


def _load_states(run_dir):
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


def _get_unit_cost(run_dir):
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


def _compute_metrics(run_dir, good):
    files = _load_states(run_dir)
    if not files:
        return None
    from ai_bazaar.utils.dataframe_builder import DataFrameBuilder
    db = DataFrameBuilder(state_files=files)
    firms_df = db.firms_in_business_over_time().sort_values("timestep")
    if firms_df.empty:
        return None
    states = db.states
    first_firms = len(states[0].get("firms", []))
    if first_firms == 0:
        return None
    bankruptcy_rate = 1.0 - int(firms_df.iloc[-1]["value"]) / first_firms

    last_state = states[-1]
    prices_at_last = [
        f["prices"].get(good)
        for f in last_state.get("firms", [])
        if f.get("in_business")
        and isinstance(f.get("prices", {}).get(good), (int, float))
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
    }


def _fallback_from_state_files(dir_candidates, logs_dir, good):
    """Compute per-seed data from raw state files when no cache exists."""
    per_seed_data = {}
    unit_costs = []

    for dirname in dir_candidates:
        model_key = dirname[len("exp1_"):] if dirname.startswith("exp1_") else dirname
        model_logs = os.path.join(logs_dir, dirname)
        if not os.path.isdir(model_logs):
            continue

        for i, n_stab in enumerate(N_STAB_VALUES):
            for j, dlc in enumerate(DLC_VALUES):
                cell_key = f"{i},{j}"
                if cell_key in per_seed_data:
                    continue
                for seed in SEEDS:
                    run_dir = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model_key)
                    if run_dir is None:
                        continue
                    m = _compute_metrics(run_dir, good)
                    if m is None:
                        continue
                    if cell_key not in per_seed_data:
                        per_seed_data[cell_key] = {
                            "bankruptcy_rate": [], "final_avg_price": [],
                            "price_std": [], "total_volume": [],
                        }
                    per_seed_data[cell_key]["bankruptcy_rate"].append(m["bankruptcy_rate"])
                    per_seed_data[cell_key]["final_avg_price"].append(m["final_avg_price"])
                    per_seed_data[cell_key]["price_std"].append(m["price_std"])
                    unit_costs.append(_get_unit_cost(run_dir))

        if per_seed_data:
            break

    unit_cost = float(np.mean(unit_costs)) if unit_costs else 1.0
    return per_seed_data, unit_cost


def load_model_data(dir_candidates, good, fig_exp1_dir, logs_dir):
    """Load per-seed metric data for one frontier model.

    Returns (per_seed_data, unit_cost) or (None, None) if no data found.
    per_seed_data: dict["row,col" -> {metric_name -> [seed_values]}]
    """
    cache_path = _find_cache(dir_candidates, good, fig_exp1_dir)
    if cache_path:
        try:
            data = load_cache_data(cache_path)
            psd = data.get("per_seed_data", {})
            uc  = data.get("unit_cost", 1.0)
            if psd:
                print(f"  Loaded cache: {cache_path}", flush=True)
                return psd, uc
        except Exception as e:
            print(f"  Cache load failed ({cache_path}): {e}", flush=True)

    print(f"  No cache for {dir_candidates[0]}, trying raw state files...", flush=True)
    psd, uc = _fallback_from_state_files(dir_candidates, logs_dir, good)
    if psd:
        return psd, uc
    return None, None


# ── Health score computation ──────────────────────────────────────────────

def compute_health_scores(all_model_psd):
    """Compute per-cell health scores with global normalization.

    Parameters
    ----------
    all_model_psd : dict[display_name -> (per_seed_data, unit_cost)]

    Returns
    -------
    dict[display_name -> dict[(k, dlc) -> {"mean", "se", "n", "seeds"}]]
    """
    all_pstd, all_pdev = [], []
    for _, (psd, uc) in all_model_psd.items():
        for key_str, cell in psd.items():
            row_idx, col_idx = (int(x) for x in key_str.split(","))
            if row_idx >= len(N_STAB_VALUES) or col_idx >= len(DLC_VALUES):
                continue
            k = N_STAB_VALUES[row_idx]
            if k not in FRONTIER_K_COMPARE:
                continue
            for s_idx in range(len(cell["bankruptcy_rate"])):
                all_pstd.append(cell["price_std"][s_idx])
                price = cell["final_avg_price"][s_idx]
                all_pdev.append(abs(price / uc - 1.0) if uc > 0 else 0.0)

    global_max_pstd = max(all_pstd) if all_pstd else 1.0
    global_max_pdev = max(all_pdev) if all_pdev else 1.0

    results = {}
    for name, (psd, uc) in all_model_psd.items():
        model_scores = {}
        for key_str, cell in psd.items():
            row_idx, col_idx = (int(x) for x in key_str.split(","))
            if row_idx >= len(N_STAB_VALUES) or col_idx >= len(DLC_VALUES):
                continue
            k   = N_STAB_VALUES[row_idx]
            dlc = DLC_VALUES[col_idx]
            if k not in FRONTIER_K_COMPARE:
                continue

            seed_scores = []
            for s_idx in range(len(cell["bankruptcy_rate"])):
                br    = cell["bankruptcy_rate"][s_idx]
                price = cell["final_avg_price"][s_idx]
                pstd  = cell["price_std"][s_idx]
                if br >= 1.0:
                    seed_scores.append(0.0)
                else:
                    s_surv  = 1.0 - br
                    s_stab  = (1.0 - pstd / global_max_pstd
                               if global_max_pstd > 0 else 1.0)
                    s_price = (1.0 - min(abs(price / uc - 1.0) / global_max_pdev, 1.0)
                               if global_max_pdev > 0 else 1.0)
                    seed_scores.append((s_surv + s_stab + s_price) / 3.0)

            if seed_scores:
                mean_s = float(np.mean(seed_scores))
                se_s   = (float(np.std(seed_scores, ddof=1) / np.sqrt(len(seed_scores)))
                          if len(seed_scores) > 1 else 0.0)
                model_scores[(k, dlc)] = {
                    "mean":  mean_s,
                    "se":    se_s,
                    "n":     len(seed_scores),
                    "seeds": seed_scores,
                }

        results[name] = model_scores
    return results


# ── Label collision avoidance ─────────────────────────────────────────────

def _spread_labels(labels, min_gap):
    """Push label y-positions apart so no two are closer than *min_gap*.

    Parameters
    ----------
    labels : list of dict with keys "y_data", "y_label", "color", "text"
        Sorted by y_data ascending.
    min_gap : float
        Minimum distance between adjacent label centres in data coords.

    Returns
    -------
    labels with "y_label" adjusted.
    """
    n = len(labels)
    if n <= 1:
        return labels
    ys = np.array([l["y_label"] for l in labels])
    for _ in range(50):
        moved = False
        for i in range(n - 1):
            gap = ys[i + 1] - ys[i]
            if gap < min_gap:
                push = (min_gap - gap) / 2.0
                ys[i]     -= push
                ys[i + 1] += push
                moved = True
        if not moved:
            break
    for i, l in enumerate(labels):
        l["y_label"] = float(ys[i])
    return labels


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_frontier_health(health_scores, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8),
                             sharey=True, constrained_layout=True)

    label_gap = 0.07  # min vertical separation in data coords

    for panel_idx, dlc in enumerate(DLC_VALUES):
        ax = axes[panel_idx]

        # Collect pending labels grouped by k for collision avoidance
        pending_labels = {}  # k -> [{"y_data", "y_label", "color", "text"}, ...]

        for display_name, _, color, marker in FRONTIER_MODELS:
            if display_name not in health_scores:
                continue
            scores = health_scores[display_name]

            ks    = sorted(k for (k, d) in scores if d == dlc and k in FRONTIER_K_COMPARE)
            if not ks:
                continue
            means = np.array([scores[(k, dlc)]["mean"] for k in ks])
            ses   = np.array([scores[(k, dlc)]["se"]   for k in ks])

            ax.plot(ks, means, color=color, marker=marker,
                    markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                    linewidth=1.6, label=display_name, zorder=4)
            ax.fill_between(ks, means - ses, means + ses,
                            alpha=0.15, color=color, zorder=2)

            for ki, mi, si, ni in zip(
                ks, means, ses,
                [scores[(k, dlc)]["n"] for k in ks],
            ):
                lbl = f"{mi:.2f}"
                if ni > 1 and si > 0.005:
                    lbl += f"\n\u00b1{si:.2f}"
                pending_labels.setdefault(ki, []).append({
                    "y_data":  mi,
                    "y_label": mi,
                    "color":   color,
                    "text":    lbl,
                })

        # Resolve overlaps per k column and draw
        for ki, items in pending_labels.items():
            items.sort(key=lambda l: l["y_data"])
            _spread_labels(items, label_gap)
            for item in items:
                displaced = abs(item["y_label"] - item["y_data"]) > 0.005
                if displaced:
                    ax.annotate(
                        "", xy=(ki, item["y_data"]),
                        xytext=(ki, item["y_label"] + 0.01),
                        arrowprops=dict(arrowstyle="-", color=item["color"],
                                        lw=0.5, alpha=0.5),
                        zorder=5,
                    )
                ax.text(ki, item["y_label"] + 0.015, item["text"],
                        ha="center", va="bottom",
                        fontsize=6.5, color=item["color"], zorder=6)

        ax.axhline(0.5, color="0.70", lw=0.7, ls="--", zorder=0)
        ax.set_title(f"dlc = {dlc}", fontweight="bold")
        ax.set_xlabel("Stabilizing firms ($k$)")
        ax.set_xticks(sorted(FRONTIER_K_COMPARE))
        ax.set_xlim(-0.35, 5.35)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)
        ax.grid(axis="x", linewidth=0.0)

        if panel_idx == 0:
            ax.set_ylabel("Health score $S$")

    handles = [
        mlines.Line2D([], [], color=c, marker=m, markersize=6,
                      markeredgecolor="white", markeredgewidth=0.6,
                      label=name)
        for name, _, c, m in FRONTIER_MODELS
        if name in health_scores
    ]
    axes[-1].legend(handles=handles, loc="lower right",
                    fontsize=8, framealpha=0.92)

    fig.suptitle("Experiment 1: Frontier Model Health Profiles",
                 fontweight="bold", fontsize=11)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path)
    print(f"\nSaved -> {output_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Frontier model health profile comparison.")
    ap.add_argument("--good",    default="food")
    ap.add_argument("--output",  default=DEFAULT_OUTPUT)
    ap.add_argument("--logs-dir", default="logs/",
                    help="Base logs directory (fallback for raw state files).")
    ap.add_argument("--fig-exp1-dir", default=_FIG_EXP1_DIR,
                    help="Base exp1 figure directory containing model cache subdirs.")
    ap.add_argument("--workers", type=int, default=8, help="(unused, kept for interface compat)")
    ap.add_argument("--model",   default="", help="(unused, kept for interface compat)")
    args = ap.parse_args()

    fig_exp1_dir = os.path.abspath(args.fig_exp1_dir)

    all_model_psd = {}
    for display_name, dir_candidates, _, _ in FRONTIER_MODELS:
        psd, uc = load_model_data(dir_candidates, args.good,
                                  fig_exp1_dir, args.logs_dir)
        if psd is not None:
            all_model_psd[display_name] = (psd, uc)
        else:
            print(f"  WARNING: no data for {display_name}, skipping.", flush=True)

    if not all_model_psd:
        print("No frontier model data found. "
              "Run exp1_run_all.py for each frontier model first.", flush=True)
        sys.exit(1)

    print(f"\nModels loaded: {list(all_model_psd.keys())}", flush=True)
    health_scores = compute_health_scores(all_model_psd)

    for name, scores in health_scores.items():
        print(f"\n  {name}:")
        for (k, dlc), s in sorted(scores.items()):
            print(f"    k={k} dlc={dlc}: S={s['mean']:.3f} +/-{s['se']:.3f}  (n={s['n']})")

    plot_frontier_health(health_scores, args.output)


if __name__ == "__main__":
    main()
