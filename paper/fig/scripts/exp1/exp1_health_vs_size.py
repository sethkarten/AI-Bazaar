"""
exp1_health_vs_size.py -- Market health score vs. model size (dense open-weight models).

Single scatter plot:
  x-axis: model parameter count (log scale, in billions)
  y-axis: composite market health score at dlc=3, k=3 (mean across seeds 8/16/64)
  error bars: min/max across seeds
  color: developer

Models: dense open-weight models with include=1 from EAS_vs_MODEL_SIZE.md.
Setting: dlc=3, k=3 (exp1_stab_3_dlc3_seed* runs).

Data loading priority per model:
  1. Heatmap cache written by exp1_run_all (paper/fig/exp1/{src}/data/exp1_heatmap_food.json)
  2. Fallback: compute from raw state files

Output: paper/fig/exp1/exp1_health_vs_size.pdf

Usage:
    python paper/fig/scripts/exp1/exp1_health_vs_size.py [--logs-dir logs/] [--good food]
"""

import argparse
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
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder
from exp1_cache import get_cache_path, is_cache_fresh, save_cache, load_cache_data

# ── Model registry (dense, include=1, from EAS_vs_MODEL_SIZE.md) ──────────
# (display_name, params_B, openrouter_model_id, developer)
MODELS = [
    ("Llama 3.2 3B",         3.0,   "meta-llama/llama-3.2-3b-instruct",               "Meta"),
    ("Gemma 3 4B",           4.0,   "google/gemma-3-4b-it",                             "Google"),
    ("Mistral 7B",           7.3,   "mistralai/mistral-7b-instruct",                   "Mistral"),
    ("Llama 3.1 8B",         8.0,   "meta-llama/llama-3.1-8b-instruct",               "Meta"),
    ("Qwen3 8B",             8.2,   "qwen/qwen3-8b",                                   "Alibaba"),
    ("Gemma 3 12B",          12.0,  "google/gemma-3-12b-it",                           "Google"),
    ("Phi-4",                14.0,  "microsoft/phi-4",                                 "Microsoft"),
    ("DS-R1-D 14B",          14.0,  "deepseek/deepseek-r1-distill-qwen-14b",           "DeepSeek"),
    ("Mistral S 24B",        24.0,  "mistralai/mistral-small-3.1-24b-instruct",        "Mistral"),
    ("Gemma 3 27B",          27.0,  "google/gemma-3-27b-it",                           "Google"),
    ("OLMo 2 32B",           32.0,  "allenai/olmo-2-32b-instruct",                     "Allen AI"),
    ("OLMo 3.1 32B",         32.0,  "allenai/olmo-3.1-32b-think",                      "Allen AI"),
    ("DS-R1-D 32B",          32.0,  "deepseek/deepseek-r1-distill-qwen-32b",           "DeepSeek"),
    ("Llama 3.3 70B",        70.0,  "meta-llama/llama-3.3-70b-instruct",               "Meta"),
    ("Llama 3.1 70B",        70.0,  "meta-llama/llama-3.1-70b-instruct",               "Meta"),
    ("DS-R1-D 70B",          70.0,  "deepseek/deepseek-r1-distill-llama-70b",          "DeepSeek"),
    ("Nemotron 70B",         70.0,  "nvidia/llama-3.1-nemotron-70b-instruct",          "NVIDIA"),
    ("Qwen2.5 72B",          72.0,  "qwen/qwen-2.5-72b-instruct",                      "Alibaba"),
    ("Llama 3.1 405B",       405.0, "meta-llama/llama-3.1-405b-instruct",              "Meta"),
    ("Hermes 3 405B",        405.0, "nousresearch/hermes-3-llama-3.1-405b",            "NousResearch"),
    ("Hermes 4 405B",        405.0, "nousresearch/hermes-4-405b",                      "NousResearch"),
]

# Developer -> Okabe-Ito color
DEV_COLORS = {
    "Meta":         "#0072B2",
    "Google":       "#009E73",
    "Mistral":      "#E69F00",
    "Alibaba":      "#56B4E9",
    "DeepSeek":     "#D55E00",
    "Microsoft":    "#CC79A7",
    "Allen AI":     "#F0E442",
    "NVIDIA":       "#000000",
    "NousResearch": "#888888",
}

# Target cell: dlc=3, k=3
TARGET_N_STAB = 3
TARGET_DLC    = 3
SEEDS         = [8, 16, 64]

# Heatmap grid layout (for extracting from heatmap cache)
HEATMAP_N_STAB_ALL = [0, 1, 2, 3, 4, 5]
HEATMAP_DLC_ALL    = [1, 3, 5]
HEATMAP_ROW = HEATMAP_N_STAB_ALL.index(TARGET_N_STAB)  # 3
HEATMAP_COL = HEATMAP_DLC_ALL.index(TARGET_DLC)         # 1

_SCRIPT_DIR  = os.path.dirname(__file__)
_FIG_EXP1_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp1"))

DEFAULT_OUTPUT = os.path.join(_FIG_EXP1_DIR, "exp1_health_vs_size.pdf")

# ── rcParams ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.4,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex":        False,
    "pdf.fonttype":       42,
})


# ── Directory helpers ──────────────────────────────────────────────────────

def model_key(or_id):
    """Convert OR model ID to filesystem key (/ -> _)."""
    return or_id.replace("/", "_")


def resolve_run_dir(logs_dir, model, seed):
    """Resolve the run directory for (k=3, dlc=3, seed)."""
    path = os.path.join(logs_dir, f"exp1_{model}_stab_{TARGET_N_STAB}_dlc{TARGET_DLC}_seed{seed}")
    return path if os.path.isdir(path) else None


# ── Metric computation ─────────────────────────────────────────────────────

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


def compute_metrics_one(run_dir, good):
    files = load_states(run_dir)
    if not files:
        return None
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


# ── Data loading per model ─────────────────────────────────────────────────

def load_cell_data(or_id, logs_base, good):
    """
    Load per-seed metrics for (k=3, dlc=3) for one model.
    Priority:
      1. Heatmap cache (paper/fig/exp1/exp1_{key}/data/exp1_heatmap_{good}.json)
      2. Compute from raw state files

    Returns list of dicts (one per available seed), or [] if no data found.
    """
    key      = model_key(or_id)
    model    = key  # same as what resolve_run_dir expects
    logs_dir = os.path.join(logs_base, f"exp1_{key}")

    # ── 1. Heatmap cache ────────────────────────────────────────────────
    heatmap_data_dir = os.path.join(_FIG_EXP1_DIR, f"exp1_{key}", "data")
    heatmap_cache    = get_cache_path(heatmap_data_dir, "exp1_heatmap", good)

    if os.path.isfile(heatmap_cache):
        try:
            raw = load_cache_data(heatmap_cache)
            if "per_seed_data" in raw:
                psd_raw = raw["per_seed_data"]
                cell_key = f"{HEATMAP_ROW},{HEATMAP_COL}"
                if cell_key in psd_raw:
                    cell = psd_raw[cell_key]
                    n_seeds = len(cell["bankruptcy_rate"])
                    records = []
                    for s in range(n_seeds):
                        records.append({
                            "bankruptcy_rate": cell["bankruptcy_rate"][s],
                            "final_avg_price": cell["final_avg_price"][s],
                            "price_std":       cell["price_std"][s],
                            "unit_cost":       raw.get("unit_cost", 1.0),
                        })
                    print(f"  [{key}] Loaded from heatmap cache ({n_seeds} seeds).", flush=True)
                    return records
        except Exception as e:
            print(f"  [{key}] Heatmap cache load failed: {e}", flush=True)

    # ── 2. Compute from raw state files ─────────────────────────────────
    records = []
    for seed in SEEDS:
        run_dir = resolve_run_dir(logs_dir, key, seed)
        if run_dir is None:
            continue
        m = compute_metrics_one(run_dir, good)
        if m:
            records.append(m)

    if records:
        print(f"  [{key}] Computed from state files ({len(records)} seeds).", flush=True)
    else:
        print(f"  [{key}] No data found.", flush=True)
    return records


# ── Health score ───────────────────────────────────────────────────────────

def compute_health_scores(all_records):
    """
    Compute per-seed and mean health scores for all models.
    Global normalization across all models.

    all_records: dict[display_name -> list of metric dicts]
    Returns: dict[display_name -> {"mean": float, "lo": float, "hi": float, "n": int}]
             or None if no data.
    """
    # Gather global max pstd and pdev
    all_pstd, all_pdev = [], []
    for records in all_records.values():
        for r in records:
            uc   = r["unit_cost"]
            all_pstd.append(r["price_std"])
            all_pdev.append(abs(r["final_avg_price"] / uc - 1.0) if uc > 0 else 0.0)

    global_max_pstd = max(all_pstd) if all_pstd else 1.0
    global_max_pdev = max(all_pdev) if all_pdev else 1.0

    results = {}
    for name, records in all_records.items():
        if not records:
            results[name] = None
            continue
        per_seed = []
        for r in records:
            br = r["bankruptcy_rate"]
            uc = r["unit_cost"]
            if br >= 1.0:
                per_seed.append(0.0)
            else:
                s_surv  = 1.0 - br
                s_stab  = 1.0 - r["price_std"] / global_max_pstd if global_max_pstd > 0 else 1.0
                s_price = 1.0 - min(abs(r["final_avg_price"] / uc - 1.0) / global_max_pdev, 1.0) \
                          if global_max_pdev > 0 else 1.0
                per_seed.append((s_surv + s_stab + s_price) / 3.0)
        results[name] = {
            "mean": float(np.mean(per_seed)),
            "lo":   float(np.min(per_seed)),
            "hi":   float(np.max(per_seed)),
            "n":    len(per_seed),
        }
    return results


# ── Jitter helper ──────────────────────────────────────────────────────────

def jitter_x(params_B_list):
    """
    For points at the same (or very close) parameter count, spread them slightly
    on the log scale so they don't overlap.
    Returns list of jittered x values (still in B).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, p in enumerate(params_B_list):
        groups[p].append(idx)

    jittered = list(params_B_list)
    for p, idxs in groups.items():
        n = len(idxs)
        if n == 1:
            continue
        # Spread evenly in log space: factor range ~±10% of log10(p)
        spread = 0.06 * np.log10(max(p, 1))
        offsets = np.linspace(-spread, spread, n)
        for i, idx in enumerate(idxs):
            jittered[idx] = 10 ** (np.log10(p) + offsets[i])
    return jittered


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Health score vs. model size scatter.")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good",    default="food")
    ap.add_argument("--output",  default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    # ── Load per-model data ──────────────────────────────────────────────
    all_records = {}
    for display_name, params_B, or_id, developer in MODELS:
        all_records[display_name] = load_cell_data(or_id, args.logs_dir, args.good)

    # ── Health scores ─────────────────────────────────────────────────────
    scores = compute_health_scores(all_records)

    # ── Build plot arrays (skip models with no data) ──────────────────────
    plot_models = [(name, p, dev)
                   for name, p, _, dev in MODELS
                   if scores.get(name) is not None]

    if not plot_models:
        print("No data found for any model. Check --logs-dir.", flush=True)
        return

    names   = [m[0] for m in plot_models]
    params  = [m[1] for m in plot_models]
    devs    = [m[2] for m in plot_models]
    means   = [scores[n]["mean"] for n in names]
    los     = [scores[n]["lo"]   for n in names]
    his     = [scores[n]["hi"]   for n in names]

    x_vals  = jitter_x(params)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)

    # Plot each point
    seen_devs = set()
    for i, (name, x, dev, mean, lo, hi) in enumerate(
            zip(names, x_vals, devs, means, los, his)):
        color  = DEV_COLORS.get(dev, "#999999")
        label  = dev if dev not in seen_devs else "_nolegend_"
        seen_devs.add(dev)

        # Error bar (min/max)
        ax.plot([x, x], [lo, hi], color=color, lw=1.0, alpha=0.6, zorder=3)

        # Point
        ax.scatter([x], [mean], s=40, color=color, edgecolors="white",
                   linewidths=0.5, zorder=4, label=label)

    # Model name labels — staggered vertically to reduce overlap
    for i, (name, x, mean) in enumerate(zip(names, x_vals, means)):
        y_offset = 0.03 + 0.025 * (i % 2)  # alternate two vertical offsets
        ax.annotate(name, xy=(x, mean), xytext=(0, 6 + 8 * (i % 2)),
                    textcoords="offset points",
                    fontsize=5.5, ha="center", va="bottom",
                    rotation=45, color="#333333", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (B)")
    ax.set_ylabel("Health score (dlc=3, $k$=3)")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(2, 600)

    # x-axis ticks at natural sizes
    xticks = [3, 4, 7, 8, 12, 14, 24, 27, 32, 70, 72, 100, 200, 405]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t}B" for t in xticks], rotation=45, ha="right", fontsize=7)

    ax.axhline(0.0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.axhline(1.0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)
    ax.grid(axis="x", which="major", linewidth=0.0)  # suppress x gridlines (log scale messy)

    # Developer legend
    handles = [
        mpatches.Patch(color=DEV_COLORS.get(dev, "#999999"), label=dev)
        for dev in sorted({m[2] for m in plot_models})
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5,
              framealpha=0.9, ncol=2)

    n_shown = len(plot_models)
    n_total = len(MODELS)
    ax.set_title(
        f"Market Health Score vs. Model Size — dense open-weight models"
        f" (dlc=3, k=3) [{n_shown}/{n_total} models with data]",
        fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
