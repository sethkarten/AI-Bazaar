"""
exp1_health_vs_size.py -- Market health score vs. model size (dense open-weight models).

Scatter plot with three series per model (k=0, k=3, k=5) at dlc=3:
  x-axis: model parameter count (log scale, in billions)
  y-axis: composite market health score
  color:  developer (Okabe-Ito)
  marker: k value (triangle=k=0, circle=k=3, square=k=5)
  error bars: min/max across seeds (k=3 and k=5 have 3 seeds; k=0 has seed=8 only)

Health score normalization is global across all models AND all k levels.

Models: dense open-weight models with include=1 from EAS_vs_MODEL_SIZE.md.
Setting: dlc=3, k in {0, 3, 5}.

Data loading priority per model/k:
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder
from exp1_cache import get_cache_path, load_cache_data

# ── Model registry (dense, include=1, from EAS_vs_MODEL_SIZE.md) ──────────
# (display_name, params_B, openrouter_model_id, developer)
MODELS = [
    ("Llama 3.2 3B",        3.0,   "meta-llama/llama-3.2-3b-instruct",               "Meta"),
    ("Gemma 3 4B",          4.0,   "google/gemma-3-4b-it",                            "Google"),
    ("Mistral 7B",          7.3,   "mistralai/mistral-7b-instruct",                   "Mistral"),
    ("Llama 3.1 8B",        8.0,   "meta-llama/llama-3.1-8b-instruct",               "Meta"),
    ("Qwen3 8B",            8.2,   "qwen/qwen3-8b",                                   "Alibaba"),
    ("Gemma 3 12B",         12.0,  "google/gemma-3-12b-it",                           "Google"),
    ("Phi-4",               14.0,  "microsoft/phi-4",                                 "Microsoft"),
    ("DS-R1-D 14B",         14.0,  "deepseek/deepseek-r1-distill-qwen-14b",           "DeepSeek"),
    ("Mistral S 24B",       24.0,  "mistralai/mistral-small-3.1-24b-instruct",        "Mistral"),
    ("Gemma 3 27B",         27.0,  "google/gemma-3-27b-it",                           "Google"),
    ("OLMo 2 32B",          32.0,  "allenai/olmo-2-32b-instruct",                     "Allen AI"),
    ("OLMo 3.1 32B",        32.0,  "allenai/olmo-3.1-32b-think",                      "Allen AI"),
    ("DS-R1-D 32B",         32.0,  "deepseek/deepseek-r1-distill-qwen-32b",           "DeepSeek"),
    ("Llama 3.3 70B",       70.0,  "meta-llama/llama-3.3-70b-instruct",               "Meta"),
    ("Llama 3.1 70B",       70.0,  "meta-llama/llama-3.1-70b-instruct",               "Meta"),
    ("DS-R1-D 70B",         70.0,  "deepseek/deepseek-r1-distill-llama-70b",          "DeepSeek"),
    ("Nemotron 70B",        70.0,  "nvidia/llama-3.1-nemotron-70b-instruct",          "NVIDIA"),
    ("Qwen2.5 72B",         72.0,  "qwen/qwen-2.5-72b-instruct",                      "Alibaba"),
    ("Llama 3.1 405B",      405.0, "meta-llama/llama-3.1-405b-instruct",              "Meta"),
    ("Hermes 3 405B",       405.0, "nousresearch/hermes-3-llama-3.1-405b",            "NousResearch"),
    ("Hermes 4 405B",       405.0, "nousresearch/hermes-4-405b",                      "NousResearch"),
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

# k levels to plot
TARGET_K_VALUES = [0, 3, 5]
TARGET_DLC      = 3
SEEDS           = [8, 16, 64]

# Marker style per k
K_MARKERS = {0: '^', 3: 'o', 5: 's'}
K_SIZES   = {0: 28,  3: 42,  5: 32}
K_ALPHAS  = {0: 0.7, 3: 1.0, 5: 0.88}
K_LABELS  = {0: '$k=0$ (no stabilization)',
             3: '$k=3$',
             5: '$k=5$ (full stabilization)'}


# Heatmap grid axes (for cache extraction)
HEATMAP_N_STAB_ALL = [0, 1, 2, 3, 4, 5]
HEATMAP_DLC_ALL    = [1, 3, 5]
HEATMAP_COL        = HEATMAP_DLC_ALL.index(TARGET_DLC)          # 1
HEATMAP_ROWS       = {k: HEATMAP_N_STAB_ALL.index(k)
                      for k in TARGET_K_VALUES}                  # {0:0, 3:3, 5:5}

_SCRIPT_DIR    = os.path.dirname(__file__)
_FIG_EXP1_DIR  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp1"))
DEFAULT_OUTPUT = os.path.join(_FIG_EXP1_DIR, "exp1_health_vs_size.pdf")
_CACHE_PATH    = os.path.join(_FIG_EXP1_DIR, "data", "exp1_health_vs_size_metrics.json")

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


# ── Helpers ────────────────────────────────────────────────────────────────

def model_key(or_id):
    return or_id.replace("/", "_")


def resolve_run_dir_for_k(logs_dir, key, k, seed):
    """Return run directory for one (model, k, seed) combination."""
    if k == 0:
        # Baseline: seed=8 only, no seed suffix
        path = os.path.join(logs_dir, f"exp1_{key}_baseline")
        return path if os.path.isdir(path) else None
    path = os.path.join(logs_dir, f"exp1_{key}_stab_{k}_dlc{TARGET_DLC}_seed{seed}")
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


# ── Data loading ───────────────────────────────────────────────────────────

def load_model_data(or_id, logs_base, good):
    """
    Load per-seed metrics for k in {0, 3, 5} at dlc=3 for one model.
    Returns dict {k: [metric_dict, ...]} — empty list if no data for that k.

    Priority per k:
      1. Heatmap cache (paper/fig/exp1/exp1_{key}/data/exp1_heatmap_{good}.json)
      2. Raw state files
    """
    key      = model_key(or_id)
    logs_dir = os.path.join(logs_base, f"exp1_{key}")

    # Try heatmap cache once for all k values
    heatmap_cache = os.path.join(
        _FIG_EXP1_DIR, f"exp1_{key}", "data",
        f"exp1_heatmap_{good}.json"
    )
    psd_raw   = {}
    unit_cost = 1.0
    if os.path.isfile(heatmap_cache):
        try:
            raw = load_cache_data(heatmap_cache)
            if "per_seed_data" in raw:
                psd_raw   = raw["per_seed_data"]
                unit_cost = raw.get("unit_cost", 1.0)
        except Exception as e:
            print(f"  [{key}] Heatmap cache load failed: {e}", flush=True)

    result = {}
    for k in TARGET_K_VALUES:
        cache_key = f"{HEATMAP_ROWS[k]},{HEATMAP_COL}"
        if cache_key in psd_raw:
            cell    = psd_raw[cache_key]
            n_seeds = len(cell["bankruptcy_rate"])
            records = [
                {
                    "bankruptcy_rate": cell["bankruptcy_rate"][s],
                    "final_avg_price": cell["final_avg_price"][s],
                    "price_std":       cell["price_std"][s],
                    "unit_cost":       unit_cost,
                }
                for s in range(n_seeds)
            ]
            result[k] = records
            print(f"  [{key}] k={k}: {n_seeds} seed(s) from cache.", flush=True)
            continue

        # Fallback: raw state files
        seeds_to_try = [8] if k == 0 else SEEDS
        records = []
        for seed in seeds_to_try:
            run_dir = resolve_run_dir_for_k(logs_dir, key, k, seed)
            if run_dir is None:
                continue
            m = compute_metrics_one(run_dir, good)
            if m:
                records.append(m)
        result[k] = records
        if records:
            print(f"  [{key}] k={k}: {len(records)} seed(s) from state files.", flush=True)
        else:
            print(f"  [{key}] k={k}: no data.", flush=True)

    return result


# ── Health score ───────────────────────────────────────────────────────────

def compute_health_scores(all_model_data):
    """
    Compute health scores for all (model, k) combinations.
    Global normalization across all models AND all k values.

    all_model_data: dict[display_name -> {k -> [metric_dicts]}]
    Returns: dict[display_name -> {k -> {"mean", "lo", "hi", "n"} or None}]
    """
    # Global normalization values
    all_pstd, all_pdev = [], []
    for k_data in all_model_data.values():
        for records in k_data.values():
            for r in records:
                uc = r["unit_cost"]
                all_pstd.append(r["price_std"])
                all_pdev.append(abs(r["final_avg_price"] / uc - 1.0) if uc > 0 else 0.0)

    global_max_pstd = max(all_pstd) if all_pstd else 1.0
    global_max_pdev = max(all_pdev) if all_pdev else 1.0

    results = {}
    for name, k_data in all_model_data.items():
        results[name] = {}
        for k, records in k_data.items():
            if not records:
                results[name][k] = None
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
            results[name][k] = {
                "mean": float(np.mean(per_seed)),
                "std":  float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                "n":    len(per_seed),
            }
    return results


# ── Jitter ─────────────────────────────────────────────────────────────────

def jitter_x(params_B_list):
    """Spread models at the same param count in log space."""
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, p in enumerate(params_B_list):
        groups[p].append(idx)
    jittered = list(params_B_list)
    for p, idxs in groups.items():
        n = len(idxs)
        if n == 1:
            continue
        spread  = 0.06 * np.log10(max(p, 1))
        offsets = np.linspace(-spread, spread, n)
        for i, idx in enumerate(idxs):
            jittered[idx] = 10 ** (np.log10(p) + offsets[i])
    return jittered


# ── Main ───────────────────────────────────────────────────────────────────

def _save_raw_cache(data: dict) -> None:
    """Persist raw metric records to a JSON cache to avoid re-reading state files."""
    os.makedirs(os.path.dirname(os.path.abspath(_CACHE_PATH)), exist_ok=True)
    # Serialize: outer key = display_name, inner key = str(k)
    serialized = {
        name: {str(k): records for k, records in k_data.items()}
        for name, k_data in data.items()
    }
    with open(_CACHE_PATH, "w") as f:
        json.dump(serialized, f)
    print(f"Raw metric cache saved -> {_CACHE_PATH}", flush=True)


def _load_raw_cache() -> dict | None:
    if not os.path.isfile(_CACHE_PATH):
        return None
    try:
        with open(_CACHE_PATH) as f:
            raw = json.load(f)
        return {
            name: {int(k): records for k, records in k_data.items()}
            for name, k_data in raw.items()
        }
    except Exception as e:
        print(f"Cache load failed ({e}), rebuilding.", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description="Health score vs. model size scatter.")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good",    default="food")
    ap.add_argument("--output",  default=DEFAULT_OUTPUT)
    ap.add_argument("--force",   action="store_true",
                    help="Ignore metric cache and reload from state files / heatmap caches.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel workers for data loading (default: 8).")
    args = ap.parse_args()

    # ── Load data (with caching) ─────────────────────────────────────────
    cached = None if args.force else _load_raw_cache()
    if cached is not None:
        print(f"Using cached metrics: {_CACHE_PATH}", flush=True)
        all_model_data = cached
    else:
        all_model_data = {}

        def _load_one(entry):
            display_name, _, or_id, _ = entry
            return display_name, load_model_data(or_id, args.logs_dir, args.good)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_load_one, entry): entry[0] for entry in MODELS}
            for fut in as_completed(futures):
                display_name, k_data = fut.result()
                all_model_data[display_name] = k_data

        _save_raw_cache(all_model_data)

    # ── Health scores (global normalization) ─────────────────────────────
    scores = compute_health_scores(all_model_data)

    # ── Build model list (keep models that have data for at least one k) ─
    plot_models = [
        (name, p, dev)
        for name, p, _, dev in MODELS
        if any(scores[name].get(k) is not None for k in TARGET_K_VALUES)
    ]

    if not plot_models:
        print("No data found for any model. Check --logs-dir.", flush=True)
        return

    names  = [m[0] for m in plot_models]
    params = [m[1] for m in plot_models]
    devs   = [m[2] for m in plot_models]
    x_base = jitter_x(params)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)

    seen_devs = set()
    for i, (name, x_b, dev) in enumerate(zip(names, x_base, devs)):
        color  = DEV_COLORS.get(dev, "#999999")
        k_scores = scores[name]

        # Collect valid (k, score) pairs for connecting line
        valid_ks = [(k, k_scores[k]) for k in TARGET_K_VALUES if k_scores.get(k) is not None]
        if not valid_ks:
            continue

        # Connecting line (vertical — all k at same x)
        line_ys = [s["mean"] for _, s in valid_ks]
        ax.plot([x_b] * len(valid_ks), line_ys,
                color=color, lw=0.6, alpha=0.35, zorder=2)

        for k, s in valid_ks:
            mean = s["mean"]

            # Error bar (only meaningful when n > 1)
            if s["n"] > 1 and s["std"] > 0:
                ax.plot([x_b, x_b], [mean - s["std"], mean + s["std"]],
                        color=color, lw=0.9, alpha=0.5, zorder=3)

            label = dev if (dev not in seen_devs and k == 3) else "_nolegend_"
            seen_devs.add(dev)
            ax.scatter([x_b], [mean],
                       s=K_SIZES[k], marker=K_MARKERS[k],
                       color=color, edgecolors="white", linewidths=0.5,
                       alpha=K_ALPHAS[k], zorder=4, label=label)

        # Model name label just above the highest point
        top_y = max(s["mean"] + s["std"] for _, s in valid_ks)
        ax.annotate(name, xy=(x_b, top_y), xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=7, ha="center", va="bottom",
                    rotation=90, color="0.3")

    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (B)")
    ax.set_ylabel("Health score (dlc=3)")
    ax.set_ylim(-0.05, 1.55)
    ax.set_xlim(2, 600)

    xticks = [3, 4, 7, 8, 12, 14, 24, 27, 32, 70, 72, 100, 200, 405]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t}B" for t in xticks], rotation=45, ha="right", fontsize=7)

    ax.axhline(0.0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.axhline(1.0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)
    ax.grid(axis="x", which="major", linewidth=0.0)

    # Legend: developer colors
    dev_handles = [
        mpatches.Patch(color=DEV_COLORS.get(dev, "#999999"), label=dev)
        for dev in sorted({m[2] for m in plot_models})
    ]
    leg1 = ax.legend(handles=dev_handles, loc="lower left", fontsize=7.5,
                     framealpha=0.9, ncol=2, title="Developer")

    # Legend: k marker shapes
    k_handles = [
        mlines.Line2D([], [], color="0.4", marker=K_MARKERS[k], linestyle="None",
                      markersize=5, label=K_LABELS[k])
        for k in TARGET_K_VALUES
    ]
    ax.legend(handles=k_handles, loc="lower right", fontsize=7.5,
              framealpha=0.9, title="Stabilizing firms")
    ax.add_artist(leg1)

    n_shown = len(plot_models)
    n_total = len(MODELS)
    ax.set_title(
        f"Market Health Score vs. Model Size — dense open-weight models"
        f" (dlc=3, k={{0,3,5}}) [{n_shown}/{n_total} models with data]",
        fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
