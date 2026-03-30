"""
exp1_eas_tokens.py -- Token usage per run and episode length for EAS sweep (Exp1 + Exp2).

5-panel figure (bar charts, models sorted by parameter count):
  (A) Exp1 input tokens / run
  (B) Exp1 output tokens / run
  (C) Exp2 input tokens / run
  (D) Exp2 output tokens / run
  (E) Exp1 avg episode length (timesteps / run)  [spans full bottom row]

Each bar = one model; height = mean across all available runs (k × seeds).
Error bars = ±1σ. Color = developer (Okabe-Ito).
Panel titles show grand total across all models and runs.
Missing models silently skipped.

Data sources:
  Exp1 tokens:   logs/exp1_{slug}/**/*_token_usage.json
  Exp2 tokens:   logs/exp2_{slug}/**/*_token_usage.json
  Episode len:   count of valid state_t*.json files per exp1 run dir

Output: paper/fig/exp1/exp1_eas_tokens.pdf

Usage:
    python paper/fig/scripts/exp1/exp1_eas_tokens.py [--logs-dir logs/] [--output ...]
"""

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = [
    ("Llama 3.2 3B",   3.0,   "meta-llama/llama-3.2-3b-instruct",            "Meta"),
    ("Gemma 3 4B",     4.0,   "google/gemma-3-4b-it",                         "Google"),
    ("Mistral 7B",     7.3,   "mistralai/mistral-7b-instruct",                "Mistral"),
    ("Llama 3.1 8B",   8.0,   "meta-llama/llama-3.1-8b-instruct",            "Meta"),
    ("Qwen3 8B",       8.2,   "qwen/qwen3-8b",                                "Alibaba"),
    ("Gemma 3 12B",    12.0,  "google/gemma-3-12b-it",                        "Google"),
    ("Phi-4",          14.0,  "microsoft/phi-4",                              "Microsoft"),
    ("DS-R1-D 14B",    14.0,  "deepseek/deepseek-r1-distill-qwen-14b",        "DeepSeek"),
    ("Mistral S 24B",  24.0,  "mistralai/mistral-small-3.1-24b-instruct",     "Mistral"),
    ("Gemma 3 27B",    27.0,  "google/gemma-3-27b-it",                        "Google"),
    ("OLMo 2 32B",     32.0,  "allenai/olmo-2-32b-instruct",                  "Allen AI"),
    ("OLMo 3.1 32B",   32.0,  "allenai/olmo-3.1-32b-think",                   "Allen AI"),
    ("DS-R1-D 32B",    32.0,  "deepseek/deepseek-r1-distill-qwen-32b",        "DeepSeek"),
    ("Llama 3.3 70B",  70.0,  "meta-llama/llama-3.3-70b-instruct",            "Meta"),
    ("Llama 3.1 70B",  70.0,  "meta-llama/llama-3.1-70b-instruct",            "Meta"),
    ("DS-R1-D 70B",    70.0,  "deepseek/deepseek-r1-distill-llama-70b",       "DeepSeek"),
    ("Nemotron 70B",   70.0,  "nvidia/llama-3.1-nemotron-70b-instruct",       "NVIDIA"),
    ("Qwen2.5 72B",    72.0,  "qwen/qwen-2.5-72b-instruct",                   "Alibaba"),
    ("Llama 3.1 405B", 405.0, "meta-llama/llama-3.1-405b-instruct",           "Meta"),
    ("Hermes 3 405B",  405.0, "nousresearch/hermes-3-llama-3.1-405b",         "NousResearch"),
    ("Hermes 4 405B",  405.0, "nousresearch/hermes-4-405b",                   "NousResearch"),
]

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

# exp2 directories that use a different slug than the standard OR-ID
EXP2_SLUG_OVERRIDES = {
    "mistralai/mistral-7b-instruct": "mistralai_mistral-7b-instruct-v0.1",
}

BAR_W = 0.65

_SCRIPT_DIR   = os.path.dirname(__file__)
_FIG_EXP1_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp1"))
DEFAULT_OUTPUT = os.path.join(_FIG_EXP1_DIR, "exp1_eas_tokens.pdf")

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


# ── Data loading ──────────────────────────────────────────────────────────────

def model_key(or_id: str) -> str:
    return or_id.replace("/", "_")


def load_token_records(logs_dir: str, exp_prefix: str, or_id: str) -> list[dict]:
    if exp_prefix == "exp2" and or_id in EXP2_SLUG_OVERRIDES:
        key = EXP2_SLUG_OVERRIDES[or_id]
    else:
        key = model_key(or_id)
    pattern = os.path.join(logs_dir, f"{exp_prefix}_{key}", "**", "*_token_usage.json")
    records = []
    for tf in glob.glob(pattern, recursive=True):
        try:
            with open(tf) as f:
                j = json.load(f)
            inp = j.get("input_tokens") or j.get("total", {}).get("input_tokens", 0)
            out = j.get("output_tokens") or j.get("total", {}).get("output_tokens", 0)
            if inp > 0:
                records.append({"input": int(inp), "output": int(out)})
        except Exception:
            pass
    return records


def load_episode_lengths(logs_dir: str, or_id: str) -> list[int]:
    """Count valid state_t*.json files in each run dir under logs/exp1_{key}/."""
    key       = model_key(or_id)
    model_dir = os.path.join(logs_dir, f"exp1_{key}")
    lengths   = []
    if not os.path.isdir(model_dir):
        return lengths
    for entry in os.listdir(model_dir):
        run_dir = os.path.join(model_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        valid = [f for f in glob.glob(os.path.join(run_dir, "state_t*.json"))
                 if os.path.getsize(f) > 0]
        if valid:
            lengths.append(len(valid))
    return lengths


def aggregate_tokens(records: list[dict]) -> dict | None:
    if not records:
        return None
    inputs  = [r["input"]  for r in records]
    outputs = [r["output"] for r in records]
    n = len(records)
    return {
        "input_mean":   float(np.mean(inputs)),
        "input_std":    float(np.std(inputs,  ddof=1)) if n > 1 else 0.0,
        "input_total":  int(sum(inputs)),
        "output_mean":  float(np.mean(outputs)),
        "output_std":   float(np.std(outputs, ddof=1)) if n > 1 else 0.0,
        "output_total": int(sum(outputs)),
        "n": n,
    }


def aggregate_episode(lengths: list[int]) -> dict | None:
    if not lengths:
        return None
    n = len(lengths)
    return {
        "mean": float(np.mean(lengths)),
        "std":  float(np.std(lengths, ddof=1)) if n > 1 else 0.0,
        "n":    n,
    }


# ── Drawing ───────────────────────────────────────────────────────────────────

def _errbar(ax, x, mean, std, color):
    cap = BAR_W * 0.3
    ax.plot([x, x], [mean - std, mean + std], color="0.25", lw=0.9, zorder=4)
    ax.plot([x - cap, x + cap], [mean - std, mean - std], color="0.25", lw=0.7, zorder=4)
    ax.plot([x - cap, x + cap], [mean + std, mean + std], color="0.25", lw=0.7, zorder=4)


def draw_token_panel(ax, plot_models, stats, metric, ylabel, title,
                     show_legend=False):
    mean_key  = f"{metric}_mean"
    std_key   = f"{metric}_std"
    total_key = f"{metric}_total"

    grand_total = sum(
        s[total_key] for s in stats.values() if s is not None
    )

    seen_devs = set()
    for i, (name, _, dev) in enumerate(plot_models):
        s = stats.get(name)
        if s is None:
            continue
        color = DEV_COLORS.get(dev, "#999999")
        mean  = s[mean_key]
        std   = s[std_key]

        bar_label = dev if dev not in seen_devs else "_nolegend_"
        seen_devs.add(dev)
        ax.bar(i, mean, width=BAR_W, color=color, alpha=0.85, zorder=3, label=bar_label)

        if s["n"] > 1 and std > 0:
            _errbar(ax, i, mean, std, color)

    ax.set_xticks(range(len(plot_models)))
    ax.set_xticklabels([m[0] for m in plot_models], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(plot_models) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)
    ax.grid(axis="x", linewidth=0.0)

    if metric == "input":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
        total_str = f"total: {grand_total / 1e6:.1f}M"
    else:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
        total_str = f"total: {grand_total / 1e3:.0f}k"

    ax.set_title(total_str, loc="right", fontsize=7.5, color="0.45", style="italic")

    if show_legend:
        handles = [
            mpatches.Patch(color=DEV_COLORS.get(dev, "#999999"), label=dev)
            for dev in sorted({m[2] for m in plot_models})
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=7.5,
                  framealpha=0.9, ncol=2, title="Developer")


def draw_episode_panel(ax, plot_models, ep_stats, title):
    seen_devs = set()
    for i, (name, _, dev) in enumerate(plot_models):
        s = ep_stats.get(name)
        if s is None:
            continue
        color = DEV_COLORS.get(dev, "#999999")
        mean  = s["mean"]
        std   = s["std"]

        bar_label = dev if dev not in seen_devs else "_nolegend_"
        seen_devs.add(dev)
        ax.bar(i, mean, width=BAR_W, color=color, alpha=0.85, zorder=3, label=bar_label)

        if s["n"] > 1 and std > 0:
            _errbar(ax, i, mean, std, color)

    ax.set_xticks(range(len(plot_models)))
    ax.set_xticklabels([m[0] for m in plot_models], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Timesteps / run")
    ax.set_xlim(-0.6, len(plot_models) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)
    ax.grid(axis="x", linewidth=0.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="EAS token usage and episode length bar charts (Exp1 + Exp2).")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--output",   default=DEFAULT_OUTPUT)
    ap.add_argument("--workers",  type=int, default=8)
    args = ap.parse_args()

    def _load(entry):
        name, _, or_id, _ = entry
        e1  = aggregate_tokens(load_token_records(args.logs_dir, "exp1", or_id))
        e2  = aggregate_tokens(load_token_records(args.logs_dir, "exp2", or_id))
        epl = aggregate_episode(load_episode_lengths(args.logs_dir, or_id))
        return name, e1, e2, epl

    exp1_stats: dict = {}
    exp2_stats: dict = {}
    ep_stats:   dict = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_load, entry): entry[0] for entry in MODELS}
        for fut in as_completed(futures):
            name, e1, e2, epl = fut.result()
            exp1_stats[name] = e1
            exp2_stats[name] = e2
            ep_stats[name]   = epl
            print(f"  {name}: exp1={'ok' if e1 else '--'}  "
                  f"exp2={'ok' if e2 else '--'}  "
                  f"ep={'ok' if epl else '--'}", flush=True)

    # Models sorted by param count (MODELS order already is size-sorted)
    plot_models = [
        (name, p, dev)
        for name, p, _, dev in MODELS
        if exp1_stats.get(name) or exp2_stats.get(name) or ep_stats.get(name)
    ]

    n1  = sum(1 for s in exp1_stats.values() if s)
    n2  = sum(1 for s in exp2_stats.values() if s)
    nep = sum(1 for s in ep_stats.values()   if s)

    # Layout: 3 rows — top two rows are 2-col token panels; bottom row is
    # full-width episode-length panel.
    fig = plt.figure(figsize=(14.0, 11.0), constrained_layout=True)
    gs  = fig.add_gridspec(3, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])   # spans both columns

    fig.suptitle("EAS Sweep — Token Usage and Episode Length",
                 fontsize=10, fontweight="bold")

    draw_token_panel(ax_a, plot_models, exp1_stats, "input",
                     "Input tokens / run", f"(A) Exp1 — input  [{n1} models]",
                     show_legend=True)
    draw_token_panel(ax_b, plot_models, exp1_stats, "output",
                     "Output tokens / run", f"(B) Exp1 — output  [{n1} models]")
    draw_token_panel(ax_c, plot_models, exp2_stats, "input",
                     "Input tokens / run", f"(C) Exp2 — input  [{n2} models]")
    draw_token_panel(ax_d, plot_models, exp2_stats, "output",
                     "Output tokens / run", f"(D) Exp2 — output  [{n2} models]")
    draw_episode_panel(ax_e, plot_models, ep_stats,
                       f"(E) Exp1 — avg episode length  [{nep} models]")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
