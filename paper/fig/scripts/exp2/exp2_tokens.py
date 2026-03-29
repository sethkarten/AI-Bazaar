"""
Fig: Experiment 2 Token Usage — input and output tokens across the K × rep_visible sweep.

Two-panel figure:
  (A) Input tokens  — grouped bars, K on x-axis, rep_visible encoded by colour,
                      mean bar height with per-seed dots
  (B) Output tokens — same structure, independent y-scale

K=0 (baseline) and sybil cells both support rep_visible ∈ {True, False} with
single or multiple seeds depending on what exists in the run directories.
Missing run dirs are silently skipped.

Directory naming (mirrors exp2.py):
  logs/{name_prefix}/{name_prefix}_k0_rep{1|0}_seed{seed}
  logs/{name_prefix}/{name_prefix}_k{k}_rep1_seed{seed}   (reputation visible)
  logs/{name_prefix}/{name_prefix}_k{k}_rep0_seed{seed}   (reputation hidden)
where name_prefix = exp2_{model_slug}  (e.g. exp2_gemini-2.5-flash).

Legacy flat layout (early prototype runs directly under logs/):
  logs/exp2_k0_rep{1|0}_seed{seed}
  logs/exp2_baseline_seed{seed}
  logs/exp2_k{k}_rep{1|0}_seed{seed}
is also checked as a fallback.

Usage:
    python paper/fig/scripts/exp2/exp2_tokens.py
    python paper/fig/scripts/exp2/exp2_tokens.py \\
        --logs-dir logs/ --name-prefix exp2_gemini-2.5-flash
    python paper/fig/scripts/exp2/exp2_tokens.py --force
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
import matplotlib.ticker as mticker
import numpy as np

# Repo root: 5 levels up from paper/fig/scripts/exp2/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from exp2_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data, infer_name_prefix

# ── rcParams ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.5,
    "lines.markersize":   5,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "grid.color":         "gray",
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":        False,
    "pdf.fonttype":       42,
})

# ── Experiment matrix (must match exp2.py) ─────────────────────────────────
K_VALUES   = [0, 3, 6, 9]
SEEDS      = [8, 16, 64]
RHO_MIN    = 0.3

# ── Colour palette (Okabe-Ito) — rep_visible encoded by colour ─────────────
REP_COLORS  = {True: "#0072B2", False: "#D55E00"}   # blue = visible, vermillion = hidden
REP_LABELS  = {True: "Rep. visible (rep1)", False: "Rep. hidden (rep0)"}

# ── Bar layout constants ───────────────────────────────────────────────────
BAR_W        = 0.30
REP_OFFSETS  = {True: -0.18, False: 0.18}   # two bars per K group
GROUP_STEP   = 1.0

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "exp2", "exp2_tokens.pdf"
)


# ── Directory resolution ───────────────────────────────────────────────────

def resolve_run_dir(logs_dir: str, name_prefix: str, k: int, rep_visible: bool, seed: int) -> str | None:
    """Return the run directory path if it exists, else None.

    Checks canonical layout first (logs/{name_prefix}/{run_name}/),
    then legacy flat layout (logs/{run_name}/).
    """
    rep_tag = "rep1" if rep_visible else "rep0"
    run_names = (
        [f"{name_prefix}_k0_{rep_tag}_seed{seed}", f"{name_prefix}_baseline_seed{seed}"]
        if k == 0
        else [f"{name_prefix}_k{k}_{rep_tag}_seed{seed}"]
    )
    for run_name in run_names:
        # Canonical layout
        canonical = os.path.join(logs_dir, name_prefix, run_name)
        if os.path.isdir(canonical):
            return canonical

        # Legacy flat layout (old prototype runs directly under logs/)
        flat = os.path.join(logs_dir, run_name)
        if os.path.isdir(flat):
            return flat

    return None


def collect_run_dirs(logs_dir: str, name_prefix: str) -> list[str]:
    dirs = []
    for k in K_VALUES:
        rep_opts = [True, False]
        for rv in rep_opts:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, name_prefix, k, rv, seed)
                if d:
                    dirs.append(d)
    return dirs


# ── Data loading ───────────────────────────────────────────────────────────

def load_token_file(run_dir: str) -> dict | None:
    """Return {"input": int, "output": int} from the run's *_token_usage.json, or None."""
    matches = glob.glob(os.path.join(run_dir, "*_token_usage.json"))
    if not matches:
        return None
    try:
        with open(matches[0]) as fh:
            j = json.load(fh)
        return {
            "input":  int(j.get("input_tokens",  0)),
            "output": int(j.get("output_tokens", 0)),
        }
    except Exception:
        return None


def load_all_tokens(logs_dir: str, name_prefix: str) -> dict:
    """Return nested dict: data[(k, rep_visible)] = [{"input": int, "output": int}, ...]."""
    data: dict = {}
    for k in K_VALUES:
        rep_opts = [True, False]
        for rv in rep_opts:
            records = []
            for seed in SEEDS:
                run_dir = resolve_run_dir(logs_dir, name_prefix, k, rv, seed)
                if run_dir is None:
                    continue
                rec = load_token_file(run_dir)
                if rec is not None:
                    records.append(rec)
            if records:
                data[(k, rv)] = records
    return data


# ── Cache serialisation ────────────────────────────────────────────────────

def _serialize(data: dict) -> dict:
    return {f"{k},{int(rv)}": records for (k, rv), records in data.items()}


def _deserialize(raw: dict) -> dict:
    out = {}
    for key, records in raw.items():
        k_str, rv_str = key.split(",")
        out[(int(k_str), bool(int(rv_str)))] = records
    return out


# ── Drawing ────────────────────────────────────────────────────────────────

def _group_center(g_idx: int) -> float:
    return g_idx * GROUP_STEP


def draw_panel(ax, data: dict, key: str, ylabel: str, panel_label: str,
               total: int = 0, show_legend: bool = False):
    """Draw one panel (key = 'input' or 'output')."""
    for g_idx, k in enumerate(K_VALUES):
        cx = _group_center(g_idx)
        rep_opts = [True, False]
        for rv in rep_opts:
            if (k, rv) not in data:
                continue
            records  = data[(k, rv)]
            vals     = [r[key] for r in records]
            mean_val = float(np.mean(vals))
            color    = REP_COLORS[rv]
            x        = cx + REP_OFFSETS[rv]

            ax.bar(
                x, mean_val,
                width=BAR_W * 0.88,
                color=color,
                alpha=0.82,
                zorder=3,
                label=REP_LABELS[rv] if g_idx == 0 else "_nolegend_",
            )

            for v in vals:
                ax.scatter(
                    x, v,
                    s=14,
                    color=color,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=5,
                )

    tick_xs = [_group_center(i) for i in range(len(K_VALUES))]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels([f"$K={k}$" for k in K_VALUES])
    ax.set_xlabel("Sybil cluster size $K$")
    ax.set_xlim(tick_xs[0] - 0.55, tick_xs[-1] + 0.55)

    if key == "input":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M")
        )
    else:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k")
        )

    ax.set_ylabel(ylabel)
    ax.set_title(panel_label, loc="left", fontweight="normal")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linewidth=0.5, color="0.85", zorder=0)
    ax.set_axisbelow(True)

    if total > 0:
        total_str = f"total: {total / 1e6:.1f} M"
        ax.set_title(total_str, loc="right", fontsize=7.5, color="0.45", style="italic")

    if show_legend:
        handles = [
            mpatches.Patch(color=REP_COLORS[rv], label=REP_LABELS[rv], alpha=0.82)
            for rv in [True, False]
        ]
        ax.legend(handles=handles, loc="upper left", framealpha=0.9)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Exp2 token usage figure.")
    ap.add_argument(
        "--logs-dir", default="logs/",
        help="Root logs directory (default: logs/).",
    )
    ap.add_argument(
        "--name-prefix", default=None,
        help=(
            "Experiment name prefix, e.g. exp2_gemini-2.5-flash. "
            "Auto-detected from the first exp2_* subdirectory in --logs-dir if omitted."
        ),
    )
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="Ignore cache and rebuild.")
    # Accepted for compatibility with exp2_run_all.py; not used by this script.
    ap.add_argument("--good", default="car")
    args = ap.parse_args()

    # Auto-detect name_prefix
    name_prefix = args.name_prefix
    if name_prefix is None:
        name_prefix = infer_name_prefix(args.logs_dir)
        print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_run_dirs(args.logs_dir, name_prefix)
    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_tokens", "tokens")

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, "tokens"):
        print(f"Using cached data: {cache_path}", flush=True)
        data = _deserialize(load_cache_data(cache_path))
    else:
        print(f"Loading runs from: {args.logs_dir}  (name_prefix={name_prefix})", flush=True)
        data = load_all_tokens(args.logs_dir, name_prefix)
        save_cache(cache_path, _serialize(data), args.logs_dir, "tokens")
        print(f"Cached data: {cache_path}", flush=True)

    n_cells = len(data)
    print(f"Cells with token data: {n_cells}", flush=True)
    for (k, rv), records in sorted(data.items()):
        tag = "rep1" if rv else "rep0"
        print(f"  K={k} {tag}: {len(records)} seed(s)", flush=True)

    total_input  = sum(r["input"]  for records in data.values() for r in records)
    total_output = sum(r["output"] for records in data.values() for r in records)
    print(f"Experiment total input tokens:  {total_input:,}", flush=True)
    print(f"Experiment total output tokens: {total_output:,}", flush=True)

    fig, (ax_in, ax_out) = plt.subplots(
        1, 2,
        figsize=(7.0, 3.2),
        constrained_layout=True,
    )
    fig.suptitle("Token Usage per Run", fontsize=10, fontweight="bold")

    draw_panel(ax_in,  data, key="input",  ylabel="Input tokens",
               panel_label="(A) Input tokens",  total=total_input,  show_legend=True)
    draw_panel(ax_out, data, key="output", ylabel="Output tokens",
               panel_label="(B) Output tokens", total=total_output, show_legend=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
