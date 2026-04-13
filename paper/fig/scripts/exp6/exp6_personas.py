"""
exp6_personas.py — Fig C: Consumer Heterogeneity at High Discovery Limit.

Two-panel figure at dlc=5:
  Left:  Mixed personas vs. homogeneous PRICE_HAWK baseline — b_r vs k.
  Right: Per-persona breakdown at k=0 — bar chart of b_r.

Expected run naming convention:
  logs/exp6_{slug}/exp6_{slug}_homogeneous_stab_{k}_dlc5_seed{seed}/
  logs/exp6_{slug}/exp6_{slug}_mixed_stab_{k}_dlc5_seed{seed}/
  logs/exp6_{slug}/exp6_{slug}_persona_{name}_k0_dlc5_seed{seed}/
    name ∈ {price_hawk, loyal, small_biz, popular, variety}

Usage:
    python paper/fig/scripts/exp6/exp6_personas.py [--llm gemini-2.5-flash] \\
        [--logs-dir logs/] [--list] [--output ...]
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
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "exp1"))
from exp1_paths import resolve_run_dir as _exp1_resolve


def llm_filesystem_slug(llm: str) -> str:
    """Convert a model name to a filesystem-safe slug."""
    s = llm.strip()
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = s.replace(":", "_")
    return s or "model"

K_VALUES  = [0, 3, 5]
SEEDS     = [8, 16, 64]
PERSONAS  = ["price_hawk", "loyal", "small_biz", "popular", "variety"]
PERSONA_LABELS = {
    "price_hawk": "PRICE_HAWK",
    "loyal":      "LOYAL",
    "small_biz":  "SMALL_BIZ",
    "popular":    "POPULAR",
    "variety":    "VARIETY",
}
# Okabe-Ito colors for 5 personas
PERSONA_COLORS = {
    "price_hawk": "#0072B2",
    "loyal":      "#E69F00",
    "small_biz":  "#009E73",
    "popular":    "#D55E00",
    "variety":    "#CC79A7",
}

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


def compute_br(run_dir):
    states = load_states(run_dir)
    if not states:
        return None
    first = sum(1 for f in states[0].get("firms", []) if f.get("in_business"))
    last  = sum(1 for f in states[-1].get("firms", []) if f.get("in_business"))
    if first == 0:
        return None
    return 1.0 - last / first


def _avg_price_per_step(states):
    """Return list of mean prices across active firms for each timestep."""
    series = []
    for s in states:
        prices = []
        for firm in s.get("firms", []):
            if firm.get("in_business"):
                prices.extend(v for v in firm.get("prices", {}).values() if v is not None)
        if prices:
            series.append(float(np.mean(prices)))
    return series


def compute_price_volatility(run_dir):
    """Std of per-timestep mean price across all timesteps."""
    states = load_states(run_dir)
    if not states:
        return None
    series = _avg_price_per_step(states)
    return float(np.std(series)) if len(series) > 1 else None


def compute_final_avg_price(run_dir):
    """Mean price across active firms in the final timestep."""
    states = load_states(run_dir)
    if not states:
        return None
    for s in reversed(states):
        prices = []
        for firm in s.get("firms", []):
            if firm.get("in_business"):
                prices.extend(v for v in firm.get("prices", {}).values() if v is not None)
        if prices:
            return float(np.mean(prices))
    return None


def resolve_condition_dir(logs_dir, slug, condition, k, seed):
    """Resolve run dir for a given condition string and k level.

    Tries the condition-qualified name first, then falls back to the plain
    name (no condition tag) for the mixed condition.  Plain-named runs are
    the mixed-persona runs (consumers may not carry an explicit persona label
    in the state files but were configured with mixed personas).
    """
    base = os.path.join(logs_dir, f"exp6_{slug}")
    # Primary: condition-qualified name
    d = os.path.join(base, f"exp6_{slug}_{condition}_stab_{k}_dlc5_seed{seed}")
    if os.path.isdir(d):
        return d
    # Fallback for mixed: plain name without condition tag
    if condition == "mixed":
        d = os.path.join(base, f"exp6_{slug}_stab_{k}_dlc5_seed{seed}")
        if os.path.isdir(d):
            return d
    return None


def resolve_persona_dir(logs_dir, slug, persona, seed):
    """Resolve run dir for a per-persona run at k=0."""
    name = f"exp6_{slug}_persona_{persona}_k0_dlc5_seed{seed}"
    d = os.path.join(logs_dir, f"exp6_{slug}", name)
    return d if os.path.isdir(d) else None


def resolve_homogeneous_dir(logs_dir, slug, k, seed):
    """Resolve homogeneous baseline from exp1 runs at dlc=5."""
    model_logs = os.path.join(logs_dir, f"exp1_{slug}")
    return _exp1_resolve(model_logs, dlc=5, n_stab=k, seed=seed, model=slug)


def load_condition_sweep(logs_dir, slug, condition):
    """Return {k: [br, ...]} for a condition across K_VALUES.

    Homogeneous baseline is loaded from exp1 runs at dlc=5.
    Mixed condition is loaded from exp6 runs (plain-named fallback).
    """
    result = {}
    for k in K_VALUES:
        brs = []
        for seed in SEEDS:
            if condition == "homogeneous":
                d = resolve_homogeneous_dir(logs_dir, slug, k, seed)
            else:
                d = resolve_condition_dir(logs_dir, slug, condition, k, seed)
            if d is None:
                continue
            br = compute_br(d)
            if br is not None:
                brs.append(br)
        if brs:
            result[k] = brs
    return result


def load_price_sweep(logs_dir, slug, condition, metric_fn):
    """Return {k: [metric, ...]} for a price metric across K_VALUES and seeds."""
    result = {}
    for k in K_VALUES:
        vals = []
        for seed in SEEDS:
            if condition == "homogeneous":
                d = resolve_homogeneous_dir(logs_dir, slug, k, seed)
            else:
                d = resolve_condition_dir(logs_dir, slug, condition, k, seed)
            if d is None:
                continue
            v = metric_fn(d)
            if v is not None:
                vals.append(v)
        if vals:
            result[k] = vals
    return result


def load_persona_brs(logs_dir, slug, persona):
    """Return [br, ...] for a persona at k=0 across seeds."""
    brs = []
    for seed in SEEDS:
        d = resolve_persona_dir(logs_dir, slug, persona, seed)
        if d is None:
            continue
        br = compute_br(d)
        if br is not None:
            brs.append(br)
    return brs


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_line_panel(ax, homo_data, mixed_data, ylabel, title, legend_loc="best"):
    """Generic line plot: homogeneous vs mixed for any per-seed metric dict {k: [vals]}."""
    for data, label, ls, color in [
        (homo_data,  "Homogeneous (PRICE_HAWK)", "-",  "#0072B2"),
        (mixed_data, "Mixed personas",           "--", "#D55E00"),
    ]:
        xs, means, mins, maxs = [], [], [], []
        for k in K_VALUES:
            vals = data.get(k)
            if not vals:
                continue
            xs.append(k)
            means.append(float(np.mean(vals)))
            mins.append(float(np.min(vals)))
            maxs.append(float(np.max(vals)))
        if not xs:
            continue
        xs = np.array(xs)
        ax.plot(xs, means, ls=ls, color=color, marker="o", markersize=5,
                label=label, zorder=4)
        ax.fill_between(xs, mins, maxs, color=color, alpha=0.15, zorder=2)

    ax.set_xticks(K_VALUES)
    ax.set_xlim(-0.3, K_VALUES[-1] + 0.3)
    ax.set_xlabel("Stabilizing Firms ($k$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc=legend_loc, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_left_panel(ax, homo_data, mixed_data):
    """b_r vs k: homogeneous vs mixed."""
    for data, label, ls, color in [
        (homo_data,  "Homogeneous (PRICE_HAWK)", "-",  "#0072B2"),
        (mixed_data, "Mixed personas",           "--", "#D55E00"),
    ]:
        xs, means, mins, maxs = [], [], [], []
        for k in K_VALUES:
            brs = data.get(k)
            if not brs:
                continue
            srs = [1.0 - b for b in brs]
            xs.append(k)
            means.append(float(np.mean(srs)))
            mins.append(float(np.min(srs)))
            maxs.append(float(np.max(srs)))
        if not xs:
            continue
        xs = np.array(xs)
        ax.plot(xs, means, ls=ls, color=color, marker="o", markersize=5,
                label=label, zorder=4)
        ax.fill_between(xs, mins, maxs, color=color, alpha=0.15, zorder=2)

    # Annotate gap at k=0 if data present (in success-rate terms)
    homo_k0  = homo_data.get(0)
    mixed_k0 = mixed_data.get(0)
    if homo_k0 and mixed_k0:
        homo_sr0  = 1.0 - float(np.mean(homo_k0))
        mixed_sr0 = 1.0 - float(np.mean(mixed_k0))
        delta = mixed_sr0 - homo_sr0
        if abs(delta) > 0.1:
            ax.annotate(f"Persona effect: {delta:+.2f} $1-b_r$",
                        xy=(0, homo_sr0),
                        xytext=(0.5, homo_sr0 + 0.15 * np.sign(delta)),
                        fontsize=7.5, color="0.3",
                        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
        else:
            ax.text(0, 0.95, "No persona effect at $dlc=5$",
                    fontsize=7.5, color="0.4", ha="left")

    ax.axhline(0.5, color="0.6", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(K_VALUES)
    ax.set_xlim(-0.3, K_VALUES[-1] + 0.3)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Stabilizing Firms ($k$)")
    ax.set_ylabel("Success Rate $1 - b_r$")
    ax.set_title("Success Rate", fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_right_panel(ax, persona_data):
    """Bar chart: per-persona b_r at k=0."""
    xs    = range(len(PERSONAS))
    means = []
    lows  = []
    highs = []

    for p in PERSONAS:
        brs = persona_data.get(p, [])
        if brs:
            means.append(float(np.mean(brs)))
            lows.append(float(np.min(brs)))
            highs.append(float(np.max(brs)))
        else:
            means.append(float("nan"))
            lows.append(float("nan"))
            highs.append(float("nan"))

    colors = [PERSONA_COLORS[p] for p in PERSONAS]
    bars = ax.bar(xs, means, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    # Range error bars
    for i, (m, lo, hi) in enumerate(zip(means, lows, highs)):
        if not np.isnan(m):
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]],
                        fmt="none", color="0.3", capsize=3, lw=1.0, zorder=4)

    ax.axhline(0.5, color="0.6", lw=0.8, ls=":", zorder=1)

    # Annotate PRICE_HAWK bar as baseline
    hawk_idx = PERSONAS.index("price_hawk")
    if not np.isnan(means[hawk_idx]):
        ax.text(hawk_idx, means[hawk_idx] + 0.04, "baseline",
                ha="center", va="bottom", fontsize=7, color="0.4")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONAS], rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Consumer Persona ($k=0$)")
    ax.set_ylabel("Bankruptcy Rate $b_r$")
    ax.set_title("Per-Persona Breakdown ($k=0$, $dlc=5$)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── List helper ───────────────────────────────────────────────────────────────

def list_runs(logs_dir, slug):
    rows = []
    # Condition sweep runs
    for condition in ["homogeneous", "mixed"]:
        for k in K_VALUES:
            for seed in SEEDS:
                d = resolve_condition_dir(logs_dir, slug, condition, k, seed)
                # Show plain fallback path for homogeneous when condition-qualified doesn't exist
                if condition == "homogeneous":
                    canonical = os.path.join(
                        logs_dir, f"exp6_{slug}",
                        f"exp6_{slug}_stab_{k}_dlc5_seed{seed}")
                else:
                    canonical = os.path.join(
                        logs_dir, f"exp6_{slug}",
                        f"exp6_{slug}_{condition}_stab_{k}_dlc5_seed{seed}")
                rows.append(("✓" if d else "✗", f"{condition} k={k}", seed, d or canonical))
    # Per-persona runs
    for persona in PERSONAS:
        for seed in SEEDS:
            d = resolve_persona_dir(logs_dir, slug, persona, seed)
            canonical = os.path.join(
                logs_dir, f"exp6_{slug}",
                f"exp6_{slug}_persona_{persona}_k0_dlc5_seed{seed}")
            rows.append(("✓" if d else "✗", f"persona={persona}", seed, d or canonical))

    print(f"Expected runs  slug='{slug}':")
    for exists, label, seed, path in rows:
        print(f"  [{exists}] {label:28s}  seed={seed:3d}  {path}")
    print(f"\n{sum(1 for r in rows if r[0]=='✓')} / {len(rows)} runs present")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fig C: exp6 consumer personas.")
    ap.add_argument("--llm",      default="gemini-2.5-flash",
                    help="LLM model name (converted to filesystem slug)")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--list",     action="store_true",
                    help="Print expected run directories and exit without plotting.")
    ap.add_argument("--output",   default=None)
    args = ap.parse_args()

    slug     = llm_filesystem_slug(args.llm)
    logs_dir = args.logs_dir

    if args.list:
        list_runs(logs_dir, slug)
        return

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp6"))
        args.output = os.path.join(fig_dir, "exp6_personas.pdf")

    homo_data  = load_condition_sweep(logs_dir, slug, "homogeneous")
    mixed_data = load_condition_sweep(logs_dir, slug, "mixed")

    n_homo  = sum(len(v) for v in homo_data.values())
    n_mixed = sum(len(v) for v in mixed_data.values())
    print(f"  homogeneous: {n_homo} seed-cells loaded", flush=True)
    print(f"  mixed:       {n_mixed} seed-cells loaded", flush=True)

    homo_vol   = load_price_sweep(logs_dir, slug, "homogeneous", compute_price_volatility)
    mixed_vol  = load_price_sweep(logs_dir, slug, "mixed",       compute_price_volatility)
    homo_price = load_price_sweep(logs_dir, slug, "homogeneous", compute_final_avg_price)
    mixed_price = load_price_sweep(logs_dir, slug, "mixed",      compute_final_avg_price)

    if n_homo + n_mixed == 0:
        warnings.warn(f"No data found for llm='{args.llm}' (slug='{slug}') in {logs_dir}. "
                      "Saving empty figure.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.0), constrained_layout=True)
    draw_left_panel(axes[0], homo_data, mixed_data)
    _draw_line_panel(axes[1], homo_vol, mixed_vol,
                     ylabel="Price Volatility (std)",
                     title="Price Volatility",
                     legend_loc="best")
    _draw_line_panel(axes[2], homo_price, mixed_price,
                     ylabel="Final Avg. Price",
                     title="Final Avg. Price",
                     legend_loc="best")

    # Remove duplicate legends from panels 2 and 3 — panel 1 has the legend
    for ax in axes[1:]:
        ax.get_legend().remove()

    fig.suptitle("Mixed vs. Homogeneous ($dlc=5$)", fontsize=11, fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
