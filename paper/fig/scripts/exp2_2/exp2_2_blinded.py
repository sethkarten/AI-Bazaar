"""
exp2_2_blinded.py — Fig D: Detection Mechanism — Identity Memory vs. Quality Reasoning.

Three-panel grouped bar chart:
  Panel 1: Sybil detection rate
  Panel 2: Sybil revenue share
  Panel 3: Consumer surplus

Bar groups per panel (at k=K, default 6):
  1. Base buyer,         ID-visible  (exp2,   k=K, _base suffix)  — sky blue,   solid
  2. Skeptical Guardian, ID-visible  (exp2,   k=K, regular)       — vermillion, solid
  3. Skeptical Guardian, ID-blinded  (exp2_2, k=K, regular)       — vermillion, hatched
  4. [optional] Second model, ID-blinded (exp2_2, k=K)            — bluish-green, hatched

Bar 1 vs 2: effect of quality reasoning (ID memory available in both).
Bar 2 vs 3: effect of removing ID memory from the Guardian.
Bar 3 vs 4: cross-model comparison under blinded condition.

"Base buyer" runs use the --lemon-base-buyer flag and carry a _base suffix in the dir name.
"Guardian"   runs are the standard exp2/exp2_2 runs (no suffix).

Run naming:
  exp2 base     : logs/exp2_{slug}/exp2_{slug}_k{k}_rep{rep}_seed{seed}_base/
  exp2 guardian : logs/exp2_{slug}/exp2_{slug}_k{k}_rep{rep}_seed{seed}/
  exp2_2 guard  : logs/exp2_2_{slug}/exp2_2_{slug}_k{k}_rep{rep}_seed{seed}/

Usage:
    python paper/fig/scripts/exp2_2/exp2_2_blinded.py \\
        --llm gemini-3-flash-preview [--llm2 qwen3.5-guardian:latest] \\
        [--k 6] [--rep 1] [--logs-dir logs/] [--list] [--output ...]
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
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "exp2"))
try:
    from exp2_common import resolve_run_dir as _exp2_resolve
    _HAS_EXP2_COMMON = True
except ImportError:
    _HAS_EXP2_COMMON = False

SEEDS = [8, 16, 64]

# Okabe-Ito
COLOR_BASE     = "#56B4E9"   # sky blue
COLOR_GUARDIAN = "#D55E00"   # vermillion
COLOR_QWEN     = "#009E73"   # bluish-green (second model)

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8.5,
    "lines.linewidth":    1.5,
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


def llm_filesystem_slug(llm: str) -> str:
    """Convert a model name to a filesystem-safe slug (matches scripts/exp2_2.py)."""
    s = llm.strip()
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = s.replace(":", "_")
    return s or "model"


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


def compute_metrics(run_dir):
    """
    Return dict with keys:
      detection_rate   — mean(1 - sybil_passed/sybil_seen) across consumers/timesteps
      sybil_rev_share  — mean lemon_market_sybil_revenue_share
      consumer_surplus — mean lemon_market_avg_consumer_surplus
    Returns None if run_dir missing or no lemon-market data.
    """
    if not os.path.isdir(run_dir):
        return None
    states = load_states(run_dir)
    if not states:
        return None

    det_vals     = []
    rev_vals     = []
    surplus_vals = []

    for s in states:
        for c in s.get("consumers", []):
            sybil_seen   = c.get("sybil_seen_total", 0)
            sybil_passed = c.get("sybil_passed_total", 0)
            if sybil_seen and sybil_seen > 0:
                det_vals.append(1.0 - sybil_passed / sybil_seen)

        rev = s.get("lemon_market_sybil_revenue_share")
        if rev is not None:
            rev_vals.append(float(rev))

        surplus = s.get("lemon_market_avg_consumer_surplus")
        if surplus is not None:
            surplus_vals.append(float(surplus))

    premium_vals = []
    for s in states:
        consumers = s.get("consumers", [])
        honest_hits, sybil_hits = [], []
        for cdata in consumers:
            if not isinstance(cdata, dict):
                continue
            h_seen   = cdata.get("honest_seen_total",   0) or 0
            h_passed = cdata.get("honest_passed_total", 0) or 0
            s_seen   = cdata.get("sybil_seen_total",    0) or 0
            s_passed = cdata.get("sybil_passed_total",  0) or 0
            if h_seen > 0:
                honest_hits.append((h_seen - h_passed) / h_seen)
            if s_seen > 0:
                sybil_hits.append((s_seen - s_passed) / s_seen)
        if honest_hits and sybil_hits:
            premium_vals.append(np.mean(honest_hits) - np.mean(sybil_hits))

    if not rev_vals and not surplus_vals:
        return None

    return {
        "detection_rate":      float(np.mean(det_vals))      if det_vals      else float("nan"),
        "sybil_rev_share":     float(np.mean(rev_vals))      if rev_vals      else float("nan"),
        "consumer_surplus":    float(np.mean(surplus_vals))  if surplus_vals  else float("nan"),
        "detection_premium":   float(np.mean(premium_vals))  if premium_vals  else float("nan"),
    }


def resolve_run_dir(logs_dir, exp, slug, k, rep_visible, seed, base_buyer=False):
    """Resolve a run directory for either exp2 or exp2_2."""
    rep_tag = "rep1" if rep_visible else "rep0"
    suffix  = "_base" if base_buyer else ""
    prefix  = exp  # "exp2" or "exp2_2"
    name    = f"{prefix}_{slug}_k{k}_{rep_tag}_seed{seed}{suffix}"
    d       = os.path.join(logs_dir, f"{prefix}_{slug}", name)
    return d if os.path.isdir(d) else None


def load_condition(logs_dir, slug, exp, k, rep_visible, base_buyer=False):
    """Load metrics for one condition across all seeds."""
    records = []
    for seed in SEEDS:
        d = resolve_run_dir(logs_dir, exp, slug, k, rep_visible, seed, base_buyer)
        if d is None:
            continue
        m = compute_metrics(d)
        if m is not None:
            records.append(m)
    return records


def agg(records, key):
    """Return (mean, min, max) or (nan, nan, nan) if no data."""
    vals = [r[key] for r in records
            if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


# ── Plotting ──────────────────────────────────────────────────────────────────

METRICS = [
    ("detection_rate",    "Detection Rate",    "Higher = better", [0, 1],  False),
    ("detection_premium", "Detection Premium", "Higher = better", None,    True),
    ("sybil_rev_share",   "Revenue Share",     "Lower = better",  [0, 1],  False),
    ("consumer_surplus",  "Consumer Surplus",  "Higher = better", None,    False),
]


def make_bar_groups(slug, slug2=None):
    """
    Return bar group spec: (tick_label, color, hatch, exp, base_buyer, slug).
    If slug2 is provided, appends a 4th bar for the second model's blinded condition.
    """
    groups = [
        ("Base, vis.",     COLOR_BASE,     "",   "exp2",   True,  slug),
        ("Guard., vis.",   COLOR_GUARDIAN, "",   "exp2",   False, slug),
        ("Guard., blind.", COLOR_GUARDIAN, "//", "exp2_2", False, slug),
    ]
    if slug2:
        groups.append(
            ("Qwen Guard., blind.", COLOR_QWEN, "//", "exp2_2", False, slug2)
        )
    return groups


def draw_metric_panel(ax, metric_key, all_records, bar_groups, ylim, ylabel, subtitle,
                      hline=False):
    """Draw one grouped bar panel for a single metric."""
    xs = np.arange(len(bar_groups))

    for xi, (label, color, hatch, exp, base_buyer, slug) in enumerate(bar_groups):
        records = all_records[(exp, base_buyer, slug)]
        mean, mn, mx = agg(records, metric_key)
        bar_h = mean if not np.isnan(mean) else 0.0
        ax.bar(xi, bar_h, width=0.6, color=color, hatch=hatch,
               edgecolor="white",
               linewidth=0.5, zorder=3, alpha=0.9)
        if not np.isnan(mean):
            ax.errorbar(xi, mean, yerr=[[mean - mn], [mx - mean]],
                        fmt="none", color="0.3", capsize=3, lw=1.0, zorder=4)

    if hline:
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in bar_groups], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(subtitle, fontsize=8.5, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(ylim[0] - 0.05, ylim[1] + 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── List helper ───────────────────────────────────────────────────────────────

def list_runs(logs_dir, slug, slug2, k, rep_visible):
    bar_groups = make_bar_groups(slug, slug2)
    rows = []
    for label, _c, _h, exp, base_buyer, grp_slug in bar_groups:
        for seed in SEEDS:
            d = resolve_run_dir(logs_dir, exp, grp_slug, k, rep_visible, seed, base_buyer)
            rep_tag = "rep1" if rep_visible else "rep0"
            suffix = "_base" if base_buyer else ""
            canonical = os.path.join(
                logs_dir, f"{exp}_{grp_slug}", f"{exp}_{grp_slug}_k{k}_{rep_tag}_seed{seed}{suffix}")
            exists = "Y" if d else "N"
            rows.append((exists, label, seed, d or canonical))

    rep_str = "visible" if rep_visible else "hidden"
    print(f"Expected runs  slug='{slug}'  slug2='{slug2}'  k={k}  rep={rep_str}:")
    for exists, label, seed, path in rows:
        print(f"  [{exists}] {label:22s}  seed={seed:3d}  {path}")
    print(f"\n{sum(1 for r in rows if r[0]=='Y')} / {len(rows)} runs present")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fig D: exp2_2 blinded detection.")
    ap.add_argument("--llm",      default="gemini-3-flash-preview",
                    help="Primary LLM model name (converted to filesystem slug)")
    ap.add_argument("--llm2",     default="qwen3.5-guardian:latest",
                    help="Second model for blinded-only comparison bar (omit to hide)")
    ap.add_argument("--no-llm2",  action="store_true",
                    help="Suppress the second-model bar entirely")
    ap.add_argument("--k",        type=int, default=6,
                    help="k value used for all bars (default: 6)")
    ap.add_argument("--rep",      type=int, default=1, choices=[0, 1],
                    help="Rep visibility (1=visible, 0=hidden)")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--list",     action="store_true",
                    help="Print expected run directories and exit without plotting.")
    ap.add_argument("--output",   default=None)
    args = ap.parse_args()

    slug      = llm_filesystem_slug(args.llm)
    slug2     = None if args.no_llm2 else llm_filesystem_slug(args.llm2)
    k         = args.k
    rep_vis   = bool(args.rep)
    logs_dir  = args.logs_dir
    bar_groups = make_bar_groups(slug, slug2)

    if args.list:
        list_runs(logs_dir, slug, slug2, k, rep_vis)
        return

    if args.output is None:
        fig_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp2_2"))
        args.output = os.path.join(fig_dir, "exp2_2_blinded.pdf")

    # Load all conditions — keyed by (exp, base_buyer, slug)
    all_records = {}
    for _label, _color, _hatch, exp, base_buyer, grp_slug in bar_groups:
        key = (exp, base_buyer, grp_slug)
        if key in all_records:
            continue
        all_records[key] = load_condition(logs_dir, grp_slug, exp, k, rep_vis, base_buyer)
        btype = "base" if base_buyer else "guardian"
        n = len(all_records[key])
        print(f"  {exp} {btype} [{grp_slug}]: {n} seeds loaded", flush=True)

    total = sum(len(v) for v in all_records.values())
    if total == 0:
        warnings.warn(f"No data found for llm='{args.llm}' (slug='{slug}'). Saving empty figure.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.8), constrained_layout=True)

    for ax, (metric_key, metric_name, direction, ylim, hline) in zip(axes, METRICS):
        draw_metric_panel(ax, metric_key, all_records, bar_groups, ylim,
                          ylabel=metric_name, subtitle=f"{metric_name}\n({direction})",
                          hline=hline)

    # Shared legend below all panels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_BASE,     edgecolor="white",
              label=f"Base buyer, ID-visible (exp2, k={k})"),
        Patch(facecolor=COLOR_GUARDIAN, edgecolor="white",
              label=f"Guardian, ID-visible (exp2, k={k})"),
        Patch(facecolor=COLOR_GUARDIAN, edgecolor="white", hatch="//",
              label=f"Guardian, ID-blinded (exp2-2, k={k})"),
    ]
    if slug2:
        legend_elements.append(
            Patch(facecolor=COLOR_QWEN, edgecolor="white", hatch="//",
                  label=f"Qwen Guard., ID-blinded (exp2-2, k={k})")
        )
    fig.legend(handles=legend_elements, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7,
               frameon=True, framealpha=0.9, edgecolor="0.8")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
