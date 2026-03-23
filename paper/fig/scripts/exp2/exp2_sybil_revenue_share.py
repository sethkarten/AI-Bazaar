"""
Fig Exp2-E: Experiment 2 — Deceptive Revenue Share

Single panel: sybil revenue share over time (per n_sybil, mean ± 1σ across seeds).

Usage:
    python exp2_sybil_revenue_share.py [--logs-dir logs/] [--good car] [--output ...] [--workers 8]
"""

import argparse
import concurrent.futures
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from exp2_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "lines.linewidth":   1.5,
    "lines.markersize":  5,
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "grid.color":        "gray",
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.8",
    "figure.dpi":        100,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":       False,
    "pdf.fonttype":      42,
})

SEEDS = [8, 16, 64]
N_SYBIL_VALUES = [0, 3, 6, 9, 12]
RHO_MIN = 0.3

COLORS_N_SYBIL = {
    0:  "#999999",
    3:  "#56B4E9",
    6:  "#E69F00",
    9:  "#009E73",
    12: "#D55E00",
}


# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------

def resolve_run_dir(logs_dir, n_sybil, seed):
    if n_sybil == 0:
        path = os.path.join(logs_dir, f"exp2_baseline_seed{seed}")
    else:
        path = os.path.join(logs_dir, f"exp2_sybil_{n_sybil}_rho{RHO_MIN}_seed{seed}")
    return path if os.path.isdir(path) else None


def collect_run_dirs(logs_dir):
    dirs = []
    for n in N_SYBIL_VALUES:
        for s in SEEDS:
            d = resolve_run_dir(logs_dir, n, s)
            if d:
                dirs.append(d)
    return dirs


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


# ---------------------------------------------------------------------------
# Metric extractor
# ---------------------------------------------------------------------------

def get_revenue_share_series(run_dir):
    """Return (timesteps, revenue_share_values) arrays or None."""
    files = load_states(run_dir)
    if not files:
        return None
    ts_vals = []
    for p in files:
        with open(p) as f:
            s = json.load(f)
        v = s.get("lemon_market_sybil_revenue_share")
        if v is not None:
            ts_vals.append((s["timestep"], float(v)))
    if not ts_vals:
        return None
    ts_vals.sort()
    return np.array([t for t, v in ts_vals]), np.array([v for t, v in ts_vals])


def load_one_run(run_dir):
    return get_revenue_share_series(run_dir)


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def interp_common(ts_list, val_list):
    if not ts_list:
        return None, None
    t_min = int(min(ts[0] for ts in ts_list))
    t_max = int(max(ts[-1] for ts in ts_list))
    common = np.arange(t_min, t_max + 1, dtype=float)
    arr = np.array([np.interp(common, ts.astype(float), v.astype(float))
                    for ts, v in zip(ts_list, val_list)])
    return common, arr


# ---------------------------------------------------------------------------
# Cache serialisation
# ---------------------------------------------------------------------------

def build_aggregate(results_by_n, n_sybil_list):
    """
    For each n_sybil, aggregate mean ± std across seeds.
    Returns dict: {str(n_sybil): {"ts": [...], "mean": [...], "std": [...]} or None}
    """
    agg = {}
    for n in n_sybil_list:
        seed_data = results_by_n.get(n, {})
        series = [v for v in seed_data.values() if v is not None]
        if not series:
            agg[str(n)] = None
            continue
        ts_list  = [s[0] for s in series]
        val_list = [s[1] for s in series]
        common, arr = interp_common(ts_list, val_list)
        if common is None:
            agg[str(n)] = None
            continue
        mean_v = arr.mean(axis=0)
        std_v  = arr.std(axis=0) if arr.shape[0] > 1 else np.zeros_like(mean_v)
        agg[str(n)] = {
            "ts":   common.tolist(),
            "mean": mean_v.tolist(),
            "std":  std_v.tolist(),
        }
    return agg


def _deserialize_agg(raw, n_sybil_list):
    out = {}
    for n in n_sybil_list:
        entry = raw.get(str(n))
        if entry is None:
            out[n] = None
        else:
            out[n] = {
                "ts":   np.array(entry["ts"]),
                "mean": np.array(entry["mean"]),
                "std":  np.array(entry["std"]),
            }
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_band(ax, agg_entry, color, label, lw=1.8, ls="-"):
    if agg_entry is None:
        return
    ts   = agg_entry["ts"]
    mean = agg_entry["mean"]
    std  = agg_entry["std"]
    ax.plot(ts, mean, color=color, lw=lw, ls=ls, label=label, zorder=4)
    if np.any(std > 0):
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15, zorder=3)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(revenue_share_agg):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    fig.suptitle("Deceptive Revenue Share", fontweight="bold", fontsize=10)

    # Majority threshold reference line
    ax.axhline(
        0.5,
        color="#555555",
        lw=1.2,
        ls="--",
        alpha=0.8,
        zorder=2,
        label="Majority threshold",
    )

    for n in [3, 6, 9, 12]:
        entry = revenue_share_agg.get(n)
        lbl = f"n_sybil={n}"
        plot_band(ax, entry, COLORS_N_SYBIL[n], lbl)

    ax.set_ylabel("Sybil revenue share")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Timestep")
    ax.set_title(
        "Fraction of step revenue captured by sybil cluster (mean ± 1\u03c3 across seeds)"
    )
    ax.legend(loc="best")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp2 Fig E: Deceptive Revenue Share")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="car")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "exp2", "exp2_sybil_revenue_share.pdf"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Ignore cache and rebuild from scratch")
    args = parser.parse_args()

    # n_sybil=0 has no sybil firms — only aggregate over {3,6,9,12}
    SYBIL_ONLY = [3, 6, 9, 12]

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_sybil_revenue_share", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        raw = load_cache_data(cache_path)
        revenue_share_agg = _deserialize_agg(raw["revenue_share"], SYBIL_ONLY)
    else:
        jobs = []
        for n in SYBIL_ONLY:
            for seed in SEEDS:
                run_dir = resolve_run_dir(args.logs_dir, n, seed)
                if run_dir:
                    jobs.append((n, seed, run_dir))
                else:
                    print(f"  Missing: n_sybil={n}, seed={seed}", flush=True)

        print(f"Loading {len(jobs)} runs ...", flush=True)
        results_by_n = {n: {} for n in SYBIL_ONLY}

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {
                ex.submit(load_one_run, run_dir): (n, seed)
                for n, seed, run_dir in jobs
            }
            done = 0
            total = len(jobs)
            for future in concurrent.futures.as_completed(future_map):
                n, seed = future_map[future]
                done += 1
                result = future.result()
                has_data = result is not None
                print(
                    f"  [{done}/{total}] n_sybil={n} seed={seed}"
                    f" — revenue_share={'ok' if has_data else 'empty'}",
                    flush=True,
                )
                results_by_n[n][seed] = result

        revenue_share_agg = build_aggregate(results_by_n, SYBIL_ONLY)

        cache_data = {"revenue_share": revenue_share_agg}
        save_cache(cache_path, cache_data, args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

        revenue_share_agg = _deserialize_agg(revenue_share_agg, SYBIL_ONLY)

    fig = make_figure(revenue_share_agg)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
