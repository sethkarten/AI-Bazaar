"""
Fig Exp2-A: Experiment 2 — Sybil Detection & Revenue Dominance

Row 1: Sybil pass rate over time (per n_sybil, mean ± 1σ across seeds)
Row 2: Sybil revenue share over time (per n_sybil, mean ± 1σ across seeds)

Usage:
    python exp2_sybil_detection.py [--logs-dir logs/] [--good car] [--output ...] [--workers 8]
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

# 5 levels up from paper/fig/scripts/exp2/ to project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder
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
# Metric extractors
# ---------------------------------------------------------------------------

def get_pass_rate_series(run_dir):
    """Returns (timesteps, mean_pass_rate_per_step) or None.

    For each timestep, average sybil_pass_rate_this_step across all consumers
    that reported it (i.e. those who saw sybil listings that step).
    """
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    df = db.lemon_sybil_pass_rate_per_buyer_over_time()
    if df.empty:
        return None
    per_ts = (
        df.groupby("timestep")["value"]
        .mean()
        .reset_index()
        .sort_values("timestep")
    )
    if per_ts.empty:
        return None
    return per_ts["timestep"].values, per_ts["value"].values


def get_revenue_share_series(run_dir):
    """Returns (timesteps, revenue_share) or None."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    df = db.lemon_sybil_revenue_share_over_time()
    if df.empty:
        return None
    df = df.sort_values("timestep")
    return df["timestep"].values, df["value"].values


def load_one_run(run_dir, n_sybil):
    rev = get_revenue_share_series(run_dir)
    if n_sybil == 0:
        # No sybil firms — pass rate is undefined; revenue share is always 0
        return {"pass_rate": None, "revenue_share": rev}
    pr = get_pass_rate_series(run_dir)
    return {"pass_rate": pr, "revenue_share": rev}


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def interp_common(ts_list, val_list):
    """Interpolate all series to a common integer grid; return (ts, 2-D array)."""
    if not ts_list:
        return None, None
    t_min = int(min(ts[0] for ts in ts_list))
    t_max = int(max(ts[-1] for ts in ts_list))
    common = np.arange(t_min, t_max + 1, dtype=float)
    arr = np.array([
        np.interp(common, ts.astype(float), v.astype(float))
        for ts, v in zip(ts_list, val_list)
    ])
    return common, arr


# ---------------------------------------------------------------------------
# Cache serialisation
# ---------------------------------------------------------------------------

def build_aggregate(results_by_n, metric_key, n_sybil_list):
    """
    For each n_sybil, aggregate (mean ± std) across seeds.
    Returns dict: {n_sybil_str: {"ts": [...], "mean": [...], "std": [...]}}
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
    """Convert JSON agg back to dict of {n_sybil_int: {"ts": arr, "mean": arr, "std": arr}}."""
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
# Plotting
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


def make_figure(pass_rate_agg, revenue_share_agg):
    fig, axes = plt.subplots(
        2, 1,
        figsize=(7, 6),
        constrained_layout=True,
    )
    fig.suptitle(
        "Experiment 2: Lemon Market — Sybil Detection & Revenue Dominance",
        fontweight="bold",
        fontsize=10,
    )

    # --- Row 1: Pass rate ---
    ax = axes[0]
    for n in [3, 6, 9, 12]:
        entry = pass_rate_agg.get(n)
        lbl = f"n_sybil={n}"
        plot_band(ax, entry, COLORS_N_SYBIL[n], lbl)
    ax.set_ylabel("Sybil pass rate (fraction)")
    ax.set_ylim(0, 1)
    ax.set_title("Consumer pass rate on sybil listings (mean ± 1σ across seeds)")
    ax.legend(loc="best")
    ax.set_xlabel("Timestep")

    # --- Row 2: Revenue share ---
    ax = axes[1]
    # Optional: dashed grey zero line for baseline
    ax.axhline(0, color="#999999", lw=1.0, ls="--", alpha=0.6, zorder=2, label="Baseline (n_sybil=0): 0")
    ax.axhline(0.5, color="#555555", lw=1.2, ls="--", alpha=0.8, zorder=2, label="Majority threshold (0.5)")
    for n in [3, 6, 9, 12]:
        entry = revenue_share_agg.get(n)
        lbl = f"n_sybil={n}"
        plot_band(ax, entry, COLORS_N_SYBIL[n], lbl)
    ax.set_ylabel("Sybil revenue share")
    ax.set_ylim(0, 1)
    ax.set_title("Sybil revenue share over time (mean ± 1σ across seeds)")
    ax.legend(loc="best")
    ax.set_xlabel("Timestep")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp2 Fig A: Sybil Detection")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="car")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "exp2", "exp2_sybil_detection.pdf"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Ignore cache and rebuild from scratch")
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_sybil_detection", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        raw = load_cache_data(cache_path)
        pass_rate_agg     = _deserialize_agg(raw["pass_rate"],     [3, 6, 9, 12])
        revenue_share_agg = _deserialize_agg(raw["revenue_share"], N_SYBIL_VALUES)
    else:
        # Collect jobs
        jobs = []
        for n in N_SYBIL_VALUES:
            for seed in SEEDS:
                run_dir = resolve_run_dir(args.logs_dir, n, seed)
                if run_dir:
                    jobs.append((n, seed, run_dir))
                else:
                    print(f"  Missing: n_sybil={n}, seed={seed}", flush=True)

        print(f"Loading {len(jobs)} runs ...", flush=True)
        # results_by_n[n][seed] = {"pass_rate": ..., "revenue_share": ...}
        results_by_n = {n: {} for n in N_SYBIL_VALUES}

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {
                ex.submit(load_one_run, run_dir, n): (n, seed)
                for n, seed, run_dir in jobs
            }
            done = 0
            total = len(jobs)
            for future in concurrent.futures.as_completed(future_map):
                n, seed = future_map[future]
                done += 1
                data = future.result()
                has_pr  = data["pass_rate"] is not None
                has_rev = data["revenue_share"] is not None
                print(
                    f"  [{done}/{total}] n_sybil={n} seed={seed}"
                    f" — pass_rate={'ok' if has_pr else 'empty'}"
                    f", revenue_share={'ok' if has_rev else 'empty'}",
                    flush=True,
                )
                results_by_n[n][seed] = data

        # Build per-metric seed series dicts
        pr_by_n  = {n: {s: results_by_n[n].get(s, {}).get("pass_rate")
                        for s in SEEDS} for n in N_SYBIL_VALUES}
        rev_by_n = {n: {s: results_by_n[n].get(s, {}).get("revenue_share")
                        for s in SEEDS} for n in N_SYBIL_VALUES}

        pass_rate_agg     = build_aggregate(pr_by_n,  "pass_rate",     [3, 6, 9, 12])
        revenue_share_agg = build_aggregate(rev_by_n, "revenue_share", N_SYBIL_VALUES)

        cache_data = {
            "pass_rate":     {str(n): v for n, v in pass_rate_agg.items()},
            "revenue_share": {str(n): v for n, v in revenue_share_agg.items()},
        }
        save_cache(cache_path, cache_data, args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

        # Convert string keys back to int for plotting
        pass_rate_agg     = _deserialize_agg(cache_data["pass_rate"],     [3, 6, 9, 12])
        revenue_share_agg = _deserialize_agg(cache_data["revenue_share"], N_SYBIL_VALUES)

    fig = make_figure(pass_rate_agg, revenue_share_agg)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
