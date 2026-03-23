"""
Fig Exp2-B: Experiment 2 — Market Volume Under Sybil Conditions

Row 1: Filled orders per step (per n_sybil condition, mean ± 1σ across seeds)
Row 2: Market activity breakdown (Listings / Bids / Passes) for the baseline run
       averaged across available seeds

Usage:
    python exp2_lemon_volume.py [--logs-dir logs/] [--good car] [--output ...] [--workers 8]
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

# Market activity colors (Listings / Bids / Passes)
COLOR_LISTINGS = "#0072B2"   # Okabe Blue
COLOR_BIDS     = "#009E73"   # Okabe Green
COLOR_PASSES   = "#E69F00"   # Okabe Orange


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

def get_volume_series(run_dir):
    """Returns (timesteps, filled_orders_per_step) or None."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    df = db.filled_orders_count_over_time().sort_values("timestep")
    if df.empty:
        return None
    return df["timestep"].values, df["value"].values


def get_activity_series(run_dir):
    """Returns dict {metric: (ts, values)} for Listings/Bids/Passes, or None."""
    files = load_states(run_dir)
    if not files:
        return None
    db = DataFrameBuilder(state_files=files)
    df = db.lemon_market_metrics_over_time()
    if df.empty:
        return None
    out = {}
    for metric in ["Listings", "Bids", "Passes"]:
        sub = df[df["metric"] == metric].sort_values("timestep")
        if sub.empty:
            out[metric] = None
        else:
            out[metric] = (sub["timestep"].values, sub["value"].values)
    return out


def load_one_run(run_dir):
    volume   = get_volume_series(run_dir)
    activity = get_activity_series(run_dir)
    return {"volume": volume, "activity": activity}


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def interp_common(ts_list, val_list):
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

def build_volume_agg(results_by_n):
    """Aggregate filled-orders per n_sybil across seeds."""
    agg = {}
    for n in N_SYBIL_VALUES:
        seed_data = results_by_n.get(n, {})
        series = [v["volume"] for v in seed_data.values() if v.get("volume") is not None]
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
        agg[str(n)] = {"ts": common.tolist(), "mean": mean_v.tolist(), "std": std_v.tolist()}
    return agg


def build_activity_agg(results_by_n):
    """Average baseline activity (n_sybil=0) across seeds."""
    seed_data = results_by_n.get(0, {})
    metrics = ["Listings", "Bids", "Passes"]
    out = {}
    for metric in metrics:
        series = []
        for v in seed_data.values():
            act = v.get("activity")
            if act and act.get(metric) is not None:
                series.append(act[metric])
        if not series:
            out[metric] = None
            continue
        ts_list  = [s[0] for s in series]
        val_list = [s[1] for s in series]
        common, arr = interp_common(ts_list, val_list)
        if common is None:
            out[metric] = None
            continue
        mean_v = arr.mean(axis=0)
        out[metric] = {"ts": common.tolist(), "mean": mean_v.tolist()}
    return out


def _deserialize_vol_agg(raw):
    out = {}
    for n in N_SYBIL_VALUES:
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


def _deserialize_activity_agg(raw):
    out = {}
    for metric in ["Listings", "Bids", "Passes"]:
        entry = raw.get(metric)
        if entry is None:
            out[metric] = None
        else:
            out[metric] = {
                "ts":   np.array(entry["ts"]),
                "mean": np.array(entry["mean"]),
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
    std  = agg_entry.get("std", np.zeros_like(mean))
    ax.plot(ts, mean, color=color, lw=lw, ls=ls, label=label, zorder=4)
    if np.any(std > 0):
        lo = np.maximum(mean - std, 0)
        ax.fill_between(ts, lo, mean + std, color=color, alpha=0.15, zorder=3)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(vol_agg, activity_agg):
    fig, axes = plt.subplots(
        2, 1,
        figsize=(7, 6),
        constrained_layout=True,
    )
    fig.suptitle(
        "Experiment 2: Lemon Market — Market Volume",
        fontweight="bold",
        fontsize=10,
    )

    # --- Row 1: Filled orders per step ---
    ax = axes[0]
    for n in N_SYBIL_VALUES:
        entry = vol_agg.get(n)
        ls = "--" if n == 0 else "-"
        lbl = f"n_sybil={n}" + (" (baseline)" if n == 0 else "")
        plot_band(ax, entry, COLORS_N_SYBIL[n], lbl, ls=ls)
    ax.set_ylabel("Filled orders per step")
    ax.set_ylim(bottom=0)
    ax.set_title("Filled orders per step across sybil conditions (mean ± 1σ across seeds)")
    ax.legend(loc="best")
    ax.set_xlabel("Timestep")

    # --- Row 2: Market activity breakdown (baseline only) ---
    ax = axes[1]
    metric_colors = {
        "Listings": COLOR_LISTINGS,
        "Bids":     COLOR_BIDS,
        "Passes":   COLOR_PASSES,
    }
    any_data = False
    for metric, color in metric_colors.items():
        entry = activity_agg.get(metric)
        if entry is None:
            continue
        any_data = True
        ax.plot(entry["ts"], entry["mean"], color=color, lw=1.8, label=metric, zorder=4)
    if not any_data:
        ax.text(0.5, 0.5, "no activity data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=9)
    ax.set_ylabel("Count per step")
    ax.set_ylim(bottom=0)
    ax.set_title("Market activity breakdown — baseline (no sybil), mean across seeds")
    ax.legend(loc="best")
    ax.set_xlabel("Timestep")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp2 Fig B: Market Volume")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="car")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "exp2", "exp2_lemon_volume.pdf"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Ignore cache and rebuild from scratch")
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_lemon_volume", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        raw = load_cache_data(cache_path)
        vol_agg      = _deserialize_vol_agg(raw["volume"])
        activity_agg = _deserialize_activity_agg(raw["activity_baseline"])
    else:
        jobs = []
        for n in N_SYBIL_VALUES:
            for seed in SEEDS:
                run_dir = resolve_run_dir(args.logs_dir, n, seed)
                if run_dir:
                    jobs.append((n, seed, run_dir))
                else:
                    print(f"  Missing: n_sybil={n}, seed={seed}", flush=True)

        print(f"Loading {len(jobs)} runs ...", flush=True)
        results_by_n = {n: {} for n in N_SYBIL_VALUES}

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
                data = future.result()
                has_vol = data["volume"] is not None
                has_act = data["activity"] is not None
                print(
                    f"  [{done}/{total}] n_sybil={n} seed={seed}"
                    f" — volume={'ok' if has_vol else 'empty'}"
                    f", activity={'ok' if has_act else 'empty'}",
                    flush=True,
                )
                results_by_n[n][seed] = data

        vol_agg      = build_volume_agg(results_by_n)
        activity_agg = build_activity_agg(results_by_n)

        # Convert activity_agg values to plain lists for JSON
        act_ser = {}
        for metric, entry in activity_agg.items():
            if entry is None:
                act_ser[metric] = None
            else:
                act_ser[metric] = {
                    "ts":   entry["ts"].tolist() if hasattr(entry["ts"], "tolist") else entry["ts"],
                    "mean": entry["mean"].tolist() if hasattr(entry["mean"], "tolist") else entry["mean"],
                }

        cache_data = {"volume": vol_agg, "activity_baseline": act_ser}
        save_cache(cache_path, cache_data, args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

        vol_agg      = _deserialize_vol_agg(vol_agg)
        activity_agg = _deserialize_activity_agg(act_ser)

    fig = make_figure(vol_agg, activity_agg)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
