"""
Fig Exp2-E: Deceptive Revenue Share Over Time

Fraction of step revenue captured by the sybil cluster, per (K, rep_visible) condition.
Lines coloured by K; solid = rep visible, dashed = rep hidden.
Mean ± 1σ bands across seeds.

Directory naming (mirrors exp2.py):
  logs/{name_prefix}/{name_prefix}_baseline_seed{seed}
  logs/{name_prefix}/{name_prefix}_k{k}_rep1_seed{seed}
  logs/{name_prefix}/{name_prefix}_k{k}_rep0_seed{seed}

Usage:
    python exp2_sybil_revenue_share.py [--logs-dir logs/] [--good car] [--output ...]
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
from exp2_cache import (
    get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data,
    infer_name_prefix,
)

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
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

# --- Experiment constants (must match exp2.py) ---
SEEDS    = [8, 16, 64]
K_VALUES = [3, 6, 9]
RHO_MIN  = 0.3

COLORS_K = {3: "#56B4E9", 6: "#E69F00", 9: "#009E73"}
LS_REP   = {True: "-", False: "--"}


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

def resolve_run_dir(logs_dir: str, name_prefix: str, k: int, rep_visible: bool, seed: int) -> str | None:
    rep_tag  = "rep1" if rep_visible else "rep0"
    run_name = f"{name_prefix}_k{k}_{rep_tag}_seed{seed}" if k > 0 else f"{name_prefix}_baseline_seed{seed}"
    canonical = os.path.join(logs_dir, name_prefix, run_name)
    if os.path.isdir(canonical):
        return canonical
    flat = os.path.join(logs_dir, run_name)
    if os.path.isdir(flat):
        return flat
    return None


def collect_run_dirs(logs_dir: str, name_prefix: str) -> list[str]:
    dirs = []
    for k in K_VALUES:
        for rv in [True, False]:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, name_prefix, k, rv, seed)
                if d:
                    dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# State loading
# ---------------------------------------------------------------------------

def load_state_files(run_dir: str) -> list[str]:
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


def get_revenue_share_series(run_dir: str):
    files = load_state_files(run_dir)
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
    return np.array([t for t, _ in ts_vals]), np.array([v for _, v in ts_vals])


# ---------------------------------------------------------------------------
# Aggregation
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


def build_aggregate(results: dict) -> dict:
    conditions = set((k, rv) for k, rv, _ in results)
    seeds      = set(seed for _, _, seed in results)
    agg = {}
    for k, rv in conditions:
        series = [results[(k, rv, s)] for s in seeds if (k, rv, s) in results and results[(k, rv, s)] is not None]
        if not series:
            agg[(k, rv)] = None
            continue
        common, arr = interp_common([s[0] for s in series], [s[1] for s in series])
        if common is None:
            agg[(k, rv)] = None
            continue
        agg[(k, rv)] = {
            "ts":   common.tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std":  (arr.std(axis=0).tolist() if arr.shape[0] > 1 else np.zeros(len(common)).tolist()),
        }
    return agg


def serialize_agg(agg: dict) -> dict:
    return {f"{k},{int(rv)}": v for (k, rv), v in agg.items()}


def deserialize_agg(raw: dict) -> dict:
    out = {}
    for key, v in raw.items():
        k_str, rv_str = key.split(",")
        out[(int(k_str), bool(int(rv_str)))] = (
            None if v is None else {
                "ts":   np.array(v["ts"]),
                "mean": np.array(v["mean"]),
                "std":  np.array(v["std"]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_band(ax, entry, color, label, ls="-", lw=1.8):
    if entry is None:
        return
    ts, mean, std = entry["ts"], entry["mean"], entry["std"]
    ax.plot(ts, mean, color=color, lw=lw, ls=ls, label=label, zorder=4)
    if np.any(std > 0):
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15, zorder=3)


def make_figure(agg: dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5), constrained_layout=True)
    fig.suptitle("Exp 2 — Sybil revenue share over time", fontsize=10, fontweight="bold")

    ax.axhline(0.5, color="#555555", lw=1.2, ls="--", alpha=0.8, zorder=2, label="Majority threshold (0.5)")

    for k in K_VALUES:
        for rv in [True, False]:
            entry = agg.get((k, rv))
            if entry is None:
                continue
            sat = k / 12
            rep_tag = "rep" if rv else "no-rep"
            lbl = f"K={k} ({sat:.0%}), {rep_tag}"
            plot_band(ax, entry, COLORS_K[k], lbl, ls=LS_REP[rv])

    ax.set_ylabel("Sybil revenue share")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Timestep")
    ax.legend(loc="best", fontsize=7.5)

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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    name_prefix = infer_name_prefix(args.logs_dir)
    print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_run_dirs(args.logs_dir, name_prefix)
    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_sybil_revenue_share", args.good)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        agg = deserialize_agg(load_cache_data(cache_path)["agg"])
    else:
        jobs = []
        for k in K_VALUES:
            for rv in [True, False]:
                for seed in SEEDS:
                    d = resolve_run_dir(args.logs_dir, name_prefix, k, rv, seed)
                    if d:
                        jobs.append((k, rv, seed, d))
                    else:
                        print(f"  Missing: K={k} rep={int(rv)} seed={seed}", flush=True)

        print(f"Loading {len(jobs)} runs ...", flush=True)
        results: dict = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {ex.submit(get_revenue_share_series, d): (k, rv, seed)
                          for k, rv, seed, d in jobs}
            done, total = 0, len(jobs)
            for future in concurrent.futures.as_completed(future_map):
                k, rv, seed = future_map[future]
                done += 1
                data = future.result()
                print(f"  [{done}/{total}] K={k} rep={int(rv)} seed={seed} — {'ok' if data is not None else 'empty'}", flush=True)
                results[(k, rv, seed)] = data

        agg = build_aggregate(results)
        save_cache(cache_path, {"agg": serialize_agg(agg)}, args.logs_dir, args.good)
        print(f"Cached: {cache_path}", flush=True)
        agg = deserialize_agg({"agg": serialize_agg(agg)}["agg"])

    fig = make_figure(agg)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
