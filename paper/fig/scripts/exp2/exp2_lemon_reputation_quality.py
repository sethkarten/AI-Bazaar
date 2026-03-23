"""
Fig Exp2-C: Experiment 2 — Reputation Dynamics (Honest vs Sybil Firms)

Row 1: Mean honest firm reputation over time (per n_sybil, mean ± 1σ across seeds)
Row 2: Mean sybil firm reputation over time (n_sybil ∈ {3,6,9,12}, mean ± 1σ across seeds)

Sybil/honest classification is read directly from state["firms"][*]["sybil"] boolean flag.

Usage:
    python exp2_lemon_reputation_quality.py [--logs-dir logs/] [--good car] [--output ...] [--workers 8]
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

def get_reputation_series(run_dir):
    """
    Returns dict with keys:
      "honest": (timesteps, mean_honest_rep) or None
      "sybil":  (timesteps, mean_sybil_rep)  or None

    Reads sybil flag from state["firms"][*]["sybil"].
    Falls back to checking whether the firm name starts with "sybil_".
    """
    files = load_states(run_dir)
    if not files:
        return {"honest": None, "sybil": None}

    db = DataFrameBuilder(state_files=files)
    states = db.states

    honest_ts, honest_vals = [], []
    sybil_ts, sybil_vals   = [], []

    for s in states:
        t = s["timestep"]
        firms = s.get("firms", [])
        h_reps = []
        sy_reps = []
        for f in firms:
            rep = f.get("reputation")
            if not isinstance(rep, (int, float)):
                continue
            is_sybil = f.get("sybil", False)
            if not isinstance(is_sybil, bool):
                # Fall back to name-based detection
                is_sybil = str(f.get("name", "")).startswith("sybil_")
            if is_sybil:
                sy_reps.append(float(rep))
            else:
                h_reps.append(float(rep))
        if h_reps:
            honest_ts.append(t)
            honest_vals.append(float(np.mean(h_reps)))
        if sy_reps:
            sybil_ts.append(t)
            sybil_vals.append(float(np.mean(sy_reps)))

    honest = (np.array(honest_ts), np.array(honest_vals)) if honest_ts else None
    sybil  = (np.array(sybil_ts),  np.array(sybil_vals))  if sybil_ts  else None
    return {"honest": honest, "sybil": sybil}


def load_one_run(run_dir):
    return get_reputation_series(run_dir)


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

def build_rep_agg(results_by_n, rep_key, n_sybil_list):
    """
    Aggregate mean reputation (honest or sybil) across seeds per n_sybil.
    Returns dict: {n_sybil_str: {"ts": [...], "mean": [...], "std": [...]}}
    """
    agg = {}
    for n in n_sybil_list:
        seed_data = results_by_n.get(n, {})
        series = [v[rep_key] for v in seed_data.values() if v.get(rep_key) is not None]
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
        lo = np.clip(mean - std, 0, 1)
        hi = np.clip(mean + std, 0, 1)
        ax.fill_between(ts, lo, hi, color=color, alpha=0.15, zorder=3)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(honest_rep_agg, sybil_rep_agg):
    fig, axes = plt.subplots(
        2, 1,
        figsize=(7, 6),
        constrained_layout=True,
    )
    fig.suptitle(
        "Experiment 2: Lemon Market — Reputation Dynamics",
        fontweight="bold",
        fontsize=10,
    )

    # --- Row 1: Honest firm reputation ---
    ax = axes[0]
    for n in N_SYBIL_VALUES:
        entry = honest_rep_agg.get(n)
        ls  = "--" if n == 0 else "-"
        lbl = f"n_sybil={n}" + (" (baseline)" if n == 0 else "")
        plot_band(ax, entry, COLORS_N_SYBIL[n], lbl, ls=ls)
    ax.set_ylabel("Mean honest seller reputation")
    ax.set_ylim(0, 1)
    ax.set_title("Mean honest firm reputation over time (mean ± 1σ across seeds)")
    ax.legend(loc="best")
    ax.set_xlabel("Timestep")

    # --- Row 2: Sybil firm reputation ---
    ax = axes[1]
    for n in [3, 6, 9, 12]:
        entry = sybil_rep_agg.get(n)
        plot_band(ax, entry, COLORS_N_SYBIL[n], f"n_sybil={n}")
    ax.set_ylabel("Mean sybil seller reputation")
    ax.set_ylim(0, 1)
    ax.set_title("Mean sybil firm reputation over time (mean ± 1σ across seeds)")
    ax.legend(loc="best")
    ax.set_xlabel("Timestep")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp2 Fig C: Reputation Dynamics")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="car")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "exp2", "exp2_lemon_reputation_quality.pdf"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Ignore cache and rebuild from scratch")
    args = parser.parse_args()

    data_dir   = get_data_dir(args.output)
    cache_path = get_cache_path(data_dir, "exp2_lemon_reputation_quality", args.good)
    run_dirs   = collect_run_dirs(args.logs_dir)

    if not args.force and is_cache_fresh(cache_path, run_dirs, args.logs_dir, args.good):
        print(f"Using cached data: {cache_path}", flush=True)
        raw = load_cache_data(cache_path)
        honest_rep_agg = _deserialize_agg(raw["honest_rep"], N_SYBIL_VALUES)
        sybil_rep_agg  = _deserialize_agg(raw["sybil_rep"],  [3, 6, 9, 12])
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
                has_hon  = data["honest"] is not None
                has_syb  = data["sybil"]  is not None
                print(
                    f"  [{done}/{total}] n_sybil={n} seed={seed}"
                    f" — honest_rep={'ok' if has_hon else 'empty'}"
                    f", sybil_rep={'ok' if has_syb else 'empty'}",
                    flush=True,
                )
                results_by_n[n][seed] = data

        honest_rep_agg = build_rep_agg(results_by_n, "honest", N_SYBIL_VALUES)
        sybil_rep_agg  = build_rep_agg(results_by_n, "sybil",  [3, 6, 9, 12])

        cache_data = {
            "honest_rep": honest_rep_agg,
            "sybil_rep":  sybil_rep_agg,
        }
        save_cache(cache_path, cache_data, args.logs_dir, args.good)
        print(f"Cached data: {cache_path}", flush=True)

        honest_rep_agg = _deserialize_agg(honest_rep_agg, N_SYBIL_VALUES)
        sybil_rep_agg  = _deserialize_agg(sybil_rep_agg,  [3, 6, 9, 12])

    fig = make_figure(honest_rep_agg, sybil_rep_agg)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
