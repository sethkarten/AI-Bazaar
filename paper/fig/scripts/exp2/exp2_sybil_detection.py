"""
Fig Exp2-A: Sybil Detection Rate Over Time

Tracks how well buyers avoid sybil listings across the K × rep_visible sweep.
Metric: sybil pass rate = fraction of seen sybil listings that buyers passed on.
Higher = buyers more successfully reject deceptive listings.

Two panels side-by-side:
  Left:  reputation visible to buyers (rep1 condition)
  Right: reputation hidden from buyers (rep0 condition)
Lines coloured by K (sybil cluster size); shaded bands = ±1σ across seeds.

Directory naming (from exp2.py):
  logs/{name_prefix}/{name_prefix}_baseline_seed{seed}
  logs/{name_prefix}/{name_prefix}_k{k}_rep1_seed{seed}   (rep visible)
  logs/{name_prefix}/{name_prefix}_k{k}_rep0_seed{seed}   (rep hidden)
where name_prefix = exp2_{model_slug}  (e.g. exp2_gemini-2.5-flash).

Usage:
    python paper/fig/scripts/exp2/exp2_sybil_detection.py
    python paper/fig/scripts/exp2/exp2_sybil_detection.py \\
        --logs-dir logs/ --name-prefix exp2_gemini-2.5-flash \\
        --output paper/fig/exp2/exp2_sybil_detection.pdf
    python paper/fig/scripts/exp2/exp2_sybil_detection.py --force
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

# Repo root: 5 levels up from paper/fig/scripts/exp2/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder
from exp2_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data, infer_name_prefix

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

# Okabe-Ito colours by K
COLORS_K = {
    3: "#56B4E9",  # sky blue
    6: "#E69F00",  # orange
    9: "#009E73",  # bluish green
}
LS_REP = {True: "-", False: "--"}   # solid = rep visible, dashed = rep hidden


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

def resolve_run_dir(logs_dir: str, name_prefix: str, k: int, rep_visible: bool, seed: int) -> str | None:
    """Return the run directory path if it exists, else None.

    Canonical layout (exp2.py):
        logs/{name_prefix}/{name_prefix}_baseline_seed{seed}
        logs/{name_prefix}/{name_prefix}_k{k}_rep{1|0}_seed{seed}

    Also checks the legacy flat layout (early prototype runs):
        logs/{name_prefix}_k{k}_rep{1|0}_seed{seed}   (name_prefix = 'exp2' or similar)
    """
    rep_tag = "rep1" if rep_visible else "rep0"
    if k == 0:
        run_name = f"{name_prefix}_baseline_seed{seed}"
    else:
        run_name = f"{name_prefix}_k{k}_{rep_tag}_seed{seed}"

    # Canonical: under logs/{name_prefix}/
    canonical = os.path.join(logs_dir, name_prefix, run_name)
    if os.path.isdir(canonical):
        return canonical

    # Legacy flat: directly under logs/
    flat = os.path.join(logs_dir, run_name)
    if os.path.isdir(flat):
        return flat

    return None


def collect_run_dirs(logs_dir: str, name_prefix: str) -> list[str]:
    dirs = []
    for k in [0] + K_VALUES:
        for rv in ([True] if k == 0 else [True, False]):
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


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def get_pass_rate_series(run_dir: str):
    """Mean sybil pass rate per timestep across all buyers who saw sybil listings.

    Returns (timesteps_array, values_array) or None if no sybil data.
    """
    files = load_state_files(run_dir)
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


def load_one_run(run_dir: str, k: int):
    if k == 0:
        return None   # no sybil listings → pass rate undefined
    return get_pass_rate_series(run_dir)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def interp_common(ts_list, val_list):
    """Interpolate all series onto a common integer grid; return (ts, 2-D array)."""
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


def build_aggregate(results: dict) -> dict:
    """Aggregate per-seed series into {(k, rep_visible): {"ts", "mean", "std"} | None}.

    results keyed by (k, rep_visible, seed) → (ts, vals) | None.
    """
    # Collect unique (k, rep_visible) pairs
    conditions = set((k, rv) for k, rv, _ in results)
    seeds      = set(seed for _, _, seed in results)
    agg = {}
    for k, rv in conditions:
        seed_series = [results[(k, rv, s)] for s in seeds if (k, rv, s) in results]
        seed_series = [s for s in seed_series if s is not None]
        if not seed_series:
            agg[(k, rv)] = None
            continue
        ts_list  = [s[0] for s in seed_series]
        val_list = [s[1] for s in seed_series]
        common, arr = interp_common(ts_list, val_list)
        if common is None:
            agg[(k, rv)] = None
            continue
        agg[(k, rv)] = {
            "ts":   common.tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std":  (arr.std(axis=0).tolist() if arr.shape[0] > 1
                     else np.zeros(len(common)).tolist()),
        }
    return agg


def serialize_agg(agg: dict) -> dict:
    """Convert tuple keys to str for JSON serialisation."""
    return {f"{k},{int(rv)}": v for (k, rv), v in agg.items()}


def deserialize_agg(raw: dict) -> dict:
    """Restore tuple keys from JSON."""
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
    """Two-panel figure: left = rep visible, right = rep hidden."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), constrained_layout=True, sharey=True)
    fig.suptitle(
        "Exp 2 — Sybil detection rate over time",
        fontsize=10, fontweight="bold",
    )

    panel_cfg = [
        (True,  axes[0], "Reputation visible to buyers"),
        (False, axes[1], "Reputation hidden from buyers"),
    ]

    for rep_visible, ax, title in panel_cfg:
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Timestep")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="#555555", lw=1.0, ls="--", alpha=0.6, zorder=2)

        any_plotted = False
        for k in K_VALUES:
            entry = agg.get((k, rep_visible))
            if entry is None:
                continue
            sat = k / 12  # saturation = K / NUM_TOTAL_SELLERS
            lbl = f"K={k} ({sat:.0%} sybil)"
            plot_band(ax, entry, COLORS_K[k], lbl, ls=LS_REP[rep_visible])
            any_plotted = True

        if not any_plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=9)

        ax.legend(loc="lower right", fontsize=7.5)

    axes[0].set_ylabel("Sybil pass rate (fraction passed)")
    axes[1].set_ylabel("")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp2 Fig A: Sybil Detection Rate")
    parser.add_argument(
        "--logs-dir", default="logs/",
        help="Root logs directory (default: logs/).",
    )
    parser.add_argument(
        "--name-prefix", default=None,
        help=(
            "Experiment name prefix, e.g. exp2_gemini-2.5-flash. "
            "If omitted, auto-detected from the first exp2_* subdirectory in --logs-dir."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "exp2", "exp2_sybil_detection.pdf"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Ignore cache and rebuild.")
    # Accepted for compatibility with exp2_run_all.py; not used by this script.
    parser.add_argument("--good", default="car")
    cli = parser.parse_args()

    # Auto-detect name_prefix
    name_prefix = cli.name_prefix
    if name_prefix is None:
        name_prefix = infer_name_prefix(cli.logs_dir)
        print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_run_dirs(cli.logs_dir, name_prefix)
    data_dir   = get_data_dir(cli.output)
    cache_path = get_cache_path(data_dir, "exp2_sybil_detection", "car")

    if not cli.force and is_cache_fresh(cache_path, run_dirs, cli.logs_dir, "car"):
        print(f"Using cached data: {cache_path}", flush=True)
        raw = load_cache_data(cache_path)
        agg = deserialize_agg(raw["agg"])
    else:
        # Build job list
        jobs = []
        for k in K_VALUES:
            for rv in [True, False]:
                for seed in SEEDS:
                    d = resolve_run_dir(cli.logs_dir, name_prefix, k, rv, seed)
                    if d:
                        jobs.append((k, rv, seed, d))
                    else:
                        print(
                            f"  Missing: K={k} rep={int(rv)} seed={seed} "
                            f"(expected under {name_prefix}/)",
                            flush=True,
                        )

        print(f"Loading {len(jobs)} runs ...", flush=True)
        results: dict = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=cli.workers) as ex:
            future_map = {
                ex.submit(load_one_run, d, k): (k, rv, seed)
                for k, rv, seed, d in jobs
            }
            done, total = 0, len(jobs)
            for future in concurrent.futures.as_completed(future_map):
                k, rv, seed = future_map[future]
                done += 1
                data = future.result()
                print(
                    f"  [{done}/{total}] K={k} rep={int(rv)} seed={seed}"
                    f" — {'ok' if data is not None else 'empty'}",
                    flush=True,
                )
                results[(k, rv, seed)] = data

        agg = build_aggregate(results)

        cache_data = {"agg": serialize_agg(agg)}
        save_cache(cache_path, cache_data, cli.logs_dir, "car")
        print(f"Cached: {cache_path}", flush=True)

        agg = deserialize_agg(cache_data["agg"])

    fig = make_figure(agg)
    os.makedirs(os.path.dirname(os.path.abspath(cli.output)), exist_ok=True)
    fig.savefig(cli.output)
    print(f"Saved: {cli.output}")


if __name__ == "__main__":
    main()
