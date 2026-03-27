"""
Shared directory resolution, state loading, and aggregation helpers for Exp2 figure scripts.

Import pattern in each script:
    from exp2_common import (
        SEEDS, K_VALUES, COLORS_K, LS_REP,
        resolve_run_dir, collect_all_run_dirs,
        load_state_files, interp_common, build_aggregate,
        serialize_agg, deserialize_agg, plot_band,
    )
"""

import glob
import json
import os

import numpy as np

# --- Experiment constants (must match exp2.py) ---
SEEDS    = [8, 16, 64]
K_VALUES = [3, 6, 9]   # sybil cluster sizes; 0 = baseline

# Okabe-Ito colours by K
COLORS_K = {0: "#999999", 3: "#56B4E9", 6: "#E69F00", 9: "#009E73"}
# Linestyle by rep_visible
LS_REP   = {True: "-", False: "--"}


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

def resolve_run_dir(logs_dir: str, name_prefix: str, k: int, rep_visible: bool, seed: int) -> str | None:
    """Return run directory path if it exists, else None.

    Canonical: logs/{name_prefix}/{run_name}/
    Flat fallback: logs/{run_name}/  (legacy prototype runs)
    """
    rep_tag = "rep1" if rep_visible else "rep0"
    run_names = (
        [f"{name_prefix}_k0_{rep_tag}_seed{seed}", f"{name_prefix}_baseline_seed{seed}"]
        if k == 0
        else [f"{name_prefix}_k{k}_{rep_tag}_seed{seed}"]
    )
    for run_name in run_names:
        canonical = os.path.join(logs_dir, name_prefix, run_name)
        if os.path.isdir(canonical):
            return canonical
        flat = os.path.join(logs_dir, run_name)
        if os.path.isdir(flat):
            return flat
    return None


def collect_all_run_dirs(logs_dir: str, name_prefix: str, include_baseline: bool = True) -> list[str]:
    """Collect all existing run dirs across K × rep_visible × seed."""
    dirs = []
    k_list = ([0] + K_VALUES) if include_baseline else K_VALUES
    for k in k_list:
        rep_opts = [True, False]
        for rv in rep_opts:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, name_prefix, k, rv, seed)
                if d:
                    dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# State loading
# ---------------------------------------------------------------------------

def load_state_files(run_dir: str) -> list[str]:
    """Return sorted, valid state_t*.json paths from run_dir."""
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


def load_firm_types(run_dir: str) -> dict:
    """Return {firm_name: is_sybil} from firm_attributes.json, or {} if absent."""
    attr_path = os.path.join(run_dir, "firm_attributes.json")
    if not os.path.exists(attr_path):
        return {}
    try:
        with open(attr_path) as f:
            attrs = json.load(f)
        if isinstance(attrs, list):
            return {a.get("name", f"firm_{i}"): bool(a.get("sybil", False))
                    for i, a in enumerate(attrs)}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def interp_common(ts_list, val_list):
    """Interpolate all series onto a common integer grid; return (ts_array, 2-D array)."""
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
    """Aggregate per-seed series into {(k, rep_visible): {"ts","mean","std"} | None}.

    results is keyed by (k, rep_visible, seed) → (ts_array, val_array) | None.
    """
    conditions = set((k, rv) for k, rv, _ in results)
    seeds      = set(seed for _, _, seed in results)
    agg = {}
    for k, rv in conditions:
        seed_series = [results[(k, rv, s)]
                       for s in seeds if (k, rv, s) in results
                       and results[(k, rv, s)] is not None]
        if not seed_series:
            agg[(k, rv)] = None
            continue
        common, arr = interp_common([s[0] for s in seed_series], [s[1] for s in seed_series])
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
# Plot helper
# ---------------------------------------------------------------------------

def plot_band(ax, entry, color, label, ls="-", lw=1.8, alpha_band=0.15):
    if entry is None:
        return
    ts, mean, std = entry["ts"], entry["mean"], entry["std"]
    ax.plot(ts, mean, color=color, lw=lw, ls=ls, label=label, zorder=4)
    if np.any(std > 0):
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=alpha_band, zorder=3)
