"""
Exp3 Composite EAS — Systemic Resilience (Shock Recovery).

Placeholder script.  Once Experiment 3 runs are complete, this will compute
a recovery-based EAS per model:

  EAS_exp3 = (1/3) * [ Φ̂_recovery + Φ̂_post_welfare + Φ̂_post_integrity ]

  Φ_recovery       = 1 - (recovery_time / T_max)   (faster recovery = better)
  Φ_post_welfare   = normalised post-shock consumer surplus
  Φ_post_integrity = 1 - post-shock deceptive transaction rate

For now, returns NaN for all models so that the Exp4 Pareto script can
gracefully skip Exp3 contributions.

Usage:
    python exp3_score.py [--logs-dir logs/] [--good food] [--output ...]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from exp3_common import (
    find_shock_timestep, compute_markup_ratio_series,
    compute_perstep_detection_premium_series, compute_recovery_time,
)

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_FIG_EXP3_DIR  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "exp3"))
DEFAULT_OUTPUT = os.path.join(_FIG_EXP3_DIR, "exp3_score.pdf")
_CACHE_DIR     = os.path.join(_FIG_EXP3_DIR, "data")


# ---------------------------------------------------------------------------
# Cache utilities (minimal, mirrors exp1/exp2 pattern)
# ---------------------------------------------------------------------------

def _cache_path(good: str) -> str:
    return os.path.join(_CACHE_DIR, f"exp3_score_{good}.json")


def save_cache(path: str, data: dict, logs_dir: str, good: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "_meta": {
            "logs_dir": os.path.abspath(logs_dir),
            "good":     good,
            "created":  time.time(),
        },
        "data": data,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_cache(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)["data"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# EAS computation — placeholder
# ---------------------------------------------------------------------------

def compute_exp3_eas(logs_dir: str, good: str) -> dict | None:
    """Compute Exp3 aggregate EAS.

    TODO: iterate over model run dirs, call compute_markup_ratio_series /
    compute_perstep_detection_premium_series, derive recovery times, and
    build the composite score.

    Returns {"mean": float, "se": float, "n": int} or None.
    """
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Exp3 EAS Score (placeholder)")
    ap.add_argument("--logs-dir", default="logs/")
    ap.add_argument("--good", default="food")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cp = _cache_path(args.good)
    cached = None if args.force else load_cache(cp)

    if cached is not None:
        print(f"Using cached data: {cp}", flush=True)
        agg = cached.get("aggregate_eas")
    else:
        agg = compute_exp3_eas(args.logs_dir, args.good)
        save_cache(cp, {"aggregate_eas": agg}, args.logs_dir, args.good)
        print(f"Cached: {cp}", flush=True)

    if agg:
        print(f"Exp3 EAS: {agg['mean']:.3f} ± {agg['se']:.3f}  (n={agg['n']})")
    else:
        print("Exp3 EAS: not yet computed (placeholder — run experiments first).")

    print(f"\nNo figure generated (placeholder script).")
    print(f"Output would be: {args.output}")


if __name__ == "__main__":
    main()
