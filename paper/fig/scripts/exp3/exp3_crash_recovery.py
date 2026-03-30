"""
exp3_crash_recovery.py — markup ratio recovery after supply cost shock.

Usage
-----
  python exp3_crash_recovery.py
  python exp3_crash_recovery.py --logs-dir logs/exp3a_gemini-2.5-flash/

Reads state_t*.json files under each run directory, computes the markup
ratio series mu_t = mean_price_t / mean_unit_cost_t - 1, finds the shock
timestep, and reports recovery time (steps back to within 10% of
pre-shock baseline).

# TODO: add matplotlib figure once exp3 data is available
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exp3_common import (
    find_shock_timestep,
    compute_markup_ratio_series,
    compute_recovery_time,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp3a crash recovery summary — markup ratio after supply shock"
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/exp3a_gemini-2.5-flash/",
        help="Top-level log directory containing per-run subdirectories.",
    )
    parser.add_argument(
        "--window", type=int, default=5,
        help="Pre-shock window size for baseline estimation (default: 5).",
    )
    parser.add_argument(
        "--rel-threshold", type=float, default=0.10,
        help="Relative recovery threshold (default: 0.10 = 10%%).",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        print(f"Logs directory not found: {logs_dir}")
        return

    run_dirs = sorted(d for d in logs_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No run subdirectories found under {logs_dir}")
        return

    # Header
    col_w = 70
    print(f"\n{'Run':<{col_w}}  {'shock_t':>7}  {'recovery_steps':>14}  {'baseline_mu':>11}  {'post_mu':>7}")
    print("-" * (col_w + 45))

    for run_dir in run_dirs:
        state_files = list(run_dir.glob("state_t*.json"))
        if not state_files:
            continue

        shock_t = find_shock_timestep(str(run_dir))
        ts, mu = compute_markup_ratio_series(str(run_dir))

        if len(ts) == 0:
            print(f"  {run_dir.name:<{col_w-2}}  no data")
            continue

        if shock_t is None:
            print(f"  {run_dir.name:<{col_w-2}}  {'—':>7}  {'no shock':>14}")
            continue

        recovery = compute_recovery_time(
            ts, mu, shock_t,
            window=args.window,
            rel_threshold=args.rel_threshold,
        )

        # Pre-shock baseline mu
        pre_mask = (ts >= shock_t - args.window) & (ts < shock_t)
        baseline_mu = float(mu[pre_mask].mean()) if pre_mask.any() else float("nan")

        # Mean mu in first 5 steps after shock
        post_mask = (ts > shock_t) & (ts <= shock_t + 5)
        post_mu = float(mu[post_mask].mean()) if post_mask.any() else float("nan")

        print(
            f"  {run_dir.name:<{col_w-2}}"
            f"  {shock_t:>7}"
            f"  {recovery:>14}"
            f"  {baseline_mu:>11.3f}"
            f"  {post_mu:>7.3f}"
        )

    print()
    # TODO: add matplotlib figure once exp3 data is available


if __name__ == "__main__":
    main()
