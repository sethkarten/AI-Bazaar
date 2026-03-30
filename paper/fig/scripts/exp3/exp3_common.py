"""
exp3_common.py — shared utilities for Experiment 3 analysis scripts.

Functions
---------
find_shock_timestep(run_dir)
    Return the timestep where shock.applied first becomes True.

compute_markup_ratio_series(run_dir)
    Return (timesteps, mu_t) where mu_t = mean_price_t / mean_unit_cost_t - 1.

compute_perstep_detection_premium_series(run_dir)
    Return (timesteps, delta_t) = sybil_pass_rate - honest_pass_rate per step.

compute_recovery_time(ts, metric, shock_t, ...)
    Timesteps until metric returns to within threshold of pre-shock baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_state_files(run_dir: str):
    """Yield (timestep, state_dict) pairs from state_t*.json files, sorted."""
    p = Path(run_dir)
    files = sorted(p.glob("state_t*.json"), key=lambda f: int(f.stem.split("_t")[1]))
    for f in files:
        t = int(f.stem.split("_t")[1])
        with open(f, encoding="utf-8") as fh:
            yield t, json.load(fh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_shock_timestep(run_dir: str) -> int | None:
    """Return timestep where shock.applied first becomes True from state files.

    Reads state_t*.json files in order and returns the first ``t`` where
    ``state["shock"]["applied"]`` is True.  Returns None if no shock is found.
    """
    for t, state in _iter_state_files(run_dir):
        shock = state.get("shock", {})
        if shock.get("applied", False):
            # Prefer the explicit shock_timestep field if present
            return int(shock.get("shock_timestep") or t)
    return None


def compute_markup_ratio_series(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps, mu_t) where mu_t = mean_price_t / mean_unit_cost_t - 1.

    For each state file:
    - ``mean_price_t`` = mean of ``current_prices[good]`` across firms that are
      in business and have a non-zero price.
    - ``mean_unit_cost_t`` = mean of ``supply_unit_costs[good]`` across firms.
      Falls back to ``shock.post_shock_unit_cost`` for steps after the shock if
      individual firm costs are not recorded.

    Returns arrays of equal length sorted by timestep.
    """
    timesteps = []
    mu_series = []

    post_shock_cost: float | None = None

    for t, state in _iter_state_files(run_dir):
        # Track post-shock cost from metadata once available
        shock = state.get("shock", {})
        if shock.get("applied") and post_shock_cost is None:
            psc = shock.get("post_shock_unit_cost")
            if psc is not None:
                post_shock_cost = float(psc)

        firms = state.get("firms", [])
        prices = []
        costs = []
        for firm in firms:
            # Skip bankrupt / out-of-business firms
            if not firm.get("in_business", True):
                continue
            # Prices: current_prices dict, take first/only good value
            cp = firm.get("current_prices", {})
            if cp:
                p_vals = [v for v in cp.values() if v and float(v) > 0]
                if p_vals:
                    prices.append(float(p_vals[0]))
            # Costs: supply_unit_costs dict; fallback to post_shock_cost
            suc = firm.get("supply_unit_costs", {})
            if suc:
                c_vals = [float(v) for v in suc.values() if v is not None]
                if c_vals:
                    costs.append(c_vals[0])
            elif post_shock_cost is not None:
                costs.append(post_shock_cost)

        if prices and costs:
            mean_p = float(np.mean(prices))
            mean_c = float(np.mean(costs))
            mu = (mean_p / mean_c - 1.0) if mean_c > 0 else 0.0
        else:
            mu = 0.0

        timesteps.append(t)
        mu_series.append(mu)

    return np.array(timesteps, dtype=float), np.array(mu_series, dtype=float)


def compute_perstep_detection_premium_series(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps, delta_t) from per-step pass rates in state files.

    delta_t = sybil_pass_rate_this_step - honest_pass_rate_this_step.

    These fields are written by the lemon market step under the keys
    ``sybil_pass_rate_this_step`` and ``honest_pass_rate_this_step`` at the
    top level of the state dict (or inside a ``lemon_market_stats`` block if
    present).  If neither field is found for a timestep, that timestep is
    skipped.
    """
    timesteps = []
    delta_series = []

    for t, state in _iter_state_files(run_dir):
        # Try top-level keys first
        sybil_rate = state.get("sybil_pass_rate_this_step")
        honest_rate = state.get("honest_pass_rate_this_step")

        # Fallback: inside a nested lemon_market_stats block
        if sybil_rate is None or honest_rate is None:
            lm_stats = state.get("lemon_market_stats", {})
            if sybil_rate is None:
                sybil_rate = lm_stats.get("sybil_pass_rate_this_step")
            if honest_rate is None:
                honest_rate = lm_stats.get("honest_pass_rate_this_step")

        if sybil_rate is not None and honest_rate is not None:
            timesteps.append(t)
            delta_series.append(float(sybil_rate) - float(honest_rate))

    return np.array(timesteps, dtype=float), np.array(delta_series, dtype=float)


def compute_recovery_time(
    ts: np.ndarray,
    metric: np.ndarray,
    shock_t: int,
    window: int = 5,
    rel_threshold: float = 0.10,
    abs_threshold: float = 0.05,
    max_T: int | None = None,
) -> int:
    """Timesteps to recovery after shock_t.

    Parameters
    ----------
    ts : np.ndarray
        Timestep indices (sorted).
    metric : np.ndarray
        Metric values aligned with ts.
    shock_t : int
        Timestep at which the shock was applied.
    window : int
        Number of pre-shock steps used to estimate the baseline.
    rel_threshold : float
        Relative tolerance: |metric - baseline| <= rel_threshold * |baseline|.
    abs_threshold : float
        Absolute floor tolerance (used when baseline is near zero).
    max_T : int or None
        If the metric never recovers, return this value.  Defaults to
        ts.max() - shock_t.

    Returns
    -------
    int
        Number of steps after the shock until recovery, or max_T if never.
    """
    pre_mask = (ts >= shock_t - window) & (ts < shock_t)
    baseline = float(metric[pre_mask].mean()) if pre_mask.any() else 0.0
    threshold = max(rel_threshold * abs(baseline), abs_threshold)

    post_mask = ts > shock_t
    for t, val in zip(ts[post_mask], metric[post_mask]):
        if abs(val - baseline) <= threshold:
            return int(t - shock_t)

    return int(max_T if max_T is not None else (ts.max() - shock_t))
