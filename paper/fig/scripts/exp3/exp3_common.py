"""
exp3_common.py — shared utilities for Experiment 3 analysis scripts.

Functions
---------
find_shock_timestep(run_dir)
    Return the timestep where shock.applied first becomes True.

compute_markup_ratio_series(run_dir)
    Return (timesteps, mu_t) where mu_t = mean_price_t / mean_unit_cost_t - 1.

compute_bankruptcy_series(run_dir)
    Return (timesteps, dead_count_t) — number of bankrupt firms per step.

compute_consumer_surplus_series(run_dir)
    Return (timesteps, cs_t) — per-step average buyer consumer surplus.

compute_perstep_detection_premium_series(run_dir)
    Return (timesteps, delta_t) = sybil_pass_rate - honest_pass_rate per step.

rolling_mean(values, k)
    Trailing rolling mean with window size k.

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
    - ``mean_price_t`` = mean of ``firm["prices"][good]`` across firms that are
      in business and have a non-zero price.
    - ``mean_unit_cost_t`` = mean of
      ``firm["expenses_info"]["supply_by_good"][0]["unit_cost"]`` across firms.
      Falls back to ``shock.post_shock_unit_cost`` after the shock is applied,
      or 1.0 otherwise, when ``supply_by_good`` is absent / empty for a firm.

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

            # Prices: firm["prices"] is a dict like {"food": 1.79}
            price_dict = firm.get("prices", {})
            if price_dict:
                p_vals = [float(v) for v in price_dict.values() if v is not None and float(v) > 0]
                if p_vals:
                    prices.append(p_vals[0])

            # Costs: expenses_info.supply_by_good[0].unit_cost
            cost_found = False
            expenses_info = firm.get("expenses_info", {})
            supply_by_good = expenses_info.get("supply_by_good", []) if expenses_info else []
            if supply_by_good:
                uc = supply_by_good[0].get("unit_cost")
                if uc is not None:
                    costs.append(float(uc))
                    cost_found = True

            if not cost_found:
                # Fallback: use post-shock cost if shock applied, else 1.0
                fallback = post_shock_cost if post_shock_cost is not None else 1.0
                costs.append(fallback)

        if prices and costs:
            mean_p = float(np.mean(prices))
            mean_c = float(np.mean(costs))
            mu = (mean_p / mean_c - 1.0) if mean_c > 0 else 0.0
        else:
            mu = 0.0

        timesteps.append(t)
        mu_series.append(mu)

    return np.array(timesteps, dtype=float), np.array(mu_series, dtype=float)


def compute_bankruptcy_series(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps, dead_count_t) — number of bankrupt firms per step.

    For each state file, counts firms where ``in_business`` is False.
    Timesteps are aligned with those from :func:`compute_markup_ratio_series`
    (both iterate the same ``state_t*.json`` files).
    """
    timesteps: list[int] = []
    dead_counts: list[int] = []

    for t, state in _iter_state_files(run_dir):
        firms = state.get("firms", [])
        dead = sum(1 for f in firms if not f.get("in_business", True))
        timesteps.append(t)
        dead_counts.append(dead)

    return np.array(timesteps, dtype=float), np.array(dead_counts, dtype=float)


def compute_consumer_surplus_series(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps, cs_t) — per-step average buyer consumer surplus.

    Reads ``lemon_market_avg_consumer_surplus`` from each state file.
    Timesteps where the field is absent are skipped.
    """
    timesteps: list[int] = []
    cs_series: list[float] = []

    for t, state in _iter_state_files(run_dir):
        cs = state.get("lemon_market_avg_consumer_surplus")
        if cs is not None:
            timesteps.append(t)
            cs_series.append(float(cs))

    return np.array(timesteps, dtype=float), np.array(cs_series, dtype=float)


def rolling_mean(values: np.ndarray, k: int) -> np.ndarray:
    """Trailing rolling mean with window *k*.

    For indices < k the mean is taken over all available preceding values
    (i.e. the window shrinks at the start of the series).
    """
    out = np.empty_like(values, dtype=float)
    cumsum = np.cumsum(values)
    for i in range(len(values)):
        lo = max(0, i - k + 1)
        span = i - lo + 1
        out[i] = (cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0.0)) / span
    return out


def compute_perstep_detection_premium_series(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps, delta_t) from per-step pass rates in state files.

    delta_t = sybil_pass_rate_this_step - honest_pass_rate_this_step.

    Looks for the fields in three places (in order):
      1. Top-level state keys.
      2. Inside a ``lemon_market_stats`` block.
      3. Averaged across ``consumers[]`` entries (each consumer carries its
         own ``sybil_pass_rate_this_step`` / ``honest_pass_rate_this_step``).

    If no data is found for a timestep, that timestep is skipped.
    """
    timesteps = []
    delta_series = []

    for t, state in _iter_state_files(run_dir):
        sybil_rate = state.get("sybil_pass_rate_this_step")
        honest_rate = state.get("honest_pass_rate_this_step")

        if sybil_rate is None or honest_rate is None:
            lm_stats = state.get("lemon_market_stats", {})
            if sybil_rate is None:
                sybil_rate = lm_stats.get("sybil_pass_rate_this_step")
            if honest_rate is None:
                honest_rate = lm_stats.get("honest_pass_rate_this_step")

        # Fallback: aggregate per-consumer pass rates
        if sybil_rate is None or honest_rate is None:
            consumers = state.get("consumers", [])
            sybil_vals, honest_vals = [], []
            for c in consumers:
                sv = c.get("sybil_pass_rate_this_step")
                hv = c.get("honest_pass_rate_this_step")
                if sv is not None:
                    sybil_vals.append(float(sv))
                if hv is not None:
                    honest_vals.append(float(hv))
            if sybil_vals and sybil_rate is None:
                sybil_rate = float(np.mean(sybil_vals))
            if honest_vals and honest_rate is None:
                honest_rate = float(np.mean(honest_vals))

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
