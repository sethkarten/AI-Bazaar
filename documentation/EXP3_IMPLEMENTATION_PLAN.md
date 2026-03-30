# Experiment 3: Systemic Resilience — Implementation Plan

## Overview

Experiment 3 extends Experiments 1 (Crash) and 2 (Lemon) with mid-episode adversarial
shocks. It measures how quickly LLM-driven agents detect regime changes and re-establish
equilibrium.

| Variant | Shock | Timing | Mechanism |
|---------|-------|--------|-----------|
| **Supply Shock** (Crash) | Unit cost doubles: $c → c' = \$10.00$ | $t = 25$ | Firm cost dicts mutated in-place |
| **Flood of Fakes** (Lemon) | Sybil cluster scales: $K = 5 → K = 45$ | $t = 15$ | DeceptivePrincipal spawns 40 new identities |

### Recovery Metrics

**Crash — Markup Ratio Recovery**

$$\mu_t = \frac{\bar{p}_t}{\bar{c}_t} - 1$$

Recovery at first $t > t_s$ where $|\mu_t - \bar{\mu}_{\text{pre}}| \leq 0.1 \cdot \bar{\mu}_{\text{pre}}$,
with $\bar{\mu}_{\text{pre}}$ averaged over $[t_s - 5, t_s)$.

Rationale: absolute price *should* change after a cost shock; the competitive markup ratio
should not. This reuses Exp1's price distortion metric.

**Lemon — Per-Step Detection Premium Recovery**

$$\delta_t = \mathbb{E}_{b \in B_t}\!\left[\text{honest\_purchase\_rate}_{b,t} - \text{sybil\_purchase\_rate}_{b,t}\right]$$

Recovery at first $t > t_s$ where $|\delta_t - \bar{\delta}_{\text{pre}}| \leq 0.1 \cdot \bar{\delta}_{\text{pre}}$,
with $\bar{\delta}_{\text{pre}}$ averaged over $[t_s - 5, t_s)$.

Must use **per-step** pass rates (`sybil_pass_rate_this_step`, `honest_pass_rate_this_step`),
not cumulative totals, so the metric isn't diluted by pre-shock history.

**Edge case:** if $\bar{\mu}_{\text{pre}}$ or $\bar{\delta}_{\text{pre}} < 0.05$, use absolute
threshold $\epsilon = 0.05$ instead of the 10% relative band.

---

## Phase 1: CLI & Argument Plumbing

### 1.1 Add shock arguments to `ai_bazaar/main.py::create_argument_parser()`

After the existing `--max-timesteps` block (~line 321), add:

```python
shock_group = parser.add_argument_group("Experiment 3: Shock Parameters")
shock_group.add_argument(
    "--shock-timestep", type=int, default=None,
    help="Timestep at which to inject the shock (None = no shock)."
)
shock_group.add_argument(
    "--post-shock-unit-cost", type=float, default=None,
    help="New unit cost after supply shock (Crash variant). "
         "Applied to all firms at --shock-timestep."
)
shock_group.add_argument(
    "--post-shock-sybil-cluster-size", type=int, default=None,
    help="New sybil cluster size after flood shock (Lemon variant). "
         "Identities added to DeceptivePrincipal at --shock-timestep."
)
```

### 1.2 Validation in `main()`

After argument post-processing (~line 627), add validation:

- If `--post-shock-unit-cost` is set, require `--shock-timestep` and
  `--consumer-scenario THE_CRASH`.
- If `--post-shock-sybil-cluster-size` is set, require `--shock-timestep` and
  `--consumer-scenario LEMON_MARKET`.
- Both shock types cannot be active simultaneously.

---

## Phase 2: Shock Injection in the Simulation Loop

### 2.1 Add `apply_shock()` method to `BazaarWorld` (`bazaar_env.py`)

Place near the existing `step()` method (~line 463). Two sub-methods:

#### `_apply_cost_shock(self, new_unit_cost: float)`

```python
def _apply_cost_shock(self, new_unit_cost: float):
    """Double unit cost for all firms mid-episode (Crash variant)."""
    for firm in self.firms:
        for good in firm.supply_unit_costs:
            firm.supply_unit_costs[good] = new_unit_cost
        # FixedFirmAgent uses a separate .unit_costs dict
        if hasattr(firm, "unit_costs"):
            for good in firm.unit_costs:
                firm.unit_costs[good] = new_unit_cost
    self._shock_applied = True
    self._shock_type = "cost"
    self._shock_timestep = self.timestep
    logger.info(f"SHOCK APPLIED: unit cost → {new_unit_cost} at t={self.timestep}")
```

Key details:
- Mutate `firm.supply_unit_costs` (used by `FirmAgent` for LLM pricing context)
- Mutate `firm.unit_costs` (used by `FixedFirmAgent.set_price()`, line 1313–1314 of `firm.py`)
- Firms' LLM prompt context already reads from `supply_unit_costs`, so the next pricing
  call will see the new cost with no additional wiring needed.

#### `_apply_sybil_flood(self, new_k: int)`

```python
def _apply_sybil_flood(self, new_k: int):
    """Scale sybil cluster from current K to new_k identities (Lemon variant)."""
    if not self.deceptive_principal:
        raise ValueError("No DeceptivePrincipal to flood")
    principal = self.deceptive_principal
    current_k = len(principal.identities)
    to_add = new_k - current_k
    if to_add <= 0:
        return
    for _ in range(to_add):
        principal.identity_counter += 1
        new_id = SybilIdentity(
            name=f"Seller_{principal.identity_counter}",
            ...  # mirror construction from sybil.py lines 140–153
        )
        principal.identities.append(new_id)
    # Update world.firms list so marketplace sees new sellers
    self.firms = [f for f in self.firms if f not in principal.identities[:current_k]] \
                 + list(principal.identities)
    # Re-register with ledger
    ...
    self._shock_applied = True
    self._shock_type = "sybil_flood"
    self._shock_timestep = self.timestep
    logger.info(f"SHOCK APPLIED: sybil K {current_k} → {new_k} at t={self.timestep}")
```

Key details:
- Reuse `SybilIdentity` construction from `sybil.py` `__init__` (lines 140–153).
- New identities start with `R_0 = 0.8` (same as init), so buyers must learn they're
  deceptive — this is the challenge.
- Must update `self.firms` (line 166 of init) and register new identities in the ledger
  (money, inventory) so the marketplace loop handles them.
- `self.deceptive_principal.k` should be updated to `new_k` for consistency.

### 2.2 Hook into the simulation loop (`main.py`)

In `run_marketplace_simulation()` (~line 157), inject the shock check:

```python
while not world.is_done():
    # --- Shock injection ---
    if (args.shock_timestep is not None
            and world.timestep == args.shock_timestep
            and not getattr(world, '_shock_applied', False)):
        if args.post_shock_unit_cost is not None:
            world._apply_cost_shock(args.post_shock_unit_cost)
        if args.post_shock_sybil_cluster_size is not None:
            world._apply_sybil_flood(args.post_shock_sybil_cluster_size)
    # --- End shock injection ---
    stats = world.step()
```

The shock fires **before** the step at $t_s$, so agents observe the new regime during
step $t_s$.

---

## Phase 3: State Metadata for Shocks

### 3.1 Extend `save_state()` in `bazaar_env.py` (~line 1169)

Add a `"shock"` key to the state dict:

```python
state["shock"] = {
    "applied": getattr(self, "_shock_applied", False),
    "type": getattr(self, "_shock_type", None),
    "shock_timestep": getattr(self, "_shock_timestep", None),
    "post_shock_unit_cost": getattr(self, "_post_shock_unit_cost", None),
    "post_shock_sybil_k": getattr(self, "_post_shock_sybil_k", None),
}
```

This makes every `state_t*.json` self-describing for downstream analysis, and is
backward-compatible (old runs simply have `shock.applied = False`).

---

## Phase 4: Experiment Runner Script

### 4.1 Create `scripts/exp3.py`

Model after `scripts/exp2.py`. The matrix:

**Crash variant** (extends Exp1 grid):
- Inherit Exp1's `_BASE_FIXED` args (THE_CRASH, 5 firms, 50 consumers, etc.)
- Add: `--shock-timestep 25 --post-shock-unit-cost 10.0`
- Sweep: same `k × dlc` grid as Exp1 (or a subset, e.g. `k ∈ {0, 3, 5}`, `dlc ∈ {1, 3, 5}`)
- 10 reps per cell, same seeds as Exp1

**Lemon variant** (extends Exp2 grid):
- Inherit Exp2's `_BASE_FIXED` args (LEMON_MARKET, 12 buyers, reputation, etc.)
- Add: `--shock-timestep 15 --post-shock-sybil-cluster-size 45`
- Sweep: same `k × persona` grid as Exp2 (or subset)
- Extend `--max-timesteps` to 45 (need runway post-shock for recovery)
- 10 reps per cell, same seeds as Exp2

Output logs to `logs/exp3_crash_{model}/` and `logs/exp3_lemon_{model}/`.

### 4.2 Update `documentation/RUN_COMMANDS.md`

Add Experiment 3 section with example commands for both variants.

---

## Phase 5: Recovery Metric Computation (Analysis Scripts)

### 5.1 Create `paper/fig/scripts/exp3/exp3_common.py`

Shared utilities:

```python
def find_shock_timestep(run_dir: str) -> int | None:
    """Read state files, return the timestep where shock.applied first becomes True."""
    ...

def compute_markup_ratio_series(run_dir: str, good: str = "widget") -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps, mu_t) where mu_t = mean_price_t / unit_cost_t - 1."""
    ...

def compute_perstep_detection_premium_series(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps, delta_t) using per-step pass rates (not cumulative)."""
    ...

def compute_recovery_time(
    ts: np.ndarray,
    metric: np.ndarray,
    shock_t: int,
    window: int = 5,
    rel_threshold: float = 0.10,
    abs_threshold: float = 0.05,
    max_T: int | None = None,
) -> int:
    """Timesteps to recovery. Returns steps after shock, or max_T if never."""
    pre_mask = (ts >= shock_t - window) & (ts < shock_t)
    baseline = metric[pre_mask].mean()
    threshold = max(rel_threshold * abs(baseline), abs_threshold)
    post_mask = ts > shock_t
    for t, val in zip(ts[post_mask], metric[post_mask]):
        if abs(val - baseline) <= threshold:
            return int(t - shock_t)
    return max_T or int(ts.max() - shock_t)
```

### 5.2 Create `paper/fig/scripts/exp3/exp3_crash_recovery.py`

Figure: time series of $\mu_t$ across `k` values, with vertical shock line at $t=25$
and horizontal band showing the 10% recovery zone around $\bar{\mu}_{\text{pre}}$.

Inset or companion panel: bar chart of recovery time $\tau_{\text{crash}}$ by `k`.

### 5.3 Create `paper/fig/scripts/exp3/exp3_lemon_recovery.py`

Figure: time series of per-step $\delta_t$ across `k` values, with vertical shock line
at $t=15$ and horizontal recovery band.

Inset or companion panel: bar chart of recovery time $\tau_{\text{lemon}}$ by `k`.

### 5.4 Create `paper/fig/scripts/exp3/exp3_run_all.py`

Runner that calls both figure scripts, matching the pattern of `exp1_run_all.py` and
`exp2_run_all.py`.

### 5.5 Secondary metric figures (optional, lower priority)

- `exp3_crash_bankruptcy.py`: post-shock bankruptcy rate vs. k
- `exp3_crash_volatility.py`: price volatility in recovery window
- `exp3_lemon_revenue_share.py`: sybil revenue share time series through the flood
- `exp3_lemon_welfare.py`: buyer welfare time series through the flood

---

## Phase 6: Paper Updates

### 6.1 Fix `paper/sections/experimental_setup.tex` (line 70–71)

Replace the current primary metric text:

```latex
\textbf{Primary Metrics.}
\begin{itemize}
    \item \textbf{Markup Recovery Time} (Crash): Timesteps after shock until
    $|\mu_t - \bar{\mu}_{\mathrm{pre}}| \leq 0.1\,\bar{\mu}_{\mathrm{pre}}$,
    where $\mu_t = \bar{p}_t / \bar{c}_t - 1$ is the mean price markup ratio
    and $\bar{\mu}_{\mathrm{pre}}$ is averaged over $[t_s{-}5,\; t_s)$.
    \item \textbf{Detection Premium Recovery Time} (Lemon): Timesteps after shock until
    $|\delta_t - \bar{\delta}_{\mathrm{pre}}| \leq 0.1\,\bar{\delta}_{\mathrm{pre}}$,
    where $\delta_t$ is the per-step sybil detection premium and
    $\bar{\delta}_{\mathrm{pre}}$ is averaged over $[t_s{-}5,\; t_s)$.
    \item Markets that never recover are assigned $\tau = T$ (worst case).
\end{itemize}
```

### 6.2 Update `paper/sections/results.tex`

Add Exp3 results subsection with figure references.

---

## Implementation Order & Dependencies

```
Phase 1 (CLI args)
  │
  ▼
Phase 2 (shock injection)  ←  depends on Phase 1
  │
  ├──► Phase 3 (state metadata)  ←  can parallel with Phase 4
  │
  ▼
Phase 4 (exp3 runner script)  ←  depends on Phases 1–3
  │
  ▼
  [RUN EXPERIMENTS]
  │
  ▼
Phase 5 (analysis & figures)  ←  depends on log data from Phase 4
  │
  ▼
Phase 6 (paper updates)  ←  depends on Phase 5 results
```

Phases 1–3 are the critical path. Phase 4 is mechanical once Phases 1–3 are done.
Phase 5 can be scaffolded in parallel with experimentation.

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM firms don't "see" the cost change in their prompt context | Verify `supply_unit_costs` is read fresh each step in the firm's pricing prompt construction. Add integration test. |
| New sybil identities not properly registered in ledger/marketplace | Write a unit test that adds identities mid-episode and verifies they appear in listings and can transact. |
| Per-step detection premium is too noisy with 12 buyers | Use a 3-step rolling average for the recovery check (not the pre-shock baseline). |
| Recovery threshold edge case when baseline ≈ 0 | Absolute fallback threshold $\epsilon = 0.05$. |
| Cumulative detection premium script used by mistake | Name the new function clearly (`compute_perstep_detection_premium_series`) and add a deprecation note to the cumulative version pointing to the per-step variant for Exp3. |
