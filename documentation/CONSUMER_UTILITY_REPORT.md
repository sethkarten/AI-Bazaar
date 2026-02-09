# Report: Why Consumer Utilities Often Start Very Negative

This report analyzes why consumer utility (the value in state and on dashboards) is often strongly negative in early timesteps, by tracing where it is computed and when it is recorded.

---

## 0. What gets reported as "consumer utility"

There is a single notion of consumer utility in the codebase: **total utility = CES utility from consumption minus disutility of labor**. It is computed on demand and written to state at save time.

| Concept | Where defined | Formula | Saved to state? | Used by dashboard? |
|--------|----------------|---------|------------------|--------------------|
| **Consumer utility** | `CESConsumerAgent.compute_utility()` in `consumer.py` | `goods_utility - labor_disutility` (see below) | **Yes** — `_build_consumers_state()` uses `getattr(c, "utility", 0)` | **Yes** — dashboard uses `state['consumers']` and can show mean utility |

- **CESConsumerAgent:** `c.utility` is a property that calls `compute_utility()` (CES from current inventory minus labor disutility).
- **FixedConsumerAgent:** `utility` is the sum of inventory quantities (no labor term).
- **State:** `save_state()` runs at the **end** of each `step()`, so the saved utility is the value **after** that step’s labor choice, income, orders, and market clearing.

---

## 1. Where utility is computed (the value that appears in state)

**Location:** `ai_bazaar/agents/consumer.py` (`CESConsumerAgent`)

```python
def compute_utility(self) -> float:
    """Compute total utility (CES utility from goods - disutility of labor)"""
    goods_total = 0.0
    inventory = self.inventory
    for good in self.goods:
        quantity = inventory[good]
        alpha = self.ces_params[good]
        goods_total += alpha * (quantity ** ((self.sigma - 1) / self.sigma))

    goods_utility = goods_total ** (self.sigma / (self.sigma - 1))
    labor_disutility = self.c * np.power(self.l, self.delta)

    return goods_utility - labor_disutility
```

**Constants (same file):**

- `self.c = 0.0005` (labor disutility coefficient)
- `self.delta = 3.5` (labor disutility exponent)
- `self.l` = labor hours (default **40**, then set by `choose_labor()` each step)
- `self.sigma = 5.0` (CES elasticity)

**Where it's read for state:** `ai_bazaar/env/bazaar_env.py` — `_build_consumers_state()` uses `getattr(c, "utility", 0)` when building the consumers list; for CES consumers this triggers `compute_utility()` at save time.

So the two inputs that drive the level (and sign) of utility are:

- **Goods utility** — from **current ledger inventory** via the CES formula.
- **Labor disutility** — from **current labor** `l`: `0.0005 * l^3.5`.

---

## 2. Findings

### 2.1 Labor disutility is large and convex in labor

- **Formula:** `labor_disutility = c * l^delta = 0.0005 * l^3.5`.
- With **l = 40** (default): `40^3.5 ≈ 10,119` → labor disutility ≈ **5.06**.
- With **l = 50**: `50^3.5 ≈ 883,000` → labor disutility ≈ **441.5**.

So a small increase in labor (40 → 50) multiplies the disutility by roughly 87. The term is **strongly convex** in `l`.

- **Implication:** Whenever labor is chosen high (e.g. 45–50+), the **negative** labor term is already in the hundreds. Utility will be negative unless **goods_utility** is at least that large.

---

### 2.2 Zero or low consumption makes utility strongly negative

- If **inventory is 0** for all goods, then `goods_total = 0` and `goods_utility = 0`, so  
  **utility = 0 - labor_disutility = -labor_disutility.**

So with no consumption, utility is exactly the negative of labor disutility (e.g. about **-5** at l=40 and about **-442** at l=50). This matches the pattern in your logs (e.g. consumer with 0 food and labor 50 and utility ≈ -441).

- **When does inventory stay zero?**
  - **First step:** Consumers start with empty inventory; they only get goods after the first market clearing. So the **first** time we could conceptually evaluate utility "before any consumption," it would be 0 − labor_disutility (very negative if labor is high).
  - **Later steps:** Some consumers may get **no fills** in a given step (e.g. rationing, limited supply, or orders not matched). Their inventory does not increase (or even stays 0 if they had nothing). So again utility = −labor_disutility for that step.

So **consumers with no consumption always show large negative utility** in early (or any) steps when labor is non-trivial.

---

### 2.3 Order of operations: labor first, then consumption, then state

Within each `step()` in `bazaar_env.py`:

| Order | Phase | Effect on consumer utility |
|-------|--------|----------------------------|
| 0 | Labor | Consumers choose `l` (e.g. 40–50+). Labor disutility is fixed for the rest of the step. |
| 1–4 | Supply, production, pricing | — |
| 5 | Income | Consumers receive income (no direct effect on utility formula). |
| 6 | Consumption | Consumers submit orders. |
| 7 | Market clearing | Orders are filled; ledger inventories updated. |
| … | Overhead, fees, reflection | — |
| End | **save_state()** | For each consumer, `c.utility` is read (CES from **current** inventory − labor disutility). |

So the value written to state is: **utility at end of step** = CES(current inventory) − labor_disutility(l). There is no "utility before this step's consumption"; the first time we see utility in state is **after** the first clearing. Even so, in the first step only one round of consumption has happened, and some consumers may still have zero or low inventory, so **average utility can still be very negative** in early steps.

---

### 2.4 Why "start" negative: early timesteps and rationing

- **Early timesteps:**  
  - Labor is already at default 40 or at a chosen value (often 45–50).  
  - Consumption has only just started (one or few steps).  
  - So **goods_utility** is still modest for many consumers, while **labor_disutility** is already large.  
  Result: total utility is often negative at the start of a run.

- **Rationing / no fills:**  
  If in a given step a consumer gets **no** filled orders, their inventory does not increase. Their utility for that step is again **0 − labor_disutility**, which is very negative for typical labor choices. So even in later steps, **some** consumers can have strongly negative utility, which pulls the **average** down.

- **Convex labor term:**  
  Because disutility is `l^3.5`, high labor choices (common with LLM or default 40) ensure that the negative term is large from the beginning. Utility only becomes positive once CES utility from accumulated consumption is large enough to offset that term.

---

## 3. Summary table

| Input | Source in `compute_utility` | Typical early value | Effect on "starting negative" |
|-------|-----------------------------|----------------------|-------------------------------|
| **Inventory** | Ledger (current holdings) | Often 0 or low in step 0 / early steps | Zero → goods_utility = 0 → u = −labor_disutility. Low → goods_utility modest. |
| **Labor `l`** | `choose_labor()` or default 40 | 40–50+ | Large `l` → labor_disutility in hundreds → dominates until consumption builds. |
| **CES params / sigma** | `ces_params`, `sigma` | sigma=5, alphas from persona | Affect level of goods_utility; do not remove the labor dominance early on. |

---

## 4. Why the dashboard shows very negative utility at the start

- **State:** Consumer utility in `state_t*.json` is the value of `c.utility` at **end of step** (after clearing).  
- **Dashboard:** Uses `state['consumers']` (e.g. mean of `utility`).  
- So the chart is plotting **end-of-step utility** for each timestep.

Early in the run:

1. Labor is already high (default or chosen), so labor disutility is large.
2. Consumption has only just started, so goods_utility is small or zero for many consumers.
3. Some consumers get no fills → zero goods_utility → utility = −labor_disutility.

So **consumer utilities often start very negative** because: (a) the **definition** is "CES consumption − labor disutility" with a **convex** labor term, and (b) **timing**: labor is set at the start of the step and consumption only updates after clearing, so in early steps (and for unfilled consumers) the negative labor term dominates.

---

## 5. Recommendations

1. **Scale or cap labor disutility**  
   Consider a smaller `c`, a lower `delta`, or a cap on `l` in the disutility formula so that early, high labor does not overwhelm plausible levels of goods_utility (e.g. so that utility can be positive for "moderate" consumption and labor).

2. **Report components for debugging**  
   Save (or log) **goods_utility** and **labor_disutility** separately (e.g. in state or in a debug view) so you can see whether negativity comes from zero consumption, high labor, or both.

3. **Baseline or first-step interpretation**  
   Either document that "utility at t=0" is after one round of labor and clearing (so "start" is already post-first consumption), or, if you ever add a true pre-step snapshot, document that pre-step utility is expected to be very negative (no consumption yet).

4. **Rationing and fairness**  
   If many consumers routinely get no fills in early steps, consider supply or matching so that more consumers get at least some consumption early, which would raise early average utility.
