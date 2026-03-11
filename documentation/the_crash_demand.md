# Demand Randomization in the THE_CRASH Consumer Scenario

This document describes the full demand-side mechanics of the `THE_CRASH` consumer scenario in AI-Bazaar: how many consumers participate each timestep, how each consumer chooses a firm, how willingness-to-pay expectations are formed and updated, and how these components interact to produce orders.

---

## 1. Poisson Demand Participation

**Which file:** `ai_bazaar/env/bazaar_env.py` — `_consumers_participating_this_step()`
**Also:** `ai_bazaar/main.py` — CLI argument `--poisson-demand-lambda`

At the start of each consumption phase, the environment does not automatically use all consumers. Instead it draws a random count from a Poisson distribution:

```
k ~ Poisson(lambda)
k = min(max(0, k), N)
participating = random.sample(all_consumers, k)
```

where $N$ is the total number of consumers in the simulation.

**Default for THE_CRASH:** `lambda = 30`. This is hardcoded as the scenario default in two places:

- `main.py` sets `args.poisson_demand_lambda = 30` at startup if the scenario is `THE_CRASH` and no explicit value was passed.
- `_consumers_participating_this_step()` independently checks: if `poisson_demand_lambda` is `None` and the scenario is `THE_CRASH`, it uses `lam = 30`.

**Overriding the default:** Pass `--poisson-demand-lambda <value>` at the CLI. Setting it to `None` (the default for all other scenarios) causes all consumers to participate every timestep.

**Effect:** With `lambda = 30` and (say) $N = 100$ consumers, the expected number of buyers per timestep is 30, with standard deviation $\sqrt{30} \approx 5.5$. This creates aggregate demand volatility that firms experience as unpredictable order volume, contributing to the crash dynamic.

---

## 2. Discovery Friction: Which Firms Does a Consumer See?

**Which file:** `ai_bazaar/agents/consumer.py` — `make_orders()`, THE_CRASH branch
**CLI argument:** `--discovery-limit-consumers` (default: 5)

After the participating set is determined, each consumer independently executes `make_orders()`. The consumer does not see all firm quotes; it sees at most `discovery_limit_consumers` quotes per good.

**Selection logic (THE_CRASH path):** When `firm_reputations` are all identical (the typical crash setup with no reputation heterogeneity), the code falls into random discovery:

```python
if discovery_limit > 0 and len(all_good_quotes) > discovery_limit:
    if firm_reputations and len(set(firm_reputations.values())) != 1:
        # reputation-weighted ranking (not used in THE_CRASH)
        ...
    else:
        visible_good_quotes = random.sample(all_good_quotes, discovery_limit)
```

So each consumer sees a uniformly random subset of `discovery_limit_consumers` firms per good (the `discovery_limit` passed into `make_orders` is set from this CLI arg). The consumer then picks the cheapest firm within that subset:

```python
subset = random.sample(good_quotes, subset_size)   # subset_size = max(1, len(good_quotes) // 2)
lowest_in_subset = min(subset, key=lambda q: q.price)
```

Note: there are two sources of subsampling in THE_CRASH. First, the discovery limit reduces `all_good_quotes` to `visible_good_quotes`. Second, within the visible set, the consumer draws a random half-subset (`subset_size = max(1, len(visible_good_quotes) // 2)`) before picking the cheapest. The order submitted targets `lowest_in_subset.firm_id`.

---

## 3. eWTP: Expected Willingness to Pay

**Which file:** `ai_bazaar/agents/consumer.py` — `compute_willingness_to_pay()`, `update_eWTP()`

### 3.1 Initial WTP

On the first call to `compute_willingness_to_pay(timestep)`, the consumer computes its static willingness to pay for one unit of good $g$ from the CES demand system:

$$\text{WTP}_{g,t_0} = P \cdot \left(\frac{I}{P} \cdot \alpha_g\right)^{1/\sigma}$$

where:
- $P$ is the cost-of-living price index (see Section 5),
- $I$ is the consumer's income this timestep,
- $\alpha_g$ is the CES preference weight for good $g$,
- $\sigma$ is the elasticity of substitution (default: 5.0).

On this first call, the internal reference price is initialized:

```python
self._WTP_t0[g] = wtp[g]
self._r[g]      = wtp[g]
self.eWTP[g]    = wtp[g]
```

### 3.2 The r Update Rule

After each timestep's market clearing, `update_eWTP()` is called for every consumer. The internal state variable $r_g$ tracks a smoothed expected price for good $g$.

**When a sale occurred at price $p_g$:**

$$r_{g,t+1} = (1 - B) \cdot r_{g,t} + B \cdot p_g$$

The learning rate $B$ is asymmetric depending on whether the transaction price is below or above the current reference:

| Price direction | $B$ value |
|---|---|
| $p_g < r_{g,t}$ (price fell) | $B_{\text{down}} = 0.3$ |
| $p_g \geq r_{g,t}$ (price rose) | $B_{\text{up}} = 0.1$ |

This asymmetry makes consumers adapt faster to falling prices than to rising prices — a ratchet-down effect.

**When no sale occurred (consumer did not buy):**

$r_g$ is pulled back toward $\text{WTP}_{g,t_0}$ (the consumer's fundamental value):

$$r_{g,t+1} = (1 - B_{\text{down}}) \cdot r_{g,t} + B_{\text{down}} \cdot \text{WTP}_{g,t_0}$$

This prevents $r_g$ from collapsing to zero when a consumer is priced out; their reference drifts back toward their fundamental willingness to pay.

### 3.3 eWTP as the Effective Ceiling

After each $r_g$ update, the effective expected WTP is capped at the initial value:

$$\text{eWTP}_g = \min\!\left(\text{WTP}_{g,t_0},\ r_g\right)$$

This means eWTP can never exceed the consumer's initial fundamental valuation — it can only be dragged down by a history of low observed prices.

---

## 4. eWTP as a Demand Ceiling: When Consumers Do Not Buy

**Which file:** `ai_bazaar/agents/consumer.py` — `make_orders()`, THE_CRASH branch

In THE_CRASH, `use_eWTP` is always forced to `True` regardless of the CLI flag:

```python
use_eWTP = getattr(self.args, "use_eWTP", False) or (
    getattr(self.args, "consumer_scenario", None) == "THE_CRASH"
)
```

Within the THE_CRASH branch of `make_orders()`:

```python
max_wtp = self.eWTP.get(good, float('inf'))   # eWTP is the bid ceiling
...
if lowest_in_subset.price > max_wtp:
    continue                                   # no order submitted
orders.append(
    self.create_order(lowest_in_subset.firm_id, good, demand[good], max_wtp)
)
```

The order's `max_price` field is set to `eWTP` (not to the actual firm quote price). When the market clears, an order only fills if the firm's ask price is at or below `max_price`. Therefore:

- If the cheapest visible firm's price exceeds `eWTP`, the consumer submits no order at all for that good.
- If the cheapest visible firm's price is at or below `eWTP`, the consumer submits an order willing to pay up to `eWTP`. The firm collects its ask price (not `eWTP`); the surplus `eWTP - ask` stays with the consumer.

**The crash feedback loop:** As firms undercut each other, transaction prices fall. Via the $r_g$ update rule with $B_{\text{down}} = 0.3$, consumers' eWTP ratchets downward over time. This progressively tightens the effective price ceiling that firms face, which can force prices below unit cost and trigger firm bankruptcies.

---

## 5. compute_demand(): The CES Formula

**Which file:** `ai_bazaar/agents/consumer.py` — `compute_demand()`

The quantity demanded for good $g$ at timestep $t$ follows the standard CES demand system:

$$x_g = \frac{I}{P} \cdot \alpha_g \cdot \left(\frac{p_g}{P}\right)^{-\sigma}$$

where:
- $I$ = consumer income this timestep (labor income $z + $ base endowment, after scaling),
- $\alpha_g$ = CES preference weight for good $g$ (sums to 1 across goods),
- $p_g$ = current price for good $g$ (average of available quotes, or previous timestep's price if no quotes),
- $\sigma$ = elasticity of substitution (default: 5.0),
- $P$ = the CES price index (cost of living).

**Cost of living $P$:** Computed in `compute_cost_of_living()` as:

$$P = \left(\sum_g \alpha_g \cdot p_g^{1-\sigma}\right)^{1/(1-\sigma)}$$

This is the standard CES ideal price index. If no quotes are available for a good, $p_g$ falls back to the previous timestep's observed price; if neither is available, $P$ resolves to 0 and demand is set to 0.

**Income $I$:** Consumers earn labor income $z = l \cdot v \cdot w / 10$ (labor hours $\times$ skill $\times$ wage, scaled) plus a fixed base endowment. In THE_CRASH, labor hours $l$ are fixed at 40 unless `--dynamic-labor` is enabled, at which point an LLM call re-chooses $l$ each timestep. Skill $v$ is drawn uniformly from $[1.24, 159.1]$ at construction, producing consumer heterogeneity.

---

## 6. Full Chain from Timestep Start to Order Submission

The following sequence occurs on every call to `BazaarWorld.step()` for the THE_CRASH scenario:

```
1. FIRM PHASES (parallel across active firms)
   - Firms purchase supplies, produce goods, set prices (LLM or Fixed rule)
   - Each active firm posts a Quote(firm_id, good, price) to market.quotes

2. INCOME PHASE
   - Each consumer receives labor income + base endowment
   - consumer.receive_income() → ledger.credit(consumer, I)

3. PARTICIPATION SAMPLING
   - k ~ Poisson(30), clamped to [0, N]
   - participating = random.sample(all_consumers, k)

4. CONSUMPTION PHASE (parallel across participating consumers)
   For each participating consumer:

   a. compute_demand(t)
      - compute_cost_of_living(t): average quotes → P
      - x_g = (I / P) * alpha_g * (p_g / P)^(-sigma)  for each good g

   b. Discovery filtering
      - all_good_quotes = market.quotes for this good
      - If len(all_good_quotes) > discovery_limit_consumers:
            visible = random.sample(all_good_quotes, discovery_limit_consumers)
        Else:
            visible = all_good_quotes

   c. THE_CRASH order decision (per good)
      - max_wtp = consumer.eWTP[good]
      - subset = random.sample(visible, max(1, len(visible) // 2))
      - cheapest = min(subset, key=price)
      - If cheapest.price > max_wtp: skip (no order)
      - Else: submit Order(firm_id=cheapest.firm_id, quantity=x_g, max_price=max_wtp)

5. MARKET CLEARING
   - market.clear(ledger): fills orders where ask <= max_price
   - Returns filled_orders, sales_info

6. eWTP UPDATE (all consumers, sequential)
   - For each filled order: consumer.update_eWTP(sale)
     r_g ← (1 - B) * r_g + B * p_g    [B = 0.3 if p < r, else 0.1]
     eWTP_g ← min(WTP_t0_g, r_g)
   - For each consumer with no sale: consumer.update_eWTP()
     r_g ← (1 - 0.3) * r_g + 0.3 * WTP_t0_g   (drift back toward fundamental)
     eWTP_g ← min(WTP_t0_g, r_g)

7. UTILITY / CONSUMPTION
   - Consumer utilities computed, then inventories zeroed (goods consumed)
```

---

## 7. Key Parameters Summary

| Parameter | CLI flag | Default (THE_CRASH) | Effect |
|---|---|---|---|
| Poisson lambda | `--poisson-demand-lambda` | 30 | Expected number of buyers per timestep |
| Consumer discovery limit | `--discovery-limit-consumers` | 5 | Max firms visible to one consumer per good |
| Firm discovery limit | `--discovery-limit-firms` | 0 | Max competitor firms’ prices visible to each firm when setting prices (0 = no limit) |
| eWTP enabled | `--use-eWTP` | Always true for THE_CRASH | eWTP is the order `max_price` |
| $B_{\text{down}}$ | hardcoded | 0.3 | eWTP adapts quickly to falling prices |
| $B_{\text{up}}$ | hardcoded | 0.1 | eWTP adapts slowly to rising prices |
| $\sigma$ | hardcoded | 5.0 | CES elasticity of substitution |
| Skill $v$ | `--skill` (per-consumer) | $U[1.24, 159.1]$ | Consumer income heterogeneity |
