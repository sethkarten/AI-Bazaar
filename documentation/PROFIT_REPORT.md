# Report: Why Firm Profits Are Almost Always Negative

This report analyzes why `firm.profit` (used in state and dashboards) is often negative, with focus on `update_profit(quantity_sold, price, unit_cost)` and the sources of its inputs.

---

## 0. Two profit concepts (why the chart disagrees with cash)

There are **two** profit-related values in the codebase; only one is saved to state and shown on the profit chart.

| Concept | Where defined | Formula | Saved to state? | Used by profit chart? |
|--------|----------------|---------|------------------|------------------------|
| **`f.profit`** (attribute) | `BaseFirmAgent.update_profit()` in `firm.py` | `(price - unit_cost) * quantity_sold` (per sale, **overwritten** each time) | **Yes** — `save_state()` writes `getattr(f, "profit", 0.0)` | **Yes** — dashboard reads `firm["profit"]` from state |
| **Step profit** (economic) | `FirmAgent.calculate_profit()` in `firm.py` | `total_revenue - total_expenses` (revenue from sales, minus supply, overhead, taxes) | **No** — stored only in `_timestep_stats[timestep]["profit"]` | **No** |

- **Cash** in state comes from the **ledger** (all real credits/debits: revenue from market clearing, supply at 1.0, overhead, platform fees, taxes). So cash growth reflects true net inflow.
- **Profit** in state comes from **`f.profit`**, which is set only by `update_profit()` during market clearing, using a **fixed unit_cost (default 2.0)** and **last sale only**. So the profit chart shows a **flawed metric**, not economic profit. That is why the profit chart can be very negative even when firms’ cash is growing.

---

## 1. Where profit is set (the value that appears on the chart)

**Location:** `ai_bazaar/agents/firm.py` (BaseFirmAgent)

```python
def update_profit(self, quantity_sold: float, price: float, unit_cost: float) -> float:
    """Update profit for the current timestep (used for RL rewards)"""
    self.profit = (price - unit_cost) * quantity_sold
    return self.profit
```

**Call site:** `ai_bazaar/env/bazaar_env.py` (inside the market-clearing loop, after `market.clear()`)

```python
for sale in sales_info:
    firm_name = sale["firm_id"]
    good = sale["good"]
    quantity_sold = sale["quantity_sold"]
    ...
    price = firm_prices[firm_name][good]
    ...
    firm.update_profit(
        quantity_sold,
        price,
        unit_cost=getattr(self.args, "unit_cost", 2.0),
    )
```

So the three inputs are:

- **quantity_sold** – from `sale["quantity_sold"]` (market_core)
- **price** – from `firm_prices[firm_name][good]` (env)
- **unit_cost** – from `args.unit_cost` (default **2.0**)

---

## 2. Findings

### 2.1 Unit cost is a global constant and does not match actual costs

- **In the code:** `unit_cost` is always `getattr(self.args, "unit_cost", 2.0)`. It is the same for every firm and every good.
- **In the simulation:**  
  - Supply is bought at **1.0** per unit (`supply_unit_price = 1.0` in `bazaar_env.step()`).  
  - Production turns 1 unit of supply into 1 unit of good (FixedFirmAgent splits supply evenly across goods; FirmAgent uses LLM percentages).  
  - So the **marginal cost per unit of good** from supply is **1.0**, not 2.0.

If we use `unit_cost = 2.0` in the formula:

- For any sale at price &lt; 2.0 we get **negative** profit (e.g. price 1.5 → (1.5 − 2.0) × q = −0.5q).
- Even at price 1.5, the firm’s **true** marginal cost (supply only) is 1.0, so a more accurate margin would be (1.5 − 1.0) × q = +0.5q.

So **using a fixed unit_cost of 2.0 that is higher than the actual supply cost (1.0) systematically makes reported profit more negative (or less positive)**. That is a strong candidate for “profits almost always negative.”

**Conclusion:** `update_profit` is using an **outdated/mismatched** value for `unit_cost`: a global default (2.0) that does not reflect the actual supply cost (1.0) or any per-firm or per-good cost.

---

### 2.2 Price and quantity_sold

- **quantity_sold:** Comes from `market.clear()` → `sales_info` → `sale["quantity_sold"]`, which is the quantity actually filled in `_fill_order`. So **quantity_sold is correct** and matches the cleared trade.
- **price:** The sale is executed in the market at `best_quote.price`. The env does **not** pass that price back in `sales_info`; it looks up `price = firm_prices[firm_name][good]`. Those prices are set in the same step when firms call `post_quotes(prices)` and `firm_prices[firm.name] = prices`, so in normal operation the quote price and `firm_prices[firm_name][good]` are the same. So **price is consistent** with the clearing price in the current design, but it is **inferred from env state** rather than from the actual filled quote. If there were ever multiple quotes per (firm, good) or any later overwrite of `firm_prices`, the value passed to `update_profit` could in theory diverge from the real transaction price.

So:

- **quantity_sold:** correct and not outdated.
- **price:** correct in the current flow but not taken from the sale record; a future improvement would be to include the clearing price in `sales_info` and use that in `update_profit`.

---

### 2.3 Profit is overwritten per sale, not accumulated

From the env loop:

```python
for sale in sales_info:
    ...
    firm.update_profit(quantity_sold, price, unit_cost=...)
```

Each call does `self.profit = (price - unit_cost) * quantity_sold`, so **each sale overwrites `firm.profit`**. If a firm has multiple sales in one step (e.g. multiple goods, or multiple orders for the same good), only the **last** sale in the loop contributes to the stored `profit`. Earlier sales are discarded.

Effects:

- Reported profit is **not** step-level total profit across all goods and orders.
- If the last sale happens to be the one with negative margin (e.g. one good priced below cost, another above), the whole step will show negative even when total economic profit for the step could be positive.

So the **definition** of `profit` (last sale only) is a second reason the number can look wrong or persistently negative.

---

## 2.4 Chronological order: when profit and cash are set in each step

All of the following occur in `bazaar_env.step()` (and then `save_state()`). Cash is read from the ledger at save time; `f.profit` is whatever was last written by `update_profit()`.

| Order | Phase | What happens to cash | What happens to `f.profit` |
|-------|--------|------------------------|----------------------------|
| 0 | (before step) | `start_ledger = ledger.copy()` | — |
| 1 | Labor | — | — |
| 2 | Supply | Firms pay for supply (1.0/unit); ledger debited | — |
| 3 | Production | — | — |
| 4 | Pricing | — | — |
| 5 | Income | Consumers credited | — |
| 6 | Consumption | — | — |
| 7 | (snapshot) | `pre_clearing_ledger = ledger.copy()` | — |
| 8 | **Market clearing** | Ledger: revenue credited to firms per sale | **For each sale:** `firm.update_profit(...)` → **overwrites** `f.profit` (last sale wins) |
| 9 | Cleanup | — | — |
| 10 | Overhead | Firms pay overhead; ledger debited | — |
| 11 | Platform fees | Firms pay 5% of cash; ledger debited | — |
| 12 | Reflection | — | LLM firms only: `calculate_profit()` writes to `_timestep_stats[timestep]["profit"]` (not `f.profit`) |
| 13 | Rewards | — | RL uses `getattr(firm, "profit", 0.0)` for this step |
| 14 | — | `firm_prices_last_step` updated | — |
| 15 | **save_state()** | State `"cash"` = `ledger.agent_money[f.name]` | State `"profit"` = `getattr(f, "profit", 0.0)` |

So the value written to state and used by the dashboard is **only** the one set in phase 8 (market clearing). It does **not** include overhead, fees, or taxes, and it uses the wrong unit cost and last-sale-only aggregation.

---

## 3. Summary table

| Input          | Source in `update_profit`              | Matches reality? | Note |
|----------------|----------------------------------------|------------------|------|
| quantity_sold  | `sale["quantity_sold"]` from market     | Yes              | Correct. |
| price          | `firm_prices[firm_name][good]`          | Yes in current flow | Not taken from filled quote; could diverge if logic changes. |
| unit_cost      | `args.unit_cost` (default 2.0)         | No               | Actual supply cost is 1.0; constant is too high and not per-firm/per-good. |
| Aggregation    | Overwritten per sale                   | No               | Should accumulate over all sales in the step. |

---

## 3.1 Why the profit chart is very negative despite cash growing

- **Cash** in the saved state is the firm’s actual balance from the ledger after all step activity (revenue, supply at 1.0, overhead, fees, taxes). So if firms are net positive, cash grows.
- **Profit** in the saved state is **not** economic profit. It is `f.profit` from `update_profit()`: a single-sale margin using **unit_cost = 2.0**. With real supply cost 1.0, any price in (1.0, 2.0) is profitable in reality but shows as **negative** on the chart. Plus only the **last** sale in the step is counted, so step-level profit is wrong.
- **Dataflow for the chart:** `state_t*.json` → `DataFrameBuilder.profit_per_firm_over_time()` reads `f.get("profit", 0)` from each `state["firms"]` → dashboard plots that series. So the chart is plotting the flawed `f.profit` metric, not revenue minus expenses.

---

## 4. Recommendations

1. **Unit cost**
   - Prefer a unit cost that reflects **actual** cost per unit sold in that step, e.g.:
     - Supply cost incurred this step / quantity produced (or quantity sold), or
     - A per-firm or per-good cost derived from `purchase_supplies` and production, not a global `args.unit_cost`.
   - If keeping a single default, align it with supply: e.g. 1.0 to match `supply_unit_price`, or document that it intentionally includes overhead and set it accordingly.

2. **Price**
   - Have `market.clear()` (or the env) include in each sale the **price at which the order was filled** (e.g. `best_quote.price`), and pass that into `update_profit` instead of looking up `firm_prices`. That keeps profit tied to the actual transaction price.

3. **Accumulation**
   - Change `update_profit` so that it **adds** to profit for the step instead of overwriting, e.g. `self.profit += (price - unit_cost) * quantity_sold`, and reset `self.profit` at the start of each step (e.g. at the beginning of `bazaar_env.step()` or before the clearing loop). Then the stored profit is step-level total profit across all sales.

Implementing (1) and (3) should remove the main structural reasons profits look almost always negative; (2) makes the metric robust to future changes in how prices are stored or looked up.

4. **Dashboard / state**
   - Either fix `f.profit` as above so it reflects step-level economic profit, or have `save_state()` (and thus the dashboard) use an economic-profit value. For LLM firms, `_timestep_stats[timestep]["profit"]` from `calculate_profit()` is already computed in reflection; that value could be written to state (e.g. a separate field or by updating `f.profit` after reflection) so the profit chart aligns with cash growth.

---

## 5. Implementation status (fixes applied)

The following changes were made to implement recommendations 1–3:

- **Unit cost (Rec 1):** Default `unit_cost` in `bazaar_env.step()` is now **1.0** to match `supply_unit_price` (still overridable via `args.unit_cost`).
- **Price (Rec 2):** `market_core.Market._fill_order()` now returns `(filled, quantity_sold, best_quote.price)`; `clear()` adds `"price"` to each `sales_info` entry. The env uses `sale.get("price", firm_prices[...])` when calling `update_profit`, so profit uses the actual transaction price.
- **Accumulation (Rec 3):** At the start of market clearing, each firm’s `profit` is reset to **0**. `BaseFirmAgent.update_profit()` now **accumulates** (`self.profit += margin`) instead of overwriting, and `BaseFirmAgent.__init__` initializes `self.profit = 0.0`. The value saved to state and shown on the profit chart is therefore step-level total margin across all sales (before overhead/fees/taxes).
