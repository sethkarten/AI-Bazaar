# Firm state fields in `state_t*.json`

This document describes how every field in each firm’s state is computed and stored when `BazaarWorld.save_state()` writes `state_t{timestep}.json`. The code that builds the firm state lives in `ai_bazaar/env/bazaar_env.py` in `save_state()`. The dashboard (and `DataFrameBuilder.profit_per_firm_over_time()`) reads these JSON files and plots e.g. `firm["profit"]` and `ledger.money`; the **profit chart** uses the `profit` field described below.

---

## Where it’s built

The firm state is constructed here:

```python
# ai_bazaar/env/bazaar_env.py, save_state()
"firms": [
    {
        "name": ...,
        "in_business": ...,
        "cash": ...,
        "profit": ...,
        "prices": ...,
        "inventory": ...,
        "sales_by_good": ...,
        "sales_this_step": ...,
        "diary": ...,
    }
    for f in self.firms
],
```

---

## Field-by-field

### `name`

- **Source:** `f.name`
- **Meaning:** Firm identifier (e.g. `"firm_0"`). Set when the firm is created in `BazaarWorld.__init__` and never changed.

---

### `in_business`

- **Source:** `getattr(f, "in_business", True)`
- **Meaning:** Whether the firm is still operating. Defaults to `True` if the attribute is missing.
- **Updated in:** `BaseFirmAgent` (e.g. `mark_out_of_business()`). Set to `False` when the firm cannot pay overhead or taxes; once false, the firm is skipped in supply, production, pricing, and overhead in `BazaarWorld.step()`.

---

### `cash`

- **Source:** `self.ledger.agent_money.get(f.name, 0.0)`
- **Meaning:** Firm’s current money balance. Read from the shared ledger; not a copy on the firm object.
- **Updated in:** All ledger updates in the simulation (e.g. `Ledger.credit()`): initial cash, supply purchases, revenue from market clearing, overhead, platform fees. The firm’s `cash` property in `firm.py` is `return self.ledger.agent_money[self.name]`, so this JSON value is the same as `f.cash` at save time.

---

### `profit`

- **Source:** `getattr(f, "profit", 0.0)` in `save_state()`.
- **Meaning:** Value **displayed on the profit chart** and used for RL rewards. It is **not** economic profit (revenue minus all expenses). It is the result of `update_profit(quantity_sold, price, unit_cost)` during market clearing: `(price - unit_cost) * quantity_sold` for **one** sale—the **last** sale in the step’s sales loop. Defaults to `0.0` if the attribute is missing (e.g. firm had no sales yet).
- **Updated in:** `bazaar_env.step()` → market-clearing loop: for each sale in `sales_info`, the matching firm’s `update_profit(quantity_sold, price, unit_cost=getattr(self.args, "unit_cost", 2.0))` is called. Each call **overwrites** `f.profit`; there is no accumulation. So the saved value is margin from the last sale only, using a **global unit_cost (default 2.0)**, which is higher than actual supply cost (1.0). See `documentation/PROFIT_REPORT.md` for why this makes the profit chart often negative even when firm cash is growing, and for the separate economic-profit value computed in `FirmAgent.calculate_profit()` (stored in `_timestep_stats` only, not in state).

---

### `prices`

- **Source:** `self.firm_prices_last_step.get(f.name, {}).copy()`
- **Meaning:** The prices (per good) that this firm **posted for the current timestep**. Keys are good names (e.g. `"food"`), values are floats.
- **Updated in:** `bazaar_env.step()`: during the pricing phase, each firm calls `set_price(...)` and `post_quotes(prices)`; the returned `prices` are stored in `firm_prices[firm.name]`, and at the end of the step `self.firm_prices_last_step = firm_prices.copy()`. So the JSON reflects the prices that were on the market for this timestep (and will be used as “last period” context in the next step).

---

### `inventory`

- **Source:** `dict(self.ledger.agent_inventories.get(f.name, {}))`
- **Meaning:** Firm’s current holdings per good (and `"supply"`). Keys are good names plus `"supply"`; values are quantities. This is a snapshot of the ledger’s inventory for this firm at save time.
- **Updated in:** Ledger operations: initial setup in firm init, supply purchases (adds to `"supply"`), production (consumes supply, adds to each good), and market clearing (transfers goods from firm to consumers). The firm’s `f.inventory` in code is the same dict as `self.ledger.agent_inventories[f.name]`, so this JSON matches what the firm “sees” as its inventory.

---

### `sales_by_good`

- **Source:** `dict(getattr(f, "total_quantity_sold_by_good", {}))`
- **Meaning:** **Cumulative** quantity sold per good over all timesteps so far. Keys are good names; values are non-negative floats.
- **Updated in:** `bazaar_env.step()` during market-clearing: for each sale in `sales_info`, the matching firm’s `total_quantity_sold_by_good[good]` is incremented by `quantity_sold`. Initialized in `FirmAgent` / `FixedFirmAgent` as `{good: 0.0 for good in goods}`.

---

### `sales_this_step`

- **Source:** `dict(getattr(f, "total_quantity_sold_by_good_this_timestep", {}).get(self.timestep, {}))`
- **Meaning:** Quantity sold **in the current timestep only**, per good. Keys are good names; values are quantities sold this step. Empty dict if the firm has no such attribute or no entry for this timestep.
- **Updated in:** Same market-clearing loop as above: `total_quantity_sold_by_good_this_timestep[self.timestep][good]` is incremented by `quantity_sold`. The structure is `timestep -> {good: quantity}`; we only write the slice for the current `self.timestep`.

---

### `diary`

- **Source:** `getattr(f, "diary", [])[-1:]` (last entry only)
- **Meaning:** Last diary entry: a list of one element `[timestep, text]`. Only LLM firms have a diary; it’s created in `LLMAgent` and appended to in `write_diary_entry()`. Fixed firms don’t have `diary`, so we get `[]` and then `[-1:]` is still `[]`.
- **Updated in:** After reflection in `FirmAgent.reflect()`: if `not args.no_diaries`, the firm calls the LLM to write a short reflection, then `write_diary_entry(timestep, diary_entry)` appends `(timestep, entry)` to `self.diary`. The JSON stores only the **most recent** entry for brevity.

---

## Summary table

| Field             | Source (in save_state)                                      | When / where it’s updated                                      |
|------------------|-------------------------------------------------------------|-----------------------------------------------------------------|
| `name`           | `f.name`                                                    | Set at creation, never changed.                                |
| `in_business`    | `getattr(f, "in_business", True)`                           | `mark_out_of_business()` when can’t pay overhead/taxes.        |
| `cash`          | `self.ledger.agent_money.get(f.name, 0.0)`                  | All ledger credits/debits (supply, sales, overhead, fees).     |
| `profit`        | `getattr(f, "profit", 0.0)`                                 | Market-clearing loop: `update_profit()` per sale (last sale wins); used by dashboard chart and RL. Not economic profit—see PROFIT_REPORT.md. |
| `prices`        | `self.firm_prices_last_step.get(f.name, {}).copy()`         | End of pricing phase in `step()`; one set per firm per step.    |
| `inventory`     | `self.ledger.agent_inventories.get(f.name, {})`             | Supply, production, and market clearing.                        |
| `sales_by_good` | `f.total_quantity_sold_by_good`                             | Incremented in step() market-clearing for each sale.           |
| `sales_this_step`| `f.total_quantity_sold_by_good_this_timestep[timestep]`     | Same loop; per-step slice.                                     |
| `diary`         | Last element of `f.diary` (if any)                          | `write_diary_entry()` in reflect (LLM firms only).              |
