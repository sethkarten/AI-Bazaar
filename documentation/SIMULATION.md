# Marketplace Simulation Dynamics

This document describes how the AI-Bazaar marketplace simulation runs: the order of operations each timestep, how the ledger and market work, and how money and goods flow.

## Overview

The simulation is a discrete-time loop. Each **timestep** runs the same sequence of phases. Firms produce and sell goods; consumers receive income and place orders; the market clears; then overhead and fees are applied.

**Canonical implementation:** The authoritative timestep logic lives in **`BazaarWorld.step()`** in `ai_bazaar/env/bazaar_env.py`. Training (`train_reinforce.py`), evaluation (`eval_marketplace.py`), and the bazaar_env test use this implementation.

**Alternative entry point:** `run_marketplace_simulation()` in `ai_bazaar/main.py` runs a similar loop for the CLI; it does not use `BazaarWorld` and omits the Labor phase and information asymmetry. See [BAZAAR_ENV_SYNC_REPORT.md](BAZAAR_ENV_SYNC_REPORT.md) for how the two paths are kept in sync.

---

## Timestep Order (per step)

The following order is fixed. It matters for consistency (e.g. consumers must have income before ordering; firms must have inventory before quotes). The phase numbering and description below match **`BazaarWorld.step()`** in `bazaar_env.py`.

| Phase | What happens |
|-------|----------------|
| **0. Labor** *(bazaar_env only)* | Each CES consumer (if they have `choose_labor`) chooses labor supply; income is determined by labor × wage later. Not present in `main.py`'s loop. |
| **1. Supply** | Each firm (still in business) purchases raw supplies at a unit price. Money is debited from the firm; supply is added to the firm’s ledger inventory. There is no external “supplier” agent—supply is created when purchased. |
| **2. Production** | Each firm converts its supply into finished goods (e.g. food, clothing, electronics). Supply is debited; goods are added to the firm’s inventory. |
| **3. Pricing** | Each firm sets prices and posts **quotes** to the market (one quote per good they have in stock). Firms may use last step’s prices as context. |
| **4. Income** | Each consumer receives income (e.g. labor income + base endowment). This is **injected** into the economy (credited to the consumer on the ledger). |
| **5. Consumption** | Each consumer decides orders (which firm, good, quantity, max price) and **submits** them to the market. Orders are queued; quotes are already posted. Discovery limit and firm reputations (only firms in business) shape which quotes are visible. |
| **6. Market clearing** | The market matches orders to quotes and executes trades: money is transferred from consumer to firm, and goods from firm to consumer, on the **ledger**. Partial fills occur if the consumer cannot afford the full quantity or the firm has limited stock. Sales info updates firm tracking (`total_quantity_sold_by_good`, `total_quantity_sold_by_good_this_timestep`), profit, and reputations (`update_reputation(sold, requested)`). |
| **7. Cleanup & Overhead** | Sales info is used to update each firm’s tracking (e.g. `total_quantity_sold_by_good`). The market’s quotes and order queue are cleared for the next timestep. |
| **8. Overhead** | Each firm pays a fixed overhead cost (e.g. 50 per step). Money is debited from the firm (no other agent receives it). Firms can go “out of business” if they cannot pay. |
| **9. Reflection** | Firms and consumers (if they have a `reflect` method) update internal state or logs using ledger snapshots (e.g. start of step, pre-clearing, post-clearing). In bazaar_env, trajectory rewards are then assigned for training. |

After this, the next timestep starts from phase 0 (or 1 in main). The simulation stops when `max_timesteps` is reached or all firms are out of business. In bazaar_env, `save_state()` writes `state_t{timestep}.json` after each step (see [FIRM_STATE_FIELDS.md](FIRM_STATE_FIELDS.md) for firm state fields).

---

## Ledger and Market

- **Ledger** (`ai_bazaar/market_core/market_core.py`): Single source of truth for cash and inventories.
  - `agent_money[agent_id]`: cash balance.
  - `agent_inventories[agent_id]`: dict of good → quantity.
  - Agents (firms, consumers) **reference** the ledger: e.g. `self.inventory = self.ledger.agent_inventories[self.name]`. There is no separate local copy of inventory.
- **Market**: Holds the current **order** queue and **quote** list. Clearing matches each order to a quote (same firm and good, price ≤ consumer’s max price, quantity available). The **first** matching quote is used (not necessarily the cheapest). Trades are executed via `ledger.transfer_money` and `ledger.transfer_good`. After clearing, main clears quotes and orders so each timestep starts with an empty book.

### Discovery limit (polling cap)

Consumers do not see every firm’s quotes when forming orders. The **discovery limit** (`--discovery-limit`, default 5) caps how many **firms** (per good) a consumer can “poll” for prices before submitting orders:

- For each good, the consumer only considers up to `discovery_limit` quotes (each quote is one firm). If there are more quotes than the limit, the subset is chosen at random, or by a simple score (e.g. reputation × 1/price) when reputation data is available (CES scenario).
- Set `--discovery-limit 0` for no cap (consumer sees all firms’ quotes for each good).
- This creates search friction and can reduce price competition when there are many firms.
- **Firm reputations:** When choosing which firms to “see” (under the cap), CES consumers rank by score = (1/price) × reputation. Reputation is each firm’s **fulfillment rate** (quantity delivered / quantity requested) over the last 10 transactions. After each market clear, firms are updated via `update_reputation(quantity_sold, requested_quantity)`; then the current `firm.reputation` map is passed into `make_orders` as `firm_reputations`, so the next timestep’s orders use the latest reputations.

---

## Money Flows

The economy is **not** closed: money is both injected and destroyed.

- **Injected:** Consumer income in the Income phase (`receive_income`). Money is created and credited to the consumer.
- **Destroyed:**
  - **Supply purchase:** Firms pay for supplies; that money is debited and no agent receives it (supply is just added to the firm).
  - **Overhead:** Fixed cost per firm per step; debited from the firm.
  - **Platform fee:** Percentage of firm cash; debited from the firm.

So total money in the system changes over time. For debugging or validation, you can track total ledger money and compare it to “initial cash + cumulative consumer income − cumulative supply cost − overhead − fees.”

---

## Agent Types

- **Firms:** `FirmAgent` (LLM) or `FixedFirmAgent`. Both use the same ledger and market; only decision logic (prices, supply quantity, production split) differs.
- **Consumers:** `CESConsumerAgent` (LLM, CES utility) or `FixedConsumerAgent`. Again, same ledger/market; different logic for orders and (for CES) income/labor.

---

## Key Files

| File | Role |
|------|------|
| `ai_bazaar/env/bazaar_env.py` | **Canonical:** `BazaarWorld` and `step()`; used by train, eval, and bazaar_env test. |
| `ai_bazaar/main.py` | CLI entry point; `run_marketplace_simulation()` runs its own loop (no BazaarWorld). |
| `ai_bazaar/market_core/market_core.py` | `Ledger`, `Market`, `Order`, `Quote`; matching and transfers. |
| `ai_bazaar/agents/firm.py` | Firm agents (base, LLM, fixed); supply, production, pricing, overhead. |
| `ai_bazaar/agents/consumer.py` | Consumer agents; income, orders, utility. |

---

## Verification and Cleanup

When verifying or cleaning up the simulation:

1. **Order of operations:** Ensure no code assumes a different phase order (e.g. income after orders).
2. **Ledger consistency:** All trades go through the ledger; agents should not hold separate cash/inventory that can drift.
3. **Market clearing:** Be aware that the first matching quote is used, not “best price”; document or change if you want price-based choice.
4. **Tests:** Use fixed agents and short runs to assert on ledger state and total money flow after N steps.
