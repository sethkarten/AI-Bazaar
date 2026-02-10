# Consumer Consumption Phase

Consumers in the bazaar earn utility from the goods they hold (e.g. CES utility from inventory). In the real world, those goods are consumed over time and do not accumulate indefinitely. This feature adds a **consumption phase** so that consumer inventories (goods only) are periodically zeroed out after utility is computed, while **cash is left unchanged**.

## Behavior

- **When:** The consumption phase runs at the end of each timestep, **after**:
  - Market clearing and any reflection/reward logic
  - Computation and logging of consumer utility
  - State save (in the env path), so the saved state reflects utility and inventory *before* consumption

- **What:** For each consumer, all **goods** in the ledger are set to zero. **Cash is not modified.**

- **Effect:** In the next timestep, consumers start with empty goods inventory and must purchase again. Utility in the saved state is the value derived from the inventory they held at the end of that step, before it was “consumed.”

## Command-Line Argument

- **`--consumption-interval N`** (default: `1`)

  Run the consumption phase every **N** timesteps. With the default of 1, consumers’ goods are zeroed every timestep. With `--consumption-interval 2`, consumption runs at the end of steps 1, 3, 5, …, so goods persist for two steps between consumption phases.

## Example

```bash
# Consume every timestep (default)
python -m ai_bazaar.main --use-env --max-timesteps 10

# Consume every 3 timesteps
python -m ai_bazaar.main --use-env --consumption-interval 3 --max-timesteps 10
```

## Code

- **Consumer agents:** `CESConsumerAgent` and `FixedConsumerAgent` in `ai_bazaar/agents/consumer.py` each implement `consume_inventory()`, which zeroes all goods for that consumer via the shared ledger.
- **Simulation:** The consumption phase is invoked in both the inline loop in `ai_bazaar/main.py` and in `BazaarWorld.step()` in `ai_bazaar/env/bazaar_env.py`, using `args.consumption_interval`.
