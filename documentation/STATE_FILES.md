# State files

State files are JSON snapshots of the simulation written after each timestep. The dashboard and analysis scripts read these files.

---

## Where and when

- **Code:** `BazaarWorld.save_state()` in `ai_bazaar/env/bazaar_env.py`
- **When:** Called at the end of every `step()`, after market clearing, overhead, fees, and reflection
- **Location:** `logs/` by default (overridable via `args.log_dir`)
- **Naming:** `state_t0.json`, `state_t1.json`, … (one file per timestep)

---

## Structure

Each file is a single JSON object with:

| Key | Contents |
|-----|----------|
| `timestep` | Current step index (int) |
| `ledger` | `money` (agent → balance), `inventories` (agent → {good: qty}) |
| `firms` | List of firm objects, sorted by name |
| `consumers` | List of consumer objects, sorted by name |
| `total_fees` | Platform fees collected this run (float) |

---

## How the lists are built

- **Firms:** For each firm in `self.firms`, one object is built (name, in_business, cash, profit, prices, inventory, sales, diary). Any key in the ledger that starts with `firm_` but is not in that list gets a minimal stub entry so no agent is missing. Results are sorted by name.
- **Consumers:** Same idea: one entry per consumer in `self.consumers`, then any `consumer_*` in the ledger gets a stub if missing. Sorted by name.

So every agent that has ledger state appears in the file; the dashboard can show all firms and consumers.

---

## Field details

- **Firm fields:** See [FIRM_STATE_FIELDS.md](FIRM_STATE_FIELDS.md).
- **Ledger:** `money` and `inventories` are copies of the env’s ledger at save time.
