# Stabilizing Firm Implementation

Brief description of the **Stabilizing Firm** in the B2C Crash environment.

## What it is

One or more firms (the first N LLM firms) can be run as **Stabilizing Firms**: each is prompted to keep price at or above unit cost and to consider market stability, and its chosen prices are **clamped** so they never go below unit cost.

## How to enable

- **Argument:** `--num-stabilizing-firms N` (default: 0)
- **Effect:** The first **N** LLM firms are marked as stabilizing (when using `--firm-type LLM`). All other firms behave as usual. Use `--num-stabilizing-firms 1` for a single stabilizing firm (e.g. `firm_0`).

## What changes for the stabilizing firm

1. **System prompt**  
   Extra instructions: never price below unit cost; consider market stability; avoid a race to the bottom; aim for sustainable profit.

2. **Unit cost in the price step**  
   All firms (not only the stabilizing one) see “Your unit cost per good: …” when setting price. The stabilizing firm is the one with the extra prompt and the clamp.

3. **Price floor (clamp)**  
   After the LLM chooses prices, the stabilizing firm’s prices are forced to be ≥ unit cost per good. Other firms’ prices are unchanged.

## Alignment-trace logging

- **Flag:** `--log-alignment-traces`
- **File:** `logs/<run_name>/alignment_traces.jsonl`
- **Content:** One JSON line per timestep: state at step start, each LLM firm’s prompt/response/action for pricing, and outcome (prices, profit, in_business, cash).
- **Scope:** Currently **all** LLM firms are logged when the flag is on; it is not limited to the stabilizing firm.

## Code locations

- **Args:** `ai_bazaar/main.py` — `--num-stabilizing-firms`, `--log-alignment-traces`
- **Which firms are stabilizing:** `ai_bazaar/env/bazaar_env.py` — `firm.stabilizing_firm = (i < num_stabilizing_firms)` when creating firms
- **Prompt and clamp:** `ai_bazaar/agents/firm.py` — `_create_system_prompt()`, `set_price()` (unit cost in message, then clamp when `self.stabilizing_firm`)
- **Trace snapshot and write:** `ai_bazaar/env/bazaar_env.py` — start and end of `step()` when `log_alignment_traces`

## Example run

```bash
python -m ai_bazaar.main --name crash_stabilizing_test --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 50 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit-consumers 3 --max-tokens 2000 --prompt-algo cot --no-diaries --num-stabilizing-firms 1 --log-alignment-traces --seed 8
```

See `RUN_COMMANDS.md` for more examples.
