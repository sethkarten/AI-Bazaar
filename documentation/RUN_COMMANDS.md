# Simulation Run Commands

Commands to run certain simulations. Add commands below as needed.

Run from project root. Use **`python -m ai_bazaar.main`** (works without installing the package). Alternatively, after `pip install -e .`, you can use `ai-bazaar`.

---

## Quick sanity check

---

## Visualization dashboard

Run the Streamlit dashboard to inspect simulation state (requires state files from a run that saves state, e.g. via `bazaar_env`). State files are stored under `logs/<run_name>/state_t*.json` (e.g. `logs/rep_disc_test_1/state_t0.json`). The dashboard lists runs and lets you pick one.

```bash
streamlit run ai_bazaar/viz/dashboard.py
```

Run from project root so the dashboard finds the `logs/` directory.

---

## Basic LLM marketplace

**rep_disc_test_1** — 5 Gemini firms, 20 CES consumers, discovery=2, 5 steps, CoT, RACE_TO_BOTTOM:

```bash
python -m ai_bazaar.main --firm-type LLM --num-firms 5 --num-consumers 20 --discovery-limit 2 --wandb --name rep_disc_test_1 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --firm-type LLM --num-firms 5 --num-consumers 20 --discovery-limit 2 --wandb --name rep_disc_test_2 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit 2 --wandb --name profit_calc_test_1 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit 2 --wandb --name profit_calc_test_2 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit 2 --wandb --name profit_calc_test_3 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 1 --num-consumers 5 --discovery-limit 2 --wandb --name expense_info_test_1 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 1 --num-consumers 5 --discovery-limit 2 --wandb --name utility_debug_1 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 1 --num-consumers 5 --discovery-limit 2 --wandb --name utility_debug_2 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 1 --num-consumers 5 --discovery-limit 2 --wandb --name utility_debug_3 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 1 --num-consumers 5 --discovery-limit 2 --wandb --name utility_tune_1 --max-timesteps 5 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 5 --num-consumers 20 --discovery-limit 2 --wandb --name utility_tune_2 --max-timesteps 50 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 3 --num-consumers 15 --discovery-limit 2 --wandb --name utility_tune_3 --max-timesteps 30 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 3 --num-consumers 15 --discovery-limit 2 --wandb --name utility_tune_4 --max-timesteps 30 --max-tokens 2000 --consumer-scenario EARLY_BIRD --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 1 --num-consumers 3 --wandb --name mini_test_2 --max-timesteps 2 --max-tokens 2000 --consumer-scenario RACE_TO_BOTTOM --prompt-algo cot --llm gemini-2.5-flash
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 1 --num-consumers 3 --wandb --name price_disc_mini_1 --max-timesteps 2 --max-tokens 2000 --consumer-scenario PRICE_DISCRIMINATION --prompt-algo cot --llm gemini-2.5-flash --no-diaries 
```

# Testing heterogenous supply unit cost implementation
```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 1 --num-consumers 3 --wandb --name hetero_supply_1 --max-timesteps 2 --max-tokens 2000 --consumer-scenario BOUNDED_BAZAAR --prompt-algo cot --llm gemini-2.5-flash --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-goods 4 --num-firms 1 --num-consumers 3 --name hetero_supply_2 --max-timesteps 2 --max-tokens 2000 --consumer-scenario BOUNDED_BAZAAR --prompt-algo cot --llm gemini-2.5-flash --no-diaries --seed 8
```

# Parsing stress test
```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-goods 4 --num-firms 2 --consumer-type CES --num-consumers 10 --name hetero_supply_5 --max-timesteps 10 --firm-initial-cash 5000 --consumer-scenario BOUNDED_BAZAAR --llm gemini-2.5-flash --discovery-limit 1 --max-tokens 2000 --prompt-algo cot  --no-diaries --use-parsing-agent --seed 8
```

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-goods 4 --num-firms 2 --consumer-type CES --num-consumers 10 --name hetero_supply_regparse_6 --max-timesteps 10 --firm-initial-cash 5000 --consumer-scenario BOUNDED_BAZAAR --llm gemini-2.5-flash --discovery-limit 1 --max-tokens 2000 --prompt-algo cot  --no-diaries --seed 8
```

# Heterogenous Costs/Pref Generation Test
```bash
python -m ai_bazaar.main --use-cost-pref-gen --max-supply-unit-cost 5 --use-env --firm-type LLM --num-goods 4 --num-firms 2 --consumer-type CES --num-consumers 10 --name cost_pref_gen_test_1 --max-timesteps 2 --firm-initial-cash 5000 --consumer-scenario BOUNDED_BAZAAR --llm gemini-2.5-flash --discovery-limit 1 --max-tokens 2000 --prompt-algo cot  --no-diaries --seed 8
```

# Equilibrium Initial Testing

**Run all 3 setups overnight:** from project root, run either:
- `python scripts/run_overnight_eq_tests.py`  
- `powershell -ExecutionPolicy Bypass -File scripts/run_overnight_eq_tests.ps1`  

Logs go to `logs/overnight/` (summary + per-run logs).

```bash
python -m ai_bazaar.main --use-cost-pref-gen --max-supply-unit-cost 10 --use-env --firm-type LLM --num-goods 4 --num-firms 5 --consumer-type CES --num-consumers 20 --name eq_ini_test_1 --max-timesteps 40 --firm-initial-cash 5000 --consumer-scenario BOUNDED_BAZAAR --llm gemini-2.5-flash --discovery-limit 2 --max-tokens 2000 --prompt-algo cot  --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --use-cost-pref-gen --max-supply-unit-cost 10 --use-env --firm-type LLM --num-goods 4 --num-firms 5 --consumer-type CES --num-consumers 20 --name eq_ini_test_2 --max-timesteps 40 --firm-initial-cash 5000 --consumer-scenario BOUNDED_BAZAAR --llm gemini-2.5-flash --discovery-limit 1 --max-tokens 2000 --prompt-algo cot  --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --use-cost-pref-gen --max-supply-unit-cost 10 --use-env --firm-type LLM --num-goods 4 --num-firms 5 --consumer-type CES --num-consumers 20 --name eq_ini_test_3 --max-timesteps 40 --firm-initial-cash 5000 --consumer-scenario BOUNDED_BAZAAR --llm gemini-2.5-flash --discovery-limit 5 --max-tokens 2000 --prompt-algo cot  --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --use-cost-pref-gen --max-supply-unit-cost 10 --use-env --firm-type LLM --num-goods 2 --num-firms 5 --consumer-type CES --num-consumers 20 --name eq_ini_test_4 --max-timesteps 20 --firm-initial-cash 5000 --consumer-scenario BOUNDED_BAZAAR --llm gemini-2.5-flash --discovery-limit 1 --max-tokens 2000 --prompt-algo cot  --no-diaries --seed 8 
```

# Parsing Debugging/Testing
```bash
python -m ai_bazaar.main --use-cost-pref-gen --max-supply-unit-cost 10 --use-env --firm-type LLM --num-goods 4 --num-firms 1 --consumer-type CES --num-consumers 5 --name eq_ini_test_4 --max-timesteps 20 --firm-initial-cash 1000 --consumer-scenario BOUNDED_BAZAAR --llm gemini-2.5-flash --discovery-limit 1 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8 --log-firm-prompts 
```

---
## THE CRASH
# eWTP Tests
```bash
python -m ai_bazaar.main --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 1 --consumer-type CES --num-consumers 10 --name eWTP_test_1 --max-timesteps 10 --firm-initial-cash 1000 --consumer-scenario BOUNDED_BAZAAR --llm gemini-2.5-flash --discovery-limit 5 --max-tokens 2000 --prompt-algo cot  --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 20 --name eWTP_test_2 --max-timesteps 20 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 2 --max-tokens 2000 --prompt-algo cot  --no-diaries --seed 8
```

# Crash Tests

**Run all Crash Tests from project root:** `python scripts/run_crash_tests.py`  
Logs go to `logs/crash_tests/` (summary + per-run logs).

# Large: discovery limit variation
```bash
python -m ai_bazaar.main --name crash_test_large_1 --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 8 --consumer-type CES --num-consumers 40 --max-timesteps 50 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 1 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
```bash
python -m ai_bazaar.main --name crash_test_large_2 --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 8 --consumer-type CES --num-consumers 40 --max-timesteps 50 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 2 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
```bash
python -m ai_bazaar.main --name crash_test_large_3 --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 8 --consumer-type CES --num-consumers 40 --max-timesteps 50 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 5 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
```bash
python -m ai_bazaar.main --name crash_test_large_4 --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 8 --consumer-type CES --num-consumers 40 --max-timesteps 50 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 8 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
# Hetero: heterogeneous supply costs
```bash
python -m ai_bazaar.main --name crash_test_hetero_1 --use-cost-pref-gen --max-supply-unit-cost 5 --use-env --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 40 --max-timesteps 30 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 2 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
# Cheap: low firm initial cash
```bash
python -m ai_bazaar.main --name crash_test_cheap_1 --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 40 --max-timesteps 30 --firm-initial-cash 250 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 2 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
# Single: 1 firm
```bash
python -m ai_bazaar.main --name crash_test_single_1 --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 1 --consumer-type CES --num-consumers 40 --max-timesteps 30 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 2 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

---

## Ablation-style runs

---

## Consumer scenarios

---

## Longer / larger runs

---

## Logging and reproducibility

---

## Training

---

## Reference
