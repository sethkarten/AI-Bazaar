# Simulation Run Commands

Commands to run certain simulations. Add commands below as needed.

Run from project root. Use **`python -m ai_bazaar.main`** (works without installing the package). Alternatively, after `pip install -e .`, you can use `ai-bazaar`.

---

## Quick sanity check

---

## Visualization dashboard

Run the Streamlit dashboard to inspect simulation state (requires `logs/state_t*.json` from a run that saves state, e.g. via `bazaar_env`):

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
