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
