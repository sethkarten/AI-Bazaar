# Simulation Run Commands

Commands to run certain simulations. Add commands below as needed.

Run from project root. Use `**python -m ai_bazaar.main**` (works without installing the package). Alternatively, after `pip install -e .`, you can use `ai-bazaar`.

---

## Visualization dashboard

Run the Streamlit dashboard to inspect simulation state (requires state files from a run that saves state, e.g. via `bazaar_env`). State files are stored under `logs/<run_name>/state_t*.json` (e.g. `logs/crash_baseline_test_1/state_t0.json`). The dashboard lists runs and lets you pick one.

```bash
streamlit run ai_bazaar/viz/dashboard.py
```

Run from project root so the dashboard finds the `logs/` directory.

---

## EXPERIMENT 1

**Common settings:** 5 LLM firms, 50 CES consumers, 365 timesteps, THE_CRASH, `--use-cost-pref-gen`, `--no-diaries`, `--prompt-algo cot`, `--max-tokens 2000`, `--llm gemini-2.5-flash`, `--overhead-costs 14`.

### `scripts/exp1.py` — Experiment 1 runner

`scripts/exp1.py` runs the full Experiment 1 matrix and supports flexible subsetting so you can re-run individual cells, switch models, or skip completed runs. Always run from the **project root**.

**Full matrix:** 37 runs total — 1 baseline (no stabilizing firm, dlc=3, seed=8) plus stabilizing-firm sweeps over dlc ∈ {1, 3, 5} × n_stab ∈ {1, 2, 4, 5} × seeds {8, 16, 64}.

**Fixed settings:** `--wtp-algo none`, competitive persona for all non-stabilizing firms, price-only consumer scoring (no `--crash-rep-scoring`), `--overhead-costs 14`. Run names are the log labels (e.g. `exp1_baseline`, `exp1_stab_2_dlc3_seed8`). Per-run logs go to `logs/exp1/`; state files and artifacts go to `logs/<run_name>/`.

#### Basic usage

```bash
# Run everything sequentially (default)
python scripts/exp1.py

# Run in parallel (IMPORTANT FOR EFFICIENCY) — keep workers low (2–4) to respect Gemini rate limits
python scripts/exp1.py --workers 3
```

#### Model / service

By default the script uses `gemini-2.5-flash` with VertexAI. Override with the same flags as a normal run:

```bash
# Different Gemini model
python scripts/exp1.py --llm gemini-2.0-flash

# Ollama (local GPU) — start Ollama first with OLLAMA_NUM_PARALLEL=4
python scripts/exp1.py --llm gemma3:4b --service ollama --port 11434

# vLLM local server
python scripts/exp1.py --llm google/gemma-3-4b-it --service vllm --port 8009
```

#### Filtering runs

All filters combine with AND logic. Use `--list` to preview before executing.

```bash
# Preview what would run without executing
python scripts/exp1.py --list

# Only dlc=3 cells
python scripts/exp1.py --dlc 3

# Only n_stab=4 and n_stab=5
python scripts/exp1.py --n-stab 4 5

# Only seed=8
python scripts/exp1.py --seeds 8

# Combine filters: dlc=1, n_stab=1 or 2, all seeds
python scripts/exp1.py --dlc 1 --n-stab 1 2

# Specific runs by exact label
python scripts/exp1.py --run exp1_baseline exp1_stab_2_dlc3_seed8

# Skip runs whose log directory already exists (resume a partial run)
python scripts/exp1.py --skip-existing

```

#### Experiment 1 figures

After runs have produced state files under `logs/<run_name>/`, generate figures from the **project root**. Figure scripts live in `paper/fig/scripts/exp1/` and write PDFs to `paper/fig/exp1/` by default.

```bash
# Regenerate all four Exp1 figures (heatmap, interaction,figures timeseries, score)
python paper/fig/scripts/exp1/exp1_run_all.py

# Optional arguments
#   --logs-dir DIR   directory containing run folders (default: logs/)
#   --good NAME      good name for price/volume metrics (default: food)
#   --fig-dir DIR    output directory for PDFs (default: paper/fig/exp1/)

# Single figure
python paper/fig/scripts/exp1/exp1_heatmap.py --logs-dir logs/
python paper/fig/scripts/exp1/exp1_score.py   --logs-dir logs/
```

Figure scripts expect run names produced by `exp1.py` (e.g. `exp1_baseline`, `exp1_stab_1_dlc1_seed8`). They read `state_t*.json` and `firm_attributes.json` from each run directory.

---
### Experiment 1 Individual Commands

### Baseline (no stabilizing firm)

Discovery limit consumers = 3, discovery limit firms = default. No stabilizing firm.

```bash
python -m ai_bazaar.main --name exp1_baseline --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

### Discovery Limit Consumer sweep

### Stabilizing Firm sweep (dlc = 1, default dlf)

Discovery limit consumers = 1; discovery limit firms = default (0). Vary number of stabilizing firms 1–5.

#### 1 Stabilizing Firm

```bash
python -m ai_bazaar.main --name exp1_stab_1_dlc1_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 1 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_1_dlc1_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 1 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_1_dlc1_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 1 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

#### 2 Stabilizing Firms

```bash
python -m ai_bazaar.main --name exp1_stab_2_dlc1_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 2 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_2_dlc1_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 2 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_2_dlc1_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 2 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

#### 4 Stabilizing Firms

```bash
python -m ai_bazaar.main --name exp1_stab_4_dlc1_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 4 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_4_dlc1_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 4 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_4_dlc1_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 4 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

### Stabilizing firm sweep (dlc = 3, dlf = default)

Discovery limit consumers = 3; discovery limit firms = default (0). Vary number of stabilizing firms 1–5.

#### 1 Stabilizing Firm

```bash
python -m ai_bazaar.main --name exp1_stab_1_dlc3_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 1 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_1_dlc3_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 1 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_1_dlc3_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 1 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

#### 2 Stabilizing Firms

```bash
python -m ai_bazaar.main --name exp1_stab_2_dlc3_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 2 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_2_dlc3_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 2 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_2_dlc3_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 2 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

#### 4 Stabilizing Firms

```bash
python -m ai_bazaar.main --name exp1_stab_4_dlc3_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 4 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_4_dlc3_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 4 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_4_dlc3_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 4 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

### Stabilizing firm sweep (dlc = 5, dlf = default)

Discovery limit consumers = 5 (= num_firms); discovery limit firms = default (0). Vary number of stabilizing firms 1–5.

#### 1 Stabilizing Firm

```bash
python -m ai_bazaar.main --name exp1_stab_1_dlc5_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 1 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_1_dlc5_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 1 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_1_dlc5_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 1 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

#### 2 Stabilizing Firms

```bash
python -m ai_bazaar.main --name exp1_stab_2_dlc5_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 2 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_2_dlc5_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 2 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_2_dlc5_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 2 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

#### 4 Stabilizing Firms

```bash
python -m ai_bazaar.main --name exp1_stab_4_dlc5_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 4 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_4_dlc5_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 4 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_4_dlc5_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 4 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

### 5 Stabilizing Firms

DLC 1, 3, 5 sweep; 3 seeds (8, 16, 64) each.

#### 5 Stabilizing Firms (dlc = 1)

```bash
python -m ai_bazaar.main --name exp1_stab_5_dlc1_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_5_dlc1_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_5_dlc1_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 1  --num-stabilizing-firms 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

#### 5 Stabilizing Firms (dlc = 3)

```bash
python -m ai_bazaar.main --name exp1_stab_5_dlc3_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_5_dlc3_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_5_dlc3_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 3  --num-stabilizing-firms 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

#### 5 Stabilizing Firms (dlc = 5)

```bash
python -m ai_bazaar.main --name exp1_stab_5_dlc5_seed8 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name exp1_stab_5_dlc5_seed16 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 16
```

```bash
python -m ai_bazaar.main --name exp1_stab_5_dlc5_seed64 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 500 --overhead-costs 14 --consumer-scenario THE_CRASH --discovery-limit-consumers 5  --num-stabilizing-firms 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 64
```

---

## Experiment 2

---

---

## Start vLLM server (for local/Hugging Face models)

Run this in a **separate terminal** from the project root before running a simulation with `--service vllm`. Use the same `--port` when you run the app (default is **8009**).

**Gemma 3 4B (Hugging Face):**

```bash
python -m vllm.entrypoints.openai.api_server --model google/gemma-3-4b-it --port 8009
```

**Llama 3.1 8B (example):**

```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8009
```

For gated Hugging Face models, set `HF_TOKEN` in that terminal (or run `huggingface-cli login` first). On Windows, vLLM often works best in WSL2.

---

## Run with Ollama (local GPU, no API quota)

Use [Ollama](https://ollama.com) to run a model on your machine and avoid Gemini/API limits. An **RTX 4060 Ti (8GB)** runs **7B models** well; **16GB** can run 13B quantized.

### Step 1: Install Ollama

- **Windows:** Go to [ollama.com](https://ollama.com) → **Download** → run the installer.
- **macOS:** Same download, or `brew install ollama`.
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

After install, Ollama usually starts in the background. You should see an Ollama icon in the system tray (Windows) or menu bar (macOS).

### Step 2: Confirm Ollama is running

In a terminal:

```bash
ollama list
```

If it runs and shows a list (or "no models"), the server is up. If you get a connection error, start Ollama from the Start Menu / Applications, or run `ollama serve` in a terminal.

### (Optional) Disable Ollama on startup and run it manually (Windows)

Ollama on Windows often adds itself to startup. To **stop it from starting at login**:

1. Press **Win + R**, type `shell:startup`, press Enter.
2. Delete **Ollama.lnk** (or any Ollama shortcut) in the folder that opens.
3. If you don't see it there, open **Task Manager** → **Startup** tab → find **Ollama** → right-click → **Disable**.

To **run Ollama manually with 4 parallel requests**, either use the **Ollama conda environment** (recommended) or set the variable inline:

**Option A: Ollama conda environment (sets `OLLAMA_NUM_PARALLEL=4` automatically)**

1. One-time setup from project root (PowerShell):
  ```powershell
   .\scripts\setup_ollama_env.ps1
  ```
   This creates conda env `Ollama` and sets `OLLAMA_NUM_PARALLEL=4` for it.
2. Start the server (from project root):
  ```powershell
   .\scripts\run_ollama_serve.ps1
  ```
   Or: `conda activate Ollama` then `ollama serve` in any terminal.

**Option B: Inline (no conda env)**

```powershell
$env:OLLAMA_NUM_PARALLEL = "4"; ollama serve
```

Leave the server terminal open while you use Ollama. In another terminal run `ollama list`, `ollama run <model>`, or your simulation. When done, close the Ollama terminal or press Ctrl+C.

### Step 3: Install the Python `ollama` package (for the app)

Activate your conda env, then:

```bash
pip install ollama
```

(The app's `OllamaModel` uses this to talk to the Ollama server.)

### Step 4: Pull a model

Pick one and download it (this can take a few minutes):

```bash
ollama pull llama3.1:8b
```

**Other options:**


| Model         | Size | Good for 8GB? | Good for 16GB? |
| ------------- | ---- | ------------- | -------------- |
| `llama3.1:8b` | 8B   | Yes           | Yes            |
| `llama3.2:3b` | 3B   | Yes (faster)  | Yes            |
| `mistral:7b`  | 7B   | Yes           | Yes            |
| `phi3:mini`   | ~4B  | Yes           | Yes            |
| `gemma2:2b`   | 2B   | Yes (fastest) | Yes            |
| `gemma2:9b`   | 9B   | Tight         | Yes            |


### Step 5: Run the simulation with Ollama

From the **project root**, with your conda env activated:

```bash
python -m ai_bazaar.main   --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit-consumers 2 --name local_test --max-timesteps 10 --prompt-algo cot --llm llama3.1:8b --service ollama --port 11434
```

**Important:** Use `--service ollama` and `--port 11434` (Ollama's default). The `--llm` value must match a model you pulled (e.g. `llama3.1:8b`, `mistral:7b`).

**Optional:** If your shell sets `GOOGLE_API_KEY` or `GEMINI_API_KEY`, unset them so the app doesn't try Gemini: e.g. `$env:GOOGLE_API_KEY = $null` (PowerShell) or don't set them in your activate script when using Ollama.

### Example: smaller/faster run

```bash
ollama pull llama3.2:3b
python -m ai_bazaar.main   --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit-consumers 2 --name local_fast --max-timesteps 5 --prompt-algo cot --llm llama3.2:3b --service ollama --port 11434
```

### Use more GPU (VRAM and utilization)

If you see low GPU usage (e.g. ~5GB/16GB VRAM, single-digit % utilization) with a small model like Gemma 3 4B:

1. **Process more requests in parallel** — Before starting Ollama, set:
  - **Windows (PowerShell):** `$env:OLLAMA_NUM_PARALLEL = "4"`
  - **Linux/macOS:** `export OLLAMA_NUM_PARALLEL=4`
   Then start Ollama (or restart it). Default is often 1; 4 lets Ollama run up to 4 inferences at once, which uses more VRAM and can raise utilization. With 16GB and a 4B model you can try `4` or `6`; each parallel slot needs extra context memory.
2. **Larger context (optional)** — More context = more VRAM per request. Example (before starting Ollama):
  - `$env:OLLAMA_NUM_CTX = "8192"` (PowerShell) or `export OLLAMA_NUM_CTX=8192`
   Don't set this so high that VRAM runs out.
3. **Use a larger model** — An 8B or 9B model uses more VRAM and more compute per request, so utilization and VRAM usage go up (e.g. `llama3.1:8b` or `gemma2:9b` on 16GB).
4. **Restart Ollama after setting env vars** — If Ollama is already running, close it and start it again in a terminal where you've set `OLLAMA_NUM_PARALLEL` (and optionally `OLLAMA_NUM_CTX`). Check usage with `ollama ps` while the sim runs.

---

## Run with a Hugging Face model (e.g. Gemma 3 4B) via vLLM

If you have access to a model on Hugging Face (e.g. **google/gemma-3-4b-it**), run it locally with **vLLM** and point the app at the vLLM server.

1. **Access the model on Hugging Face**
  - Open the model page (e.g. [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)), accept the license if it's gated, and ensure you're logged in.
2. **Set your Hugging Face token** (required for gated models)
  - Create a read token at [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens).
  - Then either:
    - **Windows (PowerShell):** `$env:HF_TOKEN = "hf_..."`
    - **Linux/macOS:** `export HF_TOKEN=hf_...`
  - Or log in once: `huggingface-cli login` (vLLM will use your saved token).
3. **Start the vLLM server** (see [Start vLLM server](#start-vllm-server-for-localhugging-face-models) above for the command). In a separate terminal, from project root:
  ```bash
   python -m vllm.entrypoints.openai.api_server --model google/gemma-3-4b-it --port 8009
  ```
   If the model is gated and you didn't use `huggingface-cli login`, set `HF_TOKEN` in that terminal before this command. On **Windows**, vLLM often works best in **WSL2**.
4. **Run the simulation** with vLLM and the same model id (or the short name `gemma3:4b`, which is mapped to `google/gemma-3-4b-it` in the app):
  ```bash
   python -m ai_bazaar.main   --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit-consumers 2 --name gemma4b_test --max-timesteps 10 --prompt-algo cot --llm google/gemma-3-4b-it --service vllm --port 8009
  ```
   Or use the short name: `--llm gemma3:4b`.

---