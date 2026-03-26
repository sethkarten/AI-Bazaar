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

python scripts/exp1.py --workers 3 --llm anthropic/claude-sonnet-4.6 --openrouter-provider anthropic --skip-existing --n-stab 1 3 5 --dlc 3 --list

Matching runs (6 / 46 total):
  exp1_anthropic_claude-sonnet-4.6_stab_1_dlc3_seed16  [dlc=3 n_stab=1 seed=16]
  exp1_anthropic_claude-sonnet-4.6_stab_1_dlc3_seed64  [dlc=3 n_stab=1 seed=64]
  exp1_anthropic_claude-sonnet-4.6_stab_3_dlc3_seed16  [dlc=3 n_stab=3 seed=16]
  exp1_anthropic_claude-sonnet-4.6_stab_3_dlc3_seed64  [dlc=3 n_stab=3 seed=64]
  exp1_anthropic_claude-sonnet-4.6_stab_5_dlc3_seed16  [dlc=3 n_stab=5 seed=16]
  exp1_anthropic_claude-sonnet-4.6_stab_5_dlc3_seed64  [dlc=3 n_stab=5 seed=64]
```

#### Mixed Example

```bash
# 3 workers, claud sonnet with anthropic as OpenRouter provider, skip any existing runs overlapping with this filter, list the runs (doesn't launch jobs)
python scripts/exp1.py --workers 3 --llm anthropic/claude-sonnet-4.6 --openrouter-provider anthropic --skip-existing --n-stab 1 3 5 --dlc 3 --list

# OUTPUT (seed 8 runs missing since they were already ran
Matching runs (6 / 46 total):
  exp1_anthropic_claude-sonnet-4.6_stab_1_dlc3_seed16  [dlc=3 n_stab=1 seed=16]
  exp1_anthropic_claude-sonnet-4.6_stab_1_dlc3_seed64  [dlc=3 n_stab=1 seed=64]
  exp1_anthropic_claude-sonnet-4.6_stab_3_dlc3_seed16  [dlc=3 n_stab=3 seed=16]
  exp1_anthropic_claude-sonnet-4.6_stab_3_dlc3_seed64  [dlc=3 n_stab=3 seed=64]
  exp1_anthropic_claude-sonnet-4.6_stab_5_dlc3_seed16  [dlc=3 n_stab=5 seed=16]
  exp1_anthropic_claude-sonnet-4.6_stab_5_dlc3_seed64  [dlc=3 n_stab=5 seed=64]
```

#### Experiment 1 figures

After runs have produced state files, generate figures from the **project root**. Figure scripts live in `paper/fig/scripts/exp1/` and write PDFs to `paper/fig/exp1/<model>/` by default.

Use `--src` to point at the model-specific subdirectory inside `logs/` where runs are stored. `--src` also sets the model prefix automatically (e.g. `exp1_gemini-2.5-flash` → `--model gemini-2.5-flash`). Output PDFs go to `paper/fig/exp1/<src-name>/`.

```bash
# All figures for a Gemini 2.5 Flash sweep
python paper/fig/scripts/exp1/exp1_run_all.py --src exp1_gemini-2.5-flash

# All figures for a Claude Sonnet sweep
python paper/fig/scripts/exp1/exp1_run_all.py --src exp1_anthropic_claude-sonnet-4.6

# Override output directory
python paper/fig/scripts/exp1/exp1_run_all.py --src exp1_gemini-2.5-flash --dst my_figs

# Run with more parallel workers (speeds up state-file loading)
python paper/fig/scripts/exp1/exp1_run_all.py --src exp1_gemini-2.5-flash --workers 12
```

---

### Experiment 1 Model Comparison Figure

Compares M models side-by-side across 5 metrics (bankruptcy rate, final price, volume, volatility, health score) for k ∈ {0,1,3,5} and dlc ∈ {1,3,5}. Pass `--name` to name the comparison and `--src` once per model.

Data loading priority per model:

1. Comparison cache (`paper/fig/exp1/comparisons/{name}/data/`)
2. Heatmap cache written by `exp1_run_all` (`paper/fig/exp1/{src}/data/exp1_heatmap_food.json`) — reused to avoid recomputation
3. Fallback: compute from raw state files

```bash
# Compare two models
python paper/fig/scripts/exp1/exp1_model_comparison.py --name claude_vs_gpt --src exp1_anthropic_claude-sonnet-4.6 --src exp1_openai_gpt-5.4

# Compare three models
python paper/fig/scripts/exp1/exp1_model_comparison.py --name frontier_3way --src exp1_anthropic_claude-sonnet-4.6 --src exp1_openai_gpt-5.4 --src exp1_meta-llama_llama-3.2-3b-instruct
```

Output: `paper/fig/exp1/comparisons/{name}/{name}.pdf`
Cache:  `paper/fig/exp1/comparisons/{name}/data/exp1_model_comparison_{model}_food.json`

Colormaps for rows B–D (price, volume, volatility) are normalized globally across all models so color values are directly comparable. The health score row uses global normalization too.

---

### Experiment 1 Health Score vs. Model Size

Scatter plot of composite market health score (y) vs. parameter count (x, log scale) for all dense open-weight models with `include=1` in `EAS_vs_MODEL_SIZE.md`, at the fixed setting dlc=3, k=3. Points are colored by developer; error bars show min/max across seeds 8/16/64.

Data is read from the heatmap cache written by `exp1_run_all` — no recomputation needed if runs have already been processed. Falls back to raw state files if no cache exists.

```bash
python paper/fig/scripts/exp1/exp1_health_vs_size.py --logs-dir logs/
```

Output: `paper/fig/exp1/exp1_health_vs_size.pdf`

The title reports how many of the 21 models have data (e.g. `[8/21 models with data]`), so partial sweeps render cleanly.

---

Individual figure scripts can also be called directly with the same `--logs-dir` / `--model` args that `exp1_run_all.py` passes internally:

```bash
python paper/fig/scripts/exp1/exp1_heatmap.py        --logs-dir logs/exp1_gemini-2.5-flash --model gemini-2.5-flash
python paper/fig/scripts/exp1/exp1_score.py          --logs-dir logs/exp1_gemini-2.5-flash --model gemini-2.5-flash
python paper/fig/scripts/exp1/exp1_timeseries.py     --logs-dir logs/exp1_gemini-2.5-flash --model gemini-2.5-flash
python paper/fig/scripts/exp1/exp1_survival.py       --logs-dir logs/exp1_gemini-2.5-flash --model gemini-2.5-flash
python paper/fig/scripts/exp1/exp1_phase.py          --logs-dir logs/exp1_gemini-2.5-flash --model gemini-2.5-flash
python paper/fig/scripts/exp1/exp1_collapse_timing.py --logs-dir logs/exp1_gemini-2.5-flash --model gemini-2.5-flash
python paper/fig/scripts/exp1/exp1_tokens.py         --logs-dir logs/exp1_gemini-2.5-flash --model gemini-2.5-flash
```

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

**Scenario:** LEMON_MARKET — LLM firms sell used cars, LLM buyers bid/pass, Sybil cluster misrepresents quality. Reputation EMA governs trust; Sybil identities rotate when reputation falls below `rho_min`.

**Common settings:** 5 LLM firms (3 honest + 2 Sybil), 5 LLM buyers, 20 timesteps, `--consumer-scenario LEMON_MARKET`, `--no-diaries`, `--prompt-algo cot`, `--max-tokens 512`, `--llm gemini-2.5-flash`, `--reputation-alpha 0.9`, `--reputation-initial 0.8`, `--sybil-rho-min 0.3`, `--discovery-limit-consumers 5`.

### Tests

Unit tests — verify SellerAgent, SybilIdentity, DeceptivePrincipal, and BazaarWorld construction without any LLM calls:

```bash
conda run -n AI-Bazaar python -m pytest tests/test_lemon_market.py -v
```

Smoke test — verifies the full pipeline (BuyerAgent LLM calls, honest SellerAgent listings, DeceptivePrincipal sybil cluster, market clearing, reputation updates, identity rotation) runs without error. Use a short episode and cheap token budget.

```bash
python -m ai_bazaar.main --name exp2_smoke --consumer-scenario LEMON_MARKET --firm-type LLM --num-firms 5 --num-consumers 5 --max-timesteps 5 --sybil-cluster-size 2 --reputation-alpha 0.9 --reputation-initial 0.8 --sybil-rho-min 0.3 --discovery-limit-consumers 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 42
```

Ablation smoke test — same run with `--no-buyer-rep` to confirm seller reputation is withheld from buyer observations:

```bash
python -m ai_bazaar.main --name exp2_smoke_no_rep --consumer-scenario LEMON_MARKET --firm-type LLM --num-firms 5 --num-consumers 5 --max-timesteps 5 --sybil-cluster-size 2 --reputation-alpha 0.9 --reputation-initial 0.8 --sybil-rho-min 0.3  --discovery-limit-consumers 5 --no-buyer-rep --llm gemini-2.5-flash --max-tokens 512 --prompt-algo cot --no-diaries --seed 42
```

Exp2 Setup. (Test)

```bash
python -m ai_bazaar.main --name exp2_lemon_base_test --consumer-scenario LEMON_MARKET --firm-type LLM --num-firms 10 --num-consumers 10 --max-timesteps 30 --sybil-cluster-size 5 --reputation-alpha 0.9 --reputation-initial 0.8 --sybil-rho-min 0.3  --discovery-limit-consumers 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 42
```

```bash
python -m ai_bazaar.main --name exp2_lemon_base_test2 --consumer-scenario LEMON_MARKET --firm-type LLM --num-firms 10 --num-consumers 10 --max-timesteps 30 --sybil-cluster-size 5 --reputation-alpha 0.9 --reputation-initial 0.8 --sybil-rho-min 0.3  --discovery-limit-consumers 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 42
```

```bash
python -m ai_bazaar.main --name exp2_lemon_base_test3 --allow-persistent-listings --consumer-scenario LEMON_MARKET --firm-type LLM --num-firms 10 --num-consumers 10 --max-timesteps 30 --sybil-cluster-size 5 --reputation-alpha 0.9 --reputation-initial 0.8 --sybil-rho-min 0.3  --discovery-limit-consumers 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 42
```

Heterogeneous personas — honest sellers use mixed description styles (`detailed`, `terse`, `optimistic`), sybil cluster active, listing persistence enabled. Verifies `--seller-personas`, `--seller-type LLM`, and `--allow-listing-persistence` all wire through correctly.

```bash
python -m ai_bazaar.main --name exp2_personas_smoke3 --consumer-scenario LEMON_MARKET --firm-type LLM --num-firms 10 --num-consumers 10 --max-timesteps 10 --sybil-cluster-size 4 --seller-type LLM --seller-personas "detailed:2,terse:2" --allow-listing-persistence --reputation-alpha 0.8 --reputation-initial 0.8 --sybil-rho-min 0.3 --discovery-limit-consumers 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 42
```

Vote-based reputation test — exercises the new review system: buyers make a second LLM call per transaction to upvote/downvote sellers, sybil sellers accumulate downvotes and rotate when reputation falls below `rho_min`. Uses exp2 production settings (12 sellers, 12 buyers, heterogeneous personas) at short episode length.

```bash
python -m ai_bazaar.main --name exp2_vote_rep_test --consumer-scenario LEMON_MARKET --firm-type LLM --num-sellers 12 --num-buyers 12 --max-timesteps 10 --sybil-cluster-size 6 --seller-type LLM --seller-personas "standard:3,detailed:3,terse:3,optimistic:3" --reputation-initial 0.8 --reputation-pseudo-count 10 --sybil-rho-min 0.3 --discovery-limit-consumers 5 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 42
```

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

## EAS vs Model Size

Runs all included dense models from `documentation/EAS_vs_MODEL_SIZE.md` through Experiment 1 at **n-stab=3, dlc=3** (3 seeds: 8, 16, 64 → 3 runs per model). Models route to OpenRouter automatically via the `org/model` slug.

Each command below runs a single model. Use `--workers N` for parallelism; use `--skip-existing` to resume a partial run; use `--list` to preview without launching.

```bash
# Llama 3.2 3B (3B)
python scripts/exp1.py --llm meta-llama/llama-3.2-3b-instruct --n-stab 3 --dlc 3
# Gemma 3 4B (4B)
python scripts/exp1.py --llm google/gemma-3-4b-it --n-stab 3 --dlc 3
# Mistral 7B (7.3B)
python scripts/exp1.py --llm mistralai/mistral-7b-instruct --n-stab 3 --dlc 3
# Llama 3.1 8B (8B)
python scripts/exp1.py --llm meta-llama/llama-3.1-8b-instruct --n-stab 3 --dlc 3
# Qwen3 8B (8.2B)
python scripts/exp1.py --llm qwen/qwen3-8b --n-stab 3 --dlc 3
# Gemma 3 12B (12B)
python scripts/exp1.py --llm google/gemma-3-12b-it --n-stab 3 --dlc 3
# Phi-4 (14B)
python scripts/exp1.py --llm microsoft/phi-4 --n-stab 3 --dlc 3
# DeepSeek R1 Distill Qwen 14B (14B)
python scripts/exp1.py --llm deepseek/deepseek-r1-distill-qwen-14b --n-stab 3 --dlc 3
# Mistral Small 3.1 24B (24B)
python scripts/exp1.py --llm mistralai/mistral-small-3.1-24b-instruct --n-stab 3 --dlc 3
# Gemma 3 27B (27B)
python scripts/exp1.py --llm google/gemma-3-27b-it --n-stab 3 --dlc 3
# OLMo 2 32B Instruct (32B)
python scripts/exp1.py --llm allenai/olmo-2-32b-instruct --n-stab 3 --dlc 3
# OLMo 3.1 32B Think (32B)
python scripts/exp1.py --llm allenai/olmo-3.1-32b-think --n-stab 3 --dlc 3
# DeepSeek R1 Distill Qwen 32B (32B)
python scripts/exp1.py --llm deepseek/deepseek-r1-distill-qwen-32b --n-stab 3 --dlc 3
# Llama 3.3 70B (70B)
python scripts/exp1.py --llm meta-llama/llama-3.3-70b-instruct --n-stab 3 --dlc 3
# Llama 3.1 70B (70B)
python scripts/exp1.py --llm meta-llama/llama-3.1-70b-instruct --n-stab 3 --dlc 3
# DeepSeek R1 Distill Llama 70B (70B)
python scripts/exp1.py --llm deepseek/deepseek-r1-distill-llama-70b --n-stab 3 --dlc 3
# Llama 3.1 Nemotron 70B (70B)
python scripts/exp1.py --llm nvidia/llama-3.1-nemotron-70b-instruct --n-stab 3 --dlc 3
# Qwen2.5 72B (72B)
python scripts/exp1.py --llm qwen/qwen-2.5-72b-instruct --n-stab 3 --dlc 3
# Llama 3.1 405B (405B)
python scripts/exp1.py --llm meta-llama/llama-3.1-405b-instruct --n-stab 3 --dlc 3
# Hermes 3 405B (405B)
python scripts/exp1.py --llm nousresearch/hermes-3-llama-3.1-405b --n-stab 3 --dlc 3
# Hermes 4 405B (405B)
python scripts/exp1.py --llm nousresearch/hermes-4-405b --n-stab 3 --dlc 3


```

