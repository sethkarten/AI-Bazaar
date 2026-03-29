# Simulation Run Commands

Commands to run certain simulations. Add commands below as needed.

Run from project root. Use `**python -m ai_bazaar.main**` (works without installing the package). Alternatively, after `pip install -e .`, you can use `ai-bazaar`.

---

## Visualization dashboard

Run the Streamlit dashboard to inspect simulation state (requires state files from a run that saves state, e.g. via `bazaar_env`). State files are stored under `logs/<run_name>/state_t*.json` (e.g. `logs/crash_baseline_test_1/state_t0.json`). The dashboard lists runs and lets you pick one.

```bash
streamlit run ai_bazaar/viz/dashboard.py
```

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

---

### `scripts/exp1_eas_sweep.py` — EAS vs. Model Size sweep

Runs the full Experiment 1 matrix for every dense open-weight model listed with `include=1` in `EAS_vs_MODEL_SIZE.md` (21 models, 3B–405B), all via OpenRouter. Each model gets its own `logs/exp1_{model_slug}/` subdirectory. A single `--workers` pool is shared across all models, so you can parallelise across models and cells simultaneously.

**Total runs:** 21 models × 7 runs each = 147 runs (before any filtering) — baseline (k=0, seed=8) + dlc=3 × k={3,5} × seeds={8,16,64}.

**Models (dense, include=1):**


| Display name                | Params   | OpenRouter ID                                                         |
| --------------------------- | -------- | --------------------------------------------------------------------- |
| Llama 3.2 3B                | 3B       | `meta-llama/llama-3.2-3b-instruct`                                    |
| Gemma 3 4B                  | 4B       | `google/gemma-3-4b-it`                                                |
| Mistral 7B                  | 7.3B     | `mistralai/mistral-7b-instruct-v0.1`                                  |
| Llama 3.1 8B                | 8B       | `meta-llama/llama-3.1-8b-instruct`                                    |
| Qwen3 8B                    | 8.2B     | `qwen/qwen3-8b`                                                       |
| Gemma 3 12B                 | 12B      | `google/gemma-3-12b-it`                                               |
| Phi-4                       | 14B      | `microsoft/phi-4`                                                     |
| ~~DeepSeek R1 Distill 14B~~ | ~~14B~~  | ~~`deepseek/deepseek-r1-distill-qwen-14b`~~ — removed from OpenRouter |
| Mistral Small 3.1 24B       | 24B      | `mistralai/mistral-small-3.1-24b-instruct`                            |
| Gemma 3 27B                 | 27B      | `google/gemma-3-27b-it`                                               |
| DeepSeek R1 Distill 32B     | 32B      | `deepseek/deepseek-r1-distill-qwen-32b`                               |
| Llama 3.3 70B               | 70B      | `meta-llama/llama-3.3-70b-instruct`                                   |
| Llama 3.1 70B               | 70B      | `meta-llama/llama-3.1-70b-instruct`                                   |
| DeepSeek R1 Distill 70B     | 70B      | `deepseek/deepseek-r1-distill-llama-70b`                              |
| Nemotron 70B                | 70B      | `nvidia/llama-3.1-nemotron-70b-instruct`                              |
| Qwen2.5 72B                 | 72B      | `qwen/qwen-2.5-72b-instruct`                                          |
| ~~Llama 3.1 405B~~          | ~~405B~~ | ~~`meta-llama/llama-3.1-405b-instruct`~~ — removed from OpenRouter    |
| Hermes 3 405B               | 405B     | `nousresearch/hermes-3-llama-3.1-405b`                                |
| Hermes 4 405B               | 405B     | `nousresearch/hermes-4-405b`                                          |


```bash
# Dry-run: print all matching runs grouped by model, no execution
python scripts/exp1_eas_sweep.py --list

# Sequential (default)
python scripts/exp1_eas_sweep.py

# Parallel across runs (recommended: keep low to respect rate limits)
python scripts/exp1_eas_sweep.py --workers 4

# Skip any run whose log directory already exists (safe resume)
python scripts/exp1_eas_sweep.py --workers 4 --skip-existing

# Only the dlc=3, k=3 cell across all models (the health-vs-size plot target)
python scripts/exp1_eas_sweep.py --dlc 3 --n-stab 3 --workers 4

# Subset to specific models by OR ID or display-name substring
python scripts/exp1_eas_sweep.py --models llama-3.2-3b gemma-3-4b --workers 2

# Only seed=8 for a fast first pass
python scripts/exp1_eas_sweep.py --seeds 8 --workers 8 --skip-existing

# Prefer a specific OpenRouter provider
python scripts/exp1_eas_sweep.py --openrouter-provider Together --workers 4
```

Outputs:

- Per-run state files: `logs/exp1_{model_slug}/{run_name}/`
- Per-run stdout logs: `logs/exp1_{model_slug}/{run_name}_{timestamp}.log`
- Summary log: `logs/exp1_eas_sweep_{timestamp}.log`

After the sweep finishes, run `exp1_run_all.py --src exp1_{model_slug}` per model to generate figures, or `exp1_health_vs_size.py` to generate the health-vs-size scatter (reads heatmap caches automatically).

---

#### Experiment 1 figures

After runs have produced state files, generate figures from the **project root**. Figure scripts live in `paper/fig/scripts/exp1/` and write PDFs to `paper/fig/exp1/<model>/` by default.

Use `--src` to point at the model-specific subdirectory inside `logs/` where runs are stored. `--src` also sets the model prefix automatically (e.g. `exp1_gemini-2.5-flash` → `--model gemini-2.5-flash`). Output PDFs go to `paper/fig/exp1/<src-name>/`.

```bash
# Regenerate all Exp1 figures — reads from logs/exp1_gemini-2.5-flash/, writes to paper/fig/exp1/exp1_gemini-2.5-flash/
python paper/fig/scripts/exp1/exp1_run_all.py --src exp1_gemini-2.5-flash

# --dst overrides the output subdirectory (defaults to --src name)
python paper/fig/scripts/exp1/exp1_run_all.py --src exp1_gemini-2.5-flash --dst my_run

# Optional arguments
#   --src DIR        subdirectory within logs/ to read from
#   --dst DIR        subdirectory within paper/fig/exp1/ to write to (default: --src name)
#   --logs-dir DIR   base logs directory (default: logs/)
#   --good NAME      good name for price/volume metrics (default: food)
#   --fig-dir DIR    base output directory for PDFs (default: paper/fig/exp1/)

# Single figure
python paper/fig/scripts/exp1/exp1_heatmap.py --logs-dir logs/exp1_gemini-2.5-flash
python paper/fig/scripts/exp1/exp1_score.py   --logs-dir logs/exp1_gemini-2.5-flash
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

## EXPERIMENT 2

**Scenario:** LEMON_MARKET — LLM sellers (honest + Sybil cluster) post used-car listings; LLM buyers bid/pass based on description, price, and seller reputation. Sybil identities rotate when their rolling-window reputation drops below `rho_min`. Ablation: reputation visible vs. hidden.

**Common settings:** 12 total LLM sellers (honest = 12 − K, sybil = K), 12 LLM buyers, 50 timesteps, `--discovery-limit-consumers 3`, `--no-diaries`, `--prompt-algo cot`, `--max-tokens 2000`, `--llm gemini-2.5-flash`, `--reputation-initial 0.8`, `--reputation-pseudo-count 10`, `--sybil-rho-min 0.3`. Seller personas distributed evenly across `standard/detailed/terse/optimistic` for the honest slot count.

**Buyer/Seller LLM split:** Use `--buyer-llm` and `--seller-llm` to assign different models to buyer and seller agents. Both fall back to `--llm` if unset. This allows fixing the seller model and sweeping buyer capability independently (the design used by `exp2_eas_sweep.py`).

**Per-role OpenRouter provider:** Use `--buyer-openrouter-provider` and `--seller-openrouter-provider` to route each role's model through a specific OpenRouter provider. Both fall back to `--openrouter-provider` if unset, which falls back to OpenRouter auto-selection. This is necessary when the buyer and seller models are served by different providers.

### Tests

Unit tests — verify SellerAgent, SybilIdentity, DeceptivePrincipal, and BazaarWorld construction without LLM calls:

```bash
conda run -n AI-Bazaar python -m pytest tests/test_lemon_market.py -v
```

Smoke test — full pipeline (BuyerAgent LLM calls, honest SellerAgent listings, DeceptivePrincipal sybil cluster, market clearing, rolling-window reputation, identity rotation). K=2, 10 honest sellers, 2 sybil, 12 total:

```bash
conda run -n AI-Bazaar python -m ai_bazaar.main --name exp2_smoke --consumer-scenario LEMON_MARKET --firm-type LLM --num-sellers 12 --num-buyers 12 --max-timesteps 5 --sybil-cluster-size 2 --seller-type LLM --seller-personas "standard:3,detailed:3,terse:2,optimistic:2" --reputation-initial 0.8 --reputation-pseudo-count 10 --sybil-rho-min 0.3 --discovery-limit-consumers 3 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 42
```

Vote-based reputation test — K=6 (50% saturation), 6 honest + 6 sybil = 12 total, at short episode length:

```bash
conda run -n AI-Bazaar python -m ai_bazaar.main --name exp2_vote_rep_test --consumer-scenario LEMON_MARKET --firm-type LLM --num-sellers 12 --num-buyers 12 --max-timesteps 10 --sybil-cluster-size 6 --seller-type LLM --seller-personas "standard:2,detailed:2,terse:1,optimistic:1" --reputation-initial 0.8 --reputation-pseudo-count 10 --sybil-rho-min 0.3 --discovery-limit-consumers 3 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 42
```

---

### `scripts/exp2.py` — Experiment 2 runner

`scripts/exp2.py` runs the full 3×3×2 matrix and supports flexible subsetting. Always run from the **project root**.

**Full matrix:** 24 runs — 6 baseline (K=0 × repvisible ∈ {True,False} × seeds {8,16,64}) + 18 sybil grid (K ∈ {3,6,9} × repvisible ∈ {True,False} × seeds {8,16,64}).

**Fixed settings:** `--num-sellers 12` always; honest = 12 − K; sybil saturation 25% / 50% / 75%; `rho_min=0.3`; `discovery-limit-consumers=3`; `max-timesteps=50`. Run logs go to `logs/exp2/`; state files go to `logs/<run_name>/`.

#### Basic usage

```bash
# Run everything sequentially (default)
python scripts/exp2.py

# Run in parallel — keep workers low (2–4) to respect API rate limits
python scripts/exp2.py --workers 3
```

#### Model / service

```bash
# Different Gemini model
python scripts/exp2.py --llm gemini-3-flash-preview

# OpenRouter (e.g. Claude Sonnet via Anthropic)
python scripts/exp2.py --llm anthropic/claude-sonnet-4-6 --openrouter-provider anthropic

# Ollama (local GPU) — start Ollama first with OLLAMA_NUM_PARALLEL=4
python scripts/exp2.py --llm gemma3:4b --service ollama --port 11434

# vLLM local server
python scripts/exp2.py --llm google/gemma-3-4b-it --service vllm --port 8009
```

#### Split buyer / seller LLM and provider

Use `--buyer-llm` and `--seller-llm` to assign different models to buyers vs. sellers (honest + sybil principal). Either arg falls back to `--llm` when omitted. Use `--buyer-openrouter-provider` and `--seller-openrouter-provider` to pin each role to a specific OpenRouter provider; both fall back to `--openrouter-provider`.

```bash
# Fix sellers at Gemma 12B (Together), buyers at Claude Sonnet (Anthropic)
python scripts/exp2.py --seller-llm google/gemma-3-12b-it --seller-openrouter-provider Together --buyer-llm anthropic/claude-sonnet-4-6 --buyer-openrouter-provider anthropic

# Same provider for both — use the shared flag
python scripts/exp2.py --seller-llm google/gemma-3-12b-it --buyer-llm meta-llama/llama-3.1-8b-instruct --openrouter-provider Together

# Fix sellers, leave buyers at the --llm default
python scripts/exp2.py --llm gemini-2.5-flash --seller-llm google/gemma-3-12b-it
```

#### Prompt algorithm

```bash
# Default is cot; override for ablation or faster runs
python scripts/exp2.py --prompt-algo io
python scripts/exp2.py --prompt-algo cot
```

#### Prompt logging

```bash
# Log buyer bid/review prompts to lemon_agent_prompts.jsonl
python scripts/exp2.py --log-buyer-prompts

# Log seller (honest LLM + sybil principal) prompts
python scripts/exp2.py --log-seller-prompts

# Log both
python scripts/exp2.py --log-buyer-prompts --log-seller-prompts
```

#### Filtering runs

All filters combine with AND logic. Use `--list` to preview before executing.

```bash
# Preview all matching runs without executing
python scripts/exp2.py --list

# Baseline only (K=0)
python scripts/exp2.py --k 0

# Only K=3 and K=6 sybil cells
python scripts/exp2.py --k 3 6

# Only rep-visible cells (1=visible, 0=hidden)
python scripts/exp2.py --rep-visible 1

# Only rep-hidden cells
python scripts/exp2.py --rep-visible 0

# Only seed=8
python scripts/exp2.py --seeds 8

# Combine filters: K=9, rep hidden, seeds 8 and 16
python scripts/exp2.py --k 9 --rep-visible 0 --seeds 8 16

# Specific runs by exact label
python scripts/exp2.py --run exp2_gemini-2.5-flash_k0_rep1_seed8 exp2_gemini-2.5-flash_k6_rep0_seed16

# Skip runs whose log directory already exists (resume a partial sweep)
python scripts/exp2.py --skip-existing
```

#### Experiment 2 figures

After runs have produced state files under `logs/<run_name>/`, generate figures from the **project root**. Figure scripts live in `paper/fig/scripts/exp2/` and write PDFs to `paper/fig/exp2/` by default.

```bash
# Regenerate all Exp2 figures — reads from logs/exp2_gemini-2.5-flash/, writes to paper/fig/exp2/exp2_gemini-2.5-flash/
python paper/fig/scripts/exp2/exp2_run_all.py --src exp2_gemini-2.5-flash

# --dst overrides the output subdirectory (defaults to --src name)
python paper/fig/scripts/exp2/exp2_run_all.py --src exp2_gemini-2.5-flash --dst my_run

# Optional arguments
#   --src DIR        subdirectory within logs/ to read from
#   --dst DIR        subdirectory within paper/fig/exp2/ to write to (default: --src name)
#   --logs-dir DIR   base logs directory (default: logs/)
#   --good NAME      good name for price/volume metrics (default: car)
#   --fig-dir DIR    base output directory for PDFs (default: paper/fig/exp2/)
#   --workers N      parallel load workers per script (default: 8)
#   --force          ignore cache and rebuild from scratch

# Single figure
python paper/fig/scripts/exp2/exp2_sybil_detection.py          --logs-dir logs/exp2_gemini-2.5-flash
python paper/fig/scripts/exp2/exp2_lemon_volume.py             --logs-dir logs/exp2_gemini-2.5-flash
python paper/fig/scripts/exp2/exp2_lemon_reputation_quality.py --logs-dir logs/exp2_gemini-2.5-flash
python paper/fig/scripts/exp2/exp2_lemon_consumer_welfare.py   --logs-dir logs/exp2_gemini-2.5-flash
python paper/fig/scripts/exp2/exp2_sybil_revenue_share.py      --logs-dir logs/exp2_gemini-2.5-flash
python paper/fig/scripts/exp2/exp2_market_collapse.py          --logs-dir logs/exp2_gemini-2.5-flash
```

---

### `scripts/exp2_eas_sweep.py` — EAS vs. Model Size sweep (Exp2)

Runs the full Experiment 2 matrix for every dense open-weight model from `EAS_vs_MODEL_SIZE.md` (17 models, 3B–405B), all via OpenRouter. **Buyer model is swept**; seller model is **fixed** via `--seller-llm` (required). This design isolates buyer sophistication from seller listing quality, giving a clean measure of how buyer model capability affects lemon market outcomes against a constant adversarial threat level.

Each buyer model gets its own `logs/exp2_{buyer_slug}/` subdirectory. A single `--workers` pool is shared across all models.

**Total runs:** 17 buyer models × 24 runs each = 408 runs (before filtering) — K ∈ {0,3,6,9} × rep_visible ∈ {True,False} × seeds {8,16,64}.

```bash
# Dry-run: print all matching runs grouped by buyer model, no execution
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --list

# Sequential (default)
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it

# Parallel (recommended: keep low to respect API rate limits)
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --workers 4

# Skip any run whose log directory already exists (safe resume)
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --workers 4 --skip-existing

# Only rep-visible cells and K=3 across all buyer models
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --rep-visible 1 --k 3

# Subset to specific buyer models by OR ID or display-name substring
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --models llama-3.2-3b gemma-3-4b --workers 2

# Only seed=8 for a fast first pass
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --seeds 8 --workers 8 --skip-existing

# All buyer models routed through Together
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --openrouter-provider Together --workers 4

# Buyer and seller on different providers
python scripts/exp2_eas_sweep.py \
  --seller-llm google/gemma-3-12b-it --seller-openrouter-provider Together \
  --buyer-openrouter-provider Together --workers 4
```

Outputs:

- Per-run state files: `logs/exp2_{buyer_slug}/{run_name}/`
- Per-run stdout logs: `logs/exp2_{buyer_slug}/{run_name}_{timestamp}.log`
- Summary log: `logs/exp2_eas_sweep_{timestamp}.log`

After the sweep finishes, run `exp2_run_all.py --src exp2_{buyer_slug}` per buyer model to generate figures.

---

## Extract Sybil Principal Prompts (Exp2)

After running Exp2 with prompt logging enabled, extracts Sybil principal
conversations and compiles them across seeds into one file per (K, rep) cell,
written to `logs/exp2_<model>/data/`. Each row is stamped with `_seed` and `_run`.

- `k{K}_{rep}_sybil_prompts.json` — all `agent == "sybil_principal"` rows
- `k{K}_{rep}_sybil_tier_refusals.json` — `call == "sybil_tier"` rows where the model refused the task

```bash
# Default: anthropic_claude-sonnet-4.6, all K (0 3 6 9), both reps, all seeds
python scripts/extract_sybil_prompts.py

# Specify model
python scripts/extract_sybil_prompts.py --model gemini-2.5-flash

# Subset of K values or rep conditions
python scripts/extract_sybil_prompts.py --model gemini-2.5-flash --k 3 6 9
python scripts/extract_sybil_prompts.py --model gemini-2.5-flash --rep rep1

# Restrict to specific seeds
python scripts/extract_sybil_prompts.py --model gemini-2.5-flash --seed 8 16
```

---

## Analyze Lemon Market Prompt Logs (Exp2)

Loads buyer and seller prompt logs (k0 baseline + k3 sybil run) for a given model and prints quantitative metrics plus paper-ready qualitative examples. Requires those runs to have been executed with `--log-buyer-prompts --log-seller-prompts`.

```bash
# Default: anthropic_claude-sonnet-4.6, all K values (0 3 6 9), first available seed
python scripts/analyze_lemon_prompts.py

# Specify model (looks in logs/exp2_<model>/)
python scripts/analyze_lemon_prompts.py --model gemini-2.5-flash

# Subset of K values
python scripts/analyze_lemon_prompts.py --model gemini-2.5-flash --k 0 3 6
python scripts/analyze_lemon_prompts.py --model gemini-2.5-flash --k 6 9

# Pin a specific seed when multiple are available
python scripts/analyze_lemon_prompts.py --model gemini-2.5-flash --seed 16

# Override output path (default: logs/exp2_<model>/data/prompt_analysis.txt)
python scripts/analyze_lemon_prompts.py --model gemini-2.5-flash --output logs/prompt_analysis.txt
```

Outputs: deception rate, pass rate, description style stats (word count, model-hallucination rate), best buyer sybil-detection example, best deception-success example, and a side-by-side honest vs. sybil listing.

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

Runs all included dense models from `documentation/EAS_vs_MODEL_SIZE.md` through Experiment 1. Models route to OpenRouter automatically via the `org/model` slug.

### `scripts/exp1_eas_sweep.py` — Full sweep across all dense models

Runs the **complete** Experiment 1 matrix (baseline + dlc ∈ {1,3,5} × n_stab ∈ {1..5} × seeds {8,16,64} = 46 runs per model) across all 21 dense models in a single invocation. A shared `ThreadPoolExecutor` dispatches all runs across all models so `--workers` controls total parallelism.

Logs per model go to `logs/exp1_{model_slug}/`; a sweep-level summary log is written to `logs/exp1_eas_sweep_{timestamp}.log`.

```bash
# All 21 models, sequential
python scripts/exp1_eas_sweep.py

# 4 parallel runs across all models
python scripts/exp1_eas_sweep.py --workers 4 --skip-existing

# Preview all runs without launching
python scripts/exp1_eas_sweep.py --list

# Only the dlc=3, k=3 cell (produces data for exp1_health_vs_size.py)
python scripts/exp1_eas_sweep.py --dlc 3 --n-stab 3 --workers 4 --skip-existing

# Subset of models by OR ID substring
python scripts/exp1_eas_sweep.py --models llama mistral --workers 4 --skip-existing

# With a preferred OpenRouter provider
python scripts/exp1_eas_sweep.py --workers 4 --openrouter-provider Together --skip-existing
```

All filter flags (`--dlc`, `--n-stab`, `--seeds`, `--skip-existing`, `--list`) work the same as in `exp1.py`. `--models` accepts one or more substrings matched against the OR model ID or display name.

---

### Individual model commands (dlc=3, k=3 only)

For targeted single-cell runs at **n-stab=3, dlc=3** (3 seeds per model). Use `--workers N` for parallelism; use `--skip-existing` to resume.

Each command below runs a single model. Use `--workers N` for parallelism; use `--skip-existing` to resume a partial run; use `--list` to preview without launching.

```bash
# Llama 3.2 3B (3B)
python scripts/exp1.py --llm meta-llama/llama-3.2-3b-instruct --n-stab 3 --dlc 3
# Gemma 3 4B (4B)
python scripts/exp1.py --llm google/gemma-3-4b-it --n-stab 3 --dlc 3
# Mistral 7B (7.3B)
python scripts/exp1.py --llm mistralai/mistral-7b-instruct-v0.1 --n-stab 3 --dlc 3
# Llama 3.1 8B (8B)
python scripts/exp1.py --llm meta-llama/llama-3.1-8b-instruct --n-stab 3 --dlc 3
# Qwen3 8B (8.2B)
python scripts/exp1.py --llm qwen/qwen3-8b --n-stab 3 --dlc 3
# Gemma 3 12B (12B)
python scripts/exp1.py --llm google/gemma-3-12b-it --n-stab 3 --dlc 3
# Phi-4 (14B)
python scripts/exp1.py --llm microsoft/phi-4 --n-stab 3 --dlc 3
# DeepSeek R1 Distill Qwen 14B — REMOVED from OpenRouter
# python scripts/exp1.py --llm deepseek/deepseek-r1-distill-qwen-14b --n-stab 3 --dlc 3
# Mistral Small 3.1 24B (24B)
python scripts/exp1.py --llm mistralai/mistral-small-3.1-24b-instruct --n-stab 3 --dlc 3
# Gemma 3 27B (27B)
python scripts/exp1.py --llm google/gemma-3-27b-it --n-stab 3 --dlc 3
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
# Llama 3.1 405B — REMOVED from OpenRouter
# python scripts/exp1.py --llm meta-llama/llama-3.1-405b-instruct --n-stab 3 --dlc 3
# Hermes 3 405B (405B)
python scripts/exp1.py --llm nousresearch/hermes-3-llama-3.1-405b --n-stab 3 --dlc 3
# Hermes 4 405B (405B)
python scripts/exp1.py --llm nousresearch/hermes-4-405b --n-stab 3 --dlc 3


```

---

## Appendix Experiments

Run commands for experiments intended for the paper appendix. These use the same `scripts/exp1.py` infrastructure as the main experiments.

---

### Consumer Personas

Runs the standard Experiment 1 matrix (n-stab=3, dlc=3) with `--enable-consumer-personas`, which assigns deterministic behavioral types to consumers in round-robin order: **LOYAL** (boosts recently purchased firms), **SMALL_BIZ** (boosts low-market-share firms), **REP_SEEKER** (weights reputation heavily), **VARIETY** (avoids last-purchased firm). Compare EAS results against the baseline (no personas) runs to measure demand-side heterogeneity effects.

Run with `gemini-2.5-flash` (or swap `--llm` for any model):

```bash
python scripts/exp1.py --llm gemini-2.5-flash --n-stab 3 --dlc 3 --enable-n-stab-3 --enable-consumer-personas
```

With parallelism and skip-existing:

```bash
python scripts/exp1.py --llm gemini-2.5-flash --n-stab 3 --dlc 3 --enable-n-stab-3 --enable-consumer-personas --workers 3 --skip-existing
```

