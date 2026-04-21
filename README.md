# AI-Bazaar

[Python 3.10+](https://www.python.org/downloads/)
[License: MIT](https://opensource.org/licenses/MIT)
[Tests](https://pytest.org/)
[arXiv](https://arxiv.org/abs/2507.15815)

> *Left: base LLM agents fail — firms crash prices into bankruptcy (B2C), and a Sybil principal floods the market with deceptive listings (C2C). Right: aligned agents restore equilibrium — Stabilizing Firms hold a price floor; Skeptical Guardians detect and reject the Sybil cluster. See the [simulation design diagram](fig/sim_design.pdf) for a full architectural overview.*

## Research Overview

As AI agents increasingly operate autonomously in digital marketplaces, their collective behavior introduces systemic risks that standard alignment — targeting helpfulness, harmlessness, and honesty — does not address. An agent that is individually rational can drive a market into collapse through locally optimal but globally destructive decisions. We define **Economic Alignment** as the property of contributing to stable, fair markets rather than chaotic or exploitative ones, and introduce **Agent Bazaar** to benchmark it.

We study two canonical failure modes. **THE_CRASH**: in B2C markets, LLM firms engage in a destructive undercutting race until prices fall below unit cost, triggering mass bankruptcy — an LLM-native analog of the 2010 Flash Crash. **THE_LEMON_MARKET**: in C2C markets, a Sybil principal operates *K* coordinated seller identities, rotating them when reputation degrades to perpetuate fraud at scale, an amplified version of Akerlof's market for lemons.

For each failure mode, AI-Bazaar tests intervention mechanisms: **Stabilizing Firms** enforce a price floor against the undercutting spiral; **Skeptical Guardians** detect and reject deceptive listings. We evaluate frontier and open-weight models (3B–405B) across both scenarios and introduce the **Economic Alignment Score (EAS)**, a unified scalar aggregating stability, integrity, welfare, and profitability into a single cross-model metric.

Built on [LLM Economist](https://github.com/sethkarten/LLM-Economist/), extending it with agent-agent goods trading, firm/buyer/seller agents, and a Streamlit visualization dashboard.

---

## Installation

### 1. Create a conda environment

```bash
conda create -n ai-bazaar python=3.12 -y
conda activate ai-bazaar
```

### 2. Install the package

```bash
# From PyPI
pip install ai-bazaar

# Or development install from source
git clone https://github.com/sethkarten/AI-Bazaar.git
cd AI-Bazaar
pip install -e .
```

### 3. Set API keys

Set the keys for whichever LLM providers you plan to use:

```bash
# Google Gemini (Google AI Studio)
export GOOGLE_API_KEY="your_google_key"
# or for Vertex AI: set up Application Default Credentials
# gcloud auth application-default login

# OpenAI
export OPENAI_API_KEY="your_openai_key"

# Anthropic
export ANTHROPIC_API_KEY="your_anthropic_key"

# OpenRouter (access many models through one API)
export OPENROUTER_API_KEY="your_openrouter_key"
```

---

## Local LLM Setup

### Ollama

[Ollama](https://ollama.com) runs models locally on your GPU without any API quota.

**Install:**

- Windows/macOS: download from [ollama.com](https://ollama.com)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh`

**Pull a model and start the server:**

```bash
ollama pull llama3.1:8b
ollama serve           # leave this terminal open
```

To allow parallel requests (recommended for simulations; ensure sufficient GPU memory to hold 4 instances of the model before setting):

```bash
# Linux/macOS
export OLLAMA_NUM_PARALLEL=4 && ollama serve

# Windows (PowerShell)
$env:OLLAMA_NUM_PARALLEL = "4"; ollama serve
```

### vLLM

[vLLM](https://github.com/vllm-project/vllm) serves Hugging Face models via an OpenAI-compatible API. On Windows, vLLM works best in WSL2.

```bash
pip install vllm

# Start a server (run in a separate terminal)
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-4b-it \
    --port 8009
```

For gated models, set your Hugging Face token first:

```bash
export HF_TOKEN="hf_..."
# or: huggingface-cli login
```

Use `--llm google/gemma-3-4b-it --service vllm --port 8009` in your simulation command to point at this server.

---

## Quick Start

```bash
# THE_CRASH: 5 LLM firms, 50 consumers, 20 timesteps (Gemini)
python -m ai_bazaar.main \
  --consumer-scenario THE_CRASH \
  --firm-type LLM --num-firms 5 --num-consumers 50 \
  --use-cost-pref-gen --no-diaries --prompt-algo cot \
  --llm gemini-2.5-flash --max-timesteps 20 --name crash_test

# LEMON_MARKET: 12 sellers (3 Sybil), 12 LLM buyers
python -m ai_bazaar.main \
  --consumer-scenario LEMON_MARKET \
  --num-sellers 12 --num-buyers 12 \
  --sybil-cluster-size 3 --reputation-initial 0.8 \
  --no-diaries --prompt-algo cot \
  --llm gemini-2.5-flash --max-timesteps 20 --name lemon_test

# Local model via Ollama
python -m ai_bazaar.main \
  --consumer-scenario THE_CRASH \
  --firm-type LLM --num-firms 3 --num-consumers 20 \
  --llm llama3.1:8b --service ollama --port 11434 \
  --max-timesteps 10 --name local_test
```

### Visualization Dashboard

After running a simulation, inspect results in the Streamlit dashboard:

```bash
streamlit run ai_bazaar/viz/dashboard.py
```

State files are stored at `logs/<run_name>/states.json`. The dashboard lists all available runs and lets you explore per-timestep market state.

---

## Project Structure

```
AI-Bazaar/
├── ai_bazaar/              # Main package
│   ├── agents/             # Agent implementations
│   │   ├── firm.py         # LLM firm / Stabilizing firm
│   │   ├── buyer.py        # LLM buyer / Skeptical Guardian
│   │   ├── seller.py       # LLM seller / Sybil principal
│   │   ├── consumer.py     # CES consumer
│   │   ├── planner.py      # Tax planner (legacy)
│   │   └── worker.py       # Worker agent (legacy)
│   ├── models/             # LLM provider integrations
│   │   ├── openai_model.py
│   │   ├── gemini_model.py
│   │   ├── vllm_model.py
│   │   ├── openrouter_model.py
│   │   └── base.py
│   ├── env/                # BazaarEnv simulation environment
│   ├── market_core/        # Market clearing and mechanics
│   ├── utils/              # Shared utilities
│   ├── viz/                # Streamlit dashboard
│   │   └── dashboard.py
│   └── main.py             # Entry point and CLI
├── scripts/                # Experiment runners
│   ├── exp1.py             # THE_CRASH main sweep
│   ├── exp1_eas_sweep.py   # THE_CRASH × open-weight model sweep
│   ├── exp2.py             # LEMON_MARKET main sweep
│   ├── exp2_2.py           # LEMON_MARKET (no seller IDs ablation)
│   ├── exp2_eas_sweep.py   # LEMON_MARKET × buyer model sweep
│   ├── exp3.py             # Adversarial shock experiments
│   ├── exp3_open_weights_sweep.py
│   ├── exp5.py             # Discovery limit firms (DLF) ablation
│   ├── exp6.py             # Consumer procedural personas
│   ├── analyze_lemon_prompts.py
│   ├── compile_listing_corpus.py
│   ├── consolidate_states.py
│   └── extract_sybil_prompts.py
├── data/                   # Demographic and corpus data
├── documentation/          # Run commands and model reference
├── eval_logs/              # Reference evaluation data
├── examples/               # Usage examples
├── experiments/            # Experiment runner framework
├── tests/                  # Test suite
├── OPEN_WEIGHTS_MODELS.md  # Supported open-weight models
└── README.md
```

---

## CLI Reference

Run from the project root:

```bash
python -m ai_bazaar.main [OPTIONS]
```

### Agent Configuration


| Argument                     | Default | Description                                                                                                                                                           |
| ---------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--firm-type`                | `FIXED` | `LLM` or `FIXED` firm agents                                                                                                                                          |
| `--num-firms`                | `5`     | Number of firms (THE_CRASH)                                                                                                                                           |
| `--num-consumers`            | `50`    | Number of consumers                                                                                                                                                   |
| `--num-sellers`              | —       | Alias for `--num-firms` in LEMON_MARKET                                                                                                                               |
| `--num-buyers`               | —       | Alias for `--num-consumers` in LEMON_MARKET                                                                                                                           |
| `--num-stabilizing-firms`    | `0`     | Number of Stabilizing Firms (THE_CRASH). First N LLM firms get the stabilizing prompt and enforce price ≥ unit cost                                                   |
| `--seller-type`              | `FIXED` | LEMON_MARKET: `LLM` generates descriptions; `FIXED` uses templates                                                                                                    |
| `--sybil-cluster-size`       | `0`     | LEMON_MARKET: number of Sybil identities (last K of `--num-sellers`). `0` = no Sybil                                                                                  |
| `--firm-personas`            | —       | Comma-separated `persona:count` pairs for non-stabilizing firms (e.g. `competitive:3,volume_seeker:2`). Valid: `competitive`, `volume_seeker`, `reactive`, `cautious` |
| `--seller-personas`          | —       | LEMON_MARKET: comma-separated `persona:count` for honest sellers. Valid: `standard`, `detailed`, `terse`, `optimistic`                                                |
| `--disable-firm-personas`    | off     | Strip behavioral archetypes from all firm prompts                                                                                                                     |
| `--enable-consumer-personas` | off     | THE_CRASH: assign behavioral personas to CES consumers round-robin (`LOYAL`, `SMALL_BIZ`, `REP_SEEKER`, `VARIETY`)                                                    |
| `--unit-cost`                | `2.0`   | Unit cost of production                                                                                                                                               |
| `--max-supply-unit-cost`     | `1.0`   | Upper bound on randomly drawn per-firm unit costs                                                                                                                     |
| `--firm-initial-cash`        | `500.0` | Starting cash balance for each firm                                                                                                                                   |
| `--overhead-costs`           | `14.0`  | Fixed overhead cost per timestep per firm                                                                                                                             |
| `--firm-markup`              | `0.50`  | FIXED firm: markup over unit cost                                                                                                                                     |
| `--firm-tax-rate`            | `0.05`  | Tax rate on firm cash each timestep                                                                                                                                   |
| `--use-cost-pref-gen`        | off     | Generate heterogeneous supply costs and CES preferences via the heterogeneity module                                                                                  |
| `--use-gen-ces`              | off     | Generate CES parameters via LLM for consumers                                                                                                                         |


### Simulation Parameters


| Argument                             | Default          | Description                                                                                                                              |
| ------------------------------------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--consumer-scenario`                | `RACE_TO_BOTTOM` | `RACE_TO_BOTTOM`, `EARLY_BIRD`, `PRICE_DISCRIMINATION`, `RATIONAL_BAZAAR`, `BOUNDED_BAZAAR`, `THE_CRASH`, `LEMON_MARKET`                 |
| `--consumer-type`                    | `CES`            | `CES` or `FIXED` consumer agents                                                                                                         |
| `--max-timesteps`                    | `100`            | Simulation length                                                                                                                        |
| `--discovery-limit-consumers`        | `3`              | Max firms a consumer polls before ordering (`0` = no limit)                                                                              |
| `--discovery-limit-firms`            | `0`              | Max competitors each firm observes (`0` = no limit)                                                                                      |
| `--wtp-algo`                         | `none`           | `none` (always order), `wtp` (CES willingness-to-pay), `ewtp` (expected WTP)                                                             |
| `--poisson-demand-lambda`            | —                | Poisson arrival rate for consumer participation per step. Default: all consumers participate (THE_CRASH defaults to 0.6 × num_consumers) |
| `--info-asymmetry`                   | off              | Enable noisy competitor price observations for firms                                                                                     |
| `--crash-rep-scoring`                | off              | THE_CRASH: score quotes by reputation/price instead of 1/price                                                                           |
| `--dynamic-labor`                    | off              | Re-sample CES labor each timestep (vs. fixed at t=0)                                                                                     |
| `--consumption-interval`             | `1`              | Run consumer inventory consumption every N timesteps                                                                                     |
| `--num-goods`                        | `1`              | Number of goods in the simulation                                                                                                        |
| `--fixed-consumer-quantity-per-good` | `10.0`           | Quantity per good for FIXED consumers                                                                                                    |
| `--listing-corpus`                   | —                | LEMON_MARKET: path to pre-compiled listing corpus (eliminates seller LLM calls). See `scripts/compile_listing_corpus.py`                 |
| `--allow-listing-persistence`        | off              | LEMON_MARKET: carry unsold listings forward instead of discarding each step                                                              |


### Lemon Market Parameters


| Argument                    | Default | Description                                                                                         |
| --------------------------- | ------- | --------------------------------------------------------------------------------------------------- |
| `--reputation-initial`      | `0.8`   | Initial seller reputation R₀ (default `1.0` when no Sybil)                                          |
| `--reputation-pseudo-count` | `10`    | Rolling vote-window size N. `reputation = upvotes in last N / N`                                    |
| `--sybil-rho-min`           | `0.3`   | Sybil rotation threshold: when R < rho_min, spawn new identity at `--reputation-initial`            |
| `--no-buyer-rep`            | off     | Withhold seller reputation from buyer observations (ablation)                                       |
| `--no-seller-ids`           | off     | Omit seller identifiers from buyer observations; listings get ephemeral per-round labels (ablation) |
| `--lemon-base-buyer`        | off     | Minimal buyer prompt with no transaction history (ablation)                                         |


### LLM Configuration


| Argument                       | Default     | Description                                                                              |
| ------------------------------ | ----------- | ---------------------------------------------------------------------------------------- |
| `--llm`                        | `llama3:8b` | Model name. Examples: `gemini-2.5-flash`, `gpt-4o`, `meta-llama/llama-3.1-8b-instruct`   |
| `--buyer-llm`                  | —           | LEMON_MARKET: model for buyer agents (falls back to `--llm`)                             |
| `--seller-llm`                 | —           | LEMON_MARKET: model for honest sellers and Sybil principal (falls back to `--llm`)       |
| `--stab-llm`                   | —           | THE_CRASH: model name for Stabilizing Firms (e.g. a vLLM LoRA adapter alias like `stab`) |
| `--service`                    | `vllm`      | `vllm` or `ollama` for local models                                                      |
| `--buyer-service`              | —           | LEMON_MARKET: service for buyer agents (falls back to `--service`)                       |
| `--seller-service`             | —           | LEMON_MARKET: service for seller agents (falls back to `--service`)                      |
| `--port`                       | `8009`      | Port for LLM service                                                                     |
| `--buyer-port`                 | —           | LEMON_MARKET: port for buyer LLM (falls back to `--port`)                                |
| `--seller-port`                | —           | LEMON_MARKET: port for seller LLM (falls back to `--port`)                               |
| `--gemini-backend`             | auto        | `studio` (API key) or `vertex` (Vertex AI). Auto-detects from env vars                   |
| `--openrouter-provider`        | —           | Preferred OpenRouter provider(s) (e.g. `anthropic`, `Together`)                          |
| `--buyer-openrouter-provider`  | —           | LEMON_MARKET: OpenRouter provider for buyer agents                                       |
| `--seller-openrouter-provider` | —           | LEMON_MARKET: OpenRouter provider for seller agents                                      |
| `--prompt-algo`                | `io`        | `io` (input-output) or `cot` (chain-of-thought)                                          |
| `--history-len`                | `3`         | Timesteps of history sent in each firm prompt                                            |
| `--best-n`                     | `3`         | Best-N slab size for Stabilizing Firm prompts (`0` to disable)                           |
| `--max-tokens`                 | `1000`      | Maximum output tokens per LLM call                                                       |
| `--timeout`                    | `30`        | LLM call timeout in seconds                                                              |
| `--use-parsing-agent`          | off         | Use a secondary LLM call to repair malformed JSON responses                              |


### Logging and Output


| Argument                   | Default  | Description                                                              |
| -------------------------- | -------- | ------------------------------------------------------------------------ |
| `--name`                   | `""`     | Run name (used as log directory label)                                   |
| `--log-dir`                | `logs`   | Base directory for output files                                          |
| `--seed`                   | `42`     | Random seed                                                              |
| `--wandb`                  | off      | Enable Weights & Biases logging                                          |
| `--no-diaries`             | off      | Disable strategic diary entries in agent prompts                         |
| `--log-firm-prompts`       | off      | Log firm prompt/response pairs to file                                   |
| `--log-crash-firm-prompts` | off      | THE_CRASH: append firm prompts to `crash_agent_prompts.jsonl`            |
| `--log-buyer-prompts`      | off      | LEMON_MARKET: append buyer prompts to `lemon_agent_prompts.jsonl`        |
| `--log-seller-prompts`     | off      | LEMON_MARKET: append seller/Sybil prompts to `lemon_agent_prompts.jsonl` |
| `--log-alignment-traces`   | off      | Log `(state, prompt, response, outcome)` tuples for SFT data collection  |
| `--reward-type`            | `PROFIT` | `PROFIT` or `REVENUE` reward signal                                      |


### Shock Parameters (Experiment 3)


| Argument                          | Default | Description                                             |
| --------------------------------- | ------- | ------------------------------------------------------- |
| `--shock-timestep`                | —       | Timestep at which to inject the shock                   |
| `--post-shock-unit-cost`          | —       | New unit cost after supply shock (THE_CRASH)            |
| `--post-shock-sybil-cluster-size` | —       | New Sybil cluster size after flood shock (LEMON_MARKET) |


---

## Running Experiments

All experiment scripts live in `scripts/` and must be run from the **project root**. Use `--list` to preview runs without executing, and `--skip-existing` to resume partial sweeps.

### Experiment 1 — THE_CRASH

5 LLM firms, 50 CES consumers, 365 timesteps. Sweeps stabilizing firm count and consumer discovery limit.

**Full matrix (54 runs):** baseline (k=0) over dlc ∈ {1,3,5} × seeds {8,16,64} + stabilizing-firm sweeps over dlc ∈ {1,3,5} × n_stab ∈ {1,2,3,4,5} × seeds {8,16,64}.

```bash
# Run all 54 cells sequentially
python scripts/exp1.py

# Parallel (keep workers low to respect API rate limits)
python scripts/exp1.py --workers 3

# Use a different model
python scripts/exp1.py --llm gemini-2.5-flash

# Local model via Ollama
python scripts/exp1.py --llm gemma3:4b --service ollama --port 11434

# Via OpenRouter with a specific provider
python scripts/exp1.py --llm anthropic/claude-sonnet-4-6 --openrouter-provider anthropic

# Filter: only dlc=3, n_stab=1 or 3, seed=8
python scripts/exp1.py --dlc 3 --n-stab 1 3 --seeds 8

# Resume a partial run
python scripts/exp1.py --skip-existing --workers 3

# Preview matching runs without launching (RECOMMENDED BEFORE LAUNCHING A JOB)
python scripts/exp1.py --dlc 3 --n-stab 1 3 --list
```

**Filter flags:** `--dlc`, `--n-stab`, `--seeds`, `--run` (exact labels), `--skip-existing`

Logs: `logs/exp1_<model>/`

---

#### Experiment 1 — EAS × Model Size sweep

Runs the full Exp1 matrix for all dense open-weight models (3B–405B) via OpenRouter. See `OPEN_WEIGHTS_MODELS.md` for the full model list.

```bash
# All models, 4 parallel workers
python scripts/exp1_eas_sweep.py --workers 4 --skip-existing

# Only dlc=3, k=3 cells (for the health-vs-size scatter)
python scripts/exp1_eas_sweep.py --dlc 3 --n-stab 3 --workers 4

# Subset of models by name substring
python scripts/exp1_eas_sweep.py --models llama-3.2-3b gemma-3-4b --workers 2
```

---

### Experiment 2 — LEMON_MARKET

12 sellers (honest = 12 − K, Sybil = K), 12 LLM buyers, 50 timesteps. Sweeps Sybil saturation and reputation visibility.

**Full matrix (24 runs):** K ∈ {0,3,6,9} × rep_visible ∈ {True,False} × seeds {8,16,64}.

```bash
# All 24 cells
python scripts/exp2.py --seller-llm google/gemma-3-12b-it

# Parallel
python scripts/exp2.py --seller-llm google/gemma-3-12b-it --workers 3

# Split buyer and seller models
python scripts/exp2.py \
  --seller-llm google/gemma-3-12b-it --seller-openrouter-provider Together \
  --buyer-llm anthropic/claude-sonnet-4-6 --buyer-openrouter-provider anthropic

# Prompt logging
python scripts/exp2.py --log-buyer-prompts --log-seller-prompts

# Filter: only K=3 and K=6, rep hidden
python scripts/exp2.py --k 3 6 --rep-visible 0
```

**Filter flags:** `--k`, `--rep-visible` (1=visible, 0=hidden), `--seeds`, `--run`, `--skip-existing`

---

#### Experiment 2-2 — No Seller IDs ablation

Identical to Exp2 with `--no-seller-ids` hardwired, isolating whether buyers can detect lemons without cross-round seller tracking.

```bash
python scripts/exp2_2.py --llm gemini-2.5-flash --workers 3
```

---

#### Experiment 2 — EAS × Buyer Model sweep

Sweeps buyer model capability against a fixed seller model.

```bash
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --workers 4 --skip-existing
```

---

### Experiment 3 — Adversarial Shocks

Applies mid-episode shocks to measure market resilience.


| Sub-experiment | Scenario     | Shock                          | Timing |
| -------------- | ------------ | ------------------------------ | ------ |
| **exp3a**      | THE_CRASH    | Unit cost $1 → $10             | t = 25 |
| **exp3b**      | LEMON_MARKET | Sybil cluster → 80% saturation | t = 15 |


**Full matrix (36 runs):** 18 crash (n_stab ∈ {1,3,5} × dlc ∈ {3,5} × seeds) + 18 lemon (k_initial ∈ {3,6,9} × rep_visible × seeds).

```bash
# All 36 runs
python scripts/exp3.py

# Only crash or lemon sub-experiment
python scripts/exp3.py --experiment crash
python scripts/exp3.py --experiment lemon

# Override the model under test
python scripts/exp3.py --test-llm anthropic/claude-sonnet-4-6 --openrouter-provider anthropic
```

You can also run shocks directly:

```bash
# Supply shock at t=25
python -m ai_bazaar.main \
  --consumer-scenario THE_CRASH \
  --firm-type LLM --num-firms 5 --num-consumers 50 \
  --use-cost-pref-gen --overhead-costs 14 --max-timesteps 100 \
  --shock-timestep 25 --post-shock-unit-cost 10.0 \
  --llm gemini-3-flash-preview --seed 8

# Sybil flood at t=15
python -m ai_bazaar.main \
  --consumer-scenario LEMON_MARKET \
  --num-sellers 12 --num-buyers 12 \
  --sybil-cluster-size 3 --reputation-initial 0.8 \
  --max-timesteps 50 \
  --shock-timestep 15 --post-shock-sybil-cluster-size 36 \
  --llm gemini-3-flash-preview --seed 8
```

---

### Experiment 5 — Discovery Limit Firms (DLF) ablation

Mirrors Exp1 but sweeps firm-side price discovery (`--discovery-limit-firms`) with consumer discovery held at dlc=3.

```bash
python scripts/exp5.py --workers 3

# Filter by DLF value
python scripts/exp5.py --dlf 3 --n-stab 1 2
```

---

### Experiment 6 — Consumer Procedural Personas

Tests demand-side heterogeneity with behavioral consumer personas (`LOYAL`, `SMALL_BIZ`, `PRICE_HAWK`, `POPULAR`, `VARIETY`) at dlc=5.

```bash
python scripts/exp6.py

# Parallel with a specific model
python scripts/exp6.py --llm gemini-2.5-flash --workers 3
```

---

## Running on HPC (Slurm)

For large sweeps on a GPU cluster, start a single vLLM server with LoRA serving to route base model and adapter requests to one GPU:

```bash
# Start vLLM with LoRA adapters
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.5-9B \
  --enable-lora \
  --lora-modules stab=./models/ai-bazaar-checkpoints/crash_stabilizer \
                 guardian=./models/ai-bazaar-checkpoints/lemon_guardian \
  --port 8000 --gpu-memory-utilization 0.7

# Run exp1 against the base model, with Stabilizing Firms using the LoRA adapter
python scripts/exp1.py \
  --llm ./models/Qwen3.5-9B \
  --stab-llm stab \
  --service vllm --port 8000 \
  --workers 5 --skip-existing
```

The listing corpus (`data/listing_corpus.json`) eliminates seller LLM calls in lemon experiments, saving compute:

```bash
python -m ai_bazaar.main \
  --consumer-scenario LEMON_MARKET \
  --num-sellers 12 --num-buyers 12 \
  --sybil-cluster-size 3 \
  --listing-corpus data/listing_corpus.json \
  --buyer-llm guardian --service vllm --port 8000 \
  --max-timesteps 50 --seed 8

# Rebuild the corpus from existing logs
python scripts/compile_listing_corpus.py
```

---

## Testing

```bash
# Run the full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=ai_bazaar --cov-report=html

# Individual test modules
pytest tests/test_bazaar_env.py -v
pytest tests/test_lemon_market.py -v
pytest tests/test_buyer_agent.py -v
```

Most tests are self-contained. Tests that make live API calls (`test_models.py`, `test_advanced_usage.py`) require the relevant API keys.

---

## Supported Models

Any model accessible via the following backends is supported:


| Backend                    | Flag                                             | Examples                                                       |
| -------------------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| Google Gemini (AI Studio)  | `--llm gemini-2.5-flash`                         | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview` |
| Google Vertex AI           | `--llm gemini-2.5-flash --gemini-backend vertex` | Same model IDs                                                 |
| OpenAI                     | `--llm gpt-4o`                                   | `gpt-4o`, `gpt-4o-mini`, `gpt-5.4`                             |
| Anthropic (via OpenRouter) | `--llm anthropic/claude-sonnet-4-6`              | Any Anthropic model slug                                       |
| OpenRouter                 | `--llm org/model-name`                           | Any model on [openrouter.ai](https://openrouter.ai/models)     |
| Ollama (local)             | `--service ollama --llm llama3.1:8b`             | Any model in `ollama list`                                     |
| vLLM (local)               | `--service vllm --llm hf/model-id`               | Any HF model or LoRA alias                                     |


See `OPEN_WEIGHTS_MODELS.md` for the full list of tested open-weight models and their OpenRouter IDs.

---

## Citation

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

This framework builds on the LLM Economist:

```bibtex
@article{karten2025llm,
  title={LLM Economist: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra},
  author={Karten, Seth and Li, Wenzhe and Ding, Zihan and Kleiner, Samuel and Bai, Yu and Jin, Chi},
  journal={arXiv preprint arXiv:2507.15815},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.