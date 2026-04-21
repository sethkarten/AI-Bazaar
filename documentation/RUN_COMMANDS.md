# Simulation Run Commands

Run from the **project root**. Use `python -m ai_bazaar.main` (works without installing the package). After `pip install -e .` you can also use `ai-bazaar` directly.

---

## Visualization Dashboard

Inspect simulation state in the Streamlit dashboard. State files are stored at `logs/<run_name>/states.json`.

```bash
streamlit run ai_bazaar/viz/dashboard.py
```

---

## EXPERIMENT 1 — THE_CRASH

**Scenario:** 5 LLM firms, 50 CES consumers, 365 timesteps. Firms compete on price and can spiral into bankruptcy. Sweeps consumer discovery limit (dlc) and stabilizing firm count (n_stab).

**Common settings:** `--use-cost-pref-gen`, `--no-diaries`, `--prompt-algo cot`, `--max-tokens 2000`, `--overhead-costs 14`, `--llm gemini-2.5-flash`.

### `scripts/exp1.py` — Main sweep

**Full matrix (54 runs):** baseline (k=0) over dlc ∈ {1,3,5} × seeds {8,16,64} + stabilizing-firm sweeps over dlc ∈ {1,3,5} × n_stab ∈ {1,2,3,4,5} × seeds {8,16,64}.

Logs go to `logs/exp1_<model>/`; per-run state files go to `logs/<run_name>/`.

```bash
# Run all 54 cells sequentially
python scripts/exp1.py

# Parallel (keep workers low to respect API rate limits)
python scripts/exp1.py --workers 3

# Different model
python scripts/exp1.py --llm gemini-2.0-flash

# Local model via Ollama (start Ollama first with OLLAMA_NUM_PARALLEL=4)
python scripts/exp1.py --llm gemma3:4b --service ollama --port 11434

# Via OpenRouter
python scripts/exp1.py --llm anthropic/claude-sonnet-4-6 --openrouter-provider anthropic
```

**Filtering runs** — all filters combine with AND logic; use `--list` to preview without executing:

```bash
python scripts/exp1.py --list                          # preview all runs
python scripts/exp1.py --dlc 3                         # only dlc=3 cells
python scripts/exp1.py --n-stab 4 5                    # only n_stab=4 and 5
python scripts/exp1.py --seeds 8                       # only seed=8
python scripts/exp1.py --dlc 1 --n-stab 1 2            # combine filters
python scripts/exp1.py --run exp1_baseline exp1_stab_2_dlc3_seed8  # exact labels
python scripts/exp1.py --skip-existing --workers 3     # resume partial run
```

---

### `scripts/exp1_eas_sweep.py` — EAS × Model Size sweep

Runs the full Exp1 matrix for every dense open-weight model (3B–405B) via OpenRouter. See `OPEN_WEIGHTS_MODELS.md` for the full model list.

**Total runs:** ~21 models × 15 cells each. Logs per model go to `logs/exp1_{model_slug}/`.

```bash
python scripts/exp1_eas_sweep.py --list                          # preview
python scripts/exp1_eas_sweep.py --workers 4 --skip-existing     # full sweep
python scripts/exp1_eas_sweep.py --dlc 3 --n-stab 3 --workers 4  # health-vs-size cell only
python scripts/exp1_eas_sweep.py --models llama-3.2-3b gemma-3-4b --workers 2  # subset of models
python scripts/exp1_eas_sweep.py --openrouter-provider Together --workers 4
```

All `exp1.py` filter flags (`--dlc`, `--n-stab`, `--seeds`, `--skip-existing`, `--list`) are supported. `--models` matches substrings against OpenRouter model ID or display name.

---

## EXPERIMENT 2 — LEMON_MARKET

**Scenario:** 12 sellers (honest = 12 − K, Sybil = K), 12 LLM buyers, 50 timesteps. A Sybil principal operates K coordinated identities and rotates them when reputation drops below `rho_min`. Sweeps Sybil saturation (K) and reputation visibility.

**Common settings:** `--discovery-limit-consumers 3`, `--no-diaries`, `--prompt-algo cot`, `--max-tokens 2000`, `--reputation-initial 0.8`, `--reputation-pseudo-count 10`, `--sybil-rho-min 0.3`. Honest seller personas distributed evenly across `standard/detailed/terse/optimistic`.

**Buyer/Seller LLM split:** Use `--buyer-llm` and `--seller-llm` to assign different models to each role. Use `--buyer-openrouter-provider` and `--seller-openrouter-provider` to route through different providers.

### `scripts/exp2.py` — Main sweep

**Full matrix (24 runs):** K ∈ {0,3,6,9} × rep_visible ∈ {True,False} × seeds {8,16,64}.

```bash
# Run all 24 cells (always specify --seller-llm)
python scripts/exp2.py --seller-llm google/gemma-3-12b-it

# Parallel
python scripts/exp2.py --seller-llm google/gemma-3-12b-it --workers 3

# Split buyer and seller models / providers
python scripts/exp2.py \
  --seller-llm google/gemma-3-12b-it --seller-openrouter-provider Together \
  --buyer-llm anthropic/claude-sonnet-4-6 --buyer-openrouter-provider anthropic

# Prompt logging
python scripts/exp2.py --log-buyer-prompts --log-seller-prompts
```

**Filtering runs:**

```bash
python scripts/exp2.py --list
python scripts/exp2.py --k 0                       # baseline only
python scripts/exp2.py --k 3 6                     # K=3 and K=6
python scripts/exp2.py --rep-visible 1             # reputation visible only
python scripts/exp2.py --rep-visible 0             # reputation hidden only
python scripts/exp2.py --k 9 --rep-visible 0 --seeds 8 16
python scripts/exp2.py --skip-existing
```

---

### `scripts/exp2_2.py` — No seller IDs ablation

Identical to `exp2.py` with `--no-seller-ids` hardwired on. Listings receive ephemeral per-round labels; `seller_id` is absent from transaction history. Isolates whether buyers can avoid lemons using only description quality and reputation — without cross-round seller tracking.

Logs go to `logs/exp2_2_{buyer_slug}/`; run names use prefix `exp2_2_`.

```bash
python scripts/exp2_2.py --llm gemini-2.5-flash --workers 3
python scripts/exp2_2.py --buyer-llm meta-llama/llama-3.1-8b-instruct --seller-llm google/gemma-3-12b-it
python scripts/exp2_2.py --llm gemini-2.5-flash --rep-visible 1 --seeds 8
python scripts/exp2_2.py --llm gemini-2.5-flash --skip-existing
```

All `exp2.py` flags are supported except `--no-seller-ids` (always active).

---

### `scripts/exp2_eas_sweep.py` — EAS × Buyer Model sweep

Runs the full Exp2 matrix across all dense open-weight buyer models, with a **fixed seller model** (`--seller-llm` required). Isolates buyer sophistication from seller listing quality.

```bash
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --workers 4 --skip-existing
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --rep-visible 1 --k 3
python scripts/exp2_eas_sweep.py --seller-llm google/gemma-3-12b-it --models llama-3.2-3b gemma-3-4b
python scripts/exp2_eas_sweep.py \
  --seller-llm google/gemma-3-12b-it --seller-openrouter-provider Together \
  --buyer-openrouter-provider Together --workers 4
```

Logs per buyer model go to `logs/exp2_{buyer_slug}/`.

---

### Prompt analysis utilities (Exp2)

**Extract Sybil principal prompts** — after running Exp2 with `--log-seller-prompts`, compiles Sybil conversations across seeds into one file per (K, rep) cell at `logs/exp2_<model>/data/`:

```bash
python scripts/extract_sybil_prompts.py                          # default: claude-sonnet-4.6
python scripts/extract_sybil_prompts.py --model gemini-2.5-flash
python scripts/extract_sybil_prompts.py --model gemini-2.5-flash --k 3 6 9 --rep rep1 --seed 8 16
```

Outputs: `k{K}_{rep}_sybil_prompts.json` and `k{K}_{rep}_sybil_tier_refusals.json`.

**Analyze lemon market prompts** — loads buyer/seller prompt logs and prints deception rate, pass rate, description style stats, and paper-ready qualitative examples. Requires runs with `--log-buyer-prompts --log-seller-prompts`.

```bash
python scripts/analyze_lemon_prompts.py                          # default: claude-sonnet-4.6
python scripts/analyze_lemon_prompts.py --model gemini-2.5-flash
python scripts/analyze_lemon_prompts.py --model gemini-2.5-flash --k 0 3 6 --seed 16
python scripts/analyze_lemon_prompts.py --model gemini-2.5-flash --output logs/prompt_analysis.txt
```

---

## EXPERIMENT 3 — Adversarial Shocks

**Scenario:** Extends Experiments 1 and 2 with mid-episode shocks to measure market resilience.

| Sub-experiment | Scenario | Shock | Timing |
|---|---|---|---|
| **exp3a** Supply Shock | THE_CRASH | Unit cost ~$1 → $10 | t = 25 |
| **exp3b** Flood of Fakes | LEMON_MARKET | Sybil cluster → 80% saturation | t = 15 |

**Recovery metrics:**
- **exp3a** — Markup ratio recovery: first t > 25 where `|μ_t − μ̄_pre| ≤ 0.1 · μ̄_pre`
- **exp3b** — Detection premium recovery: first t > 15 where `|δ_t − δ̄_pre| ≤ 0.1 · δ̄_pre`
- Markets that never recover are assigned τ = T (max_timesteps)

### `scripts/exp3.py` — Main sweep

**Full matrix (36 runs):** 18 crash (n_stab ∈ {1,3,5} × dlc ∈ {3,5} × seeds) + 18 lemon (k_initial ∈ {3,6,9} × rep_visible ∈ {T,F} × seeds). Logs go to `logs/exp3_{test-llm}/`.

```bash
python scripts/exp3.py                             # all 36 runs
python scripts/exp3.py --workers 3
python scripts/exp3.py --experiment crash          # only crash sub-experiment
python scripts/exp3.py --experiment lemon          # only lemon sub-experiment

# Override the model under test (firm for crash, buyer for lemon)
python scripts/exp3.py --test-llm anthropic/claude-sonnet-4-6 --openrouter-provider anthropic
python scripts/exp3.py --test-llm my-finetuned-model --seller-llm google/gemma-3-12b-it
python scripts/exp3.py --test-llm gemma3:4b --service ollama --port 11434
```

**Filtering runs:**

```bash
python scripts/exp3.py --list
python scripts/exp3.py --experiment crash --n-stab 1 3 --dlc 3
python scripts/exp3.py --experiment lemon --k 3 --rep-visible 1
python scripts/exp3.py --seeds 8
python scripts/exp3.py --skip-existing
```

**Running shocks manually:**

```bash
# Supply shock: unit cost → $10 at t=25
python -m ai_bazaar.main \
  --consumer-scenario THE_CRASH \
  --firm-type LLM --num-firms 5 --num-consumers 50 \
  --use-cost-pref-gen --overhead-costs 14 --max-timesteps 100 \
  --shock-timestep 25 --post-shock-unit-cost 10.0 \
  --llm gemini-3-flash-preview --seed 8 --name exp3a_test

# Sybil flood: cluster → 80% saturation at t=15 (k_initial=3 → flood_k=36)
python -m ai_bazaar.main \
  --consumer-scenario LEMON_MARKET \
  --num-sellers 12 --num-buyers 12 \
  --sybil-cluster-size 3 --reputation-initial 0.8 --sybil-rho-min 0.3 \
  --max-timesteps 50 \
  --shock-timestep 15 --post-shock-sybil-cluster-size 36 \
  --buyer-llm gemini-3-flash-preview --seller-llm gemini-3-flash-preview \
  --seed 8 --name exp3b_test
```

---

### `scripts/exp3_open_weights_sweep.py` — Open-weights model sweep

Runs `exp3.py` once per model in the sweep list with a fixed `--seller-llm` for lemon runs.

```bash
python scripts/exp3_open_weights_sweep.py --seller-llm google/gemma-3-12b-it
python scripts/exp3_open_weights_sweep.py --seller-llm google/gemma-3-12b-it --skip-existing
python scripts/exp3_open_weights_sweep.py --seller-llm google/gemma-3-12b-it --experiment lemon
python scripts/exp3_open_weights_sweep.py --seller-llm google/gemma-3-12b-it --models llama-3.2-3b gemma-3-4b
python scripts/exp3_open_weights_sweep.py --seller-llm google/gemma-3-12b-it --list
```

---

## EXPERIMENT 5 — Discovery Limit Firms (DLF) Ablation

**Purpose:** Mirror Experiment 1's consumer discovery limit (dlc) sweep on the firm side — isolate how many competitor prices each firm observes, holding consumer discovery fixed at dlc=3.

| Experiment | What varies | Held fixed |
|---|---|---|
| **Exp1** | `dlc` ∈ {1,3,5} | `dlf` = 0 (no limit) |
| **Exp5** | `dlf` ∈ {1,3,5} | `dlc` = 3 |

**Common settings:** Same as Experiment 1 (5 firms, 50 consumers, 365 timesteps, overhead 14).

### `scripts/exp5.py` — DLF sweep

**Full matrix (54 runs):** same layout as `exp1.py` but sweeping dlf instead of dlc. Logs go to `logs/exp5_<model>/`.

```bash
python scripts/exp5.py --workers 3
python scripts/exp5.py --llm gemini-2.0-flash --workers 3
python scripts/exp5.py --llm gemma3:4b --service ollama --port 11434
```

**Filtering runs:**

```bash
python scripts/exp5.py --list
python scripts/exp5.py --dlf 3
python scripts/exp5.py --n-stab 4 5
python scripts/exp5.py --dlf 1 --n-stab 1 2
python scripts/exp5.py --skip-existing --workers 3
```

---

## EXPERIMENT 6 — Consumer Procedural Personas

**Purpose:** Isolate demand-side heterogeneity effects in THE_CRASH. Consumer behavioral personas are assigned round-robin — `LOYAL`, `SMALL_BIZ`, `PRICE_HAWK`, `POPULAR`, `VARIETY` — biasing firm selection beyond price. Held at dlc=5 to amplify discovery effects.

**Design:** n_stab ∈ {0,3,5} × seeds {8,16,64} = 9 runs. All other settings match Exp1/Exp5.

### `scripts/exp6.py` — Consumer personas sweep

```bash
python scripts/exp6.py
python scripts/exp6.py --workers 3
python scripts/exp6.py --n-stab 0          # baseline only (no stabilizing firm)
python scripts/exp6.py --list
python scripts/exp6.py --skip-existing --workers 3
```

Supports `--llm`, `--service`, `--port`, `--stab-llm` passthrough flags.

**Targeted cell (for comparison with Exp1/Exp5 baselines):** run the Exp1 matrix at n_stab=3, dlc=3 with consumer personas enabled:

```bash
python scripts/exp1.py --llm gemini-2.5-flash --n-stab 3 --dlc 3 --enable-consumer-personas --workers 3 --skip-existing
```

---

## Local LLM Setup

### vLLM

Start a vLLM server in a **separate terminal** before running any simulation with `--service vllm`. Default port is **8009**.

```bash
pip install vllm

# Gemma 3 4B
python -m vllm.entrypoints.openai.api_server --model google/gemma-3-4b-it --port 8009

# Llama 3.1 8B
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8009
```

For gated models: `export HF_TOKEN="hf_..."` or run `huggingface-cli login` before starting the server. On Windows, vLLM works best in WSL2.

**Run a simulation against the vLLM server:**

```bash
python -m ai_bazaar.main \
  --firm-type LLM --num-firms 3 --num-consumers 20 \
  --llm google/gemma-3-4b-it --service vllm --port 8009 \
  --max-timesteps 10 --name vllm_test
```

---

### Ollama

[Ollama](https://ollama.com) runs models locally without any API quota.

**Install:** download from [ollama.com](https://ollama.com) (Windows/macOS) or `curl -fsSL https://ollama.com/install.sh | sh` (Linux).

```bash
# Pull a model and start the server (leave terminal open)
ollama pull llama3.1:8b
ollama serve

# Allow parallel requests (set before starting Ollama)
# Linux/macOS:
export OLLAMA_NUM_PARALLEL=4 && ollama serve
# Windows (PowerShell):
$env:OLLAMA_NUM_PARALLEL = "4"; ollama serve
```

**Recommended models by VRAM:**

| Model | 8 GB | 16 GB |
|-------|------|-------|
| `gemma2:2b` | Yes | Yes |
| `llama3.2:3b` | Yes | Yes |
| `mistral:7b` | Yes | Yes |
| `llama3.1:8b` | Yes | Yes |
| `gemma2:9b` | Tight | Yes |

**Run a simulation with Ollama:**

```bash
python -m ai_bazaar.main \
  --firm-type LLM --num-firms 2 --num-consumers 10 \
  --llm llama3.1:8b --service ollama --port 11434 \
  --max-timesteps 10 --name ollama_test
```

Use `--service ollama --port 11434`. The `--llm` value must match an installed model (`ollama list`).

---

### Downloading models from HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login    # one-time, required for gated models

# Download a model
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./models/Qwen3.5-9B

# Download a private repo (e.g. LoRA checkpoints)
git clone https://huggingface.co/machineExMachina/ai-bazaar-checkpoints ./models/ai-bazaar-checkpoints

# Download specific files only
huggingface-cli download Qwen/Qwen3.5-9B --include "*.safetensors" "*.json" --local-dir ./models/Qwen3.5-9B
```

Re-running the same command resumes an interrupted download.

---

## HPC / LoRA Serving

For large sweeps on a GPU cluster, a single vLLM server with LoRA serving routes base model and adapter requests to one GPU. This is the setup used for fine-tuned Stabilizing Firm and Skeptical Guardian experiments.

### Setup

```bash
# Download base model and adapters
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./models/Qwen3.5-9B
git clone https://huggingface.co/machineExMachina/ai-bazaar-checkpoints ./models/ai-bazaar-checkpoints
```

Expected model layout:
```
models/
  Qwen3.5-9B/
  ai-bazaar-checkpoints/
    crash_stabilizer/   # stab LoRA adapter
    lemon_guardian/     # guardian LoRA adapter
```

### Start vLLM with LoRA

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.5-9B \
  --enable-lora \
  --lora-modules stab=./models/ai-bazaar-checkpoints/crash_stabilizer \
                 guardian=./models/ai-bazaar-checkpoints/lemon_guardian \
  --port 8000 --gpu-memory-utilization 0.7
```

One GPU serves three model names:

| Request model | Weights |
|---|---|
| `./models/Qwen3.5-9B` | Base model |
| `stab` | Crash stabilizing firm adapter |
| `guardian` | Lemon guardian buyer adapter |

### Run experiments against the LoRA server

```bash
# Crash experiments: base model for non-stabilizing firms, stab adapter for stabilizing
python scripts/exp1.py \
  --llm ./models/Qwen3.5-9B --stab-llm stab \
  --service vllm --port 8000 --workers 5 --skip-existing

# Lemon experiments: base model (sellers use corpus), guardian adapter for buyers
python scripts/exp2.py \
  --llm ./models/Qwen3.5-9B --buyer-llm guardian \
  --listing-corpus data/listing_corpus.json \
  --service vllm --port 8000 --workers 5 --skip-existing
```

The `--stab-llm` and `--buyer-llm` flags are supported by all experiment scripts.

### Listing corpus feeder

Pre-compiling seller/sybil descriptions eliminates LLM calls on the seller side in lemon experiments. Pass `--listing-corpus data/listing_corpus.json` to activate:

```bash
# Rebuild the corpus from existing exp2 logs
python scripts/compile_listing_corpus.py
python scripts/compile_listing_corpus.py --log-dirs logs/exp2_gemini-3-flash-preview logs/exp2_my_run
```

Corpus coverage (51 Gemini exp2 runs): ~4,700 honest entries per quality tier, ~11,700 sybil entries.

### Slurm

For SLURM-based clusters:

```bash
# Submit jobs (adjust REPO_ROOT and partition to match your cluster)
sbatch --export=REPO_ROOT=/path/to/AI-Bazaar,WORKERS=5 della_lemon.sh
sbatch --export=REPO_ROOT=/path/to/AI-Bazaar,WORKERS=5 della_crash.sh

# Monitor
squeue -u $USER
tail -f logs/lemon_<JOBID>.log
scancel <JOBID>
```

Estimated runtimes (single A100, `--workers 3`):

| Job | Experiments | Runs | Wall time |
|---|---|---|---|
| lemon | Exp2 (no-seller-IDs) + Exp3 lemon | ~33 | 24h |
| crash | Exp3 crash + Exp5 DLF + Exp6 personas | ~69 | 48h |
