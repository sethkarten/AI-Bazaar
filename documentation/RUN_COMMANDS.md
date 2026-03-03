# Simulation Run Commands

Commands to run certain simulations. Add commands below as needed.

Run from project root. Use **`python -m ai_bazaar.main`** (works without installing the package). Alternatively, after `pip install -e .`, you can use `ai-bazaar`.

---

## Quick sanity check

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

If it runs and shows a list (or “no models”), the server is up. If you get a connection error, start Ollama from the Start Menu / Applications, or run `ollama serve` in a terminal.

### Step 3: Install the Python `ollama` package (for the app)

Activate your conda env, then:

```bash
pip install ollama
```

(The app’s `OllamaModel` uses this to talk to the Ollama server.)

### Step 4: Pull a model

Pick one and download it (this can take a few minutes):

```bash
ollama pull llama3.1:8b
```

**Other options:**

| Model            | Size | Good for 8GB? | Good for 16GB? |
|------------------|------|---------------|----------------|
| `llama3.1:8b`    | 8B   | Yes           | Yes            |
| `llama3.2:3b`    | 3B   | Yes (faster)  | Yes            |
| `mistral:7b`     | 7B   | Yes           | Yes            |
| `phi3:mini`      | ~4B  | Yes           | Yes            |
| `gemma2:2b`      | 2B   | Yes (fastest) | Yes            |
| `gemma2:9b`      | 9B   | Tight         | Yes            |

### Step 5: Run the simulation with Ollama

From the **project root**, with your conda env activated:

```bash
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit 2 --name local_test --max-timesteps 10 --prompt-algo cot --llm llama3.1:8b --service ollama --port 11434
```

**Important:** Use `--service ollama` and `--port 11434` (Ollama’s default). The `--llm` value must match a model you pulled (e.g. `llama3.1:8b`, `mistral:7b`).

**Optional:** If your shell sets `GOOGLE_API_KEY` or `GEMINI_API_KEY`, unset them so the app doesn’t try Gemini: e.g. `$env:GOOGLE_API_KEY = $null` (PowerShell) or don’t set them in your activate script when using Ollama.

### Example: smaller/faster run

```bash
ollama pull llama3.2:3b
python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit 2 --name local_fast --max-timesteps 5 --prompt-algo cot --llm llama3.2:3b --service ollama --port 11434
```

---

## Run with a Hugging Face model (e.g. Gemma 3 4B) via vLLM

If you have access to a model on Hugging Face (e.g. **google/gemma-3-4b-it**), run it locally with **vLLM** and point the app at the vLLM server.

1. **Access the model on Hugging Face**
   - Open the model page (e.g. [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)), accept the license if it’s gated, and ensure you’re logged in.

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
   If the model is gated and you didn’t use `huggingface-cli login`, set `HF_TOKEN` in that terminal before this command. On **Windows**, vLLM often works best in **WSL2**.

4. **Run the simulation** with vLLM and the same model id (or the short name `gemma3:4b`, which is mapped to `google/gemma-3-4b-it` in the app):
   ```bash
   python -m ai_bazaar.main --use-env --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit 2 --name gemma4b_test --max-timesteps 10 --prompt-algo cot --llm google/gemma-3-4b-it --service vllm --port 8009
   ```
   Or use the short name: `--llm gemma3:4b`.

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

## Large: discovery limit variation
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
# Real Test
```bash
python -m ai_bazaar.main --name crash_test_1 --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 8 --consumer-type CES --num-consumers 40 --max-timesteps 150 --firm-initial-cash 5000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --use-parsing-agent --discovery-limit 8 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
```bash
python -m ai_bazaar.main --name crash_test_2 --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 8 --consumer-type CES --num-consumers 40 --max-timesteps 150 --firm-initial-cash 5000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --use-parsing-agent --discovery-limit 8 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
# Experiment 1: B2C Instability
```bash
python -m ai_bazaar.main --name crash_experiment_1_test_1 --use-cost-pref-gen --max-supply-unit-cost 1 --use-env --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 50 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --use-parsing-agent --discovery-limit 3 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
## Crash Environment: Baseline
```bash
python -m ai_bazaar.main --name crash_baseline_test_1 --use-cost-pref-gen --firm-type LLM --num-firms 1 --consumer-type CES --num-consumers 10 --max-timesteps 10 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --firm-markup 50 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
```bash
python -m ai_bazaar.main --name crash_baseline_test_2 --use-cost-pref-gen --firm-type FIXED --num-firms 1 --consumer-type CES --num-consumers 10 --max-timesteps 10 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --firm-markup 50 --max-tokens 2000 --no-diaries --seed 8
```

## Crash Environment: Parallelized Firms, Improved Parsing, Long Horizon Tests
```bash
python -m ai_bazaar.main --name crash_horizon_test_1 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 100 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 3 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```
```bash
python -m ai_bazaar.main --name crash_horizon_test_2 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 100 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 3 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

**Stabilizing Firm** (prompt + price floor ≥ unit cost) and **alignment-trace logging** for SFT:
```bash
python -m ai_bazaar.main --name crash_stabilizing_proto_test_1 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 10 --max-timesteps 50 --firm-initial-cash 5000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 3 --max-tokens 2000 --prompt-algo cot --no-diaries --stabilizing-firm --log-alignment-traces --seed 8
```
Traces are written to `logs/<run_name>/alignment_traces.jsonl` (one JSON object per line: state, firms' prompt/response/action, outcome).


# Lemon Market Tests

Used-car (lemon) market: firms list cars (quality from QUALITY_DICT), consumers order if CS = E[q]*max_wtp - price > 0; seller reputation (R_new = alpha*R + (1-alpha)*q) applies.

```bash
python -m ai_bazaar.main --name lemon_test_nosybil_1 --use-cost-pref-gen --use-env --firm-type LLM --num-firms 1 --consumer-type CES --num-consumers 10 --max-timesteps 10 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --llm gemini-2.5-flash --use-parsing-agent --discovery-limit 5 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

With optional reputation alpha (default 0.9):

```bash
python -m ai_bazaar.main --name lemon_test_nosybil_2 --use-env --firm-type LLM --num-firms 3 --consumer-type CES --num-consumers 10 --max-timesteps 10 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --reputation-alpha 0.9 --llm gemini-2.5-flash --discovery-limit 5 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name lemon_test_nosybil_3 --use-env --firm-type LLM --num-firms 10 --consumer-type CES --num-consumers 10 --max-timesteps 10 --firm-initial-cash 1000 --consumer-scenario LEMON_MARKET --reputation-alpha 0.9 --llm gemini-2.5-flash --discovery-limit 5 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

**Lemon market unit tests** (no LLM, fast):

```bash
python -m pytest tests/test_lemon_market.py -v
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
