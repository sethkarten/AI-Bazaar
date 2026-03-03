# Simulation Run Commands

Commands to run certain simulations. Add commands below as needed.

Run from project root. Use `**python -m ai_bazaar.main**` (works without installing the package). Alternatively, after `pip install -e .`, you can use `ai-bazaar`.

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
python -m ai_bazaar.main   --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit 2 --name local_test --max-timesteps 10 --prompt-algo cot --llm llama3.1:8b --service ollama --port 11434
```

**Important:** Use `--service ollama` and `--port 11434` (Ollama's default). The `--llm` value must match a model you pulled (e.g. `llama3.1:8b`, `mistral:7b`).

**Optional:** If your shell sets `GOOGLE_API_KEY` or `GEMINI_API_KEY`, unset them so the app doesn't try Gemini: e.g. `$env:GOOGLE_API_KEY = $null` (PowerShell) or don't set them in your activate script when using Ollama.

### Example: smaller/faster run

```bash
ollama pull llama3.2:3b
python -m ai_bazaar.main   --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit 2 --name local_fast --max-timesteps 5 --prompt-algo cot --llm llama3.2:3b --service ollama --port 11434
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
   python -m ai_bazaar.main   --firm-type LLM --num-firms 2 --num-consumers 10 --discovery-limit 2 --name gemma4b_test --max-timesteps 10 --prompt-algo cot --llm google/gemma-3-4b-it --service vllm --port 8009
  ```
   Or use the short name: `--llm gemma3:4b`.

---

## Visualization dashboard

Run the Streamlit dashboard to inspect simulation state (requires state files from a run that saves state, e.g. via `bazaar_env`). State files are stored under `logs/<run_name>/state_t*.json` (e.g. `logs/crash_baseline_test_1/state_t0.json`). The dashboard lists runs and lets you pick one.

```bash
streamlit run ai_bazaar/viz/dashboard.py
```

Run from project root so the dashboard finds the `logs/` directory.

---

## THE CRASH

Algorithmic Instability ("The Crash"): supply shock and price volatility. Use `--consumer-scenario THE_CRASH` and `--use-cost-pref-gen` for the paper setup.

**Run all Crash Tests from project root:** `python scripts/run_crash_tests.py`  
Logs go to `logs/crash_tests/` (summary + per-run logs).

### Ollama 4-parallel (CRASH test)

Start Ollama with `OLLAMA_NUM_PARALLEL=4` (e.g. via your Ollama conda env or `$env:OLLAMA_NUM_PARALLEL="4"; ollama serve`), then from project root with your app conda env activated:

```bash
python -m ai_bazaar.main --name crash_test_ollama_4parallel --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 2 --consumer-type CES --num-consumers 10 --max-timesteps 10 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm llama3.2:3b --service ollama --port 11434 --discovery-limit 2 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

Use a model you have (e.g. `llama3.1:8b`).

### Baseline (LLM vs FIXED firm)

```bash
python -m ai_bazaar.main --name crash_baseline_test_1 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type FIXED --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 50 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --discovery-limit 3 --firm-markup 50 --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

### THE CRASH 365 timesteps (no stabilizing firm)

Three runs varying the seed (same setup, different seeds for replication):

```bash
python -m ai_bazaar.main --name crash_365_seed8 --use-cost-pref-gen --max-supply-unit-cost 1   --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 5000 --consumer-scenario THE_CRASH --discovery-limit 3 --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name crash_365_seed42 --use-cost-pref-gen --max-supply-unit-cost 1   --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 5000 --consumer-scenario THE_CRASH --discovery-limit 3 --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 9
```

```bash
python -m ai_bazaar.main --name crash_365_seed123 --use-cost-pref-gen --max-supply-unit-cost 1   --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 5000 --consumer-scenario THE_CRASH --discovery-limit 3 --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 10
```

### THE CRASH with stabilizing firm

One firm is a stabilizing firm (volatility dampener); use `--stabilizing-firm`. Same scale as the 365-step runs for comparison:

```bash
python -m ai_bazaar.main --name crash_stabilizing_365 --use-cost-pref-gen --max-supply-unit-cost 1   --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 365 --firm-initial-cash 5000 --consumer-scenario THE_CRASH --discovery-limit 3 --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --stabilizing-firm --seed 8
```

---

## Lemon Market Tests

Used-car (lemon) market: firms list cars (quality from QUALITY_DICT), consumers order if CS = E[q]*max_wtp - price > 0; seller reputation (R_new = alpha*R + (1-alpha)*q) applies. Use `--consumer-scenario LEMON_MARKET`.

### Lemon market proto-run (short sanity check)

Minimal run to verify the lemon market pipeline (1 firm, 5 consumers, 5 timesteps):

```bash
python -m ai_bazaar.main --name lemon_proto_1 --firm-type LLM --num-firms 1 --consumer-type CES --num-consumers 5 --max-timesteps 5 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

### Lemon market (no Sybil)

```bash
python -m ai_bazaar.main --name lemon_test_nosybil_1 --use-cost-pref-gen --firm-type LLM --num-firms 1 --consumer-type CES --num-consumers 50 --max-timesteps 10 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name lemon_test_nosybil_1_seed9 --use-cost-pref-gen --firm-type LLM --num-firms 1 --consumer-type CES --num-consumers 50 --max-timesteps 10 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 9
```

```bash
python -m ai_bazaar.main --name lemon_test_nosybil_1_seed10 --use-cost-pref-gen --firm-type LLM --num-firms 1 --consumer-type CES --num-consumers 50 --max-timesteps 10 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 10
```

With optional reputation alpha (default 0.9):

```bash
python -m ai_bazaar.main --name lemon_test_nosybil_2  --firm-type LLM --num-firms 3 --consumer-type CES --num-consumers 10 --max-timesteps 50 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --reputation-alpha 0.9 ---llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

```bash
python -m ai_bazaar.main --name lemon_test_nosybil_3 --firm-type LLM --num-firms 10 --consumer-type CES --num-consumers 10 --max-timesteps 50 --firm-initial-cash 1000 --consumer-scenario LEMON_MARKET --reputation-alpha 0.9 --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

### Lemon market with Sybil cluster

Last few firms are Sybil identities (one principal, multiple seller IDs). Use `--sybil-cluster-size` to mark the Sybil cluster:

```bash
python -m ai_bazaar.main --name lemon_sybil_1 --use-cost-pref-gen --firm-type LLM --num-firms 10 --consumer-type CES --num-consumers 50 --max-timesteps 50 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --sybil-cluster-size 3 --llm gemma3:4b --service ollama --port 11434 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

**Lemon market unit tests** (no LLM, fast):

```bash
python -m pytest tests/test_lemon_market.py -v
```

---

## Crash and Lemon (short, fixed labor, Gemini Flash)

Short runs for crash and lemon, no dynamic labor

### Crash (100 steps, no stabilizing firm)

```bash
python -m ai_bazaar.main --name crash_100_flash_1 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 100 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --llm gemini-2.5-flash --discovery-limit 3 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

### Crash (100 steps, stabilizing firm)
```bash
python -m ai_bazaar.main --name crash_100_flash_1 --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 50 --max-timesteps 100 --firm-initial-cash 1000 --consumer-scenario THE_CRASH --stabilizing-firm --llm gemini-2.5-flash --discovery-limit 3 --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

### Lemon (50 steps): no Sybil

```bash
python -m ai_bazaar.main --name lemon_50_flash_nosybil_1 --use-cost-pref-gen --firm-type LLM --num-firms 10 --consumer-type CES --num-consumers 50 --max-timesteps 50 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

### Lemon (50 steps): with Sybil cluster

```bash
python -m ai_bazaar.main --name lemon_50_flash_sybil_1 --use-cost-pref-gen --firm-type LLM --num-firms 10 --consumer-type CES --num-consumers 50 --max-timesteps 50 --firm-initial-cash 5000 --consumer-scenario LEMON_MARKET --sybil-cluster-size 3 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```

---

## Scratch
```bash
python -m ai_bazaar.main --name gemini_test --use-cost-pref-gen --max-supply-unit-cost 1 --firm-type LLM --num-goods 1 --num-firms 5 --consumer-type CES --num-consumers 5 --max-timesteps 10 --firm-initial-cash 5000 --consumer-scenario THE_CRASH --discovery-limit 3 --llm gemini-2.5-flash --max-tokens 2000 --prompt-algo cot --no-diaries --seed 8
```