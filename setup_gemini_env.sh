#!/usr/bin/env bash
# setup_gemini_env.sh
# Creates (or updates) the AI-Bazaar conda environment, sets Google Cloud
# environment variables, installs the project, and launches ADC authentication.

set -e

ENV_NAME="AI-Bazaar"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== AI-Bazaar Gemini/Vertex AI Environment Setup ==="

# ── 1. Create or update conda environment ────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/4] Conda env '${ENV_NAME}' already exists — updating..."
    conda env update -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml" --prune
else
    echo "[1/4] Creating conda env '${ENV_NAME}'..."
    conda env create -f "${SCRIPT_DIR}/environment.yml"
fi

# ── 2. Install the project in editable mode ───────────────────────────────────
echo "[2/4] Installing ai-bazaar project (editable, no heavy deps)..."
conda run -n "${ENV_NAME}" pip install -e "${SCRIPT_DIR}" --no-deps --quiet

# ── 3. Set persistent conda environment variables ────────────────────────────
echo "[3/4] Setting conda environment variables..."
conda env config vars set -n "${ENV_NAME}" \
    GOOGLE_CLOUD_PROJECT=pokeagent-013 \
    VERTEX_LOCATION=us-central1

echo "    GOOGLE_CLOUD_PROJECT=pokeagent-013"
echo "    VERTEX_LOCATION=us-central1"

# ── 4. Google Cloud ADC authentication ───────────────────────────────────────
echo "[4/4] Launching Google Cloud authentication (browser window will open)..."
gcloud auth application-default login --project pokeagent-013

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate your environment with:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "Run a simulation with:"
echo "    python -m ai_bazaar.main --llm gemini-3-flash-preview"
