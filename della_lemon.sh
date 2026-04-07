#!/bin/bash
#SBATCH --job-name=bazaar-lemon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=chij
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lemon_%j.log
#SBATCH --error=logs/lemon_%j.err

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$(pwd):$PYTHONPATH

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/$(whoami)/AI-Bazaar}"
cd "$REPO_ROOT"
module load cudatoolkit/12.6
conda activate ai-bazaar
mkdir -p logs

BASE_MODEL="${BASE_MODEL:-./models/Qwen3.5-9B}"
GUARDIAN_LORA="${GUARDIAN_LORA:-./models/ai-bazaar-checkpoints/lemon_guardian}"

# Start vLLM server: base model + guardian LoRA adapter
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --enable-lora \
    --lora-modules guardian="$GUARDIAN_LORA" \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests &
VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
while ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 5
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM failed to start"; exit 1; }
done
echo "vLLM ready."

COMMON="--llm $BASE_MODEL --buyer-llm guardian --service vllm --port 8000 \
        --listing-corpus data/listing_corpus.json \
        --workers ${WORKERS:-3} --skip-existing"

python scripts/exp2.py $COMMON --no-seller-ids
python scripts/exp3.py --experiment lemon $COMMON

kill $VLLM_PID
echo "Done. lemon experiments complete."
