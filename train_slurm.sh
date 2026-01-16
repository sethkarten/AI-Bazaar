#!/bin/bash
#SBATCH --job-name=bazaar-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# Set offline modes
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment .venv not found. Please run installation via ssh first."
    exit 1
fi

# Start vLLM server in the background for inference
# We use the same model we are training, or a base model? 
# Usually we want the agents to use the current policy.
# For simplicity, let's have agents use the base model for now, 
# or we can use unsloth for in-process inference which is easier.

# However, following instructions "vLLM for inference":
uv run python3 -m vllm.entrypoints.openai.api_server \
    --model "unsloth/gemma-3-4b-it-bnb-4bit" \
    --port 8000 \
    --gpu-memory-utilization 0.4 \
    --disable-log-requests &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 5
done
echo "vLLM is ready."

# Use uv run to ensure the right environment
uv run python3 -m ai_bazaar.train.train_reinforce \
    --llm "unsloth/gemma-3-4b-it-bnb-4bit" \
    --port 8000 \
    --num_episodes 10 \
    --num_iterations 100 \
    --lr 5e-5 \
    --num-firms 3 \
    --num-consumers 10 \
    --max-timesteps 100 \
    --firm-type LLM \
    --consumer-type CES \
    --log-dir logs

# Cleanup
kill $VLLM_PID
