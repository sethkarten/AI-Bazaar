#!/bin/bash
#SBATCH --job-name=bazaar-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=ailab
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# Set offline modes
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
export UV_OFFLINE=1

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment .venv not found. Please run installation via ssh first."
    exit 1
fi

# Run training directly using the virtual environment to avoid uv network checks
. .venv/bin/activate

# Start vLLM server in the background for high-utilization inference
# Using 4-bit quantization and sharing GPU memory
python3 -m vllm.entrypoints.openai.api_server \
    --model "./models/gemma-3-4b-it-bnb-4bit" \
    --port 8000 \
    --gpu-memory-utilization 0.4 \
    --disable-log-requests &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 5
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM failed to start. Check logs."
        exit 1
    fi
done
echo "vLLM is ready."

# Run training. Agents will connect to vLLM server on port 8000.
python3 -m ai_bazaar.train.train_reinforce \
    --llm "./models/gemma-3-4b-it-bnb-4bit" \
    --port 8000 \
    --num_episodes 5 \
    --num_iterations 100 \
    --lr 5e-5 \
    --num-firms 3 \
    --num-consumers 10 \
    --max-timesteps 20 \
    --firm-type LLM \
    --consumer-type CES \
    --service vllm \
    --log-dir logs

# Cleanup
kill $VLLM_PID
