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

# Run training directly using the virtual environment to avoid uv network checks
. .venv/bin/activate

python3 -m ai_bazaar.train.train_reinforce \
    --llm "./models/gemma-3-4b-it-bnb-4bit" \
    --num_episodes 5 \
    --num_iterations 100 \
    --lr 5e-5 \
    --num-firms 3 \
    --num-consumers 10 \
    --max-timesteps 20 \
    --firm-type LLM \
    --consumer-type CES \
    --log-dir logs
