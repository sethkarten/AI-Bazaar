#!/bin/bash
# Wrapper for GPU Manager to run in the pre-installed uv venv on della clusters

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate pre-installed venv
source .venv/bin/activate

# Force offline modes
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Run training
python3 ai_bazaar/train/train_reinforce.py "$@"
