#!/bin/bash
# Wrapper for GPU Manager to run in the pre-installed 'llm' environment on della clusters

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate existing stable environment on the cluster
# We use the full path to the anaconda env to be safe
source /home/sk9014/anaconda3/bin/activate llm

# Force offline modes
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Run training
python3 ai_bazaar/train/train_reinforce.py "$@"
