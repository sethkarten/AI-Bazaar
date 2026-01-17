#!/bin/bash
# Wrapper for GPU Manager to run in the pre-installed uv venv on della clusters

echo "=== Slurm Wrapper Started ==="
echo "Current directory: $(pwd)"
echo "Arguments: $@"

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# Activate pre-installed venv
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "ERROR: .venv/bin/activate not found!"
    exit 1
fi

# Force offline modes
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Run training
echo "Starting training script..."
python3 ai_bazaar/train/train_reinforce.py "$@"
echo "=== Slurm Wrapper Finished ==="
