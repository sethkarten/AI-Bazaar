#!/bin/bash
# Wrapper for GPU Manager to run in the pre-installed venv on della clusters

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate pre-installed venv
# Note: On della clusters, the manager puts the repo in /scratch/gpfs/CHIJ/milkkarten/PROJECT
source .venv/bin/activate

# Run training
python3 ai_bazaar/train/train_reinforce.py "$@"
