#!/bin/bash
# Local test launcher for 2x5090 using the existing 'llm' conda environment

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate existing stable environment
source /home/milkkarten/anaconda3/bin/activate llm

# Run training
python3 ai_bazaar/train/train_reinforce.py "$@"
