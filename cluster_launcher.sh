#!/bin/bash
# Wrapper for GPU Manager to run in the pre-installed uv venv on remote clusters

echo "=== Cluster Launcher Started ==="
echo "Current directory: $(pwd)"
echo "Arguments: $@"

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# Activate pre-installed venv
if [ -f ".venv/bin/activate" ]; then
    echo "Activating local .venv..."
    source .venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    echo "Activating parent .venv..."
    source ../.venv/bin/activate
else
    echo "ERROR: .venv/bin/activate not found in $(pwd) or parent!"
    exit 1
fi

# Force offline modes for HuggingFace (but NOT wandb)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Extract log directory from arguments for unique stdout/stderr capture
LOG_DIR="logs"
RUN_NAME="unknown"
WANDB_MODE="online"
for arg in "$@"; do
    case "$prev_arg" in
        --log-dir)
            LOG_DIR="$arg"
            ;;
        --run_name)
            RUN_NAME="$arg"
            ;;
        --wandb_mode)
            WANDB_MODE="$arg"
            ;;
    esac
    prev_arg="$arg"
done

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Get job ID from environment (set by GPU Manager)
JOB_ID="${GPU_MANAGER_JOB_ID:-unknown}"

# Write job metadata to log directory
cat > "$LOG_DIR/job_info.json" << EOF
{
  "job_id": "$JOB_ID",
  "run_name": "$RUN_NAME",
  "log_dir": "$LOG_DIR",
  "wandb_mode": "$WANDB_MODE",
  "started_at": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "pwd": "$(pwd)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
}
EOF

echo "Job ID: $JOB_ID"
echo "Run name: $RUN_NAME"
echo "Log directory: $LOG_DIR"
echo "WandB mode: $WANDB_MODE"
echo "Job info written to: $LOG_DIR/job_info.json"

# Create unique log files for this job
STDOUT_LOG="$LOG_DIR/stdout.log"
STDERR_LOG="$LOG_DIR/stderr.log"

# Run training with redirected output
echo "Starting training script..."
echo "Redirecting stdout to: $STDOUT_LOG"
echo "Redirecting stderr to: $STDERR_LOG"
python3 ai_bazaar/train/train_reinforce.py "$@" > "$STDOUT_LOG" 2> "$STDERR_LOG"
EXIT_CODE=$?
echo "=== Cluster Launcher Finished (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE

