#!/bin/bash

# Base configuration
LLM_BASE="./models"
NUM_EPISODES=2
NUM_ITERATIONS=2
LR=5e-5

# Function to submit a job
submit_job() {
    local NAME=$1
    local LLM=$2
    local REWARD=$3
    local DIARIES=$4
    local DISCOVERY=$5
    local ASYMMETRY=$6
    local PORT=$7

    local DIARY_FLAG=""
    if [ "$DIARIES" == "no" ]; then
        DIARY_FLAG="--no-diaries"
    fi

    local ASYM_FLAG=""
    if [ "$ASYMMETRY" == "yes" ]; then
        ASYM_FLAG="--info-asymmetry"
    fi

    # Create a custom slurm script for this ablation
    local SLURM_FILE="test_slurm_${NAME}.sh"
    
    cat > ${SLURM_FILE} <<EOF
#!/bin/bash
#SBATCH --job-name=test-${NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=ailab
#SBATCH --output=logs/test_${NAME}_%j.log
#SBATCH --error=logs/test_${NAME}_%j.err

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

. .venv/bin/activate

# Start vLLM server
python3 -m vllm.entrypoints.openai.api_server \\
    --model "${LLM}" \\
    --port ${PORT} \\
    --gpu-memory-utilization 0.4 \\
    --disable-log-requests &
VLLM_PID=\$!

echo "Waiting for vLLM to start on port ${PORT}..."
while ! curl -s http://localhost:${PORT}/v1/models > /dev/null; do
    sleep 5
    if ! kill -0 \$VLLM_PID 2>/dev/null; then
        echo "vLLM failed to start."
        exit 1
    fi
done

python3 -m ai_bazaar.train.train_reinforce \\
    --llm "${LLM}" \\
    --port ${PORT} \\
    --num_episodes ${NUM_EPISODES} \\
    --num_iterations ${NUM_ITERATIONS} \\
    --lr ${LR} \\
    --num-firms 3 \\
    --num-consumers 10 \\
    --max-timesteps 20 \\
    --firm-type LLM \\
    --consumer-type CES \\
    --reward-type "${REWARD}" \\
    --discovery-limit ${DISCOVERY} \\
    ${ASYM_FLAG} \\
    ${DIARY_FLAG} \\
    --service vllm \\
    --log-dir logs/results_${NAME}

kill \$VLLM_PID
EOF

    sbatch ${SLURM_FILE}
}

# Run short test for a subset first to verify
submit_job "test-base" "${LLM_BASE}/gemma-3-4b-it-bnb-4bit" "PROFIT" "yes" 5 "no" 8001
submit_job "test-qwen" "${LLM_BASE}/Qwen2.5-7B-Instruct-bnb-4bit" "PROFIT" "yes" 5 "yes" 8002
