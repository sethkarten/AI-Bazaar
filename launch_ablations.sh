#!/bin/bash

# Base configuration
LLM_BASE="./models"
NUM_EPISODES=5
NUM_ITERATIONS=50
LR=5e-5

# Function to submit a job
submit_job() {
    local NAME=$1
    local LLM=$2
    local REWARD=$3
    local DIARIES=$4
    local DISCOVERY=$5
    local PORT=$6

    local DIARY_FLAG=""
    if [ "$DIARIES" == "no" ]; then
        DIARY_FLAG="--no-diaries"
    fi

    # Create a custom slurm script for this ablation
    local SLURM_FILE="train_slurm_${NAME}.sh"
    
    cat > ${SLURM_FILE} <<EOF
#!/bin/bash
#SBATCH --job-name=bazaar-${NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=ailab
#SBATCH --output=logs/train_${NAME}_%j.log
#SBATCH --error=logs/train_${NAME}_%j.err

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
    ${DIARY_FLAG} \\
    --service vllm \\
    --log-dir logs/results_${NAME}

kill \$VLLM_PID
EOF

    sbatch ${SLURM_FILE}
}

# 1. Baseline (Gemma 3 4B, Profit, Diaries, Discovery 5)
submit_job "baseline" "${LLM_BASE}/gemma-3-4b-it-bnb-4bit" "PROFIT" "yes" 5 8001

# 2. Reward Ablation (Revenue)
submit_job "revenue" "${LLM_BASE}/gemma-3-4b-it-bnb-4bit" "REVENUE" "yes" 5 8002

# 3. Reasoning Ablation (No Diaries)
submit_job "nodiaries" "${LLM_BASE}/gemma-3-4b-it-bnb-4bit" "PROFIT" "no" 5 8003

# 4. Friction Ablation (No Friction)
submit_job "nofriction" "${LLM_BASE}/gemma-3-4b-it-bnb-4bit" "PROFIT" "yes" 0 8004

# 5. Model Comparison (Qwen 3)
submit_job "qwen3" "${LLM_BASE}/Qwen2.5-7B-Instruct-bnb-4bit" "PROFIT" "yes" 5 8005

# 6. Model Comparison (Ministral 3)
submit_job "ministral3" "${LLM_BASE}/mistral-7b-instruct-v0.3-bnb-4bit" "PROFIT" "yes" 5 8006

# 7. Model Comparison (OLMo 3)
submit_job "olmo3" "${LLM_BASE}/OLMo-2-1124-7B-Instruct-bnb-4bit" "PROFIT" "yes" 5 8007
