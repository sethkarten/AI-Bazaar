#!/bin/bash
#SBATCH --job-name=rl-v5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=ailab
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# Offline modes
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export UV_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/gpfs/CHIJ/milkkarten/AI-Bazaar
source .venv/bin/activate
mkdir -p logs/reinforce

# REINFORCE++ stabilizing firm: 4-bit QLoRA, dual-GPU, gradient accumulation
# GPU 0: training + stabilizing firm inference
# GPU 1: frozen base model for non-stabilizing firms + KL reference
#
# v5 changes: removed dynamic batch doubling (OOM fix), explicit OOM recovery,
# micro_bs=8 default, SFT warmup before RL
python -m ai_bazaar.train.train_reinforce \
    --llm "./models/Qwen3.5-9B" \
    --num_episodes ${NUM_EPISODES:-32} \
    --num_iterations ${NUM_ITERATIONS:-100} \
    --lr ${LR:-5e-6} \
    --num-firms 5 \
    --num-consumers 50 \
    --num-stabilizing-firms 1 \
    --firm-personas "${FIRM_PERSONAS:-competitive:2,reactive:1,volume_seeker:1}" \
    --max-timesteps ${MAX_TIMESTEPS:-32} \
    --max-tokens 256 \
    --firm-type LLM \
    --consumer-type CES \
    --consumer-scenario THE_CRASH \
    --discovery-limit-consumers ${DLC:-3} \
    --overhead-costs 14 \
    --firm-initial-cash 1000 \
    --num-goods 1 \
    --use-cost-pref-gen \
    --timeout 3 \
    --advantage_clip 3.0 \
    --grad_clip_norm 0.5 \
    --reward_weights "0.4,0.3,0.3" \
    --survival_bonus 5.0 \
    --format_reward_weight 2.0 \
    --sft_warmup ${SFT_WARMUP:-500} \
    --sft_epochs ${SFT_EPOCHS:-5} \
    --train_batch_size ${TRAIN_BS:-32} \
    --micro_batch_size ${MICRO_BS:-8} \
    --inference_batch_size 64 \
    --kl_coeff 0.2 \
    --lora_r ${LORA_R:-16} \
    --quant_bits ${QUANT_BITS:-4} \
    --run_name "${RUN_NAME:-della_v5}" \
    --seed ${SEED:-42} \
    --log-dir logs/reinforce
