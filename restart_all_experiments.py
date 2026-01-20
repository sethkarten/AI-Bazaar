#!/usr/bin/env python3
"""Restart all 8 experiments on Pikachu with fixes and comprehensive logging."""
import requests
import json
import time

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"
LLM_BASE = "/data3/milkkarten/AI-Bazaar/models"

# All 8 variants (ministral3 excluded - not supported)
EXPERIMENTS = [
    {
        "name": "nodiaries",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "discovery": 5,
        "no_diaries": True,
        "asymmetry": False,
        "batch_size": 8,
        "gpu_memory": 48,
    },
    {
        "name": "nofriction",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "discovery": 0,
        "no_diaries": False,
        "asymmetry": False,
        "batch_size": 8,
        "gpu_memory": 48,
    },
    {
        "name": "asymmetry",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "discovery": 5,
        "no_diaries": False,
        "asymmetry": True,
        "batch_size": 8,
        "gpu_memory": 48,
    },
    {
        "name": "baseline",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "discovery": 5,
        "no_diaries": False,
        "asymmetry": False,
        "batch_size": 8,
        "gpu_memory": 48,
    },
    {
        "name": "revenue",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "REVENUE",
        "discovery": 5,
        "no_diaries": False,
        "asymmetry": False,
        "batch_size": 8,
        "gpu_memory": 48,
    },
    {
        "name": "final-gemma",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "discovery": 5,
        "no_diaries": False,
        "asymmetry": True,
        "batch_size": 8,
        "gpu_memory": 48,
    },
    {
        "name": "qwen3",
        "llm": f"{LLM_BASE}/Qwen3-8B-unsloth-bnb-4bit",
        "reward": "PROFIT",
        "discovery": 5,
        "no_diaries": False,
        "asymmetry": True,
        "batch_size": 4,  # Larger model, smaller batch
        "gpu_memory": 48,
    },
    {
        "name": "olmo3",
        "llm": f"{LLM_BASE}/Olmo-3-7B-Think-unsloth-bnb-4bit",
        "reward": "PROFIT",
        "discovery": 5,
        "no_diaries": False,
        "asymmetry": True,
        "batch_size": 4,  # Larger model, smaller batch
        "gpu_memory": 48,
    },
]

def submit_experiment(exp):
    """Submit a single experiment to GPU Manager."""
    args = [
        "--llm", exp["llm"],
        "--port", "0",
        "--num_episodes", "50",
        "--num_iterations", "50",
        "--reward-type", exp["reward"],
        "--discovery-limit", str(exp["discovery"]),
        "--run_name", f"v3-ablation-{exp['name']}",
        "--log-dir", f"logs/results_{exp['name']}",
        "--wandb_mode", "online",
        "--train_batch_size", str(exp["batch_size"]),
        "--format_reward_weight", "1.0",
    ]

    if exp["no_diaries"]:
        args.append("--no-diaries")

    if exp["asymmetry"]:
        args.append("--info-asymmetry")

    config = {
        "repo": REPO,
        "branch": BRANCH,
        "script": "cluster_launcher.sh",
        "args": args,
        "gpu_count": 1,
        "gpu_memory_min_gb": exp["gpu_memory"],
        "time_limit_hours": 24,
        "prefer_resource": "Pikachu",
    }

    print(f"\n{'='*80}")
    print(f"Submitting: {exp['name']}")
    print(f"  Model: {exp['llm'].split('/')[-1]}")
    print(f"  Reward: {exp['reward']}, Discovery: {exp['discovery']}")
    print(f"  No diaries: {exp['no_diaries']}, Asymmetry: {exp['asymmetry']}")
    print(f"  Batch size: {exp['batch_size']}, GPU memory: {exp['gpu_memory']}GB")
    print(f"{'='*80}")

    response = requests.post(GPU_MANAGER_URL, json=config)

    if response.status_code in [200, 201]:
        result = response.json()
        job_id = result.get('id', 'N/A')
        status = result.get('status', 'unknown')
        gpu = result.get('gpu_ids', 'pending')
        print(f"✅ SUCCESS: Job {job_id} - Status: {status}, GPU: {gpu}")
        return job_id
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text[:200]}")
        return None

def main():
    print("\n" + "="*80)
    print("RESTARTING ALL EXPERIMENTS WITH FIXES")
    print("="*80)
    print(f"\nBranch: {BRANCH}")
    print(f"Latest commit: 524d63b (logging improvements + all fixes)")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Resource: Pikachu (8× A6000 48GB)")
    print(f"\nFixes included:")
    print("  ✅ Filter None from full_texts before tokenization")
    print("  ✅ Defensive checks for None in tokenizer output")
    print("  ✅ Per-job stdout/stderr logging")
    print("  ✅ job_info.json with job ID and metadata")
    print("  ✅ Comprehensive WandB logging (20+ metrics)")
    print("  ✅ Detailed console output with timings")

    input("\nPress ENTER to start submitting jobs (or Ctrl+C to cancel)...")

    job_ids = []

    for i, exp in enumerate(EXPERIMENTS):
        job_id = submit_experiment(exp)
        if job_id:
            job_ids.append((exp['name'], job_id))

        # Stagger submissions to ensure proper GPU assignment
        if i < len(EXPERIMENTS) - 1:
            print(f"\nWaiting 10 seconds before next submission...")
            time.sleep(10)

    print("\n" + "="*80)
    print("SUBMISSION COMPLETE")
    print("="*80)
    print(f"\nSubmitted {len(job_ids)}/{len(EXPERIMENTS)} jobs successfully:")
    for name, job_id in job_ids:
        print(f"  {name:20s} → {job_id}")

    if len(job_ids) < len(EXPERIMENTS):
        print(f"\n⚠️  Warning: {len(EXPERIMENTS) - len(job_ids)} jobs failed to submit")

    print("\n" + "="*80)
    print("MONITORING")
    print("="*80)
    print("\nCheck job status:")
    print("  curl -s http://localhost:8080/api/jobs | jq '.[] | select(.status==\"running\") | {run_name: .args[13], gpu: .gpu_ids}'")
    print("\nCheck logs:")
    print("  ssh Pikachu 'tail -f /data3/milkkarten/AI-Bazaar/logs/results_baseline/stdout.log'")
    print("\nCheck GPU utilization:")
    print("  ssh Pikachu 'nvidia-smi'")
    print("\nCheck WandB:")
    print("  https://wandb.ai/princeton-ai/ai-bazaar")

if __name__ == "__main__":
    main()
