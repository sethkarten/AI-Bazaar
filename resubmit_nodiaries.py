#!/usr/bin/env python3
"""Resubmit nodiaries job with batching fixes."""
import requests
import json

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"
LLM_BASE = "/data3/milkkarten/AI-Bazaar/models"

config = {
    "repo": REPO,
    "branch": BRANCH,
    "script": "cluster_launcher.sh",
    "args": [
        "--llm", f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "--port", "0",
        "--num_episodes", "50",
        "--num_iterations", "50",
        "--reward-type", "PROFIT",
        "--discovery-limit-consumers", "5",
        "--run_name", "v3-ablation-nodiaries",
        "--log-dir", "logs/results_nodiaries",
        "--wandb_mode", "online",
        "--train_batch_size", "8",
        "--format_reward_weight", "1.0",
        "--no-diaries",
    ],
    "gpu_count": 1,
    "gpu_memory_min_gb": 48,
    "time_limit_hours": 24,
    "prefer_resource": "Pikachu",
}

print("Submitting nodiaries job to Pikachu...")
print(f"Config: {json.dumps(config, indent=2)}")

response = requests.post(GPU_MANAGER_URL, json=config)
if response.status_code == 200:
    result = response.json()
    print(f"\n✅ Job submitted successfully!")
    print(f"Job ID: {result.get('job_id', 'N/A')}")
    print(f"Resource: {result.get('resource', 'N/A')}")
    print(f"GPU: {result.get('gpu_ids', 'N/A')}")
else:
    print(f"\n❌ Job submission failed: {response.status_code}")
    print(f"Error: {response.text}")
