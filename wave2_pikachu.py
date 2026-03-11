#!/usr/bin/env python3
"""Submit Wave 2 to Pikachu: 6 remaining variants (baseline + large models)."""
import requests
import time

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"
LLM_BASE = "/data3/milkkarten/AI-Bazaar/models"

# Wave 2: Baseline comparisons + large models
WAVE2_CONFIGS = [
    {
        "name": "baseline",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "no_diaries": False,
        "discovery": 5,
        "asymmetry": False,
        "batch_size": 8,
    },
    {
        "name": "revenue",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "REVENUE",
        "no_diaries": False,
        "discovery": 5,
        "asymmetry": False,
        "batch_size": 8,
    },
    {
        "name": "final-gemma",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "no_diaries": False,
        "discovery": 5,
        "asymmetry": True,
        "batch_size": 8,
    },
    {
        "name": "qwen3",
        "llm": f"{LLM_BASE}/Qwen3-8B-unsloth-bnb-4bit",
        "reward": "PROFIT",
        "no_diaries": False,
        "discovery": 5,
        "asymmetry": True,
        "batch_size": 4,  # Reduced for larger model
    },
    {
        "name": "olmo3",
        "llm": f"{LLM_BASE}/Olmo-3-7B-Think-unsloth-bnb-4bit",
        "reward": "PROFIT",
        "no_diaries": False,
        "discovery": 5,
        "asymmetry": True,
        "batch_size": 4,  # Reduced for larger model
    },
    {
        "name": "ministral3",
        "llm": f"{LLM_BASE}/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit",
        "reward": "PROFIT",
        "no_diaries": False,
        "discovery": 5,
        "asymmetry": True,
        "batch_size": 4,  # Reduced for larger model
    },
]

def submit_job(config):
    args = [
        "--llm", config["llm"],
        "--port", "0",
        "--num_episodes", "50",
        "--num_iterations", "50",
        "--reward-type", config["reward"],
        "--discovery-limit-consumers", str(config["discovery"]),
        "--run_name", f"v3-ablation-{config['name']}",
        "--log-dir", f"logs/results_{config['name']}",
        "--wandb_mode", "online",
        "--train_batch_size", str(config["batch_size"]),
        "--format_reward_weight", "1.0",
    ]

    if config.get("no_diaries"):
        args.append("--no-diaries")
    if config.get("asymmetry"):
        args.append("--info-asymmetry")

    payload = {
        "repo": REPO,
        "branch": BRANCH,
        "script": "cluster_launcher.sh",
        "args": args,
        "gpu_count": 1,
        "gpu_memory_min_gb": 40,  # Ensure exclusive GPU
        "time_limit_hours": 48,
        "prefer_resource": "Pikachu",
    }

    print(f"Submitting {config['name']}...", flush=True)
    try:
        resp = requests.post(GPU_MANAGER_URL, json=payload)
        if resp.status_code == 201:
            job_id = resp.json().get("id")
            print(f"  ✅ Success! Job ID: {job_id}")
            return job_id
        else:
            print(f"  ❌ Failed: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("=== WAVE 2: Submitting 6 remaining variants to Pikachu ===")
    print(f"Expected duration: ~7-8 hours per variant (50 iterations)")
    print(f"Larger models (8B) may take slightly longer")
    print()

    job_ids = []
    for config in WAVE2_CONFIGS:
        jid = submit_job(config)
        if jid:
            job_ids.append((config['name'], jid))
        time.sleep(90)  # 90s stagger delay

    print()
    print("=== Wave 2 Submitted ===")
    for name, jid in job_ids:
        print(f"  {name}: {jid}")
    print()
    print("Monitor with: curl -s http://localhost:8080/api/jobs | jq '.[] | select(.resource==\"Pikachu\")'")
