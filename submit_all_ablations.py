import requests
import json
import time

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"

# Configuration matrix
LLM_BASE = "./models"
ABLATIONS = [
    {
        "name": "baseline",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "diaries": True,
        "discovery": 5,
        "asymmetry": False,
        "port": 8101,
    },
    {
        "name": "revenue",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "REVENUE",
        "diaries": True,
        "discovery": 5,
        "asymmetry": False,
        "port": 8102,
    },
    {
        "name": "nodiaries",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "diaries": False,
        "discovery": 5,
        "asymmetry": False,
        "port": 8103,
    },
    {
        "name": "nofriction",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "diaries": True,
        "discovery": 0,
        "asymmetry": False,
        "port": 8104,
    },
    {
        "name": "asymmetry",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "diaries": True,
        "discovery": 5,
        "asymmetry": True,
        "port": 8105,
    },
    {
        "name": "final-gemma",
        "llm": f"{LLM_BASE}/gemma-3-4b-it-bnb-4bit",
        "reward": "PROFIT",
        "diaries": True,
        "discovery": 5,
        "asymmetry": True,
        "port": 8106,
    },
    {
        "name": "qwen3",
        "llm": f"{LLM_BASE}/Qwen3-8B-unsloth-bnb-4bit",
        "reward": "PROFIT",
        "diaries": True,
        "discovery": 5,
        "asymmetry": True,
        "port": 8107,
    },
    {
        "name": "ministral3",
        "llm": f"{LLM_BASE}/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit",
        "reward": "PROFIT",
        "diaries": True,
        "discovery": 5,
        "asymmetry": True,
        "port": 8108,
    },
    {
        "name": "olmo3",
        "llm": f"{LLM_BASE}/Olmo-3-7B-Think-unsloth-bnb-4bit",
        "reward": "PROFIT",
        "diaries": True,
        "discovery": 5,
        "asymmetry": True,
        "port": 8109,
    },
]


def submit_job(config, test_mode=True):
    args = [
        "--llm",
        config["llm"],
        "--port",
        str(config["port"]),
        "--num_episodes",
        "2" if test_mode else "5",
        "--num_iterations",
        "2" if test_mode else "50",
        "--reward-type",
        config["reward"],
        "--discovery-limit",
        str(config["discovery"]),
        "--run_name",
        f"ablation-{config['name']}",
        "--log-dir",
        f"logs/results_{config['name']}",
    ]

    if not config["diaries"]:
        args.append("--no-diaries")
    if config["asymmetry"]:
        args.append("--info-asymmetry")

    payload = {
        "repo": REPO,
        "branch": BRANCH,
        "script": "cluster_launcher.sh",  # Use bash wrapper to avoid automatic uv sync
        "args": args,
        "gpu_count": 1,
        "gpu_memory_min_gb": 40,
        "time_limit_hours": 1 if test_mode else 48,
        "prefer_resource": "Pikachu",
    }

    print(f"Submitting job: {config['name']} (Test: {test_mode})")
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production", action="store_true", help="Launch full 48h runs"
    )
    args = parser.parse_args()

    job_ids = []
    for config in ABLATIONS:
        jid = submit_job(config, test_mode=not args.production)
        if jid:
            job_ids.append(jid)
        time.sleep(2)  # Avoid hitting rate limits if any

    print(f"\nAll jobs submitted. Active Job IDs: {job_ids}")
