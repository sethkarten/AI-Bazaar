import requests
import json
import time

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"


def submit_local_test(name, reward):
    # Production version with optimized throughput parameters
    # local-5090 has 32GB VRAM per card.
    ts = int(time.time())
    payload = {
        "repo": REPO,
        "branch": BRANCH,
        "script": "cluster_launcher.sh",
        "args": [
            "--llm",
            "/media/milkkarten/data/AI-Bazaar/models/gemma-3-4b-it-bnb-4bit",
            "--port",
            "0",
            "--num_episodes",
            "50",
            "--num_iterations",
            "50",
            "--reward-type",
            reward,
            "--discovery-limit-consumers",
            "5",
            "--run_name",
            f"local-{name}-{ts}",
            "--log-dir",
            f"logs/local_{name}_{ts}",
            "--train_batch_size",
            "8",
            "--format_reward_weight",
            "1.0",
        ],
        "gpu_count": 1,
        "gpu_memory_min_gb": 32,
        "time_limit_hours": 24,
        "prefer_resource": "local-5090",
    }

    print(f"Submitting local job: {name}...")
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
    submit_local_test("baseline", "PROFIT")
    time.sleep(10)
    submit_local_test("revenue", "REVENUE")
