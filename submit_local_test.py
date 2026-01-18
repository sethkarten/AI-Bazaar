import requests
import json
import time

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"


def submit_local_test(name, reward):
    # Use a small model and small settings for quick local test
    # local-5090 has 32GB VRAM per card.
    ts = int(time.time())
    payload = {
        "repo": REPO,
        "branch": BRANCH,
        "script": "local_launcher.sh",
        "args": [
            "--llm",
            "/media/milkkarten/data/AI-Bazaar/models/gemma-3-4b-it-bnb-4bit",
            "--port",
            "0",
            "--num_episodes",
            "1",
            "--num_iterations",
            "1",
            "--reward-type",
            reward,
            "--discovery-limit",
            "5",
            "--run_name",
            f"local-{name}-{ts}",
            "--log-dir",
            f"logs/local_{name}_{ts}",
            "--train_batch_size",
            "16",
        ],
        "gpu_count": 1,
        "gpu_memory_min_gb": 24,  # Fits in 5090
        "time_limit_hours": 1,
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
