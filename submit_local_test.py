import requests
import json

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"


def submit_local_test():
    # Use a small model and small settings for quick local test
    # local-5090 has 32GB VRAM per card.
    payload = {
        "repo": REPO,
        "branch": BRANCH,
        "script": "ai_bazaar/train/train_reinforce.py",
        "args": [
            "--llm",
            "./models/gemma-3-4b-it-bnb-4bit",
            "--port",
            "8201",
            "--num_episodes",
            "1",
            "--num_iterations",
            "1",
            "--reward-type",
            "PROFIT",
            "--discovery-limit",
            "5",
            "--run_name",
            "local-test",
            "--log-dir",
            "logs/local_test",
        ],
        "gpu_count": 2,
        "gpu_memory_min_gb": 24,  # Per GPU
        "time_limit_hours": 1,
        "prefer_resource": "local-5090",
    }

    print(f"Submitting local test job to 5090...")
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
    submit_local_test()
