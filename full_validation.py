import requests
import json

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"

payload = {
    "repo": REPO,
    "branch": BRANCH,
    "script": "cluster_launcher.sh",  # Uses .venv
    "args": [
        "--llm",
        "/media/milkkarten/data/AI-Bazaar/models/gemma-3-4b-it-bnb-4bit",
        "--num_episodes",
        "50",
        "--num_iterations",
        "50",
        "--format_reward_weight",
        "1.0",
        "--train_batch_size",
        "8",
        "--run_name",
        "local-validation-baseline",
        "--port",
        "0",
    ],
    "gpu_count": 1,
    "gpu_memory_min_gb": 32,
    "time_limit_hours": 24,
    "prefer_resource": "local-5090"
}

response = requests.post(GPU_MANAGER_URL, json=payload)
print(f"Status code: {response.status_code}")
print(f"Response text: {response.text}")
if response.ok:
    try:
        job_data = response.json()
        print(f"\n✅ Full validation job submitted successfully!")
        print(f"Job ID: {job_data.get('id')}")
        print(f"Expected duration: 15-20 hours (20 min/iteration × 50 iterations)")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
else:
    print(f"Error submitting job: {response.status_code}")
