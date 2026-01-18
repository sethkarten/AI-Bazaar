import requests
import json

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"

payload = {
    "repo": REPO,
    "branch": BRANCH,
    "script": "local_launcher.sh",  # Uses 'uv run'
    "args": [
        "--llm",
        "/media/milkkarten/data/AI-Bazaar/models/gemma-3-4b-it-bnb-4bit",
        "--num_episodes",
        "20",
        "--num_iterations",
        "5",
        "--format_reward_weight",
        "1.0",
        "--run_name",
        "sweep3-formatreward1",
        "--port",
        "0",
    ],
    "gpu_count": 1,
    "gpu_memory_min_gb": 32,
    "time_limit_hours": 2,
    "prefer_resource": "local-5090"
}

response = requests.post(GPU_MANAGER_URL, json=payload)
print(f"Status code: {response.status_code}")
print(f"Response text: {response.text}")
if response.ok:
    try:
        print(f"Job submitted: {response.json()}")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
else:
    print(f"Error submitting job: {response.status_code}")
