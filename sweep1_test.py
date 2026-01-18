import requests
import json

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"

payload = {
    "repo": REPO,
    "branch": BRANCH,
    "script": "cluster_launcher.sh",  # Uses 'uv run'
    "args": "--llm /media/milkkarten/data/AI-Bazaar/models/gemma-3-4b-it-bnb-4bit --num_episodes 20 --num_iterations 1 --max-timesteps 10 --run_name sweep1-episodes20",
    "gpu_count": 1,
    "gpu_memory_min_gb": 32,
    "time_limit_hours": 1,
    "prefer_resource": "local-5090"
}

response = requests.post(GPU_MANAGER_URL, json=payload)
print(f"Job submitted: {response.json()}")
