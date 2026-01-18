import requests
import json

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
REPO = "git@github.com:sethkarten/AI-Bazaar.git"
BRANCH = "Market-v0"

payload = {
    "repo": REPO,
    "branch": BRANCH,
    "script": "cluster_launcher.sh",
    "args": [
        "--llm",
        "/media/milkkarten/data/AI-Bazaar/models/gemma-3-4b-it-bnb-4bit",
        "--num_episodes",
        "20",
        "--num_iterations",
        "1",
        "--max-timesteps",
        "10",
        "--run_name",
        "sweep2-batching80ms",
        "--port",
        "0",
    ],
    "gpu_count": 1,
    "gpu_memory_min_gb": 32,
    "time_limit_hours": 1,
    "prefer_resource": "local-5090"
}

response = requests.post(GPU_MANAGER_URL, json=payload)
print(f"Status code: {response.status_code}")
print(f"Response text: {response.text}")
if response.ok:
    try:
        job_data = response.json()
        print(f"\n✅ Sweep 2 (batching) job submitted successfully!")
        print(f"Job ID: {job_data.get('id')}")
        print(f"\nExpected behavior:")
        print(f"  - Request queue fills with ~140 requests per phase")
        print(f"  - Batch sizes: 20-32 items (check logs)")
        print(f"  - GPU power: >250W (partial saturation)")
        print(f"  - Iteration 0 completes successfully")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
else:
    print(f"Error submitting job: {response.status_code}")
