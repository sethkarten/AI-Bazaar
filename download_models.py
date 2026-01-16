#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import os

models = [
    "unsloth/gemma-3-4b-it-bnb-4bit",
    "unsloth/Qwen3-7B-Instruct-bnb-4bit",
    "unsloth/Ministral-3-8B-Instruct-bnb-4bit",
    "unsloth/OLMo-3-7B-Instruct-bnb-4bit",
]


base_dir = "models"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for m in models:
    name = m.split("/")[-1]
    target_dir = os.path.join(base_dir, name)
    if os.path.exists(target_dir):
        print(f"Model {name} already exists. Skipping.")
        continue

    print(f"Downloading {m}...")
    try:
        snapshot_download(repo_id=m, local_dir=target_dir)
        print(f"Successfully downloaded {name}.")
    except Exception as e:
        print(f"Failed to download {m}: {e}")
