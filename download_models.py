#!/usr/bin/env python3
import os
import subprocess

models = [
    "unsloth/gemma-3-4b-it-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Olmo-3-7B-Think-unsloth-bnb-4bit",
    "unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit",
]

base_dir = "models"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for m in models:
    name = m.split("/")[-1]
    target_dir = os.path.join(base_dir, name)

    # Check for actual model files
    if os.path.exists(target_dir) and any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir(target_dir)
        if not os.path.isdir(os.path.join(target_dir, f))
    ):
        print(f"Model {name} already exists. Skipping.")
        continue

    print(f"Downloading {m} to {target_dir}...")
    # Use huggingface-cli for better reliability and progress tracking
    cmd = [
        "huggingface-cli",
        "download",
        m,
        "--local-dir",
        target_dir,
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully downloaded {name}.")
    except Exception as e:
        print(f"Failed to download {m}: {e}")
