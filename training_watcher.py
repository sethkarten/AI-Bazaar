#!/usr/bin/env python3
import subprocess
import time
import re
from datetime import datetime

# Configuration
CLUSTER_HOST = "della-gpu"
PROJECT_DIR = "AI-Bazaar"
HEARTBEAT_FILE = f"{PROJECT_DIR}/train_heartbeat.txt"
CHECK_INTERVAL = 300  # 5 minutes
MAX_IDLE_TIME = 3600  # 1 hour


def run_ssh_cmd(cmd):
    try:
        result = subprocess.run(
            ["ssh", CLUSTER_HOST, f"cd {PROJECT_DIR} && {cmd}"],
            capture_all=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def check_training():
    print(f"[{datetime.now()}] Checking training status...")

    # Check SLURM queue
    squeue = run_ssh_cmd("squeue -u $USER | grep bazaar-t")
    if not squeue:
        print("🚨 Job not found in squeue!")
        # Check latest log for crash
        last_err = run_ssh_cmd("ls -t logs/*.err | head -n 1 | xargs cat | tail -n 10")
        print(f"Last error output:\n{last_err}")
        return False

    job_id = squeue.split()[0]
    print(f"✅ Job {job_id} is running.")

    # Check Heartbeat
    heartbeat = run_ssh_cmd(f"cat train_heartbeat.txt")
    try:
        last_ts = float(heartbeat)
        idle_sec = time.time() - last_ts
        print(
            f"💓 Last heartbeat: {datetime.fromtimestamp(last_ts)} ({idle_sec / 60:.2f}m ago)"
        )

        if idle_sec > MAX_IDLE_TIME:
            print(f"🚨 WARNING: Job {job_id} is idle for >1 hour! Possible hang.")
    except:
        print("⚠️  Could not read heartbeat timestamp.")

    # Check GPU utilization
    gpu_info = run_ssh_cmd(
        "squeue -j "
        + job_id
        + " -o %N -h | xargs -I {} ssh {} nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
    )
    print(f"📊 GPU Util: {gpu_info}")

    return True


if __name__ == "__main__":
    while True:
        check_training()
        time.sleep(CHECK_INTERVAL)
