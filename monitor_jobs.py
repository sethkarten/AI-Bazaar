#!/usr/bin/env python3
"""Continuous job monitoring with alerts for failures."""
import requests
import time
import subprocess
import json
from datetime import datetime

GPU_MANAGER_URL = "http://localhost:8080/api/jobs"
CHECK_INTERVAL = 300  # 5 minutes

EXPECTED_JOBS = [
    "v3-ablation-nofriction",
    "v3-ablation-asymmetry",
    "v3-ablation-baseline",
    "v3-ablation-revenue",
    "v3-ablation-final-gemma",
    "v3-ablation-qwen3",
    "v3-ablation-olmo3",
    "v3-ablation-ministral3",
]

def get_jobs_status():
    """Get status of all Pikachu jobs."""
    resp = requests.get(GPU_MANAGER_URL)
    if resp.status_code != 200:
        return None

    jobs = resp.json()
    pikachu_jobs = [
        j for j in jobs
        if j.get("resource") == "Pikachu"
        and j.get("created_at", "").startswith("2026-01-18T16:")
    ]
    return pikachu_jobs

def get_latest_timestep(variant):
    """Get latest timestep from state files on Pikachu."""
    cmd = f"""ssh Pikachu "cd /data3/milkkarten/AI-Bazaar/logs/results_{variant} && ls -t state_t*.json 2>/dev/null | head -1 | sed 's/.*state_t//' | sed 's/.json//'"  """
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except:
        pass
    return "N/A"

def check_vllm_errors(variant):
    """Check for VLLM errors in logs."""
    cmd = f"""ssh Pikachu "cd /data3/milkkarten/AI-Bazaar/logs/results_{variant} && tail -20 vllm*.log 2>/dev/null | grep -E 'ERROR|ValueError|No available memory' | head -3" """
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except:
        pass
    return None

def main():
    """Main monitoring loop."""
    print("=" * 60)
    print("ABLATION MATRIX JOB MONITOR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Check interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL // 60} minutes)")
    print("=" * 60)
    print()

    iteration = 0
    last_progress = {}

    while True:
        iteration += 1
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Check #{iteration}")
        print("-" * 60)

        jobs = get_jobs_status()
        if jobs is None:
            print("❌ Failed to fetch job status from GPU Manager")
            time.sleep(CHECK_INTERVAL)
            continue

        # Status summary
        running = [j for j in jobs if j.get("status") == "running"]
        completed = [j for j in jobs if j.get("status") == "completed"]
        failed = [j for j in jobs if j.get("status") == "failed"]

        print(f"Status: {len(running)} running, {len(completed)} completed, {len(failed)} failed")
        print()

        # Check each expected job
        alerts = []
        for variant_name in EXPECTED_JOBS:
            job = next((j for j in jobs if j.get("args", [None]*14)[13] == variant_name), None)

            if not job:
                alerts.append(f"⚠️  {variant_name}: NOT FOUND")
                continue

            status = job.get("status", "unknown")
            gpu = job.get("gpu_ids", "N/A")

            if status == "running":
                # Get latest progress
                variant_short = variant_name.replace("v3-ablation-", "")
                timestep = get_latest_timestep(variant_short)

                # Check for progress stalls
                if timestep != "N/A":
                    if variant_name in last_progress:
                        if last_progress[variant_name] == timestep:
                            alerts.append(f"⚠️  {variant_short}: STALLED at t{timestep} (no progress)")
                        else:
                            print(f"  ✅ {variant_short:15s} GPU {gpu} @ t{timestep}")
                    else:
                        print(f"  ✅ {variant_short:15s} GPU {gpu} @ t{timestep}")

                    last_progress[variant_name] = timestep
                else:
                    print(f"  🔄 {variant_short:15s} GPU {gpu} (initializing)")

                # Check for VLLM errors
                vllm_error = check_vllm_errors(variant_short)
                if vllm_error:
                    alerts.append(f"🚨 {variant_short}: VLLM ERROR detected!")

            elif status == "completed":
                variant_short = variant_name.replace("v3-ablation-", "")
                timestep = get_latest_timestep(variant_short)

                # Check if completed too early
                if timestep != "N/A":
                    try:
                        t_num = int(timestep.replace("t", ""))
                        if t_num < 1000:  # Should have ~5000 files for 50 iters * 100 timesteps
                            alerts.append(f"⚠️  {variant_short}: Completed EARLY (only t{t_num} files)")
                        else:
                            print(f"  ✅ {variant_short:15s} COMPLETED (t{timestep})")
                    except:
                        pass

            elif status == "failed":
                variant_short = variant_name.replace("v3-ablation-", "")
                error = job.get("error", "No error message")
                alerts.append(f"❌ {variant_short}: FAILED - {error[:60]}")

        # Print alerts
        if alerts:
            print()
            print("🚨 ALERTS:")
            for alert in alerts:
                print(f"  {alert}")

        # GPU utilization check
        try:
            cmd = """ssh Pikachu "nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits" """
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print()
                print("GPU Utilization:")
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) == 4:
                        gpu_id, mem, util, power = [p.strip() for p in parts]
                        print(f"  GPU {gpu_id}: {mem:>6s} MB ({util:>3s}% util, {power:>6s}W)")
        except:
            pass

        print(f"\nNext check in {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
