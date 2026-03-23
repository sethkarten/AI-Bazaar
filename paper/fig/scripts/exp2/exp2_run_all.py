"""
Master script: runs all Experiment 2 figure scripts.

Usage:
    python exp2_run_all.py [--logs-dir logs/] [--good food] [--fig-dir paper/fig/exp2/]
"""

import argparse
import os
import subprocess
import sys
import threading

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "exp2_sybil_detection.py",
    "exp2_lemon_volume.py",
    "exp2_lemon_reputation_quality.py",
    "exp2_lemon_consumer_welfare.py",
    "exp2_sybil_revenue_share.py",
    "exp2_market_collapse.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run all Exp2 figure scripts")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food")
    parser.add_argument("--fig-dir", default=os.path.join(SCRIPTS_DIR, "..", "..", "exp2"))
    parser.add_argument("--workers", type=int, default=8, help="Parallel load workers per script")
    parser.add_argument("--force", action="store_true", help="Ignore cache and rebuild from scratch")
    args = parser.parse_args()

    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    outputs = {
        "exp2_sybil_detection.py":          os.path.join(fig_dir, "exp2_sybil_detection.pdf"),
        "exp2_lemon_volume.py":             os.path.join(fig_dir, "exp2_lemon_volume.pdf"),
        "exp2_lemon_reputation_quality.py": os.path.join(fig_dir, "exp2_lemon_reputation_quality.pdf"),
        "exp2_lemon_consumer_welfare.py":   os.path.join(fig_dir, "exp2_lemon_consumer_welfare.pdf"),
        "exp2_sybil_revenue_share.py":      os.path.join(fig_dir, "exp2_sybil_revenue_share.pdf"),
        "exp2_market_collapse.py":          os.path.join(fig_dir, "exp2_market_collapse.pdf"),
    }

    # Launch all scripts in parallel
    procs = {}
    for script_name in SCRIPTS:
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        output_path = outputs[script_name]
        cmd = [
            sys.executable, script_path,
            "--logs-dir", args.logs_dir,
            "--good", args.good,
            "--output", output_path,
            "--workers", str(args.workers),
        ]
        if args.force:
            cmd.append("--force")
        print(f"Launching: {script_name}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs[script_name] = proc

    # Stream output from all procs with script name prefix
    def stream(name, proc):
        short = name.replace(".py", "")
        for line in proc.stdout:
            print(f"[{short}] {line}", end="", flush=True)
        proc.wait()

    threads = [threading.Thread(target=stream, args=(name, proc)) for name, proc in procs.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results = {name: (proc.returncode == 0, outputs[name]) for name, proc in procs.items()}

    print(f"\n{'='*60}")
    print("Summary:")
    all_ok = True
    for script_name, (success, output_path) in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {script_name} -> {output_path}")
        if not success:
            all_ok = False
    print(f"{'='*60}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
