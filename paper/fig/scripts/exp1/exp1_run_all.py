"""
Master script: runs all three Experiment 1 figure scripts.

Usage:
    python exp1_run_all.py [--logs-dir logs/] [--good food] [--fig-dir paper/fig/exp1/]
"""

import argparse
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "exp1_heatmap.py",
    "exp1_interaction.py",
    "exp1_timeseries.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run all Exp1 figure scripts")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food")
    parser.add_argument("--fig-dir", default=os.path.join(SCRIPTS_DIR, "..", "..", "exp1"))
    args = parser.parse_args()

    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    outputs = {
        "exp1_heatmap.py":    os.path.join(fig_dir, "exp1_heatmap.pdf"),
        "exp1_interaction.py": os.path.join(fig_dir, "exp1_interaction.pdf"),
        "exp1_timeseries.py": os.path.join(fig_dir, "exp1_timeseries.pdf"),
    }

    results = {}
    for script_name in SCRIPTS:
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        output_path = outputs[script_name]
        cmd = [
            sys.executable, script_path,
            "--logs-dir", args.logs_dir,
            "--good", args.good,
            "--output", output_path,
        ]
        print(f"\n{'='*60}")
        print(f"Running: {script_name}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, capture_output=False)
        success = result.returncode == 0
        results[script_name] = (success, output_path)
        if not success:
            print(f"  ERROR: {script_name} exited with code {result.returncode}")

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
