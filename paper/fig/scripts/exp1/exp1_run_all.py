"""
Master script: runs all three Experiment 1 figure scripts.

The figure scripts (heatmap, timeseries) use exp1_baseline for the no-stab baseline;
sweep runs use exp1_stab_{n_stab}_dlc{dlc}_seed{seed} (including n_stab=5).

Usage:
    python exp1_run_all.py [--logs-dir logs/] [--good food] [--fig-dir paper/fig/exp1/]
"""

import argparse
import os
import subprocess
import sys
import threading

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "exp1_heatmap.py",
    "exp1_interaction.py",
    "exp1_timeseries.py",
    "exp1_survival.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run all Exp1 figure scripts")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument("--good", default="food")
    parser.add_argument("--fig-dir", default=os.path.join(SCRIPTS_DIR, "..", "..", "exp1"))
    parser.add_argument("--workers", type=int, default=8, help="Parallel load workers per script")
    args = parser.parse_args()

    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    outputs = {
        "exp1_heatmap.py":    os.path.join(fig_dir, "exp1_heatmap.pdf"),
        "exp1_interaction.py": os.path.join(fig_dir, "exp1_interaction.pdf"),
        "exp1_timeseries.py": os.path.join(fig_dir, "exp1_timeseries.pdf"),
        "exp1_survival.py":   os.path.join(fig_dir, "exp1_survival.pdf"),
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
