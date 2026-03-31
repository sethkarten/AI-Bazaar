"""
Master script: runs all Experiment 2 figure scripts.

Usage:
    python exp2_run_all.py --src <logs-subdir> [--dst <fig-subdir>] [--good car]
    python exp2_run_all.py [--logs-dir logs/] [--good car] [--fig-dir paper/fig/exp2/]

--src sets the subdirectory within logs/ to read from.
--dst sets the subdirectory within paper/fig/exp2/ to write to (defaults to --src name).
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
    "exp2_tokens.py",
    "exp2_heatmap.py",
    "exp2_health_consumer_welfare.py",
    "exp2_score.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run all Exp2 figure scripts")
    parser.add_argument("--src", default=None,
                        help="Subdirectory within logs/ to read from (e.g. 'exp2_gemini-2.5-flash')")
    parser.add_argument("--dst", default=None,
                        help="Subdirectory within paper/fig/exp2/ to write to; defaults to --src name")
    parser.add_argument("--logs-dir", default="logs/",
                        help="Base logs directory (overridden by --src if provided)")
    parser.add_argument("--good", default="car")
    parser.add_argument("--fig-dir", default=os.path.join(SCRIPTS_DIR, "..", "..", "exp2"),
                        help="Base fig directory (overridden by --dst/--src if provided)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel load workers per script")
    parser.add_argument("--force", action="store_true", help="Ignore cache and rebuild from scratch")
    args = parser.parse_args()

    if args.src:
        logs_dir = os.path.join(args.logs_dir, args.src)
    else:
        logs_dir = args.logs_dir

    dst_name = args.dst or args.src
    if dst_name:
        fig_dir = os.path.abspath(os.path.join(args.fig_dir, dst_name))
    else:
        fig_dir = os.path.abspath(args.fig_dir)

    os.makedirs(fig_dir, exist_ok=True)

    outputs = {
        "exp2_sybil_detection.py":          os.path.join(fig_dir, "exp2_sybil_detection.pdf"),
        "exp2_lemon_volume.py":             os.path.join(fig_dir, "exp2_lemon_volume.pdf"),
        "exp2_lemon_reputation_quality.py": os.path.join(fig_dir, "exp2_lemon_reputation_quality.pdf"),
        "exp2_lemon_consumer_welfare.py":   os.path.join(fig_dir, "exp2_lemon_consumer_welfare.pdf"),
        "exp2_sybil_revenue_share.py":      os.path.join(fig_dir, "exp2_sybil_revenue_share.pdf"),
        "exp2_market_collapse.py":          os.path.join(fig_dir, "exp2_market_collapse.pdf"),
        "exp2_tokens.py":                   os.path.join(fig_dir, "exp2_tokens.pdf"),
        "exp2_heatmap.py":                  os.path.join(fig_dir, "exp2_heatmap.pdf"),
        "exp2_health_consumer_welfare.py":  os.path.join(fig_dir, "exp2_health_consumer_welfare.pdf"),
        "exp2_score.py":                    os.path.join(fig_dir, "exp2_score.pdf"),
    }

    # Launch all scripts in parallel
    procs = {}
    for script_name in SCRIPTS:
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        output_path = outputs[script_name]
        cmd = [
            sys.executable, script_path,
            "--logs-dir", logs_dir,
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
