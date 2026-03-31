"""
exp3_run_all.py — run all Experiment 3 analysis scripts.

Usage
-----
  python exp3_run_all.py --src exp3_gemini-3-flash-preview
  python exp3_run_all.py --src exp3_gemini-3-flash-preview --dst my_run
  python exp3_run_all.py --logs-dir logs/exp3_gemini-3-flash-preview/ --fig-dir paper/fig/exp3/

--src sets the subdirectory within logs/ to read from.
--dst sets the subdirectory within paper/fig/exp3/ to write to (defaults to --src name).

Crash scripts (exp3a):
  exp3_crash_heatmap.py
  exp3_crash_timeseries.py
  exp3_crash_recovery.py

Lemon scripts (exp3b):
  exp3_lemon_recovery.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _run(script, extra_args):
    cmd = [sys.executable, str(_SCRIPTS_DIR / script)] + extra_args
    print(f"\n=== Running {script} ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {script} exited with code {result.returncode}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run all Exp3 figure scripts")
    parser.add_argument("--src", default=None,
                        help="Subdirectory within logs/ to read from (e.g. 'exp3_gemini-3-flash-preview')")
    parser.add_argument("--dst", default=None,
                        help="Subdirectory within paper/fig/exp3/ to write to; defaults to --src name")
    parser.add_argument("--logs-dir", default="logs/",
                        help="Base logs directory (overridden by --src if provided)")
    parser.add_argument("--fig-dir", default=str(_SCRIPTS_DIR.parents[1] / "exp3"),
                        help="Base output directory for figures")
    parser.add_argument("--model", default="",
                        help="Model tag forwarded to sub-scripts via --model")
    parser.add_argument("--good", default="food")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.src:
        logs_dir = os.path.join(args.logs_dir, args.src)
        if not args.model and args.src.startswith("exp3_"):
            args.model = args.src[len("exp3_"):]
    else:
        logs_dir = args.logs_dir

    dst_name = args.dst or args.src
    if dst_name:
        fig_dir = os.path.abspath(os.path.join(args.fig_dir, dst_name))
    else:
        fig_dir = os.path.abspath(args.fig_dir)

    base_fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    common_kwargs = []
    if args.good:
        common_kwargs += ["--good", args.good]
    if args.workers:
        common_kwargs += ["--workers", str(args.workers)]
    if args.model:
        common_kwargs += ["--model", args.model]

    crash_kwargs = ["--logs-dir", logs_dir] + common_kwargs

    _run("exp3_crash_heatmap.py",
         crash_kwargs + ["--output", os.path.join(fig_dir, "exp3_crash_heatmap.pdf")])

    _run("exp3_crash_timeseries.py",
         crash_kwargs + ["--output", os.path.join(fig_dir, "exp3_crash_timeseries.pdf")])

    _run("exp3_crash_recovery.py",
         ["--logs-dir", logs_dir, "--fig-dir", base_fig_dir])

    _run("exp3_lemon_recovery.py",
         ["--logs-dir", logs_dir, "--fig-dir", base_fig_dir])


if __name__ == "__main__":
    main()
