"""
Master script: runs all Experiment 1 figure scripts.

Baseline (n_stab=0, dlc=3) resolves to exp1_{model}_stab_0_dlc3_seed{8|16|64}
per shared exp1_paths.resolve_run_dir; legacy single-dir exp1_{model}_baseline
(seed 8) is still accepted. Sweep runs use exp1_stab_{n_stab}_dlc{dlc}_seed{seed}.

Usage:
    python exp1_run_all.py --src <logs-subdir> [--dst <fig-subdir>] [--good food]
    python exp1_run_all.py [--logs-dir logs/] [--good food] [--fig-dir paper/fig/exp1/]

--src sets the subdirectory within logs/ to read from.
--dst sets the subdirectory within paper/fig/exp1/ to write to (defaults to --src name).
"""

import argparse
import os
import subprocess
import sys
import threading

from exp1_paths import baseline_run_dirs as _exp1_baseline_run_dirs
from exp1_paths import SEEDS as _EXP1_SEEDS

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "exp1_heatmap.py",
    "exp1_score.py",
    "exp1_timeseries.py",
    "exp1_survival.py",
    "exp1_phase.py",
    "exp1_collapse_timing.py",
    "exp1_tokens.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run all Exp1 figure scripts")
    parser.add_argument("--src", default=None,
                        help="Subdirectory within logs/ to read from (e.g. 'exp1_gemini')")
    parser.add_argument("--dst", default=None,
                        help="Subdirectory within paper/fig/exp1/ to write to; defaults to --src name")
    parser.add_argument("--logs-dir", default="logs/",
                        help="Base logs directory (overridden by --src if provided)")
    parser.add_argument("--good", default="food")
    parser.add_argument("--fig-dir", default=os.path.join(SCRIPTS_DIR, "..", "..", "exp1"),
                        help="Base fig directory (overridden by --dst/--src if provided)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel load workers per script")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    if args.src:
        logs_dir = os.path.join(args.logs_dir, args.src)
        if not args.model and args.src.startswith("exp1_"):
            args.model = args.src[len("exp1_"):]
    else:
        logs_dir = args.logs_dir

    dst_name = args.dst or args.src
    if dst_name:
        fig_dir = os.path.abspath(os.path.join(args.fig_dir, dst_name))
    else:
        fig_dir = os.path.abspath(args.fig_dir)

    os.makedirs(fig_dir, exist_ok=True)

    # Resolve baseline directories (dlc=3, n_stab=0) for each seed — matches figure scripts
    bl_paths = _exp1_baseline_run_dirs(logs_dir, args.model)
    bl_ok = sum(1 for p in bl_paths if p)
    bl_missing = [s for s, p in zip(_EXP1_SEEDS, bl_paths) if not p]
    if bl_missing:
        ex = f"exp1_{args.model}_" if args.model else "exp1_"
        print(
            f"[exp1_run_all] Warning: missing baseline dir(s) for seeds {bl_missing} under {logs_dir!r} "
            f"(expect {ex}stab_0_dlc3_seed<seed>, or legacy {ex}baseline for seed 8).",
            flush=True,
        )
    else:
        print(f"[exp1_run_all] Baseline: all {len(_EXP1_SEEDS)} seed dirs found under {logs_dir!r}.", flush=True)

    def pdf_name(stem):
        if args.model:
            return f"exp1_{args.model}_{stem}.pdf"
        return f"exp1_{stem}.pdf"

    outputs = {
        "exp1_heatmap.py":          os.path.join(fig_dir, pdf_name("heatmap")),
        "exp1_score.py":            os.path.join(fig_dir, pdf_name("score")),
        "exp1_score_3d.py":         os.path.join(fig_dir, pdf_name("score_3d")),
        "exp1_timeseries.py":       os.path.join(fig_dir, pdf_name("timeseries")),
        "exp1_survival.py":         os.path.join(fig_dir, pdf_name("survival")),
        "exp1_phase.py":            os.path.join(fig_dir, pdf_name("phase")),
        "exp1_collapse_timing.py":  os.path.join(fig_dir, pdf_name("collapse_timing")),
        "exp1_tokens.py":           os.path.join(fig_dir, pdf_name("tokens")),
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
            "--model", args.model,
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

    # ── Frontier health comparison (cross-model, reads cached data) ───────
    frontier_script = os.path.join(SCRIPTS_DIR, "exp1_frontier_health.py")
    fig_exp1_dir    = os.path.abspath(args.fig_dir)
    frontier_output = os.path.join(fig_exp1_dir, "exp1_frontier_health.pdf")
    frontier_cmd = [
        sys.executable, frontier_script,
        "--good", args.good,
        "--output", frontier_output,
        "--fig-exp1-dir", fig_exp1_dir,
        "--logs-dir", args.logs_dir,
    ]
    print(f"Launching: exp1_frontier_health.py", flush=True)
    frontier_proc = subprocess.Popen(
        frontier_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    frontier_thread = threading.Thread(
        target=stream, args=("exp1_frontier_health", frontier_proc))
    frontier_thread.start()
    frontier_thread.join()

    frontier_ok = frontier_proc.returncode == 0
    results["exp1_frontier_health.py"] = (frontier_ok, frontier_output)

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
