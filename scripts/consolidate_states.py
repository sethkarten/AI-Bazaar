#!/usr/bin/env python3
"""
Consolidate legacy per-timestep state files into a single states.json.

Scans a run directory (or a parent directory of run directories) for
state_t*.json files, sorts them by timestep, and writes states.json.

Usage:
    # Single run directory:
    python scripts/consolidate_states.py logs/exp1_gemini_stab_2_dlc3_seed0

    # Entire experiment directory (recurses into all run subdirs):
    python scripts/consolidate_states.py logs/exp1_gemini --recursive

    # Preview without writing:
    python scripts/consolidate_states.py logs/exp1_gemini --recursive --dry-run

    # Delete per-timestep files after consolidating:
    python scripts/consolidate_states.py logs/exp1_gemini --recursive --delete
"""

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_run_dirs(root: Path, recursive: bool) -> list[Path]:
    """Return directories that contain state_t*.json files."""
    if not recursive:
        return [root]
    dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.startswith("state_t") and f.endswith(".json") for f in filenames):
            dirs.append(Path(dirpath))
    return sorted(dirs)


def consolidate(run_dir: Path, dry_run: bool, delete: bool) -> bool:
    """Consolidate state_t*.json files in run_dir into states.json.

    Returns True if any work was done (or would be done in dry-run).
    """
    files = sorted(
        run_dir.glob("state_t*.json"),
        key=lambda p: int(p.stem[len("state_t"):]),
    )

    if not files:
        return False

    out_path = run_dir / "states.json"
    rel = run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir

    if dry_run:
        action = f"would write {out_path.name} ({len(files)} timesteps)"
        if delete:
            action += f", would delete {len(files)} state_t*.json files"
        print(f"  [dry-run] {rel}: {action}")
        return True

    states = []
    for f in files:
        with open(f) as fh:
            states.append(json.load(fh))

    with open(out_path, "w") as fh:
        fh.write("[\n")
        for i, state in enumerate(states):
            fh.write(json.dumps(state, indent=2, default=str))
            if i < len(states) - 1:
                fh.write(",\n")
        fh.write("\n]")

    print(f"  {rel}: wrote {out_path.name} ({len(states)} timesteps)")

    if delete:
        for f in files:
            f.unlink()
        print(f"    deleted {len(files)} state_t*.json files")

    return True


def main():
    parser = argparse.ArgumentParser(description="Consolidate state_t*.json files into states.json")
    parser.add_argument("path", type=Path, help="Run directory or parent directory to scan")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recurse into subdirectories")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing")
    parser.add_argument("--delete", action="store_true",
                        help="Delete per-timestep files after consolidating")
    args = parser.parse_args()

    if not args.path.is_dir():
        print(f"Error: {args.path} is not a directory")
        sys.exit(1)

    run_dirs = find_run_dirs(args.path, args.recursive)
    if not run_dirs:
        print("No run directories with state_t*.json files found.")
        sys.exit(0)

    done = 0
    for run_dir in run_dirs:
        if consolidate(run_dir, dry_run=args.dry_run, delete=args.delete):
            done += 1

    verb = "would process" if args.dry_run else "processed"
    print(f"\nDone. {verb} {done} run director{'y' if done == 1 else 'ies'}.")


if __name__ == "__main__":
    main()
