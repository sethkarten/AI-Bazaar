#!/usr/bin/env python3
"""
One-shot utility: rename legacy exp1 baseline log directories.

    exp1_{model}_baseline  ->  exp1_{model}_stab_0_dlc3_seed8

Scans every exp1_* subdirectory under logs/ for a child named
*_baseline and renames it.  Use --dry-run to preview without renaming.

Usage:
    python scripts/rename_baselines.py              # execute renames
    python scripts/rename_baselines.py --dry-run    # preview only
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def main():
    parser = argparse.ArgumentParser(description="Rename exp1 baseline dirs to stab_0_dlc3_seed8")
    parser.add_argument("--dry-run", action="store_true", help="Print renames without executing")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR, help="Base logs directory")
    args = parser.parse_args()

    if not args.logs_dir.is_dir():
        print(f"Logs directory not found: {args.logs_dir}")
        sys.exit(1)

    renames = []
    for exp1_dir in sorted(args.logs_dir.iterdir()):
        if not exp1_dir.is_dir() or not exp1_dir.name.startswith("exp1_"):
            continue
        for child in sorted(exp1_dir.iterdir()):
            if not child.is_dir() or not child.name.endswith("_baseline"):
                continue
            new_name = child.name.replace("_baseline", "_stab_0_dlc3_seed8")
            new_path = child.parent / new_name
            renames.append((child, new_path))

    if not renames:
        print("No *_baseline directories found.")
        return

    for old, new in renames:
        rel_old = old.relative_to(args.logs_dir)
        rel_new = new.relative_to(args.logs_dir)
        if new.exists():
            print(f"  SKIP (target exists): {rel_old} -> {rel_new}")
            continue
        if args.dry_run:
            print(f"  [dry-run] {rel_old} -> {rel_new}")
        else:
            os.rename(old, new)
            print(f"  RENAMED: {rel_old} -> {rel_new}")

    action = "would rename" if args.dry_run else "renamed"
    print(f"\nDone. {len(renames)} director{'y' if len(renames) == 1 else 'ies'} {action}.")


if __name__ == "__main__":
    main()
