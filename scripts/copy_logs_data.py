#!/usr/bin/env python3
"""
Copy non-.log artifacts from logs/ into logs-data/, preserving subdirectory layout.

Only copies from top-level run directories that do not already exist in logs-data/.
Skips files whose extension is .log (case-insensitive). Creates logs-data/ as needed.

Usage:
    python scripts/copy_logs_data.py              # copy
    python scripts/copy_logs_data.py --dry-run    # list what would be copied
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DATA_DIR = PROJECT_ROOT / "logs-data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy all non-.log files from logs/ to logs-data/"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print copies without writing")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR, help="Source tree (default: logs/)")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=LOGS_DATA_DIR,
        help="Destination tree (default: logs-data/)",
    )
    args = parser.parse_args()

    src_root = args.logs_dir.resolve()
    dst_root = args.dest_dir.resolve()

    if not src_root.is_dir():
        print(f"Source directory not found: {src_root}", file=sys.stderr)
        sys.exit(1)

    copied = 0
    considered_dirs = 0
    copied_dirs = 0

    for entry in sorted(src_root.iterdir()):
        if not entry.is_dir():
            continue
        considered_dirs += 1

        dest_top = dst_root / entry.name
        if dest_top.exists():
            continue

        copied_dirs += 1
        for src_path in sorted(entry.rglob("*")):
            if not src_path.is_file():
                continue
            if src_path.suffix.lower() == ".log":
                continue
            rel = src_path.relative_to(src_root)
            dest_path = dst_root / rel
            if args.dry_run:
                print(f"{src_path} -> {dest_path}")
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
            copied += 1

    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {copied} file(s) from {copied_dirs}/{considered_dirs} top-level directories.")


if __name__ == "__main__":
    main()
