#!/usr/bin/env python3
"""
Copy non-.log artifacts from logs/ into logs-data/, preserving subdirectory layout.

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
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".log":
            continue
        rel = path.relative_to(src_root)
        dest = dst_root / rel
        if args.dry_run:
            print(f"{path} -> {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        copied += 1

    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {copied} file(s).")


if __name__ == "__main__":
    main()
