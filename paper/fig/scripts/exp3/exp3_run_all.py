"""
exp3_run_all.py — run all Experiment 3 analysis scripts.

Usage
-----
  python exp3_run_all.py
  python exp3_run_all.py --logs-dir-crash logs/exp3a_mymodel/ --logs-dir-lemon logs/exp3b_mymodel/

Any extra arguments are forwarded to both sub-scripts.
"""

import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent

subprocess.run(
    [sys.executable, str(_SCRIPTS_DIR / "exp3_crash_recovery.py")] + sys.argv[1:]
)
subprocess.run(
    [sys.executable, str(_SCRIPTS_DIR / "exp3_lemon_recovery.py")] + sys.argv[1:]
)
