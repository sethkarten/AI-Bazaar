"""
run_figures.py — Generate all AI-Bazaar paper figures in one shot.

USAGE:
    python run_figures.py          # if AI-Bazaar conda env is already active
    conda run -n AI-Bazaar python run_figures.py   # run from any environment

CONFIGURATION:
    Edit the RUN DIRECTORIES block below to point at your log directories.
    Set any directory list to [] to skip figures that need those runs.
    Toggle the SKIP sets at the bottom to exclude individual figures.
"""

import subprocess
import sys
import os
from pathlib import Path

# Always use the AI-Bazaar conda environment's Python, regardless of how this
# script is invoked. Falls back to the current interpreter if not found.
_CONDA_PYTHON = Path(r"C:\Users\cbcro\miniconda3\envs\AI-Bazaar\python.exe")
PYTHON = str(_CONDA_PYTHON) if _CONDA_PYTHON.exists() else sys.executable

# ============================================================
# RUN DIRECTORIES — edit these to select your experiment data
# ============================================================

# THE CRASH — baseline (no stabilizing firm)
CRASH_BASELINE = [
    "logs/exp1_baseline",
]

# THE CRASH — with stabilizing firm intervention
CRASH_STABILIZING = [
    "logs/exp!_stab_1",
    "logs/exp!_stab_2", 
    "logs/exp!_stab_3",
    "logs/exp!_stab_4",
    "logs/exp!_stab_5",
]

# LEMON MARKET — baseline (honest firms only, no Sybil)
LEMON_BASELINE = [
    # "logs/lemon_50_flash_nosybil_1",
]

# LEMON MARKET — with Sybil firms present (no guardian)
LEMON_SYBIL = [
    # "logs/lemon_50_flash_sybil_1",
]

# LEMON MARKET — with Skeptical Guardian intervention
LEMON_GUARDIAN = [
    # "logs/lemon_guardian_seed42",
    # "logs/lemon_guardian_seed123",
]

# ============================================================
# SKIP — comment out entries to re-enable those figures
# ============================================================

SKIP = {
    # Placeholder scripts (no real data yet) — skip by default
    "crash_trajectory",      # placeholder
    "lemon_volume",          # placeholder
    "pareto",                # placeholder
    "sybil_detection",       # placeholder

    # Icon generation (slow, needs GEMINI_API_KEY, run once manually)
    "gen_icons",

    # Uncomment to skip individual figures:
    # "crash_price_cascade",
    # "crash_survival_curve",
    # "crash_intervention",
    # "crash_welfare",
    # "lemon_market_freeze",
    # "lemon_reputation_quality",
    # "lemon_guardian_effect",
    # "lemon_consumer_welfare",
    # "welfare_summary",
    # "methodology",
    # "teaser",
    # "gen_stability_chart",
}

# ============================================================
# INTERNALS — no need to edit below this line
# ============================================================

SCRIPTS = Path("paper/fig/scripts")


def dirs(*lists):
    """Flatten lists, skip empty strings, return as flat list."""
    return [d for lst in lists for d in lst if d]


def run(name, args):
    """Run one figure script, print status. Returns True on success."""
    if name in SKIP:
        print(f"  [SKIP]  {name}")
        return True

    cmd = [PYTHON, str(SCRIPTS / f"{name}.py")] + args
    label = f"  [RUN ]  {name}"

    # Check that all --run-dirs / --*-dirs arguments actually exist before launching
    dir_args = [a for a in args if not a.startswith("--")]
    missing = [d for d in dir_args if not Path(d).is_dir()]
    if missing:
        print(f"  [SKIP]  {name}  (missing dirs: {', '.join(missing)})")
        return True

    # If every positional dir arg was stripped (empty dirs lists), skip silently
    if not dir_args and any(a.startswith("--") and "dirs" in a for a in args):
        print(f"  [SKIP]  {name}  (no run dirs configured)")
        return True

    print(label)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [FAIL]  {name} exited with code {result.returncode}")
        return False
    return True


def build_dir_args(flag, dir_list):
    """Build ['--flag', 'dir1', 'dir2', ...] or [] if list is empty."""
    if not dir_list:
        return []
    return [flag] + dir_list


def main():
    os.chdir(Path(__file__).parent)  # always run from repo root

    print("=" * 60)
    print("AI-Bazaar Figure Generation")
    print("=" * 60)

    failures = []

    def maybe_run(name, args):
        if not run(name, args):
            failures.append(name)

    # ── No-data figures (no CLI args needed) ────────────────
    print("\n── Static / Layout Figures ──")
    maybe_run("methodology",        [])
    maybe_run("gen_stability_chart", [])
    maybe_run("teaser",             [])

    # ── THE CRASH figures ────────────────────────────────────
    print("\n── The Crash ──")

    # C1: per-firm price cascade (baseline runs)
    maybe_run("crash_price_cascade",
        build_dir_args("--run-dirs", CRASH_BASELINE))

    # C2: survival curve (baseline + stabilizing)
    maybe_run("crash_survival_curve",
        build_dir_args("--baseline-dirs",    CRASH_BASELINE)
        + build_dir_args("--stabilizing-dirs", CRASH_STABILIZING))

    # C3: intervention comparison (baseline vs stabilizing)
    maybe_run("crash_intervention",
        build_dir_args("--baseline-dirs",    CRASH_BASELINE)
        + build_dir_args("--stabilizing-dirs", CRASH_STABILIZING))

    # C4: welfare cost (baseline runs)
    maybe_run("crash_welfare",
        build_dir_args("--run-dirs", CRASH_BASELINE))

    # placeholder
    maybe_run("crash_trajectory", [])

    # ── LEMON MARKET figures ─────────────────────────────────
    print("\n── Lemon Market ──")

    # L1: market freeze / Akerlof effect (sybil runs)
    maybe_run("lemon_market_freeze",
        build_dir_args("--run-dirs", LEMON_SYBIL))

    # L2: reputation vs quality by firm type (sybil runs)
    maybe_run("lemon_reputation_quality",
        build_dir_args("--run-dirs", LEMON_SYBIL))

    # L3: guardian intervention (baseline sybil + guardian runs)
    maybe_run("lemon_guardian_effect",
        build_dir_args("--baseline-dirs", LEMON_SYBIL)
        + build_dir_args("--guardian-dirs", LEMON_GUARDIAN))

    # L4: consumer welfare harm (sybil runs)
    maybe_run("lemon_consumer_welfare",
        build_dir_args("--run-dirs", LEMON_SYBIL))

    # placeholder
    maybe_run("lemon_volume", [])

    # ── Placeholder figures ──────────────────────────────────
    print("\n── Placeholders ──")
    maybe_run("pareto",          [])
    maybe_run("sybil_detection", [])

    # ── Welfare summary (all conditions) ────────────────────
    print("\n── Summary ──")
    maybe_run("welfare_summary",
        build_dir_args("--crash-baseline-dirs",    CRASH_BASELINE)
        + build_dir_args("--crash-stabilizing-dirs", CRASH_STABILIZING)
        + build_dir_args("--lemon-baseline-dirs",    LEMON_BASELINE)
        + build_dir_args("--lemon-guardian-dirs",    LEMON_GUARDIAN))

    # ── Icon generation (skipped by default) ─────────────────
    print("\n── Icons ──")
    maybe_run("gen_icons", [])

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"DONE — {len(failures)} figure(s) failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("DONE — all figures generated successfully.")


if __name__ == "__main__":
    main()
