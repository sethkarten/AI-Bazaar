"""
Shared Experiment 1 log directory layout.

Stabilizing sweep: exp1_{model}_stab_{k}_dlc{dlc}_seed{seed} under logs/exp1_{model}/.

Baseline (no stabilizer, dlc=3): prefer exp1_{model}_stab_0_dlc3_seed{seed}
(three seeds: 8, 16, 64). Legacy single run: exp1_{model}_baseline (seed 8 only).
"""

from __future__ import annotations

import os

SEEDS = [8, 16, 64]
DLC_VALUES = [1, 3, 5]
N_STAB_VALUES = [0, 1, 2, 3, 4, 5]


def resolve_run_dir(logs_dir: str, dlc: int, n_stab: int, seed: int, model: str = ""):
    """Return run directory for Exp1 config, or None if missing."""
    if model:
        if n_stab == 0:
            if dlc != 3 or seed not in SEEDS:
                return None
            stab0 = os.path.join(logs_dir, f"exp1_{model}_stab_0_dlc3_seed{seed}")
            if os.path.isdir(stab0):
                return stab0
            if seed == 8:
                legacy = os.path.join(logs_dir, f"exp1_{model}_baseline")
                return legacy if os.path.isdir(legacy) else None
            return None
        path = os.path.join(logs_dir, f"exp1_{model}_stab_{n_stab}_dlc{dlc}_seed{seed}")
        return path if os.path.isdir(path) else None

    if n_stab == 0:
        if dlc != 3 or seed not in SEEDS:
            return None
        stab0 = os.path.join(logs_dir, f"exp1_stab_0_dlc{dlc}_seed{seed}")
        if os.path.isdir(stab0):
            return stab0
        if seed == 8:
            legacy = os.path.join(logs_dir, "exp1_baseline")
            return legacy if os.path.isdir(legacy) else None
        return None
    if n_stab == 5:
        path = os.path.join(logs_dir, f"exp1_stab_5_dlc{dlc}_seed{seed}")
        return path if os.path.isdir(path) else None
    path = os.path.join(logs_dir, f"exp1_stab_{n_stab}_dlc{dlc}_seed{seed}")
    return path if os.path.isdir(path) else None


def collect_run_dirs(logs_dir: str, model: str = "") -> list[str]:
    """All run dirs for the full dlc × n_stab × seeds matrix (existing dirs only)."""
    dirs: list[str] = []
    for n_stab in N_STAB_VALUES:
        for dlc in DLC_VALUES:
            for seed in SEEDS:
                d = resolve_run_dir(logs_dir, dlc, n_stab, seed, model=model)
                if d:
                    dirs.append(d)
    return dirs


def baseline_run_dirs(logs_dir: str, model: str = "") -> list[str | None]:
    """Resolve baseline (dlc=3, n_stab=0) dirs for each seed in SEEDS; None if missing."""
    return [resolve_run_dir(logs_dir, 3, 0, seed, model=model) for seed in SEEDS]
