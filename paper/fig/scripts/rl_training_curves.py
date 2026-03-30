"""
RL Training Curves — REINFORCE++ Stabilizing Firm (THE_CRASH)

Generates 4-panel figure from training log data:
  (a) Survival rate over iterations (stab + nonstab + market)
  (b) Price stability (mean price + std band)
  (c) Composite health score S over iterations
  (d) Curriculum progression (n_stab_avg + stage labels)

Usage:
    python rl_training_curves.py --log slurm-6197469.out --output rl_crash_training.pdf
    # Or with inline data (no log file needed):
    python rl_training_curves.py --output rl_crash_training.pdf
"""

import argparse
import re
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.8,
    "axes.grid":          True,
    "grid.alpha":         0.3,
})

# Colors
C_STAB = "#2ca02c"       # green — stabilizing firm survival
C_NONSTAB = "#d62728"    # red — non-stabilizing firm survival
C_MKT = "#1f77b4"        # blue — market overall
C_PRICE = "#ff7f0e"      # orange — price
C_SCORE = "#9467bd"      # purple — composite score
C_CURRICULUM = "#8c564b" # brown — curriculum


def parse_log(log_path):
    """Parse SLURM log for Collected lines with metrics."""
    records = []
    with open(log_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Collected"):
            rec = {}
            # Parse main line
            m = re.search(r"stab_surv=(\d+)%", line)
            if m: rec["stab_surv"] = int(m.group(1)) / 100
            m = re.search(r"nonstab_surv=(\d+)%", line)
            if m: rec["nonstab_surv"] = int(m.group(1)) / 100
            m = re.search(r"mkt=(\d+)%", line)
            if m: rec["mkt_surv"] = int(m.group(1)) / 100
            m = re.search(r"return=([-\d.]+)", line)
            if m: rec["return"] = float(m.group(1))

            # Parse metrics line (next line)
            if i + 1 < len(lines):
                ml = lines[i + 1].strip()
                m = re.search(r"price=([\d.]+)", ml)
                if m: rec["price"] = float(m.group(1))
                m = re.search(r"std=([\d.]+)", ml)
                if m: rec["price_std"] = float(m.group(1))
                m = re.search(r"gini=([\d.]+)", ml)
                if m: rec["gini"] = float(m.group(1))
                m = re.search(r"composite_S=([\d.]+)", ml)
                if m: rec["composite_S"] = float(m.group(1))
                m = re.search(r"curriculum=(\S+)", ml)
                if m: rec["curriculum"] = m.group(1)
                m = re.search(r"n_stab_avg=([\d.]+)", ml)
                if m: rec["n_stab_avg"] = float(m.group(1))
                i += 1  # skip metrics line

            records.append(rec)
        i += 1

    return records


def parse_inline():
    """Hardcoded data from bf16_r64_lr5e6_curriculum run (SLURM 6197469)."""
    raw = [
        (21, 0, 21, 1.91, 0.152, 0.016, 0.394, "5/5", 5.0, -48.03),
        (96, 0, 96, 1.98, 0.019, 0.223, 0.653, "full", 5.0, -37.66),
        (54, 67, 67, 1.85, 0.084, 0.158, 0.579, "5/5", 2.9, -23.20),
        (94, 0, 94, 2.01, 0.016, 0.361, 0.643, "full", 5.0, -42.83),
        (56, 68, 69, 1.87, 0.103, 0.161, 0.582, "5/5", 2.6, -18.54),
        (95, 0, 95, 2.01, 0.024, 0.246, 0.643, "full", 5.0, -35.14),
        (63, 62, 71, 1.89, 0.075, 0.189, 0.585, "mix4", 3.1, -24.82),
        (95, 47, 96, 2.00, 0.043, 0.213, 0.643, "full", 4.5, -29.20),
        (59, 60, 70, 1.83, 0.093, 0.242, 0.598, "5/5", 3.1, -25.91),
        (86, 0, 86, 2.06, 0.055, 0.242, 0.603, "full", 5.0, -36.41),
        (76, 64, 81, 1.88, 0.064, 0.200, 0.626, "mix3", 3.3, -23.66),
        (88, 62, 91, 2.00, 0.037, 0.266, 0.630, "full", 4.1, -32.75),
        (67, 63, 76, 1.88, 0.058, 0.203, 0.611, "mix4", 3.3, -26.10),
        (89, 38, 89, 2.01, 0.028, 0.222, 0.627, "full", 4.6, -32.43),
        (61, 62, 71, 1.84, 0.090, 0.222, 0.596, "mix4", 3.2, -27.78),
        (90, 38, 90, 2.03, 0.040, 0.196, 0.620, "full", 4.6, -30.53),
        (66, 58, 74, 1.89, 0.088, 0.205, 0.597, "mix4", 3.2, -24.20),
        (87, 50, 88, 2.02, 0.040, 0.264, 0.617, "full", 4.5, -35.10),
        (60, 64, 69, 1.96, 0.349, 0.186, 0.569, "5/5", 3.0, -25.40),
        (94, 0, 94, 2.02, 0.021, 0.230, 0.642, "full", 5.0, -33.73),
        (68, 57, 77, 1.90, 0.068, 0.158, 0.605, "mix4", 3.2, -22.25),
        (89, 50, 90, 2.02, 0.028, 0.288, 0.624, "full", 4.5, -38.59),
        (58, 57, 66, 2.09, 1.347, 0.223, 0.563, "5/5", 2.9, -24.60),
        (92, 0, 92, 2.04, 0.045, 0.278, 0.625, "full", 5.0, -38.24),
        (84, 68, 87, 1.98, 0.067, 0.223, 0.620, "mix3", 3.5, -21.27),
        (91, 53, 92, 2.00, 0.024, 0.247, 0.638, "full", 4.1, -26.95),
        (60, 61, 70, 1.86, 0.073, 0.242, 0.594, "5/5", 3.0, -25.68),
    ]
    records = []
    for ss, ns, mk, pr, ps, gi, cs, cu, na, ret in raw:
        records.append({
            "stab_surv": ss / 100, "nonstab_surv": ns / 100, "mkt_surv": mk / 100,
            "price": pr, "price_std": ps, "gini": gi, "composite_S": cs,
            "curriculum": cu, "n_stab_avg": na, "return": ret,
        })
    return records


def plot(records, output_path):
    iters = np.arange(1, len(records) + 1)

    stab = [r["stab_surv"] for r in records]
    nonstab = [r["nonstab_surv"] for r in records]
    mkt = [r["mkt_surv"] for r in records]
    price = [r["price"] for r in records]
    price_std = [min(r["price_std"], 0.5) for r in records]  # cap outliers
    composite = [r["composite_S"] for r in records]
    n_stab = [r["n_stab_avg"] for r in records]
    curriculum = [r["curriculum"] for r in records]
    returns = [r["return"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    # (a) Survival rates
    ax = axes[0, 0]
    ax.plot(iters, [s * 100 for s in stab], color=C_STAB, label="Stabilizing firms", marker="o", markersize=3)
    ax.plot(iters, [s * 100 for s in nonstab], color=C_NONSTAB, label="Non-stabilizing firms", marker="s", markersize=3)
    ax.plot(iters, [s * 100 for s in mkt], color=C_MKT, label="Market overall", marker="^", markersize=3, alpha=0.7)
    ax.set_ylabel("Survival rate (%)")
    ax.set_title("(a) Firm Survival over Training")
    ax.legend(loc="lower right")
    ax.set_ylim(-5, 105)

    # Shade mixed-mode iterations
    for i, c in enumerate(curriculum):
        if c not in ("full", "5/5"):
            ax.axvspan(i + 0.5, i + 1.5, color="yellow", alpha=0.1)

    # (b) Price stability
    ax = axes[0, 1]
    p = np.array(price)
    ps = np.array(price_std)
    ax.plot(iters, p, color=C_PRICE, label="Mean price")
    ax.fill_between(iters, p - ps, p + ps, color=C_PRICE, alpha=0.2, label="$\\pm$ 1 std")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1, label="Unit cost ($1.00)")
    ax.set_ylabel("Price ($)")
    ax.set_title("(b) Price Stability")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 3.5)

    # (c) Composite score + episode return
    ax = axes[1, 0]
    ax.plot(iters, composite, color=C_SCORE, label="Composite $S$", marker="D", markersize=3)
    ax2 = ax.twinx()
    ax2.plot(iters, returns, color="gray", alpha=0.5, linestyle="--", label="Episode return")
    ax2.set_ylabel("Episode return", color="gray")
    ax.set_ylabel("Composite score $S$")
    ax.set_xlabel("Training iteration")
    ax.set_title("(c) Composite Health Score & Return")
    ax.set_ylim(0.3, 0.75)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # (d) Curriculum progression
    ax = axes[1, 1]
    ax.plot(iters, n_stab, color=C_CURRICULUM, marker="o", markersize=4, label="Avg stabilizing firms")
    ax.set_ylabel("# Stabilizing firms (of 5)")
    ax.set_xlabel("Training iteration")
    ax.set_title("(d) Adaptive Curriculum")
    ax.set_ylim(0, 5.5)

    # Color-code curriculum stages
    stage_colors = {"5/5": "#d4edda", "mix4": "#fff3cd", "mix3": "#fce4ec", "full": "#e3f2fd"}
    for i, c in enumerate(curriculum):
        ax.axvspan(i + 0.5, i + 1.5, color=stage_colors.get(c, "white"), alpha=0.4)

    # Legend for stages
    patches = [mpatches.Patch(color=stage_colors[k], label=k) for k in ["5/5", "mix4", "mix3", "full"]]
    ax.legend(handles=patches, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", help="Path to SLURM log file")
    parser.add_argument("--output", default="paper/fig/rl_crash_training.pdf")
    args = parser.parse_args()

    if args.log and os.path.exists(args.log):
        records = parse_log(args.log)
    else:
        records = parse_inline()

    print(f"Loaded {len(records)} iterations")
    plot(records, args.output)


if __name__ == "__main__":
    main()
