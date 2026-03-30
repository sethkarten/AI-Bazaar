"""
RL Training Summary — Before vs After comparison bar chart

Shows key metrics comparing:
  - Pre-training baseline (iteration 1, no RL)
  - Post-training easy mode (5/5 stabilizing, avg of last 10 iters)
  - Post-training mixed mode (with competitors, avg of last 10 iters)

Output: fig/rl_crash_summary.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.grid":          True,
    "grid.alpha":         0.3,
})

# ── Data from bf16_r64_lr5e6_curriculum run ──────────────────────────────

# Pre-training (iteration 1)
pre = {
    "stab_surv": 21, "nonstab_surv": 0, "mkt_surv": 21,
    "price": 1.91, "price_std": 0.152, "composite_S": 0.394,
}

# Post-training: easy mode (5/5 or full with n_stab>=4.5, last 10 iters)
easy_iters = [
    {"stab_surv": 90, "nonstab_surv": 38, "price": 2.03, "price_std": 0.040, "composite_S": 0.620},
    {"stab_surv": 87, "nonstab_surv": 50, "price": 2.02, "price_std": 0.040, "composite_S": 0.617},
    {"stab_surv": 94, "nonstab_surv": 0, "price": 2.02, "price_std": 0.021, "composite_S": 0.642},
    {"stab_surv": 89, "nonstab_surv": 50, "price": 2.02, "price_std": 0.028, "composite_S": 0.624},
    {"stab_surv": 92, "nonstab_surv": 0, "price": 2.04, "price_std": 0.045, "composite_S": 0.625},
    {"stab_surv": 91, "nonstab_surv": 53, "price": 2.00, "price_std": 0.024, "composite_S": 0.638},
]
easy = {k: np.mean([d[k] for d in easy_iters]) for k in easy_iters[0]}

# Post-training: mixed mode (n_stab < 4.0, last 10 mixed iters)
mixed_iters = [
    {"stab_surv": 67, "nonstab_surv": 63, "mkt_surv": 76, "price": 1.88, "price_std": 0.058, "composite_S": 0.611},
    {"stab_surv": 61, "nonstab_surv": 62, "mkt_surv": 71, "price": 1.84, "price_std": 0.090, "composite_S": 0.596},
    {"stab_surv": 66, "nonstab_surv": 58, "mkt_surv": 74, "price": 1.89, "price_std": 0.088, "composite_S": 0.597},
    {"stab_surv": 60, "nonstab_surv": 64, "mkt_surv": 69, "price": 1.96, "price_std": 0.349, "composite_S": 0.569},
    {"stab_surv": 68, "nonstab_surv": 57, "mkt_surv": 77, "price": 1.90, "price_std": 0.068, "composite_S": 0.605},
    {"stab_surv": 58, "nonstab_surv": 57, "mkt_surv": 66, "price": 2.09, "price_std": 1.347, "composite_S": 0.563},
    {"stab_surv": 84, "nonstab_surv": 68, "mkt_surv": 87, "price": 1.98, "price_std": 0.067, "composite_S": 0.620},
    {"stab_surv": 60, "nonstab_surv": 61, "mkt_surv": 70, "price": 1.86, "price_std": 0.073, "composite_S": 0.594},
]
mixed = {k: np.mean([d[k] for d in mixed_iters]) for k in mixed_iters[0]}

# ── Figure 1: Key metrics bar chart ─────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(11, 4))

# Panel (a): Survival rates
ax = axes[0]
x = np.arange(3)
width = 0.25
bars_pre = [pre["stab_surv"], 0, pre["stab_surv"]]
bars_easy = [easy["stab_surv"], easy["nonstab_surv"], (easy["stab_surv"] + easy["nonstab_surv"]) / 2]
bars_mixed = [mixed["stab_surv"], mixed["nonstab_surv"], mixed["mkt_surv"]]

ax.bar(x - width, bars_pre, width, label="Pre-training", color="#bbb", edgecolor="black", linewidth=0.5)
ax.bar(x, bars_easy, width, label="Post (easy)", color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=0.5)
ax.bar(x + width, bars_mixed, width, label="Post (mixed)", color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(["Stab. firm", "Non-stab.", "Market"])
ax.set_ylabel("Survival rate (%)")
ax.set_title("(a) Firm Survival")
ax.legend(loc="upper left", fontsize=8)
ax.set_ylim(0, 105)

# Panel (b): Price metrics
ax = axes[1]
x = np.arange(2)
width = 0.22

ax.bar(x[0] - width, pre["price"], width, label="Pre-training", color="#bbb", edgecolor="black", linewidth=0.5)
ax.bar(x[0], easy["price"], width, label="Post (easy)", color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=0.5)
ax.bar(x[0] + width, mixed["price"], width, label="Post (mixed)", color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)

ax.bar(x[1] - width, pre["price_std"], width, color="#bbb", edgecolor="black", linewidth=0.5)
ax.bar(x[1], easy["price_std"], width, color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=0.5)
ax.bar(x[1] + width, min(mixed["price_std"], 0.5), width, color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(["Mean price ($)", "Price volatility"])
ax.set_title("(b) Price Stability")
ax.axhline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.5)
ax.legend(loc="upper right", fontsize=8)

# Panel (c): Composite score
ax = axes[2]
x = np.arange(1)
width = 0.22

ax.bar(x - width, pre["composite_S"], width, label="Pre-training", color="#bbb", edgecolor="black", linewidth=0.5)
ax.bar(x, easy["composite_S"], width, label="Post (easy)", color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=0.5)
ax.bar(x + width, mixed["composite_S"], width, label="Post (mixed)", color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(["Composite $S$"])
ax.set_ylabel("Score (0-1)")
ax.set_title("(c) Market Health Score")
ax.set_ylim(0, 0.8)
ax.legend(loc="upper left", fontsize=8)

# Add value labels
for bar_set in [ax.patches]:
    for bar in bar_set:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "rl_crash_summary.pdf")
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"Saved: {out}")


if __name__ == "__main__":
    main() if "main" in dir() else None
