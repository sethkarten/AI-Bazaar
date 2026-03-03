"""
Figure 3: Price Trajectory — The Crash (Experiment 1)

Axes/labels/series structure only. Fill data after experiments.
Output: fig/crash_price_trajectory.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Config ─────────────────────────────────────────────────────────────────
T = 50
UNIT_COST = 5.0
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "crash_price_trajectory.pdf")

# ── TODO: Replace with real data after experiments ─────────────────────────
timesteps = np.arange(T)

# Placeholder series (all NaN until filled)
series = {
    # (label, color, linestyle, linewidth)
    "Llama 3.2 (Base)":              (np.full(T, np.nan), "#d62728", "-",  1.5),
    "Gemma 3 (Base)":                (np.full(T, np.nan), "#ff7f0e", "-",  1.5),
    "Qwen 3 (Base)":                 (np.full(T, np.nan), "#bcbd22", "-",  1.5),
    "Llama 3.2 (Stabilizing Firm)":  (np.full(T, np.nan), "#2ca02c", "--", 2.0),
    "Qwen 3 (Stabilizing Firm)":     (np.full(T, np.nan), "#1f77b4", "--", 2.0),
}

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

for label, (data, color, ls, lw) in series.items():
    ax.plot(timesteps, data, label=label, color=color, linestyle=ls, linewidth=lw)

# Bankruptcy zone shading (P < cost)
ax.axhline(UNIT_COST, color="black", linestyle=":", linewidth=1.2, label=f"Unit cost (\${UNIT_COST:.2f})")
ax.fill_between(timesteps, 0, UNIT_COST, color="red", alpha=0.08, label="Bankruptcy zone ($P < c$)")

ax.set_xlim(0, T - 1)
ax.set_ylim(0, 20)           # adjust y-range after seeing real data
ax.set_xlabel("Timestep $t$", fontsize=12)
ax.set_ylabel("Average Market Price ($\\bar{P}_t$, \\$)", fontsize=12)
ax.set_title("The Crash: Price Trajectory (Exp.\\ 1)", fontsize=13)
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
