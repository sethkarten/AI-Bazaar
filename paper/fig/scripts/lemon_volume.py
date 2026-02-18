"""
Figure 4: Market Volume Decay — The Lemon Market (Experiment 2)

Dual-axis line plot: trading volume (left) + avg reputation (right).
Placeholder series only; fill data after experiments.
Output: fig/lemon_volume_reputation.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ── Config ─────────────────────────────────────────────────────────────────
T = 30
SHOCK_T = 15   # Flood of Fakes shock (Exp 3)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "lemon_volume_reputation.pdf")

# ── TODO: Replace with real data after experiments ─────────────────────────
timesteps = np.arange(T)

# Volume series (left axis)
vol_base     = np.full(T, np.nan)   # base buyer volume — expected to collapse
vol_guardian = np.full(T, np.nan)   # guardian buyer volume — expected to stabilize

# Reputation series (right axis)
rep_base     = np.full(T, np.nan)   # avg reputation with base buyers
rep_guardian = np.full(T, np.nan)   # avg reputation with guardian buyers

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()

# Volume (left axis)
l1, = ax1.plot(timesteps, vol_base,     color="#d62728", linestyle="-",  lw=2.0, label="Base buyer volume")
l2, = ax1.plot(timesteps, vol_guardian, color="#2ca02c", linestyle="-",  lw=2.0, label="Guardian buyer volume")

# Reputation (right axis)
l3, = ax2.plot(timesteps, rep_base,     color="#d62728", linestyle="--", lw=1.5, label="Avg rep. (Base buyers)")
l4, = ax2.plot(timesteps, rep_guardian, color="#2ca02c", linestyle="--", lw=1.5, label="Avg rep. (Guardians)")

# Shock marker
ax1.axvline(SHOCK_T, color="gray", linestyle=":", lw=1.5)
ax1.text(SHOCK_T + 0.3, ax1.get_ylim()[1] * 0.9 if not np.isnan(ax1.get_ylim()[1]) else 0.9,
         "Flood of Fakes", fontsize=8, color="gray", va="top")

ax1.set_xlim(0, T - 1)
ax1.set_xlabel("Timestep $t$", fontsize=12)
ax1.set_ylabel("Trading Volume $V_t$", fontsize=12)
ax2.set_ylabel("Avg. Seller Reputation $\\bar{R}_t$", fontsize=12)
ax2.set_ylim(0, 1)

# Combined legend
lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=9, loc="upper right")

ax1.set_title("The Lemon Market: Volume \\& Reputation Decay (Exp.\\ 2)", fontsize=13)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
