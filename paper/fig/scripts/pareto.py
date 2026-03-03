"""
Figure 5: Pareto Frontier — Compute vs. Economic Alignment Score (Experiment 4)

Scatter plot with Pareto frontier curve.
Placeholder structure only; fill data after experiments.
Output: fig/pareto_frontier.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os

# ── Config ─────────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "pareto_frontier.pdf")

# ── Model family metadata ───────────────────────────────────────────────────
# (family_name, param_count_B, color)
MODEL_FAMILIES = [
    ("Gemma 3",     4.0,  "#1f77b4"),  # blue
    ("Qwen 3",      7.0,  "#ff7f0e"),  # orange
    ("Ministral 3", 3.0,  "#2ca02c"),  # green
    ("OLMo 3",      7.0,  "#9467bd"),  # purple
    ("Llama 3.2",   3.2,  "#d62728"),  # red
]

# ── TODO: Replace with real EAS values after experiments ───────────────────
# Placeholder: EAS values are all NaN.
# After experiments, set:
#   eas_base[i]      = EAS of model i in base configuration
#   eas_finetuned[i] = EAS of model i in finetuned configuration
eas_base      = [np.nan] * len(MODEL_FAMILIES)
eas_finetuned = [np.nan] * len(MODEL_FAMILIES)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for i, (name, params, color) in enumerate(MODEL_FAMILIES):
    # Base: hollow circle
    ax.scatter(params, eas_base[i],
               color=color, s=80, marker="o", facecolors="none",
               edgecolors=color, linewidths=1.8, zorder=3)
    # Finetuned: filled circle
    ax.scatter(params, eas_finetuned[i],
               color=color, s=80, marker="o",
               zorder=3, label=name)
    # Dashed arrow from base to finetuned (improvement vector)
    if not (np.isnan(eas_base[i]) or np.isnan(eas_finetuned[i])):
        ax.annotate("", xy=(params, eas_finetuned[i]),
                    xytext=(params, eas_base[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, linestyle="dashed"))

# ── Pareto frontier (connect Pareto-optimal finetuned models) ──────────────
# TODO: compute actual Pareto frontier after filling in EAS values.
# Placeholder: draw nothing until data is available.
# Example (replace with real computation):
#   pareto_idx = compute_pareto(params_list, eas_finetuned)
#   ax.plot(params_list[pareto_idx], eas_finetuned[pareto_idx], 'k--', lw=1.5, label="Pareto frontier")

ax.set_xscale("log")
ax.set_xlim(1, 20)
ax.set_ylim(0, 1)
ax.set_xlabel("Model Size (Billion Parameters, log scale)", fontsize=12)
ax.set_ylabel("Economic Alignment Score (EAS)", fontsize=12)
ax.set_title("Pareto Frontier: Compute vs.\\ Alignment (Exp.\\ 4)", fontsize=13)

# Legend entries for marker style (base vs. finetuned)
hollow = mlines.Line2D([], [], color="gray", marker="o", markersize=8, markerfacecolor="none",
                        linestyle="None", label="Base")
filled = mlines.Line2D([], [], color="gray", marker="o", markersize=8,
                        linestyle="None", label="Finetuned")
family_handles = [mlines.Line2D([], [], color=c, marker="o", markersize=8,
                                 linestyle="None", label=n)
                  for n, _, c in MODEL_FAMILIES]
ax.legend(handles=[hollow, filled] + family_handles, fontsize=9, loc="lower right",
          title="Style / Model Family", title_fontsize=9)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
