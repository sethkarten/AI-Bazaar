"""
Figure 6: Sybil Detection Confusion Matrix (optional)

2x2 grid of confusion matrices:
  rows = {Base buyer, Skeptical Guardian}
  cols = {small model, large model}
Placeholder structure; fill data after experiments.
Output: fig/sybil_detection.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# ── Config ─────────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "sybil_detection.pdf")

# ── TODO: Replace with real confusion matrix data after experiments ─────────
# Each matrix is [[TN, FP], [FN, TP]] normalized by row (recall-oriented).
# Format: np.array([[true_neg_rate, false_pos_rate],
#                   [false_neg_rate, true_pos_rate]])
# Placeholder: all NaN until filled.
NaN2 = np.full((2, 2), np.nan)

matrices = {
    ("Base Buyer",         "Small Model (3B)"): NaN2.copy(),
    ("Base Buyer",         "Large Model (7B)"): NaN2.copy(),
    ("Skeptical Guardian", "Small Model (3B)"): NaN2.copy(),
    ("Skeptical Guardian", "Large Model (7B)"): NaN2.copy(),
}

labels = ["Honest", "Deceptive"]

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for ax, ((row_label, col_label), mat) in zip(axes.flat, matrices.items()):
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Honest", "Pred: Deceptive"], fontsize=8)
    ax.set_yticklabels(["True: Honest", "True: Deceptive"], fontsize=8)
    ax.set_title(f"{row_label}\n{col_label}", fontsize=9)
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            text = f"{val:.2f}" if not np.isnan(val) else "TBD"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if (not np.isnan(val) and val > 0.5) else "black",
                    fontsize=11)

fig.suptitle("Sybil Detection Confusion Matrices (Exp.\\ 2)", fontsize=13)
plt.colorbar(im, ax=axes, fraction=0.03, pad=0.04, label="Rate")
plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
