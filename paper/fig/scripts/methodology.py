"""
Figure 2: Methodology Diagram — Agent-Environment Interaction Loop
Three swim-lane diagram using matplotlib patches only.

Lane 1 (top):    Environment — market state loop
Lane 2 (middle): Agent Internals — LLM backbone -> CoT -> action
Lane 3 (bottom): Sybil Structure — Deceptive Principal -> K identities

v3: Tight layout, no dead whitespace, legend in title bar area,
Guardian callout repositioned, all content fits within lanes.

Output: fig/methodology.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "methodology.pdf")

# ── Colours ──────────────────────────────────────────────────────────────────
C_ENV = "#1a5276"   # dark blue  — environment lane
C_AGT = "#145a32"   # dark green — agent lane
C_SYB = "#6c3483"   # purple     — Sybil lane
C_ORG = "#b7600a"   # orange     — annotation callouts
C_RED = "#c0392b"

LN_ENV = "#d6eaf8"
LN_AGT = "#d5f5e3"
LN_SYB = "#e8daef"

# ── Canvas ───────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 14.0, 7.8
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Layout constants ─────────────────────────────────────────────────────────
# Title banner at top
TITLE_H  = 0.55
TITLE_Y  = FIG_H - TITLE_H   # 7.25

# Three swim lanes fill from bottom to just below title
LANE_TOP = TITLE_Y - 0.08     # 7.17
LANE_H   = 2.22               # each lane
ENV_TOP  = LANE_TOP            # 7.17
ENV_BOT  = ENV_TOP - LANE_H   # 4.95
AGT_TOP  = ENV_BOT             # 4.95
AGT_BOT  = AGT_TOP - LANE_H   # 2.73
SYB_TOP  = AGT_BOT             # 2.73
SYB_BOT  = 0.0                 # flush to bottom (slightly taller sybil lane)

# ── Helpers ──────────────────────────────────────────────────────────────────

def draw_lane(ax, y_bot, height, color, label, label_color):
    """Horizontal swim-lane background with rotated label."""
    rect = mpatches.Rectangle((0, y_bot), FIG_W, height,
                               facecolor=color, edgecolor="none",
                               alpha=0.28, zorder=0)
    ax.add_patch(rect)
    ax.text(0.22, y_bot + height / 2, label, fontsize=8.5, color=label_color,
            ha="left", va="center", fontweight="bold",
            rotation=90, zorder=2)

def box(ax, x, y, w, h, fc="#f2f3f4", ec=C_ENV, lw=1.5,
        label="", fs=9, label_color="black", bold=False,
        sublabel="", sublabel_color=None, radius=0.10, alpha=1.0):
    """Rounded box with label + optional 1-line sublabel."""
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad={radius}",
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        alpha=alpha, zorder=3)
    ax.add_patch(p)
    fw = "bold" if bold else "normal"
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.13, label,
                fontsize=fs, color=label_color, ha="center", va="center",
                fontweight=fw, zorder=4)
        sc = sublabel_color or label_color
        ax.text(x + w / 2, y + h / 2 - 0.16, sublabel,
                fontsize=fs - 1.5, color=sc, ha="center", va="center",
                fontstyle="italic", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                fontsize=fs, color=label_color, ha="center", va="center",
                fontweight=fw, zorder=4)

def arr(ax, x0, y0, x1, y1, color=C_ENV, lw=1.6, hw=0.14, hl=0.16,
        label="", label_color=None, conn="arc3,rad=0"):
    """Arrow with optional label badge."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                                color=color, lw=lw,
                                connectionstyle=conn),
                zorder=5)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        lc = label_color or color
        ax.text(mx, my + 0.18, label, fontsize=7.0, color=lc,
                ha="center", va="center", fontstyle="italic", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec=lc, lw=0.7, alpha=0.88))

def callout(ax, x, y, text, color=C_ORG, fs=7.0):
    """Small callout badge."""
    ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="#fff8f0",
                      ec=color, lw=0.9, alpha=0.95))

# ── Title banner ─────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (0.4, TITLE_Y), FIG_W - 0.8, TITLE_H,
    boxstyle="round,pad=0.08", facecolor="#f0f3f6", edgecolor="#888888",
    linewidth=1.0, zorder=3))
ax.text(FIG_W / 2, TITLE_Y + TITLE_H / 2,
        "Agent Bazaar \u2014 System Diagram",
        fontsize=13, ha="center", va="center", fontweight="bold",
        color="#1c1c1c", zorder=4)

# Legend inside the title banner area (right side)
legend_items = [
    mpatches.Patch(facecolor=LN_ENV, edgecolor=C_ENV, linewidth=0.8,
                   label="Environment"),
    mpatches.Patch(facecolor=LN_AGT, edgecolor=C_AGT, linewidth=0.8,
                   label="Agent Internals"),
    mpatches.Patch(facecolor=LN_SYB, edgecolor=C_SYB, linewidth=0.8,
                   label="Sybil Structure"),
]
leg = ax.legend(handles=legend_items, loc="center right",
                bbox_to_anchor=(0.99, (TITLE_Y + TITLE_H / 2) / FIG_H),
                fontsize=6.5, framealpha=0.0, edgecolor="none",
                ncol=3, columnspacing=1.0, handletextpad=0.4)
leg.get_frame().set_linewidth(0)

# ── Lane backgrounds ────────────────────────────────────────────────────────
draw_lane(ax, ENV_BOT, LANE_H, LN_ENV, "Environment", C_ENV)
draw_lane(ax, AGT_BOT, LANE_H, LN_AGT, "Agent\nInternals", C_AGT)
draw_lane(ax, SYB_BOT, SYB_TOP - SYB_BOT, LN_SYB, "Sybil\nStructure", C_SYB)

# Lane dividers
for yy in [ENV_BOT, AGT_BOT]:
    ax.axhline(yy, color="#aaaaaa", lw=0.9, zorder=1)

# =============================================================================
# LANE 1 — Environment  (ENV_BOT=4.95 to ENV_TOP=7.17)
# =============================================================================
BOX_ENV_Y = ENV_BOT + 0.58      # boxes start above loop arrow space
BOX_ENV_H = 1.08

# Market State
box(ax, 1.6, BOX_ENV_Y, 2.2, BOX_ENV_H, fc="#d6eaf8", ec=C_ENV, lw=1.6,
    label="Market State $s_t$", sublabel="prices, inventory, reputation",
    label_color=C_ENV, bold=True, fs=9)

# Arrow: Market State -> Agents
ENV_ARR_Y = BOX_ENV_Y + BOX_ENV_H / 2
arr(ax, 3.8, ENV_ARR_Y, 5.0, ENV_ARR_Y, color=C_ENV, lw=1.6,
    label="obs $o_t^i$", label_color=C_ENV)

# Agents
box(ax, 5.0, BOX_ENV_Y, 2.2, BOX_ENV_H, fc="#d6eaf8", ec=C_ENV, lw=1.6,
    label="Agents", sublabel="Firm | Buyer",
    label_color=C_ENV, bold=True, fs=9)

# Arrow: Agents -> Market Clearing
arr(ax, 7.2, ENV_ARR_Y, 8.6, ENV_ARR_Y, color=C_ENV, lw=1.6,
    label="action $a_t^i$", label_color=C_ENV)

# Market Clearing
box(ax, 8.6, BOX_ENV_Y, 2.6, BOX_ENV_H, fc="#d6eaf8", ec=C_ENV, lw=1.6,
    label="Market Clearing", sublabel="sort, assign, update rep",
    label_color=C_ENV, bold=True, fs=9)

# Loop arrow: Market Clearing -> Market State
# Curves below the boxes but stays inside the Environment lane
LOOP_START_X = 9.9   # bottom-center of Market Clearing
LOOP_END_X   = 2.7   # bottom-center of Market State
LOOP_Y       = BOX_ENV_Y  # bottom edge of boxes
ax.annotate("", xy=(LOOP_END_X, LOOP_Y), xytext=(LOOP_START_X, LOOP_Y),
            arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.14",
                            color=C_ENV, lw=1.4,
                            connectionstyle="arc3,rad=-0.28"),
            zorder=2)
ax.text(6.3, ENV_BOT + 0.16, "$s_{t+1}$", fontsize=7.5, color=C_ENV,
        ha="center", va="center", fontstyle="italic", zorder=6)

# Search friction callout (above observation arrow)
callout(ax, 4.4, ENV_ARR_Y + 0.68,
        "Search Friction: $\\tau$ tokens/step", color=C_ORG, fs=6.5)
ax.annotate("", xy=(4.4, ENV_ARR_Y + 0.30), xytext=(4.4, ENV_ARR_Y + 0.54),
            arrowprops=dict(arrowstyle="-", color=C_ORG, lw=0.7), zorder=5)

# =============================================================================
# LANE 2 — Agent Internals  (AGT_BOT=2.73 to AGT_TOP=4.95)
# =============================================================================
BOX_AGT_Y = AGT_BOT + 0.46
BOX_AGT_H = 1.10
AGT_ARR_Y = BOX_AGT_Y + BOX_AGT_H / 2

# LLM Backbone
box(ax, 1.6, BOX_AGT_Y, 2.0, BOX_AGT_H, fc="#d5f5e3", ec=C_AGT, lw=1.6,
    label="LLM Backbone", sublabel="Gemma, Qwen, Llama, ...",
    label_color=C_AGT, bold=True, fs=9)

# Arrow
arr(ax, 3.6, AGT_ARR_Y, 4.8, AGT_ARR_Y, color=C_AGT, lw=1.6)

# CoT Prompt
box(ax, 4.8, BOX_AGT_Y, 2.2, BOX_AGT_H, fc="#d5f5e3", ec=C_AGT, lw=1.6,
    label="CoT Prompt", sublabel="[SYS] [OBS] [THINK]",
    label_color=C_AGT, bold=True, fs=9)

# Arrow
arr(ax, 7.0, AGT_ARR_Y, 8.4, AGT_ARR_Y, color=C_AGT, lw=1.6)

# JSON Parser
box(ax, 8.4, BOX_AGT_Y, 2.0, BOX_AGT_H, fc="#d5f5e3", ec=C_AGT, lw=1.6,
    label="JSON Parser", sublabel='{"price": P, "qty": Q}',
    label_color=C_AGT, bold=True, fs=9)

# Arrow
arr(ax, 10.4, AGT_ARR_Y, 11.6, AGT_ARR_Y, color=C_AGT, lw=1.6)

# Action output
box(ax, 11.6, BOX_AGT_Y, 1.6, BOX_AGT_H, fc="#d5f5e3", ec=C_AGT, lw=1.6,
    label="Action $a_t^i$", sublabel="\u2192 Env",
    label_color=C_AGT, bold=True, fs=9)

# Vertical connector: Agents (env lane) -> CoT Prompt (agent lane)
arr(ax, 6.1, BOX_ENV_Y, 6.1, BOX_AGT_Y + BOX_AGT_H,
    color="#777777", lw=1.1, hw=0.12, hl=0.13)

# Forensic Analysis callout — inside Agent lane, above boxes on the right
# Positioned high enough to stay fully within the Agent lane
callout(ax, 12.2, AGT_TOP - 0.22,
        "Guardian: text sim. + rep/price check",
        color=C_SYB, fs=6.2)
# Pointer from callout down to the Action box
ax.annotate("", xy=(12.4, BOX_AGT_Y + BOX_AGT_H), xytext=(12.3, AGT_TOP - 0.38),
            arrowprops=dict(arrowstyle="-", color=C_SYB, lw=0.7), zorder=5)

# =============================================================================
# LANE 3 — Sybil Structure  (SYB_BOT=0.0 to SYB_TOP=2.73)
# =============================================================================
SYB_LANE_H = SYB_TOP - SYB_BOT  # 2.73
BOX_SYB_Y = SYB_BOT + 0.36
BOX_SYB_H = 1.30
SYB_ARR_Y = BOX_SYB_Y + BOX_SYB_H / 2

# Deceptive Principal
box(ax, 1.0, BOX_SYB_Y, 2.2, BOX_SYB_H, fc="#e8daef", ec=C_SYB, lw=1.6,
    label="Deceptive Principal", sublabel="1 LLM, K identities",
    label_color=C_SYB, bold=True, fs=9)

# K=5 seller identity boxes
id_xs = [3.8, 5.1, 6.4, 7.7, 9.0]
ID_W  = 1.05
for k, ix in enumerate(id_xs):
    box(ax, ix, BOX_SYB_Y, ID_W, BOX_SYB_H, fc="#e8daef", ec=C_SYB, lw=1.2,
        label=f"$S_{{{k+1}}}$", sublabel=f"$R_{{{k+1}}} \\in [0,1]$",
        label_color=C_SYB, fs=9, bold=True)
    arr(ax, 3.2, SYB_ARR_Y, ix, SYB_ARR_Y, color=C_SYB, lw=0.9,
        hw=0.10, hl=0.12)

# Shared coordination signal (dashed line across tops of S_k boxes)
COORD_Y = BOX_SYB_Y + BOX_SYB_H + 0.08
ax.plot([3.2, id_xs[-1] + ID_W], [COORD_Y, COORD_Y],
        color=C_SYB, linestyle="--", lw=0.9, zorder=4)
ax.text(6.4, COORD_Y + 0.08, "shared coordination signal",
        fontsize=6.8, color=C_SYB, ha="center", va="bottom",
        fontstyle="italic", zorder=6)

# Identity rotation callout — to the right of S5, shifted left to avoid clipping
callout(ax, 11.2, SYB_ARR_Y,
        "If $R_k<0.3$: retire $S_k$,\nspawn $S_{K+1}$ ($R_0=0.8$)",
        color=C_RED, fs=6.0)
# Pointer from S5 right edge to callout
s5_right = id_xs[-1] + ID_W
ax.annotate("", xy=(s5_right + 0.04, SYB_ARR_Y),
            xytext=(11.2 - 0.76, SYB_ARR_Y),
            arrowprops=dict(arrowstyle="-", color=C_RED, lw=0.7), zorder=5)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, bbox_inches="tight", format="pdf", dpi=150)
print(f"[v3] Saved: {OUT_PATH}")
