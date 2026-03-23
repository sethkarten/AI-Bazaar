"""
Lemon Market Simulation — LLM Action Architecture

Three horizontal swim-lanes, one complete simulation timestep.

Lane layout (top to bottom):
  Lane 1 — Seller Phase  (Honest Seller sub-row + Deceptive Principal sub-rows)
  Lane 2 — Market        (8 sequential timestep steps)
  Lane 3 — Buyer Phase   (Bid Decision LLM + Post-Purchase Review LLM)

Geometry — derived and verified before writing
----------------------------------------------
FIG: 14 × 10.5 inches.

Lane boundaries (bottom-to-top in y):
  BUY_BOT =  0.30    BUY_TOP =  3.54   BUY_H = 3.24
  MKT_BOT =  3.54    MKT_TOP =  5.42   MKT_H = 1.88
  SEL_BOT =  5.42    SEL_TOP =  9.94   SEL_H = 4.52
  TITLE_Y =  9.94

Row centres inside Seller lane:
  HON_ROW_Y  = 9.14
  SUB_DIV_Y  = 7.76    (dashed divider)
  SYB_ROW1_Y = 7.12
  SYB_ROW2_Y = 6.00

Box heights:
  BH_H = 0.82   honest row
  BH_T = 0.84   tier-decision row
  ID_H = 0.90   description row
  BH_B = 0.90   buyer row

Computed desc-box bottom:
  DESC_Y  = SYB_ROW2_Y - ID_H/2 = 6.00 - 0.45 = 5.55
  DESC_BOT = DESC_Y = 5.55   (> SEL_BOT 5.42 by 0.13 — clear margin)

Persona label y: DESC_Y - 0.16  (below box, va=top)
K-parallel label y: DESC_Y - 0.44  (below personas, inside lane)
Sybil arrow origin y: DESC_Y - 0.02  (box bottom with tiny gap)

Buyer row:
  BUY_ROW_Y = 1.90
  BUY_Y = BUY_ROW_Y - BH_B/2 = 1.45
  Box top = BUY_Y + BH_B = 2.35
  Bid callout placed BELOW box: y = BUY_Y - 0.44  (= 1.01 > BUY_BOT 0.30)

Market:
  MKT_MID = (3.54 + 5.42)/2 = 4.48
  BH_MKT = 0.86
  Box top = 4.48 + 0.43 = 4.91
  Box bot = 4.48 - 0.43 = 4.05
  Rep arc bows DOWN (rad=-0.20), anchored at box_bot - 0.04 = 4.01
  Arc nadir ≈ 4.01 - 0.25 = 3.76 > MKT_BOT 3.54  ✓ inside lane
  Rep label y = MKT_BOT + 0.26 = 3.80  ✓ inside lane

Cross-lane arrows:
  HON_ARR_X  = 10.42  (right of all seller content)
  SYB_ARR_X  = 4.83   (centre of S2 desc box)
  REV_ARR_X  = step_centres[6]  ≈ 10.0  (centre of Rep Update step)

Usage:
    python paper/fig/scripts/lemon/architecture.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH  = os.path.join(_THIS_DIR, "..", "..", "lemon", "architecture.pdf")

# ── Colours (identical to methodology.py) ────────────────────────────────────
C_ENV = "#1a5276"
C_AGT = "#145a32"
C_SYB = "#6c3483"
C_ORG = "#b7600a"
C_RED = "#c0392b"

LN_ENV = "#d6eaf8"
LN_AGT = "#d5f5e3"
LN_SYB = "#e8daef"
LN_SEL = "#eaf4fb"
LN_BUY = "#fef9e7"
C_BUY  = "#7d6608"

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 14.0, 10.5
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Lane boundaries ───────────────────────────────────────────────────────────
TITLE_H = 0.50
TITLE_Y = FIG_H - TITLE_H      # 10.00

SEL_TOP = TITLE_Y - 0.06       # 9.94

# Box heights — fixed
BH_H  = 0.82   # honest row
BH_T  = 0.84   # tier-decision row
ID_H  = 0.90   # description row
BH_MKT = 0.86  # market step
BH_B  = 0.90   # buyer row

# ── Seller lane: build bottom-up from SEL_BOT ─────────────────────────────────
# Required vertical space inside Seller lane (bottom to top):
#   0.20  bottom margin
#   ID_H  desc boxes          = 0.90   top = SEL_BOT + 0.20 + 0.90 = SEL_BOT + 1.10
#   0.34  persona labels + K-label gap
#   0.50  sub-sub-divider clearance / arrow
#   BH_T  tier box             = 0.84   top = above + 0.84
#   0.42  tier-to-subdiv gap
#   SUB_DIV (dashed line)
#   0.42  subdiv-to-honest gap
#   BH_H  honest row           = 0.82
#   0.30  top margin + SYS callout space
# Total ≈ 0.20+0.90+0.34+0.50+0.84+0.42+0.42+0.82+0.30 = 4.74
# → set SEL_BOT low enough, then derive row centres from bottom up

# Build geometry bottom-up.
# Persona labels and K-label sit BELOW the desc boxes (DESC_Y - margin space).
# "adv. tier (shared)" sits beside the vertical arrow (not a separate text row).
# Minimum gap between desc-top and tier-bottom = 0.20 (no text, just arrow).

SEL_BOT_MARGIN = 0.52   # below desc boxes: K-label (~0.14) + persona (~0.14) + 0.24 margins
DESC_TIER_GAP  = 0.22   # desc-top to tier-bottom: arrow clearance only
TIER_GAP_ABOVE = 0.46   # tier-top to SUB_DIV_Y: SYS callout fits here
SUB_HON_GAP    = 0.46   # SUB_DIV_Y to honest-box bottom

SEL_BOT = 4.46
DESC_Y    = SEL_BOT + SEL_BOT_MARGIN              # 4.98
DESC_TOP  = DESC_Y + ID_H                         # 5.88
TIER_Y    = DESC_TOP + DESC_TIER_GAP              # 6.10
TIER_TOP  = TIER_Y + BH_T                         # 6.94
SUB_DIV_Y = TIER_TOP + TIER_GAP_ABOVE             # 7.40
HON_Y     = SUB_DIV_Y + SUB_HON_GAP              # 7.86
HON_TOP   = HON_Y + BH_H                          # 8.68

# Row centres
SYB_ROW2_Y = DESC_Y  + ID_H  / 2                 # 5.43
SYB_ROW1_Y = TIER_Y  + BH_T  / 2                 # 6.52
HON_ROW_Y  = HON_Y   + BH_H  / 2                 # 8.27

MKT_TOP = SEL_BOT                                 # 4.46
MKT_BOT = 2.66
BUY_TOP = MKT_BOT                                 # 2.66
BUY_BOT = 0.86   # extra margin so Bid callout at BUY_Y-0.42 stays inside
BUY_ROW_Y = 1.90

SEL_H = SEL_TOP - SEL_BOT
MKT_H = MKT_TOP - MKT_BOT                         # 1.80
BUY_H = BUY_TOP - BUY_BOT                         # 2.12

BUY_Y    = BUY_ROW_Y - BH_B / 2                   # 1.45
MKT_MID  = (MKT_BOT + MKT_TOP) / 2                # 3.56

X_MIN, X_MAX = 0.60, 13.50

# ── Helpers ───────────────────────────────────────────────────────────────────

def draw_lane(y_bot, height, color, label, label_color, lx=0.22):
    ax.add_patch(mpatches.Rectangle(
        (0, y_bot), FIG_W, height,
        facecolor=color, edgecolor="none", alpha=0.28, zorder=0))
    ax.text(lx, y_bot + height / 2, label,
            fontsize=8.5, color=label_color,
            ha="center", va="center", fontweight="bold",
            rotation=90, zorder=2)


def box(x, y, w, h,
        fc="#f2f3f4", ec=C_ENV, lw=1.5,
        label="", fs=8.2, label_color="black", bold=False,
        sub1="", sub1_color=None,
        sub2="", sub2_color=None,
        radius=0.10, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))
    fw = "bold" if bold else "normal"
    cx, cy = x + w / 2, y + h / 2
    if sub1 and sub2:
        for txt, off, sz, col, sty, wt in [
            (label, +0.24, fs,       label_color,               "normal", fw),
            (sub1,  +0.02, fs - 1.7, sub1_color or label_color, "italic", "normal"),
            (sub2,  -0.22, fs - 2.0, sub2_color or label_color, "italic", "normal"),
        ]:
            ax.text(cx, cy + off, txt, fontsize=sz, color=col,
                    ha="center", va="center", fontweight=wt, fontstyle=sty,
                    zorder=zorder + 1)
    elif sub1:
        ax.text(cx, cy + 0.14, label, fontsize=fs, color=label_color,
                ha="center", va="center", fontweight=fw, zorder=zorder + 1)
        ax.text(cx, cy - 0.14, sub1, fontsize=fs - 1.7,
                color=sub1_color or label_color,
                ha="center", va="center", fontstyle="italic", zorder=zorder + 1)
    else:
        ax.text(cx, cy, label, fontsize=fs, color=label_color,
                ha="center", va="center", fontweight=fw, zorder=zorder + 1)


def arr(x0, y0, x1, y1, color=C_ENV, lw=1.4, hw=0.12, hl=0.14,
        conn="arc3,rad=0", zorder=5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                    color=color, lw=lw, connectionstyle=conn),
                zorder=zorder)


def callout(x, y, text, color=C_ORG, fs=6.2, zorder=6, ha="center"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center",
            zorder=zorder,
            bbox=dict(boxstyle="round,pad=0.17", fc="#fff8f0",
                      ec=color, lw=0.9, alpha=0.95))


def badge(x, y, n, color=C_ENV, zorder=8):
    ax.add_patch(plt.Circle((x, y), 0.165, color=color, zorder=zorder))
    ax.text(x, y, str(n), fontsize=6.5, color="white",
            ha="center", va="center", fontweight="bold", zorder=zorder + 1)


def stub(x0, y0, x1, y1, color, lw=0.75, zorder=5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-", color=color, lw=lw),
                zorder=zorder)


# ── Title banner ──────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (0.40, TITLE_Y), FIG_W - 0.80, TITLE_H,
    boxstyle="round,pad=0.08", facecolor="#f0f3f6", edgecolor="#888888",
    linewidth=1.0, zorder=3))
ax.text(FIG_W * 0.42, TITLE_Y + TITLE_H / 2,
        "Lemon Market Simulation \u2014 LLM Action Architecture",
        fontsize=12.5, ha="center", va="center", fontweight="bold",
        color="#1c1c1c", zorder=4)

legend_items = [
    mpatches.Patch(facecolor=LN_SEL, edgecolor=C_ENV, lw=0.8, label="Honest Seller"),
    mpatches.Patch(facecolor=LN_SYB, edgecolor=C_SYB, lw=0.8, label="Deceptive Principal"),
    mpatches.Patch(facecolor=LN_AGT, edgecolor=C_AGT, lw=0.8, label="LLM Call"),
    mpatches.Patch(facecolor=LN_ENV, edgecolor=C_ENV, lw=0.8, label="Market"),
    mpatches.Patch(facecolor=LN_BUY, edgecolor=C_BUY, lw=0.8, label="Buyer"),
]
leg = ax.legend(handles=legend_items,
                loc="center right",
                bbox_to_anchor=(0.996, (TITLE_Y + TITLE_H / 2) / FIG_H),
                fontsize=6.5, framealpha=0.0, edgecolor="none",
                ncol=5, columnspacing=0.7, handletextpad=0.4)
leg.get_frame().set_linewidth(0)

# ── Lane backgrounds ──────────────────────────────────────────────────────────
draw_lane(SEL_BOT, SEL_H, "#ede8f5", "Seller\nPhase", C_ENV,  lx=0.22)
draw_lane(MKT_BOT, MKT_H, LN_ENV,   "Market",          C_ENV,  lx=0.22)
draw_lane(BUY_BOT, BUY_H, LN_BUY,   "Buyer\nPhase",    C_BUY,  lx=0.22)

for yy in [SEL_BOT, MKT_BOT]:
    ax.axhline(yy, color="#aaaaaa", lw=0.9, zorder=1)

# Dashed sub-divider: honest / sybil halves of Seller lane
ax.axhline(SUB_DIV_Y, xmin=0.038, xmax=0.970,
           color="#999999", lw=0.65, linestyle="--", zorder=1)

# Sub-lane labels at x=0.44 — rotated, inside lane, well left of box content
ax.text(0.44, (HON_ROW_Y + SUB_DIV_Y) / 2, "Honest\nSeller",
        fontsize=6.4, color=C_ENV, ha="center", va="center",
        fontstyle="italic", rotation=90, zorder=2)
ax.text(0.44, (SUB_DIV_Y + SEL_BOT) / 2, "Deceptive\nPrincipal",
        fontsize=6.4, color=C_SYB, ha="center", va="center",
        fontstyle="italic", rotation=90, zorder=2)

# Dotted divider between sybil tier and desc rows
ax.axhline((SYB_ROW1_Y + SYB_ROW2_Y) / 2,
           xmin=0.038, xmax=0.970,
           color="#c8a8e0", lw=0.5, linestyle=":", zorder=1)

# =============================================================================
# LANE 1 — Seller Phase
# =============================================================================

# ── Honest Seller sub-row ─────────────────────────────────────────────────────
ENDOW_X, ENDOW_W = 0.68, 1.52
HLLM_X,  HLLM_W  = 2.38, 3.60
PRICE_X, PRICE_W = 6.20, 2.10
POST_X,  POST_W  = 8.56, 1.52   # right edge = 10.08

box(ENDOW_X, HON_Y, ENDOW_W, BH_H, fc=LN_SEL, ec=C_ENV, lw=1.5,
    label="Endow Car",
    sub1="quality ~ U{poor, fair, good, mint}",
    label_color=C_ENV, bold=True)
arr(ENDOW_X + ENDOW_W, HON_ROW_Y, HLLM_X, HON_ROW_Y, color=C_ENV)
badge((ENDOW_X + ENDOW_W + HLLM_X) / 2, HON_ROW_Y + 0.30, 1, C_ENV)

HLLM_CX = HLLM_X + HLLM_W / 2
box(HLLM_X, HON_Y, HLLM_W, BH_H, fc=LN_AGT, ec=C_AGT, lw=1.9,
    label="LLM Call: Honest Description",
    sub1="In: quality label, quality_value",
    sub2='Out: {"description": "<honest text>"}',
    sub2_color="#1a3a1a", label_color=C_AGT, bold=True)

# SYS callout — just above HON_TOP, inside Seller lane
HON_SYS_Y = HON_TOP + 0.30
callout(HLLM_CX, HON_SYS_Y,
        'SYS: "You are an honest used-car seller. Describe the true condition."',
        color=C_ORG, fs=6.1)
stub(HLLM_CX, HON_TOP, HLLM_CX, HON_SYS_Y - 0.18, color=C_ORG)

arr(HLLM_X + HLLM_W, HON_ROW_Y, PRICE_X, HON_ROW_Y, color=C_ENV)
badge((HLLM_X + HLLM_W + PRICE_X) / 2, HON_ROW_Y + 0.30, 2, C_ENV)

box(PRICE_X, HON_Y, PRICE_W, BH_H, fc=LN_SEL, ec=C_ENV, lw=1.5,
    label="Set Price (rule)",
    sub1="price = V\u2098\u2090\u2093 \u00d7 quality_value",
    label_color=C_ENV, bold=True)
arr(PRICE_X + PRICE_W, HON_ROW_Y, POST_X, HON_ROW_Y, color=C_ENV)

box(POST_X, HON_Y, POST_W, BH_H, fc=LN_SEL, ec=C_ENV, lw=1.5,
    label="Post Listing", sub1="\u2192 Market", label_color=C_ENV, bold=True)

# Honest listings arrow — stub right from Post box then arrow straight down
# x = 10.42, well right of all seller-lane content
HON_ARR_X = 10.42
stub(POST_X + POST_W, HON_ROW_Y, HON_ARR_X, HON_ROW_Y, color=C_ENV, lw=0.9)
arr(HON_ARR_X, HON_ROW_Y, HON_ARR_X, MKT_TOP + 0.04,
    color=C_ENV, lw=1.1, hw=0.11, hl=0.12, zorder=3)
# Badge ③ in the Honest Seller sub-row, right side — between HON_Y and HON_TOP
BADGE3_Y = HON_ROW_Y   # same height as honest row centre
badge(HON_ARR_X, BADGE3_Y, 3, C_ENV)
ax.text(HON_ARR_X + 0.14, BADGE3_Y - 0.24,
        "honest\nlistings",
        fontsize=6.0, color=C_ENV, ha="left", va="center",
        fontstyle="italic", zorder=6)

# ── Deceptive Principal — Tier Decision sub-row ───────────────────────────────
ENDOW_S_X, ENDOW_S_W = 0.68, 1.52
TIER_X,    TIER_W    = 2.38, 3.60
TIER_CX = TIER_X + TIER_W / 2

box(ENDOW_S_X, TIER_Y, ENDOW_S_W, BH_T, fc=LN_SYB, ec=C_SYB, lw=1.5,
    label="Endow Car", sub1="quality = poor (always)",
    label_color=C_SYB, bold=True)
arr(ENDOW_S_X + ENDOW_S_W, SYB_ROW1_Y, TIER_X, SYB_ROW1_Y, color=C_SYB)

box(TIER_X, TIER_Y, TIER_W, BH_T, fc=LN_AGT, ec=C_AGT, lw=1.9,
    label="LLM Call A: Tier Decision",
    sub1="In: true quality label, available tiers [poor, fair, good, mint]",
    sub2='Out: {"advertised_quality": "<tier strictly above true>"}',
    sub2_color="#1a3a1a", label_color=C_AGT, bold=True)

# SYS callout — between SUB_DIV_Y and TIER_TOP, centred in that gap
# Gap = SUB_DIV_Y - TIER_TOP = TIER_GAP_ABOVE = 0.44  ✓
TIER_CALLOUT_Y = (SUB_DIV_Y + TIER_TOP) / 2
callout(TIER_CX, TIER_CALLOUT_Y,
        'SYS: "Deceptive seller \u2014 maximise revenue by inflating advertised quality tier"',
        color=C_ORG, fs=6.1)
stub(TIER_CX, TIER_TOP, TIER_CX, TIER_CALLOUT_Y - 0.18, color=C_ORG)

# "always strictly above true tier" constraint
callout(7.46, SYB_ROW1_Y, "always strictly\nabove true tier", color=C_RED, fs=6.1)
stub(TIER_X + TIER_W + 0.06, SYB_ROW1_Y, 6.84, SYB_ROW1_Y, color=C_RED)

# ── Deceptive Principal — Description sub-row (K parallel calls) ──────────────
ID_W   = 2.50
ID_GAP = 0.22
# Three boxes: x0=0.68, x1=3.40, x2=6.12
ID_XS  = [0.68, 0.68 + ID_W + ID_GAP, 0.68 + 2 * (ID_W + ID_GAP)]

ID_SPECS = [
    ("LLM Call B \u2014 Identity S\u2081", "persona: cautious"),
    ("LLM Call B \u2014 Identity S\u2082", "persona: salesy"),
    ("LLM Call B \u2014 S\u2083 \u22ef S\u2096",  "persona: urgent / \u2026"),
]
ID_SUB2 = 'Out: {"description": "...", "price": N}'

# DESC_TOP already computed in geometry block above.
# Persona labels go just above desc boxes; K-label goes above personas.
# Gap available = TIER_Y - DESC_TOP = DESC_GAP_ABOVE = 0.36  ✓

for ix, (lbl, psub) in zip(ID_XS, ID_SPECS):
    box(ix, DESC_Y, ID_W, ID_H, fc=LN_AGT, ec=C_AGT, lw=1.5,
        label=lbl,
        sub1="In: adv. quality, identity rep, persona system prompt",
        sub2=ID_SUB2, sub2_color="#1a3a1a",
        label_color=C_AGT, bold=True, fs=7.5)
    # Persona label BELOW each box (va="top" so text hangs down from box bottom)
    ax.text(ix + ID_W / 2, DESC_Y - 0.05,
            psub, fontsize=6.0, color=C_SYB,
            ha="center", va="top", fontstyle="italic", zorder=5)

# "K calls in parallel" label — below persona labels, above SEL_BOT
# DESC_Y - 0.05 (persona top) - ~0.14 (text height) = DESC_Y - 0.19
# K-label at DESC_Y - 0.36, well above SEL_BOT (margin = 0.52)
ax.text((ID_XS[0] + ID_XS[-1] + ID_W) / 2, DESC_Y - 0.36,
        "K description calls run in parallel (ThreadPoolExecutor)",
        fontsize=6.2, color=C_SYB, ha="center", va="top",
        fontstyle="italic", zorder=6)

# Vertical arrow: Tier Decision → Description row top
arr(TIER_CX, TIER_Y, TIER_CX, DESC_TOP,
    color=C_SYB, lw=1.2, hw=0.10, hl=0.12)
ax.text(TIER_CX + 0.13, (TIER_Y + DESC_TOP) / 2,
        "adv. tier\n(shared)",
        fontsize=6.0, color=C_SYB, ha="left", va="center",
        fontstyle="italic", zorder=6)

# Identity Rotation callout — right of desc boxes, vertically centred in sybil half
ROT_CX = 11.10
ROT_Y  = (SYB_ROW1_Y + SYB_ROW2_Y) / 2
callout(ROT_CX, ROT_Y,
        "Identity Rotation  (step \u2467)\n"
        "If  R\u2096  <  \u03c1_min = 0.3 :\n"
        "  \u2514 retire  S\u2096\n"
        "  \u2514 spawn  S\u2096\u208a\u2081  ( R\u2080 = 0.8 )",
        color=C_RED, fs=6.2)
arr(ID_XS[-1] + ID_W, SYB_ROW2_Y, ROT_CX - 0.90, SYB_ROW2_Y,
    color=C_RED, lw=0.9, hw=0.09, hl=0.11)

# Sybil listings arrow — from DESC_Y straight down to MKT_TOP
# Use S3 box centre (rightmost) to separate from K-label text at centre
SYB_ARR_X = ID_XS[2] + ID_W / 2   # centre of S3 box ≈ 7.37
arr(SYB_ARR_X, DESC_Y, SYB_ARR_X, MKT_TOP + 0.04,
    color=C_SYB, lw=1.1, hw=0.11, hl=0.12, zorder=3)
ax.text(SYB_ARR_X + 0.13, (DESC_Y + MKT_TOP) / 2,
        "deceptive\nlistings",
        fontsize=6.0, color=C_SYB, ha="left", va="center",
        fontstyle="italic", zorder=6)

# =============================================================================
# LANE 2 — Market
# =============================================================================
STEP_GAP = 0.10
STEP_W   = (X_MAX - X_MIN - 7 * STEP_GAP) / 8

MKT_STEPS = [
    ("\u2460 Endow",       "cars allocated\nto sellers"),
    ("\u2461 List Phase",  "sellers\ncreate_listings()"),
    ("\u2462 Post",        "listings posted\nto market"),
    ("\u2463 Buyer Phase", "sample \u22645 listings\nper buyer"),
    ("\u2464 Clearing",    "match bids,\nfill orders"),
    ("\u2465 Review",      "LLM review\nvotes cast"),
    ("\u2466 Rep Update",  "EMA updated\nfrom votes"),
    ("\u2467 Rotation",    "retire low-R\nsybil identities"),
]

step_centres = []
for si, (lbl, sub) in enumerate(MKT_STEPS):
    bx = X_MIN + si * (STEP_W + STEP_GAP)
    box(bx, MKT_MID - BH_MKT / 2, STEP_W, BH_MKT,
        fc=LN_ENV, ec=C_ENV, lw=1.2,
        label=lbl, sub1=sub, label_color=C_ENV, bold=True, fs=7.6)
    step_centres.append(bx + STEP_W / 2)
    if si < len(MKT_STEPS) - 1:
        nx = X_MIN + (si + 1) * (STEP_W + STEP_GAP)
        arr(bx + STEP_W, MKT_MID, nx, MKT_MID,
            color=C_ENV, lw=1.0, hw=0.09, hl=0.10)

# Reputation EMA arc — bows DOWNWARD below the step boxes (rad=-0.20)
# Anchored at box_bottom - 0.04
MKT_BOX_BOT = MKT_MID - BH_MKT / 2          # = 4.48 - 0.43 = 4.05
REP_ARC_Y   = MKT_BOX_BOT - 0.04             # = 4.01
# Arc nadir ≈ REP_ARC_Y - span*rad/2 → stays above MKT_BOT 3.54  ✓

ax.annotate("",
            xy=(step_centres[0], REP_ARC_Y),
            xytext=(step_centres[6], REP_ARC_Y),
            arrowprops=dict(
                arrowstyle="-|>,head_width=0.12,head_length=0.13",
                color=C_ENV, lw=1.2,
                connectionstyle="arc3,rad=-0.20"),
            zorder=4)
# Label at arc nadir, inside Market lane (well above MKT_BOT 3.54)
ax.text((step_centres[6] + step_centres[0]) / 2, MKT_BOT + 0.36,
        "reputation EMA \u21ba  feeds into next-step listing observations",
        fontsize=6.1, color=C_ENV, ha="center", va="center",
        fontstyle="italic", zorder=5,
        bbox=dict(boxstyle="round,pad=0.10", fc="white",
                  ec=C_ENV, lw=0.5, alpha=0.92))

# Market ③ Post → Buyer Sample: dashed arc
POST_CX     = step_centres[2]
SAMP_CX_VAL = 0.68 + 1.52 / 2    # = 1.44, matches SAMP_X + SAMP_W/2 below
ax.annotate("",
            xy=(SAMP_CX_VAL, BUY_TOP - 0.04),
            xytext=(POST_CX, MKT_BOT - 0.04),
            arrowprops=dict(
                arrowstyle="-|>,head_width=0.10,head_length=0.12",
                color="#777777", lw=0.9,
                linestyle="dashed",
                connectionstyle="arc3,rad=-0.28"),
            zorder=4)
ax.text((POST_CX + SAMP_CX_VAL) / 2 - 0.18,
        BUY_TOP - 0.50,
        "listings\nvisible",
        fontsize=6.0, color="#555555", ha="right", va="center",
        fontstyle="italic", zorder=6)

# =============================================================================
# LANE 3 — Buyer Phase
# =============================================================================
# BUY_Y = 1.45  BUY_Y + BH_B = 2.35  BUY_ROW_Y = 1.90

SAMP_X, SAMP_W = 0.68, 1.52
BID_X,  BID_W  = 2.38, 3.60
ORD_X,  ORD_W  = 6.20, 1.70
REV_X,  REV_W  = 8.16, 3.72
REV_CX  = REV_X + REV_W / 2       # ≈ 10.02

# Sample Listings
box(SAMP_X, BUY_Y, SAMP_W, BH_B, fc=LN_BUY, ec=C_BUY, lw=1.5,
    label="Sample Listings",
    sub1="\u22645 visible (discovery limit)",
    label_color=C_BUY, bold=True)
arr(SAMP_X + SAMP_W, BUY_ROW_Y, BID_X, BUY_ROW_Y, color=C_BUY)
badge((SAMP_X + SAMP_W + BID_X) / 2, BUY_ROW_Y + 0.30, 4, C_BUY)

# LLM Call 1: Bid Decision
BID_CX = BID_X + BID_W / 2
box(BID_X, BUY_Y, BID_W, BH_B, fc=LN_AGT, ec=C_AGT, lw=1.9,
    label="LLM Call 1: Bid Decision",
    sub1="In: persona, market_mean_quality, txn_history[\u221210],",
    sub2="    listings \u00d7 \u22645  (description, price, seller_reputation)",
    sub2_color="#1a3a1a", label_color=C_AGT, bold=True)

# Output callout BELOW the Bid box
# BUY_Y = 1.01, callout at 1.01 - 0.42 = 0.59 > BUY_BOT 0.10  ✓
BID_CALLOUT_Y = BUY_Y - 0.42
callout(BID_CX, BID_CALLOUT_Y,
        'Out: {"decision": "bid" | "pass",  "listing_id": "<id or null>"}',
        color=C_ORG, fs=6.2)
stub(BID_CX, BUY_Y, BID_CX, BID_CALLOUT_Y + 0.20, color=C_ORG)

arr(BID_X + BID_W, BUY_ROW_Y, ORD_X, BUY_ROW_Y, color=C_BUY)
badge((BID_X + BID_W + ORD_X) / 2, BUY_ROW_Y + 0.30, 5, C_BUY)

# Submit Order
box(ORD_X, BUY_Y, ORD_W, BH_B, fc=LN_BUY, ec=C_BUY, lw=1.5,
    label="Submit Order", sub1="or pass this step",
    label_color=C_BUY, bold=True)
arr(ORD_X + ORD_W, BUY_ROW_Y, REV_X, BUY_ROW_Y, color=C_BUY)
badge((ORD_X + ORD_W + REV_X) / 2, BUY_ROW_Y + 0.30, 6, C_BUY)

# LLM Call 2: Post-Purchase Review
box(REV_X, BUY_Y, REV_W, BH_B, fc=LN_AGT, ec=C_AGT, lw=1.9,
    label="LLM Call 2: Post-Purchase Review",
    sub1="In: seller description, true quality label, quality_value",
    sub2='Out: {"vote": "upvote" | "downvote" | "abstain"}',
    sub2_color="#1a3a1a", label_color=C_AGT, bold=True)

# Cross-lane vote arrow → Market ⑦ Rep Update
# Arrow spans BUY_TOP to MKT_BOT (same boundary), badge inside Buyer lane
arr(REV_CX, BUY_TOP, REV_CX, MKT_BOT + 0.04,
    color=C_BUY, lw=1.1, hw=0.11, hl=0.12, zorder=3)
# Badge ⑦ inside Buyer lane at box-top level
BADGE7_Y = BUY_Y + BH_B + 0.28   # just above Review box top
badge(REV_CX, BADGE7_Y, 7, C_BUY)
ax.text(REV_CX + 0.14, BADGE7_Y - 0.22,
        "vote\n\u2192 rep EMA",
        fontsize=6.0, color=C_BUY, ha="left", va="center",
        fontstyle="italic", zorder=6)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
plt.savefig(OUT_PATH, bbox_inches="tight", format="pdf", dpi=150)
print(f"Saved: {OUT_PATH}")
