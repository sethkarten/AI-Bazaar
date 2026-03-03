"""Figure 1 Teaser — v10: Three-act narrative with pipeline center.
Fixes: sparkline artifacts (icon-based), rbox overlaps, zorder hierarchy,
text truncation, separator spines, sybil arc arrows, chart placement.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

ROOT    = os.path.join(os.path.dirname(__file__), "..")
ICONS   = os.path.join(ROOT, "icons")
OUT_PDF = os.path.join(ROOT, "teaser.pdf")

P = dict(
    red="#C0392B",   rfl="#FDF0EE", rmed="#E8B8B3",
    green="#1E8449", gfl="#EDF8F1", gmed="#A9D9B5",
    orange="#B7600A",ofl="#FEF9EF", omed="#F5C98A",
    purple="#6C3483",pfl="#F6EFF9",
    blue="#1A5276",  bfl="#EAF2FB",
    grey="#566573",  div="#C8CDD0",  bg="#FFFFFF",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def place_icon(ax, name, x, y, zoom=0.13, alpha=1.0, z=4):
    img = Image.open(os.path.join(ICONS, f"{name}.png")).convert("RGBA")
    ob  = OffsetImage(np.array(img), zoom=zoom)
    ob.image.axes = ax
    if alpha < 1.0:
        ob.image.set_alpha(alpha)
    ax.add_artist(AnnotationBbox(ob, (x, y), frameon=False, zorder=z, pad=0))

def rbox(ax, x, y, w, h, fc, ec, lw=1.4, alpha=1.0, r=0.14, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z))

def T(ax, x, y, s, fs=9, c="#1C2833", ha="center", va="center",
      bold=False, italic=False, z=6, family=None):
    kw = dict(fontsize=fs, color=c, ha=ha, va=va,
              fontweight="bold" if bold else "normal",
              fontstyle="italic" if italic else "normal", zorder=z)
    if family:
        kw["fontfamily"] = family
    ax.text(x, y, s, **kw)

def arr(ax, x0, y0, x1, y1, col, lw=1.4, hw=0.12, hl=0.13, conn="arc3,rad=0"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                                color=col, lw=lw, connectionstyle=conn), zorder=5)

def down_arr(ax, x, y, col, length=0.50, lw=1.8):
    ax.annotate("", xy=(x, y - length), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.10,head_length=0.11",
                                color=col, lw=lw), zorder=5)

def speech(ax, x, y, txt, fs=6.5):
    ax.text(x, y, txt, fontsize=fs, ha="center", va="center",
            color="#6E2F05", fontweight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", fc="#FFF8DC", ec="#E8A020",
                      lw=0.8, alpha=0.95))

def xmark(ax, x, y, col=None, s=0.18, lw=2.8):
    if col is None:
        col = P["red"]
    kw = dict(color=col, lw=lw, zorder=8, solid_capstyle="round")
    ax.plot([x - s, x + s], [y - s, y + s], **kw)
    ax.plot([x - s, x + s], [y + s, y - s], **kw)

# ── Canvas ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.4, 7.2), dpi=300)
ax  = fig.add_axes([0.012, 0.015, 0.976, 0.970])
ax.set_xlim(0, 14.4)
ax.set_ylim(0, 7.2)
ax.axis("off")
fig.patch.set_facecolor(P["bg"])
ax.set_facecolor(P["bg"])

# ── Column geometry ──────────────────────────────────────────────────────────
# Wider separators to prevent text overlap with panel borders
FAIL_L, FAIL_R = 0.12, 4.48
SEP1_L, SEP1_R = 4.48, 5.52       # 1.04 wide (was 0.84)
PIPE_L, PIPE_R = 5.52, 8.92
SEP2_L, SEP2_R = 8.92, 9.88       # 0.96 wide (was 0.80)
FIX_L,  FIX_R  = 9.88, 14.28

ROW_TOP = 7.08
ROW_MID = 3.50
ROW_BOT = 0.12

MID_FAIL = (FAIL_L + FAIL_R) / 2   # 2.30
MID_PIPE = (PIPE_L + PIPE_R) / 2   # 7.22
MID_FIX  = (FIX_L  + FIX_R) / 2    # 12.08

FAIL_W = FAIL_R - FAIL_L  # 4.36
PIPE_W = PIPE_R - PIPE_L  # 3.40
FIX_W  = FIX_R  - FIX_L   # 4.40

# ── Outer panel backgrounds (z=1) ───────────────────────────────────────────
rbox(ax, FAIL_L, ROW_BOT, FAIL_W, ROW_TOP - ROW_BOT,
     "#FDF2F1", P["red"], lw=2.0, r=0.20, z=1)
rbox(ax, PIPE_L, ROW_BOT, PIPE_W, ROW_TOP - ROW_BOT,
     "#FFFDF7", P["orange"], lw=2.0, r=0.20, z=1)
rbox(ax, FIX_L,  ROW_BOT, FIX_W,  ROW_TOP - ROW_BOT,
     "#F2FBF4", P["green"], lw=2.0, r=0.20, z=1)

# Separator strips — white background with colored border, rendered above panels
for sl, sr, fc, ec in [(SEP1_L, SEP1_R, "#FDFCFA", P["orange"]),
                        (SEP2_L, SEP2_R, "#F6FBF7", P["green"])]:
    rbox(ax, sl + 0.02, ROW_BOT + 0.06, sr - sl - 0.04, ROW_TOP - ROW_BOT - 0.12,
         fc, ec, lw=0.8, r=0.06, z=1.5, alpha=0.95)

# ── Header banners (z=3) ────────────────────────────────────────────────────
BANNER_Y = 6.52
BANNER_H = 0.48
rbox(ax, FAIL_L + 0.06, BANNER_Y, FAIL_W - 0.12, BANNER_H,
     P["rmed"], P["red"], lw=1.8, r=0.12, alpha=0.90, z=3)
T(ax, MID_FAIL, BANNER_Y + BANNER_H / 2, "(A) FAILURE MODES",
  fs=12, c=P["red"], bold=True)

rbox(ax, PIPE_L + 0.06, BANNER_Y, PIPE_W - 0.12, BANNER_H,
     P["omed"], P["orange"], lw=1.8, r=0.12, alpha=0.90, z=3)
T(ax, MID_PIPE, BANNER_Y + BANNER_H / 2, "(B) ALIGNMENT SFT",
  fs=12, c=P["orange"], bold=True)

rbox(ax, FIX_L + 0.06, BANNER_Y, FIX_W - 0.12, BANNER_H,
     P["gmed"], P["green"], lw=1.8, r=0.12, alpha=0.90, z=3)
T(ax, MID_FIX, BANNER_Y + BANNER_H / 2, "(C) ALIGNED AGENTS",
  fs=12, c=P["green"], bold=True)

# Row dividers inside failure and fix panels only
for x0, x1 in [(FAIL_L + 0.20, FAIL_R - 0.20), (FIX_L + 0.20, FIX_R - 0.20)]:
    ax.plot([x0, x1], [ROW_MID, ROW_MID], color=P["div"], lw=0.8,
            ls="--", zorder=3, alpha=0.45)

# ── Separator arrows & spine labels ─────────────────────────────────────────
# These now sit fully within the wider separator strips

# Separator 1: between A and B
cx1 = (SEP1_L + SEP1_R) / 2
ax.annotate("", xy=(SEP1_R - 0.16, ROW_MID), xytext=(SEP1_L + 0.16, ROW_MID),
            arrowprops=dict(arrowstyle="-|>,head_width=0.20,head_length=0.12",
                            color=P["orange"], lw=2.6), zorder=6)
T(ax, cx1, 5.50, "Economic",   fs=6.2, c=P["orange"], bold=True)
T(ax, cx1, 5.26, "Alignment",  fs=6.2, c=P["orange"], bold=True)
T(ax, cx1, 5.02, "Finetuning", fs=6.2, c=P["orange"], bold=True)

# Separator 2: between B and C
cx2 = (SEP2_L + SEP2_R) / 2
ax.annotate("", xy=(SEP2_R - 0.16, ROW_MID), xytext=(SEP2_L + 0.16, ROW_MID),
            arrowprops=dict(arrowstyle="-|>,head_width=0.20,head_length=0.12",
                            color=P["green"], lw=2.6), zorder=6)
T(ax, cx2, 5.40, "Aligned", fs=6.2, c=P["green"], bold=True)
T(ax, cx2, 5.16, "Agents",  fs=6.2, c=P["green"], bold=True)

# ═══════════════════════════════════════════════════════════════════════════════
# (A) B2C FAILURE: THE CRASH  (y: ROW_MID to BANNER_Y)
# ═══════════════════════════════════════════════════════════════════════════════

# Inner sub-panel (z=2): inset within the outer panel
B2C_INNER_BOT = ROW_MID + 0.12
B2C_INNER_TOP = BANNER_Y - 0.10
B2C_INNER_H   = B2C_INNER_TOP - B2C_INNER_BOT  # ~2.80
rbox(ax, FAIL_L + 0.12, B2C_INNER_BOT, FAIL_W - 0.24, B2C_INNER_H,
     P["rfl"], P["red"], lw=1.0, r=0.12, z=2)

# Title and subtitle
T(ax, MID_FAIL, 6.20, "B2C: The Crash", fs=10.5, c=P["red"], bold=True)
T(ax, MID_FAIL, 5.96, "Myopic undercutting \u2192 price spiral",
  fs=7.5, c=P["red"], italic=True)

# 3+2 staggered skyline of crashing buildings (left side of B2C panel)
b2c_row1_xs = [FAIL_L + 0.50, FAIL_L + 1.16, FAIL_L + 1.82]
b2c_row2_xs = [FAIL_L + 0.83, FAIL_L + 1.49]
for bx in b2c_row1_xs:
    place_icon(ax, "building_crash", bx, 5.38, zoom=0.100)
    down_arr(ax, bx, 4.96, P["red"], length=0.34)
for bx in b2c_row2_xs:
    place_icon(ax, "building_crash", bx, 4.82, zoom=0.100)
    down_arr(ax, bx, 4.40, P["red"], length=0.34)

# Crash chart icon — positioned in the right portion of the B2C panel,
# with clearance above the collapse bar.
# Collapse bar top = ROW_MID + 0.14 + 0.40 = 4.04.
CHART_X = MID_FAIL + 1.16
CHART_BOT = 4.20       # safe clearance above collapse bar top
CHART_H   = 1.02
CHART_Y   = CHART_BOT + CHART_H / 2
rbox(ax, CHART_X - 0.76, CHART_BOT, 1.52, CHART_H,
     "#FEF0EE", P["red"], lw=0.8, r=0.06, alpha=0.85, z=3)
place_icon(ax, "crash_chart_v2", CHART_X, CHART_Y, zoom=0.19, z=4)
T(ax, CHART_X, CHART_BOT - 0.10, "price trajectory",
  fs=5.8, c=P["red"], italic=True)

# Collapse status bar
COLLAPSE_Y = ROW_MID + 0.14
COLLAPSE_H = 0.40
rbox(ax, FAIL_L + 0.18, COLLAPSE_Y, FAIL_W - 0.36, COLLAPSE_H,
     P["rmed"], P["red"], lw=1.4, r=0.06, alpha=0.92, z=3)
T(ax, MID_FAIL, COLLAPSE_Y + COLLAPSE_H / 2,
  "MARKET COLLAPSE  ( P < c )", fs=7.8, c=P["red"], bold=True)

# ═══════════════════════════════════════════════════════════════════════════════
# (A) C2C FAILURE: THE LEMON MARKET  (y: ROW_BOT to ROW_MID)
# ═══════════════════════════════════════════════════════════════════════════════

C2C_INNER_BOT = ROW_BOT + 0.06
C2C_INNER_TOP = ROW_MID - 0.12
C2C_INNER_H   = C2C_INNER_TOP - C2C_INNER_BOT
rbox(ax, FAIL_L + 0.12, C2C_INNER_BOT, FAIL_W - 0.24, C2C_INNER_H,
     P["pfl"], P["purple"], lw=1.0, r=0.12, z=2)

T(ax, MID_FAIL, 3.14, "C2C: The Lemon Market", fs=10.5, c=P["purple"], bold=True)
T(ax, MID_FAIL, 2.92, "Sybil cluster floods market with fakes",
  fs=7.5, c=P["purple"], italic=True)

# Puppet master (left)
place_icon(ax, "person_puppetmaster", FAIL_L + 0.52, 2.16, zoom=0.115)
T(ax, FAIL_L + 0.52, 1.62, "Sybil\nPrincipal", fs=6.8, c=P["red"], bold=True)

# Sybil arc — 5 identities fanning from puppet master
sybil_xs = [FAIL_L + 1.22, FAIL_L + 1.66, FAIL_L + 2.10, FAIL_L + 2.54, FAIL_L + 2.98]
sybil_ys = [2.06, 2.16, 2.22, 2.16, 2.06]
for k, (sx, sy) in enumerate(zip(sybil_xs, sybil_ys)):
    place_icon(ax, "person_sybil", sx, sy, zoom=0.084, alpha=0.80)
    T(ax, sx, sy - 0.46, f"$S_{{{k+1}}}$", fs=6.2, c=P["purple"])
    speech(ax, sx, sy + 0.42, '"Mint!"', fs=5.5)
    # Fan arrows: symmetric rads centered on 0
    rad = -0.10 + k * 0.05
    arr(ax, FAIL_L + 0.86, 2.18, sx - 0.10, sy - 0.05, P["purple"],
        lw=0.65, hw=0.05, hl=0.06, conn=f"arc3,rad={rad:.2f}")

# Deceived buyer (right)
place_icon(ax, "person_buyer", FAIL_R - 0.42, 2.16, zoom=0.110)
T(ax, FAIL_R - 0.42, 1.62, "Buyer\n(deceived)", fs=6.8, c=P["grey"])
arr(ax, sybil_xs[-1] + 0.14, 2.10, FAIL_R - 0.66, 2.16,
    P["purple"], lw=1.0, hw=0.08, hl=0.09)

# Forensic miss callout (bottom of C2C panel)
FORENSIC_FAIL_Y = C2C_INNER_BOT + 0.06
FORENSIC_FAIL_H = 0.68
rbox(ax, FAIL_L + 0.18, FORENSIC_FAIL_Y, FAIL_W - 0.36, FORENSIC_FAIL_H,
     P["pfl"], P["purple"], lw=1.0, r=0.06, alpha=0.95, z=3)
place_icon(ax, "magnifier_forensic", FAIL_L + 0.52, FORENSIC_FAIL_Y + 0.40, zoom=0.078)
T(ax, FAIL_L + 0.52, FORENSIC_FAIL_Y + 0.14, "undetected",
  fs=5.8, c=P["purple"], italic=True)
T(ax, FAIL_L + 1.22, FORENSIC_FAIL_Y + 0.46,
  "\u2460 text similarity: 5 listings near-identical",
  fs=6.5, c=P["purple"], ha="left")
T(ax, FAIL_L + 1.22, FORENSIC_FAIL_Y + 0.22,
  "\u2461 Rep=0.9 but first-time seller",
  fs=6.5, c=P["purple"], ha="left")

# ═══════════════════════════════════════════════════════════════════════════════
# (B) ALIGNMENT FINETUNING PIPELINE  (full height)
# ═══════════════════════════════════════════════════════════════════════════════

# Step 1: Simulate episodes
place_icon(ax, "pipeline_traces", MID_PIPE, 5.86, zoom=0.105)
T(ax, MID_PIPE, 5.40, "\u2460 Simulate Episodes", fs=8.8, c=P["orange"], bold=True)
T(ax, MID_PIPE, 5.20, "100 ep. \u00d7 5 model families", fs=6.8, c=P["orange"], italic=True)

down_arr(ax, MID_PIPE, 4.98, P["orange"], length=0.28, lw=1.6)

# Step 2: Filter by EAS
STEP2_Y = 4.30
STEP2_H = 0.48
rbox(ax, PIPE_L + 0.18, STEP2_Y, PIPE_W - 0.36, STEP2_H,
     "#FFF3DC", P["orange"], lw=1.0, r=0.08, z=3)
place_icon(ax, "pipeline_filter", PIPE_L + 0.44, STEP2_Y + STEP2_H / 2, zoom=0.076)
T(ax, MID_PIPE + 0.20, STEP2_Y + 0.30,
  "\u2461 Filter: top-10% EAS", fs=8.0, c=P["orange"], bold=True, ha="left")
T(ax, MID_PIPE + 0.20, STEP2_Y + 0.14,
  "Economic Alignment Score", fs=6.5, c=P["orange"], italic=True, ha="left")

down_arr(ax, MID_PIPE, STEP2_Y, P["orange"], length=0.28, lw=1.6)

# Step 3: SFT via LoRA
STEP3_Y = 3.62
STEP3_H = 0.48
rbox(ax, PIPE_L + 0.18, STEP3_Y, PIPE_W - 0.36, STEP3_H,
     "#FFF3DC", P["orange"], lw=1.0, r=0.08, z=3)
place_icon(ax, "brain_lora", PIPE_L + 0.44, STEP3_Y + STEP3_H / 2, zoom=0.080)
T(ax, MID_PIPE + 0.20, STEP3_Y + 0.30,
  "\u2462 SFT via LoRA", fs=8.4, c=P["orange"], bold=True, ha="left")
T(ax, MID_PIPE + 0.20, STEP3_Y + 0.14,
  "r=16, \u03b1=32, 3 epochs", fs=6.5, c=P["orange"], italic=True, ha="left")

down_arr(ax, MID_PIPE, STEP3_Y, P["orange"], length=0.26, lw=1.6)

# Step 4: Aligned agent outputs (two side-by-side mini-boxes)
STEP4_Y = 2.92
STEP4_H = 0.52
half_w  = (PIPE_W - 0.56) / 2  # ~1.42

# B2C track (left)
rbox(ax, PIPE_L + 0.18, STEP4_Y, half_w, STEP4_H,
     P["gfl"], P["green"], lw=1.0, r=0.06, z=3)
place_icon(ax, "building_stable", PIPE_L + 0.42, STEP4_Y + 0.26, zoom=0.068)
T(ax, PIPE_L + 0.94, STEP4_Y + 0.34, "Stabilizing",
  fs=7.2, c=P["green"], bold=True, ha="left")
T(ax, PIPE_L + 0.94, STEP4_Y + 0.16, "Firm (B2C)",
  fs=6.2, c=P["green"], ha="left")

# C2C track (right)
cx2c = PIPE_L + 0.18 + half_w + 0.20
rbox(ax, cx2c, STEP4_Y, half_w, STEP4_H,
     P["bfl"], P["blue"], lw=1.0, r=0.06, z=3)
place_icon(ax, "shield_person", cx2c + 0.22, STEP4_Y + 0.26, zoom=0.074)
T(ax, cx2c + 0.74, STEP4_Y + 0.34, "Skeptical",
  fs=7.2, c=P["blue"], bold=True, ha="left")
T(ax, cx2c + 0.74, STEP4_Y + 0.16, "Guardian (C2C)",
  fs=6.2, c=P["blue"], ha="left")

T(ax, MID_PIPE, STEP4_Y - 0.12, "\u2463 Aligned Agents",
  fs=8.2, c=P["orange"], bold=True)

down_arr(ax, MID_PIPE, STEP4_Y - 0.22, P["orange"], length=0.20, lw=1.4)

# Training format callout (bottom of pipeline)
FMT_Y = ROW_BOT + 0.10
FMT_H = 1.94
rbox(ax, PIPE_L + 0.14, FMT_Y, PIPE_W - 0.28, FMT_H,
     "#FFFAF0", P["orange"], lw=0.8, r=0.08, alpha=0.95, z=3)
T(ax, MID_PIPE, FMT_Y + FMT_H - 0.14, "Training Data Format",
  fs=7.2, c=P["orange"], bold=True)

# Code lines in monospace
code_lines = [
    ("[SYSTEM]  role description",            P["grey"]),
    ("[OBS]     market state JSON",           P["grey"]),
    ("[THINK]   chain-of-thought",            P["grey"]),
    ('[ACTION]  {"price":P, "qty":Q}',        P["blue"]),
    ("EAS \u2265 top-10% \u2192 SFT target",          P["orange"]),
]
line_h  = 0.26
line_y0 = FMT_Y + 0.14
for i, (line, col) in enumerate(code_lines):
    ly = line_y0 + i * line_h
    rbox(ax, PIPE_L + 0.22, ly, PIPE_W - 0.44, 0.20,
         "#FFFFF5" if i % 2 == 0 else "#FFF8E0", P["orange"],
         lw=0.3, r=0.03, alpha=0.50, z=4)
    ax.text(PIPE_L + 0.32, ly + 0.10, line,
            fontsize=5.8, color=col, va="center", ha="left",
            fontfamily="monospace", zorder=6)

# ═══════════════════════════════════════════════════════════════════════════════
# (C) B2C FIX: STABILIZING FIRMS  (y: ROW_MID to BANNER_Y)
# ═══════════════════════════════════════════════════════════════════════════════

FIX_B2C_BOT = ROW_MID + 0.12
FIX_B2C_TOP = BANNER_Y - 0.10
FIX_B2C_H   = FIX_B2C_TOP - FIX_B2C_BOT
rbox(ax, FIX_L + 0.12, FIX_B2C_BOT, FIX_W - 0.24, FIX_B2C_H,
     P["gfl"], P["green"], lw=1.0, r=0.12, z=2)

T(ax, MID_FIX, 6.20, "B2C: Stabilizing Firms", fs=10.5, c=P["green"], bold=True)
T(ax, MID_FIX, 5.96, "Price floor held above unit cost",
  fs=7.5, c=P["green"], italic=True)

# 4 stable buildings — spread across the full panel width
sf_xs = [FIX_L + 0.52, FIX_L + 1.38, FIX_L + 2.24, FIX_L + 3.10]
for sx in sf_xs:
    place_icon(ax, "building_stable", sx, 5.36, zoom=0.100)

# Dashed price floor line below buildings
ax.plot([FIX_L + 0.26, FIX_L + 3.36], [4.78, 4.78],
        color=P["green"], ls="--", lw=2.0, zorder=4, dash_capstyle="round")
T(ax, FIX_L + 3.56, 4.78, r"$P \geq c$", fs=8.5, c=P["green"], bold=True)

# Stability chart icon — right side of panel, well above the EQ bar
# EQ bar top = ROW_MID + 0.14 + 0.40 = 4.04. Chart starts above that.
SCHART_BOT = 4.18
SCHART_H   = 0.82
SCHART_X   = MID_FIX + 1.16
SCHART_Y   = SCHART_BOT + SCHART_H / 2
rbox(ax, SCHART_X - 0.56, SCHART_BOT, 1.12, SCHART_H,
     "#EDF8F1", P["green"], lw=0.8, r=0.06, alpha=0.85, z=3)
place_icon(ax, "stability_chart", SCHART_X, SCHART_Y, zoom=0.155, z=4)
T(ax, SCHART_X, SCHART_BOT - 0.10, "price stability",
  fs=5.8, c=P["green"], italic=True)

# Equilibrium status bar
EQ_Y = ROW_MID + 0.14
EQ_H = 0.40
rbox(ax, FIX_L + 0.18, EQ_Y, FIX_W - 0.36, EQ_H,
     P["gmed"], P["green"], lw=1.4, r=0.06, alpha=0.92, z=3)
T(ax, MID_FIX, EQ_Y + EQ_H / 2,
  "EQUILIBRIUM MAINTAINED  ( Nash )", fs=7.8, c=P["green"], bold=True)

# ═══════════════════════════════════════════════════════════════════════════════
# (C) C2C FIX: SKEPTICAL GUARDIANS  (y: ROW_BOT to ROW_MID)
# ═══════════════════════════════════════════════════════════════════════════════

FIX_C2C_BOT = ROW_BOT + 0.06
FIX_C2C_TOP = ROW_MID - 0.12
FIX_C2C_H   = FIX_C2C_TOP - FIX_C2C_BOT
rbox(ax, FIX_L + 0.12, FIX_C2C_BOT, FIX_W - 0.24, FIX_C2C_H,
     P["bfl"], P["blue"], lw=1.0, r=0.12, z=2)

T(ax, MID_FIX, 3.14, "C2C: Skeptical Guardians", fs=10.5, c=P["blue"], bold=True)
T(ax, MID_FIX, 2.92, "Forensic analysis detects & rejects Sybils",
  fs=7.5, c=P["blue"], italic=True)

# Ghost sybils — matching arc layout
ghost_xs = [FIX_L + 0.52, FIX_L + 0.98, FIX_L + 1.44, FIX_L + 1.90, FIX_L + 2.36]
ghost_ys = [2.06, 2.16, 2.22, 2.16, 2.06]
for gx, gy in zip(ghost_xs, ghost_ys):
    rbox(ax, gx - 0.18, gy - 0.30, 0.36, 0.46,
         P["bfl"], P["red"], lw=0.8, alpha=0.42, r=0.04, z=3)
    place_icon(ax, "person_sybil", gx, gy, zoom=0.078, alpha=0.35)
    xmark(ax, gx, gy, s=0.16, lw=2.4)

# Guardian — prominent on right side
place_icon(ax, "shield_person", FIX_L + 3.34, 2.22, zoom=0.185)
T(ax, FIX_L + 3.34, 1.62, "Skeptical\nGuardian", fs=7.2, c=P["blue"], bold=True)

# Arrow: sybil cluster to guardian
arr(ax, ghost_xs[-1] + 0.18, 2.14, FIX_L + 3.00, 2.22,
    P["blue"], lw=1.2, hw=0.09, hl=0.10)

# Forensic detection callout
FORENSIC_FIX_Y = FIX_C2C_BOT + 0.48
FORENSIC_FIX_H = 0.56
rbox(ax, FIX_L + 0.18, FORENSIC_FIX_Y, FIX_W - 0.36, FORENSIC_FIX_H,
     "#E8F4FD", P["blue"], lw=1.0, r=0.06, alpha=0.95, z=3)
place_icon(ax, "magnifier_forensic", FIX_L + 0.50, FORENSIC_FIX_Y + 0.34, zoom=0.078)
T(ax, FIX_L + 0.50, FORENSIC_FIX_Y + 0.12, "detected",
  fs=5.8, c=P["blue"], bold=True, italic=True)
T(ax, FIX_L + 1.22, FORENSIC_FIX_Y + 0.38,
  "\u2460 cosine sim > \u03b8 \u2014 coordinated!",
  fs=6.5, c=P["blue"], ha="left")
T(ax, FIX_L + 1.22, FORENSIC_FIX_Y + 0.16,
  "\u2461 Rep + Mint price \u2014 mismatch!",
  fs=6.5, c=P["blue"], ha="left")

# Rejection status bar
REJ_Y = FIX_C2C_BOT + 0.06
REJ_H = 0.38
rbox(ax, FIX_L + 0.18, REJ_Y, FIX_W - 0.36, REJ_H,
     "#D4E6F1", P["blue"], lw=1.4, r=0.06, alpha=0.92, z=3)
T(ax, MID_FIX, REJ_Y + REJ_H / 2,
  "SYBIL CLUSTER REJECTED", fs=7.8, c=P["blue"], bold=True)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PDF, bbox_inches="tight", format="pdf", dpi=300)
print("[v10] Saved ->", OUT_PDF)
