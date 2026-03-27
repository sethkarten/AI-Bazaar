"""
Architecture Diagram: "The Crash" (Experiment 1 — Environment A)

Single-panel architecture figure illustrating the feedback loop of the B2C
multi-agent simulation used in Agent Bazaar (COLM 2026).

Three-column layout:
  Left:   Consumers  (CES, WTP, demand orders)
  Center: Market / Auctioneer  (price-priority clearing, Ledger, Bankruptcy)
  Right:  LLM Firms  (Standard + Stabilizing)

A large observation/feedback arc at the bottom closes the simulation loop.

Usage:
    python exp1_architecture.py [--output paper/fig/arch/exp1_architecture.pdf]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.5,
    "lines.markersize":   5,
    "axes.linewidth":     0.8,
    "axes.grid":          False,
    "axes.axisbelow":     True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex":        False,
})

# ---------------------------------------------------------------------------
# Color palette (Okabe-Ito + semantic)
# ---------------------------------------------------------------------------
C_LLM_DARK    = "#0072B2"   # Blue — Standard LLM
C_LLM_LIGHT   = "#D6EEFA"
C_STAB_DARK   = "#009E73"   # Green — Stabilizing
C_STAB_LIGHT  = "#C8EDDF"
C_MARKET_DARK = "#B07D00"   # Amber — Market
C_MARKET_FILL = "#FFF5CC"
C_MARKET_EDGE = "#D4A800"
C_CONSUMER    = "#8B3A8B"   # Purple — Consumers
C_CONSUMER_LT = "#F2E4F2"
C_CRASH_RED   = "#C44000"   # Vermillion — Crash risk
C_CRASH_LIGHT = "#FDDCC8"
C_OBS_DARK    = "#5533AA"   # Indigo — Observation / feedback
C_OBS_LIGHT   = "#EEE6FF"
C_BG          = "#F8F8FC"
C_DIVIDER     = "#CCCCDD"
C_LEDGER_DK   = "#806600"


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def rbox(ax, cx, cy, w, h, fc, ec, lw=1.2, radius=0.03, alpha=1.0, zorder=3):
    """Centered rounded box."""
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, fs=8.5, c="#222222", ha="center", va="center",
        fw="normal", style="normal", zorder=5, ls=1.3):
    return ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va,
                   fontweight=fw, style=style, zorder=zorder,
                   linespacing=ls)


def arrw(ax, x0, y0, x1, y1, c=C_LLM_DARK, lw=1.4, rad=0.0,
         astyle="->", zorder=5, lbl=None, lbl_c=None, lbl_fs=7.0,
         lbl_dx=0.0, lbl_dy=0.015):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=astyle, color=c, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=zorder,
    )
    if lbl:
        mx = (x0 + x1) / 2 + lbl_dx
        my = (y0 + y1) / 2 + lbl_dy
        ax.text(mx, my, lbl, fontsize=lbl_fs, color=lbl_c or c,
                ha="center", va="bottom", style="italic", zorder=zorder + 1)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure() -> plt.Figure:
    W, H = 12.0, 7.0
    fig = plt.figure(figsize=(W, H), facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ===== TITLE =====
    txt(ax, 0.5, 0.965,
        "Agent Bazaar  \u2014  Environment A: The Crash  (B2C Market)",
        fs=12, c="#111122", fw="bold")
    txt(ax, 0.5, 0.937,
        "Partially Observable Stochastic Game  \u2022  N = 5 Firms  \u2022  M = 50 CES Consumers  \u2022  T = 365 timesteps",
        fs=7.8, c="#555566", style="italic")

    # ===== COLUMN SEPARATORS =====
    for xv in [0.265, 0.555]:
        ax.plot([xv, xv], [0.07, 0.91], color=C_DIVIDER, lw=0.9, ls="--",
                alpha=0.7, zorder=1)

    # ===== COLUMN HEADER BADGES =====
    header_y = 0.903
    for cx, label_txt, fc, ec in [
        (0.132, "CONSUMERS",        C_CONSUMER_LT, C_CONSUMER),
        (0.41,  "MARKET CLEARING",  C_MARKET_FILL, C_MARKET_EDGE),
        (0.775, "LLM FIRMS",        C_LLM_LIGHT,   C_LLM_DARK),
    ]:
        rbox(ax, cx, header_y, 0.13, 0.038, fc, ec, lw=1.0, radius=0.015, zorder=6)
        txt(ax, cx, header_y, label_txt, fs=8, c=ec, fw="bold", zorder=7)

    # ============================================================
    # LEFT COLUMN: CONSUMERS
    # ============================================================
    col_cx = 0.132

    # --- CES Consumer ---
    cy1 = 0.80
    rbox(ax, col_cx, cy1, 0.20, 0.10, C_CONSUMER_LT, C_CONSUMER, lw=1.4, radius=0.03)
    txt(ax, col_cx, cy1 + 0.030, "CES Consumer", fs=9.5, c=C_CONSUMER, fw="bold")
    txt(ax, col_cx, cy1 - 0.001, "N = 50 agents", fs=7.5, c="#666680")
    txt(ax, col_cx, cy1 - 0.025, "prefs $\\alpha_i$ \u2022 income $w_i$", fs=7.5, c="#555566")

    # --- WTP Computation ---
    cy2 = 0.638
    rbox(ax, col_cx, cy2, 0.20, 0.085, "#F3EAFF", C_CONSUMER, lw=1.0, radius=0.025)
    txt(ax, col_cx, cy2 + 0.023, "WTP Computation", fs=8.5, c=C_CONSUMER, fw="bold")
    txt(ax, col_cx, cy2 - 0.013, "$\\hat{p}^i = $ CES demand($\\alpha_i, w_i, P_t$)", fs=7.2, c="#555566")

    arrw(ax, col_cx, cy1 - 0.050, col_cx, cy2 + 0.043,
         c=C_CONSUMER, lw=1.3, lbl="budget + prefs", lbl_dx=0.045, lbl_dy=0.004)

    # --- Demand Bids ---
    cy3 = 0.495
    rbox(ax, col_cx, cy3, 0.20, 0.08, C_CONSUMER_LT, C_CONSUMER, lw=1.1, radius=0.025)
    txt(ax, col_cx, cy3 + 0.022, "Demand Bids", fs=8.5, c=C_CONSUMER, fw="bold")
    txt(ax, col_cx, cy3 - 0.005, "Order: $(good,\\ Q,\\ \\hat{p}^i)$", fs=7.5, c="#555566")

    arrw(ax, col_cx, cy2 - 0.042, col_cx, cy3 + 0.040,
         c=C_CONSUMER, lw=1.3, lbl="submit order", lbl_dx=0.045, lbl_dy=0.004)

    # --- Search Friction box ---
    cy4 = 0.360
    rbox(ax, col_cx, cy4, 0.20, 0.075, "#FFF5D6", "#C09000", lw=1.0, radius=0.025)
    txt(ax, col_cx, cy4 + 0.020, "Search Friction", fs=8, c="#8B6000", fw="bold")
    txt(ax, col_cx, cy4 - 0.005, "token budget $\\tau \\ll |S_t|$", fs=7.2, c="#8B6000")
    txt(ax, col_cx, cy4 - 0.025, "partial observability", fs=7.0, c="#9B7010", style="italic")

    # M/T annotation
    txt(ax, col_cx, 0.265, "M = 50 consumers\n365 timesteps", fs=7.2, c="#555555",
        zorder=8, ls=1.5)

    # ============================================================
    # CENTER COLUMN: MARKET
    # ============================================================
    col_mx = 0.41

    # --- Main Auctioneer box ---
    my1 = 0.715
    rbox(ax, col_mx, my1, 0.22, 0.21, C_MARKET_FILL, C_MARKET_EDGE, lw=1.8, radius=0.04)
    txt(ax, col_mx, my1 + 0.082, "Auctioneer", fs=10.5, c=C_MARKET_DARK, fw="bold")
    txt(ax, col_mx, my1 + 0.052, "Market Clearing", fs=8, c=C_MARKET_DARK)
    ax.plot([col_mx - 0.087, col_mx + 0.087], [my1 + 0.034, my1 + 0.034],
            color=C_MARKET_EDGE, lw=0.7, alpha=0.7, zorder=5)

    steps = [
        "1. Sort quotes by price $P_t^i$",
        "2. Fill demand, lowest price first",
        "3. Transfer goods + cash",
        "4. Update reputation $R_t^i$",
    ]
    for k, s in enumerate(steps):
        txt(ax, col_mx, my1 + 0.012 - k * 0.030, s, fs=7.5, c="#555500")

    # Orders arrow: consumers → market
    arrw(ax, col_cx + 0.100, cy3,
         col_mx - 0.110, my1 - 0.05,
         c=C_CONSUMER, lw=1.6, rad=-0.2,
         lbl="orders", lbl_c=C_CONSUMER, lbl_dy=-0.035, lbl_dx=-0.02)

    # --- Ledger ---
    my2 = 0.485
    rbox(ax, col_mx, my2, 0.20, 0.085, "#FFFBE8", C_LEDGER_DK, lw=1.1, radius=0.025)
    txt(ax, col_mx, my2 + 0.025, "Ledger", fs=9, c=C_LEDGER_DK, fw="bold")
    txt(ax, col_mx, my2 - 0.005, "Cash $C_t^i$  \u2022  Inventory $I_t^i$", fs=7.5, c="#666600")
    txt(ax, col_mx, my2 - 0.028, "updated every timestep", fs=7.0, c="#888800", style="italic")

    arrw(ax, col_mx, my1 - 0.105, col_mx, my2 + 0.043,
         c=C_LEDGER_DK, lw=1.3, lbl="record", lbl_dx=0.032, lbl_dy=0.003)

    # --- Overhead / Taxes ---
    my3 = 0.350
    rbox(ax, col_mx, my3, 0.22, 0.075, C_CRASH_LIGHT, C_CRASH_RED, lw=1.1, radius=0.025)
    txt(ax, col_mx, my3 + 0.021, "Overhead + Taxes", fs=8.5, c=C_CRASH_RED, fw="bold")
    txt(ax, col_mx, my3 - 0.005, "paid every step  (fixed cost)", fs=7.2, c=C_CRASH_RED)
    txt(ax, col_mx, my3 - 0.027, "$C_t^i < 0 \\;\\Rightarrow\\;$ bankrupt", fs=7.2, c=C_CRASH_RED, fw="bold")

    arrw(ax, col_mx, my2 - 0.043, col_mx, my3 + 0.038,
         c=C_CRASH_RED, lw=1.3, lbl="deduct", lbl_dx=0.032, lbl_dy=0.003)

    # --- Bankruptcy Check ---
    my4 = 0.225
    rbox(ax, col_mx, my4, 0.20, 0.07, "#FFEDED", C_CRASH_RED, lw=1.2, radius=0.025)
    txt(ax, col_mx, my4 + 0.018, "Bankruptcy Check", fs=8.5, c=C_CRASH_RED, fw="bold")
    txt(ax, col_mx, my4 - 0.010, "$C_t^i < 0 \\;\\Rightarrow\\;$ firm exits", fs=7.2, c=C_CRASH_RED)

    arrw(ax, col_mx, my3 - 0.038, col_mx, my4 + 0.035,
         c=C_CRASH_RED, lw=1.3)

    # Exit arrow
    ax.annotate(
        "exit", xy=(col_mx + 0.135, my4),
        xytext=(col_mx + 0.100, my4),
        fontsize=7.2, color=C_CRASH_RED,
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=C_CRASH_RED, lw=1.0),
        zorder=6,
    )

    # Goods + surplus back to consumers
    arrw(ax, col_mx - 0.110, my1 - 0.07,
         col_cx + 0.100, cy3 - 0.01,
         c=C_MARKET_DARK, lw=1.4, rad=0.25,
         lbl="goods + surplus", lbl_c=C_MARKET_DARK, lbl_dy=-0.055, lbl_dx=-0.03)

    # ============================================================
    # RIGHT COLUMN: FIRMS
    # ============================================================
    col_fx = 0.775

    # ---- Standard Firm ----
    fy1 = 0.775
    rbox(ax, col_fx, fy1, 0.24, 0.17, C_LLM_LIGHT, C_LLM_DARK, lw=1.5, radius=0.035)
    txt(ax, col_fx, fy1 + 0.063, "Standard Firm", fs=10, c=C_LLM_DARK, fw="bold")
    txt(ax, col_fx, fy1 + 0.038, "persona: competitive", fs=7.5, c="#446688", style="italic")

    # LLM badge inside standard firm
    rbox(ax, col_fx, fy1 - 0.005, 0.185, 0.065, "#E8F4FF", C_LLM_DARK, lw=1.0, radius=0.02, zorder=6)
    txt(ax, col_fx, fy1 + 0.015, "\u25b6  LLM Inference", fs=8.5, c=C_LLM_DARK, fw="bold", zorder=7)
    txt(ax, col_fx, fy1 - 0.012, "decide  $(P_t^i,\\; Q_t^i)$", fs=7.5, c=C_LLM_DARK, zorder=7)

    # Crash badge
    fy1b = fy1 - 0.110
    rbox(ax, col_fx, fy1b, 0.24, 0.055, C_CRASH_LIGHT, C_CRASH_RED, lw=1.0, radius=0.02)
    txt(ax, col_fx, fy1b + 0.010, "Price Spiral Risk", fs=8, c=C_CRASH_RED, fw="bold")
    txt(ax, col_fx, fy1b - 0.013, "$P_t^i \\to P_t^i < c$  (undercut loop)", fs=7.2, c=C_CRASH_RED)

    # ---- Stabilizing Firm ----
    fy2 = 0.50
    rbox(ax, col_fx, fy2, 0.24, 0.18, C_STAB_LIGHT, C_STAB_DARK, lw=1.6, radius=0.035)
    txt(ax, col_fx, fy2 + 0.068, "Stabilizing Firm", fs=10, c=C_STAB_DARK, fw="bold")
    txt(ax, col_fx, fy2 + 0.042, "alignment harness", fs=7.5, c="#226644", style="italic")

    # LLM badge inside stabilizing firm
    rbox(ax, col_fx, fy2 - 0.001, 0.185, 0.065, "#D8F5E8", C_STAB_DARK, lw=1.0, radius=0.02, zorder=6)
    txt(ax, col_fx, fy2 + 0.018, "\u25b6  LLM Inference", fs=8.5, c=C_STAB_DARK, fw="bold", zorder=7)
    txt(ax, col_fx, fy2 - 0.010, "decide  $(P_t^i,\\; Q_t^i)$", fs=7.5, c=C_STAB_DARK, zorder=7)

    # Price floor badge
    fy2b = fy2 - 0.118
    rbox(ax, col_fx, fy2b, 0.24, 0.055, "#E0F7EE", C_STAB_DARK, lw=1.0, radius=0.02)
    txt(ax, col_fx, fy2b + 0.010, "Price Floor Clamp", fs=8, c=C_STAB_DARK, fw="bold")
    txt(ax, col_fx, fy2b - 0.013, "$P_t^i \\leftarrow \\max(P_t^i,\\; c)$", fs=7.2, c=C_STAB_DARK)

    # N firms annotation
    txt(ax, col_fx, 0.285,
        "N = 5 firms per run\n(k stabilizing, 5\u2212k standard)",
        fs=7.2, c="#444444", zorder=8, ls=1.5)

    # ---- Quotes: firms → market ----
    arrw(ax, col_fx - 0.120, fy1 - 0.01,
         col_mx + 0.110, my1 + 0.03,
         c=C_LLM_DARK, lw=1.6, rad=0.2,
         lbl="quotes $(P_t^i, Q_t^i)$", lbl_c=C_LLM_DARK,
         lbl_dy=0.025, lbl_dx=-0.02)

    arrw(ax, col_fx - 0.120, fy2 + 0.04,
         col_mx + 0.110, my1 - 0.06,
         c=C_STAB_DARK, lw=1.6, rad=-0.15,
         lbl="quotes $(P_t^i, Q_t^i)$", lbl_c=C_STAB_DARK,
         lbl_dy=-0.045, lbl_dx=-0.02)

    # ============================================================
    # OBSERVATION / FEEDBACK PANEL (bottom)
    # ============================================================
    obs_cx, obs_cy = 0.595, 0.148
    obs_w, obs_h = 0.42, 0.085
    rbox(ax, obs_cx, obs_cy, obs_w, obs_h, C_OBS_LIGHT, C_OBS_DARK, lw=1.4, radius=0.03, zorder=4)
    txt(ax, obs_cx, obs_cy + 0.025,
        "Observation  $o_t^i$  (token budget $\\tau$, history $h = 3$)",
        fs=8.5, c=C_OBS_DARK, fw="bold")
    txt(ax, obs_cx, obs_cy - 0.010,
        "prices $P_t$  \u2022  sales volume  \u2022  profit $r_t^i$  \u2022  competitors (sampled)",
        fs=7.5, c="#5544AA")

    # Ledger → observation
    arrw(ax, col_mx + 0.005, my4 - 0.035,
         obs_cx - obs_w / 2 + 0.03, obs_cy + 0.025,
         c=C_OBS_DARK, lw=1.3, rad=-0.3,
         lbl="profit + state", lbl_c=C_OBS_DARK, lbl_dy=-0.052, lbl_dx=0.03)

    # Observation → Standard Firm
    ax.annotate(
        "", xy=(col_fx - 0.007, fy1 - 0.085),
        xytext=(obs_cx + obs_w / 2 - 0.02, obs_cy + 0.025),
        arrowprops=dict(arrowstyle="->", color=C_OBS_DARK, lw=1.3,
                        connectionstyle="arc3,rad=-0.28"),
        zorder=5,
    )
    txt(ax, col_fx - 0.08, 0.245, "obs. context\n(history)", fs=7.0, c=C_OBS_DARK,
        style="italic", ls=1.4)

    # Observation → Stabilizing Firm
    ax.annotate(
        "", xy=(col_fx - 0.007, fy2 - 0.090),
        xytext=(obs_cx + obs_w / 2 - 0.01, obs_cy - 0.015),
        arrowprops=dict(arrowstyle="->", color=C_OBS_DARK, lw=1.3,
                        connectionstyle="arc3,rad=0.25"),
        zorder=5,
    )

    # ============================================================
    # SIMULATION LOOP ARROW (very bottom)
    # ============================================================
    ax.annotate(
        "", xy=(0.16, 0.082), xytext=(0.80, 0.082),
        arrowprops=dict(arrowstyle="->", color="#666688", lw=1.8,
                        connectionstyle="arc3,rad=0.55"),
        zorder=3,
    )
    txt(ax, 0.48, 0.048,
        "Simulation loop  (t \u2190 t + 1)",
        fs=8, c="#555577", style="italic")

    # ============================================================
    # EAS SCORE BADGE (top right)
    # ============================================================
    eas_cx, eas_cy = 0.955, 0.74
    rbox(ax, eas_cx, eas_cy, 0.077, 0.21, "#F0F4FF", "#335599", lw=1.3, radius=0.03, zorder=6)
    txt(ax, eas_cx, eas_cy + 0.083, "EAS", fs=10, c="#335599", fw="bold", zorder=7)
    txt(ax, eas_cx, eas_cy + 0.055, "Economic", fs=7, c="#335599", zorder=7)
    txt(ax, eas_cx, eas_cy + 0.036, "Alignment", fs=7, c="#335599", zorder=7)
    txt(ax, eas_cx, eas_cy + 0.017, "Score", fs=7, c="#335599", zorder=7)
    ax.plot([eas_cx - 0.030, eas_cx + 0.030],
            [eas_cy + 0.005, eas_cy + 0.005], color="#335599", lw=0.6, alpha=0.5, zorder=7)

    metrics = [
        ("$\\Phi_S$", "Stability", C_STAB_DARK),
        ("$\\Phi_I$", "Integrity", C_MARKET_DARK),
        ("$\\Phi_W$", "Welfare",   C_CONSUMER),
    ]
    for k, (sym, name, col) in enumerate(metrics):
        yy = eas_cy - 0.018 - k * 0.032
        txt(ax, eas_cx - 0.018, yy, sym, fs=8.5, c=col, zorder=7)
        txt(ax, eas_cx + 0.016, yy, name, fs=7.0, c="#445566", zorder=7)

    # ============================================================
    # LEGEND (bottom left)
    # ============================================================
    leg_cx = 0.10
    leg_cy = 0.195
    leg_items = [
        (C_LLM_DARK,    C_LLM_LIGHT,    "Standard LLM Firm"),
        (C_STAB_DARK,   C_STAB_LIGHT,   "Stabilizing Firm"),
        (C_CONSUMER,    C_CONSUMER_LT,  "CES Consumer"),
        (C_MARKET_DARK, C_MARKET_FILL,  "Market / Auctioneer"),
        (C_CRASH_RED,   C_CRASH_LIGHT,  "Crash / Bankruptcy"),
        (C_OBS_DARK,    C_OBS_LIGHT,    "Observation / Feedback"),
    ]
    txt(ax, leg_cx - 0.01, leg_cy + 0.022, "Legend", fs=8, c="#333333", fw="bold")
    for k, (ec, fc, label_txt) in enumerate(leg_items):
        yy = leg_cy - 0.002 - k * 0.030
        box = FancyBboxPatch((leg_cx - 0.067, yy - 0.009), 0.016, 0.018,
                             boxstyle="round,pad=0,rounding_size=0.003",
                             facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=7)
        ax.add_patch(box)
        ax.text(leg_cx - 0.040, yy, label_txt, fontsize=7.2,
                color="#333333", va="center", ha="left", zorder=8)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Architecture diagram: The Crash (Exp 1)")
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "arch", "exp1_architecture.pdf",
    )
    parser.add_argument("--output", default=default_out)
    args = parser.parse_args()

    fig = make_figure()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
