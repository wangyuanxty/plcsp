# -*- coding: utf-8 -*-
"""Redraw the two paper figures as conceptual illustrations.

Fig. 1  Motivated example: one decision, three tied candidates; fixed CDR vs ours.
Fig. 2  Solution overview: input -> encoder -> MA -> rule-filtered candidates ->
        scoring heads -> actions, with a two-phase training panel.

The style follows the HKUSTDial Supervisor-Skills handbook examples:
rounded panels, concrete example content, icons, red/green failure/success cues.
No data charts (no Gantt, no line/bar plots of results).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

# Okabe–Ito colour-blind-safe palette
BLUE = "#0072B2"
LBLUE = "#56B4E9"
SKY = "#D6EAF8"
ORANGE = "#E69F00"
SAND = "#FDF1DC"
GREEN = "#009E73"
MINT = "#E6F5F0"
RED = "#D55E00"
ROSE = "#FBE3D6"
GREY = "#7F7F7F"
LGREY = "#D9D9D9"
INK = "#222222"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.color": INK,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def panel(ax, x, y, w, h, fc=WHITE, ec=INK, lw=1.0, rad=0.015, z=1):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.004,rounding_size={rad}",
                       linewidth=lw, facecolor=fc, edgecolor=ec, zorder=z)
    ax.add_patch(p)
    return p


def box(ax, x, y, w, h, fc=WHITE, ec=INK, lw=0.9, rad=0.012, z=3, ls="-"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.003,rounding_size={rad}",
                       linewidth=lw, facecolor=fc, edgecolor=ec, zorder=z,
                       linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=INK, lw=1.2, style="-|>", ls="-", z=5, rad=0.0,
          ms=10):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z,
                        shrinkA=1.5, shrinkB=1.5)
    ax.add_patch(a)


def txt(ax, x, y, s, size=7, color=INK, weight="normal", ha="left",
        va="center", z=10, **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, **kw)


def machine(ax, x, y, n, color=BLUE, fc=SKY, busy=None):
    w = h = 0.085
    box(ax, x, y, w, h, fc=fc, ec=color, lw=0.9, rad=0.008)
    # spindle icon
    ax.add_patch(Circle((x + 0.020, y + 0.043), 0.011, facecolor=color,
                        edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((x + 0.034, y + 0.035), 0.030, 0.016,
                           facecolor=color, edgecolor="none", zorder=5))
    txt(ax, x + w / 2, y + 0.015, n, 5.6, color, "bold", "center")
    if busy is True:
        ax.add_patch(Circle((x + w - 0.012, y + h - 0.012), 0.010,
                            facecolor=GREEN, edgecolor="white", lw=0.5, zorder=7))
    elif busy is False:
        ax.add_patch(Circle((x + w - 0.012, y + h - 0.012), 0.010,
                            facecolor=RED, edgecolor="white", lw=0.5, zorder=7))


def agv(ax, x, y, label="V1", color=ORANGE):
    box(ax, x, y, 0.055, 0.028, fc=color, ec=color, lw=0.8, rad=0.007)
    ax.add_patch(Circle((x + 0.013, y - 0.004), 0.006, facecolor=INK,
                        edgecolor="none", zorder=6))
    ax.add_patch(Circle((x + 0.043, y - 0.004), 0.006, facecolor=INK,
                        edgecolor="none", zorder=6))
    txt(ax, x + 0.0275, y + 0.014, label, 5.0, WHITE, "bold", "center")


def lu(ax, x, y):
    box(ax, x, y, 0.070, 0.060, fc="#EFEFEF", ec=GREY, lw=0.9, rad=0.008)
    txt(ax, x + 0.035, y + 0.040, "L/U", 5.8, INK, "bold", "center")
    arrow(ax, (x + 0.035, y + 0.010), (x + 0.035, y + 0.030), color=GREY,
          lw=0.8, ms=6)


def shop_map(ax, x0, y0, w, h, chosen=None, mode="idle"):
    """Draw the same shop in every panel. `chosen` = 'M4' or 'M2'."""
    panel(ax, x0, y0, w, h, fc="#FBFBFB", ec=LGREY, lw=0.8, rad=0.012, z=1)
    # positions inside the panel
    def px(fx):
        return x0 + fx * w

    def py(fy):
        return y0 + fy * h

    lu(ax, px(0.03), py(0.34))
    machine(ax, px(0.30), py(0.66), "M1")
    machine(ax, px(0.48), py(0.66), "M2",
            busy=(True if chosen == "M2" and mode != "idle" else None))
    machine(ax, px(0.66), py(0.66), "M3")
    machine(ax, px(0.30), py(0.30), "M4",
            busy=(True if chosen == "M4" and mode != "idle" else None))
    machine(ax, px(0.48), py(0.30), "M5")
    machine(ax, px(0.66), py(0.30), "M6")
    agv(ax, px(0.12), py(0.40))

    # job cards waiting at L/U
    for i, (j, due) in enumerate((("J2", "41"), ("J3", "58"), ("J4", "22"))):
        bx = px(0.03) + i * 0.075
        by = py(0.10)
        box(ax, bx, by, 0.065, 0.070, fc=WHITE, ec=BLUE, lw=0.7, rad=0.006)
        txt(ax, bx + 0.0325, by + 0.050, j, 5.4, BLUE, "bold", "center")
        txt(ax, bx + 0.0325, by + 0.026, f"due {due}", 4.6, GREY, "normal", "center")

    # paths
    if chosen == "M4":
        pts = [(px(0.075), py(0.38)), (px(0.075), py(0.73)),
               (px(0.32), py(0.73)), (px(0.32), py(0.46))]
        col = RED
    elif chosen == "M2":
        pts = [(px(0.075), py(0.38)), (px(0.50), py(0.38)),
               (px(0.50), py(0.60))]
        col = GREEN
    else:
        pts = []
        col = GREY
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=col, lw=2.0, solid_capstyle="round", zorder=6,
                alpha=0.85)
        arrow(ax, pts[-2], pts[-1], color=col, lw=2.0, ms=11, z=7)

    # callouts
    if chosen == "M4" and mode != "idle":
        txt(ax, px(0.43), py(0.83), "M2 idle 38 min", 5.8, RED, "bold", "center")
        arrow(ax, (px(0.43), py(0.80)), (px(0.52), py(0.73)), color=RED,
              lw=0.9, ms=7, ls=":")
        txt(ax, px(0.43), py(0.20), "chosen: J4 → M4 (far)", 5.8, RED, "bold",
            "center")
    elif chosen == "M2" and mode != "idle":
        txt(ax, px(0.43), py(0.52), "M2 used at t = 12", 5.8, GREEN, "bold",
            "center")
        txt(ax, px(0.43), py(0.20), "chosen: J2 → M2 (near)", 5.8, GREEN,
            "bold", "center")


# ══════════════════════════ Figure 1 v2 ══════════════════════════

def fig1():
    fig = plt.figure(figsize=(7.2, 3.15))
    gs = fig.add_gridspec(1, 3, wspace=0.14, left=0.012, right=0.988,
                          top=0.88, bottom=0.06)

    titles = [
        "(a) One decision, three tied candidates",
        "(b) Fixed CDR: state-blind tie-break",
        "(c) Ours: state-conditioned ranking",
    ]
    for k, title in enumerate(titles):
        ax = fig.add_subplot(gs[0, k])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        txt(ax, 0.0, 0.985, title, 8.0, INK, "bold", "left", "top")

        if k == 0:
            shop_map(ax, 0.0, 0.40, 1.0, 0.56, chosen=None)
            card_y = 0.01
            box(ax, 0.02, card_y, 0.96, 0.36, fc=WHITE, ec=LGREY, lw=0.8,
                rad=0.010)
            txt(ax, 0.05, card_y + 0.315, "candidate pool · rule: EAJ + SMPT",
                6.2, GREY, "bold")
            rows = [("C1", "J2 → M2", "47.5", "6.0", GREEN),
                    ("C2", "J3 → M3", "47.5", "6.0", GREY),
                    ("C3", "J4 → M4", "47.5", "6.0", RED),
                    ("C4", "J6 → M1", "52.0", "4.0", GREY)]
            for i, (c, pair, eaj, smpt, col) in enumerate(rows):
                yy = card_y + 0.255 - i * 0.058
                txt(ax, 0.06, yy, c, 6.0, col, "bold")
                txt(ax, 0.17, yy, pair, 6.0)
                txt(ax, 0.55, yy, eaj, 6.0)
                txt(ax, 0.75, yy, smpt, 6.0)
                if i == 0:
                    txt(ax, 0.53, card_y + 0.330, "EAJ", 4.8, GREY)
                    txt(ax, 0.73, card_y + 0.330, "SMPT", 4.8, GREY)
            # tie bracket
            ax.annotate("", xy=(0.95, card_y + 0.270), xytext=(0.95, card_y + 0.095),
                        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=2.4))
            txt(ax, 0.92, card_y + 0.183, "3-way\ntie", 5.4, ORANGE, "bold",
                "center")

        elif k == 1:
            shop_map(ax, 0.0, 0.40, 1.0, 0.56, chosen="M4", mode="fail")
            box(ax, 0.02, 0.01, 0.96, 0.36, fc=ROSE, ec=RED, lw=0.9, rad=0.010)
            txt(ax, 0.05, 0.305, "fixed CDR", 6.4, RED, "bold")
            txt(ax, 0.05, 0.245, "EAJ + SMPT   →   tie-break by AGV id",
                6.0)
            txt(ax, 0.05, 0.185, "picks  C3: J4 → M4", 6.4, RED, "bold")
            txt(ax, 0.05, 0.120, "reads state?  no", 5.8, GREY)
            txt(ax, 0.05, 0.065, "result:  MS = 268.4 min", 7.0, RED, "bold")
            txt(ax, 0.70, 0.065, "▲ +16.7", 6.0, RED, "bold")

        else:
            shop_map(ax, 0.0, 0.40, 1.0, 0.56, chosen="M2", mode="good")
            box(ax, 0.02, 0.01, 0.96, 0.36, fc=MINT, ec=GREEN, lw=0.9, rad=0.010)
            txt(ax, 0.05, 0.305, "top-K + candidate scoring", 6.4, GREEN, "bold")
            txt(ax, 0.05, 0.245, "C1 0.84   C2 0.09   C3 0.07", 6.0)
            txt(ax, 0.05, 0.185, "picks  C1: J2 → M2", 6.4, GREEN, "bold")
            txt(ax, 0.05, 0.120,
                "reads: V1 pos · M2 util · J2 due", 5.8, GREY)
            txt(ax, 0.05, 0.065, "result:  MS = 251.7 min", 7.0, GREEN, "bold")
            txt(ax, 0.70, 0.065, "▼ −16.7", 6.0, GREEN, "bold")

    out = os.path.join(HERE, "fig1_motivated_example_v2")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return out


# ══════════════════════════ Figure 2 v2 ══════════════════════════

def fig2():
    fig = plt.figure(figsize=(7.2, 3.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # three big panels
    panel(ax, 0.008, 0.03, 0.170, 0.90, fc="#FBFBFB", ec=GREY, lw=0.9)
    panel(ax, 0.190, 0.03, 0.565, 0.90, fc=WHITE, ec=INK, lw=0.9)
    panel(ax, 0.767, 0.03, 0.225, 0.90, fc="#FBFBFB", ec=GREY, lw=0.9)
    txt(ax, 0.093, 0.905, "Input", 8.2, INK, "bold", "center")
    txt(ax, 0.472, 0.905, "Policy", 8.2, INK, "bold", "center")
    txt(ax, 0.879, 0.905, "Training", 8.2, INK, "bold", "center")

    # ── input panel ──
    # mini shop
    panel(ax, 0.018, 0.66, 0.150, 0.20, fc=WHITE, ec=LGREY, lw=0.7, rad=0.008)
    lu(ax, 0.026, 0.690)
    for i, (m, fx, fy) in enumerate((("M1", 0.070, 0.800), ("M2", 0.120, 0.800),
                                     ("M3", 0.070, 0.700), ("M4", 0.120, 0.700))):
        machine(ax, fx, fy, m, color=BLUE, fc=SKY)
    agv(ax, 0.095, 0.720, "V1")
    txt(ax, 0.093, 0.635, "layout + distance matrix", 5.4, GREY, "normal", "center")

    tokens = [("M", "machines", "backlog · util · dist L/U", BLUE, SKY),
              ("B", "jobs", "progress · due · location", BLUE, SKY),
              ("V", "AGVs", "status · position · queue", ORANGE, SAND),
              ("G", "global", "time · MCT mean/min/max", GREY, "#F2F2F2")]
    for i, (k, name, feats, col, fc) in enumerate(tokens):
        y = 0.555 - i * 0.105
        box(ax, 0.018, y, 0.150, 0.085, fc=fc, ec=col, lw=0.7, rad=0.008)
        ax.add_patch(Circle((0.034, y + 0.043), 0.014, facecolor=col,
                            edgecolor="none", zorder=5))
        txt(ax, 0.034, y + 0.043, k, 5.8, WHITE, "bold", "center")
        txt(ax, 0.052, y + 0.057, name, 5.8, INK, "bold")
        txt(ax, 0.052, y + 0.026, feats, 4.9, GREY)

    # ── policy panel ──
    # encoder
    box(ax, 0.205, 0.700, 0.130, 0.150, fc=SKY, ec=BLUE, lw=0.9, rad=0.010)
    txt(ax, 0.270, 0.825, "Shared Transformer", 6.3, BLUE, "bold", "center")
    txt(ax, 0.270, 0.795, "encoder", 6.3, BLUE, "bold", "center")
    for i in range(4):
        box(ax, 0.218 + i * 0.028, 0.735, 0.022, 0.045, fc=WHITE, ec=BLUE,
            lw=0.6, rad=0.004)
    txt(ax, 0.270, 0.715, "4 layers · d = 128", 5.3, GREY, "normal", "center")
    arrow(ax, (0.168, 0.775), (0.203, 0.775), color=INK, lw=1.1)

    # MA
    box(ax, 0.360, 0.700, 0.380, 0.150, fc=SAND, ec=ORANGE, lw=0.9, rad=0.010)
    txt(ax, 0.550, 0.825, "MA: rule selection", 6.6, "#9A6400", "bold", "center")
    txt(ax, 0.550, 0.797, "every ΔT / disturbance", 5.4, GREY, "normal", "center")
    rule_cards = [("PA", "12 rules", "EAJ + SMPT", 0.372),
                  ("LA", "3 rules", "MINCOST", 0.505),
                  ("RA", "3 rules", "QUEUE_EMPTY", 0.638)]
    for k, n, ex, x in rule_cards:
        box(ax, x, 0.712, 0.118, 0.070, fc=WHITE, ec=ORANGE, lw=0.7, rad=0.006)
        txt(ax, x + 0.008, 0.765, f"{k} · {n}", 5.6, ORANGE, "bold")
        txt(ax, x + 0.008, 0.735, ex, 5.0, INK)
    arrow(ax, (0.270, 0.698), (0.360, 0.775), color=GREY, lw=1.0, rad=-0.18)

    # rule-filtered candidates
    box(ax, 0.205, 0.405, 0.535, 0.260, fc="#FFF7E6", ec=ORANGE, lw=0.9,
        rad=0.010)
    txt(ax, 0.472, 0.635, "Rule-filtered candidates", 6.6, "#9A6400", "bold",
        "center")
    txt(ax, 0.472, 0.610, "strict K · unbiased boundary tie", 5.2, GREY,
        "normal", "center")
    # PA card
    box(ax, 0.215, 0.425, 0.185, 0.165, fc=WHITE, ec=LGREY, lw=0.7, rad=0.006)
    txt(ax, 0.223, 0.570, "PA · top-16", 5.6, BLUE, "bold")
    for i, s in enumerate(("C1  J2→M2  47.5", "C2  J3→M3  47.5",
                            "C3  J4→M4  47.5", "C4  J6→M1  52.0")):
        txt(ax, 0.223, 0.540 - i * 0.035, s, 4.8,
            GREEN if i == 0 else (RED if i == 2 else INK))
    # LA card
    box(ax, 0.410, 0.425, 0.155, 0.165, fc=WHITE, ec=LGREY, lw=0.7, rad=0.006)
    txt(ax, 0.418, 0.570, "LA · top-6", 5.6, ORANGE, "bold")
    for i, s in enumerate(("V1  12.0", "V2  15.2", "V3  18.4", "…")):
        txt(ax, 0.418, 0.540 - i * 0.035, s, 4.8)
    # RA card
    box(ax, 0.575, 0.425, 0.150, 0.165, fc=WHITE, ec=LGREY, lw=0.7, rad=0.006)
    txt(ax, 0.583, 0.570, "RA · binary", 5.6, GREEN, "bold")
    txt(ax, 0.583, 0.535, "NOW", 5.2, GREEN, "bold")
    txt(ax, 0.650, 0.535, "DEFER", 5.2, GREY)
    txt(ax, 0.583, 0.495, "rule filters", 4.8, GREY)
    txt(ax, 0.583, 0.465, "the allowed set", 4.8, GREY)
    arrow(ax, (0.550, 0.698), (0.550, 0.667), color=INK, lw=1.1)

    # scoring
    box(ax, 0.205, 0.095, 0.535, 0.275, fc=MINT, ec=GREEN, lw=0.9, rad=0.010)
    txt(ax, 0.472, 0.345, "Candidate scoring", 6.6, "#006B4F", "bold", "center")
    txt(ax, 0.472, 0.320, "rule embedding + candidate features", 5.2, GREY,
        "normal", "center")
    heads = [("PA head", "z_M · z_B · 9 feats", 0.215, 0.105),
             ("LA head", "z_V · ctx · 4 feats", 0.385, 0.105),
             ("RA head", "z_M · 5 feats", 0.555, 0.105)]
    for name, feats, x, y in heads:
        box(ax, x, y, 0.155, 0.180, fc=WHITE, ec=GREEN, lw=0.7, rad=0.006)
        txt(ax, x + 0.008, y + 0.150, name, 5.6, GREEN, "bold")
        txt(ax, x + 0.008, y + 0.118, feats, 4.8, GREY)
        # little score bars
        for i, v in enumerate((0.72, 0.18, 0.10)):
            ax.add_patch(Rectangle((x + 0.010, y + 0.075 - i * 0.022),
                                   0.130 * v, 0.014, facecolor=GREEN,
                                   edgecolor="none", alpha=0.75, zorder=5))
    txt(ax, 0.472, 0.070, "group z-norm if K ≥ 3 · log-softmax within K",
        5.0, GREY, "normal", "center")
    arrow(ax, (0.472, 0.403), (0.472, 0.372), color=INK, lw=1.1)

    # actions
    box(ax, 0.755, 0.405, 0.225, 0.260, fc=MINT, ec=GREEN, lw=0.9, rad=0.010)
    txt(ax, 0.8675, 0.635, "Actions", 6.6, "#006B4F", "bold", "center")
    for i, (s1, s2) in enumerate((("dispatch", "(job, machine)"),
                                  ("assign AGV", "V1 / V2 / V3"),
                                  ("maintain", "NOW / DEFER"))):
        y = 0.540 - i * 0.065
        box(ax, 0.765, y, 0.205, 0.052, fc=WHITE, ec=GREEN, lw=0.6, rad=0.005)
        txt(ax, 0.772, y + 0.033, s1, 5.4, INK, "bold")
        txt(ax, 0.772, y + 0.014, s2, 4.8, GREY)
    arrow(ax, (0.742, 0.535), (0.753, 0.535), color=INK, lw=1.1)

    # ── training panel ──
    txt(ax, 0.879, 0.865, "two-phase", 6.4, INK, "bold", "center")
    txt(ax, 0.879, 0.842, "one model", 6.4, INK, "bold", "center")
    box(ax, 0.778, 0.620, 0.205, 0.190, fc=SAND, ec=ORANGE, lw=0.8, rad=0.008)
    txt(ax, 0.790, 0.785, "A · MA only", 5.8, ORANGE, "bold")
    txt(ax, 0.790, 0.755, "G rule samples", 5.0)
    txt(ax, 0.790, 0.730, "per top-level point", 5.0)
    txt(ax, 0.790, 0.705, "group advantage", 5.0, GREY)
    txt(ax, 0.790, 0.680, "update: select heads", 5.0, GREY)
    txt(ax, 0.790, 0.650, "encoder shared", 4.8, GREY)

    box(ax, 0.778, 0.400, 0.205, 0.190, fc=MINT, ec=GREEN, lw=0.8, rad=0.008)
    txt(ax, 0.790, 0.565, "B · scoring only", 5.8, GREEN, "bold")
    txt(ax, 0.790, 0.535, "rules frozen", 5.0)
    txt(ax, 0.790, 0.510, "M execution rollouts", 5.0)
    txt(ax, 0.790, 0.485, "group advantage", 5.0, GREY)
    txt(ax, 0.790, 0.460, "update: 3 score heads", 5.0, GREY)
    txt(ax, 0.790, 0.430, "encoder shared", 4.8, GREY)

    arrow(ax, (0.880, 0.618), (0.880, 0.592), color=GREY, lw=1.0, ms=8)
    arrow(ax, (0.905, 0.592), (0.905, 0.618), color=GREY, lw=1.0, ms=8)

    box(ax, 0.778, 0.245, 0.205, 0.120, fc="#F2F2F2", ec=GREY, lw=0.7, rad=0.008)
    txt(ax, 0.790, 0.335, "loss = −Σ Â · log p", 5.2, INK, "bold")
    txt(ax, 0.790, 0.310, "no PPO clip", 5.0, GREY)
    txt(ax, 0.790, 0.285, "random MK-range train", 5.0, GREY)
    txt(ax, 0.790, 0.260, "zero-shot MK01–10", 5.0, GREY)

    # dashed feedback from training to policy
    arrow(ax, (0.767, 0.700), (0.742, 0.700), color=GREY, lw=0.9, ls="--", ms=8)
    arrow(ax, (0.767, 0.480), (0.742, 0.480), color=GREY, lw=0.9, ls="--", ms=8)
    txt(ax, 0.754, 0.715, "π", 5.0, GREY, "bold", "center")
    txt(ax, 0.754, 0.465, "∇", 5.0, GREY, "bold", "center")

    out = os.path.join(HERE, "fig2_solution_overview_v2")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig1())
    print(fig2())
