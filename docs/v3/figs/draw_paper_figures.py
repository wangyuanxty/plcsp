# -*- coding: utf-8 -*-
"""Conceptual drafts for the two paper figures (v3).

Fig. 1  Motivated example: a shop-floor decision, a fixed-rule failure, and the
        state-conditioned fix. Shop maps + decision cards, no data charts.
Fig. 2  Solution overview: input -> encoder -> MA -> rule-filtered candidates ->
        scoring heads -> actions (top), with a two-phase training strip (bottom).

Style follows the HKUSTDial Supervisor-Skills handbook examples.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#0072B2"
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


def panel(ax, x, y, w, h, fc=WHITE, ec=INK, lw=1.0, rad=0.015, z=1, ls="-"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.004,rounding_size={rad}",
                       linewidth=lw, facecolor=fc, edgecolor=ec, zorder=z,
                       linestyle=ls)
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


def machine(ax, x, y, n, color=BLUE, fc=SKY, busy=None, s=0.085):
    box(ax, x, y, s, s, fc=fc, ec=color, lw=0.9, rad=0.008)
    ax.add_patch(Circle((x + 0.020 * s / 0.085, y + 0.043 * s / 0.085),
                        0.011 * s / 0.085, facecolor=color, edgecolor="none",
                        zorder=5))
    ax.add_patch(Rectangle((x + 0.034 * s / 0.085, y + 0.035 * s / 0.085),
                           0.030 * s / 0.085, 0.016 * s / 0.085,
                           facecolor=color, edgecolor="none", zorder=5))
    txt(ax, x + s / 2, y + 0.015 * s / 0.085, n, 5.6, color, "bold", "center")
    if busy is True:
        ax.add_patch(Circle((x + s - 0.012, y + s - 0.012), 0.010,
                            facecolor=GREEN, edgecolor="white", lw=0.5, zorder=7))
    elif busy is False:
        ax.add_patch(Circle((x + s - 0.012, y + s - 0.012), 0.010,
                            facecolor=RED, edgecolor="white", lw=0.5, zorder=7))


def agv(ax, x, y, label="V1", color=ORANGE, s=0.055):
    box(ax, x, y, s, 0.028, fc=color, ec=color, lw=0.8, rad=0.007)
    ax.add_patch(Circle((x + 0.013, y - 0.004), 0.006, facecolor=INK,
                        edgecolor="none", zorder=6))
    ax.add_patch(Circle((x + s - 0.012, y - 0.004), 0.006, facecolor=INK,
                        edgecolor="none", zorder=6))
    txt(ax, x + s / 2, y + 0.014, label, 5.0, WHITE, "bold", "center")


def lu(ax, x, y, s=0.070):
    box(ax, x, y, s, 0.060, fc="#EFEFEF", ec=GREY, lw=0.9, rad=0.008)
    txt(ax, x + s / 2, y + 0.040, "L/U", 5.8, INK, "bold", "center")
    arrow(ax, (x + s / 2, y + 0.010), (x + s / 2, y + 0.030), color=GREY,
          lw=0.8, ms=6)


def badge(ax, x, y, w, text, color, fc=WHITE, size=4.4):
    box(ax, x, y, w, 0.042, fc=fc, ec=color, lw=0.7, rad=0.005)
    txt(ax, x + w / 2, y + 0.021, text, size, color, "bold", "center")


def conflict_icon(ax, x, y):
    ax.add_patch(Circle((x, y), 0.030, facecolor=RED, edgecolor="white",
                        lw=0.8, zorder=8))
    txt(ax, x, y, "!", 7.0, WHITE, "bold", "center")


def shop_map2(ax, x0, y0, w, h, mode):
    """Shared shop thumbnail. `mode` in {coupled, conflict, coordinated}."""
    panel(ax, x0, y0, w, h, fc="#FBFBFB", ec=LGREY, lw=0.8, rad=0.012, z=1)

    def px(fx):
        return x0 + fx * w

    def py(fy):
        return y0 + fy * h

    lu(ax, px(0.03), py(0.34))
    machine(ax, px(0.30), py(0.66), "M1")
    machine(ax, px(0.48), py(0.66), "M2",
            busy=(False if mode in ("coupled", "conflict") else None))
    machine(ax, px(0.66), py(0.66), "M3")
    machine(ax, px(0.30), py(0.30), "M4")
    machine(ax, px(0.48), py(0.30), "M5")
    machine(ax, px(0.66), py(0.30), "M6")
    agv(ax, px(0.12), py(0.40), "V1")
    agv(ax, px(0.80), py(0.16), "V2")

    # J4 waits at L/U
    box(ax, px(0.03), py(0.08), 0.120, 0.075, fc=WHITE, ec=BLUE, lw=0.7,
        rad=0.006)
    txt(ax, px(0.09), py(0.132), "J4", 5.0, BLUE, "bold", "center")
    txt(ax, px(0.09), py(0.108), "due", 3.6, GREY, "normal", "center")
    txt(ax, px(0.09), py(0.088), "22", 4.0, GREY, "normal", "center")

    if mode == "coupled":
        badge(ax, px(0.12), py(0.50), 0.105, "V1 booked", ORANGE)
        conflict_icon(ax, px(0.40), py(0.55))
        return

    if mode == "conflict":
        # J4 -> M2 (load path)
        pts = [(px(0.075), py(0.38)), (px(0.075), py(0.58)),
               (px(0.52), py(0.58)), (px(0.52), py(0.66))]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=BLUE, lw=2.0, solid_capstyle="round", zorder=6,
                alpha=0.85)
        arrow(ax, pts[-2], pts[-1], color=BLUE, lw=2.0, ms=11, z=7)
        # V2 deadheads to L/U along the bottom aisle
        pts2 = [(px(0.80), py(0.16)), (px(0.19), py(0.16)),
                (px(0.19), py(0.37)), (px(0.10), py(0.37))]
        xs2, ys2 = zip(*pts2)
        ax.plot(xs2, ys2, color=ORANGE, lw=2.0, solid_capstyle="round",
                zorder=6, alpha=0.85)
        arrow(ax, pts2[-2], pts2[-1], color=ORANGE, lw=2.0, ms=11, z=7)
        txt(ax, px(0.575), py(0.72), "PM", 4.6, RED, "bold")
        txt(ax, px(0.40), py(0.25), "J4 waits 26 min", 5.4, RED, "bold",
            "center")
        return

    if mode == "coordinated":
        # J4 -> M3 (green), V1 takes it
        pts = [(px(0.075), py(0.38)), (px(0.075), py(0.58)),
               (px(0.72), py(0.58)), (px(0.72), py(0.66))]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=GREEN, lw=2.0, solid_capstyle="round", zorder=6,
                alpha=0.85)
        arrow(ax, pts[-2], pts[-1], color=GREEN, lw=2.0, ms=11, z=7)
        txt(ax, px(0.40), py(0.25), "no wait", 5.4, GREEN, "bold", "center")


# ══════════════════════════ Figure 1 ══════════════════════════

def fig1():
    fig = plt.figure(figsize=(7.2, 3.35))
    gs = fig.add_gridspec(1, 3, wspace=0.14, left=0.012, right=0.988,
                          top=0.84, bottom=0.06)

    titles = [
        "(a) One job,\nthree coupled decisions",
        "(b) Independent agents:\nlocal choices conflict",
        "(c) Hierarchical MA:\ncoordinate, then execute",
    ]
    for k, title in enumerate(titles):
        ax = fig.add_subplot(gs[0, k])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        txt(ax, 0.0, 0.995, title, 7.0, INK, "bold", "left", "top",
            linespacing=1.2)

        if k == 0:
            shop_map2(ax, 0.0, 0.58, 1.0, 0.33, "coupled")
            cards = [
                ("PA · production", "J4 → M2", "M2 least utilized", BLUE, SKY),
                ("LA · logistics", "V1 nearest", "but V1 is booked", ORANGE,
                 SAND),
                ("RA · maintenance", "M2 age 118/120", "maintain now?", GREEN,
                 MINT),
            ]
            for i, (h1, h2, h3, col, fc) in enumerate(cards):
                x = 0.02 + i * 0.325
                box(ax, x, 0.30, 0.305, 0.20, fc=fc, ec=col, lw=0.8, rad=0.008)
                txt(ax, x + 0.010, 0.470, h1, 4.6, col, "bold")
                txt(ax, x + 0.010, 0.428, h2, 4.9, INK)
                txt(ax, x + 0.010, 0.390, h3, 4.7, GREY)
            conflict_icon(ax, 0.50, 0.275)
            txt(ax, 0.545, 0.275, "coupled", 5.2, RED, "bold")
            # timescale band
            box(ax, 0.02, 0.03, 0.96, 0.20, fc=WHITE, ec=LGREY, lw=0.7,
                rad=0.008)
            txt(ax, 0.04, 0.195, "decision frequencies", 4.8, GREY, "bold")
            for i, (tag, freq, act, col) in enumerate((
                    ("MA", "every ΔT", "rule set", ORANGE),
                    ("PA", "every op", "job→mach", BLUE),
                    ("LA", "every trip", "AGV", ORANGE),
                    ("RA", "completion", "NOW/DEFER", GREEN))):
                x = 0.04 + i * 0.235
                box(ax, x, 0.055, 0.220, 0.115, fc=WHITE, ec=col, lw=0.6,
                    rad=0.005)
                txt(ax, x + 0.008, 0.150, tag, 5.0, col, "bold")
                txt(ax, x + 0.008, 0.120, freq, 4.3, GREY)
                txt(ax, x + 0.008, 0.085, act, 4.6, INK)

        elif k == 1:
            shop_map2(ax, 0.0, 0.44, 1.0, 0.47, "conflict")
            box(ax, 0.02, 0.01, 0.96, 0.41, fc=ROSE, ec=RED, lw=0.9, rad=0.010)
            txt(ax, 0.05, 0.375, "independent agents", 6.2, RED, "bold")
            txt(ax, 0.05, 0.322, "PA: EAJ + SMPT  →  M2", 5.6)
            txt(ax, 0.05, 0.278, "LA: MINCOST  →  V2 (V1 booked)", 5.6)
            txt(ax, 0.05, 0.234, "RA: AGE  →  maintain M2 now", 5.6)
            txt(ax, 0.05, 0.182, "each locally optimal", 5.2, GREY)
            txt(ax, 0.05, 0.140, "together: J4 waits 26 min", 5.4, RED, "bold")
            txt(ax, 0.05, 0.075, "MS = 268.4 min", 6.8, RED, "bold")
            conflict_icon(ax, 0.88, 0.095)

        else:
            shop_map2(ax, 0.0, 0.44, 1.0, 0.47, "coordinated")
            box(ax, 0.02, 0.01, 0.96, 0.41, fc=MINT, ec=GREEN, lw=0.9,
                rad=0.010)
            txt(ax, 0.05, 0.375, "MA selects a coordinated rule set", 5.8,
                GREEN, "bold")
            txt(ax, 0.05, 0.322, "PA: EAJ + EAM  (avoid M2)", 5.4)
            txt(ax, 0.05, 0.278, "LA: MINCOST  (use V1)", 5.4)
            txt(ax, 0.05, 0.234, "RA: QUEUE_EMPTY  (defer PM)", 5.4)
            txt(ax, 0.05, 0.182, "J4 → M3; PM deferred 6 min", 5.2, GREEN,
                "bold")
            txt(ax, 0.05, 0.130, "no waiting", 5.2, GREY)
            txt(ax, 0.05, 0.075, "MS = 251.7 min", 6.8, GREEN, "bold")

    out = os.path.join(HERE, "fig1_motivated_example")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return out
# ══════════════════════════ Figure 2 ══════════════════════════

def fig2():
    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── top row: input / policy / actions ──
    panel(ax, 0.008, 0.335, 0.160, 0.645, fc="#FBFBFB", ec=GREY, lw=0.9)
    panel(ax, 0.176, 0.335, 0.630, 0.645, fc=WHITE, ec=INK, lw=0.9)
    panel(ax, 0.814, 0.335, 0.178, 0.645, fc="#FBFBFB", ec=GREY, lw=0.9)
    txt(ax, 0.088, 0.955, "Input", 7.8, INK, "bold", "center")
    txt(ax, 0.491, 0.955, "Policy", 7.8, INK, "bold", "center")
    txt(ax, 0.903, 0.955, "Actions", 7.8, INK, "bold", "center")

    # input: mini shop
    panel(ax, 0.016, 0.785, 0.144, 0.150, fc=WHITE, ec=LGREY, lw=0.7,
          rad=0.008)
    lu(ax, 0.024, 0.800, s=0.050)
    for m, fx, fy in (("M1", 0.055, 0.885), ("M2", 0.100, 0.885),
                      ("M3", 0.055, 0.815), ("M4", 0.100, 0.815)):
        machine(ax, fx, fy, m, s=0.048)
    agv(ax, 0.085, 0.838, "V1", s=0.040)
    txt(ax, 0.088, 0.770, "layout + distance matrix", 4.9, GREY, "normal",
        "center")

    tokens = [("M", "machines", "backlog · util · dist L/U", BLUE, SKY),
              ("B", "jobs", "progress · due · location", BLUE, SKY),
              ("V", "AGVs", "status · position · queue", ORANGE, SAND),
              ("G", "global", "time · MCT mean/min/max", GREY, "#F2F2F2")]
    for i, (k, name, feats, col, fc) in enumerate(tokens):
        y = 0.690 - i * 0.080
        box(ax, 0.016, y, 0.144, 0.068, fc=fc, ec=col, lw=0.7, rad=0.008)
        ax.add_patch(Circle((0.030, y + 0.034), 0.012, facecolor=col,
                            edgecolor="none", zorder=5))
        txt(ax, 0.030, y + 0.034, k, 5.2, WHITE, "bold", "center")
        txt(ax, 0.046, y + 0.047, name, 5.4, INK, "bold")
        txt(ax, 0.046, y + 0.020, feats, 4.5, GREY)

    # policy: encoder + MA
    box(ax, 0.190, 0.770, 0.125, 0.155, fc=SKY, ec=BLUE, lw=0.9, rad=0.010)
    txt(ax, 0.2525, 0.895, "Shared", 6.0, BLUE, "bold", "center")
    txt(ax, 0.2525, 0.868, "Transformer", 6.0, BLUE, "bold", "center")
    txt(ax, 0.2525, 0.841, "encoder", 6.0, BLUE, "bold", "center")
    for i in range(4):
        box(ax, 0.202 + i * 0.026, 0.802, 0.020, 0.030, fc=WHITE, ec=BLUE,
            lw=0.6, rad=0.003)
    txt(ax, 0.2525, 0.783, "4 layers · d = 128", 4.9, GREY, "normal", "center")

    box(ax, 0.330, 0.770, 0.455, 0.155, fc=SAND, ec=ORANGE, lw=0.9, rad=0.010)
    txt(ax, 0.5575, 0.895, "MA: rule selection", 6.4, "#9A6400", "bold", "center")
    txt(ax, 0.5575, 0.868, "every ΔT / disturbance", 4.9, GREY, "normal", "center")
    for k, n, ex, x in (("PA", "12", "EAJ + SMPT", 0.342),
                         ("LA", "3", "MINCOST", 0.500),
                         ("RA", "3", "QUEUE_EMPTY", 0.658)):
        box(ax, x, 0.782, 0.145, 0.068, fc=WHITE, ec=ORANGE, lw=0.7, rad=0.006)
        txt(ax, x + 0.008, 0.833, f"{k} · {n} rules", 5.2, ORANGE, "bold")
        txt(ax, x + 0.008, 0.802, ex, 4.8, INK)
    arrow(ax, (0.2525, 0.768), (0.330, 0.840), color=GREY, lw=1.0, rad=-0.18)

    # rule-filtered candidates
    box(ax, 0.190, 0.520, 0.595, 0.220, fc="#FFF7E6", ec=ORANGE, lw=0.9,
        rad=0.010)
    txt(ax, 0.4875, 0.715, "Rule-filtered candidates", 6.3, "#9A6400", "bold",
        "center")
    txt(ax, 0.4875, 0.690, "strict K · unbiased boundary tie", 4.9, GREY,
        "normal", "center")
    # PA
    box(ax, 0.200, 0.535, 0.195, 0.140, fc=WHITE, ec=LGREY, lw=0.7, rad=0.006)
    txt(ax, 0.208, 0.655, "PA · top-16", 5.2, BLUE, "bold")
    for i, s in enumerate(("C1  J2→M2  47.5", "C2  J3→M3  47.5",
                            "C3  J4→M4  47.5", "…")):
        txt(ax, 0.208, 0.625 - i * 0.026, s, 4.5,
            GREEN if i == 0 else (RED if i == 2 else INK))
    # LA
    box(ax, 0.405, 0.535, 0.165, 0.140, fc=WHITE, ec=LGREY, lw=0.7, rad=0.006)
    txt(ax, 0.413, 0.655, "LA · top-6", 5.2, ORANGE, "bold")
    for i, s in enumerate(("V1  12.0", "V2  15.2", "V3  18.4", "…")):
        txt(ax, 0.413, 0.625 - i * 0.026, s, 4.5)
    # RA
    box(ax, 0.580, 0.535, 0.195, 0.140, fc=WHITE, ec=LGREY, lw=0.7, rad=0.006)
    txt(ax, 0.588, 0.655, "RA · binary", 5.2, GREEN, "bold")
    txt(ax, 0.588, 0.625, "NOW", 4.8, GREEN, "bold")
    txt(ax, 0.650, 0.625, "DEFER", 4.8, GREY)
    txt(ax, 0.588, 0.590, "rule filters the", 4.4, GREY)
    txt(ax, 0.588, 0.565, "allowed set", 4.4, GREY)
    arrow(ax, (0.5575, 0.768), (0.5575, 0.742), color=INK, lw=1.1)

    # scoring
    box(ax, 0.190, 0.360, 0.595, 0.140, fc=MINT, ec=GREEN, lw=0.9, rad=0.010)
    txt(ax, 0.4875, 0.478, "Candidate scoring", 6.3, "#006B4F", "bold", "center")
    txt(ax, 0.4875, 0.455, "rule embedding + candidate features", 4.9, GREY,
        "normal", "center")
    for name, feats, x in (("PA head", "z_M · z_B · 9 feats", 0.200),
                            ("LA head", "z_V · ctx · 4 feats", 0.398),
                            ("RA head", "z_M · 5 feats", 0.596)):
        box(ax, x, 0.372, 0.180, 0.072, fc=WHITE, ec=GREEN, lw=0.7, rad=0.005)
        txt(ax, x + 0.006, 0.430, name, 5.0, GREEN, "bold")
        txt(ax, x + 0.006, 0.414, feats, 4.4, GREY)
        for i, v in enumerate((0.70, 0.18, 0.10)):
            ax.add_patch(Rectangle((x + 0.008, 0.390 - i * 0.009),
                                    0.150 * v, 0.006, facecolor=GREEN,
                                    edgecolor="none", alpha=0.75, zorder=5))
    arrow(ax, (0.4875, 0.518), (0.4875, 0.502), color=INK, lw=1.1)
    txt(ax, 0.4875, 0.347, "group z-norm if K ≥ 3 · log-softmax within K",
        4.7, GREY, "normal", "center")

    # actions panel
    for i, (s1, s2) in enumerate((("dispatch", "(job, machine)"),
                                  ("assign AGV", "V1 / V2 / V3"),
                                  ("maintain", "NOW / DEFER"))):
        y = 0.760 - i * 0.135
        box(ax, 0.824, y, 0.158, 0.115, fc=MINT, ec=GREEN, lw=0.7, rad=0.008)
        txt(ax, 0.834, y + 0.080, s1, 5.4, GREEN, "bold")
        txt(ax, 0.834, y + 0.045, s2, 4.7, GREY)
    arrow(ax, (0.786, 0.430), (0.822, 0.430), color=INK, lw=1.1)

    # ── bottom row: training ──
    panel(ax, 0.008, 0.020, 0.984, 0.290, fc="#F7F7F7", ec=GREY, lw=0.9)
    txt(ax, 0.500, 0.280, "Two-phase training · one model · cross-instance",
        7.4, INK, "bold", "center")

    box(ax, 0.020, 0.055, 0.300, 0.190, fc=SAND, ec=ORANGE, lw=0.8, rad=0.008)
    txt(ax, 0.032, 0.222, "A · MA only", 5.8, ORANGE, "bold")
    txt(ax, 0.032, 0.192, "G rule samples per top-level point", 4.8)
    txt(ax, 0.032, 0.166, "group advantage", 4.8, GREY)
    txt(ax, 0.032, 0.140, "update: PA / LA / RA select heads", 4.8, GREY)
    txt(ax, 0.032, 0.114, "encoder shared", 4.6, GREY)
    txt(ax, 0.032, 0.082, "phase A", 4.8, ORANGE, "bold")

    box(ax, 0.340, 0.055, 0.300, 0.190, fc=MINT, ec=GREEN, lw=0.8, rad=0.008)
    txt(ax, 0.352, 0.222, "B · scoring heads only", 5.8, GREEN, "bold")
    txt(ax, 0.352, 0.192, "rules frozen at rule*", 4.8)
    txt(ax, 0.352, 0.166, "M execution rollouts", 4.8)
    txt(ax, 0.352, 0.140, "group advantage", 4.8, GREY)
    txt(ax, 0.352, 0.114, "update: score_pa / la / ra", 4.8, GREY)
    txt(ax, 0.352, 0.082, "phase B", 4.8, GREEN, "bold")

    box(ax, 0.660, 0.055, 0.320, 0.190, fc=WHITE, ec=GREY, lw=0.8, rad=0.008)
    txt(ax, 0.672, 0.222, "loss = −Σ Â · log p", 5.6, INK, "bold")
    txt(ax, 0.672, 0.192, "no PPO clip · no critic", 4.8, GREY)
    txt(ax, 0.672, 0.166, "train: random MK-range instances", 4.8, GREY)
    txt(ax, 0.672, 0.140, "test: MK01–10 zero-shot", 4.8, GREY)
    txt(ax, 0.672, 0.114, "layout seeds disjoint", 4.8, GREY)
    txt(ax, 0.672, 0.082, "one shared encoder", 4.8, GREY)

    arrow(ax, (0.320, 0.150), (0.338, 0.150), color=GREY, lw=1.0)
    arrow(ax, (0.640, 0.150), (0.658, 0.150), color=GREY, lw=1.0)

    out = os.path.join(HERE, "fig2_solution_overview")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig1())
    print(fig2())
