# -*- coding: utf-8 -*-
"""Draft two paper figures with matplotlib.

Fig. 1  Motivated example: a rule-tied decision at a shop-floor decision point.
Fig. 2  Solution overview: top-K rule-guided hierarchical policy (v3).

Outputs (vector PDF/SVG + 300-dpi PNG preview) go to this directory.
All numbers in Fig. 1 are schematic; the caption says so. Replace them with a
traced MK01 decision before submission.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

# Okabe–Ito colour-blind-safe palette
BLUE = "#0072B2"
LBLUE = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GREY = "#7F7F7F"
LGREY = "#D9D9D9"
INK = "#222222"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def rounded(ax, x, y, w, h, fc, ec=None, lw=1.0, rad=0.02, alpha=1.0, z=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.004,rounding_size={rad}",
                       linewidth=lw, facecolor=fc, edgecolor=ec or fc,
                       alpha=alpha, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, xy0, xy1, color=INK, lw=1.1, style="-|>", ls="-", z=5, rad=0.0):
    a = FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=9,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z,
                        shrinkA=1.5, shrinkB=1.5)
    ax.add_patch(a)


# ══════════════════════════ Figure 1: motivated example ══════════════════════════

def fig1():
    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.22, 1.22],
                          wspace=0.24, left=0.015, right=0.985,
                          top=0.84, bottom=0.10)

    # ── (a) the decision point ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a) A 3-way rule tie at $t$", loc="left",
                 fontsize=8.5, fontweight="bold", pad=4)

    # shop sketch: 4 machines + L/U + AGV
    mach = {"M1": (0.08, 0.68), "M2": (0.40, 0.68),
            "M3": (0.72, 0.68), "M4": (0.72, 0.48)}
    for name, (x, y) in mach.items():
        rounded(ax, x, y, 0.17, 0.13, "#DCEBF6", BLUE, lw=0.9)
        ax.text(x + 0.085, y + 0.065, name, ha="center", va="center",
                fontsize=8, color=BLUE, fontweight="bold")
    # L/U
    rounded(ax, 0.05, 0.29, 0.19, 0.11, "#EFEFEF", GREY, lw=0.9)
    ax.text(0.145, 0.345, "L/U", ha="center", va="center", fontsize=8,
            color=INK, fontweight="bold")
    # AGV
    ax.add_patch(plt.Circle((0.72, 0.345), 0.042, facecolor=ORANGE,
                            edgecolor="white", linewidth=0.8, zorder=6))
    ax.text(0.72, 0.345, "V1", ha="center", va="center", fontsize=6.8,
            color="white", fontweight="bold", zorder=7)
    # candidate arrows (schematic)
    for (x, y), col, lab, lx, ly, lw in [
            ((0.48, 0.68), GREEN, "C1", 0.52, 0.60, 1.1),
            ((0.80, 0.68), GREY, "C2", 0.84, 0.60, 1.0),
            ((0.80, 0.48), RED, "C3", 0.62, 0.43, 1.1),
            ((0.16, 0.68), LGREY, "C4", 0.25, 0.64, 0.9)]:
        arrow(ax, (0.15, 0.40), (x, y - 0.005), color=col, lw=lw, rad=0.0)
        ax.text(lx, ly, lab, fontsize=6.8, color=col, fontweight="bold")

    # score table
    rows = [
        ("C1", "J2 → M2", "47.5", "6.0", GREEN),
        ("C2", "J3 → M3", "47.5", "6.0", GREY),
        ("C3", "J4 → M4", "47.5", "6.0", RED),
        ("C4", "J6 → M1", "52.0", "4.0", GREY),
    ]
    x0, y0, w = 0.03, 0.02, 0.94
    ax.add_patch(Rectangle((x0, y0), w, 0.21, facecolor="white",
                           edgecolor=LGREY, linewidth=0.8, zorder=3))
    ax.text(x0 + 0.02, y0 + 0.175, "cand", fontsize=6.2, color=GREY)
    ax.text(x0 + 0.21, y0 + 0.175, "pair", fontsize=6.2, color=GREY)
    ax.text(x0 + 0.55, y0 + 0.175, "EAJ", fontsize=6.2, color=GREY)
    ax.text(x0 + 0.75, y0 + 0.175, "SMPT", fontsize=6.2, color=GREY)
    for i, (c, pair, eaj, smpt, col) in enumerate(rows):
        yy = y0 + 0.132 - i * 0.034
        ax.text(x0 + 0.02, yy, c, fontsize=6.5, color=col, fontweight="bold")
        ax.text(x0 + 0.21, yy, pair, fontsize=6.5)
        ax.text(x0 + 0.55, yy, eaj, fontsize=6.5)
        ax.text(x0 + 0.75, yy, smpt, fontsize=6.5)
    # tie bracket
    ax.annotate("", xy=(x0 + w - 0.015, y0 + 0.145),
                xytext=(x0 + w - 0.015, y0 + 0.060),
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=2.2))
    ax.text(x0 + w - 0.050, y0 + 0.102, "tie", rotation=90, fontsize=6.3,
            color=ORANGE, ha="center", va="center", fontweight="bold")

    # ── (b) and (c): Gantt of the affected resources ──
    def gantt(ax, title, transport_end, j4_start, ms, note, note_col,
              m2_start, kind):
        ax.set_xlim(0, 270)
        ax.set_ylim(-0.75, 3.15)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["V1", "M2", "M4"], fontsize=7)
        ax.set_xlabel("time [min]", fontsize=7, labelpad=2)
        ax.tick_params(axis="x", labelsize=6.5, length=2)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=8.3, fontweight="bold", pad=4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # V1 transport (row 0)
        ax.broken_barh([(0, transport_end)], (-0.38, 0.76),
                       facecolors=ORANGE, edgecolor="white")
        if transport_end >= 45:
            ax.text(transport_end / 2, 0.0,
                    "L/U → M4" if kind == "bad" else "L/U → M2",
                    ha="center", va="center", fontsize=6.0, color="white",
                    fontweight="bold")
        else:
            ax.text(transport_end + 4, 0.0,
                    ("L/U → M4" if kind == "bad" else "L/U → M2")
                    + f" ({transport_end} min)",
                    ha="left", va="center", fontsize=5.8, color=ORANGE,
                    fontweight="bold")
        # M2 (row 1)
        if kind == "bad":
            ax.broken_barh([(0, transport_end)], (0.62, 0.76),
                           facecolors="white", edgecolor=RED, hatch="///",
                           linewidth=0.7)
            ax.text(transport_end / 2, 1.0, "idle", ha="center", va="center",
                    fontsize=6.0, color=RED, fontweight="bold")
        ax.broken_barh([(m2_start, 60)], (0.62, 0.76), facecolors=BLUE,
                       edgecolor="white")
        ax.text(m2_start + 30, 1.0, "J2", ha="center", va="center",
                fontsize=6.0, color="white", fontweight="bold")
        ax.broken_barh([(m2_start + 60, 55)], (0.62, 0.76),
                       facecolors=LBLUE, edgecolor="white")
        ax.text(m2_start + 87.5, 1.0, "J3", ha="center", va="center",
                fontsize=6.0, color=INK)
        # M4 (row 2)
        ax.broken_barh([(j4_start, 40)], (1.62, 0.76), facecolors=BLUE,
                       edgecolor="white")
        ax.text(j4_start + 20, 2.0, "J4", ha="center", va="center",
                fontsize=6.0, color="white", fontweight="bold")
        # MS and the one-line reading
        ax.text(0.99, 0.96, f"MS = {ms} min", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.4, fontweight="bold")
        note_y = 0.84 if kind == "bad" else 0.11
        ax.text(0.99, note_y, note, transform=ax.transAxes, ha="right",
                va="top", fontsize=6.2, color=note_col, fontweight="bold")
        ax.text(0.99, 0.02, "affected resources only", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.6, color=GREY)

    axb = fig.add_subplot(gs[0, 1])
    gantt(axb, "(b) Fixed CDR: random tie-break", 38, 38, "268.4",
          "38 min transport → M4 far", RED, 38, "bad")
    axc = fig.add_subplot(gs[0, 2])
    gantt(axc, "(c) Ours: state-conditioned ranking", 12, 92, "251.7",
          "12 min transport → M2 near", GREEN, 12, "good")

    out = os.path.join(HERE, "fig1_motivated_example")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return out


# ══════════════════════════ Figure 2: solution overview ══════════════════════════

def fig2():
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.015, 0.965, "Top-K rule-guided hierarchical policy",
            fontsize=10, fontweight="bold", va="top")

    def label(x, y, text, size=7, color=INK, weight="normal", ha="left",
              va="center"):
        ax.text(x, y, text, fontsize=size, color=color,
                fontweight=weight, ha=ha, va=va, zorder=10)

    # input
    rounded(ax, 0.015, 0.30, 0.135, 0.42, "#EAF4FB", BLUE)
    label(0.082, 0.685, "Shop state", 7.8, BLUE, "bold", "center")
    for i, t in enumerate(("M  machines", "B  jobs", "V  AGVs", "G  global")):
        label(0.03, 0.615 - i * 0.052, t, 6.8)
    label(0.082, 0.37, "+ layout / distances", 6.4, GREY, "normal", "center")

    # encoder
    rounded(ax, 0.185, 0.30, 0.15, 0.42, "#DCEBF6", BLUE)
    label(0.260, 0.665, "Shared", 7.6, BLUE, "bold", "center")
    label(0.260, 0.625, "Transformer", 7.6, BLUE, "bold", "center")
    label(0.260, 0.585, "encoder", 7.6, BLUE, "bold", "center")
    label(0.260, 0.495, "4 layers · d = 128", 6.4, GREY, "normal", "center")
    label(0.260, 0.445, "4 token types", 6.4, GREY, "normal", "center")
    arrow(ax, (0.152, 0.51), (0.183, 0.51))

    # MA
    rounded(ax, 0.395, 0.655, 0.225, 0.265, "#FDF1DC", ORANGE)
    label(0.5075, 0.875, "MA: rule selection", 7.8, "#9A6400", "bold", "center")
    for i, (k, n) in enumerate((("PA", 12), ("LA", 3), ("RA", 3))):
        x = 0.412 + i * 0.068
        rounded(ax, x, 0.755, 0.060, 0.070, "white", ORANGE, lw=0.8)
        label(x + 0.030, 0.800, k, 6.6, ORANGE, "bold", "center")
        label(x + 0.030, 0.772, str(n), 6.8, INK, "bold", "center")
    label(0.5075, 0.700, "every ΔT / disturbance", 6.3, GREY, "normal", "center")
    arrow(ax, (0.340, 0.61), (0.400, 0.705), color=GREY, rad=-0.18)
    label(0.355, 0.665, "z", 6.5, GREY, "bold")

    # rule filter
    rounded(ax, 0.395, 0.365, 0.225, 0.215, "#FFF7E6", ORANGE, lw=0.9)
    label(0.5075, 0.545, "Rule-filtered candidates", 7.4, "#9A6400", "bold", "center")
    label(0.407, 0.493, "PA: top-16 (job, machine)", 6.5)
    label(0.407, 0.456, "LA: top-6 AGVs", 6.5)
    label(0.407, 0.419, "RA: {NOW, DEFER}", 6.5)
    label(0.5075, 0.388, "strict K · unbiased boundary tie", 6.0, GREY,
          "normal", "center")
    arrow(ax, (0.5075, 0.653), (0.5075, 0.583))

    # scoring
    rounded(ax, 0.665, 0.285, 0.170, 0.400, "#E6F5F0", GREEN)
    label(0.750, 0.655, "Candidate scoring", 7.6, "#006B4F", "bold", "center")
    for i, (k, d) in enumerate((("PA head", "9 feats"), ("LA head", "4 feats"),
                                 ("RA head", "5 feats"))):
        y = 0.520 - i * 0.085
        rounded(ax, 0.680, y, 0.140, 0.065, "white", GREEN, lw=0.8)
        label(0.690, y + 0.033, k, 6.6, "#006B4F", "bold")
        label(0.810, y + 0.033, d, 6.2, GREY, "normal", "right")
    label(0.750, 0.325, "rule embedding", 6.2, GREY, "normal", "center")
    label(0.750, 0.298, "z-norm K≥3 · softmax in K", 5.9, GREY,
          "normal", "center")
    arrow(ax, (0.622, 0.470), (0.663, 0.470))

    # encoder → scoring (routed under the filter)
    ax.plot([0.335, 0.335, 0.750, 0.750], [0.300, 0.245, 0.245, 0.283],
            color=GREY, lw=1.0, zorder=4)
    arrow(ax, (0.750, 0.260), (0.750, 0.283), color=GREY)
    label(0.500, 0.257, "token embeddings z", 6.3, GREY, "normal", "center")

    # actions
    rounded(ax, 0.875, 0.335, 0.115, 0.300, "#E6F5F0", GREEN)
    label(0.9325, 0.610, "Actions", 7.8, "#006B4F", "bold", "center")
    label(0.9325, 0.535, "dispatch", 6.5, INK, "normal", "center")
    label(0.9325, 0.505, "(job, machine)", 6.1, GREY, "normal", "center")
    label(0.9325, 0.435, "assign AGV", 6.5, INK, "normal", "center")
    label(0.9325, 0.365, "maintain / defer", 6.5, INK, "normal", "center")
    arrow(ax, (0.837, 0.485), (0.873, 0.485))

    # training
    rounded(ax, 0.185, 0.045, 0.665, 0.145, "#F2F2F2", GREY, lw=0.9)
    label(0.5175, 0.163, "Two-phase training · one model · cross-instance",
          7.4, INK, "bold", "center")
    label(0.200, 0.120,
          "A: MA only — G rule samples per top-level point", 6.4)
    label(0.200, 0.088,
          "B: scoring heads only — M execution rollouts, rules frozen", 6.4)
    label(0.200, 0.056,
          "group z-normalisation · loss = −Σ Â log p · no PPO clip", 6.4)
    # dashed feedback arrows (routed outside the filter box)
    arrow(ax, (0.330, 0.190), (0.400, 0.705), color=GREY, ls="--", lw=0.9)
    arrow(ax, (0.700, 0.190), (0.700, 0.283), color=GREY, ls="--", lw=0.9)

    out = os.path.join(HERE, "fig2_solution_overview")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig1())
    print(fig2())
