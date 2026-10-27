"""Figure 1: shop environment and logistics processes, dense version.

Every label carries information: machine state, AGV state, transport stage,
the affected objective terms, and the three decision types.
Colour-blind-safe (Okabe-Ito) with shape and line-style redundancy.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)

from pubstyle import FigureStyle, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "verm": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
     "gray": "#4D4D4D"}


def rbox(ax, x, y, w, h, ec=C["gray"], ls="-", lw=0.8, fc="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))


def machine(ax, x, y, s, label, ops):
    ax.add_patch(Rectangle((x, y), s, s, facecolor="white",
                           edgecolor=C["blue"], linewidth=0.9))
    ax.text(x + s / 2, y + s * 0.64, label, ha="center", va="center",
            fontsize=5.2, fontweight="bold")
    if ops:
        ax.text(x + s / 2, y + s * 0.30, ops, ha="center", va="center",
                fontsize=4.0, color=C["gray"])
    ax.add_patch(Rectangle((x + 0.004, y - 0.018), s * 0.55, 0.006,
                           facecolor=C["blue"], edgecolor="none"))
    ax.add_patch(Rectangle((x + 0.004, y - 0.028), s * 0.35, 0.006,
                           facecolor=C["orange"], edgecolor="none"))


def agv(ax, x, y, label, state):
    busy = state.startswith("busy")
    ax.add_patch(Circle((x, y), 0.011, facecolor=C["orange"] if busy else "white",
                        edgecolor=C["orange"], linewidth=0.9))
    if busy:
        ax.plot([x - 0.005, x + 0.005], [y, y], color="white", lw=0.7)
    ax.text(x, y - 0.026, label, ha="center", va="center", fontsize=4.6)
    ax.text(x, y + 0.018, state, ha="center", va="center", fontsize=4.0,
            color=C["verm"] if busy else C["green"])


def route(ax, p, q, loaded=True, color=None, lw=1.0):
    color = color or (C["sky"] if loaded else C["gray"])
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=5.5, linewidth=lw,
        color=color, linestyle="-" if loaded else "-.", shrinkA=0.5, shrinkB=0.5))


def chip(ax, x, y, text, ec):
    rbox(ax, x, y, 0.085, 0.035, ec=ec, lw=0.8)
    ax.text(x + 0.0425, y + 0.0175, text, ha="center", va="center", fontsize=4.2)


def panel_a(ax):
    rbox(ax, 0.02, 0.16, 0.46, 0.74, ls="--")
    rbox(ax, 0.52, 0.16, 0.46, 0.74, ls="--")
    ax.text(0.25, 0.87, "Workshop 1  (3 machines, 2 AGVs)", ha="center", fontsize=5.2)
    ax.text(0.75, 0.87, "Workshop 2  (3 machines, 2 AGVs)", ha="center", fontsize=5.2)
    for i, x in enumerate((0.06, 0.19, 0.32)):
        machine(ax, x, 0.52, 0.075, f"M{i+1}", "p | e")
        if i < 2:
            rbox(ax, x + 0.080, 0.54, 0.030, 0.040, ls=":", lw=0.6)
            ax.text(x + 0.095, 0.525, "buf", ha="center", fontsize=3.4, color=C["gray"])
    for i, x in enumerate((0.56, 0.69, 0.82)):
        machine(ax, x, 0.52, 0.075, f"M{i+4}", "p | e")
        if i < 2:
            rbox(ax, x + 0.080, 0.54, 0.030, 0.040, ls=":", lw=0.6)
            ax.text(x + 0.095, 0.525, "buf", ha="center", fontsize=3.4, color=C["gray"])
    agv(ax, 0.09, 0.30, "A1", "busy")
    agv(ax, 0.25, 0.30, "A2", "free")
    agv(ax, 0.59, 0.30, "A3", "free")
    agv(ax, 0.75, 0.30, "A4", "free")
    ax.add_patch(FancyArrowPatch((0.135, 0.64), (0.555, 0.64),
                                 connectionstyle="arc3,rad=-0.22",
                                 arrowstyle="-|>", mutation_scale=7,
                                 linewidth=1.2, color=C["verm"]))
    ax.text(0.34, 0.79, r"job $J_3$: $O_2\to O_4$", ha="center", fontsize=5.0,
            color=C["verm"])
    ax.text(0.49, 0.19, "boundary", ha="center", fontsize=4.2, color=C["gray"])
    ax.text(0.02, 0.07, "decisions:", fontsize=4.4)
    chip(ax, 0.10, 0.05, "job", C["blue"])
    chip(ax, 0.20, 0.05, "machine", C["blue"])
    chip(ax, 0.30, 0.05, "AGV", C["orange"])
    ax.text(0.40, 0.07, "at each step", fontsize=4.2, color=C["gray"])
    ax.set_title("(b) Shop environment", loc="left", fontsize=6.2)


def panel_b(ax, title):
    rbox(ax, 0.02, 0.10, 0.42, 0.82, ls="--")
    rbox(ax, 0.56, 0.10, 0.42, 0.82, ls="--")
    ax.text(0.23, 0.88, "W1", ha="center", fontsize=4.8)
    ax.text(0.77, 0.88, "W2", ha="center", fontsize=4.8)
    machine(ax, 0.06, 0.58, 0.08, "M1", "")
    machine(ax, 0.62, 0.58, 0.08, "M4", "")
    agv(ax, 0.08, 0.28, "A1", "busy")
    agv(ax, 0.64, 0.28, "A3", "busy")
    route(ax, (0.08, 0.30), (0.095, 0.56), loaded=False)
    ax.text(0.032, 0.42, "empty", fontsize=4.0, rotation=90, color=C["gray"])
    route(ax, (0.14, 0.62), (0.50, 0.62), loaded=True)
    ax.text(0.30, 0.65, r"loaded $\tau_s$", fontsize=4.2, color=C["sky"])
    ax.text(0.50, 0.53, r"$\delta$", fontsize=4.8, color=C["verm"])
    ax.plot([0.50, 0.50], [0.58, 0.66], color=C["verm"], lw=1.0, ls=":")
    route(ax, (0.50, 0.62), (0.62, 0.62), loaded=True)
    ax.text(0.565, 0.65, r"$\tau_d$", fontsize=4.2, color=C["sky"])
    route(ax, (0.68, 0.56), (0.64, 0.30), loaded=False)
    ax.text(0.685, 0.42, "empty", fontsize=4.0, rotation=-70, color=C["gray"])
    ax.scatter([0.14], [0.62], marker="^", s=16, color=C["green"], zorder=5)
    ax.scatter([0.62], [0.62], marker="*", s=28, color=C["green"], zorder=5)
    ax.add_patch(Rectangle((0.62, 0.70), 0.08, 0.010, facecolor="none",
                           edgecolor=C["verm"], hatch="///", lw=0.6))
    ax.text(0.66, 0.725, "M4 idle", ha="center", fontsize=3.6, color=C["verm"])
    ax.text(0.50, 0.17, "affects idleT + transT", ha="center", fontsize=4.2,
            color=C["verm"])
    ax.set_title(title, loc="left", fontsize=5.8)


def panel_c(ax, title):
    rbox(ax, 0.04, 0.10, 0.92, 0.82, ls="--")
    ax.text(0.50, 0.88, "same workshop", ha="center", fontsize=4.8)
    machine(ax, 0.16, 0.58, 0.08, "M2", "")
    machine(ax, 0.48, 0.58, 0.08, "M3", "")
    agv(ax, 0.18, 0.28, "A2", "busy")
    route(ax, (0.18, 0.30), (0.20, 0.56), loaded=False)
    ax.text(0.135, 0.42, "empty", fontsize=4.0, rotation=90, color=C["gray"])
    route(ax, (0.24, 0.62), (0.48, 0.62), loaded=True)
    ax.text(0.35, 0.65, r"loaded $\tau$", fontsize=4.2, color=C["sky"])
    ax.scatter([0.24], [0.62], marker="^", s=16, color=C["green"], zorder=5)
    ax.scatter([0.48], [0.62], marker="*", s=28, color=C["green"], zorder=5)
    ax.text(0.50, 0.17, "affects transT only", ha="center", fontsize=4.2,
            color=C["sky"])
    ax.set_title(title, loc="left", fontsize=5.8)


def panel_d(ax, title):
    rbox(ax, 0.02, 0.10, 0.94, 0.82, ls="--")
    ax.text(0.50, 0.88, "job complete", ha="center", fontsize=4.8)
    machine(ax, 0.42, 0.58, 0.08, "M6", "")
    agv(ax, 0.20, 0.50, "A2", "free")
    agv(ax, 0.62, 0.50, "A4", "free")
    ax.text(0.46, 0.70, "no transport left", ha="center", fontsize=4.2,
            color=C["gray"])
    ax.text(0.46, 0.52, "done", ha="center", fontsize=4.2, color=C["green"])
    rbox(ax, 0.06, 0.14, 0.88, 0.22, lw=0.7)
    ax.text(0.50, 0.31, "what the decisions control", ha="center", fontsize=4.2,
            color=C["gray"])
    ax.text(0.50, 0.26, "job -> sequence", ha="center", fontsize=3.9)
    ax.text(0.50, 0.21, "machine -> PEC, idleT", ha="center", fontsize=3.9)
    ax.text(0.50, 0.16, "AGV -> transT, idleT", ha="center", fontsize=3.9)
    ax.set_title(title, loc="left", fontsize=5.8)


def panel_concept(ax):
    boxes = [("Job head", "which operation", C["blue"]),
             ("Machine head", "which machine", C["blue"]),
             ("AGV head", "which vehicle", C["orange"])]
    xs = [0.06, 0.38, 0.70]
    w, y, h = 0.24, 0.42, 0.30
    for x, (t, b, ec) in zip(xs, boxes):
        rbox(ax, x, y, w, h, ec=ec, lw=1.0)
        ax.text(x + w / 2, y + h - 0.06, t, ha="center", va="top",
                fontsize=6.0, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.04, b, ha="center", va="center",
                fontsize=5.4)
    for i in range(2):
        ax.add_patch(FancyArrowPatch((xs[i] + w, y + h / 2),
                                     (xs[i + 1], y + h / 2),
                                     arrowstyle="-|>", mutation_scale=7,
                                     lw=1.0, color=C["blue"]))
    ax.plot([0.82, 0.82, 0.18, 0.18], [y, 0.34, 0.34, y],
            color=C["gray"], ls="--", lw=0.9)
    ax.add_patch(FancyArrowPatch((0.18, 0.34), (0.18, y),
                                 arrowstyle="-|>", mutation_scale=6,
                                 lw=0.9, color=C["gray"]))
    ax.text(0.50, 0.355, "shop state", ha="center", fontsize=5.0,
            color=C["gray"])
    ax.text(0.06, 0.16, "objective:", fontsize=5.2)
    for i, (lab, col) in enumerate([("mk", C["blue"]), ("PEC", C["orange"]),
                                    ("idleT", C["verm"]), ("transT", C["sky"])]):
        x = 0.16 + i * 0.16
        rbox(ax, x, 0.10, 0.12, 0.10, ec=col, lw=0.8)
        ax.text(x + 0.06, 0.15, lab, ha="center", va="center", fontsize=4.8)
    ax.set_title("(a) Conceptual architecture: three coupled decisions",
                 loc="left", fontsize=6.2)


def main():
    fig = plt.figure(figsize=(7.2, 5.4))
    base = fig.add_axes([0, 0, 1, 1])
    base.axis("off")
    for rect, fn, title in (
            ([0.00, 0.80, 1.00, 0.18], panel_concept, None),
            ([0.00, 0.44, 1.00, 0.34], panel_a, None),
            ([0.00, 0.03, 0.32, 0.36], panel_b, "(c) First operation: two legs"),
            ([0.34, 0.03, 0.32, 0.36], panel_c, "(d) Later operation: one leg"),
            ([0.68, 0.03, 0.32, 0.36], panel_d, "(e) After the last operation")):
        sub = base.inset_axes(rect)
        sub.set_xlim(0, 1)
        sub.set_ylim(0, 1)
        sub.axis("off")
        if title is None:
            fn(sub)
        else:
            fn(sub, title)

    handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C["green"],
               markersize=5, label="pickup"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C["green"],
               markersize=7, label="drop-off"),
        Line2D([0], [0], color=C["sky"], lw=1.2, label="loaded"),
        Line2D([0], [0], color=C["gray"], lw=1.0, ls="-.", label="empty"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor=C["blue"], markersize=6, label="machine"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["orange"],
               markeredgecolor=C["orange"], markersize=6, label="AGV busy"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=C["orange"], markersize=6, label="AGV free"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=5.0)
    finalize_figure(fig, OUT / "fig_environment", formats=["pdf", "png"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=6.0, axes_linewidth=0.8))
    main()
