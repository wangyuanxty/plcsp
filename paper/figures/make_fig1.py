"""Figure 1: shop environment and logistics processes, M2 transport model.

Every label carries information: machine state, AGV state, transport stage,
the affected objective terms, and the three decision types.
Colour-blind-safe (Okabe-Ito) with shape and line-style redundancy.

M2 transport: every workshop has one load/unload station (LU).  A job enters
at the station of the workshop that hosts its first operation, so the first
operation also has a deadhead and a loaded leg.  A cross-workshop trip is
machine -> LU_src, a station transfer with no AGV, then LU_dst -> machine.
The objective counts loaded travel plus 0.5 * empty travel.
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
LU_FC = "#F7EDF3"  # light purple tint for load/unload stations


def rbox(ax, x, y, w, h, ec=C["gray"], ls="-", lw=0.8, fc="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))


def machine(ax, x, y, s, label, ops, bars=True):
    ax.add_patch(Rectangle((x, y), s, s, facecolor="white",
                           edgecolor=C["blue"], linewidth=0.9))
    ax.text(x + s / 2, y + s * 0.64, label, ha="center", va="center",
            fontsize=5.2, fontweight="bold")
    if ops:
        ax.text(x + s / 2, y + s * 0.30, ops, ha="center", va="center",
                fontsize=4.0, color=C["gray"])
    if bars:
        ax.add_patch(Rectangle((x + 0.004, y - 0.018), s * 0.55, 0.006,
                               facecolor=C["blue"], edgecolor="none"))
        ax.add_patch(Rectangle((x + 0.004, y - 0.028), s * 0.35, 0.006,
                               facecolor=C["orange"], edgecolor="none"))


def lu_box(ax, x, y, w, h, label):
    """Load/unload station: rounded purple box, distinct from a machine."""
    rbox(ax, x, y, w, h, ec=C["purple"], lw=1.0, fc=LU_FC)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=4.4, fontweight="bold", color=C["purple"])


def agv(ax, x, y, label, state):
    busy = state.startswith("busy")
    ax.add_patch(Circle((x, y), 0.011, facecolor=C["orange"] if busy else "white",
                        edgecolor=C["orange"], linewidth=0.9, zorder=4))
    if busy:
        ax.plot([x - 0.005, x + 0.005], [y, y], color="white", lw=0.7,
                zorder=5)
    ax.text(x, y - 0.036, f"{label} {state}", ha="center", va="center",
            fontsize=4.2, zorder=6,
            color=C["verm"] if busy else C["green"])


def route(ax, p, q, loaded=True, color=None, lw=1.0, rad=0.0):
    color = color or (C["sky"] if loaded else C["gray"])
    style = "-" if loaded else "-."
    extra = {} if rad == 0 else {"connectionstyle": f"arc3,rad={rad}"}
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=5.5, linewidth=lw,
        color=color, linestyle=style, shrinkA=0.5, shrinkB=0.5,
        zorder=3, **extra))


def station_transfer(ax, p, q, lw=1.4):
    """Station-to-station transfer: no AGV, dotted purple."""
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=5.0, linewidth=lw,
        color=C["purple"], linestyle=":", shrinkA=0.5, shrinkB=0.5,
        zorder=3))


def chip(ax, x, y, text, ec):
    rbox(ax, x, y, 0.085, 0.035, ec=ec, lw=0.8)
    ax.text(x + 0.0425, y + 0.0175, text, ha="center", va="center", fontsize=4.2)


def panel_a(ax):
    rbox(ax, 0.02, 0.16, 0.46, 0.74, ls="--")
    rbox(ax, 0.52, 0.16, 0.46, 0.74, ls="--")
    ax.text(0.25, 0.87, "Workshop 1  (3 machines, 3 AGVs)", ha="center",
            fontsize=5.2)
    ax.text(0.75, 0.87, "Workshop 2  (3 machines, 3 AGVs)", ha="center",
            fontsize=5.2)
    for i, x in enumerate((0.05, 0.18, 0.31)):
        machine(ax, x, 0.52, 0.075, f"M{i+1}", "p | e")
        if i < 2:
            rbox(ax, x + 0.080, 0.54, 0.030, 0.040, ls=":", lw=0.6)
            ax.text(x + 0.095, 0.525, "buf", ha="center", fontsize=3.6,
                    color=C["gray"])
    for i, x in enumerate((0.56, 0.69, 0.82)):
        machine(ax, x, 0.52, 0.075, f"M{i+4}", "p | e")
        if i < 2:
            rbox(ax, x + 0.080, 0.54, 0.030, 0.040, ls=":", lw=0.6)
            ax.text(x + 0.095, 0.525, "buf", ha="center", fontsize=3.6,
                    color=C["gray"])
    agv(ax, 0.07, 0.27, "A1", "busy")
    agv(ax, 0.19, 0.27, "A2", "free")
    agv(ax, 0.31, 0.27, "A3", "free")
    agv(ax, 0.60, 0.27, "A4", "free")
    agv(ax, 0.72, 0.27, "A5", "free")
    agv(ax, 0.84, 0.27, "A6", "free")
    lu_box(ax, 0.405, 0.66, 0.065, 0.070, "LU1")
    lu_box(ax, 0.525, 0.66, 0.065, 0.070, "LU2")
    station_transfer(ax, (0.470, 0.695), (0.525, 0.695))
    route(ax, (0.2175, 0.600), (0.405, 0.672), loaded=True,
          color=C["sky"], lw=1.2, rad=-0.08)
    route(ax, (0.558, 0.660), (0.598, 0.600), loaded=True,
          color=C["sky"], lw=1.2, rad=-0.05)
    ax.text(0.34, 0.81, r"job $J_3$: $O_2\to O_4$", ha="center", fontsize=5.0,
            color=C["verm"])
    ax.text(0.02, 0.07, "decisions:", fontsize=4.4)
    chip(ax, 0.10, 0.05, "job", C["blue"])
    chip(ax, 0.20, 0.05, "machine", C["blue"])
    chip(ax, 0.30, 0.05, "AGV", C["orange"])
    ax.text(0.40, 0.07, "at each step", fontsize=4.2, color=C["gray"])
    ax.set_title("(b) Shop environment", loc="left", fontsize=6.2)


def panel_b(ax, title):
    """Cross-workshop case: M1 -> LU1 -> LU2 -> M4."""
    rbox(ax, 0.02, 0.10, 0.42, 0.82, ls="--")
    rbox(ax, 0.56, 0.10, 0.42, 0.82, ls="--")
    ax.text(0.23, 0.88, "W1 (source)", ha="center", fontsize=4.4)
    ax.text(0.77, 0.88, "W2 (target)", ha="center", fontsize=4.4)
    machine(ax, 0.08, 0.44, 0.10, "M1", "", bars=False)
    machine(ax, 0.74, 0.44, 0.10, "M4", "", bars=False)
    lu_box(ax, 0.335, 0.62, 0.090, 0.070, "LU1")
    lu_box(ax, 0.575, 0.62, 0.090, 0.070, "LU2")
    agv(ax, 0.10, 0.235, "A1", "busy")
    agv(ax, 0.62, 0.235, "A3", "busy")
    route(ax, (0.100, 0.246), (0.115, 0.435), loaded=False)
    ax.text(0.072, 0.340, "deadhead", fontsize=4.0, rotation=90,
            color=C["gray"])
    route(ax, (0.180, 0.490), (0.335, 0.655), loaded=True)
    ax.text(0.235, 0.610, "loaded", fontsize=4.0, color=C["sky"])
    station_transfer(ax, (0.425, 0.655), (0.575, 0.655))
    ax.text(0.500, 0.790, "transfer", ha="center", fontsize=4.0,
            color=C["purple"])
    ax.text(0.500, 0.750, "no AGV", ha="center", fontsize=4.0,
            color=C["purple"])
    route(ax, (0.620, 0.246), (0.620, 0.615), loaded=False)
    ax.text(0.648, 0.440, "deadhead", fontsize=4.0, rotation=90,
            color=C["gray"])
    route(ax, (0.665, 0.655), (0.765, 0.540), loaded=True)
    ax.text(0.720, 0.630, "loaded", fontsize=4.0, color=C["sky"])
    ax.scatter([0.180], [0.490], marker="^", s=16, color=C["green"], zorder=5)
    ax.scatter([0.765], [0.540], marker="*", s=28, color=C["green"], zorder=5)
    ax.text(0.50, 0.055, "affects idleT + transport", ha="center", fontsize=4.0,
            color=C["verm"])
    ax.set_title(title, loc="left", fontsize=5.8)


def panel_c(ax, title):
    """Same workshop: entry case and later operation."""
    rbox(ax, 0.02, 0.10, 0.96, 0.82, ls="--")
    ax.plot([0.05, 0.95], [0.60, 0.60], color=C["gray"], lw=0.5,
            ls=(0, (2, 2)))
    ax.text(0.50, 0.860, "entry: job starts at LU1", ha="center", fontsize=4.4)
    # entry: deadhead to the station, then loaded station -> machine
    agv(ax, 0.13, 0.730, "A2", "busy")
    lu_box(ax, 0.30, 0.685, 0.10, 0.075, "LU1")
    machine(ax, 0.60, 0.685, 0.09, "M2", "", bars=False)
    route(ax, (0.142, 0.730), (0.300, 0.730), loaded=False)
    ax.text(0.221, 0.772, "deadhead", ha="center", fontsize=4.0,
            color=C["gray"])
    route(ax, (0.400, 0.730), (0.600, 0.730), loaded=True)
    ax.text(0.500, 0.772, "loaded", ha="center", fontsize=4.0, color=C["sky"])
    ax.scatter([0.400], [0.730], marker="^", s=16, color=C["green"], zorder=5)
    ax.scatter([0.600], [0.730], marker="*", s=28, color=C["green"], zorder=5)
    # later: deadhead to the pickup machine, then one loaded leg
    ax.text(0.50, 0.565, "later: machine -> machine", ha="center",
            fontsize=4.4)
    agv(ax, 0.15, 0.310, "A2", "busy")
    machine(ax, 0.22, 0.42, 0.09, "M2", "", bars=False)
    machine(ax, 0.62, 0.42, 0.09, "M3", "", bars=False)
    route(ax, (0.150, 0.321), (0.235, 0.415), loaded=False)
    ax.text(0.115, 0.375, "deadhead", fontsize=4.0, rotation=90,
            color=C["gray"])
    route(ax, (0.310, 0.465), (0.620, 0.465), loaded=True)
    ax.text(0.465, 0.495, "loaded", ha="center", fontsize=4.0, color=C["sky"])
    ax.scatter([0.310], [0.465], marker="^", s=16, color=C["green"], zorder=5)
    ax.scatter([0.620], [0.465], marker="*", s=28, color=C["green"], zorder=5)
    ax.text(0.50, 0.170, "affects transport only", ha="center", fontsize=4.0,
            color=C["sky"])
    ax.set_title(title, loc="left", fontsize=5.8)


def panel_d(ax, title):
    rbox(ax, 0.02, 0.10, 0.96, 0.82, ls="--")
    ax.text(0.50, 0.88, "job complete", ha="center", fontsize=4.8)
    machine(ax, 0.43, 0.56, 0.08, "M6", "", bars=False)
    agv(ax, 0.22, 0.48, "A2", "free")
    agv(ax, 0.70, 0.48, "A4", "free")
    ax.text(0.50, 0.70, "job ends at M6; no exit transport", ha="center",
            fontsize=4.0, color=C["gray"])
    ax.text(0.47, 0.52, "done", ha="center", fontsize=4.2, color=C["green"])
    rbox(ax, 0.06, 0.12, 0.88, 0.24, lw=0.7)
    ax.text(0.50, 0.315, "what the decisions control", ha="center",
            fontsize=4.2, color=C["gray"])
    ax.text(0.50, 0.265, "job -> sequence", ha="center", fontsize=3.9)
    ax.text(0.50, 0.215, "machine -> PEC, idleT", ha="center", fontsize=3.9)
    ax.text(0.50, 0.165, "AGV -> transport, idleT", ha="center", fontsize=3.9)
    ax.set_title(title, loc="left", fontsize=5.8)


def panel_concept(ax):
    # top level: one rule selector
    rbox(ax, 0.30, 0.80, 0.40, 0.15, ec=C["purple"], lw=1.0)
    ax.text(0.50, 0.895, "Rule selector agent", ha="center", va="center",
            fontsize=6.0, fontweight="bold")
    ax.text(0.50, 0.835, r"one rule per agent, every $\Delta T$ steps",
            ha="center", va="center", fontsize=4.6)

    # bottom level: three coupled shop decisions
    boxes = [("Job agent", "which operation", C["blue"]),
             ("Machine agent", "which machine", C["blue"]),
             ("AGV agent", "which vehicle", C["orange"])]
    xs = [0.06, 0.38, 0.70]
    w, y, h = 0.24, 0.38, 0.28
    for x, (t, b, ec) in zip(xs, boxes):
        rbox(ax, x, y, w, h, ec=ec, lw=1.0)
        ax.text(x + w / 2, y + h - 0.06, t, ha="center", va="top",
                fontsize=6.0, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.04, b, ha="center", va="center",
                fontsize=5.4)
        ax.add_patch(FancyArrowPatch((0.50, 0.80), (x + w / 2, y + h),
                                     arrowstyle="-|>", mutation_scale=7,
                                     lw=0.9, color=C["purple"]))
    for i in range(2):
        ax.add_patch(FancyArrowPatch((xs[i] + w, y + h / 2),
                                     (xs[i + 1], y + h / 2),
                                     arrowstyle="-|>", mutation_scale=7,
                                     lw=1.0, color=C["blue"]))
    ax.plot([0.82, 0.82, 0.18, 0.18], [y, 0.28, 0.28, y],
            color=C["gray"], ls="--", lw=0.9)
    ax.add_patch(FancyArrowPatch((0.18, 0.28), (0.18, y),
                                 arrowstyle="-|>", mutation_scale=6,
                                 lw=0.9, color=C["gray"]))
    ax.text(0.50, 0.315, "shop state", ha="center", fontsize=5.0,
            color=C["gray"])
    ax.text(0.06, 0.13, "objective:", fontsize=5.2)
    for x, cw, lab, col in ((0.16, 0.10, "mk", C["blue"]),
                            (0.29, 0.10, "PEC", C["orange"]),
                            (0.42, 0.12, "idleT", C["verm"])):
        rbox(ax, x, 0.08, cw, 0.10, ec=col, lw=0.8)
        ax.text(x + cw / 2, 0.13, lab, ha="center", va="center", fontsize=4.8)
    rbox(ax, 0.57, 0.08, 0.12, 0.10, ec=C["sky"], lw=0.8)
    ax.text(0.63, 0.13, "TEC", ha="center", va="center", fontsize=4.8)
    ax.text(0.72, 0.13, r"$T_{\mathrm{loaded}} + 0.5\,T_{\mathrm{empty}}$",
            ha="left", va="center", fontsize=4.2)
    ax.set_title("(a) Four decisions: one rule selector above three "
                 "coupled shop decisions", loc="left", fontsize=6.2)


def main():
    fig = plt.figure(figsize=(7.2, 5.4))
    base = fig.add_axes([0, 0, 1, 1])
    base.axis("off")
    for rect, fn, title in (
            ([0.00, 0.75, 1.00, 0.23], panel_concept, None),
            ([0.00, 0.40, 1.00, 0.33], panel_a, None),
            ([0.00, 0.03, 0.32, 0.36], panel_b,
             "(c) Cross-workshop transport"),
            ([0.34, 0.03, 0.32, 0.36], panel_c,
             "(d) Same-workshop transports"),
            ([0.68, 0.03, 0.32, 0.36], panel_d,
             "(e) After the last operation")):
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
        Line2D([0], [0], color=C["purple"], lw=1.4, ls=":",
               label="station transfer"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor=C["blue"], markersize=6, label="machine"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=LU_FC,
               markeredgecolor=C["purple"], markersize=6, label="LU station"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["orange"],
               markeredgecolor=C["orange"], markersize=6, label="AGV busy"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=C["orange"], markersize=6, label="AGV free"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=4.8, handlelength=1.3,
               columnspacing=0.8, handletextpad=0.4)
    finalize_figure(fig, OUT / "fig_environment", formats=["pdf", "png"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=6.0, axes_linewidth=0.8))
    main()
