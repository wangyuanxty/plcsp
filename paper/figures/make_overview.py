"""Solution overview, three-column layout (state | controller | decision)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyBboxPatch, FancyArrowPatch,
                                Patch, Rectangle)

OUT = Path(__file__).resolve().parent

from pubstyle import PALETTE, FigureStyle, apply_publication_style
BLUE = PALETTE["blue_main"]
BLUE2 = PALETTE["blue_secondary"]
TEAL = PALETTE["teal"]
RED = PALETTE["red_strong"]
GRAY = "#4D4D4D"
GREEN = PALETTE["green_3"]
ORANGE = "#E69F00"

from pubstyle import FigureStyle, apply_publication_style


def rbox(ax, x, y, w, h, ec=BLUE, fc="white", ls="-", lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))


def title(ax, x, y, text, fs=7.5):
    ax.text(x, y, text, ha="left", va="bottom", fontsize=fs,
            fontweight="bold")


def block(ax, x, y, w, h, head, body, ec=BLUE, fc="white", fs=6.2):
    rbox(ax, x, y, w, h, ec=ec, fc=fc)
    if body:
        ax.text(x + w / 2, y + h - 0.035, head, ha="center", va="top",
                fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.01, body, ha="center", va="center",
                fontsize=fs - 0.5, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, head, ha="center", va="center",
                fontsize=fs, fontweight="bold")


def arrow(ax, p, q, color=BLUE, ls="-", lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, linestyle=ls))


def mini_shop(ax, x, y, w, h):
    rbox(ax, x, y, w, h, ec=GRAY, ls="--")
    for k in range(2):
        x0 = x + 0.008 + k * (w / 2)
        rbox(ax, x0, y + 0.02, w / 2 - 0.016, h - 0.04, ec=GRAY, ls=":")
        for i in range(3):
            mx = x0 + 0.010 + i * 0.034
            ax.add_patch(Rectangle((mx, y + h * 0.52), 0.022, 0.022,
                                   facecolor="white", edgecolor=BLUE, lw=0.6))
            ax.add_patch(Circle((mx + 0.011, y + h * 0.26), 0.006,
                                facecolor="white", edgecolor=ORANGE, lw=0.6))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.48, 1.04)
    ax.axis("off")
    ax.text(0.015, 0.985, "(a) Architecture and decision flow",
            fontsize=6.4, fontweight="bold")

    # three columns
    rbox(ax, 0.015, 0.06, 0.260, 0.90, ec=GRAY, ls="--", lw=1.1)
    rbox(ax, 0.295, 0.06, 0.40, 0.90, ec=GRAY, ls="--", lw=1.1)
    rbox(ax, 0.710, 0.06, 0.275, 0.90, ec=GRAY, ls="--", lw=1.1)
    title(ax, 0.015, 0.905, "State and candidates")
    title(ax, 0.305, 0.905, "Controller")
    title(ax, 0.720, 0.905, "Decision and feedback")

    # left column
    mini_shop(ax, 0.020, 0.62, 0.245, 0.22)
    ax.text(0.142, 0.585, "shop state: graph, machine routes, AGV states", ha="center", fontsize=4.8, color=GRAY)
    block(ax, 0.020, 0.30, 0.245, 0.24, "Candidate sets",
          "operations  $O_t$\nmachines  $M_{j,o}$\nAGVs  $A_w$", ec=GRAY)
    ax.text(0.142, 0.22, "$s_t$", ha="center", fontsize=8, color=BLUE)

    # middle: selector
    block(ax, 0.315, 0.775, 0.36, 0.115, "Top-level rule selector",
          "18-dim state -> 4 / 3 / 3 rules, every 8 steps",
          ec=BLUE, fc="#EEF3FA", fs=6.0)

    # middle: encoders and heads
    ys = [0.62, 0.47, 0.32]
    enc = [("Job encoder", "33-d tokens, 2 layers"),
           ("Machine encoder", "3 groups; intra 1L; inter 3L"),
           ("AGV encoder", "AGV tokens, 1 layer")]
    heads = ["Job head", "Machine head", "AGV head"]
    for y, (h, b) in zip(ys, enc):
        block(ax, 0.315, y, 0.16, 0.11, h, b, ec=BLUE, fs=5.9)
    for y, h in zip(ys, heads):
        block(ax, 0.515, y, 0.16, 0.11, h, "score over candidates",
              ec=BLUE2, fc="#EEF3FA", fs=5.4)
        arrow(ax, (0.475, y + 0.055), (0.515, y + 0.055))
    for y in ys:
        arrow(ax, (0.30, 0.55), (0.315, y + 0.055), color=GRAY)
    arrow(ax, (0.495, 0.775), (0.495, 0.73))
    ax.text(0.50, 0.745, "rule bias", fontsize=5.4, color=BLUE)

    # middle: GRPO
    block(ax, 0.315, 0.12, 0.36, 0.13, "GRPO update",
          "group of 8 episodes | shared z-score advantage | no critic",
          ec=BLUE, fc="#EEF3FA", fs=5.9)
    arrow(ax, (0.495, 0.25), (0.495, 0.32))

    # right column
    block(ax, 0.725, 0.66, 0.26, 0.16, "Joint action",
          "operation, machine, AGV", ec=BLUE)
    block(ax, 0.725, 0.44, 0.26, 0.16, "Shop update",
          "assignment, sequence,\nAGV routes", ec=GRAY)
    rbox(ax, 0.725, 0.12, 0.26, 0.24, ec=RED, lw=1.0)
    ax.text(0.855, 0.32, "Per-head costs", ha="center", va="center",
            fontsize=5.6, fontweight="bold")
    ax.text(0.855, 0.26, "job: mk + idle", ha="center", va="center",
            fontsize=4.8)
    ax.text(0.855, 0.21, "machine: PEC + trans", ha="center", va="center",
            fontsize=4.8)
    ax.text(0.855, 0.16, "AGV: trans + wait", ha="center", va="center",
            fontsize=4.8)
    for y in ys:
        ax.plot([0.675, 0.695], [y + 0.055, y + 0.055], color=BLUE, lw=0.9)
    ax.plot([0.695, 0.695], [ys[-1] + 0.055, 0.74], color=BLUE, lw=0.9)
    arrow(ax, (0.695, 0.74), (0.725, 0.74))
    arrow(ax, (0.855, 0.66), (0.855, 0.60), color=GRAY)
    arrow(ax, (0.855, 0.44), (0.855, 0.35), color=RED)
    ax.text(0.86, 0.395, "cost change", fontsize=5.4, color=RED)
    # feedback to selector (right margin)
    arrow(ax, (0.985, 0.27), (0.998, 0.27), color=RED, ls="--")
    arrow(ax, (0.998, 0.27), (0.998, 0.845), color=RED, ls="--")
    arrow(ax, (0.998, 0.845), (0.675, 0.845), color=RED, ls="--")
    # next state (bottom margin)
    arrow(ax, (0.855, 0.44), (0.855, 0.10), color=GRAY, ls="--")
    arrow(ax, (0.855, 0.10), (0.142, 0.10), color=GRAY, ls="--")
    arrow(ax, (0.142, 0.10), (0.142, 0.30), color=GRAY, ls="--")
    ax.text(0.50, 0.075, "next state", fontsize=5.6, ha="center")


    fig.tight_layout(rect=(0, 0.04, 1, 1))
    ax.text(0.015, -0.06, "(b) Training pipeline", fontsize=6.2, fontweight="bold")
    ty, th = -0.40, 0.24
    tx = [0.015, 0.210, 0.405, 0.600, 0.795]
    tw = 0.17
    tb = [
        ("Group sampling", ["G = 8 episodes", "one lambda per group"], BLUE),
        ("Rollout", ["job / machine / AGV", "36 or 60 steps"], BLUE),
        ("Segment close", ["every 8 steps", "weighted-cost change"], RED),
        ("Updates", ["GRPO: group z-score", "PPO: 4 epochs"], BLUE2),
        ("Checkpoint", ["top-1 on", "validation"], GRAY),
    ]
    for x, (head, body, ec) in zip(tx, tb):
        rbox(ax, x, ty, tw, th, ec=ec, lw=1.0,
             fc="#EEF3FA" if ec == BLUE else "white")
        ax.text(x + tw / 2, ty + th - 0.045, head, ha="center", va="top",
                fontsize=5.6, fontweight="bold")
        ax.text(x + tw / 2, ty + th / 2 + 0.02, body[0], ha="center",
                va="center", fontsize=4.8)
        ax.text(x + tw / 2, ty + th / 2 - 0.05, body[1], ha="center",
                va="center", fontsize=4.8)
    for i in range(4):
        arrow(ax, (tx[i] + tw, ty + th / 2), (tx[i + 1], ty + th / 2))
    arrow(ax, (tx[4] + tw / 2, ty), (tx[4] + tw / 2, ty - 0.045),
          color=GRAY, ls="--")
    arrow(ax, (tx[4] + tw / 2, ty - 0.045), (tx[0] + tw / 2, ty - 0.045),
          color=GRAY, ls="--")
    arrow(ax, (tx[0] + tw / 2, ty - 0.045), (tx[0] + tw / 2, ty),
          color=GRAY, ls="--")
    ax.text(0.5, ty - 0.035, "next group", ha="center", fontsize=4.8,
            color=GRAY)

    fig.savefig(OUT / "fig_method.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_method.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_method")


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=6.5, axes_linewidth=1.0))
    main()
