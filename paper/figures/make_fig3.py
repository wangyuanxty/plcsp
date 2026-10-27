"""Figure 3: training pipeline."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pubstyle import FigureStyle, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "verm": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7", "gray": "#4D4D4D"}


def box(ax, x, y, w, h, head, body, ec, fc="white", fs=6.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h - 0.055, head, ha="center", va="top",
            fontsize=fs, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.02, body, ha="center", va="center",
            fontsize=fs - 0.6, linespacing=1.25)


def arrow(ax, p, q, color=C["blue"], ls="-", lw=1.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=7, linewidth=lw,
        color=color, linestyle=ls, shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y, h = 0.48, 0.34
    xs = [0.02, 0.18, 0.34, 0.50, 0.66, 0.82]
    w = 0.14
    items = [
        ("Group sampling", "G = 8 episodes\none $\\lambda$ per group", C["blue"]),
        ("Rule selection", "18-d state\n4 / 3 / 3 rules\nevery 8 steps", C["green"]),
        ("Rollout", "job / machine / AGV\n36 or 60 steps", C["blue"]),
        ("Segment close", "every 8 steps\nweighted-cost change", C["verm"]),
        ("GRPO update", "group z-score\nshared advantage\nno critic", C["blue"]),
        ("PPO update", "segment returns\ntop level, 4 epochs", C["purple"]),
    ]
    for x, (head, body, ec) in zip(xs, items):
        box(ax, x, y, w, h, head, body, ec, fc="#EEF3FA" if ec == C["blue"] else "white")
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))

    # per-head costs
    box(ax, 0.50, 0.10, 0.30, 0.22, "Per-head costs",
        "job: mk + idle\nmachine: PEC + trans\nAGV: trans + wait",
        C["verm"], fs=5.6)
    arrow(ax, (0.65, 0.48), (0.65, 0.32), color=C["verm"])
    ax.text(0.66, 0.40, "segment reward", fontsize=5.0, color=C["verm"])

    # checkpoint selection
    box(ax, 0.83, 0.10, 0.15, 0.22, "Checkpoint",
        "top-1 on\nvalidation", C["gray"], fs=5.4)
    arrow(ax, (0.89, 0.48), (0.89, 0.32), color=C["gray"], ls="--")

    # loop
    arrow(ax, (0.89, 0.10), (0.89, 0.04), color=C["gray"], ls="--")
    arrow(ax, (0.89, 0.04), (0.09, 0.04), color=C["gray"], ls="--")
    arrow(ax, (0.09, 0.04), (0.09, 0.48), color=C["gray"], ls="--")
    ax.text(0.49, 0.055, "next group", ha="center", fontsize=5.2, color=C["gray"])

    ax.text(0.02, 0.90, "(a) One training iteration = one group of 8 episodes",
            fontsize=6.2, fontweight="bold")
    fig.savefig(OUT / "fig_training.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_training.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=6.0, axes_linewidth=0.8))
    main()
