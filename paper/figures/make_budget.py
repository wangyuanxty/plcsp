"""Equal-budget curve: RGA objective against its evaluation budget.

The controller builds one schedule in a single pass, so it has no budget axis;
it is drawn as a horizontal line. The bottom axis is the number of schedule
evaluations, the top axis the wall time those evaluations need in the
environment simulator at 6x6 (0.0903 s per evaluation).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import (PALETTE, FigureStyle, apply_publication_style,
                      finalize_figure)

OUT = Path(__file__).resolve().parent

# 6x6: three-seed means over the 100 test instances, except the 530 point,
# which is one seed over five instances.
EVALS = np.array([530, 914, 1394, 2306, 4850])
OBJ = np.array([1036.52, 932.83, 868.20, 803.85, 762.04])
SEC_PER_EVAL = 437.8 / 4850.0

CONTROLLER_OBJ = 867.75
CONTROLLER_TIME = 0.95
CROSS_EVALS = 1394


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.45))

    ax.plot(EVALS, OBJ, color=PALETTE["teal"], marker="o", markersize=3.2,
            linewidth=1.1, zorder=3, label="RGA")
    ax.axhline(CONTROLLER_OBJ, color=PALETTE["blue_main"], linestyle="--",
               linewidth=1.2, zorder=2, label="Controller, one pass")

    ax.plot([CROSS_EVALS], [CONTROLLER_OBJ], marker="o", markersize=4.2,
            markerfacecolor="white", markeredgecolor=PALETTE["teal"],
            markeredgewidth=1.1, zorder=4)
    ax.annotate("1,394 evaluations, 126 s",
                xy=(CROSS_EVALS, CONTROLLER_OBJ), xytext=(1620, 782),
                fontsize=5.8, color=PALETTE["blue_main"], ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=PALETTE["blue_main"],
                                linewidth=0.6, shrinkA=1, shrinkB=2))
    ax.text(240, CONTROLLER_OBJ + 10, "one pass: 0.95 s", fontsize=5.8,
            color=PALETTE["blue_main"], ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

    ax.set_xlabel("RGA schedule evaluations", fontsize=7)
    ax.set_ylabel("Objective ($6\\times6$)", fontsize=7)
    ax.set_xlim(200, 5300)
    ax.set_ylim(700, 1120)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.4)

    top = ax.secondary_xaxis("top", functions=(lambda x: x * SEC_PER_EVAL,
                                               lambda t: t / SEC_PER_EVAL))
    top.set_xlabel("RGA wall time (s)", fontsize=6.6)
    top.tick_params(labelsize=6.2)
    top.spines["top"].set_visible(False)

    ax.legend(loc="lower left", fontsize=6, handlelength=1.2,
              borderpad=0.2, labelspacing=0.3)

    fig.tight_layout(pad=0.25)
    finalize_figure(fig, OUT / "fig_budget", formats=["pdf"])
    print("crossing", CROSS_EVALS, "->", CROSS_EVALS * SEC_PER_EVAL, "s")


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.8))
    main()