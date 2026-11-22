"""Equal-budget curve: RGA objective against its evaluation budget.

Three instance families.  The controller builds one schedule in a single pass,
so it has no budget axis and is drawn as a horizontal line.  The 6x6 curve is
the three-seed mean over the 100 test instances (the 530 point is one seed over
five instances); the 10x6 and 10x10 curves are three-seed means over the first
ten test instances.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import (PALETTE, FigureStyle, apply_publication_style,
                      finalize_figure)

OUT = Path(__file__).resolve().parent

EVALS = np.array([530, 914, 1394, 2306, 4850])

CURVES = {
    r"$6\times6\times2$": (np.array([1036.52, 932.83, 868.20, 803.85, 762.04]),
                           867.75),
    r"$10\times6\times2$": (np.array([1245.44, 1101.34, 997.46, 892.33, 769.26]),
                            808.67),
    r"$10\times10\times2$": (np.array([3703.24, 3339.82, 3073.87, 2734.32,
                                       2399.74]), 1836.64),
}

YLIMS = [(700, 1120), (700, 1300), (1750, 3900)]
CROSS_EVALS = 1394


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))

    for ax, (title, (obj, ours)), ylim in zip(axes, CURVES.items(), YLIMS):
        ax.plot(EVALS, obj, color=PALETTE["teal"], marker="o",
                markersize=3.0, linewidth=1.1, zorder=3, label="RGA")
        ax.axhline(ours, color=PALETTE["blue_main"], linestyle="--",
                   linewidth=1.1, zorder=2, label="Our model, one pass")
        ax.set_title(title, loc="left", fontsize=7.4)
        ax.set_xlabel("RGA schedule evaluations", fontsize=6.6)
        ax.set_xlim(200, 5300)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.2)

    axes[0].set_ylabel("Objective", fontsize=6.8)
    axes[0].plot([CROSS_EVALS], [CURVES[r"$6\times6\times2$"][1]],
                 marker="o", markersize=4.0, markerfacecolor="white",
                 markeredgecolor=PALETTE["teal"], markeredgewidth=1.0,
                 zorder=4)
    axes[0].annotate("1,394 evaluations", xy=(CROSS_EVALS, 867.75),
                     xytext=(2250, 960), fontsize=5.6,
                     color=PALETTE["blue_main"], ha="left", va="bottom",
                     arrowprops=dict(arrowstyle="-", color=PALETTE["blue_main"],
                                     linewidth=0.6, shrinkA=1, shrinkB=2))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=6.2, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.8)
    finalize_figure(fig, OUT / "fig_budget", formats=["pdf"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.8))
    main()
