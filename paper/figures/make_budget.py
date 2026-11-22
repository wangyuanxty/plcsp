"""Equal-budget curve: RGA objective against its computation time.

Three instance families.  The controller builds one schedule in a single pass,
so it has no budget axis and is drawn as a horizontal line.  The 6x6 curve is
the three-seed mean over the 100 test instances (the 530 point is one seed over
five instances); the 10x6 and 10x10 curves are three-seed means over the first
ten test instances.  Time is the number of evaluations times the per-evaluation
simulator cost at each size (437.8, 1,426 and 4,161 s for 4,850 evaluations).
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
SEC_PER_EVAL = {"6x6": 437.8 / 4850.0,
                "10x6": 1426.0 / 4850.0,
                "10x10": 4161.0 / 4850.0}

CURVES = {
    "6x6": (r"$6\times6\times2$",
            np.array([1036.52, 932.83, 868.20, 803.85, 762.04]), 867.75),
    "10x6": (r"$10\times6\times2$",
             np.array([1245.44, 1101.34, 997.46, 892.33, 769.26]), 808.67),
    "10x10": (r"$10\times10\times2$",
              np.array([3703.24, 3339.82, 3073.87, 2734.32, 2399.74]),
              1836.64),
}

YLIMS = [(700, 1120), (700, 1300), (1750, 3900)]
XLIMS = [(0, 500), (0, 1520), (0, 4400)]
CROSS_EVALS = 1394


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

    for ax, (key, (title, obj, ours)), ylim, xlim in zip(
            axes, CURVES.items(), YLIMS, XLIMS):
        t = EVALS * SEC_PER_EVAL[key]
        ax.plot(t, obj, color=PALETTE["teal"], marker="o", markersize=3.0,
                linewidth=1.1, zorder=3, label="RGA")
        ax.axhline(ours, color=PALETTE["blue_main"], linestyle="--",
                   linewidth=1.1, zorder=2, label="Our model, one schedule")
        ax.text(0.02, 0.96, title, transform=ax.transAxes, ha="left",
                va="top", fontsize=7.4, zorder=5)
        for e, y in zip(EVALS, obj):
            if key == "6x6" and e == 1394:
                continue
            first = e == EVALS[0]
            ax.annotate(f"{int(e):,}", (e * SEC_PER_EVAL[key], y),
                        textcoords="offset points",
                        xytext=(5, -7) if first else (-5, -7),
                        fontsize=5.6, color=PALETTE["teal"],
                        ha="left" if first else "right", va="top")
        ax.set_xlabel("RGA computation time (s)", fontsize=6.6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.2)

    axes[0].set_ylabel("Objective", fontsize=6.8)
    cross_t = CROSS_EVALS * SEC_PER_EVAL["6x6"]
    axes[0].plot([cross_t], [CURVES["6x6"][2]], marker="o", markersize=4.0,
                 markerfacecolor="white", markeredgecolor=PALETTE["teal"],
                 markeredgewidth=1.0, zorder=4)
    axes[0].annotate("1,394 evals\n126 s", xy=(cross_t, CURVES["6x6"][2]),
                     xytext=(150, 915), fontsize=5.6,
                     color=PALETTE["blue_main"], ha="left", va="bottom",
                     arrowprops=dict(arrowstyle="-", color=PALETTE["blue_main"],
                                     linewidth=0.6, shrinkA=1, shrinkB=2))

    cross_e = 4035.3
    t106 = cross_e * SEC_PER_EVAL["10x6"]
    axes[1].plot([t106], [CURVES["10x6"][2]], marker="o", markersize=4.0,
                 markerfacecolor="white", markeredgecolor=PALETTE["teal"],
                 markeredgewidth=1.0, zorder=4)
    axes[1].annotate("4,035 evals\n1,186 s", xy=(t106, CURVES["10x6"][2]),
                     xytext=(1000, 950), fontsize=5.6,
                     color=PALETTE["blue_main"], ha="left", va="bottom",
                     arrowprops=dict(arrowstyle="-", color=PALETTE["blue_main"],
                                     linewidth=0.6, shrinkA=1, shrinkB=2))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=6.2, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.8)
    finalize_figure(fig, OUT / "fig_budget", formats=["pdf"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.8))
    main()
