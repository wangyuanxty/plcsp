"""Main comparison: objective of every method, one panel per instance family.

The three families are not comparable across columns (the idle term grows with
the number of machines), so each panel carries its own y-range. Bars are means
over the 100 test instances; the learned methods carry the seed standard
deviation.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import (PALETTE, FigureStyle, apply_publication_style,
                      finalize_figure)

OUT = Path(__file__).resolve().parent

SIZES = [r"$6\times6\times2$", r"$10\times6\times2$", r"$10\times10\times2$"]

METHODS = [
    ("Ours", PALETTE["blue_main"], [867.75, 808.67, 1836.64],
     [11.86, 12.83, 43.45]),
    ("RGA", PALETTE["teal"], [762.04, 796.70, 2298.63], [0, 0, 0]),
    ("Wang et al.", PALETTE["red_strong"], [1075.64, 1066.24, 3153.81],
     [73.82, 102.61, 530.01]),
    ("Li et al.", PALETTE["violet"], [1160.59, 1198.09, 2738.91],
     [42.40, 76.91, 82.76]),
    ("PDR", PALETTE["neutral"], [1319.24, 1461.63, 3288.35], [0, 0, 0]),
]

# One y-range per panel: the families are not comparable with each other.
YLIM = [(0, 1520), (0, 1680), (0, 3720)]
SHORT = ["Ours", "RGA", "Wang", "Li", "PDR"]


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))

    width = 0.78
    x = np.arange(len(METHODS))

    for k, ax in enumerate(axes):
        for xi, (name, colour, vals, errs) in enumerate(METHODS):
            ax.bar(xi, vals[k], width, label=name, color=colour,
                   edgecolor="white", linewidth=0.4, zorder=2)
            if errs[k]:
                ax.errorbar(xi, vals[k], yerr=errs[k], fmt="none",
                            ecolor="#333333", elinewidth=0.7, capsize=1.4,
                            capthick=0.7, zorder=3)
        ax.set_title(SIZES[k], fontsize=7, pad=2)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.set_ylim(*YLIM[k])
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT, rotation=90, ha="center", fontsize=5.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.2, length=2.2)
        ax.grid(axis="y", linewidth=0.3, color=PALETTE["neutral"], alpha=0.6)
        ax.set_axisbelow(True)
        if k == 0:
            ax.set_ylabel("Objective", fontsize=7)
        else:
            ax.tick_params(labelleft=False)

    fig.tight_layout(pad=0.25, w_pad=0.5)
    finalize_figure(fig, OUT / "fig_main_comparison", formats=["pdf"])
    for i, s in enumerate(SIZES):
        print(s, "best", min(m[2][i] for m in METHODS), "ours", METHODS[0][2][i])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()