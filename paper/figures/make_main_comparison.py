"""Main comparison: objective of every method on the three instance families.

Grouped bars, one group per instance family, one bar per method. The learned
methods carry the seed standard deviation as an error bar; the deterministic
baselines have none.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import (PALETTE, FigureStyle, apply_publication_style,
                      finalize_figure)

OUT = Path(__file__).resolve().parent

SIZES = ["$6\\times6\\times2$", "$10\\times6\\times2$", "$10\\times10\\times2$"]

METHODS = [
    ("Ours", PALETTE["blue_main"], [867.75, 808.67, 1836.64],
     [11.86, 12.83, 43.45]),
    ("RGA", PALETTE["teal"], [762.04, 796.70, 2298.63], [0, 0, 0]),
    ("E2E", PALETTE["red_strong"], [1075.64, 1066.24, 3153.81],
     [73.82, 102.61, 530.01]),
    ("Li et al.", PALETTE["violet"], [1160.59, 1198.09, 2738.91],
     [42.40, 76.91, 82.76]),
    ("PDR", PALETTE["neutral"], [1319.24, 1461.63, 3288.35], [0, 0, 0]),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    n = len(METHODS)
    width = 0.15
    x = np.arange(len(SIZES))

    for k, (name, colour, vals, errs) in enumerate(METHODS):
        off = (k - (n - 1) / 2) * width
        ax.bar(x + off, vals, width * 0.92, label=name, color=colour,
               edgecolor="white", linewidth=0.5, zorder=2)
        if any(errs):
            ax.errorbar(x + off, vals, yerr=errs, fmt="none",
                        ecolor="#333333", elinewidth=0.7, capsize=1.6,
                        capthick=0.7, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(SIZES)
    ax.set_ylabel("Objective")
    ax.set_ylim(0, 3700)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, fontsize=6, handlelength=0.9,
              columnspacing=1.2, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, color=PALETTE["neutral"], alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(top=0.80)
    finalize_figure(fig, OUT / "fig_main_comparison", formats=["pdf"])
    for i, s in enumerate(SIZES):
        best = min(m[2][i] for m in METHODS)
        print(s, "best", best, "ours", METHODS[0][2][i])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()