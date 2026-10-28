"""Anytime comparison: objective against wall-clock time on the 6x6 test set."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pubstyle import PALETTE, FigureStyle, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent

RGA_TIME = [0.178, 0.339, 0.533, 0.823, 1.630]
RGA_OBJ = [695.61, 606.04, 550.85, 511.95, 478.04]
OURS = (0.56, 556.70)
PDR = (0.007, 951.20)
MIP = (12.3, 505.05)


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    ax.plot(RGA_TIME, RGA_OBJ, marker="o", markersize=3.6,
            color=PALETTE["blue_main"], linewidth=1.2,
            label="RGA (5 budgets)")
    ax.scatter([PDR[0]], [PDR[1]], marker="s", s=26,
               color=PALETTE["neutral"], edgecolor="black", linewidth=0.6,
               label="PDR", zorder=4)
    ax.scatter([OURS[0]], [OURS[1]], marker="*", s=110,
               color=PALETTE["red_strong"], edgecolor="black", linewidth=0.5,
               label="H-ST-GRPO", zorder=5)
    ax.scatter([MIP[0]], [MIP[1]], marker="D", s=26,
               color=PALETTE["violet"], edgecolor="black", linewidth=0.5,
               label="MIP-AGV", zorder=4)

    ax.axvline(OURS[0], color=PALETTE["red_strong"], linewidth=0.7,
               linestyle=":", zorder=1)

    ax.annotate("0.56 s", (OURS[0], 985), ha="center", va="bottom",
                fontsize=5.5, color=PALETTE["red_strong"])
    ax.annotate("556.7", (OURS[0], OURS[1]), textcoords="offset points",
                xytext=(6, -8), fontsize=5.5)
    ax.annotate("951.2", (PDR[0], PDR[1]), textcoords="offset points",
                xytext=(5, 2), fontsize=5.5)
    ax.annotate("505.1", (MIP[0], MIP[1]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=5.5)
    ax.annotate("478.0", (RGA_TIME[-1], RGA_OBJ[-1]), textcoords="offset points",
                xytext=(6, 6), ha="left", fontsize=5.5)

    ax.set_xscale("log")
    ax.set_xlim(0.004, 20)
    ax.set_ylim(450, 1000)
    ax.set_xlabel("Wall-clock time per instance (s, log scale)")
    ax.set_ylabel("Objective (6$\\times$6)")
    ax.grid(True, which="major", axis="both", linewidth=0.3,
            color=PALETTE["neutral"], alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=5.5, frameon=False,
              handletextpad=0.4, borderaxespad=0.2)

    finalize_figure(fig, OUT / "fig_anytime", formats=["pdf"])
    print("wrote fig_anytime.pdf")


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()
