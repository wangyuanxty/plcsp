"""2x2 encoder ablation heatmap (job x machine)."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import FigureStyle, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent
VALUES = np.array([[559.31, 648.81], [580.21, 676.99]])


def main():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    masked = np.ma.masked_invalid(VALUES)
    im = ax.imshow(masked, cmap="Blues_r", vmin=550, vmax=690)
    for i in range(2):
        for j in range(2):
            v = VALUES[i, j]
            ax.text(j, i, "running" if np.isnan(v) else f"{v:.1f}",
                    ha="center", va="center", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["machine ST", "machine GNN"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["job ST", "job GNN"])
    ax.set_xlabel("Machine encoder")
    ax.set_ylabel("Job encoder")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Objective")
    finalize_figure(fig, OUT / "fig_encoder_grid", formats=["pdf"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()
