"""Validation curves: mean and range over three seeds."""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import PALETTE, FigureStyle, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent
SRC = Path("C:/Users/evari/AppData/Local/Temp/figrefs/val_curves.csv")
METHODS = {"local": ("RS", PALETTE["blue_main"]),
           "nors": ("no RS", PALETTE["red_strong"])}


def main():
    rows = list(csv.DictReader(open(SRC)))
    data = {m: {} for m in METHODS}
    for r in rows:
        data[r["method"]].setdefault(int(r["ep"]), []).append(float(r["obj"]))

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for m, (label, color) in METHODS.items():
        eps = sorted(data[m])
        mean = np.array([np.mean(data[m][e]) for e in eps])
        lo = np.array([np.min(data[m][e]) for e in eps])
        hi = np.array([np.max(data[m][e]) for e in eps])
        ax.plot(eps, mean, color=color, lw=1.4, label=label)
        ax.fill_between(eps, lo, hi, color=color, alpha=0.16, linewidth=0)
        i = int(np.argmin(mean))
        ax.scatter([eps[i]], [mean[i]], color=color, s=14, zorder=5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Validation objective")
    ax.set_xlim(0, 205)
    ax.legend(loc="upper right")
    finalize_figure(fig, OUT / "fig_validation", formats=["pdf"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()
