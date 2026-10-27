"""Four separate ablation figures, each with the chart type that fits the
claim. House palette from scientific-figure-making."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

from pubstyle import (PALETTE, FigureStyle,
                       apply_publication_style, finalize_figure)



def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def encoder():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    labels = ["ST\n+ intra", "ST\nno intra", "job ST /\nmach GAT", "all GNN"]
    vals = [559.31, 624.63, 648.81, 676.99]
    cols = [PALETTE["blue_main"], PALETTE["blue_secondary"],
            PALETTE["teal"], PALETTE["red_strong"]]
    ax.bar(range(4), vals, 0.62, color=cols)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Objective (6x6)")
    ax.set_ylim(0, 760)
    for i, v in enumerate(vals):
        ax.text(i, v + 8, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    clean(ax)
    finalize_figure(fig, OUT / "fig_encoder", formats=["pdf"])
    print("wrote fig_encoder.pdf")


def algorithm():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [559.31, 575.27], w, label="GRPO",
           color=PALETTE["blue_main"])
    ax.bar(x + w / 2, [637.41, 605.14], w, label="PPO",
           color=PALETTE["red_strong"])
    ax.set_xticks(x)
    ax.set_xticklabels(["6x6", "10x6"])
    ax.set_ylabel("Objective")
    ax.set_ylim(0, 720)
    ax.legend(loc="upper right")
    for i, pair in enumerate([(559.31, 637.41), (575.27, 605.14)]):
        for off, v in zip((-w / 2, w / 2), pair):
            ax.text(i + off, v + 7, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6)
    clean(ax)
    finalize_figure(fig, OUT / "fig_algo", formats=["pdf"])
    print("wrote fig_algo.pdf")


def agv_dumbbell():
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    rows = ["6x6", "10x6"]
    rule = [573.57, 577.83]
    head = [564.05, 577.71]
    y = np.arange(len(rows))
    for i in range(len(rows)):
        ax.plot([rule[i], head[i]], [y[i], y[i]], color=PALETTE["neutral"],
                lw=1.6, zorder=1)
    ax.scatter(rule, y, s=34, color=PALETTE["red_strong"],
               label="earliest-free rule", zorder=2)
    ax.scatter(head, y, s=34, color=PALETTE["blue_main"],
               label="learned AGV head", zorder=2)
    for i in range(len(rows)):
        d = head[i] - rule[i]
        ax.text(max(rule[i], head[i]) + 0.5, y[i] + 0.13, f"{d:+.2f}",
                fontsize=6, color=PALETTE["blue_main"] if d < 0 else "black")
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.set_xlabel("Objective")
    ax.set_xlim(560, 581)
    ax.legend(loc="lower left")
    clean(ax)
    finalize_figure(fig, OUT / "fig_agv", formats=["pdf"])
    print("wrote fig_agv.pdf")


def rs_slope():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    rs = [570.84, 555.49, 543.76]
    no = [558.11, 558.35, 575.69]
    x = [0, 1]
    for i, (a, b) in enumerate(zip(no, rs)):
        color = PALETTE["blue_main"] if b < a else PALETTE["red_strong"]
        ax.plot(x, [a, b], marker="o", ms=3.2, lw=1.1, color=color,
                alpha=0.85, label=f"s{i+1} ({b-a:+.1f})")
    ax.plot(x, [np.mean(no), np.mean(rs)], marker="s", ms=4, lw=2.0,
            color="black", label=f"mean ({np.mean(rs)-np.mean(no):+.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels(["no RS", "RS"])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylabel("Objective (6x6)")
    ax.legend(loc="upper left")
    clean(ax)
    finalize_figure(fig, OUT / "fig_rs", formats=["pdf"])
    print("wrote fig_rs.pdf")


if __name__ == "__main__":
    encoder()
    algorithm()
    agv_dumbbell()
    rs_slope()
