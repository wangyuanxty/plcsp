"""Distribution figure: per-instance objective (violin + box) and per-seed
paired difference, following the AEI reference style."""

from pathlib import Path
import csv, glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
TMP = Path("C:/Users/evari/AppData/Local/Temp/figrefs")
ROOT = Path(__file__).resolve().parents[2]

from pubstyle import (PALETTE, FigureStyle,
                       apply_publication_style, finalize_figure)



def load_npz(pattern):
    return [np.load(p)["obj"] for p in sorted(TMP.glob(pattern))]


def load_mip():
    vals = []
    for p in sorted(glob.glob(str(ROOT / "plcsp_v5/results/mip_agv_100/shard*.csv"))):
        for r in csv.DictReader(open(p)):
            if r.get("best_obj"):
                vals.append(float(r["best_obj"]))
    return np.array(vals)


def violin(ax, data, positions, colors, labels):
    parts = ax.violinplot(data, positions=positions, widths=0.72,
                          showmeans=False, showmedians=False,
                          showextrema=False)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
        pc.set_edgecolor(c)
        pc.set_linewidth(0.8)
    bp = ax.boxplot(data, positions=positions, widths=0.16,
                    patch_artist=True, showfliers=False, whis=1.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(c)
        patch.set_linewidth(0.9)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(0.9)
    for i, (d, c) in enumerate(zip(data, colors)):
        ax.scatter(np.full(len(d), positions[i]) + np.random.uniform(-0.13, 0.13, len(d)),
                   d, s=1.2, color=c, alpha=0.5, linewidths=0)
        ax.scatter([positions[i]], [np.mean(d)], marker="D", s=14,
                   color="white", edgecolor=c, zorder=5, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def main():
    rs = load_npz("rs_s*.npz")
    no = load_npz("nors_s*.npz")
    mip = load_mip()
    rs_all = np.concatenate(rs)
    no_all = np.concatenate(no)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6),
                             gridspec_kw={"width_ratios": [1, 1]})

    ax = axes[0]
    violin(ax, [rs_all, no_all, mip], [1, 2, 3],
           [PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["teal"]],
           ["RS\n(3 seeds)", "no RS\n(3 seeds)", "MIP\n(100 inst.)"])
    ax.set_ylabel("Objective (6x6)")
    ax.set_ylim(350, 850)
    for pos, d in zip([1, 2, 3], [rs_all, no_all, mip]):
        ax.text(pos, np.mean(d) + 18, f"{np.mean(d):.1f}",
                ha="center", fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(a) Per-instance objective", loc="left", fontsize=7)

    ax = axes[1]
    diffs = [r - n for r, n in zip(rs, no)]
    bp = ax.boxplot(diffs, positions=[1, 2, 3], widths=0.45,
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor(PALETTE["blue_main"])
    for med in bp["medians"]:
        med.set_color("black")
    for i, d in enumerate(diffs):
        ax.scatter(np.full(len(d), i + 1) + np.random.uniform(-0.12, 0.12, len(d)),
                   d, s=1.4, color=PALETTE["blue_main"], alpha=0.5,
                   linewidths=0)
        ax.text(i + 1, np.max(d) + 3, f"{np.mean(d):+.1f}", ha="center",
                fontsize=6)
    ax.axhline(0, color=PALETTE["neutral"], lw=0.8, ls="--")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["s1", "s2", "s3"])
    ax.set_ylabel("RS - no RS")
    ax.set_title("(b) Paired difference", loc="left", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    finalize_figure(fig, OUT / "fig_distribution", formats=["pdf"])
    print("wrote fig_distribution.pdf")
    print("means", rs_all.mean(), no_all.mean(), mip.mean())


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()
