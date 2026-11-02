"""Distribution figure: per-instance objective and paired difference.

Both panels are horizontal so that the two variants and the three seeds read
as rows. Panel (a) shows the per-instance objective of the controller with and
without the rule selector; panel (b) the paired difference on each training
seed, as a percentage of the controller value.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import (PALETTE, FigureStyle, apply_publication_style,
                      finalize_figure)

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]


def load_tags(tags):
    return [np.load(ROOT / "plcsp_v5/trained_model" / t
                    / "eval_st_seed3_seed3.npz")["obj"] for t in tags]


def _marks(ax, data, pos, colour):
    """IQR bar, median dot and mean diamond for one row."""
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    ax.plot([q1, q3], [pos, pos], color=colour, linewidth=2.6,
            solid_capstyle="butt", zorder=4)
    ax.plot([med], [pos], marker="o", markersize=3.0, color="white",
            markeredgecolor=colour, markeredgewidth=0.8, zorder=5)
    ax.plot([data.mean()], [pos], marker="D", markersize=3.6, color=colour,
            markeredgecolor="white", markeredgewidth=0.6, zorder=6)


def main() -> None:
    rs = load_tags([f"m3_agv6_ours_s{s}" for s in (1, 2, 3)])
    no = load_tags([f"m3_noRS_s{s}" for s in (1, 2, 3)])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    # (a) per-instance objective, one violin per variant
    ax = axes[0]
    groups = [np.concatenate(no), np.concatenate(rs)]
    colours = [PALETTE["red_strong"], PALETTE["blue_main"]]
    pos = [1, 2]

    parts = ax.violinplot(groups, positions=pos, widths=0.66,
                          showmeans=False, showmedians=False,
                          showextrema=False, points=60)
    rng = np.random.default_rng(7)
    for body, c, row, p in zip(parts["bodies"], colours, groups, pos):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.16)
        body.set_linewidth(0.5)
        ax.scatter(p + rng.uniform(-0.13, 0.13, row.size), row, s=0.35,
                   color=c, alpha=0.30, linewidths=0, zorder=3)
        q1, med, q3 = np.percentile(row, [25, 50, 75])
        ax.plot([p, p], [q1, q3], color=c, linewidth=2.6,
                solid_capstyle="butt", zorder=4)
        ax.plot([p], [med], marker="o", markersize=3.0, color="white",
                markeredgecolor=c, markeredgewidth=0.8, zorder=5)
        ax.plot([p], [row.mean()], marker="D", markersize=3.6, color=c,
                markeredgecolor="white", markeredgewidth=0.6, zorder=6)

    ax.set_xticks(pos)
    ax.set_xticklabels(["without", "with"], fontsize=6.4)
    ax.set_xlim(0.45, 2.55)
    lo = min(g.min() for g in groups)
    hi = max(g.max() for g in groups)
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_ylabel("Objective ($6\times6$)", fontsize=7)
    ax.set_title("(a) per-instance objective", fontsize=7, pad=3)

    # (b) per-seed means of the two variants, joined per seed
    ax = axes[1]
    rs_m = [r.mean() for r in rs]
    no_m = [r.mean() for r in no]
    pos = np.arange(3)

    for i, (a, b) in enumerate(zip(no_m, rs_m)):
        ax.plot([i, i], [a, b], color="#555555", linewidth=0.8, zorder=2)
        ax.annotate(f"{b - a:+.1f}", xy=(i, (a + b) / 2), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=5.8, color=PALETTE["blue_main"])
    ax.plot(pos, no_m, "o", color=PALETTE["red_strong"], markersize=4.2,
            label="without", zorder=3)
    ax.plot(pos, rs_m, "o", color=PALETTE["blue_main"], markersize=4.2,
            label="with", zorder=3)

    ax.set_xticks(pos)
    ax.set_xticklabels(["seed 1", "seed 2", "seed 3"], fontsize=6.4)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(840, 915)
    ax.set_ylabel("Objective ($6\times6$)", fontsize=7)
    ax.set_title("(b) mean on each seed", fontsize=7, pad=3)
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.0,
              borderpad=0.2, labelspacing=0.25, ncol=2)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=6.2, length=2.2)
        a.grid(axis="y", linewidth=0.3, color=PALETTE["neutral"], alpha=0.6)
        a.set_axisbelow(True)

    fig.tight_layout(pad=0.25, w_pad=0.9)
    finalize_figure(fig, OUT / "fig_distribution", formats=["pdf"])
    print("means", np.concatenate(rs).mean(), np.concatenate(no).mean())
    print("per-seed", [round(r.mean(), 1) for r in rs],
          [round(r.mean(), 1) for r in no])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.8))
    main()