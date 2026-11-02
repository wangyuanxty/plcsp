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

    # (a) per-instance objective, one row per variant
    ax = axes[0]
    groups = [np.concatenate(no), np.concatenate(rs)]
    colours = [PALETTE["red_strong"], PALETTE["blue_main"]]
    pos = [1, 2]

    parts = ax.violinplot(groups, positions=pos, orientation="horizontal", widths=0.72,
                          showmeans=False, showmedians=False,
                          showextrema=False, points=60)
    rng = np.random.default_rng(7)
    for body, c, row in zip(parts["bodies"], colours, groups):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.16)
        body.set_linewidth(0.5)
        jitter = rng.uniform(-0.13, 0.13, row.size)
        ax.scatter(row, np.full(row.size, 1) + jitter, s=0.35, color=c,
                   alpha=0.30, linewidths=0, zorder=3)
        ax.scatter(row, np.full(row.size, 2) + jitter, s=0.35, color=c,
                   alpha=0.30, linewidths=0, zorder=3)
    for p, row, c in zip(pos, groups, colours):
        _marks(ax, row, p, c)

    ax.set_yticks(pos)
    ax.set_yticklabels(["no selector", "rule selector"], fontsize=6.4)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.10 * (hi - lo), hi + 0.10 * (hi - lo))
    ax.set_ylim(0.45, 2.55)
    ax.set_xlabel("Objective ($6\\times6$)", fontsize=7)
    ax.set_title("(a) per-instance objective", fontsize=7, pad=3)

    # (b) paired difference per seed
    ax = axes[1]
    diff = [(no[s] - rs[s]) / rs[s] * 100.0 for s in range(3)]
    for i, d in enumerate(diff):
        y = 3 - i
        c = PALETTE["blue_secondary"]
        ax.scatter(d, np.full(d.size, y) + rng.uniform(-0.14, 0.14, d.size),
                   s=0.35, color=c, alpha=0.30, linewidths=0, zorder=3)
        _marks(ax, d, y, c)

    ax.axvline(0, color="black", linewidth=0.7, zorder=1)
    pooled = np.concatenate(diff)
    _marks(ax, pooled, 3.62, PALETTE["blue_main"])
    ax.annotate(f"pooled mean {pooled.mean():+.1f}%",
                xy=(pooled.mean(), 3.62), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6, color=PALETTE["blue_main"])

    ax.set_yticks([3.62, 3, 2, 1])
    ax.set_yticklabels(["pooled", "seed 1", "seed 2", "seed 3"], fontsize=6.4)
    ax.set_ylim(0.45, 3.95)
    lo = min(pooled.min(), -5) - 6
    hi = max(pooled.max(), 5) + 6
    ax.set_xlim(lo, hi)
    ax.set_xlabel("no selector $-$ with (\\% of with)", fontsize=7)
    ax.set_title("(b) paired difference on each seed", fontsize=7, pad=3)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=6.2, length=2.2)
        a.grid(axis="x", linewidth=0.3, color=PALETTE["neutral"], alpha=0.6)
        a.set_axisbelow(True)

    fig.tight_layout(pad=0.25, w_pad=0.9)
    finalize_figure(fig, OUT / "fig_distribution", formats=["pdf"])
    print("means", np.concatenate(rs).mean(), np.concatenate(no).mean())
    print("paired mean %+.2f  range %+.1f .. %+.1f" %
          (pooled.mean(), pooled.min(), pooled.max()))


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.8))
    main()