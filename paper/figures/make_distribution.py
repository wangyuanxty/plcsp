"""Distribution figure: per-instance objective and paired difference.

Panel (a) shows the per-instance objective of the controller, of the controller
without the rule selector, and of the MIP baseline on the 6x6 test set.
Panel (b) shows the per-instance paired difference, as a percentage of the
controller value, for each training seed; positive means the selector helps.
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
import csv
import glob


def load_tags(tags):
    return [np.load(ROOT / "plcsp_v5/trained_model" / t
                    / "eval_st_seed3_seed3.npz")["obj"] for t in tags]


def load_mip():
    vals = []
    for p in sorted(glob.glob(str(ROOT / "plcsp_v5/results"
                                / "mip_agv_m2_trailing_6x6/shard*.csv"))):
        for r in csv.DictReader(open(p)):
            if r.get("best_obj"):
                vals.append(float(r["best_obj"]))
    return np.array(vals)


def _strip(ax, data, pos, colour, horizontal=False):
    rng = np.random.default_rng(7)
    for i, d in enumerate(data):
        if horizontal:
            jitter = rng.uniform(-0.10, 0.10, d.size)
            ax.scatter(d, np.full(d.size, pos + 0.0) + jitter, s=0.5,
                       color=colour, alpha=0.22, linewidths=0, zorder=2)
        else:
            jitter = rng.uniform(-0.10, 0.10, d.size)
            ax.scatter(pos + jitter, d, s=0.5, color=colour, alpha=0.22,
                       linewidths=0, zorder=2)


def main() -> None:
    rs = load_tags([f"m3_agv6_ours_s{s}" for s in (1, 2, 3)])
    no = load_tags([f"m3_noRS_s{s}" for s in (1, 2, 3)])
    mip = load_mip()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]

    groups = [np.concatenate(rs), np.concatenate(no), mip]
    colours = [PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["violet"]]
    labels = ["RS\n(3 seeds)", "no RS\n(3 seeds)", "MIP\n(100 inst.)"]
    pos = [1, 2, 3]

    parts = ax.violinplot(groups, positions=pos, widths=0.62,
                          showmeans=False, showmedians=False,
                          showextrema=False, points=60)
    for body, c in zip(parts["bodies"], colours):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.18)
        body.set_linewidth(0.6)

    for p, d, c in zip(pos, groups, colours):
        _strip(ax, [d], p, c)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([p, p], [q1, q3], color=c, linewidth=3.2,
                solid_capstyle="butt", zorder=4)
        ax.plot([p], [med], marker="o", markersize=3.4, color="white",
                markeredgecolor=c, markeredgewidth=0.9, zorder=5)
        ax.plot([p], [d.mean()], marker="D", markersize=4.2, color="white",
                markeredgecolor=c, markeredgewidth=1.0, zorder=6)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.45, 3.55)
    lo = min(g.min() for g in groups)
    hi = max(g.max() for g in groups)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("Objective (6$\\times$6)")
    ax.set_title("(a) per-instance objective", fontsize=7.5, pad=4)

    ax = axes[1]
    diff = [(no[s] - rs[s]) / rs[s] * 100.0 for s in range(3)]
    for i, d in enumerate(diff):
        y = 3 - i
        c = PALETTE["blue_secondary"]
        _strip(ax, [d], y, c, horizontal=True)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([q1, q3], [y, y], color=c, linewidth=3.2,
                solid_capstyle="butt", zorder=4)
        ax.plot([med], [y], marker="o", markersize=3.4, color="white",
                markeredgecolor=c, markeredgewidth=0.9, zorder=5)
        ax.plot([d.mean()], [y], marker="D", markersize=4.2, color="white",
                markeredgecolor=c, markeredgewidth=1.0, zorder=6)

    ax.axvline(0, color="black", linewidth=0.7, zorder=1)
    ax.set_yticks([3, 2, 1])
    ax.set_yticklabels(["seed 1", "seed 2", "seed 3"])
    ax.set_ylim(0.45, 3.55)
    pooled = np.concatenate(diff)
    lo = min(pooled.min(), -5) - 3
    hi = max(pooled.max(), 5) + 3
    ax.set_xlim(lo, hi)
    ax.set_xlabel("no RS $-$ RS (\\% of RS); positive = selector helps")
    ax.set_title("(b) paired difference on each seed", fontsize=7.5, pad=4)
    ax.annotate(f"mean {pooled.mean():+.1f}%", xy=(pooled.mean(), 3.34),
                ha="center", va="bottom", fontsize=6,
                color=PALETTE["blue_main"])

    fig.tight_layout(pad=0.3)
    finalize_figure(fig, OUT / "fig_distribution", formats=["pdf"])
    print("means", np.concatenate(rs).mean(), np.concatenate(no).mean(),
          mip.mean())
    print("paired mean %+.2f  range %+.1f .. %+.1f" %
          (pooled.mean(), pooled.min(), pooled.max()))


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()