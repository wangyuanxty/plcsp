"""Distribution figure: per-instance objective and paired difference.

Two rows, one per instance family. The left column shows the per-instance
objective of the controller with and without the rule selector; the right
column the mean of each training seed, joined per seed.
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

FAMILIES = [
    (r"$6\times6\times2$", "m3_agv6_ours_s{s}", "eval_st_seed3_seed3.npz",
     "m3_noRS_s{s}", "eval_st_seed3_seed3.npz"),
    (r"$10\times6\times2$", "m3_agv_s{s}_J10M6", "eval_st_seed3_seed3.npz",
     "m3_noRS_J10M6_s{s}", "eval_st_seed3.npz"),
]


def load_tags(tags):
    out = []
    for tag, fname in tags:
        out.append(np.load(ROOT / "plcsp_v5/trained_model" / tag / fname)["obj"])
    return out


def _violin(ax, groups, colours, pos, rng):
    parts = ax.violinplot(groups, positions=pos, widths=0.66,
                          showmeans=False, showmedians=False,
                          showextrema=False, points=60)
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


def main() -> None:
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.7),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    tags = ["(a)", "(b)", "(c)", "(d)"]
    for row_i, (label, rs_pat, rs_f, no_pat, no_f) in enumerate(FAMILIES):
        rs = load_tags([(rs_pat.format(s=s), rs_f) for s in (1, 2, 3)])
        no = load_tags([(no_pat.format(s=s), no_f) for s in (1, 2, 3)])
        colours = [PALETTE["red_strong"], PALETTE["blue_main"]]
        groups = [np.concatenate(no), np.concatenate(rs)]
        pos = [1, 2]

        ax = axes[row_i, 0]
        _violin(ax, groups, colours, pos, rng)
        ax.set_xticks(pos)
        ax.set_xticklabels(["without", "with"], fontsize=6.4)
        ax.set_xlim(0.45, 2.55)
        lo = min(g.min() for g in groups)
        hi = max(g.max() for g in groups)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_ylabel("Objective (%s)" % label, fontsize=7)
        ax.set_title(tags[2 * row_i] + " per-instance objective",
                     fontsize=7, pad=3)

        ax = axes[row_i, 1]
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
        span = max(max(rs_m), max(no_m)) - min(min(rs_m), min(no_m))
        mid = 0.5 * (max(max(rs_m), max(no_m)) + min(min(rs_m), min(no_m)))
        ax.set_ylim(mid - 0.75 * max(span, 5.0), mid + 0.75 * max(span, 5.0))
        ax.set_ylabel("Objective (%s)" % label, fontsize=7)
        ax.set_title(tags[2 * row_i + 1] + " mean on each seed",
                     fontsize=7, pad=3)
        if row_i == 0:
            ax.legend(loc="upper left", fontsize=5.8, handlelength=1.0,
                      borderpad=0.2, labelspacing=0.25, ncol=2)

    for a in axes.ravel():
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=6.2, length=2.2)
        a.grid(axis="y", linewidth=0.3, color=PALETTE["neutral"], alpha=0.6)
        a.set_axisbelow(True)

    fig.tight_layout(pad=0.25, w_pad=0.9)
    finalize_figure(fig, OUT / "fig_distribution", formats=["pdf"])
    for label, rs_pat, rs_f, no_pat, no_f in FAMILIES:
        rs = load_tags([(rs_pat.format(s=s), rs_f) for s in (1, 2, 3)])
        no = load_tags([(no_pat.format(s=s), no_f) for s in (1, 2, 3)])
        print(label, "with", round(np.concatenate(rs).mean(), 2),
              "without", round(np.concatenate(no).mean(), 2))


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.8))
    main()