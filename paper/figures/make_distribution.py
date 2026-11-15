"""Distribution figure: per-instance objective and paired difference.

Two rows, one per instance family. The left column shows the per-instance
objective of the controller with and without the rule selector; the right
column the mean of each training seed, joined per seed.

The training outputs are not part of the repository. When they are absent, the
pooled per-instance values are read from fig_distribution_data.npz, which was
digitised from the vector PDF of the earlier version; its group means agree
with the three-seed means of the rule-selection ablation to 0.01.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import FigureStyle, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DIGITISED = OUT / "fig_distribution_data.npz"

# Vivid pair, matched to the main comparison figure.
RED = "#E63946"    # without the selector
BLUE = "#0D6EFD"   # with the selector

FAMILIES = [
    (r"$6\times6\times2$", "m3_agv6_ours_s{s}", "eval_st_seed3_seed3.npz",
     "m3_noRS_s{s}", "eval_st_seed3_seed3.npz", "six6"),
    (r"$10\times6\times2$", "m3_agv_s{s}_J10M6", "eval_st_seed3_seed3.npz",
     "m3_noRS_J10M6_s{s}", "eval_st_seed3.npz", "ten6"),
]

# Three-seed means of the rule-selection ablation, used when the outputs are
# absent: (without, with).
SEED_MEANS = [
    ([887.75, 896.53, 901.36], [857.93, 880.92, 864.39]),
    ([828.55, 808.92, 820.51], [796.81, 822.29, 806.91]),
]


def load_tags(tags):
    out = []
    for tag, fname in tags:
        path = ROOT / "plcsp_v5/trained_model" / tag / fname
        if not path.exists():
            return None
        out.append(np.load(path)["obj"])
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
        ax.plot([p, p], [q1, q3], color=c, linewidth=2.8,
                solid_capstyle="butt", zorder=4)
        ax.plot([p], [med], marker="o", markersize=4.0, color="white",
                markeredgecolor=c, markeredgewidth=1.0, zorder=5)
        ax.plot([p], [row.mean()], marker="D", markersize=4.8, color=c,
                markeredgecolor="white", markeredgewidth=0.7, zorder=6)


def main() -> None:
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    tags = ["(a)", "(b)", "(c)", "(d)"]
    for row_i, (label, rs_pat, rs_f, no_pat, no_f, key) in enumerate(FAMILIES):
        rs = load_tags([(rs_pat.format(s=s), rs_f) for s in (1, 2, 3)])
        no = load_tags([(no_pat.format(s=s), no_f) for s in (1, 2, 3)])
        if rs is None or no is None:
            digitised = np.load(DIGITISED)
            no = [digitised[f"{key}_without"]]
            rs = [digitised[f"{key}_with"]]
            no_m, rs_m = SEED_MEANS[row_i]
        else:
            no_m = [r.mean() for r in no]
            rs_m = [r.mean() for r in rs]
        colours = [RED, BLUE]
        groups = [np.concatenate(no), np.concatenate(rs)]
        pos = [1, 2]

        ax = axes[row_i, 0]
        _violin(ax, groups, colours, pos, rng)
        ax.set_xticks(pos)
        ax.set_xticklabels(["without selector", "with selector"], fontsize=9.5)
        ax.set_xlim(0.45, 2.55)
        lo = min(g.min() for g in groups)
        hi = max(g.max() for g in groups)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_ylabel("Objective (%s)" % label, fontsize=10)
        ax.set_title(tags[2 * row_i] + " per-instance objective",
                     fontsize=10.5, pad=3)

        ax = axes[row_i, 1]
        pos = np.arange(3)
        for i, (a, b) in enumerate(zip(no_m, rs_m)):
            ax.plot([i, i], [a, b], color="#555555", linewidth=0.9, zorder=2)
            ax.annotate(f"{b - a:+.1f}", xy=(i, (a + b) / 2), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8.8, color=BLUE)
        ax.plot(pos, no_m, "o", color=RED, markersize=5.6,
                label="without selector", zorder=3)
        ax.plot(pos, rs_m, "o", color=BLUE, markersize=5.6,
                label="with selector", zorder=3)
        ax.set_xticks(pos)
        ax.set_xticklabels(["seed 1", "seed 2", "seed 3"], fontsize=9.5)
        ax.set_xlim(-0.45, 2.55)
        span = max(max(rs_m), max(no_m)) - min(min(rs_m), min(no_m))
        mid = 0.5 * (max(max(rs_m), max(no_m)) + min(min(rs_m), min(no_m)))
        ax.set_ylim(mid - 0.75 * max(span, 5.0), mid + 0.75 * max(span, 5.0))
        ax.set_ylabel("Objective (%s)" % label, fontsize=10)
        ax.set_title(tags[2 * row_i + 1] + " mean on each seed",
                     fontsize=10.5, pad=3)
        if row_i == 0:
            ax.legend(loc="upper left", fontsize=9, handlelength=1.0,
                      borderpad=0.2, labelspacing=0.25, ncol=2)

    for a in axes.ravel():
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=9.5, length=2.8, width=1.0)
        a.grid(axis="y", linewidth=0.3, color="#CFCECE", alpha=0.6)
        a.set_axisbelow(True)

    fig.tight_layout(pad=0.25, w_pad=0.9)
    finalize_figure(fig, OUT / "fig_distribution", formats=["pdf"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=10, axes_linewidth=1.0))
    main()