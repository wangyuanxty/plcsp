"""Distribution figure: per-instance objective and the objective terms.

Two rows, one per instance family. The left column shows the per-instance
objective of the controller with and without the rule selector. The right
column is a radar of the five objective terms, each divided by its value
without the selector, so the dashed pentagon at 1 is the baseline and the
outside is better.

The training outputs are not part of the repository. When they are absent, the
pooled per-instance values are read from fig_distribution_data.npz, which was
digitised from the vector PDF of the earlier version; its group means agree
with the three-seed means of the rule-selection ablation to 0.01. The radar
then falls back to the constants in RADAR_FALLBACK.
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

RADAR_AXES = ["Cmax", "PEC", "TEC", "Idle", "Obj."]

# (without, with) means of RADAR_AXES, used with SEED_MEANS.
RADAR_FALLBACK = {
    "six6": ((490.39, 518.11, 186.26, 1136.40, 895.21),
             (478.85, 508.75, 174.95, 1094.29, 867.75)),
    "ten6": ((618.73, 518.44, 392.91, 714.69, 819.33),
             (612.41, 519.43, 411.12, 684.28, 808.67)),
}


def load_runs(tags):
    out = []
    for tag, fname in tags:
        path = ROOT / "plcsp_v5/trained_model" / tag / fname
        if not path.exists():
            return None
        z = np.load(path)
        out.append({"obj": np.asarray(z["obj"], float),
                    "cost4": np.asarray(z["cost4"], float)})
    return out


def terms(runs):
    """Mean of Cmax, PEC, TEC, Idle and the objective over three seeds."""
    c4 = np.mean([r["cost4"].mean(axis=0) for r in runs], axis=0)
    obj = float(np.mean([r["obj"].mean() for r in runs]))
    return [c4[0], c4[1], c4[2], c4[3], obj]


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


def _radar(ax, no_terms, rs_terms, title):
    angles = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False)
    angles = angles.tolist()
    angles += angles[:1]
    base = [1.0] * (len(RADAR_AXES) + 1)
    ratio = [no / ours for no, ours in zip(no_terms, rs_terms)]
    ratio += ratio[:1]
    ax.plot(angles, base, color=RED, linewidth=1.5,
            linestyle=(0, (3, 2)), zorder=2)
    ax.plot(angles, ratio, color=BLUE, linewidth=1.6, zorder=3)
    ax.fill(angles, ratio, color=BLUE, alpha=0.14, zorder=1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_AXES, fontsize=8.5)
    ax.set_ylim(0.94, 1.08)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.set_yticklabels(["0.95", "1", "1.05"], fontsize=8.5)
    for lab in ax.get_yticklabels():
        lab.set_bbox({"facecolor": "white", "edgecolor": "none",
                      "alpha": 1.0, "pad": 0.5})
    ax.set_rlabel_position(270)
    ax.set_title(title, fontsize=9.5, pad=10)
    ax.grid(linewidth=0.3, color="#CFCECE", alpha=0.6)
    ax.spines["polar"].set_linewidth(0.5)


def main() -> None:
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(7.0, 4.45))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], hspace=0.62,
                          wspace=0.30, top=0.86)
    tags = ["(a)", "(b)", "(c)", "(d)"]
    for row_i, (label, rs_pat, rs_f, no_pat, no_f, key) in enumerate(FAMILIES):
        rs = load_runs([(rs_pat.format(s=s), rs_f) for s in (1, 2, 3)])
        no = load_runs([(no_pat.format(s=s), no_f) for s in (1, 2, 3)])
        if rs is None or no is None:
            digitised = np.load(DIGITISED)
            no_obj = [digitised[f"{key}_without"]]
            rs_obj = [digitised[f"{key}_with"]]
            no_m, rs_m = SEED_MEANS[row_i]
            radar_no, radar_rs = RADAR_FALLBACK[key]
        else:
            no_obj = [r["obj"] for r in no]
            rs_obj = [r["obj"] for r in rs]
            no_m = [r["obj"].mean() for r in no]
            rs_m = [r["obj"].mean() for r in rs]
            radar_no = terms(no)
            radar_rs = terms(rs)

        colours = [RED, BLUE]
        groups = [np.concatenate(no_obj), np.concatenate(rs_obj)]
        pos = [1, 2]

        ax = fig.add_subplot(gs[row_i, 0])
        _violin(ax, groups, colours, pos, rng)
        ax.set_xticks(pos)
        ax.set_xticklabels(["without selector", "with selector"],
                           fontsize=8)
        ax.set_xlim(0.45, 2.55)
        lo = min(g.min() for g in groups)
        hi = max(g.max() for g in groups)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_ylabel("Objective (%s)" % label, fontsize=8.5)
        ax.set_title(tags[2 * row_i] + " per-instance objective",
                     fontsize=9, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, length=2.8, width=1.0)
        ax.grid(axis="y", linewidth=0.3, color="#CFCECE", alpha=0.6)
        ax.set_axisbelow(True)

        axr = fig.add_subplot(gs[row_i, 1], projection="polar")
        _radar(axr, radar_no, radar_rs, "")
        box = axr.get_position()
        fig.text(box.x0 + 0.5 * box.width, box.y1 + 0.062,
                 tags[2 * row_i + 1] + " objective terms",
                 ha="center", va="bottom", fontsize=9.5)

    handles = [
        plt.Line2D([], [], color=RED, linewidth=1.0,
                   linestyle=(0, (3, 2))),
        plt.Line2D([], [], color=BLUE, linewidth=1.2),
    ]
    fig.legend(handles, ["without selector (baseline)", "with selector"],
               loc="upper left", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.01, 0.995), handlelength=1.6,
               columnspacing=2.0)
    finalize_figure(fig, OUT / "fig_distribution", formats=["pdf"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=9, axes_linewidth=1.0))
    main()