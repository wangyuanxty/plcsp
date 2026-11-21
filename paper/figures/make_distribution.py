"""Distribution figure: per-instance objective and the objective terms.

Two panels, three instance families each. The top panel merges the per-instance
objective of the three families into one violin plot, with the two conditions
side by side in each family. The bottom panel merges the three radars into one,
with one polygon per family; each axis is a term divided by its value without
the selector, so the dashed pentagon at 1 is the baseline and the outside is
better.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import FigureStyle, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]

RED = "#E63946"    # without the selector
BLUE = "#0D6EFD"   # with the selector
FAMILY_COLOURS = ["#9B5DE5", "#00B3B3", "#F4A261"]

FAMILIES = [
    (r"$6\times6\times2$", "m3_agv6_ours_s{s}", "eval_st_seed3_seed3.npz",
     "m3_noRS_s{s}", "eval_st_seed3_seed3.npz"),
    (r"$10\times6\times2$", "m3_agv_s{s}_J10M6", "eval_st_seed3_seed3.npz",
     "m3_noRS_J10M6_s{s}", "eval_st_seed3.npz"),
    (r"$10\times10\times2$", "m3_agv_s{s}_J10M10", "eval_st_seed3_seed3.npz",
     "m3_noRS_J10M10_s{s}", "eval_st_seed3.npz"),
]

RADAR_AXES = ["Cmax", "PEC", "TEC", "Idle", "Obj."]


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


def _violin(ax, rows, colours, positions, rng):
    for row, c, p in zip(rows, colours, positions):
        parts = ax.violinplot([row], positions=[p], widths=0.30,
                              showmeans=False, showmedians=False,
                              showextrema=False, points=60)
        body = parts["bodies"][0]
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.16)
        body.set_linewidth(0.5)
        ax.scatter(p + rng.uniform(-0.05, 0.05, row.size), row, s=0.25,
                   color=c, alpha=0.25, linewidths=0, zorder=3)
        q1, med, q3 = np.percentile(row, [25, 50, 75])
        ax.plot([p, p], [q1, q3], color=c, linewidth=2.2,
                solid_capstyle="butt", zorder=4)
        ax.plot([p], [med], marker="o", markersize=3.2, color="white",
                markeredgecolor=c, markeredgewidth=0.9, zorder=5)
        ax.plot([p], [row.mean()], marker="D", markersize=4.0, color=c,
                markeredgecolor="white", markeredgewidth=0.6, zorder=6)


def _radar(ax, ratio):
    angles = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False)
    angles = angles.tolist()
    angles += angles[:1]
    base = [1.0] * (len(RADAR_AXES) + 1)
    r = ratio + ratio[:1]
    ax.plot(angles, base, color=RED, linewidth=1.5,
            linestyle=(0, (3, 2)), zorder=2)
    ax.plot(angles, r, color=BLUE, linewidth=1.6, zorder=3)
    ax.fill(angles, r, color=BLUE, alpha=0.14, zorder=1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_AXES, fontsize=8)
    for tick in ax.xaxis.get_major_ticks():
        if tick.label1.get_text() == "PEC":
            tick.set_pad(-2)
    ax.set_ylim(0.94, 1.08)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.set_yticklabels(["0.95", "1", "1.05"], fontsize=7.5)
    for lab in ax.get_yticklabels():
        lab.set_bbox({"facecolor": "white", "edgecolor": "none",
                      "alpha": 1.0, "pad": 0.5})
    ax.set_rlabel_position(270)
    ax.grid(linewidth=0.3, color="#CFCECE", alpha=0.6)
    ax.spines["polar"].set_linewidth(0.5)


def main() -> None:
    rng = np.random.default_rng(7)
    groups_all, ratios = [], []
    for _, rs_pat, rs_f, no_pat, no_f in FAMILIES:
        rs = load_runs([(rs_pat.format(s=s), rs_f) for s in (1, 2, 3)])
        no = load_runs([(no_pat.format(s=s), no_f) for s in (1, 2, 3)])
        if rs is None or no is None:
            raise SystemExit("evaluation outputs are not available")
        groups_all.append(np.concatenate([r["obj"] for r in no]))
        groups_all.append(np.concatenate([r["obj"] for r in rs]))
        ratios.append([n / w for n, w in zip(terms(no), terms(rs))])

    fig = plt.figure(figsize=(7.0, 4.8))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.62, top=0.90)

    for col, (label, _, _, _, _) in enumerate(FAMILIES):
        groups = [groups_all[2 * col], groups_all[2 * col + 1]]
        ax = fig.add_subplot(gs[0, col])
        _violin(ax, groups, [RED, BLUE], [1, 2], rng)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["without", "with"], fontsize=8.5)
        ax.set_xlim(0.45, 2.55)
        lo = min(g.min() for g in groups)
        hi = max(g.max() for g in groups)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_ylabel("Objective", fontsize=7.5)
        ax.set_title(label, fontsize=8, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, length=2.8, width=1.0)
        ax.grid(axis="y", linewidth=0.3, color="#CFCECE", alpha=0.6)
        ax.set_axisbelow(True)

        axr = fig.add_subplot(gs[1, col], projection="polar")
        _radar(axr, ratios[col])
        axr.set_title(label, fontsize=8, pad=8)

    fig.text(0.005, 0.965, "(a) per-instance objective", fontsize=8,
             ha="left", va="bottom")
    fig.text(0.005, 0.53, "(b) objective terms, divided by the value "
             "without the selector", fontsize=8, ha="left", va="bottom")
    cond = [plt.Line2D([], [], color=RED, linewidth=2.0),
            plt.Line2D([], [], color=BLUE, linewidth=2.0)]
    fig.legend(cond, ["without selector", "with selector"],
               loc="upper right", ncol=2, fontsize=7.5, frameon=False,
               bbox_to_anchor=(1.0, 0.995), handlelength=1.4,
               columnspacing=1.4)
    finalize_figure(fig, OUT / "fig_distribution", formats=["pdf"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=9, axes_linewidth=1.0))
    main()