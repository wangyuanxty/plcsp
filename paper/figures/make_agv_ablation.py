"""AGV ablation: per-instance change when the fixed rule replaces the agent.

Each family keeps the same checkpoints. Only the AGV head changes: the learned
agent, or the earliest-arrival rule, which is the best of the three fixed rules
in Table 4. The two conditions are paired on the same test instance, so a
positive value means that the fixed rule gives the larger objective.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import FigureStyle, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "plcsp_v5/trained_model"

COLORS = ["#0D6EFD", "#00B3B3", "#2ECC71", "#E63946", "#9B5DE5"]
GREY = "#555555"

FAMILIES = [
    (r"$6\times6\times2$", "m3_agv6_ours_s{s}",
     "eval_st_seed3_seed3.npz", "eval_st_seed3_seed3rule.npz"),
    (r"$10\times6\times2$", "m3_agv_s{s}_J10M6",
     "eval_st_seed3_seed3.npz", "eval_st_seed3_seed3rule.npz"),
    (r"$10\times10\times2$", "m3_agv_s{s}_J10M10",
     "eval_st_seed3_seed3.npz", "eval_st_seed3_agvfix.npz"),
    (r"$15\times10\times2$", "plan15_J15M10E2_s{s}_local",
     "eval_st_seed3_15x10.npz", "eval_st_seed3_agvfix.npz"),
    (r"$20\times6\times3$", "plan20_J20M6E3_s{s}_local",
     "eval_st_seed3_20x6.npz", "eval_st_seed3_agvfix.npz"),
]


def paired_change(pat: str, learned_f: str, fixed_f: str) -> np.ndarray:
    """Percent change of the objective, paired on the same instance."""
    vals = []
    for s in (1, 2, 3):
        d = TR / pat.format(s=s)
        a = np.load(d / learned_f)
        b = np.load(d / fixed_f)
        la = dict(zip(a["idx"].tolist(), a["obj"].tolist()))
        fb = dict(zip(b["idx"].tolist(), b["obj"].tolist()))
        for i in sorted(set(la) & set(fb)):
            vals.append((fb[i] / la[i] - 1.0) * 100.0)
    return np.asarray(vals)


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    rows = [paired_change(pat, lf, ff) for _, pat, lf, ff in FAMILIES]
    ypos = np.arange(len(rows))[::-1]
    bp = ax.boxplot(rows, orientation="horizontal", positions=ypos, widths=0.52,
                    patch_artist=True, whis=1.5,
                    medianprops={"color": "#333333", "linewidth": 1.1},
                    whiskerprops={"color": GREY, "linewidth": 0.8},
                    capprops={"color": GREY, "linewidth": 0.8},
                    flierprops={"marker": "o", "markersize": 1.6,
                                "markerfacecolor": GREY,
                                "markeredgecolor": "none", "alpha": 0.5})
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    means = [row.mean() for row in rows]
    ax.plot(means, ypos, "D", markersize=3.6, color="white",
            markeredgecolor=GREY, markeredgewidth=0.6, zorder=5,
            linestyle="none")
    ax.axvline(0.0, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)),
               zorder=1)

    ax.set_yticks(ypos)
    ax.set_yticklabels([fam[0] for fam in FAMILIES], fontsize=7.4)
    ax.set_xlim(-32, 66)
    ax.set_xticks(np.arange(-20, 61, 20))
    ax.set_xlabel("Objective change per instance (%)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.3, color="#CFCECE", alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.2, length=2.6, width=0.9)

    fig.tight_layout(pad=0.25)
    finalize_figure(fig, OUT / "fig_agv_ablation", formats=["pdf"])
    for label, row in zip([f[0] for f in FAMILIES], rows):
        q = np.percentile(row, [25, 50, 75])
        print(label, f"q1={q[0]:+.1f} med={q[1]:+.1f} q3={q[2]:+.1f} "
              f"mean={row.mean():+.2f}")


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=10, axes_linewidth=1.0))
    main()