"""Anytime comparison: objective against wall-clock time on the 6x6 test set.

RGA points come from one run per budget (pop 50, gen in 5/10/20/50/100) with
two worker processes. The controller, PDR and MIP are single points taken from
Tables 4 and 6.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pubstyle import PALETTE, FigureStyle, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent
TMP = Path("C:/Users/evari/AppData/Local/Temp")

OURS = (0.95, 867.75)
PDR = (0.0004, 1319.24)
MIP = (11.0, 881.43)


def load_rga():
    times, objs = [], []
    for gen in (5, 10, 20, 50, 100):
        path = TMP / f"rga_anytime_m2_gen{gen}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        times.append(d["summary"]["wall_s"] / d["n_runs"])
        objs.append(d["summary"]["overall"]["mean"])
    return times, objs


def main() -> None:
    rga_time, rga_obj = load_rga()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    ax.plot(rga_time, rga_obj, marker="o", markersize=3.6,
            color=PALETTE["blue_main"], linewidth=1.2,
            label="RGA (5 budgets)")
    ax.scatter([PDR[0]], [PDR[1]], marker="s", s=26,
               color=PALETTE["neutral"], edgecolor="black", linewidth=0.6,
               label="PDR", zorder=4)
    ax.scatter([OURS[0]], [OURS[1]], marker="*", s=110,
               color=PALETTE["red_strong"], edgecolor="black", linewidth=0.5,
               label="H-ST-GRPO", zorder=5)
    ax.scatter([MIP[0]], [MIP[1]], marker="D", s=26,
               color=PALETTE["violet"], edgecolor="black", linewidth=0.5,
               label="MIP-AGV", zorder=4)

    ax.axvline(OURS[0], color=PALETTE["red_strong"], linewidth=0.7,
               linestyle=":", zorder=1)

    ax.annotate("0.95 s", (OURS[0], 1330), ha="center", va="bottom",
                fontsize=5.5, color=PALETTE["red_strong"])
    ax.annotate("%.1f" % OURS[1], (OURS[0], OURS[1]),
                textcoords="offset points", xytext=(7, -10), fontsize=5.5)
    ax.annotate("%.0f" % PDR[1], (PDR[0], PDR[1]),
                textcoords="offset points", xytext=(4, 3), fontsize=5.5)
    ax.annotate("%.1f" % MIP[1], (MIP[0], MIP[1]),
                textcoords="offset points", xytext=(0, 8), ha="center",
                fontsize=5.5)
    ax.annotate("%.0f" % rga_obj[-1], (rga_time[-1], rga_obj[-1]),
                textcoords="offset points", xytext=(6, -2), ha="left",
                fontsize=5.5)

    ax.set_xscale("log")
    ax.set_xlim(2e-4, 40)
    ax.set_ylim(620, 1360)
    ax.set_xlabel("Wall-clock time per instance (s, log scale)")
    ax.set_ylabel("Objective (6$\\times$6)")
    ax.grid(True, which="major", axis="both", linewidth=0.3,
            color=PALETTE["neutral"], alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=5.5, frameon=False,
              handletextpad=0.4, borderaxespad=0.2)
    finalize_figure(fig, OUT / "fig_anytime.pdf")
    print("rga points", list(zip(rga_time, rga_obj)))


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()
