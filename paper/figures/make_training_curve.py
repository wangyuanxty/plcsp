"""Training curves of the three seeds of the controller.

The values are the validation objectives written by the training loop to the
queue logs, one every 25 episodes. The logs are the source of record; nothing
is retyped here.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import (PALETTE, FigureStyle, apply_publication_style,
                      finalize_figure)

OUT = Path(__file__).resolve().parent
LOGS = Path(__file__).resolve().parents[2] / "plcsp_v5" / "queue_logs"

RUNS = [
    "20261028_194020_T_m3_agv6_ours_s1.log",
    "20261028_205104_T_m3_agv6_ours_s2.log",
    "20261028_223447_T_m3_agv6_ours_s3.log",
]
EPISODES = np.arange(25, 201, 25)


def read_run(name):
    text = (LOGS / name).read_text(encoding="utf-8", errors="ignore")
    vals = [float(v) for v in
            re.findall(r"each_episode_result: .*obj_100ins_mean=([0-9.]+)",
                       text)]
    if len(vals) != len(EPISODES):
        raise SystemExit(f"{name}: {len(vals)} evaluations")
    return np.array(vals)


def main() -> None:
    runs = np.vstack([read_run(r) for r in RUNS])
    mean = runs.mean(axis=0)

    fig, ax = plt.subplots(figsize=(3.4, 2.15))

    for row in runs:
        ax.plot(EPISODES, row, color=PALETTE["blue_secondary"], linewidth=0.7,
                alpha=0.55, zorder=2)
    ax.plot(EPISODES, mean, color=PALETTE["blue_main"], linewidth=1.4,
            marker="o", markersize=2.8, zorder=3, label="mean of three seeds")
    ax.fill_between(EPISODES, runs.min(axis=0), runs.max(axis=0),
                    color=PALETTE["blue_secondary"], alpha=0.13, linewidth=0,
                    zorder=1, label="seed range")

    ax.set_xlabel("Training episode", fontsize=7)
    ax.set_ylabel("Validation objective", fontsize=7)
    ax.set_xlim(15, 210)
    ax.set_ylim(830, 1110)
    ax.set_xticks(EPISODES)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.2, length=2.2)
    ax.grid(axis="y", linewidth=0.3, color=PALETTE["neutral"], alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.4,
              borderpad=0.2, labelspacing=0.3)

    fig.tight_layout(pad=0.25)
    finalize_figure(fig, OUT / "fig_training_curve", formats=["pdf"])
    print("mean", [round(v, 1) for v in mean])
    print("best per seed", [round(r.min(), 1) for r in runs])
    print("episode of best", [int(EPISODES[r.argmin()]) for r in runs])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.8))
    main()