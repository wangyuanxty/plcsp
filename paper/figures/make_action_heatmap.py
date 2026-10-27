"""Action-probability heatmap for the three top-level heads.

Reads the raw dump from diag_rs_actions.py (one entry per decision) and plots
the mean probability of each rule against the decision index inside the
episode.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SRC = Path("C:/Users/evari/AppData/Local/Temp/figrefs/probs_local_s1.json")

LABELS = [
    ["est", "eft", "spt", "mwko"],
    ["min-t", "min-e", "min-trans"],
    ["earliest", "nearest", "none"],
]
TITLES = ["(a) Job head", "(b) Machine head", "(c) AGV head"]

from pubstyle import FigureStyle, apply_publication_style, finalize_figure


def main():
    data = json.loads(SRC.read_text())
    n_dec = max(r["decision"] for r in data) + 1
    n_heads = len(data[0]["probs"])
    fig, axes = plt.subplots(1, n_heads, figsize=(7.0, 1.9),
                             gridspec_kw={"wspace": 0.42})

    for h, ax in enumerate(axes):
        n_act = len(data[0]["probs"][h])
        mat = np.zeros((n_act, n_dec))
        cnt = np.zeros(n_dec)
        for r in data:
            d = r["decision"]
            cnt[d] += 1
            for a in range(n_act):
                mat[a, d] += r["probs"][h][a]
        mat = mat / np.maximum(cnt, 1)
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n_dec))
        ax.set_xticklabels([str(i + 1) for i in range(n_dec)])
        ax.set_yticks(range(n_act))
        ax.set_yticklabels(LABELS[h])
        ax.set_xlabel("Decision index")
        ax.set_title(TITLES[h], loc="left")
        for a in range(n_act):
            for d in range(n_dec):
                v = mat[a, d]
                ax.text(d, a, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.2,
                        color="white" if v > 0.55 else "black")

    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015,
                 label="Mean probability")
    finalize_figure(fig, OUT / "fig_action_heatmap", formats=["pdf"])
    print("wrote fig_action_heatmap.pdf", "decisions", n_dec)


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()
