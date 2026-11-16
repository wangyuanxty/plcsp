"""Radar overview: six normalised axes per instance family.

Each axis is min-max normalised inside its panel, with the smallest value at
the outer radius, so the outward direction is "better" on every axis. The
figure is descriptive only; the tables and the paired tests carry the claims.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubstyle import (PALETTE, FigureStyle, apply_publication_style,
                      finalize_figure)

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "plcsp_v5/trained_model"
RES = ROOT / "plcsp_v5/results"
EXP = Path("D:/research/remote-export-2026-11-14")

AXES = ["Cmax", "PEC", "TEC", "Idle", "Obj.", "Std"]
COLOURS = {
    "Ours": PALETTE["blue_main"],
    "RGA": PALETTE["teal"],
    "Wang et al.": PALETTE["red_strong"],
    "Li et al.": PALETTE["violet"],
    "PDR": PALETTE["neutral"],
}
ORDER = ["Ours", "RGA", "Wang et al.", "Li et al.", "PDR"]

OURS = {
    "6x6": ("m3_agv6_ours_s{s}", "eval_st_seed3_seed3.npz"),
    "10x6": ("m3_agv_s{s}_J10M6", "eval_st_seed3_seed3.npz"),
    "10x10": ("m3_agv_s{s}_J10M10", "eval_st_seed3_seed3.npz"),
    "15x10": ("plan15_J15M10E2_s{s}_local", "eval_st_seed3_15x10.npz"),
    "20x6": ("plan20_J20M6E3_s{s}_local", "eval_st_seed3_20x6.npz"),
}
WANG = {
    "6x6": ("m3_agv6_e2e_s{s}", "eval_st_seed3_seed3.npz"),
    "10x6": ("m3_e2e_s{s}_J10M6", "eval_st_seed3_seed3.npz"),
    "10x10": ("m3_e2e_s{s}_J10M10", "eval_st_seed3_seed3.npz"),
    "15x10": ("wang_J15M10E2_s{s}_local", "eval_st_seed3_15x10.npz"),
    "20x6": ("wang_J20M6E3_s{s}_local", "eval_st_seed3_20x6.npz"),
}
LI = {
    "6x6": [ROOT / "plcsp_v5/baselines/li2025/results" /
            f"m2_m3_li_J6M6E2_s{s}.json" for s in (1, 2, 3)],
    "10x6": [ROOT / "plcsp_v5/baselines/li2025/results" /
             f"m2_m3_li_J10M6E2_s{s}.json" for s in (1, 2, 3)],
    "10x10": [ROOT / "plcsp_v5/baselines/li2025/results" /
              f"m2_m3_li_J10M10E2_s{s}.json" for s in (1, 2, 3)],
    "15x10": [EXP / f"15x10/baselines/li2025/results/m2_li15_J15M10E2_s{s}.json"
              for s in (1, 2, 3)],
    "20x6": [EXP / f"20x6/baselines/li2025/results/m2_li20_J20M6E3_s{s}.json"
             for s in (1, 2, 3)],
}
RGA = {
    "6x6": RES / "rga_m2_simeval_6x6.json",
    "10x6": RES / "rga_m2_simeval_J10M6E2.json",
    "10x10": RES / "rga_m2_simeval_J10M10E2.json",
    "15x10": EXP / "15x10/results/rga_m2_trailing_15x10.json",
    "20x6": RES / "rga_m2_trailing_20x6.json",
}
PDR = {
    "6x6": RES / "pdr_m2_trailing_agvrule3_6x6.json",
    "10x6": RES / "pdr_m2_trailing_agvrule3_10x6.json",
    "10x10": RES / "pdr_m2_trailing_agvrule3_10x10.json",
    "15x10": EXP / "15x10/results/pdr_m2_trailing_15x10.json",
    "20x6": RES / "pdr_m2_trailing_20x6_lu3.json",
}
SIZES = ["6x6", "10x6", "10x10", "15x10", "20x6"]
TITLES = {"6x6": r"$6\times6\times2$", "10x6": r"$10\times6\times2$",
          "10x10": r"$10\times10\times2$", "15x10": r"$15\times10\times2$",
          "20x6": r"$20\times6\times3$"}


def _npz_stats(pat, fname):
    vals = []
    for s in (1, 2, 3):
        z = np.load(TR / pat.format(s=s) / fname)
        c = np.asarray(z["cost4"], float)
        vals.append(np.column_stack([c[:, 0], c[:, 1], c[:, 2], c[:, 3],
                                     np.asarray(z["obj"], float)]))
    v = np.vstack(vals)                        # 300 x 5
    per = np.nanmean(np.vstack([x[:, 4] for x in vals]), axis=0)
    return {"Cmax": v[:, 0].mean(), "PEC": v[:, 1].mean(),
            "TEC": v[:, 2].mean(), "Idle": v[:, 3].mean(),
            "Obj.": v[:, 4].mean(), "Std": per.std(ddof=0)}


def _json_stats(paths, best_pair=False):
    recs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        rs = d["records"]
        if best_pair:
            b = min(d["summary"]["per_rule"], key=lambda r: r["mean"])
            key = (b["job_rule"], b["machine_rule"], b["agv_rule"])
            rs = [r for r in rs if (r["job_rule"], r["machine_rule"],
                                    r["agv_rule"]) == key]
        recs.append(rs)
    cols = {k: [] for k in ("mk", "pt", "transT", "idleT", "obj")}
    per_inst = {}
    for rs in recs:
        for r in rs:
            i = int(r["instance"])
            per_inst.setdefault(i, []).append(float(r["obj"]))
            for k in cols:
                cols[k].append(float(r[k]))
    per = np.array([np.mean(v) for v in per_inst.values()])
    return {"Cmax": np.mean(cols["mk"]), "PEC": np.mean(cols["pt"]),
            "TEC": np.mean(cols["transT"]), "Idle": np.mean(cols["idleT"]),
            "Obj.": np.mean(cols["obj"]), "Std": per.std(ddof=0)}


def collect():
    out = {}
    for size in SIZES:
        row = {}
        row["Ours"] = _npz_stats(*OURS[size])
        row["Wang et al."] = _npz_stats(*WANG[size])
        row["Li et al."] = _json_stats(LI[size])
        row["RGA"] = _json_stats([RGA[size]])
        row["PDR"] = _json_stats([PDR[size]], best_pair=True)
        out[size] = row
    return out


def main() -> None:
    data = collect()
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6),
                             subplot_kw={"polar": True})
    axes = axes.ravel()
    for k, size in enumerate(SIZES):
        ax = axes[k]
        row = data[size]
        for name in ORDER:
            r = []
            for a in AXES:
                vals = [row[m][a] for m in ORDER]
                lo, hi = min(vals), max(vals)
                v = row[name][a]
                r.append(1.0 if hi <= lo else (hi - v) / (hi - lo))
            r += r[:1]
            ax.plot(angles, r, color=COLOURS[name], linewidth=1.2, label=name)
            ax.fill(angles, r, color=COLOURS[name], alpha=0.05)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(AXES, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=9.5)
        for lab in ax.get_yticklabels():
            lab.set_bbox({"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.75, "pad": 0.3})
        ax.tick_params(pad=1)
        ax.set_title(TITLES[size], fontsize=9, pad=10)
        ax.grid(linewidth=0.3, color=PALETTE["neutral"], alpha=0.6)
        ax.spines["polar"].set_linewidth(0.5)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5,
               frameon=False, handlelength=1.6, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=0.4, w_pad=1.2, h_pad=1.2)
    finalize_figure(fig, OUT / "fig_radar", formats=["pdf"])
    for size in SIZES:
        print(size, {n: round(data[size][n]["Obj."], 1) for n in ORDER})


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.6))
    main()