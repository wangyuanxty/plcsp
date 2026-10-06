# -*- coding: utf-8 -*-
"""论文插图生成（scientific-figure-making 风格）。

数据来源：`docs/progress-log.md` §52（期① 实测），逐项对应如下——
  fig_energy_accounts : §52.11 表 1（13 个消融 run 的 makespan / 总能耗 / 净能耗 A 比值）
  fig_advantage       : §52.3（D 三臂 × 5 个训练种子）
  fig_layout          : 设计图，非实测（spec §3.2 的网格布局定义）

输出：[figure-name].pdf（矢量，LaTeX 用）+ .png（预览）。
运行：D:/anaconda/envs/py312/python.exe paper/figures/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "red_2": "#E9A6A1",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
}

# 双栏排版：单栏宽 3.4 in、跨栏宽 7.2 in。字号取 9（正文同级）。
STYLE = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.1,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "figure.dpi": 300,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, name: str) -> None:
    fig.tight_layout(pad=0.35)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# --------------------------------------------------------------------------
# 图 1：车间布局
# --------------------------------------------------------------------------
def fig_layout() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.75))

    n_rows, n_cols = 2, 3          # MK01：6 台机 → 3 列 × 2 行
    cell = 1.0
    w, h = 0.62, 0.62              # 机台方块边长

    # 通道格线（横竖两族）
    for r in range(n_rows + 1):
        ax.plot([0, n_cols], [r * cell, r * cell], color=PALETTE["neutral"],
                lw=2.2, zorder=1, solid_capstyle="round")
    for c in range(n_cols + 1):
        ax.plot([c * cell, c * cell], [0, n_rows], color=PALETTE["neutral"],
                lw=2.2, zorder=1, solid_capstyle="round")

    # 机台（占格）+ dock 连到所在格最近的左下角节点
    for idx in range(6):
        r, c = divmod(idx, n_cols)
        x0, y0 = c * cell + (cell - w) / 2, r * cell + (cell - h) / 2
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=PALETTE["blue_secondary"],
                               edgecolor="#272727", lw=1.0, zorder=3))
        ax.text(x0 + w / 2, y0 + h / 2, f"M{idx + 1}", ha="center", va="center",
                color="white", fontsize=7, zorder=4, fontweight="bold")
        ax.plot([x0, c * cell], [y0, r * cell], color="#767676", lw=0.9,
                ls=(0, (2, 1.6)), zorder=2)

    # 装卸站（格点外，接左边缘节点）
    lx, ly = -0.92, 0.55
    ax.add_patch(Rectangle((lx, ly - 0.13), 0.54, 0.42,
                           facecolor=PALETTE["green_3"], edgecolor="#272727",
                           lw=1.0, zorder=3))
    ax.text(lx + 0.27, ly + 0.08, "LU", ha="center", va="center",
            fontsize=7.5, zorder=4, fontweight="bold")
    ax.plot([lx + 0.54, 0.0], [ly + 0.08, ly + 0.08], color="#767676",
            lw=1.2, zorder=2)

    # 充电桩（落在通道节点上，≥2 个）
    for cx, cy in ((1.0, 2.0), (3.0, 0.0)):
        ax.plot(cx, cy, marker="s", ms=5.5, color=PALETTE["highlight"],
                markeredgecolor="#272727", markeredgewidth=0.8, zorder=5)
    ax.plot([], [], marker="s", ms=5.5, color=PALETTE["highlight"],
            markeredgecolor="#272727", markeredgewidth=0.8, ls="none",
            label="Charging station")

    # 一条示例行进路径：LU → M5 → M2（沿格线，直角走）
    route_x = [0.0, 0.0, 2.0]
    route_y = [ly + 0.08, 1.0, 1.0]
    ax.plot(route_x, route_y, color=PALETTE["blue_main"], lw=1.6, zorder=6)
    ax.plot([2.0, 1.6], [1.0, 1.42], color=PALETTE["blue_main"], lw=1.6,
            zorder=6)
    ax.add_patch(FancyArrowPatch((1.62, 1.44), (1.55, 1.52),
                                 arrowstyle="-|>", mutation_scale=8,
                                 color=PALETTE["blue_main"], zorder=6))

    ax.plot([], [], color=PALETTE["blue_main"], lw=1.6, label="AGV travel")
    ax.plot([], [], color="#767676", lw=0.9, ls=(0, (2, 1.6)), label="Dock link")

    ax.set_xlim(-1.05, n_cols + 0.15)
    ax.set_ylim(-0.18, n_rows + 0.18)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
              fontsize=7.5, handlelength=1.6, columnspacing=1.1,
              handletextpad=0.5)
    save(fig, "fig_layout")


# --------------------------------------------------------------------------
# 图 2：两种能耗口径
# --------------------------------------------------------------------------
# docs/progress-log.md §52.11 表 1
RUNS = [
    ("c-none",          49.27, 0.08316, 0.05614),
    ("c-no-production", 70.73, 0.07834, 0.03730),
    ("adv-reinforce",   71.12, 0.07837, 0.04922),
    ("m-no-route",      77.26, 0.07876, 0.04924),
    ("m-no-charge",     77.76, 0.07843, 0.04957),
    ("full",            78.90, 0.07808, 0.04737),
    ("m-no-pm",         80.20, 0.07809, 0.04846),
    ("c-no-logistics",  80.59, 0.07788, 0.04676),
    ("m-no-t3",         84.04, 0.07836, 0.04490),
    ("c-no-info",       84.31, 0.07789, 0.04644),
    ("m-no-batch",      84.56, 0.07730, 0.04348),
    ("m-no-r2",         86.69, 0.07719, 0.04362),
    ("m-no-geom",       87.18, 0.07828, 0.04480),
]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def fig_energy_accounts() -> None:
    apply_style()
    ms = np.array([r[1] for r in RUNS])
    panels = [
        ("Total energy / makespan", np.array([r[2] for r in RUNS]),
         PALETTE["red_strong"]),
        ("Net energy A / makespan", np.array([r[3] for r in RUNS]),
         PALETTE["blue_main"]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.1))
    for ax, (title, ratio, colour) in zip(axes, panels):
        # ρ 报的是 makespan 与**能耗本身**的秩相关（与正文 §6.5 同一口径）；
        # 纵轴画比值，因为比值谱的展宽才是这一节的视觉要点。
        energy = ratio * ms
        rho = _spearman(ms, energy)
        spread = (ratio.max() - ratio.min()) / ratio.mean() * 100.0
        ax.scatter(ms, ratio, s=26, color=colour, alpha=0.85,
                   edgecolor="#272727", linewidth=0.5, zorder=3)
        # 最小二乘参考线，只作视觉引导
        k, b = np.polyfit(ms, ratio, 1)
        xs = np.linspace(ms.min(), ms.max(), 50)
        ax.plot(xs, k * xs + b, color="#767676", lw=1.0, ls=(0, (3, 2)),
                zorder=2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Energy / makespan")
        ax.annotate(rf"$\rho$(makespan, energy) = {rho:.3f}", xy=(0.04, 0.06),
                    xycoords="axes fraction", fontsize=8,
                    color=colour, fontweight="bold")
        ax.annotate(f"ratio spread {spread:.1f}%", xy=(0.04, 0.17),
                    xycoords="axes fraction", fontsize=8, color="#4D4D4D")
    save(fig, "fig_energy_accounts")


# --------------------------------------------------------------------------
# 图 3：优势口径消融
# --------------------------------------------------------------------------
# docs/progress-log.md §52.3
ARMS = {
    "scalar":    [84.04, 81.24, 87.19, 79.27, 71.56],
    "reinforce": [71.12, 79.96, 75.31, 71.79, 85.85],
    "raw":       [84.97, 87.98, 90.72, 79.35, 85.41],
}
ARM_COLOUR = {
    "scalar": PALETTE["blue_main"],
    "reinforce": PALETTE["teal"],
    "raw": PALETTE["red_strong"],
}


def fig_advantage() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    rng = np.random.default_rng(0)          # 仅用于散点横向抖动

    names = list(ARMS)
    for i, name in enumerate(names):
        vals = np.array(ARMS[name], dtype=float)
        jitter = rng.uniform(-0.10, 0.10, size=vals.size)
        ax.scatter(np.full(vals.size, i) + jitter, vals, s=24,
                   color=ARM_COLOUR[name], alpha=0.9, edgecolor="#272727",
                   linewidth=0.5, zorder=3)
        # 均值与标准差
        m, sd = vals.mean(), vals.std(ddof=1)
        ax.plot([i - 0.22, i + 0.22], [m, m], color="#272727", lw=1.6,
                zorder=4)
        ax.plot([i, i], [m - sd, m + sd], color="#272727", lw=1.0, zorder=4)
        ax.annotate(f"{m:.1f}", xy=(i, m + sd + 1.0), ha="center", fontsize=8,
                    color="#272727")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Makespan")
    ax.set_xlim(-0.45, len(names) - 0.55)
    ax.annotate("5 training seeds per arm", xy=(0.5, -0.30),
                xycoords="axes fraction", ha="center", fontsize=7.5,
                color="#4D4D4D")
    save(fig, "fig_advantage")


# --------------------------------------------------------------------------
# 图 4：主对比（三目标 × 两档 × 五方法）
# --------------------------------------------------------------------------
# docs/progress-log.md §52.1
MAIN = {
    # 方法: (tier A 三元组, tier B 三元组)；None = 该档未跑
    "Rule":          ((73.01, 5.675, 0.000), (105.35, 7.936, 60.14)),
    "GRPO (ms)":     ((57.60, 4.714, 0.000), (74.09, 5.790, 5.977)),
    "GRPO (en)":     ((52.91, 4.341, 0.000), (96.93, 7.434, 97.458)),
    "GRPO (TWT)":    (None, (77.30, 6.048, 7.392)),
    "Ours":          ((49.27, 4.097, 0.000), (78.90, 6.160, 5.042)),
    "NSGA-II":       ((49.14, 4.093, 0.000), (79.56, 6.239, 42.48)),
}
MAIN_ORDER = ["Rule", "GRPO (ms)", "GRPO (en)", "GRPO (TWT)", "Ours",
              "NSGA-II"]
# 同色系靠**纹理**再编码一次（手册：绝不能只靠颜色传递信息）
MAIN_COLOUR = {
    "Rule": "#D9D9D9", "GRPO (ms)": "#AADCA9", "GRPO (en)": "#AADCA9",
    "GRPO (TWT)": "#AADCA9", "Ours": PALETTE["blue_main"],
    "NSGA-II": PALETTE["red_2"],
}
MAIN_HATCH = {
    "Rule": "", "GRPO (ms)": "", "GRPO (en)": "///", "GRPO (TWT)": "xxx",
    "Ours": "", "NSGA-II": "",
}
OBJ = [("Makespan", 0), ("Energy (kWh)", 1), ("TWT", 2)]


def fig_main() -> None:
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.6))
    for ax, (title, k) in zip(axes, OBJ):
        xs = np.arange(2)                       # 两档
        w = 0.13
        n = len(MAIN_ORDER)
        for j, m in enumerate(MAIN_ORDER):
            vals = []
            for t in (0, 1):
                trip = MAIN[m][t]
                vals.append(np.nan if trip is None else trip[k])
            off = (j - (n - 1) / 2) * w
            bars = ax.bar(xs + off, vals, width=w * 0.88,
                          color=MAIN_COLOUR[m],
                          edgecolor="#272727", linewidth=0.4,
                          hatch=MAIN_HATCH[m],
                          label=m if k == 0 else None, zorder=3)
            if m == "Ours":
                for b in bars:
                    b.set_linewidth(1.2)
                    b.set_edgecolor("#0B3A6E")
        ax.set_xticks(xs)
        ax.set_xticklabels(["Tier A", "Tier B"])
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Value", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MAIN_COLOUR[m],
                             edgecolor="#272727", linewidth=0.4,
                             hatch=MAIN_HATCH[m])
               for m in MAIN_ORDER]
    fig.legend(handles, MAIN_ORDER, loc="lower center", ncol=3, fontsize=7.0,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.35,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.93))
    save(fig, "fig_main")


# --------------------------------------------------------------------------
# 图 5：敏感性（5 个参数 × 3 水平）
# --------------------------------------------------------------------------
# docs/progress-log.md §52.9.4；水平索引 0=低 1=默认 2=高
SENS = {
    # 参数: (makespan 三元组, TWT 三元组)
    "setup time":      ((67.38, 78.90, 95.73), (0.00, 5.042, 100.07)),
    "failure rate":    ((78.81, 78.90, 84.74), (4.893, 5.042, 45.41)),
    "rework prob.":    ((73.53, 78.90, 81.99), (3.112, 5.042, 8.05)),
    "maintenance int.": ((78.90, 78.90, 79.90), (5.042, 5.042, 5.04)),
    "vehicle MTBF":    ((80.02, 78.90, 78.81), (7.730, 5.042, 5.16)),
}
SENS_MARK = ["o", "s", "^", "D", "v"]
SENS_LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def fig_sensitivity() -> None:
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.1))
    levels = np.arange(3)
    for ax, (title, k) in zip(axes, [("Makespan", 0), ("TWT", 1)]):
        for i, (name, series) in enumerate(SENS.items()):
            y = series[k]
            ax.plot(levels, y, color=PALETTE["blue_main"] if i == 0
                    else plt.cm.viridis(i / 5.0),
                    marker=SENS_MARK[i], markersize=4, linewidth=1.2,
                    linestyle=SENS_LS[i], label=name, zorder=3)
        ax.set_xticks(levels)
        ax.set_xticklabels(["low", "default", "high"])
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Value", fontsize=8)
    axes[0].legend(fontsize=7, handlelength=1.8, labelspacing=0.25)
    save(fig, "fig_sensitivity")


# --------------------------------------------------------------------------
# 图 6：消融 Δ（12 臂，带训练种子 sd 带）
# --------------------------------------------------------------------------
# docs/progress-log.md §52.8.1（Δ = arm − full，正 = 拿掉该机制后变差）
ABL = [
    ("c-none",          -29.63,  -5.04),
    ("c-no-production",  -8.17,  -2.65),
    ("adv-reinforce",    -7.78,  -3.83),
    ("m-no-route",       -1.64,  +6.58),
    ("m-no-charge",      -1.15,  +4.08),
    ("m-no-pm",          +1.30,  +0.49),
    ("c-no-logistics",   +1.69, +19.07),
    ("m-no-t3",          +5.14, +16.39),
    ("c-no-info",        +5.41, +28.06),
    ("m-no-batch",       +5.66, +34.74),
    ("m-no-r2",          +7.79, +30.19),
    ("m-no-geom",        +8.28, +26.23),
]
SEED_SD = 5.90          # 实测训练种子 sd（§4）


def fig_ablation() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    names = [r[0] for r in ABL]
    dm = np.array([r[1] for r in ABL])
    dt = np.array([r[2] for r in ABL])
    y = np.arange(len(names))
    h = 0.36

    ax.axvspan(-SEED_SD, SEED_SD, color="#F2F2F2", zorder=0)
    ax.axvline(0, color="#767676", linewidth=0.8, zorder=1)

    ax.barh(y + h / 2, dm, height=h, color=PALETTE["blue_main"],
            edgecolor="#272727", linewidth=0.4, label="Makespan", zorder=3)
    ax.barh(y - h / 2, dt, height=h, color=PALETTE["teal"],
            edgecolor="#272727", linewidth=0.4, label="TWT", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ against the all-on configuration", fontsize=8)
    ax.annotate(f"one training-seed sd ({SEED_SD:.2f})", xy=(0.0, 1.0),
                xycoords=("data", "axes fraction"), xytext=(2, -4),
                textcoords="offset points", fontsize=6.8, color="#4D4D4D",
                ha="left", va="top")
    ax.legend(fontsize=7.5, loc="lower right", handlelength=1.4)
    save(fig, "fig_ablation")


if __name__ == "__main__":
    fig_layout()
    fig_energy_accounts()
    fig_advantage()
    fig_main()
    fig_sensitivity()
    fig_ablation()
