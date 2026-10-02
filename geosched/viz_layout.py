# -*- coding: utf-8 -*-
"""布局可视化：Layout → PNG（机台 / 装卸点 / 缓冲 / 充电位 / 走廊图）——论文配图 + 共演化设计检查。

用法：
  python -m geosched.viz_layout                                    # 三类型对照图（line/U/island）
  python -m geosched.viz_layout --type U --aisle 1.2 --gap 0.7     # 单张（如共演化推荐设计）
输出：geosched/figs/*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False      # 中文字体下负号正常显示

from .env.layout import Layout, sample_layout
from .env.corridors import build_corridor_graph

C_MACHINE_FACE, C_MACHINE_EDGE = "#dce9f7", "#2b5d9e"
C_DOCK, C_BUFFER, C_CHARGER, C_CORRIDOR = "#e07b39", "#5b8c5a", "#c9a227", "#b9b9b9"


def add_legend(fig) -> None:
    """图例（挂整图底部单行——不压任何子图的坐标轴）：机台/装卸点/缓冲/充电/走廊图。"""
    from matplotlib.lines import Line2D
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=C_MACHINE_FACE, edgecolor=C_MACHINE_EDGE),
        Line2D([], [], marker="o", ls="", ms=5, color=C_DOCK),
        Line2D([], [], marker="s", ls="", ms=7, color=C_BUFFER),
        Line2D([], [], marker="*", ls="", ms=10, color=C_CHARGER),
        Line2D([], [], color=C_CORRIDOR, lw=1.2, marker=".", ms=3),
    ]
    labels = ["机台 M#（矩形=占地）", "装卸点 dock（AGV 取放位）", "缓冲区 B#",
              "充电桩 C#", "距离矩阵：相邻 dock 连线（简化，非避障路径）"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.06))


def draw(ax, lay: Layout, title: str = "", corridor: bool = True) -> None:
    """画一个布局：机台(蓝矩形)/装卸点(橙点)/缓冲(绿方)/充电桩(金星)/走廊图(灰线+节点)。

    注：走廊图 = 先验抽象路网（dock 与环形节点的直线连接，AGV 沿其最短路行走）；
    非物理通道 CAD；line 布局的锯齿 = dock 上下交替所致；U/岛式的长斜线 = 环路闭合边。
    """
    if corridor:
        try:
            g = build_corridor_graph(lay)
            for u, v in g.edges:
                (x1, y1), (x2, y2) = g.nodes[u]["pos"], g.nodes[v]["pos"]
                ax.plot([x1, x2], [y1, y2], color=C_CORRIDOR, lw=1.0, zorder=1)
            for n in g.nodes:
                x, y = g.nodes[n]["pos"]
                ax.plot(x, y, marker=".", ms=2, color="#8a8a8a", zorder=2)
        except Exception:                     # 走廊叠加失败不影响主图
            pass
    for m in lay.machines:
        ax.add_patch(Rectangle((m.x - m.w / 2, m.y - m.h / 2), m.w, m.h,
                               facecolor=C_MACHINE_FACE, edgecolor=C_MACHINE_EDGE,
                               lw=1.2, zorder=3))
        ax.text(m.x, m.y, f"M{m.id}", ha="center", va="center", fontsize=7, zorder=5)
        ax.plot(*m.dock, marker="o", ms=3.2, color=C_DOCK, zorder=4)
    for b in lay.buffers:
        ax.plot(b.x, b.y, marker="s", ms=7, color=C_BUFFER, zorder=4)
        ax.text(b.x, b.y + 0.5, f"B{b.id}", ha="center", fontsize=6, color=C_BUFFER)
    for c in lay.chargers:
        ax.plot(c.x, c.y, marker="*", ms=11, color=C_CHARGER, zorder=4)
        ax.text(c.x, c.y + 0.5, f"C{c.id}", ha="center", fontsize=6, color=C_CHARGER)
    ax.set_aspect("equal")
    ax.set_title(title or f"{lay.layout_type} (seed={lay.seed})", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default=None, choices=["line", "U", "island"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-machines", type=int, default=6)
    ap.add_argument("--aisle", type=float, default=1.5)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path("geosched/figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.type:                                  # 单张（如共演化推荐设计）
        lay = sample_layout(args.n_machines, args.type, args.seed,
                            aisle_width=args.aisle, machine_gap=args.gap)
        fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
        draw(ax, lay, f"{args.type}  aisle={args.aisle}  gap={args.gap}  seed={args.seed}")
        add_legend(fig)
        p = Path(args.out) if args.out else out_dir / f"design_{args.type}.png"
        fig.savefig(p, bbox_inches="tight")
        print(p)
    else:                                          # 三类型对照（论文背景图）
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
        for ax, lt in zip(axes, ("line", "U", "island")):
            draw(ax, sample_layout(args.n_machines, lt, args.seed,
                                   aisle_width=args.aisle, machine_gap=args.gap), lt)
        add_legend(fig)
        p = out_dir / "layouts_three_types.png"
        fig.savefig(p, bbox_inches="tight")
        print(p)


if __name__ == "__main__":
    main()
