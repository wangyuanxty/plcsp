# -*- coding: utf-8 -*-
"""论文结构图（scientific-figure-making 风格）。

与 `make_figures.py` 分开：那边是**数据图**（画实测数字），这边是**示意图**。
示意图的成本全在版式调参上，混在一起会把数据脚本撑得难以维护。

图：
  fig_motivation - Motivated Example（范式 A：运行示例 + 失败案例 + 我们的做法）
                   三栏：只排产 → 再派车（饿死机床）→ 一条链决策（不饿死）
  fig_overview   - Solution Overview：状态 → token → 编码器 → 各头 → 联合链 → 组相对优势

两个图都是**示意**，不来自实测；图注里已写明。

输出：[figure-name].pdf（矢量）+ .png（预览）。
运行：D:/anaconda/envs/py312/python.exe paper/figures/make_schematics.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

C = {
    "ink": "#272727",
    "muted": "#767676",
    "plan": "#3775BA",
    "agv": "#42949E",
    "bad": "#B64342",
    "good": "#2F7A46",
    "cut": "#B64342",
}

STYLE = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "pdf.fonttype": 42,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def _box(ax, x, y, w, h, text, *, fc, ec, tc, fs=7.2, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=fc, edgecolor=ec, linewidth=1.0, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.4)


def _arr(ax, p, q, *, color, ls="-", lw=1.0, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=8, color=color,
        linewidth=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


# --------------------------------------------------------------------------
# Motivated Example —— 运行示例：分开做会饿死机床，合起来做不会
# --------------------------------------------------------------------------
def _machine(ax, cx, yb, w, h, name, sub, sub_color, *,
             edge="#8A929B", lw=0.9, ls="-") -> None:
    """一台机床。第一行机号，第二行它当前的状态。"""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2.0, yb), w, h, boxstyle="round,pad=0,rounding_size=0.05",
        facecolor="#FFFFFF", edgecolor=edge, linewidth=lw, linestyle=ls,
        zorder=3))
    ax.text(cx, yb + h * 0.68, name, ha="center", va="center", fontsize=8.0,
            color=C["ink"], fontweight="bold", zorder=4)
    ax.text(cx, yb + h * 0.28, sub, ha="center", va="center", fontsize=8.0,
            color=sub_color, zorder=4)


def _agv(ax, cx, cy, color) -> None:
    """一辆车。画在它正在走的那条搬运线上。"""
    _box(ax, cx - 0.14, cy - 0.09, 0.28, 0.18, "AGV",
         fc=color, ec=color, tc="#FFFFFF", fs=8.0, bold=True)


def fig_motivation() -> None:
    """三栏运行示例：同一批搬运，分开决策会饿死机床，联合决策不会。

    (a) 只对机床排产 —— 车队不在输入里，两个搬运挤在同一个时间窗。
    (b) 照计划派车 —— 一辆车只能做一个，M2 空转，后面的块集体后移。
    (c) 合成一条链 —— 一辆车的一趟行程做完两个搬运，没有机床空转。
    全部为示意，不来自实测。
    """
    apply_style()
    fig = plt.figure(figsize=(7.0, 2.52))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ink, muted = C["ink"], C["muted"]
    agv_dark = "#2E6E77"          # 车队色加深一档，保证白字可读

    PW = 2.18                     # 面板宽
    PXL = (0.00, 2.41, 4.82)      # 三个面板的左边界
    PB, PT = 0.06, 2.46           # 面板下 / 上边界

    MW, MH, MGAP = 0.56, 0.42, 0.15
    MC = [0.10 + MW / 2.0 + k * (MW + MGAP) for k in range(3)]
    MYB = 1.66                    # 机床底边
    L1, L2 = 1.30, 0.80           # 两条搬运线的 y；间距要放得下 (c) 的拐角

    for px, (fc, ec) in zip(PXL, [("#F6F7F9", "#CCD2D8"),
                                  ("#FCF3F3", "#E5C6C5"),
                                  ("#F2F8F3", "#C2DBC7")]):
        ax.add_patch(FancyBboxPatch(
            (px, PB), PW, PT - PB, boxstyle="round,pad=0,rounding_size=0.07",
            facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=0))

    def head(p: int, line1: str, c1: str, line2: str) -> None:
        ax.text(PXL[p] + 0.10, PT - 0.04, line1, ha="left", va="top",
                fontsize=8.8, color=c1, fontweight="bold")
        ax.text(PXL[p] + 0.10, PT - 0.20, line2, ha="left", va="top",
                fontsize=8.0, color=muted)

    def lane_text(p: int, y: float, text: str, color: str) -> None:
        """线注：写在该条搬运线的正上方。"""
        ax.text(PXL[p] + (MC[0] + MC[1]) / 2.0, y + 0.14, text,
                ha="center", va="bottom", fontsize=8.0, color=color)

    def lane2_text(p: int, text: str, color: str) -> None:
        ax.text(PXL[p] + (MC[1] + MC[2]) / 2.0, L2 + 0.10, text,
                ha="center", va="bottom", fontsize=8.0, color=color)

    def note(p: int, text: str, color: str) -> None:
        ax.text(PXL[p] + PW / 2.0, 0.62, text, ha="center", va="top",
                fontsize=8.0, color=color, linespacing=1.45)

    # ---------------- (a) 只排产：车队不在输入里 ----------------
    px = PXL[0]
    head(0, "(a)  Schedule the machines", ink, "the vehicle is not an input")
    for k, (nm, sb, col) in enumerate([("M1", "J1", ink), ("M2", "J2", ink),
                                       ("M3", "free", muted)]):
        _machine(ax, px + MC[k], MYB, MW, MH, nm, sb, col)
    _arr(ax, (px + MC[0], L1), (px + MC[1], L1), color=C["plan"], lw=1.3)
    lane_text(0, L1, "J1 due at $t$", C["plan"])
    _arr(ax, (px + MC[1], L2), (px + MC[2], L2), color=C["plan"], lw=1.3)
    lane2_text(0, "J2 due at $t{+}1$", C["plan"])
    note(0, "Two transfers one time unit apart;\n"
            "transport is assumed to be free.", muted)

    # ---------------- (b) 再派车：一辆车，做不过来 ----------------
    px = PXL[1]
    head(1, "(b)  Dispatch the vehicle", C["bad"],
         "the plan can no longer change")
    _machine(ax, px + MC[0], MYB, MW, MH, "M1", "J1 waits", C["bad"])
    _machine(ax, px + MC[1], MYB, MW, MH, "M2", "idle", C["bad"],
             edge=C["bad"], lw=1.3, ls=(0, (3, 1.8)))
    _machine(ax, px + MC[2], MYB, MW, MH, "M3", "J2", ink)
    _arr(ax, (px + MC[0], L1), (px + MC[1], L1), color=C["bad"],
         ls=(0, (3, 2)), lw=1.3)
    lane_text(1, L1, "no vehicle free for J1", C["bad"])
    _arr(ax, (px + MC[1], L2), (px + MC[2], L2), color=agv_dark, lw=1.3)
    lane2_text(1, "J2 served", agv_dark)
    _agv(ax, px + MC[1] + 0.30, L2, agv_dark)
    note(1, "Only one transfer can be served.\n"
            "M2 starves; later blocks shift.", C["bad"])

    # ---------------- (c) 一条链：一趟行程做完两个搬运 ----------------
    px = PXL[2]
    head(2, "(c)  Decide both in one chain", C["good"],
         "the plan already knows the fleet")
    _machine(ax, px + MC[0], MYB, MW, MH, "M1", "J1", ink)
    _machine(ax, px + MC[1], MYB, MW, MH, "M2", "fed", C["good"],
             edge=C["good"], lw=1.3)
    _machine(ax, px + MC[2], MYB, MW, MH, "M3", "J2", ink)
    ax.plot([px + MC[0], px + MC[1]], [L1, L1], color=C["good"], lw=1.3,
            zorder=1, solid_capstyle="round")
    ax.plot([px + MC[1], px + MC[1]], [L1, L2], color=C["good"], lw=1.3,
            zorder=1, solid_capstyle="round")
    _arr(ax, (px + MC[1], L2), (px + MC[2], L2), color=C["good"], lw=1.3)
    _agv(ax, px + MC[1], (L1 + L2) / 2.0, agv_dark)
    lane_text(2, L1, "one tour: J1 then J2", C["good"])
    note(2, "One tour serves both transfers;\n"
            "no machine goes idle.", C["good"])

    ax.set_xlim(0.0, 7.0)
    ax.set_ylim(0.0, 2.52)
    ax.axis("off")
    save(fig, "fig_motivation")


# --------------------------------------------------------------------------
# Solution Overview
# --------------------------------------------------------------------------
def _sbox(ax, x, y, w, h, text, *, fc, ec, tc, fs=6.8, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=fc, edgecolor=ec, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def fig_overview() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    blue, ink, muted = C["plan"], C["ink"], C["muted"]
    W = 7.20

    y1, h1 = 2.90, 0.56
    _sbox(ax, 0.10, y1, 1.00, h1, "shop state\nat a decision",
          fc="#F2F2F2", ec=ink, tc=ink, fs=6.4)

    segs = [("M", "machine", "7"), ("B", "job", "9"), ("V", "vehicle", "11"),
            ("G", "global", "3"), ("Z", "aisle seg.", "4")]
    tw, tg, tx0 = 0.84, 0.12, 1.28
    for k, (tag, name, dim) in enumerate(segs):
        x = tx0 + k * (tw + tg)
        ours = (k == len(segs) - 1)
        _sbox(ax, x, y1, tw, h1, f"{tag}\n{name}  $F$={dim}",
              fc="#EAF3E6" if ours else "#DCE7F4",
              ec="#5C9E5C" if ours else blue,
              tc="#2F6B2F" if ours else blue, fs=6.4)
    tok_span = (tx0, tx0 + len(segs) * tw + (len(segs) - 1) * tg)
    tok_cx = sum(tok_span) / 2
    _arr(ax, (1.10, y1 + h1 / 2), (tx0, y1 + h1 / 2), color=ink)
    ax.text(tok_cx, y1 + h1 + 0.08,
            "tokens, padded to a common width, plus a type embedding",
            ha="center", va="bottom", fontsize=6.4, color=muted)

    y2, h2 = 2.02, 0.52
    tw2 = 2.80
    _sbox(ax, tok_cx - tw2 / 2, y2, tw2, h2,
          "Transformer   8 layers, $d$=128, no attention mask",
          fc="#DCE7F4", ec=blue, tc=blue, fs=6.8)
    _arr(ax, (tok_cx, y1), (tok_cx, y2 + h2), color=ink)

    y3, h3 = 1.22, 0.52
    heads = [("S", "machine"), ("L", "vehicle"), ("R", "route"),
             ("M", "maint."), ("C", "charge"), ("B", "batch")]
    hw, hg = 0.62, 0.10
    hspan = len(heads) * hw + (len(heads) - 1) * hg
    hx0 = tok_cx - hspan / 2
    for k, (tag, sub) in enumerate(heads):
        x = hx0 + k * (hw + hg)
        _sbox(ax, x, y3, hw, h3, f"{tag}\n{sub}", fc="#FFFFFF",
              ec=blue, tc=blue, fs=6.2)
    _arr(ax, (tok_cx, y2), (tok_cx, y3 + h3), color=ink)

    y4, h4 = 0.46, 0.44
    cw = hspan
    _sbox(ax, tok_cx - cw / 2, y4, cw, h4,
          "joint chain:  every S, L, R, M, C, B decision of one episode",
          fc="#EAF3E6", ec="#5C9E5C", tc="#2F6B2F", fs=6.6)
    _arr(ax, (tok_cx, y3), (tok_cx, y4 + h4), color=ink)

    y5, h5 = -0.30, 0.46
    bw, bgap = 2.25, 0.42
    pair = 2 * bw + bgap
    px0 = tok_cx - pair / 2
    _sbox(ax, px0, y5, bw, h5,
          "terminal reward\n$r=w\\cdot(-C_{\\max},-E,-\\mathrm{TWT})$",
          fc="#F2F2F2", ec=ink, tc=ink, fs=6.2)
    _sbox(ax, px0 + bw + bgap, y5, bw, h5,
          "group-relative advantage\n$A=z(r)$,  no critic",
          fc="#F2F2F2", ec=ink, tc=ink, fs=6.2)
    _arr(ax, (px0 + bw, y5 + h5 / 2),
         (px0 + bw + bgap, y5 + h5 / 2), color=ink)
    _arr(ax, (tok_cx, y4), (tok_cx, y5 + h5), color=ink)

    ax.set_xlim(0.0, W)
    ax.set_ylim(-0.50, 3.62)
    ax.axis("off")
    save(fig, "fig_overview")


if __name__ == "__main__":
    fig_motivation()
    fig_overview()
