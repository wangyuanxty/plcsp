# -*- coding: utf-8 -*-
"""MKT 基准的数据层（Brandimarte MK + 机台间行程时间）。

**MKT = 原始 MK 的加工数据（一字不改）+ 一张机台间行程时间矩阵 + 一个车辆数设定**
（核实于 progress-log §19.7d）。故本模块**只补运输**，不碰加工数据。

口径（与 HGS 的读取方式一致）：
布局文件是 `(m+1)×(m+1)`，**第 0 行/列是装卸站（Load/Unload）**；
机台间行程时间 = 丢 LU 后的 `m×m` 子矩阵（`raw[1:, 1:]`）。

⚠️ 该矩阵**非对称**——**但 `m=4` 是唯一例外**（恰好对称），故"非对称"只能钉在具体规模上。
⚠️ 对角线为 0——**但 `m=17` 是唯一例外**（`t[5][5]=4`，上游数据如此）。
两处异常均已实测并记入 `plcsp/data/mkt/README.md` §2，且有测试钉住。

数据来源与许可见 `plcsp/data/mkt/README.md`。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .instances import Instance, load_mk

MKT_LAYOUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mkt" / "layouts"


def load_mkt_layout(n_machines: int, *, drop_lu: bool = True) -> np.ndarray:
    """读 `{n}_machine_layout.txt`。

    `drop_lu=True`（默认）返回 `m×m` 的机台间矩阵；`False` 返回含装卸站的 `(m+1)×(m+1)` 原矩阵。
    **取不到该机台数就显式报错**——不得回退到邻近规模（Review Focus #6）。

    ⚠️ 本函数**不**做任何清洗：`m=17` 的对角第 6 项是 4、`m=4` 的矩阵恰好对称，
       都是上游数据的原样（见 README §2）。要"修正"它们须在调用侧显式做，不得在此静默抹平。
    """
    p = MKT_LAYOUT_DIR / f"{n_machines}_machine_layout.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"MKT 布局文件不存在：{p}（已发布集合为 4–18 中的 12 个机台数，见 "
            f"{MKT_LAYOUT_DIR}）——不得静默回退到其它规模")
    m = np.loadtxt(p, dtype=float)
    return m[1:, 1:] if drop_lu else m

# ── 已发表的逐实例数字（⚠️ 均为 **MKT** 口径，不是原始 MK）──
# HGS：Moon, Lee, Park. *Learning-enabled Flexible Job-shop Scheduling for Scalable Smart
#      Manufacturing.* Journal of Manufacturing Systems 77:356-367, 2024，Table IV 的 HGS 列。
# HA-DQN：Dong, Wan, Zeng. *A heuristic-assisted deep reinforcement learning algorithm for
#      flexible job shop scheduling with transport constraints.* Complex & Intelligent
#      Systems 11:210, 2025，Table 8。
# HF2021 LAHC：Homayouni & Fontes, J. Global Optimization 79(2):463-502, 2021（其车辆数为 2，
#      **与 HGS/HA-DQN 的 v=m 不同口径**——见 progress-log §19.7d 存疑点②）。
# ⚠️ 三列**都**是 MKT 口径。原始 MK 的 BKS 是另一套（MK01=40），**不得混表**。
# ⚠️ 每个方法的**车数设定**（Review Focus #1）：`"m"` = 该实例的机台数；整数 = 固定车数。
# HF2021 的 2 台是**二手转述**（§19.7d 存疑点②原文如此），引用时须保留这个限定词。
# 机读形式存在的理由：`MKT_PUBLISHED` 只有数字，下游画表/画图时若只拿它，
# "HF2021 与 HGS 不同车数"这条事实就会丢掉，两个报告档的车数会被并排当成可比。
MKT_PUBLISHED_AGV: dict[str, "str | int"] = {
    "HGS_JMS2024": "m",
    "HA_DQN_CIS2025": "m",
    "HF2021_LAHC": 2,
}


MKT_PUBLISHED: dict[str, dict[str, float]] = {
    "mk01": {"HGS_JMS2024": 153.0, "HA_DQN_CIS2025": 97.0, "HF2021_LAHC": 187.0},
    "mk02": {"HGS_JMS2024": 104.0, "HA_DQN_CIS2025": 71.0, "HF2021_LAHC": 148.0},
    "mk03": {"HGS_JMS2024": 267.0, "HA_DQN_CIS2025": 235.0, "HF2021_LAHC": 371.0},
    "mk04": {"HGS_JMS2024": 139.0, "HA_DQN_CIS2025": 129.0, "HF2021_LAHC": 225.0},
    "mk05": {"HGS_JMS2024": 374.0, "HA_DQN_CIS2025": 259.0, "HF2021_LAHC": 312.0},
    "mk06": {"HGS_JMS2024": 217.0, "HA_DQN_CIS2025": 158.0, "HF2021_LAHC": 389.5},
    "mk07": {"HGS_JMS2024": 348.0, "HA_DQN_CIS2025": 213.0, "HF2021_LAHC": 291.0},
    "mk08": {"HGS_JMS2024": 812.0, "HA_DQN_CIS2025": 669.0, "HF2021_LAHC": 846.0},
    "mk09": {"HGS_JMS2024": 529.0, "HA_DQN_CIS2025": 502.0, "HF2021_LAHC": 794.0},
    "mk10": {"HGS_JMS2024": 409.0, "HA_DQN_CIS2025": 357.0, "HF2021_LAHC": 712.5},
}


@dataclass(frozen=True)
class MktInstance:
    """MKT 实例 = 原始 MK + 行程时间矩阵 + 车辆数。

    ⚠️ **车辆数不是数据文件里的字段**（§19.7d）：HF2021 用 2 台、HGS 与 HA-DQN 用 v=m。
    **不同车数下的 Cmax 不可比**，故它只能是显式参数、且必须随结果一起报出去。
    ⚠️ P4-B 起矩阵**挂在 `base` 上**（`base.trans_time_full`，含 LU 的全矩阵）——正是它让
    "口径跟随实例"贯通整条训练/评估栈；`trans_time` 是本对象提供的**丢 LU 视图**（P4-A 语义不变）。
    """

    base: Instance
    n_agv: int
    layout_m: int

    @property
    def trans_time(self) -> np.ndarray:
        """机台间矩阵（丢 LU 的 m×m）——`base.trans_time_full[1:, 1:]` 的**视图**（只读用）。"""
        return self.base.trans_time_full[1:, 1:]

    @property
    def name(self) -> str:
        return f"{self.base.source}（MKT, m={self.layout_m}, v={self.n_agv}）"


def load_mkt(name: str, n_agv: int | None = None) -> MktInstance:
    """载入 MKT 实例。`n_agv=None` -> **v = m**（HGS/HA-DQN 的设定）。

    ⚠️ **加工数据一字不改**（有测试逐作业比对）：布局矩阵与车辆数是 MKT 相对 MK 的**全部**增量。
    ⚠️ 布局按**机台数**选（不是按实例名）——故 mk06 与 mk10 共用同一份 15 机布局。
    ⚠️ P4-B 起把实例标成 **matrix 口径**并挂上**含 LU 的全矩阵**（`drop_lu=False`）——
       `AgvSim` 的节点→矩阵下标映射需要第 0 位是 LU 的那个全矩阵。
    """
    base = load_mk(name)                      # 加工数据一字不改（有测试钉住）
    m = base.n_machines
    full = load_mkt_layout(m, drop_lu=False)  # ⚠️ **含 LU**：丢 LU 的 m×m 会被口径对象拒绝
    base = replace(base, transport="matrix", trans_time_full=full)
    return MktInstance(base=base, n_agv=(m if n_agv is None else int(n_agv)), layout_m=m)
