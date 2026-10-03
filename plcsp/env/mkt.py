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

from pathlib import Path

import numpy as np

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
