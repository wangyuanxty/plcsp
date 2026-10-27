# -*- coding: utf-8 -*-
"""**多目标指标**：Pareto 支配筛选 · spacing (SP) · hypervolume (HV)。

用途：按 **Shi et al., *Advanced Engineering Informatics* 65 (2025) 103195** 的口径复现其
Table 8–12。原文的对比协议（`docs/citation-cards.md:1680`）：

> 每个实例独立运行各方法 **20 次**，用 **Pareto 支配**概念从重复运行解中提取各方法的
> **Pareto 解集**，再比较这些解集在 **HV 与 SP** 上的表现。

══ 约定 ══

- 两个目标**都是越小越好**（`minMS`、`minTC`）。
- **SP**（spacing，Schott 1995）：前沿上各点到**最近邻**的 L1 距离的**标准差**。
  `SP = sqrt( (1/(n−1)) Σ (d̄ − d_i)² )`，`d_i = min_{j≠i} Σ_k |f_k(i) − f_k(j)|`。
  **越小越好，0 = 完全均匀。**
- **HV**（hypervolume，Zitzler & Thiele）：前沿**支配区域**的面积（2 目标），
  相对一个**参考点** `r`（须比前沿上所有点都差）。**越大越好。**

⚠️ **参考点 `r` 的选择**：原文未在卡片里记明确式子。本模块**不设默认**，强制调用方传入
（`ref`），免得悄悄用一个数就出结论。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def pareto_front(points: Iterable[Sequence[float]]) -> np.ndarray:
    """**非支配**子集（两目标、都越小越好）。返回原顺序下的子集，形状 `(m, 2)`。

    一个点被支配 ⟺ 存在另一个点**两个目标都不更差、且至少一个更小**。
    """
    P = np.asarray(list(points), dtype=float).reshape(-1, 2)
    if len(P) == 0:
        return P
    keep = np.ones(len(P), dtype=bool)
    for i in range(len(P)):
        if not keep[i]:
            continue
        for j in range(len(P)):
            if i == j or not keep[j]:
                continue
            if (P[j, 0] <= P[i, 0] and P[j, 1] <= P[i, 1]
                    and (P[j, 0] < P[i, 0] or P[j, 1] < P[i, 1])):
                keep[i] = False
                break
    return P[keep]


def spacing(front: Iterable[Sequence[float]]) -> float:
    """**SP**（越小越好）。`n < 2` 时返回 0.0（单点前沿无间距可言）。"""
    F = np.asarray(list(front), dtype=float).reshape(-1, 2)
    n = len(F)
    if n < 2:
        return 0.0
    d = np.empty(n)
    for i in range(n):
        dif = np.abs(F - F[i]).sum(axis=1)          # L1
        dif[i] = np.inf
        d[i] = dif.min()
    return float(np.sqrt(((d.mean() - d) ** 2).sum() / (n - 1)))


def hypervolume_2d(front: Iterable[Sequence[float]], ref: Sequence[float]) -> float:
    """**HV**（越大越好）。`ref` 必须**两个目标都严格大于**前沿上所有点。

    ⚠️ `ref` 若不成立（有前沿点不劣于它）会**显式报错** —— 静默算会得到一个
    "前沿贡献为 0"的假读数，而那看起来只是 HV 偏小。

    ⚠️ **本仓此前一律用原始量纲（分钟 × 物流成本）⟹ 面积上千。**
    原文那一列**全是 ≤ 1 的小数**（0.8225 / 0.3378 / 0.2642 / 0.5356 …）
    **⟹ 原文的目标一定归一化过、参考点取 `(1, 1)`。** 见 `normalize_minmax`。
    """
    F = np.asarray(list(front), dtype=float).reshape(-1, 2)
    r = np.asarray(ref, dtype=float).reshape(2)
    if len(F) == 0:
        return 0.0
    if not (F[:, 0] < r[0]).all() or not (F[:, 1] < r[1]).all():
        bad = F[(F[:, 0] >= r[0]) | (F[:, 1] >= r[1])]
        raise ValueError(
            f"参考点 {tuple(r)} 不比前沿上所有点都差：{bad.tolist()[:4]} —— "
            "HV 会少算甚至为 0（静默的假读数）。请换一个更差的参考点。")
    # 2 目标：按 f0 升序扫描，累加 (r0 − f0_i) × (r1 − 当前最高 f1)
    S = F[np.argsort(F[:, 0])]
    hv, best_f1 = 0.0, r[1]
    for f0, f1 in S:
        if f1 < best_f1:
            hv += (r[0] - f0) * (best_f1 - f1)
            best_f1 = f1
    return float(hv)


REF_NORM = (1.0 + 1e-9, 1.0 + 1e-9)
"""归一化后的参考点。**名义上是 `(1, 1)`** —— 加 `1e-9` 只是为了让 `hypervolume_2d`
那个「`ref` 必须严格大于所有前沿点」的检查不误伤边界点（归一化后最大值恰为 1）。"""


def normalize_minmax(arrays: Sequence[Iterable[Sequence[float]]]
                     ) -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """**把一组点集的两个目标各自 min–max 归一化到 `[0, 1]`**（2026-10-14 加）。

    ⚠️ **这不是「换个写法」，是另一个量**：

    | | 原始量纲 | **归一化 + `ref=(1,1)`** |
    |---|---|---|
    | HV 的量级 | **上千**（分钟 × 物流成本） | **∈ [0, 1]** |
    | 与原文可比 | ❌ | ✅（原文那列全 ≤ 1） |

    ⚠️ **它还会改排序**：本仓 MS 跨度 ≈ 150、TC 跨度 ≈ 10 ⟹ **不归一时 HV 面积
    几乎全由 MS 决定，TC 只占几个百分点；归一之后两个目标等权。**

    **基准 = 传入的全部点集合起来**（该实例上**所有方法的所有解**）。
    ⚠️ **原文没说基准按谁** —— 本仓取「全部解」，**登记在案。**

    返回 `(归一化后的各点集, (lo, hi))`；`(lo, hi)` 是归一化用的下界/上界（原始单位）。
    """
    P = [np.asarray(list(A), dtype=float).reshape(-1, 2) for A in arrays]
    allp = np.vstack(P) if P else np.zeros((0, 2))
    if len(allp) == 0:
        return P, (np.zeros(2), np.ones(2))
    lo, hi = allp.min(0), allp.max(0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)      # 常数维 ⟹ 全部映到 0
    return [(A - lo) / span for A in P], (lo, hi)
