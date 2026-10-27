# -*- coding: utf-8 -*-
"""**按 Shi et al., *Advanced Engineering Informatics* 65 (2025) 103195 的规格生成实例。**

⚠️ **这不是原文的实例** —— 原文用的是**真实航空零部件车间**的数据，**未公开**。
本模块只复现它**描述出来的规格**，两处原文没给、必须我们自己拍的量见下。

══ 原文给的（`docs/citation-cards.md:1626-1631`）══

| 项 | 值 |
|---|---|
| 车间 | 18 台多用途机器 + 1 个 L/U 站 + 6 台 AGV |
| 零件类型 | **7 类**航空零部件；各类工序数 = **8, 8, 8, 8, 9, 6, 7** |
| 工件数 `I` | **10–40 随机** |
| 加工时间 | 每道工序在可用机器上 **2–10** 随机 |
| 实例命名 | `Exp(工件数 / 工序数)`，从 `Exp(7/59)` 到 `Exp(90/801)` |
| **"每道工序的可用机器固定"** | ⚠️ **说固定，但没给是哪几台** |
| **"车间布局与两两位置间的转运路线固定且已知"** | ⚠️ **没给**（布局另见 `Fig. 4`，本仓用网格近似） |

══ 实例名的口径（本仓实测确认）══

`Exp(a/b)` 的 `b` **含每件一道「回站」虚拟工序**（原文：「为每个工件的工序序列追加一个
虚拟工序，关联完成后退回 L/U 站的物流任务」）。故：

    真实工序数 = b − I        ⟹ `Exp(90/801)` = 90 工件 / **711 道真工序**

对照 7 类零件的工序均值 `(8+8+8+8+9+6+7)/7 = 7.71`：全部实例的 `b/I − 1` 落在
**6.20–7.90**，均值 ≈ 7.7 ⟹ **口径对得上**。

══ ⚠️ 本模块里 `assumed` 的两处（论文里必须写明）══

1. **工序的可用机器集合** —— 原文只说"固定"。本模块取**每道工序随机 2–4 台**
   （`n_alts`），且**按零件类型固定**（同类型的工件工艺路线相同）。
2. **加工时间的分布** —— 原文说"2–10 随机"。本模块取**整数** `U{2,…,10}`。

**⟹ 产物只能叫"按 Shi 2025 的规模生成"，不得与本仓的 MK 系列或原文读数并列。**
"""
from __future__ import annotations

import numpy as np

from .instances import Instance

N_MACHINES_SHI: int = 18
"""原文验证环境的机器数（含 L/U 抽象成的虚拟机器时是 18+1，此处只数真实机器）。"""

N_AGV_SHI: int = 6
"""原文验证环境的 AGV 数。"""

PART_TYPES: tuple[str, ...] = (
    "compressor_disk", "turbine_disk", "fan_disk", "integral_drum",
    "compressor_front_journal", "compressor_rear_journal", "long_shaft",
)
"""原文点名的 7 类航空零部件（见其 §5 实例来源）。"""

PART_OPS: tuple[int, ...] = (8, 8, 8, 8, 9, 6, 7)
"""各类零件的**真实工序数**（与 `PART_TYPES` 一一对应）。"""

PT_LO, PT_HI = 2, 10
"""单道工序的加工时间范围（整数，闭区间）—— 原文 *"在 2–10 之间随机"*。"""

N_ALTS_LO, N_ALTS_HI = 2, 4
"""⚠️ `assumed`：每道工序的**可用机器台数**范围。原文只说"可用机器固定"，没给台数。"""


ALTS_POW_SEED = 2600
"""**幂律候选数**的独立子流子种子（2026-10-16 加）—— 见 `make_shi_routings(alts_pow=…)`。

⚠️ **不得并进主 `rng`**：主流的取数次序一变，后面所有抽样整体移位（同 `ALPHA_MIX_SEED` 那条纪律）。
"""


def alts_weights(hi: int, pow_: float) -> np.ndarray:
    """`k = 1..hi` 的**幂律权重** `k^(-pow_)`（未归一化）。

    ⚠️ **2026-10-17：本函数只服务于 `alts_dist="power"` 的**消融**路径。
    复现默认已改为 `alts_dist="uniform"`（`SHI_ALTS_DIST`），**不再经过这里**。**

    ⚠️ **动机段已删** —— 它曾引 *"Each machine **group** is fixed"* 作为「专用工序只在一两台机上
    能做」的依据，而**原文全文没有这句**（见 `SHI_ALTS_POW` 的复核段）。
    原文对候选机台数**只说了「按零件类型定」，没给分布**。

    `pow_ = 0` ⟹ 均匀；`pow_` 越大越往 `k = 1` 偏。
    """
    k = np.arange(1, hi + 1, dtype=float)
    return k ** (-float(pow_))


SHI_ALTS_POW = -2.0
"""**复现层**的候选机台数**幂律指数**（`make_shi_routings(alts_pow=…)`）。

⚠️ **它是复现口径的一部分** —— `pooled` / `eval_instance` **默认**传它（`named` 的默认仍是 `0.0`
⟹ 直接调 `named` 的老路径**逐位不变**）。

**取值 −2.0 的依据 —— ⚠️ 2026-10-17 复核后**认定不成立**：**

**✗ 旧依据（已删）**：本仓曾引 *"Each machine **group** is fixed"* 当作「能力是机台的属性」之据。
**全文检索 `machine group` 零命中 —— 原文没有这句**，它是把下面两句**揉成一句**、
**并引入了原文没有的 "group" 概念**的自造引文：

```
① "the available machines for each operation is determined based on the
    corresponding component type"                      ← 候选机台按**类型**定
② "The overall shop layout and transfer routes between two locations are
    **fixed and known**, as shown in Fig. 4"            ← **布局**固定（另一件事）
```

**✗ 另一条旧依据（§4.40 已撤）**：按「对上原文 43% 杠杆」挑的 —— 那个 43% 是
**方法与方法的差**（RN vs 展示的那条 CDR），被误当成**族内散度**的靶。

**⟹ 原文对候选机台数的**分布一字未给**（§5.1 只说「按零件类型定」）。**
**⟹ 复现默认改为 `alts_dist = "uniform"`（假设最少的一档），见 `SHI_ALTS_DIST`。**
**本常量保留，仅供**消融**用（`alts_dist="power"` 时才生效）。**
"""


SHI_ALTS_DIST = "uniform"
"""**复现层**的候选机台数**分布族**（2026-10-17 起为**默认**）—— `make_shi_routings(alts_dist=…)`。

取值 `"uniform"`（`k ~ U{1..alts_max}`，均值 5.5）的**唯一依据是「原文没给分布」**：
§5.1 只写 *"the available machines for each operation is determined based on the
corresponding component type"*，**分布一字未给** ⟹ 取**假设最少**的那一档。

⚠️ **2026-10-17 之前是 `alts_pow = −2` 的幂律（往 10 偏）** —— 那条依据已被复核推翻
（引自造引文 + 拿方法与方法的差当族内散度的靶），见 `SHI_ALTS_POW`。
⚠️ **`pooled` / `eval_instance` 默认传本值**；`named` 的默认仍是 `"power"` + `alts_pow=0.0`
⟹ 直接调 `named` 的老路径**逐位不变**。
⚠️ 本项生效时 `alts_pow` **被忽略**（均匀权重不走 `alts_weights`）。
"""

POOL_STRIDE = 7
"""训练池的第 `i` 个实例取 `named(..., seed=POOL_STRIDE * i)`。

⚠️ **必须与 `len(hits)` 互素**（`7` 与 212 互素）⟹ `7i ≡ 7i' (mod 212)` 只在 `i ≡ i'` 时成立
⟹ **池内 `i` 不同必不撞实例**（要求 `i < len(hits)` = 212）。见 `named` 的 `hits`。
"""

PT_REDRAW_SEED = 2000
"""**加工时间重抽**的种子基 —— 池的第 `i` 个用 `PT_REDRAW_SEED + i`。

原文把两件事分在两栏（`docs/citation-cards.md` §4）：

| 项 | 原文归到 | 随实例变吗 |
|---|---|---|
| 每道工序的**可用机器** | **车间配置** | ❌ 固定 |
| 每道工序的**加工时间 2–10** | **生产参数** | ✅ **随机** |

⟹ 本模块照此拆开：**机台集合恒由 `seed=0` 那张表定**（车间常量），
**加工时间由 `pt_seed` 重抽**（生产参数）。见 `make_shi_routings(pt_seed=...)`。
"""


ALPHA_MIX_SEED = 4200
"""**重尾 `α_ijk`** 的**独立子流** —— 见 `make_shi_routings(narrow_n=...)`。

⚠️ **不得并进主 `rng`**：主流的取数次序一变，后面所有抽样整体移位（同 `MACHINE_CAP_SEED` 那条纪律）。
"""


PT_PER_JOB_SEED = 2400
"""**逐作业重抽加工时间**的独立子流子种子（2026-10-16 加）—— `[本值, seed, rot_seed]`。

══ 依据（原文记号表逐字）══

```
i, i'    Indices of **jobs**, i ∈ {1, 2, ..., I}
j, j'    Indices of **operations for job i**, j ∈ {1, ..., J_i, J_i+1}
pt_ijk   Processing time of the **j-th operation of job i** on machine k
α_ijk    1 if machine k can execute **O_ij**, and 0 otherwise
I        Number of manufacturing **jobs**
```

**`pt_ijk` 的下标 `i` 是**作业**（记 "of job i"）；而 §5.1 对 `α` 另有一句
*"determined based on the corresponding **component type**"* ⟹ **两者规则不同**：**

| | 由什么定 | 同类作业之间 |
|---|---|---|
| **`α_ijk`**（可用机台） | **零件类型**（§5.1 逐字） | **相同** |
| **`pt_ijk`**（加工时间） | **作业**（记号表 "of job i" + §5.1 *"for each operation of **each job**"*） | **不同** |

**⚠️ 2026-10-16 前本仓把**两者都按类型**照抄** ⟹ **16 个作业只有 7 种、逐位完全相同**
（实测：重复次数 `[4,3,3,3,1,1,1]`）⟹ 「先派哪一个」根本没有区别，是**结构性失真**。

⚠️ **独立子流**：不得并进主 `rng`（那会把 `types` 的抽样整体移位）—— 同 `ALPHA_MIX_SEED` 那条纪律。
本值同时进 `seed`（作业→类型指派）与 `rot_seed`（池的第 i 个实例），故**两者任一变化都会换一批加工时间**。
"""


ALTS_DISTS: tuple[str, ...] = ("uniform", "power", "normal")
"""候选机台数 `k` 的**分布族**（2026-10-16 加，2026-10-17 加 `"uniform"` 并**设为复现默认**）。

- **`"uniform"`**：`k ~ U{1..alts_max}`（**均值 5.5**）—— **复现层的默认**。
- `"power"`：`p(k) ∝ k^(-alts_pow)` —— 见 `alts_weights`。
- `"normal"`：`k ~ Normal(alts_max/2, alts_max/4)` 取整、截到 `[1, alts_max]`。

⚠️ **它与 `alts_pow = 0` 不是一回事**：`alts_pow = 0` 会落到**旧的「均匀 2–4 台」**那条路
（见 `make_shi_routings` 的三分支），**不是**均匀 1–10。想均匀 1–10 必须用本项。

══ 为什么取 `"uniform"`（2026-10-17）══

**原文对「候选机台数」的分布一个字都没给。** 它只写（§5.1 逐字）：

> each job randomly corresponds to one type of component that can be processed in the shop, and
> **the number of operations for each job and the available machines for each operation is
> determined based on the corresponding component type**. … the parameter `α_ijk` reflecting
> the machines' **ability** to process each operation can be determined.

⚠️ **此前本文件与 `alts_weights` 引过一句 *"Each machine group is fixed"* ——
本仓全文检索 `machine group` **零命中：原文没有这句话。** 那是揉合了**
「候选机台按类型定」**与**「车间布局固定且已知」**两句的**自造引文**，
**并引入了原文没有的 "group" 概念 ⟹ 以它为据的 `alts_pow = −2` 依据不成立。**

**⟹ 既然原文没给分布，就取**假设最少**的那一档：均匀。**
"""

PT_DISTS: tuple[str, ...] = ("uniform", "normal")
"""加工时间的**分布族**（2026-10-16 加）。

- `"uniform"`（默认）：整数 `U{PT_LO..PT_HI}` —— 原文 *"在 2–10 之间随机"* 的自然读法。
- `"normal"`：`N(6, 2)` 取整、截到 `[2, 10]`（**均值同样是 6，只改方差**）。

⚠️ **原文没说分布** —— `"uniform"` 是自然读法，`"normal"` 是**敏感度测试**。
⚠️ 两种的**均值相同（6）** ⟹ 机台总工时不变，**只改 `SMPT` 能挑出多少差异**。
"""


def draw_pt(rng, pt_dist: str, pt_hi: int = PT_HI, pt_lo: int = PT_LO) -> float:
    """按分布族抽一个加工时间。**两处调用点共用此一处**（`make_shi_routings` /
    `make_shi_instance` 的逐作业重抽）—— 分两处写必然漂。"""
    if pt_dist == "uniform":
        return float(rng.integers(int(pt_lo), int(pt_hi) + 1))
    if pt_dist == "normal":
        mid, sd = 0.5 * (pt_lo + pt_hi), 0.25 * (pt_hi - pt_lo)
        return float(min(pt_hi, max(pt_lo, round(float(rng.normal(mid, sd))))))
    raise ValueError(f"未知的 pt_dist={pt_dist!r}；可选={PT_DISTS}")


def make_shi_routings(seed: int = 0, n_machines: int = N_MACHINES_SHI,
                      n_alts_lo: int = N_ALTS_LO, n_alts_hi: int = N_ALTS_HI,
                      pt_scale: float = 1.0, cluster_size: int | None = None,
                      pt_seed: int | None = None,
                      narrow_n: int = 0, wide_lo: int | None = None,
                      wide_hi: int | None = None,
                      alts_pow: float = 0.0, alts_max: int = 10,
                      alts_dist: str = "power", pt_dist: str = "uniform",
                      pt_hi: int = PT_HI, alts_min: int = 1, pt_lo: int = PT_LO
                      ) -> tuple[tuple[tuple[tuple[int, float], ...], ...], ...]:
    """7 类零件的**工艺路线**（按类型的固定路线）。返回 `(类型, 工序, [(机台, 时长)])`。

    ⚠️ **按零件类型固定**，不按工件 —— 同类型的工件走同一条路线（原文"可用机器固定"）。

    ⚠️ **`pt_seed`（2026-10-14 加）—— 只重抽加工时间，机台集合不动**：

    - `None`（默认）⟹ **原行为，逐位不变**（加工时间也由 `seed` 定）。
    - 给定时 ⟹ 机台集合仍由 `seed` 定（**车间常量**），加工时间另起一条流重抽。

    依据：原文把「可用机器」放在**车间配置**栏（固定）、「加工时间 2–10」放在
    **生产参数**栏（随机）—— 见 `PT_REDRAW_SEED`。

    ⚠️ **重抽发生在两条分支之后、`out.append` 之前**，故**默认路径的 `rng` 取数次序一字未动**
    （同 `MACHINE_CAP_SEED` 那条纪律）。
    """
    rng = np.random.default_rng(seed)
    pt_rng = np.random.default_rng(pt_seed) if pt_seed is not None else None
    # ⚠️ **幂律候选数**的独立子流（2026-10-16 加）：`alts_pow ≠ 0` 时才建。
    #    **正 ⟹ 往 1 偏**（专用工序）；**负 ⟹ 往 `alts_max` 偏**（通用工序）。
    _arng = (np.random.default_rng([seed, ALTS_POW_SEED])
             if (alts_pow != 0.0 or alts_dist != "power") else None)
    # ⚠️ **下限**（`alts_min`，2026-10-17 加）：只作用于 `"uniform"` 那一档；
    #    其它档仍从 1 起（`alts_pow` / `"normal"` 的语义不变）。默认 1 ⟹ 逐位不变。
    _klo = int(alts_min) if alts_dist == "uniform" else 1
    _kvals = np.arange(_klo, int(alts_max) + 1)
    if alts_dist == "uniform":
        _kprob = np.ones(len(_kvals))          # **均匀 {1..alts_max}**（均值 5.5）
    elif alts_dist == "normal":
        _kprob = np.exp(-0.5 * ((_kvals - 0.5 * alts_max) / (0.25 * alts_max)) ** 2)
    elif alts_pow != 0.0:
        _kprob = alts_weights(int(alts_max), alts_pow)
    else:
        _kprob = None
    if _kprob is not None:
        _kprob = _kprob / _kprob.sum()
    out = []
    for n_ops in PART_OPS:
        ops = []
        if _arng is not None and cluster_size is None and not (
                narrow_n and wide_lo is not None):
            # ⚠️ **幂律候选数**（2026-10-16 加，用户提出）：`k ∈ {1..alts_max}`，`p(k) ∝ k^(-alts_pow)`。
            #    **动机**见 `alts_weights`：`α_ijk` 是机台的**能力** ⟹ 专用工序只在一两台机上能做。
            #    ⚠️ **与 `cluster_size` 的区别**：`cluster_size` 把机台**排除出车间**（偷偷改 `K`），
            #    **这一条不排任何机台** —— 18 台仍全部可用，只是**每道工序能用的少**。
            for _ in range(n_ops):
                k = int(_arng.choice(_kvals, p=_kprob))
                ms = np.sort(_arng.choice(n_machines, size=k, replace=False))
                ops.append(tuple((int(m), draw_pt(_arng, pt_dist, pt_hi, pt_lo) * pt_scale)
                                 for m in ms))
        elif cluster_size is None and narrow_n and wide_lo is not None:
            # ⚠️ **重尾 `α_ijk`**（2026-10-15 加）：每类零件随机选 `narrow_n` 道工序**只给 1 台**
            #    （瓶颈工序 ⟹ 规则在那一步无差别），其余给 `(wide_lo, wide_hi)` 台。
            #    **动机**：默认的均匀 2–4 台让 `EAM`/`MINU` 几乎总选同一台 ⟹ 规则之间没有可翻转的差异。
            #    ⚠️ **独立子流**（`ALPHA_MIX_SEED`）—— 不消耗主 `rng`。
            #    ⚠️ `narrow_n = 0`（默认）⟹ **走原路径，逐位不变**。
            mix = np.random.default_rng([seed, ALPHA_MIX_SEED, n_ops])
            nidx = {int(x) for x in mix.choice(n_ops, size=min(narrow_n, n_ops),
                                              replace=False)}
            for o in range(n_ops):
                k = 1 if o in nidx else int(mix.integers(wide_lo, wide_hi + 1))
                ms = np.sort(mix.choice(n_machines, size=k, replace=False))
                ops.append(tuple((int(m),
                                  float(mix.integers(int(pt_lo), int(pt_hi) + 1)) * pt_scale)
                                 for m in ms))
        elif cluster_size is None:
            # ⚠️ **原来的路径，逐字不动**（2026-10-14）：默认必须与改动前**逐位相同**。
            #    ⚠️ **不能**用 `pool = arange(18)` 走下面那条 —— 多一次 `rng.choice` 会
            #    把后续抽样**整体移位**（实测踩过，同 `PA_TIE_SEED` 那条纪律）。
            for _ in range(n_ops):
                k = int(rng.integers(n_alts_lo, n_alts_hi + 1))
                ms = np.sort(rng.choice(n_machines, size=k, replace=False))
                ops.append(tuple((int(m),
                                  float(rng.integers(int(pt_lo), int(pt_hi) + 1)) * pt_scale)
                                 for m in ms))
        else:
            # ⚠️ **路线专属子集**（2026-10-14）：该类型先分到 `cluster_size` 台，
            #    各道工序的候选**只能从这几台里挑**。
            #    ⚠️ **为什么需要它**：默认路径每道工序都从**全部 18 台**里抽，8–9 道工序并起来
            #    覆盖 12–16 台 ⟹ **每类零件几乎能用所有机器，路线之间没有特异性**。
            #    实测：18 台里 **0 台**是「只被 1 条路线用到」的。而真实工艺里同类零件的
            #    多道工序通常落在**相关的**几台上。**原文没给 `α_ijk`，这条结构完全没约束。**
            pool = np.sort(rng.choice(n_machines, size=int(cluster_size), replace=False))
            hi = min(n_alts_hi, len(pool))
            lo = min(n_alts_lo, hi)
            for _ in range(n_ops):
                k = int(rng.integers(lo, hi + 1))
                ms = np.sort(rng.choice(pool, size=k, replace=False))
                ops.append(tuple((int(m),
                                  float(rng.integers(int(pt_lo), int(pt_hi) + 1)) * pt_scale)
                                 for m in ms))
        if pt_rng is not None:
            # ⚠️ **只换时间那一项**：`m` 逐个照抄（机台集合 = 车间常量）。
            #    顺序也照抄 ⟹ 机台候选集合与顺序**逐位不变**。
            ops = [tuple((m, float(pt_rng.integers(int(pt_lo), int(pt_hi) + 1)) * pt_scale)
                         for (m, _t) in op) for op in ops]
        out.append(tuple(ops))
    return tuple(out)


def make_shi_instance(n_jobs: int, seed: int = 0, n_machines: int = N_MACHINES_SHI,
                      rot_seed: int | None = None, n_alts_lo: int = N_ALTS_LO,
                      n_alts_hi: int = N_ALTS_HI, pt_scale: float = 1.0,
                      cluster_size: int | None = None,
                      narrow_n: int = 0, wide_lo: int | None = None,
                      wide_hi: int | None = None,
                      alts_pow: float = 0.0, alts_max: int = 10,
                      alts_dist: str = "power", pt_dist: str = "uniform",
                      pt_hi: int = PT_HI, alts_min: int = 1, pt_lo: int = PT_LO) -> Instance:
    """按规格生成一个实例。`n_jobs` = 工件数 `I`。

    `seed` ⟹ **工件→类型的指派**（原文「每个工件随机对应一类可加工零部件」）。
    `rot_seed` ⟹ **加工时间重抽**（原文「每道工序…加工时间在 2–10 之间随机」）。

    ⚠️ **2026-10-16 起：加工时间**逐作业**重抽**（`PT_PER_JOB_SEED` 子流），
    **机台序列仍照抄**由 `seed` 定的那张类型路线表。
    原文记号的 `i` 是**作业**（`pt_ijk` = "of job i"），而 `α_ijk` 的取值由**类型**定（§5.1）
    ⟹ **两者规则不同**。修之前同类作业逐位完全相同（16 个作业只有 7 种），是**结构性失真**。
    ⚠️ **因此本次修复后，凡基于旧实例的读数**一律作废**（含 `eval_instance(idx=199)`）。**

    ⚠️ **2026-10-14 前 `rot_seed` 是死的** —— 签名里有，函数体里恒传字面量 `0`
    （同 `named()` 那个 `seed`，见 `docs/v3/open-question-nhdrl.md` §4.7.2）。
    """
    routings = make_shi_routings(0, n_machines, n_alts_lo, n_alts_hi, pt_scale,
                                 cluster_size, pt_seed=rot_seed,
                                 narrow_n=narrow_n, wide_lo=wide_lo, wide_hi=wide_hi,
                                 alts_pow=alts_pow, alts_max=alts_max,
                                 alts_dist=alts_dist, pt_dist=pt_dist, pt_hi=pt_hi,
                                 alts_min=alts_min, pt_lo=pt_lo)
    rng = np.random.default_rng(seed)
    types = rng.integers(0, len(PART_TYPES), size=n_jobs)
    # ⚠️ **加工时间逐作业重抽**（2026-10-16 修，见 `PT_PER_JOB_SEED`）：
    #    **机台序列照抄**（`α_ijk` 由零件类型定，§5.1），**加工时间各作业独立**（`pt_ijk` 下标是 job）。
    #    ⚠️ 修之前是 `jobs = [list(routings[t]) for t in types]` —— **同类作业逐位完全相同**。
    pt_rng = np.random.default_rng(
        [PT_PER_JOB_SEED, int(seed), 0 if rot_seed is None else int(rot_seed)])
    jobs = []
    for t in types:
        route = routings[int(t)]
        jobs.append([tuple((m, draw_pt(pt_rng, pt_dist, pt_hi, pt_lo) * pt_scale)
                           for m, _ in op) for op in route])
    return Instance(n_jobs=n_jobs, n_machines=n_machines, jobs=jobs,
                    source=f"shi2025-spec(I={n_jobs},seed={seed},rot={rot_seed})")


OUT_N_SEEDS = 2000
"""`named` 搜索满足 `Σ工序数 == n − jobs` 的 seed 的上界。**`POOL_STRIDE` 必须与它互素。**"""


def named(n: int, jobs: int, seed: int = 0, n_alts_lo: int = N_ALTS_LO,
          n_alts_hi: int = N_ALTS_HI, pt_scale: float = 1.0,
          cluster_size: int | None = None,
          rot_seed: int | None = None,
          narrow_n: int = 0, wide_lo: int | None = None,
          wide_hi: int | None = None,
          alts_pow: float = 0.0, alts_max: int = 10,
          alts_dist: str = "power", pt_dist: str = "uniform",
          pt_hi: int = PT_HI, alts_min: int = 1, pt_lo: int = PT_LO) -> Instance:
    """按原文的命名档生成：`jobs` 个工件，**总工序数对齐 `Exp(jobs/n)` 的 `n − jobs`**。

    实现：反复换 `seed` 直到 `Σ工序数 == n − jobs`（或最接近）。原文的 `b` 含虚拟工序，
    故真实工序数是 `n − jobs`。找不到精确值时就取最接近的，并**由调用方核对**。
    """
    want = n - jobs
    # ⚠️⚠️ **2026-10-14 修真 bug**：原来是「**从 `seed` 起顺序搜，碰到第一个就停**」——
    #   而满足 `tot == want` 的 seed 在 0–2000 里只有几十个，**从任何 `seed ≤ 14` 出发
    #   都会先撞到 `seed=14`** ⟹ **`named(seed=0..14)` 全返回同一个实例**。
    #
    #   **实测**：`named(140,16,seed=0/1/2/3/4)` 的实例指纹**全部相同**（`ea5715173ebd`）。
    #   **后果**：§37.47 的「跨实例训练」（`named(..., seed=7*i)`，i=0/1/2）
    #   **三个「实例」是同一个** ⟹ 「跨实例无效」那条结论**不成立**。
    #
    #   **修法**：先把**全部**满足的 seed 收齐，再按 `seed % len(hits)` 取。
    #   ⚠️ **`seed=0` 仍取 `hits[0]`（= 14）** ⟹ **既有读数逐位不变**，只有 `seed > 0` 变。
    hits = [t for t in range(OUT_N_SEEDS)
            if sum(PART_OPS[int(x)] for x in
                   np.random.default_rng(t).integers(0, len(PART_TYPES), size=jobs)) == want]
    if hits:
        return make_shi_instance(jobs, seed=hits[seed % len(hits)],
                                 n_alts_lo=n_alts_lo, n_alts_hi=n_alts_hi,
                                 pt_scale=pt_scale, cluster_size=cluster_size,
                                 rot_seed=rot_seed, narrow_n=narrow_n,
                                 wide_lo=wide_lo, wide_hi=wide_hi,
                                 alts_pow=alts_pow, alts_max=alts_max,
                                 alts_dist=alts_dist, pt_dist=pt_dist, pt_hi=pt_hi,
                                 alts_min=alts_min, pt_lo=pt_lo)
    # 找不到精确值时取最接近的（原行为）
    best, best_gap = None, None
    for t in range(OUT_N_SEEDS):
        inst = make_shi_instance(jobs, seed=t, n_alts_lo=n_alts_lo,
                                 n_alts_hi=n_alts_hi, pt_scale=pt_scale,
                                 cluster_size=cluster_size, rot_seed=rot_seed,
                                 narrow_n=narrow_n, wide_lo=wide_lo,
                                 wide_hi=wide_hi,
                                 alts_pow=alts_pow, alts_max=alts_max,
                                 alts_dist=alts_dist, pt_dist=pt_dist, pt_hi=pt_hi,
                                 alts_min=alts_min, pt_lo=pt_lo)
        gap = abs(sum(len(o) for o in inst.jobs) - want)
        if best_gap is None or gap < best_gap:
            best, best_gap = inst, gap
    return best


def pooled(n: int, jobs: int, i: int, **kw) -> Instance:
    """**训练池的第 `i` 个实例** —— 原文口径：类型指派**与**加工时间都随实例变。

    `seed = POOL_STRIDE * i`（类型指派）· `rot_seed = PT_REDRAW_SEED + i`（加工时间）。
    ⚠️ `POOL_STRIDE` 与 `OUT_N_SEEDS` 互素 ⟹ **`i` 不同必不撞实例**（`i < OUT_N_SEEDS`）。

    ⚠️ **默认带复现口径**（`alts_pow = SHI_ALTS_POW`）—— 调用方给了就以调用方为准。
    直接调 `named`（不传 `alts_dist`）仍走旧路 ⟹ 老路径**逐位不变**。
    """
    kw.setdefault("alts_dist", SHI_ALTS_DIST)
    return named(n, jobs, seed=POOL_STRIDE * i, rot_seed=PT_REDRAW_SEED + i, **kw)


EVAL_IDX = 199
"""**评测实例的池下标 —— 固定**，不随 `n_inst` 变。

⚠️ **必须是固定的**：若取 `seed = POOL_STRIDE * n_inst`，则**换池子大小评测实例就跟着换**
（`20` 与 `100` 两次跑的读数就不可比了）。钉在 199 之后，「加池子」这个旋钮是**单变量**。

⚠️ 上限：`n_inst ≤ EVAL_IDX`。`199 < 212 = len(hits)` ⟹ 与池内任何 `i ≤ 199` 都不撞
（`7·199 ≡ 7i (mod 212)` ⟹ `199 ≡ i (mod 212)` ⟹ 无解）。
"""


def eval_instance(n: int, jobs: int, n_inst: int, idx: int = EVAL_IDX,
                  **kw) -> Instance:
    """**评测实例** —— ⚠️ **保证不在 `pooled(..., i<n_inst)` 的池里**。

    种子由**固定的** `idx` 定（默认 `EVAL_IDX`），故**换 `n_inst` 不会换评测实例**。
    `n_inst` 只用来做越界检查。

    ⚠️ **这是修 train-on-test**：此前评测用池里的实例 0，而轮换 `si = ep % n_inst`
    让实例 0 进了训练池（每 `n_inst` 个 epoch 一次）⟹ **在训练过的实例上评测**。

    ⚠️ **`idx`（2026-10-15 加）**：换**另一个**评测实例，用来查「读数是不是这一个实例的
    特例」。**默认 `EVAL_IDX` ⟹ 既有读数逐位不变。** 约束：`idx` 必须 `< n_inst`
    **或** `≥ n_inst`？—— 池是 `pooled(..., i<n_inst)`，种子 = `POOL_STRIDE*i`
    ⟹ **只要 `idx ≥ n_inst` 就不在池里**。故越界检查改为 `n_inst > idx`。

    ⚠️ **默认带复现口径**（`alts_pow = SHI_ALTS_POW`，同 `pooled`）—— 调用方给了就以调用方为准。
    """
    if n_inst > idx:
        raise ValueError(f"n_inst 必须 ≤ 评测实例下标 {idx}，实得 {n_inst}")
    kw.setdefault("alts_dist", SHI_ALTS_DIST)
    return named(n, jobs, seed=POOL_STRIDE * idx,
                 rot_seed=PT_REDRAW_SEED + idx, **kw)
