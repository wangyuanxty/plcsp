# -*- coding: utf-8 -*-
"""训练/评估环境的**唯一构造入口**（评审 F4：五份拷贝收敛成一份）。

背景：`layout + dm + NormContext` 三件套此前在 `m13_train_a` / `test_end_to_end_a` /
`test_joint_chain` / `runner.__main__` / `test_features` 各有一份**拷贝**，且口径已漂：

- 三处用 `m_ref=100.0` 占位，另两处用真实参考 makespan——同一批特征在不同入口下刻度不同；
- 三处 `sample_layout(...)` **漏传 `aisle_w` / `max_agv_capacity`**（正确写法就在
  `des.rollout()` 里）：`SimConfig(aisle_width=1.0)` 时布局几何仍按 1.5 m 生成，而
  `eff_speed` 按 1.0 降速 ⇒ **距离来自一个布局、速度按另一个布局**，几何与动力学错配且零报错。

放 `algo/` 的理由：`env/` 有层次纪律（不得依赖 `nn/`，见 `des.run_gated` 的契约），而本函数
要同时产出 `NormContext`（`nn.features`）——只有 `algo/` 能同时依赖 `env/` 与 `nn/`。
"""
from __future__ import annotations

import numpy as np

from ..env.constraints import ConstraintConfig
from ..env.corridors import build_corridor_graph, dock_distance_matrix
from ..env.des import SimConfig, reference_makespan
from ..env.instances import Instance
from ..env.layout import Layout, sample_layout
from ..nn.features import NormContext, norm_context

# 仅供**纯特征层单测**的占位 m_ref（不是生产口径，见 `build_ctx_for_unit_test`）。
UNIT_TEST_M_REF = 100.0


def build_layout_and_dm(inst: Instance, cfg: SimConfig,
                        seed_layout: int = 0) -> tuple[Layout, np.ndarray]:
    """采样布局 + 格点距离矩阵——**唯一**调用 `sample_layout` 的生产路径。

    ⚠️ `n_agv` / `max_agv_capacity` / `aisle_w` **三个参数缺一不可**：
    - 漏 `n_agv` → 布局车队 ≠ 仿真车队（`SimWorld._fleet` 会显式报错，还算幸运）；
    - 漏 `max_agv_capacity` / `aisle_w` → **静默错配**（布局几何/载量与 cfg 说的不是一回事）。
    """
    lay = sample_layout(inst.n_machines, seed=seed_layout, aisle_w=cfg.aisle_width,
                        n_agv=cfg.n_agv, max_agv_capacity=cfg.max_agv_capacity)
    return lay, dock_distance_matrix(build_corridor_graph(lay))


def build_setup(inst: Instance, cfg: SimConfig | None = None, seed_layout: int = 0,
                constraints: ConstraintConfig | None = None
                ) -> tuple[Layout, np.ndarray, NormContext]:
    """`(layout, dm, ctx)`——训练/评估/验收的**统一**环境三件套。

    - `m_ref` 取**真实参考 makespan**（`reference_makespan`，有缓存）：spec §5.3.1③ 要求
      归一化用实例静态量，且这个数与交期 `d_j = τ·M_ref` 的 `M_ref` **是同一个**
      （特征归一化与交期同源）；
    - `cfg` / `constraints` 同时进 `norm_context`（评审 F2/F3：归一标度必须与仿真同源）；
    - `seed_layout` 默认 0：`joint_chain_step` 的入口守卫前提（奖励权重 `ReferenceObjectives.of`
      固定用 seed_layout=0 的参考运行，而 `SimWorld._due_map` 用**布局**的 seed 取 M_ref
      ——两者不同源则目标口径与权重口径静默错位）。
    """
    c = cfg or SimConfig()
    lay, dm = build_layout_and_dm(inst, c, seed_layout)
    ctx = norm_context(inst, lay, m_ref=reference_makespan(inst, c, seed_layout),
                       cfg=c, constraints=constraints)
    return lay, dm, ctx


def build_ctx_for_unit_test(inst: Instance, layout: Layout, m_ref: float = UNIT_TEST_M_REF,
                            cfg: SimConfig | None = None,
                            constraints: ConstraintConfig | None = None) -> NormContext:
    """⚠️ **非生产口径**：`m_ref` 用占位常量（默认 100.0），**不跑参考运行**。

    存在的唯一理由：纯特征层单测（手搓快照、不跑仿真）不该为一次参考运行付墙钟，也不该被
    `des` 的缓存/交期耦合。与 `build_setup` 的差异是**实质**的：`m_ref` 不是真实参考 makespan，
    故一切以 `m_ref` 归一化的维（`due_margin` / `time_progress` / `at_machine` …）**数值与
    生产口径不同**——任何跑仿真/训练/评估的路径都不得用它，否则 spec §5.3.1③ 的
    "归一化用实例静态量（含真实 M_ref）"这句话就是假的。
    """
    return norm_context(inst, layout, m_ref=m_ref, cfg=cfg, constraints=constraints)
