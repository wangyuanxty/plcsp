# -*- coding: utf-8 -*-
"""TF/RDD 交期生成（⑧ 重设计）——**外生、逐作业、有跨度**的纯数据口径。

口径（用文献承认的三件套术语，不自造）
--------------------------------------
- **总工时（total work content, TWK）**：`W_j` = 作业 j 各工序**最短候选工时**之和；
- **交期松紧因子（tardiness factor, TF）**：`τ`——整批交期相对负荷下界的松紧；
- **交期跨度（due-date range, RDD）**：`R`——同批内各作业交期的相对展开幅度。

生成式（本项目按此实现）::

    LB   = max( Σ_j W_j / m ,  max_j W_j )        # 负荷下界，纯实例数据
    ρ_j  = W_j 升序排名归一化到 [0, 1]             # 排名比例；n_jobs == 1 时取 0.5
    d_j  = LB · τ · ( 1 + R · (2·ρ_j − 1) )

**为什么不用 `M_ref`**（旧口径 `d_j = τ·M_ref` 的死因，两条都要写进论文）:

1. **内生**——旧口径把交期锚在**自己的**参考调度 makespan 上。换车队规模/车速/通道宽
   就换 `M_ref`（实测 mk01：n_agv=3 → 103.42，n_agv=1 → 109.95），交期随我们的配置漂；
   交期应是问题的**外生**属性，不该随求解配置变。本模块的入口签名里**没有 cfg**。
2. **基准太弱导致目标恒 0**——`τ=0.90` 只对着**参考策略**标定，而训练后策略改进约
   12–14%，一次性把 TWT 清零（实测 makespan 89.5 < `d_j = 93.08` ⟹ TWT ≡ 0 是恒等式）。
   故标定规则里加"**改进后也不许退化**"这一条约束（见 `plcsp/m16_due_calib.py`）。

`LB` 是纯实例数据：`ΣW_j / m` 是机器产能下界、`max_j W_j` 是单件串行下界，故
`LB ≤ 任何可行调度的 makespan`（`test_workload_lower_bound_is_a_lower_bound` 钉住）。

书目
----
TWK 类交期在本项目已有引证（完整题录，核对自 `docs/citation-cards.md`）：
**《Real-time scheduling for production-logistics collaborative environment using multi-agent
deep reinforcement learning》**，Advanced Engineering Informatics（Elsevier），Vol. 65, 2025,
Article 103216，DOI 10.1016/j.aei.2025.103216（其式 3 为含物流因素的 Total Work Content 形式）。
**TF/RDD 参数组合 (τ, R) 的完整书目待核**——本模块不作书目裁定，也不得据此编造出处。
"""
from __future__ import annotations

from pathlib import Path

from .instances import Instance

# ══ 标定表：逐实例标定的 (τ, R) ══
# 由 `plcsp/m16_due_calib.py` 在 (τ, R) 网格上按"参考误期率 + 改进后误期率"规则选出
# （规则见该脚本 docstring），**不是手抄的魔数**：`plcsp/tests/test_due_calib.py` 与
# 脚本输出对拍。新增实例必须重跑标定脚本，否则 `due_dates_for` 显式报错（Review Focus #3）。
TF_RDD: dict[str, tuple[float, float]] = {
    "mk01": (2.50, 0.0),
    "mk02": (2.65, 0.3),
    "mk03": (3.40, 0.5),
    "mk04": (3.85, 0.0),
    "mk05": (1.55, 0.6),
    "mk06": (5.60, 0.2),
    "mk07": (1.90, 0.6),
    "mk08": (2.50, 0.8),
    "mk09": (2.45, 0.8),
    "mk10": (2.95, 0.9),
}


def total_work_content(inst: Instance) -> list[float]:
    """`W_j` = 作业 j 各工序**最短候选工时**之和。

    **纯实例数据**，与任何调度/配置无关（这正是它比 `τ·M_ref` 强的地方）。取最短候选
    与参考调度的工序选择口径一致（`des.reference_run` 也是每工序取最短候选）。
    """
    return [sum(min(t for _m, t in op) for op in job) for job in inst.jobs]


def workload_lower_bound(inst: Instance) -> float:
    """`LB = max(Σ_j W_j / m, max_j W_j)`——任何可行调度 makespan 的下界。

    - `Σ_j W_j / m`：每台机都要承担全部工时，机器数 m 给出产能下界；
    - `max_j W_j`：单个作业只能串行加工，其总工时给出单件下界。

    两者都只读实例数据。取 max 是因为两个下界各自都可能更紧，下界必须同时满足。
    """
    w = total_work_content(inst)
    return max(sum(w) / inst.n_machines, max(w))


def tf_rdd_due_dates(inst: Instance, tau: float, due_range: float) -> dict[int, float]:
    """逐作业交期 `d_j = LB · τ · (1 + R·(2ρ_j − 1))`。

    `ρ_j` 用 `W_j` 的**升序排名**归一化：工时长的作业交期晚（ρ 大），工时短的早。
    排名含平局时按作业号升序（`sorted` 稳定 + 显式作业号键），故结果**可复现**。
    `n_jobs == 1` 时排名无意义，取 `ρ = 0.5` ⟹ `d_0 = LB·τ`（R 不起作用）。

    ⚠️ `tau` / `due_range` 由调用方给出（`due_dates_for` 负责查表或覆盖）。
    """
    w = total_work_content(inst)
    lb = workload_lower_bound(inst)
    n = inst.n_jobs
    if n == 1:
        rho = [0.5]
    else:
        order = sorted(range(n), key=lambda j: (w[j], j))
        pos = {j: i for i, j in enumerate(order)}
        rho = [pos[j] / (n - 1) for j in range(n)]
    return {j: lb * tau * (1.0 + due_range * (2.0 * rho[j] - 1.0)) for j in range(n)}


def _instance_name(inst: Instance) -> str | None:
    """从 `Instance.source` 取出标定表键（如 `Mk01.fjs` → `mk01`）；取不出返回 None。

    ⚠️ 用文件**名**而非内容指纹：标定表本来就是逐 **实例名** 的（mk01–mk10），
    同名不同内容不在本项目的使用范围内。`gen_random` 的 source 是描述串，必不命中。
    """
    src = (inst.source or "").strip()
    if not src:
        return None
    return Path(src).stem.lower()


def due_dates_for(inst: Instance, tau: float | None = None,
                  due_range: float | None = None) -> dict[int, float]:
    """交期入口：**显式传入 ≥ 标定表**；表里没有的实例**显式报错**（不静默取默认值）。

    - `tau` / `due_range` 都给 ⟹ 直接用（调用方自担口径，可用于新实例的标定/敏感性扫描）；
    - 任一为 `None` ⟹ 查 `TF_RDD` 补缺；实例不在表里 ⟹ `ValueError`，错误信息指明
      需要跑哪个脚本（Review Focus #3：不得静默用错值、也不得 `KeyError` 崩在深处）。

    签名里**没有 cfg**——交期是外生量，不得随仿真配置变（Review Focus #1）。
    """
    if tau is None or due_range is None:
        key = _instance_name(inst)
        calib = TF_RDD.get(key) if key else None
        if calib is None:
            raise ValueError(
                f"实例 {key or inst.source or '<无 source>'} 未标定交期 (τ, R)——"
                f"TF_RDD 只覆盖 {sorted(TF_RDD)}。请跑 `plcsp/m16_due_calib.py` 标定该实例，"
                f"或显式传入 tau= 与 due_range= 两个参数（覆盖开关）。")
        tau = calib[0] if tau is None else tau
        due_range = calib[1] if due_range is None else due_range
    return tf_rdd_due_dates(inst, tau, due_range)
