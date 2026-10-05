"""联合链 GRPO（spec §5.3.4，P2 Task 7）——一条链 = 一个完整 episode 的**全部**决策。

    链 = (机台计划序列 a_S[1..n_ops]) ⊕ (派车序列 a_L[1..n_tasks]) ⊕ (路线序列 a_R[1..n_drives])
         ⊕ (维护序列 a_M[1..n_pm_decisions]) ⊕ (充电序列 a_C[1..n_charge_decisions])
    logp(链) = Σ_t log π_S(a_S[t]) + Σ_t log π_L(a_L[t]) + Σ_t log π_R(a_R[t]) + Σ_t log π_M(a_M[t])
                                                                          + Σ_t log π_C(a_C[t])
                                                                          ← 五类都取【求和】

⚠️ **R（路线）是恢复出来的第三个决策，不是新发明**：`route_logits` 当初随 ① 拥堵一并取消
（理由"无拥堵时选远路严格更差"，spec §5.3），但 ① 后来在 `eb1d1da` 恢复、**路线头没跟着
恢复**（`docs/progress-log.md` §27.3、§28）。本模块的 R 分支即那次遗漏的补建：
`AgvSim._drive` 从"恒走最短路"改为"在 k 条候选里问策略"，`roll_chain` 照 S/L 的规格记录
R 决策，链 logp 照常求和。⚠️ **默认 `route_k=1`（关闭）**——关闭档必须**逐位等于今日行为**
（黄金摘要钉死），既有全部读数才继续成立；启用传 `route_k=2`。

后接标准 GRPO：**一个**终端奖励 r → 组内 z 化（`_z`）→ 纯组内 REINFORCE（默认：
`epochs=1, clip_eps=None`）或 PPO 式裁剪（**只在 `epochs>1` 时才有意义**——`epochs=1` 时
`new` 与 `old` 同参数算出 ⟹ `ratio ≈ 1`（1e-5 内，见下），裁剪项近乎恒等、纯空转；
见 `joint_chain_step` 的守卫）。信任域按**决策**施加
（每个决策一个 ratio）：2026-10-04 修复——此前按**整条链**裁剪，而链 logp 是求和（100–460
项）⇒ 一次更新就把全部链推出带外、后续 epoch 梯度恒 0，clip 形同虚设（见 `joint_chain_step`）。

⚠️ **重算路径的编码器前向已批量化**（2026-10-04，性能批次）：`_decision_logp_terms` 把
**整组 G 条链的全部决策**堆成 `(B, N, F_MAX)`，**一次** `forward_enc`（在线路径不能批——每个
决策依赖上一刻的仿真状态；重算是事后的，可以批）。编码器是耗时主项，实测（MK01、本机、
`torch.set_num_threads(1)`）：单条前向 ≈ 6.0 ms、231 条批成一次 ≈ 402 ms（**~3.4×**），
重算从 9.31 ms/决策降到 2.31 ms/决策，`joint_chain_step` 整步 **≈2×**。
**打分头同批也批量**（2026-10-04 打分头批次）：按 `(kind, n_cand)` 分组（MK01/默认档 4 组、
全头档 7 组），不做 padding，归约长度不变——CUDA 上头与它引出的反向小图曾是整步的 ~54%
（见 `_decision_logp_terms` 与 `joint_chain_step` 的批量说明）。
**代价**：批矩阵乘的分块与单条不同 ⟹ 重算的每一项与逐决策有 ~4e-7 以内的末位漂移
（编码器批 + 打分头批两笔合计；实测 max|Δlogp| = 3.58e-07），`ratio ≡ 1` 因此从
**严格等式**降为"≈1 在 1e-5 内"（逐位钉死的地方已逐个改写，见各函数 docstring 与测试）。
⚠️ **累加次序没有变**：链级 logp 仍是逐决策 float32 顺序累加（`_sequential_float32_sum`）。

⚠️ **O1：优势有两个口径**（`joint_chain_step` 的 `adv_mode`）——`"scalar"`（默认，历史口径：
先加权求和再组内 z 化）与 `"per_objective"`（每目标各自组内 z 化、再按 w 合成）。后者对
**逐目标单位缩放不变** ⇒ w 真正控制各目标的权衡；见 `_advantages` 的 docstring。

spec §5.3.4 五条硬性约定在本模块的落点：
1. **联合链、单一优势** —— `roll_chain` 把**四头**的决策记在**同一条**链上（同一 episode），
   `joint_chain_step` 只算**一个** A（不按头分组、不按层归一化）；
2. **logp 一律取求和** —— `chain_logp` = `Σ logπ_S + Σ logπ_L + Σ logπ_R + Σ logπ_M`（**不除
   决策数**：旧实现 S 取平均、L 取求和，同一个 `ratio=exp(Δ)` 在两边含义不同）。⚠️ 求和只
   定义**链级** logp；裁剪的信任域不看它，按**逐决策**的 `decisions_logp` 算（修复，见上段）；
3. **J=1，预算全给 G** —— `joint_chain_step` 只有 G（J 个扰动取均值留给**评估**，不混进训练）；
4. **优化器 Adam** —— `policy.optim` 惰性创建为 `torch.optim.Adam`（旧的"手写 SGD + 逐元素
   clamp ±1.0"无动量无自适应，已随旧训练器删除）；
5. **L 头接编码器** —— 两头都经 `forward_enc` 拿**同一份** token 嵌入（Task 4 已改）。

⚠️ **梯度口径**（本模块最容易写错的一处）：`roll_chain` 全程 `torch.no_grad()` 采样——决策只记
**上下文**（当时的 token 特征 / 决策特征 / 逐候选特征 / 候选 / 动作），logp 事后由 `chain_logp`
（链级求和）与 `decisions_logp`（逐决策）用**当前**策略**带梯度重算**。仿真栈（SimPy）不参与
反向传播，这是必须的。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .policy import PolicyNet, v_token_index

if TYPE_CHECKING:                       # 仅类型标注：运行时导入会成环（chain_pool → group_rel）
    from .chain_pool import ChainWorkerPool
from ..env.constraints import ConstraintConfig
from ..env.corridors import build_corridor_graph
from ..env.des import CHARGE_CAND_SKIP, SimConfig, SimWorld, build_zone_map
from ..env.instances import Instance
from ..env.layout import Layout
from ..env.reward import (ReferenceObjectives, objective_vector, reward_weights,
                          scalar_reward)
from ..env.t3_budget import T3Budget, check_influenceable, dual_ascent, normalized_activation
from ..nn.features import NormContext
from ..nn.state_emb import build_geom_bias, build_tok


# 训练流步长：第 s 步第 g 条链的仿真扰动种子 = (seed0+s)*SEED_STRIDE + g。`m13_train_a` 的
# "评估流不得与训练流相交"检查复用此常量（单一来源，避免两处漂移）。
SEED_STRIDE = 1000

# R 头两个特征槽的宽度（**唯一真相**，与 `PolicyNet` 的 `n_feat_route*` 形参对应——
# `test_route_choice.py::test_route_feature_widths_match_the_head` 按它核对打分头的输入
# 宽度，两处漂开即报错）。
ROUTE_FEAT_DRIVE = 4      # `route_feat`：起点/终点/负载标志/本车号
ROUTE_FEAT_CAND = 3       # `_route_cand_feat`：长度比/区段数占比/被别的车占着的区段占比

# M（⑫ 维护）头两个特征槽的宽度（**唯一真相**，与 `PolicyNet` 的 `n_feat_pm*` 形参对应——
# `test_maintenance_head.test_pm_feature_widths_match_the_head` 按它核对打分头的输入宽度）。
PM_FEAT_DEC = 3       # `pm_feat`：该机台状态摘要（与候选无关）
PM_FEAT_CAND = 3      # `_pm_cand_feat`：逐候选（两个**动作**的后果，行序 = `des.PM_CANDS`）

# C（⑪ 充电）头两个特征槽的宽度（**唯一真相**，与 `PolicyNet` 的 `n_feat_charge*` 形参对应——
# `test_charge_head.test_charge_feature_widths_match_the_head` 按它核对打分头的输入宽度）。
CHARGE_FEAT_DEC = 3   # `charge_feat`：本车状态摘要（与候选无关）
CHARGE_FEAT_CAND = 3  # `_charge_cand_feat`：逐候选（到该桩的行驶时长 / 占用排队 / 是不是不去充）

# B（⑩ 拼批）头两个特征槽的宽度（**唯一真相**，与 `PolicyNet` 的 `n_feat_batch*` 形参对应——
# `test_batch_head.test_batch_feature_widths_match_the_head` 按它核对打分头的输入宽度）。
BATCH_FEAT_DEC = 6    # `batch_feat`：头件身份 + 该车可取队列的压力（与候选无关）
BATCH_FEAT_CAND = 3   # `_batch_cand_feat`：逐候选（批件数 / 打乱顺序的件数 / 运输距离节省）


def _z(vals: np.ndarray) -> np.ndarray:
    """组内 z 化（**唯一口径**）：(v − mean) / (std + eps)。

    eps 防"全同奖励"的 0 方差（NaN）。旧实现另带 `mean` / `loo` 两个 mode 分支——**全仓零
    调用点**（唯一调用方传的就是 "z"），已删（评审 M-6）。
    """
    return (vals - vals.mean()) / (vals.std() + 1e-9)


# 优势口径（O1）。`scalar` = 历史口径（先加权求和、再组内 z 化，逐位兼容）；`per_objective`
# = 每目标各自组内 z 化、再按 w 合成。白名单在此唯一定义，入口与算子共用（写错名不得静默回退）。
ADV_MODES = ("scalar", "per_objective")


def _check_adv_mode(adv_mode: str) -> None:
    """`adv_mode` 白名单守卫——`joint_chain_step` 入口先查（写错名的代价不该是几分钟的仿真）。"""
    if adv_mode not in ADV_MODES:
        raise ValueError(
            f"adv_mode={adv_mode!r} 未知：只接受 {ADV_MODES}。"
            "'scalar' = 先加权求和再组内 z 化（历史口径，逐位兼容）；"
            "'per_objective' = 每目标各自组内 z 化、再按 w 合成"
            "（O1：消除目标之间的相对尺度，让 w 真正控制权衡）。")


def _advantages(f_objs: np.ndarray, w: tuple[float, float, float],
                adv_mode: str) -> torch.Tensor:
    """组内优势 A（G 条链，float32）——两种口径（O1）。

    - `scalar`：r_g = Σᵢ wᵢ(−f_{g,i})，A = z(r)。⚠️ 这是**今日**的表达式（**公式**未变，
      由 `test_scalar_adv_mode_regression_pin` 的捕获摘要钉死——注意该摘要已于 2026-10-04
      批量重算批次**重捕获**：优势公式一字未动，变的是重算前向的末位）。z 只消掉加权和的
      **总尺度**，消不掉**目标之间的相对尺度**——w 只把**参考点**上的三项贡献拉平
      （wᵢ·fᵢ^ref ≡ 1/Σ），管不住三者在组内的**方差**：谁方差大谁主导 A，问题因此隐蔽。
    - `per_objective`：A = Σᵢ wᵢ·zᵢ(−fᵢ)。每个目标**各自**组内 z 化——z 对逐目标正缩放不变
      （`z(c·v)=z(v)`）——再按 w 合成，故 w 真正决定各目标的相对权重。

    `f_objs` 是 (G, 3) 的**原始目标值**（越小越好），float64（与奖励侧同精度口径）。
    """
    _check_adv_mode(adv_mode)
    if adv_mode == "scalar":
        r = np.asarray([scalar_reward(tuple(o), w) for o in f_objs], dtype=np.float64)
        return torch.tensor(_z(r), dtype=torch.float32)
    z = np.stack([_z(-f_objs[:, i]) for i in range(f_objs.shape[1])], axis=1)   # (G, 3)
    return torch.tensor(z @ np.asarray(w, dtype=np.float64), dtype=torch.float32)


def op_feat(inst: Instance, job: int, oi: int, layout: Layout | None = None) -> list[float]:
    """S 头的**决策特征**（**3 维**，Ruling T4-4）：`[oi/n_ref, 1−oi/n_ref, (len−oi)/n_ref]`。

    前两维是同一进度的正/负极坐标，第三维是剩余工序占比（`n_ref` = 全实例最长作业的工序数）。
    ⚠️ 换型信号**不在这里**——spec §5.3.1② 定死：换型是 `(机台, 作业)` 的**交互量**，
    走 S 头的**候选特征**（`mach_logits_emb` 的 `feat_cand`，见 `_mach_cand_feat`），
    不占决策特征位。brief 旧写的第 4 维 `0.0` 是恒零死维，已按 T4-4 删除。
    ⚠️ `layout` 保留在签名里（计划既定接口）——3 维语义不含布局量。
    """
    n_ref = max(max(len(j) for j in inst.jobs), 1)
    return [oi / n_ref, 1.0 - oi / n_ref, (len(inst.jobs[job]) - oi) / n_ref]


def setup_flag(snap, machine: int, job: int,
               constraints: ConstraintConfig | None = None) -> float:
    """该机台为该作业加工**是否需换型**（1.0 = 需要）——⑤ 进网的取值。

    判据同 `des.MachineSim._process`：与本机**上一件**加工的作业不同才换型（`prev_job`
    为 -1 = 该机还没加工过，同样不换）。这是 S 头候选特征的语义（spec §5.3.1②）。

    ⚠️ **⑤ 关闭时恒 0**（评审 F2）：仿真里换型时长就是 0（`MachineSim._process` 读的是同一个
    开关），特征若不读它，消融档（−生产 / None）会报出"这里要换型"的**假信号**。
    `constraints=None` = 十约束全开（与 `SimWorld` 的 None 语义一致，向后兼容）。
    ⚠️ P4-B Task 2b：任务端点可以是**装卸站**（端点号 = `n_machines`，不是机台）——站上没有
    "上一件加工的作业"，换型问题对它无意义，恒 0。少了这条守卫就是运行期 IndexError。
    """
    if constraints is not None and not constraints.setup_time:
        return 0.0
    if machine >= len(snap.machines):           # 装卸站（格点外端点）：换型语义不适用
        return 0.0
    return 0.0 if snap.machines[machine].prev_job in (-1, job) else 1.0


def task_feat(inst: Instance, snap, frm: int, to: int, oi: int, job: int,
              constraints: ConstraintConfig | None = None) -> list[float]:
    """L 头的**任务特征**（4 维）：起送机台 / 目标机台 / 目标工序序号(归一) / 该机是否需换型。

    ⚠️ 第 4 维需要**作业号**（`setup_flag(snap, to, job)`）——故 `des.run_gated` 的
    `policy_l` 回调契约在 Task 7 扩了 `job` 参数（原 `(snap, frm, to, oi, cand)` 表达不了
    本维；用 `(frm, to, oi)` 反查作业**多义**：实测 MK01 51/112、MK10 538/985 个键歧义）。

    ⚠️ P4-B Task 2b：端点可以是**装卸站**（号 = `n_machines`，作业在站入场、完工回站）——
    归一后站**恰好取 1.0**，而机台仍是 `i/n_m ∈ [0, 1)`。故机台部分的取值与旧口径**逐位不变**
    （不必重训机台→机台那部分），装卸站单独占 1.0 这个点。
    """
    n_ref = max(max(len(j) for j in inst.jobs), 1)
    n_m = max(inst.n_machines, 1)
    return [frm / n_m, to / n_m, oi / n_ref, setup_flag(snap, to, job, constraints)]


def _mach_cand_feat(snap, cand: tuple[int, ...], job: int,
                    constraints: ConstraintConfig | None = None) -> np.ndarray:
    """S 头**逐候选**特征 `(n_cand, 1)`：该机台为本作业加工是否需换型（⑤ / spec §5.3.1②）。"""
    return np.array([[setup_flag(snap, m, job, constraints)] for m in cand], dtype=np.float32)


def _agv_cand_feat(snap, layout: Layout, dm: np.ndarray, ctx: NormContext,
                   cand: tuple[int, ...], frm: int) -> np.ndarray:
    """L 头**逐候选**特征 `(n_cand, 1)`：该车到取货点的**预计行驶时长**（归一化）。

    值 = 最短路距离 ÷（车间包围盒对角 × 该车**有效**速度倍率）——倍率取
    `VehicleState.speed_factor`（= 仿真里 `agv.speed / cfg.agv_speed_mps`），⑩ 关闭时恒 1，
    故该维与**动力学一致**（不是只看几何）。格点最短路是曼哈顿式的，比值可略超 1（~1.3）。
    哨兵 `-1.0` = 该车尚未出车（`node == -1`，位置未知）—— 按 0 号节点换算会伪造
    "已在取货点"（同 `state_emb` V 段坐标维的哨兵约定：真值域外、天然可区分）。
    ⚠️ P4-B Task 2b：`frm` 可以是**装卸站**（端点号 = `n_machines`，不是机台）——作业在站
    入场，AGV 的取货点就是站自己的节点（`LuPad.node`，格点外；仿真侧读
    `entities[frm].pad.dock_node`，`LuStation` 的 `_StationPad` 正是用它）。少了这条守卫
    就是运行期 IndexError（与 `setup_flag` / `task_feat` 已修的同一类）。
    ⚠️ 触发条件（本仓实测）：**必须**先有 AGV 动过（`node >= 0`），才会走到取距离那一行。
    默认档所有投放决策都在 t=0、AGV 尚未出车（全体哨兵），故一直没炸；⑪ C 决策档在 t=0 的
    第一个空闲点就可能开车去充电，投放决策随之后移——**潜在缺陷当场变活**。
    """
    vs = snap.vehicles
    pick = (layout.machines[frm].dock_node if frm < len(layout.machines)
            else layout.lu.node)                    # 装卸站端点（P4-B Task 2b）
    out = np.empty((len(cand), 1), dtype=np.float32)
    for i, a in enumerate(cand):
        node = int(vs[a].node)
        if node < 0:
            out[i, 0] = -1.0
            continue
        dist = float(dm[node, pick])
        sp = max(float(vs[a].speed_factor), 1e-9)
        out[i, 0] = dist / max(ctx.bbox_diag * sp, 1e-9)
    return out


def route_feat(layout: Layout, aid: int, src: int, dst: int, leg: str) -> list[float]:
    """R 头的**行驶特征**（`ROUTE_FEAT_DRIVE` = 4 维）：起点节点 / 终点节点 / 是否负载段 / 本车号。

    ⚠️ 与候选**无关**（broadcast 给 k 条路径）——候选之间的差异只由 `_route_cand_feat` 携带
    （路线候选在 token 序列里没有 token，见 `PolicyNet.route_logits_emb`；`feat_drive` 与
    编码器 token 一样是"谁在开车"的上下文）。
    节点号按**图规模**归一：格点交叉口 `n_nodes` + 装卸站 1 个（`build_corridor_graph`
    的节点数口径）；`leg == "loaded"` = 负载段（空载段 = 0.0）。
    """
    n_nodes = layout.grid.n_nodes + (1 if layout.lu is not None else 0)
    n_agv = max(len(layout.agvs or []), 1)
    return [src / max(n_nodes, 1), dst / max(n_nodes, 1),
            1.0 if leg == "loaded" else 0.0, aid / n_agv]


def _route_cand_feat(snap, zof: dict[int, int], n_zones: int, cands, dm: np.ndarray,
                     aid: int) -> np.ndarray:
    """R 头**逐候选**特征 `(k, ROUTE_FEAT_CAND)`：[长度比, 区段数占比, 被别的车占着的占比]。

    这是 R 机制的**要点**：候选只有按**拥堵暴露度**打分，"选路"才有意义（否则永远选最短路，
    回到砍掉路线头时的处境）。三列各有分工：
    - **长度比** = 该候选几何长度 ÷ **第 0 条**（最短候选）的长度，≥ 1；绕行的代价。
      ⚠️ 矩阵口径下它同时是行驶时长的比例：`AgvSim._drive` 按几何把整段时长分摊到各段，
      故绕行的总时长自动按此比例变长（见该函数的"矩阵口径"段）。
    - **区段数占比** = 该路径经过的**不同**区段数 ÷ 全区段数；跨行/跨列的绕行走得区段更多。
    - **争用占比** = 该路径的区段里**当前被别的 AGV 占着**的个数 ÷ 该路径区段数（0 = 一路畅通）。
      ⚠️ **自己持的区段不算争用**（`holder == aid` 是自己的既有资源，不是冲突）。
      ⚠️ `snap.zone_holder` 为空 = 该快照不带区段表（手搓快照）⟹ 争用列恒 0，**不是哨兵**。
    `cands` 由 `k_shortest_paths` 给出（按长度升序），故 `cands[0]` 恒为最短路。
    """
    holder = snap.zone_holder
    lens = [float(sum(dm[u, v] for u, v in zip(p, p[1:]))) for p in cands]
    base = max(lens[0], 1e-9) if lens else 1.0
    out = np.empty((len(cands), ROUTE_FEAT_CAND), dtype=np.float32)
    for i, p in enumerate(cands):
        zones = list(dict.fromkeys(zof[n] for n in p))            # 去重且保序
        held = sum(1 for z in zones if z < len(holder) and holder[z] not in (-1, aid))
        out[i] = (lens[i] / base,                                  # 0 长度比（越短越好）
                  len(zones) / max(n_zones, 1),                    # 1 区段数占比
                  held / max(len(zones), 1))                       # 2 争用占比
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    """M 头特征用的 [lo, hi] 截断（与 `state_emb._safe` 同口径，两处都在归一化边界上）。"""
    return float(min(max(x, lo), hi))


def pm_feat(snap, mach: int, ctx: NormContext) -> list[float]:
    """M 头的**决策特征**（`PM_FEAT_DEC` = 3 维）：**该机台的状态摘要**（与候选无关）。

    `[pm_used_frac, in_q_fill, out_q_fill]`——"这个保养周期的进度 + 机台手上的压力"：
    - `pm_used_frac` = `pm_clock / pm_interval`：距上次保养已累计的主轴工时占比
      （与 M 段 token 的 `pm_left` 维互补：那是"还剩多少"，这是"已经用掉多少"）；
    - `in_q_fill` / `out_q_fill` = 输入/输出缓冲占用比（无界缓冲时恒 0，`_safe` 同口径）。

    ⚠️ 候选特征里的"当前积压量（现在停机的代价）"与这里的 `in_q_fill` 不同源：那里是
    **工时口径**（`backlog_min / 全实例总工时`），这里是**占位口径**（件数/容量）。
    """
    ms = snap.machines[int(mach)]
    return [_clamp(ms.pm_used_min / max(ctx.pm_interval, 1e-9), 0.0, 1.0),
            _clamp(ms.in_q_len / max(ms.in_cap, 1.0), 0.0, 1.0),
            _clamp(ms.out_q_len / max(ms.out_cap, 1.0), 0.0, 1.0)]


def _pm_cand_feat(snap, mach: int, ctx: NormContext) -> np.ndarray:
    """M 头的**逐候选**特征 `(2, PM_FEAT_CAND)`——行序 = `des.PM_CANDS`（0 现在保养 / 1 不保养）。

    三个槽（设计 §⑫ 的候选特征表）：
    - **0 距停机的主轴余量**（该候选**导致的**停机还有多少主轴工时）：行 0（现在保养）=
      `0.0`（停机就在此刻）；行 1（不保养）= `pm_left / pm_interval`（强制点还在前面）。
      ⚠️ 这一维是"选择非平凡"的**唯一来源**——它在两行上恒不同（未逾期 ⟹ `pm_left > 0`），
      故两个候选的分数不会恒等。它也是设计文档说的 `pm_left`：对"不保养"候选，
      距强制保养正是 `pm_left`；
    - **1 当前积压量** = `backlog_min / total_work_min`（该机台输入队列的**工时**）——
      "现在停机"的代价。两个候选共享同一行值：它是**当下的处境**，不是候选的后果。
    - **2 episode 进度** = `now / m_ref`（与 G 段 `time_progress` 同口径与同截断）。
    """
    ms = snap.machines[int(mach)]
    pm_left = _clamp(1.0 - ms.pm_used_min / max(ctx.pm_interval, 1e-9), 0.0, 1.0)
    backlog = ms.backlog_min / max(ctx.total_work_min, 1e-9)
    progress = _clamp(snap.now / max(ctx.m_ref, 1e-9), 0.0, 4.0)
    return np.array([[0.0, backlog, progress],
                     [pm_left, backlog, progress]], dtype=np.float32)


def charge_feat(snap, aid: int, ctx: NormContext) -> list[float]:
    """C 头的**决策特征**（`CHARGE_FEAT_DEC` = 3 维）：**本车状态摘要**（与候选无关）。

    `[battery_frac, queued_frac, progress]`——"要不要现在充"的三件事实：
    - `battery_frac` = 本车电量占比（[0,1]）。耗尽（`battery <= 0`）是**模型修复**下的硬后果：
      该车不可用直到充到 `battery_high×cap`（`des.AgvSim._depleted`）——故这一维是决策的核心；
    - `queued_frac` = 本车待办任务数 ÷ `ctx.max_queued`：**剩余工作量的代理**。
      ⚠️ 如实声明限度：快照里**没有**"剩余任务各自耗多少电"（任务在 FIFO 队列里只有端点，
      没有里程；每趟耗电还取决于批大小与路径），故用"待办件数"作预期耗电的代理，不是精确值。
    - `progress` = `now / m_ref`（与 G 段 `time_progress` 同口径与同截断）：episode 还剩多少，
      决定"晚充"的风险窗口。

    ⚠️ 设计文档把"本车电量 / 剩余任务预期耗电"列在**候选特征**行——那两量**逐候选不变**
    （是这辆车的处境，不是某根桩的后果），按本仓的槽位约定（`feat_op` / `feat_task` /
    `feat_mach` 都是"与候选无关的决策特征"）归 `feat_dec`；否则 m 行会重复同一个常量，
    既无候选区分度，又让"候选特征"槽的语义含糊。
    """
    v = snap.vehicles[int(aid)]
    return [_clamp(v.battery_frac, 0.0, 1.0),
            _clamp(v.queued / max(ctx.max_queued, 1), 0.0, 1.0),
            _clamp(snap.now / max(ctx.m_ref, 1e-9), 0.0, 4.0)]


def _charge_cand_feat(snap, layout: Layout, dm: np.ndarray, ctx: NormContext,
                      aid: int, cand: tuple[int, ...]) -> np.ndarray:
    """C 头的**逐候选**特征 `(m, CHARGE_FEAT_CAND)`：行序 = `des.charge_cands`（0 = 不去充）。

    三个槽（设计 §⑪ 的候选特征表）：
    - **0 到该桩的行驶时长**（归一化）= 最短路距离 ÷（包围盒对角 × 该车**有效**速度倍率）——
      与 L 头 `_agv_cand_feat` 同一口径与同一理由（⑩ 关闭时倍率恒 1，与动力学一致；
      真时长在 `_leg_min`，但那是 `env/` 的口径对象，特征层按几何近似——如实用几何代理）。
      "不去充"候选取 0.0（它没有桩；由第 2 列区分，不是哨兵）。
    - **1 该桩的占用/排队** = `(occupied + waiting) / capacity`（`Snapshot.chargers` 的活计数）。
      没有它，"选哪根桩"就只剩距离一个维度——忙桩要排队，正是"早充 vs 晚充"的一个代价。
      快照不带桩表（手搓 / `_cold_start`）时恒 0（约定同 `zone_holder` 为空：缺表 = 无信息）。
    - **2 是不是"不去充"**（0/1）——候选集里唯一的**非桩**动作，必须显式标出，否则它只能靠
      前两列取 0 来间接表达（"最近的空闲桩"与"不去充"会撞在同一行上）。

    哨兵 `-1.0` = 该车尚未出车（`node == -1`，位置未知），同 `_agv_cand_feat` 的约定。
    """
    v = snap.vehicles[int(aid)]
    node = int(v.node)
    sp = max(float(v.speed_factor), 1e-9)
    out = np.zeros((len(cand), CHARGE_FEAT_CAND), dtype=np.float32)
    for i, code in enumerate(cand):
        if int(code) == CHARGE_CAND_SKIP:
            out[i, 2] = 1.0
            continue
        ch = layout.chargers[int(code) - 1]
        if node < 0:
            out[i, 0] = -1.0
        else:
            out[i, 0] = float(dm[node, ch.node]) / max(ctx.bbox_diag * sp, 1e-9)
        st = snap.chargers[int(code) - 1] if int(code) - 1 < len(snap.chargers) else None
        if st is not None:
            out[i, 1] = (st.occupied + st.waiting) / max(st.capacity, 1)
    return out


# ── ⑩ 拼批头（B）：候选是**预构造的批次**（动作码 = 追加件数 s），批次在序列里没有 token ──

def _own_queue(snap, aid: int) -> list:
    """决定方那台车**可取的队列**（⑩ 拼批头的候选原料）——两种队列形状**同口径**。

    - `bound`（每车一 Store）⟹ 该车自己的队列（`QueuedTask.veh == aid`）；
    - FIFO（单个共享 Store）⟹ 整条共享队列（`veh == -1`）——任何车都从同一条队列取。

    形状由 `Snapshot.queued_tasks` **自己带**（有 `veh >= 0` 的项 = 绑定档）——不查 cfg、
    不查 SimWorld：决策记录里只有快照，重算路径必须**只凭快照**重建候选（否则 ratio≠1）。
    队列空 ⟹ 不构成决策（`des.batch_cands(0, cap) == (0,)`，仿真侧不回调），故本函数只在
    "该车确有可取任务"时被走到。
    """
    if any(int(t.veh) >= 0 for t in snap.queued_tasks):
        return [t for t in snap.queued_tasks if int(t.veh) == int(aid)]
    return [t for t in snap.queued_tasks if int(t.veh) == -1]


def _same_frm(queue, frm: int) -> list:
    """队列里**同取货点**的任务（**队列序**）——与 `des.AgvSim._same_frm_indices` 同一口径。

    ⚠️ 头件**不在** `queue` 里（它已被仿真取走、拿在手上）；本列表是"可以追加"的那些。
    """
    return [t for t in queue if int(t.frm) == int(frm)]


def batch_feat(inst: Instance, snap, ctx: NormContext, aid: int, job: int,
               frm: int, to: int, oi: int) -> list[float]:
    """B 头的**决策特征**（`BATCH_FEAT_DEC` = 6 维）：**头件身份 + 该车可取队列的压力**。

    `[job/n_jobs, frm/n_m, to/n_m, oi/n_ref, queue_len/max_queued, n_same_frm/capacity]`

    - 前四维 = 头件的身份（这趟**正在服务谁**）——与 L 头 `task_feat` 同族口径（归一化同源）；
    - `queue_len`：本车可取队列的长度 ÷ `ctx.max_queued`——"还有多少活等着"；
    - `n_same_frm / capacity`：队列里同取货点的件数 ÷ 载量——"最多能拼到几件"。
      ⚠️ 后者同时是**候选集的规模**（`des.batch_cands` 的上限）——没有它，策略看不到
      "还有没有得拼"，只能从候选特征反推。
    """
    q = _own_queue(snap, aid)
    n_same = len(_same_frm(q, frm))
    cap = max(int(snap.vehicles[int(aid)].capacity), 1)
    n_m = max(int(inst.n_machines), 1)
    n_j = max(int(inst.n_jobs), 1)
    n_ref = max(max(len(j) for j in inst.jobs), 1)
    return [int(job) / n_j, int(frm) / n_m, int(to) / n_m, int(oi) / n_ref,
            _clamp(len(q) / max(ctx.max_queued, 1), 0.0, 4.0),
            _clamp(n_same / cap, 0.0, 4.0)]


def _dock_of(layout: Layout, endpoint: int) -> int:
    """端点号 → 通道节点（机台 `0..m-1`；= m 时是装卸站）——与 `_agv_cand_feat` 同口径。"""
    m = len(layout.machines)
    return int(layout.machines[endpoint].dock_node) if endpoint < m else int(layout.lu.node)


def _batch_cand_feat(snap, layout: Layout, dm: np.ndarray, ctx: NormContext,
                     aid: int, frm: int, to: int, cands: tuple[int, ...]) -> np.ndarray:
    """B 头**逐候选**特征 `(n_cand, BATCH_FEAT_CAND)`——行序 = `cands`（= `des.batch_cands`）。

    三个槽（设计 §⑩ 的候选特征表，逐个钉）：

    0. **该批件数** = `(1 + s) / capacity`——s = 追加件数（头件已占 1 位）。唯一的"规模"信号；
    1. **打乱顺序**：批内**最后一件**在队列里的位置之前、**没被带走**的件数 ÷ `max_queued`。
       口头口径 = "插入这批会打乱多少个任务的相对顺序"：那些任务本来排在批内某件之前，
       现在要等下一趟。`s=0`（不拼）恒 0；取前缀 ⟹ 越大越打乱；
    2. **运输距离节省**（几何代理，与 L/C 头的候选特征同口径：真时长在 `env` 的 `_leg_min`，
       特征层按几何近似）：`分送 − 拼批`，均按**本模型的卸货序**（首次出现序）折算：

           分送 = Σ_{该批每件} [d(本车, 取货点) + d(取货点, 卸货点)]
           拼批 = d(本车, 取货点) + Σ 各腿（取货点 → 卸₁ → 卸₂ → …）

       归一化除以 `ctx.bbox_diag`。⚠️ 本车节点为 −1（尚未出车）⟹ `d(本车, ·)` 取 **0.0**
       （保守：不把"省一趟回空"算进去）——**不是哨兵**（0 = 无从谈起，见 `_agv_cand_feat`
       对 −1 的另一套约定：那里是"位置未知"的显式哨兵，这里只影响一个加项）。
    """
    q = _own_queue(snap, aid)
    grp_idx = [i for i, t in enumerate(q) if int(t.frm) == int(frm)]
    cap = max(int(snap.vehicles[int(aid)].capacity), 1)
    node = int(snap.vehicles[int(aid)].node)
    frm_node = _dock_of(layout, int(frm))
    d_cur = 0.0 if node < 0 else float(dm[node, frm_node])
    out = np.zeros((len(cands), BATCH_FEAT_CAND), dtype=np.float32)
    for i, code in enumerate(cands):
        take = grp_idx[:int(code)]
        stops: list[int] = []                       # 卸货点序列（首次出现序，= `des.drop_stops`）
        for e in ([int(to)] + [int(q[k].to) for k in take]):
            if e not in stops:
                stops.append(e)
        stop_nodes = [_dock_of(layout, e) for e in stops]
        separate = (1 + len(take)) * d_cur + float(
            dm[frm_node, _dock_of(layout, int(to))]) + sum(
            float(dm[frm_node, _dock_of(layout, int(q[k].to))]) for k in take)
        batched = d_cur + sum(float(dm[a, b]) for a, b in zip(
            [frm_node] + stop_nodes[:-1], stop_nodes))
        out[i, 0] = (1 + len(take)) / cap
        out[i, 1] = (0.0 if not take else (take[-1] + 1 - len(take))) / max(ctx.max_queued, 1)
        out[i, 2] = _clamp((separate - batched) / max(ctx.bbox_diag, 1e-9), -1.0, 4.0)
    return out


@dataclass
class Decision:
    """一个决策点的**全部打分输入**（采样时冻结）——`chain_logp` 据此带梯度重算 logp。

    存的是**当时的**上下文，不是事后重建的：仿真状态在变，用最新快照重算等于把策略输入
    换成另一个状态（决策与 logp 不再对应）。
    ⚠️ `cand_feat` 是**逐候选**特征（T4-3）：S 头放换型标志、L 头放"该车到取货点的预计行驶
    时长"、R 头放路径的"长度比/区段数/争用"。少了它，`chain_logp` 重算的分布与采样时的分布
    **不是同一个**（ratio≠1 的假象）。
    """
    kind: str                        # "S"（选机台）| "L"（派车）| "R"（选路）| "M"（维护时机）
                                     # | "C"（充电：不去充/去哪个桩）
    tok: np.ndarray                  # 当时的四段 token 特征 (N, F_MAX)
    seg: tuple[int, int, int, int]   # 当时的段长 (n_m, n_jobs, n_agv, 1)
    feat: np.ndarray                 # 决策特征 (F_dec,)：S=`op_feat` / L=`task_feat` / R=`route_feat` / M=`pm_feat` / C=`charge_feat`
    cand_feat: np.ndarray            # 逐候选特征 (n_cand, F_cand)
    cand: tuple[int, ...]            # 候选（S: 机台号；L: 车号；R: 0..k-1 的候选序号；M: 动作码 PM_CANDS；
                                     # C: 动作码 `charge_cands`——0 = 不去充，i≥1 = 第 i−1 号桩）
    action: int                      # 所取的动作（∈ cand）
    # 打分时**取 token 嵌入用的下标**（逐候选）。S = `cand`（机台号即序列位置，M 段在最前）；
    # L = `cand` 里的**车号**经 `v_token_index(seg)` 映射后的 V 段下标；R = 本车 V token 下标
    # 广播 k 份（路线候选是**路径**，序列里没有它们的 token，见 `PolicyNet.route_logits_emb`）；
    # M = **该机台**的 M token 下标，两个动作候选共用（候选是动作码 0/1，不是序列位置——
    # 照 S 写 `tok_idx = cand` 会去读 0/1 号机台的 token，与 L 头缺陷同型）。
    # C = **本车** V token 下标，m 个候选共用（候选是动作码 + 桩，桩在序列里没有 token，
    # 同 R；照 S 写 `tok_idx = cand` 会去读 0/1/2 行机台 token——同型缺陷）。
    # ⚠️ **L 曾与 S 共用 `tok_idx = cand`——那是缺陷，不是口径**（2026-10-04 修复）：车号
    # 0..n_agv-1 被当成序列位置，L 头实际索引 M 段机台 token，看不到任何车辆特征
    # （battery / capacity / speed_factor / st_down / zone_wait / 位置）。修前
    # `v_token_index` 只在测试里被调用、生产路径从不使用，故手搓下标的测试查不出来。
    # 改动本字段前先看 `test_policy_heads.test_l_head_reads_v_tokens_on_the_production_path`、
    # `test_maintenance_head.test_pm_head_reads_the_deciding_machines_token_on_the_production_path`
    # 与 `test_charge_head.test_c_head_reads_the_deciding_agvs_token_on_the_production_path`。
    tok_idx: np.ndarray
    # 采样那一刻策略在 `action` 上的 log 概率（无梯度标量）。它是 `decisions_logp` 带梯度
    # 重算值的**同源对照**：裁剪路径的 `old` 直接由它**逐决策**回放（`sampled_decisions_logp`，
    # 省掉一整遍"重算 old"的链前向），一致性由测试钉死。既然取自采样那一刻，它天然不受
    # "采样之后再算 old"这类重排的影响。
    # ⚠️ 2026-10-04（批量重算批次）：重算改成 (B,N,F) 一次批前向，故它与重算值只在 **1e-5 内**
    # 一致、不再逐位相同（旧命题是逐位）。实测 max|Δ| ≈ 3.6e-7；详见 `decisions_logp`。
    logp: float
    # M 决策决定的是**哪台机台**（其余头 None）——决策点逐机台触发，而候选是**动作码**
    # （0/1），机台身份在 `cand` 里表达不了。生产路径写入（`_act` 的 `mach_id`），
    # 守卫测试据此只扰动**该机台**的 M 段 token 行（见 test_maintenance_head）。
    mach: int | None = None
    # R / C 决策是**哪台车**做的（其余头 None）——理由同 `mach`：候选是路径序号 / 动作码 +
    # 桩号，**车身份在 `cand` 里表达不了**，而守卫必须知道该扰动哪一行 V token。
    # ⚠️ 不能拿 `tok_idx` 反推车号：那正是守卫要检验的量（用被检验量去选样本 = 循环论证，
    # §31 缺陷在位时守卫会在"选样本"一步就失败，测不到"扰动决定方 ⟹ logp 必变"这条实质判据）。
    agv: int | None = None
    # R2（`route_zones=True`）时 R 决策的**逐候选区段 token 下标与掩码**：
    # `zone_idx (k, m)` = 第 i 条候选路径途经区段（去重保序）在序列里的 token 下标；
    # `zone_mask (k, m)` = 1.0 标出有效位（右侧补 0）。其余头 / R2 关时 None。
    # 与 `tok_idx` 同源纪律：**采样时冻结**，重算路径照它回放（`_decision_logp_terms`）。
    zone_idx: np.ndarray | None = None
    zone_mask: np.ndarray | None = None
    # ② 几何/度量偏置（`geom_bias=True` 时）——采样那一刻由快照 + 布局算出的 `(N,N)` 矩阵，
    # 重算路径**照它回放**（不能事后重算：实体位置随时间变，重算 = 换了一份策略输入）。
    # 其余档 None。见 `state_emb.build_geom_bias`。
    geom_bias: np.ndarray | None = None


def roll_chain(inst: Instance, layout: Layout, dm: np.ndarray, cfg: SimConfig,
               policy: PolicyNet, seed: int, ctx: NormContext,
               sample: bool = True,
               generator: torch.Generator | None = None,
               constraints: ConstraintConfig | None = None,
               route_k: int = 1, route_zones: bool = False, geom_bias: bool = False,
               pm_head: bool = False,
               charge_head: bool = False, batch_head: bool = False) -> tuple[list[Decision], dict]:
    """跑一条链：仿真里每个派工点同步调策略，记录每个决策的 (token 特征, 决策特征, 候选, 动作)。

    ⚠️ **无梯度**——决策只记上下文（`torch.no_grad()` 下取样），logp 事后由 `chain_logp`
       **带梯度重算**（SimPy 栈不参与反向传播，见模块头）。
    ⚠️ **每个决策存**当时的** token/决策/候选特征——仿真状态在变，用事后最新快照重建等于把
       策略输入换成另一个状态（决策与 logp 不再对应）。存的都是 **CPU numpy**（重算时再搬到
       参数设备，见 `_decision_logp_terms`）——决策日志不占显存。
    ⚠️ **设备跟随策略**（2026-10-04 设备批次）：在线前向经 `forward_enc` 搬到 `policy.device`；
       动作采样流的设备也跟随策略（CUDA 上 `torch.multinomial` 不接受 CPU generator）——故
       `--device cuda` 时整条链在 GPU 上跑，默认档逐位不变（`torch.Generator(device='cpu')`）。

    ⚠️ 布局由调用方给（本函数**不采样布局**）：`layout.layout_seed` 必须与奖励侧参考运行同源
       ——`ReferenceObjectives.of` 固定 seed_layout=0，而特征归一化的 `m_ref` 取自**该布局**的
       参考运行（`build_setup`）。两者不同源则策略看到的刻度与 w 的刻度来自两次参考调度。
       用默认奖励口径时请传 seed=0 的布局（本任务不扩这条口径）。
       （⑧ 交期自 2026-10-03 起是**外生**量，已不参与这条同源约束——见 spec §3.5。）
    ⚠️ L 回调需要**作业号**（任务特征第 4 维 = 该机是否需换型）：`des.run_gated` 的
       `policy_l` 契约在 Task 7 由 `(snap, frm, to, oi, cand)` 扩为 `(snap, job, frm, to, oi, cand)`
       ——用 `(frm, to, oi)` 反查作业**多义**（实测 MK01 51/112、MK10 538/985 个键歧义），
       静默取错作业 = 换型特征错，故走显式契约。
    ⚠️ **`route_k`（R 层开关）**：`1`（默认）= **关闭**，`_drive` 恒走最短路、不记 R 决策、
       不消费采样流 ⟹ **逐位等于今日行为**（黄金摘要钉死，既有读数全靠它）；`≥2` = 每次行驶在
       `k_shortest_paths` 的前 `route_k` 条候选里问 `policy_r`（本任务的设计值 = 2）。
       ⚠️ R 决策**只记在候选 ≥ 2 的行驶上**：装卸站那条**桥**边（及相邻节点的直连）只有一条
       简单路径，"选路"不构成决策——记一条单候选的假决策只会给链 logp 添一个恒为 0 的项。
       ⚠️ `route_k > 1` 要求 ① 拥堵开：① 关时区段机制根本不存在，选远路**严格更差**，
       那正是当年砍掉路线头的理由；给一个"可选的死动作"会污染链 logp，故**显式报错**。
     ⚠️ **`route_zones`（R2 区段 token 开关，2026-10-05）**：默认 `False` = **关**——token
        序列与今日**逐位相同**（`seg` 仍是四元组、行数不变、R 头走原打分式），既有 `route_k=2`
        读数不受影响；`True` = 在序列**末位追加 Z 段**（每区段一个 token，`seg` 变五元组），
        且 R 头**逐候选**读该路径途经区段的 token（均值池化）。区段状态取自快照的
        `zone_holder` / `zone_wait`（后者是 `VehicleState.zone_wait` 的逐区段聚合）。
        ⚠️ `route_zones=True` 要求 `route_k > 1`：没有 R 决策时区段 token 无人消费，
        是**死输入**，故**显式报错**（同 `route_k>1` 要求 ① 拥堵的形态）。
        ⚠️ 打开后链长 / makespan 会变——R 决策的**输入**变了，不是动力学变了。
     ⚠️ **`geom_bias`（② 几何/度量偏置开关，2026-10-05）**：默认 `False` = **关**——编码器
        不做任何额外加减，链路逐位等于今日；`True` = 每个决策点由**当时的快照 + 布局**算出
        `(N,N)` 距离偏置（`state_emb.build_geom_bias`，与 `AgvSim._seg_min` 同一张距离矩阵），
        加在每层注意力分数上（`AttnLayer._attn`）。**只作"输入"用途**：让距离/拥堵可被感知；
        **不得**借它主张跨布局泛化（`method-transfer-candidates.md` §6.3 的裁定）。
        ⚠️ 偏置**冻结在 `Decision` 里**（实体位置逐决策变），重算路径照它回放；偏置档跳过
        CUDA 图快路（见 `LayoutEncoder.forward`）。
     ⚠️ **`pm_head`（⑫ 维护头开关）**：`False`（默认）= **关闭**，`MachineSim` 按规则自动保养
        （`pm_clock >= pm_interval`）、不记 M 决策、不消费采样流 ⟹ **逐位等于今日行为**
        （黄金摘要钉死，既有读数全靠它）；`True` = 一道工序加工完毕、下一件尚未上机时由策略在
        `des.PM_CANDS`（现在保养 / 不保养）里选。⚠️ 逾期（`pm_clock >= pm_interval`）时
        **强制**保养且不产生决策——规则是硬底线，策略只能把保养提前。
        ⚠️ MK01 默认 `pm_interval=120` 在 episode 内从不**逾期**（每机 ~25.5 主轴分钟）——
        机制验证须传显式短间隔档 `SimConfig(pm_interval=…)`（默认参数**不动**，见 §27.1）。
        ⚠️ `pm_head=True` 要求 ⑫ `maintenance` 开：⑫ 关时决策点不存在，给一个死动作只会
        污染链 logp，故**显式报错**（同 `route_k>1` 要求 ① 拥堵的形态）。
     ⚠️ **`charge_head`（⑪ 充电头开关）**：`False`（默认）= **关闭**，`AgvSim._maybe_charge`
        按规则补电、不记 C 决策、不消费采样流 ⟹ **逐位等于今日行为**（黄金摘要钉死）；`True` =
        AGV 在每个空闲待命点由策略在 `des.charge_cands`（{不去充} ∪ {各桩}）里选。
        ⚠️ 配套的**模型修复**（`battery <= 0` ⟹ 该车不可用）**不随本开关开关**——它是
        `AgvSim` 的语义修正，只在 ⑪ 开且电池真的到 0 时生效（默认电池 2–4 kWh 放不空，
        故默认档逐位不变；⑪ 关时电池恒 = cap，**不可能**触发）。
        ⚠️ `charge_head=True` 要求 ⑪ `charging` 开：⑪ 关时决策点不存在，给一个死动作只会
        污染链 logp，故**显式报错**（同上两条的形态）。
     ⚠️ **`batch_head`（⑩ 拼批头开关，2026-10-05）**：`False`（默认）= **关闭**——
        `AgvSim._collect_multi` 按**规则**拼批（全队列同取货点、队列序、取满容量），不记 B
        决策、不消费采样流 ⟹ 逐位等于"拼批头从未存在"（**注意**：与"multi_drop 关"不是同一
        件事——`multi_drop=True` 而 `batch_head=False` 是**规则档**，模型仍是 multi-drop）；
        `True` = AGV 在取货点、头件已取走后，在**预构造的批次候选**（`des.batch_cands`：
        追加件数 s = 0..min(capacity−1, 同取货点任务数)）里由策略选一个。
        ⚠️ 要求 `SimConfig.multi_drop=True`（单卸货点模型下决策点不存在）**且** ⑩ 异构车队开
        （⑩ 关 ⟹ 载量恒 1 ⟹ 候选集恒 `(0,)`，机制是死的）——两条**都在入口显式报错**。
        ⚠️ 候选 < 2（队列里没有同取货点任务）时**不记决策**（`_collect_batch` 直接返回单件）
        ——同 R 头的"候选 < 2 不记决策"：记一条单候选的假决策只会给链 logp 添恒 0 项。
     ⚠️ **动作采样流**（评审 I-3）：`generator=None` ⇒ 按 `torch.Generator().manual_seed(seed)`
        现建——同 seed 同调用序列的动作**逐位相同**；显式传入者自备种子（`joint_chain_step`
        自建一条并透传，组内 G 条链顺序共享）。`sample=False`（argmax）不消费该流。
        ⚠️ R 与 S/L（以及 ⑫ 的 M）共享**同一条**动作流（谁先决策谁先消费）——四头是一条链上的
        联合策略，不各开一条流（那会让"同 seed 可复现"变成分头可复现）。
     ⚠️ **`constraints` 必须透传进 `SimWorld`**（评审 F1）：省略 = 十约束全开，而 spec §6.2 的
        5 组消融正是靠这个形参区分——不透传时 5 组跑出**完全相同**的链且零报错（"约束不重要"
        的假阴性）。`None` = `ConstraintConfig()`（全开，与 `SimWorld` 的 None 语义一致）。
     ⚠️ **`ctx` 应按同一份 constraints 建**（R1/R2）：③ 的 `fail_rate` 维与 ⑧ 的 `due_margin`
        维在约束关掉时静默读 `ctx.constraints`。`joint_chain_step` 入口校验两者同源；直接调用
        本函数（如消融探针）时由调用方负责配对。
    返回 (决策序列, `run_gated` 的 metrics)。
    """
    if route_k < 1:
        raise ValueError(f"route_k={route_k} 非法：1 = 关闭路线头（恒走最短路，逐位等于今日），"
                         "≥2 = 候选路径条数（本任务的设计值 = 2）。")
    if route_zones and route_k < 2:
        raise ValueError(
            f"route_zones=True 需要 route_k > 1（实得 route_k={route_k}）：没有 R 决策时"
            "区段 token 无人消费，是**死输入**（还会加长序列、拖慢每一步）。"
            "请传 route_k≥2 或 route_zones=False。")
    cons = constraints or ConstraintConfig()
    if route_k > 1 and not cons.congestion:
        raise ValueError(
            f"route_k={route_k} 需要 ① 拥堵开启：① 关时区段机制不存在，选远路**严格更差**"
            "（spec §5.3 的原始理由）——给策略一个死动作只会污染链 logp。请开 ① 或传 route_k=1。")
    if pm_head and not cons.maintenance:
        raise ValueError(
            "pm_head=True 需要 ⑫ 维护开启：⑫ 关时 `pm_clock` 根本不累加、保养事件恒 0"
            "（决策点不存在）——给策略一个死动作只会污染链 logp。请开 ⑫ 或传 pm_head=False。")
    if charge_head and not cons.charging:
        raise ValueError(
            "charge_head=True 需要 ⑪ 充电开启：⑪ 关时电池从不增减（恒 = cap）、充电桩机制"
            "根本不存在（决策点不存在）——给策略一个死动作只会污染链 logp。"
            "请开 ⑪ 或传 charge_head=False。")
    if batch_head and not cfg.multi_drop:
        raise ValueError(
            "batch_head=True 需要 SimConfig.multi_drop=True：单卸货点模型下一趟只有一个卸货点"
            "（`_collect` 只拼同 (取货点, 卸货点)），载量上限永远用不上——决策点不存在，"
            "给策略一个死动作只会污染链 logp。请开 multi_drop 或传 batch_head=False。")
    if batch_head and not cons.heterogeneous_fleet:
        raise ValueError(
            "batch_head=True 需要 ⑩ 异构车队开启：⑩ 关时每台车的载量退化为 1"
            "（`AgvSim.capacity = 1`）⟹ 候选集恒为 `(0,)`、一次决策都不会发生——"
            "机制是死的。请开 ⑩ 或传 batch_head=False。")
    decisions: list[Decision] = []
    # ⚠️ 采样流的设备**跟随策略**（2026-10-04 设备批次）：CUDA 上 `torch.multinomial` 不接受
    #    CPU generator（`--device cuda` 时过去会直接报错）。CPU 档 `device='cpu'` 与旧行为
    #    逐位相同（`torch.Generator()` 的默认设备就是 cpu）。
    gen = (generator if generator is not None
           else torch.Generator(device=policy.device.type).manual_seed(seed))
    world = SimWorld(inst, layout, dm, cfg, graph=build_corridor_graph(layout),
                     constraints=constraints)
    # 区段映射（R 头争用特征 / R2 区段 token 的原料）：与 `SimWorld` 内部
    # `build_zone_map(layout, cfg.zone_granularity)` 同一函数、同一入参——两处不可能漂
    # （漂了候选特征会按另一套区段数错标度）。
    zof, n_zones = build_zone_map(layout, cfg.zone_granularity) if route_k > 1 else ({}, 0)
    # R2：区段 token 只在开关打开时进序列（`build_tok` 的 `zof=None` = 四段、逐位等于今日）。
    zof_tok = zof if route_zones else None

    def _act(kind: str, snap, feat: list[float], cand_feat: np.ndarray,
             cand: tuple[int, ...], aid: int | None = None,
             mach_id: int | None = None,
             cand_zones: list[list[int]] | None = None) -> int:
        tok_feat, seg = build_tok(snap, inst, layout, ctx,
                                  zof=zof_tok, n_zones=n_zones)
        tok_feat = np.asarray(tok_feat, dtype=np.float32)
        # ② 几何/度量偏置：`geom_bias=True` 时由此刻的快照 + 布局现算（实体位置逐决策变，
        # 必须**冻结在决策里**，重算路径照它回放）。区段锚只在 R2 打开（序列里有 Z 段）时才有。
        bias = (build_geom_bias(snap, inst, layout, ctx, dm, zof=zof_tok)
                if geom_bias else None)
        # R2：逐候选的区段 token 下标/掩码（候选路径的区段 → 序列下标）。Z 段在**末位**，
        # 故区段 z 的 token 下标 = 前四段行数之和 + z（`sum(seg[:4])`，与 build_tok 同源）。
        zone_idx = zone_mask = None
        if kind == "R" and cand_zones is not None:
            n_base = int(sum(seg[:4]))
            m_max = max(len(zs) for zs in cand_zones)
            zone_idx = np.zeros((len(cand_zones), m_max), dtype=np.int64)
            zone_mask = np.zeros((len(cand_zones), m_max), dtype=np.float32)
            for i, zs in enumerate(cand_zones):
                zone_idx[i, :len(zs)] = [n_base + int(z) for z in zs]
                zone_mask[i, :len(zs)] = 1.0
        with torch.no_grad():
            tok, _ = policy.forward_enc(torch.as_tensor(tok_feat).unsqueeze(0), seg, bias)
            head = {"S": policy.mach_logits_emb, "L": policy.agv_logits_emb,
                    "R": policy.route_logits_emb, "M": policy.pm_logits_emb,
                    "C": policy.charge_logits_emb, "B": policy.batch_logits_emb}[kind]
            if kind == "S":
                # 机台号 = 序列位置（M 段在最前）——S 的候选本身就是 token 下标。
                tok_idx = np.asarray(cand, dtype=np.int64)
            elif kind == "L":
                # ⚠️ L 的候选是**车号**，不是序列位置：必须经 `v_token_index` 映射到 V 段
                # token。修复前它与 S 共用 `tok_idx = cand`（缺陷，2026-10-04 修）——车号
                # 0..n_agv-1 被当成序列位置落在 M 段上，L 头读的是机台 token，对车辆特征
                # （电量/载量/速度/趴窝/位置/区段等待）完全失明。当时 `v_token_index` 只在
                # 测试里被调用、生产路径从不使用，故手搓下标的测试全绿而生产是错的。
                v_idx = v_token_index(seg)
                tok_idx = np.asarray([v_idx[int(a)] for a in cand], dtype=np.int64)
            elif kind == "M":
                # ⚠️ M 的候选是**动作码**（`des.PM_CANDS`：0 现在保养 / 1 不保养），不是序列
                # 位置、也不是机台号：两个候选同属**一台**机台（`mach_id`），故都读该机台的
                # M 段 token。照 S 写 `tok_idx = cand` 会让候选 0/1 去读 0/1 号机台的 token——
                # 维护头给别的机器打分（与 2026-10-04 修的 L 头缺陷同型，progress-log §31）。
                tok_idx = np.full(len(cand), int(mach_id), dtype=np.int64)
            elif kind == "C":
                # ⚠️ C 的候选是**动作码 + 桩号**（`des.charge_cands`：0 = 不去充，i≥1 = 第
                # i−1 号桩），**都不是序列位置**：充电桩在序列里没有 token（序列只有 M/B/V/G/Z
                # 段），"不去充"更不是实体。故 m 个候选共用**本车**的 V token（`aid`）——
                # 与 R 头同型（路线候选也没有 token）。照 S 写 `tok_idx = cand` 会让候选
                # 0/1/2 去读 0/1/2 号**机台**的 token：充电头给机台打分（§31 同型缺陷）。
                tok_idx = np.full(len(cand), v_token_index(seg)[int(aid)], dtype=np.int64)
            elif kind == "B":
                # ⚠️ B 的候选是**动作码**（`des.batch_cands` 的追加件数 s），批次本身
                # （"头件 + 队列里同取货点的前 s 件"）在序列里**没有 token**（任务是 B 段
                # 作业 token 之外的运行期实体）——故 n_cand 个候选共用**本车**的 V token
                # （`aid`），与 C/R 两头同型。照 S 写 `tok_idx = cand` 会让候选 0/1/2 去读
                # 0/1/2 号机台的 token——§31 同型缺陷。
                tok_idx = np.full(len(cand), v_token_index(seg)[int(aid)], dtype=np.int64)
            else:                                   # R
                # 路线候选在序列里没有 token（见 `PolicyNet.route_logits_emb`）：取本车 V token
                # 的下标、k 个候选共用同一份。R2 打开时另给逐候选的区段 token 下标/掩码
                # （`zone_idx` / `zone_mask`）——候选之间的分差自此不再只来自 `cand_feat`。
                tok_idx = np.full(len(cand), v_token_index(seg)[int(aid)], dtype=np.int64)
            extra = ({} if zone_idx is None else
                     {"zone_idx": torch.as_tensor(zone_idx, dtype=torch.long),
                      "zone_mask": torch.as_tensor(zone_mask, dtype=torch.float32)})
            logits = head(tok,
                          torch.as_tensor(feat, dtype=torch.float32).reshape(1, 1, -1),
                          torch.as_tensor(cand_feat, dtype=torch.float32),
                          torch.as_tensor(tok_idx, dtype=torch.long),
                          **extra)
            # 用 log_softmax（而非 softmax）取样：同一遍里就拿到所取动作的 log 概率
            # （`Decision.logp`）——裁剪路径据此省掉一整遍"重算 old"的链前向。
            lp_all = torch.log_softmax(logits.flatten(), -1)
            k = (int(torch.multinomial(lp_all.exp(), 1, generator=gen).item()) if sample
                 else int(lp_all.argmax()))
            lp_k = float(lp_all[k])
        decisions.append(Decision(kind=kind, tok=tok_feat, seg=seg,
                                  feat=np.asarray(feat, dtype=np.float32),
                                  cand_feat=np.asarray(cand_feat, dtype=np.float32),
                                  cand=cand, action=int(cand[k]), tok_idx=tok_idx,
                                  logp=lp_k, mach=mach_id,
                                  agv=(None if aid is None else int(aid)),
                                  zone_idx=zone_idx, zone_mask=zone_mask,
                                  geom_bias=bias))
        return int(cand[k])

    def policy_s(snap, job, oi, cand):
        cand = tuple(int(c) for c in cand)
        return _act("S", snap, op_feat(inst, job, oi, layout),
                    _mach_cand_feat(snap, cand, job, constraints), cand)

    def policy_l(snap, job, frm, to, oi, cand):
        cand = tuple(int(c) for c in cand)
        return _act("L", snap, task_feat(inst, snap, frm, to, oi, job, constraints),
                    _agv_cand_feat(snap, layout, dm, ctx, cand, frm), cand)

    def policy_r(snap, aid, src, dst, leg, cands):
        # 候选 = 路径（节点序列），动作 = 候选**序号**（0 = 最短路）——路径本身没法当动作号。
        # R2：逐候选给出**途经区段**（去重保序，与 `_route_cand_feat` 同一口径）——`_act`
        # 把它们映射成区段 token 下标，R 头据此读每条路径自己的区段。
        cand_zones = (None if zof_tok is None else
                      [list(dict.fromkeys(zof[n] for n in p)) for p in cands])
        return _act("R", snap, route_feat(layout, aid, src, dst, leg),
                    _route_cand_feat(snap, zof, n_zones, cands, dm, aid),
                    tuple(range(len(cands))), aid=int(aid), cand_zones=cand_zones)

    def policy_m(snap, mach, cand):
        # 候选 = **动作码**（`des.PM_CANDS`：0 现在保养 / 1 不保养），两个候选同属该机台。
        # `mach_id` 是**决定保养的那台机台**——`_act` 据此把两个候选的 token 下标都指向它
        # （候选不是序列位置，见 `_act` 的 M 分支与 `PolicyNet.pm_logits_emb`）。
        return _act("M", snap, pm_feat(snap, mach, ctx), _pm_cand_feat(snap, mach, ctx),
                    cand, mach_id=int(mach))

    def policy_c(snap, aid, cand):
        # 候选 = **动作码**（`des.charge_cands`：0 = 不去充，i≥1 = 第 i−1 号桩）。桩在 token
        # 序列里没有 token，故 m 个候选共用**本车**的 V token（`aid`）——`_act` 的 C 分支与
        # `PolicyNet.charge_logits_emb` 都按这条口径。
        return _act("C", snap, charge_feat(snap, aid, ctx),
                    _charge_cand_feat(snap, layout, dm, ctx, aid, cand), cand, aid=int(aid))

    def policy_b(snap, aid, job, frm, to, oi, cand):
        # 候选 = **动作码**（`des.batch_cands` 的追加件数 s；s=0 = 只带头件）。批次本身在
        # token 序列里没有 token（任务是运行期实体），故 n_cand 个候选共用**本车**的 V token
        # （`aid`）——`_act` 的 B 分支与 `PolicyNet.batch_logits_emb` 都按这条口径。
        return _act("B", snap, batch_feat(inst, snap, ctx, aid, job, frm, to, oi),
                    _batch_cand_feat(snap, layout, dm, ctx, aid, frm, to, cand),
                    cand, aid=int(aid))

    metrics = world.run_gated(seed_chain=seed, online_s=True,
                              policy_s=policy_s, policy_l=policy_l,
                              policy_r=None if route_k == 1 else policy_r,
                              policy_m=None if not pm_head else policy_m,
                              policy_c=None if not charge_head else policy_c,
                              policy_b=None if not batch_head else policy_b)
    return decisions, metrics


def _encoder_forward_batched(policy: PolicyNet, tok_all: torch.Tensor, seg,
                             bias_all: torch.Tensor | None,
                             chunk: int) -> torch.Tensor:
    """编码器前向——`chunk > 0` 且批超过 `chunk` 时走**分段 + 梯度检查点**。

    返回 `(B, N, d)` 的 token 嵌入。

    ## ⚠️ 为什么必须配检查点：单分段**不降峰值**

    `torch.cat([f(x₁), f(x₂), …])` 保留的计算图与**一次大前向完全相同**——每段的每一层
    中间量都要留着反向用。所以"把批切成几段分别前向"**不减少任何常驻内存**。

    只有 `torch.utils.checkpoint` 才真降：它**不存段内的中间量**，反向时按段重算。
    峰值从"整批的图"降到"**一段的图**"（另加各段的输入/输出，那很小）：

        峰值 ≈ 整批的图 × (chunk / B)

    **实测**（mk10、全开档、`route_k=2`、G=8，链长约 440 × 8 链 ≈ 3520 条决策）：
    **51 GB → 2.6 GB**。没有这一条，mk10 会被 8 GB 显存挡死。

    ## 数值

    - **梯度检查点是重算同一串算子** ⟹ 逐位相同；
    - **分段**改 GEMM 分块 ⟹ 末位会差（与既有批量化同量级，≤1e-5，见
      `test_per_decision_logp_vector_is_same_source_within_tolerance` 的容差口径）。

    ## 默认

    `chunk = 0` ⟹ **走原路（一次整批前向）⟹ 逐位不变**。本仓铁律：默认关不许动既有读数。
    调用方开它是因为**批大到会被显存挡住**。

    ⚠️ **速度的方向取决于显存是不是瓶颈**（实测 mk10 全开档、G=8、`route_k=2`）：

    | chunk | 峰值显存 | 步时 |
    |---|---|---|
    | 0（关） | **49.58 GB** | 171.4 s |
    | 256 | 1.62 GB | **125.6 s** |
    | 128 | **1.32 GB** | 126.2 s |

    三次的 `r_mean` **完全一致**（只改重算，不动采样）。**"显存省下来"与"更快"同时发生，
    不是巧合**——未分段那一路在 8 GB 卡上跑 49.58 GB 的图，一直在撞分配器的重试与换出。
    **反之，若显存本来够用，分段+检查点会更慢**（反向要按段重算）：**它是换显存的手段，
    不是提速的手段——只在显存真成瓶颈时才两头都赚。**
    """
    B = int(tok_all.shape[0])
    if chunk <= 0 or B <= chunk:
        return policy.forward_enc(tok_all, seg, bias_all)[0]

    def _one(t: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        return policy.forward_enc(t, seg, b)[0]

    parts = [torch.utils.checkpoint.checkpoint(
                _one, tok_all[i:i + chunk],
                None if bias_all is None else bias_all[i:i + chunk],
                use_reentrant=False)
             for i in range(0, B, chunk)]
    return torch.cat(parts, dim=0)


def _decision_logp_terms(decisions: list[Decision], policy: PolicyNet,
                         recompute_chunk: int = 0) -> list[torch.Tensor]:
    """每个决策的 logπ(a)（**标量张量**，带梯度），按 `decisions` 原序——唯一的打分体重算。

    `chain_logp`（求和）与 `decisions_logp`（向量化裁剪用）都从这里取项，**不可能漂开**。
    每个决策用**当时记录的** token/决策/候选特征重建打分（状态已变，不能用最新快照）。
    梯度经 `forward_enc` 同时回到**五**个头与编码器——这是"联合链"的实质（约定 1/5）。
    ⚠️ token 下标用 `d.tok_idx`（五头语义各不同，见 `Decision.tok_idx`）：S = 机台号即序列
    位置；L = 车号经 `v_token_index` 映射的 V 段下标；R = 本车 V token 广播给 k 条候选；
    M = 决定保养的那台机台的 M token，广播给 2 个动作候选；C = 本车 V token 广播给 m 个
    候选（不去充 / 各桩）。
    ⚠️ **L 的 `tok_idx` 修复前错记为 `cand`（缺陷，2026-10-04 修）**：那时这条统一路径
    忠实回放的是"索引 M 段"的错误分布——它不改既有两头任何一位的保证只对 S/R 成立。
    修复后重算路径自动跟随采样下标（`_act` 记录什么就重算什么），无需另一套映射。

    ⚠️ **编码器前向批量化**（2026-10-04，本批的性能改动）：把本列表**全部**决策的 token 堆成
    `(B, N, F_MAX)`，**一次** `forward_enc`（在线路径不能批——每个决策依赖上一刻的仿真状态；
    重算是事后的，可以批）。编码器是耗时主项：本机实测（MK01、`torch.set_num_threads(1)`、
    20 token × d=128 × 8 层）单条前向 ≈ 6.0 ms、231 条批成一次 ≈ 402 ms（每条约 1.74 ms，
    即**吞吐 ~3.4×**）；`chain_logp` 原先每个决策一次前向，就是那 2×链长 次的来源。
    - **形状前提**：全部决策的 token 形状与 `seg` 必须相同。`build_tok` 的行数是实例级常量
      （`n_m + n_jobs + n_agv + 1`，四段长度取自 `NormContext`），故在本项目的调用面上恒成立
      （`test_all_decisions_in_a_step_share_one_token_shape` 钉住）。不同则**显式报错**——
      不静默退回逐条，那会让"批没接上"变成看不见的性能回归，且掩盖上游契约变化。
    - **打分头按 `(kind, n_cand)` 分组批量**（2026-10-04 打分头批次，上一批留的口子）：
      同组一次 `PolicyNet.logits_emb_batch`，**不做 padding**——padding 会改变 `log_softmax`
      的归约长度、把 1e-5 容差撑破；同 `n_cand` 成组则归约长度不变（`log_softmax` 仍只在
      `n_cand` 上做），漂移与编码器批量化同量级。候选数逐决策不同（机台 / 车 / k 条路径 /
      2 个动作码 / 桩数+1），但**组数很少**：MK01/默认档实测 4 组（S 的 1/2/3 与 L 的 3），
      全头开启档 7 组（见 `progress-log.md` §37）。组序 `sorted`（**确定**：同 seed 同结果，
      不依赖 dict 迭代序），组内按原决策序。批的是**打分**这一步；链级 logp 的累加次序不变。
    - **末位漂移**：批前向（含批打分头）与单条前向的矩阵乘分块不同 ⟹ 结果在 1e-5 内一致、
      不是逐位相同。故 `Decision.logp`（采样时逐条算出）与这里的重算不再逐位同源，裁剪的
      `ratio ≡ 1` 随之从**严格等式**降为"≈1 在 1e-5 内"（见 `decisions_logp` 的说明）。
    """
    if not decisions:
        return []
    shapes = {d.tok.shape for d in decisions}
    segs = {d.seg for d in decisions}
    if len(shapes) != 1 or len(segs) != 1:
        raise ValueError(
            f"批量重算要求全部决策的 token 形状与 seg 相同，实得形状 {shapes}、seg {segs}。"
            "形状唯一性由 `build_tok` 的实例级常量保证（行数 = n_m+n_jobs+n_agv+1）；"
            "若上游真的让 seg 随决策变，须先改回逐决策前向并同步批量判据，不得静默错算。")
    # ⚠️ 设备跟随参数（2026-10-04 设备批次）：决策记录是 CPU numpy，这里**一次**搬到
    #    `policy.device`（`--device cuda` 时重算整段在 GPU 上）。头内的 `_to_dev` 兜底其余输入。
    tok_all = torch.stack([torch.as_tensor(d.tok, dtype=torch.float32)
                           for d in decisions]).to(policy.device)
    # ⚠️ ② 几何偏置：整组决策的 `(N,N)` 堆成 `(B,N,N)`——**必须全有或全无**（半带 = 上游
    #    接线错；静默按 None 算会让一部分决策按另一份输入重算，logp 与采样分布不同源）。
    has_bias = [d.geom_bias is not None for d in decisions]
    if any(has_bias) and not all(has_bias):
        raise AssertionError(
            "同一组决策的几何偏置有的有、有的没有——geom_bias 的接线不一致（混组会让"
            "部分决策按错误的输入重算）。")
    bias_all = (None if not all(has_bias) else
                torch.as_tensor(np.stack([d.geom_bias for d in decisions]),
                                dtype=torch.float32, device=policy.device))
    emb = _encoder_forward_batched(policy, tok_all, decisions[0].seg, bias_all,
                                   recompute_chunk)      # (B, N, d)
    # ⚠️ **打分头也批量**（2026-10-04 打分头批次）：按 `(kind, n_cand)` 分组，同组一次算完。
    #    分组而**不做 padding**：padding 会改变 `log_softmax` 的归约长度，把 1e-5 容差撑破；
    #    同 `n_cand` 的决策批在一起则归约长度不变，漂移与编码器批量化同量级（见函数 docstring）。
    #    组序 = `sorted(groups)`（**确定**：同 seed 同结果，不依赖 dict 迭代序）；组内 = 原决策序。
    groups: dict[tuple[str, int], list[int]] = {}
    for i, d in enumerate(decisions):
        groups.setdefault((d.kind, len(d.cand)), []).append(i)
    out: list[torch.Tensor | None] = [None] * len(decisions)
    for kind, n_cand in sorted(groups):
        idx = groups[(kind, n_cand)]
        # 组键含 n_cand ⟹ 同组的 token 下标/特征/候选特征形状一致；不齐时
        # `logits_emb_batch` 的 `emb[rows, tok_idx]` 当场报错，不静默。
        tok_idx = torch.as_tensor(np.stack([decisions[i].tok_idx for i in idx]),
                                  dtype=torch.long, device=policy.device)
        feat = torch.as_tensor(np.stack([decisions[i].feat for i in idx]),
                               dtype=torch.float32, device=policy.device)
        cand = torch.as_tensor(np.stack([decisions[i].cand_feat for i in idx]),
                               dtype=torch.float32, device=policy.device)
        rows = torch.as_tensor(idx, dtype=torch.long, device=policy.device)
        # ⚠️ R2：同组的 R 决策都带（或不带）区段下标——**混组显式报错**（半带 = 上游接线错，
        #    静默按 None 算会让一部分决策的 logp 与采样分布不同源）。组内每条路径的区段数
        #    可以不同：右补 0 到组内最大 m，`zone_mask` 标出有效位（补位不参与均值）。
        has_zones = [decisions[i].zone_idx is not None for i in idx]
        if any(has_zones) and not all(has_zones):
            raise AssertionError(
                f"同一 ({kind}, n_cand) 组里 R 决策的区段下标有的有、有的没有——"
                "R2 的接线不一致（混组会让部分决策按错误的分布重算）。")
        zone_kwargs: dict[str, torch.Tensor] = {}
        if all(has_zones):
            m_max = max(int(decisions[i].zone_idx.shape[1]) for i in idx)
            zi = np.zeros((len(idx), n_cand, m_max), dtype=np.int64)
            zm = np.zeros((len(idx), n_cand, m_max), dtype=np.float32)
            for b, i in enumerate(idx):
                d = decisions[i]
                assert d.zone_mask is not None
                m = int(d.zone_idx.shape[1])
                zi[b, :, :m] = d.zone_idx
                zm[b, :, :m] = d.zone_mask
            zone_kwargs = {"zone_idx": torch.as_tensor(zi, dtype=torch.long, device=policy.device),
                           "zone_mask": torch.as_tensor(zm, dtype=torch.float32,
                                                        device=policy.device)}
        logits = policy.logits_emb_batch(
            kind, emb.index_select(0, rows), tok_idx, feat, cand, **zone_kwargs)  # (B', n_cand)
        lp = torch.log_softmax(logits, dim=-1)
        for b, i in enumerate(idx):
            out[i] = lp[b, decisions[i].cand.index(decisions[i].action)]
    if any(t is None for t in out):     # 分组是 decisions 下标的一个划分，漏项即实现错误
        raise AssertionError("分组批量打分漏了决策——分组与下标不再是一一划分")
    return [t for t in out if t is not None]


def _sequential_float32_sum(terms) -> torch.Tensor:
    """Σ terms——**逐步 float32 顺序累加**（链级 logp 的唯一累加口径）。

    ⚠️ 不许换成 `torch.stack(terms).sum()`：`Tensor.sum()` 的归约次序与逐步相加不同
    （上一批实测末位差 ~1.5e-5），而 `joint_chain_step` 的 `ratio` 判据就是拿链级 logp
    与采样回放比的——归约次序一变，"同源"就只剩近似（见 `chain_logp` / `decisions_logp`
    的 1e-5 口径说明）。
    """
    total = torch.zeros(())
    for term in terms:
        total = total + term
    return total


def decisions_logp(decisions: list[Decision], policy: PolicyNet,
                   recompute_chunk: int = 0) -> torch.Tensor:
    """**逐决策** logπ(a_t) 向量 `(n_decisions,)`，**带梯度**——裁剪的信任域就建在它上面。

    它是 `chain_logp` 去掉最后那步求和：同一个打分体（`_decision_logp_terms`）、同一批
    "当时的"上下文——而现在也是**同一次编码器批前向**（整条链一次，见 `_decision_logp_terms`）。
    裁剪按**决策**而非整条链施加（见 `joint_chain_step` 的信任域段落）。

    ⚠️ **2026-10-04 批量重算后，本函数与 `sampled_decisions_logp` 只在 1e-5 内一致、不再
    逐位相同**（旧断言是 `torch.equal`）：
    - 旧命题：重算与采样回放**逐位**相同（两者都是逐决策单条前向，同一串浮点运算）；
    - 新命题：**≤1e-5 内相同**（重算改成一次 `(B,N,F)` 批前向 + 打分头分组批，
      矩阵乘分块与单条不同，本机实测 `max|Δlogp| = 3.58e-07`）；
    - 理由：编码器前向是耗时主项（本机实测单条 ≈6.0 ms；231 条批前向 ≈402 ms，吞吐 ~3.4×，
      重算整段从 9.31 ms/决策降到 2.31 ms/决策）。裁剪的 `ratio ≡ 1` 随之从严格等式降为
      "≈1 在 1e-5 内"，由 `test_per_decision_logp_vector_is_same_source_within_tolerance` 钉住。
    ⚠️ **不要**反过来用 `... .sum()` 定义 `chain_logp`：归约次序与逐步 float32 相加不同
    （实测末位差 ~1.5e-5），会把这个 1e-5 带撑破（见 `_sequential_float32_sum`）。
    """
    return torch.stack(_decision_logp_terms(decisions, policy, recompute_chunk))


def all_decisions_logp(chains: list[list[Decision]], policy: PolicyNet,
                       recompute_chunk: int = 0) -> torch.Tensor:
    """G 条链的**展平逐决策** logp `(Σn_g,)`，**带梯度**——裁剪路径用它，**一次**编码器前向。

    ⚠️ 与 `decisions_logp` 的区别只有覆盖面：后者一次一条链，本函数一次覆盖**全部 G 条链的
    全部决策**（批大小 = Σn_g）。两条路径的打分体、累加口径、1e-5 容差完全相同。
    """
    return torch.stack(_decision_logp_terms([d for ch in chains for d in ch], policy,
                                            recompute_chunk))


def chain_logp(decisions: list[Decision], policy: PolicyNet,
               recompute_chunk: int = 0) -> torch.Tensor:
    """Σ_t logπ_S(a_t) + Σ_t logπ_L(a_t)——**求和**（spec §5.3.4 约定 2），**带梯度**。

    ⚠️ 累加**必须逐步 float32**（`_sequential_float32_sum`）：与 `sampled_logp` 同序、同 dtype。
    参数未变时两者**在 1e-5 内相同**——不再是逐位相同（旧命题），因为每项的来源已从"逐决策
    单条前向"改成"(B,N,F) 一次批前向（见 `_decision_logp_terms`）；累加**次序**没变，
    变的只是各项的末位（批矩阵乘分块不同）。逐决策的值见 `decisions_logp`；两者共用同一打分体。
    """
    return _sequential_float32_sum(_decision_logp_terms(decisions, policy, recompute_chunk))


def chains_logp(chains: list[list[Decision]], policy: PolicyNet,
                recompute_chunk: int = 0) -> torch.Tensor:
    """G 条链的**链级** logp `(G,)`，**带梯度**——无裁剪训练路径用它，**一次**编码器前向。

    ⚠️ 每条链内仍是 `_sequential_float32_sum`（逐步 float32 顺序累加，与 `chain_logp`
    **同源**：单链调用时逐位相同）；批量化只动"token 嵌入算在哪"，不动"链内怎么累加"。
    """
    lengths = [len(ch) for ch in chains]
    terms = _decision_logp_terms([d for ch in chains for d in ch], policy,
                                 recompute_chunk)
    parts, r = [], 0
    for n in lengths:
        parts.append(_sequential_float32_sum(terms[r:r + n]))
        r += n
    return torch.stack(parts)


def sampled_decisions_logp(decisions: list[Decision]) -> torch.Tensor:
    """**逐决策**回放采样那一刻的 logp：向量 `(n_decisions,)`，无梯度——裁剪路径的 `old`。

    ⚠️ 与 `decisions_logp` **同序、同 dtype**（float32），故"参数未变"时两者在 **1e-5 内**
    相同——`ratio ≈ 1` 是**近似**成立（不再是严格等式：重算已改成一次编码器批前向 +
    打分头分组批，每项有 ~4e-7 以内的末位漂移，实测 `max|Δlogp| = 3.58e-07`，
    见 `_decision_logp_terms` / `decisions_logp`）。别图省事改用
    float64：那会引入 ~1e-5 的假 delta（实测 1.0000114），在裁剪边界上给出无意义的翻转。
    ⚠️ 值在 `roll_chain` 采样时就已算好（`Decision.logp`），本函数没有可省的前向。
    """
    return torch.stack([torch.tensor(d.logp, dtype=torch.float32) for d in decisions])


def sampled_logp(decisions: list[Decision]) -> torch.Tensor:
    """Σ `Decision.logp`——**链级** logp 的采样时刻值（`chain_logp` 的无梯度对照）。

    ⚠️ 裁剪路径已改**逐决策**（`sampled_decisions_logp`），本函数不再是裁剪基准；保留为
    链级口径的定义对照与同源回归判据（测试用）。累加保持逐步 float32（同 `chain_logp`）。
    ⚠️ 与 `chain_logp` 的差在 **1e-5·n** 量级内（各项末位漂移之和），不再逐位相同——
    见 `chain_logp` 的说明。
    """
    total = torch.zeros(())
    for d in decisions:
        total = total + torch.tensor(d.logp, dtype=torch.float32)
    return total


def _chain_mean(obj_flat: torch.Tensor, lengths) -> torch.Tensor:
    """把**展平的逐决策**目标按**链**平均：每条链先对自己的决策取平均，再对链取平均。

    ⚠️ 存在的理由（2026-10-04 修）：逐决策裁剪改造后，裁剪路径把 G 条链展平成 N=Σn_g 个决策
    再 `.mean()`，于是每条链的权重是 `n_g/N`——**正比于链长**；而无裁剪路径的 `obj` 是 `(G,)`
    的链 logp，`.mean()` 给每条链 `1/G`。**两条路径口径不一致**，且链长与 episode 长短相关
    ⟹ 系统性地给长（差）的调度更大的梯度权重。

    链等长时本函数与直接 `.mean()` **逐位相同**（`n_g ≡ n` ⇒ `1/G == n/N`），故不改变
    既有等长情形的行为。
    """
    parts = obj_flat.split([int(n) for n in lengths])
    return sum(p.mean() for p in parts) / len(parts)


def joint_chain_step(policy: PolicyNet, inst: Instance, layout: Layout, dm: np.ndarray,
                     cfg: SimConfig, ctx: NormContext, ref: ReferenceObjectives,
                     seed: int, G: int = 8, lr: float = 1e-3,
                     clip_eps: float | None = None,
                     epochs: int = 1,
                     constraints: ConstraintConfig | None = None,
                     adv_mode: str = "scalar",
                     route_k: int = 1, route_zones: bool = False, geom_bias: bool = False,
                     pm_head: bool = False,
                     charge_head: bool = False,
                     batch_head: bool = False,
                     recompute_chunk: int = 0,
                     parallel: bool = False,
                     pool: "ChainWorkerPool | None" = None,
                     t3_lambda: np.ndarray | None = None,
                     t3_budget: T3Budget | None = None,
                     t3_eta: float = 0.0) -> tuple[float, dict]:
    """一步联合链组训练（spec §5.3.4）。

    G 条链（**J=1**，预算全给 G：约定 3）→ 每条一个终端奖励（三目标加权标量化，
    `reward.scalar_reward`）→ **组内 z 化**（`_z`，一个优势，不按头分层）→ 组内更新。
    优化器 = Adam（`policy.optim` 惰性创建：约定 4；不再手写 SGD + 逐元素 clamp）。

    ⚠️ **`adv_mode`（O1）——优势算在标量奖励上还是逐目标上**：
    - `"scalar"`（默认）= 今日口径：`r = Σᵢ wᵢ(−fᵢ)`，`A = z(r)`。z 只消掉加权和的**总尺度**，
      **不消掉目标之间的相对尺度**——w 只把**参考点**上的三项贡献拉平（wᵢ·fᵢ^ref ≡ 1/Σ），
      管不住三者在组内的**方差**：谁方差大谁主导 A，w 名义上控制权衡、实际控制不了
      （MK01 默认 cfg 下 w≈(0.066, 0.886, 0.048)）。此口径**逐位兼容**（既有读数全靠它，
      由测试的捕获摘要钉死）。
    - `"per_objective"` = `A = Σᵢ wᵢ·zᵢ(−fᵢ)`：每个目标**各自**组内 z 化，再按 w 合成。
      z 对逐目标正缩放不变 ⇒ 换单位（J→0.1J）不改变 A，w 于是真正决定各目标的相对权重。
    未知值在入口**显式报错**（`_check_adv_mode`），不静默回退。诊断（`r_mean` / `r_std`）
    **始终**按 scalar 奖励算（它是"奖励读数"，与优势口径无关）；`A_std` 按所选口径的 A 算。

    ⚠️ **`ref` 而非裸 `w`**（评审 F5）：`w = (1/f^ref)/Σ(1/f^ref)` 只在**同一 (inst, cfg)**
    内有意义（实测 mk01 的 M_ref：n_agv=1/3/5 → 109.95/103.42/97.24；车速 0.5/1.0 →
    103.42/106.38）。旧签名收裸 `w`，P4 扫 `n_agv` 时沿用按旧 cfg 算出的 w 会**静默**落在
    错误的 Pareto 点（`ReferenceObjectives.matches` 当时零调用点）。故此处直接收
    `ReferenceObjectives`：入口校验 `ref.matches(inst, cfg)`（不匹配**显式报错**），
    `w` 由 ref **派生**——不存在"w 与 ref 不同源"的空隙。取 ref 用
    `ReferenceObjectives.of(inst, cfg)`（有缓存）。

    ⚠️ **可复现性**（评审 I-3）：动作采样由 `seed` 派生的 `torch.Generator` 锁定 ⇒ **同 seed
    同调用序列逐位可复现**（跨进程亦然：CPU 的 torch RNG 由 seed 完全确定）；`seed` 同时是
    仿真扰动流基（第 g 条链 `seed*SEED_STRIDE + g`）。旧实现走全局 torch RNG，`--seed`
    锁不住动作——多进程各跑各的，"同 seed 对照 / 种子矩阵"两件事都不成立。

    ⚠️ **裁剪只在 `epochs > 1` 时才有意义**（默认 `epochs=1, clip_eps=None` = 纯组内 REINFORCE，
    即 spec §5.3.4/理论骨架的「纯版本」）：`epochs=1` 时 `new` 与 `old` 都在**同一组参数**
    上算出 ⇒ `ratio ≈ 1` ⇒ `clamp(1, 1±ε) ≈ 1`，裁剪项**近乎恒等**、纯空转（`test_clipped_path_
    ratio_uses_sampling_time_logp` 的 `lr=0` 判据测的就是这个恒等性）。故此处显式**拒收**
    "epochs=1 + clip_eps"这个组合，
    ⚠️ `ratio` 不再是**精确** 1，因为 `old` 是采样时逐决策算的、`new` 是事后 `(B,N,F)` 一次批
    前向算的——末位漂移约 3.6e-07（2026-10-04 两批：编码器批 + 打分头批，
    见 `_decision_logp_terms`）；
    判据容差 1e-5。
    并在 `epochs=1` 时**连 `old` 都不算**——省掉一整遍链前向；`epochs>1` 时 `old` 直接取
    `Decision.logp`（采样那一刻已存，`sampled_decisions_logp` **逐决策**回放），**不额外重算**。

    ⚠️ **重算的批量口径**（2026-10-04）：两条路径都**一次**编码器前向覆盖 **G 条链的全部
    决策**——无裁剪路径用 `chains_logp`（返回 `(G,)`，链内仍逐步 float32 顺序累加），
    裁剪路径用 `all_decisions_logp`（返回展平的 `(Σn_g,)`）。**打分头也按 `(kind, n_cand)`
    分组批量**（追加批次，见 `_decision_logp_terms`）：上一批曾以"头只是两层 MLP、批头要
    padding 且会撑大漂移"为由保持逐决策；CUDA 实测推翻了这条——1920 次头调用（960 在线 +
    960 重算）占整步 35%，且反向要穿 960 张各自独立的头小图（反向 0.568 s ÷ 重算前向
    0.024 s = 24 倍，正常应 ~2 倍）。**不做 padding**：同 `(kind, n_cand)` 成组，`log_softmax`
    的归约长度不变，漂移仍在 1e-5 内（实测见 `docs/progress-log.md` §37）。

    ⚠️ **`route_k`（R 层开关，2026-10-04 恢复路线头）**：默认 `1` = 关闭，链与训练步
    **逐位等于今日**（既有读数全靠它）；传 `2` 启用——每次行驶在 2 条候选路径里由策略选，
    链长与墙钟随之上升（MK01 实测见 `docs/progress-log.md`）。R 决策与 S/L 同进链 logp、
    同吃一条采样流、同受逐决策裁剪——**四头是一条联合链**，不是四套并行策略。

    ⚠️ **`route_zones`（R2 区段 token 开关，2026-10-05）**：默认 `False` = 关闭，链与训练步
    **逐位等于今日**（含 `route_k=2` 的既有读数）；`True`（要求 `route_k>1`）时序列末位追加
    Z 段，R 头逐候选读该路径途经区段的 token。打开后链变长、决策输入变——makespan 变化是
    **预期**的（不是动力学变了）。见 `roll_chain` 的说明。

    ⚠️ **`geom_bias`（② 几何/度量偏置开关，2026-10-05）**：默认 `False` = 关闭，链与训练步
    **逐位等于今日**；`True` = 每个决策点的编码器前向加一份由**当时快照 + 布局**算出的
    `(N,N)` 距离偏置（只作"输入"用途，不得主张跨布局泛化）。偏置随决策冻结、随重算回放。

    ⚠️ **`pm_head`（⑫ 维护头开关，2026-10-04）**：默认 `False` = 规则自动保养，链与训练步
    **逐位等于今日**；`True` 启用——机台在两件之间由策略选 {现在保养, 不保养}（M 决策与
    S/L/R 同进链 logp、同吃采样流、同受逐决策裁剪）。⚠️ 需 ⑫ `maintenance` 开（入口经
    `roll_chain` 显式报错）；默认 MK01 的 `pm_interval=120` 从不逾期 ⟹ 机制验证要传
    `SimConfig(pm_interval=…)` 短间隔档（默认参数不动，见 §27.1）。

    ⚠️ **`charge_head`（⑪ 充电头开关，2026-10-04）**：默认 `False` = 规则补电，链与训练步
    **逐位等于今日**；`True` 启用——AGV 在每个空闲待命点由策略在 {不去充} ∪ {各桩} 里选
    （C 决策与其余四头同进链 logp、同吃采样流、同受逐决策裁剪）。⚠️ 需 ⑪ `charging` 开
    （入口经 `roll_chain` 显式报错）；默认电池 2–4 kWh 在一个 episode 里放不空 ⟹
    "耗尽有后果"要在**小电池档**验证（`AgvSpec(battery_kwh=…)`，默认参数不动，见 §27.1）。

    ⚠️ **`batch_head`（⑩ 拼批头开关，2026-10-05）**：默认 `False` = 规则拼批，链与训练步
    **逐位等于今日**（`multi_drop=False` 时更是"单卸货点模型"，连规则拼批都没有）；
    `True` 启用——AGV 在取货点在**预构造的批次候选**里选（`des.batch_cands` 的追加件数 s），
    B 决策与其余五头同进链 logp、同吃采样流、同受逐决策裁剪。⚠️ 需 `cfg.multi_drop=True`
    且 ⑩ 异构车队开（两条都在 `roll_chain` 入口显式报错）。

    ⚠️ **`t3_lambda` / `t3_budget` / `t3_eta`（T3 拉格朗日，2026-10-05）——三者默认全 `None`
    ⟹ 优势路径一字不改（逐位不变）**。全给时走**对偶上升**（设计 `docs/t3-design.md`）：

        r'_g = r_g − Σᵢ λᵢ·âᵢ,g          # 逐链（G 条链各有自己的 â），**在组内 z 化之前**
        A = z(r')                        # 之后照旧（唯一优势，不按头分层）
        λᵢ ← clip(λᵢ + η·(âᵢ_mean − bᵢ), 0, λ_max)   # 下一步的 λ，回传给持有者

    - **`âᵢ,g = aᵢ,g / aᵢ^ref`**：归一化的**被迫激活量**（口径见 `env/t3_budget`：
      ⑪ 用 `agv_dry_events`、⑫ 用 `pm_events_forced`——罚"策略主动做的动作"是自相矛盾）。
      归一化后才跨约束可比（六条的量纲从 1.75 到 2503）。
    - ⚠️ **λ 的持久状态不在这里**：本函数**只读** `t3_lambda`、把**下一步的 λ** 放进
      `diag["t3_lambda"]` 回传；跨步累积由**持有者**（`runner.run_training`，见
      `env/t3_budget.T3Lagrangian`）负责——这是设计 §3.4 点名最容易写错的一处。
    - ⚠️ **λ 的有效强度会被组内 z 化按 σ(r) 缩放**（设计 §3.3）：z 只消总尺度、不消分量之间的
      相对尺度，故 λ 不是"直接指定罚项占多大权重"。v1 就用 `scalar` 口径（λ 仍是单调旋钮），
      本函数因此**拒收 `adv_mode != "scalar"` + T3 的组合**（v1' 依赖 O1 切成默认，见设计 §3.3）。
    - ⚠️ **可控性守卫**（入口显式报错，不静默）：⑫ 需要维护头、③ 需要维护头**且**
      `cfg.machine_age_failure`、⑪ 需要充电头——没有对应动作时罚它是"罚策略无法控制的事"
      （④⑨ 不纳入的同一条理由）。只要其中一条不在 `t3_budget.keep` 里就不受此限。
    - 诊断：`diag["t3_lambda"]`（下一步 λ，持有者收下）、`diag["t3_lambda_used"]`（本步实际用的）、
      `diag["t3_ahat"]`（组内平均 â）、`diag["t3_pen_mean"]`（平均罚项，标量）。
      `r_mean` / `r_std` **仍是未加罚的奖励读数**（与历史口径同义，不因 T3 而变义）。

    ⚠️ **`parallel`（链级多进程，2026-10-04 并行批次）默认 `False` = 原串行路径，逐位不变。**
    `True` 时把 G 条链交给 `ChainWorkerPool`（spawn）——worker 跑
    `roll_chain` 的整段 episode（仿真 + 在线前向 + 决策记录），主进程只做重算/反向/
    优化器步（实测 MK01、G=8 时在线部分占整步 ~79%，见 `progress-log.md` §36–§38）。
    - **必须显式传 `pool`**（`ChainWorkerPool` 实例）：`parallel=True, pool=None`
      **显式报错**，不静默退回串行（那会让"并行没接上"变成看不见的性能回归）。
    - ⚠️ **采样流口径改变（第九次读数作废）**：串行档 G 条链**共用一条** `torch.Generator`、
      按链序消费；并行档每条链自建 `torch.Generator().manual_seed(seed*SEED_STRIDE + g)`
      （逐链独立且确定 —— 消费次序不再依赖跨进程调度）。**并行档与串行档同 seed 的数值
      不同**，差异来自采样流，不是实现错误。
    - ⚠️ worker 的设备由池的 `worker_device` 定：`"cpu"`（默认）时在线前向浮点路径与采样流
      设备都与串行 CUDA 档不同（CPU 与 CUDA 的 generator 是两条流，§36.9）——跨档读数不可
      逐位互比；`"cuda"` 时 worker 自建 CUDA 上下文与 GPU 副本，在线前向走 CUDA 图快路。
      两种档都不改本函数的语义，只改"在哪算"。重算/反向仍在主进程、仍跟随 `policy.device`。
    - 池常驻复用（每步只提交任务）；清理走 `pool.close()`，`runner.run_training` 的
      `finally` 已接好。

    ⚠️ **信任域按决策施加**（2026-10-04 修复，此前是**链级**裁剪的缺陷）：`logp` 是整条链的
    **求和**（约定 2，链长 100–460），故 `ratio = exp(Δ链logp)` 对单决策的微小漂移极敏感——
    一次 Adam 更新（lr=1e-3）就把每条链推过 `log(1.2)`，于是 `epochs≥2` 时**全部**链出界、
    `∂obj/∂new ≡ 0`：实测 MK01、G=2 时 `clipped_frac=1.0`、`grad_norm=0.0`，多出来的 epoch
    只烧链前向 + 反向。修复 = 把组**展平**：G 条链的全部决策拼成一个长向量（链长不等，
    `A.repeat_interleave(lengths)` 给每个决策它那条链的优势），每个决策各自一个 ratio、各自
    裁剪。`clip_eps` 于是恢复教科书含义（**每个决策**的动作概率比夹在 1±ε），与链长无关。

    ⚠️ **但展平会改变"按什么平均"**（2026-10-04 同日二修）：展平后若直接 `.mean()`，平均的
    对象是 **N=Σn_g 个决策**，每条链权重 `n_g/N`——**正比于链长**；而无裁剪路径的 `obj` 是
    `(G,)`，`.mean()` 给每条链 `1/G`。链长与 episode 长短相关 ⟹ 系统性地给长（差）的调度
    更大的梯度权重，**且两条路径口径不一致**。故裁剪路径改用 `_chain_mean`：每条链内先对
    自己的决策取平均，再对 G 条链取平均。链等长时与直接 `.mean()` 逐位相同。
    ⟹ **两条路径一律按链平均**，`loss` 首轮也重新结构性为 `-mean(A)`（与链长分布无关）。

    ⚠️ **诊断口径随修复改变**：`ratio` = **全部决策**的 ratio 均值；`clipped_frac` = 出界的
    **决策**占比（旧口径是"出界的**链**占比"，1.0 = 全裁 ⇒ 该轮无梯度信号）。

    ⚠️ **`layout.layout_seed` 必须为 0**（守卫在入口）：奖励权重 `w` 的既定来源
    `ReferenceObjectives.of` 固定用 seed_layout=0 的参考运行，而特征归一化的 `m_ref` 取自
    **该布局**的参考运行（`build_setup` 把布局 seed 传进去）——两者不同源则"按参考调度归一化"
    的刻度对不上、且**静默**（Fact F）。如需非 0 布局种子，须先让 `ReferenceObjectives`
    记录其种子再放宽此守卫。（⑧ 交期已外生，**不再**是这条约束的理由——见 spec §3.5。）

    ⚠️ **`constraints` 必须透传**（评审 F1）：`None` = 十约束全开（向后兼容），其余按消融组
    传入——不透传时 spec §6.2 的 5 组消融跑出完全相同的链（假阴性，见 `roll_chain`）。
    ⚠️ **且必须与 `ctx` 同一份**（R2）：③/⑧ 的特征静默读 `ctx.constraints`，入口校验两者
    相等，不同源**显式报错**（否则特征报"不会坏/无交期"而仿真照坏照交期，静默假信号）。

    返回 (组内 r 均值, 诊断 dict：loss / ratio / clipped_frac / grad_norm / r 均值与 std / 优势 std)。
    `ratio` / `clipped_frac` 只在裁剪路径被真算（无裁剪时报 1.0 / 0.0，= 不适用）；两者都是
    **逐决策**口径（`ratio` = 决策级比值的均值，`clipped_frac` = 出界决策占比）。`grad_norm` =
    **末轮**裁剪前的 `‖∂L/∂θ‖`——它非零即"这一轮确有梯度信号"（旧实现 `epochs≥2` 时恒为 0）。
    """
    _check_adv_mode(adv_mode)                   # O1：白名单先于一切（写错名不得先跑几分钟仿真）
    if parallel and pool is None:
        # ⚠️ 硬要求：不许静默退回串行——那样"并行没接上"会变成看不见的性能回归。
        raise ValueError(
            "parallel=True 但 pool=None：并行档必须显式传进程池（ChainWorkerPool 实例）——"
            "不静默退回串行。用法：pool = ChainWorkerPool(policy, n_workers=8)；"
            "joint_chain_step(..., parallel=True, pool=pool)；用完 pool.close()。")
    if layout.layout_seed != 0:
        raise ValueError(
            f"layout_seed={layout.layout_seed} ≠ 0：奖励权重（ReferenceObjectives.of 固定 "
            "seed_layout=0 的参考运行）与特征归一化的 m_ref（build_setup 按**该布局**的 seed "
            "取）会不同源，归一化刻度静默错位。请传 seed=0 的布局；确需非 0 种子，先扩展 "
            "ReferenceObjectives 记录它。")
    if not ref.matches(inst, cfg):
        raise ValueError(
            "ReferenceObjectives 与当前 (inst, cfg) 不同源（cfg 指纹不符）——w 是按**旧** cfg "
            "算出的，'按参考调度归一化'会在这条扫描轴上**静默**不成立（P4 扫 n_agv / 车速 / "
            "通道宽时最易踩），落在哪个 Pareto 点也就错了。请用 "
            "ReferenceObjectives.of(inst, cfg) 取当前 cfg 的参考值。")
    # ⚠️ R2：ctx 与训练的约束必须同源。③/⑧ 的特征静默读 `ctx.constraints`（R1），而仿真读
    #    这里的 `constraints`——不同源时特征会报出动力学里不存在的量（如"不会坏"而仿真照坏），
    #    与 F2 消灭的是同一类静默错配。`None` = 全开（与 `NormContext` 的构造语义一致）。
    cons = constraints or ConstraintConfig()
    if cons != ctx.constraints:
        raise ValueError(
            "ctx 的约束与训练的 constraints 不同源——ctx 由 build_setup/build_ctx_for_unit_test "
            "按另一组约束构造（③/⑧ 的特征静默读 ctx.constraints），仿真却按本组跑：特征会给"
            "策略一个动力学里不存在的信号，且静默。请把**同一份** ConstraintConfig 同时交给 "
            "ctx 的构造与 joint_chain_step（build_training_setup 返回的即该份）。")
    w = reward_weights(ref.as_tuple())          # w 由 ref 派生（评审 F5：单一来源）
    if epochs > 1 and clip_eps is None:
        raise ValueError("epochs>1 必须配 clip_eps：无裁剪时同一批数据重复计算，"
                         "结果与 epochs=1 相同（纯浪费）。")
    if clip_eps is not None and epochs <= 1:
        raise ValueError("epochs=1 配 clip_eps 是空转：new 与 old 同参数算出 ⇒ ratio≈1 "
                         "⇒ clamp 近乎恒等。请用 epochs>1，或 clip_eps=None（纯组内 REINFORCE）。")
    if policy.optim is None:
        policy.optim = torch.optim.Adam(policy.parameters(), lr=lr)
    # ── T3（拉格朗日）入口守卫：全部**显式报错**，不静默、不在跑了几分钟仿真之后才炸 ──
    if (t3_lambda is None) != (t3_budget is None):
        raise ValueError(
            "t3_lambda 与 t3_budget 必须**同时给**（λ 与预算是一对，单独给一个无意义）："
            f"实得 t3_lambda={'None' if t3_lambda is None else '非 None'}、"
            f"t3_budget={'None' if t3_budget is None else '非 None'}。")
    if t3_budget is not None:
        assert t3_lambda is not None            # 上面的一对校验已挡
        if adv_mode != "scalar":
            raise ValueError(
                f"T3 目前只支持 adv_mode='scalar'（实得 {adv_mode!r}）：罚项加在标量奖励上、"
                "组内 z 化之前（设计 §3.3 的 v1 口径）。per_objective 口径下'罚项算在哪一项'"
                "没有定义——v1'（罚项作为独立项各自 z 化）依赖 O1 切成默认，尚未实现。")
        if not t3_eta > 0.0:
            raise ValueError(f"t3_eta={t3_eta} 非法：对偶上升步长必须 > 0（T3 开着而 λ 不动"
                             "是静默空转，不是'冻结 λ'的开关——要冻结请传 t3_budget=None）。")
        lam = np.asarray(t3_lambda, dtype=np.float64)
        if lam.shape != (len(t3_budget),):
            raise ValueError(f"t3_lambda 形状 {lam.shape} ≠ T3Budget 的约束数 {len(t3_budget)}"
                             f"（keep={t3_budget.keep}）——λ 与预算不同源。")
        # 可控性守卫：没有对应动作的约束不纳入（单一真相在 `t3_budget.check_influenceable`，
        # CLI 用同一个函数在开工前先查一遍）。
        check_influenceable(t3_budget.keep, pm_head, charge_head, cfg.machine_age_failure)

    chains: list[list[Decision]] = []
    rewards: list[float] = []
    f_objs: list[tuple[float, float, float]] = []
    mets: list[dict] = []                       # T3 的激活量原料（逐链 metrics；T3 关时不用）
    if parallel:
        # ⚠️ 并行档（2026-10-04 并行批次）：G 条链各一个 worker 任务，worker 只跑 CPU；
        #    主进程只在这里收决策/指标，随后照旧做重算 + 反向 + 优化器步。
        #    ⚠️ 必须先把**当前**参数刷进共享内存镜像（上一步的 Adam 已改过参数）。
        assert pool is not None     # 入口已显式校验：parallel=True, pool=None 直接报错
        pool.sync_policy(policy)
        results = pool.run_chains(seed, G, inst=inst, layout=layout, dm=dm, cfg=cfg, ctx=ctx,
                                  constraints=cons, route_k=route_k, route_zones=route_zones,
                                  geom_bias=geom_bias, pm_head=pm_head, charge_head=charge_head,
                                  batch_head=batch_head)
        for dec, met in results:
            chains.append(dec)
            f = objective_vector(met)               # 逐目标值 (makespan, energy, TWT)，越小越好
            f_objs.append(f)
            rewards.append(scalar_reward(f, w))     # 奖励读数（与串行档同一运算次序）
            mets.append(met)
    else:
        # ⚠️ 评审 I-3：动作采样流由本步的 `seed` 派生并透传——组内 G 条链**顺序共享**同一条流
        #    （消费次序确定 ⇒ 逐位可复现），不再落到全局 torch RNG。
        #    ⚠️ 设备跟随策略（2026-10-04 设备批次）：CUDA 上必须用 CUDA generator（见 `roll_chain`）。
        gen = torch.Generator(device=policy.device.type).manual_seed(seed)
        for g in range(G):
            dec, met = roll_chain(inst, layout, dm, cfg, policy, seed * SEED_STRIDE + g, ctx,
                                  sample=True, generator=gen, constraints=cons, route_k=route_k,
                                  route_zones=route_zones, geom_bias=geom_bias,
                                  pm_head=pm_head, charge_head=charge_head,
                                  batch_head=batch_head)
            chains.append(dec)
            f = objective_vector(met)               # 逐目标值 (makespan, energy, TWT)，越小越好
            f_objs.append(f)
            rewards.append(scalar_reward(f, w))     # 奖励读数（诊断用；与历史同一运算次序）
            mets.append(met)

    # O1：两种优势口径的唯一分叉点（见 `_advantages`）。scalar 分支与历史表达式逐位相同。
    # T3：在**组内 z 化之前**逐链减罚项 `r'_g = r_g − Σᵢλᵢ·âᵢ,g`，再走同一个 `_z`
    #（设计 §3.2/§3.3 的 v1：罚项与目标共用 `scalar` 口径，λ 仍是单调旋钮）。
    if t3_budget is None:
        A = _advantages(np.asarray(f_objs, dtype=np.float64), w, adv_mode)
        t3_info: dict = {}
    else:
        assert t3_lambda is not None            # 入口守卫已校验成对
        ahat = np.stack([normalized_activation(m, t3_budget.a_ref, t3_budget.keep)
                         for m in mets])                                  # (G, n_active)
        lam_used = np.asarray(t3_lambda, dtype=np.float64)
        pen = ahat @ lam_used                                             # (G,)
        A = torch.tensor(_z(np.asarray(rewards, dtype=np.float64) - pen),
                         dtype=torch.float32)
        # ⚠️ λ 的下一步：**回传**给持有者（runner），本函数不持有任何跨步状态。
        ahat_mean = ahat.mean(axis=0)
        lam_next = dual_ascent(lam_used, ahat_mean, t3_budget.b, t3_eta)
        t3_info = {"t3_lambda": tuple(float(x) for x in lam_next),
                   "t3_lambda_used": tuple(float(x) for x in lam_used),
                   "t3_ahat": tuple(float(x) for x in ahat_mean),
                   "t3_pen_mean": float(pen.mean())}
    # ⚠️ 设备跟随参数（2026-10-04 设备批次）：优势 / 逐决策回放都必须在**参数设备**上参与
    #    损失（`--device cuda` 时 `ratio`、`obj` 全在 GPU 上）；`A` 本身留在 CPU 供诊断。
    A_dev = A.detach().to(policy.device)

    # 裁剪路径的信任域基准 = 采样那一刻 logp 的**逐决策**回放（`Decision.logp`，无梯度）——
    # 不重算，且与 `decisions_logp` / `all_decisions_logp` 同序同 dtype，故首轮
    # ratio ≈ 1（**1e-5 内**：重算已是批前向，末位与采样时逐条算出的值不同，见 `decisions_logp`）。
    # 组内链长不等：把 G 条链的全部决策拼成一个长向量，每个决策经
    # `repeat_interleave` 拿到**它那条链**的优势。
    if clip_eps is not None:
        lengths = torch.tensor([len(ch) for ch in chains], dtype=torch.long)
        old_flat = torch.cat([sampled_decisions_logp(ch) for ch in chains]).to(policy.device)
        # ⚠️ `repeat_interleave` 的 repeats 必须与输入**同设备**（CUDA 上否则 RuntimeError：
        #    index is on cpu, different from other tensors on cuda）。`lengths` 本身留在 CPU
        #    供 `_chain_mean` 的 `.split()` 用（那是纯 Python 侧）。
        adv_flat = A_dev.repeat_interleave(lengths.to(A_dev.device))
    else:
        old_flat = adv_flat = None

    loss_val, ratio_mean, clipped_frac, grad_norm = 0.0, 1.0, 0.0, 0.0
    for _ in range(max(epochs, 1)):
        if clip_eps is None:
            # ⚠️ `chains_logp` 把 **G 条链的全部决策**堆成一次编码器前向（重算的批量口径，
            #    见 `_decision_logp_terms`）——不是每条链各一次。
            new = chains_logp(chains, policy, recompute_chunk)
            obj = A_dev * new                # 纯组内 REINFORCE（理论骨架的「纯版本」）
        else:
            # 同一次批前向覆盖全部链的全部决策（展平序与 `old_flat` / `adv_flat` 一致）
            new_flat = all_decisions_logp(chains, policy, recompute_chunk)
            ratio = torch.exp(new_flat - old_flat)   # 逐**决策**概率比（与链长无关）
            obj = torch.min(ratio * adv_flat,
                            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_flat)
            ratio_mean = float(ratio.mean().item())
            # 出界比例 = 该轮 surrogate **失去梯度**的**决策**占比（1.0 = 全裁 ⇒ 这一轮纯浪费）
            clipped_frac = float(((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps))
                                 .float().mean().item())
        # ⚠️ 两条路径**都按链平均**：无裁剪的 `obj` 是 (G,) 直接 mean；
        #    裁剪的 `obj` 是展平的逐决策量，必须经 `_chain_mean` 折回按链——否则长链权重更大，
        #    且与无裁剪路径口径不一致（见 `_chain_mean` docstring）。
        loss = -(obj.mean() if clip_eps is None else _chain_mean(obj, lengths))
        policy.optim.zero_grad()
        loss.backward()
        # 范数裁剪（不是逐元素）；返回值 = 裁剪前的梯度范数，作为**非零**学习信号入诊断
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
        policy.optim.step()
        loss_val = float(loss.item())

    diag = {"loss": loss_val, "ratio": ratio_mean, "clipped_frac": clipped_frac,
            "grad_norm": grad_norm,
            "r_mean": float(np.mean(rewards)), "r_std": float(np.std(rewards)),
            "A_std": float(A.std()), **t3_info}
    for d in chains:                         # ⚠️ 决策日志**用完即弃**——不得跨 step 累积
        d.clear()
    return diag["r_mean"], diag
