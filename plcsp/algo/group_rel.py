"""联合链 GRPO（spec §5.3.4，P2 Task 7）——一条链 = 一个完整 episode 的**全部**决策。

    链 = (机台计划序列 a_S[1..n_ops]) ⊕ (派车序列 a_L[1..n_tasks]) ⊕ (路线序列 a_R[1..n_drives])
    logp(链) = Σ_t log π_S(a_S[t]) + Σ_t log π_L(a_L[t]) + Σ_t log π_R(a_R[t])   ← 三类都取【求和】

⚠️ **R（路线）是恢复出来的第三个决策，不是新发明**：`route_logits` 当初随 ① 拥堵一并取消
（理由"无拥堵时选远路严格更差"，spec §5.3），但 ① 后来在 `eb1d1da` 恢复、**路线头没跟着
恢复**（`docs/progress-log.md` §27.3、§28）。本模块的 R 分支即那次遗漏的补建：
`AgvSim._drive` 从"恒走最短路"改为"在 k 条候选里问策略"，`roll_chain` 照 S/L 的规格记录
R 决策，链 logp 照常求和。⚠️ **默认 `route_k=1`（关闭）**——关闭档必须**逐位等于今日行为**
（黄金摘要钉死），既有全部读数才继续成立；启用传 `route_k=2`。

后接标准 GRPO：**一个**终端奖励 r → 组内 z 化（`_z`）→ 纯组内 REINFORCE（默认：
`epochs=1, clip_eps=None`）或 PPO 式裁剪（**只在 `epochs>1` 时才有意义**——`epochs=1` 时
`ratio ≡ 1`，裁剪项恒等、纯空转；见 `joint_chain_step` 的守卫）。信任域按**决策**施加
（每个决策一个 ratio）：2026-10-04 修复——此前按**整条链**裁剪，而链 logp 是求和（100–460
项）⇒ 一次更新就把全部链推出带外、后续 epoch 梯度恒 0，clip 形同虚设（见 `joint_chain_step`）。

⚠️ **O1：优势有两个口径**（`joint_chain_step` 的 `adv_mode`）——`"scalar"`（默认，历史口径：
先加权求和再组内 z 化）与 `"per_objective"`（每目标各自组内 z 化、再按 w 合成）。后者对
**逐目标单位缩放不变** ⇒ w 真正控制各目标的权衡；见 `_advantages` 的 docstring。

spec §5.3.4 五条硬性约定在本模块的落点：
1. **联合链、单一优势** —— `roll_chain` 把**三头**的决策记在**同一条**链上（同一 episode），
   `joint_chain_step` 只算**一个** A（不按头分组、不按层归一化）；
2. **logp 一律取求和** —— `chain_logp` = `Σ logπ_S + Σ logπ_L + Σ logπ_R`（**不除决策数**：旧实现
   S 取平均、L 取求和，同一个 `ratio=exp(Δ)` 在两边含义不同）。⚠️ 求和只定义**链级** logp；
   裁剪的信任域不看它，按**逐决策**的 `decisions_logp` 算（修复，见上段）；
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

import numpy as np
import torch

from .policy import PolicyNet, v_token_index
from ..env.constraints import ConstraintConfig
from ..env.corridors import build_corridor_graph
from ..env.des import SimConfig, SimWorld, build_zone_map
from ..env.instances import Instance
from ..env.layout import Layout
from ..env.reward import (ReferenceObjectives, objective_vector, reward_weights,
                          scalar_reward)
from ..nn.features import NormContext
from ..nn.state_emb import build_tok


# 训练流步长：第 s 步第 g 条链的仿真扰动种子 = (seed0+s)*SEED_STRIDE + g。`m13_train_a` 的
# "评估流不得与训练流相交"检查复用此常量（单一来源，避免两处漂移）。
SEED_STRIDE = 1000

# R 头两个特征槽的宽度（**唯一真相**，与 `PolicyNet` 的 `n_feat_route*` 形参对应——
# `test_route_choice.py::test_route_feature_widths_match_the_head` 按它核对打分头的输入
# 宽度，两处漂开即报错）。
ROUTE_FEAT_DRIVE = 4      # `route_feat`：起点/终点/负载标志/本车号
ROUTE_FEAT_CAND = 3       # `_route_cand_feat`：长度比/区段数占比/被别的车占着的区段占比


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

    - `scalar`：r_g = Σᵢ wᵢ(−f_{g,i})，A = z(r)。⚠️ 这是**今日**的表达式（逐位兼容，
      由 `test_scalar_adv_mode_is_bitwise_unchanged` 的捕获摘要钉死）。z 只消掉加权和的
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
    """
    vs = snap.vehicles
    out = np.empty((len(cand), 1), dtype=np.float32)
    for i, a in enumerate(cand):
        node = int(vs[a].node)
        if node < 0:
            out[i, 0] = -1.0
            continue
        dist = float(dm[node, layout.machines[frm].dock_node])
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


@dataclass
class Decision:
    """一个决策点的**全部打分输入**（采样时冻结）——`chain_logp` 据此带梯度重算 logp。

    存的是**当时的**上下文，不是事后重建的：仿真状态在变，用最新快照重算等于把策略输入
    换成另一个状态（决策与 logp 不再对应）。
    ⚠️ `cand_feat` 是**逐候选**特征（T4-3）：S 头放换型标志、L 头放"该车到取货点的预计行驶
    时长"、R 头放路径的"长度比/区段数/争用"。少了它，`chain_logp` 重算的分布与采样时的分布
    **不是同一个**（ratio≠1 的假象）。
    """
    kind: str                        # "S"（选机台）| "L"（派车）| "R"（选路）
    tok: np.ndarray                  # 当时的四段 token 特征 (N, F_MAX)
    seg: tuple[int, int, int, int]   # 当时的段长 (n_m, n_jobs, n_agv, 1)
    feat: np.ndarray                 # 决策特征 (F_dec,)：S=`op_feat` / L=`task_feat` / R=`route_feat`
    cand_feat: np.ndarray            # 逐候选特征 (n_cand, F_cand)
    cand: tuple[int, ...]            # 候选（S: 机台号；L: 车号；R: 0..k-1 的候选序号）
    action: int                      # 所取的动作（∈ cand）
    # 打分时**取 token 嵌入用的下标**（逐候选）。S/L 就是 `cand`（机台号=序列位置；
    # 车号的历史口径——**不得改动**，改了会破坏既有读数与比例恒等）。R 的候选是**路径**，
    # 序列里没有它们的 token（见 `PolicyNet.route_logits_emb`），故这里是**本车 V token 下标
    # 广播 k 份**。单独存而不是从 `cand` 推：两类语义不同，混用会让 R 头去索引机台 token。
    tok_idx: np.ndarray
    # 采样那一刻策略在 `action` 上的 log 概率（无梯度标量）。它是 `decisions_logp` 带梯度
    # 重算值的**同源对照**：裁剪路径的 `old` 直接由它**逐决策**回放（`sampled_decisions_logp`，
    # 省掉一整遍"重算 old"的链前向），逐位一致性由测试钉死。既然取自采样那一刻，它天然不受
    # "采样之后再算 old"这类重排的影响。
    logp: float


def roll_chain(inst: Instance, layout: Layout, dm: np.ndarray, cfg: SimConfig,
               policy: PolicyNet, seed: int, ctx: NormContext,
               sample: bool = True,
               generator: torch.Generator | None = None,
               constraints: ConstraintConfig | None = None,
               route_k: int = 1) -> tuple[list[Decision], dict]:
    """跑一条链：仿真里每个派工点同步调策略，记录每个决策的 (token 特征, 决策特征, 候选, 动作)。

    ⚠️ **无梯度**——决策只记上下文（`torch.no_grad()` 下取样），logp 事后由 `chain_logp`
       **带梯度重算**（SimPy 栈不参与反向传播，见模块头）。
    ⚠️ 每个决策存**当时的** token/决策/候选特征——仿真状态在变，用事后最新快照重建等于把
       策略输入换成另一个状态（决策与 logp 不再对应）。
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
     ⚠️ **动作采样流**（评审 I-3）：`generator=None` ⇒ 按 `torch.Generator().manual_seed(seed)`
        现建——同 seed 同调用序列的动作**逐位相同**；显式传入者自备种子（`joint_chain_step`
        自建一条并透传，组内 G 条链顺序共享）。`sample=False`（argmax）不消费该流。
        ⚠️ R 与 S/L 共享**同一条**动作流（谁先决策谁先消费）——三头是一条链上的联合策略，
        不各开一条流（那会让"同 seed 可复现"变成分头可复现）。
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
    cons = constraints or ConstraintConfig()
    if route_k > 1 and not cons.congestion:
        raise ValueError(
            f"route_k={route_k} 需要 ① 拥堵开启：① 关时区段机制不存在，选远路**严格更差**"
            "（spec §5.3 的原始理由）——给策略一个死动作只会污染链 logp。请开 ① 或传 route_k=1。")
    decisions: list[Decision] = []
    gen = generator if generator is not None else torch.Generator().manual_seed(seed)
    world = SimWorld(inst, layout, dm, cfg, graph=build_corridor_graph(layout),
                     constraints=constraints)
    # 区段映射（R 头争用特征的原料）：与 `SimWorld` 内部 `build_zone_map(layout, cfg.zone_granularity)`
    # 同一函数、同一入参——两处不可能漂（漂了候选特征会按另一套区段数错标度）。
    zof, n_zones = build_zone_map(layout, cfg.zone_granularity) if route_k > 1 else ({}, 0)

    def _act(kind: str, snap, feat: list[float], cand_feat: np.ndarray,
             cand: tuple[int, ...], aid: int | None = None) -> int:
        tok_feat, seg = build_tok(snap, inst, layout, ctx)
        tok_feat = np.asarray(tok_feat, dtype=np.float32)
        with torch.no_grad():
            tok, _ = policy.forward_enc(torch.as_tensor(tok_feat).unsqueeze(0), seg)
            head = {"S": policy.mach_logits_emb, "L": policy.agv_logits_emb,
                    "R": policy.route_logits_emb}[kind]
            if kind == "R":
                # 路线候选在序列里没有 token（见 `PolicyNet.route_logits_emb`）：取本车 V token
                # 的下标、k 个候选共用同一份——打分只差在 `cand_feat`。
                tok_idx = np.full(len(cand), v_token_index(seg)[int(aid)], dtype=np.int64)
            else:
                tok_idx = np.asarray(cand, dtype=np.int64)
            logits = head(tok,
                          torch.as_tensor(feat, dtype=torch.float32).reshape(1, 1, -1),
                          torch.as_tensor(cand_feat, dtype=torch.float32),
                          torch.as_tensor(tok_idx, dtype=torch.long))
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
                                  logp=lp_k))
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
        return _act("R", snap, route_feat(layout, aid, src, dst, leg),
                    _route_cand_feat(snap, zof, n_zones, cands, dm, aid),
                    tuple(range(len(cands))), aid=int(aid))

    metrics = world.run_gated(seed_chain=seed, online_s=True,
                              policy_s=policy_s, policy_l=policy_l,
                              policy_r=None if route_k == 1 else policy_r)
    return decisions, metrics


def _decision_logp_terms(decisions: list[Decision], policy: PolicyNet) -> list[torch.Tensor]:
    """每个决策的 logπ(a)（**标量张量**，带梯度），按 `decisions` 原序——唯一的打分体重算。

    `chain_logp`（求和）与 `decisions_logp`（向量化裁剪用）都从这里取项，**不可能漂开**。
    每个决策用**当时记录的** token/决策/候选特征重建打分（状态已变，不能用最新快照）。
    梯度经 `forward_enc` 同时回到**三**个头与编码器——这是"联合链"的实质（约定 1/5）。
    ⚠️ token 下标用 `d.tok_idx`（R 与 S/L 的下标语义不同，见 `Decision.tok_idx`）——
    S/L 的 `tok_idx` 恒等于 `cand`，故这条统一路径不改既有两头的任何一位。
    """
    out: list[torch.Tensor] = []
    for d in decisions:
        tok, _ = policy.forward_enc(
            torch.as_tensor(d.tok, dtype=torch.float32).unsqueeze(0), d.seg)
        head = {"S": policy.mach_logits_emb, "L": policy.agv_logits_emb,
                "R": policy.route_logits_emb}[d.kind]
        logits = head(tok,
                      torch.as_tensor(d.feat, dtype=torch.float32).reshape(1, 1, -1),
                      torch.as_tensor(d.cand_feat, dtype=torch.float32),
                      torch.as_tensor(d.tok_idx, dtype=torch.long))
        lp = torch.log_softmax(logits.flatten(), -1)
        out.append(lp[d.cand.index(d.action)])
    return out


def decisions_logp(decisions: list[Decision], policy: PolicyNet) -> torch.Tensor:
    """**逐决策** logπ(a_t) 向量 `(n_decisions,)`，**带梯度**——裁剪的信任域就建在它上面。

    它是 `chain_logp` 去掉最后那步求和：同一个打分体（`_decision_logp_terms`）、同一批
    "当时的"上下文。裁剪按**决策**而非整条链施加（见 `joint_chain_step` 的信任域段落）。
    ⚠️ **不要**反过来用 `... .sum()` 定义 `chain_logp`：`Tensor.sum()` 的归约次序与逐步
    float32 相加不同（实测末位差 ~1e-5），会破坏 `chain_logp` 与 `sampled_logp` 的**逐位
    同源**——裁剪首轮 `ratio ≡ 1` 是**严格等式**的前提。
    """
    return torch.stack(_decision_logp_terms(decisions, policy))


def chain_logp(decisions: list[Decision], policy: PolicyNet) -> torch.Tensor:
    """Σ_t logπ_S(a_t) + Σ_t logπ_L(a_t)——**求和**（spec §5.3.4 约定 2），**带梯度**。

    ⚠️ 累加**必须逐步 float32**（与 `sampled_logp` 同序、同 dtype ⇒ 参数未变时**逐位相同**，
    裁剪首轮 ratio 才严格 ≡1）。逐决策的值见 `decisions_logp`；两者共用同一打分体。
    """
    total = torch.zeros(())
    for term in _decision_logp_terms(decisions, policy):
        total = total + term
    return total


def sampled_decisions_logp(decisions: list[Decision]) -> torch.Tensor:
    """**逐决策**回放采样那一刻的 logp：向量 `(n_decisions,)`，无梯度——裁剪路径的 `old`。

    ⚠️ 与 `decisions_logp` **同序、同 dtype**（float32），故"参数未变"时两者**逐位相同**
    ——`ratio = exp(new − old) ≡ 1` 才是严格成立（而非近似）。别图省事改用 float64：
    那会引入 ~1e-5 的假 delta（实测 1.0000114），在裁剪边界上给出无意义的翻转。
    本函数没有可省的前向：值在 `roll_chain` 采样时就已算好（`Decision.logp`）。
    """
    return torch.stack([torch.tensor(d.logp, dtype=torch.float32) for d in decisions])


def sampled_logp(decisions: list[Decision]) -> torch.Tensor:
    """Σ `Decision.logp`——**链级** logp 的采样时刻值（`chain_logp` 的无梯度对照）。

    ⚠️ 裁剪路径已改**逐决策**（`sampled_decisions_logp`），本函数不再是裁剪基准；保留为
    链级口径的定义对照与逐位同源的回归判据（测试用）。累加保持逐步 float32（同 `chain_logp`）。
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
                     route_k: int = 1) -> tuple[float, dict]:
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
    上算出 ⇒ `ratio ≡ 1` ⇒ `clamp(1, 1±ε) ≡ 1`，裁剪项**恒等**、纯空转（`test_clipped_path_
    ratio_uses_sampling_time_logp` 的 `lr=0` 判据测的就是这个恒等式）。故此处显式**拒收**
    "epochs=1 + clip_eps"这个组合，
    并在 `epochs=1` 时**连 `old` 都不算**——省掉一整遍链前向；`epochs>1` 时 `old` 直接取
    `Decision.logp`（采样那一刻已存，`sampled_decisions_logp` **逐决策**回放），**不额外重算**。

    ⚠️ **`route_k`（R 层开关，2026-10-04 恢复路线头）**：默认 `1` = 关闭，链与训练步
    **逐位等于今日**（既有读数全靠它）；传 `2` 启用——每次行驶在 2 条候选路径里由策略选，
    链长与墙钟随之上升（MK01 实测见 `docs/progress-log.md`）。R 决策与 S/L 同进链 logp、
    同吃一条采样流、同受逐决策裁剪——**三头是一条联合链**，不是三套并行策略。

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
        raise ValueError("epochs=1 配 clip_eps 是空转：new 与 old 同参数算出 ⇒ ratio≡1 "
                         "⇒ clamp 恒等。请用 epochs>1，或 clip_eps=None（纯组内 REINFORCE）。")
    if policy.optim is None:
        policy.optim = torch.optim.Adam(policy.parameters(), lr=lr)

    chains: list[list[Decision]] = []
    rewards: list[float] = []
    f_objs: list[tuple[float, float, float]] = []
    # ⚠️ 评审 I-3：动作采样流由本步的 `seed` 派生并透传——组内 G 条链**顺序共享**同一条流
    #    （消费次序确定 ⇒ 逐位可复现），不再落到全局 torch RNG。
    gen = torch.Generator().manual_seed(seed)
    for g in range(G):
        dec, met = roll_chain(inst, layout, dm, cfg, policy, seed * SEED_STRIDE + g, ctx,
                              sample=True, generator=gen, constraints=cons, route_k=route_k)
        chains.append(dec)
        f = objective_vector(met)               # 逐目标值 (makespan, energy, TWT)，越小越好
        f_objs.append(f)
        rewards.append(scalar_reward(f, w))     # 奖励读数（诊断用；与历史同一运算次序）

    # O1：两种优势口径的唯一分叉点（见 `_advantages`）。scalar 分支与历史表达式逐位相同。
    A = _advantages(np.asarray(f_objs, dtype=np.float64), w, adv_mode)

    # 裁剪路径的信任域基准 = 采样那一刻 logp 的**逐决策**回放（`Decision.logp`，无梯度）——
    # 不重算，且与 `decisions_logp` 同序同 dtype（见 `sampled_decisions_logp`），故首轮
    # ratio 逐位严格 = 1。组内链长不等：把 G 条链的全部决策拼成一个长向量，每个决策经
    # `repeat_interleave` 拿到**它那条链**的优势。
    if clip_eps is not None:
        lengths = torch.tensor([len(ch) for ch in chains], dtype=torch.long)
        old_flat = torch.cat([sampled_decisions_logp(ch) for ch in chains])
        adv_flat = A.detach().repeat_interleave(lengths)
    else:
        old_flat = adv_flat = None

    loss_val, ratio_mean, clipped_frac, grad_norm = 0.0, 1.0, 0.0, 0.0
    for _ in range(max(epochs, 1)):
        if clip_eps is None:
            new = torch.stack([chain_logp(d, policy) for d in chains])
            obj = A.detach() * new           # 纯组内 REINFORCE（理论骨架的「纯版本」）
        else:
            new_flat = torch.cat([decisions_logp(d, policy) for d in chains])
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
            "A_std": float(A.std())}
    for d in chains:                         # ⚠️ 决策日志**用完即弃**——不得跨 step 累积
        d.clear()
    return diag["r_mean"], diag
