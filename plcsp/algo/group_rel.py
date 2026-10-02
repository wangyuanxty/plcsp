"""联合链 GRPO（spec §5.3.4，P2 Task 7）——一条链 = 一个完整 episode 的**全部**决策。

    链 = (机台计划序列 a_S[1..n_ops]) ⊕ (派车序列 a_L[1..n_tasks])
    logp(链) = Σ_t log π_S(a_S[t]) + Σ_t log π_L(a_L[t])        ← 两条都取【求和】

后接标准 GRPO：**一个**终端奖励 r → 组内 z 化（`_z`）→ 纯组内 REINFORCE 或 PPO 式裁剪。

spec §5.3.4 五条硬性约定在本模块的落点：
1. **联合链、单一优势** —— `roll_chain` 把两头的决策记在**同一条**链上（同一 episode），
   `joint_chain_step` 只算**一个** A（不按头分组、不按层归一化）；
2. **logp 一律取求和** —— `chain_logp` = `Σ logπ_S + Σ logπ_L`（**不除决策数**：旧实现 S 取
   平均、L 取求和，同一个 `ratio=exp(Δ)` 在两边含义不同，clip 对单决策的约束强度差 n 倍）；
3. **J=1，预算全给 G** —— `joint_chain_step` 只有 G（J 个扰动取均值留给**评估**，不混进训练）；
4. **优化器 Adam** —— `policy.optim` 惰性创建为 `torch.optim.Adam`（旧的"手写 SGD + 逐元素
   clamp ±1.0"无动量无自适应，已随旧训练器删除）；
5. **L 头接编码器** —— 两头都经 `forward_enc` 拿**同一份** token 嵌入（Task 4 已改）。

⚠️ **梯度口径**（本模块最容易写错的一处）：`roll_chain` 全程 `torch.no_grad()` 采样——决策只记
**上下文**（当时的 token 特征 / 决策特征 / 逐候选特征 / 候选 / 动作），logp 事后由 `chain_logp`
用**当前**策略**带梯度重算**。仿真栈（SimPy）不参与反向传播，这是必须的。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .policy import PolicyNet
from ..env.corridors import build_corridor_graph
from ..env.des import SimConfig, SimWorld
from ..env.instances import Instance
from ..env.layout import Layout
from ..env.reward import objective_vector, scalar_reward
from ..nn.features import NormContext
from ..nn.state_emb import build_tok


def _z(vals: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return vals - vals.mean()
    if mode == "loo":
        n = len(vals)
        if n <= 1:
            return vals.copy()
        return np.array([vals[i] - (vals.sum() - vals[i]) / (n - 1) for i in range(n)])
    return (vals - vals.mean()) / (vals.std() + 1e-9)


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


def setup_flag(snap, machine: int, job: int) -> float:
    """该机台为该作业加工**是否需换型**（1.0 = 需要）——⑤ 进网的取值。

    判据同 `des.MachineSim._process`：与本机**上一件**加工的作业不同才换型（`prev_job`
    为 -1 = 该机还没加工过，同样不换）。这是 S 头候选特征的语义（spec §5.3.1②）。
    """
    return 0.0 if snap.machines[machine].prev_job in (-1, job) else 1.0


def task_feat(inst: Instance, snap, frm: int, to: int, oi: int, job: int) -> list[float]:
    """L 头的**任务特征**（4 维）：起送机台 / 目标机台 / 目标工序序号(归一) / 该机是否需换型。

    ⚠️ 第 4 维需要**作业号**（`setup_flag(snap, to, job)`）——故 `des.run_gated` 的
    `policy_l` 回调契约在 Task 7 扩了 `job` 参数（原 `(snap, frm, to, oi, cand)` 表达不了
    本维；用 `(frm, to, oi)` 反查作业**多义**：实测 MK01 51/112、MK10 538/985 个键歧义）。
    """
    n_ref = max(max(len(j) for j in inst.jobs), 1)
    n_m = max(inst.n_machines, 1)
    return [frm / n_m, to / n_m, oi / n_ref, setup_flag(snap, to, job)]


def _mach_cand_feat(snap, cand: tuple[int, ...], job: int) -> np.ndarray:
    """S 头**逐候选**特征 `(n_cand, 1)`：该机台为本作业加工是否需换型（⑤ / spec §5.3.1②）。"""
    return np.array([[setup_flag(snap, m, job)] for m in cand], dtype=np.float32)


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


@dataclass
class Decision:
    """一个决策点的**全部打分输入**（采样时冻结）——`chain_logp` 据此带梯度重算 logp。

    存的是**当时的**上下文，不是事后重建的：仿真状态在变，用最新快照重算等于把策略输入
    换成另一个状态（决策与 logp 不再对应）。
    ⚠️ `cand_feat` 是**逐候选**特征（T4-3）：S 头放换型标志、L 头放"该车到取货点的预计行驶
    时长"。少了它，`chain_logp` 重算的分布与采样时的分布**不是同一个**（ratio≠1 的假象）。
    """
    kind: str                        # "S"（选机台）| "L"（派车）
    tok: np.ndarray                  # 当时的四段 token 特征 (N, F_MAX)
    seg: tuple[int, int, int, int]   # 当时的段长 (n_m, n_jobs, n_agv, 1)
    feat: np.ndarray                 # 决策特征 (F_dec,)：S=`op_feat` / L=`task_feat`
    cand_feat: np.ndarray            # 逐候选特征 (n_cand, F_cand)
    cand: tuple[int, ...]            # 候选（S: 机台号；L: 车号）
    action: int                      # 所取的动作（∈ cand）


def roll_chain(inst: Instance, layout: Layout, dm: np.ndarray, cfg: SimConfig,
               policy: PolicyNet, seed: int, ctx: NormContext,
               sample: bool = True) -> tuple[list[Decision], dict]:
    """跑一条链：仿真里每个派工点同步调策略，记录每个决策的 (token 特征, 决策特征, 候选, 动作)。

    ⚠️ **无梯度**——决策只记上下文（`torch.no_grad()` 下取样），logp 事后由 `chain_logp`
       **带梯度重算**（SimPy 栈不参与反向传播，见模块头）。
    ⚠️ 每个决策存**当时的** token/决策/候选特征——仿真状态在变，用事后最新快照重建等于把
       策略输入换成另一个状态（决策与 logp 不再对应）。
    ⚠️ 布局由调用方给（本函数**不采样布局**）：`layout.layout_seed` 必须与奖励侧参考运行同源
       ——`ReferenceObjectives.of` 固定 seed_layout=0，而 `SimWorld._due_map` 用**该布局**的
       seed 取 M_ref。用默认奖励口径时请传 seed=0 的布局（本任务不扩这条口径）。
    ⚠️ L 回调需要**作业号**（任务特征第 4 维 = 该机是否需换型）：`des.run_gated` 的
       `policy_l` 契约在 Task 7 由 `(snap, frm, to, oi, cand)` 扩为 `(snap, job, frm, to, oi, cand)`
       ——用 `(frm, to, oi)` 反查作业**多义**（实测 MK01 51/112、MK10 538/985 个键歧义），
       静默取错作业 = 换型特征错，故走显式契约。
    返回 (决策序列, `run_gated` 的 metrics)。
    """
    decisions: list[Decision] = []
    world = SimWorld(inst, layout, dm, cfg, graph=build_corridor_graph(layout))

    def _act(kind: str, snap, feat: list[float], cand_feat: np.ndarray,
             cand: tuple[int, ...]) -> int:
        tok_feat, seg = build_tok(snap, inst, layout, ctx)
        tok_feat = np.asarray(tok_feat, dtype=np.float32)
        with torch.no_grad():
            tok, _ = policy.forward_enc(torch.as_tensor(tok_feat).unsqueeze(0), seg)
            head = policy.mach_logits_emb if kind == "S" else policy.agv_logits_emb
            logits = head(tok,
                          torch.as_tensor(feat, dtype=torch.float32).reshape(1, 1, -1),
                          torch.as_tensor(cand_feat, dtype=torch.float32),
                          torch.tensor(cand, dtype=torch.long))
            p = torch.softmax(logits.flatten(), -1)
            k = int(torch.multinomial(p, 1).item()) if sample else int(p.argmax())
        decisions.append(Decision(kind=kind, tok=tok_feat, seg=seg,
                                  feat=np.asarray(feat, dtype=np.float32),
                                  cand_feat=np.asarray(cand_feat, dtype=np.float32),
                                  cand=cand, action=int(cand[k])))
        return int(cand[k])

    def policy_s(snap, job, oi, cand):
        cand = tuple(int(c) for c in cand)
        return _act("S", snap, op_feat(inst, job, oi, layout),
                    _mach_cand_feat(snap, cand, job), cand)

    def policy_l(snap, job, frm, to, oi, cand):
        cand = tuple(int(c) for c in cand)
        return _act("L", snap, task_feat(inst, snap, frm, to, oi, job),
                    _agv_cand_feat(snap, layout, dm, ctx, cand, frm), cand)

    metrics = world.run_gated(seed_chain=seed, online_s=True,
                              policy_s=policy_s, policy_l=policy_l)
    return decisions, metrics


def chain_logp(decisions: list[Decision], policy: PolicyNet) -> torch.Tensor:
    """Σ_t logπ_S(a_t) + Σ_t logπ_L(a_t)——**求和**（spec §5.3.4 约定 2），**带梯度**。

    每个决策用**当时记录的** token/决策/候选特征重建打分（状态已变，不能用最新快照）。
    梯度经 `forward_enc` 同时回到两个头与编码器——这是"联合链"的实质（约定 1/5）。
    """
    total = torch.zeros(())
    for d in decisions:
        tok, _ = policy.forward_enc(
            torch.as_tensor(d.tok, dtype=torch.float32).unsqueeze(0), d.seg)
        head = policy.mach_logits_emb if d.kind == "S" else policy.agv_logits_emb
        logits = head(tok,
                      torch.as_tensor(d.feat, dtype=torch.float32).reshape(1, 1, -1),
                      torch.as_tensor(d.cand_feat, dtype=torch.float32),
                      torch.tensor(d.cand, dtype=torch.long))
        lp = torch.log_softmax(logits.flatten(), -1)
        total = total + lp[d.cand.index(d.action)]
    return total


def joint_chain_step(policy: PolicyNet, inst: Instance, layout: Layout, dm: np.ndarray,
                     cfg: SimConfig, ctx: NormContext, w: tuple[float, float, float],
                     seed: int, G: int = 8, lr: float = 1e-3,
                     clip_eps: float | None = 0.2,
                     epochs: int = 1) -> tuple[float, dict]:
    """一步联合链组训练（spec §5.3.4）。

    G 条链（**J=1**，预算全给 G：约定 3）→ 每条一个终端奖励（三目标加权标量化，
    `reward.scalar_reward`）→ **组内 z 化**（`_z`，一个优势，不按头分层）→ PPO 式裁剪更新。
    优化器 = Adam（`policy.optim` 惰性创建：约定 4；不再手写 SGD + 逐元素 clamp）。
    `clip_eps=None` ⇒ 纯组内 REINFORCE（不裁剪、不算 ratio）；`epochs` = 同批链的重用轮数。
    返回 (组内 r 均值, 诊断 dict：loss / ratio / r 均值与 std / 优势 std)。
    """
    if policy.optim is None:
        policy.optim = torch.optim.Adam(policy.parameters(), lr=lr)

    chains: list[list[Decision]] = []
    rewards: list[float] = []
    for g in range(G):
        dec, met = roll_chain(inst, layout, dm, cfg, policy, seed * 1000 + g, ctx, sample=True)
        chains.append(dec)
        rewards.append(scalar_reward(objective_vector(met), w))

    A = torch.tensor(_z(np.asarray(rewards, dtype=np.float64), "z"), dtype=torch.float32)

    with torch.no_grad():                    # 冻结旧 logp（裁剪代理的 ratio 基准）
        old = torch.stack([chain_logp(d, policy) for d in chains]).detach()

    loss_val, ratio_mean = 0.0, 1.0
    for _ in range(max(epochs, 1)):
        new = torch.stack([chain_logp(d, policy) for d in chains])
        if clip_eps is None:
            obj = A.detach() * new           # 纯组内 REINFORCE（理论骨架的「纯版本」）
        else:
            ratio = torch.exp(new - old)     # 链概率比（求和口径才成立，约定 2）
            obj = torch.min(ratio * A.detach(),
                            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A.detach())
            ratio_mean = float(ratio.mean().item())
        loss = -obj.mean()
        policy.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)   # 范数裁剪（不是逐元素）
        policy.optim.step()
        loss_val = float(loss.item())

    diag = {"loss": loss_val, "ratio": ratio_mean,
            "r_mean": float(np.mean(rewards)), "r_std": float(np.std(rewards)),
            "A_std": float(A.std())}
    for d in chains:                         # ⚠️ 决策日志**用完即弃**——不得跨 step 累积
        d.clear()
    return diag["r_mean"], diag
