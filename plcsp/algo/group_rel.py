"""联合链 GRPO（spec §5.3.4，P2 Task 7）——一条链 = 一个完整 episode 的**全部**决策。

    链 = (机台计划序列 a_S[1..n_ops]) ⊕ (派车序列 a_L[1..n_tasks])
    logp(链) = Σ_t log π_S(a_S[t]) + Σ_t log π_L(a_L[t])        ← 两条都取【求和】

后接标准 GRPO：**一个**终端奖励 r → 组内 z 化（`_z`）→ 纯组内 REINFORCE（默认：
`epochs=1, clip_eps=None`）或 PPO 式裁剪（**只在 `epochs>1` 时才有意义**——`epochs=1` 时
`ratio ≡ 1`，裁剪项恒等、纯空转；见 `joint_chain_step` 的守卫）。

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
    # 采样那一刻策略在 `action` 上的 log 概率（无梯度标量）。它是 `chain_logp` 带梯度重算值的
    # **同源对照**：裁剪路径的 `logp_old` 直接由它求和（省掉一整遍"重算 old"的链前向），
    # 一致性由测试钉死。既然取自采样那一刻，它天然不受"采样之后再算 old"这类重排的影响。
    logp: float


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
            # 用 log_softmax（而非 softmax）取样：同一遍里就拿到所取动作的 log 概率
            # （`Decision.logp`）——裁剪路径据此省掉一整遍"重算 logp_old"的链前向。
            lp_all = torch.log_softmax(logits.flatten(), -1)
            k = (int(torch.multinomial(lp_all.exp(), 1).item()) if sample
                 else int(lp_all.argmax()))
            lp_k = float(lp_all[k])
        decisions.append(Decision(kind=kind, tok=tok_feat, seg=seg,
                                  feat=np.asarray(feat, dtype=np.float32),
                                  cand_feat=np.asarray(cand_feat, dtype=np.float32),
                                  cand=cand, action=int(cand[k]), logp=lp_k))
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


def sampled_logp(decisions: list[Decision]) -> torch.Tensor:
    """Σ `Decision.logp`——采样那一刻 logp 的**无梯度回放**（裁剪路径的 `logp_old`）。

    ⚠️ 与 `chain_logp` **同序、同 dtype** 累加（float32 逐步相加），故"参数未变"时两者
    **逐位相同**——`ratio = exp(new − old) ≡ 1` 才是严格成立（而非近似）。
    别图省事改用 float64 求和：那会引入 ~1e-5 的假 delta，把 ratio 推离 1（实测 1.0000114），
    在裁剪边界上给出无意义的翻转。本函数没有可省的前向：值在 `roll_chain` 采样时就已算好。
    """
    total = torch.zeros(())
    for d in decisions:
        total = total + torch.tensor(d.logp, dtype=torch.float32)
    return total


def joint_chain_step(policy: PolicyNet, inst: Instance, layout: Layout, dm: np.ndarray,
                     cfg: SimConfig, ctx: NormContext, w: tuple[float, float, float],
                     seed: int, G: int = 8, lr: float = 1e-3,
                     clip_eps: float | None = None,
                     epochs: int = 1) -> tuple[float, dict]:
    """一步联合链组训练（spec §5.3.4）。

    G 条链（**J=1**，预算全给 G：约定 3）→ 每条一个终端奖励（三目标加权标量化，
    `reward.scalar_reward`）→ **组内 z 化**（`_z`，一个优势，不按头分层）→ 组内更新。
    优化器 = Adam（`policy.optim` 惰性创建：约定 4；不再手写 SGD + 逐元素 clamp）。

    ⚠️ **裁剪只在 `epochs > 1` 时才有意义**（默认 `epochs=1, clip_eps=None` = 纯组内 REINFORCE，
    即 spec §5.3.4/理论骨架的「纯版本」）：`epochs=1` 时 `new` 与 `logp_old` 都在**同一组参数**
    上算出 ⇒ `ratio ≡ 1` ⇒ `clamp(1, 1±ε) ≡ 1`，裁剪项**恒等**、纯空转（`test_ratio_is_one_
    for_unchanged_policy` 测的就是这个事实）。故此处显式**拒收**"epochs=1 + clip_eps"这个组合，
    并在 `epochs=1` 时**连 `logp_old` 都不算**——省掉一整遍链前向；`epochs>1` 时 `logp_old`
    直接取 `Decision.logp`（采样那一刻已存），**不额外重算**。

    ⚠️ **链级 ratio 的尺度**（本任务实测，务必知情）：`logp` 是**求和**（约定 2，链长 100–460），
    故 `ratio = exp(Δ链logp)` 对单决策变化极敏感——**一次** Adam 更新（lr=1e-3）就把链 logp 推过
    `log(1.2)`，于是 `epochs≥2` 时全部链的 ratio 出界、裁剪项恒定、**梯度恒 0**（实测 MK01、
    G=2/G=4：epoch≥1 的 `grad_norm` 全是 0.0000）⇒ 多出来的那些 epoch 是**纯浪费**。
    链级 `clip_eps=0.2` 等价于"每决策 ≈ log(1.2)/n ≈ 0.002 nats（n=100）"。要真信任域，
    `clip_eps` 得按链长放大（或改成分决策裁剪——那要动约定 2，属研究口径决策）。
    `clipped_frac` 就是这条的读数（1.0 = 全裁、该轮无梯度信号）。

    ⚠️ **`layout.layout_seed` 必须为 0**（守卫在入口）：奖励权重 `w` 的既定来源
    `ReferenceObjectives.of` 固定用 seed_layout=0 的参考运行，而 `SimWorld._due_map` 用**该
    布局**的 seed 取 M_ref（连交期本身都随之变）——两者不同源则目标口径与权重口径**静默错位**
    （Fact F）。如需非 0 布局种子，须先让 `ReferenceObjectives` 记录其种子再放宽此守卫。

    返回 (组内 r 均值, 诊断 dict：loss / ratio / clipped_frac / grad_norm / r 均值与 std / 优势 std)。
    `ratio` / `clipped_frac` 只在裁剪路径被真算（无裁剪时报 1.0 / 0.0，= 不适用）。`loss` 在裁剪
    路径的首轮**结构性为 0**（`ratio=1` ⇒ `obj=A` ⇒ `-mean(A)=0`，A 是 z 化量），故诊断另给
    **非零**的 `grad_norm`（= 裁剪前的 `‖∂L/∂θ‖`）作为"训练在不在动"的读数。
    """
    if layout.layout_seed != 0:
        raise ValueError(
            f"layout_seed={layout.layout_seed} ≠ 0：奖励权重（ReferenceObjectives.of 固定 "
            "seed_layout=0）与交期 M_ref（_due_map 用布局种子）会不同源，目标口径静默错位。"
            "请传 seed=0 的布局；确需非 0 种子，先扩展 ReferenceObjectives 记录它。")
    if epochs > 1 and clip_eps is None:
        raise ValueError("epochs>1 必须配 clip_eps：无裁剪时同一批数据重复计算，"
                         "结果与 epochs=1 相同（纯浪费）。")
    if clip_eps is not None and epochs <= 1:
        raise ValueError("epochs=1 配 clip_eps 是空转：new 与 logp_old 同参数算出 ⇒ ratio≡1 "
                         "⇒ clamp 恒等。请用 epochs>1，或 clip_eps=None（纯组内 REINFORCE）。")
    if policy.optim is None:
        policy.optim = torch.optim.Adam(policy.parameters(), lr=lr)

    chains: list[list[Decision]] = []
    rewards: list[float] = []
    for g in range(G):
        dec, met = roll_chain(inst, layout, dm, cfg, policy, seed * 1000 + g, ctx, sample=True)
        chains.append(dec)
        rewards.append(scalar_reward(objective_vector(met), w))

    A = torch.tensor(_z(np.asarray(rewards, dtype=np.float64), "z"), dtype=torch.float32)

    # 裁剪路径的 ratio 基准 = 采样那一刻的 logp（`Decision.logp`，无梯度）——不重算，
    # 且与 `chain_logp` 逐位同源（见 `sampled_logp`），故首轮 ratio 严格 = 1。
    old = torch.stack([sampled_logp(ch) for ch in chains]) if clip_eps is not None else None

    loss_val, ratio_mean, clipped_frac, grad_norm = 0.0, 1.0, 0.0, 0.0
    for _ in range(max(epochs, 1)):
        new = torch.stack([chain_logp(d, policy) for d in chains])
        if clip_eps is None:
            obj = A.detach() * new           # 纯组内 REINFORCE（理论骨架的「纯版本」）
        else:
            ratio = torch.exp(new - old)     # 链概率比（求和口径才成立，约定 2）
            obj = torch.min(ratio * A.detach(),
                            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A.detach())
            ratio_mean = float(ratio.mean().item())
            # 出界比例 = 该轮 surrogate **失去梯度**的链占比（1.0 = 全裁 ⇒ 这一轮纯浪费）
            clipped_frac = float(((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps))
                                 .float().mean().item())
        loss = -obj.mean()
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
