"""SA-GRPO 组训练器 —— M3a 版（《方法设计文档》§3.3/§3.4 的落地子集）。

范围（诚实声明，与设计文档的差异显式化）：
- 本版树 = **G 个机台计划（S 层采样）× J 个扰动实现**：叶子=(plan, seed_chain) 一次完整 episode。
  J=1 ⇒ 纯组内相对 GRPO（G 条链，组内 z 化）——退化验证口径。
  B 层（分批门控）+ L 层（AGV 选择）+ 每决策时刻树 = M3b。
- 层内归一化（BranchGRPO Eq.4 同构、跨父级聚合）：Ã = (r−μ)/σ；
  mode='z'（std 化=GRPO 式）/ 'mean' / 'loo'（留一法=RLOO 式）——消融⑧ 的三种基线。
- 更新：组内相对梯度（A 用 detach）；M3b-③ = 批量重用（K-epoch）+ PPO 式裁剪代理（clip/KL 为实践变体；
  理论骨架 §3.3 的「纯版本」= epochs=1, clip_eps=None，不含 KL/IS）。
- 声明纪律（§3.3）：只有组内方差/尺度比较，不断言"更优"。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .policy import PolicyNet
from ..env.des import rollout_evaluate
from ..env.instances import Instance


def _z(vals: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return vals - vals.mean()
    if mode == "loo":
        n = len(vals)
        if n <= 1:
            return vals.copy()
        return np.array([vals[i] - (vals.sum() - vals[i]) / (n - 1) for i in range(n)])
    return (vals - vals.mean()) / (vals.std() + 1e-9)


def _op_feat(oi: int, n_ref: int = 8) -> torch.Tensor:
    return torch.tensor([[[float(oi) / n_ref, 0.0, 0.0]]], dtype=torch.float32)  # (1,1,F)


def _cand_feat(alts, t_max: float) -> torch.Tensor:
    return torch.tensor([[[t / t_max, 0.0, 0.0] for _, t in alts]], dtype=torch.float32)  # (1,Ncand,F)


def _op_logits(policy: PolicyNet, oi: int, alts, t_max: float, tok_emb=None) -> torch.Tensor:
    """候选机台分数统一入口（M3c）：MLP 特征路径 vs 编码器 token 路径。返回 (1,1,Ncand)。

    tok_emb = policy.encode_state(...) 结果 (1,N,d)（M 段在序列前部：机台编号即序列位置）。
    """
    if tok_emb is not None:
        cand = torch.tensor([m for m, _ in alts])
        return policy.mach_logits_emb(tok_emb, _op_feat(oi), cand)
    return policy.mach_logits(_op_feat(oi), _cand_feat(alts, t_max))


def sample_plan(inst: Instance, policy: PolicyNet, t_max: float, tok_emb=None) -> list[int]:
    plan = []
    for alts_list in inst.jobs:
        ops = []
        for oi, alts in enumerate(alts_list):
            logits = _op_logits(policy, oi, alts, t_max, tok_emb)                # (1,1,Ncand)
            p = torch.softmax(logits.flatten(), -1)
            ops.append(int(torch.multinomial(p, 1).item()))    # torch（np.multinomial 精度
        plan.append(ops)                                            # sum≤1 崩溃=bug#6 收尾）
    return plan


def _l_feat(frm: int, to: int, oi: int, n_m: int, n_ref: int = 8,
            load_d: float | None = None, pos0: int = -1, pos1: int = -1) -> torch.Tensor:
    """L 层任务特征：任务 3 维 + 车状态 3 维（预演流时点：负载差/两车最近位置）。

    load_d=None → 回退纯任务特征（向后兼容旧调用方）。
    """
    if load_d is None:
        return torch.tensor([[[float(frm) / n_m, float(to) / n_m, float(oi) / n_ref]]],
                            dtype=torch.float32)
    return torch.tensor([[[float(frm) / n_m, float(to) / n_m, float(oi) / n_ref,
                           float(load_d) / 10.0, float(pos0) / n_m, float(pos1) / n_m]]],
                        dtype=torch.float32)


def local_l_step(policy: PolicyNet, inst: Instance, seed: int, J: int = 1,
                 mode: str = "z", lr: float = 1e-3, layout_type: str = "line",
                 seed_layout: int = 1, t_max: float = 10.0, n_agv: int = 2,
                 dec_idx: int | None = None, enc_state=None) -> tuple[float, dict]:
    """M3b-②-L：L 层（AGV 分派）决策点局部树 —— SA-GRPO 的物流轴（层级因果隔离）。

    结构（与 local_tree_step 同构，决策轴 = 运输任务→AGV 选择）：
      1) 计划语境 pl ~ π_S（每步重采）；预演流：轮询 agv_phi 一次 rollout → stats.task_flow
         [(job, oi_next, frm, to), ...]（des.py 新导出；任务流形状与基流紧邻）
      2) 基序列 agv_seq：π_L 按流逐任务采样（长度=|flow|；超出补轮询）
      3) 基 rollout（agv_phi=agv_seq）→ 基 makespan + 实任务流（用于决策点特征）
      4) 决策点 t（随机或指定）：候选 = n_agv 台车（枚举/A≤2 精确）；局部树 =
         仅替换 agv_seq[t] → 每候选 J 独立扰动流（A1）→ 确定性 rollout
      5) 组 = 候选节点（J 均值）；A = 组内 z 化（BranchGRPO Eq.4 同构，归一化域=同一任务点）；
         更新仅该任务 π_L logp（L 层因果隔离 → 组内优势 ≈ LoTV 物流层分量 Δ_L 的估计）。

    诚实注记（v0）：任务流以基序列为准（变体流因时序微差可能移位，视为紧邻近似）；与
      local_tree_step（S 轴）同接口、可交替训练（= 三轴 SA-GRPO 分层归因的组合实施）。

    消融⑥ 映射：本函数 = "每任务点物流树"；轮询/贪婪 = 频率下限。
    """
    from ..env.des import rollout
    n_m = inst.n_machines
    pl = sample_plan(inst, policy, t_max, enc_state)
    ph = [i % n_agv for i in range(64)]
    prev = rollout(inst, layout_type=layout_type, seed_layout=seed_layout, seed_chain=seed,
                   op_choices=pl, agv_phi=ph)
    flow = prev.get("task_flow", [])
    if not flow:
        return float(-prev["makespan"]), {"loss": 0.0, "r_mean": float(-prev["makespan"]),
                                          "dec": None, "cands": list(range(n_agv)),
                                          "p": [0.0] * n_agv, "A_std": 0.0}
    rng = np.random.default_rng(seed)
    logits_list = [policy.agv_logits(_l_feat(frm, to, oi, n_m)).flatten().detach()
                   for (_, oi, frm, to) in flow]
    agv_seq = [int(torch.multinomial(torch.softmax(lg, -1), 1).item())
               for lg in logits_list]
    base = rollout(inst, layout_type=layout_type, seed_layout=seed_layout, seed_chain=seed * 1000,
                   op_choices=pl, agv_phi=agv_seq)
    flow = base.get("task_flow", flow)
    t = int(rng.integers(0, len(flow))) if dec_idx is None else dec_idx
    job, oi, frm, to = flow[t]
    leaves: list[tuple[int, int, float]] = []
    logp_b: list[torch.Tensor] = []
    for b in range(n_agv):
        seq2 = agv_seq[:]
        seq2[t] = b
        for jj in range(J):
            r = -rollout(inst, layout_type=layout_type, seed_layout=seed_layout,
                         seed_chain=seed * 1000 + b * 100 + jj, op_choices=pl,
                         agv_phi=seq2)["makespan"]
            leaves.append((b, jj, r))
        logits = policy.agv_logits(_l_feat(frm, to, oi, n_m))[0, 0]          # (n_agv,) 带图
        logp_b.append(F.log_softmax(logits, -1)[b])
    vB = np.array([np.mean([r for (bb, _, r) in leaves if bb == b]) for b in range(n_agv)])
    A = _z(vB, mode)
    if A.std() < 1e-12:
        A = np.zeros_like(A)
    loss = -(torch.tensor(A, dtype=torch.float32) * torch.stack(logp_b)).mean()
    loss.backward()
    with torch.no_grad():
        for p in policy.parameters():
            if p.grad is not None:
                p.grad.clamp_(-1.0, 1.0)
                p.add_(-lr * p.grad)
        policy.zero_grad()
    p_c = F.softmax(policy.agv_logits(_l_feat(frm, to, oi, n_m)).detach()[0, 0], -1)
    diag = {"loss": float(loss.item()), "r_mean": float(np.mean([r for (_, _, r) in leaves])),
            "A_std": float(A.std()), "dec": t, "cands": list(range(n_agv)),
            "job_oi": (job, oi), "vB": [float(x) for x in vB],
            "p": [round(float(x), 3) for x in p_c]}
    return float(np.mean([r for (_, _, r) in leaves])), diag


def l_seq_step_gated(policy: PolicyNet, inst: Instance, seed: int, J: int = 4,
                     mode: str = "z", lr: float = 1e-3, layout_type: str = "line",
                     seed_layout: int = 1, t_max: float = 10.0, n_agv: int = 2,
                     enc_state=None, adv_mode: str = "z", ent_beta: float = 0.0) -> tuple[float, dict]:
    """L 层完整版：真·门控事件驱动 + 序列级组内相对（闭环决策，运行流内实时状态）。

    与 l_seq_step（预演态，已证伪）的机制区别：policy_l 由 transporter 在 SimPy
    运行流内【同步】调用 —— 特征 = 决策时刻的实时 stats（无预演、无分布漂移、
    无自指）；决策日志随 run_gated 回传。
    每叶 = 一次 run_gated（同扰动种子 seed*1000+7 对照 → 组内差=纯序列差）；
    更新 = π_L 在当前策略上对决策日志重算 logp（feat 存于日志）→ 序列级组内 REINFORCE。
    """
    from ..env.des import SimWorld, SimConfig
    from ..env.corridors import build_corridor_graph, dock_distance_matrix
    from ..env.layout import sample_layout
    n_m = inst.n_machines
    layout = sample_layout(n_m, seed=seed_layout, n_agv=n_agv)
    dm = dock_distance_matrix(build_corridor_graph(layout))
    world = SimWorld(inst, layout, dm, SimConfig(n_agv=n_agv))
    pl = sample_plan(inst, policy, t_max, enc_state)

    def make_sampler(sample: bool):
        def f(feat_np):
            lg = policy.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0].detach()
            return int(torch.multinomial(F.softmax(lg, -1), 1).item()) if sample \
                else int(torch.argmax(lg).item())
        return f

    outs = []
    for _j in range(J):
        out = world.run_gated(seed_chain=seed * 1000 + 7, op_choices=pl,
                              policy_l=make_sampler(sample=True))
        outs.append(out)
    rvals = [-o["makespan"] for o in outs]
    if adv_mode == "median":                       # MC-GRPO（Kim 2026）：r - median(r)，对
        A = np.array(rvals) - np.median(rvals)     # 长尾离群免疫（小 rollout 均值基线退化修复）
    else:
        A = _z(np.array(rvals), mode)
    if A.std() < 1e-12:
        A = np.zeros_like(A)
    logp_js = []
    for out in outs:
        dl = out.get("decision_log", [])
        lp = 0.0
        for (_info, feat, agv) in dl:
            logits = policy.agv_logits(torch.tensor(feat[None, None, :]))[0, 0]     # 带图
            lp = lp + F.log_softmax(logits, -1)[agv]
        logp_js.append(lp / max(len(dl), 1))
    loss = -(torch.tensor(A, dtype=torch.float32) * torch.stack(logp_js)).mean()
    if ent_beta:                              # 熵正则：防策略塌缩/退化（玩具验证：稳定修件）
        ent = 0.0
        n_dec = 0
        for out in outs:
            for (_info, feat, _agv) in out.get("decision_log", []):
                ft = torch.tensor(feat[None, None, :])
                p = F.softmax(policy.agv_logits(ft).detach()[0, 0], -1)
                ent = ent - (p * torch.log(p + 1e-9)).sum()
                n_dec += 1
        loss = loss - ent_beta * (ent / max(n_dec, 1))
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)      # 标准口径（LLM GRPO 用 AdamW，
    loss.backward()                                          # 手写 SGD+clamp 在噪声下振荡=弃用）
    opt.step()
    opt.zero_grad()
    diag = {"loss": float(loss.item()), "r_mean": float(np.mean(rvals)),
            "r_std": float(np.std(rvals)), "A_std": float(A.std()),
            "J_leaves": J, "dec_len": len(outs[0].get("decision_log", []))}
    return float(np.mean(rvals)), diag


def l_seq_step(policy: PolicyNet, inst: Instance, seed: int, J: int = 4,
               mode: str = "z", lr: float = 1e-3, layout_type: str = "line",
               seed_layout: int = 1, t_max: float = 10.0, n_agv: int = 2,
               enc_state=None) -> tuple[float, dict]:
    """L 层序列级组内相对（**L 层的正确归一化域**）：J 个完整 AGV 分派序列为一组。

    为什么不是任务点级（local_l_step 教训，如实存档）：
      物流层是**非分量可加地形**（任务分配互为替代品，全局最优=平衡；单任务点边缘值
      会把所有任务推向同一辆车 = 拥塞灾难，200 步实验 makespan 582-650 vs 轮询 347.9）。
      → L 层的组 = **整个序列**（J 个平行分配方案比较），正对应树的 L 层分支。

    结构：
      1) 预演流（轮询 1 次 rollout → stats.task_flow；形状近似即可）
      2) J 个 agv_seq ~ π_L（逐任务采样）；**同扰动种子**比较（seed_chain 固定 → 组内差=
        纯序列差，与"扰动独立流 A1"的口径差异在 docstring 注明：序列级对照用同流，更纯净）
      3) r_j = rollout(pl, seq_j, seed_chain=SEED_FIXED)；A = z(r)（组内归一化域=序列集）
      4) 更新：logp_seq_j = Σ_t log π_L(seq_j[t]|task_t)（带图）；loss = -(A_j·logp_j).mean()
    """
    from ..env.des import rollout
    n_m = inst.n_machines
    pl = sample_plan(inst, policy, t_max, enc_state)
    prev = rollout(inst, layout_type=layout_type, seed_layout=seed_layout, seed_chain=seed,
                   op_choices=pl, agv_phi=[i % n_agv for i in range(64)])
    flow = prev.get("task_flow", [])
    loads = prev.get("agv_load", [])
    if not flow:
        return float(-prev["makespan"]), {"loss": 0.0, "r_mean": float(-prev["makespan"]),
                                          "A_std": 0.0, "J_leaves": 0}

    def task_feat(i: int, frm, to, oi):
        if i < len(loads):
            (d0, d1), (p0, p1) = loads[i]
            return _l_feat(frm, to, oi, n_m, load_d=float(d0 - d1), pos0=int(p0), pos1=int(p1))
        return _l_feat(frm, to, oi, n_m)

    logits_all = [policy.agv_logits(task_feat(i, frm, to, oi)).flatten().detach()
                  for i, (_, oi, frm, to) in enumerate(flow)]
    seqs, rvals, logp_seqs = [], [], []
    for _j in range(J):
        seq = [int(torch.multinomial(torch.softmax(lg, -1), 1).item()) for lg in logits_all]
        seqs.append(seq)
        r = -rollout(inst, layout_type=layout_type, seed_layout=seed_layout,
                     seed_chain=seed * 1000 + 7,            # 同扰动种子（组内纯序列差）
                     op_choices=pl, agv_phi=seq)["makespan"]
        rvals.append(r)
        lp = 0.0
        for i, ((_, oi, frm, to), a) in enumerate(zip(flow, seq)):
            logits = policy.agv_logits(task_feat(i, frm, to, oi))[0, 0]            # 带图
            lp = lp + F.log_softmax(logits, -1)[a]
        logp_seqs.append(lp / max(len(flow), 1))
    A = _z(np.array(rvals), mode)
    if A.std() < 1e-12:
        A = np.zeros_like(A)
    loss = -(torch.tensor(A, dtype=torch.float32) * torch.stack(logp_seqs)).mean()
    loss.backward()
    with torch.no_grad():
        for p in policy.parameters():
            if p.grad is not None:
                p.grad.clamp_(-1.0, 1.0)
                p.add_(-lr * p.grad)
        policy.zero_grad()
    diag = {"loss": float(loss.item()), "r_mean": float(np.mean(rvals)),
            "r_std": float(np.std(rvals)), "A_std": float(A.std()), "J_leaves": J,
            "flow_len": len(flow)}
    return float(np.mean(rvals)), diag


def train_step(policy: PolicyNet, inst: Instance, seed: int, G: int = 8, J: int = 1,
               mode: str = "z", lr: float = 1e-3, layout_type: str = "line",
               seed_layout: int = 1, t_max: float = 10.0, epochs: int = 1,
               clip_eps: float | None = None, kl_beta: float = 0.0,
               enc_state=None) -> tuple[float, dict]:
    """一步组训练。叶子=(plan_g, seed_chain_j)→r；节点组=G 计划（J 个扰动取均值）。

    M3b-③（批量重用裁剪版）：同批 K-epoch 更新，PPO 式裁剪代理（GRPO 实践）——
      epochs=1 且 clip_eps=None ⇒ 纯组内 REINFORCE（M3a 精确行为；理论骨架 §3.3「纯版本」）。
      epochs>1 且 clip_eps=0.2 ⇒ K-epoch 裁剪（A 与 logp_old 冻结；理论不含 KL/IS——简化假设）。
    返回 (组均值 r, 诊断 dict：loss / ratio / Ã std / r 均值与 std)。
    """
    tok_emb = policy.encode_state(enc_state) if enc_state is not None else None  # (1,N,d) 带图
    leaves: list[tuple[int, int, float]] = []      # (g, j, r)
    logp_old: list[float] = []
    plans: list[list[int]] = []
    for g in range(G):
        pl = sample_plan(inst, policy, t_max, tok_emb)
        plans.append(pl)
        for j in range(J):
            r = rollout_evaluate(inst, pl, seed=seed * 1000 + g * 100 + j,
                                 layout_type=layout_type, seed_layout=seed_layout)
            leaves.append((g, j, r))
        with torch.no_grad():
            logp_old.append(float(_plan_logp(inst, pl, policy, t_max, tok_emb)))
    vG = np.array([np.mean([r for (gg, _, r) in leaves if gg == g]) for g in range(G)])
    vL = np.array([r for (_, _, r) in leaves])
    A_G = _z(vG, mode)                       # 层内（G 计划）归一化
    A_t = torch.tensor(A_G, dtype=torch.float32)
    old_t = torch.tensor(logp_old, dtype=torch.float32)
    ratio_mean = 1.0
    loss = 0.0
    for _ in range(max(epochs, 1)):          # 批量重用：同批 K-epoch（GRPO 标准做法）
        new_t = torch.stack([_plan_logp(inst, pl, policy, t_max, tok_emb) for pl in plans])
        if clip_eps is not None:
            ratio = torch.exp(new_t - old_t)
            obj = torch.min(ratio * A_t,
                            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A_t)
            ratio_mean = float(ratio.mean().item())
        else:
            obj = A_t * new_t                # 纯组内 REINFORCE
        loss = -obj.mean()
        if kl_beta:
            d = new_t - old_t
            loss = loss + kl_beta * (torch.exp(-d) - 1.0 + d).mean()   # KL(π‖π_old) 非负路径估计
        loss.backward()
        with torch.no_grad():
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.clamp_(-1.0, 1.0)
                    p.add_(-lr * p.grad)
            policy.zero_grad()
    diag = {"loss": float(loss.item()), "ratio": ratio_mean,
            "r_mean": float(vL.mean()), "r_std": float(vL.std()),
            "A_std_G": float(A_G.std()), "logp_min": float(min(old_t))}
    return float(vL.mean()), diag


def _plan_logp(inst: Instance, plan: list[int], policy: PolicyNet, t_max: float,
               tok_emb=None) -> torch.Tensor:
    """整计划 log 概率（**带梯度**——用当前网络重算采样的 log 概率，供 REINFORCE 更新）。"""
    total, n = None, 0
    for j, alts_list in enumerate(inst.jobs):
        for oi, alts in enumerate(alts_list):
            logits = _op_logits(policy, oi, alts, t_max, tok_emb)                # (1,1,Ncand)
            lp = torch.log_softmax(logits, -1)[0, 0, plan[j][oi]]
            total = lp if total is None else total + lp
            n += 1
    return total / max(n, 1)
