"""标准 SA-GRPO（设计文档 §3.3/§3.4 树式版）——"状态克隆树"实现（2026-09-07）。

「标准版」与已证伪变体的机制区分（逐条对 §3.4 伪代码）：
- **三层同树、同一步更新**：π=π_B·π_S·π_L 条件链树。此前 tree_step 只覆盖 B×S（L 不在树内，
  L 无梯度）；l_seq_step_gated 只覆盖 L 且 S 每次重采 —— 树被拆成两个独立训练器（"半树"）。
- B 层 G 支 = π_B 的 G 个 iid 采样（§3.4 字面「b_1..b_G ← iid π_B(·|x)」）；
  S 层每 b 下 J 支 = π_S(·|x,b) 的 J 个 iid 全计划采样；
  L 层每 (b,s) 下 J 支 = π_L(·|x,b,s) 的 J 个 iid AGV 序列（**真门控**：run_gated 运行内实时决策，
  非预演流——预演流版 A' 已证伪：大问题 668-850）。
- **每叶 = 世界继续一次完整 episode + 扰动流**：标准 = 全树克隆同一 ω（"分支共享前缀+扰动"，
  同 372.7 口径；pert_shared=True 默认）；每叶独立 ω = 消融⑨ 变体（骨架 §5-3 ❓建议）。
- 子树融合：节点值 r̄ = 后代叶均匀融合（无偏，Var=σ²/|叶|，BranchGRPO B.2）；
  深度 z 化 = **同父内聚合**（修正版标准：§3.3 基线判据"只依赖该层条件变量"的唯一一致口径；
  §3.4 伪代码"跨父级聚合"与判据矛盾=第 9 号 bug，cross 保留为勘测变体（norm="cross"））。
- 分层更新：θ += α·(∇ℓ(π_B,Ã_B) + ∇ℓ(π_S,Ã_S) + ∇ℓ(π_L,Ã_L))，纯 REINFORCE
  （epochs=1、无 clip/KL —— §3.3「纯净版」）；优化器 AdamW（口径同 L 层标准件）。
- 等预算：总叶数 G·J² = 对照 GRPO 链数。J=1 ⇒ 三层组同一组叶，但损失仍为三分量之和
  （分层 REINFORCE 本身，与单一 z 化的 GRPO 差异=层间归因路径）。

判定锚（记录 §一，布局 seed=1）：规则 load_min 457.1 / L 组内 GRPO 372.7。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..env.des import SimWorld, SimConfig
from ..env.corridors import build_corridor_graph, dock_distance_matrix
from ..env.layout import sample_layout
from ..env.instances import Instance
from .policy import PolicyNet
from .group_rel import _z, _mode_feat, _plan_logp, _op_logits


def _sample_plan(policy: PolicyNet, inst: Instance, t_max: float,
                 tok_emb=None) -> list[int]:
    """整计划采样（torch.multinomial 版——numpy 精度 sum≤1 崩溃教训，不再用 np）。"""
    plan = []
    for alts_list in inst.jobs:
        ops = []
        for oi, alts in enumerate(alts_list):
            logits = _op_logits(policy, oi, alts, t_max, tok_emb)                 # (1,1,Ncand)
            p = torch.softmax(logits.flatten(), -1)
            ops.append(int(torch.multinomial(p, 1).item()))
        plan.append(ops)
    return plan


def standard_tree_step(policy: PolicyNet, inst: Instance, seed: int = 0,
                       G: int = 4, J: int = 2, J_L: int | None = None, mode: str = "z",
                       lr: float = 1e-4,
                       layout_type: str = "line", seed_layout: int = 1,
                       t_max: float = 10.0, n_agv: int = 2, enc_state=None,
                       cap_opts: list[int | None] | None = None,
                       ent_beta: float = 0.0, pert_shared: bool = True,
                       norm: str = "within", reward_mode: str = "mks",
                       lam: float | None = None,
                       w_channels: tuple[float, float, float, float] | None = None,
                       topo_random: bool = False,
                       layout_obj=None, n_zones: int = 3
                       ) -> tuple[float, dict]:
    """标准 SA-GRPO 一步（§3.4 全树）：G·J² 叶；三层**同父内**归一化；三分量 REINFORCE。

    Args:
        G: B 层（π_B）iid 支数；J: S 层每支下 iid 数；J_L: L 层每 (b,s) 下叶数（默认=J）。
          总叶数 = G·J·J_L（等预算口径对照 GRPO 的 G 链——J=J_L=1 ⇒ 退化）。
        norm: "within"（默认，**修正版标准**）= 每层在**同父组**内 z 化——§3.3 基线判据
          （"基线只依赖该层条件变量"）的唯一一致口径：S 层组=(x,b_g) 下 J 支、L 层组=
          (x,b_g,s_{g,j}) 下 J 叶；B 层无父，"within" 平凡等于 "cross"。LoTV 条件塔
          （E[r|x]→E[r|x,b]→E[r|x,b,s]）的归因=条件化差异 ⇒ 同父内。
          "cross"（§3.4 伪代码字面"跨父级聚合"）= 勘测变体：与 §3.3 判据矛盾（基线混入
          其它父的条件），实测 S/L 层学不动、B 层独活（B 无父=平凡满足）——留作消融。
          "flat" = **标准 GRPO 对照**：优势 = 全部叶全局 z（无分层域）；链 logp/B/S/L 段、
          采样结构、预算与树完全一致 —— 唯一的机制差异 = 优势的域（决定"树是否有增益"）。
        ent_beta: L 头熵正则（历史教训：无正则小 rollout 下学习-退化循环；0=纯净版）。
        pert_shared: True（默认，**修正版标准**）= 全树克隆同一扰动流（记录 §七"分支共享
          前缀+扰动"；与 372.7 成功案例同口径：J 序列同种子 seed*1000+7 → 组内差=纯决策差，
          小组宽下 ω 噪声不致淹没信噪）。False = 每叶独立 ω（理论骨架 §5-3 的 ❓ 建议 =
          消融⑨ 变体；A1 原文仅要求采样源与 ω 独立，未要求叶间 ω 独立——两口径均满足 A1）。
    Returns:
        (叶均值 r, diag) —— diag 含三层 σ/比率/各头梯度范数（防"假平台"：头无梯度=实现缺陷）。
    """
    n_m = inst.n_machines
    jl = J if J_L is None else J_L          # L 层每 (b,s) 叶数（独立于 S 层 J——组内 L 归因样本数）
    caps = cap_opts or [None, max(2, inst.n_jobs // 2), max(2, inst.n_jobs // 4)]
    if layout_obj is not None:              # Phase A 共演化：外部构图（通道宽/间距/拥塞档可控）
        layout = layout_obj
    elif topo_random:                       # 训练期随机拓扑：x=布局 非退化 → LoTV 层级塔恢复
        from ..nn.state_emb import encode_state
        lt = ("line", "U", "island")[seed % 3]
        sl = int((seed // 3) % 10) + 1
        layout = sample_layout(n_m, lt, sl)
        enc_state = encode_state(inst, layout, n_agv)
    else:
        layout = sample_layout(n_m, layout_type, seed_layout)
    dm = dock_distance_matrix(build_corridor_graph(layout))
    world = SimWorld(inst, layout, dm, SimConfig(n_agv=n_agv, n_zones=n_zones,
                                                 aisle_width=layout.aisle_width))
    fb = _mode_feat(inst)                                                     # (1,4)
    if topo_random:                     # B 层特征扩展：布局类型 one-hot（否则 B 看不到拓扑
        lt_idx = ("line", "U", "island").index(lt)                            # → 无法条件响应，
        fb = torch.cat([fb, torch.nn.functional.one_hot(                       #   c4 ratio≈1 系
            torch.tensor([lt_idx]), num_classes=3).float()], dim=-1)          #   实现局限）
    tok_emb = policy.encode_state(enc_state) if enc_state is not None else None

    # —— 第 1 层 B：G 个 iid π_B 采样（3 模式可重复）——
    with torch.no_grad():
        pb = F.softmax(policy.batch_logits(fb), -1)[0]                        # (n_modes,)
        b_idx = torch.multinomial(pb, G, replacement=True)                    # (G,)
        b_vals = [caps[int(i)] for i in b_idx]

    def make_sampler():
        def f(feat_np):
            lg = policy.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0].detach()
            return int(torch.multinomial(F.softmax(lg, -1), 1).item())
        return f

    # —— 第 2/3 层 + 叶评估（每叶独立扰动流 A1）——
    plans: list[list[int]] = []
    leaves: list[tuple[int, int, int, float, list]] = []   # (g, j, k, r, decision_log)
    for g in range(G):
        for j in range(J):
            pl = _sample_plan(policy, inst, t_max, tok_emb)
            plans.append(pl)
            for k in range(jl):
                leaf_seed = seed * 1_000_000 + ((g * J + j) * jl + k)         # A1：每叶独立 ω
                if pert_shared:
                    leaf_seed = seed * 1000 + 7                               # 消融⑨：共享流
                out = world.run_gated(seed_chain=leaf_seed, op_choices=pl,
                                      batch_cap=b_vals[g], policy_l=make_sampler())
                assert out["jobs_done"] == inst.n_jobs and not out["horizon_hit"], \
                    f"叶 {(g, j, k)}: jobs={out['jobs_done']}/{inst.n_jobs} " \
                    f"horizon_hit={out['horizon_hit']} —— 运行不完整（仿真/B 门控缺陷），奖励无效"
                met = (out["makespan"], out["tardy"], out["moves"], out["energy"])
                leaves.append((g, j, k, None, out.get("decision_log", []), met))

    # —— 叶奖励映射：reward_mode="mks"=单指标；"channels"=每通道独立 z 后等权和（MO-GRPO/
    # Multi-GRPO reward-grouping 同构：各指标方差对齐、冲突信号解耦、免手调权重）——
    met = np.array([m for (_, _, _, _, _, m) in leaves], dtype=np.float32)        # (N,4)
    if reward_mode == "channels":
        sc = np.zeros(len(leaves))
        for kk in range(4):
            ck = met[:, kk]
            if ck.std() > 1e-9:
                w = w_channels[kk] if w_channels else 1.0
                sc -= w * _z(ck, mode)      # 越小越好 → -z（z 奇对称：z(-c)=-z(c)）；w=逐通道权重
    else:
        sc = -met[:, 0]
    leaves = [(g, j, k, float(v), dl, met[i])
              for i, (g, j, k, _, dl, _) in enumerate(leaves) for v in [sc[i]]]

    # —— 子树融合 + 同父内深度 z 化（修正版标准；norm="cross" = 勘测变体）——
    rLeaf = np.array([r for (_, _, _, r, _, _) in leaves])
    rS = np.array([np.mean([r for (gg, jj, _, r, _, _) in leaves if gg == g and jj == j])
                   for g in range(G) for j in range(J)])
    rB = np.array([np.mean([r for (gg, _, _, r, _, _) in leaves if gg == g]) for g in range(G)])
    if norm == "cross":
        A_B, A_S, A_L = _z(rB, mode), _z(rS, mode), _z(rLeaf, mode)
        for A in (A_B, A_S, A_L):
            if A.std() < 1e-12:
                A[:] = 0.0
    elif norm == "flat":                          # 标准 GRPO 对照：全局 z（无分层域）
        A_flat = _z(rLeaf, mode)
        if A_flat.std() < 1e-12:
            A_flat[:] = 0.0
        A_B = A_S = A_L = A_flat
    else:                                         # within：每层同父组（B 无父=整组）
        A_B = _z(rB, mode)
        if A_B.std() < 1e-12:
            A_B[:] = 0.0
        segs_S, segs_L = [], []
        for g in range(G):
            seg = _z(rS[g * J:(g + 1) * J], mode)
            segs_S.append(seg)
            for j in range(J):
                segl = _z(rLeaf[(g * J + j) * jl:(g * J + j + 1) * jl], mode)
                segs_L.append(segl)
        A_S = np.concatenate(segs_S)
        A_L = np.concatenate(segs_L)

    # —— 分层 REINFORCE（三个可学头同一步更新）——
    logp_B = F.log_softmax(policy.batch_logits(fb), -1)[0][b_idx]             # (G,) 带图
    logp_S = torch.stack([_plan_logp(inst, pl, policy, t_max, tok_emb) for pl in plans])
    lps = []
    for (_, _, _, _, dl, _) in leaves:
        T = len(dl)
        lp = torch.zeros(())
        n = 0.0
        for t_idx, (_info, feat, agv) in enumerate(dl):
            w = (lam ** (T - 1 - t_idx)) if lam is not None else 1.0   # GRPO-λ：近端决策权重高
            ft = torch.tensor(feat[None, None, :])
            lp = lp + w * F.log_softmax(policy.agv_logits(ft)[0, 0], -1)[agv]
            n += w
        lps.append(lp / max(n, 1.0))
    logp_L = torch.stack(lps)                                                 # (G·J·jl,)
    if norm == "flat":                        # 链级 logp = B 段(按叶展开) + S 段 + L 段
        A_t = torch.tensor(A_L, dtype=torch.float32)                          # (G·J·jl,)
        lp_chain = (logp_B.repeat_interleave(J * jl) + logp_S.repeat_interleave(jl)
                    + logp_L)
        loss = -(A_t * lp_chain).mean()
    else:
        loss = -((logp_B * torch.tensor(A_B, dtype=torch.float32)).mean()
                 + (logp_S * torch.tensor(A_S, dtype=torch.float32)).mean()
                 + (logp_L * torch.tensor(A_L, dtype=torch.float32)).mean())
    if ent_beta:                       # L 头熵正则（历史稳定件；0=纯净版）
        ent = torch.zeros(())
        n_dec = 0
        for (_, _, _, _, dl, _) in leaves:
            for (_info, feat, _agv) in dl:
                ft = torch.tensor(feat[None, None, :])
                p = F.softmax(policy.agv_logits(ft).detach()[0, 0], -1)
                ent = ent - (p * torch.log(p + 1e-9)).sum()
                n_dec += 1
        loss = loss - ent_beta * (ent / max(n_dec, 1))
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    loss.backward()
    gn = {}
    for name, mod in (("b", policy.b_head), ("s", policy.s_head),
                      ("stok", getattr(policy, "s_head_tok", None)),
                      ("l", policy.l_head), ("enc", policy.enc)):
        if mod is None:
            continue
        tot = sum(float(p.grad.detach().norm() ** 2)
                  for p in mod.parameters() if p.grad is not None)
        gn["g_" + name] = round(float(tot ** 0.5), 4)
    opt.step()
    opt.zero_grad()
    pB = torch.softmax(policy.batch_logits(fb), -1).detach().numpy()[0]
    diag = {"loss": float(loss.item()), "r_mean": float(rLeaf.mean()),
            "r_std": float(rLeaf.std()),
            "sig_B": round(float(np.std(rB, ddof=1) if len(rB) > 1 else 0.0), 3),
            "sig_S": round(float(np.std(rS, ddof=1) if len(rS) > 1 else 0.0), 3),
            "sig_L": round(float(np.std(rLeaf, ddof=1) if len(rLeaf) > 1 else 0.0), 3),
            "ratio_S": round(float(np.std(rS, ddof=1) / (np.std(rLeaf, ddof=1) + 1e-9)), 3),
            "ratio_B": round(float(np.std(rB, ddof=1) / (np.std(rLeaf, ddof=1) + 1e-9)), 3),
            "A_std": round(float(A_L.std()), 3),
            "norm": norm,
            "pB": [round(float(x), 3) for x in pB],
            "b_modes": [int(i) for i in b_idx],
            "dec_len": len(leaves[0][4]) if leaves else 0,
            **gn}
    return float(rLeaf.mean()), diag
