"""联合链 GRPO 的测试（P2 Task 7，spec §5.3.4）。"""
from __future__ import annotations

import torch
import pytest

from plcsp.algo.group_rel import chain_logp, joint_chain_step, roll_chain
from plcsp.algo.policy import PolicyNet
from plcsp.env.des import SimConfig
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.env.reward import ReferenceObjectives, reward_weights
from plcsp.nn.encoder import LayoutEncoder


def _setup(name="mk01"):
    """返回 (inst, layout, dm, cfg, ctx, policy)——**六个**，与 roll_chain 的签名对齐。

    ⚠️ 布局 seed 固定 0：`SimWorld._due_map` 用**该布局**的 seed 取参考 makespan，而奖励侧
    `ReferenceObjectives.of` 固定用 seed_layout=0 的参考运行——两者同源才有一致的目标口径。
    """
    from plcsp.nn.state_emb import norm_context
    inst = load_mk(name)
    cfg = SimConfig()
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    pol = PolicyNet(enc=LayoutEncoder(), n_agv=cfg.n_agv)
    return inst, lay, dock_distance_matrix(g), cfg, norm_context(inst, lay, m_ref=100.0), pol


def _weights(inst, cfg):
    """三目标权重 wᵢ = (1/fᵢ^ref)/Σ(1/fⱼ^ref)（spec §5.3.3）。

    ⚠️ brief 的测试片段里 `w` **未定义**（`joint_chain_step` 直接引用了它）——此处补上。
    """
    return reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())


@pytest.mark.unit
def test_chain_has_one_decision_per_op_and_per_task():
    """一条链的决策数 = 工序数 + 运输任务数。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, metrics = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    n_ops = sum(len(j) for j in inst.jobs)
    kinds = [d.kind for d in decisions]
    assert kinds.count("S") == n_ops
    assert kinds.count("L") == metrics["deliveries"]


@pytest.mark.unit
def test_logp_is_sum_not_mean():
    """⚠️ Review Focus：logp 必须取**求和**——现状 S 用平均、L 用求和，同一次比较里口径不一致。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    lp = chain_logp(decisions, pol)

    # 求和口径：隐式验证——logp 的绝对值应随决策数线性增长，而不是被压在 O(1)
    decisions_k = decisions[: max(4, len(decisions) // 2)]
    lp_k = chain_logp(decisions_k, pol)
    # ⚠️ 用 `.item()` 而非 `float(...)`：后者对 requires_grad 张量发 UserWarning
    # （brief 里的 `float(...)` 写法会把测试输出弄脏）。
    assert abs(lp.item()) > abs(lp_k.item()), "logp 未随决策数增长——疑似仍取平均"


@pytest.mark.unit
def test_ratio_is_one_for_unchanged_policy():
    """同一策略、同一条链 → ratio 必须为 1（这是 PPO 裁剪的前提）。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    a = chain_logp(decisions, pol)
    b = chain_logp(decisions, pol)
    assert torch.exp(b - a).item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.unit
def test_gradient_flows_to_both_heads_and_encoder():
    """⚠️ Review Focus #5：一次更新必须同时给 S 头、L 头、编码器都留下非零梯度。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    chain_logp(decisions, pol).backward()
    for name, p in (("s_head_tok", pol.s_head_tok[0].weight),
                    ("l_head_tok", pol.l_head_tok[0].weight),
                    ("enc.embed", pol.enc.embed.weight),
                    ("enc.type_emb", pol.enc.type_emb)):
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} 无梯度"


@pytest.mark.unit
def test_l_decisions_alone_send_gradient_to_encoder():
    """⚠️ 变异验证补的守卫（Task 7）：把 **L 路径**的编码器梯度砍掉（`tok.detach()`），
    上一条测试**仍然全绿**——因为 S 决策照样把梯度送到编码器，它只证明了"存在通路"。
    而 spec §5.3.4 约定 5 要的恰恰是**两条**通路都在（否则"联合"只是名义上的）。
    故单列一条：**只用 L 决策**反传，编码器与 L 头都必须有非零梯度。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    l_only = [d for d in decisions if d.kind == "L"]
    assert len(l_only) >= 20, f"L 决策太少（{len(l_only)}）——断言失去意义"
    chain_logp(l_only, pol).backward()
    for name, p in (("l_head_tok", pol.l_head_tok[0].weight),
                    ("enc.embed", pol.enc.embed.weight),
                    ("enc.type_emb", pol.enc.type_emb)):
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} 无梯度（仅 L 决策反传）"


@pytest.mark.unit
def test_joint_step_uses_adam_and_returns_diagnostics():
    inst, lay, dm, cfg, ctx, pol = _setup()
    r, diag = joint_chain_step(pol, inst, lay, dm, seed=0, G=4, cfg=cfg, ctx=ctx,
                               w=_weights(inst, cfg))
    assert isinstance(pol.optim, torch.optim.Adam), "优化器应为 Adam（spec §5.3.4 第 4 条）"
    assert set(diag) >= {"loss", "ratio", "r_mean", "r_std", "A_std"}
    assert r == pytest.approx(diag["r_mean"])
