"""联合链 GRPO 的测试（P2 Task 7，spec §5.3.4）。"""
from __future__ import annotations

import copy

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
    pol = PolicyNet(enc=LayoutEncoder())
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
def test_same_seed_reproduces_action_sequence():
    """⚠️ 评审 I-3：同 seed 跑两次 `roll_chain`，动作序列必须**逐位相同**（可复现性判据）。

    旧实现的动作采样走**全局 torch RNG**（`torch.multinomial` 不传 generator）：`seed=` 只进
    仿真扰动流，锁不住动作——多进程各跑各的、"同 seed 对照 / 种子矩阵"两件事都不成立。
    判据取 `(kind, cand, action)` 三元组全覆盖（只看 action 会漏掉"候选集也变了"的形态）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    a, _ = roll_chain(inst, lay, dm, cfg, pol, seed=7, ctx=ctx)
    b, _ = roll_chain(inst, lay, dm, cfg, pol, seed=7, ctx=ctx)
    sig_a = [(d.kind, d.cand, d.action) for d in a]
    sig_b = [(d.kind, d.cand, d.action) for d in b]
    assert len(sig_a) >= 50, f"链太短（{len(sig_a)} 决策）——判据失去意义"
    assert sig_a == sig_b, "同 seed 两次的动作序列不同——采样流没接上 seed"


@pytest.mark.unit
def test_different_seeds_give_different_actions():
    """⚠️ 评审 I-3 的另一半：不同 seed ⇒ 动作序列必须**不同**（否则 seed 根本没接进采样）。

    没有这条，"同 seed 相同"会被"采样恒走一条与 seed 无关的固定序列"蒙混过关——那同样是
    不可复现的随机性（换 seed 也复现不出差别），且与"seed 矩阵"的语义直接冲突。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    a, _ = roll_chain(inst, lay, dm, cfg, pol, seed=7, ctx=ctx)
    c, _ = roll_chain(inst, lay, dm, cfg, pol, seed=8, ctx=ctx)
    sig_a = [(d.kind, d.cand, d.action) for d in a]
    sig_c = [(d.kind, d.cand, d.action) for d in c]
    assert sig_a != sig_c, "换 seed 动作序列不变——seed 没接进动作采样"


@pytest.mark.unit
def test_logp_is_sum_not_mean():
    """⚠️ Review Focus：logp 必须取**求和**（不是平均/归一化）。

    判据 = **可加性分解**：整条链的 logp 恰等于"前半 + 后半"。任何"除以决策数"的实现都会
    **硬红**——半链的平均 ≠ 整链平均（差 2 倍量级）。旧判据（"整链 |logp| > 半链 |logp|"）
    对平均实现**碰巧也通过**，是假守卫（评审 F2，本计划第三次同类）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    lp = chain_logp(decisions, pol)
    k = len(decisions) // 2
    assert k >= 4, f"链路太短（{len(decisions)}）——分解判据失去意义"
    lp_head = chain_logp(decisions[:k], pol)
    lp_tail = chain_logp(decisions[k:], pol)
    # ⚠️ 用 `.item()`：`float(requires_grad 张量)` 会发 UserWarning（弄脏测试输出）。
    assert lp.item() == pytest.approx(lp_head.item() + lp_tail.item(), abs=1e-3), \
        "chain_logp 不可加（整链 ≠ 前半 + 后半）——疑似取平均/归一化，而非求和"


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
def test_stored_logp_matches_recomputed():
    """`Decision.logp`（采样那一刻的无梯度 logp）必须与 `chain_logp` 的重算值一致。

    裁剪路径的 `logp_old` 直接取它（省掉一整遍"重算 old"的链前向）——这条是那个省法的
    **正确性前提**：省法只允许动"算在哪"，不允许动"算出来是多少"。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    stored = sum(d.logp for d in decisions)
    fresh = chain_logp(decisions, pol).item()
    assert stored == pytest.approx(fresh, abs=1e-3), "存的 logp 与带梯度重算值不符——省法前提不成立"


@pytest.mark.unit
def test_clip_epoch_combinations_are_guarded():
    """⚠️ 裁剪只在 `epochs>1` 时有意义：`epochs=1` 时 `new` 与 `logp_old` 同参数算出 ⇒
    `ratio≡1` ⇒ `clamp` 恒等（纯空转）。两种无意义组合必须**显式报错**，不得静默空转/浪费。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    w = _weights(inst, cfg)
    with pytest.raises(ValueError, match="空转"):
        joint_chain_step(pol, inst, lay, dm, seed=0, G=2, cfg=cfg, ctx=ctx, w=w,
                         epochs=1, clip_eps=0.2)
    with pytest.raises(ValueError, match="epochs>1"):
        joint_chain_step(pol, inst, lay, dm, seed=0, G=2, cfg=cfg, ctx=ctx, w=w,
                         epochs=3, clip_eps=None)


@pytest.mark.unit
def test_clipped_path_ratio_uses_sampling_time_logp():
    """裁剪路径（**唯一**用 `old` 的分支）的 ratio 基准必须与采样那一刻同源。

    判据用 `lr=0`（不更新参数）把它变成**确定性**的：`new` 与 `old` 同参数 ⇒ `ratio ≡ 1`
    （若 `old` 来自别处/别的时刻，这里立刻 ≠1）。顺带说明：`epochs=1` 的裁剪正是同一个
    恒等式，故那条路是空转（由 `test_clip_epoch_combinations_are_guarded` 拒收）。

    ⚠️ 原 `test_ratio_is_one_for_unchanged_policy`（同策略两遍 `chain_logp` 比大小）已删：
    同策略 + 同决策 + 无 dropout ⇒ `b − a` **恒为 0**、`exp(0)` **恒为 1**，对任何实现缺陷
    都不能变红（评审 F3 的恒真判据）。本测试的 `lr=0` 判据才是该性质的**真守卫**。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    # ⚠️ 本配置必然触发 F1 的"多 epoch 饱和"警告（那正是为什么这里要配 lr=0）——显式收下它，
    # 免得测试输出带噪；警告本身由 `test_multi_epoch_clip_warns_about_saturation` 负责钉。
    with pytest.warns(UserWarning, match="链级"):
        r, diag = joint_chain_step(pol, inst, lay, dm, seed=0, G=2, cfg=cfg, ctx=ctx,
                                   w=_weights(inst, cfg), epochs=2, clip_eps=0.2, lr=0.0)
    assert -1e9 < r < 0.0, "裁剪路径的组均值奖励不有限"
    assert diag["ratio"] == pytest.approx(1.0, abs=1e-6), "old 与 new 不同源——省法算错了"
    assert diag["clipped_frac"] == 0.0, "参数没动却报出界——clipped_frac 口径错"
    assert diag["grad_norm"] > 0.0, "裁剪路径梯度范数为 0——诊断退化"


@pytest.mark.unit
def test_multi_epoch_clip_warns_about_saturation():
    """⚠️ F1（评审裁定）：`epochs>1` 在链级尺度上是**静默烧算力**——链级 logp 是求和，一次更新
    （默认 lr）就把每条链的 Δlogp 推过 `log(1+clip_eps)`，之后 `∂obj/∂new ≡ 0`，多出来的
    epoch 一个参数都不会变。**不能硬禁**（lr 足够小则合法），故必须**警告**：消息要点明机理
    与出路（调小 lr / 按链长放大 clip_eps）。这里用 `lr=1e-6` 演示"合法用法"仍会警告。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    with pytest.warns(UserWarning, match="链级"):
        joint_chain_step(pol, inst, lay, dm, seed=0, G=2, cfg=cfg, ctx=ctx,
                         w=_weights(inst, cfg), epochs=2, clip_eps=0.2, lr=1e-6)


@pytest.mark.unit
def test_nonzero_layout_seed_is_rejected():
    """Fact F 守卫：奖励权重（`ReferenceObjectives.of` 固定 seed_layout=0）与交期 M_ref
    （`SimWorld._due_map` 用**布局**种子）必须同源——非 0 布局种子**显式报错**，
    不得让目标口径与权重口径静默错位。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    lay1 = sample_layout(inst.n_machines, seed=1, n_agv=cfg.n_agv)
    dm1 = dock_distance_matrix(build_corridor_graph(lay1))
    with pytest.raises(ValueError, match="seed"):
        joint_chain_step(pol, inst, lay1, dm1, seed=0, G=2, cfg=cfg, ctx=ctx,
                         w=_weights(inst, cfg))


@pytest.mark.unit
def test_joint_step_uses_adam_and_returns_diagnostics():
    inst, lay, dm, cfg, ctx, pol = _setup()
    r, diag = joint_chain_step(pol, inst, lay, dm, seed=0, G=4, cfg=cfg, ctx=ctx,
                               w=_weights(inst, cfg))
    assert isinstance(pol.optim, torch.optim.Adam), "优化器应为 Adam（spec §5.3.4 第 4 条）"
    assert set(diag) >= {"loss", "ratio", "clipped_frac", "grad_norm", "r_mean", "r_std", "A_std"}
    assert r == pytest.approx(diag["r_mean"])
    # ⚠️ loss 在裁剪路径首轮**结构性为 0**（ΣA=0）；无裁剪路径非 0。故诊断必须另带一个
    # **非零**的学习信号读数——`grad_norm` = 裁剪前的 ‖∂L/∂θ‖。
    assert diag["grad_norm"] > 0.0, "grad_norm 为 0——诊断退化，看不出训练是否在动"


@pytest.mark.unit
def test_joint_step_bitwise_reproducible_given_seed():
    """⚠️ 评审 I-3：`joint_chain_step` 在**同 seed + 同起点**下逐位可复现（训练入口的判据）。

    判据 = 两份**同起点**的 policy 副本各跑一步同 seed（G=2），奖励与诊断读数必须**精确
    相等**（同进程、同参数 ⇒ 同一串浮点运算逐位相同）。这条钉的是 **generator 真被透传进
    `roll_chain`**——只测 `roll_chain` 自己盖不住它（默认参数会自己兜底建 generator，透传
    断了照样"看起来可复现"）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    w = _weights(inst, cfg)
    pa, pb = copy.deepcopy(pol), copy.deepcopy(pol)      # 同起点（深拷贝前共享同一份初始权重）
    ra, da = joint_chain_step(pa, inst, lay, dm, cfg, ctx, w, seed=3, G=2)
    rb, db = joint_chain_step(pb, inst, lay, dm, cfg, ctx, w, seed=3, G=2)
    assert (ra, da["r_std"], da["grad_norm"]) == (rb, db["r_std"], db["grad_norm"]), (
        f"同 seed 同起点的两步训练读数不同：r={ra} vs {rb}"
        f"（r_std {da['r_std']} vs {db['r_std']}，grad_norm {da['grad_norm']} vs {db['grad_norm']}）"
        "——动作采样没有逐位复现")
