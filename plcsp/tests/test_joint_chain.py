"""联合链 GRPO 的测试（P2 Task 7，spec §5.3.4）。"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch
import pytest

import plcsp.algo.group_rel as group_rel
from plcsp.algo.group_rel import (_advantages, chain_logp, decisions_logp,
                                  joint_chain_step, roll_chain,
                                  sampled_decisions_logp, sampled_logp)
from plcsp.algo.policy import PolicyNet
from plcsp.algo.setup import build_layout_and_dm, build_setup
from plcsp.env.constraints import ABLATION_GROUPS, ConstraintConfig
from plcsp.env.des import SimConfig
from plcsp.env.instances import load_mk
from plcsp.env.reward import ReferenceObjectives
from plcsp.nn.encoder import LayoutEncoder


def _setup(name="mk01"):
    """返回 (inst, layout, dm, cfg, ctx, policy)——**六个**，与 roll_chain 的签名对齐。

    ⚠️ 环境三件套一律走 `algo.setup.build_setup`（评审 F4 收敛：此前本处用 `m_ref=100.0`
    占位，且 `sample_layout` 漏传 `aisle_w` / `max_agv_capacity`）。布局 seed 固定 0：
    特征归一化的 `m_ref` 按**该布局**的 seed 取，而奖励侧 `ReferenceObjectives.of` 固定用
    seed_layout=0 的参考运行——两者同源才有一致的归一化刻度。
    """
    inst = load_mk(name)
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    pol = PolicyNet(enc=LayoutEncoder())
    return inst, lay, dm, cfg, ctx, pol


def _ref(inst, cfg):
    """参考调度三目标 f^ref——`joint_chain_step` 的 `ref` 形参（`w` 由它内部派生，评审 F5）。

    ⚠️ 旧签名收裸 `w`：测试自己算 `reward_weights(...)` 再传进去，P4 扫 `n_agv` 时沿用旧 cfg
    的 w 是**静默**的（`ReferenceObjectives.matches` 当时零生产调用点）。
    """
    return ReferenceObjectives.of(inst, cfg)


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
def test_ablation_groups_produce_different_chains():
    """⚠️ F1（评审裁定）：训练路径必须把 `ConstraintConfig` 透传进 `SimWorld`。

    症状：`roll_chain` 直接 `SimWorld(...)` 构造（不传 constraints）⇒ 十约束默认全开，
    spec §6.2 的 5 组消融（Full/−物流/−生产/−信息/None）跑出**完全相同**的链且**零报错**
    ——最自然的解读会变成"约束不重要"，**假阴性会让人砍掉本来重要的约束**。

    判据：5 组在同一 `(inst, seed)` 下产生**互不相同**的链。链 = 决策序列（`Decision`），
    故签名取 `(makespan, len(decisions), 决策内容哈希)`：
    - 四个改变动力学的组（Full/−物流/−生产/None）在 makespan 上就分开；
    - **−信息**（只关 ⑧ 交期）按设计**不改动力学**（交期只进 metric 与特征，见
      `SimWorld._due`），它的区分度只能来自决策内容（快照 due 维 → B 段特征）——
      只比 makespan 会把它误判成"无差异"，那正是"关掉的开关没接线"的形状。
    """
    inst, _lay, _dm, cfg, _ctx, pol = _setup()
    sigs = {}
    for name, cons in ABLATION_GROUPS.items():
        # 每组用**该组口径**的 ctx（⑩ 关 → 载量标度退化；评审 F2）——spec §6.2 的正确用法。
        lay, dm, ctx = build_setup(inst, cfg, constraints=cons)
        dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, constraints=cons)
        digest = hashlib.sha256(b"".join(
            d.tok.tobytes() + d.feat.tobytes() + d.cand_feat.tobytes()
            + np.asarray([d.action], dtype=np.int64).tobytes() for d in dec)).hexdigest()[:16]
        sigs[name] = (round(float(met["makespan"]), 6), len(dec), digest)
    assert len(set(sigs.values())) == len(sigs), (
        f"消融组跑出相同的链——constraints 没有进 SimWorld：{sigs}")


@pytest.mark.unit
def test_chain_setup_features_are_silent_when_constraint_off():
    """⚠️ F2（评审裁定）：约束在**特征构造链**里不得留下假信号——⑤ 关时换型维必须恒 0。

    `roll_chain` 把 constraints 透传进 `_mach_cand_feat`（S 头候选特征）与 `task_feat`
    （L 头任务特征第 4 维）**两条**通路；只看 `setup_flag` 单函数盖不住"回调里没传下去"。
    全开配置必须有非零（否则判据恒真，抓不住任何东西）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    full = ConstraintConfig()
    dec_on, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, constraints=full)
    dec_off, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx,
                            constraints=full.with_off("setup_time"))
    s_on = np.array([float(d.cand_feat.max()) for d in dec_on if d.kind == "S"])
    s_off = np.array([float(d.cand_feat.max()) for d in dec_off if d.kind == "S"])
    assert s_on.max() > 0.0, "全开配置没有换型信号——判据失去意义"
    assert s_off.max() == 0.0, f"⑤ 关时 S 头换型特征仍报 {s_off.max()}——没读约束开关"
    l_off = np.array([float(d.feat[3]) for d in dec_off if d.kind == "L"])
    assert l_off.max() == 0.0, f"⑤ 关时 L 头任务特征第 4 维仍报 {l_off.max()}"


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
    # ⚠️ 2026-10-04 编码器改**分段投影**：S 头读 M 段 token、L 头读 V 段 token，
    # 故梯度应分别落到 `proj[0]`（M）与 `proj[2]`（V）——见 `encoder.py` 模块 docstring。
    for name, p in (("s_head_tok", pol.s_head_tok[0].weight),
                    ("l_head_tok", pol.l_head_tok[0].weight),
                    ("enc.proj[M]", pol.enc.proj[0].weight),
                    ("enc.proj[V]", pol.enc.proj[2].weight),
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
    # ⚠️ 分段投影下，这里的判据与旧版**相同**（不是更严）。
    #    曾想加一条反向判据"M 段投影不该有梯度"，**实测证伪**：注意力是全连接的
    #    （spec §5.1 删掉了轴掩码），V 段 token 经 8 层注意力后混合了全部段 ⟹
    #    L 头的梯度会回传到**每一段**的投影。分段投影改变的是"各段用了哪些列"，
    #    **不是梯度的可达性**。故此处只断言 V 段投影非零（这才是 L 路径的入口）。
    for name, p in (("l_head_tok", pol.l_head_tok[0].weight),
                    ("enc.proj[V]", pol.enc.proj[2].weight),
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
    ref = _ref(inst, cfg)
    with pytest.raises(ValueError, match="空转"):
        joint_chain_step(pol, inst, lay, dm, seed=0, G=2, cfg=cfg, ctx=ctx, ref=ref,
                         epochs=1, clip_eps=0.2)
    with pytest.raises(ValueError, match="epochs>1"):
        joint_chain_step(pol, inst, lay, dm, seed=0, G=2, cfg=cfg, ctx=ctx, ref=ref,
                         epochs=3, clip_eps=None)


@pytest.mark.unit
def test_clipped_path_ratio_uses_sampling_time_logp():
    """裁剪路径（**唯一**用 `old` 的分支）的 ratio 基准必须与采样那一刻同源。

    判据用 `lr=0`（不更新参数）把它变成**确定性**的：`new` 与 `old` 同参数 ⇒ `ratio ≈ 1`
    （若 `old` 来自别处/别的时刻，这里立刻 ≠1）。⚠️ 修复后裁剪是**逐决策**的（旧实现按整条
    链裁剪——见 `test_clip_band_is_per_decision_not_per_chain`）。
    顺带说明：`epochs=1` 的裁剪正是同一个恒等式，故那条路是空转（由
    `test_clip_epoch_combinations_are_guarded` 拒收）。

    ⚠️ **旧命题 → 新命题（2026-10-04，批量重算批次）**：
    - 旧：`ratio` 是**精确** 1.0（断言写 `== 1.0`）；理由 = `new` 与 `old` 逐位相同。
    - 新：`ratio` 在 **1e-5 内**等于 1；理由 = `new` 改由一次编码器批前向 + 打分头分组批
      算出，而 `old` 是采样时逐决策单条前向的值——批矩阵乘的分块不同 ⟹ 末位漂移约 3.6e-7
      （单链 route_k=2 实测 `max|Δlogp| = 3.58e-07`；`ratio` 与 1 的偏差 < 4e-7）。
      容差取 1e-5 与硬要求 1 一致。
    - `clipped_frac` / `grad_norm` 的判据不受影响（带是 [0.8, 1.2]，漂移差几个数量级）。

    ⚠️ 原 `test_ratio_is_one_for_unchanged_policy`（同策略两遍 `chain_logp` 比大小）已删：
    同策略 + 同决策 + 无 dropout ⇒ `b − a` **恒为 0**、`exp(0)` **恒为 1**，对任何实现缺陷
    都不能变红（评审 F3 的恒真判据）。本测试的 `lr=0` 判据才是该性质的**真守卫**。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    r, diag = joint_chain_step(pol, inst, lay, dm, seed=0, G=2, cfg=cfg, ctx=ctx,
                               ref=_ref(inst, cfg), epochs=2, clip_eps=0.2, lr=0.0)
    assert -1e9 < r < 0.0, "裁剪路径的组均值奖励不有限"
    assert diag["ratio"] == pytest.approx(1.0, abs=1e-5), \
        "old 与 new 不同源——逐决策同源不再成立（ratio 有超出末位漂移的假 delta）"
    assert diag["clipped_frac"] == 0.0, "参数没动却报出界——clipped_frac 口径错"
    assert diag["grad_norm"] > 0.0, "裁剪路径梯度范数为 0——诊断退化"


@pytest.mark.unit
def test_multi_epoch_clip_trust_region_is_live():
    """⚠️ 训练方法缺陷修复的**头号判据**：`epochs>1` 的信任域必须真的在起作用。

    症状（修复前实测，MK01、G=2、`epochs=2, clip_eps=0.2, lr=1e-3`）：裁剪施加在**链级**
    ratio 上，而 `logp` 是整条链的求和（约定 2，链长 100–460）——一次 Adam 更新就把每条链的
    Δlogp 推过 `log(1.2)`，于是**全部**链出界、`∂obj/∂new ≡ 0`：`clipped_frac=1.0`、
    `grad_norm=0.0`，第 2 个 epoch 的损失面是平的（只剩 Adam 动量的余波），却照样付链前向
    + 反向。裁剪下沉到**每个决策**后：单个决策的漂移约 0.002–0.05 nats，带内大多数决策仍在
    带内 ⇒ `clipped_frac < 1.0`、`grad_norm > 0`（`grad_norm` 取的是**末轮**裁剪前的梯度
    范数，故 `> 0` 直接说明第 2 个 epoch 的梯度信号非零）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    r, diag = joint_chain_step(pol, inst, lay, dm, seed=0, G=2, cfg=cfg, ctx=ctx,
                               ref=_ref(inst, cfg), epochs=2, clip_eps=0.2, lr=1e-3)
    assert -1e9 < r < 0.0, "裁剪路径的组均值奖励不有限"
    assert diag["clipped_frac"] < 1.0, (
        f"全部决策都被裁掉（clipped_frac={diag['clipped_frac']}）——信任域仍按链级算，"
        "第 2 个 epoch 的梯度恒 0")
    assert diag["grad_norm"] > 0.0, "末轮梯度范数为 0——第 2 个 epoch 没有梯度信号"


# ================= 批量重算（编码器 + 打分头各自一次批量） =================
# 背景（本批的动机，docs/progress-log.md §34/§36/§37）：`roll_chain` 的在线前向是**逐决策**的
# （每个决策依赖上一刻的仿真状态），不能批；但 `chain_logp` / `decisions_logp` 的重算是
# **事后**的——全部决策的 token 形状相同（实例级常量），可堆成 (B,N,F) 一次前向。
# 编码器是 CPU 配置耗时主项（本机实测单条 ≈6.0 ms、231 条批成一次 ≈402 ms，吞吐 ~3.4×）；
# CUDA 配置上打分头 + 它引出的逐决策反向小图占整步 ~54%（§36.8），故头也按 `(kind, n_cand)`
# 分组批量（**不 padding**：padding 会改 `log_softmax` 的归约长度），见 `_decision_logp_terms`。


def _reference_terms(decisions, policy):
    """**旧路径**的参照实现：逐决策单条前向——只存在于测试里，不参与生产。

    它是"批量重算没有改变算出来是多少"的对照物：生产路径已改成一次批前向，
    没有第二份实现就分不清"批算对了但有末位漂移"与"批路径接错了"。
    """
    out = []
    for d in decisions:
        tok, _ = policy.forward_enc(
            torch.as_tensor(d.tok, dtype=torch.float32).unsqueeze(0), d.seg)
        head = {"S": policy.mach_logits_emb, "L": policy.agv_logits_emb,
                "R": policy.route_logits_emb, "M": policy.pm_logits_emb,
                "C": policy.charge_logits_emb}[d.kind]
        logits = head(tok,
                      torch.as_tensor(d.feat, dtype=torch.float32).reshape(1, 1, -1),
                      torch.as_tensor(d.cand_feat, dtype=torch.float32),
                      torch.as_tensor(d.tok_idx, dtype=torch.long))
        out.append(torch.log_softmax(logits.flatten(), -1)[d.cand.index(d.action)])
    return out


@pytest.mark.unit
def test_all_decisions_in_a_step_share_one_token_shape():
    """⚠️ 批量重算的**形状前提**（先验证、再依赖）：整组决策的 token 形状与 `seg` 必须**唯一**。

    依据：`build_tok` 的行数 = n_m + n_jobs + n_agv + 1，四段长度都是**实例级常量**
    （`NormContext` 的 n_m/n_jobs/n_agv 在一次运行里不变），故形状不随决策变。
    这条把"可以堆成 (B,N,F)"钉成机器可判的判据：日后若有人让 `seg` 随决策变（如逐决策
    增删 token），`_decision_logp_terms` 的批量路径会**显式报错**，而不是静默算错。
    ⚠️ 两条不同 seed 的链都查：批量要跨 G 条链，前提是跨链也同形。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec_a, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2)
    dec_b, _ = roll_chain(inst, lay, dm, cfg, pol, seed=1, ctx=ctx, route_k=2)
    shapes = {(d.tok.shape, d.seg) for d in (*dec_a, *dec_b)}
    assert len(shapes) == 1, f"决策的 token 形状/seg 不唯一——批量前提不成立：{shapes}"
    assert dec_a[0].seg == (inst.n_machines, inst.n_jobs, cfg.n_agv, 1), \
        f"seg 不是实例级常量：{dec_a[0].seg}"


@pytest.mark.unit
def test_batched_recompute_matches_the_per_decision_reference_logp():
    """⚠️ 硬要求 1（前向侧）：批量重算的 logp 必须与逐决策重算一致（≤1e-5）。

    (B,N,F) 批前向与单条前向的矩阵乘分块不同 ⟹ 末位漂移（实测嵌入层 ~2e-6）。这是本批
    **刻意**接受的代价（换来编码器调用次数从 ~2×链长 降到 1），故把"漂移量级"钉成判据：
    超过 1e-5 就说明批路径真的算错了（批维接错 / seg 混用 / 漏了某条决策），不是浮点末位。
    ⚠️ 用 route_k=2：S/L/R 三类决策都走同一条重算路径。
    """
    from plcsp.algo.group_rel import _decision_logp_terms
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2)
    assert len(dec) >= 50, f"链太短（{len(dec)} 个决策）——判据失去意义"
    batched = torch.stack(_decision_logp_terms(dec, pol)).detach()
    reference = torch.stack(_reference_terms(dec, pol)).detach()
    d = float((batched - reference).abs().max())
    assert d < 1e-5, f"批量与逐决策重算的 logp 最大差 {d}——批路径算错了（非浮点末位）"


@pytest.mark.unit
def test_batched_recompute_matches_the_per_decision_reference_gradients():
    """⚠️ 硬要求 1（梯度侧）：批量重算回传的**每个参数**的梯度必须与逐决策一致（≤1e-5）。

    只比 logp 不够：批量路径若在反传里接错（detach、漏项、批维错位），前向值可能仍然接近，
    梯度却少了一路或错位。判据取全部参数上的最大逐元素差。
    """
    from plcsp.algo.group_rel import _decision_logp_terms
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2)

    def _grads(build):
        pol.zero_grad(set_to_none=True)
        torch.stack(build(dec, pol)).sum().backward()
        return {k: p.grad.detach().clone() for k, p in pol.named_parameters()
                if p.grad is not None}

    g_batched = _grads(_decision_logp_terms)
    g_reference = _grads(_reference_terms)
    assert set(g_batched) == set(g_reference) and g_batched, "两条路径的参数集不同——判据失去意义"
    worst = max(float((g_batched[k] - g_reference[k]).abs().max()) for k in g_batched)
    assert worst < 1e-5, f"批量与逐决策的梯度最大差 {worst}——反传路径算错了（非浮点末位）"


@pytest.mark.unit
def test_head_scores_are_grouped_batched_by_kind_and_candidate_count(monkeypatch):
    """⚠️ 打分头批量化（2026-10-04 打分头批次）：重算路径按 `(kind, n_cand)` 分组批量打分。

    判据四条：
    1. 重算路径**零**次逐决策头调用（`*_logits_emb`）——头真批了，不是接了个没用的 API；
    2. 批量调用次数 = 不同 `(kind, n_cand)` 组数，且各组的决策数与决策表一致（多一组 =
       分组键写错；少一组 = 漏决策）；
    3. 组序 = `sorted`（**确定**：同 seed 同结果，不依赖 dict 迭代序）；
    4. 五头全覆盖（route_k=2 + pm/charge）——头名映射写错（如 C 头指到 S 头）会被逐 kind
       的组数与候选数当场抓住。

    ⚠️ **不 padding**是本仓的选择（padding 会改 `log_softmax` 的归约长度、撑破 1e-5 容差）：
    组数必须远小于决策数（MK01/全头配置实测 7 组 vs 3400+ 决策）。组数若退化成逐决策，
    本判据当场红——那说明该改分组键，不是该放宽判据。
    """
    from plcsp.algo.group_rel import _decision_logp_terms
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2,
                        pm_head=True, charge_head=True)
    expected: dict[tuple[str, int], int] = {}
    for d in dec:
        expected[(d.kind, len(d.cand))] = expected.get((d.kind, len(d.cand)), 0) + 1
    assert {k for k, _ in expected} == {"S", "L", "R", "M", "C"}, \
        f"这条链没覆盖五个头：{sorted({d.kind for d in dec})}——判据失去意义"

    calls: list[tuple[str, int, int]] = []
    per_decision = {"n": 0}
    orig_batch = pol.logits_emb_batch

    def _counting_batch(kind, emb, tok_idx, feat_dec, feat_cand):
        calls.append((kind, int(emb.shape[0]), int(feat_cand.shape[1])))
        return orig_batch(kind, emb, tok_idx, feat_dec, feat_cand)

    def _counting_head(*_a, **_k):
        per_decision["n"] += 1
        raise AssertionError("重算路径仍在逐决策调打分头——批量没接上")

    monkeypatch.setattr(pol, "logits_emb_batch", _counting_batch)
    for name in ("mach_logits_emb", "agv_logits_emb", "route_logits_emb",
                 "pm_logits_emb", "charge_logits_emb"):
        monkeypatch.setattr(pol, name, _counting_head)
    terms = _decision_logp_terms(dec, pol)
    assert len(terms) == len(dec), f"项数 {len(terms)} != 决策数 {len(dec)}"
    assert per_decision["n"] == 0, "重算路径逐决策调了打分头"
    got: dict[tuple[str, int], int] = {}
    for kind, batch, n_cand in calls:
        got[(kind, n_cand)] = got.get((kind, n_cand), 0) + batch
    assert got == expected, f"分组批量与决策的分组不符：批量 {got} vs 期望 {expected}"
    assert len(calls) == len(expected), f"批量调用 {len(calls)} 次 != 组数 {len(expected)}"
    assert [k for k, _, _ in calls] == [k for k, _ in sorted(expected)], \
        f"组序不是 sorted（确定性判据）：{[k for k, _, _ in calls]}"


@pytest.mark.unit
def test_grouped_head_batching_is_deterministic_across_calls():
    """⚠️ 分组顺序**确定**（同 seed 同结果）：同一批决策两次重算必须**逐位相同**。

    组序用 `sorted`、组内用原决策序 ⟹ 每次的批组成与行序完全一致；CPU 上同一串 kernel
    是确定性的，故判据取 `torch.equal`（不是 allclose）。若日后有人把组序改成依赖 dict
    迭代序或集合序，这条会红。
    """
    from plcsp.algo.group_rel import _decision_logp_terms
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2)
    a = torch.stack(_decision_logp_terms(dec, pol)).detach()
    b = torch.stack(_decision_logp_terms(dec, pol)).detach()
    assert torch.equal(a, b), "同一批决策两次重算不逐位相同——分组顺序不确定"


@pytest.mark.unit
def test_recompute_batches_all_chains_into_one_encoder_forward(monkeypatch):
    """⚠️ 硬要求：批量必须跨 **G 条链的全部决策**一次前向，不是一条链一次。

    判据 = 数 `forward_enc` 的调用次数与批大小。若重算仍写在"每条链一次"的循环里，编码器
    调用次数会是 G（本判据当场红），G× 的墙钟也就省不下来。两条训练路径（链级 `chains_logp`
    与展平 `all_decisions_logp`）都要覆盖。
    """
    from plcsp.algo.group_rel import (all_decisions_logp, _decision_logp_terms,
                                      chains_logp)
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec_a, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2)
    dec_b, _ = roll_chain(inst, lay, dm, cfg, pol, seed=1, ctx=ctx, route_k=2)
    chains = [dec_a, dec_b]
    n_total = len(dec_a) + len(dec_b)

    calls = {"n": 0, "sizes": []}
    orig = pol.forward_enc

    def _counting(tok_feat, seg, bias=None):
        x = tok_feat if torch.is_tensor(tok_feat) else torch.as_tensor(tok_feat)
        calls["n"] += 1
        calls["sizes"].append(int(x.shape[0]))
        return orig(tok_feat, seg, bias)      # 透传几何偏置（默认 None）

    monkeypatch.setattr(pol, "forward_enc", _counting)
    lp = chains_logp(chains, pol)
    assert calls["n"] == 1 and calls["sizes"] == [n_total], \
        f"链级重算的编码器调用 {calls}——不是一次覆盖全部 {n_total} 个决策"
    assert lp.shape == (2,), f"chains_logp 形状应为 (G,)，实得 {tuple(lp.shape)}"
    calls["n"], calls["sizes"] = 0, []
    flat = all_decisions_logp(chains, pol)
    assert calls["n"] == 1 and calls["sizes"] == [n_total], \
        f"逐决策重算的编码器调用 {calls}——不是一次覆盖全部决策"
    assert flat.shape == (n_total,), f"展平向量形状应为 (N,)，实得 {tuple(flat.shape)}"
    assert calls["sizes"] == [n_total], "第二次调用不是单次批前向"
    # 同一批输入下，chains_logp / all_decisions_logp / _decision_logp_terms 必须互相一致：
    # 三条入口共用同一个打分体，漂开就是"两条训练路径口径不同"（本项目反复出现的一类缺陷）。
    terms = _decision_logp_terms([d for ch in chains for d in ch], pol)
    assert float(chains_logp([dec_a], pol)[0].detach()) == \
        float(chain_logp(dec_a, pol).detach()), \
        "chains_logp 与 chain_logp 的累加口径不同源——两条训练路径会漂开"
    assert torch.allclose(lp.detach()[0], chain_logp(dec_a, pol).detach(), atol=1e-5)
    assert torch.allclose(flat.detach(), torch.stack(terms).detach(), atol=1e-5)


@pytest.mark.unit
def test_per_decision_logp_vector_is_same_source_within_tolerance():
    """⚠️ 逐决策裁剪的**同源**前提（`lr=0` 时 ratio≈1 靠它才是成立的近似）：

    1. `decisions_logp`（带梯度重算）与 `sampled_decisions_logp`（采样回放）在 **1e-5 内**
       相同（元素级 `torch.allclose`）；
    2. `chain_logp` 仍是**逐步 float32 顺序求和**——与手写逐步累加**逐位相同**，且该累加与
       `torch.stack(...).sum()` 在当前数据上**确实不同**（否则归约次序的判据恒真、失去区分力）。

    ⚠️ **旧命题 → 新命题（2026-10-04，批量重算批次）**：
    - 旧：重算与采样回放**逐位相同**（`torch.equal`）。理由 = 两者都走"逐决策单条前向"，
      同一串浮点运算、同一个归约次序。
    - 新：重算与采样回放**在 1e-5 内相同**。理由 = 重算改成 `(B,N,F)` **一次批前向**
      （编码器前向是耗时主项：实测 231 个决策的重算从 9.31 ms/决策降到 2.31 ms/决策，
      整步 ~2×），打分头再按 `(kind, n_cand)` 分组批（§37）。批矩阵乘的分块与单条不同 ⟹
      每项有约 **4e-7 以内**的末位漂移（本仓实测 `max|Δlogp| = 3.58e-07`，编码器批 +
      打分头批两笔合计）——这是**刻意接受**的代价，不是缺陷。裁剪首轮 `ratio ≡ 1`
      随之从**严格等式**降为"≈1 在 1e-5 内"。
    ⚠️ 累加**次序**没变（第 2 条仍逐位钉死）——变的只是每一项的末位。
    """
    from plcsp.algo.group_rel import _decision_logp_terms
    # ⚠️ 显式播种（2026-10-05）：`_setup()` 建策略**不播种**，故本条的取值取决于**全局 RNG
    # 在 pytest 会话里的当时状态**（哪些文件先跑）。本批新增 `test_batch_head.py` 后，第 2 条
    # 断言（"逐步累加与 `.sum()` 在本数据上必须不同"——一条**数据相关**的区分力判据）恰好翻转。
    # 播种把数据钉死，判据本身一字未动（既不放宽也不删除）。
    torch.manual_seed(0)
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    new_vec = decisions_logp(decisions, pol).detach()
    old_vec = sampled_decisions_logp(decisions)
    assert new_vec.shape == (len(decisions),), f"逐决策向量形状 {tuple(new_vec.shape)} 不对"
    d = float((new_vec - old_vec).abs().max())
    assert d < 1e-5, (
        f"逐决策重算与采样回放的最大差 {d} ≥ 1e-5——分决策裁剪的 ratio≈1 前提不成立"
        "（超过 1e-5 说明批路径算错了，不是浮点末位）")
    # 累加口径：chain_logp 必须仍是**逐步 float32 顺序累加**——与手写逐步和逐位相同。
    terms = _decision_logp_terms(decisions, pol)
    manual = torch.zeros(())
    for t in terms:
        manual = manual + t
    vec_sum = torch.stack(terms).sum()
    assert torch.equal(chain_logp(decisions, pol).detach(), manual.detach()), \
        "chain_logp 不再是逐步 float32 顺序累加（换成了 .sum() 之类的归约？）"
    assert not torch.equal(manual.detach(), vec_sum.detach()), \
        "本数据上逐步累加与 .sum() 恰好逐位相同——归约次序的判据在此失去区分力"
    # 链级：顺序累加 + 各项末位漂移 ⟹ 与采样回放不再逐位相同，只在 n·2e-7 量级内一致。
    assert float(chain_logp(decisions, pol).detach()) == pytest.approx(
        float(sampled_logp(decisions)), abs=1e-3), \
        "chain_logp 与采样回放的链级口径漂开——累加次序或 dtype 变了"


@pytest.mark.unit
def test_clip_band_is_per_decision_not_per_chain(monkeypatch):
    """⚠️ `clip_eps` 的**粒度**判据：带是加在**每个决策**的 ratio 上，不是整条链的。

    构造一条**合成链**（复用真实链的决策，但不跑仿真：`roll_chain` 被打桩）：逐个决策把
    存下的 logp 改成 `重算值 − δ`（δ=0.15 nats）。于是同一份数据上
    - 每个决策的 ratio = e^δ ≈ 1.162 ∈ [0.8, 1.2]（**带内**）；
    - 链级 Δlogp = n·δ ≥ 3·0.15 = 0.45 > log(1.2) ≈ 0.182（**出界**）。

    旧实现（链级裁剪）会把整条链裁掉：`clipped_frac=1.0`、`grad_norm=0`、`ratio=e^{nδ}`。
    分决策裁剪才给出 `clipped_frac=0`、`ratio≈e^δ`、`grad_norm>0`。`lr=0` 保证两轮同参数，
    ratio 只由 δ 决定（不掺参数更新的漂移）。两条链用**不同** seed 的真实决策，否则
    `ΣA=0` 会让两条同源链的梯度逐位抵消，`grad_norm>0` 失去意义。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec_a, met_a = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    dec_b, met_b = roll_chain(inst, lay, dm, cfg, pol, seed=1, ctx=ctx)
    assert len(dec_a) >= 3 and len(dec_b) >= 3, "链太短——n·δ 盖不过 log(1.2)"
    delta = 0.15
    with torch.no_grad():
        for d in (*dec_a, *dec_b):
            # 单决策的 `chain_logp` = 零 + 该项 ⇒ 恰是该决策带梯度重算 logp 的逐位同源值
            d.logp = float(chain_logp([d], pol)) - delta
    calls = {"g": 0}

    def _fake_roll(*_args, **_kwargs):
        g, calls["g"] = calls["g"], calls["g"] + 1
        dec, met = (dec_a, met_a) if g == 0 else (dec_b, met_b)
        # 两条链奖励拉开（makespan 差 10）⇒ 组内 z 化后 A≠0，裁剪才有非零梯度可言
        return dec, dict(met, makespan=float(met["makespan"]) + 10.0 * g)

    monkeypatch.setattr(group_rel, "roll_chain", _fake_roll)
    _r, diag = joint_chain_step(pol, inst, lay, dm, cfg, ctx, _ref(inst, cfg),
                                seed=0, G=2, epochs=2, clip_eps=0.2, lr=0.0)
    assert diag["clipped_frac"] == 0.0, "每个决策都在带内，却报出界——裁剪还是链级的"
    assert diag["ratio"] == pytest.approx(math.exp(delta), abs=1e-6), \
        "ratio 不是逐决策比的均值（链级比值会是 e^{nδ}）"
    assert diag["grad_norm"] > 0.0, "整链出界被裁 ⇒ 梯度恒 0——裁剪没有下沉到决策"


@pytest.mark.unit
def test_nonzero_layout_seed_is_rejected():
    """Fact F 守卫：奖励权重（`ReferenceObjectives.of` 固定 seed_layout=0 的参考运行）与
    特征归一化的 `m_ref`（按**布局**种子取）必须同源——非 0 布局种子**显式报错**，
    不得让归一化刻度静默错位。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    lay1, dm1 = build_layout_and_dm(inst, cfg, seed_layout=1)
    with pytest.raises(ValueError, match="seed"):
        joint_chain_step(pol, inst, lay1, dm1, seed=0, G=2, cfg=cfg, ctx=ctx,
                         ref=_ref(inst, cfg))


@pytest.mark.unit
def test_joint_step_rejects_reference_from_other_cfg():
    """⚠️ F5（评审裁定）：`ReferenceObjectives.matches()` 必须进训练路径。

    症状：它此前**只被测试调用**（`grep '.matches('` 仅命中测试），而 P4 扫 `n_agv` 时极易
    沿用**按旧 cfg 算出的 w**——`w` 决定落在哪个 Pareto 点，"按参考归一化"在这条扫描轴上
    静默不成立。故 `joint_chain_step` 直接收 `ReferenceObjectives`（不再收裸 `w`）：
    指纹不匹配即显式报错，且权重由 ref **派生**（不存在"w 与 ref 不同源"的空隙）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    stale = ReferenceObjectives.of(inst, SimConfig(n_agv=3))        # 旧 cfg 的参考值
    assert stale.matches(inst, SimConfig(n_agv=3))
    with pytest.raises(ValueError, match="指纹|不同源"):
        joint_chain_step(pol, inst, lay, dm, SimConfig(n_agv=3, tau=0.5), ctx, stale,
                         seed=0, G=2)


@pytest.mark.unit
def test_joint_step_rejects_ctx_from_other_constraints():
    """⚠️ R2：`ctx` 的约束与训练的约束必须同源——不同源**显式报错**，不得静默给假信号。

    症状（F2 同型）：`build_training_setup(constraints=off)` 建的 ctx 与训练实际跑的动力学
    分属两组开关时，③/⑧ 的特征静默读的是 `ctx`（R1），仿真读的是 `constraints` ⟹ 特征报
    "不会坏 / 无交期"而仿真照坏照交期——策略收到一个动力学里不存在的信号。
    入口指纹不符即报错（同 `ref.matches` 的 F5 守卫形状）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    with pytest.raises(ValueError, match="约束"):
        joint_chain_step(pol, inst, lay, dm, cfg, ctx, _ref(inst, cfg), seed=0, G=2,
                         constraints=ConstraintConfig().with_off("machine_failure"))


@pytest.mark.unit
def test_joint_step_uses_adam_and_returns_diagnostics():
    inst, lay, dm, cfg, ctx, pol = _setup()
    r, diag = joint_chain_step(pol, inst, lay, dm, seed=0, G=4, cfg=cfg, ctx=ctx,
                               ref=_ref(inst, cfg))
    assert isinstance(pol.optim, torch.optim.Adam), "优化器应为 Adam（spec §5.3.4 第 4 条）"
    assert set(diag) >= {"loss", "ratio", "clipped_frac", "grad_norm", "r_mean", "r_std", "A_std"}
    assert r == pytest.approx(diag["r_mean"])
    # ⚠️ loss 在裁剪路径首轮**结构性为 0**（ΣA=0）；无裁剪路径非 0。故诊断必须另带一个
    # **非零**的学习信号读数——`grad_norm` = 裁剪前的 ‖∂L/∂θ‖。
    assert diag["grad_norm"] > 0.0, "grad_norm 为 0——诊断退化，看不出训练是否在动"


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="本环境 torch 无 CUDA（CPU-only 构建）——GPU 侧须用 "
                           "D:/anaconda/envs/py312/python.exe 显式跑")
def test_gpu_recompute_matches_cpu_and_follows_the_device():
    """⚠️ CUDA 侧验收（默认门禁里**自动跳过**）：策略搬上 GPU 后

    1. 批重算的逐决策 logp 与 CPU 在 1e-5 内一致（同一份权重、同一批决策）；
    2. 张量确实落在 CUDA 上（"设备跟随参数"，而不是悄悄退回 CPU）；
    3. `chains_logp` / `joint_chain_step` 在 GPU 上跑得通（含 CUDA 动作采样流）。

    ⚠️ **跨设备不可逐位复现**（如实记下）：动作采样流的设备跟随策略，CPU 与 CUDA 的
    `torch.Generator` 是两条不同的流 ⟹ 同 seed 在 CPU / GPU 上采样出**不同**的链。
    "同 seed 逐位可复现"的既有承诺因此是**逐设备**的；论文的种子对照必须在同一设备上做。
    """
    from plcsp.algo.group_rel import _decision_logp_terms, chains_logp
    inst, lay, dm, cfg, ctx, _ = _setup()
    torch.manual_seed(11)
    pol_cpu = PolicyNet(enc=LayoutEncoder())
    dec, _met = roll_chain(inst, lay, dm, cfg, pol_cpu, seed=0, ctx=ctx, route_k=2)
    pol_gpu = PolicyNet(enc=LayoutEncoder()).to("cuda")
    pol_gpu.load_state_dict(pol_cpu.state_dict())
    with torch.no_grad():
        terms_cpu = torch.stack(_decision_logp_terms(dec, pol_cpu))
        terms_gpu = torch.stack(_decision_logp_terms(dec, pol_gpu))
        assert terms_gpu.device.type == "cuda", "重算没有跟随参数设备（仍在 CPU？）"
        d = float((terms_cpu - terms_gpu.cpu()).abs().max())
    assert d < 1e-5, f"GPU 与 CPU 的批重算最大差 {d}——跨设备批路径算错了"
    assert chains_logp([dec], pol_gpu).device.type == "cuda", "链级 logp 没落在 GPU 上"
    r, diag = joint_chain_step(pol_gpu, inst, lay, dm, cfg, ctx, _ref(inst, cfg),
                               seed=0, G=2, route_k=2)
    assert math.isfinite(r) and math.isfinite(float(diag["grad_norm"])), \
        f"GPU 训练步给出非有限读数：r={r} diag={diag}"
    # ⚠️ 裁剪路径也必须过一遍：它的 `repeat_interleave` / `old_flat` 是**另一条**设备路径
    #    （实测漏改一次：repeats 留在 CPU ⟹ `index is on cpu, different from cuda:0`）。
    r2, diag2 = joint_chain_step(pol_gpu, inst, lay, dm, cfg, ctx, _ref(inst, cfg),
                                 seed=0, G=2, route_k=2, epochs=2, clip_eps=0.2, lr=0.0)
    assert math.isfinite(r2) and diag2["clipped_frac"] == 0.0, \
        f"GPU 裁剪路径读数异常：r={r2} diag={diag2}"


@pytest.mark.unit
def test_joint_step_bitwise_reproducible_given_seed():
    """⚠️ 评审 I-3：`joint_chain_step` 在**同 seed + 同起点**下逐位可复现（训练入口的判据）。

    判据 = 两份**同起点**的 policy 副本各跑一步同 seed（G=2），奖励与诊断读数必须**精确
    相等**（同进程、同参数 ⇒ 同一串浮点运算逐位相同）。这条钉的是 **generator 真被透传进
    `roll_chain`**——只测 `roll_chain` 自己盖不住它（默认参数会自己兜底建 generator，透传
    断了照样"看起来可复现"）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    ref = _ref(inst, cfg)
    pa, pb = copy.deepcopy(pol), copy.deepcopy(pol)      # 同起点（深拷贝前共享同一份初始权重）
    ra, da = joint_chain_step(pa, inst, lay, dm, cfg, ctx, ref, seed=3, G=2)
    rb, db = joint_chain_step(pb, inst, lay, dm, cfg, ctx, ref, seed=3, G=2)
    assert (ra, da["r_std"], da["grad_norm"]) == (rb, db["r_std"], db["grad_norm"]), (
        f"同 seed 同起点的两步训练读数不同：r={ra} vs {rb}"
        f"（r_std {da['r_std']} vs {db['r_std']}，grad_norm {da['grad_norm']} vs {db['grad_norm']}）"
        "——动作采样没有逐位复现")

@pytest.mark.unit
def test_clipped_loss_weights_chains_equally_not_by_length():
    """⚠️ 两条损失路径必须对**链**等权——裁剪路径曾按**决策**平均，导致长链权重更大。

    背景：无裁剪路径 `obj` 是 `(G,)` 的链 logp，`obj.mean()` 按 **G 条链**平均（每条 `A_g/G`）。
    逐决策裁剪改造后，裁剪路径把 G 条链展平成 N=Σn_g 个决策再 `.mean()`，于是每条链的权重
    变成 `n_g/N`——**正比于链长**。链长与 episode 长短相关 ⟹ 系统性地给长（差）的调度
    更大的梯度权重，且两条路径口径不一致。

    修法：`_chain_mean` —— 每条链先对自己的决策取平均，再对 G 条链取平均。
    """
    from plcsp.algo.group_rel import _chain_mean

    # 链等长时，两种算法**完全一致**（这保证修复不改变既有等长情形的行为）
    flat_eq = torch.tensor([1.0, 3.0, 2.0, 4.0])
    assert torch.allclose(_chain_mean(flat_eq, [2, 2]), flat_eq.mean())

    # 链不等长时，按链平均 ≠ 按决策平均
    flat = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])       # 链0 两个决策、链1 三个决策
    by_chain = _chain_mean(flat, [2, 3])
    assert not torch.allclose(by_chain, flat.mean()), "仍按决策平均——长链被加权了"
    assert torch.allclose(by_chain, torch.tensor((5.0 + 0.0) / 2.0))


@pytest.mark.unit
def test_two_loss_paths_agree_when_chains_are_equal_length():
    """⚠️ 链等长时，裁剪与不裁剪两条路径对链的加权必须相同（口径一致性）。

    这是上一条的端到端对照：构造等长链时两者的**链权重**应一致；
    不等长时才允许不同（因为按链平均本来就与按决策平均不同）。
    """
    from plcsp.algo.group_rel import _chain_mean

    flat = torch.arange(6, dtype=torch.float32)          # 三条等长链
    assert torch.allclose(_chain_mean(flat, [2, 2, 2]), flat.mean())


# ================= O1：逐目标优势（adv_mode="per_objective"） =================
# 背景：`scalar` 把三目标**先加权求和再组内 z 化**。z 只消掉**总尺度**，不消掉**目标之间的
# 相对尺度**——哪个目标的奖励方差大，A 就由它主导，w 名义上控制权衡、实际控制不了。
# `per_objective` 改成**每个目标各自组内 z 化、再按 w 合成**：z_i 对逐目标正缩放不变，
# w 才能真正决定"各目标的相对权重"。

# MK01 口径权重（任务书给出；默认 cfg 实测 ≈(0.066, 0.886, 0.048)，能量项权重最大），
# 用于判据贴近真实口径。
MK01_W = (0.066, 0.880, 0.055)


@pytest.mark.unit
def test_adv_mode_rejects_unknown_value():
    """`adv_mode` 只认两个值——未知值必须**显式报错**（不得静默落到 scalar 或新口径）。

    静默回退是本项目反复出现的失效形态（F1/F2/F5 同型）：名字写错时跑出来的曲线会被当成
    "新机制没效果"，而实际上机制根本没接上。守卫必须在**跑链之前**（写错名的代价不该是
    几分钟的仿真墙钟）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    with pytest.raises(ValueError, match="adv_mode"):
        joint_chain_step(pol, inst, lay, dm, seed=0, G=2, cfg=cfg, ctx=ctx,
                         ref=_ref(inst, cfg), adv_mode="per-objective")


@pytest.mark.unit
def test_scalar_adv_mode_regression_pin():
    """⚠️ O1 硬要求 1（**改写版**）：`adv_mode="scalar"` 的**优势公式**不得被重构改动。

    仓里所有已记录的读数（消融表、训练曲线）都建立在今天的 scalar 口径上——重构不得悄悄
    改动它。判据 = **捕获参照**（固定初始化跑一步，比 `state_dict` 的 sha256）：
    `torch.manual_seed(1234)` → MK01 上跑一步（G=2, seed=0）→ 最终 `state_dict` 的 sha256
    （同 seed 同起点下**跨进程逐位可复现**，见 `test_joint_step_bitwise_reproducible_given_seed`。）
    任一个比特被改动都会让摘要变化——这比"两次调用互相相等"强，后者对两种实现都恒真。

    ⚠️ **旧命题 → 新命题（2026-10-04，批量重算批次）**：
    - 旧：摘要 `9333163ad5910498e62468b0892d827cb9262274e55f9e0399607a93c843d06b`，主张
      "scalar 路径的**输出**与改造前**逐位**相同"。
    - 新：摘要 `6a50aafcd51e4d7c7e0679557869a729ab46766d72759d913ccac2cfaa8ade4a`，主张
      "scalar 的**优势公式**未变（`r = Σwᵢ(−fᵢ)` → `A = z(r)` 一字未动），参数更新自本批起逐位稳定"。
    - 理由：本批把重算的编码器前向从"逐决策单条"改成 `(B,N,F)` **一次批前向**（编码器是
      耗时主项，实测整步 ~2×）。批矩阵乘的分块与单条不同 ⟹ 反传梯度有 ~3.6e-6 的末位差
      ⟹ Adam 更新后的参数末位不同 ⟹ 摘要**必然**改变。这是刻意接受的代价，不是公式变了。
    - ⚠️ **后果（如实写明）**：既往**训练曲线不再逐位可复现**（末位差随步数放大）；
      但**同 seed 的奖励序列不受影响**——动作采样路径（`roll_chain` 的在线前向）一行未动，
      `r` 逐位相同。短程对照与"同 seed 可复现"仍成立，长程数字须重跑才能引用。

    ⚠️ **再捕获（2026-10-04，打分头批量批次）**：摘要改为
    `003583718aa166e56b995d16f679d91c4dad76933861fb398d66e03226d387fa`。理由与上一批同型：
    重算路径的**打分头**从"逐决策单条"改成按 `(kind, n_cand)` 分组的**批量打分** ⟹ 每项
    logp 有 ~3.6e-7 的末位漂移（实测 `max|Δlogp| = 3.58e-07`；`test_batched_recompute_matches_the_per_decision_reference_logp`
    实测）⟹ Adam 更新后的参数末位不同 ⟹ 摘要必然改变。**优势公式仍未动**；动作采样路径
    一行未动 ⟹ 同 seed 的奖励序列仍逐位不变。

    ⚠️ 2026-10-04 早先的三次重捕获（R 头恢复排除 / L 头 token 下标修复 / ⑫⑪ 新头排除）见
    `docs/progress-log.md`；本条记**两次**由数值批次引起的重捕获（批量重算、打分头批量）。
    三个新头（`r_head_tok.*` / `pm_head_tok.*` / `c_head_tok.*`）的排除理由不变：默认关态
    它们拿不到梯度，其参数是新增结构、不进本条"scalar 口径"的证据链。
    ⚠️ **2026-10-05（R2 区段 token）追加排除** `enc.proj_z.*` 与 `enc.zone_type_emb`：同理由
    ——`route_zones=False`（默认）时 Z 段不存在，这两个参数 grad 为 None。**摘要值未重捕获**
    （既有参数逐位不变，已用改造前的参数摘要对拍核实）。
    """
    torch.manual_seed(1234)                     # 网络初始化锚点（捕获参照时的同一序列）
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    ref = ReferenceObjectives.of(inst, cfg)
    torch.manual_seed(1234)                     # 与捕获脚本逐字对齐（ref 之前/之后各锚一次）
    pol = PolicyNet(enc=LayoutEncoder())
    joint_chain_step(pol, inst, lay, dm, cfg, ctx, ref, seed=0, G=2, adv_mode="scalar")
    h = hashlib.sha256()
    for k, v in sorted(pol.state_dict().items()):
        # 排除默认关态**拿不到梯度**的新增结构：R 头 / ⑫M 头 / ⑪C 头（历史批次），
        # R2 的区段投影与区段类型嵌入（2026-10-05），以及 ⑩ 拼批头 `b_head_tok`（2026-10-05）
        # ——它们只在 `route_zones=True` / `batch_head=True` 时被调用，默认配置 grad 为 None
        # （不进 `clip_grad_norm_`、不进 Adam），故排除它们
        # **不削弱**本条对 scalar 口径默认路径的钉法；摘要值因此无需重捕获。
        if k.startswith(("r_head_tok.", "pm_head_tok.", "c_head_tok.", "b_head_tok.",
                         "enc.proj_z.", "enc.zone_type_emb")):
            continue
        h.update(k.encode("utf-8"))
        h.update(v.detach().numpy().tobytes())
    assert h.hexdigest() == "003583718aa166e56b995d16f679d91c4dad76933861fb398d66e03226d387fa", \
        "scalar 路径的输出与捕获参照不再逐位相同——训练轨迹的回归基准变了"


@pytest.mark.unit
def test_per_objective_advantage_is_invariant_to_objective_unit_rescaling():
    """⚠️ O1 的**头号性质**（机制存在的理由）：per_objective 对**逐目标单位缩放不变**，
    scalar 不是。

    `_z` 对正缩放不变（`_z(c·v) = _z(v)`），而"先加权求和、再 z 化"把这个不变性毁掉：
    某目标的量纲一变，它在加权和里的方差占比就变，A 跟着变 ⟹ w 名义上控制权衡、实际被
    目标的物理单位控制。把 energy 全部 ×10（等价于换单位：J → 0.1J），per_objective 的优势
    必须**不变**，scalar 必须**变**——这正是"w 真的在控制权衡"的判据。
    """
    w = MK01_W
    f = np.array([[100.0, 50.0, 10.0],
                  [110.0, 45.0, 20.0],
                  [95.0, 60.0, 5.0],
                  [105.0, 55.0, 15.0]], dtype=np.float64)
    f10 = f.copy()
    f10[:, 1] *= 10.0                          # energy 换单位（其余目标原样）
    a_po, a_po10 = _advantages(f, w, "per_objective"), _advantages(f10, w, "per_objective")
    a_sc, a_sc10 = _advantages(f, w, "scalar"), _advantages(f10, w, "scalar")
    assert torch.allclose(a_po, a_po10, atol=1e-6), \
        "per_objective 未做到逐目标尺度不变——机制没生效"
    assert not torch.allclose(a_sc, a_sc10, atol=1e-6), \
        "scalar 竟然对 energy 单位缩放不敏感——对照组失效，本判据失去意义"
    assert not torch.allclose(a_po, a_sc, atol=1e-6), \
        "两种口径给出相同优势——adv_mode 没有接线"


@pytest.mark.unit
def test_per_objective_advantage_still_responds_to_weights():
    """⚠️ O1 硬要求 3：per_objective 下 `w` 必须仍然**起作用**（否则机制是惰性的）。

    这是上一条的对照：尺度不变性只允许来自 z，不允许把 w 一起抹掉。z_i 各自独立于 w，
    只有合成那一步乘 w——故换 w 必然改变 A。
    """
    f = np.array([[100.0, 50.0, 10.0],
                  [110.0, 45.0, 20.0],
                  [95.0, 60.0, 5.0],
                  [105.0, 55.0, 15.0]], dtype=np.float64)
    a = _advantages(f, MK01_W, "per_objective")
    b = _advantages(f, (0.8, 0.1, 0.1), "per_objective")   # 权重挪到 makespan
    assert not torch.allclose(a, b, atol=1e-6), \
        "改 w 而优势不变——w 在 per_objective 下失效（机制惰性）"


@pytest.mark.unit
def test_per_objective_step_returns_finite_diagnostics():
    """O1 硬要求 4：per_objective 下 A 有限，既有诊断（A_std / r_mean / r_std…）照常可用。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    r, diag = joint_chain_step(pol, inst, lay, dm, cfg, ctx, _ref(inst, cfg),
                               seed=0, G=2, adv_mode="per_objective")
    assert set(diag) >= {"loss", "ratio", "clipped_frac", "grad_norm",
                         "r_mean", "r_std", "A_std"}
    assert all(math.isfinite(float(v)) for v in diag.values()), f"诊断出现非有限值：{diag}"
    assert r == pytest.approx(diag["r_mean"]), "返回值与 r_mean 诊断不一致"
    assert diag["A_std"] > 0.0, "组内优势全同——判据失去意义"
    assert diag["grad_norm"] > 0.0, "梯度范数为 0——诊断退化"


@pytest.mark.unit
def test_per_objective_step_is_invariant_to_energy_unit_rescaling(monkeypatch):
    """端到端版的头号性质：**整步参数更新**对 energy 单位缩放不变（per_objective）/ 会变（scalar）。

    直接测 `_advantages` 只盖住算子本身；这条钉住 `joint_chain_step` **真的把逐目标值喂进了
    新口径**——若 adv_mode 接错线（例如仍走加权和），两种口径的更新会**逐位相同**，立刻变红。

    构造：真实链当载体（logp/梯度有意义），奖励字典用**手写指标**——G=4 且各目标图案
    互不共线。⚠️ 不能用 G=2：两点组内 z 恒为 ±1，**任何**口径都对尺度不敏感，对照会失效
    （本测试初版正是踩了这个坑）。`joint_chain_step` 用完会 `clear()` 决策表，故打桩必须
    返回副本，否则第二次调用看到空链。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    ref = _ref(inst, cfg)
    dec_a, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    dec_b, _ = roll_chain(inst, lay, dm, cfg, pol, seed=1, ctx=ctx)
    # 各目标图案互不共线（mk/en/twt 的组内 z 分别为 [.45,-1.34,1.34,-.45] 等三个不同方向），
    # 于是"先加权求和"与"逐目标 z 化"必然给出不同 A；energy ×10 时前者的 A 变化、后者不变。
    craft = [{"makespan": 100.0, "energy": 50.0, "tardy_twt": 10.0},
             {"makespan": 110.0, "energy": 45.0, "tardy_twt": 20.0},
             {"makespan": 95.0, "energy": 60.0, "tardy_twt": 5.0},
             {"makespan": 105.0, "energy": 55.0, "tardy_twt": 15.0}]
    state = {"g": 0, "scale": 1.0}

    def _fake_roll(*_a, **_k):
        g, state["g"] = state["g"], state["g"] + 1
        dec = dec_a if g % 2 == 0 else dec_b
        met = dict(craft[g])
        met["energy"] *= state["scale"]
        return list(dec), met

    monkeypatch.setattr(group_rel, "roll_chain", _fake_roll)

    def _run(mode: str, scale: float) -> dict:
        state["g"], state["scale"] = 0, scale
        p = copy.deepcopy(pol)
        joint_chain_step(p, inst, lay, dm, cfg, ctx, ref, seed=0, G=4, lr=1e-3, adv_mode=mode)
        return p.state_dict()

    def _max_diff(a: dict, b: dict) -> float:
        return max(float((a[k] - b[k]).abs().max()) for k in a)

    po1, po10 = _run("per_objective", 1.0), _run("per_objective", 10.0)
    sc1, sc10 = _run("scalar", 1.0), _run("scalar", 10.0)
    d_po, d_sc, d_modes = _max_diff(po1, po10), _max_diff(sc1, sc10), _max_diff(po1, sc1)
    assert d_modes > 1e-4, \
        f"两种口径的整步更新相同（最大参数差 {d_modes}）——adv_mode 没有接线"
    assert d_sc > 1e-5, \
        f"scalar 的整步更新对 energy 单位不敏感（最大参数差 {d_sc}）——对照组失效"
    assert d_po < 1e-6, f"per_objective 的整步更新随 energy 单位变化（最大参数差 {d_po}）"
