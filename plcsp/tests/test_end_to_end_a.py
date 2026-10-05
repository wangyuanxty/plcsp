# -*- coding: utf-8 -*-
"""A 端到端验收（P2 Task 8）——三条断言钉住"P2 真的接通了"，每条各配一条**牙齿**测试：

1. **编码器输入非零且是活的**（P0 遗留的"tok_feat 全零"必须已消失——这是 P2 的头号验收项）
   ＋ `test_zeroed_features_would_be_caught`（变异守卫：把 `build_tok` 换回全零，断言必须变红）；
2. **30 步内组均值奖励有改善**（"在学"而非"跑通"）；
3. **训练后 argmax 策略优于规则基线**（严格 `<`，加权标量化 A 的验收线）
   ＋ `test_untrained_control_does_not_beat_rule_baseline`（防假绿对照：**未训练**的同口径
   argmax 必须**不**优于规则）——评审 F2 补：没有它，③ 的绿色只能靠"信我"。

⚠️ **预算（R1 裁定）**：本文件的全部训练预算是 **30 步 × G=4**。MK01 上
`joint_chain_step(G=8)` 实测 18.2 s/步（Task 7）⇒ 300 步 ≈ 1.5 h，**不能进单元测试**。
正式的 300 步运行由脚本 `plcsp/m13_train_a.py` 承担（后台跑、metrics 落 run_dir）。

⚠️ 三条测试只用**一次**训练（模块级 fixture `a_run`）：每条各训一遍会把墙钟翻三倍。
断言 1 自带一次 `roll_chain`（~0.5 s），与训练解耦，故 `-k` 单跑它也成立。

⚠️ **动作采样由 `seed` 锁定**（评审 I-3 修后）：`roll_chain` 用
`torch.Generator().manual_seed(seed)` 采样（`joint_chain_step` 自建一条并透传）——**同 seed
同调用序列逐位可复现**，跨进程亦然。`torch.manual_seed(0)` 在本文件的 fixture 里仍要钉，
但它现在只为**网络初始化**（初始化走全局 RNG，不受 `seed=` 影响）；论文里的种子对照照此
办理（详见 `m13_train_a.py` 的 docstring）。
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
import time

import numpy as np
import pytest
import torch

from plcsp.algo.group_rel import joint_chain_step, roll_chain
from plcsp.algo.policy import PolicyNet
from plcsp.algo.setup import build_setup
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk
from plcsp.env.reward import ReferenceObjectives
from plcsp.m13_train_a import EVAL_SEED_BASE       # 评估种子起点：唯一定义处（评审 M-13）
from plcsp.nn.encoder import LayoutEncoder

STEPS, G, LR = 30, 4, 3e-4        # R1 裁定的小预算（见模块头）
EVAL_SEEDS = 10                   # argmax 评估的扰动种子数（J>1 只用于评估，spec §5.3.4 约定 3）
# ⚠️ **评估种子必须避开训练流**（评审 F1）：`joint_chain_step` 第 s 步第 g 条链用
# `seed_chain = (seed0+s)*1000 + g`，本文件 seed0=0 ⇒ 30 步 × G=4 的训练流 ∈ [0, 29004)。
# 旧值 10_000 **与第 10 步的训练流正面相撞**（10000..10003 —— 10 个评估流里 4 个是训练流），
# "评估集与训练集分离"这句话当时是**假的**。安全条件与 `m13_train_a.assert_eval_seed_isolated`
# 同形式（本文件 seed0=0）：**`steps ≤ EVAL_SEED_BASE // 1000 = 1000`**——条件是 `seed0+steps`
# 而非 `steps`（评审 I-2）。`EVAL_SEED_BASE` 自 `m13_train_a` 导入：常量只有一处定义（M-13）。


def _setup(name: str = "mk01"):
    """返回 (inst, layout, dm, cfg, ctx, policy)——与 `roll_chain` 的签名对齐。

    ⚠️ 环境三件套一律走 `algo.setup.build_setup`（评审 F4 收敛：此前五份拷贝里三份漏传
    `aisle_w` / `max_agv_capacity`、三份用 `m_ref=100.0` 占位）。布局 seed 固定 0 与真实的
    参考 makespan 归一化都在 `build_setup` 里定死，本文件不再自留口径。
    """
    inst = load_mk(name)
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    pol = PolicyNet(enc=LayoutEncoder())
    return inst, lay, dm, cfg, ctx, pol


@dataclass(frozen=True)
class _ARun:
    """一次 A 验收训练的读数（fixture 产出，测试只读）。"""
    rewards: tuple[float, ...]          # 每步的组均值奖励（30 个）
    trained: tuple[float, ...]          # 训练后 argmax 策略的 makespan（EVAL_SEEDS 个种子）
    rule: float                         # 规则基线 makespan（每工序最短候选 + 规则派车）
    seconds: float                      # 训练墙钟 [s]（不含评估）


@pytest.fixture(scope="module")
def a_run() -> _ARun:
    """**一次** 30 步 × G=4 的联合链训练 + argmax 评估——测试 2/3 共用，避免重复训练。

    评估用**argmax**（`sample=False`）而非采样：A 要交付的是"训练出来的策略"，采样的方差
    会把 30 步的小信号淹没。评估种子从 `EVAL_SEED_BASE = 10**6` 起——**与训练流严格分离**
    （训练第 s 步第 g 条链的流是 `s*1000+g`，30 步最多到 29003；旧值 10_000 会撞第 10 步，评审 F1）。
    """
    torch.manual_seed(0)     # ⚠️ 必须在 `_setup()` **之前**：**网络初始化**走全局 torch RNG
    #                          （动作采样自评审 I-3 起由 `seed=` 派生，不再吃全局流）——不钉
    #                          则每次运行的初始权重与 30 步读数都不同（验收数字不可复现）。
    #                          钉在序列开头后整条轨迹（初始化 + 每步采样）被完全确定 ⇒ 可复现。
    inst, lay, dm, cfg, ctx, pol = _setup()
    ref = ReferenceObjectives.of(inst, cfg)      # ref 进训练入口（评审 F5：w 由 ref 派生）
    t0 = time.time()
    rewards = []
    for s in range(STEPS):
        r, _diag = joint_chain_step(pol, inst, lay, dm, cfg, ctx, ref, seed=s, G=G, lr=LR)
        rewards.append(float(r))
    secs = time.time() - t0
    trained = tuple(float(roll_chain(inst, lay, dm, cfg, pol, seed=EVAL_SEED_BASE + s, ctx=ctx,
                                     sample=False)[1]["makespan"])
                    for s in range(EVAL_SEEDS))
    rule = float(rollout(inst, seed_chain=0, cfg=cfg)["makespan"])
    print(f"\n[A-e2e] {STEPS} 步 × G={G}：{secs:.0f} s（{secs / STEPS:.1f} s/步）｜"
          f"r 前 10 = {mean(rewards[:10]):.2f} → 后 10 = {mean(rewards[-10:]):.2f}｜"
          f"argmax makespan = {mean(trained):.1f} vs 规则 {rule:.1f}"
          f"（训练后 {EVAL_SEEDS} 种子 {[round(x, 1) for x in trained]}）")
    return _ARun(rewards=tuple(rewards), trained=trained, rule=rule, seconds=secs)


@pytest.mark.unit
def test_encoder_input_is_nonzero_and_live():
    """⚠️ 头号验收项：编码器收到的 token 特征**必须非零**，且随仿真状态**变化**。

    P0 的遗留：`tok_feat` 全零（旧 `encode_state` 的动态量从未接入，几何一删就只剩 0），
    实测一次 `train_step` 编码器被调用 **0 次**（spec §5.3.1 现状表 #1/#2）。
    判据两条：
    - **非零**：任一个决策的整张 `(N, F_MAX)` 若全零 ⇒ 该决策等于喂零给编码器；
    - **是活的**：首末决策的特征**逐位不同**（仿真状态在变，特征必须跟着变）——否则
      "非零"可能只是一份常量在复制（比全零更隐蔽的假接通）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, metrics = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    assert len(decisions) >= 50, f"决策数 {len(decisions)} 太少，覆盖不到在线 S/L 两头的通路"

    sums = np.array([float(np.abs(d.tok).sum()) for d in decisions])
    assert sums.min() > 0.0, (f"第 {int(sums.argmin())}/{len(decisions)} 个决策的 token 特征"
                              f"**全零**——编码器等于吃零输入（P0 遗留 bug 未修复）")
    assert not np.allclose(decisions[0].tok, decisions[-1].tok), \
        "首末决策的 token 特征逐位相同——特征没接活状态（常量伪装成特征）"

    # 段长必须与实例/车队规模一致（seg 由各段实际行数推出，不硬编码实例规模）
    n_m, n_b, n_v, _g = decisions[0].seg
    assert (n_m, n_b, n_v) == (inst.n_machines, inst.n_jobs, cfg.n_agv)
    assert metrics["jobs_done"] == inst.n_jobs, "链没跑完（horizon 掐表）——验收不成立"


@pytest.mark.unit
def test_zeroed_features_would_be_caught(monkeypatch):
    """⚠️ **变异验证**（本仓惯例，见 `test_joint_chain.py::test_l_decisions_alone_send_gradient_to_encoder`）
    ——把 `build_tok` 换回 **P0 的全零输出**，断言 1 必须当场变红。

    理由：断言 1 是"P0 全零输入已消失"的守卫；若它对"全零"不敏感，它就只是**恒真守卫**
    （本计划的评审已被假守卫坑过三次）。此处**复现 P0 缺陷**（动态量没接上 ⇒ 特征恒零），
    确认判据 `min|tok| 求和 > 0` 会失败——即断言 1 对**它要抓的那个缺陷**是敏感的。
    """
    import plcsp.algo.group_rel as gr

    inst, lay, dm, cfg, ctx, pol = _setup()
    real = gr.build_tok

    def _zeroed(snap, inst_, layout_, ctx_, **kw):    # P0 缺陷复现：整张特征恒零
        tok, seg = real(snap, inst_, layout_, ctx_, **kw)   # 透传 R2 的 zof/n_zones（默认 None/0）
        return np.zeros_like(tok), seg

    monkeypatch.setattr(gr, "build_tok", _zeroed)
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    sums = np.array([float(np.abs(d.tok).sum()) for d in decisions])
    assert not (sums.min() > 0.0), \
        "全零输入竟然通过了断言 1 的判据——那条断言是恒真的，抓不住 P0 缺陷"


@pytest.mark.unit
def test_reward_improves_within_30_steps(a_run: _ARun):
    """30 步内组均值奖励必须有改善——A 的"在学"判据（不是"跑通"）。

    口径：**前 10 步均值 vs 后 10 步均值**（单步对比会被 G=4 的组内噪声淹没）。
    奖励 = 三目标加权标量化（越小越好的三目标取负、按参考调度归一化），故"改善"= 变大。
    """
    early, late = mean(a_run.rewards[:10]), mean(a_run.rewards[-10:])
    assert late > early, (f"30 步后组均值奖励未改善：前 10 步 {early:.3f} → 后 10 步 {late:.3f}"
                          f"（{STEPS} 步 × G={G}，lr={LR}；逐步 r="
                          f"{[round(x, 1) for x in a_run.rewards]}）")


@pytest.mark.unit
def test_trained_argmax_beats_rule_baseline(a_run: _ARun):
    """A 的验收线：**训练后**的 argmax 策略在 MK01 上优于规则基线（严格 `<`）。

    规则基线 = `rollout`（每工序取最短候选 + 规则派车，`seed_chain=0`）——即 `M_ref` 的
    同一次运行（与 `ReferenceObjectives` 同源），故这条断言的分母与奖励口径一致。

    ⚠️ **R1 预警过"30 步可能反超不了"**，故本条**按实测确定**（同进程、`torch.manual_seed(0)`，
    布局 seed 0、默认 `SimConfig`）：训练后 **81.8** vs 规则 **103.4**（**−20.9%**），
    逐种子 `[75.0, 78.8, 87.8, 80.8, 79.1, 87.0, 92.3, 86.7, 75.0, 75.1]`。**反超余量足够，
    故保留严格 `<`，不降级**；牙齿由 `test_untrained_control_does_not_beat_rule_baseline` 提供
    （未训练同口径 argmax = **233.2**，规则的 2.25× ⇒ 不训练时本断言必红）。

    📌 **评审 I-3 留痕**（本波）：上面这组读数取自**动作采样接 `seed` 之后**的采样流
    （`torch.Generator().manual_seed(s)`）。修 I-3 前（动作走全局 torch RNG、`seed=` 锁不住）
    同口径读数是 **85.0**（−17.8%）——**旧值跨进程不可复现**（每次进程一个样），故改记新值；
    顺带：`r` 前 10 步均值 −25.91 → 后 10 步 −14.33（"在学"判据的读数）。

    📌 **评审 F1 留痕**：修 F1 前本条的读数是 86.8（−16.1%）——那次评估种子 `10_000..10_009`
    里有 4 个是**第 10 步训练过的流**（`s*1000+g`）。修成 `10**6` 后重测为 85.0（当时仍是
    I-3 之前的采样流）：**泄漏在数值上是噪声级的（还把数字压低了一点），但"评估集与训练集
    分离"这句话之前是假的**——论文 setup 节要写的是修后这条。
    """
    trained, rule = mean(a_run.trained), a_run.rule
    assert trained < rule, (
        f"训练后 argmax makespan {trained:.1f} 未优于规则基线 {rule:.1f}"
        f"（10 种子 {[round(x, 1) for x in a_run.trained]}；30 步 × G={G} 的小预算）")


@pytest.mark.unit
def test_untrained_control_does_not_beat_rule_baseline():
    """⚠️ **防假绿对照**（评审 F2）：断言 ③ 的全部价值在"训练**确实**起了作用"——故这里造一个
    **全新未训练**的 `PolicyNet`，在**同一些评估种子**上跑同口径 argmax，断言它**不优于**规则基线。

    没有这条，"训练后 81.8 < 103.4"只能要求审稿人**信我**（原来那份对照只写在 docstring 与报告
    里，仓库里没有任何东西能证明断言 ③ 会红）。与断言 ① 的变异守卫同理，这条是 ③ 的**牙齿**。
    `torch.manual_seed(0)` 与 fixture 同一起点 ⇒ 这里的"未训练"就是 `a_run` 训练前的那一版参数。
    实测：**233.2 vs 规则 103.4（2.25×）**——不训练时断言 ③ 必红。

    只取 **2** 个扰动种子：对照要的是"量级判据"（它差一个数量级，2 个种子足够），省墙钟。
    """
    torch.manual_seed(0)                     # 必须与 a_run 的起点一致 ⇒ 同一份初始权重
    inst, lay, dm, cfg, ctx, pol = _setup()
    scores = [float(roll_chain(inst, lay, dm, cfg, pol, seed=EVAL_SEED_BASE + s, ctx=ctx,
                               sample=False)[1]["makespan"]) for s in range(2)]
    rule = float(rollout(inst, seed_chain=0, cfg=cfg)["makespan"])
    print(f"[A-e2e] 未训练对照（同 {len(scores)} 个评估种子的 argmax）："
          f"makespan = {mean(scores):.1f} {[round(x, 1) for x in scores]} vs 规则 {rule:.1f}")
    assert mean(scores) > rule, (
        f"**未训练**策略的 argmax makespan {mean(scores):.1f} 竟然不劣于规则基线 {rule:.1f}"
        f"（{len(scores)} 种子 {[round(x, 1) for x in scores]}）——那断言 ③ 就是假绿："
        f"'训练后更优'可能只是随便初始化的功劳，而不是训练的")


@pytest.mark.unit
def test_eval_seed_condition_covers_seed0():
    """⚠️ 评审 I-2：安全条件是 **`seed0 + steps ≤ 1000`**，不是 `steps < 1000`。

    训练流 `(seed0+s)*1000 + g` 随 `--seed` **整体平移**——`--seed 700 --steps 300` 会让评估流
    落回训练流（同一个 bug 在 seed 维度复发），而 `EVAL_SEED_BASE` 那句保证会被抄进论文 setup
    节。故条件由 `m13_train_a.assert_eval_seed_isolated` 在入口**硬查**（不满足即报错，不静默）；
    本测试钉住边界与越界两侧。
    """
    from plcsp.m13_train_a import assert_eval_seed_isolated
    assert_eval_seed_isolated(0, 1000)          # 边界内：不报错
    assert_eval_seed_isolated(700, 300)         # 700+300 = 1000：恰好安全
    with pytest.raises(ValueError, match="评估集与训练集"):
        assert_eval_seed_isolated(700, 301)     # 700+301 > 1000：训练流撞上评估流
    with pytest.raises(ValueError, match="评估集与训练集"):
        assert_eval_seed_isolated(999, 2)
