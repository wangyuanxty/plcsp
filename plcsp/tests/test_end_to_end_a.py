# -*- coding: utf-8 -*-
"""A 端到端验收（P2 Task 8）——三条断言钉住"P2 真的接通了"：

1. **编码器输入非零且是活的**（P0 遗留的"tok_feat 全零"必须已消失——这是 P2 的头号验收项）；
2. **30 步内组均值奖励有改善**（"在学"而非"跑通"）；
3. **训练后 argmax 策略优于规则基线**（严格 `<`，加权标量化 A 的验收线）。

⚠️ **预算（R1 裁定）**：本文件的全部训练预算是 **30 步 × G=4**。MK01 上
`joint_chain_step(G=8)` 实测 18.2 s/步（Task 7）⇒ 300 步 ≈ 1.5 h，**不能进单元测试**。
正式的 300 步运行由脚本 `plcsp/m13_train_a.py` 承担（后台跑、metrics 落 run_dir）。

⚠️ 三条测试只用**一次**训练（模块级 fixture `a_run`）：每条各训一遍会把墙钟翻三倍。
断言 1 自带一次 `roll_chain`（~0.5 s），与训练解耦，故 `-k` 单跑它也成立。

⚠️ **训练 API 自己不带动作采样种子**（Task 7 记录）：`roll_chain` / `joint_chain_step` 的动作
采样走**全局 torch RNG**，不是 `seed=` 那条流。故**只传 seed 不能锁定动作**——跨进程复现必须
在调用序列**开头**自己钉 `torch.manual_seed`（本文件的 fixture 就是这么做的，故它的 30 步读数
可跨进程复现）；论文里的种子对照照此办理（详见 `m13_train_a.py` 的 docstring）。
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
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.env.des import SimConfig, reference_makespan, rollout
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.reward import ReferenceObjectives, reward_weights
from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.state_emb import norm_context

STEPS, G, LR = 30, 4, 3e-4        # R1 裁定的小预算（见模块头）
EVAL_SEEDS = 10                   # argmax 评估的扰动种子数（J>1 只用于评估，spec §5.3.4 约定 3）


def _setup(name: str = "mk01"):
    """返回 (inst, layout, dm, cfg, ctx, policy)——与 `roll_chain` 的签名对齐。

    ⚠️ 布局 seed 固定 0：`SimWorld._due_map` 用**该布局**的 seed 取参考 makespan，而奖励侧
    `ReferenceObjectives.of` 固定用 seed_layout=0 的参考运行——两者同源才有一致的目标口径
    （`joint_chain_step` 入口有守卫，非 0 种子直接报错）。

    归一化标度 `m_ref` 取**真实参考 makespan**（`reference_makespan`，有缓存不额外付代价）而非
    占位常量：spec §5.3.1③ 要求归一化用实例静态量，且这个数与交期 `d_j = τ·M_ref` 的 `M_ref`
    是**同一个**（特征归一化与交期同源）。**本文件与 `m13_train_a.py::build_setup` 同口径。**
    """
    inst = load_mk(name)
    cfg = SimConfig()
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    pol = PolicyNet(enc=LayoutEncoder(), n_agv=cfg.n_agv)
    ctx = norm_context(inst, lay, m_ref=reference_makespan(inst, cfg))
    return inst, lay, dock_distance_matrix(g), cfg, ctx, pol


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
    会把 30 步的小信号淹没。评估种子从 10_000 起，与训练种子（0..29）分离。
    """
    torch.manual_seed(0)     # ⚠️ 必须在 `_setup()` **之前**：网络初始化与动作采样都走**全局
    #                          torch RNG**（Task 7）——不钉种子则每次运行的初始权重与 30 步
    #                          读数都不同（验收数字不可复现、还可能偶发变红）。钉在序列开头后
    #                          整条轨迹（初始化 + 每步采样）被完全确定 ⇒ 本 fixture 可跨进程复现。
    inst, lay, dm, cfg, ctx, pol = _setup()
    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    t0 = time.time()
    rewards = []
    for s in range(STEPS):
        r, _diag = joint_chain_step(pol, inst, lay, dm, cfg, ctx, w, seed=s, G=G, lr=LR)
        rewards.append(float(r))
    secs = time.time() - t0
    trained = tuple(float(roll_chain(inst, lay, dm, cfg, pol, seed=10_000 + s, ctx=ctx,
                                     sample=False)[1]["makespan"])
                    for s in range(EVAL_SEEDS))
    rule = float(rollout(inst, seed_chain=0, cfg=cfg)["makespan"])
    print(f"\n[A-e2e] {STEPS} 步 × G={G}：{secs:.0f} s（{secs / STEPS:.1f} s/步）｜"
          f"r 前 10 = {mean(rewards[:10]):.2f} → 后 10 = {mean(rewards[-10:]):.2f}｜"
          f"argmax makespan = {mean(trained):.1f}（{EVAL_SEEDS} 种子）vs 规则 {rule:.1f}")
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

    def _zeroed(snap, inst_, layout_, ctx_):          # P0 缺陷复现：整张特征恒零
        tok, seg = real(snap, inst_, layout_, ctx_)
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

    ⚠️ **R1 预警过"30 步可能反超不了"**，故本条**实测定档**（同进程、`torch.manual_seed(0)`，
    布局 seed 0、默认 `SimConfig`）：训练后 **86.8** vs 规则 **103.4**（**−16.1%**），
    而未训练的同口径 argmax 是 **238.0**（规则的 2.30×）。**反超余量足够，故保留严格 `<`，
    不降级**；对照组见 Task 8 报告与 `progress-log.md` §十八——它才是"训练确实起了作用"的判据。
    """
    trained, rule = mean(a_run.trained), a_run.rule
    assert trained < rule, (
        f"训练后 argmax makespan {trained:.1f} 未优于规则基线 {rule:.1f}"
        f"（10 种子 {[round(x, 1) for x in a_run.trained]}；30 步 × G={G} 的小预算）")
