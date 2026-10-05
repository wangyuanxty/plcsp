# -*- coding: utf-8 -*-
"""T3 拉格朗日（`docs/t3-design.md`）——**机制**层的验收测试。

覆盖设计的 §6 七条判据里"能在测试里判"的部分：

1. **默认关 ⟹ 逐位不变**：`joint_chain_step` 不带 T3 形参时，参数摘要等于既有的捕获参照
   （`test_joint_chain.test_scalar_adv_mode_regression_pin` 的同一条摘要）。
2. **λ ≡ 0 时与关态逐位相同**：同一配置下传 `t3_lambda=0` 得到的摘要与上一条**同值**。
3. **对偶上升方向**：`â > b ⟹ λ 升`；`â < b ⟹ λ 降`且不为负——用**真实一步**的 diag 验证
   接线（纯函数的逐条性质在 `test_t3_budget.py`）。
4. **② 的计数两处都记**：单卸货点配置只走 `run` 的投递循环、multi-drop 配置只走
   `_deliver_multi` 的循环——两种配置都要 > 0（删任一处计数器都会让对应配置恒 0）。
5. **⑫ 的拆分**：规则配置 `chosen == 0`；策略配置主动保养时 `chosen > 0`；`pm_events == forced + chosen`。
6. **退化守卫（上界）**：动作使用率的判据能区分退化的策略（恒不充 / 恒充）与非退化策略；
   **训练后的**真实读数在 `m13_train_a --t3 --eval-every` 的 `charge_action_usage` / `pm_now_rate`。
7. **③ 纳入后可控**：役龄开时"提前保养 ⟹ 故障变少"；役龄关时该因果消失（对照组）。

⚠️ 本文件不跑长训练（那是 `m13_train_a --t3` 的活）；`runner` 跨步持有 λ 的接线用**桩 step_fn**
单测（快、且直测生产路径的那段代码）。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from plcsp.algo.group_rel import joint_chain_step
from plcsp.algo.policy import PolicyNet
from plcsp.algo.runner import METRICS_NAME, run_training
from plcsp.algo.setup import build_layout_and_dm, build_setup
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import (CHARGE_CAND_SKIP, PM_CAND_DEFER, PM_CAND_NOW, SimConfig,
                           SimWorld, rollout)
from plcsp.env.instances import load_mk
from plcsp.env.layout import AgvSpec
from plcsp.env.reward import ReferenceObjectives
from plcsp.env.t3_budget import (T3_CONSTRAINTS, T3_LAMBDA_MAX, T3Budget, T3Lagrangian,
                                 action_usage, dual_ascent, normalized_activation,
                                 usage_is_degenerate)
from plcsp.nn.encoder import LayoutEncoder

# 既有捕获参照（`test_joint_chain.test_scalar_adv_mode_regression_pin` 的同一条摘要，
# 那两个文件各自持一份、注明同源——两条钉的是同一条默认路径，一个改变该配置行为的改动会同时翻红）。
SCALAR_PIN_DIGEST = "003583718aa166e56b995d16f679d91c4dad76933861fb398d66e03226d387fa"
# 该摘要**不覆盖**的新增结构（默认关态拿不到梯度）：R2 区段投影/类型嵌入、⑩ 拼批头。
_DIGEST_SKIP = ("r_head_tok.", "pm_head_tok.", "c_head_tok.", "b_head_tok.",
                "enc.proj_z.", "enc.zone_type_emb")

# 只需 S/L 两头的约束子集（不需要维护/充电头即可开 T3）——摘要测试用。
CHEAP_KEEP = ("congestion", "finite_buffer", "setup_time")

# ⑪ 退化守卫用的小电池验证档（同 `test_charge_head`：默认 2–4 kWh 放不空 ⟹ 没有充电决策可言）。
SMALL_BATTERY_KWH = 0.10


def _digest(policy) -> str:
    """`state_dict` 的 sha256（排除默认关态拿不到梯度的新增结构）——同既有捕获参照的口径。"""
    h = hashlib.sha256()
    for k, v in sorted(policy.state_dict().items()):
        if k.startswith(_DIGEST_SKIP):
            continue
        h.update(k.encode("utf-8"))
        h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def _step(inst, lay, dm, cfg, ctx, ref, G=2, **kw):
    """固定初始化（`manual_seed(1234)`）跑一步——与捕获参照逐字对齐的锚点。"""
    torch.manual_seed(1234)
    pol = PolicyNet(enc=LayoutEncoder())
    r, diag = joint_chain_step(pol, inst, lay, dm, cfg, ctx, ref, seed=0, G=G, **kw)
    return pol, r, diag


@pytest.fixture(scope="module")
def mk01():
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    ref = ReferenceObjectives.of(inst, cfg)
    return inst, lay, dm, cfg, ctx, ref


@pytest.fixture(scope="module")
def step_off(mk01):
    """T3 关态的一步（摘要 + diag）——本模块多条判据的参照。"""
    inst, lay, dm, cfg, ctx, ref = mk01
    pol, _r, diag = _step(inst, lay, dm, cfg, ctx, ref)
    return _digest(pol), diag


# ─────────────────── 1/2. 默认关逐位不变；λ≡0 与关态逐位相同 ───────────────────

@pytest.mark.unit
def test_t3_off_matches_the_pinned_scalar_digest(mk01, step_off):
    """判据 1：T3 形参不给时，链路**逐位等于既有捕获参照**（默认关 ⟹ 逐位不变）。"""
    digest, _diag = step_off
    assert digest == SCALAR_PIN_DIGEST, \
        "T3 加入后默认路径的参数摘要变了——既有全部训练读数不再成立"


@pytest.mark.unit
def test_lambda_zero_is_bit_identical_to_t3_off(mk01, step_off):
    """判据 2：`λ ≡ 0` 时罚项恒 0 ⟹ 与关态**逐位相同**（"罚项没生效"要能被测出来）。

    ⚠️ 这条不是恒真：罚项路径改的是"优势怎么算"（`z(r − Σλ·â)` vs `_advantages` 的
    `z(r)`）。若接线写错（如把 â 加而不是减、或在 z 化之后加），λ=0 也会漂开。
    """
    inst, lay, dm, cfg, ctx, ref = mk01
    budget = T3Budget(inst, keep=CHEAP_KEEP)
    pol, _r, diag = _step(inst, lay, dm, cfg, ctx, ref,
                          t3_lambda=np.zeros(len(budget)), t3_budget=budget, t3_eta=0.1)
    assert _digest(pol) == step_off[0], "λ≡0 配置与 T3 关态不再逐位相同"
    for k in ("loss", "r_mean", "r_std", "A_std", "grad_norm"):
        assert float(diag[k]) == float(step_off[1][k]), f"λ≡0 配置的 {k} 与关态不同"


# ─────────────────── 3. 对偶上升：方向、上下界、罚项进优势 ───────────────────

@pytest.mark.unit
def test_penalty_enters_the_advantage_and_lambda_follows_dual_ascent(mk01):
    """判据 3（接线）：一步真实运行的 diag 必须满足 `λ_next = clip(λ + η(â − b), 0, λ_max)`，
    且 `â` 的**上下都动**（有升有降——只升不降说明 b 或 â 的接线错了）。

    ⚠️ **G=4**，不是 G=2：G=2 时组内 z 化把优势**恒**压成 (−1, +1)（两点 z 化的形状与
    数值无关：`z = (v−mean)/(std+eps)`，两点恒为 ±1）⟹ 罚项在那里**结构性地**无效。
    这正是设计 §3.3 的警告（z 只消总尺度、不消相对尺度）在小 G 上的极端形态——见
    `test_group_size_two_makes_the_penalty_inert`。

    ⚠️ 与 λ=0 的对照：`r_mean` 必须**不变**（罚项只进优势，不改仿真、不改奖励读数）；
    但 `loss` 会变（优势变了 ⟹ 梯度变了）。
    """
    inst, lay, dm, cfg, ctx, ref = mk01
    budget = T3Budget(inst, keep=CHEAP_KEEP)
    # 先用 λ=0 量一次 â（同时充当 r_mean 的对照），再把 b 取在 â 的中间 ——
    # **保证**上下都有（不依赖某次测量的运气）
    _pol0, _r0, d0 = _step(inst, lay, dm, cfg, ctx, ref, G=4,
                           t3_lambda=np.zeros(len(budget)), t3_budget=budget, t3_eta=0.1)
    ahat0 = np.asarray(d0["t3_ahat"])
    mid = float((ahat0.min() + ahat0.max()) / 2.0)
    b2 = copy.copy(budget)
    b2.b = np.full(len(budget), mid)

    eta, lam_used = 0.1, np.full(len(budget), 1.0)
    _pol, _r, diag = _step(inst, lay, dm, cfg, ctx, ref, G=4, t3_lambda=lam_used,
                           t3_budget=b2, t3_eta=eta)
    ahat = np.asarray(diag["t3_ahat"])
    lam_next = np.asarray(diag["t3_lambda"])
    assert np.all(lam_next >= 0.0) and np.all(lam_next <= T3_LAMBDA_MAX), \
        f"λ 越界：{lam_next}（必须 ∈ [0, {T3_LAMBDA_MAX}]）"
    assert np.allclose(lam_next, dual_ascent(lam_used, ahat, b2.b, eta)), \
        f"λ 的更新式与对偶上升不符：{lam_next} != {dual_ascent(lam_used, ahat, b2.b, eta)}"
    assert np.any(lam_next > lam_used) and np.any(lam_next < lam_used), \
        f"λ 只有单向变化（â={ahat}、b={b2.b}）——对偶上升的接线可疑"
    assert float(diag["t3_pen_mean"]) > 0.0, "罚项 > 0 时 diag 的罚项均值却是 0——没进优势"
    assert float(diag["r_mean"]) == float(d0["r_mean"]), \
        "罚项改了 r_mean——它只该进优势，不该改奖励读数（也不能改仿真）"
    assert float(diag["loss"]) != float(d0["loss"]), \
        "加了罚项但 loss 与 λ=0 配置相同——优势没被改动，罚项是装饰性的"


@pytest.mark.unit
def test_group_size_two_makes_the_penalty_inert(mk01):
    """⚠️ **如实钉住的限度**（设计 §3.3 的警告）：G=2 时组内 z 化把优势压成 (−1, +1)，
    与数值无关 ⟹ 罚项**结构性地**进不了优势（A 逐位相同、参数更新相同）。

    这不是 T3 的缺陷，是"优势只有一个标量、且组内 z 化"的口径在小 G 上的后果：
    `z(v) = (v−mean)/(std+eps)` 在 n=2 时恒为 (−1, +1)·(1/1)（`eps=1e-9` 相对 |a−b|/2 可忽略）。
    ⟹ **T3 的 G 必须 ≥ 3**（本仓主实验配置 G=8）。本条把这条限度写成可执行的判据，
    免得将来有人用 G=2 的读数断言"T3 无效"。
    """
    inst, lay, dm, cfg, ctx, ref = mk01
    budget = T3Budget(inst, keep=CHEAP_KEEP)
    _p0, _r0, d0 = _step(inst, lay, dm, cfg, ctx, ref, G=2,
                         t3_lambda=np.zeros(len(budget)), t3_budget=budget, t3_eta=0.1)
    _p1, _r1, d1 = _step(inst, lay, dm, cfg, ctx, ref, G=2,
                         t3_lambda=np.full(len(budget), 5.0), t3_budget=budget, t3_eta=0.1)
    assert float(d1["t3_pen_mean"]) != float(d0["t3_pen_mean"]), "前提：罚项确实变了"
    assert float(d1["A_std"]) == float(d0["A_std"]), "G=2 时 z 化的形状应不随罚项变"
    assert float(d1["loss"]) == float(d0["loss"]), \
        "G=2 下罚项竟然改变了 loss——上一条的说明该更新了"


@pytest.mark.unit
def test_t3_guards_reject_wrong_wiring_before_any_simulation(mk01):
    """入口守卫：λ 与预算必须成对、adv_mode 必须 scalar、η 必须 > 0、λ 形状必须对得上。

    这些都在**跑链之前**查（写错名/写错形状的代价不该是几分钟仿真）。
    """
    inst, lay, dm, cfg, ctx, ref = mk01
    budget = T3Budget(inst, keep=CHEAP_KEEP)
    zero = np.zeros(len(budget))
    with pytest.raises(ValueError, match="同时给"):
        _step(inst, lay, dm, cfg, ctx, ref, t3_lambda=zero)          # 只给 λ
    with pytest.raises(ValueError, match="同时给"):
        _step(inst, lay, dm, cfg, ctx, ref, t3_budget=budget)        # 只给预算
    with pytest.raises(ValueError, match="scalar"):
        _step(inst, lay, dm, cfg, ctx, ref, adv_mode="per_objective",
              t3_lambda=zero, t3_budget=budget, t3_eta=0.1)
    with pytest.raises(ValueError, match="eta"):
        _step(inst, lay, dm, cfg, ctx, ref, t3_lambda=zero, t3_budget=budget, t3_eta=0.0)
    with pytest.raises(ValueError, match="形状"):
        _step(inst, lay, dm, cfg, ctx, ref, t3_lambda=np.zeros(len(budget) + 1),
              t3_budget=budget, t3_eta=0.1)
    with pytest.raises(ValueError, match="⑫"):
        bad = T3Budget(inst, keep=("maintenance",))
        _step(inst, lay, dm, cfg, ctx, ref, t3_lambda=np.zeros(len(bad)),
              t3_budget=bad, t3_eta=0.1)                             # pm_head=False


# ─────────────────── 4. ② 的计数两处都记 ───────────────────

@pytest.mark.unit
def test_buffer_block_min_is_recorded_on_both_delivery_paths():
    """判据 4：② 的"因缓冲满被迫等待"时长在**两处**投递循环各记一笔。

    路径隔离（`AgvSim.run`）：`multi_drop=False` 只走单卸货点段的循环，
    `multi_drop=True` 直接 `continue` 进 `_deliver_multi` 的循环——一次运行只可能触发一处。
    ⚠️ 删掉任一处计数器 ⟹ 对应配置的 `buffer_block_min` 掉到 0，本测试红（已实地做变异检查）。
    ⑫ 关（`finite_buffer=False`）时缓冲无界（`inf`）⟹ 循环根本不进 ⟹ 恒 0（对照）。
    """
    inst = load_mk("mk01")
    single = rollout(inst, seed_chain=0, cfg=SimConfig())
    multi = rollout(inst, seed_chain=0, cfg=SimConfig(multi_drop=True))
    off = rollout(inst, seed_chain=0, cfg=SimConfig(),
                  constraints=ConstraintConfig(finite_buffer=False))
    assert single["buffer_block_min"] > 0.0, "单卸货点配置没记到阻塞时长（漏了 `run` 的循环？）"
    assert multi["buffer_block_min"] > 0.0, "multi-drop 配置没记到阻塞时长（漏了 `_deliver_multi`？）"
    assert off["buffer_block_min"] == 0.0, "② 关态不该有任何阻塞"


# ─────────────────── 5. ⑫ 的拆键 ───────────────────

@pytest.mark.unit
def test_pm_events_split_counts_only_forced_on_the_rule_path():
    """判据 5：规则配置 ⟹ `chosen == 0`（规则不会主动提前保养）、`forced == pm_events`；
    策略配置主动保养 ⟹ `chosen > 0`；恒等式 `pm_events == forced + chosen` 两种配置都成立。
    """
    inst = load_mk("mk01")
    cfg = SimConfig(pm_interval=10.0)          # 短间隔验证档：默认 120 在 mk01 从不逾期
    lay, dm = build_layout_and_dm(inst, cfg)
    pick_first = lambda snap, job, frm, to, oi, cand: cand[0]        # noqa: E731

    def _run(policy_m):
        return SimWorld(inst, lay, dm, cfg).run_gated(seed_chain=1, policy_l=pick_first,
                                                      policy_m=policy_m)

    rule = _run(None)
    defer = _run(lambda snap, m, cand: PM_CAND_DEFER)
    now = _run(lambda snap, m, cand: PM_CAND_NOW)
    for name, met in (("规则", rule), ("不保养", defer), ("现在保养", now)):
        assert met["pm_events"] == met["pm_events_forced"] + met["pm_events_chosen"], \
            f"{name} 配置：pm_events 不再是 forced + chosen 之和（既有读数的口径被破坏）"
    assert rule["pm_events_forced"] == rule["pm_events"] > 0 and rule["pm_events_chosen"] == 0, \
        f"规则配置应全部是 forced：{rule['pm_events_forced']}/{rule['pm_events']}/" \
        f"{rule['pm_events_chosen']}"
    assert defer["pm_events_chosen"] == 0 and defer["pm_events_forced"] == rule["pm_events_forced"], \
        "「永远不保养」配置应与规则配置同记账（强制底线 = 规则）"
    assert now["pm_events_chosen"] > 0, "「永远现在保养」配置没有记到主动保养——策略的动作没进 chosen"
    assert now["pm_events_forced"] <= rule["pm_events_forced"], \
        ("主动提前保养**不该增加**逾期强制（提前归零时钟）——实得 "
         f"now={now['pm_events_forced']} > rule={rule['pm_events_forced']}")


# ─────────────────── 6. 退化守卫（上界）能区分退化配置 ───────────────────

@pytest.mark.unit
def test_action_usage_guard_distinguishes_degenerate_charge_policies():
    """判据 6（判据本身）：⑪ 的动作使用率能区分"恒不充 / 恒充"与"非退化"三种配置。

    两半：
    - **真机制**：小电池验证档 + `run_gated(policy_c=…)`，三个回调各自记账 ⟹ 使用率 0.0 / 1.0 / 中间；
    - **判据**：把记下的动作喂进 `action_usage` + `usage_is_degenerate`（生产路径的那个函数），
      退化配置必须被判退化、非退化配置必须不被判。

    ⚠️ 上界守卫盯**动作分布**、不盯 λ 的大小（设计 §1.1：③ 与 ⑫ 推同一个动作，单看 λ 会低估
    推动力）。真实的"训练后"读数在 `m13_train_a --t3 --eval-every` 的 `charge_action_usage`。
    """
    inst = load_mk("mk01")
    cfg = SimConfig(battery_low=0.0)           # 规则配置到 0 才补电 ⟹ 车会跑干（同 §33.4）
    lay, dm = build_layout_and_dm(inst, cfg)
    for i, a in enumerate(lay.agvs):           # 小电池验证档（同 test_charge_head）
        lay.agvs[i] = AgvSpec(id=a.id, speed_factor=a.speed_factor,
                              capacity=a.capacity, battery_kwh=SMALL_BATTERY_KWH)
    pick_first = lambda snap, job, frm, to, oi, cand: cand[0]        # noqa: E731

    def _run(cb):
        acts: list[int] = []

        def logging_cb(snap, aid, cands):
            a = int(cb(snap, aid, cands))
            acts.append(a)
            return a

        met = SimWorld(inst, lay, dm, cfg).run_gated(seed_chain=3, policy_l=pick_first,
                                                     policy_c=logging_cb)
        assert len(acts) >= 5, f"C 决策太少（{len(acts)}）——使用率没有区分力"
        # `action_usage` 只读 `.kind` / `.action`（鸭子类型）：生产路径的 `Decision` 与这里的
        # 轻量替身走同一个函数、同一段判据。
        usages = [SimpleNamespace(kind="C", action=a) for a in acts]
        return action_usage(usages, "C", lambda a: a != CHARGE_CAND_SKIP), met

    n = {"k": 0}

    def _biased(snap, aid, cands):
        # 每 5 次决策才去充一次（确定性）——非退化，且足以让车不至于永久趴窝
        n["k"] += 1
        return cands[-1] if n["k"] % 5 == 0 else CHARGE_CAND_SKIP

    skip, skip_met = _run(lambda snap, aid, cands: CHARGE_CAND_SKIP)
    always, always_met = _run(lambda snap, aid, cands: cands[-1])
    biased, biased_met = _run(_biased)

    assert skip == 0.0 and usage_is_degenerate(skip), "恒不充配置的使用率应为 0 且被判退化"
    assert skip_met["horizon_hit"], \
        ("恒不充 ⟹ 电量耗尽后该车永久不可用 ⟹ episode 跑不完（§33 的既定语义）——"
         f"实得 horizon_hit={skip_met['horizon_hit']}")
    assert always == 1.0 and usage_is_degenerate(always), "恒充配置的使用率应为 1 且被判退化"
    assert not always_met["horizon_hit"], "恒充配置应能跑完（它是过度充电、不是趴窝）"
    assert 0.0 < biased < 1.0 and not usage_is_degenerate(biased), \
        (f"非退化策略的使用率 {biased} 落在中间带之外（判据失去区分力）；"
         f"跑完={not biased_met['horizon_hit']}")


# ─────────────────── 7. ③ 纳入后：役龄让故障可被策略影响 ───────────────────

@pytest.mark.unit
def test_failures_respond_to_early_maintenance_only_under_aging():
    """判据 7：**役龄开**时"提前保养 ⟹ 役龄不累积 ⟹ 故障变少"；**役龄关**时该因果消失。

    做法：把机台故障率调高（本配置专用布局，MTBF 短 ⟹ 信号强）、`machine_age_beta=3`（敏感性
    扫的上端，风险曲线上翘更陡）、短 `pm_interval`（役龄在 episode 内真的走完 [0, η]）。
    两个同 seed、同确定性派车（`policy_l` 取队首）的运行只差 M 回调：恒保养 vs 恒不保养。
    ⚠️ 对照组（役龄关）是**必须的**：没有它，"故障变少"可能只是 PM 停机改变了时间轴上的
    抽签次序，而不是役龄被归零。
    """
    inst = load_mk("mk01")
    fail_rate = 0.10                            # 本配置专用（默认 ~0–0.003 ⟹ 信号太弱）
    pick_first = lambda snap, job, frm, to, oi, cand: cand[0]        # noqa: E731

    def _run(aging, policy_m):
        cfg = SimConfig(machine_age_failure=aging, machine_age_beta=3.0, pm_interval=20.0)
        lay, dm = build_layout_and_dm(inst, cfg)
        for i, mp in enumerate(lay.machines):
            lay.machines[i] = replace(mp, fail_rate=fail_rate)
        return SimWorld(inst, lay, dm, cfg).run_gated(
            seed_chain=1, policy_l=pick_first, policy_m=policy_m)

    now = lambda snap, m, cand: PM_CAND_NOW                        # noqa: E731
    defer = lambda snap, m, cand: PM_CAND_DEFER                    # noqa: E731
    aged_now, aged_defer = _run(True, now), _run(True, defer)
    plain_now, plain_defer = _run(False, now), _run(False, defer)

    assert aged_now["fail_events"] < aged_defer["fail_events"], \
        (f"役龄开时提前保养没有减少故障：now={aged_now['fail_events']} "
         f"defer={aged_defer['fail_events']}——③ 不可控，不该纳入 T3")
    assert aged_defer["fail_events"] > 0, "役龄配置的'不保养'侧一次故障都没有——判据没有区分力"
    assert plain_now["fail_events"] >= plain_defer["fail_events"], \
        (f"役龄**关**时故障也随保养减少（now={plain_now['fail_events']} "
         f"defer={plain_defer['fail_events']}）——因果链不是'役龄被归零'，判据失效")
    # 归一化后的 â 同向（a^ref 是冻结表里的 ③ 列；此处只做同向核对）
    a_ref = T3Budget(inst, keep=("machine_failure",)).a_ref[0]
    hat = lambda met: normalized_activation(met, np.array([a_ref]),       # noqa: E731
                                            ("machine_failure",))[0]
    assert hat(aged_now) < hat(aged_defer), "â_③ 没有随动作下降——T3 的 ③ 罚项收不到信号"


# ─────────────────── runner 跨步持有 λ（设计 §3.4 的坑） ───────────────────

@pytest.mark.unit
def test_run_training_holds_lambda_across_steps(tmp_path):
    """设计 §3.4 点名的"最容易写错的一处"：λ 是**跨步状态**，由 `runner` 持有并回注。

    用桩 `step_fn`（不跑仿真）直测生产路径的那段代码：第 s 步收到的 λ 必须是第 s−1 步回传的；
    初值 0；metrics 每行带 `t3_ahat` / `t3_lambda_used`（观测 λ 轨迹与 â 收敛值用）。
    """
    inst = load_mk("mk01")
    budget = T3Budget(inst, keep=("congestion",))
    t3 = T3Lagrangian(budget, eta=0.1)
    seen: list[list[float]] = []

    def stub_step(policy, inst_, seed, **kw):
        seen.append([float(x) for x in kw["t3_lambda"]])
        nxt = float(kw["t3_lambda"][0]) + 0.5           # 假的"对偶上升"
        return 1.0, {"loss": 0.0, "t3_lambda": (nxt,), "t3_lambda_used": tuple(kw["t3_lambda"]),
                     "t3_ahat": (0.25,), "t3_pen_mean": 0.0}

    rd = tmp_path / "run_t3"
    run_training(PolicyNet(), None, steps=3, step_fn=stub_step, step_kwargs={},
                 seed0=0, run_dir=str(rd), save_every=10, t3=t3)
    assert seen == [[0.0], [0.5], [1.0]], f"λ 没有跨步累积（收到 {seen}）——它被每步重置了"
    assert np.allclose(t3.lam, [1.5]), "训练结束后 runner 手里的 λ 不是最后回传的值"
    recs = [json.loads(line) for line in (rd / METRICS_NAME).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert [r["t3_lambda_used"] for r in recs] == [[0.0], [0.5], [1.0]], \
        "metrics 里的 λ 轨迹与注入的不符（观测不到 λ 的收敛过程）"
    assert [r["t3_ahat"] for r in recs] == [[0.25]] * 3, "metrics 丢了 â 的读数"


@pytest.mark.unit
def test_run_training_without_t3_does_not_inject_t3_kwargs(tmp_path):
    """`t3=None`（默认）⟹ 训练环**不注入**任何 T3 形参——默认配置逐位不变的接线前提。"""
    seen = []

    def stub_step(policy, inst_, seed, **kw):
        seen.append(sorted(kw))
        return 0.0, {"loss": 0.0}

    run_training(PolicyNet(), None, steps=2, step_fn=stub_step, step_kwargs={"G": 2},
                 seed0=0, run_dir=str(tmp_path / "run_plain"), save_every=10)
    assert seen == [["G"], ["G"]], f"默认配置注入了额外形参：{seen}"


# ─────────────────── 表与训练的一致性（约束全集可开） ───────────────────

@pytest.mark.unit
def test_all_six_constraints_build_a_budget_for_mk01():
    """六条约束的预算可建、`b` 与 `a^ref` 同长同序——`T3_CONSTRAINTS` 是表列序的唯一真相。"""
    b = T3Budget(load_mk("mk01"))
    assert b.keep == T3_CONSTRAINTS and len(b) == 6
    assert b.a_ref.shape == (6,) and b.b.shape == (6,)
    assert np.all(b.a_ref > 0.0)
