"""三目标奖励的测试（P2 Task 6，spec §4.2）。"""
from __future__ import annotations

import pytest

from plcsp.env.reward import (ReferenceObjectives, objective_vector, reward_weights,
                              scalar_reward)
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk


@pytest.mark.unit
def test_objective_vector_reads_three_objectives():
    r = {"makespan": 100.0, "energy": 8.0, "tardy_twt": 12.5}
    assert objective_vector(r) == (100.0, 8.0, 12.5)


# ─────────────────── 完成度守卫（2026-10-06，`progress-log` §52.9） ───────────────────
# Bug：`objective_vector` 只读三项实测值。episode 没跑完时三项全都"看起来更好"
# （makespan 是部分完工的最大值、energy 按更小的 makespan 计、TWT 只算已完工的作业）
# ⟹ "少干活 / 让车队趴窝"是奖励吸引子（⑪ 验证档实测：2/10 完工却报 46.22）。


def _dead_fleet_and_rule(inst, cfg, lay, dm, seed_chain=3):
    """同一 (inst, cfg, 布局) 下的两条真实 episode：规则补电（跑完）vs 恒不充（趴窝）。

    口径与 `test_t3_lagrangian` 的退化守卫一致（小电池 + `battery_low=0` + 恒不充）。
    实测（mk01、0.15 kWh、seed_chain=3）趴窝臂停在 **8/10**——正是奖励吸引子的形态：
    部分完工的 makespan/energy/TWT 三项都更小，**原始奖励反而更好**。
    """
    from plcsp.env.des import CHARGE_CAND_SKIP, SimWorld

    pick_first = lambda snap, job, frm, to, oi, cand: cand[0]        # noqa: E731
    done = SimWorld(inst, lay, dm, cfg).run_gated(seed_chain=seed_chain, policy_l=pick_first)
    dead = SimWorld(inst, lay, dm, cfg).run_gated(
        seed_chain=seed_chain, policy_l=pick_first,
        policy_c=lambda snap, aid, cands: CHARGE_CAND_SKIP)
    return done, dead


def _raw_reward(met, w) -> float:
    """**旧式**奖励（三项实测值直接加权）——守卫之前的口径，用来证明吸引子真的存在。"""
    return -sum(wi * float(met[k]) for wi, k in
                zip(w, ("makespan", "energy", "tardy_twt")))


def _small_battery_world(inst, kwh: float = 0.15):
    from plcsp.algo.setup import build_layout_and_dm
    from plcsp.env.layout import AgvSpec

    cfg = SimConfig(battery_low=0.0)               # 规则档到 0 才补电（同 §33.4 口径）
    lay, dm = build_layout_and_dm(inst, cfg)
    for i, a in enumerate(lay.agvs):
        lay.agvs[i] = AgvSpec(id=a.id, speed_factor=a.speed_factor, capacity=a.capacity,
                              battery_kwh=kwh)     # 小电池验证档（同 test_charge_head）
    return cfg, lay, dm


@pytest.mark.unit
def test_guard_is_bit_identical_when_episode_finished():
    """① **跑完的 episode 奖励逐位不变**——全部既有读数（A 主表、13 个消融 run、D/E/F）靠它。

    ⚠️ 断言用 `==`（不是 approx）：守卫是**分支**，跑完时一个浮点运算都不许多做。
    """
    inst = load_mk("mk01")
    w = reward_weights(ReferenceObjectives.of(inst, SimConfig()).as_tuple())
    r = rollout(inst, seed_chain=1, cfg=SimConfig())
    assert not r["horizon_hit"] and r["jobs_done"] == inst.n_jobs, "本判据的前提（跑完）不成立"

    raw = (float(r["makespan"]), float(r["energy"]), float(r["tardy_twt"]))
    assert objective_vector(r) == raw, "跑完的 episode 被守卫动了——既有读数作废"
    # 与"今日"表达式逐位相同（旧式 = 三项实测值直接加权）
    today = -sum(wi * fi for wi, fi in zip(w, raw))
    assert scalar_reward(objective_vector(r), w) == today
    # 手搓 metrics（无 `horizon_hit` 键）按"跑完"处理——与今日行为一致
    assert objective_vector({"makespan": 1.0, "energy": 2.0, "tardy_twt": 3.0}) == (1.0, 2.0, 3.0)


@pytest.mark.unit
def test_guard_makes_a_real_incomplete_episode_strictly_worse():
    """② 未跑完的奖励**严格更差**——同一 (实例, cfg, 布局) 上：规则补电（跑完）vs 恒不充（趴窝）。

    ⚠️ 本判据同时钉住**吸引子真的存在**（旧式奖励下趴窝臂反而更好）——否则本测试
    失去区分力（"趴窝恰好更差"不能证明守卫在起作用）。
    """
    inst = load_mk("mk01")
    cfg, lay, dm = _small_battery_world(inst)
    done, dead = _dead_fleet_and_rule(inst, cfg, lay, dm)

    assert not done["horizon_hit"] and done["jobs_done"] == inst.n_jobs, "对照臂没跑完"
    assert dead["horizon_hit"], "趴窝臂竟然跑完了"
    assert 0 < dead["jobs_done"] < inst.n_jobs, (
        f"趴窝臂只完成 {dead['jobs_done']}/{inst.n_jobs}——本判据要的是**部分完工**的吸引子")
    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    # 吸引子：旧口径下趴窝臂的奖励**更好**（三项都因少干活而变小）
    assert _raw_reward(dead, w) > _raw_reward(done, w), "本判据失去区分力（旧口径下没有吸引子）"
    r_done = scalar_reward(objective_vector(done), w)
    r_dead = scalar_reward(objective_vector(dead), w)
    assert r_dead < r_done, f"未跑完的奖励 {r_dead} 不严格差于跑完的 {r_done}"
    assert r_done == _raw_reward(done, w), "跑完的奖励被守卫动了"


@pytest.mark.unit
def test_guard_dominates_a_worst_case_finished_episode():
    """② 的强形式：**任何**跑完的 episode 都严格更优。

    用 `des.py` 自己的定义造一个"最差的跑完"字典：完工时刻 ≤ `horizon`；
    能耗 ≤ 能量模型的**结构上界**（机床/AGV 三态时长之和恒等于 makespan，功率取各档上端）；
    TWT ≤ 作业数 × makespan（等权）。守卫必须严格大于这三条界。
    """
    from plcsp.energy import AGV_EMPTY_KW, AGV_IDLE_KW, AGV_LOADED_KW, MACHINE_TIERS, SHOP_FIXED_KW

    inst = load_mk("mk01")
    cfg = SimConfig()
    r = rollout(inst, seed_chain=1, cfg=cfg)
    h = float(r["horizon"])
    max_m_kw = max(max(t["proc"][1], t["idle"][1]) for t in MACHINE_TIERS.values())
    max_a_kw = max(AGV_IDLE_KW, AGV_EMPTY_KW, AGV_LOADED_KW)
    worst = dict(r, horizon_hit=False, makespan=h,
                 energy=(max_m_kw * inst.n_machines + max_a_kw * cfg.n_agv
                         + SHOP_FIXED_KW) * h / 60.0,
                 tardy_twt=float(inst.n_jobs) * h)
    hit = dict(r, horizon_hit=True)

    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    assert scalar_reward(objective_vector(hit), w) < scalar_reward(objective_vector(worst), w), (
        "守卫没有严格压过'最差跑完'——上界三元组不够大")
    # 分量级：三项都必须严格更大（分量级占优才是"任何权重下都成立"的保证）
    f_hit, f_worst = objective_vector(hit), objective_vector(worst)
    assert all(a > b for a, b in zip(f_hit, f_worst))


@pytest.mark.unit
def test_guard_survives_t3_penalty():
    """③ 与 T3 罚项共存：`r' = r − Σλᵢâᵢ` 加在守卫**之后**，且 `λ ≤ T3_LAMBDA_MAX`（clip 上界）。

    用真实两条 metrics + 真实 T3 归一化；λ 取算法**能达到的最大值**（对偶上升被 clip 到
    `T3_LAMBDA_MAX`）——罚项差因此有界，守卫的差远大于它。
    """
    from plcsp.env.t3_budget import T3_LAMBDA_MAX, T3Budget, normalized_activation

    inst = load_mk("mk01")
    cfg, lay, dm = _small_battery_world(inst)
    done, dead = _dead_fleet_and_rule(inst, cfg, lay, dm)
    budget = T3Budget(inst)                      # 几何口径 mk01 六条全在冻结表里
    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    lam = T3_LAMBDA_MAX
    pen_done = lam * float(normalized_activation(done, budget.a_ref, budget.keep).sum())
    pen_dead = lam * float(normalized_activation(dead, budget.a_ref, budget.keep).sum())
    assert (scalar_reward(objective_vector(dead), w) - pen_dead
            < scalar_reward(objective_vector(done), w) - pen_done), \
        "T3 罚项把守卫的序翻过来了（λ 的 clip 上界没有兜住）"


@pytest.mark.unit
def test_guard_requires_horizon_keys():
    """守卫要的四个键缺一不可——**显式报错**，不静默退回实测值（静默退回就是原 bug）。"""
    with pytest.raises(ValueError, match="完成度守卫"):
        objective_vector({"makespan": 1.0, "energy": 1.0, "tardy_twt": 1.0, "horizon_hit": True})


@pytest.mark.unit
def test_weights_are_inverse_reference_and_sum_to_one():
    w = reward_weights((100.0, 8.0, 20.0))
    assert sum(w) == pytest.approx(1.0)
    # 参考值越小 → 权重越大（"相对参考各改进一个单位，贡献相同"）
    assert w[1] > w[2] > w[0]


@pytest.mark.unit
def test_weights_reject_zero_reference():
    """参考值为 0（如 TWT=0 的实例）不得产生 inf 权重——必须显式报错或兜底。"""
    with pytest.raises(ValueError):
        reward_weights((100.0, 8.0, 0.0))


@pytest.mark.unit
def test_scalar_reward_is_monotone_in_each_objective():
    """三个目标各自变小（更好）时，奖励必须上升——这是"多目标"最容易被写反的地方。"""
    w = reward_weights((100.0, 8.0, 20.0))
    base = scalar_reward((100.0, 8.0, 20.0), w)
    assert scalar_reward((90.0, 8.0, 20.0), w) > base      # makespan ↓
    assert scalar_reward((100.0, 7.0, 20.0), w) > base     # energy ↓
    assert scalar_reward((100.0, 8.0, 10.0), w) > base     # TWT ↓


@pytest.mark.unit
def test_reference_objectives_cached_and_consistent_with_m_ref():
    """f^ref 与 M_ref 必须取自**同一次**参考运行（口径一致）。"""
    from plcsp.env.des import reference_makespan
    inst = load_mk("mk01")
    ref = ReferenceObjectives.of(inst, SimConfig())
    assert ref.makespan == pytest.approx(reference_makespan(inst, SimConfig()), rel=1e-9)
    # ⚠️ 三项都须为正的**实测值**——参考运行内交期被短路，TWT 若直接取 dict 恒为 0，
    #    这里会红（0 不是"实测值"，且会让 `reward_weights` 显式报错）。
    assert ref.energy > 0.0 and ref.twt > 0.0
    # 实例指纹：奖励绝对值只在同一实例内有意义，跨实例混用须能被挡住
    assert ref.matches(inst, SimConfig()) and not ref.matches(load_mk("mk07"), SimConfig())


@pytest.mark.unit
def test_reference_objectives_fingerprint_covers_cfg():
    """⚠️ 评审 I-1：f^ref（进而 w）随 **cfg** 变——同实例不同 cfg 的参考值不得互相 `matches`。

    失败场景：P4 扫 `n_agv`（1/3/5）时沿用默认 cfg 算出的 w；指纹若只挡实例，M_ref 已变
    （实测 mk01：n_agv=1/3/5 → 109.95/103.42/97.24；车速 0.5/1.0 → 103.42/106.38）而 w 未变
    ——"按参考调度归一化"在这条扫描轴上**静默**不成立、零报错。交期覆盖开关 `tau` / `due_range`
    同理（⑧ 重设计后是两个字段）：TWT 分量按 TF/RDD 交期事后算，任一变化 TWT（进而 w）跟着变。
    """
    inst = load_mk("mk01")
    ref3 = ReferenceObjectives.of(inst, SimConfig(n_agv=3))
    ref5 = ReferenceObjectives.of(inst, SimConfig(n_agv=5))
    assert ref3.as_tuple() != ref5.as_tuple(), "两 cfg 的 f^ref 实测竟然相同——本判据失去意义"
    assert ref3.matches(inst, SimConfig(n_agv=3)) and ref5.matches(inst, SimConfig(n_agv=5))
    assert not ref3.matches(inst, SimConfig(n_agv=5)), "换了车队规模旧参考值仍 matches——指纹漏 cfg"
    assert not ref5.matches(inst, SimConfig(n_agv=3))
    assert not ref3.matches(inst, SimConfig(n_agv=3, tau=0.50)), "τ 变了仍 matches——指纹漏 τ"
    assert not ref3.matches(inst, SimConfig(n_agv=3, due_range=0.50)), "R 变了仍 matches——指纹漏 R"
    # `None` = 默认 cfg（与 `of` / `reference_run` 的 None 语义一致），**不是**"跳过 cfg 检查"
    assert ref3.matches(inst, SimConfig()) and not ref5.matches(inst, None)


@pytest.mark.unit
def test_end_to_end_reward_on_real_rollout():
    inst = load_mk("mk01")
    ref = ReferenceObjectives.of(inst, SimConfig())
    w = reward_weights(ref.as_tuple())
    r = rollout(inst, seed_chain=1, cfg=SimConfig())
    got = scalar_reward(objective_vector(r), w)
    assert got < 0.0, "奖励应为负（三项都是'越小越好'）"


@pytest.mark.unit
def test_reference_weights_are_pinned_to_the_due_date_caliber():
    """⚠️ **本测试把 `f^ref` / `w` 钉在交期口径上**——口径一改它就红，逼改的人同步更新文档。

    实测（TF/RDD 冻结表，mk01，`SimConfig()`，**P4-B Task 2b 之后**）：
      `f^ref = (117.66, 8.77, **161.52**)` → `w = (0.066, 0.886, 0.048)`（TWT 占 **4.8%**）。
    **为什么它与交期绑定**：`wᵢ = (1/fᵢ^ref)/Σ`，而 TWT 是**事后按当前交期**从参考运行的
    `completes` 算的 —— 交期变 ⟹ `f^ref` 的第三分量变 ⟹ w 变。

    ⚠️ **本批（P4-B Task 2b）的两处变更**，都是结构变化不是调参：
    ① 作业改为**在装卸站入场、完工回站** ⟹ 参考 makespan 103.42 → **117.66**、运输时长
       7.73 → 8.77（每个作业多两条 LU 负载段）；
    ② 几何口径的 `TF_RDD` 随之**重标**（mk01 的 τ/R：2.45/0.2 → 2.75/0.3）⟹ TWT 124.05 → 161.52。
    历史（供追溯，**均已作废**）：旧口径 `d_j = τ·M_ref` 下 `f^ref = (103.42, 7.73, 19.51)`
    → `w = (0.051, 0.680, 0.269)`（TWT 占 26.9%）。
    ⚠️ **改口径的人必须同时改**：本测试、`spec §5.3.3`、`INDEX §5.8`、`progress-log` 里引用
    `f^ref`/`w` 的地方（权重失衡是**研究决定**，不在本测试的处置范围内）。
    """
    from plcsp.env.reward import ReferenceObjectives, reward_weights

    ref = ReferenceObjectives.of(load_mk("mk01"), SimConfig())
    f = ref.as_tuple()
    w = reward_weights(f)
    assert f == pytest.approx((117.66, 8.77, 161.52), rel=1e-3)
    assert w == pytest.approx((0.0660, 0.8859, 0.0481), rel=1e-3)
    assert w[2] == pytest.approx(0.048, abs=5e-3), "TWT 的奖励份额（P4-B 重标后应约 4.8%）"
