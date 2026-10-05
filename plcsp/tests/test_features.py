"""token 特征构造的测试（P2 Task 2）。"""
from __future__ import annotations

import numpy as np
import pytest

import plcsp.env.des as des_mod
from plcsp.algo.group_rel import setup_flag
from plcsp.algo.setup import build_ctx_for_unit_test, build_layout_and_dm
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, SimWorld
from plcsp.env.instances import load_mk
from plcsp.env.snapshot import JobState, MachineState, Snapshot, VehicleState
from plcsp.nn.features import F_B, F_G, F_M, F_MAX, F_V, FEATURE_NAMES, norm_context
from plcsp.nn.state_emb import (build_tok, global_features, job_features,
                                machine_features, vehicle_features)


def _ctx_and_snap(name="mk01", done=False):
    """⚠️ ctx 走 `build_ctx_for_unit_test`（**占位 m_ref**，非生产口径）——本文件是纯特征层
    测试，手搓快照/不跑训练，不应为一次参考运行付墙钟（详见该函数的 docstring）。"""
    inst = load_mk(name)
    cfg = SimConfig()
    lay, dm = build_layout_and_dm(inst, cfg)
    w = SimWorld(inst, lay, dm, cfg)
    if done:
        w.run(seed_chain=1)
    return inst, lay, w, build_ctx_for_unit_test(inst, lay)


def _snap(now=0.0, machines=(), jobs=(), vehicles=(), n_done=0, in_flight=0):
    """手搓快照（**不碰仿真**）——供"按名核对列序"这类需要**指定值**的断言用。

    ⚠️ `zone_holder=()`：手搓快照没有区段表（只有跑仿真的快照才有），
    路线候选特征的争用维按"无争用"读（见 `test_route_choice.py`）。
    """
    return Snapshot(now=now, machines=tuple(machines), jobs=tuple(jobs),
                    vehicles=tuple(vehicles), n_done=n_done, in_flight=in_flight,
                    zone_holder=())


def _backlog_snap(inst, backlog_min):
    """全机台同积压的合成快照（跨实例归一化断言用，不碰仿真）。"""
    m = MachineState(backlog_min=float(backlog_min), in_q_len=0, in_cap=2.0, out_q_len=0,
                     out_cap=2.0, busy=False, remaining_min=0.0, pm_used_min=0.0,
                     fail_rate=0.0, prev_job=-1)
    return _snap(machines=(m,) * inst.n_machines)


@pytest.mark.unit
def test_field_counts_match_declared_widths():
    """⚠️ Review Focus #2：字段名清单与声明宽度必须逐段相等——防静默错位。

    ⚠️ 2026-10-04：① 拥堵 +1（V 段 `zone_wait`）、④ 返工 +1（B 段 `rework_cnt`）
    ⟹ F_B 8→9、F_V 10→11、F_MAX 10→11。末位追加，既有列序不动。
    """
    assert len(FEATURE_NAMES["M"]) == F_M == 7
    assert len(FEATURE_NAMES["B"]) == F_B == 9
    assert len(FEATURE_NAMES["V"]) == F_V == 11
    assert len(FEATURE_NAMES["G"]) == F_G == 3
    assert F_MAX == 11, "F_MAX 应由四段最大值推出——加维后忘了跟？"


@pytest.mark.unit
def test_build_tok_is_one_padded_tensor():
    """输入是**单张** `(N, F_MAX)`——四段按行拼接、列不足处补零（spec §5.3.1）。"""
    inst, lay, w, ctx = _ctx_and_snap()
    tok, seg = build_tok(w.snapshot(), inst, lay, ctx)
    n_m, n_b, n_v, n_g = seg
    assert seg == (inst.n_machines, inst.n_jobs, SimConfig().n_agv, 1)
    assert tok.shape == (n_m + n_b + n_v + n_g, F_MAX)
    # 补零列必须**恒为 0**（M 段 7:、B 段 8:、G 段 3:）
    assert np.all(tok[:n_m, F_M:] == 0.0)
    assert np.all(tok[n_m:n_m + n_b, F_B:] == 0.0)
    assert np.all(tok[-n_g:, F_G:] == 0.0)


@pytest.mark.unit
def test_features_are_finite_and_bounded():
    """归一化后不得出现 NaN/inf，且不应有远超 [0,1] 量级的失控维。"""
    inst, lay, w, ctx = _ctx_and_snap(done=True)
    tok, _ = build_tok(w.snapshot(), inst, lay, ctx)
    assert np.isfinite(tok).all(), "出现 NaN/inf"
    assert np.abs(tok).max() < 20.0, f"有维失控：max|·|={np.abs(tok).max():.1f}"


@pytest.mark.unit
def test_backlog_feature_is_normalized_by_total_work():
    """⚠️ Review Focus #1：积压维必须除以实例总工时——否则 MK01 与 MK10 差 12 倍。

    ⚠️ Review F2：本条此前**只断言 `NormContext` 的量级**，从不穿过特征函数——把积压维
    改回**绝对分钟**它照样全绿（假回归守卫）。故下面两条都补上：
    ① `NormContext` 的标度值；② **真的调用 `machine_features`** 的跨实例断言。
    """
    inst_s, _, ws, ctx_s = _ctx_and_snap("mk01")
    inst_l, _, wl, ctx_l = _ctx_and_snap("mk10")
    assert ctx_l.total_work_min > ctx_s.total_work_min * 10, "两实例总工时应有量级差"
    assert ctx_s.total_work_min == pytest.approx(153.0, rel=0.05)
    assert ctx_l.total_work_min == pytest.approx(1847.0, rel=0.05)

    # ② 穿过特征函数：相同的**相对**积压（各占总工时的 20%）必须映到**同一个**归一化值。
    #    绝对分钟下二者会是 30.6 vs 369.4（差 12 倍），故这条能钉死"除以总工时"。
    frac = 0.2
    f_s = machine_features(_backlog_snap(inst_s, frac * ctx_s.total_work_min), ctx_s)
    f_l = machine_features(_backlog_snap(inst_l, frac * ctx_l.total_work_min), ctx_l)
    assert np.allclose(f_s[:, 0], frac, rtol=1e-6), f"MK01 积压维未按总工时归一：{f_s[0, 0]}"
    assert np.allclose(f_l[:, 0], frac, rtol=1e-6), f"MK10 积压维未按总工时归一：{f_l[0, 0]}"
    assert f_s[0, 0] == pytest.approx(f_l[0, 0], rel=1e-6), "同一相对积压在两实例上不同标度"


@pytest.mark.unit
def test_seg_lengths_come_from_actual_rows_not_hardcoded():
    """⚠️ Review Focus #2：seg 由各段**实际行数**推出，不得硬编码实例规模。"""
    inst, lay, w, ctx = _ctx_and_snap("mk10")
    tok, seg = build_tok(w.snapshot(), inst, lay, ctx)
    assert seg[0] == inst.n_machines == 15
    assert seg[1] == inst.n_jobs == 20
    assert tok.shape[0] == sum(seg)


@pytest.mark.unit
def test_undispatched_vehicle_node_sentinel_is_distinguishable():
    """⚠️ Review F1：`node == -1`（尚未出车）**不是 0 号节点**。

    `des.py` 在 `agv.pos_node is None` 时发哨兵 -1；新造世界里**每一台车**都是这个状态
    （正是 `test_build_tok_is_one_padded_tensor` 用的那种快照）。若按 0 号节点取坐标，
    就会发出**伪造的网格角落几何**，且与"真的停在 0 号节点"**无法区分**。
    """
    inst, lay, w, ctx = _ctx_and_snap()                  # 未 run：车队尚未出车
    snap = w.snapshot()
    assert all(v.node == -1 for v in snap.vehicles), "前提：新造世界的车都还没出车"
    tok, seg = build_tok(snap, inst, lay, ctx)
    n_m, n_b, n_v, _ = seg
    xy = tok[n_m + n_b:n_m + n_b + n_v, 4:6]
    assert np.all(xy == -1.0), f"哨兵应编成 (−1,−1)（真实坐标 ∈ [0,1)）：{xy.tolist()}"
    # 与"真的停在 0 号节点"可区分——0 号节点是网格原点，归一后是 (0,0)
    x0, y0 = ctx.node_xy[0]
    assert not np.allclose(xy[0], (x0 / ctx.bbox_diag, y0 / ctx.bbox_diag)), "哨兵与 0 号节点同码"


@pytest.mark.unit
def test_feature_order_matches_feature_names():
    """⚠️ Review F3：`FEATURE_NAMES` 的**顺序即列偏移**——错序 = 静默错配。

    此前只有长度断言（`test_field_counts_match_declared_widths`），构造函数里的**列序**
    其实无人守。此条手搓一个每维取值可辨认的快照，**逐名**核对「第 i 个名字的值落在第 i 列」。
    （期望**值**按 spec 公式算，此条核对的是**位置**。）
    """
    inst, lay, w, ctx = _ctx_and_snap()
    node = lay.grid.node_id(1, 1)                        # 非 0 号节点：x/y 都不为 0
    nx, ny = ctx.node_xy[node]
    snap = _snap(
        now=1.5 * ctx.m_ref,                             # due_margin = (2.0 − 1.5) = 0.5
        machines=(MachineState(backlog_min=0.30 * ctx.total_work_min, in_q_len=2, in_cap=4.0,
                               out_q_len=1, out_cap=4.0, busy=True,
                               remaining_min=0.40 * ctx.total_work_min,
                               pm_used_min=0.10 * ctx.pm_interval,
                               fail_rate=0.60 * ctx.max_fail_rate, prev_job=-1),
                  ) * ctx.n_m,
        jobs=(JobState(done_ops=1, total_ops=4, remaining_min=0.35 * ctx.total_work_min,
                       finished=False, due=2.0 * ctx.m_ref,
                       at_machine=1, in_transit=True, on_agv=2, rework_cnt=2),
              ) * ctx.n_jobs,
        vehicles=(VehicleState(status=2, node=node, queued=1, battery_frac=0.7,
                               capacity=min(2, ctx.max_capacity), speed_factor=1.1,
                               zone_wait=3.0),
                  ) * ctx.n_agv,
        n_done=3, in_flight=2)

    checks = (
        ("M", machine_features(snap, ctx)[0], {
            "backlog": 0.30, "in_fill": 2 / 4.0, "out_fill": 1 / 4.0, "busy": 1.0,
            "remaining_frac": 0.40, "pm_left": 0.90,
            "fail_rate": 0.60 * ctx.max_fail_rate / max(ctx.max_fail_rate, 1e-9)}),
        ("B", job_features(snap, ctx)[0], {
            "progress": 1 / 4, "remaining_work": 0.35, "due_margin": 0.5, "finished": 0.0,
            "at_machine": 1 / max(ctx.n_m, 1), "in_transit": 1.0,
            "on_agv": 2 / max(ctx.n_agv, 1), "weight": 1.0,
            "rework_cnt": 2 / 4.0}),
        ("V", vehicle_features(snap, ctx)[0], {
            "st_idle": 0.0, "st_empty": 0.0, "st_loaded": 1.0, "st_down": 0.0,
            "node_x": nx / ctx.bbox_diag, "node_y": ny / ctx.bbox_diag,
            "queued": 1 / max(ctx.max_queued, 1), "battery": 0.7,
            "capacity": min(2, ctx.max_capacity) / max(ctx.max_capacity, 1),
            "speed_factor": 1.1,
            "zone_wait": 3.0 / max(ctx.zone_wait_limit, 1e-9)}),
        ("G", global_features(snap, ctx)[0], {
            "time_progress": 1.5, "done_frac": 3 / max(ctx.n_jobs, 1),
            "in_flight": 2 / max(ctx.max_queued, 1)}),
    )
    for key, row, expected in checks:
        names = FEATURE_NAMES[key]
        assert len(names) == len(row) == len(expected)
        for col, name in enumerate(names):
            assert row[col] == pytest.approx(expected[name], rel=1e-5, abs=1e-6), (
                f"{key} 段第 {col} 列应是「{name}」={expected[name]}，实测 {row[col]}")


@pytest.mark.unit
def test_setup_flag_reads_constraint_switch():
    """⚠️ F2（评审裁定）：⑤ 关闭时 `setup_flag` 必须恒 0——仿真里换型时长就是 0。

    旧实现只看 `prev_job != job`，从不读 `ConstraintConfig.setup_time` ⟹ 消融 run（−生产 /
    None）下特征**照样报 1.0**，给策略一个"这里要换型"的假信号（真阴性变假阳性）。
    """
    inst, _lay, _w, _ctx = _ctx_and_snap()
    m = MachineState(backlog_min=0.0, in_q_len=0, in_cap=2.0, out_q_len=0, out_cap=2.0,
                     busy=False, remaining_min=0.0, pm_used_min=0.0, fail_rate=0.0,
                     prev_job=0)
    snap = _snap(machines=(m,) * inst.n_machines)
    full, off = ConstraintConfig(), ConstraintConfig().with_off("setup_time")
    assert setup_flag(snap, 0, 1, full) == 1.0      # 换作业 + 全开：1.0
    assert setup_flag(snap, 0, 0, full) == 0.0      # 同作业：恒 0（与约束无关）
    assert setup_flag(snap, 0, 1, off) == 0.0, "⑤ 关时仍报换型——特征层没读约束开关"


@pytest.mark.unit
def test_capacity_feature_degenerates_when_hetero_off():
    """⚠️ F2：⑩ 关时仿真里**所有**车容量退化为 1（`AgvSim.capacity = spec.capacity if
    hetero else 1`），归一标度 `ctx.max_capacity` 必须跟着退化——否则载量维报
    `1/2 = 0.5`（MK01 实测），等于告诉策略"每台车都只装了半满"（假信号）。
    """
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, dm = build_layout_and_dm(inst, cfg)
    full = ConstraintConfig()
    off = full.with_off("heterogeneous_fleet")

    def capacity_col(cons):
        w = SimWorld(inst, lay, dm, cfg, constraints=cons)
        ctx = norm_context(inst, lay, m_ref=100.0, constraints=cons)
        tok, seg = build_tok(w.snapshot(), inst, lay, ctx)
        n_m, n_b, n_v, _ = seg
        return tok[n_m + n_b:n_m + n_b + n_v, 8]

    on = capacity_col(full)
    assert len(set(on.tolist())) > 1, "前提：⑩ 开时车队异构（载量不同）——否则本判据无区分度"
    deg = capacity_col(off)
    assert np.allclose(deg, 1.0), (
        f"⑩ 关时载量维应退化为同构满值 1.0，实测 {deg.tolist()}"
        f"（ctx.max_capacity 仍按 layout 原始规格算？）")


@pytest.mark.unit
def test_pm_interval_scale_comes_from_cfg():
    """⚠️ F3（评审裁定）：`norm_context` 的 `pm_interval` 必须来自 `SimConfig`，不得硬编码 120。

    可达路径：`m11_constraint_binding` 的极端参数探针正是 `cfg.pm_interval /= 10` → 12.0；
    那里 `pm_left = 1 − pm_clock/120` 而真实间隔是 12，`pm_clock` 一过 120 就被 clip 到 0，
    该维**静默死掉**（spec §9.2 还把 pm_interval 列在 assumed 参数栏要求做敏感性分析）。
    """
    inst, _lay, _w, _ctx = _ctx_and_snap()
    m = MachineState(backlog_min=0.0, in_q_len=0, in_cap=2.0, out_q_len=0, out_cap=2.0,
                     busy=False, remaining_min=0.0, pm_used_min=6.0, fail_rate=0.0,
                     prev_job=-1)
    snap = _snap(machines=(m,) * inst.n_machines)
    pm_col = lambda cfg: machine_features(                      # noqa: E731
        snap, norm_context(inst, _lay, m_ref=100.0, cfg=cfg))[0, 5]
    assert pm_col(SimConfig(pm_interval=120.0)) == pytest.approx(1.0 - 6.0 / 120.0)
    assert pm_col(SimConfig(pm_interval=12.0)) == pytest.approx(1.0 - 6.0 / 12.0), \
        "pm_left 没接 cfg.pm_interval——极端参数（12 min）下该维静默死掉"


@pytest.mark.unit
def test_due_margin_distinguishes_jobs():
    """⚠️ Review Focus #4（**改写**自旧特征化测试"due_margin 是冗余维"，⑧ 重设计 Task 3）。

    旧口径（共同交期 `d_j = τ·M_ref`）下 `due_margin = (d_j − now)/M_ref = τ − time_progress`：
    同一快照内**所有**作业 token 同值 ⟹ 对"作业之间"零区分度（旧测试断言的就是这个）。
    新口径 `d_j = LB·τ·(1 + R·(2ρ_j − 1))`（`due_dates.tf_rdd_due_dates`）**逐作业** ⟹ 有区分度。

    判据双向：① 交期本身逐作业取不同值（前提）；② `job_features` 的第 2 维对不同作业**不同**
    ——即不再是"全等于 clip(τ − time_progress)"那种共同交期形状。此处显式取 `due_range=0.8`
    （跨度最宽），与 Task 1 的同类判据一致；冻结表的 R 下限 0.20 已保证任意实例都有跨度
    （`test_due_wiring.test_due_margin_is_no_longer_redundant` 用真实表值钉 mk01/mk08）。
    """
    from plcsp.env.due_dates import tf_rdd_due_dates

    inst, lay, _w, _ctx = _ctx_and_snap()
    m_ref = 100.0                                   # 本文件惯例：占位 m_ref（纯特征层单测）
    ctx = build_ctx_for_unit_test(inst, lay, m_ref)
    due = tf_rdd_due_dates(inst, tau=2.5, due_range=0.8)
    assert len(set(round(v, 6) for v in due.values())) > 1, "前提不成立：交期还是共同交期"
    jobs = [JobState(done_ops=0, total_ops=4, remaining_min=0.4 * m_ref, finished=False,
                     due=due[j], at_machine=-1, in_transit=True, on_agv=0, rework_cnt=0)
            for j in range(inst.n_jobs)]
    snap = _snap(now=1.0 * m_ref, jobs=jobs)
    col = job_features(snap, ctx)[:, 2]
    # 未触 [-2,2] 裁剪边界（否则"不同"会被 clip 抹平，判据空过）
    assert np.abs(col).max() < 2.0, "due_margin 撞上裁剪边界，本判据在此取值下不成立"
    assert len(np.unique(np.round(col, 6))) > 1, \
        "due_margin 对所有作业同值——又退回共同交期了（该维仍是冗余维）"


@pytest.mark.unit
def test_fail_rate_feature_is_silent_when_machine_failure_off():
    """⚠️ R1a（F2 同类残留）：③ 关闭时 `fail_rate` 维必须恒 0——不得留下"这台机会坏"的假信号。

    仿真侧：`MachineSim._process` 在 ③ 关时直接一次跑完（`fail_events` 恒 0）；特征侧此前
    照报 `pad.fail_rate / max_fail_rate` ⟹ 策略读到一个**动力学里不存在**的量（真阴性变假阳性，
    与 ⑤/⑩ 已修的 F2 同型）。
    判据双向：全开配置必须非零（否则"关时恒 0"恒真、抓不住任何东西），关态必须全 0。
    """
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, _dm = build_layout_and_dm(inst, cfg)
    m = MachineState(backlog_min=0.0, in_q_len=0, in_cap=2.0, out_q_len=0, out_cap=2.0,
                     busy=False, remaining_min=0.0, pm_used_min=0.0, fail_rate=0.01,
                     prev_job=-1)
    snap = _snap(machines=(m,) * inst.n_machines)
    full = ConstraintConfig()
    off = full.with_off("machine_failure")
    col = lambda cons: machine_features(                            # noqa: E731
        snap, build_ctx_for_unit_test(inst, lay, constraints=cons))[:, 6]
    assert col(full).max() > 0.0, "全开配置 fail_rate 维恒 0——判据失去意义"
    assert np.all(col(off) == 0.0), (
        f"③ 关时 fail_rate 维仍报 {col(off).max()}——特征层没读约束开关（假信号）")


@pytest.mark.unit
def test_due_margin_feature_is_silent_when_due_dates_off():
    """⚠️ R1b（F2 同类残留）：⑧ 关闭时 `due_margin` 维必须恒 0——不得留下"有交期"的假信号。

    仿真侧：`SimWorld._due_map` 在 ⑧ 关时返回 `{}`，快照 `JobState.due` 落 0.0 哨兵；
    特征侧此前照算 `(0 − now)/M_ref`（不是 0，长 episode 会饱和到 −2）⟹ 策略读到一个
    **并不存在**的交期紧迫度。故即便快照里带着 `due` 值，开关关掉也必须静默。
    """
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, _dm = build_layout_and_dm(inst, cfg)
    m_ref = 100.0
    # ⚠️ ⑧ 重设计后 `SimConfig().tau` 是 None（覆盖开关）——此处要的是"一个非零交期"，直接用
    # 标定值量级的具体数（本测试随后显式关掉 ⑧，交期数值本身不参与断言）。
    job = JobState(done_ops=0, total_ops=4, remaining_min=0.4 * m_ref, finished=False,
                   due=2.5 * m_ref, at_machine=-1, in_transit=True, on_agv=0, rework_cnt=0)
    snap = _snap(now=0.5 * m_ref, jobs=(job,) * inst.n_jobs)
    full = ConstraintConfig()
    off = full.with_off("due_dates")
    col = lambda cons: job_features(                                # noqa: E731
        snap, build_ctx_for_unit_test(inst, lay, m_ref, constraints=cons))[:, 2]
    assert col(full).max() > 0.0, "全开配置 due_margin 维恒 0——判据失去意义"
    assert np.all(col(off) == 0.0), (
        f"⑧ 关时 due_margin 维仍报 {col(off).max()}——特征层没读约束开关（假信号）")


# ══ ① 拥堵（zone_wait）与 ④ 返工（rework_cnt）：静默 + 活性 ══════════════════
# 两类判据成对：**关掉恒 0**（R1c/R1d，不得留假信号）且**开着真动**（不得是死维）。
# 前者防"约束关掉的假信号"，后者防"维恒零的假特征"——本仓把两者都当缺陷。


@pytest.mark.unit
def test_zone_wait_feature_is_silent_when_congestion_off():
    """⚠️ R1c（R1a/R1b 同型残留）：① 关闭时 `zone_wait` 维必须恒 0——不得留"在等区段"的假信号。

    仿真侧：① 关时 `_drive` 根本不申请区段（直行分支），等待状态永不存在；特征侧仍须读
    `ctx.constraints.congestion` 静默——快照里带着值也**不得**外泄给策略。
    判据双向：全开配置必须非零（否则"关时恒 0"恒真、抓不住任何东西），关态必须全 0。
    """
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, _dm = build_layout_and_dm(inst, cfg)
    v = VehicleState(status=1, node=0, queued=0, battery_frac=0.5, capacity=1,
                     speed_factor=1.0, zone_wait=4.0)
    snap = _snap(vehicles=(v,) * SimConfig().n_agv)
    full = ConstraintConfig()
    off = full.with_off("congestion")
    col = lambda cons: vehicle_features(                                    # noqa: E731
        snap, build_ctx_for_unit_test(inst, lay, constraints=cons))[:, 10]
    assert col(full).max() > 0.0, "全开配置 zone_wait 维恒 0——判据失去意义"
    assert np.all(col(off) == 0.0), (
        f"① 关时 zone_wait 维仍报 {col(off).max()}——特征层没读约束开关（假信号）")


@pytest.mark.unit
def test_rework_cnt_feature_is_silent_when_rework_off():
    """⚠️ R1d（R1a/R1b 同型残留）：④ 关闭时 `rework_cnt` 维必须恒 0——不得留"返过工"的假信号。

    仿真侧：④ 关时 `MachineSim.run` 的重做环一次不进（`rework_events` 恒 0）、逐作业计数恒 0；
    特征侧仍须读 `ctx.constraints.rework` 静默——快照里带着值也不得外泄。
    判据双向：全开配置必须非零，关态必须全 0。
    """
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, _dm = build_layout_and_dm(inst, cfg)
    m_ref = 100.0
    job = JobState(done_ops=0, total_ops=4, remaining_min=0.4 * m_ref, finished=False,
                   due=0.0, at_machine=-1, in_transit=False, on_agv=-1, rework_cnt=2)
    snap = _snap(jobs=(job,) * inst.n_jobs)
    full = ConstraintConfig()
    off = full.with_off("rework")
    col = lambda cons: job_features(                                        # noqa: E731
        snap, build_ctx_for_unit_test(inst, lay, m_ref, constraints=cons))[:, 8]
    assert col(full).max() > 0.0, "全开配置 rework_cnt 维恒 0——判据失去意义"
    assert np.all(col(off) == 0.0), (
        f"④ 关时 rework_cnt 维仍报 {col(off).max()}——特征层没读约束开关（假信号）")


def _spy_zone_waits(monkeypatch, rec: list, w, ctx) -> None:
    """观察每一次区段等待的**解除瞬间**（测试专用，不新增仿真事件、不扰动轨迹）。

    ⚠️ 采样点有两个：① 内层生成器刚让出（等待刚登记，值应为 0）；② `yield` 恢复的**那一瞬**
    ——此刻 `pending` 尚未清、`env.now` 已是获准/超时时刻，故快照里的 `zone_wait` 正是这次
    等待的时长。等 `wait_zone` 返回后再取就恒为 0（车已不在等待）。
    ⚠️ 恢复瞬间的采样必须放在**再次 `gen.send` 之前**：`send` 会把内层生成器一路跑完并清状态，
    放在它后面就再也取不到非零值（探针会静默退化成"只采到 0"，本测试第一版就踩了这个坑）。
    ⚠️ 与 `test_snapshot._spy_agv` 同法：必须**透明转发 send 值**，不得把事件值吞成 None。
    `w.zm is self` 滤掉嵌套 episode——交期要跑参考调度，那是另一个 world。
    """
    orig = des_mod.ZoneManager.wait_zone

    def spy(self, agv, z, limit=None):
        gen, sent, yielded = orig(self, agv, z, limit), None, False

        def sample() -> None:
            if w.zm is self:
                snap = w.snapshot()
                rec.append((float(self.env.now), int(agv),
                            float(snap.vehicles[agv].zone_wait),
                            float(vehicle_features(snap, ctx)[agv, 10])))

        while True:
            if yielded:                        # 恢复瞬间：等待已结束、状态尚未清
                sample()
            try:
                ev = gen.send(sent)
            except StopIteration as stop:
                return stop.value
            yielded = True
            sample()                           # 刚让出：等待刚登记（值 = 0）
            sent = yield ev

    monkeypatch.setattr(des_mod.ZoneManager, "wait_zone", spy)


@pytest.mark.unit
def test_zone_wait_feature_is_live_in_contended_run(monkeypatch):
    """① 开 + 强争用：`zone_wait` 维必须真的出现非零、且随等待时长变化（**不得是死维**）。

    场景：`zone_granularity="row"`（整行一个区段，争用强）+ 2 min 等待上限。探针在每次等待
    解除的瞬间取快照与特征列（见 `_spy_zone_waits`），逐样本核对归一化口径。
    """
    inst = load_mk("mk01")
    cfg = SimConfig(zone_granularity="row", zone_wait_limit=2.0)
    lay, dm = build_layout_and_dm(inst, cfg)
    w = SimWorld(inst, lay, dm, cfg)
    ctx = build_ctx_for_unit_test(inst, lay, cfg=cfg)
    rec: list = []
    _spy_zone_waits(monkeypatch, rec, w, ctx)
    r = w.run(seed_chain=0)
    assert not r["horizon_hit"], "掐表了——本判据的前提（跑完）不成立"
    assert r["zone_wait"]["n"] > 0, "该配置没逼出区段等待——判据失去意义"
    assert rec, "探针没夹到等待样本（观察失效）"
    waits = [x[2] for x in rec]
    cols = [x[3] for x in rec]
    assert max(waits) > 0.0, "快照的 zone_wait 恒 0——没接上 ZoneManager 的等待状态"
    assert max(cols) > 0.0, "① 开且确有等待时 zone_wait 维恒 0——死维"
    assert len({round(c, 6) for c in cols}) > 1, "zone_wait 维不随等待时长变化——无区分度"
    for (ts, _agv, waited, col) in rec:
        assert col == pytest.approx(min(waited / cfg.zone_wait_limit, 1.0), abs=1e-6), (
            f"t={ts}: 快照 zone_wait={waited} → 特征 {col} 与 "
            f"zone_wait_limit={cfg.zone_wait_limit} 口径不符")


@pytest.mark.unit
def test_rework_cnt_tracks_simulation_rework_events():
    """④ 开 + 高返工率：`JobState.rework_cnt` 必须真的计数（**不得是死维**）。

    ⚠️ `p_rework=0.6` 而非 1.0——返工是 `while rng.random() < p` 的重做环，p=1 会**死循环**。
    判据：① 逐作业计数之和 == 仿真 `rework_events`（同一分支里一起 ++，口径最硬）；
    ② 特征第 8 维非零且逐作业不同（归一化后仍有区分度）。
    """
    inst = load_mk("mk01")
    cfg = SimConfig(p_rework=0.6)
    lay, dm = build_layout_and_dm(inst, cfg)
    w = SimWorld(inst, lay, dm, cfg)
    r = w.run(seed_chain=1)
    assert r["rework_events"] > 0, "该配置没逼出返工——判据失去意义"
    snap = w.snapshot()
    cnt = [js.rework_cnt for js in snap.jobs]
    assert sum(cnt) == r["rework_events"], (
        f"逐作业返工计数之和 {sum(cnt)} ≠ 仿真事件数 {r['rework_events']}——"
        "快照没接上 ④ 的计数")
    assert max(cnt) > 0, "④ 开且确有返工事件，但快照的 rework_cnt 全 0"
    ctx = build_ctx_for_unit_test(inst, lay, cfg=cfg)
    col = job_features(snap, ctx)[:, 8]
    assert col.max() > 0.0, "④ 开且确有返工时 rework_cnt 维恒 0——死维"
    assert len({round(v, 6) for v in col.tolist()}) > 1, \
        "rework_cnt 对所有作业同值——无区分度（退化成常数维）"
