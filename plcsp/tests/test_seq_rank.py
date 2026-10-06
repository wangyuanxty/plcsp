# -*- coding: utf-8 -*-
"""工序排序入口（`seq_rank`）——**基线侧**的验收判据。

设计在册：`docs/superpowers/specs/2026-10-06-sequencing-design.md` §3（DES 入口）、
§5（基线侧注入）、§7（判据）。

**本文件只覆盖基线侧**（`rollout` / `SimWorld.run` 的静态优先表）。策略侧第七个头 Q
（`run_gated(policy_q=...)`）**不在本批**——它要重训，故 `run_gated` 的形参一个都没动。

三条判据在此落地：
- 判据 1c（§3.4）：关态 `type(m.in_q) is simpy.Store`——**构造层面**的"逐位不变"，
  不是"子类忠实地做了同样的事"；且 `seq_rank` **不进 `SimConfig`** ⟹ `_cfg_key` 不变。
- 判据 2b/2c（§7）：打开后 `seq_reorders > 0` 且 makespan 与关态**不同**。
- 判据 8（§7）：`rollout(seq_rank=...)` 能把指定顺序执行出来；同 rank 退回 FIFO。
"""
from __future__ import annotations

import inspect

import simpy
import pytest

from plcsp.algo.setup import build_setup
from plcsp.env.des import SimConfig, SimWorld, _PriorityStore, check_seq_rank, rollout
from plcsp.env.instances import Instance, load_mk

# ⚠️ 判据 0 实测（本文件用的默认档）：机会率 = `seq_gets_ge2 / in_q_gets` ≈ 22%（MK01）/
#    49%（MK07）/ 28%（MK10）——机制不是死的（⑩ 拼批头的教训：机会率 ≈ 0 则先报告再建）。


def _setup(name: str = "mk01"):
    inst = load_mk(name)
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    return inst, lay, dm, cfg, ctx


def _flat_rank(inst: Instance, value: int = 0) -> list[list[int]]:
    """全同 rank（= 无优先级）——按契约必然退回 FIFO 队首（spec §5.1）。"""
    return [[value] * len(job) for job in inst.jobs]


# ══ 判据 1c：关态是裸 `simpy.Store`（构造层面）══

def test_closed_state_keeps_a_bare_simpy_store():
    """`seq_rank=None` ⟹ `MachineSim.in_q` 是**裸 `simpy.Store`**（spec §3.4）。

    这是"默认关 ⟹ 逐位不变"的**最强形式**：不是"子类忠实做了同样的事"，而是**同一个类**。
    若这条翻红，既有 18 个黄金摘要的"逐位不变"就只剩"逻辑等价"这一档解释力了。
    """
    inst, lay, dm, cfg, _ = _setup()
    w = SimWorld(inst, lay, dm, cfg)
    w.run(seed_chain=0)
    assert w.machines, "机台表为空——判据没被验到"
    for m in w.machines:
        assert type(m.in_q) is simpy.Store, (
            f"关态机台 {m.pad.id} 的输入缓冲不是裸 simpy.Store（{type(m.in_q).__name__}）")


def test_open_state_swaps_the_machines_store_but_not_the_lu_station():
    """开态：机台换 `_PriorityStore`；⚠️ `LuStation.in_q` **必须**仍是裸 `Store`。

    装卸站到站即完工，**不是**加工缓冲（spec §3.3 第 8 行）——把排序透传给它，
    会让"回站落点"也走排序回调（且它没有 `pad.id` 可传），是一处真缺陷。
    """
    inst, lay, dm, cfg, _ = _setup()
    w = SimWorld(inst, lay, dm, cfg)
    w.run(seed_chain=0, seq_rank=_flat_rank(inst))
    assert all(isinstance(m.in_q, _PriorityStore) for m in w.machines)
    assert type(w.lu.in_q) is simpy.Store, "装卸站的 in_q 被排序机制碰了（spec §3.3 第 8 行）"
    # 装载上限必须透传给 `Store`：不透传时 `capacity` 静默变 `inf`，② 有限缓冲失效
    for m in w.machines:
        pad = m.pad
        want = pad.in_cap if m.con.finite_buffer else float("inf")
        assert m.in_q.capacity == want, f"机台 {pad.id} 的容量没透传：{m.in_q.capacity} ≠ {want}"


def test_seq_rank_is_not_a_simconfig_field():
    """`seq_rank` **不进 `SimConfig`**（spec §3.5）：进 cfg 就会进 `_cfg_key` ⟹ `M_ref` 失效。"""
    assert not any("seq" in f.name for f in SimConfig.__dataclass_fields__.values()), (
        "seq_rank 混进了 SimConfig——参考运行缓存 M_ref 会全部失效（spec §3.5）")


# ══ 判据 2b/2c、判据 8：打开后真的重排，且行为真的变了 ══

def test_rank_table_reorders_and_changes_the_makespan():
    """非平凡 rank 表：**真的重排了**（`seq_reorders > 0`）且 makespan 与关态**不同**。

    rank 取 `-oi`（"进度深的先做"）——每一对候选的 `(job, oi)` 组合都不同，故这是
    一张**非平凡**表（全同表只会退回 FIFO，见下一条）。
    """
    inst, lay, dm, cfg, _ = _setup("mk01")
    fifo = SimWorld(inst, lay, dm, cfg).run(seed_chain=0)
    rank = [[-oi for oi in range(len(job))] for job in inst.jobs]
    ranked = SimWorld(inst, lay, dm, cfg).run(seed_chain=0, seq_rank=rank)

    # 关态也有机会（判据 0/2a），但不重排（没有回调可言）
    assert fifo["seq_gets_ge2"] > 0, "关态机会率恒 0——机制是死的（⑩ 拼批头的先例）"
    assert fifo["seq_reorders"] == 0
    # 开态：决策真的发生了（不是"回调被调了"）
    assert ranked["seq_gets_ge2"] > 0
    assert ranked["seq_reorders"] > 0, "有候选却没有一次真重排——排序表没生效"
    # 行为真的变了（判据 2c）
    assert ranked["makespan"] != fifo["makespan"], "排序打开后 makespan 与 FIFO 逐位相同"
    # 完成度守卫：两条都跑完，"makespan 不同"才可比（spec §7 判据 2c 的纪律）
    assert not fifo["horizon_hit"] and not ranked["horizon_hit"]


def test_flat_rank_table_is_bit_identical_to_fifo():
    """全同 rank ⟹ 退回 FIFO（spec §5.1 / §8 第 6 条）——规则栏是 (A) 的一个特例。

    ⚠️ 这是**解码器自身的证据**：走上 `_PriorityStore` 这条路（多一次回调、多一处弹件点）
    而指标逐位相同，说明"排序开/关"的差别**只**来自选择本身，不来自通路开销。
    """
    inst, lay, dm, cfg, _ = _setup("mk01")
    fifo = SimWorld(inst, lay, dm, cfg).run(seed_chain=0)
    flat = SimWorld(inst, lay, dm, cfg).run(seed_chain=0, seq_rank=_flat_rank(inst))
    for k in ("makespan", "energy", "travel_time_total", "moves", "deliveries",
              "jobs_done", "completes", "seq_gets_ge2"):
        assert flat[k] == fifo[k], f"全同 rank 表下 {k} 变了：{flat[k]!r} ≠ {fifo[k]!r}"
    assert flat["seq_reorders"] == 0


def test_uncovered_ops_are_allowed_and_counted_as_inf():
    """`None` = 未覆盖 = `+inf`（spec §5.1）——部分覆盖是**合法**输入，不是畸形输入。"""
    inst = Instance(n_jobs=2, n_machines=2, jobs=[[[(0, 5.0)], [(1, 3.0)]], [[(1, 4.0)]]])
    rows = check_seq_rank(inst, [[1, None], [None]])
    assert rows == [[1.0, float("inf")], [float("inf")]]


def test_partially_covered_rank_table_runs():
    """半覆盖表也能跑（未覆盖的工序排最后）——不报错、不静默改语义。"""
    inst, lay, dm, cfg, _ = _setup("mk01")
    rank = _flat_rank(inst)
    rank[0][0] = -10                      # 只给 (作业 0, 工序 0) 一个高优先级
    met = SimWorld(inst, lay, dm, cfg).run(seed_chain=0, seq_rank=rank)
    assert met["jobs_done"] == inst.n_jobs


def test_rollout_entry_accepts_seq_rank():
    """`rollout(seq_rank=...)` 是**基线侧**的入口（NSGA-II / 规则基线走它，spec §5.1）。"""
    inst = load_mk("mk01")
    fifo = rollout(inst, seed_layout=0, seed_chain=0)
    rank = [[-oi for oi in range(len(job))] for job in inst.jobs]
    ranked = rollout(inst, seed_layout=0, seed_chain=0, seq_rank=rank)
    assert ranked["seq_reorders"] > 0
    assert ranked["makespan"] != fifo["makespan"]


def test_run_gated_takes_no_seq_rank_argument():
    """**接口纪律**（判据 8 后半）：`run_gated` **没有** `seq_rank` 形参。

    在线路径只有 `policy_q` 一个排序入口（静态表要在线用就包一个闭包）——
    两个入口同传会产生"谁说了算"的歧义，故靠**签名**而不是运行时守卫来保证。
    """
    params = inspect.signature(SimWorld.run_gated).parameters
    assert "seq_rank" not in params, "run_gated 多了 seq_rank 形参（spec §5.1 的接口纪律）"


# ══ 判据 8 前半 + §3.2：Store 侧的强制（只在 n ≥ 2 调回调、非候选动作显式报错）══

def test_priority_store_decides_only_when_there_is_a_choice():
    """只在**真有得选**（≥ 2 件）时调回调；1 件时直接弹队首（spec §3.2，Store 侧强制）。"""
    env = simpy.Environment()
    calls: list[tuple[int, tuple[int, ...]]] = []
    st = _PriorityStore(env, capacity=float("inf"), mach=3, stats={},
                        decide=lambda m, jobs: calls.append((m, jobs)) or 1)
    got = []

    def consumer():
        got.append((yield st.get()))
        got.append((yield st.get()))

    st.put((10,))
    st.put((11,))
    env.process(consumer())
    env.run()
    assert calls == [(3, (10, 11))], f"回调调用次数/实参不对：{calls}"
    assert [g[0] for g in got] == [11, 10], "弹件没按动作序号（1 = 后到那件）"


def test_priority_store_rejects_a_non_candidate_action():
    """非候选动作**显式报错**（同其余六头的写法）——静默回退会掩盖"下标写错"这类缺陷。"""
    env = simpy.Environment()
    st = _PriorityStore(env, capacity=float("inf"), mach=0, stats={},
                        decide=lambda m, jobs: 7)          # 候选只有 2 件
    st.put((10,))
    st.put((11,))
    # `StoreGet.__init__` → `_trigger_get` → `_do_get` **同步**弹件，故异常就地抛出
    # （这正是 `MachineSim.run` 走的同一条路径，不是手搓）
    with pytest.raises(ValueError, match=r"动作码非法"):
        st.get()


# ══ 畸形输入必须报错，不得静默忽略 ══

@pytest.mark.parametrize("bad,err", [
    ("not-a-list", TypeError),                             # 外形就不对
    ([[0]], ValueError),                                   # 行数 ≠ 作业数
    ([[0, 0], [0]], ValueError),                           # 行内长度 ≠ 该作业工序数
    ([[0], "x"], TypeError),                               # 行不是序列
    ([[0], [1.5]], TypeError),                             # 元素不是 int
    ([[0], [True]], TypeError),                            # bool 不是 int（显式拒绝）
    ([[0], [0, 0]], ValueError),                           # 行太长（同样按长度查）
])
def test_malformed_rank_is_rejected(bad, err):
    """形态不符**报错**，不静默忽略——静默忽略会把"基线没接上"伪装成"接了没用"。"""
    inst = Instance(n_jobs=2, n_machines=2, jobs=[[[(0, 5.0)]], [[(1, 4.0)]]])
    with pytest.raises(err):
        check_seq_rank(inst, bad)


def test_malformed_rank_is_rejected_at_the_run_entry():
    """生产入口（`rollout` / `run`）同样报错——判据不能只在一个纯函数上手搓。"""
    inst = load_mk("mk01")
    with pytest.raises(ValueError):
        rollout(inst, seq_rank=[[0]] * 3, seed_chain=0)
