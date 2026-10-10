# -*- coding: utf-8 -*-
"""RA 保养规则（v3 新增，3 条）—— 见 `docs/v3/design.md` §4.5。

⚠️ **触发点是给定的**（`des.MachineSim._pm_after_op`：工序完工、下一件未上机），
规则只决定"在那个窗口里保不保"。**逾期是硬底线 —— 规则只能提前，不能推迟。**
"""
from __future__ import annotations

import pytest

from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp_v3.env.des import PM_CAND_DEFER, PM_CAND_NOW, SimConfig, SimWorld
from plcsp_v3.env.instances import load_mk
from plcsp_v3.env.layout import sample_layout
from plcsp_v3.hrl.rules import (N_PM_RULES, PM_RULE_NAMES, _resolve_pm_rule,
                                pm_rule_policy)

PM_INTERVAL = 30.0
"""⚠️ 与默认 120 不同 —— **MK01 一局只 ~73 min，`pm_interval=120` 永不触发**（§110 的教训）。"""


def _world(cfg: SimConfig) -> SimWorld:
    inst = load_mk("mk01")
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    return SimWorld(inst, lay, dock_distance_matrix(g), cfg, graph=g)


def _chosen(rule: str, a: float) -> int:
    """跑一局，返回**未逾期时主动保养**的次数（`pm_events_chosen`）。"""
    cfg = SimConfig(pm_interval=PM_INTERVAL)
    m = _world(cfg).run_gated(seed_chain=0,
                              policy_m=pm_rule_policy(rule, a * PM_INTERVAL))
    return int(m["pm_events_chosen"])


@pytest.mark.unit
def test_rule_table_shape():
    """三条规则、名字对得上；非法值**显式报错**。"""
    assert N_PM_RULES == 3
    assert PM_RULE_NAMES == ("AGE", "QUEUE_EMPTY", "OLDEST")
    for i, name in enumerate(PM_RULE_NAMES):
        assert _resolve_pm_rule(i) == name
        assert _resolve_pm_rule(name) == name
    with pytest.raises(ValueError, match="未知的保养规则名"):
        _resolve_pm_rule("NOPE")
    with pytest.raises(ValueError, match="越界"):
        _resolve_pm_rule(N_PM_RULES)


@pytest.mark.unit
def test_negative_threshold_is_rejected():
    """阈值必须 >= 0 —— 负值会让"役龄 >= 阈值"恒真（每次都保）。"""
    with pytest.raises(ValueError, match="须 >= 0"):
        pm_rule_policy("AGE", -1.0)


@pytest.mark.unit
def test_the_three_rules_are_distinguishable():
    """**设计 §13.1 步 2 的断言**：三条规则在同一局上给**不同**的 `pm_events`。

    ⚠️ **不是每个阈值上都不同** —— `A = 1` 时三条全 0（退化成"只靠逾期强制"，
    见 `pm_rule_policy` 的 docstring）。故这里取 `A = 0.2`（实测分离度最大）。

    ⚠️ 判据的严格程度：`AGE` ⊇ `OLDEST` ⊇ `QUEUE_EMPTY`（各多一条"还要…"）
    ⟹ 主动保养次数应当是 **`AGE` 最多、`QUEUE_EMPTY` 最少**。
    """
    a = 0.2
    n = {r: _chosen(r, a) for r in PM_RULE_NAMES}
    assert len(set(n.values())) == 3, f"A={a} 上三条规则不可辨：{n}"
    assert n["AGE"] >= n["OLDEST"] >= n["QUEUE_EMPTY"], f"严格度排序反了：{n}"
    assert n["AGE"] > 0, "AGE 一次都没主动保养 —— 阈值没生效"


@pytest.mark.unit
def test_threshold_at_one_never_acts_proactively():
    """`A = 1` ⟹ **恒不主动**（`pm_clock >= pm_interval` 那一刻已被逾期分支接管）。

    这条钉住"规则只能把保养**提前**，不能推迟"这个语义。
    """
    for r in PM_RULE_NAMES:
        assert _chosen(r, 1.0) == 0, f"{r} 在 A=1 时仍然主动保养了"


@pytest.mark.unit
def test_policy_only_answers_with_the_two_candidate_codes():
    """回调**只答候选码** —— `_pm_after_op` 会校验（`choice not in PM_CANDS` ⟹ 报错）。"""
    cfg = SimConfig(pm_interval=PM_INTERVAL)
    pol = pm_rule_policy("AGE", 0.5 * PM_INTERVAL)
    seen: set[int] = set()

    class _Spy:
        def __call__(self, snap, mach, cands):
            seen.add(pol(snap, mach, cands))
            return PM_CAND_DEFER          # 永远不保，只观察

    _world(cfg).run_gated(seed_chain=0, policy_m=_Spy())
    assert seen, "回调一次都没被调到 —— 决策点不存在？"
    assert seen <= {PM_CAND_NOW, PM_CAND_DEFER}, f"答了非候选码：{seen}"

    return None
