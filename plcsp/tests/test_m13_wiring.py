"""`m13_train_a` 的**全开配置接线**（2026-10-05）。

## 为什么单独立一个测试文件

全开配置的六个开关里，**每一个都改变策略看到的输入或动作空间**。训练开、评估关 =
**用另一个策略评估**，而且**静默**——这正是 `_make_eval_fn` 的 docstring 反复警告的那一类。
（先例：`pm_head` 的注释早就写了这句，但当时**没有测试钉住它**。）

故本文件钉两件事：
1. **评估侧**把每个开关**透传**给 `roll_chain`（新增开关时若只改了训练侧，这里会红）；
2. **CLI** 暴露每个开关（否则"全开"只能靠改代码）。
"""
from __future__ import annotations

import sys

import pytest

from plcsp import m13_train_a as m13
from plcsp.env.des import SimConfig
from plcsp.env.instances import load_mk

# 链级开关（改输入 / 动作空间）——训练与评估**都必须**逐字相同
CHAIN_SWITCHES = ("route_zones", "geom_bias", "pm_head", "charge_head")
# SimConfig 级开关（改动力学）——随 `cfg` 到达两侧，同样必须同源
CFG_SWITCHES = ("agv_failover", "machine_age_failure")


def _eval_fn(**kw):
    """按本文件的口径建一个评估回调（mk01 + 默认 cfg）。

    ⚠️ 自 2026-10-05 起评估回调还要算 **energy / TWT**（`experiment-plan` §H 的三分量、
    实验 E 的前置）⟹ 它现在真的会碰 `inst` 与 `cfg`，替身不能再传 `None`。
    用 **mk01** 而非 `gen_random`：交期标定表是**逐实例名**的，合成实例查不到 (τ, R) 会显式报错。
    """
    return m13._make_eval_fn(load_mk("mk01"), None, None, SimConfig(), None, **kw)


def _spy(monkeypatch) -> dict:
    """把 `m13.roll_chain` 换成一个只记录 kwargs 的替身。"""
    seen: dict = {}

    def fake(*_a, **kw):
        seen.clear()
        seen.update(kw)
        # 完成度两键：`_make_eval_fn` 自 2026-10-05（T3 批）起一并报它们——未跑完的 episode
        # 的 makespan 是**部分完工**的最大值，单看会把掐表读成改进。
        # `energy` / `completes`：三目标补记后评估会读它们（TWT 由 completes + 交期事后算）。
        return [], {"makespan": 1.0, "jobs_done": 1, "horizon_hit": False,
                    "energy": 0.0,
                    "completes": {j: 1.0 for j in range(load_mk("mk01").n_jobs)}}

    monkeypatch.setattr(m13, "roll_chain", fake)
    return seen


def test_eval_fn_forwards_every_chain_switch(monkeypatch):
    """评估回调必须把**每一个**链级开关透传给 `roll_chain`。

    漏掉任何一个 ⟹ 训练开、评估关 ⟹ 评估的是**另一个策略**，且**不报错**。
    """
    seen = _spy(monkeypatch)
    fn = _eval_fn(seeds=1, rule=1.0, route_k=2, route_zones=True, geom_bias=True,
                  pm_head=True, charge_head=True)
    fn(policy=None)

    assert seen["route_k"] == 2
    for k in CHAIN_SWITCHES:
        assert seen[k] is True, f"评估侧没透传 {k}——训练开、评估关会静默地用另一个策略评估"


def test_eval_fn_defaults_are_all_off(monkeypatch):
    """默认配置：每个开关为假。**这是"默认关 ⟹ 逐位不变"契约的评估侧一半。**"""
    seen = _spy(monkeypatch)
    _eval_fn(seeds=1, rule=1.0)(policy=None)

    assert seen["route_k"] == 1
    for k in CHAIN_SWITCHES:
        assert seen[k] is False, f"{k} 的默认值不是 False——默认配置就不再逐位不变了"
    assert seen["sample"] is False, "评估必须走 argmax（sample=False）"


def test_cli_exposes_every_baseline_switch(capsys):
    """CLI 必须暴露全开配置的每个开关。

    否则"全开"只能靠改代码——而改代码时训练/评估两侧**极易只改一边**。
    """
    with pytest.raises(SystemExit):          # --help 在 parse_args 处退出，不会跑到训练
        sys.argv = ["m13_train_a.py", "--help"]
        m13.main()
    out = capsys.readouterr().out

    for flag in ("--route-k", "--route-zones", "--geom-bias",
                 "--pm-head", "--charge-head") + tuple(f"--{s.replace('_', '-')}"
                                                       for s in CFG_SWITCHES):
        assert flag in out, f"CLI 少了 {flag}"


def test_cfg_switches_exist_on_simconfig():
    """`③/⑨` 的开关在 `SimConfig` 上、且**默认关**。

    ⚠️ 它们是**改动力学**的：默认关是"既有读数不被污染"的前提。
    """
    from plcsp.env.des import SimConfig

    c = SimConfig()
    for k in CFG_SWITCHES:
        assert getattr(c, k) is False, f"SimConfig.{k} 的默认值不是 False"


def test_eval_fn_reports_both_rule_columns(monkeypatch):
    """评估必须**同时**报两栏规则 makespan（2026-10-05，A 主对比）。

    - `rule_makespan` = 全约束锚（f^ref / m_ref 的来源）——**刻意不随 constraints 漂**，不得改动；
    - `rule_makespan_same_constraints` = **同约束集**的规则 rollout：约束组消融（`c-*`）的
      policy 跑的是另一个问题，拿全约束的规则当基线是"易问题比难问题"
      （`progress-log.md` §52.2 第 2 条）——只有这栏才是同口径对照。

    两栏缺一不可：删锚 ⟹ f^ref 失去参照；删同约束栏 ⟹ 消融表失真。
    """
    from plcsp.env.constraints import ABLATION_GROUPS

    _spy(monkeypatch)                       # 评估链走替身（本测试只关心规则两栏）
    calls: list[dict] = []

    def fake_rollout(inst, seed_chain=0, cfg=None, constraints=None):
        calls.append({"seed_chain": seed_chain, "constraints": constraints})
        return {"makespan": 42.0}

    monkeypatch.setattr(m13, "rollout", fake_rollout)
    fn = _eval_fn(seeds=1, rule=117.77, constraints=ABLATION_GROUPS["None"])
    out = fn(policy=None)

    assert out["rule_makespan"] == 117.77, "全约束锚那一栏被动了"
    assert out["rule_makespan_same_constraints"] == 42.0
    assert calls == [{"seed_chain": 0, "constraints": ABLATION_GROUPS["None"]}], (
        "同约束集的规则必须用**训练那一份** constraints、seed_chain=0（与锚同一条随机链）")
    out2 = fn(policy=None)                  # 惰性缓存：第二次评估不再重跑规则 rollout
    assert len(calls) == 1, "同约束集的规则被反复重算（它是确定性的，一次即可）"
    assert out2["rule_makespan_same_constraints"] == 42.0
