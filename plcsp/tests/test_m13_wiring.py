"""`m13_train_a` 的**基线档接线**（2026-10-05）。

## 为什么单独立一个测试文件

基线档的六个开关里，**每一个都改变策略看到的输入或动作空间**。训练开、评估关 =
**用另一个策略评估**，而且**静默**——这正是 `_make_eval_fn` 的 docstring 反复警告的那一类。
（先例：`pm_head` 的注释早就写了这句，但当时**没有测试钉住它**。）

故本文件钉两件事：
1. **评估侧**把每个开关**透传**给 `roll_chain`（新增开关时若只改了训练侧，这里会红）；
2. **CLI** 暴露每个开关（否则"基线全开"只能靠改代码）。
"""
from __future__ import annotations

import sys

import pytest

from plcsp import m13_train_a as m13

# 链级开关（改输入 / 动作空间）——训练与评估**都必须**逐字相同
CHAIN_SWITCHES = ("route_zones", "geom_bias", "pm_head", "charge_head")
# SimConfig 级开关（改动力学）——随 `cfg` 到达两侧，同样必须同源
CFG_SWITCHES = ("agv_failover", "machine_age_failure")


def _spy(monkeypatch) -> dict:
    """把 `m13.roll_chain` 换成一个只记录 kwargs 的替身。"""
    seen: dict = {}

    def fake(*_a, **kw):
        seen.clear()
        seen.update(kw)
        return [], {"makespan": 1.0}

    monkeypatch.setattr(m13, "roll_chain", fake)
    return seen


def test_eval_fn_forwards_every_chain_switch(monkeypatch):
    """评估回调必须把**每一个**链级开关透传给 `roll_chain`。

    漏掉任何一个 ⟹ 训练开、评估关 ⟹ 评估的是**另一个策略**，且**不报错**。
    """
    seen = _spy(monkeypatch)
    fn = m13._make_eval_fn(None, None, None, None, None, seeds=1, rule=1.0,
                           route_k=2, route_zones=True, geom_bias=True,
                           pm_head=True, charge_head=True)
    fn(policy=None)

    assert seen["route_k"] == 2
    for k in CHAIN_SWITCHES:
        assert seen[k] is True, f"评估侧没透传 {k}——训练开、评估关会静默地用另一个策略评估"


def test_eval_fn_defaults_are_all_off(monkeypatch):
    """默认档：每个开关为假。**这是"默认关 ⟹ 逐位不变"契约的评估侧一半。**"""
    seen = _spy(monkeypatch)
    m13._make_eval_fn(None, None, None, None, None, seeds=1, rule=1.0)(policy=None)

    assert seen["route_k"] == 1
    for k in CHAIN_SWITCHES:
        assert seen[k] is False, f"{k} 的默认值不是 False——默认档就不再逐位不变了"
    assert seen["sample"] is False, "评估必须走 argmax（sample=False）"


def test_cli_exposes_every_baseline_switch(capsys):
    """CLI 必须暴露基线档的每个开关。

    否则"基线全开"只能靠改代码——而改代码时训练/评估两侧**极易只改一边**。
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
