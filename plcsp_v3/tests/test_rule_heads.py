# -*- coding: utf-8 -*-
"""v3 的两个新头模块 —— `RuleSelectHeads`（上层）· `RuleParamHeads`（下层）。

设计依据：`docs/v3/design.md` §3（**硬路由**）· §4.2–§4.6（**每条规则一个 head**）。
"""
from __future__ import annotations

import torch
import pytest

from plcsp_v3.hrl.agents import RuleParamHeads, RuleSelectHeads

D = 128


# ══════════════════ 上层：三个独立分类头 ══════════════════

@pytest.mark.unit
def test_select_heads_shape_and_independence():
    """三个头**各自**出 logits（形状 = 各组的类数），**不是一个联合分类**。"""
    sel = RuleSelectHeads(D, {"PA": 12, "LA": 3, "RA": 3})
    out = sel(torch.randn(5, D))
    assert set(out) == {"PA", "LA", "RA"}
    assert tuple(out["PA"].shape) == (5, 12)
    assert tuple(out["LA"].shape) == (5, 3)
    assert tuple(out["RA"].shape) == (5, 3)


@pytest.mark.unit
def test_select_heads_empty_is_rejected():
    with pytest.raises(ValueError, match="不能是空的"):
        RuleSelectHeads(D, {})


# ══════════════════ 下层：每规则一个 head + 硬路由 ══════════════════

@pytest.mark.unit
def test_param_heads_per_rule_sizes_can_differ():
    """**每条规则的参数个数可以不同**（设计 §4.2/§4.4/§4.5 —— 所以是字典不是整数）。"""
    par = RuleParamHeads(D, {"SPT": 2, "EDD": 2, "LATEST_IN_TIME": 1, "AGE": 1})
    x = torch.randn(3, D)
    assert tuple(par(x, "SPT").shape) == (3, 2)
    assert tuple(par(x, "LATEST_IN_TIME").shape) == (3, 1)


@pytest.mark.unit
def test_param_heads_are_hard_routed():
    """**硬路由的核心断言**：没被选的 head **不影响输出**（设计 §3）。

    做法：把**未选中**的那个 head 的权重整个换掉，看输出**是否逐位不变**。
    ⚠️ 这是"上层不可替代"的机械化证据 —— 若是 one-hot 共享头，换掉任意权重都会改变输出。
    """
    par = RuleParamHeads(D, {"SPT": 2, "EDD": 2})
    x = torch.randn(3, D)
    par.eval()
    with torch.no_grad():
        before = par(x, "SPT").clone()
        par.heads["EDD"].weight.fill_(999.0)          # 只动**未选中**的那个
        par.heads["EDD"].bias.fill_(999.0)
        after = par(x, "SPT")
    assert torch.equal(before, after), "未选中的 head 竟然影响了输出 —— 不是硬路由"


@pytest.mark.unit
def test_param_heads_unknown_rule_is_rejected():
    """未知规则**显式报错**，不静默退回"随便挑一个"。"""
    par = RuleParamHeads(D, {"SPT": 2})
    with pytest.raises(ValueError, match="未知的规则"):
        par(torch.randn(2, D), "NOPE")


@pytest.mark.unit
def test_param_heads_ctx_is_required_when_declared():
    """声明了 `ctx_dim` 就**必须**给上下文 —— 静默补 0 会让"头没接上"变成无声的坏策略。"""
    par = RuleParamHeads(D, {"MINCOST": 1}, ctx_dim=4)
    x = torch.randn(2, D)
    with pytest.raises(ValueError, match="逐决策上下文"):
        par(x, "MINCOST")
    assert tuple(par(x, "MINCOST", torch.randn(2, 4)).shape) == (2, 1)


@pytest.mark.unit
def test_param_heads_empty_is_rejected():
    with pytest.raises(ValueError, match="不能没有规则"):
        RuleParamHeads(D, {})


@pytest.mark.unit
def test_param_heads_batch_independent():
    """**批内各行互相独立**（与编码器同一纪律：没有跨 batch 的运算）。"""
    par = RuleParamHeads(D, {"SPT": 2})
    par.eval()
    x = torch.randn(4, D)
    with torch.no_grad():
        full = par(x, "SPT")
        one = par(x[:2], "SPT")
    assert torch.allclose(full[:2], one, atol=1e-6), "批内不独立"
