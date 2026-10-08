# -*- coding: utf-8 -*-
"""两层策略的网络（步 4）：四个 agent 前向通 + 四个 `B` 矩阵的偏置可测。

验收（设计 §14.2 步 4）：**前向通；`g` 的偏置可测（扫 `g` 看动作分布变）**。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plcsp_v2.env import layout as layout_mod                     # noqa: E402
from plcsp_v2.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v2.env.des import SimConfig, SimWorld                  # noqa: E402
from plcsp_v2.env.instances import load_mk                        # noqa: E402
from plcsp_v2.hrl.agents import (G_BINS, G_DIM, LA_CTX_DIM,  # noqa: E402
                                 HierarchicalPolicy, g_to_vec)
from plcsp_v2.hrl.rules import N_LA_RULES, N_PA_RULES             # noqa: E402
from plcsp_v2.nn.features import norm_context              # noqa: E402
from plcsp_v2.nn.state_emb import build_tok                       # noqa: E402


def _fwd(net, tok, seg, bins=None):
    """统一前向入口。

    ⚠️ **LA 头声明了 `LA_CTX_DIM` 的逐决策任务上下文 ⟹ 必须给。** 不给会**显式报错**
    （`agents.py` 里那条 guard）——静默补 0 会让"这个头没接上"变成无声的坏策略。
    这里给 0 是**调用方在声明**"本测试不关心任务身份"，不是默认值。
    """
    ctx = torch.zeros(tok.shape[0], LA_CTX_DIM)
    return net(tok, seg, bins, la_ctx=ctx)


def _tok_seg(name: str = "mk01"):
    inst = load_mk(name)
    cfg = SimConfig()
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    w = SimWorld(inst, lay, dock_distance_matrix(g), cfg, graph=g)
    ctx = norm_context(inst, lay, 75.87, cfg)
    arr, seg = build_tok(w.snapshot(), inst, lay, ctx)
    return torch.as_tensor(arr).unsqueeze(0), seg, len(lay.chargers)


@pytest.mark.unit
def test_forward_shapes_and_masks():
    tok, seg, n_ch = _tok_seg()
    net = HierarchicalPolicy(n_chargers=n_ch)
    out = _fwd(net, tok, seg)
    b = tok.shape[0]
    assert out.g_logits.shape == (b, G_DIM, G_BINS)
    assert out.g_bins.shape == (b, G_DIM) and out.g.shape == (b, G_DIM)
    assert out.pa.shape == (b, N_PA_RULES) == (b, 12)
    assert out.la.shape == (b, N_LA_RULES) == (b, 3)
    assert out.ra_charge.shape == (b, 1 + n_ch)
    assert out.ra_maint.shape == (b, 2)
    # 掩码是**唯一一处定义**，且四段拼接与 `seg` 一致
    mk = HierarchicalPolicy.masks(seg)
    assert mk["PA"] == (0, seg[0] + seg[1]) and mk["RA_maint"] == (0, seg[0])
    assert mk["LA"] == (seg[0] + seg[1], seg[0] + seg[1] + seg[2])
    # ⚠️ **RA·充电 = V ∪ C**（2026-10-09 补的 C 段）：此前桩没进 token 序列，
    # 充电头看不到桩忙不忙 ⟹ "选哪根桩"只剩距离一个维度。
    assert mk["RA_charge"] == (seg[0] + seg[1], seg[0] + seg[1] + seg[2] + seg[3])
    assert mk["C"][1] - mk["C"][0] == seg[3] == n_ch > 0
    assert (out.g >= 0).all() and (out.g <= 1).all()


@pytest.mark.unit
def test_g_binning_is_scale_free_and_covers_all_bins():
    """`g` 全是**比例** ⟹ 换实例定义不变；`n_bins=4` 时档心 = 0.125/0.375/0.625/0.875。"""
    bins = torch.arange(G_BINS).repeat(G_DIM, 1)      # (3, 4)：每维都走遍四档
    v = g_to_vec(bins)
    assert torch.allclose(v[0], torch.tensor([0.125, 0.375, 0.625, 0.875]))
    assert (v > 0).all() and (v < 1).all(), "取档心 ⟹ 两端也落在开区间（偏置不在边界退化）"


@pytest.mark.unit
def test_b_matrices_are_zero_initialised_so_g_has_no_effect_at_the_start():
    """⚠️ **零初始化的那一面代价**：起步时扫 `g` **看不到任何变化**。

    这是**有意**的（见 `hrl/agents.py` 的模块 docstring）：初始策略 = 不带 `g` 的普通策略，
    与"关掉顶层"的消融同分布；`B` 学出来的才叫"`g` 改变了什么"。
    **有人误以为坏了**时，这条断言就是答案。
    """
    tok, seg, n_ch = _tok_seg()
    net = HierarchicalPolicy(n_chargers=n_ch)
    for name, b in net.b_matrices().items():
        assert b.shape[-1] == G_DIM, name
        assert torch.all(b == 0), f"{name} 的 B 不是零初始：{b}"
    g0 = torch.zeros(1, G_DIM, dtype=torch.long)
    g3 = torch.full((1, G_DIM), G_BINS - 1, dtype=torch.long)
    a = _fwd(net, tok, seg, g0)
    b = _fwd(net, tok, seg, g3)
    assert torch.allclose(a.pa, b.pa) and torch.allclose(a.la, b.la)
    assert torch.allclose(a.ra_charge, b.ra_charge) and torch.allclose(a.ra_maint, b.ra_maint)


@pytest.mark.unit
def test_sweeping_g_moves_the_action_distribution_once_b_is_learned():
    """✅ **验收（设计 §14.2 步 4）**：`B` 非零后，**扫 `g` 动作分布真的会变**。

    做法：给四个 `B` 灌一组随机值（等价于"已经学过"的中间状态），再在两个极端 `g`
    上比 logits。**逐头都要变** —— 任一头的 `B` 没接上，这条就红。
    """
    tok, seg, n_ch = _tok_seg()
    net = HierarchicalPolicy(n_chargers=n_ch)
    gen = torch.Generator().manual_seed(0)
    for mod in (net.pa, net.la, net.ra_charge, net.ra_maint):
        with torch.no_grad():
            mod.b.copy_(torch.randn(mod.b.shape, generator=gen))
    # 再**显式**造一个必然翻转排序的偏置：目标 0 压低动作 0、抬高动作 1
    with torch.no_grad():
        net.pa.b[0, 0] = -10.0
        net.pa.b[1, 0] = +10.0
    lo = _fwd(net, tok, seg, torch.zeros(1, G_DIM, dtype=torch.long))
    hi = _fwd(net, tok, seg, torch.full((1, G_DIM), G_BINS - 1, dtype=torch.long))
    for head in ("pa", "la", "ra_charge", "ra_maint"):
        d = (getattr(hi, head) - getattr(lo, head)).abs().max().item()
        assert d > 1e-6, f"{head} 对 g 无反应——B 没接上（最大差 {d}）"
    # 12 条规则的**排序**也要能被 g 改变（不只是数值平移）
    assert not torch.equal(lo.pa.argsort(dim=-1), hi.pa.argsort(dim=-1)), \
        "g 只平移了分数、没改变排序 ⟹ 动作分布没变"


@pytest.mark.unit
def test_bias_is_a_pure_linear_additive_term():
    """`logits = base + B·g` 是**可加**的：同一 `g` 下给 `B` 加一行 `[1,0,0]`，
    每条动作的分应**齐步**抬 `g₀`。

    ⚠️ 传给网络的是**档下标** `g_bins`，连续量由 `g_to_vec` 换算（档心 `(i+0.5)/n`）。
    """
    tok, seg, n_ch = _tok_seg()
    net = HierarchicalPolicy(n_chargers=n_ch)
    bins = torch.tensor([[0, 2, 3]])
    base = _fwd(net, tok, seg, bins).pa
    with torch.no_grad():
        net.pa.b.add_(torch.tensor([[1.0, 0.0, 0.0]] * N_PA_RULES))
    after = _fwd(net, tok, seg, bins).pa
    expect = float(g_to_vec(bins)[0, 0])                  # = (0+0.5)/4 = 0.125
    assert expect == pytest.approx(0.125)
    delta = after - base
    assert torch.allclose(delta, torch.full_like(delta, expect), atol=1e-5),         f"偏置不是齐步常数：{delta.flatten()[:4].tolist()} vs {expect}"


@pytest.mark.unit
def test_missing_la_task_context_is_an_explicit_error():
    """⚠️ LA 头声明了逐决策上下文 ⟹ **不给就报错**。

    静默补 0 会让"任务身份没接上"变成**无声的坏策略**——LA 头会以为每趟任务的
    起讫都是 0 号节点。与"掩码为空报错"同一条纪律。
    """
    tok, seg, n_ch = _tok_seg()
    net = HierarchicalPolicy(n_chargers=n_ch)
    with pytest.raises(ValueError, match="逐决策上下文"):
        net(tok, seg, torch.zeros(1, G_DIM, dtype=torch.long))


@pytest.mark.unit
def test_empty_mask_is_an_explicit_error():
    """掩码为空必须**显式报错**——静默给 0 会让"这个 agent 没接上"变成无声的坏策略。"""
    tok, seg, n_ch = _tok_seg()
    net = HierarchicalPolicy(n_chargers=n_ch)
    z = torch.zeros(1, 4, 128)
    with pytest.raises(ValueError, match="掩码段为空"):
        net.pa(z, (2, 2), None)


@pytest.mark.unit
def test_forward_is_deterministic_and_batch_rows_are_independent():
    tok, seg, n_ch = _tok_seg()
    net = HierarchicalPolicy(n_chargers=n_ch)
    a = _fwd(net, tok, seg, torch.zeros(1, G_DIM, dtype=torch.long)).pa
    b = _fwd(net, tok, seg, torch.zeros(1, G_DIM, dtype=torch.long)).pa
    assert torch.allclose(a, b), "同输入必须同输出（无 dropout / 无随机）"
    both = torch.cat([tok, tok], dim=0)
    c = _fwd(net, both, seg, torch.zeros(2, G_DIM, dtype=torch.long)).pa
    assert torch.allclose(c[0], c[1], atol=1e-5), "批内各行独立（无跨 batch 注意力）"
    assert all(np.isfinite(v) for v in c.flatten().tolist())
