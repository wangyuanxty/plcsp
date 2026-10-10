# -*- coding: utf-8 -*-
"""两层策略的网络（v3）：前向形状 · 掩码 · **硬路由**。

⚠️ **2026-10-10 改到 v3 口径**（`docs/v3/progress-log.md` §9.5）。
本文件原属 v2，测的是 **`g`（3 维 × 4 档）+ 三个 `B` 矩阵的加性偏置**。
**v3 把这两样都删了**（设计 `docs/v3/design.md` §8）⟹ **四条失去对象的测试一并删除**：

| 删掉的 | 原测什么 | 为什么 v3 没有对象 |
|---|---|---|
| `test_g_binning_is_scale_free_and_covers_all_bins` | `g_to_vec` 的档心 | `g` 在 v3 不存在 |
| `test_b_matrices_are_zero_initialised_…` | `B` 零初始化 ⟹ 起步扫 `g` 无变化 | **没有 `B`** |
| `test_sweeping_g_moves_the_action_distribution_…` | `B` 学非零后扫 `g` 动作变 | 同上 |
| `test_bias_is_a_pure_linear_additive_term` | `logits = base + B·g` 可加 | 同上 |

**接替它们的是**（设计 §3"上层的不可替代性"）：

- `test_the_selected_rule_routes_to_its_own_param_head` —— **硬路由**的机械化证据；
- `test_frozen_rules_are_used_verbatim` —— 冻结档下规则号**不漂**。

⚠️ **`G_DIM` / `G_BINS` / `g_to_vec` 在 `agents.py` 里已是死代码**（v3 无人调用），
**清理由后续单独一步做**（`g_pareto.py` 等 v2 实验脚本还 import 着它们）。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plcsp_v3.env import layout as layout_mod                     # noqa: E402
from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v3.env.des import SimConfig, SimWorld                  # noqa: E402
from plcsp_v3.env.instances import load_mk                        # noqa: E402
from plcsp_v3.hrl.agents import (LA_CTX_DIM, N_LA_PARAMS,  # noqa: E402
                                 N_PA_JOB_PARAMS, N_PA_MACH_PARAMS, N_RA_PARAMS,
                                 PA_PARAM_TOTAL, HierarchicalPolicy, split_pa_rule)
from plcsp_v3.hrl.rules import (N_LA_RULES, N_PA_RULES,  # noqa: E402
                                N_PM_RULES, PA_JOB_RULES, PA_MACHINE_RULES)
from plcsp_v3.nn.features import norm_context              # noqa: E402
from plcsp_v3.nn.state_emb import build_tok                       # noqa: E402


def _fwd(net, tok, seg, rules=None):
    """统一前向入口。

    ⚠️ **LA 头声明了 `LA_CTX_DIM` 的逐决策任务上下文 ⟹ 必须给。** 不给会**显式报错**
    （`RuleParamHeads._with_ctx` 的 guard）——静默补 0 会让"这个头没接上"变成无声的坏策略。
    这里给 0 是**调用方在声明**"本测试不关心任务身份"，不是默认值。
    """
    ctx = torch.zeros(tok.shape[0], LA_CTX_DIM)
    return net(tok, seg, rules=rules, la_ctx=ctx)


def _tok_seg(name: str = "mk01"):
    inst = load_mk(name)
    cfg = SimConfig()
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    w = SimWorld(inst, lay, dock_distance_matrix(g), cfg, graph=g)
    ctx = norm_context(inst, lay, 75.87, cfg)
    arr, seg = build_tok(w.snapshot(), inst, lay, ctx)
    return torch.as_tensor(arr).unsqueeze(0), seg


@pytest.mark.unit
def test_forward_shapes_and_masks():
    """三个**独立**分类头 + 四个参数头；掩码是**唯一一处定义**。"""
    tok, seg = _tok_seg()
    net = HierarchicalPolicy()
    b = tok.shape[0]
    out = _fwd(net, tok, seg)
    # ── 上层：三个独立分类头（**不是一个联合分类**，设计 §4.6）──
    assert set(out.rule_logits) == {"PA", "LA", "RA"}
    assert out.rule_logits["PA"].shape == (b, N_PA_RULES) == (b, 12)
    assert out.rule_logits["LA"].shape == (b, N_LA_RULES) == (b, 3)
    assert out.rule_logits["RA"].shape == (b, N_PM_RULES) == (b, 3)
    for k, lg in out.rule_logits.items():
        assert out.rule_sel[k].shape == (b,), k
        assert 0 <= int(out.rule_sel[k].min()) and int(out.rule_sel[k].max()) < lg.shape[-1]
    # ── 下层：连续参数 μ；**每条规则只出自己那几个**（硬路由 ⟹ 宽度是"那一条"的）──
    assert out.param_mu["PA"].shape == (b, PA_PARAM_TOTAL) == (b, 3)
    assert out.param_mu["LA"].shape == (b, N_LA_PARAMS)
    assert out.param_mu["RA"].shape == (b, N_RA_PARAMS)
    # ── critic 是 opt-in（`test_flat.py` 有专测）──
    assert out.value is None
    # ── 掩码：四段拼接与 `seg` 一致 ──
    mk = HierarchicalPolicy.masks(seg)
    assert mk["PA"] == (0, seg[0] + seg[1]) and mk["RA_maint"] == (0, seg[0])
    assert mk["LA"] == (seg[0] + seg[1], seg[0] + seg[1] + seg[2])
    assert len(seg) == 4 and seg[3] == 1


@pytest.mark.unit
def test_the_selected_rule_routes_to_its_own_param_head():
    """⚠️ **v3 的核心接缝**：上层选的规则**真的**决定了走哪个参数 head。

    **这是"上层不可替代"的机械化证据**（设计 §3）：把**未选中**的 head 权重整个换成 999
    ⟹ 输出**逐位不变**；再把**选中**的那个换掉 ⟹ **必须变**。
    若是"共享头 + one-hot 拼输入"，换掉**任意**权重都会改变输出。

    ⚠️ 这比 `test_rule_heads.py` 的那条更强：那条测的是**模块**的 `forward`，
    这条测的是**整网 `forward` 的 `_route` 用对了 `sel`**（`sel` 与 `split_pa_rule` 的对应）。
    """
    tok, seg = _tok_seg()
    net = HierarchicalPolicy()
    rule_pa = 1
    rules = {"PA": torch.tensor([rule_pa]), "LA": torch.tensor([0]),
             "RA": torch.tensor([0])}
    a = _fwd(net, tok, seg, rules).param_mu["PA"]
    ji, mi = split_pa_rule(rules["PA"])
    assert PA_JOB_RULES[int(ji)] in net.pa_job.keys
    assert PA_MACHINE_RULES[int(mi)] in net.pa_mach.keys
    # ① 换掉**未选中**的作业 head ⟹ 输出逐位不变
    with torch.no_grad():
        for k in net.pa_job.keys:
            if k != net.pa_job.keys[int(ji)]:
                for p in net.pa_job.heads[k].parameters():
                    p.fill_(999.0)
    b = _fwd(net, tok, seg, rules).param_mu["PA"]
    assert torch.allclose(a, b), "换掉未选中的 head 竟改变了输出 ⟹ 不是硬路由"
    # ② 换掉**选中**的那个 ⟹ 必须变
    with torch.no_grad():
        for p in net.pa_job.heads[net.pa_job.keys[int(ji)]].parameters():
            p.fill_(999.0)
    c = _fwd(net, tok, seg, rules).param_mu["PA"]
    assert not torch.allclose(b, c), "换掉选中的 head 却没改变输出 ⟹ 路由没接上"


@pytest.mark.unit
def test_frozen_rules_are_used_verbatim():
    """给定 `rules` ⟹ `rule_sel` **逐位等于**给的号（阶段 B / 执行档）。

    ⚠️ 这条钉的是**规则号 ↔ head 不错位**（设计 §13.3 ③）：
    `rule_sel` 漂一格，下游就会用**另一条规则**的参数头。
    """
    tok, seg = _tok_seg()
    net = HierarchicalPolicy()
    rules = {"PA": torch.tensor([7]), "LA": torch.tensor([2]), "RA": torch.tensor([1])}
    out = _fwd(net, tok, seg, rules)
    for k, want in rules.items():
        assert int(out.rule_sel[k]) == int(want), k


@pytest.mark.unit
def test_rule_head_sizes_match_the_rule_space():
    """三个头的类数 = 规则空间（`12 / 3 / 3`）；**每条规则的参数个数可以不同**。"""
    net = HierarchicalPolicy()
    assert net.select.heads["PA"].out_features == N_PA_RULES == 12
    assert net.select.heads["LA"].out_features == N_LA_RULES == 3
    assert net.select.heads["RA"].out_features == N_PM_RULES == 3
    # 下层：PA 的作业侧每条 2 个、机台侧每条 1 个（设计 §4.2 / §4.3）
    assert all(net.pa_job.heads[r].out_features == N_PA_JOB_PARAMS for r in net.pa_job.keys)
    assert all(net.pa_mach.heads[r].out_features == N_PA_MACH_PARAMS
               for r in net.pa_mach.keys)
    assert set(net.pa_job.keys) == set(PA_JOB_RULES)
    assert set(net.pa_mach.keys) == set(PA_MACHINE_RULES)


@pytest.mark.unit
def test_missing_la_task_context_skips_the_la_head_and_direct_calls_still_raise():
    """⚠️ **没给任务上下文 ⟹ LA 头不算**（`param_mu["LA"] is None`），**不是补 0 再算**。

    为什么这样定：LA 的动作是"从 3 条派车规则里挑参数"，而**规则的自变量含这一趟的任务**
    —— **没有任务就无从打分**。补 0 等于报出一个"每趟任务都一样"的分布 ⟹ 那是**无声的坏策略**。

    ⚠️ **两件事要一起钉住**：
    1. `forward(la_ctx=None)` ⟹ `param_mu["LA"] is None`（**明确地说"这次不决策 LA"**）；
    2. **直接调 LA 头不给 `ctx_feat` 仍然报错** —— 那一层不许静默。
    """
    tok, seg = _tok_seg()
    net = HierarchicalPolicy()
    out = net(tok, seg)
    assert out.param_mu["LA"] is None, "没给任务上下文时不该算出 LA 的分布"
    assert out.param_mu["PA"] is not None and out.param_mu["RA"] is not None
    with pytest.raises(ValueError, match="逐决策上下文"):
        net.la.forward_all(torch.zeros(1, 128), None)


@pytest.mark.unit
def test_empty_mask_is_an_explicit_error():
    """掩码段为空必须**显式报错**——静默给 `nan` 会让"这个 agent 没接上"变成无声的坏策略。

    ⚠️ **这条护栏是补回来的**：v2 的 `_MaskedScoreHead` 有它；v3 改用段内 `mean` 之后
    **丢了**（`mean` 对空切片给的是 `nan`，不报错）⟹ `HierarchicalPolicy._pool` 补回。
    """
    net = HierarchicalPolicy()
    z = torch.zeros(1, 4, 128)
    with pytest.raises(ValueError, match="掩码段"):
        net._pool(z, (2, 2), "PA")
    with pytest.raises(ValueError, match="掩码段"):
        net._pool(z, (3, 1), "LA")


@pytest.mark.unit
def test_forward_is_deterministic_and_batch_rows_are_independent():
    tok, seg = _tok_seg()
    net = HierarchicalPolicy()
    a = _fwd(net, tok, seg).param_mu["PA"]
    b = _fwd(net, tok, seg).param_mu["PA"]
    assert torch.allclose(a, b), "同输入必须同输出（无 dropout / 无随机）"
    both = torch.cat([tok, tok], dim=0)
    c = _fwd(net, both, seg).param_mu["PA"]
    assert torch.allclose(c[0], c[1], atol=1e-5), "批内各行独立（无跨 batch 注意力）"
    assert all(np.isfinite(v) for v in c.flatten().tolist())
