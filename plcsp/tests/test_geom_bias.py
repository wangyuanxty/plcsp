# -*- coding: utf-8 -*-
"""几何/度量偏置（② 拥堵的"距离可感知"）——`geom_bias` 开关（默认关）。

设计（`docs/mechanism-designs.md` 的「几何/度量偏置」节，开工前定死）：

    b(i, j) = −GEOM_BIAS_W · d(anchor_i, anchor_j) / bbox_diag   （两个锚都存在）
            = 0                                                   （任一锚缺失）

`d` 取自 `dm`（`dock_distance_matrix(build_corridor_graph(layout))`——与 `AgvSim._seg_min`
同一张矩阵）；偏置加在**每层注意力分数**上（`AttnLayer._attn`），与 `type_emb` 正交。

⚠️ **只作"输入"**：本文件不主张任何跨布局/跨拓扑泛化（`method-transfer-candidates.md`
§6.3 的裁定）——只验"距离真的进了分数"与"关掉逐位不变"。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from plcsp.algo.group_rel import chain_logp, decisions_logp, roll_chain
from plcsp.algo.policy import PolicyNet
from plcsp.algo.setup import build_setup
from plcsp.env.des import SimConfig, SimWorld, build_zone_map
from plcsp.env.instances import load_mk
from plcsp.env.snapshot import JobState, MachineState, Snapshot, VehicleState
from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_MAX
from plcsp.nn.state_emb import GEOM_BIAS_W, build_geom_bias, build_tok

# 黄金摘要：与 `test_route_choice.GOLDEN_CHAIN_DIGEST` 同一口径的独立脚本值（**改造前**捕获，
# 2026-10-05）。`geom_bias=False`（默认）时它必须逐字复现——偏置的接线若在关态下动了任何一位
# （多算一次快照/多走一条分支），这里翻红。
GEOM_OFF_CHAIN_DIGEST = "4241fec403248727280c2aacc888569f03b7d3b5b50a53dc10f2fc0031479228"


def _setup(name="mk01"):
    inst = load_mk(name)
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    return inst, lay, dm, cfg, ctx


def _chain_digest(dec, met) -> str:
    h = hashlib.sha256()
    for d in dec:
        h.update(d.kind.encode())
        h.update(np.ascontiguousarray(d.tok, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(d.feat, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(d.cand_feat, dtype=np.float32).tobytes())
        h.update(np.asarray(d.cand, dtype=np.int64).tobytes())
        h.update(np.asarray(d.tok_idx, dtype=np.int64).tobytes())
        h.update(np.asarray([d.action], dtype=np.int64).tobytes())
        h.update(np.asarray([d.logp], dtype=np.float32).tobytes())
    h.update(np.asarray([met["makespan"], met["travel_time_total"],
                         float(met["energy"]), float(met["deliveries"])],
                        dtype=np.float64).tobytes())
    return h.hexdigest()


def _ms(busy=False, remaining=0.0):
    return MachineState(backlog_min=0.0, in_q_len=0, in_cap=2.0, out_q_len=0, out_cap=2.0,
                        busy=busy, remaining_min=remaining, pm_used_min=0.0,
                        fail_rate=0.0, prev_job=-1)


def _js(at_machine=-1, on_agv=-1):
    return JobState(done_ops=0, total_ops=3, remaining_min=0.0, finished=False, due=0.0,
                    at_machine=at_machine, in_transit=on_agv >= 0, on_agv=on_agv, rework_cnt=0)


def _vs(node):
    return VehicleState(status=0, node=node, queued=0, battery_frac=1.0, capacity=1,
                        speed_factor=1.0, zone_wait=0.0)


# ────────────────────────── 1. 关态逐位不变 ──────────────────────────

@pytest.mark.unit
def test_geom_bias_off_is_bit_identical_to_baseline():
    """判据 1：`geom_bias=False`（默认）⟹ 链路逐位等于改造前——黄金摘要钉死。"""
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(20261005)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    assert all(d.geom_bias is None for d in dec), "关态不得记录几何偏置"
    assert _chain_digest(dec, met) == GEOM_OFF_CHAIN_DIGEST, \
        "geom_bias 关态的链路变了——既有读数不再成立"


@pytest.mark.unit
def test_zero_bias_is_a_bitwise_no_op_in_the_encoder():
    """判据 1（编码器层对照）：喂**全零偏置**与不喂偏置，输出必须 `torch.equal`。

    这条把"`bias=None` 走原表达式"钉成机器可判的判据——若实现写成"scores + 0·bias"式的
    恒等变形，浮点末位仍可能漂（`x + 0.0` 对 −0.0 会翻符号）；逐位相等才是硬要求。
    """
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(3)
    pol = PolicyNet(enc=LayoutEncoder())
    snap = SimWorld(inst, lay, dm, cfg).snapshot()
    tok, seg = build_tok(snap, inst, lay, ctx)
    x = torch.as_tensor(tok, dtype=torch.float32).unsqueeze(0)
    o_off, _ = pol.forward_enc(x, seg)
    o_zero, _ = pol.forward_enc(x, seg, np.zeros((x.shape[1], x.shape[1]), dtype=np.float32))
    assert torch.equal(o_off, o_zero), "全零偏置不是逐位无操作——关态路径被动过"


# ────────────────────────── 2. 偏置的数值契约 ──────────────────────────

@pytest.mark.unit
def test_build_geom_bias_numeric_contract():
    """判据 2：对称、对角 0、≤ 0（负号 = 近的高分）、按 `dm/bbox_diag` 定标、缺锚为 0。

    手搓 3 机台 + 2 作业 + 2 车：机台锚 = 各自 `dock_node`；作业 0 在机台 0 上、作业 1
    **无锚**（不在机上也没上车）；车 0 在机台 1 的 dock（与机台 1 同锚 ⟹ 偏置 = 0）、
    车 1 尚未出车（node=-1 ⟹ 无锚）。G 恒无锚。
    """
    inst, lay, dm, cfg, ctx = _setup()
    snap = Snapshot(now=0.0, machines=(_ms(),) * inst.n_machines,
                    jobs=(_js(at_machine=0), _js()), vehicles=(),
                    n_done=0, in_flight=0, zone_holder=())
    v0 = _vs(lay.machines[1].dock_node)
    v1 = _vs(-1)
    snap = Snapshot(now=0.0, machines=snap.machines, jobs=snap.jobs, vehicles=(v0, v1),
                    n_done=0, in_flight=0, zone_holder=())
    g = build_geom_bias(snap, inst, lay, ctx, dm)
    # ⚠️ 行数取 `ctx.n_agv`（与 `build_tok` 的 V 段同源），不是快照里的车数——快照只给了 2 台，
    #    第 3 台的状态缺失 ⟹ 无锚（正是"缺锚 = 0"的一种）。
    n_m, n_b, n_v = inst.n_machines, inst.n_jobs, ctx.n_agv
    N = n_m + n_b + n_v + 1
    assert g.shape == (N, N) and g.dtype == np.float32
    assert np.allclose(g, g.T), "偏置必须对称"
    assert np.all(np.diag(g) == 0.0), "对角（自己到自己）必须为 0"
    assert g.max() <= 0.0
    # 机台 0 ↔ 机台 1：−W·d/bbox_diag
    d01 = float(dm[lay.machines[0].dock_node, lay.machines[1].dock_node])
    assert g[0, 1] == pytest.approx(-GEOM_BIAS_W * d01 / ctx.bbox_diag, rel=1e-6)
    # 车 0 与机台 1 同锚 ⟹ 偏置 0；车 1 无锚 ⟹ 整行/整列 0
    assert g[n_m + n_b + 0, 1] == pytest.approx(0.0), "同锚实体之间的距离应为 0"
    assert np.all(g[n_m + n_b + 1, :] == 0.0) and np.all(g[:, n_m + n_b + 1] == 0.0), \
        "未出车的车（node=-1）必须无锚 ⟹ 整行/整列为 0"
    assert np.all(g[n_m + n_b + 2, :] == 0.0), "快照里缺失的车（第 3 台）必须无锚"
    # 作业 1 无锚 ⟹ 整行/整列 0；作业 0 在机台 0 上 ⟹ 与机台 0 同锚（0）
    assert np.all(g[n_m + 1, :] == 0.0)
    assert g[n_m + 0, 0] == pytest.approx(0.0)
    # G token（末位）无锚
    assert np.all(g[-1, :] == 0.0)
    # 单调性：取两台机台与更远的一对（用 dm 直接核对至少一处"更远 ⟹ 更负"）
    far = [(i, j) for i in range(n_m) for j in range(n_m) if i < j]
    vals = [(float(dm[lay.machines[i].dock_node, lay.machines[j].dock_node]), g[i, j])
            for i, j in far]
    dmin = min(v[0] for v in vals)
    dmax = max(v[0] for v in vals)
    if dmax > dmin:
        assert min(v[1] for v in vals) < max(v[1] for v in vals), "距离越远偏置应越负"


@pytest.mark.unit
def test_geom_bias_with_zone_tokens_covers_the_z_segment():
    """R2 + 几何一起打开时，偏置矩阵的行数必须含 Z 段（与 `build_tok` 的 seg 同源）。"""
    inst, lay, dm, cfg, ctx = _setup()
    zof, n_zones = build_zone_map(lay, cfg.zone_granularity)
    snap = SimWorld(inst, lay, dm, cfg).snapshot()
    tok, seg = build_tok(snap, inst, lay, ctx, zof=zof, n_zones=n_zones)
    g = build_geom_bias(snap, inst, lay, ctx, dm, zof=zof)
    assert g.shape == (tok.shape[0], tok.shape[0]) == (sum(seg), sum(seg)), \
        f"带 Z 段时偏置行数应与 token 行数一致：{g.shape} vs {tok.shape}"
    # 两个不同区段的锚不同 ⟹ 偏置非 0（区段进入了距离几何）
    n_base = int(sum(seg[:4]))
    assert g[n_base + 0, n_base + 1] < 0.0, "不同区段之间应有非零距离偏置"


# ────────────────────────── 3. 偏置真的进分数 ──────────────────────────

@pytest.mark.unit
def test_nonzero_bias_changes_attention_output_and_scores():
    """判据 2（网络层）：非零偏置必须改变注意力输出与候选分数。"""
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(3)
    pol = PolicyNet(enc=LayoutEncoder())
    snap = SimWorld(inst, lay, dm, cfg).snapshot()
    tok, seg = build_tok(snap, inst, lay, ctx)
    x = torch.as_tensor(tok, dtype=torch.float32).unsqueeze(0)
    g = build_geom_bias(snap, inst, lay, ctx, dm)
    assert np.any(g < 0.0), "前提：该快照的距离偏置非全零"
    o_off, _ = pol.forward_enc(x, seg)
    o_on, _ = pol.forward_enc(x, seg, g)
    assert not torch.allclose(o_off, o_on), "非零偏置没有改变编码器输出——偏置没进分数"


@pytest.mark.unit
def test_machine_token_scores_change_with_a_perturbed_bias():
    """只改偏置的一行 ⟹ 该 token 的打分必须变（偏置是**活的**输入，不是装饰）。"""
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(3)
    pol = PolicyNet(enc=LayoutEncoder())
    snap = SimWorld(inst, lay, dm, cfg).snapshot()
    tok, seg = build_tok(snap, inst, lay, ctx)
    x = torch.as_tensor(tok, dtype=torch.float32).unsqueeze(0)
    g = np.zeros_like(build_geom_bias(snap, inst, lay, ctx, dm))
    cand = (0, 1, 2)
    feat = torch.zeros(1, 1, 3)
    cand_feat = torch.zeros(1, 3, 1)
    idx = torch.tensor(cand)
    base_emb = pol.forward_enc(x, seg, g)
    base = pol.mach_logits_emb(base_emb[0], feat, cand_feat, idx)
    g2 = g.copy()
    g2[3, :] = -0.5                     # 3 号机台对全场"变远"
    g2[:, 3] = -0.5
    changed = pol.mach_logits_emb(pol.forward_enc(x, seg, g2)[0], feat, cand_feat, idx)
    assert not torch.allclose(base, changed), "改了偏置行，S 头分数却不变——偏置未被消费"


# ────────────────────────── 4. 链级接线 ──────────────────────────

@pytest.mark.unit
def test_geom_bias_changes_decision_logp_on_the_production_path():
    """判据 3：`geom_bias=True` 的决策 logp 必须与关态不同（同一初始化、逐决策比）。

    ⚠️ 只比 logp 不比动作：未训练策略下采样落点可能恰好相同（动作相同并不表示输入没变）。
    """
    inst, lay, dm, cfg, ctx = _setup()
    out = {}
    for label, kw in (("off", {}), ("on", {"geom_bias": True})):
        torch.manual_seed(20261005)
        pol = PolicyNet(enc=LayoutEncoder())
        dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, **kw)
        out[label] = dec
    a, b = out["off"], out["on"]
    assert len(a) == len(b)
    assert any(d.geom_bias is not None for d in b)
    dmax = max(abs(x.logp - y.logp) for x, y in zip(a, b))
    assert dmax > 1e-6, "打开几何偏置后逐决策 logp 一位没变——偏置没有进生产路径"


@pytest.mark.unit
def test_geom_bias_batched_recompute_matches_single():
    """判据 4：带偏置的批量重算与逐决策单条前向同源（≤1e-5）。"""
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(11)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, geom_bias=True)
    sub = dec[:60]
    batched = decisions_logp(sub, pol).detach()
    ref = []
    for d in sub:
        tok, _ = pol.forward_enc(torch.as_tensor(d.tok, dtype=torch.float32).unsqueeze(0),
                                 d.seg, d.geom_bias)
        head = {"S": pol.mach_logits_emb, "L": pol.agv_logits_emb, "R": pol.route_logits_emb,
                "M": pol.pm_logits_emb, "C": pol.charge_logits_emb}[d.kind]
        logits = head(tok, torch.as_tensor(d.feat, dtype=torch.float32).reshape(1, 1, -1),
                      torch.as_tensor(d.cand_feat, dtype=torch.float32),
                      torch.as_tensor(d.tok_idx, dtype=torch.long))
        ref.append(torch.log_softmax(logits.flatten(), -1)[d.cand.index(d.action)])
    diff = float((batched - torch.stack(ref)).abs().max().detach())
    assert diff < 1e-5, f"带偏置的批量重算与逐决策差 {diff}——批路径接错了"


@pytest.mark.unit
def test_geom_bias_gradient_reaches_encoder():
    """判据 4：偏置配置的链 logp 反传照常到编码器（偏置只改输入，不改梯度通路）。"""
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(13)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, geom_bias=True)
    chain_logp(dec, pol).backward()
    g = pol.enc.proj[0].weight.grad
    assert g is not None and g.abs().sum() > 0, "偏置配置下编码器没有梯度"


@pytest.mark.unit
def test_geom_bias_works_together_with_route_zones():
    """② + R2 同时打开必须跑得通（偏置行数含 Z 段、R 决策照常进链）。"""
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(17)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2,
                          route_zones=True, geom_bias=True)
    assert not met["horizon_hit"] and met["jobs_done"] == inst.n_jobs, "②+R2 配置跑不完"
    r = next(d for d in dec if d.kind == "R")
    assert r.geom_bias is not None and r.zone_idx is not None
    assert r.geom_bias.shape == (sum(r.seg), sum(r.seg)) == (r.tok.shape[0], r.tok.shape[0])
    assert r.tok.shape[1] == F_MAX
    assert float(decisions_logp([r], pol).detach()) == pytest.approx(r.logp, abs=1e-5)
