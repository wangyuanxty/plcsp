# -*- coding: utf-8 -*-
"""`nn/` 公开 API 的规格测试（2026-10-09）——覆盖 `features` / `state_emb` / `encoder` / `cuda_graph`。

这一段此前**完全没有跑在 v2 上的测试**（名字只在 `plcsp/tests/` 里出现过，见 §80.3）。
特征段的列序**有契约**（`FEATURE_NAMES` 按名核对、ckpt 兼容），所以逐维核是必须的。
"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp_v2.env import layout as layout_mod
from plcsp_v2.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp_v2.env.des import SimConfig, SimWorld, reference_makespan
from plcsp_v2.env.instances import load_mk
from plcsp_v2.nn import state_emb
from plcsp_v2.nn.cuda_graph import CudaGraphForward
from plcsp_v2.nn.encoder import AttnLayer, LayoutEncoder
from plcsp_v2.nn.features import (F_B, F_G, F_M, F_MAX, F_V, FEATURE_NAMES,
                                  SEG_SLICE, NormContext, norm_context)


def _world(name: str = "mk01") -> SimWorld:
    inst = load_mk(name)
    cfg = SimConfig()
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    return SimWorld(inst, lay, dock_distance_matrix(g), cfg, graph=g)


@pytest.mark.unit
def test_feature_widths_and_column_order_are_frozen():
    """列序/维宽是**契约**（有测试按名核对、ckpt 兼容）——改它就是破坏兼容。"""
    assert (F_M, F_B, F_V, F_G) == (7, 8, 8, 3)
    assert F_MAX == max(F_M, F_B, F_V, F_G) == 8
    for k, n in (("M", F_M), ("B", F_B), ("V", F_V), ("G", F_G)):
        assert len(FEATURE_NAMES[k]) == n, k
        assert SEG_SLICE[k] == slice(SEG_SLICE[k].start, SEG_SLICE[k].start + n)
    assert FEATURE_NAMES["B"][1] == "remaining_work"
    assert FEATURE_NAMES["V"][-1] == "battery"


@pytest.mark.unit
def test_norm_context_and_token_tensor_shape():
    w = _world()
    ctx = norm_context(w.inst, w.layout, reference_makespan(w.inst), SimConfig())
    assert isinstance(ctx, NormContext) and ctx.n_jobs == w.inst.n_jobs
    assert ctx.n_m == w.inst.n_machines and ctx.n_agv == w.cfg.n_agv
    tok, seg = state_emb.build_tok(w.snapshot(), w.inst, w.layout, ctx)
    assert seg == (ctx.n_m, ctx.n_jobs, ctx.n_agv, 1)
    assert tok.shape == (sum(seg), F_MAX) and tok.dtype == np.float32
    # 补零列恒 0（spec §5.3.1）
    assert np.all(tok[:ctx.n_m, F_M:] == 0.0)
    assert np.all(tok[ctx.n_m:ctx.n_m + ctx.n_jobs, F_B:] == 0.0)
    assert np.all(tok[-1, F_G:] == 0.0)


@pytest.mark.unit
def test_segment_feature_builders_agree_with_the_concatenated_tensor():
    w = _world()
    snap = w.snapshot()
    ctx = norm_context(w.inst, w.layout, reference_makespan(w.inst), SimConfig())
    m = state_emb.machine_features(snap, ctx)
    j = state_emb.job_features(snap, ctx)
    v = state_emb.vehicle_features(snap, ctx)
    g = state_emb.global_features(snap, ctx)
    assert m.shape == (ctx.n_m, F_M) and j.shape == (ctx.n_jobs, F_B)
    assert v.shape == (ctx.n_agv, F_V) and g.shape == (1, F_G)
    assert np.isfinite(m).all() and np.isfinite(j).all()
    assert (m >= -2.0).all() and (j[:, 4] >= -1.0).all()   # _safe 的钳位
    assert (v[:, 7] >= 0.0).all() and (v[:, 7] <= 1.0).all()  # battery_frac
    tok, seg = state_emb.build_tok(snap, w.inst, w.layout, ctx)
    assert np.allclose(tok[:seg[0], :F_M], m)


@pytest.mark.unit
def test_geom_bias_is_symmetric_zero_diagonal_and_non_positive():
    w = _world()
    ctx = norm_context(w.inst, w.layout, reference_makespan(w.inst), SimConfig())
    dm = dock_distance_matrix(w.g)
    b = state_emb.build_geom_bias(w.snapshot(), w.inst, w.layout, ctx, dm)
    n = b.shape[0]
    assert b.shape == (n, n)
    assert np.allclose(np.diag(b), 0.0)
    assert np.allclose(b, b.T)
    assert (b <= 1e-9).all(), "距离衰减先验：偏置恒 ≤ 0（缺锚 = 0）"


@pytest.mark.unit
def test_encoder_forward_and_layer_defaults():
    """层数默认 **4**（2026-10-02 已裁定、2026-10-09 才落地 —— 见 §79.2）。"""
    torch = pytest.importorskip("torch")
    enc = LayoutEncoder()
    assert enc.n_layers == 4 and enc.d_model == 128 and enc.feat_dim == F_MAX
    assert isinstance(enc.layers[0], AttnLayer)
    tok = torch.zeros(1, 6, F_MAX)
    seg = (3, 1, 1, 1)                       # seg 是**逐段行数的元组**，不是张量
    z, pooled = enc(tok, seg)                # forward 返回 (逐 token, 池化)
    assert z.shape[-2:] == (6, 128) and pooled.shape[-1] == 128
    z2, _ = enc(tok, seg)                    # 确定性（无 dropout）
    assert torch.allclose(z, z2)


@pytest.mark.unit
def test_cuda_graph_wrapper_is_inert_on_cpu():
    """CPU 档下 CUDA 图**恒不生效**——这是"默认档保持 CPU-only"的结构性保证。"""
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, 3)
    # ⚠️ 构造时就**显式拒绝** CPU 张量——这是"默认档保持 CPU-only"的结构性保证，
    # 比"静默不生效"更强：想用 CUDA 图必须自己拿 CUDA 张量来。
    with pytest.raises(ValueError, match="只接受 CUDA 张量"):
        CudaGraphForward(lambda t: t + 1.0, x)


