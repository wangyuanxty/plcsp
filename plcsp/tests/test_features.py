"""token 特征构造的测试（P2 Task 2）。"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.des import SimConfig, SimWorld
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.nn.features import F_B, F_G, F_M, F_MAX, F_V, FEATURE_NAMES, norm_context
from plcsp.nn.state_emb import build_tok


def _ctx_and_snap(name="mk01", done=False):
    inst = load_mk(name)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=3)
    g = build_corridor_graph(lay)
    w = SimWorld(inst, lay, dock_distance_matrix(g), SimConfig(), graph=g)
    if done:
        w.run(seed_chain=1)
    return inst, lay, w, norm_context(inst, lay, m_ref=100.0)


@pytest.mark.unit
def test_field_counts_match_declared_widths():
    """⚠️ Review Focus #2：字段名清单与声明宽度必须逐段相等——防静默错位。"""
    assert len(FEATURE_NAMES["M"]) == F_M == 7
    assert len(FEATURE_NAMES["B"]) == F_B == 8
    assert len(FEATURE_NAMES["V"]) == F_V == 10
    assert len(FEATURE_NAMES["G"]) == F_G == 3


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
    """⚠️ Review Focus #1：积压维必须除以实例总工时——否则 MK01 与 MK10 差 12 倍。"""
    inst_s, _, ws, ctx_s = _ctx_and_snap("mk01")
    inst_l, _, wl, ctx_l = _ctx_and_snap("mk10")
    assert ctx_l.total_work_min > ctx_s.total_work_min * 10, "两实例总工时应有量级差"
    # 同一物理积压量在两实例上应映到相近的归一化值
    assert ctx_s.total_work_min == pytest.approx(153.0, rel=0.05)
    assert ctx_l.total_work_min == pytest.approx(1847.0, rel=0.05)


@pytest.mark.unit
def test_seg_lengths_come_from_actual_rows_not_hardcoded():
    """⚠️ Review Focus #2：seg 由各段**实际行数**推出，不得硬编码实例规模。"""
    inst, lay, w, ctx = _ctx_and_snap("mk10")
    tok, seg = build_tok(w.snapshot(), inst, lay, ctx)
    assert seg[0] == inst.n_machines == 15
    assert seg[1] == inst.n_jobs == 20
    assert tok.shape[0] == sum(seg)
