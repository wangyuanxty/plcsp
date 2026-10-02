# -*- coding: utf-8 -*-
"""快照 + 实例 + 布局 → 编码器输入（spec §5.3.1）。

四段宽度不同（F_M=7 / F_B=8 / F_V=10 / F_G=3），故各自算完后**补齐到 `F_MAX` 后按序列
拼成单张 `(N, F_MAX)`**（spec §5.3.1），并给出 `seg` 标出各段行数。
"""
from __future__ import annotations

import numpy as np

from ..env.instances import Instance
from ..env.layout import Layout
from ..env.snapshot import Snapshot
# `norm_context` 用冗余别名转出（explicit re-export）——`NormContext` 的构造入口在
# `features.py`（与字段定义同处一地），此处只为保持 `state_emb.norm_context` 这个既有入口可用。
from .features import (F_B, F_M, F_MAX, F_V, SEG_SLICE, NormContext,
                       norm_context as norm_context)


def _safe(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def machine_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(n_m, 7)。顺序 = FEATURE_NAMES["M"]，**不得改序**（有测试按名核对）。"""
    out = np.zeros((ctx.n_m, F_M), dtype=np.float32)
    for i, m in enumerate(snap.machines):
        out[i] = (
            m.backlog_min / ctx.total_work_min,                     # 0 backlog
            _safe(m.in_q_len / max(m.in_cap, 1.0), 0.0, 1.0),       # 1 in_fill
            _safe(m.out_q_len / max(m.out_cap, 1.0), 0.0, 1.0),     # 2 out_fill
            1.0 if m.busy else 0.0,                                 # 3 busy
            m.remaining_min / max(ctx.total_work_min, 1e-9),        # 4 remaining_frac
            _safe(1.0 - m.pm_used_min / max(ctx.pm_interval, 1e-9), 0.0, 1.0),  # 5 pm_left
            m.fail_rate / max(ctx.max_fail_rate, 1e-9),             # 6 fail_rate
        )
    return out


def job_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(n_jobs, 8)。"""
    out = np.zeros((ctx.n_jobs, F_B), dtype=np.float32)
    for j, js in enumerate(snap.jobs):
        out[j] = (
            js.done_ops / max(js.total_ops, 1),                     # 0 progress
            js.remaining_min / ctx.total_work_min,                  # 1 remaining_work
            # TODO(P2-Task6)：`Snapshot` 无逐作业交期 → 此维为**占位**（now/M_ref − 1 近似）；
            # Task 6 把 `due_j` 放进 `JobState` 后改为 (due_j − now)/M_ref。
            (0.0 if js.finished else
             _safe((snap.now - 0.0) / ctx.m_ref - 1.0, -2.0, 2.0)),  # 2 due_margin
            1.0 if js.finished else 0.0,                            # 3 finished
            _safe(js.at_machine / max(ctx.n_m, 1), -1.0, 1.0),      # 4 at_machine
            1.0 if js.in_transit else 0.0,                          # 5 in_transit
            _safe(js.on_agv / max(ctx.n_agv, 1), -1.0, 1.0),        # 6 on_agv
            1.0,                                                    # 7 weight（等权，见 §5.3.1）
        )
    return out


def vehicle_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(n_agv, 10)。"""
    out = np.zeros((ctx.n_agv, F_V), dtype=np.float32)
    for a, v in enumerate(snap.vehicles):
        st = [0.0] * 4
        st[int(_safe(v.status, 0, 3))] = 1.0     # `status` ∈ {0,1,2,3} 独热；越界钳到 3（故障态）
        # ⚠️ x/y 必须取自**节点坐标**，不能都用节点号（那等于丢掉了几何，只剩序号）
        nx, ny = ctx.node_xy[max(v.node, 0)]
        out[a] = (*st,
                  nx / ctx.bbox_diag,                                # 4 node_x
                  ny / ctx.bbox_diag,                                # 5 node_y
                  _safe(v.queued / max(ctx.max_queued, 1), 0.0, 1.0),   # 6 queued
                  _safe(v.battery_frac, 0.0, 1.0),                   # 7 battery
                  v.capacity / max(ctx.max_capacity, 1),             # 8 capacity
                  _safe(v.speed_factor, 0.0, 2.0))                   # 9 speed_factor
    return out


def global_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(1, 3)。"""
    return np.array([[
        _safe(snap.now / ctx.m_ref, 0.0, 4.0),                      # 0 time_progress
        snap.n_done / max(ctx.n_jobs, 1),                           # 1 done_frac
        _safe(snap.in_flight / max(ctx.max_queued, 1), 0.0, 4.0),   # 2 in_flight
    ]], dtype=np.float32)


def build_tok(snap: Snapshot, inst: Instance, layout: Layout,
              ctx: NormContext) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """快照 → **单张** `(N, F_MAX)` 特征张量 + `seg`（spec §5.3.1）。

    四段各自算完后按 `SEG_SLICE` 填进对应行，**列不足处保持 0**（补零）。
    故 `tok[:n_m, 7:]`、`tok[n_m:n_m+n_jobs, 8:]`、`tok[-1, 3:]` 恒为 0。

    ⚠️ `seg` 由**各段实际行数**推出，**不硬编码实例规模**（Review Focus #2）。
    """
    parts = (("M", machine_features(snap, ctx)), ("B", job_features(snap, ctx)),
             ("V", vehicle_features(snap, ctx)), ("G", global_features(snap, ctx)))
    seg = tuple(p.shape[0] for _k, p in parts)                    # (n_m, n_jobs, n_agv, 1)
    tok = np.zeros((sum(seg), F_MAX), dtype=np.float32)
    r = 0
    for key, p in parts:
        tok[r:r + p.shape[0], SEG_SLICE[key]] = p                 # 补零列保持 0
        r += p.shape[0]
    return tok, seg
