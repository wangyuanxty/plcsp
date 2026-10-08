# -*- coding: utf-8 -*-
"""快照 + 实例 + 布局 → 编码器输入（spec §5.3.1）。

四段宽度不同（F_M=7 / F_B=8 / F_V=9 / F_G=3；v2：④ 删后 F_B 9→8、⑩ 删后 F_V 11→9），故各自算完后**补齐到 `F_MAX` 后按序列
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


# 几何/度量偏置的固定系数（**不可学**，沿用旧 `GeomBias` 的"固定系数"约定）：
# `b(i,j) = −W · d(anchor_i, anchor_j) / bbox_diag`。取 1.0 是因为注意力分数 `q·k/√d_h`
# 本身就是 O(1) 量级，而归一化距离 ∈ [0, ~1.4]。负号 = 距离越近、分数越高（距离衰减先验）。
GEOM_BIAS_W = 1.0


def build_geom_bias(snap: Snapshot, inst: Instance, layout: Layout,
                    ctx: NormContext, dm: np.ndarray) -> np.ndarray:
    """几何/度量偏置矩阵 `(N, N)`——**只在 `geom_bias=True` 时构造**（② 拥堵的"距离可感知"）。

    口径（设计节「几何/度量偏置」定死）：

        b(i, j) = −GEOM_BIAS_W · d(anchor_i, anchor_j) / bbox_diag        （两个锚都存在）
                = 0                                                        （任一锚缺失）

    - `d` 取自 `dm`（`dock_distance_matrix(build_corridor_graph(layout))`）——**与仿真
      `AgvSim._seg_min` 的行驶时长同一张矩阵**（单位 [m]），不另造距离量；
    - 锚：M = 该机台的 `dock_node`；B = `at_machine` 的 dock，否则 `on_agv` 所在车的
      `node`（≥0 才有），否则**无锚**；V = 该车 `node`（`-1` = 尚未出车 ⟹ 无锚）；G = **无锚**。
      （⚠️ **v2：① 拥堵去后 Z 段已删**（2026-10-08）—— 原"区段最小成员节点"锚不再存在。）
    - 缺锚的行/列全 0（**中性，不是哨兵**）；对角恒 0；不可达（`inf`）按缺锚处理
      （不把 NaN/inf 送进 softmax）。

    ⚠️ **只作"输入"**：本项只让距离/拥堵可被感知；**不得**借它主张跨布局/跨拓扑泛化
    （`method-transfer-candidates.md` §6.3 的裁定——那是 2026-10-02 砍掉的卖点）。
    """
    n_m = int(inst.n_machines)
    n_b = int(inst.n_jobs)
    n_v = int(ctx.n_agv)
    n_base = n_m + n_b + n_v + 1                      # 四段行数（G 在末位）
    anchors: list[int | None] = [None] * n_base
    for i, mp in enumerate(layout.machines):
        anchors[i] = int(mp.dock_node)
    for j, js in enumerate(snap.jobs):
        if j >= n_b:
            break
        a: int | None = None
        if 0 <= int(js.at_machine) < n_m:
            a = int(layout.machines[int(js.at_machine)].dock_node)
        elif 0 <= int(js.on_agv) < len(snap.vehicles):
            node = int(snap.vehicles[int(js.on_agv)].node)
            if node >= 0:
                a = node
        anchors[n_m + j] = a
    for a_i, v in enumerate(snap.vehicles):
        if a_i >= n_v:
            break
        node = int(v.node)
        anchors[n_m + n_b + a_i] = node if node >= 0 else None
    N = len(anchors)
    out = np.zeros((N, N), dtype=np.float32)
    scale = GEOM_BIAS_W / max(ctx.bbox_diag, 1e-9)
    for i in range(N):
        ai = anchors[i]
        if ai is None:
            continue
        for j in range(i + 1, N):
            aj = anchors[j]
            if aj is None:
                continue
            d = float(dm[ai, aj])
            if not np.isfinite(d):
                continue                              # 不可达 ⟹ 按缺锚（0），不传 inf
            out[i, j] = out[j, i] = -scale * d
    return out


def machine_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(n_m, 7)。顺序 = FEATURE_NAMES["M"]，**不得改序**（有测试按名核对）。

    ⚠️ **第 6 维（fail_rate）在 ③ `machine_failure` 关闭时恒 0**（R1a，F2 同型残留）：
    仿真里 `MachineSim._process` ③ 关时一次跑完、`fail_events` 恒 0，特征若照报
    `pad.fail_rate / max_fail_rate`，策略就看到一个**动力学里不存在**的故障风险（假信号）。
    恒 0 是"不会坏"的**真值**（不是哨兵）；维宽/列序不随配置变（spec §5.3.1 字段表 + ckpt 兼容）。
    """
    out = np.zeros((ctx.n_m, F_M), dtype=np.float32)
    for i, m in enumerate(snap.machines):
        out[i] = (
            m.backlog_min / ctx.total_work_min,                     # 0 backlog
            # ⚠️ **v2（② 去后）**：无容量可除 ⟹ 改用实例静态量 `n_jobs` 归一。
            # 队列长度 <= 作业总数（每作业对某台机最多贡献 1 件），故比值在 [0,1]。
            _safe(m.in_q_len / max(ctx.n_jobs, 1), 0.0, 1.0),        # 1 in_queue
            _safe(m.out_q_len / max(ctx.n_jobs, 1), 0.0, 1.0),       # 2 out_queue
            1.0 if m.busy else 0.0,                                 # 3 busy
            m.remaining_min / max(ctx.total_work_min, 1e-9),        # 4 remaining_frac
            _safe(1.0 - m.pm_used_min / max(ctx.pm_interval, 1e-9), 0.0, 1.0),  # 5 pm_left
            (m.fail_rate / max(ctx.max_fail_rate, 1e-9)
             if ctx.constraints.machine_failure else 0.0),          # 6 fail_rate（③ 关恒 0）
        )
    return out


def job_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(n_jobs, 8)。顺序 = FEATURE_NAMES["B"]，**不得改序**（有测试按名核对）。

    ⚠️ **第 2 维（due_margin）在 ⑧ `due_dates` 关闭时恒 0**（R1b，F2 同型残留）：⑧ 关意味着
    该实例**无交期**（`_due_map` 返回 `{}`，快照 `due` 落 0.0 哨兵）；照读哨兵会得到
    `(0 − now)/M_ref`（不是 0，长 episode 饱和到 −2），等于报出一个不存在的交期紧迫度。
    恒 0 = "无交期信息"；维宽/列序不随配置变。

    ⚠️ **2026-10-08（v2）：④ 返工的 `rework_cnt` 维已删**（F_B 9 → 8）。
    """
    out = np.zeros((ctx.n_jobs, F_B), dtype=np.float32)
    for j, js in enumerate(snap.jobs):
        out[j] = (
            js.done_ops / max(js.total_ops, 1),                     # 0 progress
            js.remaining_min / ctx.total_work_min,                  # 1 remaining_work
            # 2 due_margin = (due_j − now)/M_ref（spec §5.3.1，交期 `due` 见 `JobState`）。
            # ⚠️ **不再是共同交期形状**（2026-10-03 ⑧ 重设计后）：交期改为**逐作业**的 TF/RDD
            # 口径 `d_j = LB·τ·(1+R(2ρ_j−1))`，冻结表的 R 下限 0.20 保证**10/10 实例**的 d_j
            # 互不相同 ⟹ 此维在任意实例上都随作业不同、含新信息。
            # （旧口径 `d_j = τ·M_ref` 对所有作业同值，那时它 ≡ τ − Global[0]，确为冗余维。）
            (_safe((js.due - snap.now) / ctx.m_ref, -2.0, 2.0)
             if ctx.constraints.due_dates else 0.0),                # 2 due_margin（⑧ 关恒 0）
            1.0 if js.finished else 0.0,                            # 3 finished
            _safe(js.at_machine / max(ctx.n_m, 1), -1.0, 1.0),      # 4 at_machine
            1.0 if js.in_transit else 0.0,                          # 5 in_transit
            _safe(js.on_agv / max(ctx.n_agv, 1), -1.0, 1.0),        # 6 on_agv
            1.0,                                                    # 7 weight（等权，见 §5.3.1）
        )
    return out


def vehicle_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(n_agv, 9)。顺序 = FEATURE_NAMES["V"]，**不得改序**（有测试按名核对）。

    ⚠️ **v2：⑩ 去后由 11 维降到 9 维**（2026-10-08）—— 去掉 `capacity` 与 `speed_factor`。

    ⚠️ **v2：① 去后 `zone_wait` 维已删**（2026-10-08）。
    """
    out = np.zeros((ctx.n_agv, F_V), dtype=np.float32)
    for a, v in enumerate(snap.vehicles):
        st = [0.0] * 4
        st[int(_safe(v.status, 0, 3))] = 1.0     # `status` ∈ {0,1,2,3} 独热；越界钳到 3（故障态）
        # ⚠️ x/y 必须取自**节点坐标**，不能都用节点号（那等于丢掉了几何，只剩序号）。
        # ⚠️ `node == -1` 是**哨兵**（`des.py` 在"尚未出车"`pos_node is None` 时发 -1），
        # **不是 0 号节点**：若按 0 号取坐标，每台未出车的车都会拿到**伪造的网格角落坐标**，
        # 且与"真的停在 0 号节点"**无法区分**（Review F1）。故哨兵编成 (−1, −1)——
        # 真实节点坐标归一后 ∈ [0,1)，−1 落在值域外，天然可区分。
        if v.node >= 0:
            nx, ny = ctx.node_xy[v.node]
            x, y = nx / ctx.bbox_diag, ny / ctx.bbox_diag
        else:
            x, y = -1.0, -1.0
        out[a] = (*st,
                  x,                                                 # 4 node_x
                  y,                                                 # 5 node_y
                  _safe(v.queued / max(ctx.max_queued, 1), 0.0, 1.0),   # 6 queued
                  _safe(v.battery_frac, 0.0, 1.0))                   # 7 battery
                  # ⚠️ **v2**：原 8 `capacity` / 9 `speed_factor`（⑩ 去）与
                  # 原 `zone_wait`（① 去）三维已删（2026-10-08）
    return out


def global_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(1, 3)。顺序 = FEATURE_NAMES["G"]，**不得改序**（有测试按名核对）。"""
    return np.array([[
        _safe(snap.now / ctx.m_ref, 0.0, 4.0),                      # 0 time_progress
        snap.n_done / max(ctx.n_jobs, 1),                           # 1 done_frac
        _safe(snap.in_flight / max(ctx.max_queued, 1), 0.0, 4.0),   # 2 in_flight
    ]], dtype=np.float32)


# ⚠️ **v2：① 去后 `zone_features` 已删**（2026-10-08）—— 无区段 token


def build_tok(snap: Snapshot, inst: Instance, layout: Layout,
              ctx: NormContext) -> tuple[np.ndarray, tuple[int, ...]]:
    """快照 → **单张** `(N, F_MAX)` 特征张量 + `seg`（spec §5.3.1）。

    四段各自算完后按 `SEG_SLICE` 填进对应行，**列不足处保持 0**（补零）。
    故 `tok[:n_m, F_M:]`、`tok[n_m:n_m+n_jobs, F_B:]`、`tok[-1, F_G:]` 恒为 0。

    ⚠️ `seg` 由**各段实际行数**推出，**不硬编码实例规模**（Review Focus #2）。

    ⚠️ **v2：① 拥堵去后 Z 段（区段 token）已删**（2026-10-08）—— `seg` **恒为四元**。
    """
    parts = [("M", machine_features(snap, ctx)), ("B", job_features(snap, ctx)),
             ("V", vehicle_features(snap, ctx)), ("G", global_features(snap, ctx))]
    seg = tuple(p.shape[0] for _k, p in parts)                    # 恒为 (n_m, n_jobs, n_agv, 1)
    tok = np.zeros((sum(seg), F_MAX), dtype=np.float32)
    r = 0
    for key, p in parts:
        tok[r:r + p.shape[0], SEG_SLICE[key]] = p                 # 补零列保持 0
        r += p.shape[0]
    return tok, seg
