"""M3c 桥：Instance + Layout → 编码器输入（tok_feat / seg / dist / conf）。

《方法设计文档》§2.2 token 规范的落地子集（M3c v0，静态度）：
- M 段 = 机台 token：静态几何 4 = [u/L, v/L, d(质心)/L, corridor_deg]（主轴相对坐标 ÷ char_len；
  d(质心) 为 §2.2 "d(m,hub)" 的 v0 近似，hub 实体待 ② 事件驱动状态接入时对齐）；
  动态 6 = [in/out 缓冲占用、占道、负载、电量、等待] —— M3c 桩 0，动态值接口 = M3b-②。
- B 段 = 每作业 1 个"工序进度 token"（n_b = n_jobs）：10 维桩 0（当前工序/剩余时长等动态），
  静态版无语义 —— 动态接入点同 ②。
- V 段 = 车辆 token（n_v = n_agv）：10 维桩 0（路点/负载/电量等）。
- dist：M-M 用装卸点欧氏距离/char_len（v0；A* 走廊距离 = §2.1 nav 段口径，后续对齐），跨段 0。
- conf：M-M 用 conf_sim（v0 占道 0 → 全 1），跨段 0。
公共不变性（特征函数同一来源）：平移→相对坐标；旋转→PCA 主轴；缩放→÷char_len；拓扑→变长。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..env.instances import Instance
from ..env.layout import Layout
from .features import geometry_features, conf_sim

F_TOKEN = 10              # token_feature_dim(4, 6) = 几何 4 + 动态 6
F_GEOM, F_DYN = 4, 6


@dataclass
class EncState:
    tok_feat: torch.Tensor   # (1, N, F_TOKEN)
    seg: tuple[int, int, int]
    dist: torch.Tensor       # (1, N, N)
    conf: torch.Tensor       # (1, N, N)

    @property
    def tok(self) -> torch.Tensor:
        return self.tok_feat


def encode_state(inst: Instance, layout: Layout, n_agv: int = 2) -> EncState:
    """实例 + 布局 → 编码器输入（M/B/V 三段；确定性：同 (inst, layout) → 同张量）。"""
    nm, nb, nv = inst.n_machines, inst.n_jobs, n_agv
    gf = geometry_features(layout)
    L = max(gf.char_len, 1e-9)
    u, v = gf.proj[:, 0] / L, gf.proj[:, 1] / L
    d_center = np.linalg.norm(gf.proj, axis=1) / L                 # ⊂ §2.2 hub 距离 v0 近似
    m_static = np.stack([u, v, d_center, gf.corridor_deg], axis=1)  # (nm, 4)
    m_dyn = np.zeros((nm, F_DYN), dtype=np.float64)                 # 占道/缓冲… 桩（② 接入）
    m_feat = np.concatenate([m_static, m_dyn], axis=1)              # (nm, F_TOKEN)
    b_feat = np.zeros((nb, F_TOKEN), dtype=np.float64)              # 作业进度 桩（② 接入）
    v_feat = np.zeros((nv, F_TOKEN), dtype=np.float64)              # 车辆状态 桩（② 接入）
    tok = torch.tensor(np.concatenate([m_feat, b_feat, v_feat], axis=0),
                       dtype=torch.float32).unsqueeze(0)            # (1, N, F)
    xy = np.array([m.dock for m in layout.machines], dtype=float)
    dn = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1) / L   # (nm,nm) 欧氏/char_len
    D = np.zeros((nm + nb + nv, nm + nb + nv), dtype=np.float64)
    D[:nm, :nm] = dn
    C = np.zeros_like(D)
    C[:nm, :nm] = conf_sim(np.zeros(nm))                            # 占道 0 → 全 1
    return EncState(tok_feat=tok, seg=(nm, nb, nv),
                    dist=torch.tensor(D).float().unsqueeze(0),
                    conf=torch.tensor(C).float().unsqueeze(0))
