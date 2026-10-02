"""Instance + Layout → 编码器输入（tok_feat / seg）。

⚠️ 2026-10-02 状态（**必读**）：
  **几何特征已整条删除**（几何/感知路线砍除，见 `progress-log.md` §12.6），`dist` / `conf`
  两个张量随之取消——`LayoutEncoder.forward` 现在只收 `(tok_feat, seg)`。

  **当前 token 特征全为零向量**，因为旧版除 4 位静态几何外的所有位（动态 6 位、作业位、车辆位）
  **从来就是桩 0**，从未接入过真实状态。
  → **P2（骨架）必须重新设计 token 特征**；在此之前编码器等价于吃零输入。

保留 `Layout` 形参：P2 需要它取机台数/车辆数等结构信息（以及后续可能重新引入的**距离矩阵**，
那是搬运时间的来源，与已砍的"几何进网络"是两回事）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..env.instances import Instance
from ..env.layout import Layout
from .features import token_feature_dim

F_TOKEN = token_feature_dim()          # 2026-10-02: 10 → 6（几何 4 位已删）


@dataclass
class EncState:
    tok_feat: torch.Tensor   # (1, N, F_TOKEN)
    seg: tuple[int, int, int]

    @property
    def tok(self) -> torch.Tensor:
        return self.tok_feat


def encode_state(inst: Instance, layout: Layout, n_agv: int = 2) -> EncState:
    """实例 + 布局 → 编码器输入（M/B/V 三段；确定性：同 (inst, layout) → 同张量）。

    ⚠️ 全部为零桩——待 P2 定义特征语义（见模块 docstring）。
    """
    nm, nb, nv = inst.n_machines, inst.n_jobs, n_agv
    m_feat = np.zeros((nm, F_TOKEN), dtype=np.float64)     # 机台 token（P2 定义）
    b_feat = np.zeros((nb, F_TOKEN), dtype=np.float64)     # 工序 token（P2 定义）
    v_feat = np.zeros((nv, F_TOKEN), dtype=np.float64)     # 车辆 token（P2 定义）
    tok = torch.tensor(np.concatenate([m_feat, b_feat, v_feat], axis=0),
                       dtype=torch.float32).unsqueeze(0)   # (1, N, F)
    return EncState(tok_feat=tok, seg=(nm, nb, nv))
