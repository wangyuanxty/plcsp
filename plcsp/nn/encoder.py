"""四段 token 编码器（《方法设计文档》§2.3：128/4/8）——**统一宽度 + 类型嵌入**。

序列 = [M...M (机台) | B...B (作业) | V...V (车辆) | G (全局)]，四段行数 (n_m, n_jobs, n_agv, 1)。
输入是**单张** `(N, F_MAX)` 张量（列定义见 `features.py`：各段补零到 `F_MAX=11`）：

- **每段一个自己的 `Linear`**（`proj`，2026-10-04 改）：M/B/V/G 各自投到 `d_model`，
  每段只吃自己那几列（7/9/11/3）。**推翻 P2 的"单个共享 Linear"决定**——旧方案白乘补零列
  （G 段只有 3/10 列有效），且同一列在不同段语义不同（第 0 列在 M 段是 `backlog`、
  在 V 段是 `st_idle`），共享权重被迫用一组系数解释两个意思。
  ⚠️ **A/B 实测测不出性能差别**（见 `test_per_segment_projection` 的 docstring）——
  这是**工程整洁**，**论文里不作为贡献**。
- **类型嵌入 `type_emb (4, d_model)`**：由 `seg` 给每个 token 定类型 id（0=M/1=B/2=V/3=G）。
  ⚠️ **仍然必需**：四段投到**同一个 d 维空间**，注意力算 q·k 相似度时，
  没有任何东西保证不同类型的 token 落在可区分的位置。类型嵌入是那个显式抓手。
- **全连接注意力**（spec §5.1）：原双轴块稀疏掩码（`block_mask` / `DualAxisLayer`）整体删除。
- **补零列必须恒为 0**（spec §5.3.1）：`SEG_SLICE` 之外的列一旦非零即抛 `ValueError`——**不静默清零**。
  单 Linear 会照吃 `W[:, 7:]` 这类补零列权重，上游列偏移写错必须当场炸出来（快速失败纪律）。

⚠️ 2026-10-02：**几何偏置（GeomBias）已移除**——几何/度量感知路线整条砍除
（见 `progress-log.md` §12.6）。本模块不再接收 dist/conf。
⚠️ 2026-10-03：**P2 Task 3 重写**——旧版三类 token 被迫同宽（单一 `Linear(6, ·)`）、无类型
嵌入，`mask='full'` 下 `seg` 完全不被使用，网络分不出 M/B/V（spec §5.3.1 现状表 #3）。
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import F_B, F_G, F_M, F_MAX, F_V, SEG_SLICE

_SEG_KEYS = ("M", "B", "V", "G")          # 与 seg 元组的段序一一对应（不可改动）
_SEG_DIMS = (F_M, F_B, F_V, F_G)          # 各段实际列数（= `SEG_SLICE` 的宽度）


class AttnLayer(nn.Module):
    """一层标准全连接注意力（原 `DualAxisLayer`——轴掩码与其形参一并删除）。"""

    def __init__(self, d: int = 128, h: int = 4):
        super().__init__()
        self.d, self.h, self.dh = d, h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.oproj = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.h, self.dh).transpose(1, 2)
        k = k.view(B, N, self.h, self.dh).transpose(1, 2)
        v = v.view(B, N, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)   # 全连接：无掩码
        w = F.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).reshape(B, N, self.d)
        return self.oproj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ln1(self._attn(x))
        return x + self.ln2(self.ffn(x))


class LayoutEncoder(nn.Module):
    """统一宽度编码器（128/4/8）。`(N, F_MAX)` token 特征 → 每 token 嵌入 + 全局上下文。

    ⚠️ `feat_dim` 必须等于 `F_MAX`（`SEG_SLICE` 的列布局按 `F_MAX` 定死）。
    """

    N_SEG_TYPES = 4                                # M / B / V / G

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 8,
                 feat_dim: int = F_MAX):
        super().__init__()
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.feat_dim = feat_dim
        # **分段投影**：每段一个 Linear，只吃自己那几列（见模块 docstring）。
        self.proj = nn.ModuleList([nn.Linear(dim, d_model) for dim in _SEG_DIMS])
        # 类型嵌入**必需**：四段投到同一个 d 维空间，注意力需要显式类型抓手。小随机初值
        # （BERT 式 0.02）——若零初值，特征全同的 token 会退化成完全相同的嵌入。
        self.type_emb = nn.Parameter(torch.randn(self.N_SEG_TYPES, d_model) * 0.02)
        self.layers = nn.ModuleList([AttnLayer(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, tok_feat: torch.Tensor, seg: tuple[int, int, int, int]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """tok_feat: **单张** (1, N, F_MAX)；seg=(n_m, n_jobs, n_agv, n_g=1)。

        返回 (token 嵌入 (1, N, d), 全局上下文 (1, d)=均值池化)。
        `seg` 用来生成每个 token 的**类型 id**（决定加哪个 `type_emb`）、
        核对补零列为 0，并核对总长。
        """
        n_m, n_b, n_v, n_g = seg
        N = n_m + n_b + n_v + n_g
        assert tok_feat.shape[1] == N, f"tok_feat 行数 {tok_feat.shape[1]} != sum(seg)={N}"
        tid = torch.cat([torch.full((n,), i, dtype=torch.long)
                         for i, n in enumerate(seg)])          # 每个 token 的类型 id
        self._require_zero_padding(tok_feat, seg)
        # 分段投影：各段只取自己那几列，拼回 (N, d) 后再进注意力。
        parts, r = [], 0
        for i, n in enumerate(seg):
            if n:
                parts.append(self.proj[i](tok_feat[0, r:r + n, :_SEG_DIMS[i]]))
            r += n
        x = (torch.cat(parts, dim=0) + self.type_emb[tid]).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return x, x.mean(dim=1)

    def _require_zero_padding(self, tok_feat: torch.Tensor,
                              seg: tuple[int, int, int, int]) -> None:
        """补零列（`SEG_SLICE` 之外）必须恒为 0，否则抛 `ValueError`——**不静默清零**。

        ⚠️ 分段投影**只取各段自己的列**（`tok_feat[..., :_SEG_DIMS[i]]`），补零列不参与计算——
        故此处不再有"补零列被白乘"的风险。但守卫**照旧保留**：上游列偏移写错（把某个特征
        写到别的段的位置上）会让**有效列**错位，静默通过就变成看不见的 bug，故仍然快速失败。
        补零列本应恒 0，见 spec §5.3.1。
        """
        r = 0
        for key, n in zip(_SEG_KEYS, seg):
            sl = SEG_SLICE[key]
            pad = list(range(sl.start or 0)) + list(range(sl.stop, self.feat_dim))
            if n and pad and bool((tok_feat[0, r:r + n][:, pad] != 0).any()):
                raise ValueError(
                    f"{key} 段的补零列 {pad} 非 0（行 {r}..{r + n - 1}）：补零列本应恒 0，"
                    f"见 spec §5.3.1——上游列偏移写错？")
            r += n
