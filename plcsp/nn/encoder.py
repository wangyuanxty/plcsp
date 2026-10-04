"""四段 token 编码器（《方法设计文档》§2.3：128/4/8）——**统一宽度 + 类型嵌入**。

序列 = [M...M (机台) | B...B (作业) | V...V (车辆) | G (全局)]，四段行数 (n_m, n_jobs, n_agv, 1)。
输入是 **(B, N, F_MAX)** 张量（`B ≥ 1`；列定义见 `features.py`：各段补零到 `F_MAX=11`）：

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
    """统一宽度编码器（128/4/8）。`(B, N, F_MAX)` token 特征 → 每 token 嵌入 + 全局上下文。

    ⚠️ `feat_dim` 必须等于 `F_MAX`（`SEG_SLICE` 的列布局按 `F_MAX` 定死）。
    ⚠️ **批维（B ≥ 1）**（2026-10-04 批量重算批次）：训练的重算路径把一条链乃至整组 G 条链的
    全部决策堆成一批、**一次**前向——编码器是耗时主项（实测 MK01 单条 ≈6.0 ms、231 条批成
    一次 ≈402 ms，**~3.4×**）。批内各元素**互相独立**（注意力只在 N 轴），故逐元素结果与单条
    前向数值一致（仅矩阵乘分块的末位漂移，≤1e-5）。
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
        """tok_feat: **(B, N, F_MAX)**（B ≥ 1）；seg=(n_m, n_jobs, n_agv, n_g=1)。

        返回 (token 嵌入 (B, N, d), 全局上下文 (B, d)=按 token 均值池化)。
        `seg` 用来生成每个 token 的**类型 id**（决定加哪个 `type_emb`）、
        核对补零列为 0，并核对总长。

        ⚠️ **两维（N,F）的旧形式已不接受**：批维是重算路径的性能来源（一条链 100–460 个
        决策一次前向；本机实测 231 条批前向 ≈402 ms vs 逐条 ≈1.38 s，**吞吐 ~3.4×**），
        隐式补批维会让"批没接上"变成静默错。单条请显式传 `(1, N, F)`（`PolicyNet.forward_enc`
        负责补维）。
        ⚠️ 批内**每行各自独立**：没有任何跨 batch 的注意力/归一化，故逐行结果与单条前向
        数值一致（仅 BLAS 分块带来的末位漂移，≤1e-5）。
        """
        n_m, n_b, n_v, n_g = seg
        N = n_m + n_b + n_v + n_g
        assert tok_feat.dim() == 3, \
            f"tok_feat 必须是 (B,N,F) 三维张量，收到 shape={tuple(tok_feat.shape)}"
        assert tok_feat.shape[1] == N, f"tok_feat 行数 {tok_feat.shape[1]} != sum(seg)={N}"
        tid = torch.cat([torch.full((n,), i, dtype=torch.long)
                         for i, n in enumerate(seg)])          # 每个 token 的类型 id
        self._require_zero_padding(tok_feat, seg)
        # 分段投影：各段只取自己那几列，拼回 (B, N, d) 后再进注意力（**批维在 0 轴**）。
        parts, r = [], 0
        for i, n in enumerate(seg):
            if n:
                parts.append(self.proj[i](tok_feat[:, r:r + n, :_SEG_DIMS[i]]))
            r += n
        x = torch.cat(parts, dim=1) + self.type_emb[tid]
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
        ⚠️ **逐 batch 元素检查**（2026-10-04 批量重算批次）：只查第 0 行等于对批内其余决策
        开门——重算把几百个决策堆成一批，任何一条的上游错位都必须当场炸出来。
        ⚠️ **一次归约、一次设备同步**（2026-10-04 设备批次）：全部段的结果先在**设备上**用 `|`
        合并，最后只做**一次** Python `bool()`。旧版每段各做一次 `bool(...any())`——在 CUDA 上
        每次 `bool()` 都是一次设备同步，会打断流水线（重算改成 GPU 批前向的收益会被它吃掉）。
        ⚠️ **已知限度（如实写明）**：`bool()` 本身仍是一次同步——做不到零同步（要抛异常就必须要
        结果）。**为什么可接受**：批量化后每个训练步在设备上只有**一次**重算前向（在线路径
        在 CPU 上无同步代价），实测同步约 10–50 µs，对 B≥230 的 GPU 前向（~25 ms/次）<0.2%。
        真要零同步就得**不检查**——那正是本仓反复拒绝的"静默"（见 `_require_zero_padding`
        的快速失败纪律），故选择保留检查、把同步压到一次。
        """
        r = 0
        bad = torch.zeros((), dtype=torch.bool, device=tok_feat.device)
        hits: list[tuple[str, int, int, list[int]]] = []
        for key, n in zip(_SEG_KEYS, seg):
            sl = SEG_SLICE[key]
            pad = list(range(sl.start or 0)) + list(range(sl.stop, self.feat_dim))
            if n and pad:
                bad = bad | (tok_feat[:, r:r + n][..., pad] != 0).any()
                hits.append((key, r, r + n, pad))
            r += n
        if not hits or not bool(bad):            # 唯一的设备同步点（见 docstring）
            return
        for key, lo, hi, pad in hits:            # 仅在报错路径上重查，定位出错的那一段
            if bool((tok_feat[:, lo:hi][..., pad] != 0).any()):
                raise ValueError(
                    f"{key} 段的补零列 {pad} 非 0（行 {lo}..{hi - 1}）：补零列本应恒 0，"
                    f"见 spec §5.3.1——上游列偏移写错？")
