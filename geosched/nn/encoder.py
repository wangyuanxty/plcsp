"""双轴解耦轴向注意力编码器（《方法设计文档》§2.3：128/4/8，轴解耦块稀疏）。

设计：序列 = [M...M (机台) | B...B (批次/工序) | V...V (车辆)]，三段规模 n_m/n_b/n_v。
- **生产轴**：M↔M、M↔B、B↔B 块内注意力（并行机-批次交互）
- **物流轴**：V↔V、V↔B、B↔B 块内注意力（批次-车辆交互）
- **几何偏置** b(i,j) = w_d·(d_ij/char_len) + w_c·conf_sim(i,j) **真正加进注意力打分**
- 复杂度：每层两个带掩码注意力，块规模各自独立（∑O(块²) 替代 O(N²)——"双轴解耦"即该掩码结构）
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_SEG_INDEX = {"M": 0, "B": 1, "V": 2}


def block_mask(seq_len: int, seg: tuple[int, int, int], axis: str, device=None) -> torch.Tensor:
    """轴掩码（True=允许注意力）。seg=(n_m, n_b, n_v)；axis='prod'|'logi'。"""
    nm, nb, nv = seg
    starts = {"M": 0, "B": nm, "V": nm + nb}
    allowed = (("M", "M"), ("M", "B"), ("B", "M"), ("B", "B")) if axis == "prod" else \
              (("V", "V"), ("V", "B"), ("B", "V"), ("B", "B"))
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for (a, b) in allowed:
        ia0, ja0 = starts[a], starts[b]
        sa, sb = seg[_SEG_INDEX[a]], seg[_SEG_INDEX[b]]
        mask[ia0:ia0 + sa, ja0:ja0 + sb] = True
    return mask


class DualAxisLayer(nn.Module):
    """一层双轴注意力（生产轴→物流轴→FFN），几何偏置加入打分。"""

    def __init__(self, d: int = 128, h: int = 4):
        super().__init__()
        self.d, self.h, self.dh = d, h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.oproj = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def _axis_attn(self, x: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.h, self.dh).transpose(1, 2)
        k = k.view(B, N, self.h, self.dh).transpose(1, 2)
        v = v.view(B, N, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        scores = scores + bias.unsqueeze(1)                       # 几何偏置（B,N,N → 分头广播）
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)     # 轴掩码（不可见块 -inf）
        w = F.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).reshape(B, N, self.d)
        return self.oproj(out)

    def forward(self, x: torch.Tensor, prod_m: torch.Tensor, logi_m: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        x = x + self.ln1(self._axis_attn(x, prod_m, bias))
        x = x + self.ln2(self._axis_attn(x, logi_m, bias))
        return x + self.ffn(x)


class GeomBias(nn.Module):
    """b(i,j) = w_d·(d_ij/char_len) + w_c·conf_sim(i,j)。"""

    def __init__(self, w_d: float = 1.0, w_c: float = 1.0):
        super().__init__()
        self.w_d, self.w_c = w_d, w_c

    def forward(self, dist: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        return self.w_d * dist + self.w_c * conf


class LayoutEncoder(nn.Module):
    """双轴轴向注意力编码器（128/4/8）。输入 token 特征 → 每 token 嵌入 + 全局上下文。"""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 8,
                 feat_dim: int = 10, w_d: float = 1.0, w_c: float = 1.0,
                 mask: str = "axial"):
        super().__init__()
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.mask = mask                              # 'axial'=双轴解耦（默认）| 'full'=全注意力
        self.embed = nn.Linear(feat_dim, d_model)
        self.bias = GeomBias(w_d, w_c)
        self.layers = nn.ModuleList([DualAxisLayer(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, tok_feat: torch.Tensor, seg: tuple[int, int, int],
                dist: torch.Tensor, conf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tok_feat: (B,N,F)；dist/conf: (B,N,N) 几何偏置输入（跨段处 conf 可为 0）；seg=(nm,nb,nv)。

        返回 (token 嵌入 (B,N,d), 全局上下文 (B,d)=均值池化)。
        """
        B, N, _ = tok_feat.shape
        if dist.dim() == 2:
            dist = dist.unsqueeze(0)
        if conf.dim() == 2:
            conf = conf.unsqueeze(0)
        x = self.embed(tok_feat)
        bias = self.bias(dist.to(x.dtype), conf.to(x.dtype))       # (B,N,N) 类型守卫（几何特征常为 float64）
        if self.mask == "full":
            pm = torch.ones(N, N, dtype=torch.bool, device=x.device).unsqueeze(0).expand(B, -1, -1)
            lm = pm.clone()
        else:
            pm = block_mask(N, seg, "prod", x.device).unsqueeze(0).expand(B, -1, -1)
            lm = block_mask(N, seg, "logi", x.device).unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pm, lm, bias)
        x = self.ln(x)
        return x, x.mean(dim=1)
