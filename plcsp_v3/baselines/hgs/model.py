# -*- coding: utf-8 -*-
"""HGS 网络（原文 §IV-C 异构编码器 + §IV-D 三阶段解码器）。

**逐字参数**（原文 §V-A2）：`L=2` 编码层、`d_h=128`、`d_e=1`、`d_z=16`、`d_ff=512`、
`Z=8` 头、`d_q=d_k=d_v=8`。

实现口径（原文未写死的两处，显式记）：

- **AN**：原文只说"instance normalization [25]"。本实现用 `LayerNorm(d_h)`（逐节点、在特征维上
  归一化）——图节点就是这个语境下的 "instance"。
- **边嵌入**：`d_e=1` ⟹ 边是**标量**。层内更新 `h_xy ← W_e3·σ̃_xy`（原文 Eq.9）；多头时取各头
  `σ̃` 的均值。同一条边的两个方向值相同（原文 §IV-C 明说 `h_kij` 与 `h_ijk` 同值）。

结构：

- 节点 = 工序 ∪ 机台 ∪ 车辆（原文 Eq.7 的邻居规则：工序看机台与车辆；机台看工序与机台；
  车辆看工序）。自环保留（残差需要）。
- 第 1 层 = 子编码器（类间共享一套 HMHA）；第 2 层 = 全局编码器（另一套权重，邻居规则不变）。
- 解码器三段：上下文 `h_c = [mean(H) ‖ h_glimpse]`，依次选 **工序 → 机台 → 车辆**；
  每段先做多头注意力聚合候选，再用单头 `C·tanh(q·k/√d_k)`（Eq.19，C=10）出 logits。
- 复合动作概率 = 三段概率之积（Eq.26）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import GraphState

NEG_INF = -1e9
CLIP_C = 10.0                 # 原文 Eq.19 的裁剪常数
N_CLASSES = 3                 # 0 = 工序、1 = 机台、2 = 车辆


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对 `mask` 为真的位置取 log-softmax。全 False 时显式报错（调用方应先保证有候选）。"""
    if not bool(mask.any()):
        raise ValueError("候选集为空——调用方必须先保证至少一个可行动作")
    return torch.log_softmax(logits.masked_fill(~mask, NEG_INF), dim=-1)


class Hmha(nn.Module):
    """异构多头注意力（原文 Eq.8–10）+ AN + FF（Eq.14）。"""

    def __init__(self, dh: int, dz: int, dff: int, n_heads: int, dv: int, de: int = 1):
        super().__init__()
        self.dh, self.dz, self.Z, self.dv = dh, dz, n_heads, dv
        self.q = nn.ModuleList([nn.Linear(dh, n_heads * dv) for _ in range(N_CLASSES)])
        self.k = nn.ModuleList([nn.Linear(dh, n_heads * dv) for _ in range(N_CLASSES)])
        self.v = nn.ModuleList([nn.Linear(dh, n_heads * dv) for _ in range(N_CLASSES)])
        # 增广兼容度 Eq.8：`W^e1_xy / W^e2_xy` 随**节点类对**变（原文记法）⟹ 每个目标类一套；
        # 类对里的源类差异由 q/k/v（也是按类分的）与边特征本身承担。
        # 形状：每头一套 W_e1 ∈ R^{dz×(1+d_e)}、W_e2 ∈ R^{1×dz}
        self.e1_w = nn.Parameter(torch.randn(N_CLASSES, n_heads, dz, 1 + de) * 0.1)
        self.e1_b = nn.Parameter(torch.zeros(N_CLASSES, n_heads, dz))
        self.e2_w = nn.Parameter(torch.randn(N_CLASSES, n_heads, 1, dz) * 0.1)
        self.e2_b = nn.Parameter(torch.zeros(N_CLASSES, n_heads))
        # 边更新 Eq.9：每头 W_e3 ∈ R^{d_e×1}（d_e=1 ⟹ 标量）；多头取均值
        self.e3_w = nn.Parameter(torch.randn(N_CLASSES, n_heads) * 0.1)
        self.out = nn.ModuleList([nn.Linear(dv, dh) for _ in range(N_CLASSES * n_heads)])
        self.ln1 = nn.LayerNorm(dh)                        # AN（见模块 docstring）
        self.ln2 = nn.LayerNorm(dh)
        self.ff1 = nn.Linear(dh, dff)
        self.ff2 = nn.Linear(dff, dh)

    def forward(self, h: torch.Tensor, cls: torch.Tensor, mask: torch.Tensor,
                edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`h` (M,dh)、`cls` (M,)、`mask` (M,M) 允许的邻居、`edge` (M,M) 标量边特征。"""
        M, Z, dv = h.shape[0], self.Z, self.dv
        q = torch.empty(M, Z, dv)
        for c in range(N_CLASSES):
            idx = cls == c
            if bool(idx.any()):
                q[idx] = self.q[c](h[idx]).view(-1, Z, dv)
        k_src, v_src = torch.empty(M, Z, dv), torch.empty(M, Z, dv)
        for c in range(N_CLASSES):
            idx = cls == c
            if bool(idx.any()):
                k_src[idx] = self.k[c](h[idx]).view(-1, Z, dv)
                v_src[idx] = self.v[c](h[idx]).view(-1, Z, dv)
        sigma = torch.einsum("izd,jzd->ijz", q, k_src) / (dv ** 0.5)
        inp = torch.stack([sigma, edge.unsqueeze(-1).expand(M, M, Z)], dim=-1)   # (M,M,Z,2)
        aug = torch.einsum("ijzk,izdk->ijzd", inp, self.e1_w[cls]) + self.e1_b[cls].unsqueeze(1)
        sig = (torch.einsum("ijzd,izd->ijz", F.relu(aug), self.e2_w[cls][:, :, 0, :])
               + self.e2_b[cls].unsqueeze(1))
        sig = sig.masked_fill(~mask.unsqueeze(-1), NEG_INF)
        alpha = torch.softmax(sig, dim=1)
        hp = torch.einsum("ijz,jzd->izd", alpha, v_src)            # (M,Z,dv)
        out = torch.zeros(M, self.dh)
        for c in range(N_CLASSES):
            idx = (cls == c).unsqueeze(-1)
            for z in range(Z):
                out = out + self.out[c * Z + z](hp[:, z, :]) * idx
        h = self.ln1(h + out)
        h = self.ln2(h + self.ff2(F.relu(self.ff1(h))))
        edge_new = torch.einsum("ijz,iz->ij", sig, self.e3_w[cls]) / Z   # 多头均值（d_e=1）
        edge_new = edge_new.masked_fill(~mask, 0.0)
        return h, edge_new


class DecoderMha(nn.Module):
    """解码器用的标准多头注意力（Eq.2–6）：上下文为 query，候选节点为 key/value。"""

    def __init__(self, dh: int, n_heads: int, dv: int):
        super().__init__()
        self.Z, self.dv, self.dh = n_heads, dv, dh
        self.q = nn.Linear(dh, n_heads * dv)
        self.k = nn.Linear(dh, n_heads * dv)
        self.v = nn.Linear(dh, n_heads * dv)
        self.out = nn.ModuleList([nn.Linear(dv, dh) for _ in range(n_heads)])

    def forward(self, ctx: torch.Tensor, nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        Z, dv = self.Z, self.dv
        q = self.q(ctx).view(Z, dv)
        k = self.k(nodes).view(-1, Z, dv)
        v = self.v(nodes).view(-1, Z, dv)
        sigma = torch.einsum("zd,izd->iz", q, k) / (dv ** 0.5)
        sigma = sigma.masked_fill(~mask.unsqueeze(-1), NEG_INF)
        alpha = torch.softmax(sigma, dim=0)                        # 候选维
        hp = torch.einsum("iz,izd->zd", alpha, v)
        out = torch.zeros(self.dh)
        for z in range(Z):
            out = out + self.out[z](hp[z])
        return out


class HgsNet(nn.Module):
    """HGS 策略网络：`forward(graph, h_glimpse) -> 三段分布`。"""

    def __init__(self, dh: int = 128, de: int = 1, dz: int = 16, dff: int = 512,
                 n_heads: int = 8, dv: int = 8, layers: int = 2):
        super().__init__()
        self.cfg = dict(dh=dh, de=de, dz=dz, dff=dff, n_heads=n_heads, dv=dv, layers=layers)
        self.op_in = nn.Linear(7, dh)
        self.mach_in = nn.Linear(4, dh)
        self.veh_in = nn.Linear(4, dh)
        self.layers = nn.ModuleList([Hmha(dh, dz, dff, n_heads, dv, de) for _ in range(layers)])
        self.ctx_in = nn.Linear(2 * dh, dh)
        self.dec_op = DecoderMha(dh, n_heads, dv)
        self.dec_mach = DecoderMha(dh, n_heads, dv)
        self.dec_veh = DecoderMha(dh, n_heads, dv)

    # ── 编码 ──

    def encode(self, g: GraphState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 `(h_op, h_mach, h_veh, edge)`——`edge` 是第 L 层的边嵌入（Eq.21/23 要用）。"""
        op = torch.as_tensor(g.op_feat, dtype=torch.float32)
        ma = torch.as_tensor(g.mach_feat, dtype=torch.float32)
        ve = torch.as_tensor(g.veh_feat, dtype=torch.float32)
        h = torch.cat([self.op_in(op), self.mach_in(ma), self.veh_in(ve)], dim=0)
        N, m, v = g.n_ops, g.n_machines, g.n_agv
        cls = torch.cat([torch.zeros(N, dtype=torch.long),
                         torch.ones(m, dtype=torch.long),
                         torch.full((v,), 2, dtype=torch.long)])
        mask, edge = self._graph_tensors(g)
        for layer in self.layers:
            h, edge = layer(h, cls, mask, edge)
        return h[:N], h[N:N + m], h[N + m:], edge

    @staticmethod
    def _graph_tensors(g: GraphState) -> tuple[torch.Tensor, torch.Tensor]:
        """邻居掩码与标量边特征。规则 = 原文 Eq.11–13（含自环）。"""
        N, m, v = g.n_ops, g.n_machines, g.n_agv
        M = N + m + v
        mask = torch.zeros(M, M, dtype=torch.bool)
        edge = torch.zeros(M, M)
        om = torch.as_tensor(g.om_edge, dtype=torch.float32)
        ov = torch.as_tensor(g.ov_edge, dtype=torch.float32)
        mm = torch.as_tensor(g.mm_edge, dtype=torch.float32)
        ops, mchs, vehs = slice(0, N), slice(N, N + m), slice(N + m, M)
        mask[ops, mchs] = mask[mchs, ops] = True      # 工序 ↔ 机台
        mask[ops, vehs] = mask[vehs, ops] = True      # 工序 ↔ 车辆
        mask[mchs, mchs] = True                       # 机台 ↔ 机台（满载行程）
        mask[ops, ops] = mask[vehs, vehs] = True      # 自环（残差）
        edge[ops, mchs] = om
        edge[mchs, ops] = om.T
        edge[ops, vehs] = ov
        edge[vehs, ops] = ov.T
        edge[mchs, mchs] = mm
        return mask, edge

    # ── 解码：三段分布 ──

    def decode(self, h_op: torch.Tensor, h_mach: torch.Tensor, h_veh: torch.Tensor,
               edge: torch.Tensor, g: GraphState, h_glimpse: torch.Tensor, op_i: int | None = None,
               mach_k: int | None = None) -> dict:
        """算出三段 logits（masked）。`op_i`/`mach_k` 给定时跳过上一段的采样（求 logp 用）。"""
        eligible = torch.as_tensor(g.eligible)
        h_all = torch.cat([h_op, h_mach, h_veh], dim=0)
        ctx = self.ctx_in(torch.cat([h_all.mean(dim=0), h_glimpse], dim=0))
        om_l = edge[:g.n_ops, g.n_ops:g.n_ops + g.n_machines]     # 第 L 层边嵌入（Eq.21）
        ov_l = edge[:g.n_ops, g.n_ops + g.n_machines:]            # Eq.23
        out: dict = {}
        # ① 工序（Eq.18–20）
        ctx = self.dec_op(ctx, h_op, eligible)
        logits = CLIP_C * torch.tanh((ctx * h_op).sum(-1) / (self.cfg["dv"] ** 0.5))
        lp_op = _masked_log_softmax(logits, eligible)
        if op_i is None:
            op_i = int(torch.multinomial(lp_op.exp(), 1).item())
        out["op_i"] = op_i
        out["logp_op"] = lp_op[op_i]
        out["ent_op"] = -(lp_op.exp() * lp_op).sum()
        # ② 机台（Eq.21–22）
        compat = torch.as_tensor(g.compat[op_i])
        mach_mask = compat & torch.as_tensor(g.idle_mach)
        nodes = h_mach + om_l[op_i].unsqueeze(-1)
        ctx = self.dec_mach(ctx, nodes, mach_mask)
        logits = CLIP_C * torch.tanh((ctx * h_mach).sum(-1) / (self.cfg["dv"] ** 0.5))
        lp_m = _masked_log_softmax(logits, mach_mask)
        if mach_k is None:
            mach_k = int(torch.multinomial(lp_m.exp(), 1).item())
        out["mach_k"] = mach_k
        out["logp_mach"] = lp_m[mach_k]
        out["ent_mach"] = -(lp_m.exp() * lp_m).sum()
        # ③ 车辆（Eq.23–24）
        veh_mask = torch.as_tensor(g.idle_veh)
        nodes = h_veh + ov_l[op_i].unsqueeze(-1)
        ctx = self.dec_veh(ctx, nodes, veh_mask)
        logits = CLIP_C * torch.tanh((ctx * h_veh).sum(-1) / (self.cfg["dv"] ** 0.5))
        lp_v = _masked_log_softmax(logits, veh_mask)
        veh_u = int(torch.multinomial(lp_v.exp(), 1).item())
        out["veh_u"] = veh_u
        out["logp_veh"] = lp_v[veh_u]
        out["ent_veh"] = -(lp_v.exp() * lp_v).sum()
        out["h_glimpse"] = h_op[op_i] + h_mach[mach_k] + h_veh[veh_u]
        return out

    @staticmethod
    def _require_candidates(mask: torch.Tensor, what: str) -> None:
        """候选集为空 = 调用方给了不可行的状态（动作空间必须至少有一个可行动作）。"""
        if not bool(mask.any()):
            raise ValueError(f"{what}的候选集为空——调用方必须先保证至少一个可行动作")

    def decode_greedy(self, h_op, h_mach, h_veh, edge, g, h_glimpse) -> dict:
        """贪心三段：用于 greedy rollout baseline（原文 Alg.1 第 17 行）。"""
        eligible = torch.as_tensor(g.eligible)
        h_all = torch.cat([h_op, h_mach, h_veh], dim=0)
        ctx = self.ctx_in(torch.cat([h_all.mean(dim=0), h_glimpse], dim=0))
        om_l = edge[:g.n_ops, g.n_ops:g.n_ops + g.n_machines]
        ov_l = edge[:g.n_ops, g.n_ops + g.n_machines:]
        self._require_candidates(eligible, "工序")
        ctx = self.dec_op(ctx, h_op, eligible)
        logits = CLIP_C * torch.tanh((ctx * h_op).sum(-1) / (self.cfg["dv"] ** 0.5))
        op_i = int(logits.masked_fill(~eligible, NEG_INF).argmax())
        compat = torch.as_tensor(g.compat[op_i])
        mach_mask = compat & torch.as_tensor(g.idle_mach)
        self._require_candidates(mach_mask, "机台")
        nodes = h_mach + om_l[op_i].unsqueeze(-1)
        ctx = self.dec_mach(ctx, nodes, mach_mask)
        logits = CLIP_C * torch.tanh((ctx * h_mach).sum(-1) / (self.cfg["dv"] ** 0.5))
        mach_k = int(logits.masked_fill(~mach_mask, NEG_INF).argmax())
        veh_mask = torch.as_tensor(g.idle_veh)
        self._require_candidates(veh_mask, "车辆")
        nodes = h_veh + ov_l[op_i].unsqueeze(-1)
        ctx = self.dec_veh(ctx, nodes, veh_mask)
        logits = CLIP_C * torch.tanh((ctx * h_veh).sum(-1) / (self.cfg["dv"] ** 0.5))
        veh_u = int(logits.masked_fill(~veh_mask, NEG_INF).argmax())
        return {"op_i": op_i, "mach_k": mach_k, "veh_u": veh_u,
                "h_glimpse": h_op[op_i] + h_mach[mach_k] + h_veh[veh_u]}
