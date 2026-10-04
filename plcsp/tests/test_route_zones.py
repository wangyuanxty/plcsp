# -*- coding: utf-8 -*-
"""R2 区段 token（① 拥堵的上下文相关排序）——把**区段**变成 token 进注意力。

设计（`docs/mechanism-designs.md` 的「R2 区段 token」节，开工前定死）：
- 开关 `route_zones`（默认关）；关时 token 序列**逐位不变**（`seg` 仍是四元组）。
- 打开时在序列**末位追加第五段 Z**：每区段一个 token，`seg` 变五元组
  （`(n_m, n_jobs, n_agv, 1, n_zones)`）。
- 区段状态取自快照：`zone_holder`（占用）+ `zone_wait`（逐区段最大当前等待，
  由 `VehicleState.zone_wait` 聚合）+ `layout` 坐标（区段身份）。
- R 头**逐候选**读该候选路径途经区段的 token（均值池化）——候选之间的分差自此
  **不再只来自 `feat_cand`**。

本文件的判据即设计节的验收判据 1–5。
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch

from plcsp.algo.group_rel import (decisions_logp, roll_chain)
from plcsp.algo.policy import PolicyNet, v_token_index
from plcsp.algo.setup import build_setup
from plcsp.env.des import SimConfig, build_zone_map
from plcsp.env.instances import load_mk
from plcsp.env.snapshot import Snapshot, VehicleState
from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_MAX, F_Z
from plcsp.nn.state_emb import build_tok, zone_features

# 黄金摘要：**改造前**（2026-10-05，HEAD=95e94d7）在 MK01、`route_k=2`、`route_zones=False`
# 上捕获的链路指纹（决策类型 + 全部打分上下文 + tok_idx + 动作 + 采样 logp + 关键指标）。
# 它钉的是"R2 关档 ⟹ 既有 route_k=2 读数逐位不变"——R2 的接线（build_tok 的 zof、
# `_act` 的 zone 分支、R 头的新参数）若在关档下动了任何一位，这里立刻翻红。
# 同一次捕获还复算了 `route_k=1` 的摘要（与 test_route_choice.GOLDEN_CHAIN_DIGEST 同源口径的
# 独立脚本，值为 4241fec403248727280c2aacc888569f03b7d3b5b50a53dc10f2fc0031479228，
# 与改造前逐位相同——默认档的保证由既有黄金摘要继续守着，这里只钉 route_k=2 关档）。
R2_OFF_CHAIN_DIGEST = "9fe15b4f293e74bf43e47652c079a13f54395a066dc6e03f4c1eff1c88b0a7c8"


def _setup(name="mk01"):
    inst = load_mk(name)
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    return inst, lay, dm, cfg, ctx


def _chain_digest(dec, met) -> str:
    """链路指纹——与改造前捕获脚本逐字同序（含 `tok_idx`）。"""
    h = hashlib.sha256()
    for d in dec:
        h.update(d.kind.encode())
        h.update(np.ascontiguousarray(d.tok, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(d.feat, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(d.cand_feat, dtype=np.float32).tobytes())
        h.update(np.asarray(d.cand, dtype=np.int64).tobytes())
        h.update(np.asarray(d.tok_idx, dtype=np.int64).tobytes())
        h.update(np.asarray([d.action], dtype=np.int64).tobytes())
        h.update(np.asarray([d.logp], dtype=np.float32).tobytes())
    h.update(np.asarray([met["makespan"], met["travel_time_total"],
                         float(met["energy"]), float(met["deliveries"])],
                        dtype=np.float64).tobytes())
    return h.hexdigest()


class _NoMixEncoder(torch.nn.Module):
    """逐 token **恒等**嵌入（无跨 token 注意力）——只为隔离"R 头读了哪些 token"这一件事。

    真编码器是全连接注意力：扰动任一 token 都会经注意力改变所有 token 的上下文嵌入，
    "读错区段"的头也跟着变——判据被掩盖（§31.2 的实测）。逐 token 恒等嵌入切断这条泄漏后，
    `tok_c[i] = tok[V] + mean(该候选的区段 token)` 的**路径相关性**才是可判的量，
    而 `_act` / `decisions_logp` 仍走生产代码。
    """

    d_model = F_MAX

    def forward(self, x, seg):
        return x, x.mean(dim=1)


# ────────────────────────── 1. 关档逐位不变 ──────────────────────────

@pytest.mark.unit
def test_route_zones_off_is_bit_identical_to_baseline():
    """判据 1：`route_zones=False`（默认）⟹ 既有 `route_k=2` 链路逐位不变——黄金摘要钉死。

    摘要含每条决策的 tok/feat/cand_feat/cand/tok_idx/action/logp 与
    makespan/travel/energy/deliveries。R2 的接线若在关档下动了 `build_tok` 的行数、
    R 头的打分式或采样流，这里立刻变红。
    """
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(20261005)                 # 与改造前捕获脚本逐字对齐的初始化锚点
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2)
    assert [d.kind for d in dec].count("R") == 115, "关档的 R 决策数变了——先查采样流"
    assert all(d.seg == (inst.n_machines, inst.n_jobs, cfg.n_agv, 1) for d in dec), \
        "关档的 seg 必须是四元组（不得出现 Z 段）"
    assert all(d.zone_idx is None and d.zone_mask is None for d in dec), \
        "关档不得记录任何区段 token 下标"
    assert _chain_digest(dec, met) == R2_OFF_CHAIN_DIGEST, \
        "R2 关档的 route_k=2 链路变了——既有读数不再成立"


@pytest.mark.unit
def test_route_zones_requires_route_k_gt_1():
    """没有 R 决策时区段 token 无人消费：**显式报错**，不静默加长序列。"""
    inst, lay, dm, cfg, ctx = _setup()
    pol = PolicyNet(enc=LayoutEncoder())
    with pytest.raises(ValueError, match="route_zones"):
        roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=1, route_zones=True)


# ────────────────────────── 2. 打开档：区段 token 真的进序列 ──────────────────────────

@pytest.mark.unit
def test_zone_tokens_extend_the_sequence_and_record_candidate_indices():
    """判据 2：打开后 `seg` 五元、行数 = 四段和 + `n_zones`；R 决策记录逐候选区段下标。

    区段号与 `build_zone_map(layout, cfg.zone_granularity)` **同源**（仿真侧也是这一份）；
    下标必须落在 `[n_base, n_base + n_zones)`，掩码只为**途经**区段置 1。
    """
    inst, lay, dm, cfg, ctx = _setup()
    zof, n_zones = build_zone_map(lay, cfg.zone_granularity)
    torch.manual_seed(20261005)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2,
                          route_zones=True)
    assert not met["horizon_hit"] and met["jobs_done"] == inst.n_jobs, "R2 档跑不完"
    n_base = inst.n_machines + inst.n_jobs + cfg.n_agv + 1          # Z 段起点
    segs = {d.seg for d in dec}
    assert segs == {(inst.n_machines, inst.n_jobs, cfg.n_agv, 1, n_zones)}, \
        f"R2 打开后 seg 应为五元组且区段数同源：{segs}"
    assert all(d.tok.shape == (n_base + n_zones, F_MAX) for d in dec), \
        "token 行数必须 = 四段和 + 区段数"
    r_dec = [d for d in dec if d.kind == "R"]
    assert len(r_dec) >= 50, f"R 决策太少（{len(r_dec)}）——判据失去意义"
    assert all(d.zone_idx is not None and d.zone_mask is not None for d in r_dec), \
        "R 决策必须记录逐候选区段下标（否则 R2 只是多了几个没人读的 token）"
    for d in r_dec:
        zi, zm = d.zone_idx, d.zone_mask
        assert zi.shape == zm.shape == (len(d.cand), zi.shape[1])
        valid = zm > 0
        assert np.all(zi[valid] >= n_base) and np.all(zi[valid] < n_base + n_zones), \
            f"有效区段 token 下标越界：{zi[valid].min()}..{zi[valid].max()}" \
            f"（合法 [{n_base},{n_base + n_zones})）"
        # 掩码是"左对齐的 1 + 右侧补位"（`_act` 右补 0；补位下标取 0，被掩码挡住）
        counts = zm.sum(axis=1)
        assert (counts >= 1).all(), "每条候选路径至少经过一个区段"
        for i in range(zm.shape[0]):                  # 逐行左对齐
            n_valid = int(counts[i])
            assert np.all(zm[i, :n_valid] == 1.0) and np.all(zm[i, n_valid:] == 0.0), \
                f"掩码不是左对齐的 1 段：{zm[i]}"
    # 同 seed 逐位可复现（R2 不额外消费采样流）
    dec_b, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2,
                          route_zones=True)
    sig = lambda ds: [(d.kind, d.cand, d.action, d.logp) for d in ds]     # noqa: E731
    assert sig(dec) == sig(dec_b), "R2 档同 seed 不再可复现"


@pytest.mark.unit
def test_zone_feature_is_not_a_dead_dim():
    """判据 2（特征层）：区段 token 的占用/等待维**读快照**——改快照必须改特征。

    手搓快照：4 个节点、1 台车、2 号区段被 1 号车持着。`zone_holder` / `zone_wait` 各改一次，
    只有对应维变——把"区段状态真的进了 token"钉成机器可判的判据（防死维）。
    """
    inst, lay, dm, cfg, ctx = _setup()
    zof, n_zones = build_zone_map(lay, cfg.zone_granularity)
    v = VehicleState(status=1, node=0, queued=0, battery_frac=1.0, capacity=1,
                     speed_factor=1.0, zone_wait=0.0)
    snap0 = Snapshot(now=1.0, machines=(), jobs=(), vehicles=(v,), n_done=0, in_flight=0,
                     zone_holder=tuple(-1 for _ in range(n_zones)),
                     zone_wait=tuple(0.0 for _ in range(n_zones)))
    f0 = zone_features(snap0, ctx, zof, n_zones)
    assert f0.shape == (n_zones, F_Z), f"区段特征形状应为 (n_zones, F_Z)：{f0.shape}"
    assert np.all(f0[:, 0] == 0.0) and np.all(f0[:, 1] == 0.0)
    held = list(snap0.zone_holder)
    held[2] = 1
    waited = list(snap0.zone_wait)
    waited[2] = ctx.zone_wait_limit * 0.5
    snap1 = Snapshot(now=1.0, machines=(), jobs=(), vehicles=(v,), n_done=0, in_flight=0,
                     zone_holder=tuple(held), zone_wait=tuple(waited))
    f1 = zone_features(snap1, ctx, zof, n_zones)
    assert f1[2, 0] == pytest.approx(1.0), f"2 号区段被持着，busy 维应为 1：{f1[2]}"
    assert f1[2, 1] == pytest.approx(0.5, abs=1e-6), f"等待维应 = 等待/上限：{f1[2]}"
    assert np.all(f1[[0, 1, 3], 0] == 0.0) and np.all(f1[[0, 1, 3], 1] == 0.0), \
        "只有 2 号区段的状态变了——别的区段 token 不得跟着动"
    # 身份维（坐标）存在且非退化：不同区段的坐标不同
    xy = f0[:, 2:4]
    assert xy.shape == (n_zones, 2) and np.isfinite(xy).all()
    assert len({tuple(np.round(r, 6)) for r in xy}) >= 2, "区段坐标全同——身份维退化"


@pytest.mark.unit
def test_build_tok_default_is_four_segments_and_zone_rows_are_zero_padded():
    """`build_tok` 的默认档（`zof=None`）仍是四段；带 Z 段时只填 `SEG_SLICE["Z"]` 的列。"""
    inst, lay, dm, cfg, ctx = _setup()
    from plcsp.env.des import SimWorld
    snap = SimWorld(inst, lay, dm, cfg).snapshot()
    tok4, seg4 = build_tok(snap, inst, lay, ctx)
    assert len(seg4) == 4 and tok4.shape == (sum(seg4), F_MAX)
    zof, n_zones = build_zone_map(lay, cfg.zone_granularity)
    tok5, seg5 = build_tok(snap, inst, lay, ctx, zof=zof, n_zones=n_zones)
    assert len(seg5) == 5 and seg5[:4] == seg4 and seg5[4] == n_zones
    assert tok5.shape == (sum(seg5), F_MAX)
    assert np.array_equal(tok5[:sum(seg4)], tok4), "Z 段不得改变前四段的任何一位"
    zrows = tok5[sum(seg4):]
    assert np.all(zrows[:, F_Z:] == 0.0), "区段行的补零列必须恒 0（编码器守卫）"


# ────────────────────────── 3. 候选之间的分差真的经过区段 token ──────────────────────────

@pytest.mark.unit
def test_r_scores_change_with_zone_state_even_when_cand_features_are_equal():
    """判据 3：两条候选的 `feat_cand` **完全相同**，只有途经区段不同 ⟹ 分数必须不同。

    这是"候选分差不再只来自 `feat_cand`"的直接判据：路径 A 读一个**空闲**区段的 token，
    路径 B 读一个**被占**区段的 token；`cand_feat` 两行同值。R2 之前两条候选**恒同分**。
    """
    inst, lay, dm, cfg, ctx = _setup()
    zof, n_zones = build_zone_map(lay, cfg.zone_granularity)
    n_base = inst.n_machines + inst.n_jobs + cfg.n_agv + 1
    v = VehicleState(status=1, node=0, queued=0, battery_frac=1.0, capacity=1,
                     speed_factor=1.0, zone_wait=0.0)
    holder = [-1] * n_zones
    holder[1] = 7                                   # 1 号区段被 7 号车占着
    snap = Snapshot(now=1.0, machines=(), jobs=(), vehicles=(v,), n_done=0, in_flight=0,
                    zone_holder=tuple(holder), zone_wait=tuple(0.0 for _ in range(n_zones)))
    tok_feat, seg = build_tok(snap, inst, lay, ctx, zof=zof, n_zones=n_zones)
    torch.manual_seed(4)
    pol = PolicyNet(enc=LayoutEncoder())
    with torch.no_grad():
        for p in pol.r_head_tok.parameters():            # 显式扰动：不赌某个初始化恰好敏感
            p.add_(0.1 * torch.randn_like(p))
    tok, _ = pol.forward_enc(torch.as_tensor(tok_feat).unsqueeze(0), seg)
    v_idx = v_token_index(seg)[0]
    drive = torch.zeros(1, 1, 4)
    cand = torch.tensor([[[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]])    # 两行完全相同
    zi = torch.tensor([[n_base + 0], [n_base + 1]])              # 空闲区段 vs 被占区段
    zm = torch.ones(2, 1)
    logits = pol.route_logits_emb(tok, drive, cand, torch.tensor([v_idx, v_idx]), zi, zm)
    assert logits.shape == (1, 1, 2)
    d = float((logits[0, 0, 1] - logits[0, 0, 0]).item())
    assert abs(d) > 1e-8, (
        "`feat_cand` 相同、途经区段不同，两条候选却同分——区段 token 没有到达 R 头打分")
    # 对照：不传区段（R2 关档口径）时两行同值必须同分——证明差异确实来自区段 token
    plain = pol.route_logits_emb(tok, drive, cand, torch.tensor([v_idx, v_idx]))
    assert torch.allclose(plain[0, 0, 0], plain[0, 0, 1]), \
        "关档口径下两行同值候选不应有分差——本判据的对照失效"


@pytest.mark.unit
def test_r2_gradient_reaches_zone_projection_and_head():
    """判据 5：链 logp 反传必须给区段投影（`proj_z`）与 R 头留下非零梯度。"""
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(5)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2,
                           route_zones=True)
    from plcsp.algo.group_rel import chain_logp
    chain_logp(dec, pol).backward()
    for name, p in (("enc.proj_z.weight", pol.enc.proj_z.weight),
                    ("enc.zone_type_emb", pol.enc.zone_type_emb),
                    ("r_head_tok[0]", pol.r_head_tok[0].weight)):
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} 无梯度"


# ────────────────────────── 4. 生产路径守卫（§31 纪律） ──────────────────────────

@pytest.mark.unit
def test_r_head_reads_the_candidates_own_zones_on_the_production_path():
    """判据 4（§31）：扰动**该候选途经区段**的 token ⟹ logp 必变；扰动**谁都不经过**的
    区段 ⟹ logp 逐位不变。

    ⚠️ 守卫**走 `roll_chain` 记录的下标**（不自己算），经 `decisions_logp`（生产重算路径）
    重算，并用逐 token 恒等嵌入切断注意力泄漏——否则"扰动任一行都变"会让判据恒真。
    ⚠️ 负判据是关键的一半：池化只该读**自己路径**的区段。若实现把 R 头写成"读所有区段"
    （例如忘了掩码、或平均了全序列），正判据照样绿、负判据必红。
    """
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(7)
    pol = PolicyNet(enc=_NoMixEncoder())
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2,
                           route_zones=True)
    r_dec = [d for d in dec if d.kind == "R"]
    assert len(r_dec) >= 20, f"R 决策太少（{len(r_dec)}）——守卫失去意义"
    n_base = int(sum(r_dec[0].seg[:4]))
    n_zones = int(r_dec[0].seg[4])
    d = r_dec[0]
    base = decisions_logp([d], pol).detach().clone()
    # 正判据：扰动**1 号候选**途经的一个区段 token（逐列不同的扰动，非均匀平移）
    z_hit = int(d.zone_idx[1, 0])
    delta = np.zeros((1, F_MAX), dtype=np.float32)
    delta[0, :F_Z] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d2 = copy.deepcopy(d)
    d2.tok[z_hit:z_hit + 1] += delta
    assert not torch.equal(base, decisions_logp([d2], pol).detach()), (
        f"扰动候选 1 途经区段（token {z_hit}）后该 R 决策的 logp 不变——R 头没读它自己的区段")
    # 负判据：找一个**两条候选都不经过**的区段——扰动它必须逐位不变
    used = set()
    for i in range(d.zone_idx.shape[0]):
        used.update(int(z) for z in d.zone_idx[i][np.asarray(d.zone_mask[i]) > 0])
    unused = [n_base + z for z in range(n_zones) if (n_base + z) not in used]
    assert unused, "两条候选覆盖了全部区段——负判据取不到样本（换一条 R 决策/加大 k）"
    d3 = copy.deepcopy(d)
    d3.tok[unused[0]:unused[0] + 1] += delta
    assert torch.equal(base, decisions_logp([d3], pol).detach()), (
        f"扰动**没有候选经过**的区段（token {unused[0]}）却改变了 logp——"
        "R 头读了不属于任何候选的区段（掩码/池化写错）")


@pytest.mark.unit
def test_batched_recompute_carries_the_zone_readout():
    """批量重算（`decisions_logp`）与逐决策单条前向在 R2 下仍同源（≤1e-5）。

    批量打分入口 `logits_emb_batch` 的区段池化必须与逐决策的 `route_logits_emb` 同算式
    （组内右补 0 到最大区段数 + 掩码）。这条把"批量路径漏了区段"挡在门外。
    """
    inst, lay, dm, cfg, ctx = _setup()
    torch.manual_seed(9)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2,
                           route_zones=True)
    r_dec = [d for d in dec if d.kind == "R"][:20]
    assert r_dec, "没有 R 决策"
    batched = decisions_logp(r_dec, pol).detach()
    ref = []
    for d in r_dec:
        tok, _ = pol.forward_enc(torch.as_tensor(d.tok, dtype=torch.float32).unsqueeze(0), d.seg)
        logits = pol.route_logits_emb(
            tok, torch.as_tensor(d.feat, dtype=torch.float32).reshape(1, 1, -1),
            torch.as_tensor(d.cand_feat, dtype=torch.float32),
            torch.as_tensor(d.tok_idx, dtype=torch.long),
            torch.as_tensor(d.zone_idx, dtype=torch.long),
            torch.as_tensor(d.zone_mask, dtype=torch.float32))
        ref.append(torch.log_softmax(logits.flatten(), -1)[d.cand.index(d.action)])
    dt = float((batched - torch.stack(ref)).abs().max().detach())
    assert dt < 1e-5, f"R2 的批量与逐决策重算差 {dt}——批量路径的区段池化接错了"
