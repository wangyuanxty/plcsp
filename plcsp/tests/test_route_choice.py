"""路线决策（R 头）的测试——恢复被砍掉的 `route_logits`（progress-log §27.3/§28）。

背景（必须保留在测试里，防"再次被砍"）：`route_logits` 当初随 ① 拥堵一并取消，理由是
"无拥堵时选远路严格更差"（spec §5.3）。但 ① 后来在 `eb1d1da` 恢复、路线头**没跟着恢复**
（见 `docs/progress-log.md` §27.3、§28）。本文件的判据即那次遗漏的验收：

1. **关掉路线头 ⟹ 逐位等于今日**（黄金摘要钉死）——所有既有读数立在这条上；
2. **候选特征必须让选择非平凡**：仅改变区段争用，两条候选的分数必须不同；
3. **k 条候选必须真的互不相同**、按长度升序、不足 k 时优雅退化。
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import pytest

from plcsp.algo.group_rel import (chain_logp, decisions_logp, roll_chain,
                                  sampled_decisions_logp)
from plcsp.algo.policy import PolicyNet, v_token_index
from plcsp.algo.setup import build_layout_and_dm, build_setup
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.corridors import build_corridor_graph, k_shortest_paths, shortest_node_path
from plcsp.env.des import SimConfig, SimWorld
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.mkt import load_mkt
from plcsp.env.snapshot import Snapshot
from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_MAX, SEG_SLICE

# 黄金摘要：在 MK01 上跑出的链路指纹，把"路线头关闭 ⟹ 逐位等于既有行为"钉成机器可判的
# 判据——任何对 `_drive`/`roll_chain` 的改动若在关闭档下动了行为，此处立刻变红（其余测试
# 都盖不住"多抽了一个随机数"或"路径选法变了"这类静默漂移）。
# ⚠️ 2026-10-04 重捕获（L 头 token 下标修复，不是路线头改动）：旧值
# `403f68e3ba380857e14a94ada025d667a2cc7e0790abef755ca2845f371c31b0` 钉的是带缺陷的链路
# ——`_act` 把 L 的**车号**当序列位置，L 头读 M 段机台 token、对车辆特征完全失明；修复后
# L 的分数与采样动作都变，摘要必须换新基准。路线头关闭档本身的"逐位稳定"仍由本测试守着。
GOLDEN_CHAIN_DIGEST = "e67a71292fe33c16e64cdcfdc6a6e8104cd06c546a3c4377ca1b9eb9dab55f79"


def _setup(name="mk01"):
    """(inst, layout, dm, cfg, ctx, policy)——与 `roll_chain` 的签名对齐（同 test_joint_chain）。"""
    inst = load_mk(name)
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    pol = PolicyNet(enc=LayoutEncoder())
    return inst, lay, dm, cfg, ctx, pol


def _chain_digest(dec, met) -> str:
    """链路指纹：决策类型 + 全部打分上下文 + 动作 + 采样 logp + 关键指标。"""
    h = hashlib.sha256()
    for d in dec:
        h.update(d.kind.encode())
        h.update(np.ascontiguousarray(d.tok, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(d.feat, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(d.cand_feat, dtype=np.float32).tobytes())
        h.update(np.asarray(d.cand, dtype=np.int64).tobytes())
        h.update(np.asarray([d.action], dtype=np.int64).tobytes())
        h.update(np.asarray([d.logp], dtype=np.float32).tobytes())
    h.update(np.asarray([met["makespan"], met["travel_time_total"],
                         float(met["energy"]), float(met["deliveries"])],
                        dtype=np.float64).tobytes())
    return h.hexdigest()


def _enc_inputs(n_m=6, n_b=10, n_v=3):
    """(1, N, F_MAX) 的合法 token 张量（补零列恒 0，见编码器的快速失败守卫）。"""
    t = torch.zeros(1, n_m + n_b + n_v + 1, F_MAX)
    for start, n, key in ((0, n_m, "M"), (n_m, n_b, "B"), (n_m + n_b, n_v, "V"),
                          (n_m + n_b + n_v, 1, "G")):
        t[0, start:start + n, SEG_SLICE[key]] = torch.randn(n, SEG_SLICE[key].stop)
    return t, (n_m, n_b, n_v, 1)


# ────────────────────────── 1. k 条候选路径 ──────────────────────────

@pytest.mark.unit
def test_k_shortest_paths_are_distinct_simple_and_ascending():
    """k 条候选必须**互不相同**、都是简单路径、按长度升序（最短在前）。

    "候选都长一个样"是 R 头最隐蔽的失效形态：动作空间退化成单点，策略学不到任何东西，
    而链路里仍会多出一堆 logp ≡ 0 的假决策。故这里逐条核对**路径本身**。
    ⚠️ k=5 而非 3：网格上曼哈顿等长的路很多（实测 mk01 的前 3 条全是 12.9 m），
    取到第 5 条才出现严格更长的绕行——"升序"要连**严格变大**一起测，否则全等长也蒙混过关。
    """
    lay = sample_layout(6, seed=0)
    g = build_corridor_graph(lay)
    src, dst = lay.machines[0].dock_node, lay.machines[5].dock_node
    paths = k_shortest_paths(g, src, dst, 5)
    assert len(paths) == 5, f"网格上应有 5 条不同路径，实得 {len(paths)}"
    assert paths[0] == shortest_node_path(g, src, dst), "第 0 条必须是最短路（关闭档的基准）"
    seen = {tuple(p) for p in paths}
    assert len(seen) == len(paths), f"候选路径有重复：{paths}"
    for p in paths:
        assert p[0] == src and p[-1] == dst, f"端点不对：{p}"
        assert len(set(p)) == len(p), f"不是简单路径（含环）：{p}"
        assert all(g.has_edge(u, v) for u, v in zip(p, p[1:])), f"路径含不存在的边：{p}"
    lens = [float(sum(g[u][v]["weight"] for u, v in zip(p, p[1:]))) for p in paths]
    assert lens == sorted(lens), f"未按长度升序：{lens}"
    assert lens[0] < lens[-1], f"5 条候选应含严格更长的绕行：{lens}"


@pytest.mark.unit
def test_k_shortest_paths_degrades_gracefully():
    """候选不足 k 时返回实际条数（不得抛错、不得凑数）。

    ⚠️ 用小图（三节点的**链** 0-1-2）测退化：本仓的格点图上几乎处处有绕行
    （连相邻节点都有 0→4→5→1 这种），只有链上"桥边"两侧才只剩一条简单路径。
    """
    import networkx as nx
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    assert k_shortest_paths(g, 0, 2, 3) == [[0, 1, 2]], "链上只有一条简单路径"
    assert k_shortest_paths(g, 0, 1, 3) == [[0, 1]], "相邻节点只有直连这一条简单路径"


# ────────────────────────── 2. 候选择优特征 ──────────────────────────

def _cycle_snap_and_zof():
    """4 节点环 0-1-2-3-0（单位边）+ 手搓快照：两条**等长**候选 0-1-2 与 0-3-2。

    两条候选的长度、区段数完全相同，唯一差别是 3 号区段**被别的车占着**。
    这正是判据 2 要的形状：只改争用，不改长度。
    """
    from plcsp.algo.group_rel import _route_cand_feat          # 被测函数（模块私有，测试直调）
    n = 4
    dm = np.full((n, n), np.inf)
    for i in range(n):
        dm[i, i] = 0.0
        dm[i, (i + 1) % n] = dm[(i + 1) % n, i] = 1.0
    # 环上的两两最短路（0↔2 两条都是 2）
    dm[0, 2] = dm[2, 0] = 2.0
    dm[1, 3] = dm[3, 1] = 2.0
    zof = {i: i for i in range(n)}                 # node 粒度：节点号 = 区段号
    snap = Snapshot(now=0.0, machines=(), jobs=(), vehicles=(),
                    n_done=0, in_flight=0, zone_holder=(-1, -1, -1, 3))
    return _route_cand_feat, snap, zof, dm


@pytest.mark.unit
def test_route_candidate_features_read_zone_contention():
    """判据 2（特征层）：两条**等长**候选，仅区段争用不同 ⟹ 候选特征必须不同，且差异只落在争用维。

    ⚠️ 假守卫的形状：候选特征只放长度 ⟹ 两条候选同分，R 头退化成"永远选最短路"的常数策略，
    链路里那堆 R 决策全是空转。故本判据同时钉住"长度维相同"与"争用维不同"。
    """
    feat_fn, snap, zof, dm = _cycle_snap_and_zof()
    cands = [(0, 1, 2), (0, 3, 2)]                 # 等长 2；后者经过被别车占着的 3 号区段
    f = feat_fn(snap, zof, len(zof), cands, dm, aid=0)
    assert f.shape == (2, 3), f"R 候选特征形状应为 (k, 3)，实得 {f.shape}"
    assert f[0, 0] == pytest.approx(1.0) and f[1, 0] == pytest.approx(1.0), \
        f"两条候选等长，归一化长度维应同为 1.0：{f[:, 0]}"
    assert f[0, 1] == pytest.approx(f[1, 1]), f"区段数应相同：{f[:, 1]}"
    assert f[0, 2] == pytest.approx(0.0), f"0 号候选一路畅通，争用维应为 0：{f[:, 2]}"
    assert f[1, 2] > f[0, 2], \
        f"被别的车占着的候选（1 号）争用维必须更大：{f[:, 2]}"
    # 自己占着不算争用：aid=3 时 3 号区段是本车自己的，争用维回到 0
    g = feat_fn(snap, zof, len(zof), cands, dm, aid=3)
    assert g[1, 2] == pytest.approx(0.0), f"自己持的区段不是争用：{g[:, 2]}"


@pytest.mark.unit
def test_route_score_changes_when_only_contention_differs():
    """判据 2（网络层）：**扰动过的**策略下，仅争用不同的两条候选必须得到不同分数。

    这条把"特征里有争用"升级成"争用真的影响分数"。只测特征不看打分，接线断了也照样绿
    （特征算对了，但头没读它）。
    """
    torch.manual_seed(11)
    pol = PolicyNet(enc=LayoutEncoder())
    with torch.no_grad():                       # 显式扰动：不依赖某个恰好敏感的初始化
        for p in pol.r_head_tok.parameters():
            p.add_(0.1 * torch.randn_like(p))
    tok_feat, seg = _enc_inputs()
    tok, _ = pol.forward_enc(tok_feat, seg)
    v_idx = v_token_index(seg)[0]
    idx = torch.tensor([v_idx, v_idx])
    drive = torch.zeros(1, 1, 4)
    # 两条候选：长度维与区段数维相同，只有争用维不同（同 `_cycle_snap_and_zof` 的口径）
    cand = torch.tensor([[[1.0, 0.5, 0.0], [1.0, 0.5, 1.0]]])
    logits = pol.route_logits_emb(tok, drive, cand, idx)
    assert logits.shape == (1, 1, 2), f"R 头输出形状应为 (1,1,k)，实得 {tuple(logits.shape)}"
    d = float((logits[0, 0, 1] - logits[0, 0, 0]).item())
    assert abs(d) > 1e-8, "仅争用不同却同分——争用信号没有到达打分"


@pytest.mark.unit
def test_route_head_gradient_reaches_encoder():
    """R 头必须与 S/L 同走编码器（否则"路线也是策略的一部分"是名义上的）。"""
    torch.manual_seed(2)
    pol = PolicyNet(enc=LayoutEncoder())
    tok_feat, seg = _enc_inputs()
    tok, _ = pol.forward_enc(tok_feat, seg)
    v_idx = v_token_index(seg)[0]
    out = pol.route_logits_emb(tok, torch.zeros(1, 1, 4),
                               torch.ones(1, 2, 3), torch.tensor([v_idx, v_idx])).sum()
    out.backward()
    g = pol.enc.proj[2].weight.grad      # V 段投影（R 头读本车 V token）
    assert g is not None and g.abs().sum() > 0, "R 头梯度没到编码器（V 段投影）"


@pytest.mark.unit
def test_route_feature_widths_match_the_head():
    """⚠️ `ROUTE_FEAT_*`（`group_rel` 的特征宽度）与 R 头的输入宽度必须一致。

    两处分别在 `group_rel` 与 `PolicyNet` 的默认形参里，漂开就是静默错位（打分头的
    第一层会把决策特征与候选特征切在错误的列上）——用断言把两处钉在一起。
    """
    from plcsp.algo.group_rel import ROUTE_FEAT_CAND, ROUTE_FEAT_DRIVE
    pol = PolicyNet(enc=LayoutEncoder())
    assert pol.r_head_tok[0].in_features == pol.enc.d_model + ROUTE_FEAT_DRIVE + ROUTE_FEAT_CAND


# ────────────────────────── 3. 链集成 ──────────────────────────

@pytest.mark.unit
def test_route_decisions_join_the_chain_and_its_logp():
    """判据 5：R 决策必须进链、且**进链 logp**——否则"策略选路"只是名义。

    ⚠️ 只断言"链里出现 R"会漏掉"logp 里没有 R"的形态：那些决策就成了不产生梯度的装饰。
    故这里**抽掉 R 再算链 logp**，值必须变。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, route_k=2)
    kinds = [d.kind for d in dec]
    n_r = kinds.count("R")
    assert n_r >= 10, f"R 决策太少（{n_r}）——判据失去意义"
    assert any(len(d.cand) >= 2 for d in dec if d.kind == "R"), \
        "没有任何 R 决策带 2 条候选——候选集退化了"
    r_dec = [d for d in dec if d.kind == "R"]
    assert all(d.cand == tuple(range(len(d.cand))) for d in r_dec), \
        "R 决策的候选应是 0..k-1 的序号（路径没有 token，动作即候选序号）"
    assert all(0 <= d.action < len(d.cand) for d in r_dec)
    vec = decisions_logp(dec, pol)
    assert vec.shape == (len(dec),), f"逐决策 logp 未覆盖 R：{tuple(vec.shape)}"
    old = sampled_decisions_logp(dec)
    assert vec.tolist() == old.tolist(), "含 R 后逐决策 logp 的采样回放与重算不再逐位同源"
    lp_all = float(chain_logp(dec, pol).detach())
    lp_no_r = float(chain_logp([d for d in dec if d.kind != "R"], pol).detach())
    assert lp_all != lp_no_r, "R 决策没有进链 logp——路线头是名义上的"


@pytest.mark.unit
def test_route_enabled_chain_is_reproducible_and_differs():
    """同 seed 下开路线头仍逐位可复现；且与关闭档**不同**（R 真的进了链路）。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    a, met_a = roll_chain(inst, lay, dm, cfg, pol, seed=3, ctx=ctx, route_k=2)
    b, _ = roll_chain(inst, lay, dm, cfg, pol, seed=3, ctx=ctx, route_k=2)
    sig = lambda dec: [(d.kind, d.cand, d.action) for d in dec]      # noqa: E731
    assert sig(a) == sig(b), "开路线头后同 seed 不再可复现（R 的采样流没接上 seed）"
    base, met_base = roll_chain(inst, lay, dm, cfg, pol, seed=3, ctx=ctx)
    assert sig(a) != sig(base), "开/关路线头的链完全相同——R 是名义决策"
    assert met_a["travel_time_total"] != met_base["travel_time_total"], \
        "R 决策没有改变任何行驶结果"


@pytest.mark.unit
def test_route_head_off_is_bit_identical_to_baseline():
    """判据 1：路线头**关闭**（默认 `route_k=1`）时链路逐位等于既有行为——黄金摘要钉死。

    摘要含每条决策的 tok/feat/cand_feat/cand/action/logp 与 makespan/travel/energy。
    任何"多抽一个随机数、路径选法变了、快照多算了一个量"的漂移都会翻红。
    ⚠️ 2026-10-04 重捕获：L 头 token 下标修复（`_act` 车号→V 段 token）改变的是 **L 决策的
    输入**，采样动作与指标随之变化，故摘要换新基准。旧基准 `403f68e3...` 钉的是带缺陷的
    链路（L 头读机台 token、看不见任何车辆特征）。本测试的**判据**（关闭档逐位稳定）不变。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    torch.manual_seed(20261004)                 # 与黄金摘要生成时同一初始化
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    assert [d.kind for d in dec].count("R") == 0, "默认档不得产生 R 决策"
    assert _chain_digest(dec, met) == GOLDEN_CHAIN_DIGEST, \
        "关闭路线头后链路变了——既有读数不再成立"


@pytest.mark.unit
def test_route_head_works_on_the_matrix_caliber():
    """矩阵口径（MKT）下 R 同样成立：候选/争用仍由区段给出，行程时长仍按矩阵分摊。

    ⚠️ 矩阵口径的绕行**不改公式**：逐段时长 = 矩阵整段时长 × 该段几何 ÷ 最短路几何，
    绕行的总时长因此自动按几何比变长（见 `AgvSim._drive` 的"矩阵口径"段）。本判据要求
    "跑得通、有 R 决策、不掐表"——把矩阵分支被路线改动打坏（如时长按几何口径重算）挡在门外。
    ⚠️ ⑪ 充电**必须关**：矩阵只覆盖机台与装卸站，**充电桩没有矩阵项**——⑪ 开着时
    `_leg_min` 按 `transport_unmapped='raise'` 显式报错。这是既有口径（同
    `test_transport_wiring._exact_caliber_constraints`），与本机制无关。
    """
    inst = load_mkt("mk01").base
    cfg = SimConfig()
    cons = ConstraintConfig().with_off("charging")
    lay, dm, ctx = build_setup(inst, cfg, constraints=cons)
    torch.manual_seed(5)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx,
                          constraints=cons, route_k=2)
    assert any(d.kind == "R" for d in dec), "矩阵口径下没有产生 R 决策"
    assert met["horizon_hit"] is False and met["jobs_done"] == inst.n_jobs, \
        f"矩阵口径 + 路线头跑不完：horizon_hit={met['horizon_hit']}，jobs={met['jobs_done']}"


@pytest.mark.unit
def test_route_choices_actually_steer_the_agv():
    """端到端：`policy_r` 真的决定 AGV 走哪条路（不是记录了一个没人读的决策）。

    判据 = 同一局、同一派车规则下，强制"永远选最长的候选"必须让行驶时长**变大**——
    绕行在时间上有代价。这条盖住"决策记了但 `_drive` 没用它"的形态。
    """
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, dm = build_layout_and_dm(inst, cfg)
    pick_first = lambda snap, job, frm, to, oi, cand: cand[0]        # noqa: E731
    base = SimWorld(inst, lay, dm, cfg).run_gated(seed_chain=1, policy_l=pick_first)
    far = SimWorld(inst, lay, dm, cfg).run_gated(
        seed_chain=1, policy_l=pick_first,
        policy_r=lambda snap, aid, src, dst, leg, cands: len(cands) - 1)
    assert far["travel_time_total"] > base["travel_time_total"], (
        f"强制绕行后行驶时长未变：base={base['travel_time_total']:.3f} "
        f"far={far['travel_time_total']:.3f}——R 决策没有被 `_drive` 消费")


@pytest.mark.unit
def test_route_choice_requires_congestion():
    """① 拥堵关闭时不得启用路线头——**显式报错**，不静默退化。

    理由正是当初砍掉路线头的那句：路线只为绕开拥堵而存在；① 关时选远路严格更差，
    此时给一个"可以选"的动作头等于给策略一个**死决策**（还会污染链 logp）。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    off = ConstraintConfig(congestion=False)
    lay2, dm2, ctx2 = build_setup(inst, cfg, constraints=off)
    with pytest.raises(ValueError, match="拥堵"):
        roll_chain(inst, lay2, dm2, cfg, pol, seed=0, ctx=ctx2,
                   constraints=off, route_k=2)


# ────────────────────────── 4. 快照带区段占用 ──────────────────────────

@pytest.mark.unit
def test_snapshot_reports_zone_holders():
    """路线候选的争用信号来自快照：`zone_holder[z]` = 持有该区段的 AGV 号，-1 = 空闲。"""
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, dm = build_layout_and_dm(inst, cfg)
    w = SimWorld(inst, lay, dm, cfg)
    assert len(w.snapshot().zone_holder) == w.zm.n, "区段表长度必须等于区段数"
    w.zm.holder[0] = 2
    snap = w.snapshot()
    assert snap.zone_holder[0] == 2 and snap.zone_holder[1] == -1, \
        f"区段占用没有进快照：{snap.zone_holder[:4]}"
    w.zm.holder[0] = None
    assert w.snapshot().zone_holder[0] == -1, "释放后应回到 -1（空闲）"
