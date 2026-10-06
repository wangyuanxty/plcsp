# -*- coding: utf-8 -*-
"""⑩ 拼批头（B）——把「这趟带哪几件」从规则交还策略（`docs/mechanism-designs.md` §⑩）。

**决策点**：AGV 在取货点、头件已取走后。**动作**：候选 = **预构造的批次**
（`des.batch_cands` 的追加件数 s = 0..min(capacity−1, 同取货点任务数)，头件已占 1 位）。
s = 0 = 不拼批（只带头件 = 单件行为）。

**为什么选"预构造批次"而不是"逐任务二值决策"**（§45.4 的评估在新模型下重做）：
- 两组在"队列里没有同取货点任务"时都退化（候选 < 2 ⟹ 不记决策）；
- 同向组内的**非前缀子集被支配**（同一串载货段、更差的 FIFO 顺序）⟹ 逐任务版多出来的
  表达力没有对应收益；
- 逐任务版要引入"同批内先判谁"这一层**决策顺序**，与既有"一组候选一次打分"的骨架不同构，
  且链长上界变成"每个同取货点任务一个决策"；预构造版的链长上界 = 取货次数（MK10 实测
  38 次 B 决策 / 227 次取货，见 `progress-log.md` §47）。
  ⟹ **选预构造批次**；逐任务二值决策**不做**（登记为不做，不是待办）。

**判据**（与设计 §⑩ 的验收 + 本仓纪律一致）：
1. `batch_head=False`（默认）⟹ 不记 B 决策、链路逐位等于规则配置（黄金摘要在下）；
2. 候选特征三个槽（批件数 / 打乱顺序 / 距离节省）**逐候选不同**，且**只改候选特征分数会变**
   （§31 纪律：守卫必须能不通过——错误实现的变异检查写在注释里）；
3. 生产路径的 token 下标守卫：B 头读**决定方那台车**的 V token（不得把动作码当下标）；
4. 策略的选择**改变运输**（强制 s=0 vs s=max 两种配置读数不同）；
5. B 决策进链 logp、带梯度、同 seed 可复现；
6. 仿真侧的候选集与特征侧重构**同源**（`des.batch_cands` 唯一真相）。
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch

from plcsp.algo.group_rel import (BATCH_FEAT_CAND, BATCH_FEAT_DEC, _batch_cand_feat,
                                  _own_queue, _same_frm, batch_feat, chain_logp,
                                  decisions_logp, roll_chain)
from plcsp.algo.policy import PolicyNet, v_token_index
from plcsp.algo.setup import build_setup
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, SimWorld, batch_cands
from plcsp.env.instances import load_mk
from plcsp.env.snapshot import QueuedTask, Snapshot, VehicleState
from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_MAX, SEG_SLICE

# ── 1. 关态：不记 B 决策 + 规则配置链路指纹（**本批实现后捕获**，作回归钉子） ──
# 规则配置（`multi_drop=True` + `batch_head=False`）是**新路径**（改造前不存在），故这两条
# 摘要捕获自本批实现之后；它们的用途是钉住"以后动 ⑩ 不得悄悄改规则配置"。
#   kinds={'S':55,'L':65} n=120（mk01）｜{'S':115,'L':105} n=220（mk07）
# ⚠️ 2026-10-06 重捕：机床待机功率由 P^u 改为 Table 9 的 Standby Power（见 energy.py 模块 docstring）。
# 已实证决策 / makespan / travel / deliveries 逐位不变，只有 met[energy] 变。
BATCH_OFF_DIGEST = {
    "mk01": "da11e92e3c1fae1c043985e7420d1a070f655304a3e15031358b6475e2553425",
    "mk07": "8b80b0bdf58a086dc58a7259a0dc12cf65a7b5aab7addf6aab0c240345031827",
}


def _setup(name: str = "mk01", *, multi_drop: bool = True):
    inst = load_mk(name)
    cfg = SimConfig(multi_drop=multi_drop)
    lay, dm, ctx = build_setup(inst, cfg)
    return inst, lay, dm, cfg, ctx, PolicyNet(enc=LayoutEncoder())


def _chain_digest(dec, met) -> str:
    """链路指纹（同 `test_charge_head._chain_digest` 口径：含 tok_idx / 动作 / 采样 logp）。"""
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
    h.update(np.asarray([met["makespan"], met["travel_time_total"], float(met["energy"]),
                         float(met["deliveries"])], dtype=np.float64).tobytes())
    return h.hexdigest()


@pytest.mark.unit
@pytest.mark.parametrize("name", ["mk01", "mk07"])
def test_batch_head_off_records_no_b_decision_and_matches_the_rule_digest(name):
    """判据 1：`batch_head=False`（默认）⟹ 不记 B 决策，链路逐位等于规则配置。"""
    # ⚠️ 建策略的**抽签序**必须与捕获摘要的脚本逐字相同（先 seed → 再建唯一一个策略）：
    # `_setup` 会多建一个策略（丢弃），多消费一份初始化抽签 ⟹ 摘要必然不同。
    torch.manual_seed(20261005)
    inst = load_mk(name)
    cfg = SimConfig(multi_drop=True)
    lay, dm, ctx = build_setup(inst, cfg)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, batch_head=False)
    assert all(d.kind != "B" for d in dec), "关态不得出现 B 决策"
    assert _chain_digest(dec, met) == BATCH_OFF_DIGEST[name], \
        f"{name} 的规则配置链路变了——既有读数不再成立"


@pytest.mark.unit
def test_batch_head_requires_multi_drop_and_heterogeneous_fleet():
    """两条前置**显式报错**（不静默给死动作）：单卸货点模型 / ⑩ 关（载量恒 1）。"""
    torch.manual_seed(0)
    for cfg, cons, want in ((SimConfig(multi_drop=False), ConstraintConfig(), "multi_drop"),
                            (SimConfig(multi_drop=True),
                             ConstraintConfig().with_off("heterogeneous_fleet"),
                             "异构车队")):
        inst = load_mk("mk01")
        lay, dm, ctx = build_setup(inst, cfg, constraints=cons)
        with pytest.raises(ValueError, match=want):
            roll_chain(inst, lay, dm, cfg, PolicyNet(enc=LayoutEncoder()), seed=0, ctx=ctx,
                       constraints=cons, batch_head=True)


@pytest.mark.unit
def test_batch_feature_widths_match_the_head():
    """⚠️ `BATCH_FEAT_*`（`group_rel` 的宽度）与 B 头的输入宽度必须一致——漂开即静默错位。"""
    pol = PolicyNet(enc=LayoutEncoder())
    assert pol.b_head_tok[0].in_features == pol.enc.d_model + BATCH_FEAT_DEC + BATCH_FEAT_CAND


# ── 2. 候选特征：三个槽的语义 + "只改候选特征分数会变" ──

def _snap_with_queue(tasks, *, node=0, capacity=3):
    """手搓快照：一台车 + 队列里的若干任务（`QueuedTask`，veh=0 = 绑定配置）。"""
    v = VehicleState(status=0, node=node, queued=len(tasks), battery_frac=1.0,
                     capacity=capacity, speed_factor=1.0, zone_wait=0.0)
    return Snapshot(now=0.0, machines=(), jobs=(), vehicles=(v,), n_done=0, in_flight=len(tasks),
                    zone_holder=(), queued_tasks=tuple(tasks))


def _qt(job, frm, to, oi=0, veh=0):
    return QueuedTask(veh=veh, job=job, frm=frm, to=to, oi=oi)


@pytest.mark.unit
def test_batch_candidate_features_carry_the_three_designed_slots():
    """三个候选槽逐条钉：**批件数** / **打乱顺序** / **运输距离节省**（设计 §⑩ 的候选特征表）。"""
    _inst, lay, dm, cfg, ctx, _ = _setup()
    # 队列序：[同取货点(1) 的 A、别的取货点(2) 的 X、同取货点(1) 的 B]；头件 (frm=1 → to=0)
    snap = _snap_with_queue([_qt(1, 1, 2), _qt(2, 2, 3), _qt(3, 1, 3)])
    cands = batch_cands(2, 3)                     # = (0, 1, 2)：追加 0 / 1 / 2 件
    f = _batch_cand_feat(snap, lay, dm, ctx, 0, 1, 0, cands)
    assert f.shape == (3, BATCH_FEAT_CAND)
    # 0 批件数：1/3、2/3、3/3（头件已占 1 位）
    assert f[:, 0] == pytest.approx([1 / 3, 2 / 3, 1.0])
    # 1 打乱顺序：不拼 = 0；追加 A（队列第 0 位）也不越过任何件；追加 B 越过 X 一件
    assert f[:, 1] == pytest.approx([0.0, 0.0, 1.0 / ctx.max_queued])
    # 2 运输距离节省：**直接钉算术**（不靠实现的循环形状；口径 = 分送 − 拼批，见函数 docstring）
    def d(a, b):
        return float(dm[lay.machines[a].dock_node, lay.machines[b].dock_node])

    cur = lay.machines[1].dock_node               # 本车 node=0 ⟹ d_cur = dm[0, 取货点]
    d_cur = float(dm[0, cur])
    assert f[0, 2] == pytest.approx(0.0), "不拼批（s=0）谈不上节省，该槽必须恰为 0"
    sep1 = 2 * d_cur + d(1, 0) + d(1, 2)          # 两件各跑一趟
    bat1 = d_cur + d(1, 0) + d(0, 2)              # 一趟：取货点 → 0 号卸货点 → 2 号卸货点
    assert f[1, 2] == pytest.approx((sep1 - bat1) / ctx.bbox_diag)
    sep2 = 3 * d_cur + d(1, 0) + d(1, 2) + d(1, 3)
    bat2 = d_cur + d(1, 0) + d(0, 2) + d(2, 3)    # 卸货序 = 首次出现序（0 → 2 → 3）
    assert f[2, 2] == pytest.approx((sep2 - bat2) / ctx.bbox_diag)
    # 决策特征：头件身份 + 队列压力（同取货点 2 件 / 载量 3）
    df = batch_feat(_inst, snap, ctx, 0, job=9, frm=1, to=0, oi=0)
    assert len(df) == BATCH_FEAT_DEC
    assert df[4] == pytest.approx(3 / ctx.max_queued) and df[5] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_only_the_candidate_features_move_the_score():
    """判据 2（brief 的硬要求）：**只改候选特征 ⟹ 分数必须变**（该槽是活的，不是装饰）。

    ⚠️ 变异检查（写成守卫的理由）：若 `batch_logits_emb` 把 `feat_cand` 忽略（例如误接到
    `feat_batch` 上），本判据**当场失败**——`base` 与 `changed` 会逐位相同。
    """
    _inst, lay, dm, cfg, ctx, _ = _setup()
    snap = _snap_with_queue([_qt(1, 1, 2), _qt(2, 1, 3)])
    cf = _batch_cand_feat(snap, lay, dm, ctx, 0, 1, 0, batch_cands(2, 3))
    torch.manual_seed(3)
    pol = PolicyNet(enc=LayoutEncoder())
    tok_feat = torch.zeros(1, 2 + 3 + 1 + 1, F_MAX)
    seg = (2, 3, 1, 1)
    for start, n, key in ((0, 2, "M"), (2, 3, "B"), (6, 1, "G")):
        tok_feat[0, start:start + n, SEG_SLICE[key]] = torch.randn(n, SEG_SLICE[key].stop)
    tok, _ = pol.forward_enc(tok_feat, seg)
    idx = torch.full((len(cf),), v_token_index(seg)[0], dtype=torch.long)   # 本车 V token 广播
    base = pol.batch_logits_emb(tok, torch.zeros(1, 1, BATCH_FEAT_DEC), torch.as_tensor(cf), idx)
    changed = pol.batch_logits_emb(tok, torch.zeros(1, 1, BATCH_FEAT_DEC),
                                   torch.as_tensor(cf) + 0.5, idx)
    assert base.shape == (1, 1, len(cf)), f"B 头输出形状应为 (1,1,n_cand)：{tuple(base.shape)}"
    assert not torch.allclose(base, changed), "只改候选特征打分却不变——该槽是死的"


# ── 3. 生产路径的 token 下标守卫 ──

class _NoMixEncoder(torch.nn.Module):
    """逐 token **恒等**嵌入（无跨 token 注意力）——隔离"B 头索引了哪一行"（同 §31 的纪律）。"""

    d_model = F_MAX

    def forward(self, x, seg, bias=None):
        return x, x.mean(dim=1)


@pytest.mark.unit
def test_b_head_reads_the_deciding_agvs_token_on_the_production_path():
    """⚠️ B 头必须读**决定方那台车**的 V 段 token——不得把候选动作码（s）当下标。

    缺陷形态：`_act` 若照 S 头写 `tok_idx = cand`，动作码 {0,1,2} 会被当成序列位置，B 头去读
    **0/1/2 行机台**的 token（与 §31 的 L 头缺陷、`test_charge_head` 的 C 头守卫同型）。
    守卫走 `roll_chain`（生产路径）→ 挑一条**由 aid≥1 的车**做出的 B 决策 → 只扰动该车 V 行，
    `decisions_logp` 必须变；扰动别的车必须逐位不变。
    ⚠️ 用 mk10（B 决策多、含 aid≥1）——mk01 只有 1 次 B 决策（默认布局只有 0 号车载量 2）。
    """
    torch.manual_seed(7)
    inst, lay, dm, cfg, ctx, _ = _setup("mk10")
    pol = PolicyNet(enc=_NoMixEncoder())
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, batch_head=True)
    b_dec = [d for d in dec if d.kind == "B"]
    assert len(b_dec) >= 10, f"B 决策太少（{len(b_dec)}）——判据失去意义"
    v_rows = v_token_index(b_dec[0].seg)
    d = next((x for x in b_dec if x.agv is not None and x.agv >= 1), None)
    assert d is not None, "链路里没有 aid≥1 的车做出的 B 决策——守卫区分不出错下标"
    aid = int(d.agv)
    assert np.array_equal(d.tok_idx, np.full(len(d.cand), v_rows[aid], dtype=np.int64)), (
        f"B 决策记录的 token 下标 {d.tok_idx.tolist()} 应全部指向 {aid} 号车的 V token"
        f"（下标 {v_rows[aid]}）——候选是动作码，不是序列位置")
    base = decisions_logp([d], pol).detach().clone()
    delta = np.zeros((1, F_MAX), dtype=np.float32)
    delta[0, :3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)     # 逐列不同，非均匀平移
    d2 = copy.deepcopy(d)
    d2.tok[v_rows[aid]:v_rows[aid] + 1] += delta
    assert not torch.equal(base, decisions_logp([d2], pol).detach()), \
        f"扰动 {aid} 号车（决定方）的 V token 后 B 决策的 logp 不变——B 头读的不是这台车"
    other = 0 if aid != 0 else 1
    d3 = copy.deepcopy(d)
    d3.tok[v_rows[other]:v_rows[other] + 1] += delta
    assert torch.equal(base, decisions_logp([d3], pol).detach()), \
        f"扰动 {other} 号车（非决定方）却改变了 aid={aid} 的 B 决策——B 头读了不相干的 token"


# ── 4. 仿真侧与特征侧同源 ──

@pytest.mark.unit
def test_sim_and_feature_side_agree_on_the_candidate_set():
    """候选集**唯一真相**：`des.batch_cands`。仿真侧给回调的 `cands` 必须与"按快照重构"的一致。

    重构 = `_own_queue(snap, aid)` 里同取货点的件数 → `batch_cands(·, capacity)`。两边漂开
    （如仿真看队列、特征看别的车）会让策略给 A 批打分、实际取走 B 批——链 logp 与动作错位。
    ⚠️ 判据在**回调内部**取（回调同时拿到仿真给的 `cands` 与"当时"的快照），故它测的正是
    "生产路径上两侧是否同源"，不是手搓的两份数据。
    """
    inst, lay, dm, cfg, _ctx, _ = _setup("mk10")
    calls: list = []

    def policy_b(snap, aid, job, frm, to, oi, cands):
        n_same = len(_same_frm(_own_queue(snap, aid), frm))
        calls.append((int(aid), tuple(int(c) for c in cands), n_same,
                      int(snap.vehicles[aid].capacity)))
        return int(cands[-1])                      # 强制取最大批

    met = SimWorld(inst, lay, dm, cfg).run_gated(seed_chain=1, policy_b=policy_b)
    assert calls, "⑩ 的回调一次都没被调用——判据失去意义"
    for aid, cands, n_same, cap in calls:
        assert cands == batch_cands(n_same, cap), (
            f"仿真给的候选 {cands} ≠ 按快照重构的 {batch_cands(n_same, cap)}"
            f"（aid={aid}，同取货点 {n_same} 件，载量 {cap}）")
    assert met["batch_ge2"] > 0, "强制取最大批却一件都没拼上——候选集接错了"


# ── 5. 策略的选择改变运输（验收判据 ③） ──

@pytest.mark.unit
def test_batch_choice_changes_transport():
    """验收 ③：**同一个世界、同一 seed**，强制 s=0（不拼）与 s=max（拼满）两种配置读数不同。

    这是"决策点真的在运输动力学上生效"的直接判据（不是只看 logp）。
    """
    inst = load_mk("mk10")
    cfg = SimConfig(multi_drop=True)
    lay, dm, _ctx = build_setup(inst, cfg)
    out = {}
    for tag, fn in (("zero", lambda sn, a, j, f, t, o, cs: 0),
                    ("max", lambda sn, a, j, f, t, o, cs: int(cs[-1]))):
        w = SimWorld(inst, lay, dm, cfg)
        met = w.run_gated(seed_chain=1, policy_b=fn)
        out[tag] = met
    avg = {k: v["batch_items"] / max(v["batch_trips"], 1) for k, v in out.items()}
    assert out["zero"]["batch_ge2"] == 0, "强制不拼却出现了 >1 件的批次——策略没接上"
    assert avg["max"] > avg["zero"], \
        "强制拼满与强制不拼的**平均每趟件数**没有差别——决策点没接上动力学"
    assert out["max"]["batch_ge2"] > 0, "强制拼满没有产生 >1 件的批次——拼批没有发生"
    assert out["max"]["batch_trips"] < out["zero"]["batch_trips"], \
        "强制拼满没有减少取货次数——拼批没有发生"
    assert (out["max"]["trips"], out["max"]["travel_time_total"]) != \
        (out["zero"]["trips"], out["zero"]["travel_time_total"]), "两种配置的运输读数相同"


# ── 6. 进链、有梯度、可复现 ──

@pytest.mark.unit
def test_b_decisions_enter_the_chain_with_grad_and_are_reproducible():
    """判据 5：B 决策进链 logp（带梯度到编码器），且同 seed 逐位可复现。"""
    torch.manual_seed(20261005)
    inst, lay, dm, cfg, ctx, _ = _setup("mk07")
    rng_state = torch.get_rng_state()          # 复现对照从**同一抽签点**起（见文末 pol2）
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, batch_head=True)
    b_dec = [d for d in dec if d.kind == "B"]
    assert b_dec and met["batch_ge2"] >= 1, "mk07 上应有拼批发生（存在性）"
    lp = chain_logp(dec, pol)
    assert lp.requires_grad, "链 logp 必须带梯度（B 决策要能反向到编码器）"
    lp.backward()
    grads = [p.grad for p in pol.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "反向没有产生任何梯度"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _n, p in
               pol.named_parameters() if _n.startswith("b_head_tok")), \
        "B 头自己没有梯度——它不在链上"
    # 同 seed 复现（新策略实例、**同一抽签点**）
    torch.set_rng_state(rng_state)
    pol2 = PolicyNet(enc=LayoutEncoder())
    dec2, _ = roll_chain(inst, lay, dm, cfg, pol2, seed=0, ctx=ctx, batch_head=True)
    assert [(d.kind, d.action) for d in dec] == [(d.kind, d.action) for d in dec2]
    b2 = [d for d in dec2 if d.kind == "B"]
    assert len(b_dec) == len(b2), "同 seed 的 B 决策条数不同"
    for x, y in zip(b_dec, b2):                       # 候选数可不同 ⟹ 逐条比，不 stack
        assert x.cand == y.cand
        np.testing.assert_array_equal(x.cand_feat, y.cand_feat)
        np.testing.assert_array_equal(x.feat, y.feat)


@pytest.mark.unit
def test_batch_head_on_default_config_produces_multi_item_batches():
    """存在性（拼批头配置）：默认配置（`n_agv=3`、`SimConfig(multi_drop=True)`）**真的出现** >1 件的批次。

    ⚠️ 与 `test_multidrop_transport` 的规则配置判据**不同**：这里策略可能主动选 s=0（不拼），
    故读数 ≤ 规则配置。MK01 只有 1 次候选（默认布局只有 0 号车载量 2）⟹ 本判据用 mk07/mk10。
    """
    for name in ("mk07", "mk10"):
        torch.manual_seed(20261005)
        inst, lay, dm, cfg, ctx, _ = _setup(name)
        pol = PolicyNet(enc=LayoutEncoder())
        _dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, batch_head=True)
        assert met["batch_ge2"] >= 1, f"{name}：拼批头配置没有出现 >1 件的批次"
