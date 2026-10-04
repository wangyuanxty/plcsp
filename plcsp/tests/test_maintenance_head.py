"""⑫ 维护头（M）——把「何时停机保养」从规则交还策略（`docs/mechanism-designs.md` §⑫）。

**规则 → 策略**：`MachineSim` 现在在一道工序加工完毕时自动检查 `pm_clock >= pm_interval`
（默认 120 min 主轴工时）并强制停机 `pm_duration`；策略只能**看见** `pm_left`（M 段特征），
不能**动作**。本机制新增 M 决策：机台在**两件之间**（一道工序加工完毕、下一件尚未上机）
问策略 {现在保养, 不保养}，逐候选打分与 S/L/R 同构。

**判据**（与设计文档 §⑫ 的验收一致）：
1. `pm_head=False`（默认）⟹ 链路逐位等于今日（黄金摘要钉死，既有读数全靠它）；
2. 短间隔档下 M 决策**真的发生**、进链 logp、梯度到 M 头、同 seed 可复现；
3. 动作**有效果**：强制「永远现在保养」把首次保养提前到规则不可能达到的时刻；
4. **规则仍是硬底线**：逾期（`pm_clock >= pm_interval`）**强制**保养、不产生决策——策略
   只能把保养提前，不能推迟过强制点（设计 §⑫「推迟：强制停机落在更晚」）。

⚠️ **守卫必须走生产路径、不得自己提供下标**（2026-10-04 L 头缺陷的教训，progress-log §31）：
M 的两个候选同属**一台**机台（动作码 0/1），若照 S 头写 `tok[0, cand]`，候选 0/1 会去读
0/1 号机台的 token——维护头评的不是决定保养的那台机。守卫经 `roll_chain` → `decisions_logp`
重算路径，并用逐 token **恒等**嵌入的哑编码器**切断注意力泄漏**（真编码器下扰动任一行都会
经注意力改变所有 token 的上下文嵌入，判据被掩盖：§31.2 的实测 −1.0927 → −1.0873）。

⚠️ **默认参数不动**（§27.1 纪律一）：MK01 每机负载仅 ~25.5 主轴分钟 < 120，PM **从不触发**
（`test_constraints_production` 已记）。本文件一律用**显式短间隔** `SimConfig(pm_interval=10)`
验证——改默认会为第二个不必要的理由作废全部读数。
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
import pytest

from plcsp.algo.group_rel import (PM_FEAT_CAND, PM_FEAT_DEC, chain_logp,
                                  decisions_logp, roll_chain, sampled_decisions_logp)
from plcsp.algo.policy import PolicyNet
from plcsp.algo.setup import build_layout_and_dm, build_setup
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import PM_CAND_DEFER, PM_CAND_NOW, PM_CANDS, SimConfig, SimWorld
from plcsp.env.instances import load_mk
from plcsp.env.snapshot import MachineState, Snapshot
from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_MAX, SEG_SLICE

# 验证用的**短间隔档**（默认仍是 120.0，见模块 docstring）：MK01 机台负载 ~25.5 min，
# 10 min 间隔下每台机在整个 episode 里都会被强制保养 1–2 次。
PM_SHORT = 10.0

# 黄金摘要：`pm_head=False` 档的链路指纹（决策类型 + 全部打分上下文 + tok_idx + 动作 +
# 采样 logp + 关键指标）。**捕获自改造前的树**：先在改造前的工作树（HEAD=39f095f）上算出，
# 改造后再用 `git worktree` 在当时的 HEAD（9bbcc1e，期间只有文档提交推进、代码同 39f095f）
# 复算一次，两处同值——它不是实现完之后现编的基准：
#   kinds={'S':55,'L':65,'R':0} n=120，makespan=118.8647282376667
# 任何"多抽一个随机数 / 多取一次快照 / 下标映射变了"的关闭档漂移都会翻红。
PM_HEAD_OFF_DIGEST = "5a06255d6d57d2f507c7186a198c11ec056454a359265809c80da40c98f4c5ae"


def _setup(interval=PM_SHORT):
    """(inst, layout, dm, cfg, ctx, policy)——与 `roll_chain` 的签名对齐（同 test_route_choice）。

    `interval=None` ⇒ **默认** `SimConfig()`（`pm_interval=120`）——黄金摘要钉的是默认档。
    """
    inst = load_mk("mk01")
    cfg = SimConfig() if interval is None else SimConfig(pm_interval=interval)
    lay, dm, ctx = build_setup(inst, cfg)
    pol = PolicyNet(enc=LayoutEncoder())
    return inst, lay, dm, cfg, ctx, pol


def _chain_digest(dec, met) -> str:
    """链路指纹（含 `tok_idx`——它把"头读哪一行"也钉进摘要）。"""
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


def _enc_inputs(n_m=4, n_b=5, n_v=2):
    """(1, N, F_MAX) 的合法 token 张量（补零列恒 0，见编码器的快速失败守卫）。"""
    t = torch.zeros(1, n_m + n_b + n_v + 1, F_MAX)
    for start, n, key in ((0, n_m, "M"), (n_m, n_b, "B"), (n_m + n_b, n_v, "V"),
                          (n_m + n_b + n_v, 1, "G")):
        t[0, start:start + n, SEG_SLICE[key]] = torch.randn(n, SEG_SLICE[key].stop)
    return t, (n_m, n_b, n_v, 1)


class _NoMixEncoder(torch.nn.Module):
    """逐 token **恒等**嵌入（无跨 token 注意力）——只为隔离"M 头索引了哪一行"这一件事。

    真编码器是全连接注意力：扰动 M 段某一行会经注意力改变其余 token 的上下文嵌入，
    "读错行的头"也跟着变——判据被掩盖（§31.2 实测）。逐 token 恒等嵌入切断这条泄漏后，
    "分数依赖哪一行"才是可判的量，而 `_act` / `decisions_logp` 仍走生产代码。
    `d_model = F_MAX` 使 `PolicyNet` 按 11 维 token 建头。
    """
    d_model = F_MAX

    def forward(self, x, seg, bias=None):
        return x, x.mean(dim=1)


# ────────────────────────── 1. 关闭档逐位不变（摘要在前文） ──────────────────────────

@pytest.mark.unit
def test_pm_head_off_is_bit_identical_to_baseline():
    """判据 1：`pm_head=False`（默认）⟹ 链路逐位等于改造前——黄金摘要钉死。

    摘要含每条决策的 tok/feat/cand_feat/cand/**tok_idx**/action/logp 与
    makespan/travel/energy/deliveries。这条是所有既有读数成立的前提。
    """
    inst, lay, dm, cfg, ctx, _ = _setup(interval=None)   # 黄金摘要锚在**默认** cfg 上
    torch.manual_seed(20261004)                 # 与捕获脚本逐字对齐的初始化锚点
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, pm_head=False)
    assert [d.kind for d in dec].count("M") == 0, "关闭档不得产生 M 决策"
    assert _chain_digest(dec, met) == PM_HEAD_OFF_DIGEST, \
        "关闭维护头后链路变了——既有读数不再成立"


@pytest.mark.unit
def test_pm_head_requires_maintenance_on():
    """⑫ 关闭时不得启用维护头——**显式报错**，不静默退化。

    ⑫ 关掉时 `MachineSim` 连 `pm_clock` 都不累加（决策点根本不存在）：给一个"可选的
    死动作"只会污染链 logp。与 `route_k>1` 要求 ① 拥堵同一形态。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    off = ConstraintConfig(maintenance=False)
    lay2, dm2, ctx2 = build_setup(inst, cfg, constraints=off)
    with pytest.raises(ValueError, match="维护|⑫"):
        roll_chain(inst, lay2, dm2, cfg, pol, seed=0, ctx=ctx2,
                   constraints=off, pm_head=True)


# ────────────────────────── 2. M 决策真的进链 ──────────────────────────

@pytest.mark.unit
def test_pm_decisions_join_the_chain_and_its_logp():
    """判据 2：短间隔档下 M 决策必须出现、进链 logp、有梯度、同 seed 可复现。

    ⚠️ 只断言"链里出现 M"会漏掉"logp 里没有 M"的形态（那些决策成了不产生梯度的装饰），
    故这里**抽掉 M 再算链 logp**，值必须变。
    """
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, pm_head=True)
    assert not met["horizon_hit"] and met["jobs_done"] == inst.n_jobs, \
        f"维护头档跑不完：horizon_hit={met['horizon_hit']} jobs={met['jobs_done']}"
    m_dec = [d for d in dec if d.kind == "M"]
    assert len(m_dec) >= 20, f"M 决策太少（{len(m_dec)}）——判据失去意义"
    assert all(d.cand == PM_CANDS for d in m_dec), \
        f"M 的候选应是动作码 {PM_CANDS}：{ {d.cand for d in m_dec} }"
    assert all(d.mach is not None and 0 <= d.mach < inst.n_machines for d in m_dec), \
        "M 决策必须记下决定保养的那台机台"
    assert all(0 <= d.action < len(PM_CANDS) for d in m_dec)
    # 采样回放与带梯度重算同源（裁剪路径的正确性前提）——
    # ⚠️ 2026-10-04 批量重算批次：重算改成 (B,N,F) 一次批前向，与采样回放只在末位漂移内
    # 一致（≤1e-5），不再逐位相同。见 test_joint_chain 的同源判据（旧断言是逐位相等）。
    vec = decisions_logp(dec, pol).detach()
    assert vec.shape == (len(dec),), f"逐决策 logp 未覆盖 M：{tuple(vec.shape)}"
    assert torch.allclose(vec, sampled_decisions_logp(dec), atol=1e-5), \
        "含 M 后逐决策 logp 的采样回放与重算漂开超过 1e-5（批路径算错了？）"
    # M 决策真的进链 logp（不是只记录）
    lp_all = float(chain_logp(dec, pol).detach())
    lp_no_m = float(chain_logp([d for d in dec if d.kind != "M"], pol).detach())
    assert lp_all != lp_no_m, "M 决策没有进链 logp——维护头是名义上的"
    # 梯度到 M 头（M 决策参与训练）
    chain_logp(dec, pol).backward()
    g = pol.pm_head_tok[0].weight.grad
    assert g is not None and g.abs().sum() > 0, "M 头没有梯度——它的决策不进训练"
    # 同 seed 逐位可复现
    dec_b, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, pm_head=True)
    sig = lambda ds: [(d.kind, d.cand, d.action) for d in ds]      # noqa: E731
    assert sig(dec) == sig(dec_b), "开维护头后同 seed 不再可复现（M 的采样流没接上 seed）"


# ────────────────────────── 3. 候选特征与打分头 ──────────────────────────

@pytest.mark.unit
def test_pm_candidate_features_make_the_choice_nontrivial():
    """判据 2（特征层 + 网络层）：两个候选的特征行必须**不同**，且改它分数必须变。

    两个候选同属一台机台：若 `feat_cand` 两行相同，两个候选的分数**恒等**——决策退化
    成不可学的掷硬币（梯度对两个动作相同）。故"行 0（现在保养）≠ 行 1（不保养）"是
    机制成立的硬前提，不是风格问题。
    """
    from plcsp.algo.group_rel import _pm_cand_feat, pm_feat
    inst, lay, dm, cfg, ctx, pol = _setup()
    ms = MachineState(backlog_min=12.0, in_q_len=2, in_cap=3.0, out_q_len=1, out_cap=3.0,
                      busy=True, remaining_min=0.0, pm_used_min=4.0, fail_rate=0.01,
                      prev_job=0)
    snap = Snapshot(now=30.0, machines=(ms,), jobs=(), vehicles=(),
                    n_done=0, in_flight=0, zone_holder=())
    dec_f = pm_feat(snap, 0, ctx)
    cand_f = _pm_cand_feat(snap, 0, ctx)
    assert len(dec_f) == PM_FEAT_DEC and cand_f.shape == (2, PM_FEAT_CAND), \
        f"特征宽度与声明不符：dec={len(dec_f)} cand={cand_f.shape}"
    # 行 0 = 现在保养（停机就在此刻 ⟹ 距停机 0）；行 1 = 不保养（距强制点 = pm_left）
    assert cand_f[0, 0] == pytest.approx(0.0), f"现在保养行的余量应为 0：{cand_f}"
    assert cand_f[1, 0] == pytest.approx(1.0 - 4.0 / PM_SHORT), \
        f"不保养行的余量应为 pm_left：{cand_f}"
    assert not np.allclose(cand_f[0], cand_f[1]), "两个候选的特征行相同——决策不可学"
    # 网络层：只改候选特征，打分必须变（该槽是活的）
    torch.manual_seed(3)
    pol = PolicyNet(enc=LayoutEncoder())
    tok_feat, seg = _enc_inputs()
    tok, _ = pol.forward_enc(tok_feat, seg)
    idx = torch.tensor([1, 1])                       # 1 号机台的 token（两个候选共用）
    base = pol.pm_logits_emb(tok, torch.zeros(1, 1, PM_FEAT_DEC),
                             torch.as_tensor(cand_f), idx)
    changed = pol.pm_logits_emb(tok, torch.zeros(1, 1, PM_FEAT_DEC),
                                torch.as_tensor(cand_f) + 0.5, idx)
    assert base.shape == (1, 1, 2), f"M 头输出形状应为 (1,1,2)：{tuple(base.shape)}"
    assert not torch.allclose(base, changed), "只改候选特征打分却不变——该槽是死的"


@pytest.mark.unit
def test_pm_feature_widths_match_the_head():
    """⚠️ `PM_FEAT_*`（`group_rel` 的宽度）与 M 头的输入宽度必须一致——漂开即静默错位。"""
    pol = PolicyNet(enc=LayoutEncoder())
    assert pol.pm_head_tok[0].in_features == pol.enc.d_model + PM_FEAT_DEC + PM_FEAT_CAND


# ────────────────────────── 4. 生产路径的 token 下标守卫 ──────────────────────────

@pytest.mark.unit
def test_pm_head_reads_the_deciding_machines_token_on_the_production_path():
    """⚠️ M 头在生产路径上必须读**决定保养的那台机台**的 M 段 token——不得把动作码当下标。

    缺陷形态（与 §31 L 头同型）：`_act` 若照 S 头写 `tok_idx = cand`，而 M 的候选是动作码
    {0, 1}，两个候选会去读 **0 号与 1 号机台**的 token——"维护头"在给别的机器打分，且
    `cand=(0,1)` 只在测试手搓时才"看起来对"。故守卫**走 `roll_chain`**（`_act` 真正记录
    下标）→ 挑一条**由 m≥2 号机台**做出的决策 → 只扰动**该机台**的 M 段行 → 经
    `decisions_logp`（生产重算路径）重算，logp 必须变。

    ⚠️ 用逐 token 恒等嵌入（`_NoMixEncoder`）切断注意力泄漏，理由同 §31.2：真编码器下
    扰动任意一行都会改变所有 token 的上下文嵌入，缺陷在位时也会"变"——判据恒真。
    ⚠️ 负判据（扰动**别的**机台的行 ⟹ logp 必须逐位不变）是抓 `tok_idx = cand` 的另一半：
    缺陷在位时扰动 0/1 号机会改变候选 0/1 的分数。
    """
    torch.manual_seed(7)
    pol = PolicyNet(enc=_NoMixEncoder())
    inst, lay, dm, cfg, ctx, _ = _setup()
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, pm_head=True)
    m_dec = [d for d in dec if d.kind == "M"]
    assert len(m_dec) >= 20, f"M 决策太少（{len(m_dec)}）——判据失去意义"
    d = next((d for d in m_dec if d.mach is not None and d.mach >= 2), None)
    assert d is not None, "链路里没有 m≥2 号机台做出的 M 决策——守卫区分不出错下标"
    assert np.array_equal(d.tok_idx, np.full(len(d.cand), int(d.mach), dtype=np.int64)), (
        f"M 决策记录的 token 下标 {d.tok_idx.tolist()} 应两个候选都指向 "
        f"{d.mach} 号机台（动作码不是序列位置）")
    base = decisions_logp([d], pol).detach().clone()
    delta = np.zeros((1, F_MAX), dtype=np.float32)
    delta[0, :3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)   # 逐列不同，非均匀平移
    d2 = copy.deepcopy(d)
    d2.tok[int(d.mach):int(d.mach) + 1] += delta
    after = decisions_logp([d2], pol).detach()
    assert not torch.equal(base, after), (
        f"扰动 {d.mach} 号机台的 token 后该 M 决策的 logp 不变——"
        f"M 头读的不是这台机（记录下标 {d.tok_idx.tolist()}）")
    other = 0 if d.mach != 0 else 1
    d3 = copy.deepcopy(d)
    d3.tok[other:other + 1] += delta
    after3 = decisions_logp([d3], pol).detach()
    assert torch.equal(base, after3), (
        f"扰动 {other} 号机台（非决定方）的 token 却改变了 m={d.mach} 的 M 决策——"
        "M 头读了不相干的 token")


# ────────────────────────── 5. 动作有效果（生产路径的 liveness） ──────────────────────────

@pytest.mark.unit
def test_pm_defer_matches_the_rule_and_now_moves_the_first_pm_earlier():
    """判据 3：策略的动作真的决定保养**何时发生**，且"不保养"档逐位等于规则档。

    三个同 seed、同确定性派车（`policy_l`）的运行：
    - **规则档**（`policy_m=None`）：今日行为；
    - **永远不保养档**：逾期前不主动停 ⟹ 应与规则档**逐位相同**（强制底线 = 规则）；
    - **永远现在保养档**：首次保养发生在**第一个决策点**（第一道工序下机）。
      规则档的首次保养不可能早于 `pm_interval`（主轴工时 ≤ 墙钟，累计 10 min 主轴至少要
      10 min 墙钟）⟹ 首次保养时刻被**动作**提前了，这是"决策有效果"的直接读数。
    """
    inst = load_mk("mk01")
    cfg = SimConfig(pm_interval=PM_SHORT)
    lay, dm = build_layout_and_dm(inst, cfg)
    pick_first = lambda snap, job, frm, to, oi, cand: cand[0]        # noqa: E731
    trace = []

    def pol_now(snap, m, cand):
        trace.append((float(snap.now), int(m), float(snap.machines[m].pm_used_min)))
        return PM_CAND_NOW

    def pol_defer(snap, m, cand):
        return PM_CAND_DEFER

    rule = SimWorld(inst, lay, dm, cfg).run_gated(seed_chain=1, policy_l=pick_first)
    defer = SimWorld(inst, lay, dm, cfg).run_gated(seed_chain=1, policy_l=pick_first,
                                                   policy_m=pol_defer)
    now = SimWorld(inst, lay, dm, cfg).run_gated(seed_chain=1, policy_l=pick_first,
                                                 policy_m=pol_now)
    for name, met in (("规则", rule), ("不保养", defer), ("现在保养", now)):
        assert not met["horizon_hit"] and met["jobs_done"] == inst.n_jobs, \
            f"{name}档未跑完：horizon_hit={met['horizon_hit']} jobs={met['jobs_done']}"
    assert defer["pm_events"] == rule["pm_events"] > 0, \
        (f"「永远不保养」档的保养事件数 {defer['pm_events']} ≠ 规则档 {rule['pm_events']}——"
         "强制底线不是规则原本的行为")
    assert defer["makespan"] == rule["makespan"] and defer["completes"] == rule["completes"], \
        "「永远不保养」档与规则档的时序不同——决策点改变了规则档的行为"
    assert now["pm_events"] > defer["pm_events"], \
        (f"「永远现在保养」档的保养事件数 {now['pm_events']} 未超过规则档 "
         f"{defer['pm_events']}——动作没有被执行")
    assert trace, "策略档没有调用 M 回调"
    first_now = min(t for t, _m, _c in trace)     # 「现在保养」⟹ 保养就在此刻开始
    assert first_now < PM_SHORT, (
        f"首个决策点 t={first_now:.2f} 不早于 pm_interval={PM_SHORT}——"
        "规则档的首次保养下界被打破，本判据的'提前'不成立")
