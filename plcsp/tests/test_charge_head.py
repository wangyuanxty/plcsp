"""⑪ 充电头（C）+ 耗尽模型修复——把「去哪充 / 充不充」从规则交还策略（`docs/mechanism-designs.md` §⑪）。

本文件盖**两件事**，缺一不可：

1. **模型修复（耗尽有后果）**：`battery <= 0` ⟹ 该车**不可用**（不接新任务），
   充到 `battery_high × battery_cap` 才恢复。判定在**任务边界**——该车空闲待命、不在行驶中、
   **不持任何区段锁**。⚠️ 这是刻意的：`_drive` 持锁行驶，车若死在持锁状态会把整条走廊
   堵死（死锁），故车必须能跑完**已开始**的行程。
   没有这条修复，充电是**纯成本、零收益**（行驶 + 充电时长 + 计入能耗目标），
   最优策略是"永不充"，C 决策退化。
   ⚠️ 电池只在 ⑪ `charging` 开启时增减，故 **⑪ 关态本修复不可能触发**——既有 ⑪ 关读数
   （含 `ABLATION_GROUPS["-物流"]` 与 `None` 组）继续有效。

2. **C 决策**：`AgvSim._maybe_charge` 的规则（低电 → 最近**空闲**桩）改为策略决策，
   候选 = {不去充} ∪ {各充电桩}，逐候选打分，与 S/L/R/M 四头同构。
   ⚠️ **候选在 token 序列里没有 token**：充电桩不是 M/B/V/G 四段中的任何一种实体，
   "不去充"更不是。故全部候选共用**本车**的 V token（与 R 头"候选是路径、广播本车 V token"
   同型）；候选之间的分数差只能来自 `feat_cand`，上下文只经 GELU 的非线性调节敏感度。

**判据**（与设计文档 §⑪ 的验收一致）：
1. `charge_head=False`（默认）⟹ 链路逐位等于今日（黄金摘要钉死）；⑪ 关态同样逐位不变；
2. C 决策真的进链 logp、有梯度、同 seed 可复现；
3. **生产路径**的 token 下标守卫（不得自己提供下标）+ 变异检查（见 `docs/progress-log.md` §31）；
4. liveness 两半：小电池验证档**真的会耗尽**（`agv_dry_events`）；「永远现在充」策略相对规则配置
   **可测地改变行为**。

⚠️ **默认参数不动**（§27.1 纪律一）：`layout.sample_layout` 的默认电池 2–4 kWh 在一个
episode 里放不空（实测 mk01 全程每车耗电 ~0.1–0.27 kWh）。故验证一律用**显式小电池验证档**
（`AgvSpec(battery_kwh=0.10)` + `battery_low=0.0`，同 `m11_constraint_binding` 的先例）——
改默认会为第二个不必要的理由作废全部读数。
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch

from plcsp.algo.group_rel import (CHARGE_FEAT_CAND, CHARGE_FEAT_DEC, chain_logp,
                                  charge_feat, decisions_logp, roll_chain,
                                  sampled_decisions_logp)
from plcsp.algo.policy import PolicyNet, v_token_index
from plcsp.algo.setup import build_layout_and_dm, build_setup
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, SimWorld, charge_cands
from plcsp.env.instances import load_mk
from plcsp.env.layout import AgvSpec
from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_MAX, SEG_SLICE

# 黄金摘要：`charge_head=False` 配置的链路指纹（决策类型 + 全部打分上下文 + tok_idx + 动作 +
# 采样 logp + 关键指标）。**捕获自改造前的树**（`git stash` 前的 HEAD，脚本在系统临时目录），
# 并与 `test_maintenance_head.PM_HEAD_OFF_DIGEST` **同值**——这两条钉的是同一条默认路径，
# 一个改变该配置行为的改动会同时翻红两处。
#   kinds={'S':55,'L':65} n=120，makespan=118.8647282376667
CHARGE_HEAD_OFF_DIGEST = "5a06255d6d57d2f507c7186a198c11ec056454a359265809c80da40c98f4c5ae"

# ⑪ 关态的链路指纹——**同样捕获自改造前的树**。本修复的"不可能触发"由此证明：
# 关掉 ⑪ 后电池恒为 `battery_cap`（`_drain` / `_drain_idle` / `_maybe_charge` 全部直接返回），
# `battery <= 0` 恒不成立，耗尽门永不生效 ⟹ 链路逐位不变。
CHARGING_OFF_DIGEST = "dc766fd04a6d042cd65ca924e3e2727311f49cf19072ad739a7847551b8304d5"

# 验证用的**小电池验证档**：电池 0.10 kWh + `battery_low=0.0`。
# ⚠️ `battery_low` 必须为 0：默认 0.20 时规则在 0.02 kWh 就补电，电池**到不了 0**
# （实测 mk01 默认配置最小 2.64 kWh、0.3 kWh 配置最小 0.098 kWh）——那样本判据测不到任何东西。
# `low=0` 时规则只在**恰好 0** 动手，车必然先跑干（实测 3 个种子均触 0）。
SMALL_BATTERY_KWH = 0.10


def _setup(charge=False, constraints=None):
    """(inst, layout, dm, cfg, ctx, policy)——与 `roll_chain` 的签名对齐（同 test_maintenance_head）。"""
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg, constraints=constraints)
    return inst, lay, dm, cfg, ctx, PolicyNet(enc=LayoutEncoder())


def _small_battery_setup(charge_off=False):
    """小电池验证档的 (inst, layout, dm, cfg)：车队电池全换成 `SMALL_BATTERY_KWH`，`battery_low=0`。"""
    inst = load_mk("mk01")
    cfg = SimConfig()
    cfg.battery_low = 0.0
    lay, dm = build_layout_and_dm(inst, cfg)
    for i, a in enumerate(lay.agvs):
        lay.agvs[i] = AgvSpec(id=a.id, speed_factor=a.speed_factor,
                              capacity=a.capacity, battery_kwh=SMALL_BATTERY_KWH)
    cons = ConstraintConfig(charging=False) if charge_off else ConstraintConfig()
    return inst, lay, dm, cfg, cons


def _chain_digest(dec, met) -> str:
    """链路指纹（含 `tok_idx`——它把"头读哪一行"也钉进摘要）。与 test_maintenance_head 同口径。"""
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
    """逐 token **恒等**嵌入（无跨 token 注意力）——只为隔离"C 头索引了哪一行"这一件事。

    真编码器是全连接注意力：扰动 V 段某一行会经注意力改变其余 token 的上下文嵌入，
    "读错行的头"也跟着变——判据被掩盖（§31.2 实测 −1.0927 → −1.0873）。逐 token 恒等嵌入
    切断这条泄漏后，"分数依赖哪一行"才是可判的量，而 `_act` / `decisions_logp` 仍走生产代码。
    `d_model = F_MAX` 使 `PolicyNet` 按 11 维 token 建头。
    """

    d_model = F_MAX

    def forward(self, x, seg, bias=None):
        return x, x.mean(dim=1)


# ────────────────────────── 1. 关态逐位不变（摘要在前文） ──────────────────────────

@pytest.mark.unit
def test_charge_head_off_is_bit_identical_to_baseline():
    """判据 1：`charge_head=False`（默认）⟹ 链路逐位等于改造前——黄金摘要钉死。"""
    inst, lay, dm, cfg, ctx, _ = _setup()
    torch.manual_seed(20261004)                 # 与捕获脚本逐字对齐的初始化锚点
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, charge_head=False)
    assert [d.kind for d in dec].count("C") == 0, "关态不得产生 C 决策"
    assert _chain_digest(dec, met) == CHARGE_HEAD_OFF_DIGEST, \
        "关闭充电头后链路变了——既有读数不再成立"


@pytest.mark.unit
def test_charging_off_is_bit_identical_to_baseline():
    """判据 1（模型修复的"不可能触发"）：⑪ 关态 ⟹ 链路逐位等于改造前。

    ⑪ 关时电池恒为 `battery_cap`（`_drain` / `_drain_idle` / `_maybe_charge` 全部直接返回），
    `battery <= 0` 恒不成立——耗尽门永不生效。这是既有 ⑪ 关读数（-物流 / None 消融组）继续
    有效的前提，由**改造前**捕获的摘要钉死。
    """
    off = ConstraintConfig(charging=False)
    inst, lay, dm, cfg, ctx, _ = _setup(constraints=off)
    torch.manual_seed(20261004)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx,
                          constraints=off, charge_head=False)
    assert _chain_digest(dec, met) == CHARGING_OFF_DIGEST, \
        "⑪ 关态的链路变了——耗尽后果在 ⑪ 关时被触发了（本修复必须不可能）"


@pytest.mark.unit
def test_charge_head_requires_charging_on():
    """⑪ 关闭时不得启用充电头——**显式报错**，不静默退化（同 route_k>1 要求 ① 的形态）。

    ⑪ 关时电池从不增减（恒 = cap）、充电桩机制根本不存在：给策略一个"可选的死动作"
    只会污染链 logp。
    """
    off = ConstraintConfig(charging=False)
    inst, lay, dm, cfg, ctx, _ = _setup(constraints=off)
    with pytest.raises(ValueError, match="充电|⑪"):
        roll_chain(inst, lay, dm, cfg, PolicyNet(enc=LayoutEncoder()),
                   seed=0, ctx=ctx, constraints=off, charge_head=True)


# ────────────────────────── 2. C 决策真的进链 ──────────────────────────

@pytest.mark.unit
def test_charge_decisions_join_the_chain_and_its_logp():
    """判据 2：C 决策必须出现、进链 logp、有梯度、同 seed 可复现。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, charge_head=True)
    assert not met["horizon_hit"] and met["jobs_done"] == inst.n_jobs, \
        f"充电头配置跑不完：horizon_hit={met['horizon_hit']} jobs={met['jobs_done']}"
    c_dec = [d for d in dec if d.kind == "C"]
    assert len(c_dec) >= 10, f"C 决策太少（{len(c_dec)}）——判据失去意义"
    n_ch = len(lay.chargers)
    assert all(d.cand == charge_cands(n_ch) for d in c_dec), \
        f"C 的候选应是动作码 {charge_cands(n_ch)}：{ {d.cand for d in c_dec} }"
    assert all(0 <= d.action < len(d.cand) for d in c_dec)
    # 采样回放与带梯度重算同源（裁剪路径的正确性前提）——
    # ⚠️ 2026-10-04 批量重算批次：重算改成 (B,N,F) 一次批前向，与采样回放只在末位漂移内
    # 一致（≤1e-5），不再逐位相同。见 test_joint_chain 的同源判据（旧断言是逐位相等）。
    vec = decisions_logp(dec, pol).detach()
    assert vec.shape == (len(dec),), f"逐决策 logp 未覆盖 C：{tuple(vec.shape)}"
    assert torch.allclose(vec, sampled_decisions_logp(dec), atol=1e-5), \
        "含 C 后逐决策 logp 的采样回放与重算漂开超过 1e-5（批路径算错了？）"
    # C 决策真的进链 logp（不是只记录）
    lp_all = float(chain_logp(dec, pol).detach())
    lp_no_c = float(chain_logp([d for d in dec if d.kind != "C"], pol).detach())
    assert lp_all != lp_no_c, "C 决策没有进链 logp——充电头是名义上的"
    # 梯度到 C 头（C 决策参与训练）
    chain_logp(dec, pol).backward()
    g = pol.c_head_tok[0].weight.grad
    assert g is not None and g.abs().sum() > 0, "C 头没有梯度——它的决策不进训练"
    # 同 seed 逐位可复现
    dec_b, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, charge_head=True)
    sig = lambda ds: [(d.kind, d.cand, d.action) for d in ds]      # noqa: E731
    assert sig(dec) == sig(dec_b), "开充电头后同 seed 不再可复现（C 的采样流没接上 seed）"


# ────────────────────────── 3. 候选特征与打分头 ──────────────────────────

@pytest.mark.unit
def test_charge_candidate_features_make_the_choice_nontrivial():
    """判据 2（特征层 + 网络层）：候选行必须**不同**，且改它分数必须变。

    "不去充"候选与"去某桩"候选若特征行相同，两个方向的分数只能靠 token 相同的那份上下文
    区分——决策退化成不可学的掷硬币。故 `feat_cand` 必须携带候选间差异（桩距 / 占用 /
    "是不是不去充"），这是机制成立的硬前提。
    """
    from plcsp.algo.group_rel import _charge_cand_feat
    from plcsp.env.snapshot import ChargerState, Snapshot, VehicleState
    _inst, lay, dm, cfg, ctx, _p = _setup()
    v = VehicleState(status=1, node=0, queued=1, battery_frac=0.3, capacity=1,
                     speed_factor=1.0, zone_wait=0.0)
    snap = Snapshot(now=10.0, machines=(), jobs=(), vehicles=(v,), n_done=0, in_flight=0,
                    zone_holder=(), chargers=(ChargerState(occupied=1, waiting=0, capacity=1),
                                             ChargerState(occupied=0, waiting=2, capacity=1)))
    dec_f = charge_feat(snap, 0, ctx)
    cand_f = _charge_cand_feat(snap, lay, dm, ctx, 0, charge_cands(len(lay.chargers)))
    assert len(dec_f) == CHARGE_FEAT_DEC and cand_f.shape == (len(lay.chargers) + 1, CHARGE_FEAT_CAND), \
        f"特征宽度与声明不符：dec={len(dec_f)} cand={cand_f.shape}"
    assert not np.allclose(cand_f[0], cand_f[1]), "'不去充'与'去 0 号桩'的特征行相同——决策不可学"
    assert cand_f[0, 2] == pytest.approx(1.0) and cand_f[1, 2] == pytest.approx(0.0), \
        f"'是不是不去充'列反了：{cand_f[:, 2]}"
    # 占用维取的是**逐桩的活计数**（行序：cand[1] = 0 号桩、cand[2] = 1 号桩）
    assert cand_f[1, 1] == pytest.approx(1.0) and cand_f[2, 1] == pytest.approx(2.0), \
        f"桩占用/排队维没按桩取值：{cand_f[:, 1]}"
    assert not np.allclose(cand_f[1, :2], cand_f[2, :2]), "两根桩的行在候选槽上完全同值"
    # 网络层：只改候选特征，打分必须变（该槽是活的）
    torch.manual_seed(3)
    pol = PolicyNet(enc=LayoutEncoder())
    tok_feat, seg = _enc_inputs()
    tok, _ = pol.forward_enc(tok_feat, seg)
    idx = torch.full((len(cand_f),), v_token_index(seg)[1], dtype=torch.long)   # 本车 V token 广播
    base = pol.charge_logits_emb(tok, torch.zeros(1, 1, CHARGE_FEAT_DEC),
                                 torch.as_tensor(cand_f), idx)
    changed = pol.charge_logits_emb(tok, torch.zeros(1, 1, CHARGE_FEAT_DEC),
                                    torch.as_tensor(cand_f) + 0.5, idx)
    assert base.shape == (1, 1, len(cand_f)), f"C 头输出形状应为 (1,1,m)：{tuple(base.shape)}"
    assert not torch.allclose(base, changed), "只改候选特征打分却不变——该槽是死的"


@pytest.mark.unit
def test_charge_feature_widths_match_the_head():
    """⚠️ `CHARGE_FEAT_*`（`group_rel` 的宽度）与 C 头的输入宽度必须一致——漂开即静默错位。"""
    pol = PolicyNet(enc=LayoutEncoder())
    assert pol.c_head_tok[0].in_features == pol.enc.d_model + CHARGE_FEAT_DEC + CHARGE_FEAT_CAND


# ────────────────────────── 4. 生产路径的 token 下标守卫 ──────────────────────────

@pytest.mark.unit
def test_c_head_reads_the_deciding_agvs_token_on_the_production_path():
    """⚠️ C 头在生产路径上必须读**决定方那台车**的 V 段 token——不得把候选动作码当下标。

    候选（不去充 / 各充电桩）在序列里**没有 token**，故全部候选共用本车 V token（同 R 头）。
    缺陷形态：`_act` 若照 S 头写 `tok_idx = cand`，候选动作码 {0,1,2} 会被当成序列位置，
    C 头去读 **0/1/2 行的 M 段（机台）token**——"充电头"评的是机台，且 `cand=(0,1,2)` 只在
    测试手搓时才"看起来对"（与 §31 的 L 头缺陷同型）。故守卫**走 `roll_chain`**（`_act`
    真正记录下标）→ 挑一条**由 aid≥1 的车**做出的 C 决策 → 只扰动**该车**的 V 段行 →
    经 `decisions_logp`（生产重算路径）重算，logp 必须变；扰动**别的车**的行必须逐位不变。

    ⚠️ 用逐 token 恒等嵌入（`_NoMixEncoder`）切断注意力泄漏，理由同 §31.2。
    """
    torch.manual_seed(7)
    pol = PolicyNet(enc=_NoMixEncoder())
    inst, lay, dm, cfg, ctx, _ = _setup()
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, charge_head=True)
    c_dec = [d for d in dec if d.kind == "C"]
    assert len(c_dec) >= 10, f"C 决策太少（{len(c_dec)}）——判据失去意义"
    v_rows = v_token_index(c_dec[0].seg)
    # ⚠️ 用 `d.agv`（决定方车号）选样本，**不**拿 `tok_idx` 反推——tok_idx 正是本守卫要
    # 检验的量，用它选样本是循环论证：错误实现下会先在"选样本"一步失败，测不到实质判据。
    d = next((x for x in c_dec if x.agv is not None and x.agv >= 1), None)
    assert d is not None, "链路里没有 aid≥1 的车做出的 C 决策——守卫区分不出错下标"
    aid = int(d.agv)
    assert np.array_equal(d.tok_idx, np.full(len(d.cand), v_rows[aid], dtype=np.int64)), (
        f"C 决策记录的 token 下标 {d.tok_idx.tolist()} 应全部指向 {aid} 号车的 V token"
        f"（下标 {v_rows[aid]}）——候选是动作/桩号，不是序列位置")
    base = decisions_logp([d], pol).detach().clone()
    delta = np.zeros((1, F_MAX), dtype=np.float32)
    delta[0, :3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)   # 逐列不同，非均匀平移
    d2 = copy.deepcopy(d)
    d2.tok[v_rows[aid]:v_rows[aid] + 1] += delta
    after = decisions_logp([d2], pol).detach()
    assert not torch.equal(base, after), (
        f"扰动 {aid} 号车（决定方）的 V token 后该 C 决策的 logp 不变——"
        f"C 头读的不是这台车（记录下标 {d.tok_idx.tolist()}）")
    other = 0 if aid != 0 else 1
    d3 = copy.deepcopy(d)
    d3.tok[v_rows[other]:v_rows[other] + 1] += delta
    after3 = decisions_logp([d3], pol).detach()
    assert torch.equal(base, after3), (
        f"扰动 {other} 号车（非决定方）的 token 却改变了 aid={aid} 的 C 决策——"
        "C 头读了不相干的 token")


# ────────────────────────── 5. liveness（两半） ──────────────────────────

@pytest.mark.unit
def test_small_battery_agvs_really_run_dry_and_recover():
    """判据 4a：⑪ 开 + 小电池 ⟹ 车**真的跑到 0**（不可用），且**必须能恢复**（不死锁）。

    规则配置（C 关）也走耗尽门——这就是模型修复的落点：没有它，车在 0 电量照样接活，
    "充电"是纯成本零收益。`agv_dry_events` = 任务边界上发现本车耗尽的次数。
    ⚠️ 断言"跑完 + 全部作业完成"是**防死锁**的读数：耗尽门只在任务边界（不持锁）判定，
    车总能跑完已开始的行程并补上电。
    """
    inst, lay, dm, cfg, cons = _small_battery_setup()
    for s in (0, 1):
        r = SimWorld(inst, lay, dm, cfg, constraints=cons).run(seed_chain=s)
        assert r["jobs_done"] == inst.n_jobs and not r["horizon_hit"], \
            f"小电池验证档未跑完（seed={s}）——耗尽门把仿真卡死了"
        assert r["battery_min_kwh"] == 0.0, \
            f"小电池验证档没跑到 0（min={r['battery_min_kwh']}）——本判据的前提不成立"
        assert r["agv_dry_events"] > 0, \
            f"车跑干了却不记为不可用（seed={s}）——耗尽没有后果，充电仍是无收益的成本"
        assert r["charge_events"] > 0, "跑干后没补过电——恢复路径断了"


@pytest.mark.unit
def test_charging_off_never_produces_dry_events():
    """判据 4a 的反面：⑪ 关 ⟹ 电池恒为 cap，`agv_dry_events == 0`（本修复不可能触发）。"""
    inst, lay, dm, cfg, cons = _small_battery_setup(charge_off=True)
    world = SimWorld(inst, lay, dm, cfg, constraints=cons)
    r = world.run(seed_chain=0)
    assert r["agv_dry_events"] == 0, "⑪ 关态出现耗尽——电池在没有 ⑪ 时也被扣了"
    # 电池恒为满（`battery_frac == 1.0`）；`battery_min_kwh` 的 0.0 是"无耗电记录"哨兵
    # （stats 里的 `inf` 在返回时折成 0.0），**不是**真的掉到 0。
    assert all(v.battery_frac == pytest.approx(1.0) for v in world.snapshot().vehicles), \
        "⑪ 关态电池应当恒为容量"
    assert r["battery_min_kwh"] == 0.0, "⑪ 关态 battery_min_kwh 应是 '无耗电记录' 哨兵 0.0"
    assert r["jobs_done"] == inst.n_jobs and not r["horizon_hit"]


@pytest.mark.unit
def test_depletion_gate_never_fires_while_holding_a_zone_lock(monkeypatch):
    """⚠️ 死锁防线（直接读数）：耗尽门**只在车不持任何区段锁时**才判定。

    `_drive` 持锁行驶；车若死在持锁状态，别的车会永远等这条走廊（死锁）。故本测试在
    `_depleted` 返回 True 的每一次上，记录该车是否正持有区段——必须全部为"未持有"。
    """
    from plcsp.env.des import AgvSim
    held = []
    orig = AgvSim._depleted

    def spy(self):
        dry = orig(self)
        if dry:
            held.append(any(h == self.aid for h in self.zm.holder.values()))
        return dry

    monkeypatch.setattr(AgvSim, "_depleted", spy)
    inst, lay, dm, cfg, cons = _small_battery_setup()
    r = SimWorld(inst, lay, dm, cfg, constraints=cons).run(seed_chain=0)
    assert held, "耗尽门一次都没触发——判据失去意义"
    assert not any(held), "耗尽判定发生在车持有区段锁时——那会把走廊堵死（死锁）"
    assert r["jobs_done"] == inst.n_jobs and not r["horizon_hit"]


@pytest.mark.unit
def test_always_charge_now_policy_moves_behaviour_versus_the_rule():
    """判据 4b：动作**有效果**——强制「永远现在充」相对规则配置可测地改变行为。

    同 seed、同确定性派车（`policy_l`），只有 C 回调不同：
    - **规则配置**（`policy_c=None`）：`battery_low=0` ⟹ 只在恰好 0 时补电（"跑干再充"）；
    - **永远现在充**：每个空闲点都去 0 号桩 ⟹ 充电次数更多、时序不同（行驶/排队/充电都是成本）。
    """
    inst, lay, dm, cfg, cons = _small_battery_setup()
    pick_first = lambda snap, job, frm, to, oi, cand: cand[0]       # noqa: E731
    trace = []

    def always(snap, aid, cand):
        trace.append((float(snap.now), int(aid)))
        return cand[1]                       # 1 = 0 号桩（0 = 不去充）

    rule = SimWorld(inst, lay, dm, cfg, constraints=cons).run_gated(
        seed_chain=1, policy_l=pick_first)
    now = SimWorld(inst, lay, dm, cfg, constraints=cons).run_gated(
        seed_chain=1, policy_l=pick_first, policy_c=always)
    for name, met in (("规则", rule), ("永远现在充", now)):
        assert met["jobs_done"] == inst.n_jobs and not met["horizon_hit"], \
            f"{name} 配置未跑完：jobs={met['jobs_done']} horizon_hit={met['horizon_hit']}"
    assert trace, "策略配置没有调用 C 回调"
    assert now["charge_events"] > rule["charge_events"], \
        (f"「永远现在充」的充电次数 {now['charge_events']} 未超过规则配置 "
         f"{rule['charge_events']}——动作没有被执行")
    assert now["makespan"] != rule["makespan"], "动作被执行了却没改变时序"
