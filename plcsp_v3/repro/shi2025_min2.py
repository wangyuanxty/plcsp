# -*- coding: utf-8 -*-
"""**调度侧逐层加**：动作空间从 K 条规则起，一根一根加。

══ 这条线在整条调试线上的位置 ══

```
1. 最小版（只 PA、9 条规则）                        ✅
2. 修掉查实的缺陷（跨链 GAE / 参照表口径 / 特征标准化）  ✅
3. **从 2 条规则起，逐层加**                        ← **本文件**
4. 加到 9 条 ⟹ 换大实例 ⟹ 完整配置
```

**加法固定、各档可比**：按 20 场景**实测 makespan** 排序，取**均匀间隔**的 K 条。
K=2 时 = {最好, 最差}。

⚠️ **判据是「greedy 的 makespan 有没有到本档最好那条」** —— 本档里最好那条是**全局**最好，
塌到它就是最优。**「有没有状态依赖」是下一层的判据，不在本层。**
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402
import torch  # noqa: E402

import shi2025_min_pa as M  # noqa: E402
from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import (N_AGV_SHI, POOL_STRIDE,  # noqa: E402
                                        PT_REDRAW_SEED, eval_instance,
                                        make_shi_instance, pooled)
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES,  # noqa: E402
                                PA_RULE_NAMES, apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build, dims_from_shop_len  # noqa: E402


def sig(inst) -> tuple:
    """实例指纹 —— 用于断言「评测实例不在训练池里」。"""
    return tuple(sorted((len(o), o[0][0], sum(t for _m, t in o))
                        for j in inst.jobs for o in j))

# 按 20 场景实测 makespan 从好到差（读数见 `v3_proxy_vs_truth.py`）：
#   EAJ+SMPT 147.6 · SPT+SMPT 169.7 · EAJ+EAM 169.8 · SPT+EAM 191.2 · EAJ+MINU 198.1
#   SRPT+SMPT 207.7 · SRPT+EAM 231.6 · SPT+MINU 242.7 · SRPT+MINU 279.6
# ⚠️ **2026-10-13 修**：此前写 `[3, 0, 5, 1, 4, 6, 8, 2, 7]` —— 位置 3 和 7 对调，
# **与上面这行注释不符**（242.7 排在了 198.1 和 191.2 前面）。**读数不受影响**：
# K=2/3/5 取的都是位置 0/2/4/6/8，恰好躲开错位的那两个。**日志里那张表是误导的，已改成排序后打印。**
RANK = [3, 0, 5, 2, 4, 6, 8, 1, 7]
K = int(os.environ.get("MIN_K", "2"))
# ⚠️ `MIN_ACT`：**直接指定动作集合**，绕过 RANK/均匀间隔。
# 用它把「动作**条数**」与「动作之间**离得多近**」拆开（K 阶梯把这两件事绑在一起了）：
#   MIN_ACT=3,7  → 差 90%（已测：100%）
#   MIN_ACT=3,0  → 差 15%
#   MIN_ACT=0,5  → 差 **0.06%**（两条规则几乎一样）
_ACT_ENV = os.environ.get("MIN_ACT", "")
ACT = (tuple(int(x) for x in _ACT_ENV.split(",")) if _ACT_ENV
       else (tuple(RANK[round(i * (len(RANK) - 1) / (K - 1))] for i in range(K))
             if K > 1 else (RANK[0],)))
K = len(ACT)
# ⚠️ **训练 seed 必须能换**（2026-10-13）：此前写死 `torch.manual_seed(0)` ⟹
# **整条 K 阶梯都是单 seed** ⟹「K=9 失败」「`{3,2}` 失败」都可能是这一个 seed 的偶然。
# 换 seed 只动网络初始化（场景由 `seed_chain` 独立控制，不受影响）—— 单变量。
SEED = int(os.environ.get("MIN_SEED", "0"))

# ⚠️ **环境规格旋钮**（2026-10-17 加，用户指定训练配置）：
#    `MIN_SHOP_LEN` / `MIN_SPEED` / `MIN_HLO` / `MIN_HHI` / `MIN_ALTSMIN` / `MIN_ALTSMAX`
#    `MIN_SHI2025=1` ⟹ 原文口径（AGV 故障开、机台故障/保养/交期/有限缓冲关）
#    + 两处账本修复（`agv_plan_pos` / `agv_act_book`）。
#    ⚠️ **不设这些环境变量 ⟹ 与旧行为逐位相同。**
_SHOP = os.environ.get("MIN_SHOP_LEN", "")
SHOP_LEN = float(_SHOP) if _SHOP else None
_SPEED = os.environ.get("MIN_SPEED", "")
SPEED = float(_SPEED) if _SPEED else None
_HLO = os.environ.get("MIN_HLO", "")
HLO = float(_HLO) if _HLO else None
_HHI = os.environ.get("MIN_HHI", "")
HHI = float(_HHI) if _HHI else None
_ALO = os.environ.get("MIN_ALTSMIN", "")
ALTS_MIN = int(_ALO) if _ALO else None
_AHI = os.environ.get("MIN_ALTSMAX", "")
ALTS_MAX = int(_AHI) if _AHI else None
SHI2025 = os.environ.get("MIN_SHI2025", "0") == "1"
# ⚠️ **实验 B 的变规模训练**（2026-10-18）：`MIN_VARY=1` 时训练池逐实例随机规模。
VARY = os.environ.get("MIN_VARY", "0") == "1"
JOBS_LO = int(os.environ.get("MIN_JOBS_LO", "10"))
JOBS_HI = int(os.environ.get("MIN_JOBS_HI", "40"))
# ⚠️ **新布局口径**（2026-10-18）：显式机位/走廊/L-U；`MIN_CELL=0` ⟹ 走 `shop_len`。
CELL = float(os.environ.get("MIN_CELL", "0"))
AISLE = float(os.environ.get("MIN_AISLE", "0"))
LU = float(os.environ.get("MIN_LU", "0"))
# ⚠️ 新布局口径下也要显式给排布：`random` + 5×5（与 CDR 表同口径）。
SCHEME = os.environ.get("MIN_SCHEME", "random")
# ⚠️ **特征标准化**（2026-10-17 加，用户选的第 1 项）：把 24 维 PA 特征压到 O(1)。
#    默认 `0` ⟹ 旧行为逐位不变；本次训练用 `MIN_FEAT_STD=1`。
FEAT_STD = os.environ.get("MIN_FEAT_STD", "0") == "1"


def _inst_kw() -> dict:
    """实例生成参数：只有显式给了上下界才切 `uniform`。"""
    kw: dict = {}
    if ALTS_MIN is not None:
        kw["alts_min"] = ALTS_MIN
    if ALTS_MAX is not None:
        kw["alts_max"] = ALTS_MAX
    if kw:
        kw["alts_dist"] = "uniform"
    return kw


def _sim_over() -> dict:
    """仿真覆盖：只有显式给了才写。**不覆盖 `agv_mttr_lo/hi`**（原文 5–10）。"""
    over: dict = {"due_range": 0.6}
    if SPEED is not None:
        over["agv_speed_mps"] = SPEED
    if HLO is not None:
        over["agv_handling_lo"] = HLO
    if HHI is not None:
        over["agv_handling_hi"] = HHI
    if SHI2025:
        over["agv_plan_pos"] = 1.0
        over["agv_act_book"] = 1.0
        # ⚠️ **同机台相邻工序省略搬运**（2026-10-17，照原文）。
        over["skip_same_machine_move"] = True
    return over


def _reward_scale(sp) -> float:
    """`MIN_REWARD_NORM=shi` ⟹ `(max_ms − min_ms)`（原文 Eq.50–51 的归一化）。

    ⚠️ `reward_norm="none"` 时 `sp.bounds is None`（跳过 36 条规则扫描）⟹ 退回 100.0。
    """
    if os.environ.get("MIN_REWARD_NORM", "raw") != "shi":
        return 1.0
    if sp.bounds is None:
        return 100.0
    lo, hi = sp.bounds
    span = float(hi[0]) - float(lo[0])
    return span if span > 0 else 1.0


def _calibrate_feats(specs, nfs, rules=(0, 3, 8)):
    """**特征标准化统计**（用户选的第 1 项）：跑 `len(specs) × len(rules)` 局，
    收集每个决策点的 24 维原始特征，返回 `(mean, std, n_const, n_points)`。

    ⚠️ **固定统计量**（不随训练更新）⟹ 回放记忆里的特征口径前后一致。
    ⚠️ 用 3 条规则覆盖好/中/差的状态分布；特征只依赖状态，不依赖动作。
    """
    feats: list[np.ndarray] = []
    for sp, nfi in zip(specs, nfs):
        for k in rules:
            def _pol(view, pool, tie_rng=None, _k=k, _nf=nfi):
                feats.append(np.asarray(_nf.pa(view.snap, 0.0, (5.0, 5.0)),
                                        dtype=float))
                return apply_pa_rule(_k, view, pool, tie_rng)
            sp.world().run_gated(seed_chain=701_000, policy_pa=_pol,
                                 policy_l=la_rule_policy(0), dt_events=0)
    arr = np.asarray(feats, dtype=float)
    raw_sd = arr.std(axis=0)
    n_const = int(np.sum(raw_sd <= 1e-8))
    mu = arr.mean(axis=0)
    sd = np.where(raw_sd > 1e-8, raw_sd, 1.0)
    return mu, sd, n_const, len(feats)


def build_spec(it):
    """训练池 / 评测实例的统一 `build`（`inst_obj=it`）。"""
    kw: dict = dict(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="none",
                    tau=2.70, sim_overrides=_sim_over(), inst_obj=it,
                    shi2025=SHI2025)
    if SHOP_LEN is not None:
        kw.update(dims_from_shop_len(SHOP_LEN, it.n_machines, layout_cols=5),
                  layout_rows=5, layout_scheme="random")
    if CELL > 0.0:
        kw.update(cell_w=CELL, cell_h=CELL, aisle_w=AISLE,
                  lu_connector_m=LU, layout_scheme=SCHEME)
        # ⚠️ **不传 `layout_cols/layout_rows`**：`build` 内部的 `grid_shape(n_machines)`
        #    自己算最接近正方形的行列（18 台 ⟹ 5 列 × 4 行）。
    return build(**kw)[0]


class RollK:
    """与 `M.Roll` 同形，动作空间缩到 K 条规则。`eps > 0` 时按原文做 ε-greedy。

    ⚠️ **局末必须调 `finish(makespan)`** —— 否则回报漏掉整条尾巴（见 `finish`）。
    """

    def __init__(self, net, spec, nf, greedy=False, eps: float = 0.0, on_push=None,
                 reward_scale: float = 1.0, feat_mu=None, feat_sd=None):
        self.net, self.nf, self.greedy, self.eps = net, nf, greedy, eps
        # ⚠️ **奖励归一化**（2026-10-17）：`MIN_REWARD_NORM=shi` ⟹ `(max_ms − min_ms)`。
        self.reward_scale = float(reward_scale)
        # ⚠️ **特征标准化**（2026-10-17，用户选的第 1 项）：固定 μ/σ，`None` = 不标准化。
        self.feat_mu, self.feat_sd = feat_mu, feat_sd
        self.buf, self._cm = [], None
        self._f, self._a, self._lp, self._v = None, 0, 0.0, 0.0
        self.first_cm = None
        # ⚠️ **回放记忆钩子**（2026-10-14）：`on_push(entry, v_next, terminal)`
        self.on_push = on_push

    @torch.no_grad()
    def _pick(self, f):
        ft = torch.as_tensor(f)[None]
        if self.greedy:
            d = self.net.dist(ft)
            a = int(d.logits.argmax())
        else:
            # ⚠️ **采样与 log-prob 必须同源**（2026-10-13 修）：ε>0 时用混合分布 `b`。
            d = self.net.behavior(ft, self.eps)
            a = int(d.sample())
        return a, float(d.log_prob(torch.tensor(a))), float(self.net.value(ft)[0])

    def finish(self, makespan: float) -> None:
        """**局末收尾**：补上最后一个决策的奖励（见 `M.Roll.finish`，§37.44）。"""
        if self._cm is None:
            return
        self.buf.append((self._f, self._a, self._lp, self._v,
                         -(float(makespan) - self._cm) / self.reward_scale))
        self._cm = None
        # ⚠️ `terminal=True` ⟹ 记忆用它当**段界**（不许跨局拼接）。
        if self.on_push is not None:
            self.on_push(self.buf[-1], 0.0, True)

    def policy_pa(self, view, pool, tie_rng=None):
        snap = view.snap
        cm = M.cm_of(snap)                     # ⚠️ 口径由 `MIN_CM` 选（`bound` / `plan`）
        raw = self.nf.pa(snap, 0.0, (5.0, 5.0))
        if self.feat_mu is not None:
            # ⚠️ **必须转回 float32**：`mu/sd` 是 float64，直接送进网络会
            #    `mat1 and mat2 must have the same dtype`（2026-10-17 实测踩过）。
            raw = ((np.asarray(raw, dtype=np.float64) - self.feat_mu)
                   / self.feat_sd).astype(np.float32)
        if self.first_cm is None:
            self.first_cm = cm
        if self._cm is not None:
            self.buf.append((self._f, self._a, self._lp, self._v,
                             -(cm - self._cm) / self.reward_scale))
        self._cm = cm
        self._f = raw
        self._a, self._lp, self._v = self._pick(self._f)
        # ⚠️ **记忆推进在 `_pick` 之后** —— 此刻 `self._v` = `V(S_{t+1})`，
        #    正是刚写进去那条 transition 的切片末端代入值。
        if self.on_push is not None and len(self.buf) >= 2:
            self.on_push(self.buf[-1], self._v, False)
        return apply_pa_rule(ACT[self._a], view, pool, tie_rng)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(0, (0.0,))(view, cand_v))


def main() -> None:
    # ⚠️ **2026-10-14 删掉 `MIN_STEPS` / `MIN_CHAIN`** —— 旧循环的旋钮，新循环不用。
    #   新旋钮是 `MIN_EPOCHS` / `MIN_MINIB`（见下）。
    # ⚠️ **实例规模旋钮**（2026-10-13）：`Exp(16/140)` 是 **16 工件 / 18 台机**，
    # **工件比机台还少 ⟹ 车间不拥挤 ⟹ 派工规则没什么可分**。
    # 原文 Table 11 用的是 `Exp(50/425)`…`Exp(90/801)`（50–90 工件）。
    N_TOT = int(os.environ.get("MIN_N", "140"))
    JOBS = int(os.environ.get("MIN_JOBS", "16"))
    # ⚠️ **跨实例训练**（2026-10-13）：原文 Algorithm 1 第 2 步是**每个 epoch 随机生成
    # 工件与故障信息初始化环境**；本仓此前**固定一个实例**。
    # `MIN_NINST=N>1` ⟹ 预建 N 个实例，训练时按 step 轮换。
    # ⚠️ **为什么预建而不是每步重建**：实测 `build(spec)` = **12.4 s**（它要跑一次参考解），
    # 而**一局仿真只要 0.057 s** ⟹ 每步重建会把成本放大约 55 倍（300 步 = 62 分钟）。
    # 预建 N 个只付 N × 12.4 s。**参照表与评测一律用实例 0，不动。**
    n_inst = int(os.environ.get("MIN_NINST", "20"))
    # ⚠️ **默认已改成 20**（2026-10-14）：原文 Algorithm 1 第 2 步是「每个 epoch 随机生成
    #   工件与故障相关信息初始化环境」。真按 1000 个生成 = **3.4 小时/seed**（`build` = 12.4 s）
    #   ⟹ 本仓预建 20 个**轮换**，是**近似**（登记在案）。
    # ⚠️ **实测（§37.47）：20 个轮换 vs 单实例，读数「逐位相同」** —— 这条没效果。
    #   但要「对齐论文」就得开着。嫌慢可传 `MIN_NINST=1`。
    _ik = _inst_kw()
    if VARY:
        _vr = np.random.default_rng([20261018, SEED])
        insts = []
        for i in range(n_inst):
            jobs_i = int(_vr.integers(JOBS_LO, JOBS_HI + 1))
            insts.append(make_shi_instance(
                jobs_i, seed=POOL_STRIDE * (20_000 + i),
                rot_seed=PT_REDRAW_SEED + 20_000 + i,
                alts_dist="uniform",
                alts_min=ALTS_MIN if ALTS_MIN is not None else 1,
                alts_max=ALTS_MAX if ALTS_MAX is not None else 10,
                pt_dist="uniform", pt_lo=2, pt_hi=10))
    else:
        insts = [pooled(N_TOT, JOBS, i, **_ik) for i in range(n_inst)]
    specs = [build_spec(it) for it in insts]
    nfs = [NHFeatures(it, sp.layout,
                      speed_m_per_min=sp.cfg.agv_speed_mps * 60.0)
           for it, sp in zip(insts, specs)]
    # ⚠️ **特征标准化**（2026-10-17）：固定 μ/σ，来自训练池的 3 条规则 × 20 实例。
    feat_mu = feat_sd = None
    if FEAT_STD:
        feat_mu, feat_sd, n_const, n_pts = _calibrate_feats(specs, nfs)
        print(f"  特征标准化：校准 {len(specs) * 3} 局 · **{n_pts} 个决策点** · "
              f"常数维 **{n_const}/24** · |μ| 中位 {np.median(np.abs(feat_mu)):.3f} · "
              f"σ 中位 {np.median(feat_sd):.3f}", flush=True)
    # ⚠️⚠️ **2026-10-14 修 train-on-test**：此前评测用 `specs[0]`，而轮换 `si = ep % n_inst`
    #    让实例 0 进了训练池（每 20 个 epoch 一次）⟹ **在训练过的实例上评测**。
    #    改用 `eval_instance`，并**断言它不在池里**。见 `open-question-nhdrl.md` §4.7.2。
    ev_it = eval_instance(N_TOT, JOBS, n_inst, **_ik)
    spec = build_spec(ev_it)
    nf = NHFeatures(ev_it, spec.layout,
                    speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
    assert all(sig(it) != sig(ev_it) for it in insts), "评测实例进了训练池"
    print(f"  实例池：**{n_inst}** 个（真工序 "
          + "/".join(f"{sum(len(o) for o in it.jobs)}" for it in insts[:6])
          + ("…" if n_inst > 6 else "") + "）"
          + f"　评测实例**不在池里**（真工序 {sum(len(o) for o in ev_it.jobs)}）", flush=True)
    print(f"  环境：shop_len(旧标定)={SHOP_LEN} · speed={SPEED} · h={HLO}–{HHI} · "
          f"alts={ALTS_MIN}–{ALTS_MAX}（uniform）· shi2025={SHI2025} · "
          f"layout={SCHEME} {spec.layout.grid.n_rows}×{spec.layout.grid.n_cols} · "
          f"critic_tanh={M.CRITIC_TANH} · "
          f"reward_norm={os.environ.get('MIN_REWARD_NORM', 'raw')}"
          f"（scale={_reward_scale(spec):.1f}）· feat_std={FEAT_STD}", flush=True)
    scs = [100_000 + 13 * i for i in range(20)]

    # ── 基线①：**本档**（policy 够得到的集合 = `ACT` × `MINCOST`，因为 `policy_l` 钉死）──
    ref = {}
    for k in ACT:
        ref[k] = float(np.mean([
            float(spec.world().run_gated(
                seed_chain=sc,
                policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, tr),
                policy_l=la_rule_policy(0), dt_events=0)["makespan"])
            for sc in scs]))
    best = min(ref, key=lambda k: ref[k])

    # ── 基线②：**原文口径的 CDR** = 9 PA × 3 LA = **27 条** ──
    # ⚠️⚠️ **2026-10-14 加**：此前只报基线①却印成「本档最好 CDR」，而**判据用的是它**。
    #    原文的 CDR 是 *"27 CDRs generated by **pairwise combinations** of PA's and LA's
    #    actions"*（PAGE 16）。**实测（grid · scale 1）：① 147.6 而 ② 136.0 —— 差 7.8%。**
    g_ref: dict[tuple[int, int], float] = {}
    for _k in range(N_AEI2025_RULES):
        for _j in range(len(LA_RULE_NAMES)):
            g_ref[(_k, _j)] = float(np.mean([
                float(spec.world().run_gated(
                    seed_chain=sc,
                    policy_pa=lambda v, p, tr, k=_k: apply_pa_rule(k, v, p, tr),
                    policy_l=la_rule_policy(_j), dt_events=0)["makespan"])
                for sc in scs]))
    gbest = min(g_ref, key=lambda t: g_ref[t])
    print(f"**K = {K}** 条规则 · Exp({JOBS}/{N_TOT}) · 20 场景均值 · "
          f"随机基线 {1 / K:.1%} · 训练 seed {SEED}\n")
    print("  ── 基线① 本档（`ACT` × `MINCOST`，= policy 够得到的集合）──")
    for k in sorted(ACT, key=lambda k: ref[k]):        # ⚠️ 按读数排序打印，不按 RANK 顺序
        print(f"    {PA_RULE_NAMES[k]:<12}{ref[k]:>8.1f}"
              + ("　← 本档最好" if k == best else ""))
    print("\n  ── 基线② 原文口径的 27 条 CDR（9 PA × 3 LA）MS 升序前 6 ──")
    for t in sorted(g_ref, key=lambda t: g_ref[t])[:6]:
        print(f"    {PA_RULE_NAMES[t[0]] + '+' + LA_RULE_NAMES[t[1]]:<28}{g_ref[t]:>8.1f}"
              + ("　← **全局最好**" if t == gbest else ""))
    print(f"\n  ⚠️ 基线① 最好 {ref[best]:.1f}　基线② 最好 {g_ref[gbest]:.1f}"
          f"　差 {(g_ref[gbest] - ref[best]) / ref[best]:+.1%}"
          f"　⟹ **判据一律用基线②**（原文口径）\n", flush=True)

    # ⚠️ **2026-10-14：训练循环整个换成原文 Algorithm 1 + Table 6 的形状。**
    #   旧形状：`MIN_STEPS` 个 step × `MIN_CHAIN` 条链 ⟹ 每 step 一批 **492** 个样本、更新 1 次。
    #   新形状：`MIN_EPOCHS` 个 epoch，**每 epoch = 1 局**（原文第 2 步「每个 epoch 随机生成
    #   工件与故障相关信息初始化环境」）⟹ 每局按 **minibatch = 32** 切着更新。
    #   ⚠️ **一处刻意的偏差**：原文「攒满 32 个就更新、用完即弃」，本仓**先跑完整局、
    #   算完 GAE 再切 32** —— 见 `M.update_mb` 的 docstring（切片后各自算 GAE 会
    #   把 §37.44.1 那个假终止 bug 搬回来）。**更新次数与有效批大小与原文一致。**
    epochs = int(os.environ.get("MIN_EPOCHS", "1000"))
    mb = int(os.environ.get("MIN_MINIB", "32"))         # 原文 Table 6：minibatch = 32

    torch.manual_seed(SEED)
    net = M.Head(M.F_PA)
    net.pi[2] = torch.nn.Linear(M.HID_A, K)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    bi = ACT.index(best)

    # ⚠️ **回放记忆**（2026-10-14）：照原文 Algorithm 1 第 3–4 步 —— 逐决策点把 transition
    #   存进记忆，**攒满 `bs` 就更新、用完即清**。容量 = `bs` = 32（Table 6 的 `C_high`；
    #   `citation-cards.md:1606` 写明这个设置是为了让**上层 agent 近似在线**训练）。
    # ⚠️ **与旧口径的差别**：旧的是「跑完整局 → 整局算 GAE → 局末切 32 更新」；
    #   新的是「**局内**每 32 个决策点更新一次」。
    # ⚠️ **切片末端的 GAE 代入值必须传 `V(S_{t+1})`**（`M.Memo` 逐条绑好了）——
    #   局内切片不是终止，传 0 就是 §37.44.1 那个假终止 bug。
    memo = M.Memo(bs=mb)
    n_upd = 0
    cur = {"eps": 0.0}

    def on_push(entry, v_next, terminal) -> None:
        """写一条进记忆；满了就取一批更新；局末把余下的那一段冲掉。"""
        nonlocal n_upd
        memo.push(entry, v_next)
        while memo.full():
            seg, boot = memo.take()
            M.update_mb(net, seg, opt, mb=mb, eps=cur["eps"], boot=boot)
            n_upd += 1
        if terminal:                       # ⚠️ 段界：不许跨局拼接
            seg, boot = memo.flush()
            if len(seg) >= 2:
                M.update_mb(net, seg, opt, mb=mb, eps=cur["eps"], boot=boot)
                n_upd += 1

    for ep in range(1, epochs + 1):
        # ⚠️ ε 的步进单位 = **epoch**（原文 Algorithm 1 第 5 步在每个 epoch 末更新 ε）
        eps = 0.0 if os.environ.get("MIN_EPS", "1") == "0" else M.eps_at(ep)
        cur["eps"] = eps
        si = ep % n_inst if n_inst > 1 else 0     # ⚠️ 跨实例：按 epoch 轮换
        sp, nfi = specs[si], nfs[si]
        r = RollK(net, sp, nfi, eps=eps, on_push=on_push,
                  reward_scale=_reward_scale(sp),
                  feat_mu=feat_mu, feat_sd=feat_sd)
        m = sp.world().run_gated(seed_chain=9_000_000 + ep,
                                 policy_pa=r.policy_pa,
                                 policy_l=r.policy_l, dt_events=0)
        r.finish(float(m["makespan"]))                    # ⚠️ 补终局（§37.44）
        if ep == 1:                                       # 一次性自检
            _s = sum(b[4] for b in r.buf)
            _t = (r.first_cm - float(m["makespan"])) / _reward_scale(sp)
            assert np.isclose(_s, _t, atol=1e-6), \
                f"回报未覆盖终局：Σr={_s:.4f} vs C₀−M={_t:.4f}"
        if ep % max(1, epochs // 6) == 0 or ep == 1:
            c = Counter(b[1] for b in r.buf)
            n = sum(c.values())
            print(f"  epoch {ep:>4}/{epochs}　ε {eps:.3f}　本局 {n} 点　"
                  f"选本档最好那条 {c.get(bi, 0) / n:6.1%}"
                  f"　makespan {float(m['makespan']):.1f}　更新 {n_upd} 次", flush=True)

    net.eval()
    cnt, ms_pol = Counter(), []
    _pre = []
    for sc in scs:
        r = RollK(net, spec, nf, greedy=True, reward_scale=_reward_scale(spec),
                  feat_mu=feat_mu, feat_sd=feat_sd)
        m = spec.world().run_gated(seed_chain=sc, policy_pa=r.policy_pa,
                                   policy_l=r.policy_l, dt_events=0)
        r.finish(float(m["makespan"]))          # ⚠️ 否则最后一个决策不入 buf
        cnt.update(b[1] for b in r.buf)
        ms_pol.append(float(m["makespan"]))
        if r.buf:
            f = torch.as_tensor(np.stack([b[0] for b in r.buf]), dtype=torch.float32)
            with torch.no_grad():
                # ⚠️ **`Tanh` 之前**那层（下标 2 = 最后一个 `Linear`）
                _pre.append(net.pi[2](net.pi[1](net.pi[0](f))).numpy())

    # ⚠️ **两道闸一起看**（`Tanh` 于 2026-10-14 加回，见 §37.48）：
    #   `|pre|` = `Tanh` 之前的幅度（判据 `≥3` ⟹ `1−tanh² ≤ 0.01` ⟹ 梯度被掐）
    #   `p_max` = `Tanh` 之后 `softmax` 的最大概率（**上界 = `e/(e+(K−1)/e)`**）
    if _pre:
        raw = np.concatenate(_pre)
        pre = np.abs(raw).ravel()
        with torch.no_grad():
            pmax = torch.softmax(torch.tanh(torch.as_tensor(raw)), -1)\
                .max(1).values.numpy()
        print(f"  `Tanh` 之前 |pre|：中位 {np.median(pre):.2f}"
              f"　**≥3 的占比 {(pre >= 3).mean():.1%}**"
              f"　|　`tanh` 之后 softmax 最大概率：中位 {np.median(pmax):.3f}"
              f"　上界 {np.e / (np.e + (K - 1) / np.e):.3f}（均匀 {1 / K:.3f}）")
    n = sum(cnt.values())
    share = cnt.get(bi, 0) / n
    mean_ms = float(np.mean(ms_pol))
    gtag = f"{PA_RULE_NAMES[gbest[0]]}+{LA_RULE_NAMES[gbest[1]]}"
    # ⚠️ **判据改用基线②**（原文口径的 27 条）。基线① 只作诊断。
    ok_bin = mean_ms <= ref[best] * 1.01
    ok_all = mean_ms <= g_ref[gbest] * 1.01
    print(f"\n  greedy：选本档最好那条 {share:.1%}　（随机 {1 / K:.1%}）")
    # ⚠️ **必须看实际分布**（2026-10-13）：`{3,0}` 实测「选对率」只有 61.6%，
    # 而 makespan 146.3 已追平最好 CDR 147.6 —— 那 38.4% 的「没选中」大多落在
    # **选哪条都一样**的决策点上。**光看选对率会误判成败，要看它偏到哪条 + makespan。**
    print("  greedy 分布：" + "　".join(
        f"{PA_RULE_NAMES[ACT[k]]}:{v / n:.0%}（该规则纯用 {ref[ACT[k]]:.1f}）"
        for k, v in sorted(cnt.items(), key=lambda x: -x[1])))
    print(f"  策略 makespan **{mean_ms:.1f}**")
    print(f"    基线① 本档最好 {ref[best]:.1f}（{PA_RULE_NAMES[best]}）"
          f"　⟹ {'✅' if ok_bin else '❌'}"
          f"（{(mean_ms - ref[best]) / ref[best]:+.1%}）")
    print(f"    基线② 全局最好 {g_ref[gbest]:.1f}（{gtag}）"
          f"　⟹ {'✅' if ok_all else '❌'}"
          f"（{(mean_ms - g_ref[gbest]) / g_ref[gbest]:+.1%}）")
    print(f"  ⟹ **K={K}　对原文口径的 CDR：{'✅ 赢' if ok_all else '❌ 没赢'}**")


if __name__ == "__main__":
    main()
