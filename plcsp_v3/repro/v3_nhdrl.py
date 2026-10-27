# -*- coding: utf-8 -*-
"""**NHDRL 复现**（Shi et al., *Advanced Engineering Informatics* 65 (2025) 103195）。

══ 这个文件是什么 ══

**一版干净的复现，没有补丁开关。** 只有四个运行参数（`V3_A`/`V3_B`/`V3_STEPS`/
`V3_CHAIN`/`V3_CKPT`），其余全部**按原文写死**。

⚠️ **2026-10-12 重写**：此前本文件长出了 **12 个环境开关**，其中一半是为**已被实测证伪**的
变体加的（架构开关、三种奖励归一化、`cal` 的标定常数……）。那些变体的结论已登记在
`docs/v3/progress-log.md` §37，**代码里不再留痕迹** —— 留着的开关下一次还会误导人。

══ 照原文的哪几处 ══

| 项 | 出处 | 本文件 |
|---|---|---|
| 三个 agent：OA(2) → PA(9) → LA(3) | Fig. 3 | `NHDRL` |
| Actor：输入→ReLU→**256**→ReLU→输出→**Tanh**；Critic：输入→ReLU→**128**→ReLU→1 | **Table 4** | `ActorCritic` |
| 奖励 `R = w1·x + w2·(p/maxp)·x`，`x` 按 OA 选的目标取 MS 或 TC | **Eq. 50–53** | `RewardFn` |
| `ms_t` = *"the difference in MS before and after state transition"* | §4.2.2 | `cm_of` 的增量 |
| `TC = c·Σy_v + uc·Σ(nct−nst+lct−lst)` | **Eq. 2** | `tc_cum` |
| OA/PA/LA 的 lr = **1e-4 / 1e-4 / 1e-3**；γ = **0.99** | **Table 5** | `LR_BY_HEAD` |
| ε-greedy，ε 初值 **0.9**、下限 **0.1** | **Table 6** | `EPS0` / `EPS_LB` |
| clip **0.2** | Table 6 | `ppo_update` |
| 实例 `Exp(a/b)`：18 机、6 AGV、工序数 8/8/8/8/9/6/7、加工 2–10 | §5.1 | `env/shi_instances.py` |

══ ⚠️ 本文件**无法**照原文的三处（已登记，不是近似）══

1. **逐步物流账本**：原文 Eq.2 的 `Σ(nct−nst+lct−lst)` 是**逐任务**量。本仓的物流成本
   原本只有整局累计 ⟹ TC 支逐步恒为 0。**2026-10-12 用 `Snapshot.vehicles[v].busy_min`
   （累计空载+负载行驶工时）差分补上。**
2. **`tt_ll'`（两地间运输时间）原文没给数** —— 只说 *"fixed and known, as shown in Fig. 4"*。
   本仓用几何口径（走廊图最短路）。**这是假设，要写进论文。**
3. **布局**：原文是 Fig. 4，本仓用 `layout_seed` 采样。**T1 待办。**

⚠️ **训练量**：原文 Table 6 是 **1000 epoch**（Fig.5 显示 ~400 收敛）。
本文件的 `V3_STEPS` 是**步**，每步 = `V3_CHAIN` 局 ⟹ **要按原文跑就得给够步数。**
"""
import os
import sys

sys.path.insert(0, os.path.dirname(  # 仓根：本文件在 plcsp_v3/repro/ 下
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from plcsp_v3.env.des import reference_makespan  # noqa: E402
from plcsp_v3.env.nhdrl_features import F_LA, F_OA, F_PA, NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import (POOL_STRIDE,  # noqa: E402
                                        PT_REDRAW_SEED, eval_instance,
                                        make_shi_instance, pooled)
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES,  # noqa: E402
                                PA_RULE_NAMES, apply_pa_rule, la_rule_policy)
from shi2025_min_pa import Memo  # noqa: E402  回放记忆（原文 Alg.1 第 3–4 步）


def sig(inst) -> tuple:
    """实例指纹 —— 用于断言「评测实例不在训练池里」。"""
    return tuple(sorted((len(o), o[0][0], sum(t for _m, t in o))
                        for j in inst.jobs for o in j))

N_OA_ACT, N_PA_ACT, N_LA_ACT = 2, 9, 3
HID, HID_V = 256, 128                              # Table 4
LR_BY_HEAD = {"oa": 1e-4, "pa": 1e-4, "la": 1e-3}  # Table 5
GAMMA, LAM, CLIP = 0.99, 0.95, 0.2                 # Table 5 / Table 6
# ⚠️ **Critic 输出层加 `Tanh`**（照原文 Table 4）—— 默认 0 = v3 原行为。
CRITIC_TANH = os.environ.get("V3_CTANH", "0") == "1"
# ⚠️ **回报归一化标度**（2026-10-17）：原文 Critic 输出是 Tanh（Table 4），
#   但本仓 γ=0.99 下的回报 Σγ^t·r ≈ 79，纯 Tanh 装不下。
#   开 Tanh 时把奖励除以 `RET_SCALE`，让回报落在 (−1,1)；策略优势再乘回同一标度。
#   默认 1 ⟹ 不改旧行为；`V3_CTANH=1` 时由调用方给 100。
RET_SCALE = float(os.environ.get("V3_RET_SCALE", "1"))
# ⚠️ **训练分布**（2026-10-17，用户要求照原文）：原文每 epoch 随机生成实例，
#    工件数 `I ∈ [10,40]`。`V3_VARY=1` ⟹ 训练池按该分布生成（默认关 ⟹ 旧行为）。
VARY = os.environ.get("V3_VARY", "0") == "1"
JOBS_LO = int(os.environ.get("V3_JOBS_LO", "10"))
JOBS_HI = int(os.environ.get("V3_JOBS_HI", "40"))
JOB_SEED = int(os.environ.get("V3_JOB_SEED", "0"))
# ⚠️ **强制 OA**（2026-10-18，诊断用）：`-1` = 正常训练；`0` = 恒选 MS；`1` = 恒选 TC。
#    强制时**不写 OA 缓冲、不更新 OA**，只隔离 PA/LA。
FORCE_OA = int(os.environ.get("V3_FORCE_OA", "-1"))
# ⚠️ **强制 LA**（2026-10-18，诊断用）：`-1` = 正常；`0/1/2` = 固定 LA 规则。
FORCE_LA = int(os.environ.get("V3_FORCE_LA", "-1"))
# ⚠️ **固定尺度 MS 奖励**（2026-10-18，诊断）：`1` ⟹ MS 支用
#    `−d_ms / 该实例的 reference makespan`，**不走运行极值 min–max**。
#    理由：PA-only 成功用的是固定尺度 `−d/792.6`；变规模时 min–max 会把不同规模的
#    增量分布等权化，可能改变策略的最优动作。
MS_FIXED = os.environ.get("V3_MS_FIXED", "0") == "1"
# ⚠️ **优势归一化**（2026-10-18，诊断）：`1` ⟹ 每个更新批内的 ADV 做 z-score。
#    共享奖励 + 每 epoch 一个实例 ⟹ 优势方差大；原文 Eq.49 没写归一化，但这是 PPO 常规手段。
ADV_NORM = os.environ.get("V3_ADV_NORM", "0") == "1"
# ⚠️ **终局 shaping**（2026-10-18，照 v3b）：最后一 transition 加
#    `TERM_BONUS·(LB_MS/MS + LB_TC/TC)`；单位与奖励一致（÷RET_SCALE）。
TERM_BONUS = float(os.environ.get("V3_TERM_BONUS", "0"))
# ⚠️ **AGV 台数**（2026-10-18，用户要求可配置）：原文 6；默认 6 ⟹ 旧行为不变。
NAGV = int(os.environ.get("V3_NAGV", "6"))
# ⚠️ **v3b 训练配方**（2026-10-18，全部默认关/原文口径）：
OA_COMBINED = os.environ.get("V3_OA_COMBINED", "0") == "1"
ENT = float(os.environ.get("V3_ENT", "0"))
LOGIT_SCALE = float(os.environ.get("V3_LOGIT_SCALE", "1"))
# ⚠️ **部分 MS 代理是否含 AGV 运输**（2026-10-18，修 A）：默认 0 = 旧 MCT。
CM_AGV = os.environ.get("V3_CM_AGV", "0") == "1"
# ⚠️ **TC 归一化口径**（2026-10-18）：`1` ⟹ 论文 Eq.53 的**单任务解析上下界**：
#    `tc_min = uc·min(正行程)`、`tc_max = c + uc·2·max(行程)`。
#    默认 0 ⟹ 运行 min/max（旧 v3b 口径）。
TC_ANALYTIC = os.environ.get("V3_TC_ANALYTIC", "0") == "1"
W1, W2, MAXP = 0.8, 0.2, 10.0                      # Eq. 50/52
C_COMMISSION, UC_TRANSPORT = 7.0, 0.5              # Eq. 2 的 c 与 uc
EPS0, EPS_LB, EPS_DECAY = 0.9, 0.1, 0.99            # Table 6 + Alg.1 第 5 步

# ⚠️ **环境规格旋钮**（2026-10-17，与 `shi2025_min2.py` 同口径）：
ALTS_MIN = int(os.environ.get("V3_ALTSMIN", "1"))
ALTS_MAX = int(os.environ.get("V3_ALTSMAX", "10"))
SPEED = float(os.environ.get("V3_SPEED", "0.5"))
HLO = float(os.environ.get("V3_HLO", "0.0"))
HHI = float(os.environ.get("V3_HHI", "0.0"))
LAYOUT_COLS = int(os.environ.get("V3_COLS", "0")) or None
LAYOUT_ROWS = int(os.environ.get("V3_ROWS", "0")) or None
# ⚠️ **新布局口径**（2026-10-18）：显式机位/走廊/L-U；`V3_CELL=0` ⟹ 走 `shop_len`。
CELL = float(os.environ.get("V3_CELL", "0"))
AISLE = float(os.environ.get("V3_AISLE", "0"))
LU = float(os.environ.get("V3_LU", "0"))
# ⚠️ **偏好采样**（2026-10-18）：原文训练/评测都抽 p_MS ∈ 0..10。
#    默认 0 = 旧行为（固定 5/5）⟹ 既有读数不变；复现层显式传 1。
PREF_RANDOM = os.environ.get("V3_PREF_RANDOM", "0") == "1"
SHI = os.environ.get("V3_SHI", "1") == "1"
# ⚠️ **特征标准化**（2026-10-17）：三头各自 Z-score，固定 μ/σ。默认关 ⟹ 旧行为不变。
FEAT_STD = os.environ.get("V3_FSTD", "0") == "1"


def _inst_kw() -> dict:
    return {"alts_min": ALTS_MIN, "alts_max": ALTS_MAX, "alts_dist": "uniform"}


def _sim_over() -> dict:
    """仿真覆盖：**不覆盖 `agv_mttr_lo/hi`**（原文 5–10）。"""
    over: dict = {"due_range": 0.6}
    if SPEED is not None:
        over["agv_speed_mps"] = SPEED
    if HLO is not None:
        over["agv_handling_lo"] = HLO
    if HHI is not None:
        over["agv_handling_hi"] = HHI
    if SHI:
        over["agv_plan_pos"] = 1.0
        over["agv_act_book"] = 1.0
        # ⚠️ **同机台相邻工序省略搬运**（2026-10-17，照原文）。
        over["skip_same_machine_move"] = True
    return over


class ActorCritic(nn.Module):
    """照原文 **Table 4**：Actor `输入→ReLU→256→ReLU→输出→Tanh`；Critic `输入→ReLU→128→ReLU→1`。

    ⚠️ actor 输出层的 **Tanh 是原文逐字**（把 logits 压到 (−1,1)，softmax 后分布本来就平）。
    """

    def __init__(self, f_dim: int, n_act: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(f_dim, HID), nn.ReLU(),
                                   nn.Linear(HID, n_act), nn.Tanh())
        # ⚠️ **Critic 输出层**（`V3_CTANH=1`，2026-10-14）：
        #   原文 **Table 4 第三行**写 `1 | Tanh` ⟹ `V ∈ (−1, 1)`。
        #   ⚠️ **但本仓的 `RET`（累积回报）量级是 ~79**（`r ∈ [0,1]`、`γ=0.99`）——
        #   实测：单步奖励 mean 0.785、Σγ^t·r ≈ **79** ⟹ `Tanh` 装不下。
        #   **⟹ 加它是照字面，但会撞一个原文自己没写清的标定问题**（`RET` 的量级）。
        #   默认 `0` ⟹ v3 原行为（无 Tanh）。
        self.critic = (nn.Sequential(nn.Linear(f_dim, HID_V), nn.ReLU(),
                                     nn.Linear(HID_V, 1), nn.Tanh())
                       if CRITIC_TANH else
                       nn.Sequential(nn.Linear(f_dim, HID_V), nn.ReLU(),
                                     nn.Linear(HID_V, 1)))

    def dist(self, f: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor(f) * LOGIT_SCALE)

    def behavior(self, f: torch.Tensor, eps: float) -> torch.Tensor:
        """**行为分布** `b = (1−ε)·π + ε/K`（2026-10-14，`progress-log` §37.44.2）。

        ⚠️ **ε>0 时采样和 log-prob 都必须用 `b`，PPO 的 ratio 也必须用 `b`。**
        此前 `_pick` 采样走「ε 概率均匀随机」、却记 `log π`，而 `ppo_update` 又按 `π` 算 ratio
        ⟹ **样本来自 `b`，PPO 当成来自 `π`，口径不一致。**
        （同一个 bug 在 `shi2025_min_pa.py` 已修。）
        """
        p = torch.softmax(self.actor(f) * LOGIT_SCALE, -1)
        if eps <= 0.0:
            return p
        return (1.0 - eps) * p + eps / p.shape[-1]

    def value(self, f: torch.Tensor) -> torch.Tensor:
        return self.critic(f)[:, 0]


class NHDRL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.oa = ActorCritic(F_OA, N_OA_ACT)
        self.pa = ActorCritic(F_PA, N_PA_ACT)
        self.la = ActorCritic(F_LA, N_LA_ACT)

    def heads(self) -> dict[str, ActorCritic]:
        return {"oa": self.oa, "pa": self.pa, "la": self.la}


def cm_of(snap) -> float:
    """**部分调度的 MS 代理**。

    ⚠️ 原文 Eq.1 的 MS 含运输/回站；本仓原实现只取 `max(mct_assigned)`（机台完工），
    不含 AGV 运输。对齐探针实测：该代理与**真 MS 秩相关 −0.43**（越优化越差）。
    `V3_CM_AGV=1` ⟹ `max(max(MCT), max(agv_act))`，把派车计划完工（含空载+负载腿）
    也算进去。默认 0 ⟹ 旧行为逐位不变。

    ⚠️ **2026-10-18 优先用 `snap.ms_plan`**（已 assign 工序 + 回站段的预计完工最大值，
    与原文 Eq.50–51 的 `MS` 同义）；`ms_plan=0`（离线路径）才回退旧口径。
    """
    v = float(getattr(snap, "ms_plan", 0.0))
    if v > 0.0:
        return v
    mct = max((float(x) for x in snap.mct_assigned), default=0.0)
    if not CM_AGV:
        return mct
    act = max((float(x) for x in (snap.agv_act or ())), default=0.0)
    return max(mct, act)


def tc_cum(snap) -> float:
    """**累计物流成本的可变部分** = `uc · Σ_v AW_v(t)`（`Snapshot.vehicles[v].busy_min`）。

    照原文 **Eq.2**：`uc·Σ(nct−nst+lct−lst)` = 空载段 + 负载段的运输时长之和。
    `c·Σy_v`（启车成本）在车队恒全启用时是常数，不产生逐步梯度，故不进增量。

    ⚠️ **这是 2026-10-12 补的缺口**：此前 `_close` 把 `tc_step` 恒传 `0.0` ⟹
    **TC 支的奖励与状态、动作完全无关** ⟹ OA 一旦选 TC 就再也收不到回 MS 支的梯度。
    实测补上后：非零增量从 0% 升到 **62%**，不同取值从 **1 种**升到 **176–287 种**。
    """
    return UC_TRANSPORT * sum(float(v.busy_min) for v in snap.vehicles)


def tc_of(m: dict, n_agv: int) -> float:
    """**整局**物流成本（评测用）= `c·Σy_v + uc·Σ(空载 + 负载)`，照原文 Eq.2。"""
    st = m["energy_breakdown"]["agv_states_min"]
    return C_COMMISSION * n_agv + UC_TRANSPORT * (float(st["empty"]) + float(st["loaded"]))


def tc_bounds(spec) -> tuple[float, float]:
    """论文 Eq.53 的**单任务解析上下界**（与外部 `tc_min/tc_max` 同口径）。

    ```
    tc_min = uc · min(正行程)
    tc_max = c  + uc · 2 · max(行程)
    ```
    """
    w = spec.world()
    nodes = [int(m.dock_node) for m in spec.layout.machines]
    if spec.layout.lu is not None:
        nodes.append(int(spec.layout.lu.dock_node))
    legs = []
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            t = float(w.leg_min(a, b))
            if t > 1e-9:
                legs.append(t)
    if not legs:
        return 0.0, 1.0
    return (UC_TRANSPORT * min(legs),
            C_COMMISSION + UC_TRANSPORT * 2.0 * max(legs))


class RewardFn:
    """**Eq. 50–53**：`R = w1·x + w2·(p/maxp)·x`，`x` 按 OA 选的目标取 MS 或 TC。

    两支各自 **min–max 归一**到 `[0,1]`（原文 Eq. 51/53 用观测极值）。

    ⚠️ **归一化不是可选项 —— 它是 OA 有意义的前提**：两支不归一时，单步代价的量纲不同
    （实测 `|d_ms|` 均值 1.17 min、`|tc_step|/uc` 0.62 min，差 1.87 倍）⟹
    **OA 会去选"扣得少"的那一支，而那个选择与调度质量无关**。
    实测：不归一时两支回报差 **2.6 倍**、两支配成的 makespan 却几乎相同（247.4 vs 246.9）；
    归一到 `[0,1]` 后两支回报差 **6%**。**见 `progress-log` §37.19。**
    """

    def __init__(self) -> None:
        self.ms_lo, self.ms_hi = 0.0, 1e-6
        self.tc_lo, self.tc_hi = 0.0, 1e-6
        # ⚠️ **论文 Eq.53 的解析上下界**（`V3_TC_ANALYTIC=1` 时由 `Rollout` 填）。
        self.tc_min: float | None = None
        self.tc_max: float | None = None
        # ⚠️ **固定尺度**（`V3_MS_FIXED=1`）：由训练循环按实例的 reference makespan 填。
        self.ms_scale = 1.0

    def __call__(self, d_ms: float, tc_step: float, oa_ms: bool,
                 prefs: tuple[float, float]) -> float:
        """选中支的奖励。两支的极值都在 `both()` 里更新。"""
        r_ms, r_tc = self.both(d_ms, tc_step, prefs)
        return r_ms if oa_ms else r_tc

    def both(self, d_ms: float, tc_step: float,
             prefs: tuple[float, float]) -> tuple[float, float]:
        """→ `(R_MS, R_TC)`。两支极值都更新；`OA_COMBINED` 用 `0.5·(R_MS+R_TC)`。"""
        self.ms_lo, self.ms_hi = min(self.ms_lo, d_ms), max(self.ms_hi, d_ms)
        self.tc_lo, self.tc_hi = min(self.tc_lo, tc_step), max(self.tc_hi, tc_step)
        x_ms = (self.ms_hi - d_ms) / max(self.ms_hi - self.ms_lo, 1e-9)
        if self.tc_min is not None and self.tc_max is not None:
            # 论文 Eq.53：单任务解析上下界；夹到 [0,1]（同机台省略时 tc_step 可为 0）。
            x_tc = ((self.tc_max - tc_step)
                    / max(self.tc_max - self.tc_min, 1e-9))
            x_tc = min(1.0, max(0.0, x_tc))
        else:
            x_tc = (self.tc_hi - tc_step) / max(self.tc_hi - self.tc_lo, 1e-9)
        g_ms = W1 + W2 * (prefs[0] / MAXP)
        g_tc = W1 + W2 * (prefs[1] / MAXP)
        r_ms = (-float(d_ms) / max(self.ms_scale, 1e-9)
                if MS_FIXED else g_ms * x_ms / RET_SCALE)
        r_tc = g_tc * x_tc / RET_SCALE
        return r_ms, r_tc


class Rollout:
    """一局：逐决策点记 `(state, act, logp, value, reward)` —— **三个 agent 各一份**。

    OA/PA/LA 的调用次序照原文 Fig. 3 的决策传递：OA 先定目标 → PA 出派工 → LA 出派车。
    本仓仿真在**两个回调**里分别问 PA 与 LA，故 **OA+PA 放 `policy_pa`**、**LA 放 `policy_l`**，
    LA 复用最近一次 OA 的目标决策。
    """

    def __init__(self, net: NHDRL, spec, nf: NHFeatures, rew: RewardFn,
                 gen: torch.Generator, prefs=(5.0, 5.0), greedy: bool = False,
                 eps: float = 0.0, on_push=None, feat_stats=None,
                 force_oa: int = -1, force_la: int = -1) -> None:
        self.net, self.nf = net, nf
        self.rew, self.gen, self.prefs, self.greedy = rew, gen, prefs, greedy
        if TC_ANALYTIC:
            rew.tc_min, rew.tc_max = tc_bounds(spec)
        self.eps = float(eps)
        # ⚠️ **强制 OA / LA**（-1 = 不强制）；强制时不写对应缓冲。
        self.force_oa = int(force_oa)
        self.force_la = int(force_la)
        # ⚠️ **三头各自的特征标准化**（2026-10-17）：`{key: (mu, sd)}`，`None` = 不标准化。
        self.feat_stats = feat_stats or {}
        self.spec = spec
        # ⚠️ **终局 shaping 的下界**（v3b）：LB_MS = max_job Σ min(pt)；
        #    LB_TC = c + uc·Σ_jobs（L/U→首工序机台 + 末工序机台→L/U 的最短腿）。
        self.lb_ms = max(
            sum(min(float(t) for _m, t in op) for op in job)
            for job in spec.inst.jobs)
        _lu = int(spec.layout.grid.lu_dock_node)
        _s = 0.0
        for job in spec.inst.jobs:
            _f = min(self.nf._leg(_lu, spec.layout.machines[int(m)].dock_node)
                     for m, _t in job[0])
            _l = min(self.nf._leg(spec.layout.machines[int(m)].dock_node, _lu)
                     for m, _t in job[-1])
            _s += _f + _l
        self.lb_tc = 7.0 + 0.5 * _s
        # ⚠️ **回放记忆钩子**（2026-10-14）：`on_push(key, entry, v_next, terminal)`
        self.on_push = on_push
        self.buf: dict[str, list] = {"oa": [], "pa": [], "la": []}
        self._cm, self._tc, self._prev, self._la_pend = None, None, None, None
        self._oa_last, self._job, self._mach = 0, 0, 0

    def _std(self, key: str, f) -> np.ndarray:
        """按训练池校准的 `μ/σ` 标准化该头的特征（转回 float32）。"""
        st = self.feat_stats.get(key)
        if st is None:
            return f
        mu, sd = st
        return ((np.asarray(f, dtype=np.float64) - mu) / sd).astype(np.float32)

    @torch.no_grad()
    def _pick(self, key: str, f: np.ndarray):
        """⚠️ **采样与 log-prob 同源**（2026-10-14，§37.44.2）：ε>0 时都走 `b = (1−ε)π + ε/K`。

        ⚠️ 仍用**每链自己的 `self.gen`**（不走 torch 全局 RNG），否则同一步各链采样同源
        ⟹ 组内方差为 0（§37.37.2）。
        """
        head = self.net.heads()[key]
        ft = torch.as_tensor(f)[None]
        p = head.behavior(ft, self.eps)[0]
        v = float(head.value(ft)[0])
        a = (int(head.actor(ft)[0].argmax()) if self.greedy
             else int(torch.multinomial(p, 1, generator=self.gen)))
        return a, float(torch.log(p[a].clamp_min(1e-12))), v

    def policy_pa(self, view, pool, tie_rng=None):
        snap = view.snap
        cm, tcx = cm_of(snap), tc_cum(snap)
        oa_a, oa_lp, oa_v = self._pick("oa", self._std("oa", self.nf.oa(snap, self.prefs)))
        if self.force_oa >= 0:
            oa_a = int(self.force_oa)
        self._oa_last = oa_a
        pa_a, pa_lp, pa_v = self._pick(
            "pa", self._std("pa", self.nf.pa(snap, float(oa_a), self.prefs)))
        # ⚠️ **先取新值、再收尾上一步**（2026-10-14 调序）：`_close` 要把新状态的价值
        #    `V(S_{t+1})` 一并交给回放记忆 —— 它是**局内切片**末端 GAE 的代入值。
        #    原来 `_close` 在 `_pick` **之前**调 ⟹ 那个值还没算出来。
        if self._cm is not None:
            self._close(cm, tcx, oa_v, pa_v)
        self._cm, self._tc = cm, tcx
        self._prev = (snap, oa_a, oa_lp, oa_v, pa_a, pa_lp, pa_v)
        # ⚠️ **2026-10-17 修 bug**：LA 的状态必须用 **PA 刚选中的 (工件, 机台)**。
        #    此前写的是 `self._job = pool[0][0]` + `policy_l` 里 **机台恒传 0**
        #    ⟹ LA 的 f33–f48（任务相关特征）**全部算错**。
        _job, _mach = apply_pa_rule(pa_a, view, pool, tie_rng)
        self._job, self._mach = int(_job), int(_mach)
        return _job, _mach

    def policy_l(self, view, cand_v):
        # ⚠️ **LA 的状态在这里才算得出来**（它要 PA 刚选中的 (工件, 机台)）⟹ 连状态一起存，
        # 否则 `_close` 时拿不到，`ppo_update` 会把整条 LA 缓冲跳过（实测 loss 恒 0）。
        f_la = self._std("la", self.nf.la(view.snap, self._job, self._mach,
                                          float(self._oa_last), self.prefs))
        la_a, la_lp, la_v = self._pick("la", f_la)
        if self.force_la >= 0:
            la_a = int(self.force_la)
        self._la_pend = (f_la, la_a, la_lp, la_v)
        return int(la_rule_policy(la_a, (0.0,))(view, cand_v))

    def _close(self, cm: float, tc: float, v_oa: float, v_pa: float,
               v_la: float = 0.0, terminal: bool = False) -> None:
        """收尾上一步：算奖励、写三份缓冲，并把 transition 推给回放记忆。

        `v_oa` / `v_pa` / `v_la` = **下一步三个 head 各自的价值 `V(S_{t+1})`** ——
        局内切片末端 GAE 的代入值（见 `M.Memo`）。`terminal=True` = 局末（代入值 0）。
        """
        if self._prev is None:
            return
        (_s, oa_a, oa_lp, oa_v, pa_a, pa_lp, pa_v) = self._prev
        r = self.rew(cm - self._cm, tc - self._tc, oa_a == 0, self.prefs)
        if terminal and TERM_BONUS > 0.0 and cm > 0.0:
            # ⚠️ **终局 shaping**（v3b）：只加在最后一 transition 上。
            _tc_full = float(tc) + 7.0 * float(self.spec.cfg.n_agv)
            r += TERM_BONUS * (self.lb_ms / cm
                               + self.lb_tc / max(_tc_full, 1e-9)) / RET_SCALE
        self.buf["pa"].append((self._std("pa", self.nf.pa(_s, float(oa_a), self.prefs)),
                               pa_a, pa_lp, pa_v, r))
        if self.force_oa < 0:
            self.buf["oa"].append((self._std("oa", self.nf.oa(_s, self.prefs)),
                                   oa_a, oa_lp, oa_v, r))
        if self.on_push is not None:
            if self.force_oa < 0:
                self.on_push("oa", self.buf["oa"][-1], v_oa, terminal)
            self.on_push("pa", self.buf["pa"][-1], v_pa, terminal)
        if self._la_pend is not None:
            f_la, la_a, la_lp, la_v = self._la_pend
            if self.force_la < 0:
                self.buf["la"].append((f_la, la_a, la_lp, la_v, r))
                if self.on_push is not None:
                    self.on_push("la", self.buf["la"][-1], v_la, terminal)
            self._la_pend = None
        self._prev = None

    def finish(self, snap, makespan: float) -> None:
        """**局末收尾**（2026-10-14，`progress-log` §37.44.1）。`snap` = 终局快照。

        `_close` 只在**下一次** `policy_pa` 被调用时才执行 ⟹ **最后一次没有「下一次」**
        ⟹ 它和它之后的收成（加工 / 运输 / 回站）**永远不进回报**。
        （复现侧同病实测：末值 `104.5` vs 真 makespan `157.3` —— **差 52.8 分钟，占 34%**。）

        ⚠️ 终局的 `cm` 用**真 makespan**，不用 `cm_of(snap)` —— §37.43.3 已证
        `cm_of` 在最后一个决策点之后收不到那一段。
        """
        if self._prev is None:
            return
        # ⚠️ 终局：三个 head 的代入值都是 **0**（真终止），且 `terminal=True` ⟹
        #    记忆用它当**段界**，不许跨局拼接。
        self._close(float(makespan), tc_cum(snap), 0.0, 0.0, 0.0, terminal=True)


def _gae(rew: list, val: list, boot: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """GAE。⚠️ `boot` = 序列末端那个状态的代入值 `V(S_{T+1})`（2026-10-14 加）。

    - **整局** ⟹ 末端是真终止（`finish()` 已补到真完工时刻）⟹ `boot = 0.0` ✅
    - **局内切片**（回放记忆攒满 32 就更新）⟹ 末端**不是**终止 ⟹ **必须传 `V(S_{T+1})`**，
      否则每片都成了假终止（§37.44.1）。
    """
    adv, g = np.zeros(len(rew)), 0.0
    v = list(val) + [float(boot)]
    for t in reversed(range(len(rew))):
        d = rew[t] + GAMMA * v[t + 1] - v[t]
        g = d + GAMMA * LAM * g
        adv[t] = g
    return adv, adv + np.asarray(val)


def ppo_update(head: ActorCritic, buf: list, optim, eps: float = 0.0,
               mb: int = 32, boot: float = 0.0, shuffle: bool = False) -> float:
    """PPO-clip + GAE。`buf` = `[(state, a, logp_old, v_old, r)]`。**值函数系数 vc = 0.5。**

    ⚠️ **`eps` 必须与采样时相同**（2026-10-14，§37.44.2）：ratio 的分母 `LP0` 记自 `b`，
    分子就也得是 `b`。按 `π` 算 ratio 就是拿在策略的式子去解释离策略样本。

    ⚠️ **`mb` = minibatch 大小，照原文 Table 6 = 32**（2026-10-14 加）。
    此前一批用**整条链**（8 链拼起来 ≈ 992）。
    **GAE 仍在整局上算** —— 切片只切 PPO 更新的粒度。
    理由：`_gae()` 末端按 `V = 0` bootstrap，**整局末端是真终止**（`finish()` 已把回报
    补到真完工时刻），`V=0` 对；**若先按 32 切、各自算 GAE，每片末端都成了假终止
    ⟹ 把 §37.44.1 那个 bug 搬回来。**
    """
    rows = [b for b in buf if b[0] is not None and len(b) == 5]
    if len(rows) < 2:
        return 0.0
    S = torch.as_tensor(np.stack([r[0] for r in rows]), dtype=torch.float32)
    A = torch.as_tensor([r[1] for r in rows], dtype=torch.long)
    LP0 = torch.as_tensor([r[2] for r in rows], dtype=torch.float32)
    adv, ret = _gae([r[4] for r in rows], [r[3] for r in rows], boot=boot)
    ADV = torch.as_tensor(adv, dtype=torch.float32) * RET_SCALE
    RET = torch.as_tensor(ret, dtype=torch.float32)
    return _ppo_apply(head, S, A, LP0, ADV, RET, optim, eps, mb, shuffle)


def _ppo_apply(head: ActorCritic, S, A, LP0, ADV, RET, optim,
               eps: float, mb: int, shuffle: bool) -> float:
    """在**已算好优势/回报**的样本上做 PPO-clip；`shuffle` 控制随机 minibatch。"""
    n = len(A)
    if n < 2:
        return 0.0
    if ADV_NORM:
        # ⚠️ **先归一化优势，再打乱/切 minibatch**（2026-10-18，诊断）。
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    if shuffle:
        perm = torch.randperm(n)
        S, A, LP0 = S[perm], A[perm], LP0[perm]
        ADV, RET = ADV[perm], RET[perm]
    out = 0.0
    for _ in range(4):
        tot = 0.0
        for i in range(0, n, mb):
            sl = slice(i, i + mb)
            d = torch.distributions.Categorical(probs=head.behavior(S[sl], eps))
            ratio = torch.exp(d.log_prob(A[sl]) - LP0[sl])
            pi = -torch.min(ratio * ADV[sl],
                            torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ADV[sl]).mean()
            loss = pi + 0.5 * ((head.value(S[sl]) - RET[sl]) ** 2).mean()
            if ENT > 0.0:
                # ⚠️ **熵奖励**（2026-10-18 补实现）：`ENT` 此前只定义未使用 ⟹
                #    `V3_ENT=0.05` 实际没有生效。这里按外部 v3b 加 `−ENT·H(b)`。
                loss = loss - ENT * d.entropy().mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tot += float(loss.detach())
        out = tot / max(1, (n + mb - 1) // mb)
    return out


def ppo_update_segments(head: ActorCritic, segs: list, optim, eps: float = 0.0,
                        mb: int = 32, shuffle: bool = True) -> float:
    """**跨局 `C_low=500` 回放**：按局分别算 GAE，拼起来再随机抽 minibatch。

    ⚠️ GAE **不能跨局递推** —— 每段用自己最后一条的 `v_next` 当 bootstrap。
    """
    S, A, LP0, ADV, RET = [], [], [], [], []
    for seg, boot in segs:
        if not seg:
            continue
        adv, ret = _gae([r[4] for r in seg], [r[3] for r in seg], boot=boot)
        for r, a_, t_ in zip(seg, adv, ret):
            S.append(r[0])
            A.append(r[1])
            LP0.append(r[2])
            ADV.append(float(a_) * RET_SCALE)
            RET.append(float(t_))
    if len(A) < 2:
        return 0.0
    return _ppo_apply(head,
                      torch.as_tensor(np.stack(S), dtype=torch.float32),
                      torch.as_tensor(A, dtype=torch.long),
                      torch.as_tensor(LP0, dtype=torch.float32),
                      torch.as_tensor(ADV, dtype=torch.float32),
                      torch.as_tensor(RET, dtype=torch.float32),
                      optim, eps, mb, shuffle)


def ppo_update_onestep(head: ActorCritic, batch: list, optim,
                       eps: float = 0.0, mb: int = 32,
                       shuffle: bool = False) -> float:
    """**LA 的 1-step TD**（2026-10-18）：从 500 的滚动缓冲里随机抽 32 条。

    `batch` = `[(entry, v_next)]`；`entry = (s, a, logp, v, r)`。
    `adv = r + γ·V(s') − V(s)`；`ret = adv + V(s)`。
    ⚠️ 随机样本不连续 ⟹ **不能算 GAE**；原文 Table 6 的 `C_low=500` 就是回放池。
    """
    if len(batch) < 2:
        return 0.0
    S = torch.as_tensor(np.stack([b[0][0] for b in batch]), dtype=torch.float32)
    A = torch.as_tensor([b[0][1] for b in batch], dtype=torch.long)
    LP0 = torch.as_tensor([b[0][2] for b in batch], dtype=torch.float32)
    V = torch.as_tensor([b[0][3] for b in batch], dtype=torch.float32)
    R = torch.as_tensor([b[0][4] for b in batch], dtype=torch.float32)
    V2 = torch.as_tensor([b[1] for b in batch], dtype=torch.float32)
    delta = R + GAMMA * V2 - V
    ADV = delta * RET_SCALE
    RET = delta + V
    return _ppo_apply(head, S, A, LP0, ADV, RET, optim, eps, mb, shuffle)


def _calibrate_feats(specs, nfs, rules=((0, 0), (3, 0), (8, 2))) -> dict:
    """三头特征的固定 `μ/σ`（训练池 × 3 组规则；特征只依赖状态）。"""
    F: dict = {k: [] for k in ("oa", "pa", "la")}
    for sp, nfi in zip(specs, nfs):
        for k, j in rules:
            h: dict = {}

            def pol_pa(view, pool, tie_rng=None, _k=k, _nf=nfi, _h=h):
                snap = view.snap
                F["oa"].append(_nf.oa(snap, (5.0, 5.0)))
                F["pa"].append(_nf.pa(snap, 0.0, (5.0, 5.0)))
                job, mach = apply_pa_rule(_k, view, pool, tie_rng)
                _h["job"], _h["mach"] = int(job), int(mach)
                return job, mach

            def pol_l(view, cand_v, _j=j, _nf=nfi, _h=h):
                F["la"].append(_nf.la(view.snap, _h.get("job", 0),
                                      _h.get("mach", 0), 0.0, (5.0, 5.0)))
                return int(la_rule_policy(_j, (0.0,))(view, cand_v))

            sp.world().run_gated(seed_chain=701_000, policy_pa=pol_pa,
                                 policy_l=pol_l, dt_events=0)
    out = {}
    for key, rows in F.items():
        arr = np.asarray(rows, dtype=float)
        raw_sd = arr.std(axis=0)
        sd = np.where(raw_sd > 1e-8, raw_sd, 1.0)
        out[key] = (arr.mean(axis=0), sd)
    return out


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    # ⚠️ **2026-10-14：训练循环照原文 Algorithm 1 重写。**
    #   旧形状：`V3_STEPS` 步 × `V3_CHAIN` 链 ⟹ 每步一批 ≈ 992（8 × 124）、更新一次。
    #   新形状：`V3_EPOCHS` 个 epoch、**每 epoch = 1 局**（原文第 2 步「每个 epoch 随机生成
    #   工件与故障相关信息初始化环境」），每局按 **minibatch 32**（Table 6）切着更新。
    epochs = int(os.environ.get("V3_EPOCHS", "1000"))
    mb = int(os.environ.get("V3_MINIB", "32"))          # Table 6：minibatch = 32
    n_inst = int(os.environ.get("V3_NINST", "20"))
    # ⚠️ **复现口径 = 默认**（2026-10-14 改）：本文件是**复现层**，默认就该是原文的配置。
    #   此前默认是 v3 原行为（`grid` · `scale 1` · `shi2025` 关）⟹ 每次要手打三个环境变量，
    #   **打漏一个就静默跑成别的配置**。改成默认即原文口径，要跑别的再显式覆盖。
    #   ⚠️ 代价：**此前所有 `v3_nhdrl.py` 的读数都作废**（它们跑的是 `grid`）。
    scheme = os.environ.get("V3_SCHEME", "fig4")
    shi = os.environ.get("V3_SHI", "1") == "1"
    ck = os.environ.get("V3_CKPT", "")
    from plcsp_v3.q1_flat import (SHI_SHOP_LEN, build,
                                  dims_from_shop_len)
    # ⚠️ **`V3_SHOPLEN`（2026-10-16）**：车间长度 [m]，取代旧的 `V3_SCALE`（无因次倍率）。
    shoplen = float(os.environ.get("V3_SHOPLEN", str(SHI_SHOP_LEN)))

    torch.manual_seed(0)
    # ⚠️ **实例池**：原文每 epoch 随机生成新实例；本仓 `build(spec)` = **12.406 s**（§37.46.4）
    #    ⟹ 1000 个 = **3.4 小时/seed** ⟹ 预建 `V3_NINST` 个**轮换**（近似，登记在案）。
    #
    # ⚠️⚠️ **2026-10-14 修两处**（见 `docs/v3/open-question-nhdrl.md` §4.7）：
    #   ① **`pooled`** —— 原文把「可用机器」（车间配置，固定）与「加工时间 2–10」
    #      （生产参数，随机）**分在两栏**。此前本仓连加工时间也冻死在同一张表里。
    #   ② **评测用 `eval_instance`** —— 此前评测用 `specs[0]`，而轮换 `si = (ep-1) % n_inst`
    #      让实例 0 进了训练池 **50 次** ⟹ **在训练过的实例上评测**（train-on-test）。
    if VARY:
        # ⚠️ **标定池**（只用于特征标准化）；训练实例**每 epoch 现生成**（见下）。
        _jrng = np.random.default_rng([20261017, JOB_SEED])
        jobs_i = _jrng.integers(JOBS_LO, JOBS_HI + 1, size=n_inst)
        insts = [make_shi_instance(
                     int(j), seed=POOL_STRIDE * (1000 + i),
                     rot_seed=PT_REDRAW_SEED + 1000 + i,
                     alts_dist="uniform", alts_min=ALTS_MIN, alts_max=ALTS_MAX,
                     pt_dist="uniform", pt_lo=2, pt_hi=10)
                 for i, j in enumerate(jobs_i)]
        fail_i = _jrng.integers(1, 9, size=n_inst)
        print("  标定池规模：" + " ".join(str(int(j)) for j in jobs_i), flush=True)
    else:
        insts = [pooled(b, a, i, **_inst_kw()) for i in range(n_inst)]
        fail_i = np.zeros(n_inst, dtype=int)

    def _build_one(it, nfail: int = 0):
        """统一 `build`（`reward_norm="none"` ⟹ **0.07–0.12 s/实例**）。"""
        kw = dict(seed=0, pm_interval=30, target_failures=4, n_agv=NAGV,
                  finite_buffer=True, layout_seed=0, reward_norm="none",
                  tau=2.70, sim_overrides=_sim_over(), inst_obj=it,
                  agv_n_failures=int(nfail),
                  layout_scheme=scheme,
                  layout_cols=LAYOUT_COLS, layout_rows=LAYOUT_ROWS,
                  shi2025=shi,
                  **dims_from_shop_len(shoplen, it.n_machines,
                                       layout_cols=LAYOUT_COLS))
        if CELL > 0.0:
            # ⚠️ 显式尺寸覆盖 `dims_from_shop_len` 的旧口径换算。
            kw.update(cell_w=CELL, cell_h=CELL, aisle_w=AISLE,
                      lu_connector_m=LU)
        return build(**kw)[0]

    specs = [_build_one(it, fail_i[i]) for i, it in enumerate(insts)]
    nfs = [NHFeatures(it, sp.layout,
                      speed_m_per_min=sp.cfg.agv_speed_mps * 60.0)
           for it, sp in zip(insts, specs)]
    if FEAT_STD:
        feat_stats = _calibrate_feats(specs, nfs)
        print("  特征标准化：" + "　".join(
            f"{k}({len(v[0])}维)" for k, v in feat_stats.items()), flush=True)
    else:
        feat_stats = None
    ev_it = eval_instance(b, a, n_inst, **_inst_kw())
    spec = _build_one(ev_it, 0)
    nf = NHFeatures(ev_it, spec.layout,
                    speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
    assert all(sig(it) != sig(ev_it) for it in insts), "评测实例进了训练池"
    net = NHDRL()
    opts = {k: torch.optim.Adam(h.parameters(), lr=LR_BY_HEAD[k])
            for k, h in net.heads().items()}
    print(f"NHDRL · Exp({a}/{b}) · 真工序 {sum(len(o) for o in insts[0].jobs)} · "
          f"{sum(p.numel() for p in net.parameters())} 参数 · "
          f"{epochs} epoch（每 epoch 一局）· minibatch {mb} · 实例池 {n_inst} · "
          f"lr={LR_BY_HEAD} · ε {EPS0}→{EPS_LB} · 奖励 Eq.50–53\n", flush=True)

    import time
    t0 = time.time()
    # ⚠️ **`RewardFn` 全局唯一**：原文 Eq.51/53 的 `max_ms`/`min_ms` 是
    #    「**整段交互中**观察到的极值」。每 epoch 新建会让归一化的起点每局重置。
    rew = RewardFn()
    # ⚠️ **回放记忆**（2026-10-14）：照原文 Algorithm 1 第 3–4 步 —— 逐决策点把 transition
    #   存进**各自**的记忆，**攒满 `bs` 就更新、用完即清**。
    #   原文 Table 6：`C_high = 32`（OA/PA）、`C_low = 500`（LA）。
    # ⚠️ **本仓三个 head 都用「容量 = `bs` = 32」那一档**：原文的 minibatch 是从记忆里
    #   **随机抽**的，而 GAE 要在**连续轨迹**上算 —— 随机子集上算不出优势。
    #   **`C_low = 500` 未实现，登记在案。**
    memos = {k: Memo(bs=mb) for k in ("oa", "pa")}
    # ⚠️ **LA：容量 500、触发 32**（2026-10-18 修）：原文 Table 6 的 `C_low=500`
    #    是**回放池容量**；更新触发与 OA/PA 一样是 **minibatch=32**。
    #    1-step TD（随机样本不连续，不能算 GAE）。
    memos["la"] = Memo(bs=mb, cap=500, one_step=True, trigger=mb)
    la_rng = np.random.default_rng([20261018, JOB_SEED])
    n_since = {"la": 0}
    n_upd = {"oa": 0, "pa": 0, "la": 0}
    cur = {"eps": 0.0}

    def on_push(key: str, entry: tuple, v_next: float, terminal: bool) -> None:
        """写一条进该 head 的记忆；满了就更新；局末把余下的那一段冲掉。

        ⚠️ **LA 走 `C_low=500` 的整段路线**（2026-10-17）：先攒满 500，
        在**整段上算 GAE**，再**随机抽 32 条**做 PPO 更新。
        OA/PA 仍是 `C_high=32` 的在线路线。
        """
        mm = memos[key]
        mm.push(entry, v_next)
        if key == "la":
            # ⚠️ **LA：每 32 个决策点更新一次**，从 500 的滚动缓冲里随机抽 32 条，
            #    1-step TD（不是 GAE）。2026-10-18 修，见 `Memo`。
            n_since["la"] += 1
            if n_since["la"] >= mb:
                n_since["la"] = 0
                if len(mm.seg) >= mb:
                    batch = mm.sample(mb, la_rng)
                    ppo_update_onestep(net.heads()[key], batch, opts[key],
                                       eps=cur["eps"], mb=mb)
                    n_upd[key] += 1
            return
        while mm.full():
            seg, boot = mm.take()
            ppo_update(net.heads()[key], seg, opts[key], eps=cur["eps"],
                       mb=mb, boot=boot)
            n_upd[key] += 1
        if terminal:                       # ⚠️ 段界：不许跨局拼接
            seg, boot = mm.flush()
            if len(seg) >= 2:
                ppo_update(net.heads()[key], seg, opts[key], eps=cur["eps"],
                           mb=mb, boot=boot)
                n_upd[key] += 1

    for ep in range(1, epochs + 1):
        # ⚠️ **ε 递推照原文 Algorithm 1 第 5 步** `ε ← max(ε*0.99, LB)`；
        #    步进单位 = **epoch**（原文第 5 步在每个 epoch 末更新 ε）。
        eps = max(EPS0 * (EPS_DECAY ** (ep - 1)), EPS_LB)
        cur["eps"] = eps
        if VARY:
            # ⚠️ **照原文 Algorithm 1 第 2 步**：**每个 epoch 现生成一个新实例**。
            #    `reward_norm="none"` 后 `build` 只要 0.07–0.12 s ⟹ 1000 个不再需要 3.4 h。
            _er = np.random.default_rng([20261017, JOB_SEED, ep])
            jobs_ep = int(_er.integers(JOBS_LO, JOBS_HI + 1))
            inst_ep = make_shi_instance(
                jobs_ep, seed=POOL_STRIDE * (10_000 + ep),
                rot_seed=PT_REDRAW_SEED + 10_000 + ep,
                alts_dist="uniform", alts_min=ALTS_MIN, alts_max=ALTS_MAX,
                pt_dist="uniform", pt_lo=2, pt_hi=10)
            sp = _build_one(inst_ep, int(_er.integers(1, 9)))
            nfi = NHFeatures(inst_ep, sp.layout,
                             speed_m_per_min=sp.cfg.agv_speed_mps * 60.0)
            si = jobs_ep
        else:
            si = (ep - 1) % n_inst if n_inst > 1 else 0
            sp, nfi = specs[si], nfs[si]
        if MS_FIXED:
            # ⚠️ **固定尺度**：按当前实例的 reference makespan。
            rew.ms_scale = float(reference_makespan(sp.inst, sp.cfg))
        p_ms = (int(np.random.default_rng([20261018, ep]).integers(0, 11))
                if PREF_RANDOM else 5)
        r = Rollout(net, sp, nfi, rew, torch.Generator().manual_seed(ep),
                    prefs=(float(p_ms), float(10 - p_ms)),
                    eps=eps, on_push=on_push, feat_stats=feat_stats,
                    force_oa=FORCE_OA, force_la=FORCE_LA)
        w = sp.world()                     # ⚠️ 抓住同一个 world —— `finish` 要它的终局快照
        m = w.run_gated(seed_chain=9_000_000 + ep, policy_pa=r.policy_pa,
                        policy_l=r.policy_l, dt_events=0)
        r.finish(w.snapshot(), float(m["makespan"]))   # ⚠️ 补终局（§37.44.1）
        # ⚠️ **带 epoch 后缀**：只存一个路径会被覆盖，之后没法看它是收敛了还是没训够。
        if ck and ep % max(1, epochs // 20) == 0:
            base, ext = os.path.splitext(ck)
            torch.save({"state_dict": net.state_dict(),
                        "feat_stats": feat_stats}, f"{base}_e{ep:04d}{ext}")
        # ⚠️ 每 1% 打一次 + 带累计秒数：慢下来才看得出来。
        if ep % max(1, epochs // 100) == 0 or ep == 1:
            print(f"  epoch {ep:>4}/{epochs}  {time.time() - t0:8.1f}s  "
                  f"{'规模' if VARY else '实例'} {si}  本局 OA/PA/LA={len(r.buf['oa'])}/"
                  f"{len(r.buf['pa'])}/{len(r.buf['la'])}  "
                  f"更新 {n_upd['oa']}/{n_upd['pa']}/{n_upd['la']}",
                  flush=True)
    print(f"\n  训练完成：{epochs} 局 · 更新次数 OA/PA/LA = {n_upd['oa']}/"
          f"{n_upd['pa']}/{n_upd['la']}（原文 ≈ 1000 × 123 / 32 = 3844）· "
          f"用时 {time.time() - t0:.0f} s", flush=True)

    # ⚠️ **训练结束**：LA 的滚动缓冲按 32 抽一次（剩余不足 32 的就地用完）。
    if len(memos["la"].seg) >= 2:
        _b = memos["la"].sample(min(mb, len(memos["la"].seg)), la_rng)
        ppo_update_onestep(net.heads()["la"], _b, opts["la"],
                           eps=cur["eps"], mb=mb)
        n_upd["la"] += 1

    if ck:
        torch.save({"state_dict": net.state_dict(), "feat_stats": feat_stats}, ck)
        print(f"  权重已存 {ck}", flush=True)

    # ══════════ 评测（2026-10-14 新增）══════════
    #
    # ⚠️ **这版此前只有训练、没有评测** —— 跑完只打三个 loss，**读不出 makespan**，
    #    所以「有没有赢过常数派工规则」根本无从判断。
    #
    # ⚠️ **口径与复现最小版逐字一致**（这样两边的数可以直接并排）：
    #   `greedy=True` · 同 20 个场景 `100_000 + 13i` · 同 `run_gated`。
    #
    # ⚠️⚠️ **2026-10-14 修标错**：此前这里只扫 9 条 PA、把 LA **钉死在 `MINCOST`**
    #     (`la_rule_policy(0)`)，却印成「9 条 CDR」。
    #     **9 × 1 ≠ 27。** 原文的 CDR 是 *"27 CDRs generated by **pairwise combinations**
    #     of PA's and LA's actions"*（PAGE 16）= **9 × 3**。**已改成扫满 27 条。**
    #     实测（fig4 · scale 12.5）：`MINCOST` 列最好 490.5，而 27 条最好 **424.0**
    #     —— 基准被低估 **13.5%**。见 `docs/v3/open-question-nhdrl.md` §4.7.1。
    net.eval()
    ev_rew = RewardFn()
    if MS_FIXED:
        ev_rew.ms_scale = float(reference_makespan(spec.inst, spec.cfg))
    scs = [100_000 + 13 * i for i in range(20)]
    ref: dict[tuple[int, int], tuple[float, float]] = {}
    for k in range(N_AEI2025_RULES):
        for j in range(len(LA_RULE_NAMES)):
            rows = [spec.world().run_gated(
                seed_chain=sc,
                policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, tr),
                policy_l=la_rule_policy(j), dt_events=0) for sc in scs]
            ref[(k, j)] = (float(np.mean([r["makespan"] for r in rows])),
                           float(np.mean([r["logistics_cost"] for r in rows])))
    best = min(ref, key=lambda t: ref[t][0])
    print(f"\n  {len(ref)} 条 CDR（20 场景均值）· 按 MS 升序前 9：")
    print(f"      {'PA + LA':<28}{'MS':>8}{'TC':>10}")
    for t in sorted(ref, key=lambda t: ref[t][0])[:9]:
        print(f"      {PA_RULE_NAMES[t[0]] + '+' + LA_RULE_NAMES[t[1]]:<28}"
              f"{ref[t][0]:>8.1f}{ref[t][1]:>10.1f}"
              + ("　← 最好" if t == best else ""))

    ms_pol = []
    oa_c: dict[int, int] = {}
    pa_c: dict[int, int] = {}
    la_c: dict[int, int] = {}
    for i, sc in enumerate(scs):
        p_ms = (int(np.random.default_rng([2000, i]).integers(0, 11))
                if PREF_RANDOM else 5)
        r = Rollout(net, spec, nf, ev_rew,
                    torch.Generator().manual_seed(0), greedy=True,
                    prefs=(float(p_ms), float(10 - p_ms)),
                    feat_stats=feat_stats, force_oa=FORCE_OA,
                    force_la=FORCE_LA)
        w = spec.world()
        m = w.run_gated(seed_chain=sc, policy_pa=r.policy_pa,
                        policy_l=r.policy_l, dt_events=0)
        r.finish(w.snapshot(), float(m["makespan"]))
        ms_pol.append(float(m["makespan"]))
        for b in r.buf["oa"]:
            oa_c[b[1]] = oa_c.get(b[1], 0) + 1
        for b in r.buf["pa"]:
            pa_c[b[1]] = pa_c.get(b[1], 0) + 1
        for b in r.buf["la"]:
            la_c[b[1]] = la_c.get(b[1], 0) + 1

    tot = lambda d: max(1, sum(d.values()))            # noqa: E731
    print("\n  greedy 评测（20 场景）：")
    print("    OA 目标：" + "　".join(
        f"{'MS' if k == 0 else 'TC'}:{v / tot(oa_c):.0%}"
        for k, v in sorted(oa_c.items(), key=lambda x: -x[1])))
    print("    PA 规则：" + "　".join(
        f"{PA_RULE_NAMES[k]}:{v / tot(pa_c):.0%}"
        for k, v in sorted(pa_c.items(), key=lambda x: -x[1])))
    print("    LA 规则：" + "　".join(
        f"{k}:{v / tot(la_c):.0%}" for k, v in sorted(la_c.items(), key=lambda x: -x[1])))
    mean_pol = float(np.mean(ms_pol))
    tag = f"{PA_RULE_NAMES[best[0]]}+{LA_RULE_NAMES[best[1]]}"
    print(f"\n  策略 makespan **{mean_pol:.1f}**　最好 CDR {ref[best][0]:.1f}（{tag}）"
          f"　TC {ref[best][1]:.1f}")
    print(f"  ⟹ **NHDRL 3-agent　"
          f"{'✅ 赢过最好 CDR' if mean_pol < ref[best][0] else '❌ 没赢'}**")


if __name__ == "__main__":
    main()
