# -*- coding: utf-8 -*-
"""**复现的最小版本：只留 PA 一个智能体。**

══ 为什么从这个开始 ══

复现 `v3_nhdrl.py` 查了一整天「为什么学不出来」，改的全是高层（奖励口径、网络、
特征、口径）。**正确顺序是反的**：先把复现缩到最小，确认这一小步**对得上原文**，
再逐层加。加了才坏 ⟹ 那一层就是病灶。

══ 这一版是什么 ══

| 层 | 原文 | **本版** |
|---|---|---|
| 智能体 | OA + PA + LA | **只留 PA** |
| 目标 | MS 与 TC（OA 选） | **只 MS** |
| 奖励 | Eq.50–53（min–max 归一 + 权重） | **`−d_ms`**（望远镜求和 = `−makespan`） |
| 动作 | PA 9 条规则 | **同样是 9 条** |
| 实例 | `Exp(a/b)` | 同 |
| 评测 | 20 局 → HV/SP | **规则分布 + 实测 makespan** |

**奖励取最简形**是有意的：不加归一、不加权重、不分两支。**先确认"奖励→优势→梯度→策略"
这条回路通不通**，通了再按 Eq.50–53 加回去。

══ 判据（分层，从弱到强）══

1. **训练曲线上升**（原文 Fig. 5 就是这个）。
2. **训练后偏向实测更好的规则** —— 与 9 条 CDR 的实测 makespan 对照。
3. **赢过最好 CDR** —— 这条是原文 Table 9 的主张，**最小版不预设一定过**。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from plcsp_v3.env.nhdrl_features import F_PA, NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

STEPS = int(os.environ.get("MIN_STEPS", "300"))
N_CHAIN = 4
LA_FIXED = 0
HID_A, HID_V = 256, 128
"""**照原文 Table 4**（`docs/citation-cards.md:1607`）：

| 网络 | 层 | 节点数 | 激活函数 |
|---|---|---|---|
| Actor | 隐藏层 | **256** | ReLU |
| Critic | 隐藏层 | **128** | ReLU |

⚠️ **2026-10-14 改**：此前两个都是 **64** —— 与原文不符，且是本仓唯一没动过的
「训练规模」缺口之一（§37.47.6）。
"""
LR = float(os.environ.get("MIN_LR", "1e-4"))
"""学习率。**默认 = 原文 Table 5 的 PA lr = `1e-4`**（2026-10-14 改，此前默认是 `1e-3`）。

⚠️ **改默认值的依据**：那个 `1e-3` 是在**旧配置**（更新批 492 ≈ 一整局）下量出来的。
现在批是 **minibatch 32**（照 Table 6）⟹ **原来「批大所以步长可以大」的理由作废。**
**代码默认必须与论文一致，否则不传环境变量就跑歪。**

⚠️ **2026-10-13 更正**：本仓此前写死 `1e-4`，注释称「此前我写 1e-3，大 10 倍；
§22.3 记过它会过冲」——**那两个引用都指错了**：
`docs/v3/progress-log.md` §22.3 讲的是 **MKT 交期标定**，与 lr 无关；
§22.1/22.2 的原话是「**`lr = 1e-4` 下也涨**」，那是为了说明「涨不是因为 lr 太大」，
**不是「1e-3 会过冲」**。**⟹ 当初把 lr 降 10 倍的依据不存在。**

**实测对照（K=9、1000 步、20 场景均值，CDR = 147.6）**：

| lr | 3e-5 | 1e-4 | **1e-3** | 3e-3 |
|---|---|---|---|---|
| makespan | 155.2 | 157.9 | **152.4** | 162.4 |

**⟹ `1e-3` 最好，但**仍然没到 147.6**。差值 3.7 个百分点。**

⚠️ **同时登记的判断**：`1e-4 → 1e-3` 只值 3.7 个点，**说明这不是「对 lr 敏感」**——
一个正常的 PG，lr 变 10 倍应只改收敛速度、不该改成败。**更可能是「步长 × 步数」总量偏小**
（`300 步 × 1e-4 = 0.03`，而 9 路 softmax 要形成强偏好需要 ~2–4 个 logit 单位）。
"""

# ══════════ 两道探索闸（`Tanh` 于 2026-10-14 删过又加回，见 §37.48）══════════
#
# ⚙️ **闸一：Actor 输出层 `Tanh`**（`docs/citation-cards.md:1607`，Table 4：
#    Actor 输出层激活 = Tanh）。**回**把 logits 压在 **(−1, 1)**
#    ⟹ **`softmax` 在数学上不可能是 delta**。
#    （K=2/3/9 时最大概率的上界分别 0.881 / 0.787 / 0.480。）
#
# **一天的来回（记下来，别再走一遍）**：
#   - 删掉后跑了 4 档（无激活 · 256/128 · minibatch 32 · 1000 epoch · lr 1e-3 / 1e-4），
#     **两个 seed 全失败**：seed 0 = **279.6 / 179.4**，seed 1 = **169.7**（最好 CDR 147.6）。
#   - 诊断：**logit 极差 38.9–50.5、`softmax` 最大概率 1.000** ⟹ 策略一上来就是 delta。
#   - ⚠️ **删它的依据不假但不足以定论**（§37.46）：原文 Table 4 的 `Tanh` 判不了真假
#     （不开源 · 表里有笔误 · 同任务姊妹篇用多元高斯、根本没这一格），
#     且三种输出激活（无 / `Tanh` / `GELU`）**实测都没赢过**。
#     **⟹ `Tanh` 不是病灶，但删掉它会让起点更糟。加回来，按原文。**
#
# ⚙️ **闸二：ε-greedy**（`docs/citation-cards.md:1605`：「以概率 ε 向 actor 输出的
#    动作概率分布加随机噪声…ε 逐步下降至下界 LB」；Table 6：ε 初值 0.9、下界 0.1；
#    更新式 `ε ← max(ε*0.99, LB)`）。
#
# ⚠️ **两道闸解决的是同一件事的两半**：`Tanh` 管**上界**（分布不能变成 delta），
#    ε 管**下界**（即使塌了也留一条采样通道）。§37.37 实测：两道都缺会**冻死**
#    （float32 下非最大动作的概率和恰为 0，148800 个样本一个都采不到）。
#
# ⚠️ **Critic 输出层原文也是 Tanh，本仓**故意不加**：
#    原文的回报经 Eq.50–53 归一化（量级 ~1），本仓的 `−d(cm)` 是**分钟量级（几百）**
#    ⟹ Tanh 会把价值头压在 (−1,1) 直接饱和、学不了。**这个差异登记在案。**
EPS0, EPS_LB, EPS_DECAY = 0.9, 0.1, 0.99

# ⚠️ **`BOUND` 已删（2026-10-14）。** 它 2026-10-13 从地板测试（§37.45.4）引进来
#    （理由：「原文 Table 4 只写 `Tanh`，没写输出尺度」），**当天测一次就撤。**
#
# **撤的依据 —— 单变量对照，seed 0，其余逐字相同（256/128 · minibatch 32 ·
#  1000 epoch · lr 1e-4 · `Tanh` 在）：**
#
# | | `BOUND=1` | `BOUND=4` |
# |---|---|---|
# | greedy 分布 | `EAJ+SMPT` **100%** | `EAJ+SMPT` 86% / `SPT+EAM` 14% |
# | 策略 makespan | **147.6** ✅ | **148.1** ✅ |
# | `softmax` π_max 中位 | 0.156（上界 0.480） | 0.250（上界 0.997） |
#
# **⟹ 两档都过、差 0.5（噪声内），`BOUND=1` 略好 ⟹ 不换 ⟹ 不留开关。**
#
# ⚠️ **顺带更正 §37.45.4 那条读数**：原记录说 `BOUND=4` 修好了 `BOUND=1` 的
#   「三条并列 26%/26%/26%」。**但同一段还记着 `BOUND=4` 那次的 `|pre|` 中位 = 1.71（没饱和）**
#   ⟹ **那是那一次训练轨迹的偶然，不是 `BOUND` 的性质。**
#
# **实测机制**：`|pre|` 饱和（中位 11.8 / 14.1，≥3 占比 98.6% / 100%）⟹ `tanh` 把 logits
#   压成**二值** `{+B, −B}` ⟹ **`π_max` 由「有几个正臂」决定，不由 `BOUND` 决定。**
#   （4 正 5 负时：`B=1` → `e/(4e+5/e)` = **0.214**；`B=4` → `e⁴/(4e⁴+5e⁻⁴)` = **0.250**。
#   实测 0.156 / 0.250，对得上。）


def eps_at(step: int) -> float:
    """原文 `ε ← max(ε*0.99, LB)`。`step` 从 1 数起。"""
    return max(EPS0 * (EPS_DECAY ** step), EPS_LB)


# ⚙️ **奖励那个量：`MIN_CM`**（2026-10-13）
#
# `"bound"`（默认，本仓原口径）：`now + max_m(remaining_min + backlog_min)`
#   —— **机台负载下界**：只看**当下队列**里有多少活，不看计划表。
# `"plan"`：**`max_k mct_assigned[k]`** —— 按 `pa_mct` 账本（`des.py:2238` 推进
#   `mct[k] ← max(mct[k], 物理清空时刻, now) + pt`）算的、
#   **全部已排定工序的完成时刻**。**这才是原文 Eq.50–51 的 MS**
#   （*"Makespan for all jobs in **the schedule**"*）。
#
# ⚠️ 两者**望远镜求和都收敛到真 makespan**，但**逐步形状完全不同** ⟹
#    「奖励 → 优势 → 梯度」的信用分配也不同。这一条此前从未测过（§37.43）。
# ⚠️ 已知缺陷（两者都有）：`mct_assigned` 在**第一次派工前全是 0** ⟹
#    最前面几个决策点的奖励会是 0。
CM_KIND = os.environ.get("MIN_CM", "plan")
"""**默认已改成原文口径 `plan`**（2026-10-14）。原文 Eq.50–51 的 MS 是
*"Makespan for all jobs **in the schedule**"* = `max_k mct_assigned[k]`，**那就是 `plan`。**"""

# ⚠️ **GAE 的 γ / λ 也可换**（2026-10-13，排查用）。
# 原文 Table 5：γ = 0.99。但「望远镜求和 = −makespan」**只在未折扣时成立**：
#   Σ γ^t (C_t − C_{t+1}) = C₀ + (γ−1)·Σ γ^{t−1} C_t − γ^{T−1}·C_T
# γ<1 时终局项被压到 `γ^{T−1}`（123 步时 ≈ 0.293），中间各 `C_t` 还各带 `(γ−1)` 的系数
# ⟹ **优化目标是一个三段混合，不是终局 makespan**（§37.44）。
# `MIN_GAMMA=1 MIN_LAM=1` 才让「−Δcm 就是 −makespan」这句话成立。
GAMMA = float(os.environ.get("MIN_GAMMA", "0.99"))
LAM = float(os.environ.get("MIN_LAM", "0.95"))


def cm_of(snap) -> float:
    """奖励里那个量 —— 由 `CM_KIND` 选口径。**训练与评测必须用同一个。**"""
    if CM_KIND == "plan":
        return max((float(x) for x in snap.mct_assigned), default=0.0)
    return float(snap.now) + max(
        (float(m.remaining_min) + float(m.backlog_min) for m in snap.machines),
        default=0.0)


# ⚠️ **Critic 输出层 `Tanh`**（原文 Table 4：`citation-cards.md:1614–1616`）。
#    **纯 `tanh` 成立的前提：价值目标是 O(1)。** 原文的奖励先按 Eq.50–53 归一化；
#    本仓原来直接喂 `−d(cm)`（分钟量级）⟹ 直接加 `tanh` 会饱和。
#    **正确做法 = 先归一化奖励**：`MIN_REWARD_NORM=shi` 时每步奖励除以
#    `(max_ms − min_ms)`（MS 的可达范围），回报落在 (−1, 0) ⟹ 纯 `tanh` 可表示。
#    ⚠️ **默认 `raw` + `MIN_CRITIC_TANH=0` ⟹ 旧行为逐位不变。**
CRITIC_TANH = os.environ.get("MIN_CRITIC_TANH", "0") == "1"
REWARD_NORM = os.environ.get("MIN_REWARD_NORM", "raw")


class Head(nn.Module):
    """PA 一个头：24 维状态 → K 个 logit + 一个价值。

    **网络结构照原文 Table 4**（Actor 隐层 **256**、Critic 隐层 **128**、ReLU）：

    ```
    Actor   输入 → ReLU → **256** → ReLU → 输出(动作维) → **Tanh**
    Critic  输入 → ReLU → **128** → ReLU → 输出(1)     → **Tanh**（开关见下）
    ```

    ⟹ **logits ∈ (−1, 1)** ⟹ **`softmax` 在数学上不可能是 delta**，探索不会死。
    （K=2/3/9 时最大概率的上界分别 0.881 / 0.787 / 0.480。）

    ⚠️ **2026-10-14 一天内来回一次，记在这里**：
    先按用户指令删掉 `Tanh` 跑了 4 档（无激活 · 256/128 · minibatch 32 · 1000 epoch ·
    lr 1e-3 / 1e-4）—— **两个 seed 都失败**（seed 0 = 279.6 / 179.4，seed 1 = 169.7；
    最好 CDR 147.6），且 **logit 极差 38.9–50.5、`softmax` 最大概率 1.000**（策略是 delta）。
    **⟹ 又加回来了。** 见 `progress-log` §37.48。

    ⚠️ **Critic 输出层原文是 `Tanh`**（Table 4）。**2026-10-17 按用户要求加纯 `Tanh`**：
    前提是**奖励先归一化** —— `MIN_REWARD_NORM=shi` 时每步奖励除以 `(max_ms − min_ms)`
    （原文 Eq.50–51 的归一化），回报落在 (−1, 0) ⟹ `tanh` 可表示。
    `MIN_CRITIC_TANH=1` 打开，默认关；**不归一化而直接开 `Tanh` 会饱和**。

    ⚠️ **`nn.Sequential` 的下标 2 仍是最后一个 `Linear`**（`Tanh` 在下标 3；
    `shi2025_min2.py` 等替换下标 2 来改动作数 —— **不受 `Tanh` 存废影响**）。
    """

    def __init__(self, f_dim: int = F_PA) -> None:
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(f_dim, HID_A), nn.ReLU(),
                                nn.Linear(HID_A, 9), nn.Tanh())
        self.v = nn.Sequential(nn.Linear(f_dim, HID_V), nn.ReLU(),
                               nn.Linear(HID_V, 1))
        if CRITIC_TANH:
            # ⚠️ **下标 3**（最后一项）；`shi2025_min2.py` 只替换 `pi[2]`，不受影响。
            #     纯 `Tanh`：要求回报已归一化到 (−1,1)（见 `REWARD_NORM`）。
            self.v.append(nn.Tanh())

    def dist(self, f):
        return torch.distributions.Categorical(logits=self.pi(f))

    def behavior(self, f, eps: float):
        """**行为分布** `b = (1−ε)·π + ε/K`（2026-10-13 修，见 `Roll._pick`）。

        ⚠️ **ε>0 时，采样和 log-prob 都必须用 `b`，PPO 的 ratio 也必须用 `b`。**
        此前采样走 ε 均匀、log-prob 却记 `log π`，ratio 算 `π_new/π_old`
        —— **样本来自 `b`，PPO 当成来自 `π`，口径不一致。**
        """
        if eps <= 0.0:
            return self.dist(f)
        p = torch.softmax(self.pi(f), -1)
        p = (1.0 - eps) * p + eps / p.shape[-1]
        return torch.distributions.Categorical(probs=p)

    def value(self, f):
        return self.v(f)[:, 0]


class Roll:
    """一局：奖励 = `−(CM(t) − CM(t−1))`。

    ⚠️ **2026-10-13：`Standardizer` / `calibrate` 已删。** 它当初是为了「初始策略 91% 集中」
    加的补丁，但实测**不修任何东西**（§37.34.5）——`K=9` 有它 ❌、没它 ❌。
    **它遮住的是特征自身的尺度问题。**
    **删掉 `Tanh` 之后，唯一挡着「初始策略变成 delta」的就是 ε**（见上方「探索闸」）。

    ⚠️ **局末必须调 `finish(makespan)`**，否则回报**漏掉整条尾巴**（见 `finish`）。
    """

    def __init__(self, net, spec, nf, greedy=False, eps: float = 0.0,
                 on_push=None, reward_scale: float = 1.0):
        self.net, self.nf, self.greedy, self.eps = net, nf, greedy, eps
        # ⚠️ **奖励归一化标度**（2026-10-17）：`MIN_REWARD_NORM=shi` 时传
        #    `(max_ms − min_ms)`；默认 1.0 ⟹ 旧行为逐位不变。
        self.reward_scale = float(reward_scale)
        self.buf, self._cm = [], None
        self._f, self._a, self._lp, self._v = None, 0, 0.0, 0.0
        self.first_cm = None                   # 供回报求和的自检用
        # ⚠️ **回放记忆的钩子**（2026-10-14）：每写完一条 transition 调一次
        #    `on_push(entry, v_next, terminal)`。`v_next` 是**该条之后那个状态的 `V`**，
        #    正好当切片末端 GAE 的代入值（见 `gae` 的 `boot`）。
        self.on_push = on_push

    @torch.no_grad()
    def _pick(self, f):
        ft = torch.as_tensor(f)[None]
        if self.greedy:
            d = self.net.dist(ft)
            a = int(d.logits.argmax())
        else:
            # ⚠️ **采样与 log-prob 必须同源**（2026-10-13 修）：ε>0 时用**混合分布** `b`。
            d = self.net.behavior(ft, self.eps)
            a = int(d.sample())
        return a, float(d.log_prob(torch.tensor(a))), float(self.net.value(ft)[0])

    def finish(self, makespan: float) -> None:
        """**局末收尾：把最后一个决策的奖励补上**（2026-10-13，§37.44）。

        ⚠️ **不加这个，回报就漏掉整条尾巴。** `policy_pa` 只在**下一次**被调用时才记
        上一个动作的奖励 ⟹ **最后一次永远没有「下一次」** ⟹ 它和它之后的收成都不进回报。
        实测（场景 `100000`）：最后决策点 `cm = 104.5`，真 makespan `157.3`
        —— **尾巴 52.8 分钟（占 34%）从未进入任何一条奖励**。
        """
        if self._cm is None:
            return
        self.buf.append((self._f, self._a, self._lp, self._v,
                         -(float(makespan) - self._cm) / self.reward_scale))
        self._cm = None
        # ⚠️ **`terminal=True`** ⟹ 记忆用它当**段界**（代入值 0，真终止），
        #    不许跨局拼接（否则上一局的终局项会被当成下一局的延续）。
        if self.on_push is not None:
            self.on_push(self.buf[-1], 0.0, True)

    def policy_pa(self, view, pool, tie_rng=None):
        snap = view.snap
        cm = cm_of(snap)                       # ⚠️ 口径由 `CM_KIND` 选（见上）
        raw = self.nf.pa(snap, 0.0, (5.0, 5.0))
        if self.first_cm is None:
            self.first_cm = cm
        if self._cm is not None:
            self.buf.append((self._f, self._a, self._lp, self._v,
                             -(cm - self._cm) / self.reward_scale))
        self._cm = cm
        self._f = raw
        self._a, self._lp, self._v = self._pick(self._f)
        # ⚠️ **记忆推进必须在 `_pick` 之后** —— 此刻 `self._v` 是 `V(S_{t+1})`，
        #    正是刚写进去那条 transition 的切片末端代入值。
        if self.on_push is not None and len(self.buf) >= 2:
            self.on_push(self.buf[-1], self._v, False)
        return apply_pa_rule(self._a, view, pool, tie_rng)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(LA_FIXED, (0.0,))(view, cand_v))

def gae(rew, val, gamma=0.99, lam=0.95, boot: float = 0.0):
    """GAE。

    ⚠️ **`boot` = 序列末端那个状态的代入值 `V(S_{T+1})`**（2026-10-14 加）。
    - **整局**（`finish()` 已把回报补到真完工时刻）⟹ 末端是**真终止** ⟹ `boot = 0.0` ✅
    - **局内切片**（回放记忆攒满 32 就更新）⟹ 切片末端**不是终止** ⟹
      **必须传 `V(S_{T+1})`**，否则每片都成了假终止（§37.44.1）。
    """
    adv, g = np.zeros(len(rew)), 0.0
    v = list(val) + [float(boot)]
    for t in reversed(range(len(rew))):
        g = rew[t] + gamma * v[t + 1] - v[t] + gamma * lam * g
        adv[t] = g
    return adv, adv + np.asarray(val)


class Memo:
    """**回放记忆**（原文 Algorithm 1 第 3–4 步 ＋ Table 6）。

    > 3. 逐决策点依框架逻辑交互（OA → PA → LA），存储 transition `(S_t, A_t, R_t, S_{t+1})`
    >    到各自记忆。
    > 4. **记忆中样本数达到 `bs` 时**，三者分别用 PPO 目标（式 49）经 Adam 更新。

    **容量 = `bs` = 32**（原文 Table 6 的 `C_high`，且 `citation-cards.md:1606` 写明
    这个「容量 = minibatch」的设置是为了让**上层 agent 近似在线**训练）
    ⟹ **攒满 32 就更新、用完即清。**

    ⚠️ **切片末端必须传 `V(S_{t+1})`**，见 `gae` 的 `boot`。
    ⚠️ **局末必须 `flush()`** —— 否则最后不足 32 条的那一段永远不更新，
    而且**跨局拼接**会把上一局的终局项当成下一局的延续。
    """

    def __init__(self, bs: int = 32, cap: int | None = None,
                 one_step: bool = False, trigger: int | None = None) -> None:
        self.bs = int(bs)
        # ⚠️ **容量与触发分开**（2026-10-18 修）：原文 `C_high=32`（OA/PA）、
        #    `C_low=500`（LA），而**更新触发都是 minibatch=32**。
        #    `one_step` ⟹ 滚动缓冲 + 1-step TD（LA 的随机样本不连续，算不了 GAE）。
        self.cap = int(cap) if cap is not None else self.bs
        self.one_step = bool(one_step)
        self.trigger = int(trigger) if trigger is not None else self.cap
        # ⚠️ **逐条存 `(entry, v_next)`** —— 代入值必须跟**被取出的那一段的最后一条**绑定。
        #    若只存一个标量，`take()` 取出前 32 条时会错用**后面**那条的 `v_next`。
        self.seg: list[tuple[tuple, float]] = []
        # ⚠️ **跨局回放**（2026-10-17）：`bounds` = 每局结束的位置（不含）。
        #    切段时按它拆，**GAE 不跨局递推**。
        self.bounds: list[int] = []

    def push(self, entry: tuple, v_next: float) -> None:
        """记一条 transition。`v_next` = **该条之后那个状态的 `V`**。"""
        self.seg.append((entry, float(v_next)))
        if self.one_step and len(self.seg) > self.cap:
            self.seg.pop(0)          # 滚动缓冲：丢掉最老的

    def full(self) -> bool:
        # ⚠️ **LA（one_step）只看触发（32）**；OA/PA 仍看容量（=32）。
        return len(self.seg) >= (self.trigger if self.one_step else self.cap)

    def sample(self, n: int, rng) -> list[tuple[tuple, float]]:
        """从缓冲里随机抽 `n` 条（**不删除**）。给 1-step TD 用。"""
        k = min(int(n), len(self.seg))
        idx = rng.choice(len(self.seg), size=k, replace=False)
        return [self.seg[int(i)] for i in idx]

    def take(self) -> tuple[list, float]:
        """取出**最前面连续的一批**（`bs` 条），返回 `(seg, boot)`。剩下的留到下一批。

        ⚠️ `cap > bs` 时（LA），只取前 `bs` 条，其余继续留在记忆里。
        **仍取连续段**：GAE 要求段内相邻；随机子集算不出优势。
        """
        chunk = self.seg[: self.bs]
        self.seg = self.seg[self.bs:]
        return [x[0] for x in chunk], chunk[-1][1]

    def take_all(self) -> tuple[list, float]:
        """**交出整段**（`cap` 条），返回 `(seg, boot)`，并清空。

        ⚠️ 给「**先在整段上算 GAE、再随机抽 minibatch**」这条路用（原文 `C_low=500`）。
        `boot` = 最后一条之后那个状态的 `V`；整段末端若跨局，由 `flush()` 的 0.0 处理。
        """
        chunk, self.seg = self.seg, []
        return [x[0] for x in chunk], (chunk[-1][1] if chunk else 0.0)

    def mark_terminal(self) -> None:
        """记下**一局结束**的位置（不清空）。`take_segments()` 按它切段。"""
        if self.seg:
            self.bounds.append(len(self.seg))

    def take_segments(self) -> list[tuple[list, float]]:
        """按局切分整个缓冲，返回 `[(entries, boot), ...]`，并清空。

        ⚠️ **跨局回放**（`C_low=500`）：LA 的过渡跨多局累积；GAE 必须**按局**
        分别递推（不能把上一局的终局项当成下一局的延续）。
        """
        bounds = self.bounds + [len(self.seg)]
        out: list[tuple[list, float]] = []
        start = 0
        for b in bounds:
            chunk = self.seg[start:b]
            if chunk:
                out.append(([x[0] for x in chunk], float(chunk[-1][1])))
            start = b
        self.seg, self.bounds = [], []
        return out

    def flush(self) -> tuple[list, float]:
        """**局末**：交出剩下的一段，末端代入值 = **0**（真终止）。取完即清。"""
        chunk, self.seg = self.seg, []
        return [x[0] for x in chunk], 0.0


def update(net, chains, opt, clip=0.2, vc=0.5, epochs=4, eps: float = 0.0) -> float:
    """`chains` = **每条链各自的 buffer 列表**。`eps` = 该批采样时的 ε（`π` 与 `b` 的桥）。

    ⚠️ **必须按链算 GAE，不能把各链拼成一条**（2026-10-12 修）：
    `gae` 是沿序列递推的 `g = d + γλ·g`。权重是 `(γλ)^k` ⟹ **有效视野 `1/(1−γλ) ≈ 16.8` 步**
    （`γλ = 0.9405`；67 步处权重只剩 1.6%）。
    ⚠️ **2026-10-13 更正**：此前这里写「有效视野 ≈ 67 步 ⟹ 过半 advantage 错」——**夸大了**。
    跨链拼接仍然错（尾部十几步污染），但**不是「过半」**。
    """
    S, A, LP0, ADV, RET = [], [], [], [], []
    for buf in chains:
        # ⚠️ **判据是「这条链非空」，不是「长度 ≥ 2」**（2026-10-12 修）：
        # 原来写 `len(buf) < 2` ⟹ **单步的局会被整条丢掉**（地板测试就是这么静默失败的）。
        # GAE 对长度 1 的序列完全成立（`adv[0] = r − v`），没有理由丢。
        # 「样本够不够」由函数末尾 `len(A) < 2` 统一判。
        if not buf:
            continue
        adv, ret = gae([b[4] for b in buf], [b[3] for b in buf],
                       gamma=GAMMA, lam=LAM)
        adv = np.asarray(adv, dtype=float)
        S += [b[0] for b in buf]
        A += [b[1] for b in buf]
        LP0 += [b[2] for b in buf]
        ADV += list(adv)
        RET += list(ret)
    if len(A) < 2:
        return 0.0
    S = torch.as_tensor(np.stack(S), dtype=torch.float32)
    A = torch.as_tensor(A, dtype=torch.long)
    LP0 = torch.as_tensor(LP0, dtype=torch.float32)
    ADV = torch.as_tensor(np.asarray(ADV), dtype=torch.float32)
    # ⚠️ **2026-10-14：全批 z-score 已删。** 原文 Eq.49 直接用 `AF_t`，**没有归一化**；
    #    而且原文的奖励经 Eq.50–53 已归一化到 ~1 量级。本仓的 `−d(cm)` 是分钟量级，
    #    **那条 z-score 是在替一个没归一化的奖励收场**（先把奖励换成 Eq.50–51 再删它才是对的）。
    RET = torch.as_tensor(np.asarray(RET), dtype=torch.float32)
    out = 0.0
    for _ in range(epochs):
        d = net.behavior(S, eps)           # ⚠️ 必须与采样同源（ε>0 时是混合分布）
        ratio = torch.exp(d.log_prob(A) - LP0)
        loss = (-torch.min(ratio * ADV, torch.clamp(ratio, 1 - clip, 1 + clip) * ADV).mean()
                + vc * ((net.value(S) - RET) ** 2).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        out = float(loss.detach())
    return out


def update_mb(net, buf, opt, mb=32, clip=0.2, vc=0.5, epochs=4, eps: float = 0.0,
              boot: float = 0.0) -> float:
    """**原文 Algorithm 1 第 4 步 + Table 6**：`minibatch size = 32`。

    ⚠️ **GAE 仍在整局上算，切片只切 PPO 更新的粒度。**
    理由：`gae()` 里 `v = list(val) + [0.0]` 把序列末端当作**真终止态**（`V=0`）。
    整局的末端**确实**是真终止 —— `finish(makespan)` 已把回报补到真完工时刻（§37.44.1）
    ⟹ `V=0` 正确。**若先按 32 步切片、再各自算 GAE，每片末端都成了假终止
    ⟹ 把 §37.44.1 那个 bug 原样搬回来。**

    ⚠️ **与 `update()` 的唯一区别就是这一处**：`update()` 一批用完整条链（~123）；
    本函数一批用 32。**其余（clip / vc / 4 epochs / 全批优势归一）一字不动。**

    **⟹ 它把「原文的 3844 次小步更新」和「本仓的 300 次大步更新」拆成单变量。**
    """
    if not buf:
        return 0.0
    adv, ret = gae([b[4] for b in buf], [b[3] for b in buf], gamma=GAMMA, lam=LAM,
                   boot=boot)
    S = torch.as_tensor(np.stack([b[0] for b in buf]), dtype=torch.float32)
    A = torch.as_tensor([b[1] for b in buf], dtype=torch.long)
    LP0 = torch.as_tensor([b[2] for b in buf], dtype=torch.float32)
    ADV = torch.as_tensor(np.asarray(adv), dtype=torch.float32)
    # ⚠️ **2026-10-14：全批 z-score 已删。** 原文 Eq.49 直接用 `AF_t`，**没有归一化**；
    #    而且原文的奖励经 Eq.50–53 已归一化到 ~1 量级。本仓的 `−d(cm)` 是分钟量级，
    #    **那条 z-score 是在替一个没归一化的奖励收场**（先把奖励换成 Eq.50–51 再删它才是对的）。
    RET = torch.as_tensor(np.asarray(ret), dtype=torch.float32)
    n, out = len(buf), 0.0
    for _ in range(epochs):
        tot = 0.0
        for i in range(0, n, mb):
            s, a = S[i:i + mb], A[i:i + mb]
            d = net.behavior(s, eps)       # ⚠️ 必须与采样同源
            ratio = torch.exp(d.log_prob(a) - LP0[i:i + mb])
            advb = ADV[i:i + mb]
            loss = (-torch.min(ratio * advb,
                               torch.clamp(ratio, 1 - clip, 1 + clip) * advb).mean()
                    + vc * ((net.value(s) - RET[i:i + mb]) ** 2).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        out = tot / max(1, (n + mb - 1) // mb)
    return out


def train_eval(spec, nf, scs, ref, best_k, tag: str) -> None:
    """训一次、评一次、报三层判据。

    ⚠️ **2026-10-13：此处原来挂着一条**已作废**的结论，现删除。**
    原文（2026-10-12 记）：「把逐步 `−d_ms` 换成末步一次性 `−makespan`，结果**逐位相同**
    ⟹ **病灶不在「信用怎么摊」**」——**这句不能成立，两个原因：**

    1. 那次实验跑在**旧代码**上（无 `Tanh`、策略**根本不动**）。策略不动 ⟹ 换任何目标都不动
       ⟹ 「逐位相同」是**真结果，但证明不了任何事**。
    2. `γ<1` 时两种摊法**数学上不是同一个目标**，不可能逐位相同（§37.44.3）。

    **⟹ 该实验作废；「信用分配」这条**从未被排除过**。**
    """
    torch.manual_seed(0)
    net = Head()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    hist = []
    for step in range(1, STEPS + 1):
        chains = []                      # ⚠️ **按链留着**，不能拼——GAE 跨链会互相污染
        eps = eps_at(step)
        for i in range(N_CHAIN):
            r = Roll(net, spec, nf, eps=eps)                 # 原文闸二
            m = spec.world().run_gated(seed_chain=9_000_000 + step,
                                       policy_pa=r.policy_pa,
                                       policy_l=r.policy_l, dt_events=0)
            r.finish(float(m["makespan"]))                  # ⚠️ 补终局（§37.44）
            if step == 1 and i == 0:                        # 一次性自检：回报是否覆盖终局
                _s = sum(b[4] for b in r.buf)
                _t = r.first_cm - float(m["makespan"])
                assert np.isclose(_s, _t, atol=1e-6), \
                    f"回报未覆盖终局：Σr={_s:.4f} vs C₀−M={_t:.4f}"
            chains.append(r.buf)
        ls = update(net, chains, opt, eps=eps)
        flat = [x for ch in chains for x in ch]
        mean_r = float(np.mean([x[4] for x in flat]))
        hist.append(mean_r)
        if step % max(1, STEPS // 6) == 0 or step == 1:
            c = Counter(b[1] for b in flat)
            top = " ".join(f"{PA_RULE_NAMES[k]}:{v / len(flat):.0%}"
                           for k, v in c.most_common(3))
            print(f"  step {step:>4}/{STEPS}　样本 {len(flat)}　loss {ls:+.3f}　"
                  f"**单步奖励均值 {mean_r:+.3f}**　本批分布 {top}", flush=True)

    net.eval()
    cnt, ms_pol = Counter(), []
    for sc in scs:
        r = Roll(net, spec, nf, greedy=True)
        m = spec.world().run_gated(seed_chain=sc, policy_pa=r.policy_pa,
                                   policy_l=r.policy_l, dt_events=0)
        r.finish(float(m["makespan"]))          # ⚠️ 否则最后一个决策不入 buf
        cnt.update(b[1] for b in r.buf)
        ms_pol.append(float(m["makespan"]))
    n = sum(cnt.values())
    print(f"\n  greedy 评测（20 场景，{n} 个决策点）：")
    for k, v in cnt.most_common(5):
        print(f"      {PA_RULE_NAMES[k]:<12}{v:>6}　{v / n:6.1%}"
              f"　（该规则纯用时 {ref[k]:.1f}）")
    top1 = cnt.most_common(1)[0][0]
    print(f"\n  ══ 【{tag}】判据 ══")
    print(f"  1. **训练曲线上升**：首档 {hist[0]:+.3f} → 末档 "
          f"{np.mean(hist[-len(hist) // 6:]):+.3f}　"
          f"{'✅' if np.mean(hist[-len(hist) // 6:]) > hist[0] else '❌'}"
          f"　（原文 Fig. 5 就是这个）")
    print(f"  2. **偏向好规则**：最常选 {PA_RULE_NAMES[top1]}"
          f"（实测 {ref[top1]:.1f}）　最好 CDR = {PA_RULE_NAMES[best_k]}"
          f"（{ref[best_k]:.1f}）　"
          f"{'✅' if top1 == best_k else '❌'}")
    print(f"  3. **赢过最好 CDR**：策略 {np.mean(ms_pol):.1f} vs "
          f"常数 {ref[best_k]:.1f}　"
          f"{'✅' if np.mean(ms_pol) < ref[best_k] else '❌'}"
          f"　（原文 Table 9 的主张；**最小版不预设一定过**）")


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout,
                    speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
    scs = [100_000 + 13 * i for i in range(20)]

    print(f"复现·最小版 · Exp({a}/{b}) · 真工序 {sum(len(o) for o in inst.jobs)} · "
          f"**只留 PA，9 条规则**\n")
    ref = {}
    for k in range(9):
        # ⚠️ **平局口径必须与策略评测一致**（2026-10-12 修）：
        # 此前这里传 `tie_rng=None`（确定性平局），而策略评测走 `run_gated` 的**环境随机平局**。
        # 实测同一条规则、同样 20 场景：`None` **153.27** vs 环境随机 **147.59**（差 5.68）
        # ⟹ 那张参照表**不可比**，曾让判据 3 假通过。**传 `tr`，与策略同口径。**
        ms = [float(spec.world().run_gated(
            seed_chain=sc, policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, tr),
            policy_l=la_rule_policy(LA_FIXED), dt_events=0)["makespan"]) for sc in scs]
        ref[k] = float(np.mean(ms))
        print(f"  参照 {k + 1}/9　{PA_RULE_NAMES[k]:<12}{ref[k]:>8.1f}", flush=True)
    best_k = min(ref, key=lambda k: ref[k])
    print("  9 条 CDR 的实测 makespan（20 场景均值）：")
    for k in sorted(ref, key=lambda k: ref[k]):
        print(f"      {PA_RULE_NAMES[k]:<12}{ref[k]:>8.1f}"
              f"{'   ← 最好' if k == best_k else ''}")

    train_eval(spec, nf, scs, ref, best_k, tag="奖励 = −d_ms（逐步）")


if __name__ == "__main__":
    main()
