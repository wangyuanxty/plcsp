# -*- coding: utf-8 -*-
"""**`K=3 / seed 9` 为什么冻死** —— 300 步、约 59000 个样本，策略一步没动。

══ 事实（§37.36，已测）══

起点锁死在 `SRPT+MINU`（279.6，最差那条），选对率全程 **0.0%**，
greedy 分布 `SRPT+MINU` **100%**，终点 makespan **279.6 = 起点**。
**而同样起点、动作数 2 的那档，300 步爬回了最优（147.6）。**

══ 待查的机制（三条互斥的猜想）══

| # | 猜想 | 若成立，应看到 |
|---|---|---|
| **A** | **探索死了**：softmax 是 delta，另外两个动作概率 ~`e^{-27}`，**采样永远采不到** ⟹ 它们永远拿不到梯度 | 初始概率里非选中动作是 0 量级；加 ε 探索就活 |
| **B** | **梯度算错**：`update()` 里优势/ratio/loss 有 bug | 梯度范数为 0 或方向恒定 |
| **C** | **全批优势归一化把唯一信号抹掉了** | 优势的均值恒为 0；逐链回报**全都一样** |

⚠️ **C 与 A 是同一根链条的两段**：既然样本全是同一个动作，
**减掉全批均值就等于把「这个动作很差」这条信息整个删掉** ⟹ 剩下的只有仿真噪声。

══ 本脚本 ══

`FREEZE_MODE=log`（默认）：跑训练，逐步记诊断量。
`FREEZE_MODE=eps`：**同一份起点，只加 ε 探索**（`FREEZE_EPS`，默认 0.1）。
  —— 这是单变量反事实：**能活 ⟹ 病灶是探索，不是 PPO。**
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
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import PA_RULE_NAMES, apply_pa_rule, la_rule_policy  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

K = int(os.environ.get("FREEZE_K", "3"))
SEED = int(os.environ.get("FREEZE_SEED", "9"))
STEPS = int(os.environ.get("FREEZE_STEPS", "300"))
NCH = int(os.environ.get("FREEZE_CHAIN", "4"))
MODE = os.environ.get("FREEZE_MODE", "log")            # "log" | "eps"
EPS = float(os.environ.get("FREEZE_EPS", "0.1"))
RANK = [3, 0, 5, 2, 4, 6, 8, 1, 7]
ACT = (tuple(RANK[round(i * (len(RANK) - 1) / (K - 1))] for i in range(K))
       if K > 1 else (RANK[0],))


class RollE:
    """与 `RollK` 同形；`eps > 0` 时**以 ε 概率均匀抽一个动作**（只加这一个变量）。"""

    def __init__(self, net, spec, nf, eps: float = 0.0, greedy: bool = False) -> None:
        self.net, self.nf, self.eps, self.greedy = net, nf, eps, greedy
        self.buf, self._cm = [], None
        self._f, self._a, self._lp, self._v = None, 0, 0.0, 0.0
        self.rng = np.random.default_rng(12345)

    @torch.no_grad()
    def _pick(self, f):
        ft = torch.as_tensor(f)[None]
        d = self.net.dist(ft)
        if self.greedy:
            a = int(d.logits.argmax())
        elif self.eps > 0.0 and self.rng.random() < self.eps:
            a = int(self.rng.integers(0, K))          # **均匀探索**
        else:
            a = int(d.sample())
        return a, float(d.log_prob(torch.tensor(a))), float(self.net.value(ft)[0])

    def policy_pa(self, view, pool, tie_rng=None):
        snap = view.snap
        cm = float(snap.now) + max(
            (float(m.remaining_min) + float(m.backlog_min) for m in snap.machines),
            default=0.0)
        raw = self.nf.pa(snap, 0.0, (5.0, 5.0))
        if self._cm is not None:
            self.buf.append((self._f, self._a, self._lp, self._v, -(cm - self._cm)))
        self._cm = cm
        self._f = raw
        self._a, self._lp, self._v = self._pick(self._f)
        return apply_pa_rule(ACT[self._a], view, pool, tie_rng)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(0, (0.0,))(view, cand_v))


def main() -> None:
    inst = named(140, 16)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    scs = [100_000 + 13 * i for i in range(20)]
    ref = {k: float(np.mean([float(spec.world().run_gated(
        seed_chain=sc, policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, tr),
        policy_l=la_rule_policy(0), dt_events=0)["makespan"]) for sc in scs]))
        for k in ACT}
    best = min(ref, key=lambda k: ref[k])
    print(f"K={K} · seed={SEED} · 模式 **{MODE}**"
          + (f" · ε={EPS}" if MODE == "eps" else "")
          + " · 动作集合 " + " ".join(f"{PA_RULE_NAMES[a]}({ref[a]:.0f})" for a in ACT))

    torch.manual_seed(SEED)
    net = M.Head(M.F_PA)
    net.pi[2] = torch.nn.Linear(M.HID_A, K)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)

    # ── 起点：**精确**概率（不是三位小数，要看非选中动作到底有多小）──
    r0 = RollE(net, spec, nf)
    spec.world().run_gated(seed_chain=9_000_001, policy_pa=r0.policy_pa,
                           policy_l=r0.policy_l, dt_events=0)
    F0 = torch.as_tensor(np.stack([b[0] for b in r0.buf]), dtype=torch.float32)
    with torch.no_grad():
        P0 = torch.softmax(net.pi(F0), -1).mean(0).numpy()
    print("  起点策略（状态平均）："
          + "　".join(f"{PA_RULE_NAMES[ACT[j]]} {P0[j]:.3e}" for j in range(K)))
    print(f"  非最大动作的概率之和 = {1.0 - P0.max():.3e}"
          f"　（每步采 {NCH} 链×~124 点 ⟹ 期望采到次数 ≈ "
          f"{(1.0 - P0.max()) * NCH * len(r0.buf) * STEPS:.2e}）\n")

    bi = ACT.index(best)
    print(f"  {'step':>5}{'p_max':>10}{'熵':>9}{'梯度范数':>11}"
          f"{'逐链回报(4条)':>34}{'选对率':>9}")
    for step in range(1, STEPS + 1):
        chains, rets = [], []
        for _i in range(NCH):
            r = RollE(net, spec, nf, eps=(EPS if MODE == "eps" else 0.0))
            m = spec.world().run_gated(seed_chain=9_000_000 + step,
                                       policy_pa=r.policy_pa, policy_l=r.policy_l,
                                       dt_events=0)
            chains.append(r.buf)
            rets.append(-float(m["makespan"]))
        if step == 1:
            before = [p.detach().clone() for p in net.pi.parameters()]
        M.update(net, chains, opt)
        gnorm = float(np.sqrt(sum(float((p.grad ** 2).sum())
                                 for p in net.pi.parameters() if p.grad is not None)))
        if step % max(1, STEPS // 6) == 0 or step == 1:
            flat = [b for ch in chains for b in ch]
            F = torch.as_tensor(np.stack([b[0] for b in flat]), dtype=torch.float32)
            with torch.no_grad():
                P = torch.softmax(net.pi(F), -1).mean(0).numpy()
            c = Counter(b[1] for b in flat)
            n = sum(c.values())
            print(f"  {step:>5}{float(P.max()):>10.6f}"
                  f"{float(-(P * np.log(P + 1e-12)).sum()):>9.4f}{gnorm:>11.3e}"
                  f"{'  '.join(f'{x:>7.1f}' for x in rets):>34}"
                  f"{c.get(bi, 0) / n:>9.1%}", flush=True)

    if MODE == "log":
        d = [float((p - q).abs().max().detach()) for p, q in
             zip(net.pi.parameters(), before)]
        print(f"\n  300 步后，策略头参数相对起点的最大改动：**{max(d):.3e}**"
              f"　（≈0 ⟹ 策略根本没动）")
    net.eval()
    cnt, ms = Counter(), []
    for sc in scs:
        r = RollE(net, spec, nf, greedy=True)
        m = spec.world().run_gated(seed_chain=sc, policy_pa=r.policy_pa,
                                   policy_l=r.policy_l, dt_events=0)
        cnt.update(b[1] for b in r.buf)
        ms.append(float(m["makespan"]))
    n = sum(cnt.values())
    print("  greedy 分布：" + "　".join(
        f"{PA_RULE_NAMES[ACT[j]]}:{v / n:.0%}" for j, v in sorted(cnt.items(),
                                                                 key=lambda x: -x[1])))
    print(f"  策略 makespan **{np.mean(ms):.1f}**　本档最好 CDR {ref[best]:.1f}")


if __name__ == "__main__":
    main()
