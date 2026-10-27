# -*- coding: utf-8 -*-
"""**地板测试（真实环境版）**：整局用一个规则，只学 9 个 logits —— 找得到最好 CDR 吗？

══ 这台子要问什么 ══

**先把「状态」整个拿掉**：不吃 24 维特征、不做逐步决策、不用长链 GAE、不涉及状态弱信息。

```
每步：采样一条规则 k（从 9 个 logits）
整局：**从头到尾只用规则 k**
回报：−makespan
更新：**同一个 `M.Head` + 同一个 `M.update`**（输入恒为常数 ⟹ 网络退化成 9 个 logits）
```

**⟹ 它把「逐步决策 / 续接回报 / 长链估计 / 状态弱信息」全部移除，
只留下「按**真实 makespan** 学哪条规则好」。**

══ 判据（这不是「环境可不可分」的测试）══

⚠️ **臂的收益本来就可分** —— CDR 表实测：

| 实例 | 最好 | 最差 | 相对差 |
|---|---|---|---|
| `Exp(16/140)` | `EAJ+SMPT` 147.6 | `SRPT+MINU` 279.6 | 89% |
| `Exp(50/425)` | `EAJ+EAM` 295.0 | `SRPT+MINU` 514.2 | 74% |

**⟹ 本台子测的是：「9 个参数 + 可区分的臂 + 300 步梯度 —— 找不找得到最好的那个？」**

| 结果 | 结论 |
|---|---|
| **找到最好那条** | 更新/采样没问题 ⟹ 病灶在**逐步决策那一层**（续接回报 / 长链估计 / 训练—greedy 落差） |
| **找不到** | **更新 / 采样本身有问题** ⟹ 跟 24 维、跟 PPO、跟信用分配**全都无关** |

⚠️ **有意复用同一个 `M.update`** —— 它把「算法实现」和「调度」两条腿拆开。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402
import torch  # noqa: E402

import shi2025_min_pa as M  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (N_AEI2025_RULES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

STEPS = int(os.environ.get("BF_STEPS", "300"))
NCH = int(os.environ.get("BF_CHAIN", "4"))
SC0 = int(os.environ.get("BF_SC0", "100000"))
N_SC = int(os.environ.get("BF_NSC", "20"))
# ⚠️ **阶梯的第二格**（2026-10-13）：`BF_REWARD=ref` ⟹ 奖励直接用**实测均值** `ref[k]`，
# **不跑仿真** ⟹ **零环境噪声**。
#   本格 = ①（确定性地板）+ **采样**，只多这一个变量。
#   过了 ⟹ ③ 的失败归因于**环境噪声**；没过 ⟹ 采样 / `M.update` 本身有问题。
REWARD = os.environ.get("BF_REWARD", "sim")        # "sim" | "ref"


class BanditRoll:
    """一局 = **整局只用一个规则**。输入恒为常数 ⟹ 网络退化成 9 个 logits。"""

    def __init__(self, net) -> None:
        self.net, self.buf = net, []

    @torch.no_grad()
    def sample(self, eps: float = 0.0) -> int:
        """按策略采样一个规则；把 `(常数特征, 动作, logp, V, 占位奖励)` 记进 buffer。

        ⚠️ **2026-10-13 修**：此前用 `net.dist(ft)`（**π**）采样，而 `M.update` 按
        `behavior(·, eps)`（**`b = (1−ε)π + ε/K`**）算 ratio ⟹ **两边口径不一致**。
        ε 早期是 0.9 ⟹ `b ≈ 均匀` ⟹ ratio ≈ 1 ⟹ 梯度被抹平。
        **采样必须和更新同源。**
        """
        f = np.ones(1, dtype=np.float32)
        ft = torch.as_tensor(f[None])
        d = self.net.behavior(ft, eps)
        a = int(d.sample())
        self.buf.append((f, a, float(d.log_prob(torch.tensor(a))),
                         float(self.net.value(ft)[0]), 0.0))
        return a

    def set_reward(self, r: float) -> None:
        f, a, lp, v, _ = self.buf[-1]
        self.buf[-1] = (f, a, lp, v, float(r))


def reinforce(net, chains, opt, eps: float = 0.0) -> None:
    """**朴素 REINFORCE**（2026-10-13，用于隔离 `M.update`）：
    不 clip、不重复 epochs、**批均值基线**、优势**不归一化**。

    ⚠️ **2026-10-13 修**：此前基线**按链**算，而本台子每条链**只有 1 步**
    ⟹ `base` 就是那一步自己的奖励 ⟹ **优势恒为 0** ⟹ 梯度恒为 0
    ⟹ **网络一步没动**（两个不同 lr 给出**完全相同**的 greedy，这就是铁证）。
    **基线必须按**整批**算。**
    """
    flat = [b for buf in chains for b in buf]
    if len(flat) < 2:
        return
    base = float(np.mean([b[4] for b in flat]))
    S = torch.as_tensor(np.stack([b[0] for b in flat]), dtype=torch.float32)
    A = torch.as_tensor([b[1] for b in flat], dtype=torch.long)
    ADV = torch.as_tensor(np.asarray([b[4] - base for b in flat]), dtype=torch.float32)
    d = net.behavior(S, eps)
    loss = -(d.log_prob(A) * ADV).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def exact_floor(net, opt, cost: np.ndarray, steps: int, best: int) -> None:
    """**确定性地板**（2026-10-13，外部 review 的建议）：**去掉环境、去掉采样**。

    直接用 9 条规则的**实测均值** `c_k` 构造期望损失：

        J(z) = Σ_k p_k(z) · c_k        （`p = softmax(pi(constant))`）

    对无 `Tanh` 的直接 logits，`∂J/∂z_k = p_k · (c_k − J)`
    ⟹ **成本低于当前均值的臂，梯度下降应抬高它的 logit。**

    **⟹ 这一档把「更新方向对不对」与「步子够不够」一刀切开：**
    **不用仿真、不用采样、没有噪声，几秒钟。**
    """
    f = torch.ones(1, 1)
    with torch.no_grad():
        p0 = torch.softmax(net.pi(f), -1)[0].numpy()
    print(f"  起点：期望 makespan Σp·c = **{float((p0 * cost).sum()):.1f}**"
          f"　最好那条的概率 {p0[best]:.1%}")
    c = torch.as_tensor(cost, dtype=torch.float32)
    for it in range(1, steps + 1):
        z = net.pi[2](net.pi[1](net.pi[0](f)))          # ⚠️ **`Tanh` 之前**那层
        p = torch.softmax(torch.tanh(z), -1)            # ⚠️ 与 `Head` 同口径（`BOUND` 已删）
        loss = (p[0] * c).sum()
        if it == 1:                                    # 只查一次梯度方向
            dz = torch.autograd.grad(loss, z, retain_graph=True)[0][0].detach().numpy()
            j = float(loss)
            print(f"  ⟹ **第 1 步的 dJ/dz**（好的臂应为**负** = 抬它的 logit）："
                  f"　最好 `{PA_RULE_NAMES[best]}` = **{dz[best]:+.4f}**"
                  f"　最差 `{PA_RULE_NAMES[int(np.argmax(cost))]}` = "
                  f"{dz[int(np.argmax(cost))]:+.4f}"
                  f"　J₀ = {j:.1f}")
            print(f"    {'✅ 方向对（好的臂 dJ/dz < 0）' if dz[best] < 0 else '❌ **方向反了**'}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % max(1, steps // 3) == 0:
            with torch.no_grad():
                p = torch.softmax(net.pi(f), -1)[0].numpy()
                print(f"    it {it:>4}/{steps}　Σp·c "
                      f"{float((p * cost).sum()):>8.1f}　最好那条 p "
                      f"{p[best]:>6.1%}　argmax "
                      f"{PA_RULE_NAMES[int(p.argmax())]}", flush=True)
    with torch.no_grad():
        p = torch.softmax(net.pi(f), -1)[0].numpy()
    print(f"  终点：Σp·c = **{float((p * cost).sum()):.1f}**"
          f"　最好那条 p **{p[best]:.1%}**　argmax "
          f"**{PA_RULE_NAMES[int(p.argmax())]}**"
          f"　{'✅ 找到' if int(p.argmax()) == best else '❌ 没找到'}")


def main() -> None:
    N = int(os.environ.get("BF_N", "140"))
    J = int(os.environ.get("BF_JOBS", "16"))
    upd = os.environ.get("BF_UPD", "ppo")               # "ppo" | "reinforce" | "exact"
    inst = named(N, J)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    scs = [SC0 + 13 * i for i in range(N_SC)]
    print(f"地板测试（真实 makespan）· Exp({J}/{N}) · {N_SC} 场景 · "
          f"{STEPS} 步 × {NCH} 链 · 同一套 `M.update`\n", flush=True)

    ref = {}
    for k in range(N_AEI2025_RULES):
        ref[k] = float(np.mean([float(spec.world().run_gated(
            seed_chain=sc, policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, tr),
            policy_l=la_rule_policy(0), dt_events=0)["makespan"]) for sc in scs]))
    best = min(ref, key=lambda k: ref[k])
    print("  9 条 CDR 的实测 makespan（20 场景均值）：")
    for k in sorted(ref, key=lambda k: ref[k]):
        print(f"      {PA_RULE_NAMES[k]:<12}{ref[k]:>8.1f}"
              f"{'   ← 最好' if k == best else ''}")
    print(flush=True)

    torch.manual_seed(0)
    net = M.Head(1)
    net.pi[2] = torch.nn.Linear(M.HID_A, N_AEI2025_RULES)
    lr = float(os.environ.get("BF_LR", str(M.LR)))      # ⚠️ **学习率可换**
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    print(f"  更新式 **{upd}** · lr **{lr:g}** · 步数 {STEPS}\n", flush=True)

    if upd == "exact":                                  # ⚠️ 确定性地板：不碰仿真
        exact_floor(net, opt, np.asarray([ref[k] for k in range(N_AEI2025_RULES)]),
                    STEPS, best)
        return

    seen: dict[int, list] = {k: [] for k in range(N_AEI2025_RULES)}
    for step in range(1, STEPS + 1):
        chains = []
        eps = M.eps_at(step)
        for _i in range(NCH):
            r = BanditRoll(net)
            k = r.sample(eps)                           # ⚠️ 与 `M.update` 同源
            sc = scs[int(rng.integers(0, N_SC))]        # **每链随机一个场景**
            if REWARD == "ref":                         # ⚠️ 第二格：**不跑仿真**
                m = float(ref[k])
            else:
                m = float(spec.world().run_gated(
                    seed_chain=sc,
                    policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, tr),
                    policy_l=la_rule_policy(0), dt_events=0)["makespan"])
            r.set_reward(-m)
            seen[k].append(m)                           # ⚠️ **采样数据的经验臂均值**
            chains.append(r.buf)
        (reinforce if upd == "reinforce" else M.update)(net, chains, opt, eps=eps)
        if step % max(1, STEPS // 6) == 0 or step == 1:
            c = Counter(int(b[1]) for ch in chains for b in ch)
            n = sum(c.values())
            print(f"  step {step:>4}/{STEPS}　采样到最好那条 {c.get(best, 0) / n:6.1%}"
                  f"　（随机 {1 / N_AEI2025_RULES:.1%}）", flush=True)

    # ⚠️ **数据本身有没有信息** —— 用**实际采样到的**样本估计各臂均值。
    # 若这个 argmin 是对的，而策略没找到 ⟹ **数据没问题，是更新式的问题。**
    emp = {k: float(np.mean(v)) for k, v in seen.items() if v}
    ebest = min(emp, key=lambda k: emp[k])
    print(f"\n  ⚠️ **采样数据的经验臂均值**（每臂 {len(seen[ebest])}~"
          f"{max(len(v) for v in seen.values())} 个样本）：")
    for k in sorted(emp, key=lambda k: emp[k])[:4]:
        print(f"      {PA_RULE_NAMES[k]:<12}{emp[k]:>8.1f}"
              f"　（真值 {ref[k]:.1f}　n={len(seen[k])}）")
    print(f"    ⟹ 经验 argmin = **{PA_RULE_NAMES[ebest]}**"
          f"　真值 argmin = **{PA_RULE_NAMES[best]}**"
          f"　{'✅ 一致' if ebest == best else '❌ 不一致'}")

    net.eval()
    with torch.no_grad():
        f1 = torch.ones(1, 1)
        pre = net.pi[2](net.pi[1](net.pi[0](f1)))[0].numpy()   # ⚠️ **`Tanh` 之前**
        p = torch.softmax(net.pi(f1), -1)[0].numpy()           # ⚠️ **`Tanh` 之后**
    top = int(p.argmax())
    print(f"\n  **greedy（argmax）= {PA_RULE_NAMES[top]}**（该规则纯用 {ref[top]:.1f}）")
    print("  逐臂概率：" + "　".join(
        f"{PA_RULE_NAMES[k]}:{p[k]:.0%}" for k in np.argsort(-p)[:5]))

    # ⚠️ **`Tanh` 饱和检查**（`Tanh` 于 2026-10-14 加回）：`tanh'(z) = 1 − tanh²(z)`
    # z=2 ⟹ 0.071　z=3 ⟹ 0.010　z=4 ⟹ 0.001
    # **一旦 |pre| ≥ 3，从损失回传到前面所有层的梯度就衰减到 1% 以下。**
    print("\n  ⚠️ **`Tanh` 之前**的 pre-activation（判据：`|pre| ≥ 3` ⟹ 梯度 ≈ 0）：")
    for k in np.argsort(-pre)[:4]:
        t = float(np.tanh(pre[k]))
        print(f"      {PA_RULE_NAMES[k]:<12}{pre[k]:>+9.3f}　tanh {t:+.4f}"
              f"　**导数 1−tanh² {1 - t * t:.4f}**　p {p[k]:.1%}")
    print(f"    全体：|pre| 中位 {np.median(np.abs(pre)):.3f}"
          f"　最小 {np.abs(pre).min():.3f}　最大 {np.abs(pre).max():.3f}"
          f"　**|pre| ≥ 3 的臂数 {int((np.abs(pre) >= 3).sum())} / {len(pre)}**")

    ok = top == best
    print(f"\n  ⟹ **{'✅ 找到最好那条' if ok else '❌ 没找到'}**"
          f"　⟹ 病灶在**{'逐步决策那一层' if ok else '更新 / 采样本身'}**")


if __name__ == "__main__":
    main()
