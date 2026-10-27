# -*- coding: utf-8 -*-
"""**按原文口径比 HV**：策略 / RN（全随机）/ CDRs（常数派工规则），三方在**同一个实例**上。

══ 原文的比较协议（`docs/citation-cards.md:1656`）══

> 每个实例**独立运行各方法 20 次**，用 Pareto 支配概念从**重复运行解**中提取各方法的
> Pareto 解集，再比较这些解集在 **HV** 与 SP 上的表现。

原文 Table 9（`Exp(16/140)`）：NHDRL **0.8225** · RN 0.3378 · CDRs 0.2642。

⚠️ **那是他们实例上的数，不可并列。** 本脚本**在本仓生成的实例上把三方都跑出来**，
基线不引用已发表数字。

══ 两个目标（都越小越好）══

- `MS` = `completes` 的最大值（仿真已算）
- `TC` = `logistics_cost` = `c·Σy_v + uc·(travel + at)`（`des.py:1877`，c=7 / uc=0.5 / at≡0）

══ ⚠️ 为什么评测必须用**采样**而不是 argmax ══

HV 是在**重复运行的解集**上算的，它**结构性地奖励多样性**：

| 方法 | 20 次运行给出 | Pareto 集 | HV |
|---|---|---|---|
| CDR | 基本是**同一个点** | 1 个点 | **天然小** |
| 随机 | 一片点 | 几十个点 | **大** |

**⟹ 用 greedy 跑，策略也只给一个点 ⟹ HV ≈ 0 ⟹ 三方全都测不出差别。**
**（这也正是我担心的：HV 的差距可能主要来自多样性，不等于「策略更聪明」。）**

══ 参考点 ══

原文**没给**。这里取**三方所有解里的 (max MS, max TC)**，并留 1% 余量。
**换参考点会改变 HV 的绝对值，但不改变三方的排序**（同参考点下可比）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402
import torch  # noqa: E402

import shi2025_min_pa as M  # noqa: E402
from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.mo_metrics import hypervolume_2d, pareto_front, spacing  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES,  # noqa: E402
                                N_LA_RULES, PA_RULE_NAMES, apply_pa_rule,
                                la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

N_RUN = int(os.environ.get("HV_RUN", "20"))
STEPS = int(os.environ.get("HV_STEPS", "300"))
SEED = int(os.environ.get("HV_SEED", "0"))
CKPT = os.environ.get("HV_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "hv_policy.pt"))


def _mt(m: dict) -> tuple[float, float]:
    """一局的 `(MS, TC)`。`MS` = `max(completes)`；`TC` = `logistics_cost`（Eq.2）。"""
    return float(m["makespan"]), float(m["logistics_cost"])


def run(spec, nf, net, mode: str, sc: int) -> tuple[float, float]:
    """跑一局，返回 `(MS, TC)`。

    `mode`：
    - `policy` —— 策略按 π **采样**（不是 `argmax`）
    - `rn`     —— **PA 与 LA 都随机**（照原文 *"randomly selecting action for each agent"*）
    - `cdr:<k>,<j>` —— PA 规则 `k` × LA 规则 `j` 的固定组合（原文的 27 条 = 9 × 3）
    """
    if mode == "policy":
        r = M.Roll(net, spec, nf)                 # eps=0 ⟹ 纯 π 采样（**不是 argmax**）
        m = spec.world().run_gated(seed_chain=sc, policy_pa=r.policy_pa,
                                   policy_l=r.policy_l, dt_events=0)
        return _mt(m)

    rng = np.random.default_rng([sc, 99])
    if mode == "rn":
        # ⚠️ **2026-10-14 改**：原文的 RN 是 *"randomly selecting action for **each agent**"*
        #    ⟹ **LA 也必须随机**。此前这里 LA 固定走 `la_rule_policy(0)`（只有 PA 随机）。
        # ⚠️ **本版没有 OA 层** ⟹ 随机不出 OA —— **这条在最小版上天生对不满**，
        #    要完全对齐得用 3-agent 那条（`v3_hv3.py`）。
        def pol(view, pool, tie_rng=None):
            a = int(rng.integers(0, N_AEI2025_RULES))
            return apply_pa_rule(a, view, pool, tie_rng)

        m = spec.world().run_gated(
            seed_chain=sc, policy_pa=pol,
            # ⚠️ **2026-10-15 修**：同 `v3_hv3.run_rn` —— 随机选**规则**，再由规则落到车上。
            #    （此前返回规则下标 `0..2` 当车号用；`n_agv=6` 时越界检查不触发，
            #     实际变成「随机派 0/1/2 号车」，半个车队闲置。）
            policy_l=lambda v, cv: int(la_rule_policy(
                int(rng.integers(0, N_LA_RULES)))(v, cv)), dt_events=0)
        return _mt(m)

    k, j = (int(x) for x in mode.split(":")[1].split(","))
    m = spec.world().run_gated(
        seed_chain=sc,
        policy_pa=lambda v, p, tr=None: apply_pa_rule(k, v, p, tr),
        policy_l=la_rule_policy(j), dt_events=0)
    return _mt(m)


def main() -> None:
    inst = named(int(os.environ.get("HV_N", "140")),
                 int(os.environ.get("HV_JOBS", "16")))
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    scs = [100_000 + 13 * i for i in range(N_RUN)]
    print(f"HV 口径 · Exp({int(os.environ.get('HV_JOBS', '16'))}/"
          f"{int(os.environ.get('HV_N', '140'))}) · 每方法 {N_RUN} 次独立运行 · "
          f"参考点 = 三方 (max MS, max TC)\n", flush=True)

    # ── 策略：先训，再**采样**评测 ──
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location="cpu")
        net = M.Head(M.F_PA)
        net.pi[2] = torch.nn.Linear(M.HID_A, N_AEI2025_RULES)
        net.load_state_dict(sd)
        print(f"  策略：从 ckpt 读入 `{CKPT}`")
    else:
        torch.manual_seed(SEED)
        net = M.Head(M.F_PA)
        net.pi[2] = torch.nn.Linear(M.HID_A, N_AEI2025_RULES)
        opt = torch.optim.Adam(net.parameters(), lr=M.LR)
        for step in range(1, STEPS + 1):
            chains = []
            eps = M.eps_at(step)
            for _i in range(4):
                r = M.Roll(net, spec, nf, eps=eps)
                m = spec.world().run_gated(seed_chain=9_000_000 + step,
                                           policy_pa=r.policy_pa,
                                           policy_l=r.policy_l, dt_events=0)
                r.finish(float(m["makespan"]))
                chains.append(r.buf)
            M.update(net, chains, opt, eps=eps)
        torch.save(net.state_dict(), CKPT)
        print(f"  策略：训练 {STEPS} 步完成，**已存 ckpt** `{CKPT}`")

    # ── 两个方法（原文 Table 9 的前两列）──
    pts: dict[str, np.ndarray] = {}
    for tag in ("policy", "rn"):
        pts[tag] = np.asarray([run(spec, nf, net if tag == "policy" else None,
                                   tag, sc) for sc in scs])
        print(f"  {tag:<8}{len(pts[tag]):>4} 局完成", flush=True)

    # ── 27 条 CDR（原文：9 个 PA 动作 × 3 个 LA 动作的配对组合）──
    # ⚠️ **2026-10-14 改**：此前只做 9 条（PA 变、LA 固定）—— 与原文的 27 条不符。
    cdr_pts, cdr_tag = [], []
    for k in range(N_AEI2025_RULES):
        for j in range(N_LA_RULES):
            cdr_pts.append(np.asarray([run(spec, nf, None, f"cdr:{k},{j}", sc)
                                       for sc in scs]))
            cdr_tag.append((k, j))
    print(f"  cdr    {len(cdr_pts)} 条 × {N_RUN} 局 = {len(cdr_pts) * N_RUN} 局完成\n",
          flush=True)

    # ── 逐方法的 MS / TC 明细 ──
    print(f"  {'方法':<9}{'解数':>5}{'MS 最小':>10}{'MS 均值':>10}{'MS 最大':>10}"
          f"{'TC 最小':>10}{'TC 均值':>10}{'TC 最大':>10}")
    print("  " + "-" * 76)
    for tag, P in pts.items():
        print(f"  {tag:<9}{len(P):>5}{P[:, 0].min():>10.1f}{P[:, 0].mean():>10.1f}"
              f"{P[:, 0].max():>10.1f}{P[:, 1].min():>10.1f}{P[:, 1].mean():>10.1f}"
              f"{P[:, 1].max():>10.1f}")
    print("  " + "-" * 76)
    print("  ⚠️ `MS`/`TC` 是**两个目标分开看**；HV 是**两个目标一起看**。两者排序可以不同。")

    # ── 参考点 ──
    allp = np.vstack(list(pts.values()) + cdr_pts)
    ref = (float(allp[:, 0].max()) * 1.01, float(allp[:, 1].max()) * 1.01)
    print(f"\n  参考点 = ({ref[0]:.1f}, {ref[1]:.1f})（**本脚本自定**，原文未给）\n")

    # ── HV / SP ──
    hv = {}
    print(f"  {'方法':<9}{'Pareto 点数':>12}{'HV':>12}{'SP':>10}")
    print("  " + "-" * 43)
    for tag, P in pts.items():
        front = pareto_front(P)
        hv[tag] = hypervolume_2d(front, ref)
        sp = spacing(front) if len(front) > 2 else float("nan")
        print(f"  {tag:<9}{len(front):>12}{hv[tag]:>12.4f}{sp:>10.4f}")
    print("  " + "-" * 43)
    print(f"\n  **策略 / RN = {hv['policy'] / hv['rn']:.2f}×**")

    # ── 27 条规则：逐条 HV ＋ 最好的 3 条（原文那个列表）＋ 优良率 ──
    cdr_hv = np.asarray([hypervolume_2d(pareto_front(P), ref) for P in cdr_pts])
    cdr_ms = np.asarray([P[:, 0].mean() for P in cdr_pts])
    cdr_tc = np.asarray([P[:, 1].mean() for P in cdr_pts])
    print("\n  ── 27 条 CDR 里 **MS 均值最好的 3 条**（原文同款列）──")
    print(f"  {'PA 规则':<12}{'LA 规则':<16}{'MS 均值':>10}{'TC 均值':>10}{'HV':>10}")
    for i in np.argsort(cdr_ms)[:3]:
        k, j = cdr_tag[i]
        print(f"  {PA_RULE_NAMES[k]:<12}{LA_RULE_NAMES[j]:<16}"
              f"{cdr_ms[i]:>10.1f}{cdr_tc[i]:>10.1f}{cdr_hv[i]:>10.4f}")
    print(f"\n  27 条 CDR 的 HV：最大 **{cdr_hv.max():.4f}**　中位 {np.median(cdr_hv):.4f}"
          f"　最小 {cdr_hv.min():.4f}")
    print(f"  **策略 / 最好 CDR = {hv['policy'] / cdr_hv.max():.2f}×**")
    n_win = int((hv["policy"] >= cdr_hv).sum())
    print(f"  **优良率（胜过或等于的规则数 / 27）= {n_win}/{len(cdr_pts)} = "
          f"{n_win / len(cdr_pts):.0%}**")
    print("\n  ⚠️ **原文 Table 9 没有「CDRs 汇总」这一列** —— 它是「27 条规则**逐条**报 HV」"
          "＋「优良率」。这里的 `策略/最好 CDR` 与优良率才是原文那两个量。")
    print("  ⚠️ 原文 `Exp(16/140)`：NHDRL 0.8225 · RN 0.3378 · 最好 CDR 0.2642 ·"
          " DQN 0.5487 · DDPG 0.5628（**他们的实例，不可并列**）")


if __name__ == "__main__":
    main()
