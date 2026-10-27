# -*- coding: utf-8 -*-
"""**核 `LATEST_IN_TIME`（原文 Rule 3）**：它凭什么赢过 `MINCOST`？

══ 为什么值得单独核 ══

27 条 CDR 里 **MS 最小的前 5 名全是 `LATEST_IN_TIME`**（另见 `open-question-nhdrl.md` §4.8.2）。
而它的式子是（`hrl/rules.py:464-468`）：

```python
arrive = max(free_at, now) + empty_leg(aid)
return (0, -arrive) if arrive <= ready_at else (1, arrive)
```

**两级**：① 赶得上（`arrive <= ready_at`）的优先；② 里面取**最晚**到达。

⚠️ **`ready_at ≡ now` 时它塌成 `MINCOST`**（`des.py:2520` 写明这是旧路径的预期行为）。
所以它能赢，**前提是 `ready_at` 确实落在未来且分辨力足够**。

本探针量四件事：
1. `ready_at − now` 的分布（**是不是真的在未来**）
2. 有几个 AGV 赶得上（**第二级有没有内容**）
3. `LATEST_IN_TIME` 与 `MINCOST` **选车不同的比例**
4. 逐车 `arrive` 的**分散度**（分散度 = 0 ⟹ 这条规则没有可用的信息）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import apply_pa_rule, la_rule_policy  # noqa: E402
from plcsp_v3.q1_flat import SHI_SHOP_LEN, build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

PA = os.environ.get("LP_PA", "EAJ+SMPT")
SCHEME = os.environ.get("LP_SCHEME", "fig4")
# ⚠️ **`LP_SHOPLEN`（2026-10-16）**：车间长度 [m]，取代旧的 `LP_SCALE`（无因次倍率）。
SHOPLEN = float(os.environ.get("LP_SHOPLEN", str(SHI_SHOP_LEN)))
SHI = os.environ.get("LP_SHI", "1") == "1"
N_SC = int(os.environ.get("LP_NSC", "20"))
OVR = os.environ.get("LP_OVR", "")

REC: list[dict] = []


def make_probe(rule: str):
    inner = la_rule_policy(rule)

    def pol(view, cand):
        arr = [max(view.vehicles[a].free_at, view.now) + view.empty_leg(a)
               for a in cand]
        busy = [float(view.vehicles[a].free_at - view.now) for a in cand]
        pick = int(inner(view, cand))
        REC.append(dict(
            dt=float(view.ready_at - view.now),
            n_elig=sum(1 for a in cand if arr[a] <= view.ready_at),
            arrive_spread=float(max(arr) - min(arr)),
            lit=pick,
            mincost=int(min(cand, key=lambda a: view.empty_leg(a))),
            # ⚠️ **车队忙不忙**（`free_at − now`）—— Rule 3 兜底唯一的依据。
            busy_med=float(np.median(busy)),
            busy_max=float(max(busy)),
            busy_any=float(np.mean([b > 1e-9 for b in busy])),
            # 兜底（最早到达）会不会选到与 MINCOST 不同的车
            by_arrive=int(min(cand, key=lambda a: arr[a])),
        ))
        return pick

    return pol


def main() -> None:
    sim_ovr = {"due_range": 0.6}
    if OVR:
        sim_ovr.update({k: float(v) for k, v in
                        (kv.split("=") for kv in OVR.split(","))})
    inst = eval_instance(140, 16, 20)
    spec = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=sim_ovr, inst_obj=inst,
                 layout_scheme=SCHEME, **dims_from_shop_len(SHOPLEN, inst.n_machines), shi2025=SHI)[0]
    pol_pa = lambda v, p, tr=None: apply_pa_rule(PA, v, p, tr)  # noqa: E731
    print(f"LA 探针 · PA={PA} · Exp(16/140) · {N_SC} 场景 · {SCHEME}/车间长 {SHOPLEN:.0f} m\n")

    for rule in ("MINCOST", "MINU", "LATEST_IN_TIME"):
        REC.clear()
        ms = []
        for i in range(N_SC):
            m = spec.world().run_gated(seed_chain=100_000 + 13 * i,
                                       policy_pa=pol_pa,
                                       policy_l=make_probe(rule), dt_events=0)
            ms.append(float(m["makespan"]))
        d = REC
        dt = np.array([r["dt"] for r in d])
        elig = np.array([r["n_elig"] for r in d])
        spr = np.array([r["arrive_spread"] for r in d])
        bmed = np.array([r["busy_med"] for r in d])
        bmax = np.array([r["busy_max"] for r in d])
        bany = np.array([r["busy_any"] for r in d])
        diff = np.mean([r["lit"] != r["mincost"] for r in d])
        darr = np.mean([r["by_arrive"] != r["mincost"] for r in d])
        print(f"── LA = {rule} ── MS 均值 {np.mean(ms):.1f}　（{len(d)} 个决策点）")
        print(f"   `ready_at − now`：中位 {np.median(dt):7.3f}　"
              f"**> 0 的占比 {(dt > 1e-9).mean():6.1%}**　最大 {dt.max():.3f}")
        print(f"   无车赶得上：{(elig == 0).mean():6.1%}　≥2 台赶上："
              f"{(elig >= 2).mean():6.1%}")
        # ── ⚠️ **车队忙不忙** —— Rule 3 兜底唯一的依据 ──
        print(f"   **车队 `free_at − now`：中位 {np.median(bmed):6.3f}　"
              f"有车在忙的时刻占比 {np.mean(bany > 0):6.1%}　"
              f"最忙那台中位 {np.median(bmax):6.3f}**")
        print(f"   兜底（最早到达）与 MINCOST 选车不同：{darr:6.1%}"
              f"　|　逐车 arrive 分散度中位 {np.median(spr):6.3f}"
              + (f"　|　本规则与 MINCOST 不同：{diff:.1%}"
                 if rule == "LATEST_IN_TIME" else ""))
        print(flush=True)


if __name__ == "__main__":
    main()
