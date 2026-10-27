# -*- coding: utf-8 -*-
"""`Exp(16/140)` 的**精确 makespan 下界**（CP-SAT）——判「我们的规则离最优多远」。

══ 这个脚本回答什么问题 ══

`docs/v3/open-question-nhdrl.md` §4.33.4 / §4.34.5 剩下的唯一缺口是**一句口径无关的话**：

    原文：最好 CDR 的 MS = 612，RN = 428   ⟹ 27 条规则打不出好工期
    本仓：最好 CDR 的 MS = 293，RN = 394   ⟹ 规则打得出，还比 RN 好

**但「293」这个数只有跟最优解比才有意义。** 两种可能：
- **最优 ≈ 250–290** ⟹ 我们的规则**已接近最优**，车间没有余量给策略 ⟹ 缺口在环境。
- **最优 ≤ 150** ⟹ 我们的规则**和原文的规则一样差 2 倍**，只是 RN 更差 ⟹ 缺口在训练。

**这个脚本把那条尺子造出来。**

══ 用的是仓库里已有的东西 ══

`plcsp/baselines/exact_cpsat.py`（Yao, Liu, Li, Gao, *IEEE T-ASE* 23:2378–2390, 2026 的 NCP2
模型用 CP-SAT 重写）。⚠️ 它只吃**矩阵口径**（`inst.trans_time_full`），而 Shi 实例走几何口径
⟹ 本脚本把几何**换算成矩阵**，口径与 `des.AgvSim.leg_min` **逐字同源**
（`dock_distance_matrix` 的最短路 ÷ `eff_speed` ÷ 车速 ÷ 60）。

══ 两处刻意的取整，方向都是**保证下界性** ══

1. 矩阵元素**向下**取到 `1e-3` min ⟹ 模型时间 ≤ DES 时间 ⟹ 模型最优 ≤ DES 最优 ⟹ **仍是下界**。
2. `include_return=True`（DES 的 `completes[j]` 就是回站时刻）、`initial_travel="free"`
   （DES 首趟不付空载）——两条都与 `ExactSpec` 的默认值一致。

⚠️ **不跑 `rule_bound`**：那条路要 `plcsp` 的老 `Instance`，本适配器不是。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # 仓根


import numpy as np  # noqa: E402

from plcsp.baselines.exact_cpsat import (ExactSpec, greedy_schedule,  # noqa: E402
                                         solve, time_scale)
from plcsp_v3.env import layout as layout_mod  # noqa: E402
from plcsp_v3.env.corridors import (build_corridor_graph,  # noqa: E402
                                    dock_distance_matrix)
from plcsp_v3.env.shi_instances import (EVAL_IDX, N_AGV_SHI,  # noqa: E402
                                        eval_instance)
from plcsp_v3.q1_flat import SHI_AISLE_W, SHI_SHOP_LEN  # noqa: E402

A = int(os.environ.get("EX_A", "16"))
B = int(os.environ.get("EX_B", "140"))
N_INST = int(os.environ.get("EX_NINST", "20"))
EIDX = int(os.environ.get("EX_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("EX_SHOPLEN", str(SHI_SHOP_LEN)))
SCHEME = os.environ.get("EX_SCHEME", "random")
TIME_LIMIT = float(os.environ.get("EX_TL", "1800"))
# ⚠️ `EX_SM=1` ⟹ **同机台相邻工序也生成零时长运输任务**（= 本仓 DES 的逐字口径）。
#    模型默认 `False`（NCP2：同机台不存在运输任务）⟹ 那 7 条任务被整条抹掉。
SAME_MACHINE = os.environ.get("EX_SM", "0") == "1"
AGV_SPEED = 0.5          # = `SimConfig.agv_speed_mps`（des.py 默认）
QUANT = 1000             # 矩阵量化到 1/1000 min（= 0.06 s）；**向下取**，见模块 docstring


def build_layout(inst):
    """与 `q1_flat.build` 的几何块**逐字一致**（否则矩阵与 DES 不同源）。"""
    n_cols = layout_mod.grid_shape(inst.n_machines, None)[1]
    pitch = SHOPLEN / (n_cols + 1)
    side = pitch - SHI_AISLE_W
    return layout_mod.sample_layout(
        inst.n_machines, seed=0, cell_w=side, cell_h=side,
        aisle_w=SHI_AISLE_W, lu_connector_m=pitch, n_cols=None,
        cap_fixed=None, scheme=SCHEME, n_agv=N_AGV_SHI)


def travel_units(lay, eff_speed: float) -> np.ndarray:
    """(m+1)×(m+1) 行程时间**整数矩阵**，**单位 = 1/1000 分钟**。

    ⚠️ **为什么整体换成 1/1000 分钟**（2026-10-16）：`exact_cpsat` 内部有**两处**
    `int(round(·))` 把时长当**整分钟**用 —— `greedy_schedule` 的 `d = int(round(tv(...)))`
    与 `_crude_upper_bound` 的 `tmax = max(max(r) for r in travel_matrix(inst))`。
    本仓矩阵是 `40 m ÷ 0.5 m/s ÷ 60 = 1.333…` 这种小数，于是：
    ① 构造解落在整分钟上 ⟹ `validate_schedule` 当场判「行程不符」；
    ② 地平线被撑到 **2.6 × 10⁹** ⟹ 60 s 内连一个可行解都找不到。
    **把加工时长与行程时间**一并**乘以 1000（都是整数）后，`time_scale ≡ 1`，
    那两处 `int(round())` 变成恒等，全链路无取整、地平线也回到正确量级。**

    **向下取整**（`np.floor`）保证模型时间 ≤ DES 时间 ⟹ **下界性不破**。
    """
    g = build_corridor_graph(lay)
    nodes = sorted(g.nodes)
    assert nodes == list(range(len(nodes))), "节点号不连续 ⟹ dm 不能按节点号直接索引"
    dm = dock_distance_matrix(g)
    m = int(lay.n_machines)
    dock = [int(lay.machines[a].dock_node) for a in range(m)] + [int(lay.lu.dock_node)]
    out = np.zeros((m + 1, m + 1), dtype=np.int64)
    for a in range(m + 1):
        for b in range(m + 1):
            mins = dm[dock[a], dock[b]] / (eff_speed * AGV_SPEED) / 60.0
            out[a, b] = int(np.floor(mins * QUANT))
    np.fill_diagonal(out, 0)
    return out


class _Adapter:
    """`exact_cpsat` 只需要这几个字段（见 `build_model` / `time_scale` / `_task_table`）。"""

    transport = "matrix"

    def __init__(self, jobs, mat):
        self.jobs = jobs
        self.trans_time_full = mat
        self.n_jobs = len(jobs)
        self.n_machines = int(mat.shape[0]) - 1


def main() -> None:
    inst = eval_instance(B, A, N_INST, idx=EIDX)
    # ⚠️ 加工时长也 ×1000 —— 与行程矩阵同单位（1/1000 min），见 `travel_units`。
    jobs = [[[(int(m), float(t) * QUANT) for m, t in op] for op in job] for job in inst.jobs]
    lay = build_layout(inst)
    eff = min(1.0, max(0.4, SHI_AISLE_W / 1.5))
    mat = travel_units(lay, eff)

    n_ops = sum(len(job) for job in jobs)
    off = mat[np.triu_indices_from(mat, k=1)] / QUANT          # → 分钟，仅供打印
    print(f"**Exp({A}/{B}) 精确 makespan** · eval idx={EIDX} · 车间 {SHOPLEN:.0f} m · "
          f"scheme={SCHEME} · {N_AGV_SHI} 台车 · 时限 {TIME_LIMIT:.0f} s · "
          f"线程 {os.environ.get('EX_WORKERS', '8')}\n", flush=True)
    print(f"  实例：{len(jobs)} 作业 · 真工序 {n_ops} · {inst.n_machines} 机台")
    print(f"  行程矩阵：非对角 {len(off)} 项，min {off.min():.2f} · 中位 "
          f"{np.median(off):.2f} · max {off.max():.2f} min／趟")
    print("  （§4.33.3 的靶：原文 5.40 min/趟 · 本仓标定 5.46）\n", flush=True)

    inst_a = _Adapter(jobs, mat)
    scale = time_scale(inst_a)
    assert scale == 1, f"`time_scale` 应恒为 1（全整数），实得 {scale}"
    print(f"  缩放：`time_scale` = {scale} ✓（全整数 ⟹ 模块内两处 `int(round(·))` 是恒等）\n",
          flush=True)

    # ── 先验「构造式可行解是不是真的可行」────────────────────────────────
    # ⚠️ 必须自证：一个 MS 远小于本仓任何读数的构造解，最常见的原因是**模型比 DES 松**。
    #    判据 = **AGV 利用率**。`Σ(负载+空载) / (v × MS) > 1` 是**物理不可能**，
    #    一旦出现就说明那份调度在我们 DES 里跑不出来（模型漏了约束）。
    gd = greedy_schedule(inst_a, N_AGV_SHI, include_return=True,
                         same_machine_transport=SAME_MACHINE, initial_travel="free")
    busy: dict[int, float] = {}
    seq: dict[int, list[dict]] = {}
    n_present = 0
    for t in gd["tasks"]:
        if t["present"] == 1:
            n_present += 1
            busy[int(t["agv"])] = busy.get(int(t["agv"]), 0.0) + (t["end"] - t["start"])
            seq.setdefault(int(t["agv"]), []).append(t)
    dead_list: list[float] = []
    for k, ts in seq.items():
        ts.sort(key=lambda x: x["start"])
        prev_dest = None
        for t in ts:
            if prev_dest is not None:
                dead_list.append(float(mat[prev_dest, int(t["origin"])]) / QUANT)
            prev_dest = int(t["dest"])
    gms = float(gd["makespan"]) / QUANT
    tot = sum(busy.values()) / QUANT
    dl = np.asarray(dead_list)
    dd = float(dl.sum())
    print(f"  构造式可行解（模型口径 · same_machine_transport={SAME_MACHINE}）："
          f"MS **{gms:.1f} min** · 运输任务 {n_present} 条")
    print(f"    AGV 负载段 {tot:.1f} min + 空载段 {dd:.1f} min = **{tot + dd:.1f} min**"
          f"　⟹ 利用率 {(tot + dd) / (N_AGV_SHI * gms) * 100:.1f}%（> 100% = 物理不可能）")
    print(f"    ⚠️ **空载 {len(dl)} 段**：0 的占 {(dl == 0).mean() * 100:.0f}% · "
          f"中位 {np.median(dl):.2f} · 均值 {dl.mean():.2f} · max {dl.max():.2f} min"
          f"　（本仓 DES 实测：空载总共 **410.7 min** ⟹ 均值 **3.1 min/段**）")
    print(f"    逐车忙 {[round(busy.get(k, 0.0) / QUANT, 1) for k in range(N_AGV_SHI)]}\n",
          flush=True)

    # ⚠️ `tier` **只能填 `"A-MKT"`**（`build_model` 的口径守卫：十机制全关的确定性档）。
    #    它在本脚本里只被 `rule_upper_bound` 用到，而那条路我们不走（`rule_bound=False`）。
    spec = ExactSpec(tier="A-MKT", n_agv=N_AGV_SHI, time_limit_s=TIME_LIMIT,
                     include_return=True, same_machine_transport=SAME_MACHINE,
                     initial_travel="free", exact_deadhead=False,
                     workers=int(os.environ.get("EX_WORKERS", "8")))
    # ⚠️ **`upper_bound` 留 `None`** ⟹ `solve` 用 `greedy_schedule` 的 makespan 当地平线
    #    （比 `_crude_upper_bound` 紧得多，且带一个可行初始解）。整份数据是整数 ⟹
    #    greedy 内那处 `int(round(·))` 现在是恒等，构造解能过 `validate_schedule`。
    res = solve(inst_a, spec, name=f"Exp({A}/{B})", rule_bound=False, warm_start=True)
    ub = "—" if res.objective is None else f"{res.objective / QUANT:.2f}"
    lb = "—" if res.dual_bound is None else f"{res.dual_bound / QUANT:.2f}"
    print(res.summary(), flush=True)
    print(f"\n  ⭐ **认证下界 LB = {lb} min**　最好可行解 UB = {ub} min"
          f"　（模块读数单位 = 1/1000 min，已换算）\n")
    if res.dual_bound is None or res.dual_bound <= 0:
        print("  ⚠️ **没有拿到非零下界** —— 时限内 CP-SAT 一条都没证出来。"
              "加大 `EX_TL` 重跑；**这个结果不能当结论用**。")
        return
    lo = float(res.dual_bound) / QUANT
    for tag, ms in (("最好 CDR（§4.33.2 均值）", 293.0), ("RN（§4.33.2 均值）", 393.8)):
        print(f"  {tag}: MS {ms:.1f} ⟹ 认证间隙 **{(ms - lo) / lo * 100:.1f}%**"
              f"（∵ 界不依赖任何策略）")


if __name__ == "__main__":
    main()
