# -*- coding: utf-8 -*-
"""T3 预算标定：在**参考调度**上测出六条受控约束的激活量 `aᵢ^ref`，产出冻结表。

标定流程（`docs/t3-design.md` §4.1，与 ⑧ 交期的 `m16_due_calib.py` 同一套做法）
-------------------------------------------------------------------------------
1. 在**参考调度**（每工序取最短候选 + AGV 轮询，`seed_layout=0`、`seed_chain=0`）上
   一次跑测出六条约束的**被迫激活量** `aᵢ^ref`（口径见 `env/t3_budget.ACTIVATION_SOURCES`）；
2. 预算 `bᵢ = ratio × aᵢ^ref`（`ratio` 默认 0.5 = 设计原文"参考档的 50%"的**归一化**形式——
   表里只冻 `aᵢ^ref`，`ratio` 是模块常量 `BUDGET_RATIO`，双侧判据调它）；
3. 逐实例、逐口径冻结成表（`env/t3_budget.A_REF` / `A_REF_MATRIX`），未标定实例显式报错。

⚠️ **标定环境是表的一部分**：`pm_interval=10`（⑫ 要真的逾期）+ `battery_low=0` +
小电池 `0.10 kWh`（⑪ 要真的耗尽）——默认档下这两条约束的激活量恒 0（实测 mk01：
默认电池全程耗 0.1–0.27 kWh、默认 `pm_interval=120` 在 mk01/mk02 从不逾期），
没有参考水平可归一化。环境的**唯一真相**在 `t3_budget.calibration_cfg` /
`calibration_layout`（脚本与对拍测试共用，两处不可能漂）。

⚠️ **换 cfg / 换布局 ⟹ 表作废**（同 ⑧ 的"误期率落带是 cfg 条件的"）。电池是**布局**的
属性，故环境必须 cfg + 布局成对钉住。

⚠️ **双侧判据（设计 §4.2）本脚本只做"可算的那一半"**：下界（"策略改进后 â 仍不得为 0"）
与上界（"动作分布不退化"）都要**跑训练后的策略**，脚本不训练 ⟹ 如实声明，不假装验过。
上界由 `plcsp/tests/test_t3_lagrangian.py` 的退化守卫（动作使用率的中间带）与训练读数承担。

⚠️ **矩阵口径（`--transport matrix`）实测：10 个实例只有 4 个六条全活**（mk01/mk05/mk08/mk10）。
另 6 个的 ② 有限缓冲在参考调度上恒 **0**（矩阵行程是分钟级 ⟹ AGV 慢，机台把输入缓冲排空了
才等到下一趟投递），mk02/mk04 的 ③ 也为 0。**参考水平为 0 ⟹ 无法归一化**（`â = a / 0`），
故这些实例**不进表**（查表显式报错）。要覆盖它们须另选标定环境（更紧的缓冲 / 更长的
episode）——登记为开放线索（`docs/progress-log.md` §49）。脚本对 a^ref = 0 的行**打印警告**，
由标定人决定是否贴回。

跑法::

    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m17_t3_calib
    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m17_t3_calib --instances mk01
    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m17_t3_calib --transport matrix

把打印出的表贴回 `plcsp/env/t3_budget.py`；`plcsp/tests/test_t3_budget.py` 会逐实例核对
"冻结的表 == 脚本重算的结果"（表是可复现产物，不是手抄魔数）。
"""
from __future__ import annotations

import argparse

from .env.t3_budget import (BUDGET_RATIO, T3_CONSTRAINTS, T3_LABELS, GEOMETRY, MATRIX,
                            calibration_cfg, instance_key, reference_activation)
from .env.instances import Instance, load_mk

TABLE_NAMES = {GEOMETRY: "A_REF", MATRIX: "A_REF_MATRIX"}


def _row_str(vals: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{v:.4f}" for v in vals) + ")"


def main() -> None:
    ap = argparse.ArgumentParser(description="T3 参考激活量标定（逐实例、逐口径）")
    ap.add_argument("--instances", default=",".join(f"mk{i:02d}" for i in range(1, 11)))
    ap.add_argument("--transport", choices=(GEOMETRY, MATRIX), default=GEOMETRY,
                    help="行程时间口径：geometry（原始 MK，默认）或 matrix（MKT）")
    ap.add_argument("--ratio", type=float, default=BUDGET_RATIO,
                    help="预算比例 b = ratio × a^ref（默认 0.5；表里不冻它，只冻 a^ref）")
    args = ap.parse_args()
    if not 0.0 < args.ratio < 1.0:
        raise SystemExit(f"--ratio {args.ratio} 非法：必须落在 (0, 1)")

    rows: dict[str, tuple[float, ...]] = {}
    hdr = "".join(f"{T3_LABELS[c]:>14}" for c in T3_CONSTRAINTS)
    print(f"口径 = {args.transport}｜约束序 = {T3_CONSTRAINTS}")
    print("标定环境 = t3_budget.calibration_cfg/layout（pm_interval=10、battery_low=0、"
          "小电池 0.10 kWh、seed_layout=0、seed_chain=0、⑧ 关）")
    print(f"{'inst':<7}" + hdr)
    for name in args.instances.split(","):
        if args.transport == MATRIX:
            from .env.mkt import load_mkt
            inst: Instance = load_mkt(name).base
        else:
            inst = load_mk(name)
        cfg = calibration_cfg(inst, args.transport)
        act = reference_activation(inst, cfg=cfg, caliber=args.transport)
        vals = tuple(act[c] for c in T3_CONSTRAINTS)
        rows[instance_key(inst) or name] = vals
        print(f"{name:<7}" + "".join(f"{v:>14.4f}" for v in vals), flush=True)
        zero = [T3_LABELS[c] for c, v in zip(T3_CONSTRAINTS, vals) if v <= 0.0]
        if zero:
            print(f"  ⚠️ {name}：{'、'.join(zero)} 的参考激活量为 0 —— 该约束**无法归一化**"
                  f"（â = a / a^ref 无定义）。请换标定环境或把它移出 T3 约束集。")
    print(f"\n# ── 贴回 plcsp/env/t3_budget.py 的 {TABLE_NAMES[args.transport]} ──")
    print(f"# 标定环境见模块 docstring；b = {args.ratio} × a^ref（归一化后即 â 的目标水平）")
    print(f"{TABLE_NAMES[args.transport]}: dict[str, tuple[float, ...]] = {{")
    for name, vals in rows.items():
        print(f'    "{name}": {_row_str(vals)},')
    print("}")
    print(f"\n# 原始单位下的预算 b = {args.ratio} × a^ref（供双侧判据复核）：")
    for name, vals in rows.items():
        print(f"#   {name}: " + "｜".join(
            f"{T3_LABELS[c]}={args.ratio * v:.4g}" for c, v in zip(T3_CONSTRAINTS, vals)))


if __name__ == "__main__":
    main()
