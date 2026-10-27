# -*- coding: utf-8 -*-
"""**派车决策的「空间」有多大** —— 与规则无关，纯几何（2026-10-17）。

══ 用户指出的区分 ══

> *"不能说是规则差别太小吧？只是各个规则实际造成的空载差不多吧"*

**对。两件事不同：**
```
① 规则没差别            ⟹ 规则集的问题
② 规则有差别、结果一样  ⟹ **车间的性质**
```
**本脚本量 ②，而且**完全不跑仿真、不看规则** —— 只看布局与车辆位置。**

══ 量什么 ══

```
在同一组车辆位置上，对同一个取货点：
  d_near = min_v leg(车 v 位置 → 取货点)
  d_rand = leg(随机一台车 → 取货点)
比值 = mean(d_rand) / mean(d_near)

⟹ 比值 ≈ 1  ⟹ **派哪台车都一样** ⟹ 派车没有空间
⟹ 比值 ≫ 1  ⟹ **空间在**，是规则没吃到
```

⚠️ 车辆位置取「六台车随机停在 18 个机台上」（不放 L/U，因为跑到后期车基本都在机台侧）。
   重复 20000 次取均值。**这不是仿真，是几何。**
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from plcsp_v3.env import layout as layout_mod  # noqa: E402
from plcsp_v3.env.corridors import (build_corridor_graph,  # noqa: E402
                                    dock_distance_matrix)
from plcsp_v3.q1_flat import SHI_AISLE_W  # noqa: E402

SHOPLEN = float(os.environ.get("GR_SHOPLEN", "240"))
N_SAMPLE = int(os.environ.get("GR_N", "20000"))
COLS = 5


def main() -> None:
    pitch = SHOPLEN / (COLS + 1)
    side = pitch - SHI_AISLE_W
    lay = layout_mod.sample_layout(
        18, seed=0, cell_w=side, cell_h=side, aisle_w=SHI_AISLE_W,
        lu_connector_m=pitch, n_cols=COLS, n_rows=None, cap_fixed=None,
        scheme="random", n_agv=6)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    dock = np.array([int(p.dock_node) for p in lay.machines])
    lu = int(lay.lu.dock_node)
    eff = min(1.0, max(0.4, SHI_AISLE_W / 1.5))
    T = dm / (eff * 0.5) / 60.0                      # 行程时长 [min]

    rng = np.random.default_rng(0)
    near, rand, far = [], [], []
    for _ in range(N_SAMPLE):
        veh = rng.choice(dock, size=6, replace=False)     # 六台车各停一台机
        pick = int(rng.choice(np.append(dock, lu)))        # 取货点（机台或 L/U）
        d = T[veh, pick]
        near.append(d.min())
        rand.append(d[int(rng.integers(0, 6))])
        far.append(d.max())
    near, rand, far = np.asarray(near), np.asarray(rand), np.asarray(far)

    print(f"**派车空间（纯几何）** · 18 机 · 6 车 · 车间 {SHOPLEN:.0f} m · "
          f"{N_SAMPLE} 次抽样\n")
    print(f"  {'':<16}{'均值':>8}{'中位':>8}{'0 的比例':>10}")
    print("  " + "-" * 42)
    for tag, v in (("最近一台", near), ("随机一台", rand), ("最远一台", far)):
        print(f"  {tag:<16}{v.mean():>8.2f}{np.median(v):>8.2f}{(v == 0).mean():>9.0%}")
    print("\n  **随机 / 最近 = {:.2f}×**　**最远 / 最近 = {:.2f}×**".format(
        rand.mean() / near.mean(), far.mean() / near.mean()))
    print("\n  ⚠️ ≈1 ⟹ 派哪台车都一样（派车没有空间）；≫1 ⟹ 空间在，是规则没吃到。")
    print("  ⚠️ 这是**几何**，不含排队、不含预订、不含规则 —— 与实现的派车规则无关。")


if __name__ == "__main__":
    main()
