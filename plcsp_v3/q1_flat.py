# -*- coding: utf-8 -*-
"""**Q1 跑数器**（设计 §12.2–§12.4）——扁平链两臂的读数。

```
python -m plcsp_v3.q1_flat --arms A B --seeds 0 1 2 --steps 200
```

**问的是**：*Learning Without Critics?*（arXiv:2511.03527）称
*"learned critics remain essential for long-horizon tasks"* —— **而调度是长时程。**
臂 A 有 critic（`R − V(s_t)`），臂 B 是现行 GRPO 口径（组内 z 化）。**其余完全相同。**

**测三样**（§12.4）：**训练曲线方差 · 收敛速度 · 扰动（③⑨）下的最终业绩。**
⚠️ **必须多种子** —— 单种子说明不了稳定性，而"稳定性"正是这里要测的东西。

══ 输出 ══

JSONL，每行一条记录（`kind` 区分）：
- `train`：每步的 `step / r_mean / r_std / loss / heads_grad / critic_grad / enc_grad`；
- `eval`：每 `--eval-every` 步的 **greedy 评测**（`makespan / energy / twt` 的均值与标准误）；
- `final`：跑完后的**扰动评测**（⑨ 按 `scaled_agv_mtbf` 激活到每局 ~4 次故障）。

⚠️ **默认写到系统临时目录**（项目硬规则 1：`*.jsonl` 不进项目根）。`--out` 可覆盖。

══ 一条必须写明的口径 ══

**两臂的优势尺度不同，这是构造性的**：臂 B 的 `z(R)` 天生均值 0、标准差 1；
臂 A 的 `R − V(s_t)` 在 `V` 还没学好时**约等于 `R`（≈ −40）**，比 z 值大 ~25 倍。
**不改。** 理由：Adam 的更新是 `lr·m̂/(√v̂+ε)`，**对整体的尺度因子近似不变**，
故两臂的**更新幅度可比**；而尺度差本身就是"学出来的基线 vs 组内基线"的一部分。
⚠️ **这条要在论文里写明**，别让审稿人以为两个梯度范数差 50 倍是我漏了归一化。
（实测 `n=8`：臂 B `heads_grad≈199`，臂 A `heads_grad≈10527`。）
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .env import layout as layout_mod
from .env.constraints import ConstraintConfig
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .env.des import SimConfig, reference_makespan, scaled_agv_mtbf
from .env.instances import Instance, load_mk
from .env.reward import EQUAL_WEIGHTS
from .hrl.agents import HierarchicalPolicy
from .hrl.flat import flat_rollout, flat_step
from .hrl.pool import ChainPool
from .hrl.train import Spec
from .nn.features import norm_context

DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "q1_flat.jsonl")

SHI_FAIL_SEED = 7300
"""**复现 Shi 2025 的「每实例故障次数 1–8」子种子**（`default_rng([layout_seed, 本值])`）。
⚠️ **必须独立成流**：借布局/故障/平局任一条流，都会把那条流整体移位（同 `PA_TIE_SEED` 的纪律）。"""

N_FAIL_HI = 8
"""原文的故障次数上界：*"每个实例的故障总次数在 **1–8** 之间随机"*。"""

SHI_SHOP_LEN = 240.0
"""**复现层**的**旧车间长度标定值 [m]**（2026-10-16 引入）—— **2026-10-18 起不再是 `build` 参数**。

新代码直接给 `cell_w/cell_h/aisle_w/lu_connector_m`。
旧调用点用 `dims_from_shop_len(...)` 迁移（它把本值换成四个显式尺寸）。

══ 几何（2026-10-16，用户裁定）══

```
机台区：**4 行 × 5 列**             （18 台机 ⟹ `grid_shape` 给 (4, 5)）
左边：  **L/U 占 1 个间距**，与机台间距一致
⟹ 节点间距 **横竖一致**（正方形网格）
⟹ pitch = shop_len / (n_cols + 1) = shop_len / 6
⟹ 车间 = 6·pitch × 4·pitch
```

⚠️ **为什么换成米**：`tt` 只由 `node_xy` 的两个和 `(cell_w+aisle_w)`、`(cell_h+aisle_w)` 决定
（`MachinePad.w/h` **全仓无消费者**，只用于画图）。所以「机位尺寸」本身不影响仿真 ——
而原来的 `cell_w=3.0 / cell_h=2.4` **全仓查不到出处**（连 `assumed` 都没标），
却通过 `3.9/4.5 = 0.867` 把网格变成了长方形。**改成正方形 + 直接给米，去掉这两个无依据的数。**

══ 标定（沿用 Fig. 6 直读的 TC 靶）══

原文 **Fig. 6(b)** 是**柱状图**，图注逐字 *"Comparison of **average objective values** …"*，左轴 MS · 右轴 TC。
`Exp(16/140)` 直读（PDF 第 17 页，8–16 倍渲染，图上的百分比标注可自校）：

| | MS | TC |
|---|---|---|
| NHDRL | ≈ 300 | ≈ 425 |
| RN | ≈ 428 | ≈ 600 |
| **best CDR** | ≈ 612 | **≈ 420** |

**⚠️ 旧记录写的 `(612, 622)` 错了** —— **622 是把那根 TC 柱按**左轴（MS 刻度）**读的**；
右轴上它 ≈ **420**。⟹ §37.50 的靶 `(608, 602)` 的 TC 高了 43%，`layout_scale` 因此被顶到 12.5。

**靶：** `TC = c·Σy_v + uc·运输`，`c=7`、`uc=0.5`（原文 §5.1），6 台车 ⟹ `42 + 0.5×运输`。
原文 best CDR 的 TC ≈ 420 ⟹ **运输工时 = 756 min**。

⚠️ **只对 TC 靶，不对 MS 靶** —— 原文 RN→CDR 的 MS 是**变差 43%**，本仓是**变好**；
把 MS「调差」才叫对齐的话，那是**抹掉要发现的缺口**。见 `open-question-nhdrl.md` §4.31.3。

⚠️ **2026-10-18：`shop_len` / `layout_scale` 都已从本函数删除。**
新代码只给 `cell_*` / `aisle_w` / `lu_connector_m`；旧调用点用 `dims_from_shop_len(...)` 迁移。

══ 标定值 `240.0`（2026-10-16 重标）══

⚠️ **173 那个值是**废的** —— 它是在 `alts_pow = 0`（旧的均匀候选数）下标的；
`SHI_ALTS_POW = -2` 换了实例 ⟹ 正确值要重标。**扫出来的结果是 240**：

```
 车间长 | 最好MS  RN MS  TC最MS | 最好TC  RN TC  TC最TC | RN HV   CDRHV
    173 | 235.4  319.9   399.9 | 301.5  406.6   301.5 | 0.4066  0.6772
   **240**| **304.3  417.5   412.3** | **424.0  546.8   424.0** | 0.4925  0.8672
 原文   | **300    428     612** | **430    600     420** | 0.3378  0.2642
```

**⟹ 车间 240 m 时「最好 MS / RN MS / 最好 TC」**三个量级全对上原文**（差 1–3%）。**

**⚠️ 而 240 不是"凑数选的" —— 它由 `TC` 靶**唯一确定**：**

```
TC = 42 + 0.5 × 运输  ·  趟数 = 124 真工序 + 16 回站 = 140
车间 240 m ⟹ 运输 764 min ⟹ 每趟 **5.46 min**
原文（Fig. 6 的 TC 420 反推）⟹ 运输 756 min ⟹ 每趟 **5.40 min**   ⟹ **差 1.1%** ✓
节点间距 = 240 / 6 = 40 m · 车速 0.5 m/s ⟹ 走一格 1.33 min ⟹ 机台两两中位 3–6 min
   —— **真实航空机加车间的量级，不夸张**（此前我用"每台机多少 m²"判"不合理"是**错的判据**）
```

⚠️ **仍有缺口**：`TC 最小那条`的 MS 本仓 **412** vs 原文 **612**（−33%）。
**即「最省运输那条规则」不够慢** —— 那是唯一剩下的缺口（见 `open-question-nhdrl.md`）。
"""

SHI_AISLE_W = 1.5
"""**复现层**的通道宽 [m] —— 也是 `des.SimConfig.eff_speed` 的**降速拐点**
（`min(1.0, max(0.4, aisle_width / 1.5))`）⟹ **取 1.5 则倍率恰好 1.0、不降速**。

⚠️ **它是**唯一**有功能含义的几何量**：`tt` 只看 `cell + aisle` 的两个和，而 `aisle_w` 另管降速。
"""

# ══════════ ⚠️ **装卸时间：曾设为 0.75，2026-10-16 归 0** ══════════
#
# **假设 (3) 逐字**（Shi et al., AEI 65:103195, 2025，PDF 第 5 页）：
#   *"The time for loading and unloading jobs on AGVs is **included in transport
#   time**."*   全文**只有这一处**提到装卸，**没有任何数值**。
#
# **两个阶段，记录在案：**
#   1. **2026-10-15**：`shi2025` 分支设 `agv_load_min = agv_unload_min = 0.75`
#      （`AGV_HANDLING_MIN`）。理由：给 0 = 把装卸整个删掉，与假设 (3) 相反。
#   2. **2026-10-16 归 0（用户裁定）**。理由：**`layout_scale` 的标定已经吸收了装卸。**
#
# **为什么归 0 才是对的：**
#   * `layout_scale` 是**按 `TC` 的总量**标的 —— 原文 Fig. 6 直读 best CDR 的 TC ≈ 420
#     ⟹ `TC = 42 + 0.5×运输` ⟹ 运输工时 = 756 min。**那 756 是**总运输工时**，
#     在原文口径下它**本来就含装卸**（假设 3 说 tt 含装卸）。
#   * ⟹ 再另设 `agv_load_min = 0.75` 是**重复计入**。
#   * ⟹ 而且 `0.75` 是**本仓编的数**，`layout_scale = 7.3` 是**从图上标的**。
#     **用「标出来的」承担装卸，比用「编的」干净**，且**不违反**假设 (3)
#     （假设只说"含在运输时间里"，**没说**要另设一个参数）。
#
# **实测支撑**（`open-question-nhdrl.md` §4.32）：
#   `scale 7.3` + 装卸 0    ⟹ TC 下界 **392.5**（原文 420 ✓）· CDR 最优点 **(0.279, 0.000)**
#   `scale 7.3` + 装卸 0.75 ⟹ TC 下界 510.1（高 21%）   · CDR 最优点 (0.016, 0.058) 又贴回角
#
# ⚠️ **机制本身不动**：`SimConfig.agv_load_min/unload_min` 仍在（默认 `0.0`），
# `sim_overrides` 仍能设 —— 扫装卸敏感性的脚本不受影响。

TARGET_FAILURES = 4.0
"""⑨ 的**激活口径**（`scaled_agv_mtbf`）：每局期望故障次数。

⚠️ 为什么必须激活：`SimConfig.agv_mtbf` 默认 480 min，而 MK01 一局只有 ~100–170 min
⟹ **默认档下 ⑨ 几乎不发生**，"故障下稳不稳"就测了个空。"""


def _reward_spec(inst, lay, g, dm, cfg, cons, reward_norm: str):
    """→ `(w, bounds)`。`bounds=None` ⟹ `scalar_reward` 用原始量纲，**逐位不变**。

    | `reward_norm` | `w` | `bounds` |
|---|---|---|
    | `"ref"`（**默认**） | `reward_weights(f^ref)` —— 现行口径 | `None` |
    | `"minmax"` | `EQUAL_WEIGHTS` | 36 条固定规则上的观测极值 |

    ⚠️ **为什么要 minmax**：原始量纲下能耗只占 3%，而 `1/f^ref` 走到另一个极端
    （标称 89%、实际驱动 19%）—— 两条都让「三目标」名不副实。
    见 `progress-log.md` §147 与 `docs/v2/current-state.md`。
    """
    if reward_norm == "none":
        # ⚠️ **快路径**（2026-10-17）：NHDRL 复现用自己的 `RewardFn`（Eq.50–53），
        #    不需要 v3 的 `w` / `bounds`。跳过 `objective_bounds` 的 **36 条规则扫描**
        #    （40 工件实测 **16 s/实例** ⟹ 40 个实例 ≈ 11 min）。默认不用 ⟹ v3 逐位不变。
        return EQUAL_WEIGHTS, None
    if reward_norm != "minmax":
        raise ValueError(f"reward_norm 只接受 ref/minmax/none，收到 {reward_norm!r}")
    from .env.des import SimWorld
    from .hrl.cdr import objective_bounds

    def _wf():                   # ⚠️ 每条都要**新建**世界（`SimWorld` 持活对象）
        return SimWorld(inst, lay, dm, cfg, graph=g, constraints=cons)

    lo, hi = objective_bounds(_wf)
    return EQUAL_WEIGHTS, (lo, hi)


def _apply_pt_pos_corr(inst, lay, corr: float):
    """**把「机台的加工快慢」与「它的位置远近」绑在一起**（2026-10-17 加，用户提出的方向）。

    ══ 为什么需要它 ══

    **当前实例**：`pt` 逐 (机台, 工序) 独立随机 ⟹ **「快」与「位置」无关**
    ⟹ `MINU`（利用率最低）均衡负载时，挑到的机台**平均距离不增不减**
    ⟹ **均衡不付运输成本** ⟹ MS 与 TC **同向** ⟹ **总有一条规则同时通吃两维**。

    **绑上相关性后**：`corr < 0` ⟹ **远处的机台更快**
    ⟹ `MINU` 的分子（已派工时）涨得慢 ⟹ 比率低 ⟹ **它持续往远处的机台送活**
    ⟹ **均衡必须付运输** ⟹ 两目标**反向** ⟹ 没有规则能同时最好。

    ══ 口径 ══

    `rank` = 该机台装卸点到 L/U 连接点最短路的**排名归一到 [0, 1]**（0 = 最近，1 = 最远）。
    `pt' = pt × (1 + corr × (rank − 0.5))`，下限 `0.2×`（不让它变成 0）。
    **`corr = 0.0`（默认）⟹ 原样返回 ⟹ 既有读数逐位不变。**

    ⚠️ **原文没有这一条** —— Fig. 4 只给拓扑、没给尺寸，也没说加工时间与位置有任何关系。
    **它是**我方推定的补充**，登记在 `docs/v3/open-question-nhdrl.md`。**
    """
    if not corr:
        return inst
    import dataclasses
    import numpy as _np
    lu = int(lay.lu.dock_node)
    dist = _np.array([float(spec_graph_path_len(lay, int(p.dock_node), lu))
                      for p in lay.machines])
    order = _np.argsort(_np.argsort(dist)).astype(float)
    rank = order / max(1.0, float(len(order) - 1))
    scale = 1.0 + float(corr) * (rank - 0.5)
    scale = _np.maximum(0.2, scale)
    jobs = [[tuple((int(m), float(t * scale[int(m)])) for m, t in op) for op in job]
            for job in inst.jobs]
    return dataclasses.replace(inst, jobs=jobs)


def spec_graph_path_len(lay, a: int, b: int) -> float:
    """两节点间的**最短行程时长** [min] —— 只给 `_apply_pt_pos_corr` 排距离用。"""
    from .env.corridors import build_corridor_graph, dock_distance_matrix
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    eff = min(1.0, max(0.4, float(lay.grid.aisle_w) / 1.5))
    return float(dm[int(a), int(b)]) / (eff * 0.5) / 60.0


def dims_from_shop_len(shop_len: float, n_machines: int, *,
                       layout_cols: int | None = None,
                       aisle_w: float = SHI_AISLE_W) -> dict:
    """**旧口径迁移用**：`shop_len` → 四个显式尺寸。

    旧语义（`pitch = shop_len / (n_cols + 1)`，L/U 占一个 pitch）：

    ```
    cell_w = cell_h = pitch − aisle_w
    lu_connector_m = pitch
    ```

    ⚠️ **新代码不要再把 `shop_len` 当主参数。**
    这个函数只给尚未迁移的复现脚本用；v3 主线直接给 `cell_*` / `aisle_w` / `lu_connector_m`。
    """
    n_cols = (int(layout_cols) if layout_cols
              else layout_mod.grid_shape(int(n_machines), None)[1])
    pitch = float(shop_len) / (n_cols + 1)
    aw = float(aisle_w)
    return {"cell_w": pitch - aw, "cell_h": pitch - aw,
            "aisle_w": aw, "lu_connector_m": pitch}


def build(seed: int, *, tau: float | None = None, disturb: bool = False,
          n_agv: int = 3, with_value: bool = False, device: str = "cpu",
          seq_ma: bool = False,
          instance: str = "mk01", pm_interval: float | None = None,
          target_failures: float | None = None, agv_n_failures: int = 0,
          finite_buffer: bool = True, constraints: ConstraintConfig | None = None,
          sim_overrides: dict | None = None, layout_seed: int | None = None,
          # ⚠️ **机位 / 走廊 / L-U 连接段的显式尺寸**（2026-10-18 取代 `layout_scale` / `shop_len`）。
          #    `None` ⟹ 默认 3.0 × 2.4 m、走廊 1.5 m、连接段 3.0 m。
          #    ⚠️ **`cell_w/cell_h` 只影响画图**（`MachinePad.w/h` 无消费者）；
          #    动力学只看 `pitch_x = cell_w + aisle_w`、`pitch_y = cell_h + aisle_w`。
          cell_w: float | None = None, cell_h: float | None = None,
          aisle_w: float | None = None, lu_connector_m: float | None = None,
          layout_cols: int | None = None,
          # ⚠️ **默认 "grid" = v3 原行为**（照 AEI 103216 Fig.2 的网格）。
          #    `"fig4"` = Shi 2025 Fig. 4 的拓扑（复现层显式传）。
          layout_scheme: str = "grid",
          # ⚠️ **簇间距**（2026-10-16 加）：列号 `≥ cluster_split` 的节点右移 `cluster_gap` 米
          #    ⟹ 车间被一条宽通道分成左右两簇。**见 `layout.GridSpec.cluster_gap`。**
          #    ⚠️ **默认 `(0.0, -1)` = 均匀 ⟹ v3 主线与既有读数逐位不变。**
          cluster_gap: float = 0.0, cluster_split: int = -1,
          # ⚠️ **网格行数覆盖**（2026-10-17 加，用户提出）：默认 `n_rows = ceil(18/n_cols)`
          #    **⟹ 5 列必然给 4 行**（20 格）。要 **5×5 = 25 格随机选 18** 得显式给行数。
          #    ⚠️ **默认 `None` ⟹ v3 主线与既有读数逐位不变。**
          layout_rows: int | None = None,
          # ⚠️ **加工时间 × 位置的相关性**（2026-10-17 加，用户提出的方向）——
          #    见 `_apply_pt_pos_corr`。`corr < 0` ⟹ **远处的机台更快**
          #    ⟹ `MINU` 会持续往远处送活 ⟹ **均衡必须付运输**。
          #    ⚠️ **默认 `0.0` ⟹ 逐位不变。**
          pt_pos_corr: float = 0.0,
          # ⚠️ **有向走廊**（2026-10-17 加，用户从原文 Fig. 4 的箭头上发现）——
          #    横向双向、纵向**单向且逐列交替**（偶数列↑ / 奇数列↓）。
          #    见 `corridors.build_corridor_graph(directed=...)`。
          #    ⚠️ **默认 `False` ⟹ 无向图，逐位不变。**
          directed_corridors: bool = False,
          mid_edge_dock: bool = False,
          cap_fixed: int | None = None,
          reward_norm: str = "minmax", inst_obj: Instance | None = None,
          shi2025: bool = False):
    """→ `(spec, net)`。**两臂走同一个 `build`** ⟹ 实例/布局/编码器/约束档全同。

    ⚠️ **`directed_corridors` / `mid_edge_dock`（2026-10-17 加，**默认关**）**：
    两者都照原文 **Fig. 4** 逐格读出 ——
    - `directed_corridors`：横向双向、**纵向单向且逐列交替**（偶数列↑ / 奇数列↓）；
    - `mid_edge_dock`：**机台只接横走廊上的一点**（不在路口）。
    ⚠️ **2026-10-17 实测后**回退为默认关**（用户裁定）：两者都只是**均匀**地抬高运输水平，
    **不改变规则之间的差距** ⟹ `TC付` 1.18 → 1.21（+2.5%），`MS付` 有效但不够。
    见 `docs/v3/open-question-nhdrl.md`。

    `tau=None`（默认）⟹ **查逐实例标定表**（`due_dates.TF_RDD`，2026-10-09 起 10/10 可用）。
    ⚠️ Q1/D5 早先的轮次用的是**显式 `tau=1.0`**（不查表）—— 两者奖励权重不同，**不可混比**。

    `disturb=True`：⑨ 按 `TARGET_FAILURES` 激活（只用于**最终评测**，不用于训练）。
    `device`：`"cpu"`（默认）/ `"cuda"`。⚠️ **两臂必须同设备**。

    ══ 两条"让约束真的触发"的开关（见 progress-log §110–§112）══
    - `pm_interval`：⑫ 保养间隔（主轴分钟）。默认 120；**MK01 一局只有 ~76 min ⟹ 永不触发**，
      **dev 档用 30**。
    - `target_failures`：⑨ 按**每局期望故障次数**反推 `agv_mtbf`（`scaled_agv_mtbf`）。
      **对位论文 Shi 2025 就是按次数注入（每实例 1–8 次），不设 MTBF** ⟹ 比固定 480 min 有依据。
    ⚠️ **2026-10-10：⑪ 充电机制删除 ⟹ `soc0` 开关一并删**（车队无电量）。

    ══ ⭐ `layout_seed`：把「布局」与「训练」两件事拆开（2026-10-10 夜）══

    `None`（默认）= 用 `seed` ⟹ **与改动前逐位相同**。
    给定 ⟹ **布局由 `layout_seed` 决定，训练随机性仍由 `seed` 决定**
    （`torch.manual_seed(seed)` + 采链的种子都跟着 `seed`）。

    **为什么需要它**：`sample_layout(seed=seed)` 意味着**每个"训练种子"换了一整套车间**
    （输入缓冲容量、机台故障率、几何全变，实测三套）。后果是
    **报出的标准误里混的是「布局之间」的方差，不是「训练之间」的方差** ——
    消融实验的机制效应（几个百分点）被布局方差（几十个百分点）盖住，
    **三个种子分辨不出任何东西**（§130.3 实测）。

    **用法**：消融实验**固定 `layout_seed`、只变 `seed`** ⟹ 方差小一个量级；
    「跨布局泛化」另做一组（固定 `seed`、变 `layout_seed`）—— **那本来也该是单独一节**。
    """
    torch.manual_seed(seed)
    # `inst_obj` 给定 ⟹ **直接用它**，`instance` 只当名字用（如 `shi_instances` 生成的实例）。
    # `None`（默认）⟹ 照旧 `load_mk(instance)`，**既有读数逐位不变**。
    inst = load_mk(instance) if inst_obj is None else inst_obj
    cfg = SimConfig(tau=tau, n_agv=n_agv)
    # ⚠️ `SimConfig` / `AgvSpec` 都是**普通 dataclass** ⟹ 用 `dataclasses.replace`，
    # 不用 `_replace`（那是 NamedTuple 的方法，写错是 AttributeError 而不是静默）。
    mtbf = scaled_agv_mtbf(inst, TARGET_FAILURES) if disturb else None
    if target_failures is not None:
        mtbf = scaled_agv_mtbf(inst, target_failures)
    over = {}
    if mtbf is not None:
        over["agv_mtbf"] = mtbf
    if pm_interval is not None:
        over["pm_interval"] = float(pm_interval)
    # ⚠️ **`sim_overrides` 最后应用 ⟹ 它优先**（2026-10-10 修）。
    #
    # 原先它**在最前面**，于是被后面的 `pm_interval=` / `agv_mtbf` **覆盖掉** ⟹
    # 调用方以为自己在覆盖，其实没有。**实测踩过**：`h_sensitivity` 扫 `pm_interval`
    # 的 15/60/120 三档，读数**全部是 0.0%** —— 因为每一档都被基准 30 盖回去了，
    # 扫的根本不是那个参数。**零报错、零异常，只是一张全 0 的表。**
    #
    # ⚠️ **语义**：具名参数（`pm_interval=` / `target_failures=` …）是**常用入口**，
    # `sim_overrides` 是**通用逃生口** —— 逃生口应当能盖住常用入口，反之则没有意义。
    if shi2025:
        # ⚠️ **修复时间照原文**（2026-10-14 加）：*"每次故障的**修理时间在 5–10 之间随机**"*
        #    （`docs/citation-cards.md:1509`）。
        #    `SimConfig.agv_mttr_lo/hi` 的默认是 **10.0 / 10.0**（= v3 原来那个固定 `agv_mttr`）
        #    ⟹ **复现层必须改这两条，否则修复时间是死的 10**。
        # ⚠️ **放在 `sim_overrides` 之前** —— 逃生口必须仍能盖住它（见上）。
        over["agv_mttr_lo"], over["agv_mttr_hi"] = 5.0, 10.0
        # ⚠️ **装卸时间：不另设参数**（2026-10-16 归 0，用户裁定）
        #    假设 (3) 说 tt **含装卸**；而 `layout_scale` 是按 **TC 的总量**标的，
        #    那个总量**本来就含装卸** ⟹ 再设 `agv_load_min` 是**重复计入**。
        #    理由与实测见文件头那段注释，以及 `open-question-nhdrl.md` §4.32。
        #    `SimConfig.agv_load_min/unload_min` 仍是 0.0（默认）⟹ 这里**什么都不设**。
        #    `sim_overrides` 仍能设它（逃生口纪律）——`TT_OVR="agv_load_min=1.5,…"`。
        # ⚠️ **LA 平局随机源**（2026-10-15 加）：口径**照 PA 层** —— `des.py` 对 PA 的
        #    `tie_rng` 逐字写着 *"照 AEI2025 的**随机平局**"* ⟹ **原文的平局是随机的**。
        #    LA 层此前**漏了这条**，平局按**车号升序**打破 —— 而车号是**任意标签**
        #    （车队完全同构、全部在 L/U 起步）⟹ 那是任意偏置。
        #    **实测代价：`MINCOST` 的 MS 差 16%（585.6 → 489.2）、t=0 的 16 次投放全给 V0。**
        #    见 `open-question-nhdrl.md` §4.28。
        over["la_tie_random"] = True
    if sim_overrides:
        over.update(sim_overrides)
    if shi2025 and not agv_n_failures:
        # ⚠️ **原文口径**（PDF 第 5–6 页）：*"每个实例的**故障总次数在 1–8 之间随机**"*
        # ⟹ **次数是【实例】的属性，不是【每一次运行】的** —— 所以从**实例自己的子流**抽，
        #   抽完就固定（同一实例的 20 次重复运行看到同一组故障，这正是原文 Table 8 里
        #   CDR 的 SP = 0.0000 / 空白的原因）。
        # ⚠️ **为什么不用 `target_failures`（泊松）**：泊松的期望 ≠ 固定次数，
        #   而且 6 台车各跑一条泊松流 ⟹ 实测全队 **22 次**（原文是 1–8）。
        # ⚠️ 也用**独立子流**（`SHI_FAIL_SEED`），与布局/故障/平局各流分开 —— 同 `PA_TIE_SEED` 那条纪律。
        agv_n_failures = int(np.random.default_rng(
            [seed if layout_seed is None else int(layout_seed), SHI_FAIL_SEED]
        ).integers(1, N_FAIL_HI + 1))
    if agv_n_failures:
        # ⚠️ **按次数注入**（口径照 Shi et al. 2025）—— 比 `target_failures`（只定期望）更接近对位论文。
        # ⚠️ **抽签上界必须在这里算**：`_build_entities` 自己算会递归（参考解也走那条路）。
        # 而且用**不含注入**的 cfg 算 —— 否则"注入改动力学 ⟹ 改 T ⟹ 改注入"自指。
        over["agv_fail_horizon"] = float(
            reference_makespan(inst, dataclasses.replace(cfg, **over) if over else cfg))
        over["agv_n_failures"] = int(agv_n_failures)
    if over:
        cfg = dataclasses.replace(cfg, **over)
    # ⚠️ **尺寸口径（2026-10-18：`layout_scale` / `shop_len` 都已删除）**：
    #    只接受 `cell_w/cell_h/aisle_w/lu_connector_m`。
    #    ⚠️ **L/U 占一个 pitch**：车间总宽 = `n_cols*pitch_x + lu_connector_m`。
    _cw = 3.0 if cell_w is None else float(cell_w)
    _ch = 2.4 if cell_h is None else float(cell_h)
    _aw = 1.5 if aisle_w is None else float(aisle_w)
    _lc = 3.0 if lu_connector_m is None else float(lu_connector_m)
    if min(_cw, _ch, _lc) <= 0.0 or _aw < 0.0:
        raise ValueError(
            "cell_w/cell_h/lu_connector_m 必须 > 0，aisle_w 必须 ≥ 0；"
            f"收到 {(cell_w, cell_h, aisle_w, lu_connector_m)}")
    lay = layout_mod.sample_layout(inst.n_machines,
                                   seed=seed if layout_seed is None else int(layout_seed),
                                   cell_w=_cw, cell_h=_ch, aisle_w=_aw,
                                   lu_connector_m=_lc, n_cols=layout_cols,
                                   cap_fixed=cap_fixed, scheme=layout_scheme,
                                   n_agv=cfg.n_agv, n_rows=layout_rows,
                                   mid_edge_dock=mid_edge_dock,
                                   cluster_gap=cluster_gap, cluster_split=cluster_split)
    g = build_corridor_graph(lay, directed=bool(directed_corridors))
    dm = dock_distance_matrix(g)
    # ⚠️ **加工时间 × 位置的相关性**（`pt_pos_corr`，2026-10-17，用户提出的方向）——
    #    必须在布局定下来**之后**做（要按机台到 L/U 的距离排位）。
    #    **默认 `0.0` ⟹ 原样返回 ⟹ 逐位不变。**
    inst = _apply_pt_pos_corr(inst, lay, pt_pos_corr)
    # ② 有限缓冲：**约束对象要同时给仿真与特征层**（两份不同源 = 假信号）。
    # ⚠️ **2026-10-10：② 的默认值翻成 `True`**（用户裁定「在报告档打开」）⟹
    # 这里**恒传一个真对象**，不再用 `None`。`None` 的语义是"全开"，
    # 而"全开"现在**恰好等于** `ConstraintConfig()` —— 那就直接给它。
    #
    # ⚠️ `constraints=`（2026-10-10 加）：给**整个**约束对象，供**约束组消融**用
    # （`ABLATION_GROUPS` 的五个组）。给了它就忽略 `finite_buffer`。
    if constraints is not None:
        cons = constraints                       # 显式给了就听它的（约束组消融用）
    elif shi2025:
        # ⚠️ **复现 Shi 2025 的口径**（2026-10-14）：
        #   - **机台故障必须关** —— 该文 PDF 第 5 页假设 (4)：
        #     *"Machine breakdowns, path conflicts, and AGV charging are ignored."*
        #   - **⑫ 预防性维护关** —— 它是**我们 v3 自己的扩展**，不是原文的。
        #   - **② 有限缓冲关** —— 原文的假设 4 条 + 约束 (1)–(20) 里**没有任何缓冲上限**
        #     （只有约束 (5)「每台机一次加工一个」）。**它是我们 v3 的机制**（v1 遗留，
        #     2026-10-10 用户裁定「在报告档打开」）⟹ **复现时关**。
        #     ⚠️ **实测它对本实例无效**（容量 1/2/3/随机 读数差 ≤ 0.5，见 §37.50）——
        #     **但语义上仍须关**，因为原文没有这个机制。
        #   `shi2025=False` 可退回主线的「约束全开」档。
        #   - **⑧ 交期/拖期关** —— 原文的两个目标只有 `MS`（Eq.1）和 `TC`（Eq.2），
        #     **没有拖期**；假设列表与约束 (1)–(20) 里也没有交期。**它是我们 v3 的 ⑧。**
        # ⚠️ **⟹ 五条里只剩 `agv_failure` 开** —— 那正是原文唯一的扰动。
        cons = ConstraintConfig(finite_buffer=False, machine_failure=False,
                                maintenance=False, due_dates=False)
    else:
        cons = ConstraintConfig(finite_buffer=bool(finite_buffer))
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg,
                       constraints=cons, dm=dm)
    w, bnd = _reward_spec(inst, lay, g, dm, cfg, cons, reward_norm)
    # ⚠️ **2026-10-10：⑪ 去后构造网络不再要 `n_chargers`**（充电头是唯一随布局变宽的头）。
    net = HierarchicalPolicy(with_value=with_value, seq_ma=seq_ma)
    # ⚠️ **设备跟随策略**（`rollout.net_device`）：搬一次，之后 rollout/重算自己按参数所在设备走。
    return Spec(inst, lay, g, dm, cfg, ctx, w, cons, bnd), net.to(device)


def constraints_from_ckpt(ck: dict) -> ConstraintConfig:
    """从 ckpt 里恢复**约束档**（`b_ablation` / `c_mechanism` 存了它；`q1_flat` 没存）。

    ⚠️ **必须与训练时同源**：特征层按 `ctx.constraints` 决定哪些量静默，
    仿真按同一份走动力学。给错档 ⟹ 策略看到的世界与它学过的不是同一个。

    ⚠️ **`q1_flat` 存的 ckpt 没有这个字段** ⟹ 回落到默认档。
    那对 `q1_flat` / `d5_nested` 是**对的**（它们本来就用默认档训），
    对别的来源则要另想（见 `resolve_train_env` 的同一条纪律）。
    """
    saved = ck.get("constraints")
    if not saved:
        return ConstraintConfig()
    return ConstraintConfig(**saved)


def resolve_train_env(ck: dict, *, pm_interval: float | None = None,
                      agv_n_failures: int = 0) -> dict:
    """把**训练时用的环境参数**还原出来，供"载入 ckpt 再评测"的跑数器用。

    优先级：**命令行显式给的 > ckpt 里存的 > 默认档**。

    ⚠️⚠️ **不还原它，评测就在另一个环境里做** —— 实测（`Full_s0.pt`，同一份策略）：

    | 评测环境 | clean `makespan` |
    |---|---|
    | 默认档（`pm_interval=120`、`agv_n_failures=0`） | **63.98** |
    | 训练档（`pm_interval=30`、`agv_n_failures=4`） | **87.63** |

    **差 37%。** 而后者才是 `d5_nested` 里那个逐位对得上的数。
    ⟹ **`d1_goal` / `g_pareto` / `h_sensitivity` 若不还原，读到的全是错环境的数。**

    ⚠️ **2026-10-10 之前的 ckpt 没有 `env` 字段**（那三个训练跑数器当时还没存它）
    —— 故**必须**用命令行显式补；`missing_env_warning` 就是给这种情况的。

    ⚠️ **`h_sensitivity` 是例外**：它的**整个目的**就是"换参数重评" ⟹
    它要的是**训练档做基准、再逐参数偏离**，故基准必须还原、扫描点则显式覆盖。
    """
    env: dict = {}
    saved = ck.get("env") or {}
    if saved.get("pm_interval") is not None:
        env["pm_interval"] = float(saved["pm_interval"])
    if saved.get("agv_n_failures"):
        env["agv_n_failures"] = int(saved["agv_n_failures"])
    if pm_interval is not None:                      # 命令行优先
        env["pm_interval"] = float(pm_interval)
    if agv_n_failures:
        env["agv_n_failures"] = int(agv_n_failures)
    return env


def missing_env_warning(ck: dict, env: dict) -> str | None:
    """ckpt 与命令行都没说环境时，返回一句**必须打出来的警告**（否则返回 `None`）。

    **不静默**：静默的后果是"读到一张 37% 偏好的表而毫无察觉"。
    """
    if env or (ck.get("env") or {}):
        return None
    return ("⚠️⚠️ **ckpt 里没有 `env` 字段、命令行也没给 `--pm-interval` / "
            "`--agv-n-failures`** ⟹ 本次评测跑在**默认档**上。\n"
            "     若该策略是在别的参数下训的，**这里读到的数不是它的业绩**"
            "（实测同一份策略：默认档 63.98 vs 训练档 87.63）。\n"
            "      请补 `--pm-interval 30 --agv-n-failures 4`（或该次训练实际用的值）。")


def evaluate(net: HierarchicalPolicy, spec: Spec, *, n: int, seed: int,
             dt_events: int, spread_chains: bool = True,
             force_top1: bool = False,
             fixed_rules: dict[str, int] | None = None,
             pa_limit: int | None = None,
             exact_rule: bool = False,
             tie_streams: int = 1) -> dict:
    """**greedy** 评测 `n` 个场景 × `tie_streams` 条平局流 → 三目标均值与标准误。

    ⚠️ **2026-10-10 修：`n` 条链现在跑 `n` 个**场景**（`spread_chains=True`）。**
    此前 `flat_rollout(n, seed=...)` 把 `seed*1000+i` 只喂给 `NetPolicy.gen`，
    而 **greedy 档下 `gen` 一次都不用**（`_categorical` 走 argmax、`_sample_param` 走 `u = μ`）
    ⟹ **`n` 条链逐位相同**：`makespan_se` 恒为 0（假的），且评测**白花 `n` 倍时间**。

    修法：第 `i` 条链的场景用 `seed + i`（**不同的 `seed_chain`** = 不同的机障/故障实现）
    ⟹ **`se` 现在含场景方差**。

    ⚠️ **`spread_chains=False` 退回旧口径**（全部 `seed_chain=0`）——
    **v2 的历史读数就是这么跑的**（均值可比、`se` 不可比）。

    ⚠️ **`tie_streams`（2026-10-20）**：同一场景跑 `T` 条**平局流**。
    `T=1`（默认）⟹ `tie_seed=None` ⟹ 逐位等于改动前。
    `T>1` ⟹ 第 `j` 条用 `TIE_STREAM_BASE + j`；`seed_chain` 不变 ⟹ **只变平局**。
    实测（§38.23）：同规则、同场景，平局流之间的逐实例 std **17.8 min**
    ⟹ 测 8 min 的效应需要 **≈20** 条，不是 2 条。
    """
    if tie_streams < 1:
        raise ValueError(f"tie_streams 须 >= 1，收到 {tie_streams}")
    prev = net.training
    net.eval()                                   # 关 dropout/BN（若有）——评测要确定性
    try:
        runs = []
        for i in range(n):
            for j in range(tie_streams):
                ts = None if tie_streams == 1 else TIE_STREAM_BASE + j
                runs += flat_rollout(net, spec, n=1, seed=seed * 1000 + i,
                                     dt_events=dt_events, greedy=True,
                                     seed_chain=(seed + i) if spread_chains else 0,
                                     force_top1=force_top1, fixed_rules=fixed_rules,
                                     pa_limit=pa_limit, exact_rule=exact_rule,
                                     tie_seed=ts)
    finally:
        net.train(prev)
    f = np.array([x.f for x in runs], dtype=np.float64)
    m = f.mean(axis=0)
    se = f.std(axis=0, ddof=1) / np.sqrt(len(f)) if len(f) > 1 else np.zeros(3)

    def _pair(c):
        return {"mean": float(m[c]), "se": float(se[c])}
    return {"n": n, "tie_streams": tie_streams, "runs": len(f),
            "makespan": float(m[0]), "makespan_se": float(se[0]),
            "energy": float(m[1]), "energy_se": float(se[1]),
            "twt": float(m[2]), "twt_se": float(se[2]),
            "pairs": {"makespan": _pair(0), "energy": _pair(1), "twt": _pair(2)},
            "f": f,
            "R": float(np.mean([x.R for x in runs]))}


TIE_STREAM_BASE = 10_000
"""`tie_streams` 的**平局子种子基**。

⚠️ 与 `seed_chain` 分开编号：`seed_chain` 的量级是 1–1000，
`PA_TIE_SEED=2000` / `LA_TIE_SEED=2100` 是流标识，故取 10000 起。
与既有任何子流都不撞（`[10000+j, 2000]`）。"""


CONSTRAINT_KEYS = ("fail_events", "tardy_twt", "agv_fail_events",
                   "buffer_block_min", "pm_events")
"""**五条约束**的触发计数（③机障 · ⑧拖期 · ⑨AGV故障 · ②有限缓冲 · ⑫保养）。

⚠️ **2026-10-10：⑪ 充电删除 ⟹ `charge_events` 移出本表**，`buffer_block_min` 移入 ——
本表的条目**与 `ConstraintConfig` 的五个开关一一对应**，不是历史编号的搬运。
"""


def constraint_counts(spec: Spec) -> dict:
    """**规则档**（不给任何策略）跑一局，回报五条约束各自的触发计数。

    ⚠️ **必须用规则档**（§115.3）：策略档下 `pm_events` 数的是
    **"策略多爱用这个动作"**（随机初始化的网会乱保养，实测某布局 67 次），
    与**"约束是否 binding"是两回事**。规则档才是"约束按设计触发"的口径。
    （⑪ 充电曾是本条最刺眼的例子；它已整条删除。）

    ⚠️ 它**另外跑一局仿真**（不带策略 ⟹ 很便宜，§99 实测 ~0.02–2 s）。
    为什么值得：**"跑通了但约束一次都没触发"是看不出来的** —— 那正是这几轮反复踩的坑
    （§110：MK01 上 ⑨⑪⑫ 全不触发，而所有测试都是绿的）。
    """
    m = spec.world().run_gated(seed_chain=0)
    return {k: float(m.get(k, 0)) for k in CONSTRAINT_KEYS}


def run_one(arm: str, seed: int, *, steps: int, n: int, dt_events: int,
            eval_every: int, eval_n: int, rec, device: str = "cpu",
            workers: int = 0, save_dir: str = "", env: dict | None = None) -> None:
    """一条 `(臂, 种子)` 的完整训练 + 评测。`rec` = 写一行的回调。

    `workers > 1` ⟹ 多进程并行采链（`hrl/pool.py`，**只支持 CPU**）。
    ⚠️ **并行档与单进程档不是逐位相同**（每进程 1 线程 ⟹ 浮点归约次序不同）⟹
    **同一次对比里两臂必须同档**，且报出的数字要注明档位。
    """
    env = dict(env or {})
    spec, net = build(seed, with_value=(arm in ("A", "A2")), device=device, **env)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    rec({"kind": "start", "arm": arm, "seed": seed, "steps": steps, "n": n,
         "dt_events": dt_events, "device": device, "workers": workers,
         "n_params": sum(p.numel() for p in net.parameters())})
    base = evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)
    rec({"kind": "eval", "arm": arm, "seed": seed, "step": 0, **base})

    pool = None
    if workers > 1:
        if device != "cpu":
            raise ValueError("并行档只跑 CPU（一个 GPU 上开多上下文只会互抢）——"
                             "`--workers` 与 `--device cuda` 二选一")
        pool = ChainPool(net, spec, n_workers=workers, dt_events=dt_events)
    t0 = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            out = flat_step(net, spec, optim, arm=arm, n=n, seed=seed * 10000 + step,
                            dt_events=dt_events, pool=pool)
            gn = out["grad_norm"]
            rs = [r["R"] for r in out["runs"]]
            # ⚠️ `flat_step` 返回的 loss 是**整组 8 条链的和**（真实被优化的量）。这里**除以 `n`**
            # 再记，否则两个数会读错：
            #   · 臂 A 的 `loss ≈ −9.4e4` 看着吓人，其实是 8 条链之和（每条 ≈ −1.2e4）；
            #   · 臂 B 的 `loss ≈ −12.6` **不是"没学"** —— 组内 z 化使 `Σ_i z_i ≡ 0`，
            #     8 条链的贡献**互相抵消**。**这才是 GRPO 的正常形状**，别当成 bug。
            rec({"kind": "train", "arm": arm, "seed": seed, "step": step,
                 "r_mean": float(np.mean(rs)),
                 "r_std": float(np.std(rs, ddof=1)) if n > 1 else 0.0,
                 "loss_per_chain": out["loss"] / n,
                 "policy_loss_per_chain": out["policy_loss"] / n,
                 "critic_loss_per_chain": out["critic_loss"] / n,
                 # ⚠️ `critic_loss_per_chain` ≈ `E[(V−R)²]`：初值 ≈ R² ≈ 1300 ⟹ 说明 **`V` 从 0 附近起步**
                 # （critic 头是默认小初值）。随训练下降才说明 critic 在学。
                 "heads_grad": gn["heads"], "critic_grad": gn["critic"],
                 "enc_grad": gn["encoder"],
                 "makespan": float(np.mean([r["makespan"] for r in out["runs"]])),
                 "elapsed_s": time.perf_counter() - t0})
            if eval_every and step % eval_every == 0:
                ev = evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)
                rec({"kind": "eval", "arm": arm, "seed": seed, "step": step, **ev})
    finally:
        if pool is not None:
            pool.close()

    # 最终：**扰动**（⑨ 激活）下的业绩
    dspec, _ = build(seed, disturb=True, with_value=(arm in ("A", "A2")), **env)
    dist = evaluate(net, dspec, n=eval_n, seed=1, dt_events=dt_events)
    rec({"kind": "final", "arm": arm, "seed": seed, "total_s": time.perf_counter() - t0,
         "clean": evaluate(net, spec, n=eval_n, seed=1, dt_events=dt_events),
         "disturbed": dist, "agv_mtbf": dspec.cfg.agv_mtbf,
         # ⚠️ **规则档**下的五条约束触发数（§115.3）—— 让"跑通了但约束没触发"当场可见
         "constraints": constraint_counts(spec)})

    if save_dir:
        # ⚠️ **必须存盘**：不然训好的策略丢了 ⟹ 后面要"读出 `B` 矩阵"（v2 实验计划 D6）、
        # 要"同策略在扰动环境里重评"（H 敏感性）全都做不了。§103.5 就是这么发现缺口的。
        # 连同**重建网络要的全部元信息**一起存（`with_value` / 配置），
        # 否则拿到 `.pt` 也重建不出同构的网。
        p = Path(save_dir) / f"{arm}_s{seed}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "arm": arm, "seed": seed, "steps": steps, "n": n,
                    "dt_events": dt_events, "device_trained_on": device,
                    "with_value": arm in ("A", "A2"),
                    # ⚠️ **必须存环境**（2026-10-10 加）：不存的话"载入 ckpt 再评测"的
                    # 跑数器只能猜环境，而猜错会让读数偏 37%（见 `resolve_train_env`）。
                    "env": dict(env),
                    "tau": spec.cfg.tau}, p)
        rec({"kind": "saved", "arm": arm, "seed": seed, "path": str(p)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Q1：扁平链两臂（设计 §12.2–§12.4）")
    ap.add_argument("--arms", nargs="+", default=["A", "B"], choices=["A", "A2", "B"],
                    help="A=critic(原始优势) · A2=critic+优势归一化 · B=组内 z 化")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8, help="组大小（设计 §12.2 的 G）")
    ap.add_argument("--dt-events", type=int, default=20, help="顶层定时器步长（§10②）")
    ap.add_argument("--eval-every", type=int, default=25, help="0 = 不中途评测")
    ap.add_argument("--eval-n", type=int, default=3, help="评测链数（greedy）")
    ap.add_argument("--device", default="cpu", help="cpu / cuda（⚠️ 必须用 GPU 版解释器才认 cuda）")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch 线程数（默认 1）。⚠️ 实测 §99：24 线程比 1 线程慢 2.7×")
    ap.add_argument("--workers", type=int, default=0,
                    help="多进程并行采链的进程数（0 = 顺序档）。仅 CPU；每进程 1 线程")
    ap.add_argument("--save-dir", default="",
                    help="存训练好的策略（.pt）；空 = 不存。⚠️ 不存就无法做 D6/H")
    # ── ⚠️ 让约束真的触发（§110–§112）──
    ap.add_argument("--layout-seed", type=int, default=None,
                    help="⭐ 布局种子。None = 用训练 seed（旧行为）。"
                         "**消融实验应当固定它** —— 否则标准误里混的是布局方差（§130.3）")
    ap.add_argument("--reward-norm", default="minmax", choices=("minmax",),
                    help="目标归一化口径。ref（默认）= 现行 1/f^ref，逐位不变；"
                         "minmax = 每目标按 36 条规则的可达范围归一 + 等权"
                         "（照 Shi et al., AEI 65:103195 (2025) 式 51/53）。"
                         "⚠️ 两者结论会**翻转**，见 progress-log.md §147")
    ap.add_argument("--instance", default="mk01", help="mk01–mk10")
    ap.add_argument("--tau", type=float, default=None,
                    help="交期 τ；None = 查逐实例标定表（10/10 可用）")
    ap.add_argument("--pm-interval", type=float, default=None,
                    help="⑫ 保养间隔（主轴分钟）。默认 120；dev 档用 30（MK01 一局只 76 min）")
    ap.add_argument("--no-finite-buffer", dest="finite_buffer", action="store_false",
                    help="② **关掉**有限缓冲（**默认是开的**，见 progress-log §118）。"
                         "关 ⟹ 缓冲无界，退回 2026-10-10 之前的动力学（消融用）")
    ap.add_argument("--agv-n-failures", type=int, default=0,
                    help="⑨ 按次数注入（0 = 关，用 agv_mtbf 泊松流）。照 Shi 2025 口径")
    ap.add_argument("--target-failures", type=float, default=None,
                    help="⑨ 按「每局期望故障次数」反推 agv_mtbf（对位论文 Shi 2025 同口径）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    if a.threads:
        # ⚠️ **只在跑数器里设**，不进库：改线程数会改浮点归约的结合次序 ⟹ 末位可能不同，
        # 而门禁与既有读数依赖 CPU 默认档**逐位不变**。实测（§99）：24 线程 5.72 s，
        # 8 线程 2.70 s —— 小张量被线程调度拖垮，不是算力不够。
        torch.set_num_threads(a.threads)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        # ⚠️ **快速失败**：CPU-only 的 torch 上写 `--device cuda` 静默退回 CPU
        # ⟹ 你以为在用 GPU，实际跑了 10 小时 CPU。（本仓禁用静默降级。）
        print(f"✗ 这个解释器没有 CUDA（torch {torch.__version__}）。"
              "GPU 请用 D:/anaconda/envs/py312/python.exe", file=sys.stderr)
        return 2

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ **断点续跑**：跳过已经写出 `final` 的 `(臂, 种子)`。
    # 为什么需要：一次全量是小时级，而这台笔记本目前**电源功率不足（电池 1%）**——
    # 跑到一半掉电会整个丢失，跑数器本身不写中间检查点。
    done: set[tuple[str, int]] = set()
    if out.exists():
        for ln in out.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "final":
                done.add((str(d["arm"]), int(d["seed"])))
        if done:
            print(f"续跑：已完成 {sorted(done)}，跳过", file=sys.stderr)
    with out.open("a", encoding="utf-8") as fh:
        def rec(d: dict) -> None:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps(d, ensure_ascii=False), flush=True)

        for seed in a.seeds:
            for arm in a.arms:
                if (arm, seed) in done:
                    continue
                run_one(arm, seed, steps=a.steps, n=a.n, dt_events=a.dt_events,
                        eval_every=a.eval_every, eval_n=a.eval_n, rec=rec,
                        device=a.device, workers=a.workers, save_dir=a.save_dir,
                        env={"instance": a.instance, "layout_seed": a.layout_seed, "tau": a.tau,
                             "reward_norm": a.reward_norm,
                             "pm_interval": a.pm_interval,
                             "target_failures": a.target_failures,
                             "agv_n_failures": a.agv_n_failures,
                             "finite_buffer": a.finite_buffer})
    print(f"\n写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
