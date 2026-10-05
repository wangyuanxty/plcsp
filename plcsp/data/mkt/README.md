# MKT 布局数据（`plcsp/data/mkt/`）

**MKT = 原始 Brandimarte MK 的加工数据（一字不改）+ 一张机台间行程时间矩阵 + 一个车辆数设定。**

本目录只放**运输侧**的数据。加工数据仍是 `plcsp/data/brandimarte/Mk01–Mk10.fjs`，本适配器**不碰**。

---

## 1. 数据来源

| 来源 | 位置 | 用途 |
|---|---|---|
| **一手（引用它最稳）** | 作者项目站 `https://fastmanufacturingproject.wordpress.com/2019/04/11/fjspt-instances/`（页面附 `4to18machines_layouts-1.pdf`） | 论文里的正式出处 |
| **本仓库实际取自** | `https://github.com/msh0576/FJSPT-Scheduler` → `BenchmarkDataset/dataset/{m}_machine_layout.txt` | 纯文本、免转换 |

⚠️ **许可**：`github.com/msh0576/FJSPT-Scheduler` **无 LICENSE 文件**。故本目录
**只取数据文件、不复制该仓任何代码**。若论文要求更硬的出处，引作者项目站（一手来源）。

⚠️ 该仓另有 `BenchmarkDataset/small_and_medium_instances/`（含 2、3、7 机等）与
`BenchmarkDataset/large_instances/`——**与 Brandimarte 那 12 个规模不是同一套**，
本适配器**只认 `dataset/` 下的 12 个**（`MKT_LAYOUT_DIR` 目录即为此 12 个文件），取不到就报错。

**文件逐个与上游逐字节一致**（含上游 `6_machine_layout.txt` 末尾多出的一个空行、
以及上游用 TAB 分隔——`np.loadtxt` 对两者都容忍）。保持逐字节一致是为了日后能重新核对。

---

## 2. 口径

- 原始文件是 **`(m+1) × (m+1)`**：**第 0 行 / 第 0 列是装卸站（LU, Load/Unload）**。
- **机台间行程时间矩阵 = 丢 LU 后的 `m × m` 子矩阵**，即 `raw[1:, 1:]`。
  （与 HGS 仓 `utils/utils_fjspt.py` 的读取方式一致，核实于 `progress-log §19.7d`。）
- 单位：**分钟**（与仿真时间同单位，本项目约定时间 = 分钟、坐标 = 米、能耗 = kWh）。
- **矩阵非对称**：i→j 与 j→i 可以不等（见下 §3 实测）。
- 值可以是 **半整数**（2.5 / 4.5 / … / 10.5），不是纯整数。

### 实测值域（**本机实测，非抄录**）

```
D:/anaconda/python.exe -c "import numpy as np; t=np.loadtxt('plcsp/data/mkt/layouts/10_machine_layout.txt')[1:,1:]; print(t.max())"
```

| m | 原始 shape | 丢 LU 后 | 最小（非零） | 最大 | 对角线 | 非对称? |
|---|---|---|---|---|---|---|
| 4 | 5×5 | 4×4 | 6 | 10 | 全 0 | **否（恰好对称）** |
| 5 | 6×6 | 5×5 | 3 | 15 | 全 0 | 是 |
| 6 | 7×7 | 6×6 | 3 | **17** | 全 0 | 是 |
| 8 | 9×9 | 8×8 | 2 | 10 | 全 0 | 是 |
| 10 | 11×11 | 10×10 | 2.5 | 12 | 全 0 | 是 |
| 11 | 12×12 | 11×11 | 2 | 10 | 全 0 | 是 |
| 12 | 13×13 | 12×12 | 2 | 10 | 全 0 | 是 |
| 13 | 14×14 | 13×13 | 2 | 10 | 全 0 | 是 |
| 15 | 16×16 | 15×15 | 1 | **15** | 全 0 | 是 |
| 16 | 17×17 | 16×16 | 2 | 10 | 全 0 | 是 |
| 17 | 18×18 | 17×17 | 2 | 10 | **非 0：`t[5][5]=4`** | 是 |
| 18 | 19×19 | 18×18 | 2 | 10 | 全 0 | 是 |

**全局值域：1 – 17**（非零项），取值集合 =
`{1, 2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 17}`。

### ⚠️ 两处实测得到的数据异常（**不是我们的 bug，是上游数据如此**）

1. **m = 17 的对角线非零**：`t[5][5] = 4`——第 6 台机到它自己有 4 分钟行程。
   这是 12 个布局里**唯一**对角非零的一个。若日后代码要对角为 0，必须处理这个例外。
2. **m = 4 的矩阵恰好对称**：它是 12 个里**唯一对称**的一个。
   故"矩阵非对称"这条只能钉在具体规模上（我们钉 m=10），**不能对 12 个规模一刀切**。

两处都有测试钉住（`plcsp/tests/test_mkt_data.py::test_data_quirks_are_pinned_not_assumed`）。

### ⚠️ 只钉"非对称"是不够的（终审 F3）

`形状` + `对角为 0` + `not allclose(t, t.T)` 这三条**对两种错误实现同样成立**（本机验证）：
- **裁错方向** `raw[:m, :m]`（该丢后一行一列，却丢了前一行一列）：形状同、对角同为 0、同样非对称；
- **转置** `raw[1:, 1:].T`：形状同、对角不变、非对称照样成立。

故测试**钉了方向上的具体值**：10 机布局 `t[0][1] == 3` 且 `t[1][0] == 12`
（裁错方向会变成 2/11，转置会变成 12/3）。变异验证过：两种错误实现都会让测试变红。

### ⚠️ 关于 `§19.7d` 的「M[1][2]=3 而 M[2][1]=12」——**它是对的**

P4-A 的终审里我曾把这条记为"二手读数错、实测证伪"，**那是我索引错**：
`§19.7d` 用的是 **1 基机台号**，M[1][2] = 丢 LU 后 0 基的 `t[0][1]`。
本机实测 `t[0][1] = 3`、`t[1][0] = 12`——**与 `§19.7d` 逐位相符** ✅。

（我当时按 0 基读成 `t[1][2]=2.5` / `t[2][1]=4.5`，据此写了"更正"，是错的。）
这两个值现已作为**方向敏感断言**钉在测试里——它们同时能杀掉"裁错方向"与"转置"两种错误实现
（见下 §2 末）。

---

### 哪些实例用哪份布局（**布局按机台数命名，不按实例名**）

Brandimarte MK 的 10 个实例只用到 **6 种**机台数，其余 6 份布局是给该仓 `large_instances/` 那套用的：

| 机台数 | 布局文件 | 用到它的 MK 实例 |
|---|---|---|
| 4 | `4_machine_layout.txt` | mk05 |
| 5 | `5_machine_layout.txt` | mk07 |
| 6 | `6_machine_layout.txt` | mk01、mk02 |
| 8 | `8_machine_layout.txt` | mk03、mk04 |
| 10 | `10_machine_layout.txt` | mk08、mk09 |
| 15 | `15_machine_layout.txt` | **mk06、mk10** |
| 11/12/13/16/17/18 | 同名 | （MK 系列用不到） |

⚠️ **mk06 也是 15 机**（10 工件 × 15 机台），与 mk10 **共用同一份 15 机布局**——
所以"MKT06 与 MKT10 的行程时间是同一张表"，这不是 bug。

---

## 3. 三条存疑点（逐字引自 `progress-log §19.7d`，**必须写进论文，不得隐瞒**）

1. **"行程时间随机 2–10"与公开布局矩阵矛盾**——网站文字与 HGS 正文都写 2–10，但实发文件值域到 **17**（6 机）/ **15**（15 机）。HGS 的代码用的是**矩阵**。两者不可能同时为真。
2. **车辆数未公布且各家用得不同**：HF2021 据第三方转述为 **2 台**（故其 LAHC 的 MKT01=187 远高于 HGS 的 153）；**HGS 与 HA-DQN 用 v = m**（HA-DQN 表 11 的 J-M-A 列直接读出 10-6-**6**、20-5-**5**、20-15-**15**）。⟹ **"MKT 数字可直接比"只在车辆数对齐时成立。**
3. **HGS 仓库日志 ≠ 论文 Table IV**（Mk01：日志 FIFO 158 / 论文 146；MatNet 180 / 159；而 SPT、LPT 两边相同）⟹ 论文那轮与仓库那轮的 **v 或代码版本不同**。**故不得承诺"复现 HGS 的精确数字"**，论文应写成"**同数据、同车辆数设定**下的对比"。

第 1 条本机复测**确认**（见上 §2 实测值域表：全局 1–17 且含半整数）。

---

## 4. 引用

> Homayouni, S. M., Fontes, D. B. M. M. *Production and transport scheduling in flexible job shop manufacturing systems.* **Journal of Global Optimization 79(2):463–502, 2021.** DOI 10.1007/s10898-021-00992-6.

用到这些数字的已发表方法（**均为 MKT 口径，不是原始 MK**）：

> Moon, Lee, Park. *Learning-enabled Flexible Job-shop Scheduling for Scalable Smart Manufacturing.* **Journal of Manufacturing Systems 77:356–367, 2024.**（HGS）

> Dong, Wan, Zeng. *A heuristic-assisted deep reinforcement learning algorithm for flexible job shop scheduling with transport constraints.* **Complex & Intelligent Systems 11:210, 2025.**（HA-DQN）

---

## 5. 怎么用

```python
from plcsp.env.mkt import load_mkt_layout

t = load_mkt_layout(10)            # 10×10 机台间矩阵（已丢 LU）
raw = load_mkt_layout(10, drop_lu=False)   # 11×11 原矩阵（含装卸站）
```

取不到的机台数会**显式报错**，不会回退到邻近规模。

⚠️ **接入状态**：P4-A 只落地数据与适配器；**P4-B 已把矩阵接进仿真**——口径**跟随实例**
（MKT 走矩阵、原始 MK 走几何，见 `plcsp/env/transport.py`）。P4-A 当时"`AgvSim` 仍用布局上
最短路距离"的说法**已作废**。
