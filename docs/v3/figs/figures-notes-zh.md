# 两张图的中文说明（概念草稿）

> 脚本：`docs/v3/figs/draw_paper_figures.py`
> 输出：`fig1_motivated_example.pdf/.svg/.png` · `fig2_solution_overview.pdf/.svg/.png`
> 旧版（Gantt / 通用流程图）在 `docs/v3/figs/archive/`，只留作对照。
> ⚠️ 图 1 的分数、等待时间、MS 是示意值；正式稿要换成 MK01 的真实决策轨迹。
> ⚠️ 这是 matplotlib 布局草稿。定稿建议按 handbook 用 PowerPoint / Figma 重绘，保留布局与内容。

---

## 图 1 · Motivated Example（分层多 agent）

### 放哪里

Introduction 第二段之后（第 1 页右上或第 2 页顶部）。读者应在 30 秒内看懂：
**PA / LA / RA 各自局部最优，合起来会冲突；所以需要 MA 在顶层协调。**

### 范式

Running Example + Failure Case。三栏：

| 栏 | 内容 |
|---|---|
| (a) | 一个工件 J4 的决策，同时牵动 PA、LA、RA |
| (b) | 三个 agent 各自按局部规则决策 → 冲突 → J4 等待 |
| (c) | MA 看全局状态，选一组协调的规则 → 无等待 |

### 逐栏

**(a) One job, three coupled decisions**
- 车间：L/U、M1–M6、V1、V2；J4 在 L/U 等（交期 22）。
- 三张卡：
  - **PA · production**：`J4 → M2`，因为 M2 利用率最低。
  - **LA · logistics**：V1 最近，但 V1 已被 J7 占用。
  - **RA · maintenance**：M2 役龄 118/120，想现在保养。
- 中间的红色 `!` 表示三者**耦合**。
- 下方频次带：MA 每 ΔT / 扰动；PA 每次派工；LA 每次运输；RA 每个完工窗口。
  **频次不同、动作空间不同** —— 这是分层的理由。

**(b) Independent agents: local choices conflict**
- PA：`EAJ+SMPT → M2`。
- LA：`MINCOST → V2`（V1 被占，V2 要空载绕到 L/U）。
- RA：`AGE → maintain M2 now`。
- 结果：J4 到 M2 时 M2 正在保养；V2 还在路上。**J4 等待 26 min**。
- **MS = 268.4 min**。
- 一句话：`each locally optimal; together they conflict`。

**(c) Hierarchical MA: coordinate, then execute**
- MA 看全局状态，选一组配套规则：
  - PA：`EAJ+EAM`（避开 M2）；
  - LA：`MINCOST`（等 V1 回来，不派远端 V2）；
  - RA：`QUEUE_EMPTY`（等 M2 队列空再保养）。
- 执行：`J4 → M3`；保养推迟 6 min；V1 承运。
- **MS = 251.7 min**。
- 一句话：`coordinate first, execute at each layer's own timescale`。

### 中文图注

> **图 1：三个 agent 的局部决策如何冲突，以及顶层协调如何修复。**
> (a) 工件 J4 的派工同时牵动三个 agent：PA 想把 J4 派到利用率最低的 M2；LA 想用最近的 V1，但 V1 已被 J7 占用；RA 发现 M2 的役龄达到 118/120，想立刻保养。三者频次与动作空间不同。
> (b) 三个 agent 各自按局部规则决策：J4 派到 M2，V2 被派去空载绕行，M2 同时进入保养。J4 到机台后等待 26 min，makespan 268.4 min。
> (c) MA 在顶层看全局状态，选一组协调的规则：PA 避开 M2，LA 等 V1，RA 推迟保养。J4 改派 M3，保养只推迟 6 min，makespan 251.7 min。
> 图中数值为示意，正式稿将替换为 MK01 的真实决策轨迹。

### 为什么这张图能 motivate 分层多 agent

- **动作空间不同**：PA 选 `(工件, 机台)`，LA 选 AGV，RA 选 `NOW / DEFER`。一个扁平策略要同时覆盖三种动作，难。
- **频次不同**：MA 每 ΔT / 扰动；PA 每次派工；LA 每次运输；RA 每个完工窗口。一个固定频次的单层策略无法同时对齐。
- **局部最优 ≠ 全局**：三个 agent 各自按规则都“对”，合起来 J4 等待。**需要顶层看全局状态，选配套的规则组合。**
- **这正是 MA 的职责**：MA 不改 PA/LA/RA 的底层动作，只选规则；下层在规则内执行。分层把“协调”和“执行”分开。
- 图 1 不展开 top-K 打分的细节（那是下层实现）；图 2 展开。

### 正式稿要替换

- `M2 age 118/120`、`26 min`、`6 min`、`268.4 / 251.7` 都是示意值。
- 从真实运行里选一个“PA / LA / RA 局部选择冲突”的决策点：
  - PA 的目标机台 vs RA 的保养窗口；
  - LA 的最近车 vs 该车的占用；
  - 记录局部规则与 MA 协调规则下的最终 MS。
- 如果真实 trace 没有这么大的等待差，就改用“同一决策点的最终 MS 差”，不要夸大。
- 图 1 的三种规则组合要能对应到 `PA_RULE_NAMES` / `LA_RULE_NAMES` / `PM_RULE_NAMES`。

### 与正文对应

- (a) → §3.1 四个 agent 的分工；§3.2 决策点与频次。
- (b) → §2 固定规则 / 独立规则基线。
- (c) → §3.3 顶层 MA 的规则选择；§3.4 下层执行。
- 图里的名字与正文一致：`PA`、`LA`、`RA`、`MA`、`EAJ+SMPT`、`EAJ+EAM`、`MINCOST`、`QUEUE_EMPTY`。

---

## 图 2 · Solution Overview

### 放哪里

Methodology 开头（Section 3 第一段之后）。读者跳读时，应能靠这张图找回任意小节。

### 布局

上排：**Input → Policy → Actions**。
下排：**Two-phase training**（一个模型、跨实例）。
这对应 handbook 的 LEAD 例子：推理路径在上，训练路径在下，不混在一起。

### 逐块

**Input**
- 小车间图 + 距离矩阵。
- 四类 token：
  - M 机台：backlog · util · dist L/U
  - B 作业：progress · due · location
  - V AGV：status · position · queue
  - G 全局：time · MCT mean/min/max

**Policy**
1. **Shared Transformer encoder**：4 层，d = 128，四类 token 共用一个编码器。
2. **MA: rule selection**：PA 12 路、LA 3 路、RA 3 路；每 ΔT 或扰动触发。
   例：`EAJ+SMPT` / `MINCOST` / `QUEUE_EMPTY`。
3. **Rule-filtered candidates**：
   - PA：top-16 个 `(工件, 机台)`；
   - LA：top-6 台 AGV；
   - RA：规则过滤后的 `{NOW, DEFER}`；
   - 严格 K；边界平局无偏截断。
4. **Candidate scoring**：
   - PA head：`z_M · z_B · 9 feats`；
   - LA head：`z_V · ctx · 4 feats`；
   - RA head：`z_M · 5 feats`；
   - 规则 embedding；K≥3 时组内 z 化；log-softmax 在 K 内。

**Actions**
- dispatch `(job, machine)`
- assign AGV
- maintain / defer

**Training**
- 阶段 A：只更新 MA；每个顶层点采 G 组规则；组内优势。
- 阶段 B：只更新三个打分头；规则冻结；采 M 条执行轨迹；组内优势。
- 损失：`−Σ Â · log p`；无 PPO clip；无 critic。
- 一个共享编码器；训练用随机 MK-range 实例；测试 MK01–10 零样本；布局 seed 不重叠。

### 中文图注

> **图 2：top-K 规则引导的分层策略总览。**
> 输入是车间状态的四类 token 与布局距离，经共享 Transformer 编码器得到逐 token 向量 z。
> 上层 MA 每 ΔT 或扰动选一组规则（PA 12 / LA 3 / RA 3）。
> 规则只负责筛候选：PA 取 top-16 个 `(工件, 机台)`，LA 取 top-6 台 AGV，RA 取规则过滤后的 `{NOW, DEFER}`。
> 下层三个打分头在候选集内按状态特征排序，输出派工、派车、保养/延后动作。
> 训练分两阶段：A 阶段只更新 MA，B 阶段只更新打分头；优势在组内 z 化，损失为 `−Σ Â log p`。

### 设计理由

- 核心信息是“**规则筛、打分排**”。旧架构的“规则选超参数”在图中不出现。
- 上排是推理，下排是训练；读者不会把训练回传当成在线数据流。
- 图中所有名字都能在正文找到对应小节。
- 输入和输出在边界上明确：左边 `Shop state`，右边 `Actions`。

### 待确认

- RA 现在画成“二元、无 top-K”。如果正文改成 RA 也 top-K，中间那行要改。
- 如果正文强调跨布局，可把“layout seeds disjoint”单独提出来；现在放在训练卡里。
- 图里的示例规则名只是举例，不是唯一配置。

---

## 对照 handbook 的自查

| 检查项 | 图 1 | 图 2 |
|---|---|---|
| 矢量导出（PDF/SVG） | ✅ | ✅ |
| 小画布 | ✅ 7.2 × 3.35 in | ✅ 7.2 × 3.9 in |
| 色盲安全 | ✅ Okabe–Ito；红/绿同时有文字与形状 | ✅ 颜色 + 文字标签 |
| 图注自包含 | ✅ | ✅ |
| 30 秒内看懂问题/失败 | ✅ 三个局部选择 → J4 等待 → MA 协调 | ✅ 左到右一条链 |
| 具体场景，不是抽象占位 | ✅ J4、M1–M6、V1/V2、三个规则名 | ✅ 示例规则名与候选 |
| 输入/输出明确 | ✅ L/U、三条路径、MS | ✅ Shop state → Actions |
| 名字与正文小节一致 | ✅ | ✅ |
| 分层动机可见 | ✅ PA/LA/RA 三卡 + 频次带 + 冲突 | ✅ 上推理/下训练；Policy 内三级 |
