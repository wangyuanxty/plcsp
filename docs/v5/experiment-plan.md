# v5 实验计划

> 状态：2026-10-24 起草；2026-10-24 21:00 主模型改为 GRPO。  
> 2026-10-25 01:00 machine ST 默认改为 intra1；AGV 批次钉 intra0。  
> 主模型：顶层规则选择 + job / machine / AGV 三个低层 agent，下层全用 Set Transformer；低层用 GRPO。  
> 本文件只定实验口径，不替代 `docs/v5/progress-log.md` 的读数记录。

## 0. 一句话

在主论文 E2E-MAPPO 的 MT-FJSP 上，做一个**分层多智能体 GRPO**：

- 顶层：规则选择器，每 `ΔT` 步选一个（job 规则, machine 规则）；顶层仍用 PPO。
- 低层：job / machine / AGV 三个 actor，全部用 Set Transformer。
- 低层价值：**无 critic**；组内 z-score 作 advantage。
- 训练：每组随机 λ，组内固定；评测：固定 `0.4/0.4/0.2`。
- 默认无 AGV 是环境对照；主模型开 AGV actor。

## 1. 研究问题

| 编号 | 问题 | 对照 |
|---|---|---|
| RQ1 | 下层全 ST 是否不差于原 GIN/GAT？ | E2E vs job_st vs settrans |
| RQ2 | 顶层规则选择是否改善固定目标？ | RS off vs RS on |
| RQ3 | AGV actor 是否优于规则派车？ | B（规则）vs C（actor） |
| RQ4 | AGV 资源约束的代价多大？ | A（无限 AGV）vs B（规则） |
| RQ6 | 相对 NSGA-II 的 Pareto 前沿质量如何？ | **暂不做（2026-10-25）**；λ 扫描 + HV / IGD |
| ~~RQ5~~ | ~~偏好条件化是否产生有效 trade-off？~~ | **已取消（E8）** |

注意：

- RQ3 只能在**同一 AGV 环境**内比。  
  A vs C 不能回答 actor 好坏。
- 论文的 732.5 是原 E2E 的 seed3 测试。  
  我们的 ST / AGV 口径不同，不能并列声称。

## 2. Agent 命名

### 推荐

| 名称 | 缩写 | 角色 | 动作 | 价值头 |
|---|---|---|---|---|
| **Rule Selector** | **RS** | 顶层；每 `ΔT` 步选规则对 | 12 个（job 规则 × machine 规则） | 顶层 PPO critic |
| **Job Agent** | **JA** | 选下一个工序 | 候选 job / operation | GRPO 无 critic |
| **Machine Agent** | **MA** | 选加工机器 | 候选 machine | GRPO 无 critic |
| **AGV Agent** | **AA** | 选源端 / 目标端 AGV | 候选 AGV | GRPO 无 critic |
| **Global Critic** | **GC** | 集中式 4 目标价值 | — | 只在 PPO 对照 |

完整模型工作名：

> **H-ST-GRPO** = Hierarchical Set-Transformer Multi-Agent Policy
> Optimization with Group-Relative Advantages

简称仍用 **H-ST-GRPO**。

定义：

- `H`：顶层 Rule Selector；顶层仍用 PPO。
- `ST`：JA / MA / AA 三个 actor 的 task / machine 编码器都用 Set Transformer。
- `GRPO`：低层多 actor；组内 z-score 作 advantage；无局部 / 全局 critic。
- `PPO 对照`：低层局部 critic + 集中式 global critic；GAE。

### 和原始 GRPO 的关系

原始 GRPO：

> Shao et al., “DeepSeekMath: Pushing the Limits of Mathematical
> Reasoning in Open Language Models,” arXiv:2402.03300, 2024.

借的：

- 组内相对优势（group z-score）。
- PPO clip。
- 低层无 critic。

不同的：

- 无 reference model，无 KL。
- 组 = 同一批实例的 G 个 episode，不是同一 prompt 的 G 个回答。
- 三个 actor（JA / MA / AA）共享 team advantage。
- 每组随机 λ，4 目标加权。
- 顶层 RS 是独立的分层 PPO。
- AA 同时选源端和目标端 AGV。

我们的适配：

- 分层 RS。
- 多 actor 共享优势。
- 每组随机 λ。
- 两端 AGV actor。
- ST 编码器（架构）。

写法要求：

- 首次出现写全称。
- 引用 Shao et al. 2024。
- 不说“我们提出了 GRPO”。
- 说“我们采用 group-relative 优势，并做多智能体 / 分层适配”。

说明：

- RS 当前是小 MLP，不是 ST。  
  顶层输入是 18 维状态，用 MLP 足够。
- 若要把 RS 也换成 ST，需要先把状态 token 化；列为 E6b，不进主模型。
- `OA` / `PA` / `LA` 是 v3 / NHDRL 线的名字，v5 不复用，避免混淆。

### 备选

| 名称 | 缩写 | 评价 |
|---|---|---|
| Manager | MG | 分层 RL 常见，但容易被理解为“生成子目标”；我们只是选规则 |
| Coordinator | CO | 含义太宽 |
| Transport Agent | TA | 可以；但 AGV Agent 更具体 |
| Operation Agent | OA | 与 v3 的 OA 冲突，不推荐 |

## 3. 主模型

### 3.1 组成

```text
RS: 18 维全局状态 -> 12 个规则对
    10 维摘要
    + 机器负载不均衡
    + WIP
    + AGV 忙车比例
    + AGV 平均等待
    + 最近一段 cost 变化
    + 3 个当前 λ
     每 ΔT=8 个低层 step 选一次
     规则对 = (job_rule, machine_rule)

JA: task ST tokens (33 维) -> 候选 job 打分
MA: machine ST tokens -> 候选 machine 打分
    每组机器的 m1/m2/m3 分开投影
    m1 = 6 维（有 AGV 时 10）
    m2 = 8 维
    m3 = 3 维（--st_full_context 时 8）
    -> 机器内 attention（3 token，1 层，2 头）
    -> 机器间 attention（3 层，4 头）
AA: AGV ST tokens -> 源车间 AGV 打分

GRPO（主模型）：低层无局部 critic、无全局 critic。
  组：同一批实例的 G=8 个 episode。
  组回报：每个 env/episode 的 4 个最终 cost，λ 加权成标量，组内 z-score。
  所有低层 actor 共享 team advantage。
PPO（对照）：JA / MA / AA 局部 critic + 4 目标 global critic；GAE。
```

### 3.2 规则集

Job 规则（4）：

- `est`：最早预估开始
- `eft`：最早预估完成
- `spt`：最短预估加工时间
- `mwko`：剩余工序最多

Machine 规则（3）：

- `min_t`：最短加工时间
- `min_pt`：最低加工能耗
- `min_trans`：最短运输时间

当前 RS 只控制 job / machine；不控制 AGV。  
若要把 AGV 规则也交给 RS：4 × 3 × 3 = 36 个动作，列为 E6b。

### 3.3 偏好口径

- 低层 GRPO：每组随机 λ，组内固定；`--random_weight_type eval`（代码强制）。
  组内共享同一个 λ，保证组回报可比。
- 验证 / 测试：固定 `0.4/0.4/0.2`。
- RS 的段回报：用**当前 λ**。
  - GRPO 训练：当前组的随机 λ。
  - 验证 / 测试：固定 `0.4/0.4/0.2`。
- 顶层状态含当前 λ：
  - 10 维摘要 + 机器负载不均衡 + WIP + AGV 忙车比例
    + AGV 平均等待 + 最近一段 cost 变化 + 3 个 λ = **18 维**。
  - 2026-10-25 修复：之前只有 10 维，顶层看不见 λ。
  - 2026-10-25 追加：机器负载不均衡、WIP、AGV 忙车比例、
    AGV 平均等待、最近一段 cost 变化。
- `--hier_alpha` 默认从 1.0 降到 **0.5**（2026-10-25）。
- 因此顶层是 **preference-conditioned**。
  - 低层学随机 λ 的鲁棒性。
  - 顶层按当前 λ 选规则。
- 不能声称顶层只看规则、不看偏好。  
  它现在两者都看。

### 3.4 AGV Stage-0

- 每车间独立车队；AGV 不跨车间。
- 跨车间 = 源 AGV 段 + 固定延迟 + 目标 AGV 段。
- 边界 = 目标车间第一台机器。
- 源车间 AGV：actor 选（C）或最早空闲规则（B）。
- 目标车间 AGV：与源端共用同一个 actor（C）；规则臂用最早空闲（B）。  
  2026-10-25 起改为两端都由 actor 选；旧行为是目标端固定规则。
- 开关：`--agv_dst_actor 1` 是默认；`0` = 仅源端 actor（旧 pilot）。
- B 的规则定义：源 / 目标车间内各选 `free_time` 最小的 AGV。  
  忙车也可选，等待时间计入延迟；同值按索引破平。  
  规则不看空驶距离、目的 AGV 状态。
- 旧 source-only 批次钉 `--machine_intra_attn 0`，只作 pilot。  
  2026-10-25 的新批次用 `--machine_intra_attn 1`。
- **空驶**：AGV 从当前位置到取货点的时间计入调度延迟。
  源 / 目标 AGV 的空驶都计入；TEC 仍只计载货行驶。
  （2026-10-24 修复；旧 AGV 读数作废。）
- 无 WAIT、无充电、无 L/U、无 AGV 故障。
- TEC 只计载货行驶；空驶和等待不计 TEC；无 AGV 功率。

### 3.5 AGV 三组标签（A / B / C）

A / B / C 是 **AGV 策略标签**，不是实验编号。

| 标签 | 计划编号 | 环境 | 源车间 AGV | 目标车间 AGV |
|---|---|---|---|---|
| **A** | E0 / E1 | no-AGV（无限 AGV） | 无 | 无 |
| **B** | E2 / E4a | 有限车队 | 最早空闲规则 | 最早空闲规则 |
| **C** | E3 / E4b | 有限车队 | 学习 actor | 学习 actor（同一策略） |

读法：

- A vs B：加 AGV 资源约束的代价。
- B vs C：学习 AGV actor 相对规则派车的收益。
- A vs C 不能回答 actor 好坏。

注意：

- C 是两端都学习：源端与目标端共享同一个 AGV actor。
- 主模型 E4b 用的是 C，不是 B。

## 4. 默认训练口径

除非显式说明：

```text
--encoder_type settrans
--machine_intra_attn 1
--st_token_context 1 --st_task_features 1 --st_edge_features 0
--algo grpo
--buffer_size 8 --resample_freq 8
--LR 1e-3
--use_lr_decay 0
--eval_sample 100
```

- `--machine_intra_attn 1` 是 2026-10-25 起的新默认。
- `--st_token_context 1 --st_task_features 1` 也是代码默认（2026-10-25 起）。
- 旧 checkpoint 用 intra0 训练。
  加载它们必须显式 `--machine_intra_attn 0`。
- 旧 checkpoint 用 12 维特征训练。
  加载它们必须显式 `--st_token_context 0 --st_task_features 0`。
- 新矩阵的 AGV 配置用 intra1。
  旧的 source-only / 12 维批次用 intra0，只作 pilot。

- GRPO 组：同一批实例的 G=8 个 episode。
- λ：代码强制 `--random_weight_type eval`；每组随机一个 λ，组内固定。
- PPO 同协议对照：`--algo ppo --random_weight_type 01 --buffer_size 8 --resample_freq 8`。
- job token：12 + 8 + 13 = 33 维。
- 论文学习率：Adam `1e-3`；论文 Adam `eps=1e-10`，代码默认 `1e-5`。
  复现论文口径要显式 `--lr_eps 1e-10`。
- 训练预算：**所有实验统一 200 ep**（2026-10-25 用户决定）。  
  不跑 1000。
- 2026-10-25：先跑完 200 ep 全矩阵，再决定是否补 400 ep。  
  当前批次不重启、不改预算。

## 5. 评测口径

| 项 | 口径 |
|---|---|
| 训练验证 | seed1，100 实例，每 10 ep 一次，贪心 |
| top1 | 验证 Obj 最小的 checkpoint |
| 测试 | seed3，100 实例，贪心；报 mean ± std |
| 目标 | `0.4/0.4/0.2` 的 Obj；同时报 mk / pt / trans / idle |
| λ 网格 | 暂不做（E10 暂缓） |
| 多 seed | 主结论至少 3 个训练 seed；配对比较测试实例 |
| 记录 | 每次 run 保存 full ckpt、top1、CSV、log、seed |

单 seed 只能当 pilot。  
不得用单 seed 的 1–2 个点声称方法优劣。

## 6. Baselines

### 6.1 外部已发表基线

| ID | 基线 | 状态 | 来源 / 口径 |
|---|---|---|---|
| B0 | 论文 E2E P-G | 只作参考 | 732.5 ± 96.8，seed3，100 实例，论文 Table V |
| B1 | 12 条 PDR + RA | **两个规模、两模式都已跑** | 6×6 表 B（AGV）：最好 PDR：MOR + SPT **951.20 ± 139.5**；RA **1258.30 ± 207.4**。6×6 表 A（no-AGV）：最好 PDR：MOR + SPT **913.87 ± 136.0**；RA **1085.46 ± 178.8**。10×6 表 B：最好 PDR：MOR + SEYC **1101.30 ± 187.4**；RA **1603.37 ± 275.5**。10×6 表 A：最好 **1032.77 ± 160.2**；RA **1302.95 ± 204.9**。均为 seed3 100 实例 |
| B2 | MIP / Gurobi | seed3 CSV 已有 | `tester/Solver_seed3/`；限时 60/240/600 s。6×6 no-AGV incumbent：**585.83 ± 82.7**；中位求解 14.0 s。10×6 no-AGV incumbent：**508.39 ± 52.7**；全部触 60 s 上限。**都不是已证最优界** |
| B3 | Lei et al. 2022（multi-action DRL / multi-PPO，MPGN） | **adapted 版已跑两个规模** | K. Lei, P. Guo, W. Zhao, Y. Wang, L. Qian, X. Meng, L. Tang, "A multi-action deep reinforcement learning framework for flexible job-shop scheduling problem," *Expert Systems with Applications*, vol. 205, Art. no. 117796, 2022. DOI: 10.1016/j.eswa.2022.117796；权重在 `tester/ESWA_MPPO/`。6×6：no-AGV **700.60 ± 131.5**；AGV **707.21 ± 127.8**。10×6：no-AGV **658.32 ± 117.2**；AGV **651.30 ± 111.3**。job/machine 策略 + 最早空闲 AGV 规则；机器头首层 6→10 零填充。**只能写 adapted** |
| B4 | DRL [23]（MPGN，Lei et al. 2022） | **与 B3 同一篇** | 已合并到 B3；B3 的权重是 E2E 作者的四目标改写版，不是原仓库代码 |
| B19 | RGA（Zhang et al. 2011） | **两个规模、两模式已实现** | G. Zhang, L. Gao, Y. Shi, "An effective genetic algorithm for the flexible job-shop scheduling problem," *Expert Systems with Applications*, vol. 38, no. 4, pp. 3563–3573, 2011. 6×6：AGV **478.04 ± 60.4**（实例内 23.85）；no-AGV **474.87 ± 56.0**（20.94）。10×6：AGV **565.63 ± 52.5**（26.66）；no-AGV **564.62 ± 47.8**（22.97）。pop 50 / gen 100 / 4850 evals / 3 seeds |

B19 只做单目标 RGA。  
2SGA 暂不加（用户 2026-10-25 决定）。

⚠️ 2026-10-25 22:25 核查：**E2E 原文基线含 RGA / 2SGA**，但未给 RGA 数值参数，官方仓库无 RGA 代码。  
用户决定：论文主表**暂不放 RGA**；**不写「与 E2E 相同基线」**。  
主表只放 PDR / RA / MIP / B3 / E2E P-G，这是 E2E 基线的子集。  
RGA 读数保留在 `results/` 和内部记录，不删除。  
不得写「优于所有基线」；只写「优于所比较的基线」。  
RGA 若被审稿人要求，可放附录或限制说明。

B3 与 B4 是同一篇。  
论文 [23] 就是 Lei et al. 2022。  
`tester/ESWA_MPPO/` 的权重由 E2E-MAPPO 作者改写为四目标（critic 输出 4 个值）。  
写论文时只能写 “adapted”，不能写 “复现”。

### 6.1b 经典多目标基线

| ID | 基线 | 状态 | 说明 |
|---|---|---|---|
| B13 | NSGA-II | **暂不做（2026-10-25）** | Deb et al., IEEE TEC 6(2):182–197, 2002；pymoo 0.6.2 |

2026-10-25 用户决定 E10 先不做。  
本节保留，供以后恢复。

NSGA-II 的必要性：

- 这是多目标调度论文的标准基线。
- 论文 P-G 的 λ 网格只有一个点序列，不是完整 Pareto 前沿。
- NSGA-II 能给每个实例一条前沿，用于 HV / IGD。

现状：

- `plcsp_v3/baselines/nsga2.py` 存在，但它面向 v3 的
  **三目标（makespan / energy kWh / TWT）和 v3 DES**。
- 不能直接用于 v5 的四成本 / MT-FJSP / no-AGV 环境。
- 需要重写解码器：
  - 染色体 = 工序优先级 + 机器分配。
  - 解码 = 按优先级选可开工工序，按基因选机器。
  - 评价 = v5 no-AGV 环境。
- 目标建议用 3 个：MKT / EC（PEC+SEC）/ TEC，与 λ 网格一致；
  同时报 4 个原始 cost。

### 6.2 内部复现 / 消融基线

| ID | 基线 | 状态 |
|---|---|---|
| B5 | E2E GIN/GAT（`e2e_control`） | best 703.26 @EP240；λ 网格目标 686.91 |
| B6 | E2E 固定权重 | seed3 681.98 ± 91.25 |
| B7 | `job_st`（ST job + GAT machine） | seed1 676.17；seed3 659.85 |
| B8 | `settrans`（双 ST，intra1） | 旧 intra0 固定 λ 对照 811.45（历史）；主模型口径未跑 |
| B9 | A：no-AGV（无限 AGV） | GRPO E0 / E1 待跑；PPO 历史 775.27 |
| B10 | B：AGV 规则派车 | deadhead 修复后 PPO 在跑；旧数字作废 |
| B11 | C：AGV actor | deadhead 修复后 PPO 在跑；旧 774.96 / 818.36 作废 |
| B12 | RS off / on | GRPO E0 / E1 待跑 |
| B14 | **GRPO 6×6（e2e，G=8，200 ep）** | seed3 **676.99 ± 14.3**（3 seeds） |
| B15 | PPO 同协议对照（e2e，G=8，200 ep） | seed3 **790.20 ± 147.12**（3 seeds） |
| B15b | PPO 同协议对照（settrans 33 维，G=8，200 ep） | seed3 **637.41 ± 27.4**（3 seeds；676.03 / 621.10 / 615.11）；GRPO E0 559.31，3/3 更低 |
| B16 | **GRPO 10×6（e2e，flat，200 ep）** | seed3 **613.98**（实例 std 81.79；1 seed）；论文 Table V 10×6×2 P-G 674.5 ± 88.9 |
| B17 | intra0 vs intra1（PPO，`hybrid_full`） | 验证 EP49：820.73 vs 722.98；未完成 |
| B18 | AGV deadhead 修复（PPO，hier） | 验证 EP119：actor 833.40 vs 规则 866.17；未完成 |

### 6.3 基线规则

**两张表，用途不同。**

| 表 | 环境 | 用途 |
|---|---|---|
| A：代码校验 | no-AGV | 对论文 732.5；检验代码正确性 |
| B：论文主表 | **AGV** | 比较方法优劣 |

- 表 A 不是论文主表。  
  它只回答「我们的 no-AGV 实现是否和原论文同一量级」。
- 表 B 的基线**必须带 AGV**。
- PDR-AGV：规则选工序 / 机器；运输用最早空闲 AGV。
- RGA-AGV：同一 OS/MS 解码；运输用最早空闲 AGV。
- B3-AGV：Lei et al. 2022 的 job/machine 策略 + AGV 规则。  
  只能写 “adapted”，不得写 “复现”。
- MIP-AGV：需要新的 MILP。  
  原模型没有 AGV 变量；只在 6×6 限时试，报 gap 和求解时间。
- MIP / B3 的**原模型**没有 AGV。  
  原样数字只能进表 A。
- RGA 必须报种群、代数、实际评价次数、墙钟时间、随机种子。
- RGA 是单目标基线，不是 NSGA-II；两者不混。
- 所有基线必须用同一批 seed3 100 实例、同一 `0.4/0.4/0.2` 权重。
- 论文数字和我们的复现数字必须分开列，标清来源。
- NSGA-II 暂不做。
- **2026-10-25 环境修复**：跨车间固定延时曾被计两次。
  - `find_max_arrivaTime_for_currentNode` 用整段表值，又加含延时的 `wait`。
  - 修复后用装载段覆盖值，延时只计一次。
  - 旧 E2 / E3 / E4a / E4b 读数作废。
  - 旧 E2 / E3 检查点重评：**576.04 / 557.72**。
  - E4a / E4b 顶层是 10 维，当前评测是 18 维；必须重训。
- 基线解码器与修复后的环境一致（`check_agv_env.py`，误差 <1e-3）。

### 6.4 缺口

- 表 A（no-AGV 代码校验）：PDR / RA / RGA / B3 已跑；MIP 有原 CSV。
- 表 B（AGV 主表）：
  - PDR-AGV 已实现并跑完。
  - RGA-AGV 已实现并跑完。
  - B3-AGV 已适配：**707.21 ± 127.8**（adapted）。
  - MIP-AGV 未建模。
- AGV 环境已修复；E2 / E3 / E4a / E4b 重训队列已就绪。  
  `experiment-queue-agv-fix-6x6.json`，24 个 job。
- MIP 原模型只覆盖部分规模 / 权重；6×6 seed3 有 CSV。  
  该 CSV 是限时 incumbent，不是已证最优界。
- B3 的预训练模型已在 `eval_b3.py` 跑通；只能写 adapted。
- RGA 已实现；参数、随机种子、评价次数已记录。
- NSGA-II 暂不做（E10 暂缓）；未适配 v5。
- 外部 DRL [23] 没有原始代码；有 B3 的四目标改写权重，可跑 adapted 版本。

### 6.5 三层 RS（2026-10-25，A1 + K=3）

2026-10-26 01:10 决定：**顶层回到 3 头 RS 规则先验**。  
λ / 目标权重 agent 放弃（每 phase pilot 不显著，p≈0.14）。  
RS 是条件性贡献：AGV 臂为正，无 AGV 为负，熵会坍缩。  
RS 队列仍停，等非 RS 跑完再恢复。

顶层从 1 个 12 路 (job, machine) pair 头，改成 3 个头：

| 头 | 动作数 | 规则 |
|---|---:|---|
| job | 4 | est / eft / spt / mwko |
| machine | 3 | min_t / min_pt / min_trans |
| AGV | 3 | earliest_free / nearest / random |

- 独立采样；log-prob 相加；熵相加；critic 仍是一个标量。
- E1（无 AGV）：2 个头。
- E4a / E4b（有 AGV）：3 个头。
- E4a：顶层选 AGV 规则，直接派车（`--agv_top_rule 1`）。
- E4b：AGV actor + 顶层规则 logit 先验（`--agv_top_rule 0`）。
- E2 / E3：无顶层，不变。

规则先验：

- job / machine：固定 logit 向量。
- AGV：状态相关；按当前 AGV 特征归一化到 [-1, 1]。
- source 和 destination 两个决策用同一个选中的 AGV 规则。

影响：

- 旧 E1 / E4a / E4b / rslam 读数全部作废。
- E0 / E2 / E3 不是 RS，不受影响。
- 重跑队列：`experiment-queue-rs-heads3-6x6.json` 和
  `experiment-queue-rs-heads3-10x6.json`。
- 先跑 6×6；10×6 随后。
- RS λ / 状态 / 超参扫在核心臂之后。

熵：

- 3 头熵相加，约 3.5。
- `hier_entropy=0.01` 的等效强度约为单头的 3 倍。
- 先保留，观察是否仍坍缩。

## 7. 实验矩阵

### E0：默认 6×6 基线（RS off，无 AGV）

```text
settrans
context + task，33 维
GRPO，G=8，每组随机 λ
LR 1e-3
200 ep
```

用途：主模型的 no-AGV、RS-off 基线（GRPO）。

### E1：默认 6×6 + RS

E0 + `--hier 1 --hier_interval 8 --hier_alpha 1.0`。

用途：RQ2。

### E2：AGV 规则派车 B

E0 +：

```text
--use_agv 1 --n_agv 6 --cross_delay 5 --cross_leg_ratio 0.5
--agv_rule 1 --agv_dst_actor 1
```

用途：RQ3 / RQ4 的规则基线。  
规则臂两端都用最早空闲。

### E3：AGV actor C

E0 +：

```text
--use_agv 1 --n_agv 6 --cross_delay 5 --cross_leg_ratio 0.5
--agv_dst_actor 1
```

用途：RQ3 的方法组。  
两端都由 actor 选；同一个策略。

### E4：主模型

- E4a = E0 + `--hier 1` + `--use_agv 1 --agv_rule 1 --agv_dst_actor 1`
  （RS on + AGV 规则，两端规则）
- E4b = E0 + `--hier 1` + `--use_agv 1 --agv_dst_actor 1`
  （RS on + AGV actor，两端 actor）= **主模型（GRPO）**

⚠️ GRPO + RS + 两端 AGV actor 已过 10 ep smoke。  
正式 200 ep 在 `experiment-queue-agv-both.json`，未完成。

与 E0 / E1 / E2 / E3 组成 2 × 3 的对照：

|  | 无 AGV | AGV 规则 | AGV actor |
|---|---|---|---|
| RS off | E0 | E2 | E3 |
| RS on | E1 | E4a | **E4b（主模型）** |

主模型 = RS on + AGV actor。

### E5：encoder 消融

同一 GRPO 口径，只换 encoder **家族**。  
机内 attention 属于 encoder 的一部分，不算额外变量。

- `e2e`：GIN job + GAT machine
- `job_st`：ST job + GAT machine
- `settrans`：ST job + ST machine（含 intra1，主模型）

三臂的内部融合方式不同。  
这是 encoder 家族差异，不是混杂变量。

用途：RQ1。  
旧的固定 λ / LR 3e-4 组只作历史对照，不进主表。

### E5b：machine ST 特征融合方式

目标口径：主模型 GRPO，只换 machine ST 的融合层。

- `intra0`：m1/m2/m3 拼接 → 2 层 MLP 投影 → 机器间 attention
- `intra1`：m1/m2/m3 分开线性投影 → 机器内 attention（3 token，2 头）
  → 机器间 attention（主模型，2026-10-25 起为默认）

用途：结构消融。  
考察「按特征组分开」和「机器内交互」是否必要。

旧 PPO 批次（已停）：PPO + hier + `hybrid_full`，LR 3e-4，buffer 5，6×6×2，
无 AGV，3 seeds，400 ep，`--machine_intra_attn 0/1`。

新矩阵：GRPO + settrans + 200 ep。  
`E5b_intra0` 直接对 `E0`（intra1）。

⚠️ 该口径不是主模型的 GRPO 口径。  
写进论文前，需在主模型口径下重跑，或明确标注口径差异。

当前读数（2026-10-25 00:46，EP49 / 400）：

- `intra0`：820.73（s1 825.03 / s2 774.21 / s3 862.96）
- `intra1`：722.98（s1 692.98 / s2 660.73 / s3 815.22）
- 近似 top1：785.39 vs 722.98

未完成，不下结论。

### E6：顶层消融

只在 E1 有希望时做：

- E6a：`ΔT ∈ {4, 8, 16}`；`α ∈ {0.5, 1.0, 2.0}`
- E6b：RS 动作加入 AGV 规则（12 → 36）
- E6c：RS 读 λ（preference-conditioned top level）
- E6d：规则子集（只用 est/spt/mwko 等）

### E7：10×6 跨尺寸

E0 / E1 / E4 在 10×6×2 上跑 200 ep。  
10×10×2 实测过慢，已停；先做 10×6。  
所有实验统一 200 ep，不再上 1000。

### E8（已取消）

用户决定不做偏好嵌入线。

- 代码 `pref_embed` 保留，但不进 v5 计划。
- 不再跑 21 点 λ 网格。
- 不再重跑 LR 3e-4。

### E9：多 seed

主模型、E0、E1、E2、E3 各 3 个 seed。  
配对比较同一测试实例。  
报告均值、标准差、胜率、配对检验。

已完成的第一格：GRPO 6×6 flat vs PPO 同协议，3 seeds。

- GRPO：676.99 ± 14.3
- PPO：790.20 ± 147.12
- 配对 p≈0.28（n=3，不显著；方向 3/3 一致）

### E10（暂不做）：λ 扫描 + Pareto（NSGA-II）

2026-10-25 用户决定先不做。  
以下内容保留，供以后恢复。

- 用训练好的主模型，在 21 个 λ 点跑 seed3 100 实例。
- 每个实例得到 21 个点。
- NSGA-II 在同一实例、同一 no-AGV 环境产生前沿。
- 指标：
  - 目标点 `0.4/0.4/0.2` 的 Obj
  - HV、IGD
  - 前沿点数
  - 墙钟时间、仿真次数
- 参考前沿：
  - 预先登记：所有方法前沿的非支配并集。
  - 若并集少于 2 点，只报 HV。
- 注意：这不是 E8。E8 是改网络；E10 只做评测。

## 8. 计算预算（4090，参考）

| 项 | 估计 |
|---|---|
| 6×6 200 ep | 约 50 min（独占） |
| 10×6 200 ep | 约 2.5–3 h |
| 10×10 200 ep | 实测过慢，停用 |
| λ 网格 21×100 | 暂不做（E10） |

规则：

- 上表是**独占**估计。  
  多 run 并发会明显变慢。
- 所有实验统一 **200 ep**（2026-10-25 用户决定）。
- 同一时间最多 6 个 run（当前 `max_parallel=6`）。
- 每 100 ep 存 full ckpt；每 10 ep 存验证。

## 9. 停止规则

- 200 ep 最好点明显差于对照，且最后 50 ep 不降：停，记负例。
- AGV actor（C）在 200 ep 明显差于规则（B）：停 C，不再加预算。
- RS on（E1）在 200 ep 不优于 RS off（E0）：先查 top-level 规则使用/熵，不盲目调 α。
- 10×6 在 200 ep 不优于 6×6 的相对趋势：停，不再加预算。

## 10. 当前状态（2026-10-25 18:10）

### 18:10 更新（最新）

- 6×6 的 E0 / E1 / E2 / E3 / E4a / E4b 已完成；ST 消融完成。
- 发现并修复 AGV 环境 bug：跨车间固定延时重复计费（§88）。
- 旧 E2 / E3 / E4a / E4b 的 AGV 读数作废。
- 旧 E2 / E3 检查点用固定环境重评：**E2 576.04，E3 557.72**。
- E4a / E4b 顶层是 10 维，必须重训。
  重训队列 `experiment-queue-agv-fix-6x6.json` 已就绪，24 个 job。
- E0 PPO 同协议对照 3 seeds **已完成**：**637.41 ± 27.4**（676.03 / 621.10 / 615.11）。GRPO E0 559.31，3/3 更低。
- ST 消融：settrans intra1 **559.31**；intra0 624.63；jobst 648.81；e2e 676.99。
- 10×6 矩阵未开始。
- 10×6 矩阵 **已启动（21:12）**：E0 / E1 / E4a / E4b × 3 seeds。
  - 队列：`experiment-queue-10x6-main.json`，24 个 job。
  - max_parallel 3。
  - 首轮 E0 s1 / E1 s1 / E4a s1。
- **RS 3 头重跑已启动（20:55）**：6×6 E1 / E4a / E4b 各 3 seeds。
  - 顶层：job 4 + machine 3 + AGV 3。
  - E1 用 2 头；E4a / E4b 用 3 头。
  - 队列：`experiment-queue-rs-heads3-6x6.json`。
- 旧 E1 / E4a / E4b / rslam 读数作废。
- RGA 基线：AGV **478.04**；no-AGV **474.87**。质量优于 DRL。
- 墙钟：RGA-AGV **1.63 s/实例**（远程 CPU，1 worker）；DRL-AGV **0.56 s/实例**（GPU，单评测，无并发）。DRL 约快 **2.9×**，不是数量级。

### 历史（07:17 口径）

以下为 07:17 的历史记录。  
其中 E2 / E3 / E4a / E4b 的 AGV 数字已作废，保留作对照。

02:12 停止更早的旧批次。  
02:18 启动两端 AGV 批次；该批次是 12 维，只作 pilot。  
07:17 停止该批次，启动 33 维全计划矩阵。

### 主模型（GRPO）

| 组 | 状态 | 读数 |
|---|---|---|
| GRPO 6×6 flat（e2e，no AGV，no RS） | **完成** | seed3 **676.99 ± 14.3**（3 seeds） |
| PPO 同协议对照 6×6 | **完成** | seed3 **790.20 ± 147.12**（3 seeds） |
| GRPO 10×6 flat（e2e） | **完成** | EP199 验证 609.23；seed3 **613.98**（实例 std 81.79，1 seed）；论文 Table V 674.5 ± 88.9 |
| 全计划矩阵（E0–E5b） | **在做** | 6×6 九配置 + 10×6 四配置；3 seeds；200 ep |
| 矩阵 E0（6×6，33 维，无 AGV，RS off） | **完成** | seed3 **559.31 ± 12.2**（3 seeds） |
| 矩阵 E1（6×6，+RS，旧 12 路头） | **作废** | 570.72；旧动作头不可比；3 头重跑中 |
| 矩阵 E2（6×6，AGV 规则，RS off） | **重训中** | 旧 587.06；固定环境重评 **576.04** |
| 矩阵 E3（6×6，AGV actor，RS off） | **重训中** | 旧 570.95；固定环境重评 **557.72** |
| 矩阵 E4a（6×6，RS + AGV 规则，旧 12 路头） | **作废** | 645.22；3 头重跑中 |

E1 比 E0 差：RS on 在 6×6 没有改善。  
RQ2 暂为负结果。详见 `progress-log.md` §71。

顶层诊断（E1 s1 top1，500 次决策）：

- 训练熵从 2.485 塌缩到 0.68–1.62。
- greedy argmax **500/500 = action 11**。
- action 11 = **mwko + min_trans**。
- `min_trans` 概率质量 0.937。

顶层没有学成状态相关的规则策略。  
详见 §75。

E2 比 E0 高 27.75：有限 AGV + 规则派车的代价。  
E3 比 E2 低 16.11：actor 收回 **58%** 的约束代价。  
这是 RQ3 的 B vs C 读数。详见 §72 / §73。

E4a 的 s1 崩到 745.25；去 s1 后 595.20，仍不优于 E2。  
RS + AGV 规则不稳定。详见 §76。

⚠️ 2026-10-25 13:52 顶层状态原来只有 10 维，看不见 λ。  
已改为 **18 维**：10 摘要 + 5 个状态量 + 3 个当前 λ。  
顶层段回报用**当前 λ**。  
6×6 的 E1 / E4a / E4b 旧代码作废；`rslam` 队列重跑。  
详见 `progress-log.md` §79。

### AGV（旧 source-only pilot，已停）

| 臂 | s1 | s2 | s3 | 均值 | 近似 top1 |
|---|---:|---:|---:|---:|---:|
| actor | 861.09 | 862.48 | 852.18 | **858.58** | 804.48 |
| 规则 | 820.87 | 844.98 | 873.15 | **846.33** | 798.73 |

EP199 最终验证：规则低 **12.25**。  
近似 top1：规则低 **5.75**。  
这批只作 pilot；新批次两端都由 actor 选。

### 新批次（全计划矩阵，33 维）

2026-10-25 07:17 启动。  
`experiment-queue-plan-matrix.json`，78 job（39 训练 + 39 评测）。  
10×10 不跑。

6×6×2：

- E0 / E1 / E2 / E3 / E4a / E4b
- E5_e2e / E5_jobst
- E5b_intra0
- 9 个配置 × 3 seeds = 27

10×6×2：

- E0 / E1 / E4a / E4b
- 4 个配置 × 3 seeds = 12

公共口径：

- GRPO
- 200 ep，buffer 8，resample 8
- LR 1e-3
- 33 维任务特征
- seed 1/2/3

⚠️ 旧 AGV 批次是 12 维，已停。  
6×6 的 582.59 只作 12 维 pilot，不并入本矩阵。  
详见 `progress-log.md` §68。

### machine ST 消融（旧批次，已停）

| 臂 | 最后读数 | EP | 近似 top1（EP49） |
|---|---:|---:|---:|
| intra0 | 816.90 | 129 | 785.39 |
| intra1 | 708.07 | 119 / 129 | 722.98 |

EP 不完全对齐。  
intra1 明显更好，但批次已停。  
top1 未按最后读数重算。

### PPO / 历史线

| 组 | 状态 | 读数 |
|---|---|---|
| E0 6×6 默认基线（PPO） | 历史 | seed3 775.27 ± 109.13 |
| E1 6×6 默认 + RS（PPO） | 历史 | best 793.04 @EP180；final 816.41 |
| E2 AGV 规则 B（PPO） | **旧读数作废** | 761.60 是空驶修复前 |
| E3 AGV actor C（PPO） | **旧读数作废** | 774.96 / 818.36 是空驶修复前 |
| E7 10×10 默认（PPO） | 已停 | 改跑 10×6 |
| E8 E2E + pref | **已取消** | 不进计划 |
| E10 NSGA-II / Pareto | **暂不做** | 2026-10-25 用户决定 |

GRPO 6×6 与 PPO 同协议对照：

- 同架构（e2e GIN+GAT）、同实例、同 200 ep、同 buffer 8、同评测。
- GRPO 均值低 **113.21**，std 小一个数量级。
- 3/3 seed 方向一致；n=3 的配对检验不显著（p≈0.28）。
- 这是“主模型改 GRPO”的依据；不是最终显著性结论。

### 外部基线

| 基线 | 状态 |
|---|---|
| B0 论文 P-G | 只作参考 |
| B1 12 PDR + RA | 代码已有，v5 未跑 |
| B2 MIP / Gurobi | 旧 CSV 已有，v5 未跑 |
| B3 Lei et al. 2022 | 权重已有，v5 未跑；只能写 adapted |
| B19 RGA | 未实现 |
| B13 NSGA-II | 暂不做 |

## 11. 口径风险

1. **单 seed**：主结论必须多 seed；当前 GRPO / PPO 对照已跑 3 seed，但 n=3 仍偏少。
2. **论文可比性**：论文 732.5 是原 E2E seed3；我们换 encoder / 特征 / AGV，不能并列。
3. **A vs C**：AGV 环境多一个资源约束；只能 B vs C 回答 actor 好坏。
4. **TEC 口径**：AGV TEC 只计载货行驶；与论文的运输能耗假设不同。
5. **RS 目标**：顶层用当前 λ；状态 18 维（10 摘要 + 5 个状态量 + 3 个 λ）。
   顶层是 preference-conditioned。
   旧 10 维顶层 checkpoint 不可用。
6. **edge 特征**：当前默认 0；若“全特征”包含 edge 4 维，需要单独改默认并重跑。
7. **GRPO 符号**：旧 1434.33 是 cost 符号反了；已修。旧结果不得使用。
8. **GRPO 显著性**：n=3，配对 p≈0.28；需更多 seed 才能声称显著。
9. **GRPO 结构**：终端优势广播到所有 step、team advantage 共享、G=8；这三项未改。
10. **主模型集成**：E4b（RS + AGV）还没用 GRPO 跑；当前 GRPO 结果是 flat no-AGV。
11. **PPO 对照口径**：B15 是我们协议下的对照，不是论文 PPO 的复现。
12. **machine ST 默认切换**：2026-10-25 起默认 intra1。
    旧 checkpoint 必须显式 `--machine_intra_attn 0` 才能加载。
13. **E5 解释**：机内 attention 属于 encoder 家族的一部分。
    三臂的内部融合不同，不能把差异归给单一组件。
14. **intra 消融口径**：当前是 PPO + `hybrid_full` + LR 3e-4 + buffer 5。
    不是主模型 GRPO 口径；进论文前要重跑或标注。
15. **AGV 批次架构**：当前四臂钉 intra0。
    未来用 intra1 的 AGV run 不能与本批次直接比。
16. **10×6 结果**：B16 只有 1 个 seed。不能当最终结论。
17. **RGA 口径**：B19 未实现；染色体、解码器、种群、代数未定。
    实现后不能直接与论文的 RGA 数字并列。
18. **两端 vs 源端**：2026-10-25 起两端都由 actor 选。
    旧的 source-only AGV 读数只作 pilot，不得与新结果混。
19. **目标端时点**：目标 AGV 在派发时决定，用预计边界到达时间。
    不是实际到达后再决策。
20. **新批次算法**：新批次全部是 GRPO。
    没有同协议 PPO 对照。
21. **10×10 预算**：10×10 可能跑不完 200 ep。
22. **12 维 vs 33 维**：旧的 AGV 6×6 结果 582.59 是 12 维。
    新矩阵是 33 维。两批不得混。
23. **评测默认值**：评测脚本不显式传 ST 特征开关，依赖代码默认 1。
    评测旧 12 维 checkpoint 时必须显式传 0。

## 12. 现实约束扩展（C 线）

### 原则

- 主表 E0–E9 先保持当前 base，不混入约束。（E10 暂不做）
- 约束单独一条线；每次只加一个约束。
- 先做 **eval-only 扰动**：base 训练，约束测试。
- 有信号后，再训练带约束的模型。
- 不同约束不同时加，避免多变量混淆。

### C0：base

当前环境：

- 无机器故障
- 无 AGV 故障
- 无有限缓冲
- 无交期 / 维护

### C1：机器故障

- 事件模型：每台机器 MTBF / MTTR；只在加工中触发。
- 状态：故障标志、剩余修复时间、可用性。
- 动作：故障机器的 mask。
- 先 eval-only：base 模型在故障场景下测试。
- 再训练：主模型 + 故障；与 base 对照。
- 优先级：**高**。
- 状态：**代码已写；未训练、未跑正式读数。**

### C2：AGV 故障

- 事件模型：每台 AGV MTBF / MTTR；只在运输中触发。
- 状态：故障标志、剩余修复时间。
- 动作：故障 AGV 的 mask。
- 先 eval-only；再训练。
- 与 B（规则）/ C（actor）一起比；故障下 actor 的价值更容易显现。
- 优先级：**高**。
- 状态：**代码已写；未训练、未跑正式读数。**

### C3：有限机器缓冲区

- 机器输入 / 输出缓冲容量有限。
- 阻塞：上游工序无法投放；调度必须看缓冲占用。
- 状态：每台机器缓冲占用、阻塞标志。
- 动作：不可投放的工序 mask。
- 这是环境级改动，风险高；base 稳定后再做。
- 优先级：**中高**。
- 状态：**只加了 `--buffer_capacity` 参数；逻辑未实现。**

### C4：交期 / 维护（可选）

- 交期：加入 tardiness，目标从 4 目标变成 5 目标。
- 维护：计划维护窗口，动作 mask。
- 会改变目标口径，建议单独成文或放扩展。
- 优先级：**低 / 可选**。

### C 线指标

- base 目标 + 4 个原始 cost
- 故障场景下的目标退化
- 故障率 / 修复时间曲线
- 阻塞率、缓冲占用
- AGV 利用率、等待时间
- 约束违反次数（必须为 0）

### 口径边界

- base 与约束结果分表。
- 故障模型、随机种子、MTBF/MTTR 必须预登记。
- 不得把“约束下更好”与“base 更好”混成一句话。

## 13. 批量队列

脚本：

```text
plcsp_v5/run_queue.py
plcsp_v5/experiment-queue.json
```

### 远程启动（睡觉前）

```bash
cd /root/plcsp_v5
mkdir -p queue_logs
PYTHONPATH=/root/plcsp_v5/graph-jsp-env/src \
  nohup /root/venv/bin/python run_queue.py \
  --config experiment-queue.json \
  > queue_logs/queue.log 2>&1 &
```

### 查看

```bash
tail -f queue_logs/queue.log
cat queue_logs/status.json
```

### 队列行为

- 最多同时 `max_parallel` 个 run；当前配置为 6。
- GPU 空闲少于 3 GB 时暂停启动。
- 每个 job 一个日志：`queue_logs/<时间>_<id>.log`。
- 成功写 `queue_logs/<id>.done`。
- 失败按 `retries` 重试；默认 0。
- 支持 `after` 依赖：eval 在训练完成后才跑。
- `--dry-run` 只打印命令。
- `--only E4b_main,E4a_rule` 只跑指定 job。
- `--force` 忽略 `.done`。
- `--max-parallel` 覆盖并发数。

### 当前队列

| 配置 | 内容 | 状态 |
|---|---|---|
| `experiment-queue-rs-lambda2.json` | E1 / E4a / E4b × 3 seeds，18 维，alpha=0.5 | RS 最后跑 |
| `experiment-queue-st-ablation.json` | E5_jobst + E5b_intra0 × 3 seeds；E5_e2e 评测 | **跑中（17:01）** |
| `experiment-queue-e0-ppo.json` | E0 PPO × 3 seeds，settrans，33 维，无 AGV，无 RS | watcher 等 ST 结束后启动 |
| `experiment-queue-plan-matrix.json` | 78 job | **暂停** |

RS 重跑口径：

- 6×6×2
- 33 维
- GRPO + hier
- 200 ep
- 顶层段回报用当前 λ
- 两批口径：

| 批次 | 顶层状态 | alpha |
|---|---|---:|
| rslam | 13 维（10 + 3 λ） | 1.0 |
| rslam2 | 18 维（10 + 5 状态量 + 3 λ） | 0.5 |

matrix 暂停后未跑：

- E5_jobst
- E5b_intra0
- 10×6 四配置
- E5_e2e 的 3 个评测

等 RS 结果出来后再恢复。  
详见 `progress-log.md` §79。

## 顶层信用分配实验（2026-10-26）

RS 动作直方图显示：job / AGV 头固定，machine 头少量变化。  
问题：共享 team reward + 三个头没有 per-head 信用分配。

两种方案：

| 方案 | 顶层奖励 | 基线 | 参数 |
|---|---|---|---|
| `local` | 每个头自己的局部成本增量 | 每个头独立 value | `--top_credit local` |
| `coma` | 共享 team reward | joint-action Q-critic 反事实 | `--top_credit coma` |

局部成本：

- job：`mk + idleT`
- machine：`mean PEC + transT`
- AGV：`transT + 累计 wait`

实验：

| 项 | 值 |
|---|---|
| 臂 | E4b（3 头，AGV actor + 顶层先验） |
| 实例 | 6×6×2 |
| 种子 | 1 / 2 / 3 |
| episode | 200 |
| 评测 | seed3，100 实例 |
| 队列 | `experiment-queue-credit.json` |
| 启动 | `run_credit_after.sh`，等非 RS 队列结束 |

对照：E4b team 重跑 3 个 seed。  
旧 E4b team s1 = 564.48，用于核对代码口径。

详见 `progress-log.md` §113。

## 源 AGV 边界位置：建模近似（2026-10-26）

`§3` 已登记：

- 边界位置没有建模；
- 源 AGV 送到边界后 `location = -1`；
- 下一次取货的空驶近似为 0。

它是 Stage-0 的**有记录的建模近似**，不是未记录的代码缺陷。  
审查把它标成 bug；这个标签过重。

| 项目 | 状态 |
|---|---|
| E0 无 AGV | 不受影响 |
| E2/E3/E4 AGV 臂 | 在 Stage-0 模型下有效 |
| 10×6 E2/E3 | 同模型，可比较 |
| 旧 AGV 结果 vs 新边界模型 | 不可混用 |
| 采用新模型 | AGV 臂要重跑 |
| 沿用 Stage-0 | 旧结果可用，口径写明近似 |

本地 `src_boundary_machine` 修复已回退。  
本地与远程都保持 Stage-0 近似。

详见 `progress-log.md` §114。

## RS 状态特征与 eval trend 修复（2026-10-26）

第三轮审查发现：

1. `_agv_top_stats` 用 `makespan_previous_step` 当 `now`。
   - 它是全任务 horizon。
   - AGV busy / wait 恒为 0。
   - 顶层第 12 / 13 维是死特征。
2. 验证时 `cost_trend` 恒为 0。
   - 训练时是真实 trend。
   - eval 后段状态与训练不一致。

已修：

- `now` 改为“已调度节点最大完工时间”。
- `validate.py` 每段重算 trend。
- `hier_interval < 1` 直接报错。

影响：

- 旧 RS 读数（§106 / §112）只作历史记录。
- credit 实验的 team 对照用新跑的 3 seed。
- 旧 E4b s1 564.48 不与新结果直接比。
- RS 动作直方图要重跑。

详见 `progress-log.md` §115。

## 论文贡献（2026-10-26 17:10 更新）

> 本表是写作口径，不是结果表。  
> 这里的 C1–C4 是论文贡献编号，与 §12 约束线的 C1–C4 无关。

### 主贡献

| 编号 | 贡献 | 证据 | 状态 |
|---|---|---|---|
| C1 | 双 ST 编码器：机器内 m1/m2/m3 分组投影 + 机器内 / 机器间两级 attention | §89：6×6，GRPO，no-AGV，3 seed，3/3，比 job_st 低 89.50 | 强（6×6）；10×6 补齐中 |
| C2 | 低层多智能体 GRPO：三个 actor 共享组内相对优势，无 critic | 6×6 559.31 vs PPO 637.41；10×6 575.27 vs 605.14 | 强 |
| C3 | AGV actor：源端 + 目标端由同一 actor 选 | 6×6 −9.52（3/3）；10×6 −0.12（2/3） | 中 |
| C4 | 分层规则选择（RS） | 6×6 测试：RS 556.70 vs RS off 564.05；2/3；−7.35（1.3%） | **中（待补）** |

### C4：seed3 测试（100 实例，greedy）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| RS | 570.84 | 555.49 | 543.76 | **556.70** |
| RS off | 558.11 | 558.35 | 575.69 | 564.05 |

逐 seed：

| seed | RS | RS off | 差 |
|---|---:|---:|---:|
| s1 | 570.84 | 558.11 | +12.73 |
| s2 | 555.49 | 558.35 | −2.86 |
| s3 | 543.76 | 575.69 | −31.93 |

2 好 1 差。均值 −7.35（1.3%）。

### C4：当前状态

C4 当前证据中等，待补实验后定。

### C4：待补实验

| 实验 | 作用 | 状态 |
|---|---|---|
| RS 10×6 3 seed | 规模外推 | 排队，约 02:00 |
| 动作直方图 | 判断是选规则还是固定规则 | 未跑 |

### C4 的写法

不写「RS 提升性能」。
写「规则选择在 2/3 个 seed 上降低目标值约 1.3%」。

前提：

- RS 的顶层还在按状态变化。
- 如果 entropy ≈ 0，赢的是固定规则。

### 其他信用变体

team / coma / hybrid 都跑过，记录在 `progress-log.md` §120。
论文不写这三个变体，只报 local vs RS off。

### 不进贡献

| 项 | 去处 |
|---|---|
| AGV Stage-0 建模 | 问题定义 + 限制节 |
| 动态 λ / phase 权重 | 放弃（p≈0.14） |
| 约束线（机器故障等） | 代码有，未训练 |
| MIP-AGV 基线 | 实验节；不是贡献 |

### 基线口径更新（2026-10-26）

- 旧 MIP CSV 是弱 incumbent。
- 5 个实例上 CP-SAT 平均低 137.14。
- 旧读数 585.83 不能再用。
- 两阶段 MIP-AGV 的 5 实例均值 482.05；AGV 路径只带来 −2.25。
- 论文的 MIP 对照必须用新模型重跑。

## 待办（2026-10-27）

### 跨规模泛化：只训练一个模型

- 目的：验证单一策略能否跨实例规模泛化，而不是每个规模各训一个模型。
- 做法：混合 6×6×2 与 10×6×2 训练一个模型；直接测未见过的 10×10×2，
  以及更大规模（如 15×10×2）。
- 对照：现有“每个规模单独训练”的结果（论文 Table 4）。
- 指标：零样本目标值、与单独训练的差距、单实例求解时间。
- 参考：E2E Table VII 用 20×10×5 训练的模型直测 30×10×5 / 40×10×5。
- 状态：未开始。排在当前 10×10 训练、E2E 队列、MIP / B3 补齐之后。
- 备注：现有编码器无位置编码，token 数可变，具备跨规模迁移的结构条件。



