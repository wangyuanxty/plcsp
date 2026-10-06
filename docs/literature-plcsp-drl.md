# PLCSP + DRL 文献清单（期刊侧）

> 生成日期：2026-10
> 注：标注"未核实"的条目引用前需二次核对

## 检索范围

- **时间**：2023–2026（重点 2025–2026）
- **期刊**：IEEE TII / TASE / TSMC-Systems / TCYB / TNNLS / IoT-J；Journal of Manufacturing Systems；RCIM；EJOR；Omega；IJPR；Computers & OR；ESWA；Advanced Engineering Informatics；SWEVO；Computers & Industrial Engineering；Engineering Applications of AI；Applied Soft Computing；Robotics and CIM；International Journal of Production Economics
- **检索词**：PLCSP / production-logistics collaborative scheduling / production-logistics integrated scheduling；integrated scheduling of machines and AGVs / simultaneous scheduling of machines and AGVs；FJSP + AGV + reinforcement learning / DFJSP-T / FJSP with transportation；AGV dispatching + machine scheduling + deep reinforcement learning；production and material handling joint scheduling RL；中文侧：生产-物流协同调度 / 机器-AGV 集成调度
- **本地库存交叉**："本地已有？"一列依据 `references/` 目录下 **103 篇 PDF** 的文件名判断（2026-10-06 实测；标题大致对上即算有）

---

## 表一：⭐ 核心 PLCSP+DRL（明确同时调度机器与运输资源，41 篇）

| # | 标题 | 期刊 | 年 | 方法 | AGV派车 | 本地已有？ |
|---|---|---|---|---|---|---|
| 1 | Production-logistics collaborative scheduling in dynamic flexible job shops using nested-hierarchical DRL | Adv. Eng. Informatics 65B:103195 | 2025 | 嵌套分层 MARL/MAPPO | 是 | ✅ AEI2025_ProductionLogistics_NestedHierarchical_DRL |
| 2 | Real-time scheduling for production-logistics collaborative environment using MADRL | Adv. Eng. Informatics 65:103216 | 2025 | 三智能体 MAPPO | 是 | ✅ AEI2025_ProductionLogistics_MADRL |
| 3 | Real-Time Scheduling for Flexible Job Shop With AGVs Using MARL and Efficient Action Decoding | IEEE TSMC-Systems 55(3):2120-2132 | 2025 | MARL（任务/机器/AGV） | 是 | ✅ TSMC2025_RealtimeFJSP_AGV_MARL |
| 4 | TAAGNet: graph-based MARL for integrated production and AGV scheduling in dynamic HFS | Expert Systems with Applications 306:130683 | 2026 | GNN-MARL（机器智能体+AGV智能体） | 是 | ✅ ESWA2026_TAAGNet_ProductionAGV_MARL |
| 5 | A cooperative agent DRL framework for FJSP with AGVs (CADRL) | ESWA 287:128142 | 2025 | GNN + 协作双智能体 | 是（规则） | ❌ |
| 6 | Hierarchical MARL for dynamic flexible job-shop scheduling with transportation | IJPR（在线优先：本地 PDF 只印 DOI 与 "Published online: 03 Jun 2025"，**未分配卷/期/文章号**） | 2025 | 分层 MARL（3 层）+模仿学习 | 是 | ✅ IJPR2025_HierarchicalMARL_DFJSPT |
| 7 | Integrated scheduling of multi-objective lot-streaming hybrid flowshop with AGV based on DRL | IJPR 63(4):1275-1303 | 2025 | NSGA2-MDDQN 多目标 DRL | 是 | ✅ IJPR2025_LotStreaming_HFS_AGV_DRL |
| 8 | DRL for solving efficient and energy-saving FJSP with multi-AGV | Computers & OR 181:107087 | 2025 | DQN + 复合规则 | 是 | ✅ COR2025_EnergySaving_FJSP_MultiAGV_DRL |
| 9 | Transformer-based MARL for flexible job shop scheduling with AGVs (MA-Trans) | Applied Soft Computing 193 | 2026 | Transformer-MARL（Job+AGV 双智能体） | 是 | ❌ |
| 10 | HGA-MPPO: unified heterogeneous graph attention and multi-policy PPO for AGV-assisted FJSP | SWEVO 102:102331 | 2026 | 异构图注意力 + 多策略 PPO | 是 | ✅ SWEVO2026_HGA_MPPO_AGV_FJSP |
| 11 | Hub-centric Heterogeneous Graph RL for Integrated AGV Dispatching and Unrelated Parallel Machine Scheduling | ESWA 334:134345（**卷年 2027**）| 2026（**DOI 年**；卷年/DOI 年冲突，见 `citation-cards.md:3190`）| HCHGNN + PPO | 是 | ✅ ESWA2026_HCHGNN_PPO |
| 12 | Hierarchical collaborative scheduling of workers and AGVs for DT-based distributed FJSP | C&IE 214:111882 | 2026 | 分层 MARL + DRN-DQN | 是（含工人） | ✅ CIE2026_WorkerAGV_Hierarchical_DFJSP |
| 13 | Distributed FJSP With Heterogeneous Transportation Resources Constraints via DRL and GNN | IEEE TSMC-Systems 56(5):3086-3098 | 2026 | GNN + DRL | 是 | ✅ Distributed_FJSP_Heterogeneous_Transportation |
| 14 | DRL-based memetic algorithm for energy-aware FJSP with multi-AGV | C&IE 189:109917 | 2024 | DQN + 模因算法 | 是 | ❌ |
| 15 | Dynamic scheduling for flexible job shop with insufficient transportation resources via GNN and DRL | C&IE 186:109718 | 2023 | 异构图神经网络 + DRL | 是 | ❌ |
| 16 | Dynamic Integrated Scheduling of Production Equipment and AGVs in FJSP based on DRL | Processes 12(11):2423 | 2024 | QMIX MARL（四智能体） | 是 | ❌ |
| 17 | DRL for dynamic scheduling in distributed heterogeneous FJSP with integrated inventory allocation and product delivery | EAAI 2025:112681 | 2025 | DRL | 部分（配送） | ❌ |
| 18 | A DRL optimization algorithm based on heterogeneous GNN for HFS with finite transportation resources | EAAI 161:112096 | 2025 | HGNN + DRL | 是 | ❌ |
| 19 | A flexible job shop scheduling method based on heterogeneous disjunctive graph and DRL | EAAI 2025:111356 | 2025 | PPO，O-M-A 三元组动作 | 是 | ❌ |
| 20 | Integrated process planning and scheduling considering AGVs with improved DQN | EAAI 171:114306 | 2026 | 改进 DQN | 是（IPPS+AGV） | ❌ |
| 21 | Distributed heterogeneous FJSP considering AGV transportation via improved DQN | SWEVO 94:101902 | 2025 | DQN + 组合调度规则 | 是 | ❌ |
| 22 | Energy-aware flexible open shop scheduling with multi-load AGV by graph RL assisted memetic | ESWA 314:131650 | 2026 | GAT + DQN + 模因 | 是（多载量） | ❌ |
| 23 | Dynamic flexible scheduling with transportation constraints by MARL | EAAI 134:108699 | 2024 | MARL | 是 | ❌ |
| 24 | Solving Collaborative Scheduling of Production and Logistics via DRL: Limited Transportation Resources and Charging Constraints | Applied Sciences 15(13):6995 | 2025 | 改进 PPO（CRGPPO-TKL） | 是（含充电） | ✅ |
| 25 | Green FJSP considering transportation time and machine multi-rotation speeds | SWEVO（卷期未核实） | 2025 | D3QN + 分层动作 | 是（AGV 路由） | ✅ Green flexible job-shop...pdf |
| 26 | Production-logistics cooperative scheduling（装配 + AMR） | Computers & OR 189:107409 | 2026 | **问题特定启发式（非 DRL）**（题名即 "a problem-specific heuristic"） | 是 | ✅ COR2026_ProductionLogistics_AMR |
| 27 | Integrated scheduling of material delivery and processing（FFSPDP） | C&IE 201:110863 | 2025 | DQN 超启发式（HH-DQN） | 是（送料车） | ❌ |
| 28 | Hybrid RL-assisted memetic algorithm for energy-efficient dynamic FJSP with limited transportation resources（HRLMA） | EAAI 2026:113866（未核实） | 2026 | RL + 模因 | 是 | ✅ EAAI2026_HRLMA_EnergyEfficient_DFJSP_AGV |
| 29 | A heuristic-assisted DRL algorithm for FJSP with transport constraints（HA-DQN） | Complex & Intelligent Systems | 2025 | DQN + 启发式（机器/AGV 规则） | 是（规则） | ✅ A heuristic-assisted...pdf |
| 30 | Digital twin driven dynamic scheduling of discrete manufacturing workshop with transportation resource constraint using MADRL | RCIM 95:103042 | 2025 | MADRL（多智能体 MAPPO-MC） | 是 | ✅ |
| 31 | Matrix manufacturing system layout and scheduling via GNN and multi-action DRL | JMS 82:239-253 | 2025 | GNN + 多动作 DRL | 疑（矩阵制造含 AGV） | ✅ JMS2025_MatrixManufacturing |
| 32 | Manufacturing resource-based self-organizing scheduling using MAS and DRL | JMS 79:179-198 | 2025 | MAS + MADRL（CTDE） | 部分（AGV 用启发式投标） | ❌ |
| 33 | FJSP considering AGV transport time using PPO with graph isomorphism network | Flexible Services and Manufacturing J. | 2026 | PPO + GIN | 仅运输时间 | ❌ |
| 34 | Integrated Scheduling of Multi-Objective Job Shops and Material Handling Robots with RL Guided Meta-Heuristics | Mathematics 13(1):102 | 2025 | Q-learning/SARSA + 元启发式 | 是（MHR） | ❌ |
| 35 | Learning-driven memetic algorithm for integrated distributed production and transportation scheduling | SWEVO | 2025 | RL 驱动模因算法 | 厂际运输 | ❌ |
| 36 | A joint scheduling approach for production and material handling under customized manufacturing paradigm | EJOR 332(3):730-747 | 2026 | JFMS（**非 DRL，未核实**） | 是（MHR） | ✅ EJOR2026_JointScheduling |
| 37 | Digital-twin-based AGV cluster dynamic scheduling for solar cell production workshop using DRL | Neurocomputing | 2025 | SAC/ISAC | 是（AGV 集群） | ❌ |
| 38 | An Intelligent Multi-Layer Control Architecture for Logistics Operations of Autonomous Vehicles in Manufacturing | IEEE TASE 2025:7296-7311 | 2025 | Petri 网 + DRL + MPC | 是 | ❌ |
| 39 | Digital twin-driven DRL driven by heuristics with Petri nets for real-time scheduling in robotic job shops | RCIM 97:103097 | 2026 | 启发式驱动 DRL + Petri 网 | 机器人（加工+搬运） | ✅ RCIM2026_PetriNet_DRL_RoboticJobShop |
| 40 | Constraint Programming for AGV and Machine Integrated Scheduling Problem in FMS | IEEE TASE 23:2378-2390 | 2026 | CP（**非 DRL**） | 是 | ✅ TASE2026_AGVMachine_IntegratedScheduling_CP |
| 41 | Learning-enabled flexible job-shop scheduling for scalable smart manufacturing（HGS）| JMS 77:356-367 | 2024 | **DRL 方法（非元启发式）**：单策略网络，复合动作 (工序, 机台, 车辆)，整条 episode 一个 log-prob，REINFORCE + 贪心 rollout 基线，**无 critic** | 是（车辆是动作的一维） | ✅ arXiv2402.08979_LearningEnabled_FJSP_SmartManufacturing.pdf |

**注**：#41 原表漏收，但 #10（HGA-MPPO）与 #29（HA-DQN）都以 HGS 为对比基线；它是本方向标准 DRL 基线，也是"三环节联合"的在刊先例（2026-10-06 文献审计补录）。

---

## 表二：弱相关（只做运输时间 / 无车 / 纯物流 / 非 DRL，24 篇）

| # | 标题 | 期刊 | 年 | 方法 | AGV派车 | 本地已有？ |
|---|---|---|---|---|---|---|
| W1 | A Self-Attention-Based DRL Approach for AGV Dispatching Systems | IEEE TNNLS 35(6):7911-7922 | 2024 | 自注意力 DRL | 是（仅派车） | ❌ |
| W2 | Dispatching AGVs With Battery Constraints Using DRL | C&IE 187:109678 | 2024 | DRL | 是（仅派车） | ❌ |
| W3 | Automated guided vehicle dispatching and routing integration via DT with DRL | JMS 72:492-503 | 2024 | DRL | 是（派车+路径） | ❌ |
| W4 | A Novel Mathematical Model for the FJSP With Limited AGVs | IEEE TASE 2025:7449-7462 | 2025 | MILP（非 DRL） | 是 | ❌ |
| W5 | A Knowledge-Driven Cooperative Coevolutionary Algorithm for Integrated Distributed Production and Transportation Scheduling | IEEE TASE 2025:7435-7448 | 2025 | 协同进化（非 DRL） | 是 | ❌ |
| W6 | An adaptive parallel evolutionary algorithm for AGV Scheduling Problem | IEEE TASE 2025:7361-7372 | 2025 | 进化算法（非 DRL） | 是 | ❌ |
| W7 | Digital twin-driven real-time collaborative scheduling for U-shaped automated container terminals | IJPR 63(24):10765-10790 | 2025 | PPO（码头，无机床） | 是（码头 AGV） | ✅ |
| W8 | A Reinforcement Learning Framework for Efficient Task Allocation Among AGVs in Smart Warehouse | IEEE IoT-J 12(11):16947-16961 | 2025 | DRL 编解码 + 异质注意力 | 是（仓储，无机床） | ❌ |
| W9 | Collaborative Transmission and Computation for Distributed AGV Systems: Transformer-Based MADRL | IEEE IoT-J 12(18):38113-38124 | 2025 | MADRL（通信卸载） | 否（通信侧） | ❌ |
| W10 | DRL-based dynamic integrated scheduling of AGVs and yard cranes for container terminal | EAAI 163:112912 | 2026 | RGCN-DRL | 是（码头，无机床） | ❌ |
| W11 | Reinforcement learning for joint scheduling in robotic mobile fulfillment systems | EJOR | 2026 | PPO（仓储机器人） | 是（仓储，无机床） | ❌ |
| W12 | Flexible Manufacturing Systems intralogistics: AGVs and tool sharing using CT-PN and actor-critic RL | JMS 82:405-419 | 2025 | Actor-Critic + 动作掩码 | 是（无机床调度） | ❌ |
| W13 | Energy-efficient and self-adaptive AGV scheduling based on hierarchical RL | C&IE | 2025 | 分层 RL | 是（仅 AGV） | ❌ |
| W14 | Multi-agent DRL for dynamic FJSP under uncertain processing and transport times | ESWA 270:126441 | 2025 | MARL | 仅运输时间 | ❌ |
| W15 | Multi-agent collaborative RL for dynamic FJSP under machine random failures | C&OR 196:107644 | 2026 | MARL（无 AGV） | 否 | ✅ |
| W16 | A Heterogeneous Graph RL Framework for Dynamic FJSP (TII) | IEEE TII, DOI 10.1109/TII.2025.3646962 | 2026 | 图 RL（无 AGV） | 否 | ❌ |
| W17 | A Platoon-Based Approach for AGV Scheduling and Trajectory Planning | IEEE TII 21(1):594-603 | 2025 | 非 DRL | 是（仅 AGV） | ❌ |
| W18 | A Multiagent Transformer-Based Algorithm for Multitask Dynamic Scheduling With Constrained Machines | IEEE TCYB 56(5) | 2026 | MARL-Transformer | 否 | ✅ |
| W19 | A hybrid GNN-Transformer architecture for AGV scheduling in Automated Container Terminals | C&IE 222:112334 | 2026 | GNN-Transformer + REINFORCE | 是（码头，无机床） | ✅ |
| W20 | Hierarchical agent architecture-based large-scale AGV cluster real-time motion collaboration control | ESWA 296 | 2026 | 分层智能体（运动控制） | 是（运动控制） | ❌ |
| W21 | Graph-Enhanced Actor-Critic for FJSP and AGV with Charging Constraints | Springer LNCS（**会议/书章**） | 2026 | GAT + PPO-Clip | 是 | ❌ |
| W22 | A Multi-Head DRL-Model with Spatial Pyramid Pooling for Smart Manufacturing Scheduling with Insufficient Transportation | IEEE（**会议**） | 2026 | CNN + PPO 多头 | 是 | ❌ |
| W23 | A Digital Twin-driven DRL Approach for Smart Workshop Scheduling | 未核实 | 未核实 | DRL（无 AGV） | 否 | ✅ |
| W24 | RL-Based Production Scheduling in an Industry-Based Coating Scenario | 未核实 | 未核实 | DRL（无 AGV） | 否 | ✅ |

---

## 表三：缺口清单（表一中"本地已有=❌"的条目，按补入优先级排序）

| 优先级 | # | 标题 | 期刊/年 | 为什么值得补 |
|---|---|---|---|---|
| P0 | 5 | A cooperative agent DRL framework for FJSP with AGVs (CADRL) | ESWA 287:128142, 2025 | GNN + 协作双智能体的直接同源竞争工作，与本地 HCHGNN/TAAGNet 路线高度重叠；不补则相关工作对比缺一块 |
| P0 | 9 | Transformer-based MARL for FJSP with AGVs (MA-Trans) | Applied Soft Computing 193, 2026 | 2026 最新架构范式（Transformer 骨干 + Job/AGV 双智能体解耦），报告方法趋势时的关键新证据 |
| P0 | 15 | Dynamic scheduling for FJSP with insufficient transportation resources via GNN and DRL | C&IE 186:109718, 2023 | DFJSP-ITR 方向的奠基作，被后续大量文献引用为 baseline；时间窗内最早之一 |
| P1 | 14 | DRL-based memetic algorithm for energy-aware FJSP with multi-AGV | C&IE 189:109917, 2024 | 被引 87 次，DQN+模因混合的代表作；能耗目标侧的标准对照 |
| P1 | 16 | Dynamic Integrated Scheduling of Production Equipment and AGVs in FJSP based on DRL | Processes 12(11):2423, 2024 | QMIX 四智能体（工件/机器/AGV/目标）建模，MARL 信用分配的典型实现，被引 28+ |
| P1 | 23 | Dynamic flexible scheduling with transportation constraints by MARL | EAAI 134:108699, 2024 | 运输约束 + MARL 的早期代表作，是"运输约束"子线的起点 |
| P1 | 19 | A flexible job shop scheduling method based on heterogeneous disjunctive graph and DRL | EAAI 2025:111356, 2025 | O-M-A（工序-机器-AGV）三元组动作空间设计，动作空间构造的直接参考 |
| P2 | 18 | DRL optimization algorithm based on heterogeneous GNN for HFS with finite transportation resources | EAAI 161:112096, 2025 | 混合流水车间 + 有限运输资源，与 TAAGNet（同为 HFS）形成同题对照 |
| P2 | 21 | Distributed heterogeneous FJSP considering AGV transportation via improved DQN | SWEVO 94:101902, 2025 | 分布式异构 FJSP-AGV，DQN+组合调度规则，拓展"分布式"维度的对照 |
| P2 | 27 | Integrated scheduling of material delivery and processing（FFSPDP） | C&IE 201:110863, 2025 | DQN 超启发式（HH-DQN）路线，与纯 policy network 方法形成方法论对照 |
| P2 | 22 | Energy-aware flexible open shop scheduling with multi-load AGV by graph RL assisted memetic | ESWA 314:131650, 2026 | 多载量 AGV + 图强化学习 + 模因，2026 新约束（多载量）代表 |
| P3 | 20 | Integrated process planning and scheduling considering AGVs with improved DQN | EAAI 171:114306, 2026 | IPPS 与 AGV 联合，问题边界外扩（工艺规划层） |
| P3 | 17 | DRL for dynamic scheduling in DHFJSP with integrated inventory allocation and product delivery | EAAI 2025:112681, 2025 | 库存分配 + 产品配送，把"物流"从车间内扩到车间外 |
| P3 | 32 | Manufacturing resource-based self-organizing scheduling using MAS and DRL | JMS 79:179-198, 2025 | JMS 顶刊，MAS+CTDE 框架；但 AGV 侧用启发式投标（DRL 只覆盖机器侧），价值主要在对照 |
| P3 | 33 | FJSP considering AGV transport time using PPO with graph isomorphism network | Flexible Services and Manufacturing J., 2026 | PPO + GIN 组合；但仅建模运输时间（无派车决策），作为弱对照 |
| P3 | 34 | Integrated Scheduling of Multi-Objective Job Shops and Material Handling Robots with RL Guided Meta-Heuristics | Mathematics 13(1):102, 2025 | RL 只用于选邻域算子（非端到端策略），方法论边界案例 |
| P3 | 35 | Learning-driven memetic algorithm for integrated distributed production and transportation scheduling | SWEVO, 2025 | RL 驱动模因算法 + 厂际运输 + 能耗，方法路线差异较大 |
| P3 | 37 | DT-based AGV cluster dynamic scheduling for solar cell production workshop using DRL | Neurocomputing, 2025 | 数字孪生 + AGV 集群；但仅 AGV 侧调度，无机器协同 |
| P3 | 38 | An Intelligent Multi-Layer Control Architecture for Logistics Operations of Autonomous Vehicles | IEEE TASE 2025:7296-7311, 2025 | Petri 网 + DRL + MPC 三层架构，系统架构设计参考；但生产侧耦合弱 |

**本地已有且属核心的（无需补）**：#1 #2 #3 #4 #6 #7 #8 #10 #11 #12 #13 #24 #25 #26 #28 #29 #30 #31 #36 #39 #40 #41

---

## 结论

### 1. 顶刊集中度

本方向期刊集中度很高，**ESWA、Advanced Engineering Informatics、IJPR、Computers & OR / Computers & Industrial Engineering、SWEVO** 是绝对主力：

- **ESWA**：4 篇（TAAGNet、HCHGNN-PPO、CADRL、多载量 AGV 开放车间），且 2026 年仍高频产出
- **Advanced Engineering Informatics**：2 篇（#1 #2），均为 2025 年 PLCSP 方向的标志性工作
- **IJPR**：2 篇核心（#6 #7）+ 1 篇弱相关（W7）
- **C&OR / C&IE**：核心 4 篇（#8 #14 #15 #27），弱相关 4 篇（W2 W4?/W5? 见下）
- **SWEVO**：核心 3 篇（#10 #21 #35）
- **EAAI**：核心 5 篇（#17 #18 #19 #20 #23 #28），是数量最多的单一期刊
- **JMS / RCIM**：各 2 篇（#31 #32 #39，W3 W12），顶刊但本方向命中率低于上述

### 2. 空白刊（重要负面结论）

- **IEEE TII / TCYB / TNNLS / IoT-J 几乎没有严格意义上的 PLCSP+DRL**：
  - TII：仅检索到一篇图 RL 做动态 FJSP（W16，**无 AGV**）；另一篇 AGV 调度+轨迹规划（W17）**非 DRL**
  - TNNLS：仅 AGV 派车（W1，2024），不含机器调度
  - IoT-J：两篇（W8 W9）分别是仓储 AGV 任务分配与 AGV 通信卸载，**均无机床协同调度**
  - TCYB：仅检索到"受限机器多任务动态调度"（W18），**无 AGV**
- **EJOR / Omega 基本空白**：EJOR 仅检索到仓储机器人履约系统联合调度（W11，2026）与柔性装配线+搬运机器人联合调度（#36，**疑非 DRL**）；**Omega 未检索到任何 RL+AGV 调度条目**
- 若要投 IEEE 系或 EJOR/Omega，上述空白既是机会也是风险（审稿人池与选题匹配度低）

### 3. 方法趋势

1. **异构图 / 析取图表征 + GNN/GAT 编码已成标准配置**：几乎 2025–2026 所有核心工作都采用（TAAGNet、HCHGNN、CADRL、#19 O-M-A 三元组、#10 #22 等）
2. **多智能体解耦是主流架构范式**：普遍把"机器侧决策"与"AGV 侧决策"拆成独立智能体（机器智能体 + AGV 智能体），共享或各自编码骨干，再用贡献分解奖励处理信用分配（#4 #6 #9 #12 #16）
3. **训练算法以 PPO / MAPPO 占绝对主导**：DQN / D3QN / DDQN 多与模因算法或超启发式混合使用（#8 #14 #21 #27），纯 value-based 端到端方案在 2025 后明显减少
4. **2026 年出现 Transformer 骨干**（#9 MA-Trans）、**充电 / 有限缓冲 / 有限运输资源等约束建模**显著增多（#24 #11 #12）

### 4. 未核实项清单（引用前必须二次核对）

| 位置 | 未核实内容 |
|---|---|
| 表一 #25 | SWEVO 的卷号/文章号未核实 |
| 表一 #28 | EAAI 文章号 113866 未核实 |
| 表一 #36 | 是否含 DRL 成分未核实（疑为纯邻域搜索） |
| 表一 #37 | Neurocomputing 卷期未核实 |
| 表一 #40 | 收录年份标注为 2026（TASE vol.23），需核对 |
| 表二 W23 / W24 | 期刊名未核实 |
| 表一 #13 #31 #40 | 是否含真实"AGV 派车决策"（多为"疑"）需核对原文 |

> **2026-10-06 回填（依 `paper/refs.bib`，逐条已核）**：#13 → IEEE TSMC-S 56(5):3086-3098, 2026；#26 → 问题特定启发式（非 DRL）；#30 → RCIM 95:103042, 2025；W15 → C&OR 196:107644, 2026；W19 → C&IE 222:112334, 2026。

---

## 附：检索来源

- https://www.sciencedirect.com/science/article/abs/pii/S1474034625000886 （AEI 2025 嵌套分层 DRL）
- https://www.sciencedirect.com/science/article/abs/pii/S1474034625001090 （AEI 2025 PLCSP MADRL）
- https://dl.acm.org/doi/10.1016/j.eswa.2025.130683 （TAAGNet, ESWA 306:130683）
- https://www.semanticscholar.org/paper/ab14563718903d35449ebf8469f1bcdbe518c877 （IEEE TSMC-Systems 2025 MARL+AGV）
- https://ouci.dntb.gov.ua/works/7PMM3kdx/ （CADRL, ESWA 287:128142）
- https://www.sciencedirect.com/science/article/abs/pii/S2210650226000519 （HGA-MPPO, SWEVO）
- https://www.sciencedirect.com/science/article/abs/pii/S0957417426032513 （HCHGNN-PPO, ESWA）
- https://www.sciencedirect.com/science/article/abs/pii/S1568494626003479 （MA-Trans, Applied Soft Computing）
- https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2373426 （IJPR Lot-streaming HFS AGV）
- https://www.sciencedirect.com/science/article/abs/pii/S036083522400038X （C&IE 189:109917 DQNMA）
- https://acm-stag.literatumonline.com/doi/10.1016/j.cie.2023.109718 （C&IE 186:109718 DFJSP-ITR）
- https://dl.acm.org/doi/abs/10.1016/j.cie.2026.111882 （C&IE 2026 Worker-AGV）
- https://acm-stag.literatumonline.com/doi/10.1016/j.cie.2025.110863 （C&IE 201:110863 material delivery）
- https://eurekamag.com/research/109/527/109527684.php （EJOR 2026 RMFS joint scheduling）
- https://dl.acm.org/doi/10.1016/j.engappai.2025.112096 （EAAI 161:112096 HGNN）
- https://dl.acm.org/doi/10.1016/j.engappai.2025.111356 （EAAI 111356 heterogeneous disjunctive graph）
- https://www.ebiotrade.com/newsf/2025-10/20251025084000768.htm （EAAI 163:112912 AGV+yard crane）
- https://pubmed.ncbi.nlm.nih.gov/36449577/ （TNNLS 2022/2024 AGV dispatching）
- https://ui.adsabs.harvard.edu/abs/2025IITJ...1216947L/abstract （IEEE IoT-J AGV task allocation）
- https://ieeexplore.ieee.org/abstract/document/11062477 （IEEE IoT-J Transformer MADRL AGV）
- https://gpbib.pmacs.upenn.edu/gp-html/Li_2025_jmsy.html （JMS 79:179-198 MAS+DRL）
- https://www.sciencedirect.com/science/article/abs/pii/S0736584525001632 （RCIM 97:103097 Yi & Luo）
- https://www.mendeley.com/catalogue/98b8f8b6-943c-3b76-bafc-d8b9f2ce704b/ （Applied Sciences 15(13):6995）
- 本地库存核对来源：`references/` 目录 **103 篇 PDF** 文件名（2026-10-06 实测），及 `manual-download-journal.txt` / `manual-download-2026.txt` / `manual-download-conference.txt`
