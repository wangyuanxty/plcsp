# 2026 文献风向标（PLCSP / 调度 DRL）

> 生成：2026-10（检索自 IEEE Xplore / ScienceDirect / 各会议 proceedings）。**DOI 多为检索所得，投稿引用前需二次核验。**
> 相关：`literature.md`（总表）、`citation-cards.md`（精读卡片）、`progress-log.md`（我方进展）

## 一、六条热点（期刊 + 顶会 合看）

| # | 趋势 | 期刊侧代表 | 顶会侧代表 |
|---|---|---|---|
| 1 | **异构图/图注意力 + 多策略 PPO 成事实标准**（工序/机器/AGV 建异构节点、统一动作空间同步决策）| HGA-MPPO(SWEVO 102:102331)、HCHGNN-PPO(ESWA 334:134345，卷年 2027／DOI 年 2026)、DFJSPHT(IEEE TSMC-Sys 56(5):3086-3098)、TII 22(4):2863-2874 | — |
| 2 | **表征极简化 + Transformer 取代 GNN** | — | RESCHED(ICLR'26)、RL-SPH(ICML'26) |
| 3 | **多目标/偏好条件化**（偏好向量作虚拟节点注入注意力，一次出整条 Pareto）| — | DCAN(ICLR'26) |
| 4 | **约束现实主义**：有限缓冲/死锁、充电、多行程载重、机械臂退化、**工人学习/遗忘曲线** | HCHGNN-PPO（有限缓冲/死锁）、C&OR 189:107409（机械臂退化；原文自证 "Charging of the AMRs is not considered."）、**充电** = Applied Sciences 15(13):6995（三级充电动作 + 充电桩选择）、EJOR 332(3):730-747（最优充电策略闭式解）、C&IE 214:111882（工人学习/遗忘曲线）| — |
| 5 | **RL 与精确求解器/元启发式混合**（纯端到端构造式降温）| HRLMA(EAAI 167)、LLM-MOMA(SWEVO) | RL-SPH、PUMA(KDD'26)、branch-and-bound RL(AAAI'26) |
| 6 | **LLM/Agentic AI 进入调度** | **JMS 2026 四篇**（完整标题见下）| PathWise(ICML'26)、G-STAR(KDD'26)、IDP-MCTS(ICML Workshop) |

### 六之附：JMS 2026 LLM/Agentic 四篇（完整引用，2026-10-01 核验）

| 简称 | 完整标题 | 作者 | 出处 | DOI |
|---|---|---|---|---|
| **LLM-OWA** | Multitask fine-tuning agentic AI based collaborative scheduling for flexible manufacturing systems | Gao, Gu & Ji | JMS 2026（PII S027861252600107X）| 待补 |
| **MARS** | MARS: Multi-Agent Rescheduling Framework with Large Language Models for Human-Centric Manufacturing | Park, Kim, Faturrahman & Kim（KAIST）| JMS **88:903-928**, 2026-10 | 10.1016/j.jmsy.2026.07.015 |
| **A4PS**（原误记作"4PS"）| A4PS: Agentic AI-assisted advanced planning and scheduling with large language models for smart manufacturing | Li 等 | JMS **85:207-226**, 2026 | 10.1016/j.jmsy.2026.01.003 |
| **LEA** | Coordination of LLM-Embodied Agents for manufacturing task allocation under constraints | Ye & Li | JMS **87:17-26**, 2026 | 10.1016/j.jmsy.2026.04.028 |

- **LLM-OWA** = **LLM-based Orchestrator-Workers Agents**：LeaderAgent（计划层）+ 三个执行层 agent — **ProdAgent（机床）/ TransAgent（AGV）/ AuxAgent（AR 工人）**；ISLLM 为认知核；MKF 知识织物 + 多任务微调（核心参数冻结 + DARE）。33 个基准上稳定性 +18.06%、解质量 +3.03%（vs NSGA-II/III）。
- **MARS**：两 agent（Schedule modifier 输出可执行 Python 代码 + Explainer 因果链解释）；LLaMA-3.1-8B 本地跑，不依赖闭源 API；非相关并行机问题。
- **LEA**：把 LLM 认知嵌进**物理资源本体**，RLEA（机器人）/ VLEA（AGV）；去中心化两种协同策略 RLC vs VLC；拆解场景 VLC 约束满足率 **100% vs RLC 63%**（GPT-4o），token −41%、API 调用 −68%。

## 二、结构性发现

> **"生产 + 运输/AGV 联合调度" 在 2026 顶会 = 基本空白；全部发在期刊。**

- 顶会最接近的三篇均非真联合：AAMAS'26《Multi-Agent Cooperative Transportation》（仅运输侧）、IJCAI'26《From Gridworlds to Warehouses》（仅仓库 AGV 路径）、ICML'26 RL-SPH（把生产/配送当**并列独立基准**）
- 期刊侧 2026 有 23 篇该方向新作
- **解读**：该方向的"联合建模"叙事目前由 OR/工业工程期刊承载，顶会审稿人接受度低；若要投顶会需换成方法学叙事
- ⚠️ **NeurIPS 2026 录用名单尚未公布**；网上流传的泄露名单自相矛盾，**不可引用**

## 三、必读的 2026 论文（前 6）

| 排 | 论文 | 出处 | DOI | 为什么 |
|---|---|---|---|---|
| 1 | **DFJSPHT**：Distributed FJSP with Heterogeneous Transportation | IEEE TSMC-Sys 56(5):3086-3098, 2026 | 10.1109/TSMC.2026.3656196 | 唯一把**跨厂物流**写成 MDP 的顶刊长文，问题定义最新 |
| 2 | **HGA-MPPO**：异构图表注意力 + 多策略 PPO 的 AGV 辅助 FJSP | SWEVO 102:102331, 2026 | 10.1016/j.swevo.2026.102331 | 与 TAAGNet 最同构的**强基线** |
| 3 | **HCHGNN-PPO**：hub-centric 图 RL 的 AGV 派工 + 并行机调度 | ESWA 334:134345（**卷年 2027**；DOI 年 2026，冲突已记录于 `citation-cards.md:3190`）| 10.1016/j.eswa.2026.134345 | 有限缓冲/**死锁** + 动作掩码（约束加强版模板）|
| 4 | **Production-logistics cooperative scheduling**（装配 + AMR，含机械臂退化）| C&OR 189:107409, 2026 | 10.1016/j.cor.2026.107409 | 标题即"production-logistics cooperative"，方向最正面 |
| 5 | **ATLAS**：Alibaba 学习增强调度数据集与基准 | ICLR 2026 | — | **真实生产数据 + 非全知评测协议**（实验必用）· `github.com/zhiyunjiang0810/non-clairvoyant-with-predictions` |
| 6 | **MACSIM**：Multi-Action Self-Improvement for NCO | ICLR 2026 | — | FJSP/FFSP 最强神经求解器之一，**必对比基线** · `github.com/LTluttmann/macsim` |

**备选**：EJOR 332(3):730-747（柔性装配线 + 搬运机器人联合调度，MIP+拉格朗日）；C&IE 214:111882（工人+AGV，含学习/遗忘曲线）；EAAI 167（有限运输资源：AGV+天车，节能动态 FJSP）

## 四、其他 2026 已发现的论文（简录）

**期刊**：JFMS 联合调度(EJOR 332(3))｜装配+AGV+AMR matheuristic(EJOR, in press)｜Transformer-MARL for FJSP-AGV(ASOC)｜Dynamic FJSP+AGV DRL(FGCS)｜MACD 连续动态 FJSP(IEEE TASE 23:10574-10586)｜多视图图注意力(TSMC-Sys)｜Multiagent Transformer(TCYB 56(5))｜分层双缓存调参(TCYB 56(10))｜DT 扰动识别+自适应调度(RCIM 101:103323)｜HFS+预防性维护(RCIM 97:103085)｜DT+DRL 动态 AGV(IJPR 64(1):106-124)｜ND3QN-PER 多行程多 AGV(AEI)

**顶会**：Instance-wise Adaptive Scheduling via Derivative-Free Meta-Learning(ICLR'26, `github.com/calmQ/DF-META`)｜DEFT 云工作流 MoE(ICLR'26)｜RRNCO 真实路网(ICLR'26)｜FrontierCO 基准(ICLR'26)｜EoH-S 启发式集演化(AAAI'26)｜EvoReal LLM 实例生成(AAAI'26)｜RulePlanner 设计规则→掩码(ICML'26, `github.com/Thinklab-SJTU/EDA-AI`)｜DynaSchedBench(ICML'26)｜**JSSP+运输 联合 vs 模块化协调缺口分析**(IEEE CASE'26, arXiv:2604.24117, `github.com/proto-lab-ro/jsspt-coordination-gap`)

## 五、我方位置对照

| 我方主张 | 2026 风向 | 判断 |
|---|---|---|
| 用 DRL 做 PLCSP | 期刊极卷（华中科大李新宇组一家占 3-4 篇）| ⚠️ 红海 |
| 真 AGV 派车 | 2026 已成标配 | ⚠️ 不再是新意 |
| **几何/度量感知注意力** | **已收窄**：Zhu & Peng 2026（*Algorithms* 19(4):289，RGV 轨道分 **17 段**、实时占用作边特征进 DQN）、GRAND（arXiv:2512.03194, IEEE RA-L，区段占用 + 走廊负载进派工状态）、Graphormer（NeurIPS 2021，沿最短路平均边特征）、**RRNCO**（ICLR 2026，距离偏置注入注意力）均有先例（GNNT 原文自证排除距离；TAAGNet 无成对几何项）| ⚠️ **收窄到粒度差异**：单条通道段作可独立注意力的 token |
| **三环节联合**（投放 + 排产 + 派车）| **有在刊先例（Moon 2024）**：JMS 77:356-367 已用一个策略网络 + 复合动作 (工序, 机台, 车辆) 联合决策 | ⚠️ 卖点改为**测量**联合 vs 模块化，**不是首创** |
| 证据体量 | 2026 论文普遍多规模 + 真实数据（ATLAS 级）| ❌ **我方仅 MK01 单实例 = 最大短板** |
