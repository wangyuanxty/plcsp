# INDEX — 项目索引入口

> **任何新会话从这里开始读。** 本文件是唯一入口：告诉你项目是什么、文档在哪、证据到哪一步、下一步做什么。

## 0. 工作规则（硬性）

1. **不在项目根目录留临时文件**——任何 agent/脚本的中间产物（`_xxx.txt`、`_pN.png`、提取文本、渲染图）一律放**系统临时目录**，用完即删。根目录只放正式文档。
2. **文件名一律英文**（kebab-case）；中文只出现在文件内容里。
3. **不自创术语**：描述已有概念用文献承认的词（并引用）；命名自己的方法须首次定义。详见 `progress-log.md` §12。
4. **搁置的线索必须登记**（`progress-log.md` 的"开放线索登记"节），不能只在对话里提一句。

## 1. 项目一句话

**生产-物流协同调度（PLCSP）的深度强化学习研究**：两环节联动 = 排产（FJSP）→ 物流（真 AGV 派车）；自建 SimPy 测试台 + Transformer 编码器 + 组内相对 RL（GRPO）；目标顶刊。
（2026-10-02 两处删减：① 原"**三环节** = 分批 → 排产 → 物流"——分批已砍，见 `progress-log.md` §12.5；② 原"**几何/度量感知注意力**编码器"——几何线已砍，见 §12.6。）

代码：`plcsp/`（Python 包）｜数据：`plcsp/*.jsonl`｜文献 PDF：`references/`（63 篇）｜模型：`checkpoints/`

## 2. 文档地图

| 文件 | 内容 | 谁在用 |
|---|---|---|
| **INDEX.md** | 本文件——索引入口 | 新会话第一站 |
| `method-design.md` | 方法设计（问题定义 §1.1、编码器 §2.3、算法 §3、环境 §4、实验计划 §5） | 写论文的主依据 |
| `feasibility-report.md` | 可行性分析、领域定位、文献对位表 | 立项依据 |
| `theory-bounds.md` | 理论骨架（LoTV 条件塔 / 有限样本界 v0.1） | 理论章节 |
| `literature.md` | 文献总表（编号 + 期刊 + 档次 + 定位） | 检索/引用 |
| `literature-landscape-2026.md` | **2026 风向标**（六条热点 / 必读论文 / 我方位置对照） | 选题定位 |
| `literature-plcsp-drl.md` | **PLCSP+DRL 全量清单**（核心 40 + 弱相关 24 + 缺口 19 篇 P0–P3 分级） | 找对位/找缺口 |
| `citation-cards.md` | **精读事实卡**（含公式、数字、自认局限、与我方逐点对比） | 写 related work / 找 gap |
| `progress-log.md` | 实验数字、bug 清单、负例、**开放线索 A–K**、术语纪律 | 断点续接 |
| `references/` | 63 篇 PDF + 2 个手动下载清单 | 精读 |

## 3. 当前阶段（学术 pipeline）

| Stage | 状态 |
|---|---|
| **1 RESEARCH** | ✅ 完成（文献三批精读 + 顶刊/顶会检索 + 代码库调研），**待用户确认** |
| 2 WRITE | ⏸ 未开始 |
| 2.5 → 6 | ⏸ 未开始 |

⚠️ **进入 Stage 2 前的阻塞项**：**全部实证结论目前仅在 MK01 单实例上**（10 作业 × 6 机）。见"证据状态"。

## 4. 证据状态（论文主张 vs 硬证据）

| 主张 | 现有证据 | 强度 |
|---|---|---|
| 组内相对 RL 优于规则 | flat GRPO 371.05 vs 规则 457.1（**-18.5%**，30 种子，MK01） | 🟡 单实例 |
| ~~几何/度量感知注意力 → 跨拓扑迁移~~ | **2026-10-02 已撤**：用户判断"顶会顶刊提都没提 = 无跨布局泛化需求"。自有数据亦弱——M7 显示机制仅在训练同型布局 line2 有效（−7.1%），留出的 U/岛式基本平局（369.5 vs 368.3；402.0 vs 402.0）。见 `progress-log.md` §12.6 | ⚫ 已撤 |
| ~~三环节交集的空白~~ | **2026-10-02 已删除**：分批环节砍掉后，"分批+排产+真 AGV 派车 三者齐备无先例"**不再成立**（且原论证用加工批文献支撑投放门控，属偷换概念）。详见 `progress-log.md` §12.5 | ⚫ 已撤 |
| ~~度量注入注意力无先例~~ | **2026-10-02 已撤**（随几何线一并砍）| ⚫ 已撤 |
| 树式分层 / 共演化 / 通道化多目标（+GRPO-λ） | **均为负例**（详见 progress-log.md §十三「负例补记」、§7、开放线索登记 §K） | ⚫ 不进论文 |

## 5. 待办与阻塞

**阻塞（做论文前必须补）**：多实例复现（MK02–MK10）——全部结论目前仅 MK01。

**高优先**：
1. 多实例复现 + 跨拓扑崩塌幅度测量（判天花板）
2. 对位基线：移植 `third_party/` 中 MIT 许可的基线（End-to-end-DRL-for-FJSP 等）
3. REINFORCE 消融（证明"组内相对"必要）
4. 走廊链简化升级（栅格 A*；触发条件=审稿人质疑）

**其余**：见 `progress-log.md` 开放线索 A–K。

## 5.5 P0 地基（2026-10-02 完成）

- **版本控制**：已 `git init` 并建仓（4 次提交）。`.gitignore` 排除 `third_party/`（1.5 G / 33 个嵌套 `.git`）、`references/`（410 M / 103 PDF）、`checkpoints/`（66 M）。**实际入库 71 个文件、`.git` 仅 805 K**。远程 `origin` = `https://github.com/wangyuanxty/plcsp.git`（**尚未 push**）。
- **包名**：`geosched` → **`plcsp`**（原名的 `geo` 随几何线砍除而失真）。历史/指令性文档（`progress-log.md`、spec、plan）**保留旧名不改**，以免篡改历史。
- **归档**（`plcsp/archive/`，**38 个文件**）：SA-GRPO 树 `standard_tree.py` + 6 个导入它的脚本 + `m5_*` 系列 + `m2/m3_smoke` + `m8_adv` + 全部 `*.jsonl`。**历史证据保留、不进论文**（spec §2）。
- **删除**：编码器 `GeomBias` 与 `dist`/`conf` 参数、`features.py`/`state_emb.py` 的几何上游、`PolicyNet` 的 batch 头、`des.py` 的 `batch_cap` 门控、`group_rel.py` 的 `tree_step`/`_mode_feat`（B 层树训练器）。**`run_gated` 保留**（L 层决策接口）。
- **旧 checkpoint**：`checkpoints/` 下 19 个 `.pt` **全部作废**（含已删的 `b_head.*`；且训练于 bug#12 的错误实例）。`resume_training` 已改为明确报错。
- **回归门禁**：`plcsp/tests/test_instances.py` **42 项全绿**（改名前后一致）。

> 🔴 **P0 暴露的关键发现（P2 必须处理）**：旧的 `encode_state` 产出的 token 特征，**除 4 位静态几何外全是桩 0**——动态量（缓冲占用/负载/电量/等待/占道）**从未接入过**。几何一删，编码器输入**全零**。
> **故 P2 不是"删掉几何就行"，而是必须重新设计 token 特征。**（已在 `nn/features.py`、`nn/state_emb.py` docstring 顶部写明。）

## 6. 环境

Windows 11｜Anaconda base（`D:\anaconda\python.exe`）｜Python 3.12.4｜**torch 2.14.0+cpu**（无 CUDA）｜simpy 4.1.2｜numpy 1.26.4｜matplotlib 3.11.0｜networkx 3.6.1
