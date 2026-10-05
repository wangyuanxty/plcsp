# 方法论文引证核查表（2026-10-05）

> **本文只回答一件事**：`method-transfer-candidates.md` §1 与 ⑩ 相关引文的完整出版信息是什么？
>
> **本文不修改** `docs/method-transfer-candidates.md`。是否据此更新该文，由该文作者决定。

## 0. 核查口径

| 记号 | 含义 |
|---|---|
| ✅ 已核 | 有出版社 / 会议官方记录直接支持该字段（DOI、PMLR 页、NeurIPS 元数据、arXiv 页、学会虚拟站页） |
| ⚠️ 待核 | 只有二手引文支持，或来源互相冲突 |
| 不适用 | 该 venue 没有此字段（ICLR 无卷页；RSS 按 DOI 单篇出版） |

规则：

1. 来源只列可点的链接。**"我记得"不算来源。**
2. 查不到就写「待核」，不补字。
3. 正式发表与 arXiv 预印本**分开记**。引文里能引正式版就引正式版；只能引预印本时，必须写「预印本」。
4. 我打开的每个页面都在下表的「来源」列里。打开的页面不含的字段，一律进 §3。

---

## 1. 总表

| # | 完整题名 | 作者 | venue（全名 · 卷 · 页） | 年 | 正式发表 | 来源（已打开） | 核定结果 |
|---|---|---|---|---|---|---|---|
| 1 | Constrained Policy Optimization | Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel | Proceedings of the 34th International Conference on Machine Learning（ICML 2017）· PMLR 70 · pp. 22–31 | 2017 | 是 | [PMLR](https://proceedings.mlr.press/v70/achiam17a.html) · [arXiv:1705.10528](https://arxiv.org/abs/1705.10528) | ✅ 已核 |
| 2 | Reward Constrained Policy Optimization | Chen Tessler, Daniel J. Mankowitz, Shie Mannor | 7th International Conference on Learning Representations（ICLR 2019）· poster · 不适用 | 2019 | 是 | [arXiv:1805.11074](https://arxiv.org/abs/1805.11074)（作者注 "Accepted as a poster to ICLR 2019"） | ✅ 已核（venue 由作者注确认） |
| 3 | Pointer Networks | Oriol Vinyals, Meire Fortunato, Navdeep Jaitly | Advances in Neural Information Processing Systems 28（NIPS 2015）· pp. 2692–2700 | 2015 | 是 | [NeurIPS 官网](https://papers.nips.cc/paper_files/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html) · 页码取自官方 [Metadata.json](https://papers.nips.cc/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Metadata.json) · [arXiv:1506.03134](https://arxiv.org/abs/1506.03134) | ✅ 已核 · ⚠️ 会议名见 §2.1 |
| 4 | Exploration by Random Network Distillation | Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov | 7th International Conference on Learning Representations（ICLR 2019）· poster · 不适用 | 2019 | 是 | [ICLR 2019 poster 页](https://iclr.cc/virtual/2019/poster/1093) · [arXiv:1810.12894](https://arxiv.org/abs/1810.12894) | ✅ 已核 |
| 5 | Gradient Surgery for Multi-Task Learning | Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn | Advances in Neural Information Processing Systems 33（NeurIPS 2020）· pp. 5824–5836 | 2020 | 是 | [NeurIPS 官网](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) · [arXiv:2001.06782](https://arxiv.org/abs/2001.06782) | ✅ 题名/作者/venue/年 · ⚠️ 页码待核 · ⚠️ 题名见 §2.3 |
| 6 | Multi-Task Learning as Multi-Objective Optimization | Ozan Sener, Vladlen Koltun | Advances in Neural Information Processing Systems 31（NeurIPS 2018）· pp. 527–538 | 2018 | 是 | [NeurIPS 官网](https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html) · 页码取自官方 [Metadata.json](https://papers.nips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Metadata.json) · [arXiv:1810.04650](https://arxiv.org/abs/1810.04650) | ✅ 已核 · ⚠️ 页码与常见二手引用冲突，见 §2.4 |
| 7 | Curriculum Learning | Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston | Proceedings of the 26th Annual International Conference on Machine Learning（ICML '09）· ACM · pp. 41–48 | 2009 | 是 | [DOI 10.1145/1553374.1553380](https://doi.org/10.1145/1553374.1553380)（Crossref 记录已核） | ✅ 已核 · ⚠️ 出版方见 §2.5 |
| 8 | Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping | Andrew Y. Ng, Daishi Harada, Stuart Russell | Proceedings of the Sixteenth International Conference on Machine Learning（ICML 1999）· Morgan Kaufmann · pp. 278–287 | 1999 | 是 | 无出版社在线记录。[课程存档 PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)（可下载，正文为压缩流，未能读出页码） | ✅ 题名/作者/venue/年 · ⚠️ 页码待核 · ⚠️ 无官方链接，见 §3.1 |
| 9 | POMO: Policy Optimization with Multiple Optima for Reinforcement Learning | Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, Seungjai Min | Advances in Neural Information Processing Systems 33（NeurIPS 2020）· pp. 21188–21198 | 2020 | 是 | [NeurIPS 官网](https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html) · [arXiv:2010.16011](https://arxiv.org/abs/2010.16011) | ✅ 题名/作者/venue/年 · ⚠️ 页码待核 |
| 10 | A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play | David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis | Science · vol. 362 · issue 6419 · pp. 1140–1144 | 2018 | 是 | [DOI 10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)（Crossref 记录已核） | ✅ 已核 · ⚠️ 题名与预印本见 §2.2 |
| 11 | Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks | Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, Yee Whye Teh | Proceedings of the 36th International Conference on Machine Learning（ICML 2019）· PMLR 97 · pp. 3744–3753 | 2019 | 是 | [PMLR](https://proceedings.mlr.press/v97/lee19d.html) · [arXiv:1810.00825](https://arxiv.org/abs/1810.00825) | ✅ 已核 |
| 12 | Perceiver: General Perception with Iterative Attention | Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, Joao Carreira | Proceedings of the 38th International Conference on Machine Learning（ICML 2021）· PMLR 139 · pp. 4651–4664 | 2021 | 是 | [PMLR](https://proceedings.mlr.press/v139/jaegle21a.html) · [arXiv:2103.03206](https://arxiv.org/abs/2103.03206) | ✅ 已核 |
| 13 | Conservative Q-Learning for Offline Reinforcement Learning | Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine | Advances in Neural Information Processing Systems 33（NeurIPS 2020）· pp. 1179–1191 | 2020 | 是 | [NeurIPS 官网](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html) · [arXiv:2006.04779](https://arxiv.org/abs/2006.04779) | ✅ 题名/作者/venue/年 · ⚠️ 页码待核 |
| 14 | Offline Reinforcement Learning with Implicit Q-Learning | Ilya Kostrikov, Ashvin Nair, Sergey Levine | The Tenth International Conference on Learning Representations（ICLR 2022）· poster · 不适用 | 2022 | 是 | [arXiv:2110.06169](https://arxiv.org/abs/2110.06169) · OpenReview `68n2s9ZJWF8`（两处二手来源一致，未直接打开） | ✅ 已核 |
| 15 | Decision Transformer: Reinforcement Learning via Sequence Modeling | Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch | Advances in Neural Information Processing Systems 34（NeurIPS 2021）· pp. 15084–15097 | 2021 | 是 | [NeurIPS 官网](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html) · [arXiv:2106.01345](https://arxiv.org/abs/2106.01345) | ✅ 题名/作者/venue/年 · ⚠️ 页码待核 |
| 16 | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song | Robotics: Science and Systems XIX（RSS 2023）· DOI 10.15607/RSS.2023.XIX.026 · 不适用（按 DOI 单篇出版，无页码） | 2023 | 是 | [DOI 10.15607/RSS.2023.XIX.026](https://doi.org/10.15607/RSS.2023.XIX.026)（Crossref 记录已核）· [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | ✅ 已核 |
| 17 | Multi-Agent Reinforcement Learning is a Sequence Modeling Problem | Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, Yaodong Yang | Advances in Neural Information Processing Systems 35（NeurIPS 2022）· Main Conference Track · pp. 16509–16521 · DOI 10.52202/068431-1201 | 2022 | 是 | [NeurIPS 官网](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract.html) · [DOI](https://doi.org/10.52202/068431-1201)（Crossref 记录已核）· [arXiv:2205.14953](https://arxiv.org/abs/2205.14953) | ✅ 已核 |
| 18 | QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning | Tabish Rashid, Mikayel Samvelyan, Christian Schröder de Witt, Gregory Farquhar, Jakob N. Foerster, Shimon Whiteson | Proceedings of the 35th International Conference on Machine Learning（ICML 2018）· PMLR 80 · pp. 4295–4304 | 2018 | 是 | [PMLR](https://proceedings.mlr.press/v80/rashid18a.html) · [arXiv:1803.11485](https://arxiv.org/abs/1803.11485) | ✅ 已核 · ⚠️ 页码有错值流传，见 §2.6 |
| 19 | The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games | Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu | Advances in Neural Information Processing Systems 35（NeurIPS 2022）· **Datasets and Benchmarks Track** · pp. 24611–24624 · DOI 10.52202/068431-1787 | 2022 | 是 | [NeurIPS 官网（D&B）](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html) · [DOI](https://doi.org/10.52202/068431-1787)（Crossref 记录已核）· [arXiv:2103.01955](https://arxiv.org/abs/2103.01955) | ✅ 已核 · ⚠️ track 与题名见 §2.2 |
| 20 | A History-Guided Regional Partitioning Evolutionary Optimization for Solving the Flexible Job Shop Problem with Limited Multi-load Automated Guided Vehicles | Feige Liu, Chao Lu, Xin Li | **arXiv 预印本** arXiv:2409.18742（cs.SY，交叉 cs.NE）· 14 页 · 未查到正式发表 | 2024 | **否（预印本）** | [arXiv:2409.18742](https://arxiv.org/abs/2409.18742) | ⚠️ 待核（正式发表版本未知）· ⚠️ 容量 2/3 无来源，见 §3.4 |
| 21 | Integrated Optimization of Automated Warehouse Operations and Last-Mile Transport for Differentiated On-Demand Delivery | Xiaozhu Sun, Bilal Farooq | **arXiv 预印本** arXiv:2609.19048（cs.LG）· 未查到正式发表 | 2026 | **否（预印本）** | [arXiv:2609.19048](https://arxiv.org/abs/2609.19048) | ⚠️ 待核（描述不符，见 §2.7；v1 日期矛盾，见 §3.5） |

---

## 2. 与我原来写的不一致的地方（逐条）

### 2.1 第 3 条：2015 年不能写 "NeurIPS"

我写 "NeurIPS 2015"。**2015 年该会议的正式名是 NIPS**（Advances in Neural Information Processing Systems 28）。
改名发生在 2018 年（NeurIPS 2018 起）。引文写 "NeurIPS 2015" 会被看作年代错误。
**改法**：写 `Advances in Neural Information Processing Systems 28 (NIPS 2015), pp. 2692–2700`。

### 2.2 第 19 条（MAPPO）：不是主会 track，题名也不叫 MAPPO

- **track**：我写 "NeurIPS 2022"。实际是 **NeurIPS 2022 Datasets and Benchmarks Track**。
  官方页明确标 "Datasets and Benchmarks"。**这是本仓点名要防的「track 当成主会」同型问题。**
- **题名**：正式题名是 **"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"**。
  arXiv v4 的题名（[2103.01955](https://arxiv.org/abs/2103.01955)）是 "…of **MAPPO** in Cooperative, Multi-Agent Games"。
  两个版本题名不同。正式引用按正式题名写。
- 卷页（若需）：Advances in Neural Information Processing Systems 35, pp. 24611–24624, DOI 10.52202/068431-1787。

### 2.3 第 5 条（PCGrad）：题名里没有 "PCGrad"

正式题名是 **"Gradient Surgery for Multi-Task Learning"**。PCGrad 是方法名，不是题名。
引文不能写 "PCGrad — Yu et al."，要写题名。页码 5824–5836 只有二手来源，**标待核**。

### 2.4 第 6 条（MGDA）：页码官方记录与常见引用冲突

- 官方元数据（papers.nips.cc 的 Metadata.json）：`page_first: 527`、`page_last: 538`。
- 我见到的多数二手引用写 **525–536**。
- **按官方记录写 527–538**，或标待核。这条正好说明：二手引用一致也可能整体错。

### 2.5 第 7 条（Curriculum Learning）：出版方不是 PMLR

ICML 2009 的正式出版方是 **ACM**（Proceedings of the 26th Annual International Conference on Machine Learning, pp. 41–48, DOI 10.1145/1553374.1553380）。
本仓其它 ICML 引文走 PMLR（PMLR 从 2013 年起收录 ICML）。**2009 年没有 PMLR 卷号，不能补。**

### 2.6 第 18 条（QMIX）：页码有错值流传

PMLR 官方页写 **4295–4304**。检索中见到二手引用写 4292–4301（错）。**按 PMLR 官方写。**

### 2.7 第 21 条（arXiv 2609.19048）：描述不符

我写「多容量 AGV」。**该文不是这个内容。**

- 该文题目：Integrated Optimization of Automated Warehouse Operations and Last-Mile Transport for Differentiated On-Demand Delivery。
- 内容：AGV 仓库作业 + 最后一公里运输的联合优化（DRL）。多容量/异构容量出现在**最后一公里的车辆路径问题**（MRMH-HCVRP）里，**不是 AGV 的多载容量**。
- **⟹ 不能把它当作「多容量 AGV」的文献依据。**

### 2.8 通例：方法是简称，不是题名

原文表里多处以方法简称代替题名。写进论文时，**题名要写全**。对照如下（只列不一致的）：

| 原文写法 | 正式题名 |
|---|---|
| CPO | Constrained Policy Optimization |
| RCPO | Reward Constrained Policy Optimization |
| RND | Exploration by Random Network Distillation |
| PCGrad | Gradient Surgery for Multi-Task Learning |
| MGDA | Multi-Task Learning as Multi-Objective Optimization |
| 势能奖励塑形 | Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping |
| POMO | POMO: Policy Optimization with Multiple Optima for Reinforcement Learning |
| AlphaZero | A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play |
| Set Transformer | Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks |
| Perceiver | Perceiver: General Perception with Iterative Attention |
| CQL | Conservative Q-Learning for Offline Reinforcement Learning |
| IQL | Offline Reinforcement Learning with Implicit Q-Learning |
| Decision Transformer | Decision Transformer: Reinforcement Learning via Sequence Modeling |
| Diffusion Policy | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion |
| MAT | Multi-Agent Reinforcement Learning is a Sequence Modeling Problem |
| QMIX | QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning |
| MAPPO | The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games |

一致的两条：第 1 条 CPO 与第 3 条 Pointer Networks（题名与简称相同）。
注意：**AlphaZero 的预印本与正式版题名完全不同**，不是同一题名的两个版本。

### 2.9 版本对照（该引哪一个）

| # | 正式版 | 预印本 | 引哪一个 |
|---|---|---|---|
| 1 | PMLR 70:22–31 | arXiv:1705.10528 | 正式版 |
| 2 | ICLR 2019 | arXiv:1805.11074 | 正式版 |
| 3 | NIPS 2015, 2692–2700 | arXiv:1506.03134 | 正式版 |
| 4 | ICLR 2019 | arXiv:1810.12894 | 正式版 |
| 5 | NeurIPS 33 | arXiv:2001.06782 | 正式版 |
| 6 | NeurIPS 31 | arXiv:1810.04650 | 正式版 |
| 7 | ICML '09 (ACM) | 无 | 正式版 |
| 8 | ICML 1999 | 无 | 正式版 |
| 9 | NeurIPS 33 | arXiv:2010.16011 | 正式版 |
| 10 | Science 362(6419) | arXiv:1712.01815（**另一题名**） | 正式版 |
| 11 | PMLR 97 | arXiv:1810.00825 | 正式版 |
| 12 | PMLR 139 | arXiv:2103.03206 | 正式版 |
| 13 | NeurIPS 33 | arXiv:2006.04779 | 正式版 |
| 14 | ICLR 2022 | arXiv:2110.06169 | 正式版 |
| 15 | NeurIPS 34 | arXiv:2106.01345 | 正式版 |
| 16 | RSS XIX | arXiv:2303.04137 | 正式版 |
| 17 | NeurIPS 35 | arXiv:2205.14953 | 正式版 |
| 18 | PMLR 80 | arXiv:1803.11485 | 正式版 |
| 19 | NeurIPS 35 (D&B) | arXiv:2103.01955（**题名不同**） | 正式版，且写明 D&B Track |
| 20 | 未查到 | arXiv:2409.18742 | **只能引预印本，标「预印本」** |
| 21 | 未查到 | arXiv:2609.19048 | **只能引预印本，标「预印本」** |

---

## 3. 查不到的（列清楚）

### 3.1 第 8 条：Morgan Kaufmann 1999 无出版社在线记录

ICML 1999 论文集**没有 DOI，也没有出版社在线页**。可点的链接只有作者/课程存档 PDF。
**页码 278–287 只有二手书目来源支持 ⟹ 标待核。** 引用时至少写 `ICML 1999, pp. 278–287`，并接受这一不确定性。

### 3.2 三处 NeurIPS 页码：官方不列页码

NeurIPS 官网**不显示页码**；2020/2021 年的 Metadata.json 不存在（返回 404）。
以下页码**只有二手来源**，一律标待核：
第 5 条 5824–5836 · 第 9 条 21188–21198 · 第 13 条 1179–1191 · 第 15 条 15084–15097。

对照（有官方记录、已核）：第 3 条（2015 Metadata.json）、第 6 条（2018 Metadata.json）、
第 17 条与第 19 条（Curran DOI，Crossref 已核）。

### 3.3 第 20 条（HRPEO）：未查到正式发表

检索**只找到 arXiv 预印本**（arXiv:2409.18742, 2024-09-27, 14 页）。未见期刊/会议版本。
⟹ 写进论文时必须标「预印本」。

### 3.4 第 20 条的「AGV 容量 2/3」：无来源

我写「容量 2/3」。**摘要、索引页、评论页都没有这个数**。全文 PDF 超出抓取上限，未能读到。
⟹ **标待核。** 用之前请读原文（arXiv:2409.18742）确认。

### 3.5 第 21 条的 v1 日期：与编号矛盾

arXiv:2609.19048 的编号月份是 **2026 年 9 月**，但 API 与 abs 页都记 v1 为 **2026-07-21T18:56:08Z**。
两者矛盾，**原因未查明 ⟹ 标待核**。引用时写「预印本」即可，日期先不用。

### 3.6 OpenReview 页面未能直接打开

OpenReview 有浏览器验证拦截，无法直接打开。第 2、4、14 条的 OpenReview 链接
**只由二手来源确认**（第 2 条另有两处一致来源；第 4 条与第 14 条各有会议官网/数据集页佐证）。
其 venue 与年已由其它来源核实，**引用不受影响**。

### 3.7 第 10 条：预印本不是同一篇的「另一版本题名」

AlphaZero 的 arXiv:1712.01815 题名是 "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"，
与 Science 正刊题名不同。**这是预印本，不是同一题名的两版。** 引正式版即可。

---

## 4. 一句话结论

- 第 1–19 条：**全部有正式发表版本，全部可点来源**；其中 4 处页码待核（§3.2），1 处页码按官方元数据改（第 6 条）。
- 第 20、21 条：**只有 arXiv 预印本**，且第 21 条的内容与原文描述不符。
- 最需要改的三处：**第 19 条的 track 与题名**、**第 3 条的 NIPS 会议名**、**第 21 条的内容描述**。
