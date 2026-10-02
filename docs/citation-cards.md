# Citation Cards（引用卡库）

> 合并自原「citation-cards.md / 批1-2 / 批2 / 批3 / 批4」（按精读时间分批，非主题）；主题导航见 INDEX.md。

### 引用卡库 · 批1 · 组结构谱系（facts 卡）

- 精读日期：2026-09-04 ｜ 方法：PyMuPDF 全文提取 + 逐页精读（4 篇全部读完：BranchGRPO 25/25 页、Order-Invariant 49/49 页、U-Statistic 53/53 页、EBPO 18/18 页）
- 配套：《method-design.md》§3（SA-GRPO）已先读；SA-GRPO = 按 π=π_B·π_S·π_L 层级条件采样 + 层内组相对优势归一化（LoTV 剔除 Var(E[A|层级])），无 critic；四条件 (i) 层级独立采样 (ii) 按层归一化 (iii) 各层条件策略非退化 (iv) 等预算。

> ⚠️ **状态（2026-09-29）**：批1 四张卡的"与我们的逐点对比"节，其对比骨架均建立在"SA-GRPO = 层级条件采样 + 层内归一化"之上；**该机制已被 M6 实证否决**（树 372.5 ≈ flat 371.05，无增益，见 progress-log §7）。下述各卡中的**事实、公式与原文摘录仍有效**，仅"结论/威胁判定"部分作历史参考。

---

## 《BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models》
（arXiv:2509.06040v5 [cs.CV] 29 Sep 2025；literature.md 编号 [23]，标注 ICLR 2026 已收录）

### 精确贡献
1. **结构化分支树 rollout**：把扩散/流模型的"每轨迹独立顺序 rollout"重构为树——根噪声 z0 共享，在指定分支步骤 B 注入相关噪声把单轨迹扩展为 K 个相关子轨迹（共享前缀、后分叉），**摊销计算同时保留探索多样性**（其动机段原文："replaces inefficient independent sequential rollouts with a branching structure"）。
2. **奖励融合 + 逐深度独立归一化（depth-wise advantage）**：叶子奖励按路径概率融合（Eq.3）向上传播为内部节点值，再按深度 d 做 z-score 标准化（Eq.4），把稀疏 terminal reward 变成"稠密 step-level 信号"，改进 credit assignment。
3. **宽度/深度剪枝**：只作用于反传（backprop），不动 rollout 与奖励评估；宽度剪枝（Parent-Top1 / Extreme-b）与深度剪枝（滑窗调度）。

### 关键公式
- 分支噪声（Eq.2/附录 Eq.7，s 为分支相关度）：
  ξ_b = (ξ0 + s·η_b)/√(1+s²)，ξ_b ~ N(0,I)，Cov(ξ_b,ξ_b′) = 1/(1+s²)·I（b≠b′）；每个子轨迹 z_(i+1)^(b) = μ_θ(z_i,t_i) + g_{t_i}·√h_i·ξ_b。
  **Lemma 1（单步边缘保持）/ Lemma 2（叶子边缘保持）/ Theorem 1（边界分布不变性）**："each leaf z_N^(b) has the same marginal law as a baseline SDE rollout. Hence, branching does not alter the final generator distribution." —— 分支不改变生成分布，只改变计算组织。
- 奖励融合（Eq.3，内部节点 n，后代叶 L(n)）：r̄(n) = Σ_{ℓ∈L(n)} w_ℓ^(n) r_ℓ，w_ℓ^(n) = exp(β·s_ℓ)/Σ_j exp(β·s_j)，s_ℓ = log p_beh(ℓ|n)。β=0 退化为均匀平均；β=1 为按行为策略路径概率加权（附录 B.2 证明：均匀融合无偏、Var=σ²(n)/|L(n)|；自归一化 IS 方差 O(1/ESS)；正文 softmax 版本"no longer strictly unbiased, this form provides stable training"）。
- **逐深度归一化（Eq.4）——它自己的"层内归一化"**：
  A_d(n) = ( r̄(n) − μ_d ) / (σ_d + ε)，μ_d = mean_{n∈N_d} r̄(n)，σ_d = std_{n∈N_d} r̄(n)，
  其中 N_d = 所有处于深度 d 的节点。理由原文："Nodes at the same depth share the same noise level and are thus directly comparable, while rewards across depths vary drastically due to changing noise states."
- 目标（Eq.5）：J(θ) = E[ (1/|E|)·Σ_{e∈E} min( ρ_e(θ)A(e), clip(ρ_e(θ),1−ε,1+ε)A(e) ) ]，E = 行为树上所有边（transition），ρ_e(θ) = π_θ(a_t|s_t)/π_old(a_t|s_t)。
- **其方差理论依据（附录 B.3，标题"DEPTH-WISE BASELINE (CONTROL VARIATES)"）**：A_i^(b) = r̄^(b) − r̄_i（第 i 深度 K 个兄弟组内相对），则 ∇J_group = (1/K)Σ_b A_i^(b)g_i^(b) "is an unbiased gradient estimator with **strictly smaller variance** than ∇J_single = (1/K)Σ_b r̄^(b)g_i^(b)"，除非 Cov(r̄^(b), g_i^(b)) = 0。→ 其主打理论依据确系**方差缩减 + 无偏性**（控制变量论证），另有融合无偏性、集中界（B.4 次高斯型 Pr(|r̄(n)−V(n)|≥ε|n) ≤ 2exp(−c|L(n)|ε²/(L²K_i²g_i²))）。

### 实验设置
- 基准：HPSv2.1 图像对齐（数据集 103k 训练 prompt / 400 平衡测试 prompt），主干 FLUX.1-Dev；基线 DanceGRPO、MixGRPO（同一设置）；奖励模型 HPS-v2.1 / PickScore / ImageReward / Unified Reward。视频：Wan2.1-1.3B，Video-Align motion quality 奖励。
- 树参数：深度 d=20，分支因子 K=2 → 每 rollout 16 叶子；分支步三预设 Dense (0,3,6,9)（默认）、Mixed (0,4,8,12)、Sparse (0,5,10,15)；s ∈ {1,2,4,8}（最优 s=4）。
- 训练预算：300 optimizer steps；grad accum 12；per-GPU batch 2；16× NVIDIA H200；AdamW lr 1e-5、wd 1e-4；bf16、EMA decay 0.995；分辨率 720×720；采样步 16；eta 0.3；adv clip max 5.0。全程 GRPO 超参三方法一致。

### 核心数字（Table 1 / Table 3 / §4.4–4.6）
| 方法 | NFE_old | NFE_new | 单轮迭代时长(s) | HPS-v2.1 | Pick | ImageReward | Unified |
|---|---|---|---|---|---|---|---|
| FLUX 基座 | – | – | – | 0.313 | 0.227 | 1.112 | 3.370 |
| DanceGRPO(tf=1.0) | 20 | 20 | 698 | 0.360 | 0.234 | 1.612 | 3.388 |
| DanceGRPO(tf=0.6) | 20 | 12 | 469 | 0.353 | 0.228 | 1.517 | 3.362 |
| MixGRPO(20,5) | 20 | 5 | 289 | 0.359 | 0.228 | 1.594 | 3.380 |
| BranchGRPO | 13.68 | 13.68 | 493 | 0.363 | 0.229 | 1.603 | 3.386 |
| BranchGRPO-WidPru | 13.68 | 8.625 | 314 | 0.364 | 0.231 | 1.609 | 3.383 |
| BranchGRPO-DepPru | 13.68 | 8.625 | 314 | **0.369** | **0.235** | **1.625** | **3.404** |
| BranchGRPO-Mix | 13.68 | 4.25 | **148** | 0.363 | 0.230 | 1.598 | 3.384 |

- 摘要声称"improves alignment scores by up to **16%** over DanceGRPO"；但表末值：0.369 vs 0.360 = **+0.009（+2.5%）**【存疑：16% 应指训练前中期（Fig.1 早期曲线 2.2× 收敛速度）或另一指标下的增益，正文 Table1 未给出 16% 出处，引用时需按"最终对齐 +2.5%、收敛加速 2.2×"表述，避免笔误引 16%】。
- 训练单轮 **−55%**：698s → 314s（DepPru/WidPru，=−55.0%）；**4.7×**：698s/148s = 4.72×（BranchGRPO-Mix）。视频：每轮约 8 分钟 vs DanceGRPO 20 分钟（≈2.6×）；vBench 表（500 样本）：迭代 493s vs 1352s，Motion Smoothness 0.9912 vs 0.9899。
- 可扩展性：DanceGRPO 单步 81 rollout 样本 >3500s，BranchGRPO 同规模 680s（5.2×）；K=2,3,4 → 16/81/256 叶子，"larger group sizes consistently lead to better alignment performance"（scaling law 声明，Fig.6）。
- 多样性无损证据：Inception 空间 KID=0.0057、MMD²=0.0067；CLIP 空间 KID=0.00022、MMD²=0.0149（结论："two distributions are almost indistinguishable at the semantic level"）。

### 自认局限
- 附录 D Discussion（原文）："reward fusion provides stable gradients in practice, but its **bias–variance tradeoff under different weighting schemes warrants further theoretical analysis**."——融合加权的偏差-方差权衡缺乏理论。
- Future Work 自列四项：动态分支（自适应分支因子/相关度/剪枝窗）、扩散以外范式、长高清视频需要更多验证（目前仅 WanX-1.3B I2V）、机器人动作生成。
- 附录 C.4 有专门 Failure Cases 图（无量化失败率）。
- 训练/评估均未做统计显著性（单次训练为主；预训练 FLUX 基座质量影响起点）。

### 与我们的逐点对比（SA-GRPO 层级采样 + 层内归一化 + LoTV）

> ⚠️ 本节的对比骨架建立在"SA-GRPO = 层级条件采样 + 层内归一化"之上，该机制已被 M6 否决（见 progress-log §7）。事实与摘录仍有效，结论部分仅作历史参考。

1. **同**：① 都在"组结构"上做分层（stratify）：它把组按**去噪深度**分片做逐深度 z-score，我们把组按**决策层级因果链**（π_B·π_S·π_L）分片做层内相对归一化——机制同一族（组内相对优势 + 分层归一化），它在《literature.md》中被列为"与 SA-GRPO 同属组结构×层级优势设计空间，必对比"，成立。② 都无 critic、用组内相对优势；③ 都以"方差缩减"为理论立场（它的 B.3 控制变量定理 vs 我们的 LoTV 定理）；④ 它也有"共享前缀保持边缘分布不变"（Theorem 1）——与我们"层级独立采样下条件边缘不变"是可类比的结构性质。
2. **异**：① **分层维度不同（关键判定）**：它的"层级/深度"是**生成时间维**（denoising depth，同一噪声水平才可比——引文见上）；我们的"层级"是**决策层面的因果链**（分批→排产→AGV 物流的异质笛卡尔积动作空间的内部结构）。它的归一化单位是"同一噪声水平下的树节点"，我们的是"同一父决策条件（b、s）下的兄弟决策链"。② 它的逐深度归一化是**树内节点级**（跨分支、跨父节点聚合 N_d），我们的层内归一化是**父条件内**（每一条件组独立成组；组内条件独立）。③ 它的"分层"是结构效率手段（摊销+剪枝），我们的是**方差-归因手段**（LoTV 剔除层间共享分量）；它没有 LoTV/层间方差分解这一论证，只做单层控制变量+融合无偏。④ 它的奖励融合引入了额外偏差项（softmax 加权非严格无偏），并自认偏倚-方差权衡未解；我们无融合步骤。⑤ 它的组内对比发生在"奖励聚合后"，我们发生在"结算奖励后"（不聚合，直接组内归一）。
3. **是否威胁我们的 claims**：**不威胁，反而是支持性前例**。① 其"逐深度独立归一化"表明"分片归一化优于整片广播"（它对 GRPO"prompt-level normalization, which broadcasts a single terminal reward"的批评与我们批评均匀组内归一化异曲同工）；② 它证明分片归一化是**无偏 + 严格方差更小**的——与我们的定理方向一致，可在 Related Work 引用为"时间维分片版本"；③ 需防范的审稿问题是"你与 BranchGRPO 的区别只是分层维度不同"——回答口径：**正交性论证**——时间维/深度归一化解决的是"同一噪声水平可比性"，因果链层级归一化解决的是"异质联合动作空间不可比性 + 层级间共享分量"（我们的难度维度①③）；两者可组合（理论上我们的 θ 也是深度维共享的）；名称上它叫"depth-wise"，我们叫"hierarchy-wise/conditional group"，不冲突。
4. **可直接引用的句子**：
   - "This per-depth standardization prevents late denoising steps with smaller variance from dominating, yielding process-dense and balanced credit signals."（逐深度标准化防晚期小方差步主导——与我们的"层内小 σ 组不再被层间大 σ 淹没"同构）
   - "Compared to GRPO's prompt-level normalization, which broadcasts a single terminal reward, our scheme produces stable gradients and finer credit assignment..."（对均匀广播式归一化的批评——支持我们"层内归一化"的必要性叙事）
   - "∇J_group … is an unbiased gradient estimator with strictly smaller variance than ∇J_single …"（无偏 + 严格方差更小——与我们定理同构的另一实例）

---

## 《Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning》
（ICML 2026, PMLR 306, 43rd ICML, Seoul；作者 Goudet, Suire, Goëffon, Saubion, Lamprier（LERIA, Université d'Angers）；literature.md 编号 [7]）

### 精确贡献
1. **神经网络参数化的多元 EDA**：用"每变量一个条件分布网络"的多元自回归生成模型替代经典 EDA（MIMIC/BOA）的显式变量依赖 DAG——省去每代求解 NP-hard 结构学习，参数规模 O(n·m)（多项式于实例规模）。
2. **序不变（order-invariant）训练**：不固定变量生成顺序 σ；生成顺序 σG 与训练顺序 σT 分别随机采样（其"（σ,σ）-RL-EDA"变体两者均取均匀分布），等价于"information-preserving dropout"——随机顺序的因果掩码遮蔽输入的不同子集，迫使模型对序不变、促进变量依赖识别与样本效率。
3. **GRPO 适配黑盒组合优化**：无 critic，用**组内排名尺度不变优势**（Eq.5）做策略梯度，对接 PPO-KL 目标（Eq.4/8），并给出"任意单调变换下不变"的严格理由（IGO 框架 grounding）与其理论收敛性（附录 D：目标为无偏估计、无限数据/无限容量下收敛于退化集）。
4. 大规模对照：vs Nevergrad 库 503 算法（500 伪布尔 + 496 分类有效）+ PBIL/MIMIC/BOA，跨 QUBO/NK/NK3/NAS-Bench-101、n∈{64,128,256}。

### 关键公式
- 优势（Eq.5）——**组内排名尺度不变优势**：
  A_Γ_t^λ(x) = U( rk(x, Γ_t^λ, f)/λ − 1 )，U 为非增效用函数（取 U(x)=1−2x），rk(x,Γ,f) = |{x' ∈ Γ : f(x') > f(x)}|（**排名**而非奖励值）。
  理由原文："This advantage formulation, which makes the algorithm invariant under monotone transformation of the fitness function f, is grounded in the Information-Geometric Optimization (IGO) framework."→ 只依赖**样本相对序**，f 的单调变换/尺度缩放不影响优势——这就是"scale-invariant advantage"的机制（灰盒同型：GRPO z-score 消除的是平移+线性尺度，它消除的是**一切单调变换**）。
- 生成模型：P(x) = Π_{i=1..n} π_θ(x_{σ_i}|x_{σ_<i}, σ)；每变量一个 MLP（单隐层 20 元 tanh），输入为“按 σ 因果掩码"定长向量（未生成维度置 0）。
- 目标（Eq.8，含序变体、KL 惩罚、IS 比）：L̂_λ(θ) = (1/λ)Σ_{(x_i,σ_G^i)∈Γ_t^λ} E_{σT~ξ(·|σ_G^i)} Σ_{k=1..n} [ (π_θ(x_k|σT(x_i)_<k)/π_θt(x_k|σ_G^i(x_i)_<k))·A_Γ_t^λ(x_i) − β·D_KL(π_θt(·|σ_G^i(x_i)_<k)‖π_θ(·|σT(x_i)_<k)) ]。
  **关键改序机制**：生成用 σG、训练用 σT（二者独立随机），因分母是 σG 掩码、分子是 σT 掩码，IS 比 ≠ 1 → 即使初始阶段（θ=θt, KL=0, IS 比=1 时）路径概率也会有差异；θt 与 θ 相对序不变目标训练。
- 四变体：(δ,δ) 固定序；(δ,σ) 生成固定/训练随机；(σ,δ) 生成随机/训练同序；(σ,σ) 双双均匀随机【最优】；Learned-σ（Plackett-Luce 学习生成序，收敛慢未达最优）。
- dropout 对应关系（附录 E）：随机序掩码使"某输入维度 i 在解码变量 j 时被遮蔽"的边缘概率 = 0.5（即等价于 p=0.5 的输入 dropout）；但可用输入分布不同：纯 dropout 为二项式（尖峰、难控），序+因果掩码组合则均匀铺开（Fig.5，更利于各变量见多样上下文集）。**关键性质**："using permutations of the generation orders without additional dropout is **information-preserving**"——生成时信息无损（与真 dropout 不同：dropout 在生成期引入信息损失、可致灾难性遗忘）。
- 附录 E 的 MLE 展开（为什么有效）：均匀随机序下，对维度 n 的条件对数似然展成对输入大小 k 的各项加权 w_n^k = (k−1)!(n−k)!/n!，权重随 k 递减至 k=n/2 → 模型先学边缘、再学单变量依赖、再学成对……形成"残余结构（residual structuring）"，自动过滤无关输入 → 样本效率。
- 附录 D 理论：L̂_t^λ(θ) 是 L_t^λ(θ)（Eq.32）的**无偏估计**（Lemma D.1，基于样本可交换性），无限数据+无限容量下收敛于含单解退化集。

### 实验设置
- 问题：QUBO、NK（伪布尔）、NK3（三值）；n ∈ {64,128,256}；每 (问题, n, K) 生成 10 实例；**每实例独立运行**（10 次重启、预算 10,000 次目标函数评估、预算单位=评估次数而非 GPU 时）；100 独立运行取均值；显著性 t-test p<0.001。
- 超参（Table 2）：λ=10（种群/critic-free 组大小）；每变量网络 1 隐层 20 元 tanh；U=1−2x（最优=+1、最差=−1、λ 奇数居中=0）；β=1；每代 **E=50 epochs**；Adam lr=0.001；概率裁剪 ε=0.001。复杂度 O(T·E·λ·n²h)（对比 BOA O(n³)）。
- 硬件：PyTorch 2.5/CUDA 12.4；单 QUBO n=128 预算 10k 评估 = CPU(Xeon Silver 4208) 11.5 min / V100 5 min。

### 核心数字（附录 L Table 4 / §4）
- (σ,σ)-RL-EDA 在 n=128、n=256 实例上**频获最优/显著最优**（打星=显著优于次优竞品），对 K=1（平滑）到 K=8（崎岖）全部地形不调参。
- 短预算弱点自认："may require more evaluations to converge… outperformed when considering shorter budgets (e.g., 1,000 evaluations)"——10k 预算下最好，1k 预算下被超越；n=64 小实例 10k 预算下"converges too quickly"，但 1k 预算下反而全胜。
- 三 EDA 内部：PBIL（单变量）> MIMIC/BOA（多元）——Doerr & Dufay 的"单变量可匹敌甚至超越复杂多元"结论被复现（我们的借口：多元参数爆炸、收敛慢）。
- (σ,σ) vs (σ,δ)（仅生成随机、训练固定）的差距 = 训练侧随机序（structural dropout）的贡献："The main impact is explained when comparing the green curve with the yellow curve… highlights the contribution of sampling new orders during the EDA **training** phase, which underscores the importance of the specific structural dropout at the input of each network."
- 对照 critic 变体（附录 Q，Eq.48）：(σ,σ)-RL-EDA-Critic 需调 α（NK/NK3 取 α=10、QUBO 取 0.001），**标准版在多数设置下优于 critic 版**；批评家版近双倍计算时间。
- NAS-Bench-101（真实数据集）：同超参下，1k 与 10k 预算均取最好结果。
- 计时（Table 3，QUBO 10k 评估）：(σ,σ) n=256 CPU 2940s / GPU 420s；BOA 9310s；CMApara（Nevergrad 最强）141s（仅 CPU）。

### 自认局限
- "It may require more evaluations to converge than competing methods and is therefore outperformed when considering shorter budgets"——收敛慢、短预算不占优。
- Learned-σ（显式学习生成序）"the model did not converge with the allocated budget"→ 结论："attempting to extract explicit structures in such an online search process is not beneficial"。
- 附录 Q 结论：critic 变体两大缺陷（近双倍时间 + 调参）→ 支持其无 critic 选择。
- 结论段 future work：多模态混合分布（mixture of distributions, attraction–repulsion dynamics）；未提跨实例泛化（本工作本就是每实例从头训）。
- 附录 K 自认：utilities/λ 等未调优（"we opt for simplicity and maintain a constant value"），M.4/M.5 敏感度显示 λ 与 β 对特定分布有影响。

### 与我们的逐点对比（SA-GRPO + 几何/度量信息表征 + 布局采样器零样本）

> ⚠️ 本节的对比骨架建立在"SA-GRPO = 层级条件采样 + 层内归一化"之上，该机制已被 M6 否决（见 progress-log §7）。事实与摘录仍有效，结论部分仅作历史参考。

1. **同**：① 同为"GRPO 系（无 critic）用于结构化优化/决策"，且都诉诸"组内相对优势尺度无关"；② 都处理"异质/结构化生成过程"（它是变量依赖结构，我们是决策层级结构）；③ 都用"采样时随机化 + 训练时随机化"对抗主体结构——它与我们布局采样器（同 seed 同点集）+ 拓扑泛化零样本的"变化分布内训练"精神有交集：它以随机序增强多样性，我们以参数化布局课程训练增强拓扑多样性。
2. **异**：① **它无跨实例泛化**：每实例从头训练、线上下注式 EDA（10 万评估/实例），而我们是**离线训练一个策略 → 零样本泛化到未见布局/拓扑**（拓扑泛化）；它的"随机序"是算法内随机化，我们的"随机布局"是问题域域随机化。② 它的"组"= 同一问题的 λ(<10) 个候选解，优势是**解级排名**；我们的组 = 同一决策时刻的 G 条 **b→s→l 因果链**，优势是**层级内相对**——它的优势不归属到决策层级（也不存在层级）。③ 它无时间维决策（一次解 = 一个向量，bandit），我们有 30-100 决策时刻的时序 credit assignment 问题（它没有相应处理；这也是我们"每步发一组"的依据）。④ 它的序不变性是**模型内部**不变性（生成顺序），我们的不变性是**特征层面**（PCA 主轴/char_len 归一/无绝对坐标）——两者不冲突但不可混称：它解决"生成策略不依赖变量枚举顺序"，我们解决"状态表征不依赖工厂坐标系/尺度"。⑤ 它的奖励是 fitness 本身（连续、可任意单调变换→用排名）；我们用 makespan+惩罚（尺度跨实例大→必须消除尺度，z-score 就是我们的方式）。
3. **是否威胁我们的 claims**：**不威胁**，但提供一个可预见的审稿问题——"既然排名优势对任意单调变换不变，为何你要用 z-score（只对线性变换不变）？"**拆解**：① 我们的奖励是**标量混合**（Δmakespan+Δ能耗+拖期+死锁惩罚），非单目标 fitness；排名混合权重需要预知权重排序，且等预算下排名丢弃了"差距幅度"信息（对平滑收敛不利——他们 1k 预算下也被更快收敛的方法超越，正可引用）；② 我们的优势必须在**层级内**计算，而同层内排名仍混入"层间共享分量"（LoTV 剔除的就是它——排名无法做 LoTV 分解，因为它不是幅度量）；③ z-score 保留幅度 → 可与 Dr.GRPO/EBPO 剂量式对照。可在 Related Work 中把它列为"排名不变性"极端的对照并说明我们的取舍。
4. **可直接引用的句子**：
   - "This advantage formulation, which makes the algorithm invariant under monotone transformation of the fitness function f, is grounded in the Information-Geometric Optimization (IGO) framework."（排名优势 = 单调变换不变 + IGO 谱系——citation 用）
   - "Using permutations of the generation orders without additional dropout is information-preserving. For any sampled generation order σ, the joint distribution πθt(·|σ) can fully exploit all dependencies among variables."（序随机化 = 信息保持型 dropout——支持我们"随机化不损害信息"的叙事）
   - "the model did not converge with the allocated budget… attempting to extract explicit structures in such an online search process is not beneficial when using neural estimators"（显式结构学习在线不利——支持我们"不学显式图结构、用几何不变特征"的立场）
   - "the standard version outperforms the critic-based variant in most settings"（无 critic 组相对优势战胜 critic——支持我们"无 critic"）

---

### 引用卡库 · 批1（续）· 组结构谱系（facts 卡，第 3–4 篇）

- 精读日期：2026-09-04 ｜ 两篇均为"刚从 arXiv 下载"版本号：U-Statistic = arXiv:2603.01162v3 [cs.LG] 22 Mar 2026；EBPO = arXiv:2602.05165v3 [cs.LG] 23 Feb 2026。《literature.md》尚未收录这两篇（无定位注释冲突）。

---

## 《Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic》
（arXiv:2603.01162v3，Tsinghua 数学系 / LSE 统计系 / Birmingham / USTC 管理学，通讯 Chengchun Shi；53 页）

### 精确贡献
1. **首次把 GRPO 策略梯度识别为二阶 U-统计量**（Lemma 1）：组均值基线 + 对称化把梯度写成"样本两两差"的核平均 → 直接接 Hoeffding 分解。
2. **MSE 精确刻画 + 有限样本子优界**：Theorem 2（固定 prompt 下 MSE 精确公式）、Proposition 3（minibatch 下 MSE）、Lemma 6（子优间隔有限样本界）、Theorem 8（子优间隔渐近分布，**不要求参数可辨识性**，为加权 χ² 混合）。
3. **oracle 性质与最优性**：Corollary 4/5（梯度估计器：MSE 渐近等于"知道真值函数的 oracle"，并在"基线只依赖 prompt"的估计器类中渐近 MSE 最优）、Corollary 9/10（学习策略的子优间隔同理）。
4. **最优组大小标度律**（Theorem 7）：G* = √(c3/c1)，且 G* **普遍**（不依赖采样预算 N、迭代数 n、学习率调度），只依赖数据与策略几何——直接回答"每 prompt 采多少条"。
5. **延伸至实用实现**（附录 A：reward 标准化 + IS + KL 惩罚）：Lemma 11（带标准化的梯度**不是**精确 U-统计量——核依赖数据依赖的 se(Z)，但渐近为 U-统计量）、Theorem 12（MSE 上界：偏差项 O(m²η²/ε)+O(κ²m²η²/ε³)，方差项 trace[Σ1]/G + O(1/(εG²))）、Proposition 13（一致性：DAR 目标 = E[2·arcsin(√V^πθ(X))] − κKL，衔接 Davis & Recht 的 arcsin 结果）。

### 关键公式（全部符号：X=prompt, Y=输出token序列, Z=终末奖励∈[0,2], B=prompt 批量, G=组大小, θ=参数）
- 元算法（Algorithm 1）：ĝ(θ_i) = (1/(BG))Σ_b Σ_g Σ_t ∇_θ log π_θi(Y_t^(b,g)|X^(b),Y_<t^(b,g))·(Z^(b,g) − C_i^(b,g))；
  GRPO 型基线 = **leave-one-out 组均值**：C_i^(b,g) = Z̄^(b,−g) = (Σ_{k≠g} Z^(b,k))/(G−1)（注：等价于全体组均值，尺度差 G/(G−1) 并入学习率）。
- **Lemma 1（gradient = U-statistic）**：
  ĝ_GRPO(x;θ) = C(G,2)⁻¹ Σ_{1≤i<j≤G} h((Y^(i),Z^(i)),(Y^(j),Z^(j)))，对称核
  h((Y^(i),Z^(i)),(Y^(j),Z^(j))) := (1/2)[∇_θ log π_θ(Y^(i)|x) − ∇_θ log π_θ(Y^(j)|x)]·(Z^(i) − Z^(j))。
  推导钥匙：W^(g)[Z^(g) − Z̄^(−g)] = (1/(G−1))Σ_{k≠g} W^(g)[Z^(g) − Z^(k)]（leave-one-out → 两两差分 → 对称化）。
- **Theorem 2（固定 prompt 的 MSE）**：假设 1（Z 几乎处处有界）下
  MSE(ĝ_GRPO(x;θ)) = trace[Σ_oracle(x;θ)]/G + O(E‖∇_θlog π_θ(Y|x)‖²/G²)，
  其中 Σ_oracle(x;θ) = oracle（真值函数 V^πθ 作基线）估计器的渐近协方差（×G）。首项 = 一阶投影（= oracle 估计器 MSE），残差 = 二阶退化项 O(G⁻²)。
- **Proposition 3（minibatch MSE）**：
  MSE(ĝ_GRPO(θ)) = E‖g(X;θ)−g(θ)‖²/B + trace[Σ_oracle(θ)]/(BG) + O(E‖∇_θ log π_θ(Y|X)‖²/(BG²))。
  三项 = prompt 采样方差（与 G 无关、所有算法共享）/ oracle 方差（=GRPO 首项）/ U-统计量退化残差。
- **Corollary 4（oracle）**：去掉 G 高阶项后 MSE_A(ĝ_GRPO) = MSE(ĝ_oracle) 恒等。
- **Corollary 5（最优性）**：Assumption 2（‖∇_θ log π_θ(Y|X)‖² 与 Z 在给定 X 下条件不相关）下，对**任何**形式(5)且基线只依赖 prompt 的估计器：MSE_A(ĝ_GRPO(x;θ)) ≤ MSE(ĝ(x;θ))；且在行列非退化条件下严格小于 vanilla（REINFORCE）。
- **Lemma 6（子优间隔有限样本界）**：假设 3（L-光滑 + score 函数一致有界连续）、假设 4（PL 条件：‖g(θ)‖² ≥ 2μΔ(π_θ)）、假设 5(a) 常数学习率 η_i=β<1/(2L)：
  E[Δ(π_θn)] ≤ (1−2μβ+Lμβ²)ⁿ·E[Δ(π_θ0)] + Lβ²M/(4μβ−2Lμβ²)，M = sup_θ MSE(ĝ(θ))；
  假设 5(b)（η_i = β/i）：E[Δ(π_θn)] ≤ max{(1+ε)Lβ²M/((4μβ−2)n), c/n}。→ 子优界**唯一受控于梯度估计器 MSE M**——这就是"oracle/最优性从梯度估计器传导到策略"的耦合点。
- **Theorem 7（标度律 + 最优组大小）**：子优上界依赖 B,G 的量为
  c1/B + c2/(BG) + c3/(BG²)，c1 = sup_θ E‖g(X;θ)−g(θ)‖²（数据的 prompt 方差）、c2 = sup_θ trace[Σ_oracle(θ)]、c3 = O(sup_θ E‖∇_θlog π_θ(Y|X)‖²)。
  固定预算 N = BG（每次迭代）或 N = nBG（总预算）下：
  **G* = √(c3/c1)**。
  "the optimal group size G* is universal: it is independent of the budget N, the number of iterations n and the learning rate schedule. Instead, G* depends solely on the underlying data generating process and the geometry of the policy space."（常数由数据+模型定；工程上用参考模型的几组参数估 c_i 即可放松 sup）。
- **Theorem 8（一致性 + 渐近分布）**：n→∞、B/G 固定、1/i 调度下：假设 6-9（Θ 紧 / 最优集 Θ* 闭凸且各点 Hessian 同秩 r、存在投影 Q、Q^T H Q = H* 严格负定 / 协方差矩阵 E[(ĝ−g)(ĝ−g)^T|θ_n] → Γ（可随机）/ 弱强凹 WSC：⟨g(θ), Π_Θ*(θ)−θ⟩ ≥ μ·d²(θ,Θ*)）→
  n·Δ(π_θn) →d Σ_{k=1}^{r} w_k·χ²_{1,k}（r = Hessian 秩，w_k 由梯度估计器协方差矩阵决定，非增排序）。
  三大假设解读：Assumption 6 温和（常见）；7 = **放松可辨识性**（允许最优参数形成流形，仅 r 维子空间可识别——正对 LLM 过参数化）；9 = WSC（强于 PL、弱于强凹，允许多最优）。其自述："This result is novel in two respects: (i) existing literature primarily focuses on error bounds for the suboptimality gap, which characterize its order but are not as accurate as its distribution; and (ii) classical asymptotic analyses rely on parameter identifiability, an assumption clearly violated in overparameterized LLMs."
- Corollary 9/10：w_k,GRPO − w_k,oracle = O(G⁻²)；任何"基线只依赖 prompt"的元算法 w_k ≥ w_k,oracle + O(G⁻²)→ 其子优间隔渐近不小于 GRPO。
- 附录 A 实用版（Algorithm 2）：A^(b,g) = (Z^(b,g) − Z̄^(b,−g))/se(Z^(b,•))，se = sqrt((1/(G−1))Σ_g(Z^(b,g)−Z̄^(b))²)；KL 用 K3 估计；token 级 IS 比（原文："Ideally, one would instead use the sequential IS ratio… However, it is well known that such sequential IS ratio suffers from the curse of horizon"—token 级引入偏差但方差可控，Theorem 12 上界含偏差项 O(m²η²/ε)、O(κ²m²η²/ε³)，随 η→0 消失）。

### 实验设置
- 4.1 梯度评估：合成整数算术 500 题（5 类中等难度），Qwen2.5-0.5B Base/Instruct/ICL 三档目标策略；G ∈ {4,8,16,32,64}；Monte Carlo 估计三种估计器 MSE ±95% CI。
- 4.2 最优组大小：GSM8K 上 Qwen2.5-1.5B Instruct，固定每 prompt 预算 N = BG = 1024，G ∈ {4,8,16,32,64,128}，检查点 n ∈ {200,300,400,600,800}+终态，每 G 重复 5 次取均值±95% CI；MATH 上 Qwen2.5-Math-7B，N ∈ {1024,2048,4096}，单次运行。
- 训练细节：mini-batch 512（Instruct 模型）；以 Muon 等优化器相关讨论见附录（SGD 简化分析的差距自认）。

### 核心数字
- Fig.4：三估计器 MSE 排序恒为 vanilla > GRPO-type > oracle；"with a moderately large group size (G = 8), the MSE of the GRPO-type estimator is already close to that of the oracle estimator"；G=32/64 时二者"nearly indistinguishable"；MSE 随 G 与模型能力增大而减小。
- Fig.5（GSM8K）：除 n=200 外，**G* = 32 在所有训练步一致**（n=300、800 时 G=16 与 32 很接近）；"differences in test accuracy across different values of G are mostly statistically insignificant, as only five independent runs are conducted"。
- Table 2（MATH）：N=1024 时 G* = 64（0.7817）；N=2048 时 G* = 64（0.7819，32 紧随 0.7793）；N=4096 时 **G* 移到 128**（0.7757，64 = 0.7703）；全文数值区间 ~0.767–0.782。自释："when the sampling budget is fixed at 1024, the optimal G* is larger than that for GSM8K. This shift is expected, as the constants in (11) depend on the data and the model… larger models may benefit from a larger group size."
- 该文提及 DeepSeekMath 官方 G=64——"sufficiently large for the residual term to be negligible"。

### 自认局限（原文）
- "We remark that the aforementioned results apply only to upper bounds on the suboptimality gap. Although GRPO attains a sharper upper bound, this does not necessarily imply that it achieves a strictly smaller suboptimality gap."（界 ≠ 严格更优 → 才需要 Theorem 8 的分布刻画）
- 附录 A 末尾两个 gap："(i) the original GRPO uses length normalization… although this is not used in later variants such as Dr. GRPO and GPG; (ii) simple stochastic gradient descent algorithm is no longer used in practice, and has been replaced by more sophisticated optimizers such as Muon. We leave these gaps for future research."
- 理论简化：Algorithm 1 不含 reward normalization / IS / KL（附录 A 单独处理）；分析按"每 prompt 一个 batch 的 n 次迭代"，prompt 集合有限时保证理解为"条件于此 prompt 集"。
- 实验弱项：7B 单次运行；GSM8K 5 次运行差异统计不显著；"This limitation is due to the high computational cost of training."
- Theorem 7 常数 c_i 为参数空间上确界——"may yield conservative theoretical bounds"（自注可在代表性参数点估计放松）。

### 与我们的逐点对比（SA-GRPO 四条件 / 有限样本界升级）

> ⚠️ 本节的对比骨架建立在"SA-GRPO = 层级条件采样 + 层内归一化"之上，该机制已被 M6 否决（见 progress-log §7）。事实与摘录仍有效，结论部分仅作历史参考。

1. **同**：① 它给"我们的 LoTV 方差定理"提供了**同一对象、更强结论**的标杆：GRPO 型梯度方差 = trace[Σ_oracle]/G + O(G⁻²)（而非只是"≥/≤"比较）——我们证明 SA-GRPO ≤ GRPO，可按它的口径把它精确化为**常数级推导**：层内归一化版本的 Σ_oracle^SA(θ) 与均匀版的差 = 层间共享分量（LoTV 的 Var(E[A|层级]) 恰是它的"协方差中 p 与层交叉分量"）；② 它把"组均值≈critic"（Q2）落到 Hoeffding 一阶投影 = oracle——与我们的"无 critic 但组相对≈条件值"同一论证哲学；③ 它的"基线只依赖 prompt"类（Corollary 5/10 范围）正好不含"层级条件"基线——**这是我们的接口/缝隙**；④ 它的 G* 最优组大小 = 常数（与预算/步数无关）→ 支持我们的条件 (iv) 等预算比较的充分性，并给出实验选 G 的原则（每层可按 √(c3^ℓ/c1^ℓ) 定 G_ℓ）。
2. **异**：① 其"组"= 同 prompt 的 i.i.d. 输出；我们 = 同决策时刻的**条件链**（π_B·π_S·π_L 联合采样），组内不再 i.i.d.（层级条件结构）；② 其设定是一步 bandit 化 LLM 推理；我们是 30–100 决策时刻 + 期末结算的序列——它明确把"整条输出当单动作"（"collapses the time horizon T to 1. Consequently, the problem is recast as a bandit problem"）；我们的时序 credit assignment 不在其理论范围（其"每步发一组"的时序局部化主张也不能借它的公式直接背书——需我们自己给出序列版本或逐决策时刻套用）；③ 我们的奖励 = 连续标量（makespan 主导），其 Assumption 1 只要求有界（两者兼容）；④ 它的定理假设（L-光滑 + PL + WSC + 紧参数域）面向 LLM；对调度策略网（Transformer+MLP 头）同样"不可验证但可声明"，我们可以镜像假设并引用其先例（"PL…substantially weaker than the strong concavity…accommodates landscapes with multiple global optimizers and singular Hessians"）。
3. **是否威胁我们的 claims**：**不威胁**，且是**本批最强支持 + 升级接口**。① 它对"基线类"的刻画给 SA-GRPO 的证明留了**现成接口**：把它的元算法基线类扩展为"层级条件组均值"，则 Lemma 1 的对称化、Hoeffding 分解、Theorem 2 的 MSE 公式可**逐层套用**（层内条件独立 → 层内 U-统计量；层间梯度协方差 = 共享分量 = LoTV 中被剔除项；我们"方差缩减"的精确表述 = "trace[Σ_oracle^SA(θ)] = trace[Σ_oracle^GRPO(θ)] − 层间共享分量贡献"）。② 它的 G* 律对我们的**预算分配主张**是直接支撑：等预算下最优组大小**不随预算变**——这使我们的条件 (iv) 表述"等预算比较下优势成立"比"每个 G 下都成立"更稳（我们只需在 G=G* 邻域比较即可）。③ 潜在审稿问题："你们的 LoTV 缩减是不是就是 Theorem 2 里 O(1/G) 首项的一部分？"——**是，且这是加分点**：答复口径 = 我们的 LoTV 项正是把它的 Σ_oracle 显式分解（其 Σ_oracle = 层内 + 层间，我们剔除层间）；我们可以在论文里给出"SA-GRPO 的（Theorem 2 口径）MSE ≤ GRPO 的"常数级证明。④ 他们自认的 gap（无 length norm、无 Muon、上界非精确）是"我们的有限样本界"必须避开的坑：我们的界要写"误差界"而非"优势等价"（避免它的"上界更紧 ≠ 更小"批评原样复制到我们身上：我们只陈述**方差/期望**比较，不声称子优间隔严格更小，除非也做分布刻画）。
4. **可直接引用的句子**：
   - "Crucially, the first-order projection serves as the leading term that scales at a rate of G−1 and coincides with the MSE of the oracle gradient estimator. In contrast, the second-order term decays at a faster rate of G−2, confirming its role as a higher-order residual."（U-统计量 = oracle 均值 + 残差的 MSE 结构）
   - "the optimal group size G* is universal: it is independent of the budget N, the number of iterations n and the learning rate schedule. Instead, G* depends solely on the underlying data generating process and the geometry of the policy space."（最优组大小 = 数据+模型几何的常数——直接支撑等预算比较框架）
   - "provided that Ci is a function exclusively of the random variable X (being conditionally independent of Y and Z given X), the expectation in (3) remains invariant when Z is replaced by its 'centered' version"（基线合法性的"仅依赖 X"条件——SA-GRPO 需论证"层条件组均值 = 层条件 X 函数"，这正是我们条件 (i)(ii) 的表述）
   - "This result is novel in two respects… classical asymptotic analyses rely on parameter identifiability, an assumption clearly violated in overparameterized LLMs."（不依赖可辨识性的渐近分析——可为我们调度网络的非可辨识性声明背书）

---

## 《EBPO: Empirical Bayes Shrinkage for Stabilizing Group-Relative Policy Optimization》
（arXiv:2602.05165v3 [cs.LG] 23 Feb 2026，Meta AI：Kevin Han, Yuhang Zhou, Mingze Gao, Gedi Zhou, Serena Li, Abhishek Kumar, Xiangjun Fan, Weiwei Li, Lizhu Zhang）

### 精确贡献
1. 把 GRPO 的局部组基线替换为**经验贝叶斯收缩估计器**：假设每个 prompt 的隐式成功率 θ_q ~ N(μ_glob, τ²)（全域先验），局部组均值 μ_group 向 μ_glob 收缩，收缩强度由"组内方差/组间方差"比动态决定；μ_glob、σ²、τ² 全部由 Welford 在线算法从训练史估计（无额外优化器、可扩展）。
2. 三个理论保证：饱和失败组（全零奖励）下**梯度不消失**（Theorem 3.2）、基线估计**严格更小 MSE**（Theorem 3.3）、熵衰减更保守（Proposition 3.6）；外加**聚类采样降低先验估计误差**（Proposition 3.8，topic/difficulty 聚类）。
3. 实验：G=4 主表、G ∈ {8,16,32} 组大小敏感性、difficulty-curriculum（EBPO-diff）、topic-clustering 消融（EBPO-topic vs EBPO-naive）。

### 关键公式
- 局部组均值：μ_group = (1/G)Σ_i r_i；GRPO 基线 V^GRPO = μ_group（MSE = σ²/G）。
- **EBPO 收缩基线**（Eq.2/6）：
  V_q^EB = (1 − S_q)·μ_group + S_q·μ_glob，
  **收缩因子**（Eq.3）：S_q = (σ²/G) / (σ²/G + τ²) ∈ [0,1]，
  σ² = Var(r_m,i)（组内方差），τ² = Var(r̄_m)（**观测组均值**的方差，作先验方差代理）。
- **Remark 3.1（自认保守性）**："strictly speaking we would define the prior variance as τ²_latent = Var(r̄_m) − σ²/G to remove the sampling noise. However, in online training settings, this subtraction can lead to negative variance estimates. By using the raw Var(r̄_m) as a proxy for τ², we effectively inflate the denominator, resulting in a conservative shrinkage estimator."（用原始 Var(r̄_m) 抵押 → S_q 严格落在 [0,1]，偏向局部组均值、防坍塌——软正则）。
- 最终优势（Eq.4）：Â_m,i = (r_m,i − V_qm^EB − μ_Â)/(σ_Â + ε)——其中 μ_Â、σ_Â 为**整批**原始优势的均值/std（批级归一化）。
- Welford 更新（Algorithm 1）：每批先对全体 r_m,i 更新 W_glob、对每个 μ_group,m 更新 W_means；随后 μ_glob ← mean(W_glob)、σ² ← var(W_glob)、τ² ← var(W_means)；再逐 prompt 算 S_q、V^EB、批归一化优势、走 clipped surrogate。
- **Theorem 3.2（饱和组梯度不消失）**：ri = 0 ∀i 时，GRPO：Â_raw = 0 → ∇J = 0；EBPO：V^EB = S·μ_glob > 0（需 μ_glob > 0 且批非齐次 σ_Â > 0）→ Â_raw = −S·μ_glob < 0（**非零负惩罚信号**）。
- **Theorem 3.3（MSE 严格更小）**：θ_q ~ N(μ_glob, τ²)（高斯近似；"we employ a Gaussian approximation to facilitate tractable online inference, which yields a closed-form linear shrinkage estimator"）、μ_group 为 θ_q 的无偏估计（方差 σ²/G）下，最优闭式权重 w* = Var(μ_group)/(Var(μ_group)+Var(θ_q)) = (σ²/G)/(σ²/G+τ²) = S，代入得：
  **MSE(V^EBPO) = (1−S)·σ²/G < σ²/G = MSE(V^GRPO)**（严格当 0<S<1，即 τ²>0）。
  代价：引入向 μ_glob 的**偏差**（Remark 3.4："the bias introduced by this correlation decays at a rate of O(1/T)，where T is the total number of accumulated training steps. So for large T, the global statistics act as fixed priors… keeping the policy gradient asymptotically consistent."）
- **Corollary 3.5（全局难度感知的失败惩罚）**：饱和组失败惩罚 = −S·μ_glob；"failures on globally 'easy' tasks (high μ_glob) incur larger penalties than failures on globally 'hard' tasks (low μ_glob)"。
- **Proposition 3.6（熵衰减更小）**：假设似然-奖励协方差 Ω_q 与组尺度 σ_q 独立（Remark 3.7 自认"may not strictly hold in all regimes"），用 **LoTV**（原文 Eq.7-10）：σ²_batch = E_q[σ²_q] + Var_q(μ_q)，推出 σ_batch > E_q[σ_q]（含海森式两次凸性/琴生），再由 1/x 严格凸：E_q[1/σ_q] ≥ 1/E_q[σ_q] > 1/σ_batch → **E[ΔH(π)_EBPO] < E[ΔH(π)_GRPO]**（批级分母 = 更保守更新界，防熵坍缩）。
- **Proposition 3.8（聚类采样）**：J_shuffle = Var(θ_q)；J_coherent = E_k[Var(θ_q|k)]；叠加 LoTV（Eq.10-12）：Var(θ_q) = E_k[Var(θ_q|k)] + Var_k(μ_k) → **J_coherent < J_shuffle** 当 topic 间难度不同（Var_k(μ_k)>0）——随机打乱时先验收敛到粗平均、与具体任务错配；topic 聚类使先验更锋锐。
- 与 Zeng et al.（Stein 收缩）的关系自述（Related Work）："Zeng et al. (2025) also address the high-variance bottleneck in GRPO, using Stein's Paradox… Unlike their approach, our EBPO framework uses Empirical Bayes inference to **dynamically** control shrinkage based on variance ratios, which is crucial for avoiding vanishing gradients."

### 实验设置
- 数据：训练 DAPO-Math-17K；评测 AIME2024、AIME2025、AMC23、Math-500、OlympiadBench（Pass@1，32 随机种子平均）。
- 模型：LLaMA3.1-8B、Qwen3-8B、Qwen3-14B。基线：Naive GRPO、DAPO、Dr. GRPO、EntropyMech；**同数据顺序/批大小/优化配置**（fair）。
- 实现：verl 包；mini-batch 128/回传步 → 全局 batch 512；lr 1e-6；KL 系数 β=0.001；**G=4 默认**；topic 聚类用 GPT-4.1 标注（9 个固定数域），训练 ≤1000 步、按留出验证集 Maj@16 选最优 checkpoint；difficulty 聚类用基座模型 16 rollout 的经验 pass 率、单 epoch 不 shuffle。
- 文中图均显示 200 training steps（G=4 主曲线）、图 3/4/5 的 Qwen3-8B 训练 200 步。

### 核心数字
- **Table 1（G=4，topic 聚类，Pass@1%）**：
  Qwen3-8B：EBPO-topic **64.39** avg（MATH-500 76.80 / AIME24 56.04 / AIME25 47.92 / AMC23 86.25 / Olympiad 54.93）vs GRPO 58.72、Dr GRPO 56.67、DAPO 52.63、EntropyMech 49.18（"+超过 5%"；AIME25 +5.63、Olympiad +8.94）。
  LLaMA3-8B：9.97 vs GRPO 9.00；Qwen3-14B：61.35 vs 59.88（DAPO 61.34 紧咬）。15 个 (model,dataset) 组合中 EBPO 9 个最佳。
- **Table 2（组大小敏感性，Qwen3-8B，avg Pass@1）**：
  G=8：EBPO 65.54 vs GRPO 54.26 → **+11.28%（摘要："over 11% in extremely resource-constrained settings (G = 8)"）**；DAPO 46.69。
  G=16：64.64 vs 63.16（+1.48）；G=32：**62.41 vs 62.91（−0.50，GRPO 反超）**。→ **收缩只在小组受益；G 增至 32 时 EBPO 不敌 GRPO**（且 EBPO 自身随 G 增大而下降：65.54→64.64→62.41）。
- EBPO-diff（curriculum，G=4）：Qwen3-8B 较 GRPO-diff 在 AIME24 +3.95、AIME25 +6.04；Qwen3-14B 在 AIME/Olympiad 持续领先。
- 消融（Table 3）：EBPO-topic vs EBPO-naive（Qwen3-8B）：AIME25 47.92 vs 41.88（**+6.04**）；naive 失败原因自述："the estimated prior mean μ_glob and variance τ² represent a broad, high-entropy distribution of rewards that may not accurately describe the latent reward potential of any specific prompt."
- 训练动力学：GRPO gradient norm 持续更低（饱和区消失）、EBPO 稳健非零；per-step KL：GRPO 后期尖峰，EBPO 严格有界；policy entropy 与 Maj@16/Pass@16 曲线 EBPO 全程高于 GRPO（Fig.3–5）。

### 自认局限
- Remark 3.1：τ² 用观测均值方差担保（保守、软正则，偏向局部组均值——可能系统性低估收缩）。
- Remark 3.4：向全局先验的**偏差 O(1/T) 衰减**（非零偏差；仅渐近一致）。
- Remark 3.7：Prop 3.6 的 Ω_q 独立假设"may not strictly hold in all regimes (e.g., highly saturated groups might exhibit lower covariance)… empirically validate its practical hold"。
- 高斯近似用于二元奖励（"While Beta-Binomial models are natural for binary rewards, we employ a Gaussian approximation"）——模型-先验失配风险。
- **G=32 时弱于 GRPO**（表中可见，文中只强调 G=4/8 的优势，未显式讨论 G=32 反转）。
- 依赖 GPT-4.1 做 topic 标注（外部模型 + 成本）；无调度/连续奖励域实验；全部为 LLM 数学推理。

### 与我们的逐点对比（层内归一化 + LoTV vs 向全局借强度）

> ⚠️ 本节的对比骨架建立在"SA-GRPO = 层级条件采样 + 层内归一化"之上，该机制已被 M6 否决（见 progress-log §7）。事实与摘录仍有效，结论部分仅作历史参考。

1. **同**：① 都诊断了"组均值为基线的方差/信号问题"（EBPO：小 G 高方差 + 饱和组零梯度；我们：联合空间不可比 + 层间共享分量）；② 都是**无 critic 的 GRPO 系改动**（其基线 = 局部均值与全局先验的混合，我们 = 层条件均值；其批级 σ_Â 归一化 = 我们组内 σ 归一化的批版变体）；③ 都用 **LoTV**（"Law of Total Variance"，其 Prop 3.6 的 σ²_batch = E[σ²_q] + Var(μ_q) 与我们的 Var(A) = E[Var(A|层级)] + Var(E[A|层级]) 是**同一个恒等式**——两文对偶使用同一工具）；④ 都把"共享分量"当问题：它把 Var_q(μ_q)（prompt 间共享难度）用作熵衰减论证的一部分，我们把它（层级间共享）证明为要剔除的项。
2. **异（方向相反，分层维度不同）**：① **收缩方向**：EBPO = 局部组均值 **向全局均值借强度**（shrinking toward global，跨 prompt/任务维）；SA-GRPO = 局部**层级内**组均值（只保留层条件、剔除层级间共享分量，cross-layer 借贷为零）——它收窄"组→全局"距离，我们放大"组→层条件"分辨率。② **分层维度完全不同**：EBPO 沿 **prompt/任务维**（同一个 prompt 的 G 个响应 vs 全部 prompt 的历史）；SA-GRPO 沿**决策结构维度**（同一时刻的 b·s·l 链的层级）；两者正交：可为未来组合版（在层内优势上再作 EBPO 式收缩，或把 μ_glob 换成"同层历史"）——组合顺序需防双重偏差。③ **机制性质**：EBPO = 引入偏差换方差（MSE 最优线性估计、偏差 O(1/T)）；SA-GRPO = 无方差-偏差权衡的**分解**（剔除层间分量是恒等变换下的精确分解，不引入误设定失配、无需先验/无需历史统计、无 Online 估计噪声）——这是我们的"更强"点：EBPO 的 τ² 用观测均值方差代理（Remark 3.1 自认保守）；我们用无参 LoTV 分解。④ **奖励类型**：EBPO 依赖二元 {0,1} 奖励+"成功率先验"语义（"the latent success probability of a prompt"）；我们是**连续标量**（makespan/能耗/拖期/死锁），跨实例尺度不一——"成功率"先验语义不成立；组内 z-score 与实例内归一化才是尺度无关的正确选择（同其"scale-invariant"诉求，但用幅度而非排名）。⑤ EBPO 用批级 σ_Â 归一化（把各组混入一个批尺度 → 恰是我们想剔除的"共享尺度"存在的证据）；我们用层内/组内 σ。
3. **是否威胁我们的 claims**：**方向性威胁 + 可安全拆解**。威胁点："全局借力（EBPO）在小组 G=8 实现 +11%，而你们只做局部层内归一化——为什么不做全局收缩？"**拆解**：① 证据对照：EBPO 自身数据显示 **G=32 时收缩失败（62.41 < 62.91）**——收缩的收益随 G 增大而消失甚至反转；而我们主张的是**等预算**下（总采样数一致）的方差比较，且我们的调度域 G 默认远大于 4-8（基线 G=64 量级，DeepSeekMath 官方值），其收益区间（G≤8）与我们不同；② 其收益主要来自**饱和组救梯度**（全零奖励 = 数学判题中的常见模式），我们的奖励是**连续**的、无"全零饱和"模式（makespan 永远不会全零）——其 Theorem 3.2 的适用条件在我们的域中不成立；③ 我们的 LoTV 不是"更少收缩"，而是**不同阶层的估计**：层内归一化已经天然是"向层条件值局部化"，若在层内再向全局收缩会**重新引入**我们刚剔除的共享分量（方向冲突）；④ 回应口径：EBPO 解决"组太小/饱和"问题；SA-GRPO 解决"联合空间不可比/层级耦合"问题——正交；我们可把它列为"跨 Prompt 维"基线并在消融表中作为"层内+跨层收缩"组合的对照（若组合失败/持平，则证明"剔除共享分量"是必要条件；若组合更好，则我们的表述需改为"SA-GRPO + 可选全局收缩层"——提前埋好这一组合实验，防止审稿替我们想）。
4. **可直接引用的句子**：
   - "EBPO replaces the purely local GRPO baseline with a shrinkage estimator that pulls the noisy group mean toward a global mean, μ_glob. The degree of this shrinkage is determined dynamically by the ratio of within-group variance to between-group variance."（其机制一句话版——引用并标注"跨 prompt 维"）
   - "MSE(V EBPO) = (1 − S)σ²/G" 且 "Since 0 < S < 1 (provided τ² > 0), it follows that: MSE(V EBPO) < σ²/G = MSE(V GRPO)"（严格 MSE 优势的条件 = τ²>0——注意其"严格"依赖先验方差为正；我们可对比：LoTV 分解的"严格"依赖层级间共享方差 >0，两者同构）
   - "by using the raw Var(r̄m) as a proxy for τ², we effectively inflate the denominator, resulting in a conservative shrinkage estimator"（自认先验代理保守——我们的无参分解没有这一近似）
   - "This proves that the global normalization in EBPO strictly enforces a more conservative policy update bound than GRPO, preventing the excessive entropy collapse associated with small-variance groups."（批级全局归一化=更保守更新界——与他们相反的"层内σ不变"是可讨论的另一种保守性）

---

### 批1 总结（4 篇 × SA-GRPO 的总体关系一句话）

> ⚠️ **状态（2026-09-29）**：以下四种关系定位均以"SA-GRPO = 层级条件采样 + 层内归一化"为落点，该机制已被 M6 实证否决（树 372.5 ≈ flat 371.05，见 progress-log §7）。四篇工作本身的事实、公式与原文摘录仍有效，仅"支持性前例 / 威胁判定"部分作历史参考。

BranchGRPO = **时间维分层**（机制同族、分层维度不同，支持性前例）；Order-Invariant = **排名尺度不变 + 序随机化正则**（无跨实例泛化、无层级，竞争性对照）；U-Statistic = **GRPO 梯度=U-统计量 + G* 标度律**（本批最强理论支撑与有限样本界升级接口）；EBPO = **跨 prompt 维全局收缩**（方向相反、维度正交，小 G 界的竞品与消融对象，其 G=32 反转数据可反证"局部化+分解"的必要性）。

### 引用卡库：批2·域内顶刊（全文精读事实卡）

> 精读方式：PyMuPDF 逐页全文提取 + 失表页渲染复核（文中页码标注 p#，为 PDF 页码）。
> 信源：`references\` 下三份 PDF；每篇与《method-design.md》逐点对照。口径冲突处按原文标注「存疑/纠正」。

---

## 《A Novel reinforcement learning framework based on Simplified Graph Transformer for large-scale fuzzy Flexible Job Shop Scheduling Problem》（SGFormer-SAC）

**期刊**：Engineering Applications of Artificial Intelligence, vol. 158 (2025), 111295；DOI: 10.1016/j.engappai.2025.111295（2024-12-04 收，2025-05-23 接收，2025-06-18 上线）。
**作者**：Wenquan Zhang, Fei Zhao*（通讯）, Bo Feng, Xuesong Mei（西安交通大学机械工程学院，智能制造系统国家重点实验室）。
**与《literature.md》№31 核对**：无冲突；该条注释（"简化 Graph Transformer 单层单头 + 线性化注意力 + SAC；生成集+公开集；31.23% 对 MOPNR+SPT；推理 <5s"）全部核实成立。

### 精确贡献
1. 首次把 Graph Transformer 用于**模糊析取图**（fuzzy disjunctive graph）特征提取，与 SAC 结合成 SGFormer-SAC；"We propose for the first time the use of the graph transformer method, which effectively extracts node features in disjunctive graphs"（p2）。实验证明"单层单头即可强"：p1 摘要 "We critically demonstrate that even with a single-layer, single-headed attention mechanism, remarkable competitive performance can be achieved in large-scale disjunctive graph models. This encourages a reconsideration of the design principles behind Transformers."
2. 把 Transformer 二次复杂度降到线性（见下），"at the cost of smaller GPU resources"（p3，贡献点 3）。
3. 单次训练（10×5～20×5）跨实例规模与公开基准泛化：大合成实例上对 MOPNR+SPT 平均改善 31.23%，100×60 上对 SOTA CARL 改善 11.4%，并做 ANOVA + Tukey HSD（10 次独立运行）。

### 关键公式
- **模糊模型**：三角模糊数 TFN=(t1,t2,t3)（t1 最早/t2 最可能/t3 最晚）；排序准则（Sakawa & Mori 1999）：f1(x)=(x1+2x2+x3)/4，并列时 f2(x)=x2，再并列时 f3(x)=x3−x1（p4，Eq.2 后文）。目标 Cmax = max{C_ini}（Eq.1）。
- **MDP**：状态 = 操作节点特征 x_Oij∈R⁴（总工件数 n、调度标号{0,1}、最早完成估计 C(O_ij)=C(O_i,j−1)+min(t_ijk)、已完成工序数 OP_i(t)）+ 机器节点 x_Mk∈R⁴（总数 m、忙闲标号、当前完成时间 T_t(M_k)、可加工候选数）+ O-M 对 x_O−M∈R²（t_ijk、t_ijk/工件剩余工作量比）。**已完成工序从状态中剔除**，状态随调度推进缩减为空（p4，规模友好——与我们"状态刻画不变、已完成工序不再进入"思路不同实现）。动作 = 可行 O-M 对（a_t∈A_t，|A_t|≤n×m），被调度工序立即开工 S_ij=T(t)。
- **奖励（两步差分，γ=1）**："R(a_t,s_t,s_t+1) = H_max(s_t) − H_max(s_t+1)"（Eq.3）；累计 "G = Σ_t R = H_max(s_0) − C_max"（Eq.4）。注意：与我们"期末结算为主、每步组局部化"不同，他们**每步差分即等价于最小化 Cmax**（无折扣）。
- **线性化注意力（正文核心，Eq.5–6）**：
  Q = f_Q(Z⁽⁰⁾), Q̃ = Q/‖Q‖_F；K = f_K(Z⁽⁰⁾), K̃ = K/‖K‖_F；V = f_V(Z⁽⁰⁾)；
  D = diag(1 + (1/Ñ)·Q̃(K̃ᵀ1))；Z = β D⁻¹[ V + (1/Ñ)·Q̃(K̃ᵀV) ] + (1−β)Z⁽⁰⁾。
  （f_Q/f_K/f_V 为线性前馈层；‖·‖ 为 Frobenius 范数；Ñ 为节点数；β 为残差超参。）**这就是"线性化"的做法：Q/K 做 L2 归一化后把 softmax 换成"全局和 + 自环"两项**（kernel 化全局注意力，是 Nyström 式 SGFormer 的 simplified global attention），正文声称复杂度 O(N)。
- **附录的真正线性化推导（Eq.15–17）**：softmax 注意力 → 正定核 κ(q_u,k_v)（Mercer 定理）→ 随机傅里叶/随机特征 RF 近似（Rahimi & Recht 2007）：κ(a,b)=⟨Φ(a),Φ(b)⟩≈φ(a)ᵀφ(b)，再代 ϕ∈R^m 低维特征映射（可选 PRF/Performer，Choromanski 2020）：
  "z_u^(l+1) = φ(q_u)ᵀ Σ_v φ(k_v)·v_vᵀ / (φ(q_u)ᵀ Σ_w φ(k_w))"（Eq.17），因分子分母两个求和**对每个节点 u 共享、只算一次**得 "O(N) + N·O(1) = O(N)"（p12）。→ 回答任务①：**具体做法 = Q/K 范数归一化的简化全局注意力（Eq.6）+ 附录经 Mercer 核 / RF（Performer 谱系）证明其 O(N)**；正文直接用 Eq.6，附录给 RF 推导。
- **结构信息融合**："Z_O = (1−α)Z + α·GAT(X,A)"（Eq.10），GAT 建模工序析取图邻域（Eq.7–9，LeakyReLU 系数、Softmax 归一、mask、K 头平均聚合），α=0.5。
- **解码**：score(a_t)=MLP_θπ(Z_O)（Eq.11）→ P(a_t)=Softmax(mask(score))（Eq.12）；训练用 SAC 随机采样，测试贪心取最高概率。
- **去模糊化与指标**：C_max=(C_max1+2·C_max2+C_max3)/4（Eq.13）；RPD=(C_max/C^BS_max −1)×100%（Eq.14）。

### 实验设置
- **训练集**：50,000 个随机生成模糊 FJSP 实例（规模 10×5～20×5）；验证/测试各 100 个。公开集：Lei Case1–5；模糊化 Brandimarte（FMK01–15，最大 FMk12 20×15 240 工序、FMk15 30×15 293 工序）；Hurink vdata la25–40（15 个大实例，每 5 个一组取均值）；Behnke 31–45（三组各 5 个：20×60、50×60、100×60）。模糊化方法：t1~U[σ1t, t]（σ1<1）、t2=t、t3~U[t, σ2t]（σ2>1）。
- **训练预算（回答任务②）**：E=1000 epoch，每 epoch 抽 N=100 个实例（每实例跑完整 T 步决策）入 replay buffer，之后做 SAC 梯度步（Algorithm 1；"for each epoch gradient step do"）；**墙钟训练时间 0.21 h (10×5)、0.36 h (10×10)、0.62 h (15×5)、1.33 h (15×10)、1.4 h (20×10)**。超参（随机搜索调优于 Case1，Table 1 前五组）：SAC 三路 lr=3e-4（λ_Q/λ_π/λ_γ 一致）、目标网络软更新 τ=5e-3、初始熵系数 γ=0.2、Adam；GAT 3 层、隐藏 64、多头平均聚合；单层单头 Transformer d=64、FFN 128、ReLU、**无位置编码、LayerNorm 开启**；融合 α=0.5。
- **硬件**：Intel Xeon Platinum 8383C + **单块 NVIDIA V100-SXM2-32GB**，PyTorch 1.9，Windows。
- **基线**：PDR（操作规则 FIFO/MOPNR/LWKR × 机器规则 SPT/EET）、GAT-SAC、GAT-PPO、DRL（Song 2022）、CARL（Zhang 2024b）、MPGN（Lei 2022）、GA、PPO、hCHE（Sun 2019）、CCGP（Nguyen 2013）、OR-Tools（限时 1800 s 求最优/最好解）。ANOVA + Tukey HSD 95% 区间，10 次独立运行。

### 核心数字（回答任务④⑤）
- **训练规模集（Table 3）**：对 OR-Tools gap 平均 **19.52%**（20×5 最小 12.83%）；对最佳 RL 框架（GAT-SAC）平均改善 **16.92%**；对最佳单规则改善约 **27.7%**；OR-Tools 在 15×5 仅 26% 实例得最优解，更大规模更差。
- **大合成泛化集 20×10、30×20（Table 4）**：对 GAT-SAC 平均改善 18.08%；对 MOPNR+SPT **31.23%**（p9："The gap between our approach and the optimal scheduling rule (MOPNR+SPT) becomes even more pronounced, with an average difference of approximately 31.23%. This highlights the increasing limitations of single scheduling rules as instance size grows." —— 口径：**大合成实例上 vs 最优单规则**，不是 vs 元启发式）。
- **公开集（Table 5/6）**：模糊 Brandimarte MK 系列 vs 最佳 RL（GAT-SAC）平均改善 11.61%（FMK08 例外：落后 MOPNR+EET 1.15%）；hCHE/CCGP 平均比我们好 **9.08%**（即我们**未超过**元启发式平均）；Beh41-45（100×60）对 CARL 11.4%、对 hCHE 9.2%；Fla26-40 与 CARL/MPGN 的 HSD 区间仅轻微重叠，其余站点基本不重叠。
- **推理**："notably, our method achieves a solving time of just 5 s for the 30 × 20 instances"（p9）；工业意义节（p11）："capping training time for the largest instances at 1.4 h, with inference taking only 5 s"。hCHE 最小 FMK01 需 201 s 搜索。OR-Tools 1800 s 限时下大实例得不到最优。
- ANOVA：与元启发式无统计显著差异（Fig.5a,b），但随规模增大元启发式方差显著扩大（Fig.5c,d）。

### 自认局限（原文无独立 limitation 节，可用下列原文）
- 元启发式整体仍占优："some specialized metaheuristic algorithms, such as hCHE (Sun et al., 2019) and CCGP (Nguyen et al., 2013), have achieved competitive results on specific instances, their average performance across all instances is only 9.08% better than ours"（p10）。
- 与精确解仍有 19.52% gap（p9）。
- 训练规模只到 20×5，大实例靠泛化（p7 明说"impractical to train for large-scale… we focused on developing a robust… model… train on small to medium-sized instances"）。
- 只覆盖"信息不确定性"（fuzzy 时间），**不建模动态事件**："Dynamic events, such as machine failures and urgent orders… This study focuses primarily on addressing information uncertainty"（p2）；假设全部工件/机器在 0 时刻就绪、无故障无插单（p4 假设 1–5）。
- 结论 future work 自指：多目标模糊 FJSP + 动态扰动 + 碳排放（p12）。

### 与我们的逐点对比
- **① 同**：a) 都是"单次训练、跨越未见规模泛化"的规模轴路线；b) 都主张注意力/Transformer 可以有效但不必堆深堆头；c) 都给出了训练/推理墙钟小时级的可比数据（我们可引其 0.21–1.4 h 训练 + 5 s 推理作为"训练成本量级"参照系）。
- **② 异**：a) 无任何空间/几何：状态特征只有数量/时间/比率，布局、路网、AGV 不存在——它是纯规模轴；b) 无层级因果链（无分批/物流层）；c) SAC = critics 系（双 Q + 熵正则），我们 = 无 critic 的 GRPO 系；d) 奖励 = 每步 makespan 差分，无多目标、无稀疏期末结算；e) 状态剔除已完成工序（随推进变小），我们用可变长 token + 不变特征；f) 先验图结构经 GAT 消息传递注入（我们 §2.3 对该路线的批评同样适用：GNN 建模连接性而非度量几何）。
- **③ 是否威胁我们的 claims 与拆法**：
  - **不威胁拓扑泛化/零样本主张**：其"泛化"只在规模维，未见任何布局/几何变体实验；我们的零样本=拓扑泛化，论文中分别陈述即可（规模轴已有人 31.23%，拓扑泛化仍空白——其成功恰证明"按维度泛化"可行）。
  - **轻微"名字"竞争**：我们宣称 O(N·√N) 次二次，他们宣称 O(N)。回应：不跟战——引其"单层单头即强 + 训练墙钟 <1.5 h 即收敛"作证**复杂度不是瓶颈，深网络/更优复杂度不构成核心贡献**；我们论文在复杂度上只做"工程性说明"，把贡献放在几何不变表征 + 层级优势归因（无 critic）。
  - **需修正表述处**：不要引用它为"超越元启发式"的证据——文中明确自身平均仍落后 hCHE/CCGP 9.08%；我们若与元启发式对比，须注明该立场。它对 PPO 的批评（"SAC 优于 PPO…鼓励探索"）与我们无冲突（我们基线谱系含 SAC/PPO/GRPO 三族，可作"算法族各有强主张"的横向对照句）。
- **④ 可直接引用句（原文+中译）**：
  - "We critically demonstrate that even with a single-layer, single-headed attention mechanism, remarkable competitive performance can be achieved in large-scale disjunctive graph models. This encourages a reconsideration of the design principles behind Transformers."（p1）——"我们批判性地证明即使单层单头注意力，也能在大规模析取图模型上获得显著竞争力；这促使我们重新思考 Transformer 的设计原则。"
  - "We reduce the quadratic complexity of the Transformer, O(N²), to linear complexity, O(N). This approach enhances training and inference speed when handling large-scale disjunctive graphs, at the cost of smaller GPU resources."（p3）——"我们将 Transformer 的二次复杂度 O(N²) 降至线性 O(N)，以更小的 GPU 资源代价换取大规模析取图上的训练与推理加速。"
  - "The gap between our approach and the optimal scheduling rule (MOPNR+SPT) becomes even more pronounced, with an average difference of approximately 31.23%."（p9）——"我们与最优调度规则 MOPNR+SPT 的差距进一步拉大，平均差异约 31.23%。"
  - "notably, our method achieves a solving time of just 5 s for the 30 × 20 instances."（p9）——"值得注意的是，我们的方法在 30×20 实例上仅需 5 秒求解。"
  - "This characteristic reduces the computational complexity of the global graph message passing to O(N) + N·O(1) = O(N)"（p12，附录）——"该特性将全局图消息传递的计算复杂度降至 O(N)+N·O(1)=O(N)。"

---

## 《End-to-End Multitarget Flexible Job Shop Scheduling With Deep Reinforcement Learning》（MT-FJSP / E2E-MAPPO）

**期刊**：IEEE Internet of Things Journal, vol. 12, no. 4, pp. 4420–4434, 15 February 2025；DOI: 10.1109/JIOT.2024.3485748（2024-07-30 收，2024-10-19 接收，2024-10-24 上线）。
**作者**：Rongkai Wang, Yiyang Jing, Chaojie Gu, Shibo He, Jiming Chen（浙江大学工业控制技术国家重点实验室；通讯 Chen，云端协同场景：云-边制造范式）。
**与《literature.md》№32 核对**：无冲突（"析取图+构造机器节点、弧上显式编码运输/等待时间；GNN 编码+多智能体 PPO（向量化值函数+局部 critic）"核实成立）。

### 精确贡献
1. 建模 MT-FJSP = 多目标混合整数规划：makespan（含运输时间 MKT）+ 加工能耗 PEC + 待机能耗 SEC + 运输能耗 TEC，目标 Cost_J（Eq.7），云-边制造范式下分布式车间协同（工厂 = Z 个"edge/车间"，PM 分布其上，TM 充足）。
2. 析取图 + **两个构造机器节点**（候选 PMN h_kc⁶维、已排 PMN h_ks⁸维），弧上显式编码运输时间 tt 与待机时间 tsk 传入 GNN；GIN（2 层，含边权聚合）+ GAT（3 层）编码（Eq.10–12）。
3. 端到端多智能体 PPO（job 智能体选子任务、machine 智能体选 PM，联合动作 a_t=[a_j,a_m]），**向量化值函数 + 两个局部 critic + 一个全局 critic**（Eq.14），训练期动态奖励权 ω(3)（Eq.8a 约束下 [0,1) 均匀随机），评测期换固定 ωo。

### 关键公式
- **目标**：Cost_J = ω_mk·MKT + ω_ec·(PEC+SEC) + ω_tt·TEC（Eq.7）；约束 ω_mk+ω_ec+ω_tt=1（Eq.8a）；MKT=max_i mk_i，mk_i = O_{i,|Ji|} + Σ X·t（Eq.3）；pec_i,j = Σ X·t·p（Eq.4）；bl_k = max{…}−Σ…（Eq.5，机房待机能耗）；tec_i,j = ΣΣ X·X·tt_{k,k̂}（Eq.6，工序间运输能耗）。**功率简化**："We assume the standby power of PMs and the transportation power of TMs are equal to 1."（p4）。**运输资源假设**："We assume TMs at each edge are adequate to guarantee material delivery."（p4）。
- **状态**：STN 特征 12 维（FRE：开完工 O/FT/加工能耗 pec；SRE：已排标号、入度、所选机器 id、t、p、所属工件 id、动态权重 ω(3)）；候选 PMN 6 维（时间/能耗/运输时间能力、可用标号、功率、edge id）；已排 PMN 8 维（MFT、累计 PEC/TEC/SEC、选中次数、ω(3)）。未排节点用**理想化估算规则** Eq.9 填充："FT_{i,j}(t)=FT_{i,j−1}(t)+min{t_{i,j,k}},\quad O_{i,j}(t)=FT_{i,j−1}(t);\quad pec_{i,j}(t)=min{p_{i,j,k}·t_{i,j,k}}" —— 理想估算 = 即时奖励的基准。
- **奖励（向量化）**：C⁽⁴⁾(t+1)=[MKT', PEC', SEC', TEC']；r_t⁽⁴⁾ = C⁽⁴⁾(t) − C⁽⁴⁾(t+1)；γ=1 → "G_t = C⁽⁴⁾(0) − C⁽⁴⁾(end)"，学习最大化等价于最小化 Eq.8。**分智能体奖励**："we separately utilize real-time reward of MKT and SEC for the job agent, and PEC and TEC for the machine agent for strategy learning, while r_t⁽⁴⁾ is utilized in the global critic network."（p6）。
- **Critic（回答任务①）**：Eq.14：
  V_φj⁽²⁾(s_t) = MLPφj(Ĥ_î,ĵ(t))  （job 局部值，2 维，对应 (MKT, SEC)）
  V_φm⁽²⁾(s_t) = MLPφm(Ĥ_k(t))    （machine 局部值，2 维，对应 (PEC, TEC)）
  V_φg⁽⁴⁾(s_t) = MLPφg(Ĥ_î,ĵ(t)‖Ĥ_k(t))  （全局值，4 维，对应 r_t⁽⁴⁾）
  "Corresponding to the rewards vector r_t⁽⁴⁾, we extend the state value function to be vectorized"（p8）；向量化值函数 + 向量化 advantage/target 逐元素计算（源自 GPMORL [40][41]）。
- **PPO**：L_CLIP(θ)=E_t[min(ρ_θ(t)Â_t, clip(ρ_θ(t),1−ϵ,1+ϵ)Â_t)]（Eq.15）；critic：L_V(φ)=MSE(V_ta(s_t), V_φ(s_t))，V_ta(s)=Â+V_φ(s)（Eq.16）；熵项（Eq.17）；GAE："Â_t = Σ_{t̂=t}^{end} (γλ)^{t̂−t}·(r_t̂ + γV_{t̂+1} − V_t̂)"（Eq.18，λ=0.98）。
- **训练损失**：job actor 全局损失 ~ (ω(3)ᵀ·Â_g, ρ_θj)、局部损失 ~ (ω̂(2)ᵀ·Â_j, ρ_θj)；machine 同理（用 ω̂(2)ᵀ·Â_m）；critic 用对应截断权重；四类损失系数 = **2（全局策略损失）、1（局部策略损失）、0.5（局部 critic 损失）、0.01（熵损失）**（p9）。
- **弧特征编码（回答任务②）**："The features of directed arcs are integrated into the adjacent matrix in GNN as the edge weight for the STN aggregation process."（p6）；运输时间 tt_{k,k̂}（视物理距离，同 edge/跨 edge 不同）进初始合取弧，待机时间 tsk 进新添加的有向弧（p5）。

### 实验设置
- **实例生成**：t̄=U(0,10)、p̄=U(0,10)、w̄t=U(0.8,1.2)、w̄p=U(0.8,1.2)（每 PM 的能力 = w̄t·t̄ 与 w̄t·p̄），tt 分同 edge U(0,10)/跨 edge U(0,20)；随机将部分 PM 能力置负 = 不可用（Table III）。**纯合成，无公开基准**（理由：公开基准不含 tt/p 因子）。
- **规模**：训练 6×6×2、10×6×2、20×6×3、10×10×2、15×10×2、20×10×5（I 工件 × K PMs × Z 车间）；未见于训练的**直接测试 30×10×5、40×10×5**；每规模生成训练实例 + 100 个未见验证/100 个未见测试（边训边换）。总 PEC 用平均 PEC 代替（量级处理：6×6×2 时 MKT=595 vs 总 PEC=17316）。
- **网络/超参（Table III/IV 渲染核验）**：d_STN=12、d_kc=6、d_ks=8；L_gi=2 层 GIN、L_ga=3 层 GAT；MLP 单隐层 128；actor MLP 用 tanh。Adam lr=1e-3、Adam ε=1e-10、γ=0.99、GAE λ=0.98、PPO clip ε=0.2；"Training iteration 1000"（消融用 1000 次迭代）；并行环境批 EB=16/8/4（随规模）；每 5 个 episode 重新采样实例、每 10 个 episode 验证；ReplayBuffer（TB）+ minibatch + KE 次更新（数值未提取，Table IV 底部截断）。
- **硬件/代码**：Intel Xeon Gold 6226R + NVIDIA RTX 3090 24GB；PyTorch；Python 3.9.16；Ubuntu 18.04.6；代码公开 github.com/RKWin93/E2E-MAPPO-for-MT-FJSP（⚠️ 可复现性我们可比其更高：我们同 seed 同点集）。
- **基线**：PDR（FIFO、MOR、LWKR(t)、LWKR(pt)、MWKR(t)、MWKR(pt) 选工件；SPT、SEYC 选机器）；RGA、2SGA（GA 系）；DRL [23]（MPGN 端到端多策略 PPO）；RA 随机；**MIP/Gurobi**（在线限时 60/240/600 s）。评测 δ=(x−x_b)/x_b 相对 gap；训练随机采样/验证贪心。评测默认 ωo={0.4,0.4,0.2}。

### 核心数字
- P-G（贪心）vs 最佳 PDR（MOR+SPT）：**Obj 降低 29.9%–55.0%，Std 改善 39.3%–78.0%**（p10）；P-G 优于 PDR/RA 于 **100%** 测试实例、优于 DRL[23] 于 **>60%**、优于 P-S 于 50%。
- 6×6×2：P-G 在 >10% 测试实例上优于求解器最优解（Gurobi 限时未到最优）；其他规模至少 25% 实例优于限时求解器；10×6×2 上求解器 **10 h 内无最优解**。
- 泛化（Table VII，20×10×5 模型直测）：30×10×5、40×10×5 上 P-G 持续优于 DRL[23]/RA/PDR，求解器"unable to provide solutions within a reasonable online computational time"（p12）。
- 权重实验（Fig.8）：ωo={0.1,0.1,0.8}（重 TEC）下 P-G 依权重迁移，仍为各基线中最好。
- **消融（Fig.10，6×6×2、1000 迭代、100 验证实例、对 MIP 的相对 gap 每 10 次记录——图为箱线图，以下为图读中位数，非正文数字）**：提出机制中位 gap ≈0.62；Abl.1 固定权重+加权奖励 ≈0.74；Abl.2 MLP 编候选 PMN ≈1.0（最差）；**Abl.3 移除局部 critic ≈0.85**；Abl.4 移除经验回放 ≈0.9；Abl.5 TD 替代 GAE ≈0.83。正文结论定性："The results demonstrate that the proposed mechanism significantly facilitates the policy model's strategy learning process."（p13）。**即：去局部 critic 性能恶化约相当于去 GAE，但远好于去向量化动态权重；且全局 critic 保留未消融——"无 critic 全去掉"的对照没人做过。**

### 自认局限（原文为假设性限制点，可组合引用）
- 运输资源无限："We assume TMs at each edge are adequate to guarantee material delivery."；各车间"adequate TMs"（p3/p4）——**无车辆容量约束、无调度拥堵**。
- 能耗模型简化：PM 待机功与 TM 运输功均 =1（p4）；TEC 用"平均 PEC 代替总 PEC"（p9）。
- 纯合成实例："Given that most public benchmarks can not fully cover all factors… we randomly generate synthetic instances"（p9）——无真实工厂数据、无动态事件（无故障/无插单/无波动）。
- 求解器时间窗："There is still no optimal solution over 10 h in 10×6×2 size…"（p10）——问题本身复杂度佐证。

### 与我们的逐点对比
- **① 同**：析取图图表征 + 动作掩码 + 端到端无规则辅助；"运输时间必须进表征"与我们几何注意力偏置动机一致；多目标权重 + 机器/工件分智能体与我们分层动作 b×s×l 在"动作天然分身位"上有形似。
- **② 异**：a) **critic 系最强样本**——3 个 critic（2 局部向量化 + 1 全局向量化）+ GAE + TD 目标 + 动态奖励权重，全部依赖值函数；我们 = 无 critic 组内相对优势。b) 其"分层"是**智能体分工**（job/machine 两个策略 + 各自局部 critic），不是组结构上的层级条件采样；c) 弧特征走 **GNN 边权消息传递**（离散、依赖显式图结构），我们走 **token 对的连续几何距离偏置 + 尺度归一**；d) 无空间几何（只有抽象"edge"与物理距离产生的 tt 标量）、无布局变体、无拥堵/扰动；e) 单标量加权（ωo 由用户指定 + 训练期随机），无 Pareto 面声明。
- **③ 是否威胁 / 拆法**：
  - **正面回应点（必须写入 §3.2 对照）**：其局部 critic + 消融说明"把共享/未分解的信号按档位拆开更有效"——**与我们的 LoTV 动机同向**（剔除层级共享分量），但它是"加 critic 拆解（值函数近似）"，我们是"组内相对优势拆解（无函数近似）"。拆法：① 等预算对比 PPO（其配置）vs SA-GRPO，只动优势估计器；② 引用其消融"局部 critic 摘除后 gap 由 0.62→0.85"作为**"优势信号需要分解"的独立证据**（支持我们动机），同时指出我们论文的增量 = 用组内统计量达成分解、免 critic 的显存/实现成本（其对显存无声明，我们 vs PPO 声称 −30%–50%，对 GRPO 只声称组成本不变）；③ 关键差异声明：它们是"多目标权重向量化 critic"（奖励分解），我们是"因果链层级优势归因"（时间/结构分解），两者正交，标题用词须避开"vectorized/局部 critic"。
  - **对几何偏置的叫板判定**：**不构成叫板**。它是图边离散特征（tt/tsk 标量 → GNN 邻接矩阵边权），须先构造完整析取图再做消息传递；我们是对任意 token 对的连续距离偏置 b(i,j)=w_d·d_ij/char_len+w_c·conf_sim，无需图构造、天然尺度不变、跨布局复用；而且 tt 取值直接依赖物理距离（"transportation time tt_{k,k̂} depends on the physical distance between machine m_k and m_k̂"），说明距离信息入表征是公认需求——引用他们的做法作为我们动机佐证，而非竞争。
  - **环境威胁**：无。其时限/加权/功率=1/运输无限三点，正好是我们 SimPy 拥塞管制 + 有限 AGV 的差异化空间。
- **④ 可直接引用句**：
  - "we separately utilize real-time reward of MKT and SEC for the job agent, and PEC and TEC for the machine agent for strategy learning, while r_t⁽⁴⁾ is utilized in the global critic network."（p6）——"我们分别为 job 智能体使用 MKT 与 SEC 的实时奖励、为 machine 智能体使用 PEC 与 TEC，而 r_t⁽⁴⁾ 用于全局 critic 网络。"（critic 系"信号分解"先例句）
  - "The features of directed arcs are integrated into the adjacent matrix in GNN as the edge weight for the STN aggregation process."（p6）——"有向弧特征作为边权整合进 GNN 邻接矩阵，用于子任务节点聚合。"
  - "We assume TMs at each edge are adequate to guarantee material delivery."（p4）——"我们假设各车间的运输机充足，可保证物料运送。"
  - "We conduct these studies with 1000 training iterations and 100 validation instances, computing the average relative gap to MIP(solver) every ten iterations. The results demonstrate that the proposed mechanism significantly facilitates the policy model's strategy learning process."（p12-13）——消融口径句（本文消融数字仅以图呈现）。
  - "We use average PEC to replace the total PEC for reducing the order of magnitude (e.g., MKT and total PEC are 595 and 17 316 in size 6 × 6 × 2, respectively)."（p9）——"我们用平均 PEC 代替总 PEC 以降低量级……"（其奖励尺度处理 vs 我们"奖励尺度免疫（组内相对）"卖点——可作对照句：他们须人为压量级，我们组内归一化天然免疫）。

---

## 《Green flexible job-shop scheduling considering transportation time and machine multi-rotation speeds》（GFJSPT-MMRS）

**期刊**：Swarm and Evolutionary Computation, vol. 99 (2025), 102181；DOI: 10.1016/j.swevo.2025.102181（2025-07-16 收，2025-09-17 修回，2025-09-30 接收，2025-10-11 上线）。
**作者**：Minghai Yuan*（通讯）, Zhen Zhang, Zichen Li, Yang Ye, Fengque Pei, Wenbin Gu（河海大学机电工程学院，常州）。
**与《literature.md》№35 核对**：核实成立（"首次 AGV 运输时间+多转速+多目标；D3QN 分层动作四子决策 + 分层奖励；真实案例；节能 ≤18.6%"）；补充：其为"分层=四子决策规则化 + 分层（局部/全局）奖励"的**规则选择式 D3QN**，与我们 §3.5 判定一致。

### 精确贡献
1. "This paper tackles, **for the first time**, a Green Flexible Job-Shop Scheduling Problem simultaneously considering AGV transportation time and multi-rotation-speed machines under multi-objective optimization."（p1）——首次把 AGV 运输时间 + 多转速机器 + makespan/能耗双目标合入 FJSP。
2. 五部件能耗物理模型：E_total = E_on/off + E_p + E_idle + E_agv + E_fixed（Eq.12），其中加工能耗按"空载比例 δ=0.1 + 切削比例 (1−δ)"分解，转速两档（rs∈{0,1}）。
3. **分层动作空间（四子决策层）+ 分层奖励（逐步局部 + 期末全局）**，D3QN 训练；状态为 10 维向量化特征；模拟 + 真实车间案例（新能源车企铝壳体件 15 订单 / 10 机 / 3 AGV / 34m×16m 网格布局），节能最高 18.6%。

### 关键公式（回答任务①）
- **分层动作空间**（p12–13）：四子决策层 = 工件选择、机器分配、**主轴转速档（rs∈{0,1}）**、AGV 分配；每层按领域约束过滤（工序先后、机器资格、AGV 可用、速度匹配），依专家规则，规则库源自 [40]（113 条经典调度规则），本文取 Table 2：9 条工件规则（J-FCFS/J-SPT/J-LPT/J-SPTR/J-LPTR/J-FOPNR/J-MOPNR/J-SPTSO/J-LPTSO）× 10 条机器规则（M-SPT/M-LPT/M-LUM/M-HUM/M-LEC/M-HEC/M-FNF/M-HNF/M-EET/M-LET）× 2 档转速（M-0 低速 / M-1 高速）× 6 条 AGV 规则（T-LUV/T-HUV/T-STT/T-LTT/T-LCT/T-HCT）→ **|A| = 9×10×2×6 = 1080 种复合规则组合**，动作 = 选一个复合规则。数学定义："This framework decomposes the overall scheduling problem into four sub-decision layers, each corresponding to a specific scheduling dimension. At each layer, the available actions are filtered based on domain-specific constraints… forming a rule-guided abstraction of the action space."
- **分层奖励（精确公式，Eq.44）**：
  r_t = (max_i CT_i^{t−1} − max_i CT_i^t) + (E_total^{t−1} − E_total^t)
  （局部每步奖励 = makespan 改善 + 总能耗改善；"local rewards are provided after each decision step made by the agent, while global rewards are calculated at the end of a complete scheduling cycle"，p14。）**注意**：① 该公式无权重（时间量纲与能量量纲直接相加，不同数量级混加——单位一致性弱点）；② 伪代码第 11 行写 "calculate the immediate reward rt using Eq. (55)"，全文无 Eq.55，应为笔误（奖励公式即 Eq.44）；③ "全局奖励"仅有文字定义，无显式公式。
- **状态（Eq.32–43）**：10 维向量 s(t)=[U_M_ave, U_M_std, E_mc_ave, E_mc_std, U_T_ave, U_T_std, C_J_ave, C_J_std, PR(t), CT(t)]（机器/AGV 利用率均值与标准差、机器能耗均值与标准差、工件完成率均值与标准差、工序完成比例、当前 makespan）；其中 Eq.34/35/41（能耗均值/标准差、makespan）做 min-max 归一化 Eq.42。
- **能耗模型**：P^u_{krs} = P_idle+P⁰_sp+P⁰_f（rs=0 低速）或 P_idle+P¹_sp+P¹_f（rs=1 高速）（Eq.2）；P^cut_{krs} = P^u_{krs} + P_material（Eq.3）；E_p = δ·ΣΣΣ X P^u_{krs}T_ijk + (1−δ)·ΣΣΣ X P^cut_{krs}T_ijk（Eq.6，δ=0.1）；E_idle（Eq.7–8）；E_mc = E_on/off + E_p + E_idle（Eq.9）；E_agv = ΣΣΣ Y(P^u_agv·UT + P^load_agv·LT) + P^idle_agv·WT（Eq.10，含 AGV 空驶/负载/等待功耗）；E_fixed = P0·Cmax（Eq.11）。**转速-时间耦合**：T_ijk = α·T_ijk（rs=0，α>1）或 β·T_ijk（rs=1，β<1）（Eq.26），示例 α=1.5、β=0.7。目标 f = min(Cmax, E_total)（Eq.13）；两目标求解用 Pareto 面（HV/GD/Spread 指标 Eq.48–50）。
- **D3QN（回答任务②）**：Q(s,a;θ,α,β) = V(s;θ,β) + (A(s,a;θ,α) − (1/|A|)Σ_{a'}A(s,a';θ,α))（Eq.31，dueling）；DDQN 目标（Eq.30）；ε-greedy：ε = max(0.99^e, 0.01)（Eq.45–46，e 为 episode）；MSE 损失（Eq.47）；目标网络每 C 步同步；10 维输入 → 两层并行流（V/A）→ 1080 维输出（Fig.10）。

### 实验设置
- **实例**：生成集参数（Table 5，文中表标题误录为"Transportation times…"）：工件数 {5,10,15,20,30,40,50,80,100,120,150}、机器数 {3,5,10,15,20}、AGV 数 {2,3,6}；每工件工序 U[1,5]；每工序可选机数 U[1,10]；标准加工时间 U[2,20]；AGV 运输时间用**固定运输时间表**（Table 4，20 机规模，源自 [41]）。主要对比 J5M3A2 与 J20M5A3 训练曲线；表 7 共 20 例。
- **训练预算**：**6000 episode**；奖励曲线 3000 episode 后趋于稳定（J20M5A3 约 −1300）；Taguchi L27 正交试验调参（lr∈{0.01,0.001,0.0001}、C∈{50,100,200}、batch∈{64,128,256}、buffer∈{2000,4000,8000}，HV/GD/Spread + ANOVA/Minitab）→ 最优 **lr=0.01、C=50、batch=256、buffer=8000、γ=0.99、ε 初值 1 → 最小 0.01**。
- **硬件**：Windows 10、Intel Core i5-12490F CPU 3.00GHz、16GB RAM——**纯 CPU，无 GPU**（训练成本口径极低）。
- **基线**：4 组复合规则（Rule1: J-SPT+M-SPT+M-0+T-LUV；Rule2: J-MOPNR+M-LUM+M-1+T-STT；Rule3: J-LPTR+M-LEC+M-0+T-LUV；Rule4: J-SPTSO+M-LUM+M-1+T-LUV）、Random、DQN、PPO（同状态/动作/环境；lr=0.001、clip=0.2、decay=0.95、step=100、batch=64；6000 episode；γ=0.99；buffer 8000）。无 MIP/无元启发式/无 GNN 基线。
- **案例**：新能源车企铝壳体件真实产线，15 工件 × 10 机 × 3 AGV，车间 34m×16m、2m 网格（Fig.16 网格布局 + Table 8 坐标）；M1–M4 双转速 1500/4500 rpm（Pcut 0.951/1.17 kW）；M5–M6 2000/6000 rpm；M7–M10 1000/4000 rpm；AGV同质 0.5 m/s（P_idle=0.1、P_u=0.2、P_load=0.5 kW）；P0=0.4 kW。多 AGV 动态路径用改进 **conflict-based search (CBS)** 生成动态运输时间（p21–22）。

### 核心数字（回答任务③）
- **节能 ≤18.6%**：摘要与结论 "achieving up to 18.6% reduction in total energy consumption"（p1/p24）；**文中未标注对应实例/对位基线**（存疑：表 7 为各基线 Pareto 面平均值的 Cmax/能耗对，对 PPO 的单例最大节能 J150M20A6 = (9822−9075)/9822 ≈ 7.6%，对 DQN 最大 ≈16.3%，对随机/规则组合 21–46%——18.6% 的精确口径无法复现，须谨慎引用）。
- **表 7（Cmax/总能耗，Pareto 面均值，加粗为最优）节选**：
  | 实例 | 本文(D3QN) | PPO | DQN | Random | 最佳规则 |
  |---|---|---|---|---|---|
  | J5M3A2 | **56/148** | 59/151 | 60/158 | 94/179 | 85/183 (Rule2) |
  | J20M5A3 | **210/642** | 214/648 | 219/655 | 291/809 | 233/764 (Rule4) |
  | J40M10A3 | **545/2069** | 551/2084 | 579/2113 | 599/2379 | 624/2516 (Rule4) |
  | J100M20A6 | **930/6536** | 943/6746 | 967/6899 | 1232/8693 | 993/6922 (Rule3) |
  | J150M20A6 | **1463/9075** | 1561/9822 | 1646/10838 | 1801/13850 | 1594/11556 (Rule3) |
  - 例外：**J30M10A3 上 PPO 双优（本文 434/1645 vs PPO 421/1607）**——"best performance in 90% of test instances, PPO in the remaining 10%"（p21）。
- 规则侧：Rule3 在全部 20 例能耗最低；Rule4 通常 makespan 最短。
- 训练曲线（Fig.14）：J5M3A2 收敛（Pareto 点如 (70.8,153.7)、(49.8,169.9)）；J20M5A3 奖励约 −1300 稳定（3000 episode 后）。
- 案例（Fig.17）：方案(a) 516 min/42.4 kWh；方案(b) 619 min/39 kWh → **能耗 −7.7%、makespan +20%**（文中显式给出）。
- 参数敏感性：DOE 显示参数对 HV/GD/Spread 显著影响（Figs.11–13），最优组合四参数均为边界值（lr=0.01 取上限、C=50/buffer=8000 也近边界）——**超参未做足抗性分析，调参即性能上限的风险**。

### 自认局限（原文结论段，可直接引用）
"Nevertheless, the current model assumes deterministic machine availability and static job environments. It does not yet account for uncertainties such as job interruptions, unexpected breakdowns, or dynamic arrivals. Moreover, the reward function depends on manually tuned weights, which may not generalize across all scenarios."（p24）未来工作："incorporate stochastic disruptions such as machine failures, AGV traffic congestion, and dynamic job arrivals… modeling these uncertainties explicitly in the state space and applying robust or risk-aware reinforcement learning techniques."
另：仿真中运输时间为固定表（p21 "relying solely on static transportation time tables clearly fails to meet practical requirements"），案例靠 CBS 补动态；数据"on request"（无开源代码）。

### 与我们的逐点对比
- **① 同**：都处理"工件×机器×（转速）×AGV"四级决策、运输时间入模型、makespan+能耗双目标、规则组合降动作维；其"层级 reduction"动机（"drastically reduces action dimensionality"）与我们"异质笛卡尔积联合动作难比"的动机在对**问题**的描述上同源。
- **② 异（§3.5 五家对照表用）**：
  - **本质差异**：其"分层" = **动作空间分解**（四子决策层 + 每层规则库过滤，动作 = 1080 组合规则之一）与**奖励的时域分层**（逐步局部 + 期末全局）；其学习的是一张 D3QN 对全部 1080 组合规则的**联合标量 Q 值**——层级之间**没有独立条件策略、没有条件采样、没有组内相对优势、没有层级间归因**，critic 就是 Q 网络（值函数系）。我们 = **组结构上的层级条件采样 π=π_B·π_S·π_L + 层内组相对优势归一化**（GRPO 系，无 critic，LoTV 剔除 Var(E[A|层级])）。
  - 一句话定稿：它们是"把动作空间变小"（工程化规则抽象），我们是"把优势估计器的方差变小"（统计化的组结构归因）；两者正交，不可混称"分层"。
  - 差异点之二：其策略 = **规则选择器**（RL 选启发式规则）→ 谱系上更接近"RL-辅助调度规则"（RL 选规则族，如我们基线中的 HA-DQN 风格），而我们是端到端直接动作（分批/排产/物流实体决策）。
- **③ 是否威胁 / 拆法**：
  - **名称威胁（"hierarchical"）**：§3.5 定稿句 + 差异证据引用（见 ④ 引用 1–2）。必须在正文标注：其分层=动作空间/规则库分解；我们=组（采样+归一化）维度分层，两者不在同一维度（"name guard"）。
  - **性能威胁：低**。① 无 MIP/元启发式基线（对比面 = 规则组 + DQN + PPO，且 10% 实例被 PPO 反超）；② 18.6% 口径无法复现（未标注实例与对位基线）；③ 无训练/推理墙钟声明（纯 CPU）；④ 无公开代码。引用其"90% 实例最优"时须注明对位谱系。
  - **其"自认局限"四连击=我们差异化空间的现成证据**：确定性静态（我们=扰动流）、无故障（=故障）、**无 AGV 拥堵（=拥塞管制）**、无插单（=动态插单）——四点逐一对应《方法设计文档》Intro gap 写法。
  - **对我们"层级耦合"claim 的一个加强点**：其环境假设 L/U 与机器暂存区缓冲充足、AGV 到机即卸（p3 假设 1）——回避了运输-加工耦合拥塞；我们 SimPy zone 管制层恰在其未覆盖处。
- **④ 可直接引用句**：
  - "By embedding expert-designed dispatching rules into four sub-decision layers (job, machine, speed, AGV), the framework drastically reduces action dimensionality and improves convergence and generalization."（p1）——"通过将专家规则嵌入工件、机器、转速、AGV 四个子决策层，该框架大幅降低动作维度并改善收敛与泛化。"
  - "This framework decomposes the overall scheduling problem into four sub-decision layers, each corresponding to a specific scheduling dimension… forming a rule-guided abstraction of the action space."（p12）——"该框架将整体调度问题分解为四个子决策层，每层对应一个调度维度……形成规则引导的动作空间抽象。"（§3.5 对照引语 1）
  - "Specifically, local rewards are provided after each decision step made by the agent, while global rewards are calculated at the end of a complete scheduling cycle."（p14）——"局部奖励在代理每步决策后提供，全局奖励在完整调度周期结束时计算。"（§3.5 对照引语 2：其"分层奖励"=时域局部/全局，非层级归因）
  - "Nevertheless, the current model assumes deterministic machine availability and static job environments. It does not yet account for uncertainties such as job interruptions, unexpected breakdowns, or dynamic arrivals."（p24）——"然而，当前模型假定确定性机器可用性与静态作业环境，尚未考虑工序中断、意外故障或动态到达等不确定性。"（Intro gap 现成引语）
  - "relying solely on static transportation time tables clearly fails to meet practical requirements."（p21）——"仅依赖静态运输时间表显然无法满足实际需求。"
  - "achieving up to 18.6% reduction in total energy consumption."（p24，注意与表 7 口径不一致风险）——"（所提方法）实现总能耗至多 18.6% 的降低。"

---

## 三篇合读：对《方法设计文档》的写作级结论
1. **规模轴已卷**：31.23%（SGFormer 对规则）/ 16.92%（对最佳 RL）/ 11.4%（对 CARL）——我们论文不要在"规模泛化"上正面宣称"更强"，把规模轴定位为"已有强对位、且复杂度非瓶颈（单层单头+1.4h 训练即收敛）"，我们的轴 = 几何不变表征 + 无 critic 层级归因 + 动态/拥塞环境。
2. **"信号分解有效"获得双侧证据**：MT-FJSP 的局部 critic 消融（gap 0.62→0.85）+ GFJSPT 的分层动作/奖励 + SGFormer 的"单层单头"都对"分解/简化后性能反而好"背书；我们把它引向"层内组归一化 = 无 critic 的分解"，并明确两者正交。
3. **须避开的表述陷阱**：① 不得引用 SGFormer 证明"超元启发式"；② 不得把 GFJSPT 的 18.6% 当对位精度证据（口径未标注）；③ 不得把"分层"一词让渡——三家（GFJSPT/MT-FJSP/SGFormer 无）"分层"均在动作/智能体/时域维度，我们的在组结构维度，§3.5 对照表已可按本卡逐格填写（GFJSPT=动作+规则抽象+时域奖励、MT-FJSP=双智能体+向量化/局部 critic、SGFormer=规模轴+线性化注意力）。
4. **环境差异化空间全部敞开**：三篇均无布局几何、无故障/插单、无运输资源受限+拥堵（MT-FJSP 无限 AGV、GFJSPT 固定运输表+无冲突、SGFormer 无物流）；均无零样本拓扑泛化实验；均无开源复现保障（仅 MT-FJSP 有代码——注意其 GitHub 存在）。

### 引用卡库·批3：域内对位（动态/拓扑/数字孪生）

> 精读方式：PyMuPDF 全文提取（20+18+17 页全部读完，含算法伪代码与全部表格数字）。术语保留英文；引用均为原文摘录（≤2 句/处）。
> 我们的方案简称：**Ours**（SA-GRPO 无 critic 组相对优势 + 双轴轴向注意力几何不变表征 + SimPy DES 测试台 + 参数化布局采样器；零样本=拓扑泛化+动态扰动）。

---

## 《Multi-agent collaborative reinforcement learning for dynamic flexible job shop scheduling under machine random failures》（MADAPPO）

**期刊正式信息**：Computers & Operations Research 196 (2026) 107644，DOI: 10.1016/j.cor.2026.107644（Received 3 Dec 2025 / Accepted 12 Aug 2026 / Available online 14 Aug 2026）。作者 Haoze Wu, Vladimir Golovko (corr.), Marta Chodyka, Piotr Lichograj。我们的文献表编号 [37]，§3.5 五家对照成员。

### 精确贡献
- 双智能体（JA 工件选择智能体 + MA 机器分配智能体）+ 异构图 GNN（GAT）+ 多头注意力的 PPO 族算法，解决机器随机故障的 DFJSP；目标 = min 最大完工时间、最大化机器负载均衡率、最小化机器空闲率。
- 原文："a dual-agent proximal policy optimization (PPO) algorithm integrating graph neural networks (GNN) and multi-head attention mechanisms"；"creates a dynamic hierarchical scheduling framework with two-stage collaborative processing of job selection and machine allocation"。
- 论文自报三创新（Abstract + §1）：① 再建模（工件-机器异构图拓扑特征提取）；② 算法（"decoupled workpiece agent and machine agent collaborative MADAPPO framework"）；③ 多目标复合奖励（含故障惩罚 R_interrupt = −10 与故障标志 f_flag）。

### 关键公式
- MMDP 建模：π* = argmax_{π1,π2} E_{π1,π2}[Σ_t γ^t R(s_t, a1t, a2t)]，**π1: S→Δ(A1)（JA），π2: S×A1→Δ(A2)（MA，条件于 JA 的 a1）**；联合动作 a_t = (a1t, a2t)；共享全局状态（异构图 G=(V,E,X)，V = 工件节点 ∪ 机器节点，E ⊆ V_j×V_m 为可行加工关联）与共享全局奖励 R = ω1·R_makespan + ω2·R_balance + ω3·R_idle + R_extra，ω = (0.6, 0.2, 0.2)；R_makespan = max(0, 10 − 3×(C_max(t)/T_min − 1))；R_balance = 6×B(t)；R_idle = 8×min(1, (1/m)Σ η_Ml(t))。
- **"新型优势估计"精确公式（critic 系，非无 critic）**：每智能体各一个值网络 V1(φ1)、V2(φ2)，GAE——
  **A_t = Σ_{k=0}^{T−t−1} (γλ)^k δ_{t+k}，δ_t = R_t + γV(s_{t+1}) − V(s_t)**（式 21）；
  Algorithm 2 第 3 行："calculate the advantage functions A1 and A2 **and standardize them**"。policy 目标为 PPO-clip：L1(θ1) = E[min(r1t·A1t, clip(r1t,1−ε,1+ε)·A1t)] − βH(π1)（式 22/23）；值网络 MSE 损失（式 24/25）。
  结论：**优势 = 双 critic（GAE）+ 批内标准化，属局部值函数系；无任何组内相对/无 critic 成分**——与我们 §3.5 表格"优势来自 critic/局部值，非组内相对"完全一致（原判正确，无需纠正）。
- 其余：GAT 层 GAT(X,E) = LayerNorm(GATConv(X, d_gnn, h, False) + X)（式 18）；π1 = Softmax(Linear(LeakyReLU(Linear(H_J))))（式 19）；V1(s_t) = Linear(LeakyReLU(Linear(H_global)))（式 20）。
- 奖励另含（Algorithm 1）：R_balance ← 1 − σ_load/μ_load；R_recovery ← (T_normal/(T_fault,total + T_recover,total))×exp(−ΔC_max/C_max,init)；R_interrupt = −10（故障中断惩罚）；故障 = 空闲机上按"预设故障概率"随机触发，故障时长采样于 [T_min, T_max]，入故障队列 Q_fault，倒计时恢复（Algorithm 1 行 1–14）；故障时若同机续做 Δ_ijl = t_l^f − s_ijl 不重复（数学约束 7–11），换机则重做（z_ijll' = 1）。

### 实验设置
- 实例：Brandimarte MK 改造为 DMFMK01–10（"The faulty machines are set as M1 and M2, failing at time steps 3 and 5 respectively, with durations of 3 and 4 units of time. For ease of comparison, the faults are identical."）——**故障场景固定唯一，非随机过程采样**；规模 10×6 小 / 15×8、15×4、10×10 中 / 20×5、20×10、20×15 大。
- 训练：**800 episodes**（"Until 800 episodes, the training loss converges…thus, 800 episodes is uniformly adopted"）；在 800 个随机 10×5 实例上训练，测 Brandimarte（无故障集，§4.2.2）。超参：lr=3e-5、GAT 3 层、clip=0.1、γ=0.99、head=4（Table 3 误印 0.1）、batch=32、hidden=128、熵权 β=0.01。
- 硬件：AMD Ryzen 7 9700X @3.8 GHz / 64 GB RAM / RTX 5060 Ti 16 GB；Python 3.13 + PyTorch，Windows 11/PyCharm。
- 基线：GA、PSO（元启发式）；D5QN、AC-SD、Single-PPO（DRL）。指标 RPD（式 26：RPD = (C_max−Best)/Best×100）、故障延迟率 R_d = (C_max^d − C_max)/C_max（式 27）、R_balance、R_idle、HV 与 IGD。

### 核心数字（逐表）
- **表 5**（无故障、Brandimarte）：RPD_avg：**MADAPPO 3.22** / D5QN 5.79 / Single-PPO 16.07 / PSO 18.06 / GA 27.58 / AC-SD 28.88；"represents a 44.38% performance improvement over the metaheuristic algorithm and the traditional DRL algorithm"。MK05 C_max = 174 vs Best=173，RPD=0.58；MK02/06/07 RPD = 7.69/3.17/4.86。
- **表 6/7**（故障集）：R_d^avg：MADAPPO **0.78** vs GA 1.00 / PSO 1.16 / D5QN 1.12 / AC-SD 0.81 / Single-PPO 0.89；"reduced by 27.78% (metaheuristic) and 17.02% (traditional DRL)"；DMFMK06 MADAPPO C_max^d=140（D5QN=212）；DMFMK08 R_d=0.33 全场最低。
- **表 8/9**：MADAPPO R_balance^avg=**0.80**、R_idle^avg=**0.21**（vs GA 0.66/0.33，PSO 0.69/0.31；vs D5QN 0.72/0.29→"improves by 11.65% / reduces by 26.74%"）。
- **表 12**：HV = **569.5**（最高，次高 D5QN 566.8）、IGD = **101.6**（最低，次低 AC-SD 115.6）。
- 消融收敛回合数（Fig.12）：baseline ≈30 轮 / Ablation1（去多头注意力）50 / Ablation2（去双智能体→单智能体）55 / **Ablation3（去 GAE→"ordinary advantage estimation"）95**，且 Ablation3 损失最高、波动最大、完工时间最高。

### 消融设置（对它自己的优势估计）
- 在 DMFMK02 上三组二值消融："Ablation_1 with the multi-head attention mechanism removed; Ablation_2 with the two-agent module removed and using a single agent; **Ablation_3 with the GAE removed and using ordinary advantage estimation**"。
- 即：**对自己的"新型优势估计"只做了 GAE vs 普通优势（即非 GAE 的 TD/回报基线）的比较，无组内相对/无 critic 对照**；结论泛化为"各组件均可去除后退化"，未逐组件独立量化（只有汇总曲线，无表格）。我们做 GRPO 系对照时可直接引用此"设计可参照、深度不足"点。

### 自认局限（原文结论段）
- "The current scheduling scenarios mainly focus on single-type random machine faults, and multi-objective indicators such as scheduling energy consumption are not involved."
- 问题假设（§2.1.1 约束 2）："**Without considering transportation efficiency and machine changeover costs**, all workpieces and machines are in a ready state at time 0"——**完全无运输/AGV 环节**；资源离线，无法应对动态插单（只有故障）；故障场景实验为固定唯一故障（M1/M2 于 t=3/5，时长 3/4）。

### 与我们的逐点对比
1. **同**：① 事件驱动式逐步决策（每步 JA→MA 各一动作），与我们的"事件驱动"决策时机同族；② 分层条件结构：π2 条件于 a1（π2: S×A1→Δ(A2)），与我们 π_B·π_S·π_L 因果链同属"分层条件决策"；③ 故障为随机触发（概率+时长区间），与我们"泊松故障事件流"都是随机过程（但他们实验又固定为唯一场景）；④ 多目标奖励含机器空闲/负荷均衡（我们主目标是 makespan 为主，含拥堵/拖期/死锁惩罚——部分重叠）。
2. **异**：① **优势估计为 critic 系**：双值网络 GAE + 批内标准化，无任何组内相对；我们 = 无 critic、组内相对、层级归因（LoTV 剔除共享分量）——算法族完全不同（PPO-clip 系 vs GRPO 系）。② **分层 = 动作流分层（两智能体串行分工），分层发生在智能体/动作维度；我们的分层发生在组结构维度（采样+归一化）**——设计文档 §3.5 定稿句"动作空间/决策流程分解 vs 组结构分层，两者正交，不可混称"依据成立。③ 双智能体**参数完全独立**（θ1/θ2、φ1/φ2 各一套，共享经验池+全局奖励）；我们 = 单一 π 网络的因子化条件头（π_B·π_S·π_L），共享网络编码。④ **无拓扑/几何信息**：图 = 工件-机器可行性二部图，节点特征 4 维（可用时间/进度/可用性/状态/故障），**无任何车间布局、位置、距离信息**。⑤ 无运输约束；无 AGV/缓冲/拥堵。⑥ 它的"泛化"指跨规模（800 随机 10×5 训练→测 MK 全谱），不是跨布局/拓扑。
3. **是否威胁**：不威胁。是"无 critic 层级归因"的反面证据（强势对照）；其"novel advantage estimation"实为 GAE+标准化，称不上新颖，五家对照表中可注明"其自证优势的消融（Ablation3）仅到 GAE vs 非 GAE 为止，未到组内相对层级"。另注意：**它没有物流环节**，作为我们 12 基线列表中"PPO 系"成员只对位机器故障这一扰动维度，对位时须声明"仅比排产层"。
4. **可直接引用**：
   - "the agents employ a homogeneous design that does not distinguish between two distinct decision-making tasks—workpiece allocation and machine routing—and lacks hierarchical decoupling mechanisms, resulting in an excessively large action space dimension"（批判外部；可转述"它呼吁分层解耦但止步于双智能体"）。
   - "A_t = Σ_{k=0}^{T−t−1} (γλ)^k δ_{t+k}, δ_t = R_t + γV(s_{t+1}) − V(s_t)"（"其优势来自时序差分值函数，我们来自组内相对（无 critic），两者不可混称"）。
   - "two-stage collaborative processing of job selection and machine assignment"（"两层因果链 ≠ 三层因果链；且其第二层仅条件于 a1，未含批量/物流条件"）。

---

## 《A hybrid GNN–Transformer architecture for AGV scheduling in Automated Container Terminals》（GNNT）

**期刊正式信息**：Computers & Industrial Engineering 222 (2026) 112334，DOI: 10.1016/j.cie.2026.112334（Received 8 Jan 2026 / Accepted 19 Aug 2026）。作者 Xu Gao, Fang Yu (corr.), Yongsheng Yang（上海海事大学）。我们的文献表编号 [33]，拓扑泛化最直接对位论文。

### 精确贡献
- GNN（GCN 栈）显式编码码头路网拓扑 + Transformer 编码任务时序依赖，端到端从系统状态到"任务序列 τ"，POMO-REINFORCE 训练；HHS 推理策略；青岛港**钦州** U 型自动化码头（Beibu Gulf Port Qinzhou）案例。
- 原文："existing models often treat terminal layouts as unstructured coordinate data, resulting in weak topological representation"; "It explicitly encodes the road network topology using a GNN and captures task dependencies through a Transformer"。
- 同时作者低姿态声明："The contribution lies in the integration and application of these two complementary neural network families to the AGV task sequencing problem, rather than in the invention of new network components."

### 关键公式与图构造
- **图**：路网有向图 G=(V,E)，**V = 219 节点**（4 个 QC 位置 + 32 个堆场块位置 + 183 个路口/转向点），E 有向弧 = "physical connectivity and one-way traffic constraints"（单向由邻接矩阵不对称显式编码）；边权 = 物理距离（米，来自码头 CAD）；Dijkstra 预计算全对最短路径 → 恒定速度 v=6.0 m/s 折算时间，**仅用于 reward 计算**。**关键：GCN 编码"不含显式距离边权"**——"the Map Encoder focuses on learning the latent topological features **without explicit distance-based edge weights** to ensure numerical stability during training"；节点初始化 = **逐节点可学习嵌入向量**（无坐标/几何特征、无不变性设计）。
- 任务编码：任务 i = (起点 S_i, 终点 E_i)；h_i^(0) = W_emb·concat(E_map[S_i], E_map[E_i]) + b_emb（式 17）→ λ 层 MHA+FFN（式 18–20，u_ij = q_i^T k_j/√d_key，α_ij = softmax，MHA 拼接 W_O）。**Transformer 接法：GNN 输出的地图嵌入表 E_map → 任务嵌入 → 标准自注意力 encoder → 自回归 decoder**（非"GNN 输出直接当网格"，是"地图嵌入索引式"接入）。
- 解码器（即 policy）：状态 s_t = (H_enc, π_1…π_{t−1}, M_t)（式 21）；动作 a_t = j = 选择下一个未访问任务（掩码 M_t）；**低层全部交给仿真**："AGV availability times, current AGV positions, QC/YC availability, resource occupation, time windows, congestion, and traffic states…delegated to the simulator"；任务-AGV 指派 = **round-robin**："task π_t is assigned to AGV l = ((t−1) mod L) + 1"；奖励 R(τ) = −T_max（式 22，稀疏期末）。
- **POMO-REINFORCE**：∇_θJ(θ) = E_{s~S, τ~p_θ(·|s)}[(R(τ) − b(s))∇_θlog p_θ(τ|s)]（式 23）；**基线 b(s) = (1/P)Σ_{i=1..P} R(τ_i)**（式 24，同实例 POMO 多起点的平均回报——实例级组均值基线，等价"按实例组的组相对归一化"、无 critic）；损失 L(θ) = −(1/(B·P))Σ_i Σ_j (R(τ_ij) − b(s_i)) log p_θ(τ_ij|s_i)（式 25）。
- **HHS（Hierarchical Hybrid Sampling）"层次混合采样"**：内层 = **N 个平行解码环境，每个强制从任务 i 起解**（确定性广度、覆盖所有起点的"多起点"），且解码改为**多项式随机采样**而非 POMO 原版贪心；外层 = 内层整体独立重复 **M 次**（"repeats its execution completely and independently for M times"）→ 候选池 M×N 个序列，取最小 makespan。**"层次"= 推理搜索的双层（广度×深度），不是策略分层/因果链分层**——与我们的"分层"主义不可混称，论文写作时须点名这层区别。

### 实验设置
- 环境：钦州港 U 型码头（4 QC 区、32 堆场块、219 节点），任务起点从 QC 集、终点从堆场块集均匀随机采样；**固定测试集 1000 实例 × 50 任务/实例**；AGV 规模 5/8/12；v=6.0 m/s，T_QC = T_YC = 100.0 s 固定。
- 架构：d=128，GNN 4 层，Transformer 6 层，head 8，FFN 512，logit clip 10；训练：Adam lr=1e-4，wd=1e-6，cos 衰减 γ=0.98，batch 64，**150 epochs × 100,000 episodes/epoch（≈1500 万 episode）**，grad clip 1.0，seed 1234；硬件 **NVIDIA RTX 3090**（训练+推理；GA/C-PSO 给出 30,000 次评估=100 群×300 迭代以保基线稳健）。
- 基线：GA（pop 100、gen 300、pc 0.95、pm 0.05）、C-PSO（100 粒子、300 迭代、ω=0.8）、ALNS、Transformer-Only（TO，去 GNN 的消融版）。
- 指标：Makespan（s，越低越好）、全测试集计算时间（s，1000 实例）、Empty rate（%）、平均 Wait time（s/任务，仅含 QC/YC 资源互斥 FIFO 排队延迟——原文："It does not include delays from traffic intersections, path conflicts, or vehicle congestion"）。

### 核心数字
| 规模 | GNNT makespan | TO | ALNS | GA | C-PSO | GNNT 空驶率 | 全集计算时间 |
|---|---|---|---|---|---|---|---|
| 5 AGV | **6655.39** | 6729.49 | 6705.96 | 6715.28 | 6882.55 | **36.55%** | 294 s |
| 8 AGV | 4153.33 | 4205.56 | **4151.22** | 4155.25 | 4270.43 | **37.65%** | 312 s |
| 12 AGV | **2886.16** | 2906.15 | 2908.85 | 2892.36 | 2968.52 | **34.55%** | 301 s |
- 推理：**0.29–0.31 s/实例**（完整 HHS；1000 实例约 300 s），"over two orders of magnitude faster"（Abstract；**正文 §5.2 又写 "over four orders of magnitude"——前后不一致**）；Wait time 8/12-AGV 时略高于部分基线（13.02 s / 23.39 s，8-AGV 时 ALNS 11.60 s），作者自释为"以低空驶率换等待"的 trade-off；95% CI 置信区间表（表 6–8）齐全——**统计严谨性好**。
- 空驶率降低幅度：Abstract "reducing the empty rate by **up to 4.44 percentage points**"（= 12-AGV 对 GA 38.99−34.55）。

### 自认局限（逐字）
- Abstract："**the current study is limited to semi-static environments with fixed travel times and does not account for real-time congestion, path conflicts, or equipment failures; further validation under more dynamic and uncertain conditions remains necessary.**"
- 假设 (7)："Real-time dynamic factors such as path conflicts, traffic congestion, moving obstacles, and equipment failures are deliberately excluded from the current scope."
- Conclusion 四限制：① sim-to-real gap（"including mechanical failures, sensor errors, and communication delays"）；② "**The current GNN encodes only the static topology of the road network, and macro-level decisions rely on relatively fixed travel time estimates. Consequently, the model cannot detect or avoid congestion that emerges in real time from multi-AGV interactions.**"；③ 中心化"decision brain"不易转化为单 AGV 分布式避碰；④ 黑盒不可解释。
- 未来方向：多保真仿真+故障注入、**ST-GNN**（把实时交通流作为动态图属性）、**HRL（高层 GNN-Transformer 生成宏观序列 + 低层多智能体局部路径规划）**——与我们"高层任务分配+走廊选择、低层 RCS 托管"边界声明同构，可引作行业分工佐证。

### 与我们的逐点对比（拓扑泛化重点）
1. **同**：① 都显式建模"路网而非裸坐标"，其 Intro 批评"models based on attention mechanisms or pure Transformers often simplify tasks into isolated coordinate points, lacking explicit encoding of the complex physical road network topology"——与我们"图编码器建模连接性、坐标无感知距离"的批评方向一致；② 都用 DES 仿真器验算 makespan（它的 App. A 算法即 DES 求值器）；③ 都以 makespan 为主目标 + 空驶率（我们物流环节的效率指标）；④ 都是"高层序列/分配决策 + 低层仿真/管制"职责切分。
2. **异**：① **编码对象不同**：它编码**图连接（连通性+单向约束）**，节点无几何特征、无不变性、无语义；我们编码**度量几何**（PCA 主轴、char_len 归一、距离偏置 b(i,j)=w_d·d_ij/char_len+w_c·conf_sim），且对平移/旋转/缩放不变。② **它无距离边权进编码器**（只有 reward 有用到最短路），"connects nodes but forgets how far"——正是我们 GNN 批评定稿句"图编码器建模连接性而非度量几何"的活例证。③ **泛化实验缺失**："the same architecture can be directly applied to any other ACT by simply providing the corresponding node coordinates and connectivity data"（claimed）但 **1000 测试实例 = 同一固定 U 型布局下不同任务集；从未做"未见布局/拓扑变体"实验**——且由于节点特征是可学习逐节点嵌入（查表），换布局换节点集意味着嵌入表失效，所谓"直接应用"需重训或重映射，claim 实际未验证。**结论：它不是拓扑零样本，是"单拓扑、零样本于任务分布"**；我们（训练 300 布局/测试 100 未见图谱布局 × 3 seed，全部几何不变特征）才是拓扑泛化零样本的唯一实验性证据——该论文反过来支撑我们 gap：他们"显式编码拓扑"是静态单图拓扑，泛化只到任务分布。④ 它无 critic（实例级组基线，式 24/25 与我们 GRPO 家族近亲）；但它的"组"=同一实例的 POMO 多起点，**无层级**，正可作"既有组归一化无层级结构"的对照。⑤ 它的"层次"（HHS 双层采样）与我们的"分层"（因果链组结构）同名不同义，必须点名防混淆。
3. **是否威胁**：不威胁主要 claims；反而支持两点：a) "显式拓扑编码优于裸坐标"（他们 TO 消融证明 +3.8~4.7 空驶率百分点、makespan 改善 sv T0）；b) "我们拓扑泛化零样本实验前所未有"。**唯一需包装的差异**：他们可能主张"GNN 拓扑编码已解决拓扑表征问题"，我们的应答 = "你编码连接性且从未跨拓扑评测；我们编码几何+不变性并做未见布局零样本"。其 8-AGV 时 makespan 曾略输 ALNS（4153.33 vs 4151.22），提醒我们"8 AGV 尺度 DRL 与启发式差异小"，我们表述为"与 ALNS 保持竞争"即可。
4. **可直接引用**：
   - "the Map Encoder focuses on learning the latent topological features without explicit distance-based edge weights"（"其编码无距离信息，距离只在奖励；我们把距离写进注意力偏置——度量几何进入表征入口"）。
   - "The current GNN encodes only the static topology of the road network, and macro-level decisions rely on relatively fixed travel time estimates. Consequently, the model cannot detect or avoid congestion that emerges in real time from multi-AGV interactions."（"其自认无实时拥堵感知；我们以 zone 管制层 + 匿名化占道计数建模拥堵"）。
   - "the study is limited to semi-static environments with fixed travel times and does not account for real-time congestion, path conflicts, or equipment failures"（"半静态边界；我们=动态扰动全谱：故障/低电/插单/波动/拥堵"）。

---

## 《Digital twin driven dynamic scheduling of discrete manufacturing workshop with transportation resource constraint using multi-agent deep reinforcement learning》（DTDRL-DS / MAPPO-MC）

**期刊正式信息**：Robotics and Computer-Integrated Manufacturing 95 (2025) 103042，DOI: 10.1016/j.rcim.2025.103042（Received 28 Dec 2024 / Accepted 22 Apr 2025 / Online 1 May 2025）。作者 Sai Geng、Shaohua Huang（共同一作）、Yu Guo (corr.) 等，南京航空航天大学 + 宁波工程学院 + 中国运载火箭技术研究院。我们文献表编号 [12]。

### 精确贡献
- DTDRL-DS 框架：数字孪生环境（DTE）既是 MADRL 训练环境，又作为"扰动影响评估"的仿真手段；RTDM 重调度触发判别机制按需决定是否重调度；MAPPO-MC（多评论家 PPO）同时解"机选、排序、AGV 指派"三决策。
- 原文："the digital twin environment is constructed to provide a high-fidelity training environment for scheduling agents and serve as a simulation means for evaluating the impact of disturbances"; "a rescheduling trigger discriminator mechanism is designed to dynamically determine the necessity of rescheduling"。
- 问题级创新：****在-缓冲/出-缓冲（in-buffer/out-buffer）**，把"转序等待+运输时间"显式建模："The innovation of this paper is to take into account the limited transportation resources and buffer zones to more realistically reflect the waiting time and transit time of the job, rather than assuming that the job will immediately move to another designated machine for processing after the operation is finished."

### "数字孪生"在此文的实际含义（与我们切割）
- **数据来源**：物理车间经 IIoT（UWB 超宽带 + RFID 传感器）采集实时生产数据 → 虚拟车间；DTE 由四类模型构建（"geometric model, physical model, behavioral model and rule model"），其中**规则模型"extracted and aggregated from workshop operational rules and historical data"**（含生产调度规则/资源分配规则/设备健康管理规则）。
- **建模工具 = Plant Simulation 软件**："we used Plant Simulation software to construct six different DTEs to train MAPPO-MC, addressing the 18 experimental cases"；即"数字孪生"= 商业 DES/仿真软件建模 + 实时数据同步 + 虚拟-实况一致性偏差（ΔCD）比对。**无参数化布局采样、无布局拓扑变体、无布局 seed 复现**；DTE 更新机制仅泛泛"physical workshop provides real data for DT to drive the model update"，**没有明确的模型重训/更新伪代码**（只在 RULE 模型层提"从历史数据提取/聚合"）。
- **与我们对比**：我们的"参数化布局采样器（点集+属性，同 seed 同点集）+ layout_descriptor.json（world 段 + nav 段）+ SimPy 单后端"是**可复现、可测试、无商业软件依赖的 DT-lite**；RCIM 的 DT 是**单厂高保真重资产孪生**——两者可互称互补：他们证明"孪生=训练环境+评估手段"的行业必要叙事（引用支撑），而我们证明"同一描述器 schema 训练/部署同源 → 零样本机制保证"（他们没做、也做不到跨布局，因为模型绑定单厂布局）。

### 关键公式（MAPPO-MC）
- MMDP = (S, A, R, P)；状态 S = {S_B, S_M, S_T, S_O}（buffer/machine/transport/order 四子空间，特征数随机器与 AGV 数变化；表 2：缓冲在/出数量、平均等待、本序/余序平均加工时间；机器类型/状态/利用率；AGV 载货类型/状态/剩余任务时间；订单已加工比例/总剩余工序数）。
- **动作空间**：JSA 从 8 个 job-sorting **PDR** 中选、MSA 从 4 个 machine-selection **PDR** 中选、ASA 直接选 **AGV 编号** a_a,k。**三个智能体同时决策**（"the three agents, MSA, JSA and ASA, make decisions at the same time"），共享同一状态空间 S——与 MADAPPO 的串行条件结构不同，也与我们的 π_B·π_S·π_L 条件链不同（他们的"条件结构"只存在于语义：同一决策点=同一工件，动作并行）。
- **奖励（稀疏问题+shaping）**：R_t = −(1/m)(Σ_{i=1..m} IT_i^t − Σ_{i=1..m} IT_i^{t−1})（式 22，相邻决策点间机器平均空闲量之差取负）；**IR_t = R_t × C(x)**（式 24），C(x) = 1/(1+e^{κ(n/N−χ)})（式 23，sigmoid 进度系数，κ∈(0,4]，χ∈(0,1)——"平滑地随订单完成进度递增即时奖励"）。
- **多评论家加权优势（本文核心创新）**：A_t = Σ_{k=0}∞ γλ^k δ_{t+1}（式 25，GAE，δ_t = r_t + γQ(s_{t+1},a_t) − Q(s_t,a_t)）；
  **A_i^weighted = ϖ·A_G + (1−ϖ)·A_i**（式 26，ϖ∈(0,1)，默认 0.5）。
  分工：**3 个独立 critic 每人输出 action-value Q_i(s_t, a_i,t)（"The critic network in this paper is constructed using the action-value function instead of the commonly used state-value function"——与"local 看本智能体"对应）；1 个 global critic 输出 Q_G(s_t, a_1t, a_2t, a_3t)（"comprehensive assessment of the effects of all agents' behaviors on the scheduling system from a global perspective"）**；两者各自算 GAE 后加权合成为该智能体优势；actor 走 PPO-clip 目标（Algorithm 1：argmax E[Σ_t min(π_θk/π_θ·A_weighted_i, clip(·,1−ε,1+ε)·A_weighted_i)]）；critic 更新=E[Σ(Q_i − A_i)²]（按原文印刷，实际为 TD 误差平方，印刷似有笔误）。
- **RTDM 触发判别（公式级）**：
  - ΔCD = {ΔCD_E, ΔCD_J, ΔCD_O}（式 9，虚拟-实况一致性偏差三分量）；ΔCD_E = D_M + D_E = Σ|ME^r_k − ME^v_k| + Σ|TE^r_x − TE^v_x|（式 10–12，设备 0/1 故障态差）；ΔCD_J = Σ_i Σ_j |DJ^r_ij − DJ^v_ij|（式 13–15，实际/计划加工时长差）；ΔCD_O = |NJ^r − NJ^v|（式 16–18，订单数差）。
  - 判定：ΔCD=0 继续；ΔCD_O≠0 → **订单波动扰动，立即重调度**；ΔCD_O=0 → 生产过程扰动，进 Step 4 影响评估：DTE 重仿真得扰动后完成时间 T_d，**ξ = |T_d − T_o|/T_o × 100%**（式 19）；阈值 **δ = argmin(|T_due − T_D|/T_D × 100%)**（式 20，由订单交付期与各扰动场景模拟完成时间预计算的最小值）；**ξ ≥ δ → 重调度；ξ < δ → 维持原方案**。
  - 决策点定义："when a job sits on the loading area or has just been completed at a machine and is waiting for assignment for subsequent operations (selection of the target machine and AGV), this moment is recorded as the decision point"——**事件驱动**（工件就位/完工待分配时）。

### 实验设置
- 实例：**18 例** —— m∈{6, 13}，l∈{2, 3, 4}，n∈{10, 20, 40}（无标准基准，"datasets with different scales are generated"；**自生成数据集**）；6 个 DTE（(m,l) 组合数）由 Plant Simulation 构建；另加真实工程案例 = 上海某离散制造车间，13 机 × 3 AGV × **8 工件**（订单交付期 303 min）。
- 硬件：Python 3.8，**Intel Core i5-12400 @2.50 GHz，16 GB RAM（纯 CPU，无 GPU 信息）**；超参（表 4）：train episodes=800、ep 池=816、batch=51、repeat=4、γ=0.9、λ_GAE=0.95、clip=0.12、actor lr=1e-4、critic/global-critic lr=2e-4、**ϖ（advantage weight factor）=0.5**；Optune+手工调参。
- 基线：6 条复合 PDR（FIFO+EET+SPTA 等条式）；DRL 对照：PPO、MAPPO、HDQN、MAPPO-R（原奖励）、MAPPO-GC（仅全局 critic）、MAPPO-SV（state-value critic）。
- 扰动设定（工程案例用固定时间窗）：紧急插单 J9 于 t=91 min 插入；M10 故障 40–64 min；AGV3 故障 90–120 min——**非随机过程采样，固定场景**；假设 (9)"repair time of machine or AGV remain constant"、假设 (5)"**Path conflict for AGVs is not considered**"。

### 核心数字
- vs 6 条复合 PDR：最优 makespan 改善 **11.8%–17.2%**（原文结论段"/performance improvement ranges from 11.8% to 17.2% compared to six well-known composite PDRs"）；18 例中 11 例的最差值仍优于 PDR；案例分析：13 机×3 AGV×40 工件 min 895（Rule1=1024）等。
- vs DRL：“MAPPO-MC average performance improved **8.4% vs PPO、4.5% vs MAPPO、4.2% vs MAPPO-GC、2.1% vs MAPPO-SV、1.9% vs MAPPO-R、1.7% vs HDQN**”（均为六模型平均值百分比；箱线图 BRPD/ARPD/WRPD 均最窄最优；排名 HDQN 第二）。
- AGV 配置灵敏度（13 机）：2 AGV→1158 min，3 AGV→923，4 AGV→908；"the use of 4 AGVs reduces order completion time by 27.5% compared to using 2 AGVs, and by 1.6% compared to using 3 AGVs" → 结论"3 AGVs are the optimal transportation equipment configuration"——**AGV 数量优化建议的副产品**。
- RTDM 工程案例：δ=4.8%；插单 J9（ΔCD_O≠0，直接重调度）→ 计划 276 → 288 min（+12，ξ=4.3%）；M10 故障 → 280 min（+4，**ξ=1.8% < δ → 不重调度**）；AGV3 故障 → 301 min（+25，**ξ=9.1% ≥ δ → 重调度** → 281 min，仅延 5 min）。
- 训练预算：800 episodes/场景（与 MADAPPO 同为 800，无 GPU ）、18 例各跑 20 次。

### 自认局限（原文）
- "The current study mainly focuses on reducing the completion time, which is a single research objective. In addition, while transportation resource constraints were considered, **the study did not address AGV path planning and collision avoidance**."
- 结论：单目标；无路径规划/避碰；未来 = 多能(能耗/拖期/成本)、更有效的车间专用 MADRL。
- 隐含局限：假设 (5) 无路径冲突、假设 (9) 修复时间恒定、假设 (10) 加工时间固定；扰动为时间窗固定场景（非随机过程）；DT 依赖 Plant Simulation、无跨布局复用。

### 与我们的逐点对比（数字孪生对位 + 触发机制对位）
1. **同**：① 数据-模型同源的"孪生训练/部署"思想（我们把"训练/部署共用同一描述器特征函数"精确化为零样本机制保证，他们只是口号+高保真仿真）；② 决策点 = 事件驱动（缓冲就位/完工待分配）——与我们"机器空闲/工件到达/故障触发"同族；③ 三智能体分工（JSA 工件排序≈π_B·π_S 语义、MSA 机选=π_S、ASA 选 AGV=π_L）与我们的三层因果链问题分解同构；④ 都含运输资源约束（AGV 有限 + 缓冲等待）——与"生产-物流协同"同域；⑤ PPO-clip 系底层。
2. **异**：① **动作空间设计差异大**：JSA/MSA 的动作 = **从预定义 PDR 中选规则**（8/4 选项），ASA 才直选 AGV——即"学规则选择"而非"学直接实体选择"（他们自述原因：直接选工件/机器号动作空间过大+后期无效动作）；**我们是直接实体选择 + 条件分解控制维度**（我们的动作空间乘积级 10×30×10 是"问题复杂性"而非"实现包袱"）。② **并行决策而非条件链**：三 agent 同时动作、共享 S（各 critic 又只看自己的动作/全局联合动作），与我们 π_B·π_S·π_L 的条件采样（b 先、再 s、再 l）不同——我们的"因果链"比他们强；他们靠"多 critic 加权"补协作,与我们的"层级组归一化（无 critic）"是两条正交路线（**五家对照第二行常用素材**）。③ **优势 = critic 系（加权 GAE，ϖ=0.5）**，且多 critic 分工 = 全局 critic（看联合动作）+ 局部 critic（action-value 看本代理）——反驳点：批评家数量不等于层次归因；他们无"组"概念。④ 孪生口径：他们 = 单厂 Plant Simulation（依赖商业软件、无布局参数化、无跨布局/拓扑验证）；我们 = 描述器抽象（world/nav 两段）+ 参数化采样器（点集可复现、seed 控制、拓扑课程化）——**切割点：他们的 DT 是"一个具体工厂的复刻"，我们的是"一族工厂的参数化生成 + 来源无关描述器"**。
3. **是否威胁**：**部分需防守**。威胁点：a) RTDM 的论证"频繁重调度破坏生产稳定"（"Overly frequent dynamic scheduling may lead to constant adjustments in resource allocation, which can disrupt production continuity and stability"）**对我们的"事件驱动、每事件重决策"是直接质疑**——审稿人可能问"你们是否评估扰动影响再重调度"。拆法：① 我们的高频 ≈ 决策频率而非重调度频率（SimPy DES 事件驱动本就接近"逐事件再优化"，行业主流 [11][13]）；② 我们的奖励含拖期/稳定惩罚、布局 seed 复现使消融可比；③ 把 RTDM 定位为"可插拔的元层触发策略"，作为扩展/消融项（触发阈值 ξ≥δ 或纯事件驱动）——**建议在论文 Related work 明确响应**，而非装作没看见。b) "缓冲（in/out-buffer）等待时间"我们未显式建模时会被点名（他们 problem 级创新点）——建议我们环境规格中加"暂存区容积/等待"要素（现已含 buffer 容量在布局参数内，写清即可）。c) **不威胁**的：多 critic 加权 vs 我们无 critic 组相对——两族不同，不构成 claim 冲突；他们的"改善 1.7%–8.4% vs DRL"是值函数框架内自证，与我们 GRPO 系无反方向证据。
4. **可直接引用**：
   - "Overly frequent dynamic scheduling may lead to constant adjustments in resource allocation, which can disrupt production continuity and stability."（"这正是我们决策时机设计需要预先回应的论点；我们将 RTDM 视为元层触发策略并做消融"）。
   - "A_i^weighted = ϖ·A_G + (1 − ϖ)·A_i"（"批评家数量的分工 ≠ 层级归因；批评家系与组相对系正交"）。
   - "The innovation of this paper is to take into account the limited transportation resources and buffer zones…. rather than assuming that the job will immediately move to another designated machine for processing"（"缓冲显式化是生产-物流协同的必要条件，我们的环境在布局表中已含 buffer 容积"）。
   - "MAPPO-MC…performance improvements ranging from 1.7 % to 8.4 % compared to DRL methods."（"多评论家自证增益小、且全在 critic 框架内"）。

---

## 批3 综合结论速览（写作提示）

> ⚠️ **状态（2026-09-29）**：以下判定中的"轴"表述已按术语纪律改写为"环节 / 几何·度量信息"（见 progress-log §12.3b）；三篇的事实、数字与原文摘录仍有效。凡以"我们 = 无 critic 层级归因（LoTV）"为依据的对比落点，已随 M6 否决树式分层而下调——现役算法口径 = 组内相对多环节（flat GRPO，见 progress-log §7）。

1. **MADAPPO**：critic 系双 GAE + 批标准化，"新型优势估计"名实不符（其自身消融只到 GAE vs 非 GAE）；两层因果链（π2|a1）因而是我们 π_B·π_S·π_L 三层的"弱化同族"；无运输/无几何信息/无拓扑泛化——是"分层但非组结构、有 critic 但非组相对"的现成反例。
2. **GNNT**：拓扑泛化最近邻。显式编码=连通性+单向+节点可学习嵌入（无距离进编码器、无不变性）；**无跨布局泛化实验**（1000 实例同布局），"可迁移到任意码头"声明未被验证；自认"无法感知实时拥堵/冲突/故障"。→ 我们拓扑零样本+几何注意力偏置的实验设计正好补它的洞：写作角度"连接性 vs 度量几何，静态单图 vs 参数化变拓扑"。
3. **RCIM [12]**：多 critic 加权（ϖA_G+(1−ϖ)A_i，critic 用 Q(s,a)）vs 我们无 critic 组归属——正交；RTDM 的"先评估后重调度"是我们事件驱动设计必须正面回应的论点；其 DT=单厂 Plant Simulation，反衬我们描述器抽象+参数化采样的可测试性；其未处理路径冲突/避碰（我们 zone 管制层）。

### 引用卡库·批4：域内新作（全文精读事实卡）

> 精读方式：PyMuPDF 全文提取（19 页全部读完，含 Table 1 规则表、Table 2 超参表、Table 3 全部 49 行正交实验、Table 4 四环境结果、结论段与参考文献）。术语保留英文；引用均为原文摘录（≤2 句/处）。**凡属本文重算而非原文给出的数字，均显式标注"（本文重算）"**。
> 我们的方案简称：**Ours**（SA-GRPO 无 critic 组相对优势 + 双轴轴向注意力几何不变表征 + SimPy DES 测试台 + 参数化布局采样器；零样本=拓扑泛化+动态扰动）。
> 阅读警示：本文是"**期刊来源可信、但方法学密度极低**"的一类论文——全文只有 8 个编号公式（(2.1)–(2.5) 约束、(3.1)–(3.3) 奖励），**没有一个 PPO 公式、没有一个 GNN/图卷积公式**；且存在多处表-文自相矛盾（见"事实核验与内部矛盾"小节）。用作对比时须先声明这一点，否则容易被审稿人反问"你为何拿它当同类对照"。

---

## 《Intelligent scheduling optimisation for whole-vehicle stamping production via proximal policy optimisation deep reinforcement learning》（GPVSM）

**期刊正式信息**：International Journal of Production Research（IJPR，Taylor & Francis），**在线优先（Published online: 20 Mar 2026），PDF 未见卷/期/页码（卷期未核实，属 Online First）**；DOI: **10.1080/00207543.2026.2646338**（Received 24 September 2025 / Accepted 9 March 2026 / Published online 20 March 2026）。封面页另显示 Article views: 127、Citing articles: 1（PDF 快照时点）。
**作者与机构**：Yanjuan Hu^a（通讯，yanjuan_hu@126.com）、Changhua Yin^a、Shijia Zhao^b、Yan Zhou^b、Jiashun Si^b。
^a School of Mechatronic Engineering, Changchun University of Technology（长春工业大学机电工程学院）；^b Ministry of Industry and Information Technology Equipment Industry Development Center, Beijing（工信部装备工业发展中心）。
**资助**：国家自然科学基金 no. 52575559，"Research on Intelligent Scheduling Method for Automobile Manufacturing Stamping Resources under Cloud-Edge-End Collaboration"。
**代码/数据**：**无开源代码**（全文无 GitHub 链接）；Data availability statement = "The authors confirm that the data supporting the findings of this study are available, and it can be provided upon reasonable request."

### 精确贡献

原文自报三条（Abstract）+ 四条（§1），两处口径不一致，均照录：

- Abstract 三条："(1) designing a scheduling rule combinatorial optimization method based on PPO deep reinforcement learning, which dynamically selects optimal rules through GNN; (2) adopting a hybrid scheduling rule strategy that integrates multiple rule advantages to improve efficiency; (3) designing a hyperparameter optimization method based on orthogonal experiments, systematically evaluating combinations to select optimal configuration."
- §1 四条："(1) An intelligent scheduling architecture tailored to whole-vehicle stamping production; (2) A PPO-driven hybrid rule-selection policy that dynamically composites multiple dispatching rules; (3) A graph-neural-network encoder that extracts expressive topological features from stamping-task graphs; (4) Comprehensive experiments that validate the generality and efficacy of GPVSM in realistic stamping shops."
- 自我定位句："The innovation of this paper lies in combining deep reinforcement learning with whole-vehicle stamping production task scheduling, overcoming the limitations of traditional stamping task scheduling methods in handling dynamic tasks."
- 实质贡献核验：**贡献 = "在（自建）整车冲压四工序环境里，用 PPO 替代 DQN 来选调度规则，并用正交实验调超参"**。规则选择式 DRL（rule selection）本身是 2019–2024 年已有范式（本文自己综述里就列了 Lu et al. 2024 的 DDQN 选规则、Gui et al. 2023 的复合动作选规则、Lin et al. 2019 的 DQN 选优先级规则），**"PPO 选规则"属于把已有范式换一个算法**；所谓"GNN 编码器"在正文中无任何公式与结构定义，且 §4.1.2 明写用 CNN 提特征（详见"内部矛盾"）。

### 关键公式/模型

**全文仅 8 个编号公式，全部照录如下（无遗漏）。**

数学模型（§3.2.2，目标为最小化最大完工时间）：
- f1 = min( max_{1≤j≤n} C_j )，s_jh + x_jhm × p_jhm ≤ c_jh，c_jh ≤ s_j(h+1)（式 2.1）
- C_max = max_{1≤i≤n} C_i，Σ c_Jih ≤ C_max（式 2.2）
- E_jh ≥ 0, ∀j,h（式 2.3）；E_jh ≥ E_j(h−1) + t_j(h−1)（式 2.4）；E_j′h = E_jh + t_j′h + s_jh,j′h′（式 2.5）

奖励函数（§4.3，**唯一与学习相关的公式**）：
- **U = Σ_{i=1}^{n} Σ_{j=1}^{m} OT_ij / (m ∗ C_max)**（式 3.1）——机器利用率
- **r = U′ − U**（式 3.2）
- **R = Σ r**（式 3.3）
- 原文限定："At the initial static scheduling instant, the machine-utilisation matrix is zero, U0 = 0."
- 原文目标-奖励错配自述："**The optimisation objective of this study is to minimise the maximum completion time Cmax. The reward value is composed of machine efficiency.**"——**声明目标是 min C_max，但奖励里不含 C_max 项**；因 U0 = 0，累加奖励 R 退化为"末态机器利用率"。全文无 advantage、无 clip、无 entropy、无 TD 误差、无值函数损失。

**状态空间（§4.1）**：
- "The GPVSM algorithm represents the scheduling state by a **disjunctive graph** and encodes it as a **3-D tensor of size J × M × r**, where J is the number of jobs, M is the number of machines, and r is the number of feature channels."
- 图定义："The disjunctive graph is defined by the node set N, the directed arc set A and the undirected disjunctive edge set E; the initial state is given by this graph. … undirected disjunctive edges are progressively oriented into directed conjunctive edges until a complete directed acyclic graph is obtained."
- 节点特征（**仅 2 维**）："every node is described by a **2-D vector**. The first entry is binary and indicates whether the node has already been scheduled; the second entry is **f_a(O_jh, s_t)**, a lower-bound estimate of the completion time of the operation O_jh at decision epoch t."
- 三个特征通道（§4.1.1）：**processing-time channel**（每工序在每机器上的期望加工时间，不可加工处置 0）、**schedule-result channel**（各工序完工时间，初始化为零矩阵）、**machine-utilisation channel**（各机器当前利用率，已在 [0,1]，不再归一化）。
- **编码器实现（§4.1.2）——关键句**："The three feature channels described above are stacked into a 3-D tensor and **processed by a CNN for feature extraction**. … Owing to **translational invariance and local receptive fields**, the network efficiently handles spatially structured data and generalises well even with small samples. The tensor format employed in this paper is m × n × r … Alternating **convolutional** and non-linear activation layers … **Pooling layers** down-sample … Finally, **fully-connected layers** …"
- §4.1.3 标题与 §4.1.2 **完全重名**（两节都叫 "State-Data dimension construction"），且 §4.1.3 内容其实是 DRL 算法罗列（DQN/DDQN/PPO 介绍），非状态维度——**排版/写作错误**。
- PPO 描述（§4.1.3，全文对 PPO 的全部数学描述）："PPO algorithm is an improvement to the Actor-Critic framework … PPO algorithm based on A2C algorithm, through the gradient method solves the optimal equation, while achieving action policy effect improvement, effectively improves policy space exploration capability."——**无公式**。

**动作空间（§4.2）**：动作 = **从规则库中选一条调度规则**，由该规则再从未排工序集 J 中选下一道工序。
- 原文："To overcome the myopia of a single action space, the proposed model uses a deep reinforcement learning algorithm to **select dispatching rules** according to the current shop state. … The actions listed in Table 1 constitute the algorithmic action space used to **select the next operation from the set of unscheduled operations J**."
- 决策被分成两步："Stamping-shop scheduling is decomposed into two sub-steps: **machine assignment** and **operation sequencing**."（但正文未说明两步各自如何由规则选择实现——**未核实**）
- Table 1 号称 CDR（composite dispatching rules）+ SDR（single dispatching rules）共 **12 条**：SPT、LPT、SPT+SSO、LPT+LSO、SPT∗TWK、LPT∗TWKR、LPT∗TWK、SPT∗TWKR（左列 8 条）；SROP、GROP、SRPT、LRPT（右列 4 条）。
- **本文核验出的表缺陷（可引用）**：12 条中有 **4 对公式完全重复**——SPT+SSO = arg min(p_jhm + R_jh) 与 SPT∗TWK 完全同式；LPT+LSO = arg max(p_jhm + R_jh) 与 LPT∗TWK 完全同式；SROP 与 SRPT 同为 arg min R_jh；GROP 与 LRPT 同为 arg max R_jh。**去重后实际只有 8 条不同规则**，动作空间被虚报 1.5 倍。

### 实验设置

- **环境/测试台**："a flexible stamping-job-shop testbed is built that simulates **three parallel stamping lines**; each line is divided into **four machine groups: blanking, cleaning, stamping and inspection**. Every machine has an associated **buffer with sufficient capacity** for work-in-progress storage."
- **粒度混淆（未核实）**：§3.1 说四类设备各"treated as an integrated machine/unit"（即聚合为 4 个决策单元），§5 训练又写 "**15 jobs and 4 machines**"；而 §5 测试台说"三条并行冲压线 × 每线四机群"= 12 机群。**到底 4 台还是 12 台，原文自相矛盾，标为未核实。**
- **训练/正交实验规模**："multiple intelligent algorithms are compared under identical training parameters: **15 jobs and 4 machines**. Three runs per hyper-parameter setting showed noticeable variance; to reduce this effect, the mean of the three runs is recorded."；"Orthogonal experiments use the same instance with a **task-size interval [1100, 1500]**."（该区间与"15 jobs"关系不明，疑为完工时间量级而非任务数——**未核实**）
- **超参（Table 2，正交因子 4 个）**：Optimiser ∈ {Adagrad, Adamax, AdamW, Adadelta, RMSprop}（5 水平）；γ ∈ {0.1, 0.3, 0.5, 0.6, 0.8, 0.99}（6 水平）；lr ∈ {0.1, 0.01, 0.001, 0.0001}（4 水平）；Batch Size ∈ {32, 64, 128, 256}（4 水平）；另有 **pooling size**（池化尺寸）作为第 4 因子，但 Table 2 **未列出其水平**。
- **优选结果（原文）**："In Table 3, the settings are **optimiser = Adamax, learning rate = 0.1, discount factor = 0.01, and pooling size = 32**. … The rank of importance is: **pooling size > discount factor > optimiser > learning rate**."
- **训练预算**：**原文未给出 episode 数、时间步数或训练时长**（全文 "episode" 仅出现 1 次，为泛述 "The reward increases with training episodes"）。**这是重大信息缺失。**
- **硬件/软件**："All experiments were conducted on a computer equipped with an **Intel(R) Xeon(R) w5-2445 3.10 GHz CPU, 128 GB RAM, and an NVIDIA RTX A6000 GPU**. The algorithms were implemented in **Python using the PyTorch framework**."
- **基线（列全）**：
  - **精确算法（表头记作 BB，正文称 "exact algorithm"，未给出 solver 名称/求解时限——未核实）**
  - **DRL：DQN、Dueling DQN、DDQN、PPO（本文 GPVSM）**；正文另提及 **A2C**（只在 Fig.10 收敛曲线出现，未进 Table 4）
  - **参照系说明**："Network architecture: **target and online networks are identical** to ensure fair comparison and to exclude performance deviations caused by architectural differences."（即：**四个基线用的都是同一套网络结构**，只换算法；那"GNN 编码器"在消融意义上从未被单独检验）
  - **无元启发式基线、无规则基线（SPT 等只作动作空间，未作对照跑分）**
- **评估指标**：r̄（average reward，≈末态机器利用率）、ā（称 "pathlength"，即总完工时间/最大完工时间）、ratio（以精确算法值为 100% 的相对比）。**无 RPD、无标准差/置信区间、无统计检验。**
- **实例**：A/B/C/D 四个"随机生成实例"（§5.2："validation is conducted through **randomly generating 4 instances**; the four groups of experiments are of different scales of **path length (r̄)** where **path length (ā) is the total completion time**"——**r̄/ā 符号在此句中被写反，属原文笔误**）。"Ten runs per instance were averaged and recorded in Table 4 to ensure fairness."
- **未兑现的实验承诺**：§1 写 "Section 5 presents empirical results on **synthetic and real production traces**, demonstrating superior makespan, utilisation, and generalisability against **benchmark dispatching rules and state-of-the-art RL schedulers**."——**实际 §5 无任何真实产线数据（四实例均为随机生成），也没有与任何调度规则或元启发式对比**。属 claim 与实验不符，可直接引用。

### 核心数字

**Table 4（唯一主结果表；原文列 = BB / DQN / Dueling DQN / DDQN / PPO；ratio 以 BB 为 100%）**

| 环境 | 指标 | BB(exact) | DQN | Dueling DQN | DDQN | **PPO** |
|---|---|---|---|---|---|---|
| A | r̄ | **缺（原表未给出）** | 0.509 | 0.506 | 0.506 | **0.509** |
| A | ā | 1206 | 1124 | 1123 | 1123 | **1116** |
| A | ratio | 1 | 93.2% | 93.1% | 93.1% | **92.5%** |
| B | r̄ | **缺** | 0.6422 | 0.6423 | 0.6424 | **0.7000** |
| B | ā | 722 | 723 | 723 | 724 | **661** |
| B | ratio | 1 | 100.1% | 100.1% | 100.2% | **91.5%** |
| C | r̄ | **缺** | 0.5056 | 0.5055 | 0.5054 | **0.5260** |
| C | ā | 1100 | 1040 | 1040 | 1040 | **999** |
| C | ratio | 1 | 94.5% | 94.5% | 94.5% | **90.8%** |
| D | r̄ | **缺** | 0.5915 | 0.5914 | 0.5914 | **0.6236** |
| D | ā | 880 | 879 | 878 | 878 | **833** |
| D | ratio | 1 | 1 | 99.7% | 99.7% | **94.6%** |

- **表缺陷（经逐词坐标核验）**：Table 4 的 r̄ 行在 **A/B/C/D 四个环境下都只有 4 个数值、缺 BB（exact）列**，而 ā 与 ratio 行均为 5 值。即"平均奖励"一栏精确算法值缺失，无法与 DRL 在奖励口径上对比（原文未说明原因）。
- PPO 相对精确算法的完工时间改善：A **7.5%**、B **8.5%**、C **9.2%**、D **5.4%**（由 ratio 反读：92.5/91.5/90.8/94.6）。
- PPO 相对**最强 DRL 基线（DQN 系）**的完工时间改善：A 1116 vs 1123 = **0.6%**；B 661 vs 723 = **8.6%**；C 999 vs 1040 = **3.9%**；D 833 vs 878 = **5.1%**。**环境 A 下与 DQN 系几乎不可区分。**
- 奖励口径下的自证更弱：**环境 A 中 PPO r̄ = 0.509，与 DDQN 的 0.509 完全并列**，而原文仍写 "the PPO algorithm results under different situations have higher reward values"。

**Table 3（49 组正交实验；原文只给每格 (完工时间, 奖励)，未给任何汇总统计。以下为本文重算，仅作事实核验用）**

- DQN：完工时间 min **1155** / max 1193 / **mean 1167.9 / sd 7.4**；奖励 mean **0.5164**
- Dueling DQN：min **1156** / max 1182 / mean 1167.7 / sd 6.6；奖励 mean 0.5164
- DDQN：min **1156** / max 1186 / mean 1167.0 / sd 6.3；奖励 mean 0.5164
- **PPO：min 1111 / max 1404 / mean 1199.8 / sd 69.3；奖励 mean 0.5039**
- **结论（本文重算，可直接用作我们论证）**：PPO 的**最好格点确实最好（1111）**，但其**均值最差（1199.8 vs 1167.0–1167.9）、方差高一个数量级（sd 69.3 vs 6.3–7.4）、平均奖励也最低（0.5039 vs 0.5164）**。原文的"PPO 更优"实际是**在 49 个格点中挑最大值**；而 Table 3 明文写 "Due to data instability during the training process, the average value of 3 runs was selected."，且**无独立验证集**——属典型的选择偏差（best-of-grid），原文对此的表述是 "occasionally, there exists one set of experimental results significantly better than the average results, which has a certain degree of randomness."（自认随机性，却仍按最大值选参数）。
- 原文对 PPO 的负面自述（重要，可用于反驳"PPO 万能"叙事）："**the PPO algorithm is extremely sensitive to parameters, its performance variation amplitude is large and wide-ranging**"；"PPO scheduling achieves optimal effect with stable results, but **easily falls into the extrema and cannot explore**, requiring appropriate parameters"；"PPO and A2C converge to better intervals than DQN, DDQN and Duelling-DQN; **PPO learns fastest, whereas A2C requires far longer learning time and is impractical**."
- 精确算法侧的自述："The **exact algorithm cannot learn multiple scheduling rules** in experiments, so the experimental constraint is limited to the shortest processing time. … but computation time increases progressively with model size, completion effectiveness declines, and it simultaneously places significant computational pressure on equipment."——即 **BB 被限死在"最短加工时间"单一规则下运行**，因此 Table 4 的 ratio 基线是**被刻意弱化的精确算法**，"改善 7.5%–9.2%" 的分母不可比。

### 自认局限（原文，逐字）

结论段（§6）：
- "Meanwhile, although hyperparameter tuning and the integration of scheduling rules are crucial to algorithmic learning performance, **their complex implementation logic still poses certain challenges for industrial deployment**."
- "**However, this method does not always yield optimal solutions in every situation, and the stability of the algorithm and application complexity warrant further investigation.**"
- 未来工作三条（逐字）："Future research work will proceed in several directions: **(1) Adopt more advanced workshop state extraction models; (2) Consider deterministic perturbation factors including the uncertainty of sporadic equipment failures and time errors in manual operations; (3) extend the experimental scenario to include more realistic factors – equipment breakdowns, material-supply fluctuations, etc.** – to continuously refine the strategy and provide stronger technical support for the intelligent upgrading of flexible stamping shops and the manufacturing industry as a whole."

建模假设中的硬性排除（§3.2.2，逐字，"这是它自己关掉的门"）：
- "(2) Machines operate continuously, and **unexpected breakdowns are disregarded**;"
- "(3) The processing time of a workpiece comprises setup time, transportation time, and machining time;"
- "(4) Processing tasks on a machine are **non-preemptive**;"
- "(5) **All jobs are available at the initial time t = 0.**"
- §3.2 建模约定："each operation can be processed on only one machine and cannot be interrupted; every job must follow the technological sequence; **auxiliary times such as transport, tool and fixture changes are included in the processing time**; each machine can handle at most one operation at a time; **inter-machine transport time is negligible**."
- §3.1："In this paper, **only the influence of automated equipment is considered, and the impact of labour allocation on production tasks is ignored.**"

**归纳**：它自认的局限 = ①无故障/无动态扰动（"dynamic" 只在标题与综述里，实验里没有）；②无运输（运输时间被吸收进加工时间且声明可忽略）；③无抢占、无分批；④超参调优工程复杂、不宜部署；⑤状态提取方法仍不够好。

### 与我们的逐点对比

1. **同**：① 问题域同为"多工序、多机、有前后道工序约束的车间排产"，且都显式分"机器分配 + 工序排序"两个子问题（原文："the problem comprises two sub-problems: (1) sequencing of processing tasks … and (2) task allocation, i.e. determining the machine assignment for each processing task."）；② 都用"事件驱动式逐步派工"（每步从未排工序集选一道）；③ 都用规则/优先级作为动作语义（我们是因子化条件头 π_B·π_S·π_L 直选实体，它是选规则再由规则选实体）；④ 都以 makespan 为核心指标并用"相对最优/精确解的百分比"作 ratio；⑤ 都用 PPO 系做底层（它 PPO-clip，我们 GRPO 族）。
2. **异**（**三条差异化全部落空，见下**）：
   - **真 AGV 派车：无。** 运输被写死进加工时间、且声明车间内运输时间可忽略（"inter-machine transport time is negligible"；"auxiliary times such as transport … are included in the processing time"）。全文 "AGV" 仅出现 1 次，在参考文献里（Hu, Yang, Xiao, and Wang 2023 的集装箱码头 AGV 路径规划）。缓冲只是"capacity sufficient"的无限暂存区，无容量约束、无拥堵、无车辆资源。**vs Ours：真派车（空驶/拥堵/车辆资源/电量）完全缺失。**
   - **几何/度量进网络：无。** 状态是 jobs×machines×3 通道张量（加工时间/完工时间/机器利用率），图节点特征仅 2 维（是否已排 + 完工时间下界），**全篇无坐标、无距离、无布局、无车间拓扑**（关键词 layout / coordinate / 距离均 0 命中，唯一 "distances" 出现在"初始随机状态到收敛区间的距离"这一无关语境）。最接近的表述是 §4.1.2 "**Owing to translational invariance and local receptive fields**, the network efficiently handles spatially structured data"——这是**CNN 的标准套话**，其"空间结构"指 jobs×machines **抽象矩阵的行列索引**，不是物理度量；**既非"度量张量注入注意力"，也非"标量几何特征"，而是根本没有任何几何量**。**vs Ours：几何/度量感知注意力（PCA 主轴、char_len 归一、距离偏置）在此文完全无对应物，差异化成立。**
   - **分批（batching）决策：无。** 全文 "batch" 仅 3 次命中，全部是神经网络训练 minibatch（Table 2 的 Batch Size 列）。工件约束明写 "each operation can be processed on only one machine and cannot be interrupted"、"Processing tasks on a machine are **non-preemptive**"、"each machine can handle at most one operation at a time"。**冲压/落料工艺本有天然的套裁/合批（blanking nesting）语义，本文完全未建模。** **vs Ours：分批决策完全缺失。**
   - 其他差异：⑥ 它的"dynamic/uncertain"只存在于标题，实验环境为确定性静态（无故障、无插单、无波动），四实例均为随机生成；我们 = 动态扰动全谱 + 参数化布局采样；⑦ 它有 critic（PPO-clip 系），我们无 critic 组相对（GRPO 族）；⑧ 它的"GNN"无任何图算子公式且 §4.1.2 自述用 CNN，我们的几何注意力有完整公式。
3. **是否威胁**：**不构成新颖性威胁，反而是三重反向证据。** ① 它是 IJPR 2026 的新作，却是"无物流环节、无几何/度量信息、无分批环节"的纯排产论文——可作为"域内主流工作尚未触及我们的三条差异化"的**时效性证据**（引作 "even in the most recent IJPR work on whole-vehicle stamping scheduling, transportation is assumed negligible 'inter-machine transport time is negligible' and no geometric/layout state is used"）。② 它的方法学薄弱（8 个公式、无 PPO/GNN 公式、无 episode 数、无置信区间、Table 3 最优格点挑选、Table 4 r̄ 缺 BB 列、规则表 4 对重复、§4.1.2/§4.1.3 同名、r̄/ā 符号写反、§1 承诺"real production traces + 与调度规则对比"而 §5 全未兑现）——是**"域内论文只做算法替换、不做机制创新"的活样本**，可用于我们的 gap 论证（"既有工作把 PPO 换个壳就发顶刊"）。③ 风险点（需防守）：**它把"PPO + 规则选择 + 正交调参"包装成"智能调度架构"发表在 IJPR**，说明该刊对"应用场景新 + 算法常规"是接受的；审稿人可能用同类标准质疑我们为何需要 GRPO 而非 PPO。拆法：我们的差异化不在"换算法"，而在**三条此前无人建模的决策环节（分批/真派车/几何度量）** + **组相对优势的层级归因（无 critic）** + **可复现的参数化测试台**；并且我们有它完全没有的东西——**完整公式（PPO-clip 在此文没有公式）、收敛证据、统计检验、消融**。
4. **可直接引用**：
   - "**inter-machine transport time is negligible**"（§3.2）／"auxiliary times such as transport, tool and fixture changes are included in the processing time"（§3.2）——"**物流被折进加工时间**：这正是我们'真 AGV 派车'环节要切割的对象；IJPR 2026 的整车冲压新作仍以'运输可忽略'为前提"。
   - "Machines operate continuously, and **unexpected breakdowns are disregarded**"（§3.2.2 假设 2）＋"**(5) All jobs are available at the initial time t = 0**"（假设 5）——"**其'动态调度'是标题级的**，实验环境为确定性静态；我们的动态扰动全谱与参数化布局采样是其能力的严格超集"。
   - "the PPO algorithm is **extremely sensitive to parameters**, its performance variation amplitude is large and wide-ranging"（§5.1）——"**连论文作者自己都承认 PPO 参数敏感、方差巨大**；我们的组相对优势设计正是为降方差而来（其 Table 3 重算 sd：PPO 69.3 vs DQN 系 6.3–7.4，恰好量化了这一点）"。
   - "**the problem comprises two sub-problems: (1) sequencing of processing tasks … and (2) task allocation, i.e. determining the machine assignment**"（§3.2.2）——"**只有两层（排序+机选），没有批量层、没有物流层**；我们的 π_B·π_S·π_L 是其两层结构的严格扩展"。
   - （备选，用于自证对比的诚实性）"using the exact algorithm value as a percentage reference"＋"The exact algorithm **cannot learn multiple scheduling rules** in experiments, so the experimental constraint is limited to the shortest processing time."——"**其 ratio 分母的精确算法被限死在 SPT 单规则**，故 92.5%–94.6% 这类数字不可与我们同口径比较"。

### 可复用性

- **①可直接借用**（低风险、拿来即用）：
  - **正交实验调超参（Taguchi orthogonal array）作为附录补充实验**：4 因子（optimiser / lr / γ / pooling-or-hidden size）多水平，报 range R 排序。它给了完整的 49 行原始表，可直接引用作为"正交设计用于 DRL 排产超参"的先例。
  - **奖励 = 机器利用率增量（U = Σ OT/(m·C_max)，r = U′−U，R = Σr）**：结构简单，可作为我们奖励消融里的一个"利用率塑形"对照项（注意其目标-奖励错配：声明 min C_max 却只奖利用率——**可作为反面教材引用**）。
  - **规则库对照物**：Table 1 的 12 条规则（去重后 8 条）是一份现成的派工规则清单，可用于我们的规则基线集合（SPT / LPT / argmin(p+R) / argmax(p+R) / argmin(p·R) / argmax(p·R) / argmin R / argmax R）。
  - **"target and online networks are identical" 的公平比较声明**：可直接引作"基线算法共用同一网络骨架以保证公平"的表述范式。
- **②需改造**：
  - 其"GNN 状态编码"**无法复用**（无结构定义、无公式、正文自述为 CNN）。若要在我们论文里把它列为"GNN 系对照"，必须明确写"该文只给出概念描述，未给出可复现的图算子"——否则构成对读者的误导。
  - 其"PPO 选规则"动作语义：若移植为我们的基线，需改造为**在因子化动作空间上做规则级 PPO**，而非工件级直接选择；改造要点 = 保持我们的事件驱动决策时点不变，仅将 π 的头部输出从"实体打分"换成"规则打分"。**建议改造而非直接引用**，因为若原样移植，其动作空间（8–12 条规则）与我们的实体级动作空间不可比。
- **③移植为基线工作量**：**约 4–6 人天**（估算依据：无开源代码，需从零实现；但环境极简——4 台机器/四工序/无运输/无故障、无抢占、无限缓冲，SimPy 环境下建模 ≤1 天；PPO+CNN 编码器 1–2 天（该文无 GNN 实现可抄，直接用 MLP/CNN）；正交调参 49 组 × 3 次 ≈ 2 天算力机时；结果整理 0.5 天）。**若还要复刻"三条并行冲压线 × 四机群"的 12 机群版本，再加 1–2 人天**。风险提示：因缺 episode 数与收敛判据，需自行设训练预算并声明"原值未公开"，否则复现结果不可比。
- **是否有开源代码**：**无。** 全文无 GitHub/代码链接；Data availability statement 仅称 "available … upon reasonable request"。

---

## 事实核验与内部矛盾（写作时的"防守清单"）

以下 8 条均为逐词核验（PyMuPDF 坐标级检查）得出，**可直接用于回应审稿人或说明"此文献不足为凭"**：

1. **无 PPO 公式**：全文无 clip objective、无 advantage、无 entropy、无 value loss；对 PPO 的全部描述只有 §4.1.3 一句话。
2. **无 GNN 公式**：无 GCN/消息传递/聚合公式；§4.1.2 明写状态张量由 **CNN** 处理（conv + pooling + FC），与标题/摘要的 "GNN-based" 冲突。**这是"名不副实"的核心证据。**
3. **Table 4 的 r̄ 行在四个环境下均缺 BB（exact）列**，即 4 值对 5 列；ā 与 ratio 行为 5 值。经逐词 x 坐标核验确认非提取错误。
4. **Table 1 的 12 条规则中有 4 对公式完全重复**（SPT+SSO≡SPT∗TWK；LPT+LSO≡LPT∗TWK；SROP≡SRPT；GROP≡LRPT），去重后仅 8 条。动作空间规模被虚报。
5. **超参表述自相矛盾**：正文称最优 "**discount factor = 0.01**"，但 Table 2 的 γ 水平为 {0.1, 0.3, 0.5, 0.6, 0.8, 0.99}，**不含 0.01**；0.01 是 lr 的水平之一。疑为 lr/γ 串位，**真实 γ 无法确定（未核实）**。
6. **符号写反**：§5.2 "four groups of experiments are of different scales of **path length (r̄)** where **path length (ā)** is the total completion time"——同一句内 r̄ 与 ā 指代冲突；Table 4 表头亦写作 "pathlengths" 却对应完工时间量级（661–1206），与 "path length" 语义不符。
7. **§4.1.2 与 §4.1.3 标题完全相同**（"State-Data dimension construction"），且 §4.1.3 内容为 DRL 算法介绍，与标题无关。
8. **实验与承诺不符**：§1 承诺 "synthetic and **real production traces**" 与 "against **benchmark dispatching rules and state-of-the-art RL schedulers**"，但 §5 只有 4 个随机生成实例，**无真实产线数据，无任何调度规则/元启发式对照**。
9. 补充：**全文未报告 episode 数、训练时长、收敛判据**；**未报告任何标准差或置信区间**（vs 我们已有 95% CI 与多 seed 协议）。

---

## 速答：四个关键问题（对应我们的三条差异化）

### 1. 它是不是 DRL？具体用什么算法？

**是 DRL。算法 = PPO（PPO-clip 系，Actor-Critic 框架）+ 一个"状态编码器"。** 但有三点必须同时说明：
- 它自称 **GPVSM = Graph Neural Network + PPO**（"This paper proposes a whole-vehicle stamping production scheduling algorithm based on Graph Neural Network and Proximal Policy Optimization (GPVSM)"）；
- **但正文从未给出任何图算子/PPO 的数学定义**，且 §4.1.2 明写编码器是 **CNN**（"processed by a CNN for feature extraction"，含卷积/池化/全连接）。因此严格表述应为："**声称 GNN+PPO，实为 PPO + CNN 张量编码 + 规则选择，无任何图神经网络实现细节可考**"。
- 它是 **on-policy PPO 做高层"规则选择"**（动作 = 8–12 条派工规则之一），不是端到端选工件/机器。
- 对照基线也是它自己换算法的产物：DQN / Dueling DQN / DDQN / PPO / A2C（A2C 仅在收敛图中）。
- **与 Ours 的关系**：同为 PPO 之后的策略梯度族，但**它是 critic 系（PPO-clip）**，我们是**无 critic 的组相对（GRPO 族）**——两族正交，不冲突；且它自己承认 PPO 参数敏感、方差大（其 Table 3 重算 sd = 69.3，是 DQN 系的 ~10 倍）。

### 2. 它有没有运输/AGV 决策？

**没有。完全无运输/AGV 环节。**（明确回答"**无**"）
- 原文证据 1（§3.2 建模约定，逐字）："auxiliary times such as **transport**, tool and fixture changes **are included in the processing time**; each machine can handle at most one operation at a time; **inter-machine transport time is negligible**."
- 原文证据 2（§3.2.2 假设 3，逐字）："(3) The processing time of a workpiece comprises setup time, **transportation time**, and machining time;"——运输被吸收为加工时间的一部分，**不进决策状态、不进奖励、不进约束**。
- 原文证据 3：缓冲只是"Every machine has an associated buffer with **sufficient capacity** for work-in-progress storage"——**无限容量暂存区，无容量约束、无排队、无拥堵**。
- 原文证据 4：全文 "AGV" 仅 1 次命中，且**出现在参考文献**（Hu, H., Xurui Yang, Shichang Xiao, and F. Wang. 2023. "Anti-conflict AGV Path Planning in Automated Container Terminals…"），正文从未建模车辆。
- **故：既不是"真派车"（无空驶/无拥堵/无车辆资源），也不是"固定运输时间/仅分配"——是连运输时间都被声明可忽略的"零运输"设定。** 与 Ours 的**真 AGV 派车**构成最干净、最无争议的差异化切面。

### 3. 它有没有几何/布局信息进决策状态？

**没有。**（明确回答"**无**"）
- 状态 = disjunctive graph → **jobs × machines × 3 通道**张量（processing-time / schedule-result / machine-utilisation）；图节点特征**仅 2 维**（是否已排 + 完工时间下界 f_a(O_jh, s_t)）。全部为**时间/状态类标量**，无一个几何量。
- 关键词核验：**layout = 0 次、coordinate = 0 次**；"distance" 仅 1 次且语境为"初始随机状态到收敛区间的距离"，与车间几何无关。
- 最接近几何的表述是 §4.1.2 "**Owing to translational invariance and local receptive fields**, the network efficiently handles spatially structured data and generalises well even with small samples."——**这是 CNN 的通用套话**，其"空间结构"指 jobs×machines **抽象矩阵**的行/列索引，**不是物理空间**。
- **判定：既非"度量张量注入注意力"，也非"标量几何特征"——是零几何。** 我们"几何/度量感知注意力（PCA 主轴 + char_len 归一 + 距离偏置 b(i,j)）"在此文**无任何对应物**，差异化主张完全成立，且此例证可用于"即便 IJPR 2026 的车间级新作也完全不含布局/度量信息"。

### 4. 它有没有分批（batching）决策？

**没有。**（明确回答"**无**"）
- "batch" 全文仅 3 次命中，**全部为神经网络训练的 minibatch**（Table 2 的 "Batch Size" 列：32/64/128/256），与生产分批无关。
- 反证（§3.2 建模约定，逐字）："each operation can be processed on **only one machine** and **cannot be interrupted**"；"(4) Processing tasks on a machine are **non-preemptive**"；"**each machine can handle at most one operation at a time**"。
- 值得注意的是：**落料/冲压工艺本有天然的套裁（nesting）与合批（同材质/同厚度板料共模）语义**，本文作为"整车冲压"专文却**完全未建模**——这既是我们"分批决策"环节的差异化证据，也可作为"领域常识未被既有工作吸收"的论据。

---

## 批4 综合结论速览（写作提示）

> ⚠️ **状态（2026-09-29）**：以下判定中的"轴"表述已按术语纪律改写为"环节 / 几何·度量信息"（见 progress-log §12.3b）；GPVSM / TAAGNet 的事实、原文摘录与"三连空缺"论据仍有效。凡以"层级归因（LoTV 剔除共享分量）"为我方技术卖点的表述，已随 M6 否决树式分层而下调——现役口径见 progress-log §7。

1. **GPVSM 的定位**：IJPR 2026 在线优先的"整车冲压 + PPO"新作，**期刊层级高、方法学密度极低**（全文 8 个公式，0 个 PPO/GNN 公式）。它的价值不在方法，而在**时效性证据**——"截至 2026 年 3 月，IJPR 上最新的整车冲压调度工作仍然：运输时间可忽略、无布局/几何状态、无分批、无故障、无插单"。
2. **三条差异化逐条命中"无"**：真 AGV 派车 **无**（"inter-machine transport time is negligible"）；几何/度量进网络 **无**（状态仅 3 通道时间类张量，节点特征 2 维；"translational invariance" 是 CNN 套话，非几何注入）；分批决策 **无**（"non-preemptive"，batch 仅指训练 minibatch）。**三连空缺，可逐条引用原文**。
3. **可作为"域内算法替换式研究"的批判样本**：PPO 换掉 DQN 即称"架构创新"；Table 3 的 49 组正交实验中 PPO **均值最差（1199.8 vs 1167.0）、方差最大（sd 69.3 vs 6.3–7.4）**，仅靠 best-of-grid 挑出最优点；Table 4 的 ratio 分母是**被限死在 SPT 单规则**的"精确算法"。→ 我们写 gap 时可用"既有工作以参数敏感性极高的算法替换 + 网格择优作为主要证据"一语概括。
4. **风险与拆法**：唯一需防守的是"**IJPR 接受'应用场景新 + 算法常规'**"这一事实，可能被审稿人用来质疑我们为何必须用 GRPO 族。拆法 = 强调我们贡献在**决策环节**（分批/真派车/几何度量，此三者本文全无）与**优势估计的组结构**（无 critic 的层级归因），而非"换算法"；并指出本文**连 PPO 的目标函数都没有写出**，不构成方法学对标物。
5. **引用建议**：**不列入我们的 DRL 对比基线表**（问题是零运输/零几何/零分批，且方法不可复现）；**只作为"域内最新现状（related work）"与"运输可忽略的过时假设"的引文使用**，引 2–3 句即可（首选 "inter-machine transport time is negligible" 与 "unexpected breakdowns are disregarded"）。若审稿人要求"与最新 IJPR 工作对比"，再以 4–6 人天移植一个"相同环境下的 PPO 选规则基线"作为附录。

---
---

### 批4 第二张卡：TAAGNet（ESWA 2026）——**本批最接近我们的一篇，威胁等级：中**

> 精读方式：PyMuPDF 全文提取（33 页，含 Table 1–11、附录 A/B/C 与 Algorithm 1 伪代码全读）。术语保留英文；引用均为原文摘录（≤2 句/处）。
> **阅读警示（先看这句）**：本篇与我们**在"生产-物流协同 + 异构图 + 注意力 + MARL"这三层上高度同构**，是本批唯一"必须认真对待"的对手。但它在**几何/度量**上完全空白、在**分批**上完全空白、在**空驶/拥堵/路径冲突**上自认未做；且它的动作空间是**规则对选择**（非实体级端到端），策略网络虽是 GATv2 但注意力分数里**没有任何成对几何量**。**结论：不构成对我们"度量感知注意力"的机制性抢先，但显著抬高了我们 related work 的举证门槛。**

---

## 《TAAGNet: A graph-based multi-agent reinforcement learning framework for integrated production and AGV scheduling in dynamic hybrid flow shop with uncertain sequencing》

**期刊正式信息**：Expert Systems With Applications（ESWA，Elsevier），**Volume 306, 2026, Article 130683**；DOI: **10.1016/j.eswa.2025.130683**。Received 13 July 2025 / Revised 22 November 2025 / Accepted 1 December 2025 / **Available online 6 December 2025**。
**作者与机构**：Weixiang Xu^a、**Xiaochuan Luo^a,b（通讯，luoxch@mail.neu.edu.cn）**、Yejian Zhao^a、Yulin Zhang^c。
^a College of Information Science and Engineering, Northeastern University, Shenyang 110819, China（东北大学信息科学与工程学院）；^b State Key Laboratory of Synthetical Automation for Process Industries, Northeastern University（流程工业综合自动化全国重点实验室）；^c University of Picardie Jules Verne, Saint Quentin, 02100, France。
**资助**：National Key Research and Development Program of China（2024YFB3312100）；National Natural Science Foundation of China（U24A20100）。
**代码/数据**：**无开源代码**（全文 "github" 0 命中、无代码链接）。Data availability statement 只描述数据构造（"The training datasets were constructed from fixed-seed preset scenarios … testing datasets were exclusively derived from stochastic configurations initialized with distinct random seeds."），**未提供仓库、未提供实例文件**（附录 C 以文本形式给出了少量静态实例的工序/加工时间/交期与 GA 调度解，**这是本文唯一可复现的部分**）。

### 精确贡献

原文自报三条（§1，Abstract 为同一批的压缩版）：

1. "**Unified heterogeneous graph with explicit queue modeling**: We propose a novel heterogeneous graph to represent the workshop state, which uniquely captures not only the relationships between jobs, machines, and AGVs, but also the explicit, dynamic queuing sequences for both production and transport resources."
2. "**Decoupled multi-agent collaborative decision architecture**: … comprising a MA and an AA. Both agents share a TAAGNet … However, they learn independent decision policies for machine assignment/sequencing and AGV assignment/sequencing, respectively, achieving decoupling and coordination of complex decisions."
3. "**Separated contribution-based reward function design**: We design separate reward functions for the machine and AGV agents. This mechanism not only evaluates the immediate impact of each decision on job timeliness but also incentivizes collaboration between agents by quantifying the contribution of decisions to resource load balancing, effectively addressing the multi-agent credit assignment problem."

**实质贡献核验**：贡献 = ①**把"队列顺序"显式编码成有向类型边**（此前 Moon 2023 / Zhang 2023 / Yuan 2025 的异构图只建"资格边"）；②**两个异构 agent 各自选复合派工规则对**（MA/AA 分离）；③**手工设计的分离式贡献奖励**。三点都是**表征层/工程层**创新，**无一条涉及几何度量、分批或空驶/拥堵**。
- 与既有工作的划界（原文自陈，可用于我们的 novelty 论证）："Existing models, often rooted in the classical disjunctive graph paradigm, represent the system state by modeling eligibility relationships (e.g., which jobs can be processed by which machines) and **dynamically deleting edges to represent resource occupation**. This approach, while valid, is insufficient … it **fails to explicitly capture a critical piece of state information: the ordered queuing sequence for contended resources**."

### 关键公式/模型

**问题模型（§3.2）**：目标 **min 总拖期**（非 makespan，因动态到达下 makespan 不适用）：`Minimize Z = Σ T_i`（式 1），`T_i = max(C_{i,n_i} − D_i, 0)`。
- 运输时间（**唯一出现距离的建模式**）：`τ(W_u, W_v) = dist(W_u, W_v) · t_unit`，其中 "`dist(W_u, W_v)` be the **distance (measured in segments)** between work centers `W_u` and `W_v`"。**注意：全文从未给出该距离矩阵的任何具体数值，也未说明车间是线形还是网状布局——"segments" 是唯一线索（几何布局未核实）。**
- 运输完工：`CT^trans_{i,k} = ST_{i,k} + Σ_a Y_{i,k,a} · τ(W^k_i, W^{k+1}_i)`（式 6）——**只计"载货段"里程**。
- 优先级：`S_{i,1} ≥ Arr_i`（式 2）；`S_{i,k} ≥ CT^trans_{i,k−1}`（式 3，**机-车耦合的唯一硬约束**）；`ST_{i,k} ≥ C_{i,k}`（式 4）；机/车容量互斥（式 9、10）；非负（式 11）。分配唯一性 `Σ X_{i,k,m} = 1`（式 7）、`Σ Y_{i,k,a} = 1`（式 8）。
- **建模假设（§3，逐字）**："(1) all of the processing operations are **non-preemptive**; (2) no need to rework; (3) yield losses are not considered; **(4) buffer capacity is infinite**; (5) machines and AGVs can setup immediately; (6) the time for jobs to enter and leave the system is ignored."

**状态（§4.1.2）= 单张异构图 `G^{H,t}_scene = (N, E)`，节点=作业/机器/AGV 三类，每类原始特征 5 维（逐维照录，这就是全文的全部状态）**：

| 节点 | 5 维原始特征（**逐维**） |
|---|---|
| 作业 `J_i` | `J_cpt`（当前工序加工时间；被 AGV 运输时置 −1）、`J_rpt`（剩余总加工时间）、`J_ttd`（到交期剩余时间）、`J_slack`（松弛 `S_i = TTD_i − RPT_i`）、`J_wt`（当前等待时间） |
| 机器 `M_k` | `M_idx`（**全局编号**）、`M_pf`（加工状态 0/1）、`M_ct`（队列内总加工时间）、`M_at`（到机器可用的剩余时间）、`M_util`（利用率） |
| AGV `A_a` | `A_idx`（**全局编号**）、`A_tf`（运输状态 0/1）、`A_ql`（运输队列长度）、`A_eft`（**估计总运输时间**，= Σ_{J∈A_o} τ_{i,o}）、`A_util`（利用率） |

- **编码**：`M^t_k = concat([1,0,0], M^{t,raw}_k)`、`A^t_a = concat([0,1,0], A^{t,raw}_a)`、`J^t_i = concat([0,0,1], J^{t,raw}_i)`（3 维 type id 拼接）。Table 2 记 "Machine/AGV/Job node feature dimension = **5**"，与"5 原始 + 3 type id = 8"口径略有出入（**未核实**，疑 Table 2 只计原始特征）。
- **边（只有 4 种离散类型，无任何连续属性）**：type 0 = 机器队列首作业→机器（红）；type 1 = 机器队列内后继作业→前驱作业（绿）；type 2 = AGV 队列首作业→AGV（黄）；type 3 = AGV 队列内后继作业→前驱作业（紫）。**边只有"类型"这一个属性，没有任何距离/时长/权重**。
- **MDP 转移是异步的（§4.1.5）**：事件驱动（工序完工 / 运输完成为决策点），非等步长。

**网络 TAAGNet（§4.2）**（逐条照录）：
1. **类型专用编码器**：`h^{(0),k}_v = ReLU(W^k_enc X^{raw,k}_v + b^k_enc)`（式 23）。
2. **边类型嵌入**：`e^{(0)}_uv = Embedding(e^type_uv)`，离散边型 → `R^{d_h}`，再过 LayerNorm（式 24）。
3. **GATv2Conv 双层**（Brody et al., 2021）：`s^{(1)}_vu = a^{(1)T} LeakyReLU(W^{(1)}_att [h^{(0)}_v ‖ h^{(0)}_u ‖ e^{(0)}_vu])`（式 25），`α = softmax_u(s)`（式 26），聚合（式 27），**8 头**（式 28）。
4. **全局注意力模块**：`A_global = softmax(QK^T / √d_k · M_batch)`（式 30，`M_batch` 为同图实例掩码），`H_global = A_global V W^O`（式 31），**加权残差融合 `X^{(1)}_res = X^{(1)} + β·H_global`，β = 0.1**，再过降维 MLP。
5. **第二层 GATv2** + skip connection（式 32）。
6. **池化**：对 machine / AGV / job 三类分别 mean pooling（式 33），三类结果再 mean → `p^type_b`；同时全局 mean pooling → `p^global_b`；拼接 `p^final_b = [p^type_b ‖ p^global_b]` → **MLP 输出 Q 值**（式 34）。
7. 损失 = **Smooth L1 Loss**；优化器 **AdamW**；lr **10⁻³ → 10⁻⁴ 几何衰减**。

**动作空间（§4.1.3）——关键：不是实体级选择，而是"规则对选择"**：
- MA：`a_MA = (r_MA_assign, r_MA_seq)`；`R_MA_assign = {CT, TT, SQ}`（3 条：最早完工 / 队列总加工时间最短 / 队列最短）；`R_MA_seq = {MOD, CRSPT, DPTLWKRS, PTWINQS}`（4 条）→ **12 个复合动作**，`Q_b ∈ R^{3×4}`。
- AA：`a_AA = (r_AA_assign, r_AA_seq)`；`R_AA_assign = {SPTA, STTT, MQL, ND}`（4 条：最短取货时间 / 最短总运输时间 / 最短队列 / **最近距离 ND**）；`R_AA_seq = {MS, CR, NPTWINQS, EDD}`（4 条）→ **16 个复合动作**。
- **ND 规则原文（距离进入决策的唯一入口）**："(4) **ND (Nearest distance AGV)**: Select the AGV closest to the job's current pickup location. If the AGV is transporting, the distance is calculated from its **destination** to the job's pickup point; if idle, from its **current location**."
- **SPTA 规则原文（空驶时间的唯一入口）**："The pickup time is estimated as the time for the AGV to complete all tasks in its current queue, **plus the travel time to the current job's location to perform the pickup**."

**奖励（§4.1.4，分离式贡献奖励，逐式照录）**：
- MA 分配奖励：`r^MA_assign = −tanh(ΔVar^MA_{W_c} · λ^MA_LB)`（式 16，`λ^MA_LB = 0.01`），ΔVar = 分配后减分配前的工作中心内机器累计加工时间方差。
- MA 排序奖励：`r^MA_seq = ΔS^MA_i · F^MA_i / λ^MA_{S_scale}`（式 18），临界度 `F^MA_i = (1 − Ŝ/(|Ŝ|+α^MA)) · (0.7 + 0.3·κ_proc,i)`（式 17，**α^MA = 50**）。
- `r^MA_t = r^MA_assign + clip(r^MA_seq, −1, 1)`（式 19）。
- AA 分配奖励：`r^AA_assign = −tanh(ΔVar^AA_A · λ^AA_LB)`（式 20，`λ^AA_LB = 0.1`）。
- AA 排序奖励：`r^AA_seq = (r^{AA_seq}_{slack} + r^{AA_seq}_{wc_load}) · λ^AA_{SEQ_scale}`（式 21，`λ = 0.1`），其中 `r_{slack} = (τ̄_loser·F^AA_chosen) − (τ_chosen·F̄^AA_loser)`，`r_{wc_load} = L̄^next_{Wc,loser} − L^next_{Wc,chosen}`。
- `r^AA_t = r^AA_assign + clip(r^AA_seq, −1, 1)`（式 22）。
- **明确拒绝 QMIX 式值分解（原文）**："We **abandon** the approach of explicitly learning joint Q-functions and mixing networks … This design explicitly attributes contributions to specific resources to address multi-agent credit assignment, rather than relying on implicit value decomposition."

### 实验设置

- **仿真**："The simulation environment for the production system was developed using the **SimPy** library … based on a **discrete-event modeling** paradigm."；实现用 **PyTorch + PyTorch Geometric**。
- **默认场景（Table 2 "Scene"）**：作业加工时间 `U[5,25]`；目标系统利用率 0.8（实验另测 **70% / 80% / 90%**）；**交期紧度因子 3**；**单位运输时间 = 2**（泛化时测 1 与 3）；**机器 6 台**；**AGV 3 台**；**工作中心 3 个**（每中心 2 台并行同构机）。
- **动态性**：作业**泊松到达**、到达时刻不可预知；`E(load factor) = E(t)·w / (E(interval)·m) × 100%`（式 35）；交期 `TTD^base_i = α_i · Σ t_{i,l}`、`D_i = NOW + TTD^base_i`（式 36–37）。
- **评测协议**：**100 次 Monte Carlo 仿真**，每次重采样到达间隔/加工时间/交期紧度，**同一组参数在所有方法上跑同一套**，仿真时长 **1000 时间单位**；三种负载对应约 **110 / 124 / 137 个作业到达**。
- **训练规模**："The training process spans a total of **100,000 simulated time units**"；AA 的 loss "stabilizing at an extremely low level after approximately **5000 iterations**"。
- **超参（Table 2）**：attention heads 8；MLP 隐层 2、宽度 64；dropout 0；γ = **0.8**；batch = **128**；replay = **500,000**；action network 更新频率 5；target network 更新频率 250；ε **0.5 → 0.2**。
- **算力**："an **Intel i7-14700K CPU** and an **NVIDIA RTX 3090 GPU**"；离线训练 **约 8 分钟/单 agent**（对比 DRL1 约 3 分钟、DRL2 约 2 分钟）。
- **基线（三类，列全）**：
  1. **规则组合**：先单因子扫描 4 条机器分配规则（Table 4：CT 1576.35 / TT 1648.31 / SQ 1752.28 / **UT 7326.84**）、机器排序规则（Fig. 6）、5 条 AGV 分配规则（Table 5：STTT 1238.30 / SPTA 1242.25 / EAT 1264.28 / **ND 1270.08** / MQL 1274.89）与 AGV 排序规则（Fig. 7）；最终取 8 个组合（CT、CRSPT/DPTLWKRS、SPTA/STTT、MS/CR 的交叉）。
  2. **两个复现的 DRL 方法**：DRL1 = Li et al. (2025b)、DRL2 = Liu et al. (2022)，**作者在自己环境里重实现**（"we re-implemented their core logic within our simulation environment"）。
  3. **元启发式（仅静态）**：GA（种群 100、**最大代数仅 5**、交叉 0.8、变异 0.1）、SA（500 次迭代、T₀=100、降温 0.95）、PSO（种群 30、100 次迭代）。
- **指标**：RTTA（相对总拖期优势，**基线恒为 FIFO+CT+SPTA+MS**，"the larger, the better"）与 **Win rate**；静态实验另加 **tardy job 计数**。
- **统计**：**全文无显著性检验、无置信区间、无标准差数值**（"significance" 0 命中、"confidence" 0 命中、"standard deviation" 0 命中）；离散度只用**箱线图**呈现（100 次 Monte Carlo 的分布）。
- **泛化测试**：机器数/工作中心数/单位运输时间三个维度各自增减，另加**大规模场景零样本部署（不重训）**。

### 核心数字

**动态基准（§5.1，相对最优规则组合的改进幅度，原文自报）**
- 仅 MA 决策：**约 4–12%**（vs DPTLWKRS+CT 与 PTWINQS+CT）
- 仅 AA 决策：**约 2–9%**
- 双 agent 联合：**约 3–12%**

**Table 6（与近年 DRL 方法对比，平均总拖期；表头为 "DRL1/TAAGNet"、"DRL2/TAAGNet"，即每格 "基线 / 本文"）**

| 场景（工作中心, 机器, AGV, 利用率, 单位运输时间） | DRL1 | **TAAGNet** | DRL2 | **TAAGNet** |
|---|---|---|---|---|
| 3, 6, 3, 80%, 2 | 1243 | **1216** | 1228 | **1223** |
| 3, 6, 3, 80%, 3 | 1705 | **1666** | 1673 | **1664** |
| 3, 6, 3, 90%, 2 | 2572 | **2526** | 2546 | **2526** |
| 3, 12, 3, 80%, 2 | 702 | **673** | 677 | **668** |
| 5, 10, 5, 80%, 1 | 1791 | **1732** | 1739 | **1726** |

- 10 组对比全部取胜，但**优势幅度仅 0.2%–4.3%**（最大为 3,12,3 场景的 702→673 = 4.1%；最小为 1228→1223 = 0.4%）。**无方差、无重复次数、无检验。**

**Table 8（大规模零样本泛化，平均总拖期；括号内为最强 PDR 组合）**

| 场景（工作中心, 机器, AGV, 单位时间） | MA | AA |
|---|---|---|
| 3, 9, 4, 2 | **1075** (1085) | **1071** (1076) |
| 3, 12, 5, 2 | **1061** (1068) | **1065** (1068) |
| 6, 12, 6, 1 | **2325** (2350) | **2381** (2407) |
| 7, 14, 7, 1 | **3206** (3215) | **3235** (3251) |

- 原文泛化结论："in a large-scale scenario with **7 work centers, 14 machines, and 7 AGVs**, TAAGNet continues to outperform the best-performing PDR combination."（优势 0.3%–1.2%——**已接近噪声量级**）

**Table 7（消融，默认场景 MA / AA 拖期；基准 1223 / 1212）**

| 消融项 | MA | AA |
|---|---|---|
| **TAAGNet（完整）** | **1223** | **1212** |
| Remove machine load info | 1268 | 1229 |
| Remove job slack info | **1275** | 1231 |
| Remove machine/AGV utilization | 1251 | 1233 |
| Remove machine/AGV status flag | 1249 | 1235 |
| Remove assignment reward | **1280** | 1234 |
| Remove sequencing reward | 1234 | 1229 |
| Remove 'tanh' from assignment reward | 1256 | 1236 |
| Remove slack benefit from r_seq (AA) | – | 1228 |
| Remove load benefit from r_seq (AA) | – | 1230 |
| Remove edge connections | 1250 | **1256** |
| Remove graph attention mechanism | 1265 | 1245 |
| Remove global interaction module | 1271 | 1234 |
| Remove type-aware pooling | 1267 | 1244 |
| 推理用 ε = 0.05 / 0.1 | 1274 / 1250 | 1240 / 1246 |

- **可引用**：AA 侧所有消融的退化幅度都在 **1.3%–3.6%**（1228–1256 vs 1212），**远小于 MA 侧**——即"AGV agent 的图注意力与奖励结构贡献微弱"。且 **MA 侧"移除整层图注意力"（1265）与"移除一个机器负载标量特征"（1268）的退化几乎相同**，说明该文图注意力对排产决策的边际价值与其一个手工标量特征相当。

**静态实例对比（Table 10 + §5.4，元启发式 vs TAAGNet）——本文最重要的负面结果**
- 10 / 15 / 20 作业各 10 个随机实例；**GA、SA、PSO 在多数实例上把总拖期降到 0（RTTA = 100%）**；TAAGNet 的 RTTA 落在 **73.68%–100%**，tardy jobs 为 **1–12 个**（元启发式为 0–4 个）。
- 原文自述（逐字）："**the metaheuristics demonstrate a significant advantage in solution quality, achieving 10–20 % higher RTTA than TAAGNet across the tested static instances.**"；"the genetic algorithm can **reduce the total tardiness to zero for most small- to medium-scale problems**."
- **→ 即：一旦允许离线全局搜索（哪怕 GA 只跑 5 代），本文方法在解质量上全面落后 10–20 个百分点。**

**Table 11（CPU 时间，秒）**

| 作业数 | 决策类型 | 规则 | TAAGNet | GA | SA | PSO |
|---|---|---|---|---|---|---|
| 10 | 单次决策 | 3.69e−4 | 6.21e−3 (MA) / 6.07e−3 (AA) | – | – | – |
| 10 | 全部决策 | 8.85e−3 | 0.382 | 2.16 | 1.44 | 0.41 |
| 15 | 全部决策 | 1.36e−2 | 0.577 | 3.32 | 2.24 | 0.64 |
| 20 | 全部决策 | 1.91e−2 | 0.874 | 4.67 | 2.89 | 1.79 |

- **可引用**：单次决策 TAAGNet 为 **6.2e−3 s**，是规则（3.7e−4 s）的 **约 17 倍**，但比 GA 的"整体重优化"（2.16–4.67 s）快 **2–3 个数量级**——**这是它唯一真正站得住的卖点（实时性），也是我们 DES 测试台可以对标的地方**。

### 自认局限（原文，逐字，§6）

1. **"Model fidelity and real-world complexity**: Although our model integrates AGV transport, real-world logistics systems involve more complex constraints such as **AGV battery management, path conflicts, and traffic control**. Effectively incorporating these fine-grained physical constraints into the graph representation and state transitions is a critical step toward enhancing the model's practical utility."
2. **"Depth of multi-agent collaboration**: The current framework achieves **implicit collaboration among agents through separated rewards**. Exploring more advanced multi-agent collaboration algorithms, such as **value decomposition or communication mechanisms under the Centralized Training Decentralized Execution (CTDE) framework**, could further enhance collaborative efficiency and resolve deeper resource conflicts."
3. **"Explainability and trust**: Deep learning models are often perceived as 'black boxes,' which is a barrier in industrial decision-making …"

**归纳**：它自认的局限 = ①**无空驶/无路网冲突/无交通管制/无电量**（"path conflicts, and traffic control" 完全未建模，仅列未来工作）；②**协作只是"分离奖励"的隐式协作**，未用 CTDE/值分解/通信；③可解释性。**它没有自认、但客观上存在的局限**：静态解质量输给 GA/SA/PSO 10–20%；无统计检验；无代码；几何布局从未给出。

### 与我们的逐点对比

1. **同（重合度是本批最高，必须正视）**：
   - ① **问题域几乎一致**：dynamic hybrid flow shop + **生产与 AGV 运输一体化调度**（多工作中心、每中心并行同构机、AGV 往返运输工序间工件）；
   - ② **异构图状态**：节点 = 作业 / 机器 / AGV 三类，且**显式建模队列顺序**（我们也是队列/工序/车辆实体图）；
   - ③ **图注意力骨干**：GATv2 双层 + 多头 + 全局注意力 + 类型感知池化（我们：轴向/几何注意力 + 池化）；
   - ④ **MARL + 分工决策**：两个异构 agent 分工（它 MA/AA，我们批次/工序/物流头）；
   - ⑤ **事件驱动、异步 MDP**：都在事件点（工序完工 / 运输完成）决策，非等步长；
   - ⑥ **自建参数化仿真测试台 + 泛化测试**（它 SimPy + 规模/运输时间扫描；我们 SimPy DES + 参数化布局采样器）；
   - ⑦ **都强调实时性/毫秒级在线决策**；
   - ⑧ 都用 DDQN/Q-learning 系的"无策略梯度"路线差（我们 GRPO 是策略梯度，**这一点上我们不重合**）。
2. **异（**我们的四条差异化，此文的命中情况**）**：
   - **① 几何/度量进网络：无。**（详见 Q2）状态 5 维节点特征里**无一个几何量**（`M_idx`/`A_idx` 是**编号不是坐标**）；边只有 4 种离散类型、**无边权**；注意力 `s_vu = a^T LeakyReLU(W[h_v‖h_u‖e_uv])` **无任何成对几何偏置**。全文 **layout = 0 命中、coordinate = 0 命中、Euclidean/Manhattan = 0 命中、invariant = 0 命中**。距离仅以两种方式"擦边"：**(a)** 数学模型中运输时长 `τ = dist·t_unit`（**但该距离矩阵全文未给任何数值，布局未核实**）；**(b)** 节点标量 `A_eft`（估计总运输时间）**聚合式间接**携带距离，以及动作空间里 **ND / SPTA 两条规则**的规则语义。**→ 判定：不构成"度量张量注入注意力分数"，二者不同构。**
   - **② 真派车：是"真派车"，但**空驶未形式化、无拥堵/无冲突/无路径规划**。**（详见 Q1）`Y_{i,k,a}` 是逐运输任务的车辆选择变量；ND 规则显式用"AGV 当前位置或目的地"到取货点的距离；SPTA 显式把"到取货点的空驶时间"计入估计。但式 (6) **只计载货里程**；"empty/deadhead/空驶"关键词 **0 命中**；collision / conflict / deadlock / path planning **全部 0 命中**；唯一 "congestion" 出现在引言的动机句（且语义是**机器队列积压**而非车辆路网拥堵）。**自认**："path conflicts, and traffic control" 属未来工作。
   - **③ 分批决策：无。**（详见 Q3）**"batch" 15 次命中全部是"训练 minibatch / 图 batch"**；无 sublot / wave / lot streaming；假设 (4) "**buffer capacity is infinite**"；each job 以整件为单位、非抢占。
   - **④ 组相对 / 无 critic 优势估计：无。**它是 **DDQN（value-based，两个独立 agent 各自 Q 网络）**；**明确拒绝 mixing network**；credit assignment 靠**手工奖励工程**而非**组内相对优势**。我们 = 无 critic 的组相对策略梯度。
   - 其他差异：⑤ **动作空间是"规则对选择"（12 / 16 个复合规则），策略无法直接对实体打分**——它学的是"何时用哪条规则"，不是"选哪台车/哪道工序"；我们 = 因子化条件头直接在实体上取分布。⑥ 它**在静态实例上输给 GA/SA/PSO 10–20%**；⑦ 它**无统计检验、无 CI、无代码**；⑧ 目标函数是**总拖期**而非 makespan。
3. **是否威胁：中（必须防守，但不致命）。**
   - **升到"中"的理由**：这是目前检索到的**唯一一篇与我们在"生产+AGV 一体化 + 异构图 + 图注意力 + MARL 分工"四层同时重合**的新作（ESWA 2026，一区 TOP），且它**确实实现了真派车**（`Y_{i,k,a}` + ND/SPTA 规则），并明确把"队列顺序"作为核心状态创新——**这与我们"物流实体图 + 队列/工序拓扑"的表述存在被审稿人认定为"已有工作"的真实风险**。
   - **没有升到"高"的理由（三条防线）**：
     1. **几何/度量信息完全空白**。它连"布局"这个词都没有；距离矩阵数值从未给出；注意力分数是 type-aware 而非 metric-aware。**我们"把度量张量注入注意力分数"的机制主张，在此文无任何对应物——这一点可以写死。** 且它连 AGV 的**当前坐标都不在状态里**（只有 `A_idx` 编号与 `A_eft` 聚合运输时间），意味着**策略无法在状态层面区分"同一台车的远近"**——它只能在动作层通过 ND 规则"委托"给启发式。这在方法上是一个**可被我们正面攻击的结构性缺陷**。
     2. **空驶/拥堵/路径冲突自认未做**（"path conflicts, and traffic control" 是对手自己写的 future work）。
     3. **分批环节完全空白**，且它把缓冲设为无限（"buffer capacity is infinite"），连容量约束都没有。
   - **反向利用**：它的 **Table 10 + §5.4 是"DRL 端到端调度在静态问题上打不过 5 代 GA"的硬证据**（"10–20 % higher RTTA than TAAGNet"），可支撑我们"不主张端到端 DRL 全面替代元启发式，而主张在**动态/实时 + 多环节耦合**场景下的优势"的定位。同时它的 **Table 7 显示 AA 侧（AGV）所有消融退化仅 1.4–3.6%**，可用于论证"**仅做图注意力对 AGV 决策的边际贡献有限，真正的增益来自决策环节与几何表征**"。
4. **可直接引用（中英对照）**：
   - "**real-world logistics systems involve more complex constraints such as AGV battery management, path conflicts, and traffic control**"（§6 局限 1）／"现实物流系统还涉及 AGV 电量管理、路径冲突与交通管制等更复杂的约束"——**用它自己的话证明"AGV 路网冲突/空驶未被建模"**。
   - "The current framework achieves **implicit collaboration among agents through separated rewards**."（§6 局限 2）／"当前框架仅通过分离奖励实现 agent 间的隐式协作"——**证明其多 agent 协作停在奖励工程层，未触及优势估计结构**（对照我们的组内相对）。
   - "**buffer capacity is infinite**"（§3 假设 4）／"缓冲区容量无限"——**证明无容量耦合、无波次门控、无分批**。
   - "the **metaheuristics demonstrate a significant advantage in solution quality, achieving 10–20 % higher RTTA than TAAGNet across the tested static instances**"（§5.4）／"在静态实例上，元启发式在解质量上具有显著优势，RTTA 比 TAAGNet 高 10–20%"——**DRL 阵营的自我承认**，可用于我们的定位论证。
   - （备选，用于"域内最新工作仍缺几何"的时效性举证）"This design choice … allows the AGV agent to 'see' the **downstream congestion**"（§2.4）——注意此句的 "congestion" 指**机器队列积压**，**不是车辆路网拥堵**，引用时须加注，否则会被反用。

### 可复用性

- **①可直接借用**：
  - **显式队列拓扑的异构图建模范式**（队列首→资源节点 type 0/2；队内后继→前驱 type 1/3）——这是一份干净的"把排队顺序编码进图结构"的现成配方，**且它自己论证了为什么资格边/析取图不够**（"dynamically deleting edges to represent resource occupation … fails to explicitly capture … the ordered queuing sequence"），这段论证可直接引作我们图的动机构建依据。
  - **贡献式分离奖励的两条公式**（`−tanh(ΔVar·λ)` 的负载均衡项；`ΔS·F/λ_scale` 的松弛-临界度项）——结构简单、可直接移植为我们奖励消融的对照项。
  - **AGV 侧规则库**（SPTA/STTT/MQL/**ND** × MS/CR/NPTWINQS/EDD）与**机器侧规则库**（CT/TT/SQ × MOD/CRSPT/DPTLWKRS/PTWINQS），以及 Table 3 的完整 20+ 条单规则清单（ATC/COVERT/MON/…）——**一份现成的派工规则基线集合**。
  - **评估协议**：100 次 Monte Carlo + 同参数跨方法复用 + 三档负载 + RTTA/win-rate 双指标，可直接对齐。
  - **实时性对比表（Table 11）的呈现范式**：单次决策 / 全部决策分开计时，规则 vs DRL vs 元启发式三方同表——**我们的 DES 测试台可直接产出同构表**。
- **②需改造**：
  - 其**动作空间（规则对）与我们（实体级因子化动作）不同构**，若要作为基线，须改造为"在因子化头之上允许规则作为额外动作"，或干脆只保留其**网络骨干**（类型编码 + GATv2 + 全局注意力 + 类型池化）而换成我们的动作头——**后者才是公平比较**。
  - 其 `A_eft` 是**聚合式距离信号**；若我们要做"几何消融对照"，需把它显式替换为坐标/距离偏置版本，才能构成消融。
- **③移植为基线工作量**：**约 6–9 人天**（无开源代码；SimPy 环境与我们同构度极高，1–2 天；TAAGNet 骨干（GATv2 ×2 + 全局注意力 + 类型池化）在 PyTorch Geometric 下有现成算子，2–3 天；规则对动作空间 + 分离奖励 1–2 天；100 次 Monte Carlo × 3 负载 + 泛化扫描的算力机时 1–2 天）。**风险提示**：原文未给距离矩阵数值，复现时须自设布局并声明"原距离定义（segments）未公开具体取值"。
- **是否有开源代码**：**无。** 无 GitHub 链接、无代码可用性声明；仅附录 C 给出少量静态实例与 GA 解。

---

## 速答：四个关键问题（TAAGNet）

### Q1. AGV 决策形式：真"派车"吗？有空驶/拥堵/路径规划吗？机-车耦合怎么处理？

**是"真派车"（有车辆选择 + 队列排序 + 车辆位置被规则使用），但空驶未进数学模型、拥堵/冲突/路径规划完全不存在。**

- **真派车的证据（形式化）**：`Y_{i,k,a}` 二元变量 = 每个工序间运输任务分配到具体 AGV（式 8：`Σ_a Y_{i,k,a} = 1`）；式 10 是 AGV 容量互斥；AGV 节点有队列 `A_o` 与队列长度 `A_ql`；AA 的动作含"选哪台 AGV 执行"与"AGV 空闲时从队列里选哪个任务"两个子决策。§3 明确："there are **no restrictions on which AGV can be selected for a transport task**"（全车可服务全任务）。
- **车辆位置被使用，但只在规则里**：ND（"If the AGV is transporting, the distance is calculated from its **destination** … if idle, from its **current location**"）与 SPTA（"plus the **travel time to the current job's location** to perform the pickup"）。**但 AGV 的当前坐标不在状态特征里**（AGV 节点 5 维：`A_idx / A_tf / A_ql / A_eft / A_util`），所以网络无法在状态层面对"远近"做条件化，只能整体选择"是否采用 ND 这条规则"。
- **空驶（deadhead）：部分存在、但未形式化。** SPTA 的"pickup time"里含空驶；但**式 (6) 运输完工只计载货段** `τ(W^k_i, W^{k+1}_i)`；全文 "empty" / "deadhead" / "idle travel" **0 命中**；空驶不进目标函数、不进容量约束。**标为"仅以启发式估计形式存在，未建模"（部分/未形式化）。**
- **拥堵 / 冲突 / 路径规划：全部没有。** "collision" 0、"conflict" 0、"deadlock" 0、"path plan" 0 命中；唯一 "congestion"（§2.4）语义是**机器侧下游队列积压**（"allows the AGV agent to 'see' the downstream congestion … prioritizing jobs destined for less congested work centers"），**不是车辆路网拥堵**。原文自认（§6）："real-world logistics systems involve more complex constraints such as AGV battery management, **path conflicts, and traffic control**"——**列为未来工作**。此外**无路网/布局定义**：`dist(W_u, W_v)` 只说 "measured in **segments**"，**全文未给任何距离数值、未给车间布局图**（layout 0 命中）。
- **机-车耦合怎么处理**：**唯一硬耦合是式 (3)**：`S_{i,k} ≥ CT^trans_{i,k−1}`（后道工序必须等前道运输完成）；**运输时间显式进约束**：式 (6) `CT^trans = ST + τ(W^k_i, W^{k+1}_i)`。**状态层面**：AGV 队列被显式编码为 type 2/3 边 + AGV 节点特征，即"运输资源占用"以**队形拓扑**进入图。**但机器与 AGV 之间没有联合动作、没有联合 Q、没有混合网络**——两 agent 各选各的规则对，靠**分离奖励**隐式协调（自认"implicit collaboration"）。还有一条重要简化：**假设 (4) buffer capacity is infinite**，所以机器侧没有排队容量瓶颈，机器-AGV 之间不存在"缓冲区满→AGV 无法卸货"这类强耦合。

### Q2. 几何/距离如何进网络？（最关键）

**核心判定：距离既不进状态特征、也不进注意力分数；它只以"聚合标量"与"规则语义"两种间接方式擦边。TAAGNet ≠ "把度量张量注入注意力分数"。**

- **状态里有没有位置/坐标/距离？——没有。**
  - 节点特征（**逐维**，§4.1.2，三类各 5 维）：作业 = `J_cpt / J_rpt / J_ttd / J_slack / J_wt`；机器 = `M_idx / M_pf / M_ct / M_at / M_util`；AGV = `A_idx / A_tf / A_ql / A_eft / A_util`。**逐维核验：无一个坐标、无一个距离、无一个方位。`M_idx`/`A_idx` 是"global index"（编号），不是几何位置。**
  - 关键词核验：**layout = 0、coordinate = 0、Euclidean = 0、Manhattan = 0、spatial = 0、invariant = 0、permutation = 0、symmetr = 0**。"distance" 仅 5 次命中：`dist(W_u,W_v)` 的定义（1 次）、ND 规则描述（2 次）、Table 3 的 ND 规则名（1 次）。**"location" 6 次中 4 次是 ND/SPTA 规则描述，2 次是英文词 "allocation" 的子串。**
- **标量喂 MLP 还是注意力偏置？——严格说两者都不是"几何注入"。**
  - 距离信息**只以聚合标量的形式**出现：`A_eft = Σ_{J_i∈A_o} τ_{i,o}`（"Estimated total transportation time of AGV A_o"），因 `τ = dist·t_unit` 而**间接**携带距离；类比地，`A_ql`（队列长度）携带拥堵的弱代理。这些标量经 §4.2 的**类型专用线性编码器 + GATv2** 进网络。**没有成对距离、没有距离偏置、没有坐标编码。**
  - **注意力分数逐字核验（式 25）**：`s^{(1)}_vu = a^{(1)T} LeakyReLU(W^{(1)}_att [h^{(0)}_v ‖ h^{(0)}_u ‖ e^{(0)}_vu])`；全局注意力（式 30）：`A_global = softmax(QK^T/√d_k · M_batch)`。**唯一的偏置项是 `M_batch`（同图实例掩码），唯一的关系输入是 `e^{(0)}_uv = Embedding(e^type_uv)`（4 种离散边型的可学习嵌入）。→ 无任何度量张量、无任何成对几何先验、无 RBF/距离核。**
- **图的节点/边特征（逐维，即上文表格）**：**节点 3 类 × 5 维原始 + 3 维 type id；边 1 维离散类型（0/1/2/3），无边权、无边特征向量。**（这是全文给出的全部图信息，无遗漏。）
- **不变性设计：无。** 全文无 translation/rotation/scale/permutation invariance 的讨论或设计（"invariant" 0 命中）。由于**状态里根本没有坐标**，平移不变性是"因为它没有几何所以平凡成立"，**不是设计出来的**。
- **距离进入"决策"的两条真实通路（必须诚实说明，否则会被审稿人反用）**：**(a)** 动作空间的 **ND 规则**（最近距离 AGV）与 **SPTA 规则**（最短取货时间，含空驶到取货点）——策略可以"选择使用距离感知的启发式"，但这是**动作语义层面的委托，不是表征层面的感知**；**(b)** 节点标量 `A_eft` 的聚合运输时间。**这两条都不同于也无法替代"把度量张量注入注意力分数"。**
- **→ 对我方机制主张的最终判定**：**不等同。** 我们"几何/度量感知注意力（坐标/距离→注意力偏置/成对先验，含不变性设计）"在 TAAGNet 中**没有任何对应物**；它是一篇 **type-aware** 注意力论文，不是 **metric-aware** 注意力论文。**可正面主张的差异化句**："TAAGNet 的节点特征不含任何坐标或距离，其 GATv2 注意力分数仅由节点嵌入与离散边型嵌入构成，距离仅通过一条聚合标量（估计总运输时间）与两条启发式规则（ND/SPTA）间接影响决策——**该文没有任何成对几何先验，也没有不变性设计**。"

### Q3. 有没有分批决策？

**没有。完全无生产分批（batching）环节。**
- **"batch" 全文 15 次命中，逐条核验：全部是"训练 minibatch"或"图 batch"**（如 "trained on mini-batches randomly sampled from this buffer"、"Sample a random mini-batch of N transitions"、Table 2 的 "Training batch size = 128"、"M_batch is a batch mask"、池化的 "for each batch b"）。**无一次指生产批量。**
- 相关词核验：**sublot = 0、wave = 0、lot streaming 相关 0 命中**（"lot" 5 次命中里 5 次都是 "plot" 的子串）。
- 反证（§3 假设，逐字）："(1) all of the processing operations are **non-preemptive**"；"(4) **buffer capacity is infinite**"；§3："**Each AGV can transport only one job at a time**"；"no need to rework; yield losses are not considered"。
- **含义**：工件以**整件**为单位流转（每次运输一件、每台机器同时一道工序），**不存在子批拆分、批量合并、波次门控或运输合批**（注意：AGV 一次只运一件，连"拼车合批"都没有）。**→ 我们的"分批（波次门控）"环节在此文完全空缺。**

### Q4. 方法细节

- **MARL 具体形态**：**不是 MAPPO、不是 QMIX、不是独立 PPO。** 是 **两个独立 DDQN agent（MA + AA）**，共享 **同一套 TAAGNet 骨架**但**参数独立训练**（"The network is shared by the MA and the AA (**with independently trained parameters**), differing only in the dimensionality of the output layer"）。**无 centralized critic、无 mixing network、无 CTDE、无通信**；原文明确："We **abandon** the approach of explicitly learning joint Q-functions and mixing networks"。credit assignment 靠**手工的分离贡献奖励**。
- **网络骨干**：**TAAGNet = 类型专用线性编码器（式 23）→ 边类型嵌入（式 24）→ 2 层 GATv2Conv（8 头，式 25–28）→ 全局自注意力模块（式 29–31，β=0.1 残差，式 30 为 `softmax(QK^T/√d_k · M_batch)`）→ 降维 MLP（2 隐层 × 64）→ 第二层 GATv2（式 32）→ 类型感知 mean pooling（式 33）+ 全局 mean pooling → 拼接 → 决策 MLP 输出 Q 值（式 34）**。损失 **Smooth L1**；优化器 **AdamW**；lr **1e−3 → 1e−4 几何衰减**；LayerNorm 广泛使用；dropout 0。
- **奖励设计**：见上文逐式照录。**MA = 负载均衡分配奖励（−tanh(ΔVar·0.01)）+ 松弛变化×临界度排序奖励**；**AA = 负载均衡分配奖励（−tanh(ΔVar·0.1)）+ 排序奖励（相对松弛收益 + 目的地工作中心负载收益）×0.1**；两项排序奖励均 clip 到 [−1,1]。**无全局共享奖励、无 makespan 项、无能耗项、无空驶惩罚项。**
- **规模与算力**：默认 **3 工作中心 × 2 并行机 = 6 机、3 台 AGV**；训练 **10 万个仿真时间单位**，**约 8 分钟/agent**（i7-14700K + RTX 3090）；评测 **100 次 Monte Carlo × 3 档利用率（70/80/90%）**，单次仿真 1000 时间单位、约 110–137 个到达作业；泛化最大到 **7 工作中心 / 14 机 / 7 AGV**（零样本、不重训）。**规模上比我们（若我们的布局采样支持更大规模）小；算力需求低。**
- **无统计检验**：无 p 值、无置信区间、无标准差数值（"significance"/"confidence"/"standard deviation" 全 0 命中），只有箱线图。

---
---

### 批4 第三张卡：AEI 103216（李新宇组，PLCSP + MAPPO）——**"PLCSP"缩写与"生产-物流协同"框架的出处，威胁等级：中**

> 精读方式：PyMuPDF 全文提取（**20 页**，含 Table 1–16、Algorithm 1 伪代码、式 (1)–(21)、Fig. 1–16 图注与图面核验）。术语保留英文；引用均为原文摘录（≤2 句/处）。**凡属本文重算而非原文给出的数字，均显式标注"（本文重算）"**。
> **阅读警示（先看这句）**：本篇**在"问题命名与框架叙事"层面对我们是直接对位**——它**首次（与本批 AEI 103195 同济篇并列）把 "production-logistics collaborative scheduling problem (PLCSP)" 写成了正式缩写并用满全文 33 次**；但在**方法机制层面与我们几乎零重叠**：**无图、无注意力、无几何、无分批、无拥堵/路径**，网络是 128-64-64 的纯全连接 MLP，真正的"选择"由**手写规则优先级向量**完成。**结论：它抢的是"术语与问题框定"，不是"机制"；必须引用、必须划界、不必恐惧。**

---

## 《Real-time scheduling for production-logistics collaborative environment using multi-agent deep reinforcement learning》

**期刊正式信息**：Advanced Engineering Informatics（AEI，Elsevier），**Volume 65, 2025, Article 103216**；DOI: **10.1016/j.aei.2025.103216**。Received 6 October 2024 / Revised 5 February 2025 / Accepted 14 February 2025 / **Available online 23 February 2025**。
**作者与机构**：Yuxin Li、**Xinyu Li（通讯，lixinyu@mail.hust.edu.cn）**、Liang Gao。State Key Laboratory of Intelligent Manufacturing Equipment and Technology, School of Mechanical Science and Engineering, **Huazhong University of Science and Technology（华中科技大学机械科学与工程学院，智能制造装备与技术全国重点实验室）**, Wuhan 430074。
**资助**：National Key R&D Program of China **2023YFB4705004**；NSFC **U21B2029**；Fundamental Research Funds for the Central Universities **2024BRA004**；Key R&D Program of Hubei **2021AAB001**。
**代码/数据**：**无开源代码**（全文 "github" 0 命中、无代码可用性声明）。Data availability statement = **"No data was used for the research described in the article."**（注意：与 §4.7 声称"从历史生产数据抽取真实实例"**自相矛盾**）。**无 Supplementary 可获取内容**：正文 3 处指向 "Supplementary Material"（复杂度分析、MAPPO 细节、Method Wang 训练结果），但 **Appendix A 只给了一条 DOI 链接，未随 PDF 提供该补充材料**——**这三处关键内容本文未核实**。

### 精确贡献

原文自报三条（§1，Abstract 为压缩版）：

1. "**A novel real-time scheduling framework is proposed for PLCSP.** Specifically, a new logistics task release moment is proposed, which can **reserve lots of AGV preparation time and avoid unnecessary premature decisions**."
2. "A training algorithm based on **multi-agent proximal policy optimization (MAPPO)** [54] is proposed to deal with the huge action space of PLCSP with large-scale order. **The action space and action space pruning strategy are designed for each agent**, in which the former ensures the sufficient exploration, and the latter reduces the learning difficulty by excluding poor actions."
3. "**The state spaces of three agents are in a serial relationship**, which enables each agent to fully understand the information of the corresponding decision object. Meanwhile, **a reward function is designed to optimize tardiness based on job classification** at each decision point."

**实质贡献核验**：①新物流任务释放时刻 = **单工件流水线门控**（见 Q4，概念上与我们的"波次门控"相邻但**非同构**）；②三智能体分层 + 规则权重动作空间 + 软剪枝（**这是全文唯一有密度的方法贡献，但本质是"用 RL 调规则权重"**）；③状态串联 + 势函数式奖励（见 Q5）。**三条均不涉及图/注意力/几何/分批/拥堵。**

### 关键公式/模型

**问题模型（§2.1）**：目标 **min 总加权拖期 TWT**（非 makespan）：
`min TWT = min Σ_{i=1}^{n} ( w_i · max{0, C_i − d_i} )`（式 1），`C_i` = "**the completion time of last logistics task for job Ji**"（**含返库运输**）。
- **机器是固定的**（这句决定问题性质，逐字）："**Each operation Oij has the corresponding machine Mij and processing time PTij.**" → **JSP（作业车间），非 FJSP；无机器柔性、无机器选择决策。**
- **问题性质自陈**："**PLCSP in this paper is essentially the job shop scheduling problem (JSP) considering limited AGVs**, which is a more complex NP-hard problem."
- **实时调度的定义自陈**："**Real-time scheduling refers to the online allocation of AGVs for operations.**"
- **耦合核心自陈**："the essence of PLCSP is to provide a series of **operation-AGV pairs**, that is, **to determine the AGV of each operation**."
- **规模的量级**："at a certain moment, there are 300 operations to be assigned and 10 available AGVs. The number of potential **operation-AGV pairs is 3000**."；§3.2："its order of magnitude is generally **10³ ~ 10⁴**"。

**假设（Table 3，6 条，逐字）**：
1. "All jobs, machines and AGVs are available at time zero."
2. "**One machine can process at most a job at a time, and one AGV can transport at most a job at a time.**"（原文此句有笔误"transport at most **a** job"）
3. "**The buffer of a machine is infinite.**"
4. "The loading and unloading time of a job on a machine are neglected."
5. "**The machine breakdown, AGV failure, and AGV charging are not considered.**"
6. "**Job preemption on machine/AGV is forbidden.**"

**运行逻辑（§2.3.3，关键：机器侧不学习）**：
- 机器："the traditional running logic of machine is 'first assign first service (FAFS)', or 'first come first service (FCFS)'. This paper proposes a new running logic …, which is '**first urgent first service (FUFS)**'. It means that when the machine is idle and there are some arrived jobs in the buffer, the machine selects **the job with the biggest estimated tardiness cost** for processing."（**FUFS 是消融选出的手写规则，非学习所得**）
- AGV："**The running logic of AGV is FAFS.**"（先分配先服务）

**七个物流任务释放时刻（§2.3.2 + Fig. 4）**：文献 [5]（Cai et al., IJPR 2023）定义 t4/t5/t6/t7 与最早释放时刻 **t1**；本文**新增 t2、t3**，并提出替代 t1 的新机制（逐字）："At the beginning, the job will release the logistics task of its first operation. Afterwards, **when there is only one logistics task that has been assigned an AGV but has not been completed for this job, and there is no task of this job in the logistics task pool**, this job will release the next logistics task and add this task to the logistics task pool."（**= 每工件最多 1 个在途已派任务 + 池中最多 1 个待派任务 → 单工件 WIP 门控**）

**状态（§3.1 + Table 4）**：**六个类别 S1–S6，全部是统计聚合标量，无节点、无图、无坐标**。
- **S1｜all jobs in the order**：(1) 订单完成率；(2–3) 全部工件完成率的 {mean, std}。
- **S2｜all machines**：(1–2) 全部机器利用率的 {mean, std}；(3–4) 全部 `CTM_x`（机器完工在制任务 + 缓冲区全部任务的最短时间）的 {mean, std}。
- **S3｜all AGVs**：(1–2) 全部 AGV 利用率的 {mean, std}；(3–4) 全部 `CTA_y`（AGV 完工在运任务 + 任务栏全部任务的最短时间）的 {mean, std}。
- **S4｜logistics task pool**：(1) 任务数 p；(2–3) 全部任务对应加工时间的 {mean, std}；(4–5) `RTLP_b`（工件剩余时间）的 {mean, std}；(6–9) `EAT_b`（工件最早可用时间）的 {mean, std, min, max}；(10–13) `ETC_b`（估计拖期成本）的 {mean, std, min, max}；(14) 全部任务对应机器的平均利用率；(15) 全部任务对应机器的平均 `CTM_x`。
- **S5｜the filtered job set**："S5 is similar to S4, but their objects are different."（筛后集是物流任务池的子集）
- **S6｜the selected job**（**唯一逐实体向量，且维数随 AGV 台数 l 变化**）："(1–l) the **CTA_y of each AGV**; (l+1–2l) the **PUT_y of each AGV for the selected job**; (2l+1) the EAT_b of the selected job; (2l+2) the ETC_b of the selected job; (2l+3) the CTM_x of the machine for the selected job."
- **actor 输入分工**：job filter = `{S1,S2,S3,S4}`；job selection = `{S1,S2,S3,S5}`；AGV selection = `{S2,S3,S6}`。
- **归一化**："each state feature needs to undergo **Z-Score normalization** before being input into the agent."（mean/std 由"随机动作策略"跑大量实例统计而来）
- **两条关键量（式 7、8）**：`CTA_y = RTIT_y + Σ_{g=1}^{h} ABCT_g`（AGV 忙到何时）；**`PUT_y = Δt(Mx1, Mx2)`**，其中原文："Define the **destination of ABLTh as Mx1**, and the **pickup location of the selected job as Mx2**."（**= 空驶时间，Δt 为运输时间矩阵查表**）
- **估计拖期成本（式 6）**：`ETC_b = SOT_b · w_b if SOT_b ≥ 0; SOT_b / w_b if SOT_b < 0`，其中 `SOT_b = t + RTLP_b − d_b`（**松弛为正时乘权重、为负时除权重——一个非标准的非对称设计**）。

**动作空间 + 剪枝（§3.2，全文方法核心）——三个 agent 的输出维数极小**：
- **job filter agent**：输出 **4 维** 规则权重 `a_t^JF = [ω_BETC, ω_LWKR, ω_CR, ω_EDD]`；`FPJF = Norm( Sigmoid(a_t^JF) · [PV_BETC, PV_LWKR, PV_CR, PV_EDD] ) + ω_SEAT · PV_SEAT`（式 12）→ 取优先级最高的**至多 30 个**工件（"which contains **up to 30 jobs**"）。软剪枝 = ① **Sigmoid**（保证权重为正）；② **SEAT 调整项**（偏好 `EAT_b = 0` 的立即可运工件，`ω_SEAT = 50`）。
- **job selection agent**：输出 **4 维** proto-action 向量 `a_t^JS`；每个候选工件有属性向量 `â_t^JS = [RTLP_e, ETC_e, w_e·d_e, CR_e]`（式 13 线性映射到 [−1,1]）；`FPJS = Norm( ‖a_t^JS − â_t^JS‖₂ ) + ω_SEAT · PV_SEAT`（式 14）→ 取最高优先级。软剪枝 = **SEAT**。
- **AGV selection agent**：输出 **2 维** 规则权重 `a_t^AS = [ω_SCT, ω_SPUT]`；`FPAS = Norm( a_t^AS · [PV_SCT, PV_SPUT] ) + ω_SCPT · PV_SCPT`（式 15）→ 取最高优先级。软剪枝 = **SCPT**（偏好 `CTA_y + PUT_y` 最小的车，`ω_SCPT = 5`）。
- **PDR 库（Table 5，逐条）**：工件侧 BETC（最大 ETC 优先）/ LWKR（最小 RTLP 优先）/ CR（最小临界比）/ EDD / SEAT（最小 EAT 优先）；AGV 侧 SCT（最小 CTA）/ SPUT（最小 PUT）/ SCPT（最小 CTA+PUT）。
- **归一化定义（式 9–11）**：`Norm([x_p]) = [(x_p − min)/(max − min)]`；`PV_Rule = Norm([x_p])`。

**奖励（§3.3）——势函数式（potential-based）望远镜奖励，三 agent 共享**：
- "since the three agents are in a cooperative relationship, **they adopt the same reward function**."
- `ETC_i^R(t) = w_i · max{0, EC_i − d_i}`（式 16），`EC_i` 分四种情形：(a) 已完工 → `EC_i = C_i`；(b) 有在制/在运任务 → `EC_i = t + RTJ_i`；(c) 在 L/U 或机器缓冲区等运输 → `EC_i = t + RTJ_i + (Σ_{y=1}^{l} RTIT_y)/l`（**用平均在运时间近似等待**）；(d) 在机器缓冲区等加工 → `EC_i = t + RTJ_i + RTIP_x`。
- `TETC_t^R = Σ_{i=1}^{n} ETC_i^R(t)`；**`r_t = TETC_t^R − TETC_{t+1}^R`**（式 17）。
- **有效性证明（式 18，原文）**：`R = Σ_{t=1}^{F} r_t = TETC_{t=1}^R − TETC_{t=done}^R = −TWT`，其中 "TETC_{t=1}^R and TETC_{t=done}^R … are **0 and TWT**"。
- **→ 判定：这是以 Φ(s) = −TETC^R(s) 为势函数的势函数奖励塑形（PBRS），与本任务目标严格同号且无偏；但它同时意味着"每步奖励只反映启发式估计的拖期变化"，不是真实拖期增量。**
- **无 makespan 项、无能耗项、无空驶惩罚项、无拥堵/死锁项、无层级/组内归因。**

### 实验设置

- **仿真与实现**：Python 自建仿真环境（未说明是否 SimPy/DES 库）；未提及 GPU；未提框架版本。
- **规模（§4.1，逐条）**：**车间 20 台机器 + 10 台 AGV**（**全文只有这一种车间配置**）。(1) 初始工件数 / 动态插入工件数：`randi[100,600]` / `randi[50,250]`；(2) 每工件工序数 `randi[4,10]`；(3) 工序加工时间 `randi[20,40]`；(4) 工件权重 `[1,2,4]`，占比 `[20%,60%,20%]`；(5) 到达间隔 ~ `exp(1/λ_new)`，`λ_new ∈ randi[80,110]`；(6) 交期用 **TWK 法**，`d_i = a_i + c·(TPT_i + TDT_i)`，`c = 1.5`；(7) "The transportation time matrix in this paper is obtained by **halving the travel distance matrix in [5]**."
- **训练协议**：**1500 episodes**，**每个 epoch 随机生成一个新问题**；"The DNNs of the **1420th episode** in the convergence stage are selected to conduct the subsequent performance comparisons."（**= 单一模型，跨全部 12 个数据集测试，无逐数据集重训**）
- **超参（Table 6）**：每个 actor / critic 均为**全连接 DNN，三个隐层 128 / 64 / 64**；lr **0.0001 → 0.00001**（衰减）；触发更新的样本容量 **6400**；minibatch **512**；每条 transition 复用 **8** 次；clip ε **0.2**；熵系数 σ_E **0.01**；γ **0.99**；噪声向量长度 **10**；ω_SEAT **50**；ω_SCPT **5**。调参方式 = **random search** [62]。
- **算法细节（§3.4 + Algorithm 1）**：**MAPPO**（引 Yu et al., NeurIPS 2022 [54]），**三 actor + 三 critic，共 6 个网络、6 个损失**；**GAE + PopArt** 归一化优势与 reward-to-go；actor 损失 = PPO clip 目标 + 熵项（式 19）；critic 损失 = **Huber loss**（式 20）。**全局状态构造（逐字）**："The global state s_i in (20) is the concatenated state `concat(o_t^{jf}, x→)`, where `x→` is a gaussian noise vector `x→ ~ N(0, σ²)`."；"This paper adopts the noise of **periodic shuffle (100 episodes)**, and it can mitigate the multi-agent policies overfitting."（**即三个 critic 都吃同一个 job filter 观测 + 噪声，非各自局部 critic**）
- **动作采样**：训练时 "each actor network samples an action from **Multivariate Gaussian Distribution**"；在线时 "**the output mean vector is directly used as an action**"（确定性）。
- **评测协议（§4.4）**：**12 个数据集 = {TN_job ∈ 150–300 / 300–500 / 500–700 / 700–850} × {λ_new ∈ 80–90 / 90–100 / 100–110}**，每个数据集 **10 个随机问题** → **共 120 个实例**。指标 = **SP（superior proportion）**。
- **基线（列全）**：
  1. **16 条复合规则** = 8 条工件规则（SPT / LPT / BETC / LWKR / CR / EDD / WSPT / S/RT）× 2 条 AGV 规则（**SCT** 与 **SCPT**，由 §4.3 消融选出）；
  2. **Method GP**（据 [32] 用本文规则 + 二叉树遗传编程，得 GPR1–GPR4 四条规则）；
  3. **Method Luo** [10]（R-DRL：两 agent 做工件选择 + AGV 分配，动作空间用规则表示）；
  4. **Method Park** [18]（S-DRL：proto-action 向量 + 欧氏距离解码，"can effectively solve production scheduling problems considering **large-scale order**"）；
  5. **Method Wang** [64]（**单 agent PPO + GAT**，端到端选 operation-AGV 对）。
- **消融（§4.3）**：24 条复合规则 + 3 个随机初始化的 MAPPO 模型，分别加/不加四个策略（SEAT 调整、SCPT 调整、Sigmoid 操作、机器运行逻辑 FCFS/FAFS/FUFS）。

### 核心数字

**（A）vs 16 条复合规则（Table 7，SP 定义为 `(TWT_rule − TWT_MAPPO)/TWT_MAPPO`，见下方"内部矛盾"）**
- 原文自报："the average SPs of proposed method to **10 composite rules on all datasets are greater than 50 %**, and other average SPs are **greater than 13.6 %**."
- **逐格核验（本文重算）**：SCT 侧的 8 条规则 SP = **50.9–300.1**；SCPT 侧的 SPT（**82.3–97.4**）与 LPT（**184.4–208.0**）也 > 50 → 合计恰为 10 条；其余 6 条 = **13.6–36.9**。
- **最强基线 = LWKR + SCPT，SP = 13.6–17.8**（随问题规模上升）；**若换算成常规相对改进 `(TWT_r − TWT_M)/TWT_r = SP/(1+SP)`，则约为 12.0%–15.1%（本文重算）**。→ **"对比 SPT+LPT 这类弱规则 SP > 150%"不能当头条；与最强规则的真实差距约 12–15%。**

**（B）vs GP 与三个 DRL 方法（Table 8）**
- 原文自报："the average SPs of proposed method to **all comparison methods on all datasets are greater than 5 %**, and **most SPs are greater than 10 %**"；§4.4.2 又写 "the average performance improvement is **at least 6.2 %**"。
- **逐格最小值（本文重算）**：GPR1 ≥ 6.6、GPR2 ≥ 8.7、GPR3 ≥ 8.8、GPR4 ≥ 13.6、Luo ≥ 10.4、**Park ≥ 6.2**、Wang ≥ 82.4。→ **最弱差距出现在 Method Park（6.2，300–500 / λ=90–100），与"at least 6.2%"吻合。**
- **Method Wang（PPO+GAT）SP = 82.0–98.3，被彻底击穿**（原文归因见下）。

**（C）消融（§4.3，全部为"加 vs 不加"的同模型对比）**
- SEAT 调整："Most of red bars exceed 50 %. Therefore, after adding the SEAT adjustment, the performance of each method has been improved, and **many of them have increased by more than 50 %**."
- SCPT 调整："Each purple bar is close to 100 %. Therefore, after adding SCPT adjustment, **the performance of each MAPPO model is almost doubled**."
- Sigmoid 操作："All SPs are greater than 0 and **most of SPs exceed 20 %**."
- 机器运行逻辑：FUFS 在 27 个方法中取得最优的比例 **59.3%**，故选用 FUFS。

**（D）实时性与部署成本（§4.6）**
- 单决策点在线耗时（50 个大规模实例、每实例数千决策点）：**mean = 0.0029486 s**（逐实例 0.00238–0.00380 s）。
- 端到端处理时间与内存（Table 10，**i7-10875H @ 2.30 GHz + 16 GB RAM**）：MAPPO 168.78→188.52 MB、5.67 s（150 工件）→ **90.73 s（850 工件）**；规则 59.19→62.89 MB、2.20 s→61.71 s。→ **MAPPO 比规则慢约 1.5–2.6×、多占约 3× 内存。**
- 训练成本（Table 11）：单 episode（500 工件）**64.84 s**；单次算法更新 **2.91 s**；**一次完整训练（1500 episodes）= 15.41 h**；峰值内存 **2.26 GB**。

**（E）案例研究（§4.7）**
- 场景：**中国船舶结构件加工车间，6 台机器（车/割/铣/镗/划线/钳）、2 台 AGV**，原料与成品均在仓库；运输时间矩阵 Table 12（秒，**对称矩阵**，仓库↔M5 = 150，M3↔M4 = 60 等）。
- 小规模真例（30 初始 + 20 动态工件）：TWT = **173 / 159 / 221**（四规则分别 188–211 / 176–198 / 244–248），SP = **8.67–24.53%**；计算耗时 **0.997 / 0.964 / 1.009 s**。
- 大规模真例（300 初始 + 100/200/300 动态）：TWT = **21047 / 19775 / 21731**（规则 22423–26939），SP = **13.39–23.97%**；计算耗时 **24.544 / 29.058 / 36.285 s**。
- **注意**：训练用的是 20 机 / 10 车，案例是 6 机 / 2 车，而 **S6 的输入维数 = 2l + 3 随 l 变化** → **是否重训原文未说明（未核实）**；结合 §4.6.3(4) "When facing with different workshop details, the DRL algorithm needs to undergo a new training"，**高度疑为重新训练**。**这是全文最应追问的一点。**

### 自认局限（原文，逐字）

1. "However, compared to other real-time scheduling methods, **the proposed method cannot achieve the best solution on all instances**. The superiority and stability of DRL-based scheduling algorithm deserve further exploration."
2. "Meanwhile, **we will consider the AGV charging and the fuzzy transportation time caused by AGV collision avoidance in PLCSP in future work**. **Machine flexibility** and auxiliary resource constraints are also worthy of further study. In addition, it is worth exploring how to deal with **more disturbance events such as machine breakdown and worker absence**."
3. "Like other studies [6,65,66], **this paper only considers one common event, i.e. new job arrivals.** However, the real-time scheduling framework proposed in this paper is **scalable**. For example, we can mask each job with faulty machine at each decision point."
4. **（对"我们能做什么"最关键的一条，逐字）**："The largest instance in [64] has hundreds of operations. However, the smallest instance in this paper has **thousands of operations**. **GNNs represented by GAT are difficult to accurately form the embedding of large disjunctive graph.** Furthermore, the agent cannot achieve reasonable end-to-end selection through training. Therefore, **when facing production scheduling problems considering large-scale order, it is worth studying to design an embedded strategy that can accurately handle large-scale GNN.**"
5. **（部署层自认，逐字）**："Because the details of a workshop are not always fixed. … **When facing with different workshop details, the DRL algorithm needs to undergo a new training to optimize the scheduling capability of agent.**"

**归纳**：它自认 = ①**不能在所有实例上最优**；②**无 AGV 充电、无避碰/拥堵导致的运输时间模糊性、无机器柔性**；③**只考虑新工件到达一种扰动**；④**（自认）大规模 GNN 嵌入是未解难题、并公开邀请他人去做**；⑤**换车间细节必须重训**。**它没有自认、但客观存在的**：无几何/坐标进状态、无分批、无路网/冲突、SP 指标定义自相矛盾、Table 8 标题与内容不符、算力口径前后矛盾、Supplementary 不可得、无统计检验、无代码。

### 与我们的逐点对比

| 对比维度 | 我方（plcsp） | AEI 103216 | 判定 |
|---|---|---|---|
| 问题类型 | 分批 → 并行机/工序 → 物流三环节；含机器柔性 | **纯 JSP**（"Each operation Oij has the corresponding machine Mij"），机器固定 | **我异** |
| **分批（波次门控）** | 显式动作 `b_t`，`π_B(b|s)` | **完全没有** | **我异（可写死）** |
| 排产 | 学习所得（工序/机器指派） | **手写 FUFS 规则**（消融选出），不学习 | **我异（可写死）** |
| 物流/真派车 | 逐任务选车 + 空驶 + zone 拥堵 + 路网走廊 | **逐任务选车**（`PUT_y = Δt(Mx1,Mx2)` 显式含空驶），**但运输时间 = 固定对称矩阵查表，无拥堵/冲突/路径** | 部分重合 |
| 几何/度量进状态 | 坐标/距离 → **注意力偏置/成对先验 + 不变性设计** | **零**：S1–S5 全为 mean/std/min/max 聚合；S6 为逐 AGV 标量，**无坐标、无成对量、无不变性** | **我异（可写死）** |
| 网络骨干 | 几何/度量感知注意力编码器 | **纯 FC MLP 128-64-64，无图、无注意力、无边** | **我异** |
| 算法 | 组内相对 RL（GRPO 族，无 critic） | **MAPPO（3 actor + 3 critic，GAE + PopArt + Huber）** | **我异** |
| 联合动作空间 | 因子化条件头，`π_B·π_S·π_L` | **不是实体级选择**：三个 agent 只输出 **4 / 4 / 2 维**连续向量，真实选择由**手写优先级向量加权**完成 | **我异（机制层不同构）** |
| 布局/规模泛化 | 参数化布局采样器 + 未训练布局零样本 | **单一车间（20 机/10 车）训练；自认"换车间细节须重训"** | **我异（可写死）** |
| 目标函数 | makespan 主 + 能耗/拖期/拥堵 | **TWT（总加权拖期）** | 我异 |
| 评价严谨性 | 计划含种子化多布局统计检验 | **无 p 值/CI/标准差检验；SP 定义前后矛盾** | **我优** |

1. **同（必须正视的重合面）**：
   - ① **术语与问题框定完全重合**：它把 "**production-logistics collaborative scheduling problem (PLCSP)**" 作为正式缩写使用 **33 次**（同济 AEI 103195 使用 92 次）——**我们在《进展记录》§十二 记的"文献中无此缩写、本项目自创"是错的，必须更正**（见下方"对我方记录的直接更正"）。
   - ② **动态扰动（新工件到达）+ 实时决策点 + 大规模订单**的问题叙事一致。
   - ③ **逐任务的真派车**，且**显式计入空驶**（`PUT_y = Δt(Mx1, Mx2)`、SPUT/SCPT 规则）——这是我们"真 AGV 派车"主张上**唯一需要防守的重合点**。
   - ④ **多智能体分工决策**（它 3 个 agent，我们 3 个头）——分工粒度不同但"分解大动作空间"的动机一致。
   - ⑤ **强调在线毫秒级决策与工业部署**（它 2.9 ms/决策点）。
   - ⑥ **都用"释放时刻/门控"限制过早决策**（它的新释放时刻 vs 我们的波次门控）——**概念相邻，必须主动划界**。
2. **异（我们的四条差异化，此文的命中情况）**：
   - **① 几何/度量感知注意力：完全空白。** 关键词核验：**coordinate = 0、invariant/invariance = 0、pairwise = 0、zone = 0、spatial = 0、attention/GNN 仅出现在他人文献引用中**。S1–S5 是 mean/std/min/max 聚合；**S6 是"按 AGV 索引顺序拼接的定长向量（维数 2l+3）"→ 连 AGV 的置换等变性都没有（本文推断，依据 = Table 4 的索引区间式定义 + 纯 FC 骨干）**。几何的唯一通路是**标量 `PUT_y`/`CTA_y` 与手写规则 SCT/SPUT/SCPT**，即**在动作语义层"委托"给启发式，不在表征层"感知"**。**→ 判定：与"把度量张量注入注意力分数"完全不同构。**
   - **② 分批：完全空白。** "batch" 全文 **4 次命中全部是训练 minibatch / batch size**，外加参考文献 [52] 标题中的 "batch processing"；**sublot = 0、wave = 0、lot streaming = 0**（"lot" 9 次命中为 "lots of" / "a lot of" / 参考文献 "economic lot scheduling"）。反证 = Table 3 假设 6 非抢占、假设 2 一车一工件、假设 3 无限缓冲。**→ 我们的分批环节（分批/波次门控）在此文完全空缺。**
   - **③ 真派车的完整版（拥堵/冲突/路径）：没有。** "congestion" = **0**、"conflict" = **0**、"deadhead" = **0**、"path planning" = **0**；"collision" 仅 1 次且**在 future work 里**。运输时间 = **固定、对称、与负载和路况无关的矩阵**（Table 12 对称）。**→ 我们的"zone 拥堵 + 空驶 + 走廊"环节，它只覆盖了"空驶时间估计"这一小块。**
   - **④ 组内相对 / 无 critic 优势估计：没有。** 它是**教科书式 MAPPO**（3 critic + GAE + PopArt），credit assignment 靠**共享的势函数奖励**；**无组结构、无相对优势、无层级归因**。
   - **其他差异**：⑤ **动作空间的"大"是叙事性的**——它宣称 10³–10⁴ 的 operation-AGV 对，但**策略实际只输出 4/4/2 维连续向量**，组合选择交给手写优先级；我们是在实体上直接取分布。⑥ 它**机器侧完全不学习**（FUFS 手写）。⑦ 它**只有一个车间配置**，换车间须重训。
3. **是否威胁：中（术语/框架层面高，机制层面低）。**
   - **升到"中"的理由（两条，都必须回应）**：
     1. **术语主权**：它（与同济篇）是 **"PLCSP" 缩写的实际出处与使用者**。我们若在论文中把 PLCSP 当作"本项目自创"缩写，是**可被一句话击穿的硬伤**；反之，正确做法是**引用它并声明我们沿用此术语、同时指出我们覆盖了它未覆盖的两个环节**。
     2. **"真派车 + 空驶"的重合**：它的 `PUT_y = Δt(Mx1, Mx2)`、SPUT、SCPT 说明"用车辆位置→取货点的空驶时间选车"**已经有人在 AEI 2025 做了**。我们的派车贡献**不能再表述为"首次考虑 AGV 空驶"**，必须升级为"**空驶 + 拥堵 + 走廊/路网 + 与分批/排产的联合**"。
   - **没有升到"高"的理由（四条防线）**：
     1. **几何/度量信息零命中**（coordinate/invariant/pairwise/spatial 全 0，骨干是 MLP 无注意力）——**我们"度量张量注入注意力"的机制主张在此文无任何对应物，可以写死。**
     2. **分批环节零命中**，且缓冲无限、一车一工件、非抢占。
     3. **拥堵/冲突/路径全部缺失**，且**它自己把它写进了 future work**（"the fuzzy transportation time caused by AGV collision avoidance"）。
     4. **它自己的数字显示"规则做了大部分工作"**：加 SEAT（一个可用性偏好项）→ 多数方法提升 > 50%；加 SCPT（一个"选 CTA+PUT 最小的车"的手写规则）→ **每个 MAPPO 模型性能近乎翻倍**。**→ 这恰好可被我们反用：在它的框架里，纯启发式规则项就吃掉了绝大部分收益，学习部分的净边际贡献被压缩**；我们可据此论证"真正的增益来自决策环节与几何表征，而非在规则输出上再套一层 MLP"。
   - **反向利用（三条硬证据）**：
     - ① 它**公开邀请**我们做的那件事："when facing production scheduling problems considering large-scale order, **it is worth studying to design an embedded strategy that can accurately handle large-scale GNN**"——**这是 AEI 2025 正刊上的、来自最近亲缘团队的、对我们技术路线（几何/度量感知图嵌入）的公开背书**，可直接引作 motivation。
     - ② 它的 **Table 8 Method Wang（PPO+GAT）SP = 82–98%** 是"D **朴素图注意力 + 端到端在大规模上彻底失败**"的硬数据，可用于我们的"为什么不能只是套一个 GAT"论证。
     - ③ 它自认 "**When facing with different workshop details, the DRL algorithm needs to undergo a new training**"——**我们的参数化布局采样 + 未训练布局零样本正好打在这一点上**。
4. **可直接引用（中英对照，4 条）**：
   - "**PLCSP in this paper is essentially the job shop scheduling problem (JSP) considering limited AGVs**"（§1）／"本文的 PLCSP 本质上是**考虑有限 AGV 的作业车间调度问题**"——**证明"生产-物流协同"这一叙事下的既有最强工作，其问题本体仍是经典 JSP：无机器柔性、无分批、无三类环节耦合。**
   - "**when facing production scheduling problems considering large-scale order, it is worth studying to design an embedded strategy that can accurately handle large-scale GNN**"（§4.4.2）／"**面向大规模订单的生产调度问题，设计能够准确处理大规模图神经网络的嵌入策略值得研究**"——**最近亲缘课题组公开指出的空白，正是我们的技术定位。**
   - "**we will consider the AGV charging and the fuzzy transportation time caused by AGV collision avoidance in PLCSP in future work**"（§5）／"**未来工作将考虑 AGV 充电、以及 AGV 避碰所造成的模糊运输时间**"——**用它自己的话证明"AGV 路网冲突/拥堵未被建模"。**
   - "**When facing with different workshop details, the DRL algorithm needs to undergo a new training to optimize the scheduling capability of agent.**"（§4.6.3）／"**面对不同的车间细节时，DRL 算法需要重新训练以优化 agent 的调度能力**"——**证明其无布局/规模零样本能力。**
   - （备选）"**after adding SCPT adjustment, the performance of each MAPPO model is almost doubled**"（§4.3）／"加入 SCPT 调整后，每个 MAPPO 模型的性能几乎翻倍"——**证明其收益主要来自一条手写启发式规则项，而非学习到的表征。**

### 可复用性

- **①可直接借用**：
  - **三智能体"筛选 → 选择 → 分配"的串联分解范式**（filter / select / assign），以及"**动作空间按 agent 三维切分**"的论证（"the multi-agent architecture divides the whole huge action space into three dimensions"）——与我们"分批/排产/物流"三头的**动机表述可以互相加强**，但**必须声明机制不同**（它是规则权重，我们是实体因子化）。
  - **规则权重化动作空间 + 软剪枝（`Norm(·) + ω·PV_rule`）**：这是一份干净的"S-DRL 混合"配方，可直接作为我们的**消融对照臂**（"当不采用实体级因子化头、改为规则权重化头时，性能如何"）。
  - **PDR 库 + 复合规则基线集**：8 条工件规则 × 2–4 条 AGV 规则（含 SCT/SPUT/SCPT/ISPUT），连同 Table 7 的完整 SP 数值，可直接对齐为我们的规则基线表。
  - **势函数式奖励的证明范式**（式 16–18，`Σr_t = −TWT`）：一段**教科书级的"奖励与目标严格同号"证法**，可直接移植/改写为我们奖励设计的附录证明模板。
  - **实时性/内存/训练成本三表（Table 9/10/11）的呈现格式**，以及"单决策点平均耗时"这一指标口径。
  - **物流任务释放时刻（t1–t7）的分类与图示**（Fig. 4）：一份完整的"运输任务何时释放"设计空间清单，**是我们"波次门控（何时成批放行）"的天然对照物**——我们在写"释放时机"的 related work 时可引它。
- **②需改造**：
  - 其动作头（规则权重）与我们的因子化实体头**不同构**；若要作为基线，**只应移植其"规则权重化 + SEAT/SCPT 软剪枝"的解码层**，骨干换成我们的编码器，才是公平消融。
  - 其状态 S1–S5 的 mean/std 聚合**天然丢失实体身份**，无迁移价值；S6 的定长逐 AGV 向量**与 AGV 台数绑死**，若要用必须改为集合/图编码（**这正是我们的贡献点**）。
  - 目标函数是 TWT，我们是 makespan 主 + 多目标附项；比数字时必须换算口径。
- **③移植为基线工作量**：**约 5–7 人天**（无开源代码；环境是我们已有的 SimPy DES，1 天；三 agent MLP + MAPPO（可用现成 MAPPO 实现）1–2 天；规则权重化动作层 + SEAT/SCPT 软剪枝 1–2 天；12 数据集 × 10 问题 + 训练 15 h 机时 1–2 天）。**风险提示**：其运输时间矩阵来自 [5] 且"取半"，**原矩阵未在本文给出**；S6 维数与 AGV 台数耦合，**换车间必须改网络**——移植时须固定同规模并显式声明。
- **是否有开源代码**：**无。** 无 GitHub 链接、无代码可用性声明；**且 Data availability = "No data was used for the research described in the article."**，与其 §4.7 的"真实车间实例"表述冲突。

---

## 事实核验与内部矛盾（写作时的"防守清单"）

1. **⚠️ SP 指标定义前后自相矛盾（最严重）**。§4.3 式 (21)：`SP = (TWT_B − TWT_A)/TWT_B × 100%`（分母 = 基线 B）；§4.4 却写：`SP = (TWT_X − TWT_MAPPO)/TWT_MAPPO × 100%`（分母 = 本文 MAPPO）。**判别依据（本文重算）**：Table 7 出现 **299.5 / 300.1 / 208.0** 等 > 100% 的值，而 (21) 的形态恒满足 `SP ≤ 100%`（因 TWT_A ≥ 0）。**→ 实际生效的是 §4.4 的口径（分母 = MAPPO）。** 后果：**Table 7 的 "SP > 50% 对 10 条规则" 是"基线比本文差 50%"，不是"本文比基线好 50%"**；换算成常规相对改进 `SP/(1+SP)`，对最强规则 LWKR+SCPT 的真实改进是 **12.0%–15.1%（本文重算）**。**引用时必须换算或加注，否则会被审稿人当作夸大。**
2. **Table 8 标题与内容不符**：标题为 "Comparison with GP and **two** DRL methods"，但表内含 **Luo / Park / Wang 三个** DRL 方法（列头 GPR1–4 + Luo + Park + Wang），正文也明确写 "**four** methods are compared"。
3. **算力口径前后矛盾**：§4.2 称 "runs on a server with **Intel(R) Xeon(R) W-3365 CPU @ 2.70 GHz and 125-GB RAM**"；Table 9/10/11 的注却写 "runs on a computer with **Intel(R) Core(TM) i7-10875H CPU @ 2.30 GHz and 16.0 GB RAM**"。**两组数据来源未说明，也未提及任何 GPU。**
4. **案例研究规模与训练规模不一致、是否重训未说明（未核实）**：训练 = 20 机 / 10 车；案例 = **6 机 / 2 车**。而 **S6 输入维数 = 2l + 3 随 AGV 台数 l 变化**，网络输入层不可能直接复用。**原文未说明是否重训**；结合 §4.6.3(4) 的自陈，**高度疑为重新训练**——但**未经原文证实，标"未核实"**。
5. **"贡献 2"的措辞与实现不符**：摘要/贡献称设计动作空间与剪枝以"确保充分探索"，但**三处剪枝（Sigmoid、SEAT ω=50、SCPT ω=5）在数值上都是强先验**——`FPJF` 中 `Norm(·) ∈ [0,1]` 而 `ω_SEAT·PV_SEAT ∈ [0,50]`，**SEAT 项贡献约 98% 的动态范围，实质上接近硬掩码**（仅在全部候选都 EAT≠0 时失效）；SCPT 项占 `FPAS` 的 5/6 动态范围，**为强先验而非剪枝**。这与"Sigmoid 提升 > 20%、SCPT 近乎翻倍"的消融结果一致。**→ 结论：本文的收益主要由两条手写规则项提供（本文推断，依据 = ω 取值与消融幅度）。**
6. **"reward function considering job classification" 措辞漂移**：摘要与贡献 3 说"基于工件分类"的奖励，结论说"a reward function considering **four kinds of jobs**"，而 §3.3 实际只是**四种 `EC_i` 估计情形**，并非工件被分类。**引用时不要沿用"工件分类"的说法。**
7. **Supplementary Material 不可得**：复杂度分析、MAPPO 细节、Method Wang 的训练结果**三处关键内容均在补充材料中，而本 PDF 未附、仅给 DOI 链接**（Appendix A）。**这三处未核实**——尤其 **Method Wang 的训练细节缺失，使 Table 8 中 SP = 82–98% 的对比无法复核**（虽然 Method Wang 引的是 [64] Wang et al., TNNLS 2024，可按该文自行复现）。
8. **无统计检验**：**无 p 值、无置信区间、无显著性检验**（"significance" 0 命中）；离散度仅以 mean/std（跨 10 个问题的 SP 标准差）与箱线图/小提琴图呈现。
9. **Table 1 的定位自陈**：本文在综述表中把自己归为 "JSP-AGVs / MAPPO / total weighted tardiness"——**作者自己承认问题本体是 JSP-AGVs，而非 FJSP-AGVs。** 这是最有力的划界引文。

---

## 对我方记录的直接更正（高优先级，必须处理）

**《progress-log.md》§十二 的条目"`| PLCSP | 本项目自创（…）——文献中无此缩写 |`" 是事实错误，必须更正：**

- **AEI 65 (2025) 103216（李新宇组，本卡）在摘要第一句即定义并使用 `PLCSP`，全文出现 33 次**："With the extensive application of automated guided vehicle (AGV), **production-logistics collaborative scheduling problem (PLCSP)** becomes challenging for enterprises."
- **AEI 65 (2025) 103195（同济 Shi/Qiao 组）同样定义并使用 `PLCSP`，全文出现 92 次**："Over recent years, several scholars have gradually explored the **production-logistics collaborative scheduling problem (PLCSP)** …"
- **正确写法**：**PLCSP 是既有文献已在使用的缩写，非本项目自创。** 我们应在论文中**引用这两篇作为术语出处**，然后声明"本文沿用该术语，并进一步覆盖其未涉及的分批环节与度量感知表征"。**若继续按"自创缩写"处理，属于可被一句话击穿的事实错误。**
- **处置（2026-09-29 复核）：该行动项已失效/已完成。** `progress-log.md` §12.2 已更正为"PLCSP 是既有缩写，须引用 AEI 103216 / AEI 103195"；`feasibility-report.md` 与 `method-design.md` 中**均不存在**"自创缩写"表述（原指向的 `feasibility-report.md:122`、`method-design.md:11` 两处均无此口径，行号亦已因后续修订偏移）→ 无需再改。

---

## 速答：六个关键问题（AEI 103216）

### Q1. 问题设定：调度什么？机器/工件/AGV 各是什么角色？job shop 还是 flow shop？三环节覆盖了哪些？

**是 JSP（作业车间），不是 flow shop、也不是 FJSP。三环节里只覆盖了"物流"一环，且该环被简化；"排产"交出给手写规则；"分批"完全没有。**

- **问题性质（逐字）**："**PLCSP in this paper is essentially the job shop scheduling problem (JSP) considering limited AGVs**"。Table 1 自陈行：`this paper | JSP-AGVs | MAPPO | total weighted tardiness`。
- **机器**：`M = {M_x, x = 1,…,m}`，**20 台**。**工件的每道工序对应哪台机器是预先给定的**——"Each operation Oij has the **corresponding machine Mij** and processing time PTij." → **无机器选择决策、无柔性**（作者自认"Machine flexibility … worthy of further study"）。**机器侧的加工排序由 FUFS 手写规则决定，不学习**（"when the machine is idle and there are some arrived jobs in the buffer, the machine selects **the job with the biggest estimated tardiness cost**"）。缓冲区无限。
- **工件**：`J = {J_i, i = 1,…,n}`，静态初始批 + 动态到达（到达时刻与类型不可预测）；权重 `w_i ∈ {1,2,4}`；交期 `d_i = a_i + 1.5·(TPT_i + TDT_i)`（TWK）。**工件以整件流转，不可拆分、不可抢占**（假设 6）。
- **AGV**：`V = {V_y, y = 1,…,l}`，**10 台**，全部起始于 L/U（装卸站）；负责机器与 L/U 之间的工件转运；**一次只运一件**（假设 2）；完成后若无新任务则**停在原地**、任务栏内按 FAFS 执行；每工件全部工序完成后需**返运回 L/U** 入库。
- **决策变量**：**"operation-AGV pairs"**——即为每道待运工序选一台 AGV（agent 逐决策点选一个"工件 + AGV"）。
- **三环节覆盖情况**：
  - **① 分批（batching）：完全没有。** 无子批、无合批、无波次、无运输合批（一车一件）。
  - **② 排产（parallel-machine / 工序调度）：不属于学习范畴。** 机器固定（JSP）、工序顺序固定、机器侧排序 = FUFS 规则。**唯一与"排产"沾边的是"物流任务释放时刻"门控**（§2.3.2），即通过延迟发布下一道工序的运输任务来间接影响在制品的推进节奏。
  - **③ 物流（真 AGV 派车）：覆盖，且是全文唯一的学习对象**，但**运输时间 = 固定对称矩阵查表**，无拥堵/冲突/路径规划。
- **评价目标**：`min TWT`，且 `C_i` 定义为"**last logistics task** for job Ji"（**含返库**）。

### Q2. AGV 决策形式：真"派车"吗？有空驶吗？有拥堵/冲突/路径规划吗？AGV 与机器如何耦合？

**是"真派车"（逐运输任务选车、显式计入空驶时间），但拥堵/冲突/路径规划完全不存在，"派车"建立在固定运输时间矩阵之上。**

- **真派车的证据（逐条）**：
  - 决策语义：§2.1 "the essence of PLCSP is to provide a series of **operation-AGV pairs**, that is, to determine **the AGV of each operation**"；§2.2 "select a job from the logistics task pool and **assign it an AGV**"。
  - 车辆状态进入决策：`CTA_y = RTIT_y + Σ_{g=1}^{h} ABCT_g`（式 7，AGV 忙到何时）；**`PUT_y = Δt(Mx1, Mx2)`（式 8）**。
  - AGV 侧规则：`SCT`（最小 CTA）、`SPUT`（最小 PUT）、`SCPT`（最小 CTA+PUT）——**都是"逐车"打分**。
  - 状态 S6：**逐台 AGV 的 `CTA_y` 与 `PUT_y` 都被喂给 AGV selection agent**（"(1–l) the CTA_y of each AGV; (l+1–2l) the PUT_y of each AGV for the selected job"）。
- **空驶（deadhead）：存在，且被显式形式化（这是必须承认的重合点）**。原文："Define the **destination of ABLTh as Mx1**, and the **pickup location of the selected job as Mx2**. Hence … `PUT_y = Δt(Mx1, Mx2)`." —— 即**"AGV 完成队列中最后一个任务后所在位置 → 新任务取货点"的行驶时间**，正是空驶。§4.3 的 SCPT 消融也明确说其作用是 "makes agent prefer to select the AGV that can **arrive at the pickup location in a short time**"。Fig. 4 的 AGV 时间轴亦画出 "**pickup part**" 与 "**stay and wait**" 两段。
  - **但**：`Δt` 是**固定、对称、与负载和路况无关的矩阵**（案例 Table 12 逐格对称：仓库↔M5 = 150、M3↔M4 = 60 …），**空驶与载货用同一矩阵、无速度/载重差异**。
- **拥堵 / 冲突 / 路径规划：全部没有。** 关键词核验：**"congestion" = 0、"conflict" = 0、"deadhead" = 0、"zone" = 0、"path planning" = 0、"routing" 无路网含义**；**"collision" 仅 1 次，且出现在 future work**："the fuzzy transportation time caused by **AGV collision avoidance**"。**无路网图、无路段容量、无交叉口、无死锁处理**。
- **机-车耦合怎么处理**：
  - **硬耦合（时间层）**：工件第 j 道工序必须在其第 j−1 道工序**运输完成**后才能开工；`RTLP_b`（式 2/4）把 `PT` 与 `TT` 串成链；`EAT_b`（式 3/5）给出工件的"最早可用时刻"。
  - **释放时刻门控（机制层）**：只有当该工件"**恰有 1 个已派未完成的运输任务、且池中没有该工件的任务**"时，才发布下一道运输任务 → **每工件在制运输任务数被卡在 1**。
  - **优先级层**：`ETC_b`（式 6）把"加工剩余 + 运输剩余 + 交期 + 权重"揉成一个标量，作为 BETC 规则与状态 S4/S5 的核心。
  - **求解层**：**机器侧（FUFS）与物流侧（三 agent）是两条独立的、非联合的决策通道**；没有联合动作、没有联合 Q、没有通信；**"协同"体现为共享同一势函数奖励 + 状态互相可见**。
  - **重要简化**：缓冲区无限（假设 3）→ **"缓冲区满导致 AGV 无法卸货"这类强耦合被消除**；AGV 充电/故障不考虑（假设 5）。

### Q3. 几何/布局/距离信息：进状态了吗？标量还是注意力偏置？有不变性设计吗？

**核心判定：几何以"逐实体标量 + 手写规则优先级"两种形式进入决策，但既不进注意力（无注意力），也无任何不变性设计。AEI 103216 与"把度量张量注入注意力分数"完全不同构。**

- **状态里有没有位置/坐标/距离？——没有坐标，只有"运输时间"这个派生标量。**
  - **S1–S5：全部是 mean/std/min/max 统计聚合**（见上文 Table 4 逐条），**维度与实体数无关，实体身份被抹平**。
  - **S6：唯一的逐实体向量**，含逐台 AGV 的 `CTA_y` 与 `PUT_y`（`PUT_y` 由 `Δt(Mx1, Mx2)` 派生，**携带成对距离信息**），以及被选工件的 EAT/ETC 与其目标机器的 CTM。
  - **关键词核验**：**"coordinate" = 0、"layout" 5 次命中全部指"车间布局"这一名词（Fig. 2/Fig. 16 的布置图、部署章节的"workshop layout"要素），无一次作为状态特征**；**"distance" 6 次命中 = `dist` 的定义（本文未给）+ Method Park 的"欧氏距离解码"（3 次）+ "halving the travel distance matrix in [5]"（1 次）+ 式 (14) 的**属性向量 L2 距离（非物理距离）**；**"invariant"/"invariance" = 0、"pairwise" = 0、"spatial" = 0、"zone" = 0。**
- **标量喂 MLP 还是注意力偏置？——都不是，走的是"手写优先级向量"这条第三通路。**
  - 骨干是 **纯全连接 MLP（128-64-64）**，**没有注意力、没有图、没有边、没有消息传递**；几何标量（`PUT_y`、`CTA_y`）以**定长向量**喂进 MLP。
  - **距离影响"选择"的真实通路是规则**：`FPAS = Norm(a_t^AS · [PV_SCT, PV_SPUT]) + ω_SCPT · PV_SCPT`（式 15），其中 `PV_SPUT`、`PV_SCPT` 都是**对 `PUT_y`/`CTA_y` 做 min-max 归一化得到的优先级向量**。→ **距离通过"被写成一条规则"来影响决策，而非通过网络学到成对先验。**
  - **→ 判定：既不是"标量特征注入"，也不是"注意力偏置注入"，而是"启发式规则优先级注入"。**
- **不变性设计：无。** 无 translation/rotation/scale/permutation 不变性的讨论或设计（"invariant" 0 命中）。更进一步（**本文推断，依据 = Table 4 的索引区间式定义 + 纯 FC 骨干**）：**S6 是"按 AGV 索引顺序拼接的定长向量（维数 2l+3）"，因此对 AGV 的置换既不等变也不不变，且输入维数与 AGV 台数 l 绑死**——这也是它"换车间细节必须重训"的结构性根因。
- **→ 对我方机制主张的最终判定**：**不等同，且差距比 TAAGNet 更大。** TAAGNet 至少还有一层 GATv2 注意力与离散边型；**本文连注意力层都没有**。"把度量张量注入注意力分数"在此文**没有任何对应物**。**可正面主张的差异化句**："AEI 103216 的调度状态由六类统计聚合标量构成，其中不含任何坐标；其策略网络为 128-64-64 的全连接 MLP，不含注意力层、不含图结构，因此不存在任何成对几何先验或不变性设计。距离信息仅通过两条手写规则的优先级向量（`PV_SPUT`、`PV_SCPT`）与逐 AGV 的标量 `PUT_y` 影响决策。"

### Q4. 有没有分批决策？

**没有。完全无生产分批（batching）环节。但存在一个**概念相邻**的机制——"物流任务释放时刻门控"，必须主动划界以免被审稿人认作"已有波次门控"。**

- **"batch" 全文 4 次命中，逐条核验：全部是训练 minibatch / batch size**（"B is the batch size"、Table 6 的 "minibatch size B = 512"、"the batch size and learning rate"、以及参考文献 [52] 标题中的 "batch processing"）。**无一次指生产批量。**
- 相关词核验：**sublot = 0、wave = 0、lot streaming 相关 0 命中**（"lot" 9 次命中 = "lots of AGV preparation time" / "a lot of historical data" / 参考文献 "economic lot scheduling"）。
- 反证（Table 3，逐字）：假设 6 "**Job preemption on machine/AGV is forbidden**"；假设 2 "**one AGV can transport at most a job at a time**"；假设 3 "**The buffer of a machine is infinite**"。
- **含义**：工件以**整件**为单位流转；**不存在子批拆分、批量合并、成组加工、运输合批或波次放行**。→ **我们的"分批（波次门控）"环节在此文完全空缺。**
- **⚠️ 概念相邻项（必须主动划界）**：它的**新物流任务释放时刻**（§2.3.2）规定"每工件最多 1 个已派未完成运输任务 + 池中最多 1 个待派任务"才发布下一道运输任务。这在**动机**上与我们的"波次门控"同源（都是**通过控制释放时机来抑制过早决策、限制在制品**，且它明确说目的是 "reserve lots of AGV preparation time and **avoid unnecessary premature decisions**"），但：
  - 它是**单工件级、串行、拉式门控（每工件独立，不形成批次）**；
  - **不产生批次对象、不涉及批量大小决策、不涉及成组加工或成组运输**；
  - **不是学习所得**，是**写死的规则**（"this job will release the next logistics task"）。
  - **→ 判定：非同构。写作时应在"释放时机"层面引用它（related work），并明确我们的分批环节是"批次形成 + 批量大小 + 波次门控"的**动作维度**，与之不在同一层次**。**若回避不提，反而可能被审稿人当作"已有工作"。**

### Q5. 方法细节

- **MARL 具体形态**：**MAPPO**（引 Yu et al., NeurIPS 2022 [54]）；**三个独立 agent = job filter / job selection / AGV selection**，**每个 agent 各有 1 个 actor + 1 个 critic，共 6 个网络、6 个损失**。**无 mixing network、无值分解（VDN/QMIX 仅在引言被提及为"著名方法"）**。**critic 是集中的**：三者的全局状态统一为 `concat(o_t^{jf}, x→)`（job filter 观测 + 长度 10 的高斯噪声向量，每 100 episodes 周期打乱），**即三个 critic 吃同一个全局状态**，而非各自局部观测。**三 agent 严格串行**（filter → selection → AGV），"the action of the former agent affects the input of the latter agent"。
- **三个智能体分别做什么（逐条）**：
  | agent | 输入状态 | 动作（输出） | 输出去向 | 软剪枝 |
  |---|---|---|---|---|
  | **job filter** | `{S1,S2,S3,S4}` | **4 维** PDR 权重 `[ω_BETC, ω_LWKR, ω_CR, ω_EDD]` | Sigmoid → 4 条优先级向量加权和 `Norm(Z)` → `+ω_SEAT·PV_SEAT` → **取 top-≤30 工件**成筛选集 | Sigmoid（保正）+ SEAT（ω=50） |
  | **job selection** | `{S1,S2,S3,S5}` | **4 维** proto-action 向量 | 与每个候选的属性向量 `[RTLP, ETC, w·d, CR]`（映射到 [−1,1]）求 **L2 距离** → `Norm(‖·‖₂)` → `+ω_SEAT·PV_SEAT` → **取 1 个工件** | SEAT（ω=50） |
  | **AGV selection** | `{S2,S3,S6}` | **2 维** PDR 权重 `[ω_SCT, ω_SPUT]` | `Norm(a·[PV_SCT, PV_SPUT])` → `+ω_SCPT·PV_SCPT` → **取 1 台 AGV** | SCPT（ω=5） |
- **网络骨干**：**全连接 DNN，三个隐层 128 / 64 / 64**（actor 与 critic 各自同构）；**无 GNN、无 Transformer、无注意力、无 RNN**。状态特征先做 **Z-Score 归一化**（mean/std 由随机策略预跑统计）。动作分布 = **多元高斯（训练）**，在线取**均值（确定性）**。
- **训练细节**：GAE 估计优势 + **PopArt** 归一化优势与 reward-to-go；actor 损失 = PPO clip（ε=0.2）+ 熵项（σ_E=0.01）；critic 损失 = **Huber**；γ=0.99；lr 1e-4 → 1e-5；容量 6400 触发更新；minibatch 512；**每条 transition 复用 8 次**；1500 episodes，每 episode 随机新问题；取 **第 1420 个 episode** 的权重做全部对比。
- **动作空间如何剪枝（标题卖点，逐条核验）**：
  - **job filter**：① **Sigmoid** 把 4 个权重压成正数（保证 4 条规则都起正向作用）；② **SEAT 调整项**（偏好 `EAT_b = 0` 的立即可运工件，**ω_SEAT = 50**，通过消融比较后取 50）。
  - **job selection**：**SEAT 调整项**（同上，ω_SEAT = 50）。
  - **AGV selection**：**SCPT 调整项**（偏好 `CTA_y + PUT_y` 最小的车，**ω_SCPT = 5**；Fig. 7 的扫参结论："when **ω_SCPT = 5**, the proposed method obtains the minimum TWT in the convergence stage"）。
  - **⚠️ 数值核验（本文重算）**：`Norm(·) ∈ [0,1]`，而 `ω_SEAT·PV_SEAT ∈ [0,50]`（SEAT 占约 **98%** 动态范围）、`ω_SCPT·PV_SCPT ∈ [0,5]`（SCPT 占 **5/6**）。**→ 三者实质都是强先验甚至近似硬掩码，而非"剪枝"意义上的候选剔除**；且消融显示**收益主要来自它们**（SEAT：多数 > 50%；SCPT：近乎翻倍；Sigmoid：多数 > 20%）。
- **奖励设计**：**三 agent 共享同一奖励**（"they adopt the same reward function"）= **势函数式望远镜奖励** `r_t = TETC_t^R − TETC_{t+1}^R`（式 17），并证明 `Σ r_t = −TWT`（式 18）。`ETC_i^R(t) = w_i·max{0, EC_i − d_i}`，`EC_i` 分四种情形（已完工 / 有在制或运 / 等运输（用平均在运时间 `(Σ RTIT_y)/l` 补）/ 等加工（加 `RTIP_x`））。**无 makespan 项、无能耗项、无空驶惩罚、无拥堵/死锁惩罚、无组内比较项。**
- **状态特征逐维**：见上文 Table 4 的 S1–S6 全量照录（**这是全文给出的全部状态，无遗漏**）。
- **规模与算力**：**20 机 / 10 车**（唯一配置）；规模最大 **850 个总工件、每工件 4–10 道工序**（→ 最大实例"数千道工序"）；训练 **15.41 h / 1500 episodes**（**CPU 口径矛盾，且未提 GPU**）；在线 **2.9486 ms/决策点**。
- **无统计检验**：无 p 值、无 CI、无显著性检验。

### Q6. 实验：实例来源与规模、基线（列全）、核心数字、算力与训练时长

- **实例来源**：**全部随机生成**（§4.1 的 7 条分布规则），**仅案例研究声称"从历史生产数据抽取"**（但 Data availability 又称"No data was used"——**矛盾**）。**无标准基准（无 Brandimarte/Kacem/Hurink），因此与我们常用的 MK 系列不可直接比数。**
- **规模**：训练分布 = 100–600 初始工件 + 50–250 插入工件；测试 = TN_job **150–850** × λ_new **80–110** 的 **12 个数据集 × 10 问题 = 120 实例**；单实例最多"数千决策点"；案例最大 = 300 初始 + 300 动态。
- **基线（列全，共 5 类 23 个）**：
  1. **16 条复合规则**：工件侧 8 条（SPT / LPT / BETC / LWKR / CR / EDD / WSPT / S/RT）× AGV 侧 2 条（**SCT** / **SCPT**）；
  2. **GP 的 4 条进化规则** GPR1–GPR4（据 [32]）；
  3. **Method Luo** [10]（R-DRL，两 agent，规则动作空间）；
  4. **Method Park** [18]（S-DRL，proto-action + 欧氏距离解码）；
  5. **Method Wang** [64]（单 agent PPO + **GAT**，端到端 operation-AGV 对选择）。
  （消融另有 24 条复合规则 + 3 个随机初始化 MAPPO 模型作对照。）
- **核心数字**：
  - vs **最强规则 LWKR+SCPT**：SP = **13.6–17.8**（换算成常规相对改进 ≈ **12.0–15.1%**，本文重算）；
  - vs **GPR1–GPR4 / Luo / Park**：平均 SP 全部 > 5%，多数 > 10%，**最小 6.2**（Park）→ 常规相对改进 **约 5.8%–15%**；
  - **vs Method Wang（PPO+GAT）：SP = 82.0–98.3**（近乎碾压）；
  - 消融：SEAT → 多数 > 50%；SCPT → 近乎翻倍；Sigmoid → 多数 > 20%；FUFS 在 27 方法中占优 59.3%；
  - 在线：**2.9486 ms/决策点**；850 工件端到端 **90.73 s**（规则 61.71 s）；
  - 案例：小实例 SP 8.67–24.53%（耗时 ~1 s）；大规模实例 SP 13.39–23.97%（耗时 24.5–36.3 s）。
- **算力与训练时长**：**§4.2 称 Xeon W-3365 + 125 GB RAM；Table 9/10/11 注称 i7-10875H @ 2.30 GHz + 16 GB RAM（口径矛盾，均未提 GPU）**；训练 **1500 episodes / 15.41 h**，单 episode（500 工件）64.84 s，单次更新 2.91 s，峰值内存 2.26 GB。
- **无统计检验**：无 p 值、无置信区间；离散度只用 mean/std（跨 10 问题的 SP）与箱线图/小提琴图。


---

# 附：2026-09/10 补录卡片（PLCSP 对位批）

> 来源：本轮精读的 7 篇 PLCSP 顶刊（AEI 同济/TSMC 李新宇组/TASE CP/JMS 自组织 + IJPR×2/COR）

### Batch A — 4 篇论文客观摘要卡片

> 说明：只做客观复述，不含任何评价或"与我方对比"。
> 中间提取文本位于系统临时目录，已在完成后删除。
> 提取工具：PyMuPDF 1.28.0（`import fitz`）。

---

## 卡片 1：AEI2025 — Production-Logistics Nested-Hierarchical DRL

### 1. 题录

- **标题**：Production-logistics collaborative scheduling in dynamic flexible job shops using nested-hierarchical deep reinforcement learning
- **作者**：Jiaxuan Shi (a), Fei Qiao (b), Juan Liu (b), Yumin Ma (b, 通讯), Dongyuan Wang (b), Chen Ding (b)
  - a: Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University, Shanghai 201210, China
  - b: School of Electronics and Information Engineering, Tongji University, Shanghai 201804, China
- **期刊**：Advanced Engineering Informatics, Vol. 65 (2025), 文章号 103195
- **年**：2025（Received 24 August 2024; Revised 10 January 2025; Accepted 7 February 2025; Available online 4 March 2025）
- **DOI**：10.1016/j.aei.2025.103195
- **关键词**：Flexible job shop; Production-logistics collaborative scheduling; Dynamic scheduling; Deep reinforcement learning; Nested-hierarchical framework
- **资助**：NSFC 62133011, 62273260, 62373288

### 2. 它解决什么问题

- **问题类型**：动态柔性作业车间（Flexible Job Shop, FJS）中的**生产—物流协同调度问题（DFJS-PLCSP）**。论文将其定位为比"典型 AGV 调度问题"更难的变体：典型 AGV 调度只优化物流侧决策以服务已定的生产计划，而本文的 DFJS-PLCSP 同时决定 4 类决策——① 为每个工序分配合适机器；② 安排每台机器上工序的加工顺序；③ 为每个物流任务分配合适 AGV；④ 安排每台 AGV 上物流任务的执行顺序。因此生产调度与物流调度可**互相影响、协同**，而非物流单方面被动响应。
- **涉及资源**：18 台多用途机器（+ 一个抽象为虚拟机器的装卸站 L/U station）+ 6 台 AGV（验证环境）；多个工件在 L/U 站与机器之间由 AGV 转运。物流过程分两段：non-load stage（空载取件）与 load stage（载件送达）；工件全部工序完成后还需经由物流过程返回 L/U 站。
- **优化目标**：**同时**最小化 makespan（MS）与总物流成本（TC）。这是双目标（非加权单目标）。
  - 目标函数（式 1、式 2）：
    - `minMS = max_{1≤i≤I, 1≤j≤Ji+1}(ct_ij)`
    - `minTC = c·Σ_{v=1..V} y_v + uc·Σ_{i=1..I} Σ_{j=1..Ji+1} (nct_ij − nst_ij + lct_ij − lst_ij)`
    - 其中 `c` = 启用一台 AGV 的启动成本（commissioning cost），`uc` = 单位运输时间的物流成本。验证中 `uc = 0.5`、`c = 7`。
    - 发生故障后目标函数修正为式 (48)：`minTC = c·Σ y_v + uc·( Σ(nct−nst+lct−lst) + at )`，`at` 为被中断物流任务已消耗的运输时间（AGV 空闲时故障则 `at = 0`）。
- **考虑的扰动**：**物流设备（AGV）故障**。论文强调这是被既有动态 PLCSP 研究忽视的扰动——既有研究多关注机器故障与新工件到达，只有 [36] 和 [45] 考虑过物流设备故障，但它们不同时优化生产与物流两类目标。
  - 三类故障场景：SC1 = AGV 空闲时故障；SC2 = non-load 阶段故障；SC3 = load 阶段故障。
  - SC1 就地修理；SC2 移动到起点与终点之间离自己更近的位置修理（避免路中修理干扰其他车间活动）；SC3 同 SC2，且载运工件同步移动——若同步移动到 load 阶段终点则不算被中断任务，可直接在目标机器上继续加工。
  - 受影响任务 = 故障 AGV 上被中断的物流任务（SC1 中不存在） + 故障时刻尚未开始其物流过程的那些工序的生产任务与对应物流任务。
- **约束与假设**（原文逐条）：
  1. All jobs, machines, and AGVs are available initially.
  2. Once an operation starts processing, it cannot be interrupted until completed.
  3. The time for loading and unloading jobs on AGVs is included in transport time.
  4. Machine breakdowns, path conflicts, and AGV charging are ignored.
- **建模补充**：为每个工件的工序序列追加一个**虚拟工序**，关联"完成后退回 L/U 站"的物流任务；L/U 站抽象为能力无限的**虚拟机器**。模型含大量非线性约束（式 8、10、14），通过辅助 0-1 变量 `φ`、`ϕ`、`z` 线性化（式 25–39）。故障响应附加约束为式 (40)–(47)。实例中每个实例的故障次数在 1–8 之间随机，修理时间在 5–10 之间随机；一台 AGV 修复前不会再次故障，同一时刻不超过一台 AGV 故障。故障时间在"最短调度周期"（基于实例规模与专家经验估计的理想值）内随机发生。

### 3. 它怎么做（NHDRL）

**方法全貌**：Nested-Hierarchical Deep Reinforcement Learning（NHDRL）——多智能体 + 分层强化学习思想 + MAPPO 训练。三处分层设计：① 嵌套分层框架；② 为每个 agent 定制 DRL 关键要素（含主观优化偏好信息）；③ MAPPO 训练机制。

**三个 agent 及嵌套层级**（Fig. 3）：

| Agent | 角色 | 层级位置 |
|---|---|---|
| OA（Objective agent，目标智能体） | 上层控制器（controller），选择当前临时优化目标 | 最上层，位于 PA 与 LA 之上 |
| PA（Production agent，生产智能体） | 决定生产调度（选工件 + 选机器） | 下层之上层 |
| LA（Logistics agent，物流智能体） | 决定物流调度（选 AGV） | 最下层 |

- 决策传递逻辑：每个决策点，OA 先选定目标 → 目标决策传给 PA 和 LA → PA 依据自身观测 + OA 目标决策输出生产派工动作 → 动作同时传给环境与 LA → LA 结合环境额外观测 + OA 目标决策输出 AGV 派工动作 → 环境执行全部动作进入新状态并返回奖励。PA 与 LA **共享同一奖励函数**，且共同影响状态转移与奖励反馈；虽然在决策时分别决策，但并非完全独立，训练中会被驱动去相互协作。
- 与既有方法对比（Table 2）：既有 DRL 方法（[25]、[61]）集中式单智能体、无法多目标优化；本文既支持多目标优化，又契合现实中的"生产—物流分权管理"情形。

**状态定义**（统计式降维：对状态集合取 mean / min / max 三种统计量以固定维度）：

- **OA：31 个状态特征** `{f1,…,f31}`：
  1. f1 = 机器数（含虚拟机器）`K+1`
  2. f2–f4 = 真实机器利用率 `MU_k(t)` 的 mean/min/max
  3. f5–f7 = 真实机器最早可用时间 `MCT_k(t)` 的 mean/min/max
  4. f8 = AGV 数量 `V`
  5. f9 = 已启用 AGV 数量（`AV(t) = {v | y_v = 1}` 的元素个数）
  6. f10–f12 = AGV 利用率 `AU_v(t)` 的 mean/min/max
  7. f13–f15 = AGV 最早可用时间 `ACT_v(t)` 的 mean/min/max
  8. f16 = 需处理工序数（含虚拟工序）`Σ_i (Ji+1)`
  9. f17 = 已完成工序累计消耗的总物流成本 `c·Σ_v y_v + uc·Σ_v AW_v(t)`
  10. f18 = 剩余待加工工序数
  11. f19 = 未完成工件数（`CJ(t) = {i | OP_i(t) < Ji+1}` 的大小）
  12. f20 = 全部工序完成率 `Σ_i OP_i(t) / Σ_i (Ji+1)`
  13. f21–f23 = 工件完成率 `CRJ_i(t)` 的 mean/min/max
  14. f24–f26 = 所有未完成工件最早待加工工序的平均加工时间 `APT_{i,OP_i(t)+1}` 的 mean/min/max
  15. f27–f29 = 所有工件剩余加工时间 `RTJ_i(t)` 的 mean/min/max
  16. **f30 = 对 MS 的优化偏好**
  17. **f31 = 对 TC 的优化偏好**
  - 关键定义：`OP_i(t) = Σ_j Σ_k λ_ijk`（已分配工序数）；`CRJ_i(t) = OP_i(t)/(Ji+1)`；`APT_ij = (Σ_k α_ijk·pt_ijk)/(Σ_k α_ijk)`；`RTJ_i(t) = Σ_{j=OP_i(t)}^{Ji+1} APT_ij`；`MCT_k(t) = max_{i,j≤OP_i(t)}(λ_ijk·ct_ij)`；`MU_k(t) = (Σ_i Σ_{j≤OP_i(t)} λ_ijk·pt_ijk)/MCT_k(t)`；`ACT_v(t) = max_{i,j≤OP_i(t)}(x_ijv·lct_ij)`；`AT_v(t)` = 截至 t 因 AGV v 故障导致任务中断而累计的额外运输时间；`AW_v(t) = ΣΣ x_ijv(nct−nst+lct−lst) + AT_v(t)`；`AU_v(t) = AW_v(t)/ACT_v(t)`。
  - **管理者优化偏好**（subjective information）：以 0–10 的整数互补量化，数值越大表示越需要优化对应目标。偏好由企业运营态势或生产需求触发（交期紧则偏好 MS、成本压力则偏好 TC、无特殊影响则接受折中解）。量化到固定区间是为了保证特征一致与稳定，不影响收敛。
- **PA：24 个特征** = f1–f7（机器资源利用信息） + f16–f29（工序加工信息） + f30–f32（决策需求）。其中 **f32 为 OA 的目标决策**（上层对下层的控制信号）。
- **LA：30 个特征** = f8–f15（AGV 资源利用信息） + f16–f18 与 f33–f48（工序加工信息） + f30–f32（决策需求）。f33–f48 为与所分配物流任务直接相关的信息，例如：
  - f33 = 所分配工序 O_ij 的最早可运输时刻 `ct_{i,OP_i(t)}`
  - f34 = load 阶段终点位置；f35 = non-load 阶段终点位置
  - f36 = LT_ij 最早期望到达时刻（所分配机器 k 的最早可用时间）`MCT_k(t)`
  - f37–f39 = 已启用 AGV 完成所分配物流任务全部阶段所需时间的 mean/min/max，`TV_v(t) + tt_{l'l}`
  - f40–f42 = 对应所需成本的 mean/min/max，`uc·(TV_v(t) + tt_{l'l})`
  - f43–f45 = 完成 non-load 阶段所需时间的 mean/min/max，`TV_v(t)`
  - f46–f48 = 对应所需成本的 mean/min/max，`uc·TV_v(t)`
  - `TV_v(t) = tt_{l''l'}`，其中 `l''` 是 AGV v 完成上一任务后的位置（若尚未分配任务则为 L/U 站位置），`l'` 是 `O_{i,OP_i(t)}` 所分配机器位置（若 `OP_i(t)=0` 则为 L/U 站位置）。
  - 状态转移后与所分配物流任务相关的状态值填 0。

**动作空间**：

- **OA：2 个动作** —— Action 1 选择优化 MS；Action 2 选择优化 TC。
- **PA：9 个复合派工规则（CDR）** —— 3 个工件选择规则 × 3 个机器分配规则：

| 规则 | 描述 |
|---|---|
| Rule 1 | SPT（选最早待加工工序加工时间最短的工件） + SMPT（选加工时间最短的可用机器） |
| Rule 2 | SPT + MINU（选利用率最低的可用机器） |
| Rule 3 | SPT + EAM（选最早可用机器） |
| Rule 4 | EAJ（选能被最早加工的工件） + SMPT |
| Rule 5 | EAJ + MINU |
| Rule 6 | EAJ + EAM |
| Rule 7 | SRPT（选剩余加工时间最短的工件） + SMPT |
| Rule 8 | SRPT + MINU |
| Rule 9 | SRPT + EAM |

- **LA：3 个动作**：
  - Rule 1：选完成所分配物流任务所需运输成本最低的 AGV
  - Rule 2：选利用率最低的 AGV
  - Rule 3：选能在"所分配工序最早可运输时刻"之前最晚完成 no-load 阶段的 AGV；若无合格 AGV，则选能最早完成 no-load 阶段的 AGV
- 补充：PA / LA 应用派工规则后若多个工件/机器/AGV 满足条件，则随机选一个。该设计保证解满足同工件工序顺序约束与加工能力约束。

**奖励函数**（两套，按 OA 当前选择的临时目标选择性调用；两个奖励信号按权重聚合）：

- MS 相关奖励（式 50、51）：
  - `RMS_t = w1·nms_t + w2·( p_ms / maxp · nms_t )`
  - `nms_t = (ms_t − min_ms)/(max_ms − min_ms)` 若 `max_ms ≠ min_ms`；否则 `= 1`
- TC 相关奖励（式 52、53）：
  - `RTC_t = w1·ntc_t + w2·( p_tc / maxp · ntc_t )`
  - `ntc_t = (max_tc − tc_t)/(max_tc − min_tc)` 若 `max_tc ≠ min_tc`；否则 `= 1`
- 含义：`ms_t` = 状态转移前后 MS 之差（目标优化程度信号）；`tc_t` = 状态转移后产生的物流成本；两者均先归一化以缓解目标切换导致的奖励尺度波动、并利于稳定收敛。`p_ms`、`p_tc` 分别为对 MS / TC 的优化偏好，`maxp` 为单个目标可能的最大偏好值。`w1` 为"目标优化程度"信号权重，`w2` 为"偏好满足程度"信号权重。
- 权重调节示例（原文）：当管理者强调满足优化偏好时，可适当增大偏好满足度奖励信号的权重；当管理者无特定偏好时，可适当增大目标优化程度奖励信号的权重，引导 agent 在两个目标间取得良好折中。本文非敏感性实验部分设 `w1 = 0.8`、`w2 = 0.2`，以优先保证两目标的均衡优化、确保与不考虑偏好的基线方法公平比较。

**训练机制（MAPPO，Algorithm 1）**：

- 输入：训练 epoch 数 L；OA/PA 回放记忆容量 `C_high`；LA 回放记忆容量 `C_low`；minibatch 大小 `bs`；ε-greedy 参数 ε；ε 下界 `LB`；PPO 其他参数。
- 流程要点：
  1. 分别初始化三者 actor 网络 `π_oa(θ_oa)`、`π_pa(θ_pa)`、`π_la(θ_la)` 与 value 网络 `v_oa(ω_oa)`、`v_pa(ω_pa)`、`v_la(ω_la)`，以及各自回放记忆 `D_oa(C_high)`、`D_pa(C_high)`、`D_la(C_low)`。
  2. 每个 epoch 随机生成工件与故障相关信息初始化环境。
  3. 逐决策点依框架逻辑交互（OA → PA → LA），存储 transition `(S_t, A_t, R_t, S_{t+1})` 到各自记忆。
  4. 记忆中样本数达到 `bs` 时，三者分别用 PPO 目标（式 49）经 Adam 随机梯度上升更新。
  5. `ε ← max(ε*0.99, LB)`。
- **决策点定义**：一个工序被分配的时刻，或发生 AGV 故障的时刻。
- **PPO 目标（式 49）**：`π* = argmax_{π(θ)} E_t[ min( ratio_t(θ)·AF_t, clip(ratio_t(θ), 1−cl, 1+cl)·AF_t ) ]`，其中 `ratio_t(θ) = π(s_t,a_t;θ)/π_old(s_t,a_t;θ_old)`，`cl` 为 clip 参数，`AF_t` 为用 GAE 估计的优势函数。
- **探索**：ε-greedy 结合噪声——以概率 ε 向 actor 输出的动作概率分布加随机噪声以影响动作选择，ε 逐步下降至下界 LB。
- **非平稳性处理**（关键设计）：由于嵌套层级结构，下层 agent 行为变化会使上层 agent 面临非平稳问题——下层策略更新后，同一上层决策可能触发下层不同反馈，使上层在更新前积累的经验失效。为此训练机制**将上层 agent 的回放记忆容量设为与 minibatch 大小一致**，使上层 agent 近似在线（online）模式训练。
- **网络结构（Table 4）**：

| 网络 | 层 | 节点数 | 激活函数 |
|---|---|---|---|
| Actor | 输入层 | 该 agent（OA/PA/LA）的状态特征维度 | ReLU |
| Actor | 隐藏层 | 256 | ReLU |
| Actor | 输出层 | 该 agent 的动作维度 | Tanh |
| Critic | 输入层 | 该 agent 的状态特征维度 | ReLU |
| Critic | 隐藏层 | 128 | ReLU |
| Critic | 输出层 | 1 | Tanh |

- **超参数（Table 6）**：训练 epoch 数 L = 1000；OA/PA/LA 回放记忆容量 = 32/32/500；minibatch size = 32；ε 初值 = 0.9；ε 下界 = 0.1；clip 参数 = 0.2。
- **超参数调优（Table 5、Table 7）**：网格搜索 3 水平学习率 × 3 水平折扣因子 = 9 组合；学习率水平 1/2/3 分别为 OA,PA,LA = (5e-5,5e-5,5e-4) / (1e-4,1e-4,1e-3) / (5e-4,5e-4,5e-3)；折扣因子水平 1/2/3 = 0.59 / 0.79 / 0.99。综合值 CV（式 54）归一化后求和，越小越好。结论：学习率取水平 2、折扣因子取水平 3（即 OA/PA/LA 学习率 = 1e-4 / 1e-4 / 1e-3，三者折扣因子 = 0.99）。
- **实施阶段**：与单次训练过程类似，但省略记忆存储（Algorithm 1 第 18 行）与耗时网络更新（第 21–26 行），因此可对常规决策或扰动做近实时（near real-time）响应；动作选择时丢弃 ε-greedy，直接选概率最高的动作。

### 4. 它怎么验证

**实例来源与规模**：以真实**航空零部件生产车间**（典型复杂 FJS）为验证环境。

- 车间配置：18 台多用途机器 + 1 个 L/U 站 + 6 台 AGV；可加工 7 类航空零部件（压气机盘 compressor disk、涡轮盘 turbine disk、风扇盘 fan disk、整体鼓筒 integral drums、压气机前轴颈 compressor front journals、压气机后轴颈 compressor rear journals、长轴 long shafts）；各类型工序数为 8, 8, 8, 8, 9, 6, 7；每道工序的可用机器固定；车间布局与两两位置间的转运路线固定且已知。
- 环境参数：`K = 18`，位置数 `L = 18`，`V = 6`；`uc = 0.5`，`c = 7`。
- 生产参数：待加工工件数 `I` 在 10–40 之间随机；每个工件随机对应一类可加工零部件；`Ji` 取值于 [6, 9]；每道工序在可用机器上的加工时间在 2–10 之间随机。
- 扰动参数：每个实例的故障总次数在 1–8 随机；故障在最短调度周期内随机发生并随机指派给某台 AGV；每次故障的修理时间在 5–10 随机。
- 实例命名："Exp (a/b)"，a = 工件数，b = 工序数。测试实例从 `Exp(7/59)` 到 `Exp(90/801)`；大规模实例为 `Exp(50/425)`、`Exp(60/512)`、`Exp(70/623)`、`Exp(80/702)`、`Exp(90/801)`；静态小规模对比 Gurobi 的实例为 `Exp(2/16)`、`Exp(3/23)`、`Exp(4/32)`、`Exp(5/36)`、`Exp(5/39)`。
- 另设计了一个**物流资源更稀缺的仿真车间**用于模型评估：1 个 L/U 站 + 8 台多用途机器 + 3 台 AGV；5 类工件，工序数 7, 7, 7, 6, 6（原文此处写 `K = 18, L = 18, V = 6, Ji ∈ [6,7]`，与前文描述不一致，疑为原文笔误）。

**对比基线（列全）**：

1. **27 条复合派工规则（CDRs）** —— 由 PA 的 9 个动作与 LA 的 3 个动作两两组合生成；表中只列表现最好的 3 条规则。
2. **RN**（Random version of NHDRL）—— 每个决策点随机为每个 agent 选动作，用于验证学习效果。
3. **DQN-based method** [29]（Luo, 2020, Appl. Soft Comput.）
4. **DDPG-based method** [13]（Gui et al., 2023, Comput. Ind. Eng.）
5. **NSGA-II-R** —— NSGA-II [8] + right-shift rescheduling（右移重调度）
6. **NSGA-II-C** —— NSGA-II + complete rescheduling（完全重调度）
   - NSGA-II 编码：三维等长编码（第 1 维工序序列信息、第 2 维机器分配、第 3 维 AGV 分配）；顺序解码；交叉 = 第 1 维用 precedence operation crossover，第 2/3 维用 multipoint preservative crossover；变异 = 第 1 维 swap mutation，第 2/3 维 replace mutation；种群 30，交叉率 0.8，变异率 0.3；迭代数分别设 100 和 200。
7. **Gurobi 11.0.3**（商用求解器）——双目标用经典加权和法转单目标，两目标权重均 0.5；最大执行时间 3600 s；仅在**无扰动的静态实例**上比较（因所提模型只能在无扰动情形下用 Gurobi 求解）。
8. **消融变体**：N-OA（去掉上层 OA）、N-CM（用集中式管理替代生产/物流分权管理，状态特征为 OA 状态特征 + 目标决策的组合，动作为 27 条 CDR）、N-PA（用最佳生产派工规则替代 PA）、N-LA（用最佳 AGV 派工规则替代 LA）。
9. **敏感性变体**：N-OP（`w1=0.8, w2=0.2`，侧重目标折中优化）、N-PS（`w1=0.2, w2=0.8`，侧重满足偏好）。
10. **PLISP**（Production and Logistics Independent Scheduling Model）——传统生产与物流独立、顺序调度的模型，用于评估所提 DFJS-PLCSP 模型本身。

**评估指标**：

- 目标函数值（MS、TC）平均值
- **SP（spacing）indicator** [4]（分布均匀性）—— 越小越好
- **HV（hypervolume）indicator** [65]（收敛性与多样性）—— 越大越好
- **Excellent rate**（优越率）：所提方法优于或等于的规则数 / 规则总数
- 运行时间（Time）：各方法输出有效完整调度方案所需时间
- **Gap**（式 55、56）：`Gap_MS = (g^NHDRL_MS − g^Gurobi_MS)/g^Gurobi_MS`，`Gap_TC = (g^NHDRL_TC − g^Gurobi_TC)/g^Gurobi_TC`
- **CV**（式 54）：`CV = Σ_{ob=1..OB} (ρ^ave_ob − ρ^min_ob)/(ρ^max_ob − ρ^min_ob)`，用于超参数设计
- 比较协议（与相关研究 [30,31,47] 一致）：每个实例独立运行各方法 **20 次**，用 Pareto 支配概念从重复运行解中提取各方法的 Pareto 解集，再比较这些解集在 HV 与 SP 上的表现。

**实验环境（硬件/软件）**：Python 3.6 + PyCharm；调用 Numpy、Torch、Pandas 等库；Windows server，Intel Xeon Gold 6146 3.2 GHz CPU，128 GB RAM，NVIDIA GeForce RTX 2080ti GPU。

### 5. 核心结果（照抄原文数据）

**训练**：训练 1000 个 epoch；累计奖励随 epoch 增加逐步上升并最终收敛到稳定区间；**平均每个 epoch 耗时 6.78 s**，达到收敛（约 400 epochs）所需训练时间可接受。每个 epoch 结束在测试实例 `Exp(16/140)` 上测试并记录累计奖励。

**与其他实时调度方法比较（Table 8 SP / Table 9 HV，18 个测试实例）**：

- HV 典型值：`Exp(7/59)` NHDRL 0.8175 vs RN 0.3352 vs CDRs 0.5356 vs DQN 0.5356 vs DDPG 0.5356（Excellent rate 100%）；`Exp(16/140)` NHDRL 0.8225 vs RN 0.3378 vs CDRs 0.2642；`Exp(20/173)` NHDRL 0.8151 vs RN 0.3386 vs CDRs 0.2634 vs DQN 0.2442 vs DDPG 0.2435；`Exp(34/305)` NHDRL 0.7889 vs RN 0.2475 vs CDRs 0.1788 vs DQN 0.1651 vs DDPG 0.1619。表中 HV 列 Excellent rate 几乎全部为 100%。
- SP 典型值：NHDRL 多数实例为 0.0000（如 `Exp(7/59)`、`Exp(8/70)`、`Exp(10/86)`、`Exp(16/140)`、`Exp(18/156)`、`Exp(20/173)` 等），个别为 0.0002（`Exp(21/185)`）、0.0015（`Exp(23/203)`）、0.0087（`Exp(36/316)`）；RN 为 0.0000–0.0514，CDR 为 0.0000–0.0072。
- 原文结论：NHDRL 在几乎所有实例上 SP 与 HV 均优于 RN；HV 上优于 CDRs 与其他 DRL 方法，说明其 Pareto 解集有更优的收敛性与多样性。

**大规模实例（Table 10 SP / Table 11 HV）**：

- HV：`Exp(50/425)` NHDRL 0.7073 vs RN 0.2711 vs CDRs 0.3537 vs DQN 0.3536 vs DDPG 0.3535；`Exp(60/512)` 0.7100 vs 0.2224 vs 0.3314；`Exp(70/623)` 0.6905 vs 0.2383 vs 0.2413；`Exp(80/702)` 0.6865 vs 0.2381 vs 0.2175；`Exp(90/801)` 0.6599 vs 0.2164 vs 0.2046。Excellent rate 全部 100%。
- SP：NHDRL 全部为 0.0000；RN 为 0.0024–0.0148；CDRs 为 0.0001–0.0013。

**与元启发式（NSGA-II-R / NSGA-II-C）比较（Table 12，5 个实例）**：

| 实例 | NHDRL HV | NHDRL SP | NHDRL Time (s) | NSGA-II-R 100 迭代 HV | NSGA-II-R 200 迭代 HV | NSGA-II-C 100 迭代 HV | NSGA-II-C 200 迭代 HV |
|---|---|---|---|---|---|---|---|
| Exp (12/98) | 0.796 | 0.000 | 0.62 | 0.398 (89.90 s) | 0.508 (135.06 s) | 0.289 (160.26 s) | 0.393 (242.78 s) |
| Exp (16/132) | 0.748 | 0.003 | 0.81 | 0.424 (118.15 s) | 0.577 (177.43 s) | 0.338 (223.43 s) | 0.528 (335.56 s) |
| Exp (20/176) | 0.645 | 0.000 | 1.37 | 0.309 (155.69 s) | 0.433 (235.04 s) | 0.279 (295.17 s) | 0.386 (446.73 s) |
| Exp (23/192) | 0.584 | 0.000 | 1.54 | 0.355 (171.21 s) | 0.449 (255.76 s) | 0.231 (312.15 s) | 0.379 (466.08 s) |
| Exp (27/240) | 0.558 | 0.003 | 2.06 | 0.327 (215.75 s) | 0.392 (320.49 s) | 0.274 (371.23 s) | 0.366 (564.04 s) |

- 原文结论：NHDRL 在所有实例上 HV 与 SP 均优于基线；基线获得有效完整方案所需运行时间显著更长，且随问题规模增大而增加；NHDRL 的运行时间显著低于基线，具有更优的时间效率。

**与商用求解器（Gurobi）比较（Table 13，静态小规模实例）**：

| 实例 | NHDRL MS | NHDRL TC | NHDRL Time (s) | Gurobi MS | Gurobi TC | Gurobi Time (s) | Gap MS | Gap TC |
|---|---|---|---|---|---|---|---|---|
| Exp (2/16) | 70.7 | 47.5 | 0.12 | 70.7 | 47.5 | 82 | 0.00% | 0.00% |
| Exp (3/23) | 75.5 | 74.0 | 0.15 | 75.5 | 71.0 | 3600 | 0.00% | 4.23% |
| Exp (4/32) | 75.8 | 99.0 | 0.16 | 76.3 | 91.0 | 3600 | −0.66% | 8.79% |
| Exp (5/36) | 82.2 | 115.5 | 0.22 | 89.0 | 120.0 | 3600 | −7.64% | −3.75% |
| Exp (5/39) | 82.2 | 107.5 | 0.23 | 83.4 | 119.5 | 3600 | −1.44% | −10.04% |

- 原文结论：多数实例中 Gurobi 通常需要耗尽最大允许时间才能找到高质量解，而 NHDRL 能在显著更短时间内取得与 Gurobi 相近或略差的解，在解质量与计算成本间取得良好平衡。

**消融实验**：

- NHDRL vs N-OA（Table 14）：NHDRL 在 SP 与 HV 上均更优。8 个实例的 HV：如 `Exp(9/78)` 0.7620 vs 0.4253；`Exp(11/94)` 0.8078 vs 0.4904；`Exp(17/148)` 0.7858 vs 0.5675。
- NHDRL vs N-CM / N-PA / N-LA（Table 15）：8 个实例的 HV：如 `Exp(9/78)` NHDRL 0.7620 / N-CM 0.5604 / N-PA 0.5418 / N-LA 0.3038；`Exp(11/94)` 0.8078 / 0.2007 / 0.6381 / 0.3405；`Exp(32/276)` 0.7915 / 0.2385 / 0.4785 / 0.1681。
- 原文注明：NHDRL 的 TC 结果优于 N-PA 与 N-LA；尽管 NHDRL 在 TC 上略差于 N-CM，但综合 MS 与 TC 来看 NHDRL 结果更均衡，而非像 N-CM 那样以过度牺牲 MS 换取 TC 的小幅改善。

**敏感性分析（Fig. 9）**：N-PS 在偏好目标上更好但另一目标较差（更专注优化偏好目标以满足管理者需求）；N-OP 在偏好目标上相对 N-PS 表现较弱，但在另一目标上取得不可忽略的优化（强调两目标的均衡优化）。原文结论：NHDRL 能按权重设置达到期望的学习效果。

**模型评估（Table 16，DFJS-PLCSP vs PLISP，20 次重复平均）**：

| 环境 | 实例 | MS (DFJS-PLCSP) | MS (PLISP) | TC (DFJS-PLCSP) | TC (PLISP) |
|---|---|---|---|---|---|
| 原始对比环境 | Exp (11/94) | 189.05 | 342.50 | 284.60 | 367.00 |
| 原始对比环境 | Exp (14/119) | 257.87 | 490.50 | 370.40 | 477.00 |
| 原始对比环境 | Exp (18/160) | 388.68 | 487.30 | 498.33 | 644.00 |
| 原始对比环境 | Exp (21/182) | 491.64 | 621.40 | 592.70 | 748.00 |
| 原始对比环境 | Exp (23/205) | 518.13 | 841.20 | 648.80 | 813.50 |
| 新增仿真环境 | Exp (9/68) | 339.63 | 407.10 | 336.00 | 361.60 |
| 新增仿真环境 | Exp (14/107) | 627.41 | 700.00 | 494.21 | 499.10 |
| 新增仿真环境 | Exp (21/160) | 924.46 | 971.30 | 798.14 | 817.40 |

- 原文结论：无论在哪个环境，所提模型在 MS 与 TC 上均优于 PLISP；Fig. 11 的 Gantt 图显示所提模型的调度更紧凑、AGV 利用率更高。

### 6. 它自己承认的局限（原文逐字引用 + 中文翻译）

> **原文**："In the future, more practical factors, such as machine breakdown, path conflicts, and AGV charging, can be incorporated into the proposed model to further enhance its practicality."
>
> **中文**：未来可将更多实际因素，如机器故障、路径冲突和 AGV 充电，纳入所提模型以进一步增强其实用性。

> **原文**："Meanwhile, attempts can also be made to extend the proposed method in the following aspects. First, since the agent's actions in the proposed method are formed by directly selecting the dispatching rules based on the designers' scheduling experience, this method is not objective enough in action design. More effective action design strategies, such as genetic programming or data mining, can be investigated to improve the interpretability of current action design."
>
> **中文**：同时，也可在以下方面尝试对所提方法进行扩展。第一，由于所提方法中智能体的动作是基于设计者的调度经验直接选择派工规则而形成的，该方法在动作设计上不够客观。可研究更有效的动作设计策略，如遗传编程或数据挖掘，以提升当前动作设计的可解释性。

> **原文**："Second, although the proposed method improves the comprehensiveness of agents' state feature elements by integrating both subjective and objective information, it ignores to enrich the state representation types. Using solely numerical states may limit the agents' in-depth understanding of environments. Hence, future research can attempt to incorporate multi-modal states to improve the agents' environmental perception. Meanwhile, using the specific distributions to characterize the state sets regarding resource utilization or operation processing situation is also a future attempt, which may provide agents with rich shop state information and drive their learning."
>
> **中文**：第二，尽管所提方法通过融合主观与客观信息提升了智能体状态特征要素的全面性，但它忽视了丰富状态表示类型这一点。仅使用数值型状态可能限制智能体对环境的深入理解。因此，未来研究可尝试引入多模态状态以提升智能体的环境感知能力。同时，用具体分布来刻画关于资源利用或工序加工情况的状态集合也是一个未来尝试方向，这可能为智能体提供丰富的车间状态信息并驱动其学习。

> **原文**："Moreover, the integration of human-related factors within the proposed method is still relatively single. Driven by the human-centric paradigm advocated by Industry 5.0, it is also desirable to incorporate more human-related factors, such as decision-making experience of human experts or their decision-making preference in other aspects, to improve the proposed method's human involvement."
>
> **中文**：此外，所提方法中与人的因素相关的整合仍相对单一。在工业 5.0 所倡导的以人为本范式驱动下，也宜纳入更多与人的因素相关的内容，如人类专家的决策经验或他们在其他方面的决策偏好，以提升所提方法的"人的参与度"。

（另注：假设中原文已自陈 "Machine breakdowns, path conflicts, and AGV charging are ignored."——机器故障、路径冲突与 AGV 充电被忽略。）

### 7. 有什么可以拿来用

- **开源代码**：无。文中未提供任何代码仓库链接。**数据可用性声明（原文）**："Data will be made available on request."（数据可应要求提供）。
- **公开数据集**：无公开数据集；实例由作者面向某航空零部件车间自建（`Exp(a/b)` 命名体系）。
- **可借鉴的评测协议**：
  - **双目标实时调度评测协议**：每实例独立运行各方法 **20 次** → 用 Pareto 支配从重复运行中提取 Pareto 解集 → 比较 **HV**（收敛性+多样性）与 **SP**（分布均匀性）；并给出 **Excellent rate = 优于或等于的规则数 / 规则总数** 来对 27 条 CDR 做整体对比。这是一套可直接复用的、面向"实时调度方法 vs 大规模规则集"的公平比较范式。
  - **RN（Random version）作为下界对照**：把自身方法的动作随机化作为 sanity check，验证智能体确实学到了一组高效行为策略，而非随机选择可行动作。方法简单但说服力强。
  - **4 组消融设计**：N-OA（去上层目标 agent）、N-CM（分权 → 集权）、N-PA / N-LA（下层 agent → 最佳固定规则），分别验证"多目标分层""分权管理""学习型 agent 优于固定规则"三个论点。
  - **超参数综合值 CV（式 54）**：多目标下把各目标归一化后求和作为单一标量评分来选超参，避免只看单一目标。
  - **Gurobi 对比的正确姿势**：限定在**无扰动的静态小规模实例**上、设 3600 s 上限、双目标用加权和（权重 0.5/0.5）转单目标，并报告 MS/TC 两个 Gap。
  - **与元启发式对比时同时报告解质量与运行时间**（Table 12），并区分 Iteration-100 / Iteration-200 两档。
- **有价值的公式**：
  - **双目标奖励函数（式 50–53）**：`R_t = w1·(归一化目标优化程度) + w2·(偏好值/最大偏好 × 归一化目标优化程度)`，用权重在"目标折中"与"满足管理者偏好"之间切换——这是把**人类偏好/主观信息量化进 DRL 奖励**的一个具体可抄写法（偏好用 0–10 整数互补量化，`maxp = 10`）。
  - **总物流成本目标（式 2 / 式 48）**：`TC = c·(启用 AGV 数) + uc·(总运输时间) + uc·at`——把"启用成本"与"单位时间运输成本"解耦的写法，可直接迁移到物流成本建模。
  - **AGV 故障三场景（SC1/SC2/SC3）及其修复策略**，以及对应的中断响应约束（式 40–47）：SC3 中"载运工件随 AGV 同步移动到 repair location，若同步到达 load 阶段终点则不算中断任务"这一处理很具体，可作为"AGV 故障"仿真的参考实现。
  - **非平稳性对策**：把上层 agent 的回放记忆容量设为等于 minibatch 大小，使其近似在线训练，从而缓解分层结构中下层策略更新引起的上层经验失效——对任何分层/多智能体 DRL 都适用。
  - **统计式状态降维**：对不定长状态集合（机器集合、AGV 集合、工件集合）统一取 mean/min/max 三个统计量构造固定维度特征（OA 31 维、PA 24 维、LA 30 维），并说明该方法"与应用环境无关、具有鲁棒性"。
  - **决策点定义**：`决策点 = 一个工序被分配的时刻 或 发生 AGV 故障的时刻`——事件驱动的决策点定义可直接复用。

---

## 卡片 2：TSMC2025 — Real-time FJSP with AGVs, MARL + Efficient Action Decoding

### 1. 题录

- **标题**：Real-Time Scheduling for Flexible Job Shop With AGVs Using Multiagent Reinforcement Learning and Efficient Action Decoding
- **作者**：Yuxin Li, Qingzheng Wang, Xinyu Li (通讯, Member IEEE), Liang Gao (Senior Member IEEE), Ling Fu, Yanbin Yu, Wei Zhou
  - Yuxin Li, Qingzheng Wang, Xinyu Li, Liang Gao：State Key Laboratory of Intelligent Manufacturing Equipment and Technology, Huazhong University of Science and Technology, Wuhan 430074, China
  - Ling Fu, Yanbin Yu：Department of Simulation and Digital Twin, Siemens Technology, Shanghai 200082, China
  - Wei Zhou：Department of Simulation and Digital Twin, Siemens Technology, Wuhan 430074, China
- **期刊**：IEEE Transactions on Systems, Man, and Cybernetics: Systems, Vol. 55, No. 3, March 2025, pp. 2120–2132
- **年**：2025（Received 11 September 2024; Revised 4 November 2024; Accepted 8 December 2024; Date of publication 7 January 2025; Date of current version 19 February 2025）
- **DOI**：10.1109/TSMC.2024.3520381
- **Supplementary**：作者提供补充材料，彩色图见 https://doi.org/10.1109/TSMC.2024.3520381
- **Index Terms**：Automated guided vehicle (AGV), disturbance events, flexible job shop, multiagent reinforcement learning (MARL)
- **资助**：NSFC 52188102, U21B2029；中央高校基本科研业务费 2024BRA004；湖北省重点研发计划 2021AAB001
- **Associate Editor**：D. O. Olson

### 2. 它解决什么问题

- **问题类型**：**带 AGV 的动态柔性作业车间调度问题（DFJSP-AGVs, Dynamic Flexible Job Shop Scheduling Problem with AGVs）**。原文指出该问题三个特征：① 机器柔性、有限 AGV、生产—物流过程高度耦合使问题非常复杂；② 客户对交货期有要求；③ 车间扰动事件频繁发生——这使静态调度方法（如 MILP）不可行，因为每次事件都会让原方案与实际生产不一致。
- **涉及资源**：多类型机器（每类型含多台，每台有输入缓冲区 input buffer 与输出缓冲区 output buffer）+ 多台 AGV（初始都在 AGV 区域）。另有原材料区与成品区。
- **优化目标**：**最小化总拖期成本（total tardiness cost, TTC）**：
  - `min TTC = min Σ_{i=1..n} ( w_i · max(0, C_i − D_i) )`（式 1）
  - `w_i` 为工件 `J_i` 的权重（代表紧急程度），`C_i` 为工件 `J_i` 的完成时间。
- **考虑的 4 类扰动事件**（Section III-B，原文列举）：
  1. **New Job Arrivals（新工件到达）**：新工件的到达间隔服从指数分布 `exp(1/MTBA)`，`MTBA` 为平均到达间隔。考虑物流因素后的 MTBA（式 2）：
     `MTBA = (N_ope · TPT)/(U · Σ_{x=1..m} N_x) + (N_ope + 1) · TLT/(U · l)`
     其中 `U` 为车间利用率，`N_ope` 为工件的平均工序数，`TPT` 为所有加工任务的平均耗时，`TLT` 为所有物流任务中配送部分的平均耗时。含义：到达间隔 = 总任务量 / 车间（加工与物流）产能。`U` 越高，新工件到达越频繁。
  2. **Machine Breakdowns and Repairs（机器故障与修复）**：① 机器故障时立即停止加工，修复后恢复加工能力；② 对故障时正在加工的工件，采用 **from scratch with the interrupt-repeat mode**（从头重做、中断—重复模式）；③ 机器修复时间不可预测。两次故障间隔与修复时间分别服从 `exp(1/MTBF)` 与 `exp(1/MTTR)`。车间故障水平（式 4）：`Ag = MTTR/(MTBF + MTTR)`。MTTR 固定时 `Ag` 越大则 MTBF 越小、故障越频繁。
  3. **Job Reworks（工件返工）**：因材料质量、工人操作不当等原因导致工件质量不合格，需要返工。
  4. **Fuzzy Transportation Time（模糊运输时间）**：因路径拥堵、AGV 加速、AGV 减速等原因，AGV 实际运输时间会在预定时间基础上波动。
- **工件交期**（式 3，Total Work Content 方法 + 物流因素）：
  `D_i = A_i + DDT_i · Σ_{j=1..N_i} ( Σ_{k=1..N_x} TPT_ijxk / N_x ) + DDT_i · (N_i + 1) · TLT`
  `DDT_i` 为工件 `J_i` 的交期松紧度（due date tightness），越小越可能延迟。`A_i`、`D_i` 分别为工件 `J_i` 的到达时间与交期。
- **约束与假设**：原文将假设放在 "Section A of Supplementary Material"（补充材料 A 节），正文未逐条列出。正文描述：相邻两道工序对应不同的机器类型；工件按序加工；新工件到达时先存放于原材料区，全部加工完成后需返回成品区存放。

### 3. 它怎么做

**方法全貌**：**多智能体强化学习（MARL）实时调度方法 + 高效动作解码算法**，训练算法为 **MAPPO**（Multi-Agent PPO）。整体是"离线训练 + 在线应用"两模式。

**整体框架（Fig. 2）**：

- 车间环境与决策系统通过实时信息交互（工件数据、机器数据、AGV 数据）；决策系统主要包括 **task pool（任务池）** 与 **三个 agent 的神经网络模型**。
- **Task pool** 存两类任务：`TPool_1` 目的地是机器（对应某工件的一道工序，含一个 processing task 与一个 logistics task）；`TPool_2` 目的地是成品区（对应工件的最后一个物流任务，**只含一个 logistics task**）。每个 logistics task 由 pickup 部分与 delivery 部分组成。
- **决策点定义**：当任务池非空且存在一台或多台空闲 AGV 时，当前时刻即为决策点。
- 机器运行逻辑：机器空闲时用 **FCFS** 从输入缓冲区取已到达的工件加工。AGV 运行逻辑：AGV 空闲时用 **FAFS（first assign first service）** 从任务列表中选一个物流任务执行。
- **Task release（任务释放）**：开始时工件释放其第一道工序的任务；此后当工件只剩一个已被分配 AGV 但尚未完成的任务时，释放其下一个任务。该释放时刻**既减少工件等待 AGV 的时间，又避免不必要的过早决策**（原文基于 [23] 的 earliest task release moment 改进而来）。决策后任务池删除已分配任务。

**多智能体调度架构（Fig. 3）—— 三个 agent，串行（serial）关系**：

| 步骤 | Agent | 输入 | 输出 |
|---|---|---|---|
| Step 1 | **Task selection agent (TS)** | 归一化的全局状态 + 任务池状态 `S_TS = {SF1, SF2, SF3, SF4}` | 规则权重向量，经动作解码 + 剔除不可用任务（所有机器候选都故障的任务）后从任务池选出一个任务 |
| Step 2 | **Machine selection agent (MS)** | 归一化全局状态 + 机器候选状态 `S_MS = {SF1, SF2, SF3, SF5}` | 若选中任务属于 `TPool_1`，经动作解码 + 剔除故障机器后选一台机器；若属于 `TPool_2`，动作置为 **do-nothing action δ0** |
| Step 3 | **AGV selection agent (AS)** | 归一化全局状态 + AGV 候选状态（全部 AGV）`S_AS = {SF1, SF2, SF3, SF6}` | 经动作解码选一台 AGV 运输 |

- 输入输出采用串行关系，使每个 agent 都能明确自己的决策对象并使用其详细数据作为输入。

**状态空间（Table I + Table II）**：实时车间状态分六部分，`{SF1, SF2, SF3}` 为全局状态，`{SF4, SF5, SF6}` 描述不同的决策对象。三个 agent 的状态空间分别为 `S_TS = {SF1, SF2, SF3, SF4}`、`S_MS = {SF1, SF2, SF3, SF5}`、`S_AS = {SF1, SF2, SF3, SF6}`。

- 大多数状态特征是统计型的（statistical），使每个 agent 能通过输入状态理解实时生产信息。
- **归一化**：采用 **Z-Score normalization** 缓解不同状态特征尺度差异带来的负面影响。
- 关键定义（Section IV-B，式 5–11）：
  - `RTP_g`（式 5）= 任务剩余时间：`Σ_{j=1..Ng}(TPT_gj − FTPT_gj) + Σ_{a=1..Ng+1}(TDLT_ga − FTDLT_ga)`
  - `EATP_g`（式 6）= 最早可用时间：`Σ_{a=1..Jth}(TDLT_ga − FTDLT_ga) + Σ_{j=1..Jth}(TPT_gj − FTPT_gj)`
  - `EWTP_g`（式 7）= 估计加权拖期：`max(0, t + RTP_g − D_g) · w_g`
  - `CTM_xk`（式 8）= 机器最早完成时间：`RTINGM_xk + Σ_{w=1..Nxk} TPT_xkw`
  - `DPT_gxk`（式 9）= `Δt(L1, L2) + TPT_gxk`（`L1` 为任务 `TP_g` 前一任务的位置，`L2` 为机器 `M_xk` 的位置，`Δt` 为位置间运输时间）
  - `CTA_y`（式 10）= AGV 最早完成时间：`RTINGA_y + Σ_{e=1..Ny} TLT_ye`
  - `PUT_yg`（式 11）= AGV 取件时间：`Δt(L1, L2)`（`L1` 为物流任务 `TLT_yNy` 的目的地，`L2` 为所选任务 `TP_g` 的取件位置）

**动作空间（PDR weighting + PDR adjustment）**：

- **Task selection agent**：涉及的 PDR 为 ① BEWT / ② LWKR / ③ CR / ④ EDD / ⑤ S/RT —— 分别选 `EWTP_g` 最大 / `RTP_g` 最小 / 关键比最小 / `D_g` 最早 / 单位剩余时间松弛最小的工件。agent 输出**规则权重向量**：
  - `a_TS(t) = [w_BEWT, w_LWKR, w_CR, w_EDD, w_S/RT]`（式 12）
- **Machine selection agent**：涉及 PDR 为 ① SURM / ② SDPT / ③ SCTM —— 分别选利用率最小 / `DPT_gxk` 最小 / `CTM_xk` 最小的机器。
  - `a_MS(t) = [w_SURM, w_SDPT, w_SCTM]`（式 13）
- **AGV selection agent**：涉及 PDR 为 ① SPUT / ② SCTA / ③ SCPT —— 分别选 `PUT_yg` 最小 / `CTA_y` 最小 / `CTA_y + PUT_yg` 最小的 AGV。
  - `a_AS(t) = [w_SPUT, w_SCTA, w_SCPT]`（式 14）

**动作解码算法（Algorithm 1）**——核心创新：

1. 计算 PDR 优先级集合 `(PDR_TS, PDR_MS, PDR_AS)`。
2. 通过 **PDR weighting（PDR 加权）** 与 **PDR adjustment（PDR 调整）** 得到所有任务、所有机器候选、所有 AGV 候选的优先级向量 `(PV_TS, PV_MS, PV_AS)`。
3. 以 task selection agent 为例：
   - 定义 PDR `p ∈ P`，`P = {BEWT, LWKR, CR, EDD, S/RT}`。定义 PDR p 的生产属性向量 `[SV_{p,g}, g=1..NTP]`（例如 BEWT 或 LWKR 的 `SV_{p,g}` 为 `EWTP_g` 或 `−RTP_g`）。用 Min-Max 归一化得（式 15）：`PV_p = (SV_{p,g} − SV^min_p)/(SV^max_p − SV^min_p)`。
   - 对 `a_TS(t)` 做 Sigmoid（式 16）：`Sigmoid(a_TS(t)) = [ 1/(1 + e^{−w_p}), w_p ∈ a_TS(t) ]`。
   - 计算 `X = Sigmoid(a_TS(t))·PDR_TS`（形状 `R^{1×5} · R^{5×NTP} = R^{1×NTP}`）（式 17）：`x_g = Σ_{p∈P} ( 1/(1+e^{−w_p}) · (SV_{p,g} − SV^min_p)/(SV^max_p − SV^min_p) )`。
   - 半部分 `Y = Norm(Sigmoid(a_TS(t))·PDR_TS) = Norm(X)`（式 18）：`Y = (x_g − x^min_g)/(x^max_g − x^min_g)`。**Y 代表 PDR weighting**，确保 agent 能基于不同状态选择不同任务（覆盖所有可能的解）。
   - 另一半 `Z = w_TS · PV_SEAT`（式 19）。`w_TS` 为调整权重。规则 **SEAT** 表示选 `EATP_g` 最小的任务优先。**Z 代表 PDR adjustment**，确保任务选择逻辑接近规则 SEAT，即倾向于选 `EATP_g` 较小的任务，可减少 AGV 等待工件的时间。
   - 最终决策依据（式 20）：`P_g = (x_g − x^min_g)/(x^max_g − x^min_g) + w_TS · (SV_{p,g} − SV^min_p)/(SV^max_p − SV^min_p)`，`P_g` 为任务池中任务 `TP_g` 的优先级。
   - 原文解释：由于 P 中所有 PDR 都对降低 TTC 有正作用，用 Sigmoid（式 16）把 `a_TS(t)` 变换为含正值的向量，从而使 agent 在好的轨迹方向上探索，最终提升生产效率。
   - 机器选择同理（Algorithm 1 第 8–9 行），使用规则 **SCTM** 做调整；AGV 选择同理（第 15–16 行），使用规则 **SCTA** 做调整。`w_MS`、`w_AS` 为对应的调整权重。规则 SCTM 与 SCTA 使工件能尽快开始加工与运输。
4. 机器选择部分设置 **do-nothing action δ0**（`δ0 = [0, 0, 0]`）以处理"选中任务只是工件的最后一个物流任务、不需要机器分配"的情形。

**奖励函数（Algorithm 2）**——基于工件的估计加权拖期 EWT，并考虑机器空闲时间：

```
1: for job i = 1 to n do
2:    if job J_i has been finished then
3:       Calculate EWT_i = max(0, C_i − D_i) · w_i
4:    else
5:       Calculate EWT_i = max(0, t + RT_i − D_i) · w_i
6:    end if
7: next for
8: Calculate r_t = Σ_{i=1..n} EWT_i(t) − Σ_{i=1..n} EWT_i(t+1)
9: Calculate r_t = r_t − α · ( Σ_{x=1..m} Σ_{k=1..Nx} ITM_xk ) / ( Σ_{x=1..m} N_x )
```

- `RT_i` 为工件 `J_i` 的剩余时间（用式 5 计算）；`EWT_i(t)` 为工件 `J_i` 在时刻 t 的 EWT；`ITM_xk` 为机器 `M_xk` 在 t 与 t+1 之间的空闲时间；`α` 表示机器空闲时间对奖励的影响程度。
- 第 7 行（此处应为原文编号第 8 行）表示鼓励 agent 做出降低每个转移步拖期的决策；第 8 行（原文第 9 行）表示 agent 会在训练中尽量让所有机器保持忙碌。
- 三个 agent 在每个决策点共同行动以完成一个工件的资源分配并优化 TTC，因此 **三个 agent 共享同一奖励**。

**MAPPO 训练算法（Algorithm 3）**：

- 为三个 agent 分别初始化 actor 网络 `π_TS(θ_TS)`、`π_MS(θ_MS)`、`π_AS(θ_AS)`，critic 网络 `v_TS(φ_TS)`、`v_MS(φ_MS)`、`v_AS(φ_AS)`，以及三个记忆缓冲区 `MB_TS`、`MB_MS`、`MB_AS`。
- 每个 episode 用问题 `PROB` 初始化车间环境；直到所有工件返回成品区为止循环：执行 Algorithm 1 分配资源 → 执行三个动作、机器与 AGV 继续运行 → 在执行过程中更新任务池 → 观测新状态 `{SF1,SF2,SF3,SF4}_{t+1}` → 按 Algorithm 2 计算奖励 → 若为终止状态则捕获观测 `{SF5,SF6}_{t+1}` → 存 transition 进对应缓冲区。
- 若发生扰动事件，执行对应处理策略（第 18–20 行）。
- 每 episode 形成轨迹 τ，用 **GAE + PopArt** 计算优势估计 `Â`；计算 reward-to-go `R̂` 并用 **PopArt** 归一化。
- 达到记忆更新容量 `MUC` 后用 Adam 更新 actor 与 critic，然后**重新生成一个新的训练问题 PROB**（第 28 行）——每个 epoch 用随机生成的训练问题以增强泛化。
- **Actor 目标（式 21，最大化）**：
  `L(θ) = (1/B)·Σ_{i=1..B} ( min(r_i(θ)·A_i, clip(r_i(θ), 1−ϵ, 1+ϵ)·A_i) + σ·S[π(o_i;θ)] )`
  其中 B 为 minibatch 大小；θ 为 `{θ_TS, θ_MS, θ_AS}` 之一；`A_i` 为 `Â` 中的优势估计；`ϵ` 为裁剪参数；`σ` 为熵系数；`S` 为策略熵；`o_i` 为三个观测 `{S_TS(t), S_MS(t), S_AS(t)}` 之一。
- **概率比（式 22）**：`r_i(θ) = π(a_i|o_i;θ) / π(a_i|o_i;θ_old)`。
- **Critic 损失（式 23，最小化）**：
  `L(φ) = (1/B)·Σ_{i=1..B} ( max( H(v(s_i;φ) − R̂_i), H(clip(v(s_i;φ), v(s_i;φ_old)−ϵ, v(s_i;φ_old)+ϵ) − R̂_i) ) )`
- **Critic 输入状态（式 24）**：`s_i = concat( {SF1, SF2, SF3, SF4}, N(0, ε²) )`，其中 `N(0, ε²)` 为高斯噪声向量。**每 100 个 episode 随机生成一个新的噪声向量**，以缓解多智能体策略过拟合（overfitting）。
- **Huber 损失（式 25）**：`L_H(x1, x2) = 0.5·(x1−x2)²` 若 `|x1−x2| < 1`；否则 `= |x1−x2| − 0.5`。原文指出它能提升训练稳定性。
- **探索**：训练过程中，actor 网络给出的每个权重从**均值等于输出值、标准差为 1 的高斯分布**中采样，使 agent 获得不同轨迹并从中学习。应用过程中，actor 网络的每个输出值**直接作为 PDR 权重**使用。

**扰动事件处理策略（Section IV-G）**：核心思路是**把扰动事件的发生转化为车间状态的变化**，便于 agent 感知；agent 通过相关实例的训练学会有效处理扰动事件。

1. **New Job Arrivals**：把新工件的第一个任务放入任务池，让 agent 为其分配机器与 AGV。
2. **Machine Breakdowns**：机器故障时，确定相应被扰乱的任务并放入任务池，包括 ① 正被故障机器加工的任务；② 故障机器输入缓冲区中的任务；③ 上述工件的已提前分配的任务。agent 基于车间状态与工件紧急度重新分配这些任务。机器修复完成后，确定相应被扰乱的任务并放入任务池，包括 ① 同类机器输入缓冲区中的任务；② 上述工件的已提前分配的任务（**注意排除正在配送中的任务**），防止修复后的机器长时间空闲。
3. **Job Rework**：返工事件发生时，把返工工序对应的任务放入任务池，agent 重新分配机器与 AGV。
4. **Fuzzy Transportation Time**：DRL agent 通过相关训练学会在模糊运输时间下合理分配制造资源。

**性能对比指标**：优越比例 `SP^X_Y`（式 26）：`SP^X_Y = (TTC_Y − TTC_X)/TTC_X · 100%`，其中 `TTC_X`、`TTC_Y` 分别为方法 X 与方法 Y 的 TTC。`SP^X_Y` 越大，方法 X 相对方法 Y 越好。（注：原文此处表述为 "The larger SP^X_Y, the better method X is compared to method Y"，即 X 为所提方法、Y 为对比方法。）

### 4. 它怎么验证

**实例来源与规模**：

- 用 **Tecnomatix Plant Simulation** 软件基于所定义车间布局建立模型（Fig. 4 给出 2-D 与 3-D 模型），用于测试 AGV 在不同位置间的运输时间，并测量考虑路径占用（path congestion）、AGV 加速、AGV 减速等因素下时间波动的概率与幅度。
- 生产配置：
  1. 机器类型数 `m = 8`，`N_x = [2, 3, 2, 3, 2, 3, 3, 2]`（即共 18 台机器）
  2. AGV 数 `l = 10`
  3. 初始工件数 `20`
  4. 工件工序数 `randi[4, 10]`
  5. 每道工序加工时间 `randi[10, 30]`（单位：min）
  6. 每个配送任务的运输时间范围 `[17, 204]`（单位：second）
  7. 工件权重 `w_i ∈ {1, 2, 4}`，对应比例 20%、60%、20%
  8. 交期松紧度 `DDT_i = randf[1, DDT_H]`，`DDT_H ∈ {1.5, 2}`
- 扰动事件设置：
  1. 车间利用率 `U ∈ {0.9, 0.95}`
  2. 新工件到达数 `J_new = randi[100, 200]`
  3. 平均修复时间 `MTTR = 3 · TPT`
  4. 车间故障水平 `Ag ∈ {0.025, 0.05}`
  5. 工件返工概率 `JRP = {0.03, 0.06}`
  6. 通过所建 Plant Simulation 模型（考虑路径占用、AGV 加减速）发现：AGV 执行任务时运输时间波动的概率约为 **10%**，波动幅度为 **[0, 10%]**
  - 符号约定："randi" 与 "randf" 分别表示整数与实数的均匀分布。

**对比基线（列全）**：

1. **10 条复合派工规则（composite PDRs）** `R_i, i = 1..10` —— 由常用的工件排序 PDR、机器选择 PDR、AGV 选择 PDR 组合执行而形成（从候选中挑出表现最好的 10 条）。
2. **Genetic Programming (GP)** [21]
3. **四个流行 DRL 方法**：
   - Luo et al. [4]
   - Park and Park [27]
   - Li et al. [21]
   - Lei et al. [26]
   - 注：因考虑四类扰动事件，**所有方法都被重新训练**。
4. **消融对照**：
   - 动作解码四策略消融：分别去掉 Sigmoid operation、TS adjustment、MS adjustment、AS adjustment 之一。
   - 奖励函数消融：X = 所提奖励（Algorithm 2 第 8 行含机器空闲时间）；Y = 常用的 EWT 差值奖励（仅 Algorithm 2 第 7 行）。

**评估指标**：

- `SP^X_Y`（式 26）优越比例（%），以 mean/standard deviation 形式报告
- 总拖期成本 TTC（收敛曲线：每点为一组 5 个实例上的均值，配 90% 置信区间阴影）
- 箱线图（Fig. 7、Fig. 8）、热力图（Fig. 9）

**实验环境（硬件/软件）**：PyTorch 实现；服务器 Intel Xeon W-3365 CPU @ 2.70 GHz，125 GB RAM；超参数用 **random search** [37] 搜索最优。部分内容（运输时间矩阵、超参数调优、对比方法细节、部分消融/对比结果、实时性测试、工厂仿真模型）放在 Supplementary Material 的 Section D–I。

**验证协议**：

- 设 3 个问题设置（不同参数），每个问题设置含 **5 个随机生成的实例**；用训练过程中保存的 DNN 在这些实例上评估收敛性与泛化性；从收敛阶段挑选表现最好的 DNN 用于后续对比实验。
- 动作解码消融：随机生成 **10 个 MAPPO 模型与 20 个实例**。
- 奖励函数消融：分别基于 X 与 Y 训练两次，在 **30 个随机生成的实例**上测试。
- 与复合 PDR 对比：基于三参数 `(J_new, DDT_H, U)` 生成 **12 组实例，每组含 5 个实例**。
- 鲁棒性对比：Fig. 7(a) 六组参数 `(Ag/J_new)`；Fig. 7(b) 六组参数 `(JRP/J_new)`；Fig. 7(c) 三组参数 `(J_new)`；**每组含 10 个随机生成的实例**。
- 与 GP / DRL 对比：**60 个实例**（考虑新工件到达）；扰动下对比共 **150 个实例**（各方法均重新训练）。
- 模糊运输时间下，因同一实例上方法可能得到不同解，故一个方法在一个实例上**运行 5 次取均值**。
- 行为分析：在 3 个实例（100 / 150 / 200 个工件）上测试训练好的 DNN，收集 agent 输出的 PDR 权重向量做热力图。

### 5. 核心结果（照抄原文数据）

**动作解码四策略消融（Table III）**：以 mean/standard deviation 表示 10 个 MAPPO 模型在 20 个实例上的 `SP^X_Y`。原文结论："It can be observed that each strategy has brought performance improvements to the proposed method."（可以观察到每个策略都为所提方法带来了性能提升。）原因：① Sigmoid 操作能有效整合各种面向拖期的 PDR；② 任务池中有些任务是提前放入的、其工件可能正在机器上加工，若选中这些任务会导致 AGV 长时间等待，规则 SEAT 可避免该情况；③ 规则 SCTM 能平衡各机器负载并尽量让每台机器长时间运行；④ 规则 SCTA 能使每个物流任务尽快执行。

**奖励函数消融（Fig. 6）**："The proposed method achieves better results in 83.3% of instances."（所提方法在 **83.3% 的实例**上取得更好结果。）箱线图显示 30 个 `SP^X_Y` 的**平均值为 5.66%**。原文结论：加入机器空闲时间后，所提奖励能更好地引导 agent 的训练与学习。

**与复合 PDR 比较（Table IV，考虑新工件到达，12 组 × 5 实例）**："The proposed method can obtain better solutions than composite PDR on different instances. **The performance improvement is at least 14.58%, with most exceeding 20%.**"（所提方法在不同实例上都能取得比复合 PDR 更好的解。性能提升**至少 14.58%，多数超过 20%**。）

**鲁棒性（Fig. 7，四类扰动）**：

- Fig. 7(a)（新工件到达 + 机器故障）：箱线图中所提方法**仅在一个实例上低于 R8，为 −0.54%**。
- Fig. 7(b)（新工件到达 + 工件返工）：所提方法**在所有实例上都能取得更好结果**。
- Fig. 7(c)（新工件到达 + 模糊运输时间）：每个柱代表 10 个 `SP^X_Y` 的均值，柱上方三个数字分别为均值、标准差、中位数。
- 原文总结："In summary, the proposed method can obtain better schemes on instances under various disturbance events, and the performance improvement is mostly more than 20%."

**与 GP 及四个 DRL 方法比较（Table V，60 个实例，考虑新工件到达）**：原文："The superior proportion mostly exceeds 5%."（优越比例多数超过 5%。）

**扰动下的对比（Fig. 8，150 个实例，各方法均重新训练）**：

- 在三个子图中，所提方法**仅在少数实例上比其他方法差，最差幅度接近 −10%**；在多数实例上能取得更好解。
- 对 GP 规则、Luo、Park、Li 的**性能提升多数超过 10%**；对 **Lei 的提升多数超过 5%**。

**行为分析（Fig. 9，100/150/200 工件三个实例）**：机器选择 agent **更偏向于基于 SCTM 决策**，因为规则 SCTM 能有效平衡所有机器负载并使每个工件尽快开始加工；工件选择 agent 与 AGV 选择 agent 在 PDR 调整下能在不同时刻偏好不同规则，从而实现合理决策。

**对基线方法表现不佳的原因分析（原文第 V-F 节）**：

1. 单一 PDR 只适合解决部分调度实例，而且解质量不令人满意。
2. GP 虽通过进化克服了 PDR 的缺点，但它仍是一种**固定的决策逻辑**，无法与调度过程中不断变化的车间状态形成映射。
3. 由于基于规则的动作空间，Luo 方法或 Li 方法在每个决策点只能选择属性最大/最小的工件，因此**错过许多好解**，训练效果差。
4. 尽管端到端或特殊动作设计能探索所有可能的解，但 Park 方法或 Lei 方法**容易在 DFJSP-AGVs 的巨大解空间中迷失**。

**摘要中的总括数字**（Abstract 原文）："Comparison experiments show that the proposed method outperforms the priority dispatching rules, genetic programming and four popular reinforcement learning (RL)-based methods, with performance improvements mostly exceeding 10%."

### 6. 它自己承认的局限（原文逐字引用 + 中文翻译）

> **原文**："However, compared to other methods, the proposed method cannot achieve the best solution on all instances. The superiority and stability of algorithm deserve further exploration."
>
> **中文**：然而，与其他方法相比，所提方法并不能在所有实例上都取得最优解。算法的优越性与稳定性值得进一步探索。

> **原文**："Therefore, future research directions are as follows: 1) integrate more domain knowledge into the action space decoding; 2) use state-of-the-art deep learning models; and 3) design better reward function based on domain knowledge or inverse reinforcement learning."
>
> **中文**：因此，未来的研究方向如下：1) 将更多领域知识融入动作空间解码；2) 使用最先进的深度学习模型；3) 基于领域知识或逆强化学习设计更好的奖励函数。

（另注：原文在鲁棒性实验处自陈在少数实例上不如基线——"the proposed method only achieves poor solutions on a few instances compared to other methods, with the worst amplitude being near –10%."；以及在奖励消融处自陈只在 83.3% 实例上更好。）

### 7. 有什么可以拿来用

- **开源代码**：无。文中未提供代码仓库或数据可用性声明；补充材料（含运输时间矩阵、超参数调优、对比方法细节、实时性测试、工厂仿真模型）以 Supplementary Material 形式随论文提供（DOI 10.1109/TSMC.2024.3520381）。
- **公开数据集**：无公开数据集；实例由作者自建（基于 Tecnomatix Plant Simulation 生成的车间布局与运输时间）。
- **可借鉴的评测协议**：
  - **`SP^X_Y` 优越比例指标（式 26）**：`SP^X_Y = (TTC_Y − TTC_X)/TTC_X · 100%`，并以 mean/standard deviation 形式在"多模型 × 多实例"网格上报告（例如 10 个 MAPPO 模型 × 20 个实例）。这是把"随机训练波动"纳入统计的一个很实用的做法。
  - **`SP_{−EDD}` 变体**（本文姊妹篇 P4 的式 26 同一思路）：以某条规则为基线计算优越比例。
  - **扰动鲁棒性的分组箱线图协议**：按参数组合 `(Ag/J_new)`、`(JRP/J_new)`、`(J_new)` 分组，每组 10 个实例，箱线图 + 均值/标准差/中位数三重数字。
  - **所有方法在引入新扰动时全部重新训练**——避免"只重训自己、基线用旧模型"的不公平比较。
  - **每 epoch 随机生成训练问题 / 实例**以增强泛化（本文与 P4 均采用）。
  - **模糊运输时间下运行 5 次取均值**——处理随机性运输时间的合理做法。
  - **行为分析用热力图**展示 agent 输出的规则权重随时间/实例的偏移（Fig. 9），可直接复用为"策略可解释性"证据。
- **有价值的公式**：
  - **动作解码（式 17–20）——最有借鉴价值**：`P_g = Norm( Sigmoid(a(t)) · PDR ) + w · PV_adjust`。即 **最终优先级 = 规则加权组合（覆盖全部解空间）+ 单一规则调整项（引导方向性探索）**。这套 "PDR weighting + PDR adjustment" 同时解决了"基于规则的动作空间错过好解"与"端到端动作空间在巨大解空间迷失"两个问题，是介于两者之间的一个具体可抄的折中设计。
  - **含物流因素的 MTBA（式 2）**：`MTBA = (N_ope·TPT)/(U·ΣN_x) + (N_ope+1)·TLT/(U·l)`——新工件到达率的建模方式，把加工产能与物流产能并列。
  - **含物流因素的交期（式 3）**：`D_i = A_i + DDT_i·Σ_j(Σ_k TPT_ijxk/N_x) + DDT_i·(N_i+1)·TLT`——Total Work Content 法的物流扩展版。
  - **车间故障水平（式 4）**：`Ag = MTTR/(MTBF + MTTR)`——单参数刻画车间故障严重度的简洁写法。
  - **奖励函数（Algorithm 2）**：`r_t = [Σ EWT_i(t) − Σ EWT_i(t+1)] − α·(Σ_x Σ_k ITM_xk)/(Σ_x N_x)`，即**拖期改善量减去归一化机器空闲时间**。EWT 用 `max(0, t + RT_i − D_i)·w_i` 对未完工工件做在线估计，使奖励在每一步都可计算——这是"以拖期为目标的稠密奖励"的标准写法。
  - **多智能体过拟合对策（式 24）**：critic 输入 `s_i = concat(状态, N(0, ε²))`，**每 100 episodes 重新生成噪声向量**。
  - **任务释放时刻（task release moment）**："当工件只剩一个已被分配 AGV 但尚未完成的任务时，释放其下一个任务"——既减少工件等 AGV 时间又避免过早决策，可直接迁移。
  - **决策点定义**：任务池非空 且 存在一台或多台空闲 AGV。
  - **do-nothing action δ0 = [0,0,0]**：处理"选中任务只有物流任务、无需机器分配"的情形。

---

## 卡片 3：TASE2026 — CP for AGV & Machine Integrated Scheduling

### 1. 题录

- **标题**：Constraint Programming for AGV and Machine Integrated Scheduling Problem in Flexible Manufacturing System
- **作者**：Youjie Yao, Qihao Liu, Xinyu Li (通讯, Member IEEE), Liang Gao (Senior Member IEEE)
  - 全部作者：State Key Laboratory of Intelligent Manufacturing Equipment and Technology, Huazhong University of Science and Technology, Wuhan 430074, China
- **期刊**：IEEE Transactions on Automation Science and Engineering, Vol. 23, 2026, pp. 2378–2390
- **年**：2026（Received 27 January 2025; Revised 27 May 2025 and 20 October 2025; Accepted 26 December 2025; Date of publication 12 January 2026; Date of current version 22 January 2026）
- **DOI**：10.1109/TASE.2025.3650678
- **Supplementary**：有可下载补充材料，见 https://doi.org/10.1109/TASE.2025.3650678
- **Index Terms**：Flexible manufacturing system, constraint programming, integrated scheduling, AGV and machine scheduling
- **资助**：NSFC 52188102；教育部基础与交叉学科突破计划 JYB2025XDXM208；湖北省战略科技人才培养计划 2024DJA033；中国博士后科学基金 2025M781288
- **Associate Editor**：C.-B. Yan；**Editor**：J. Li

### 2. 它解决什么问题

- **问题类型**：**柔性制造系统中的 AGV 与机器集成调度问题（AMISP-FMS, AGV and Machine Integrated Scheduling Problem in Flexible Manufacturing Systems）**。可视为 FJSP 的扩展，引入车间运输资源后成为 AGV 与机器的**双资源调度问题**。论文将该问题分解为**四个相互关联的决策子问题**，必须同时求解：
  1. **Machine Assignment for Processing Tasks（加工任务的机器分配）**——每道加工任务须从能力各异的可用机器集合中选一台。机器选择影响加工时间，并影响其后运输任务的路线与时长。
  2. **Processing Task Sequencing（加工任务排序）**——确定每台机器上加工任务的顺序，直接影响各加工任务的开始与完成时间。
  3. **AGV Assignment for Transportation Tasks（运输任务的 AGV 分配）**——给定有限 AGV 数，每个运输任务（在机器间或往返仓库移动工件）须分配给一台可用 AGV。
  4. **Transportation Task Sequencing（运输任务排序）**——分配 AGV 后，须对运输任务排序以最小化空载行驶时间、避免不必要延迟。
  - 四个子问题紧密关联：工件的加工任务只能在对应运输任务完成后开始；运输任务只能在前道工序的加工任务完成后开始；机器分配决定运输任务的存在性与载货行程，因为加工任务的机器选择决定了运输任务的起点与终点。
- **涉及资源**：`m` 台加工能力各异的机器 + `K` 台 AGV（负责机器与仓库之间的在制品转运）+ 一个仓库（warehouse）。原文指出 AMISP-FMS 由 Deroussi 和 Norre [16] 首次提出，并伴随 **FJSPT 基准数据集**（含 10 个实例）。
- **优化目标**：**最小化 makespan**（定义为所有工件在机器上的最大完成时间）：
  - `MinMax{ endOf(ProTask_i) }, i ∈ Last`（式 2）——即最后一道工序加工任务的最晚结束时间。
- **本文针对的具体缺陷（核心问题）**：既有 CP 模型（Ham [15] 的 CP1 与 CP2）**忽略了一个 AMISP-FMS 的基本特征**——当同一工件的相邻两道工序被分配到同一台机器时，这两道工序之间**不需要运输**。但 CP1 和 CP2 仍把这些不存在的运输任务纳入 AGV 调度过程，引入了不必要的空载行程，导致**错误的结果（incorrect optimal solutions）**。
  - 实例（Fig. 2）：EX21 实例中，工件 2 的相邻两道工序 O21 与 O22 都分配到机器 M2。由于纳入了 O22 的不存在运输任务，其加工开始时间被不必要地从 54 延迟到 56。实际上 O21 与 O22 之间无需运输，这一延迟不应发生。
- **运输任务存在性判据（原文关键结论）**："a transportation task does not exist if and only if adjacent operations of the same job are assigned to the same machine; otherwise, a transportation task must exist whenever the two operations' processing tasks are assigned to different machines."
  中文：**当且仅当同一工件的相邻工序被分配到同一台机器时，运输任务不存在**；否则，只要两道工序的加工任务被分配到不同机器，运输任务就必须存在。
- **约束与假设**（原文逐条）：
  - All jobs and AGVs start at the warehouse.（所有工件与 AGV 从仓库出发）
  - Only one task can be executed at any given moment by either the machine or the AGV, and preemption is not allowed.（任一时刻机器或 AGV 只能执行一个任务，不允许抢占）
  - The travel time of AGVs in the workshop depends solely on the origin and destination of their tasks, and constraints such as acceleration and deceleration, collision avoidance, and power are not considered during transport.（AGV 在车间内的行驶时间仅取决于任务的起点与终点；运输过程中不考虑加减速、避碰、电量等约束）
  - Each AGV can carry only one job at a time, and the buffer capacity of the machines, as well as the loading and unloading time of jobs, are negligible.（每台 AGV 一次只能运载一个工件；机器缓冲区容量与工件装卸时间可忽略）

### 3. 它怎么做

**方法全貌**：**精确方法（exact method）——约束规划（Constraint Programming, CP）**。不是 MARL、不是元启发式。论文提出**两个新的 CP 模型 NCP1 与 NCP2**，两者共享相同的目标函数（式 2），区别在建模方式。此外，论文在构建 CP 模型之前先建立了对应的 **MILP 模型**（基于四个决策子问题及其关系），其详细约束方程在补充材料 Section A。

- 原文定位：与 MILP 不同，CP 模型通过在解空间中系统搜索满足所有约束的解来获得最优解；且 CP 在处理复杂离散决策问题与非线性约束时往往更高效。

**两种建模思路（Section IV-A）**：

1. **思路一（→ NCP1）：利用 CP 中的条件约束（conditional constraints）**。当同一工件的相邻工序被分配到同一台机器时，显式约束这些工序的运输任务在**同一台 AGV 上连续执行**。这确保两个运输任务之间的空载时间为 0，从而避免造成延迟的不必要空载行程。
2. **思路二（→ NCP2）：利用问题分析中识别出的运输任务存在性，把运输任务变量设为可选（optional）**。在工件的运输任务与机器分配之间建立相应约束，从而避免把不必要的复杂度引入求解过程。若相邻工序分配到同一台机器，运输任务被**完全省略**。这减少了模型中的变量与约束数量，简化问题并提升求解效率。

**CP 约束函数（Section IV-B，原文列举）**：

- `endBeforeStart(Ta, Tb)`：时间约束函数。确保任务 `Ta` 必须在任务 `Tb` 开始前完成。
- `endOf(Ta)`：返回任务 `Ta` 的完成时间。
- `presenceOf(Ta)`：返回布尔值，指示任务 `Ta` 是否存在。
- `alternative(Ta, Sa)`：替代函数。给定一组替代任务 `Sa`，表示任务 `Ta` 可在一组替代资源上执行，但只能选其中之一。
- `noOverlap(Seq, Tt)`：排序函数。确保序列 `Seq` 中所有任务在执行时不能重叠，`Tt` 为可选参数，表示任务之间的转移时间（transfer time）。
- `prev(Seq, Ta, Tb)`：相邻函数。若任务 `Ta` 与 `Tb` 存在，则任务 `Ta` 正好在任务序列 `Seq` 中紧接于 `Tb` 之前，即 `Ta` 与 `Tb` 相邻。
- `IfThen(Cond, Cons)`：条件函数。当条件 `Cond` 满足时，约束 `Cons` 成立。

**NCP1 的索引与变量**：

- 索引：`i, i'` = 工序索引，`i, i' ∈ I`（`I` 为工序总数）；`j, j'` = 工位（station）索引，`j, j' = 0, 1, …, m`，`0` 代表仓库，`m` 为机器总数；`k, k'` = AGV 索引，`k, k' ∈ K`。
- 集合：`First` = 每个工件的第一道工序集合；`Last` = 每个工件的最后一道工序集合；`MS_i` = 工序 `O_i` 加工任务的可用机器集合。
- 参数：`Tt` = 转移参数，运输任务之间的转移时间（空载时间）。
- 变量（interval variables 区间变量）：`ProTask_i`（工序 `O_i` 的加工任务）；`ProInMac_ij`（工序 `O_i` 的加工任务在机器 `j` 上执行）；`TransTask_i`（工序 `O_i` 的运输任务）；`TransTaskInVeh_ijj'k`（工序 `O_i` 的运输任务从工位 `j` 到工位 `j'`、由 AGV `k` 执行）；`TaskInMacSeq_j`（机器 `j` 上的加工任务序列变量）；`TaskInVehSeq_k`（AGV `k` 上的运输任务序列变量）。

**NCP1 的关键约束（式 1 与式 3–18）**：

- 式 (1)（NCP1 特有的附加约束，Fig. 5 图解）：`x^k_{kl}·x_{k(l−1)} − 1 ≥ M·(y^m_{kl} + y^m_{k(l−1)} − 2)`，其中 `x` 表示运输任务的相邻决策变量。该约束**仅在相邻两道工序被分配到同一台机器时激活**；它在图解中把相邻工序对应的两个运输任务强行捆绑为**由同一台 AGV 连续执行**的任务。对第二道工序，由于在同一台机器上加工，其对应运输任务成为零距离、零载时行程（起点终点同位置）。
- 式 (3)：`alternative(ProTask_i, [ProInMac_ij]), j ∈ MS_i`——加工任务的可用性。
- 式 (4)：`alternative(TransTask_i, [TransTaskInVeh_ijj'k])`，`i ∈ First: j = 0, j' ∈ MS_i; i < First: j ∈ MS_{i−1}, j' ∈ MS_i`——运输任务的可用性，其起点与终点由机器分配决定（第一道工序的起点为仓库）。
- 式 (5)：`endBeforeStart(TransTask_i, ProTask_i)`——运输任务与加工任务的相互依赖关系，加工任务只能在运输任务完成后开始。
- 式 (6)：`endBeforeStart(ProTask_{i−1}, ProTask_i), i < First`——同一工件的工序先后约束。
- 式 (7)：`presenceOf(TransTaskInVeh_{i0jk}) ≤ presenceOf(ProInMac_ij), i ∈ First, j ∈ MS_{i−1}`。
- 式 (8)：`presenceOf(TransTaskInVeh_{ijj'k}) ≤ presenceOf(ProInMac_{(i−1)j}) * presenceOf(ProInMac_{ij'}), i < First, j ∈ MS_{i−1}, j' ∈ MS_i`——运输任务的存在性与机器分配相关。第一道工序只固定起点；后续工序起点与终点都须确定。
- 式 (9)：`Σ_j Σ_{j'} Σ_k presenceOf(TransTaskInVeh_{ijj'k}) = 1, i ∈ I`——任一工序的运输任务必须存在。
- 式 (10)–(13)：当同一工件的相邻两道工序被分配到同一台机器时，对应运输任务必须由同一台 AGV 且紧接执行。式 (10) `prev(TaskInVehSeq_k, TransTaskInVeh_{(i−1)0jk}, TransTaskInVeh_{ijj'k}), j ≠ j', k ∈ K`；式 (11) `IfThen( presenceOf(ProInMac_{(i−1)j}) & presenceOf(ProInMac_{ij'}) , presenceOf(TransTaskInVeh_{ijj'k}) ≥ presenceOf(TransTaskInVeh_{(i−1)0jk}) )`——分别处理 AGV 选择（10、12）与任务顺序相邻（11、13）。
- 式 (14)：`endBeforeStart(ProTask_{i−1}, TransTaskInVeh_{ijj'k})`——当同一工件的相邻工序选择不同机器时，运输任务应只在前道工序的加工任务完成后开始。
- 式 (15)：`TaskInMacSeq_j = [ProInMac_ij], i ∈ I, j ∈ m`——定义每台机器上可能的任务集合。
- 式 (16)：`noOverlap(TaskInMacSeq_j), j ∈ m`——机器上加工任务的析取（disjunctive）约束，确保任一时刻只能执行一个任务。
- 式 (17)：`TaskInVehSeq_k = [TransTaskInVeh_{ijj'k}], k ∈ K`——由于 AGV 同质（homogeneous），AGV 的任务序列包含所有工序对应的运输任务。
- 式 (18)：`noOverlap(TaskInVehSeq_k, Tt), k ∈ K`——AGV 上运输任务的析取约束。原文说明：工序任务之间排序会产生空载行驶时间，由转移时间 `Tt` 表示。

**NCP2 的约束（按四个决策子问题组织，式 19–30）**：

1. **Machine Assignment for Processing Tasks**：式 (19) `alternative(ProTask_i, [ProInMac_ij]), j ∈ MS_i`——每个加工任务须分配到至少一台可用机器，不同机器上加工时间不同。
2. **Processing Task Sequencing Problem**：式 (20) `TaskInMacSeq_j = [ProInMac_ij], i ∈ I, j ∈ m`；式 (21) `noOverlap(TaskInMacSeq_j), j ∈ m`——同一机器上的不同加工任务**不需要转移时间**，可直接用 `noOverlap` 排序。
3. **AGV Assignment for Transportation Tasks**：分配运输任务前须根据加工任务分配确定其是否存在。由于工件的初始位置不在机器上，每个工件第一道工序的运输任务**必然存在**，其起点确定（仓库用 0 表示）。
   - 式 (22)：`presenceOf(TransTask_i) = 1, i ∈ First`——第一道工序的运输任务存在。
   - 式 (23)：`presenceOf(TransTaskInVeh_{i0jk}) ≤ presenceOf(ProInMac_ij), i ∈ First, j ∈ MS_{i−1}`——机器分配决定终点。
   - 式 (24)（**NCP2 的核心**）：`presenceOf(TransTaskInVeh_{ijj'k}) = 1 − ( presenceOf(ProInMac_{(i−1)j}) * presenceOf(ProInMac_{ij'}) ), i < First, j ∈ MS_{i−1}, j' ∈ MS_i`——其他运输任务的存在性取决于机器分配，**当相邻工序分配到同一机器时运输任务被完全省略**。
   - 式 (25)：`presenceOf(TransTask_i) = Σ_j Σ_{j'} Σ_k presenceOf(TransTaskInVeh_{ijj'k}), i < I`——只有当运输任务存在时才为任务做 AGV 分配。
4. **Transportation Task Sequencing Problem**：对存在的运输任务排序时，须考虑运输任务之间的空载行程（取件），空载行程的时间约束可通过在函数中设置转移时间矩阵实现。
   - 式 (26)：`alternative(TransTask_i, [TransTaskInVeh_{ijj'k}]), i ∈ First: j = 0, j' ∈ MS_i; i < First: j ∈ MS_{i−1}, j' ∈ MS_i`
   - 式 (27)：`noOverlap(TaskInVehSeq_k, Tt), k ∈ K`
5. **Relationship Between Transportation Task and Processing Task**：
   - 式 (28)：`endBeforeStart(TransTask_i, ProTask_i)`——基于工件初始位置，同一工序的运输任务必须先于加工任务。
   - 式 (29)：`endBeforeStart(TransTask_{i−1}, ProTask_i), i < First`——下一道工序的运输任务只能在本道工序的加工任务完成后开始。
   - 式 (30)：`endBeforeStart(ProTask_{i−1}, ProTask_i), i < First`——若前道工序的运输任务不存在，则工件必须满足工序之间的约束。

**NCP1 与 NCP2 的主要区别**：仅在于建模方式不同，两者共享相同目标函数（式 2）。

### 4. 它怎么验证

**实例来源（三个基准）**：

1. **FJSPT benchmark**——文献中最常用的基准（Deroussi 和 Norre [16] 提出，含 10 个实例）。但因其对可选机器的约束，FJSPT **不涵盖"同一工件相邻工序分配到同一台机器"的场景**。其 10 个实例的最优性已被 MILP 与 CP 方法证明 [14][15]。
2. **EX benchmark**——Kumar et al. [18] 从 job shop scheduling 扩展到 flexible job shop scheduling 而建立。含 **7 个工件集（job sets）与 4 个布局（layouts）**，每个实例由一个工件集与一个布局构成（如 EX74 = 工件集 7 + 布局 4）。EX 实例每道工序有 **3 台可用机器**，因柔性更高而更复杂。EX 还可扩展：把工件集的加工时间翻倍、布局的运输时间减半，得到新实例记为 **EX740**；加工时间三倍、运输时间减半记为 **EX741**。文中"57 个 EX 实例（两组）"即来自该扩展。
   - 注：现有 MILP 模型求解每个 EX 实例最长需 **36,000 秒**（Ham [15] 报告），仍有 **13 个实例的最优性未被证明**。
3. **La benchmark**——著名的 FJSP 基准，用于评估所提 CP 模型在**大规模问题**上的表现。为适用于所考虑的 AMISP-FMS，采用了 **Homayouni et al. [22] 提出的运输调度（transportation schedule）**。La 实例中**运输任务总是必需的**。

**对比基线（列全）**：

1. **LAHC** [13]——late acceptance hill climbing，元启发式（Homayouni and Fontes）
2. **CP2** [15]——Ham 的第二个 CP 模型（把 pickup 与 delivery 合并为单个任务）；**本文用相同编码语言与实验环境复现了该模型，计算时间限制为 3600 秒**（因原论文未求解 EX 基准）
3. **CP1** [15]——Ham 的第一个 CP 模型（把运输任务拆为 pickup 与 delivery 两个独立任务）——仅在文献综述中讨论，未列入实验对比表
4. **MILP** [14]——Yao et al. 精炼的 MILP 模型（当前最有效的数学规划模型）
5. **DCGA** [24]——Han et al. 的双种群协同遗传算法（最佳表现的元启发式算法，优于 [13] 的 LAHC）；**因原文献未报告运行时间数据，DCGA 的计算时间未纳入比较**
6. **NCP1 / NCP2**——本文提出的两个模型

**评估指标**：

- **Best**：各方法找到的最好解
- **计算时间（computation time）**：秒
- **Optimality gap**：CPLEX 给出的最优性间隙。**gap = 0% 表示解可证最优**（可视为该实例的最优性已确立 [33]）；正的 gap 表示找到的是最优可行解，但其全局最优性尚未被正式证明。
- **NbVar / NbCon**：变量数与约束数（用于比较两种建模方式）

**实验环境（硬件/软件）**：两个 CP 模型用 **C++ 实现**，用 **CPLEX 12.10** 求解；计算机配置 **i9-13900HX @ 2.2 GHz 处理器、40 GB RAM**。所有实例均用 CPLEX 求解。

**案例研究**：使用来自**航空航天领域某真实机加工车间**的数据。该 FMS 包含 5 种加工设备，各有不同能力；每个工件的加工流程涉及多台机器，加工时间与机器选择各异；此外使用 **2 台 AGV** 在加工中心之间运输工件。Case 1 为工件 1–5，Case 2 为全部 10 个工件。详细加工信息与运输时间见补充材料 Section E。La 基准在 K = 5 时的结果见补充材料 Section D 的 Table S-II；t/p ≤ 0.25 时的结果见 Table S-I。

### 5. 核心结果（照抄原文数据）

**NCP1 vs NCP2 的变量与约束数（Table II，EX 基准）**：两模型的**决策变量数无差异**（因为只在约束处理上不同），但 **NCP2 在所有工件集上都减少了约束数**。特别是 EX 基准的工件集 4（EX JS4），**约束减少幅度达 20.71%**；**平均约束数减少约 13.87%**。原文结论：NCP2 在约束规模优化上有显著潜力，能有效减少求解过程中需要处理的约束数量。

**FJSPT 基准（Table III）**：CP2、MILP、NCP1、NCP2 **在所有实例上找到相同的最好解**；LAHC 在部分实例上找到略差的解。原文："The proposed NCP1 and NCP2 models perform optimally in terms of solution efficiency, with significantly lower computation times than the other methods, especially on difficult instances like FJSPT7."（所提 NCP1 与 NCP2 在求解效率上表现最优，计算时间显著低于其他方法，在 FJSPT7 等困难实例上尤为明显。）表格中 J–O–M–A 分别表示每个实例的工件数、工序数、机器数与 AGV 数。注：**NCP2 的平均计算时间略高于 NCP1，但仍显著低于 CP2 与 MILP。**

**EX 基准（Table IV，t/p > 0.25）**：

- CP2 **未能考虑工件无需运输的场景，导致解劣于其他方法**。
- MILP、LAHC、DCGA、NCP1、NCP2 在**多数实例上给出相同的最好解**。
- **NCP1 与 NCP2 更新了 EX72 与 EX84 实例的最好已知解（best-known solutions）**，细节见补充材料 Section C。
- 两模型得到相同最优解，但求解效率差异显著：**NCP2 的平均计算时间最短，为 96.57 秒**，大幅优于 NCP1。（原文："NCP2 achieved the shortest average computation time of 96.57 seconds, outperforming NCP1 by a considerable margin."）
- 当两个模型都给出最好解值时，**gap 值为 0%**，确认所得解对该实例确实最优。
- **关键结论**："Importantly, for all 57 EX instances across the two groups of EX instances, both NCP1 and NCP2 achieved a gap of 0%, confirming that the solutions obtained are provably optimal. Compared with the existing exact models, CP2 and MILP, the proposed models are the first to prove the optimality for all EX instances."
  中文：重要的是，对于两组 EX 实例中的全部 **57 个 EX 实例**，NCP1 与 NCP2 都取得了 **0% 的 gap**，确认所得解可证最优。与现有精确模型 CP2 与 MILP 相比，所提模型**首次证明了所有 EX 实例的最优性**。
- 在 t/p ≤ 0.25 与 t/p > 0.25 两种测试条件下，**NCP2 始终在计算时间上优于 NCP1，且结果更稳定、波动更小**（Fig. 7）。
- Fig. 6 给出两模型在 EX 基准上的计算时间箱线图。

**La 基准（Table V，K = 3）**：

- **LA13 是所有实例中最难求解的**。现有模型 **CP2 与 NCP2 在 3600 秒运行时间内未能为 LA13 找到最优解**；相比之下，**所提模型 NCP1 仅在 1746.75 秒内就获得了最优解**。
- 当所有 CP 模型都求得全部实例的最优解时（K = 5），**NCP1 与 NCP2 的求解效率显著高于现有模型 CP2**。
- NCP1 与 NCP2 相比，求解时间几乎相同，NCP1 略少于 NCP2。
- 原文结论：即使在一般情形下，所提模型不仅给出正确的最优解，还在求解效率上展现显著优势。

**案例研究（Table VI）**：

- **Case 1（工件 1–5）**：现有模型给出的**最优解为 1870，是错误的（incorrect optimal solution）**；**所提模型正确识别出最优解为 1860**。
- **Case 2（全部 10 个工件）**：各模型都成功识别出最优解，且随着工件数增加，**瓶颈资源发生转移（the bottleneck resource shifting due to the increase in the number of jobs）**。
- 计算效率上，**NCP2 表现出最高的计算效率**。

**讨论（Section V-G）的核心论断**：

- 建模需遵循两条原则：① 模型必须涵盖所有必要的调度约束，确保解空间的完整性与准确性；② 模型必须精炼简洁，在不损害解质量的前提下提升计算效率。
- 关于原则 ①：现有 CP 模型未考虑 AMISP-FMS 中的一个关键情形，导致产生不必要的运输任务并得到错误的最优解。所提模型通过确保**运输任务仅在必要时才生成**来纠正该问题，从而在所有场景下持续给出正确的最优解。Table IV 显示所提模型获得的最优解与现有上界解一致，验证了模型的正确性。
- 关于原则 ②：约束被设计得既准确又简洁，显著提升计算效率。
- 最终论断："Notably, it improves the best-known solutions for EX72 and EX84, and is the first exact model to prove the optimality of all EX instances."

**摘要中的总括论断**："The results show that the proposed model outperforms existing approaches in both solution quality and efficiency. Notably, the proposed model updates the best-known solutions for the EX72 and EX84 instances, and proves the optimality of all EX instances for the first time."

### 6. 它自己承认的局限（原文逐字引用 + 中文翻译）

> **原文**："Despite these advancements, the model remains computationally intensive for certain challenging instances. Future research will therefore investigate hybrid approaches combining CP with meta-heuristic algorithms to further enhance search efficiency."
>
> **中文**：尽管取得了这些进展，对于某些具有挑战性的实例，该模型的计算仍然十分密集（computationally intensive）。因此，未来研究将探讨将 CP 与元启发式算法相结合的混合方法，以进一步提升搜索效率。

（另注：论文对既有工作提出了明确批评，可作为理解其定位的补充——针对 Ham [15] 的 CP 模型："However, these models fail to consider scenarios in which certain operations of the job do not require transportation, resulting in incorrect optimal solutions."（然而，这些模型未能考虑工件的某些工序不需要运输的场景，导致产生错误的最优解。）；以及 "both CP models introduced by Ham [15] contain critical flaws that can result in incorrect optimal solutions in certain scenarios."（Ham [15] 提出的两个 CP 模型都含有严重缺陷，在某些场景下会导致错误的最优解。）；"Nevertheless, both models overlook a fundamental feature of AMISP-FMS: when two adjacent operations of the same job are assigned to the same machine, no transportation is required between those operations."（然而，两个模型都忽视了 AMISP-FMS 的一个基本特征：当同一工件的相邻两道工序被分配到同一台机器时，这些工序之间不需要运输。））

### 7. 有什么可以拿来用

- **开源代码**：无。文中未提供代码仓库链接。补充材料（含 MILP 约束方程、EX72/EX84 最好已知解细节、t/p ≤ 0.25 结果、K = 5 的 La 结果、案例研究数据）可应要求从 https://doi.org/10.1109/TASE.2025.3650678 下载。
- **公开数据集（本卡片中最有价值的可复用资产）**：
  - **FJSPT benchmark**——Deroussi 和 Norre [16] 提出，含 **10 个实例**，AMISP-FMS 领域最常用，其最优性已被证明（本文、MILP [14] 与 CP2 [15] 都得到相同最好解）。
  - **EX benchmark**——Kumar et al. [18] 从 job shop 扩展到 flexible job shop；**7 个工件集 × 4 个布局 = 28 个基础实例**（如 EX74 = 工件集 7 + 布局 4），每道工序 3 台可用机器；可扩展为 **EX740**（加工时间 ×2、运输时间 ÷2）与 **EX741**（加工时间 ×3、运输时间 ÷2），本文共讨论 **57 个 EX 实例**。所有 57 个实例的最优性在本文中首次被全部证明。
  - **La benchmark**（FJSP 经典基准）——用于大规模测试；需配套 **Homayouni et al. [22] 的运输调度方案**才能用于 AMISP-FMS。
  - **案例数据**：航空领域真实机加工车间，5 种加工设备 + 2 台 AGV，10 个工件，分 Case 1（工件 1–5）与 Case 2（全部 10 个工件）。
- **可借鉴的评测协议**：
  - **以"最优性 gap = 0%"作为最优性判据**，并明确区分"找到最好可行解（正 gap）"与"可证最优（gap = 0%）"——这是精确方法论文的标准写法，可作为"我们的启发式解距离最优有多远"的量化基准。
  - **分 t/p 阈值分档报告**（t/p > 0.25 与 t/p ≤ 0.25）——运输时间与加工时间之比是区分 AMISP-FMS 实例难易的关键参数，按此分档报告很有价值。
  - **同时报告 NbVar 与 NbCon** 来定量说明建模方式的优劣（本文用此证明 NCP2 平均减少 13.87% 约束）。
  - **复现基线并声明环境与时间上限**（复现 CP2 时明确"相同编码语言与实验环境、3600 秒上限"），以及**因原文献未报告运行时间而不比较时间**（DCGA）——严谨的对比声明范式。
  - **以"更新最好已知解（更新 BKS）"作为精确方法的成果指标**（EX72、EX84），而非只与启发式比百分比。
- **最有价值的公式/建模思想**：
  - **运输任务存在性判据**：`运输任务不存在 ⟺ 同一工件的相邻工序分配到同一台机器`。这是 AMISP-FMS（以及任何带 AGV 的 FJSP）建模中最容易被漏掉的一条，直接决定了模型解的正确性。对应的 CP 表达（NCP2 式 24）：
    `presenceOf(TransTaskInVeh_ijj'k) = 1 − ( presenceOf(ProInMac_{(i−1)j}) * presenceOf(ProInMac_ij') )`
    ——**运输任务变量的 presence 由机器分配变量的乘积取反决定**，非常干净。
  - **NCP1 式 (1)**：`x^k_{kl}·x_{k(l−1)} − 1 ≥ M·(y^m_{kl} + y^m_{k(l−1)} − 2)`——用大 M 形式的**条件激活约束**，只在相邻工序同机器时把两个运输任务捆绑给同一 AGV 连续执行（避免中间产生空载行程）。
  - **CP 约束函数集合**（`endBeforeStart` / `endOf` / `presenceOf` / `alternative` / `noOverlap(Seq, Tt)` / `prev` / `IfThen`）：这是一套可直接照搬到 OR-Tools CP-SAT 等现代 CP 求解器上的建模词汇；其中 **`noOverlap(Seq, Tt)` 用 transfer time 参数表达空载行程**是本文的一个实用技巧。
  - **four-subproblem 分解框架**（机器分配 / 加工排序 / AGV 分配 / 运输排序）及其相互依赖图（Fig. 3）：可作为一个通用的"带运输 FJSP"问题分解模板。
  - **零距离、零载时行程（zero-distance, zero-load-time trip）**：把"同机器相邻工序"的运输任务退化为起点终点相同的空行程的建模方式，可作为平滑处理而非硬性删除的替代方案（NCP1 的做法）。

---

## 卡片 4：JMS2025 — Self-Organizing Scheduling via MAS + DRL

### 1. 题录

- **标题**：Manufacturing resource-based self-organizing scheduling using multi-agent system and deep reinforcement learning（Technical paper）
- **作者**：Yuxin Li, Qihao Liu, Xinyu Li (通讯), Liang Gao
  - 全部作者：State Key Laboratory of Intelligent Manufacturing Equipment and Technology, School of Mechanical Science and Engineering, Huazhong University of Science and Technology, Wuhan 430074, China
- **期刊**：Journal of Manufacturing Systems, Vol. 79 (2025), pp. 179–198
- **年**：2025（Received 23 September 2024; Received in revised form 30 November 2024; Accepted 10 January 2025; Available online 24 January 2025）
- **DOI**：10.1016/j.jmsy.2025.01.004
- **Keywords**：Multi-agent deep reinforcement learning; Multi-agent system; Production-logistics; Self-organizing scheduling; Disturbance events
- **资助**：NSFC 52188102 与 U21B2029；中央高校基本科研业务费 2024BRA004
- **Supplementary**：补充数据见 doi:10.1016/j.jmsy.2025.01.004

### 2. 它解决什么问题

- **问题类型**：**智慧工厂调度问题（SMSP, Smart factory Scheduling Problem）**，原文明确说明其本质是 **带有限 AGV 的 FJSP（FJSP with limited AGVs）**。原文指出智慧工厂调度问题的三个特征：① 制造资源丰富，典型设备包括机器与 AGV，生产—物流协同带来额外的决策与约束；② 每道工序有多个可选的异构设备，使调度更具挑战性；③ 扰动事件在实践中不可避免，使原调度方案失效。
- **涉及资源**：多台机器 `M = {M_k, k = 1, …, N_M}`（每台有操作平台、输入缓冲区、输出缓冲区）+ 多台 AGV `V = {V_l, l = 1, …, N_V}` + 原材料区（raw material area）+ 成品区（finished product area）。工件 `J_i` 有 `NO_i` 道工序 `{O_i,j, j = 1, …, NO_i}`，每道工序的加工需 AGV 把工件从当前位置运到加工机器（一个物流任务），随后机器加工（一个加工任务）；最后一道工序加工完成后，AGV 需把工件从机器运到成品区（一个物流任务）。因此 **`J_i` 有 `NO_i` 个加工任务与 `NO_i + 1` 个物流任务**。每个工件有到达时间 `AT_i`、交期 `DD_i`、权重 `w_i`（代表交付紧急程度）。
- **优化目标**：**最小化总拖期成本（TTC, total tardiness cost）**：
  - `TTC = Σ_{i=1..NJ} ( max(0, CT_i − DD_i) · w_i )`（式 1）
  - `NJ` 为工件数；`CT_i` 为工件 `J_i` 的完成时间。
- **四个子问题**（原文沿用 [30] 的分解）：① 加工任务分配（决定每个加工任务的机器）；② 加工任务排序（决定每台机器上加工任务的执行顺序）；③ 物流任务分配（决定每个物流任务的 AGV）；④ 物流任务排序（决定每台 AGV 上物流任务的执行顺序）。SMSP 即实时地同时求解上述子问题。
- **约束与假设**（原文逐条）：
  1. At the beginning, all machines and AGVs are available, and all jobs are ready for transportation and processing.（开始时所有机器与 AGV 可用，所有工件已就绪可被运输与加工）
  2. Each machine can only perform one processing task at a time, and each AGV can only perform one logistics task at a time.（每台机器一次只能执行一个加工任务，每台 AGV 一次只能执行一个物流任务）
  3. The buffer of each machine is infinite.（每台机器的缓冲区无限）
  4. Any logistics task of any job can be performed by any of all AGVs.（任一工件的任一物流任务可由所有 AGV 中的任意一台执行）
  5. The job's loading/unloading time of machines/AGVs is ignored.（工件在机器/AGV 上的装卸时间忽略）
  6. The time consumption caused by the charging, collision avoidance, acceleration, deceleration of AGV is negligible.（AGV 充电、避碰、加速、减速造成的时间消耗可忽略）
  7. The failure of AGVs is not considered.（不考虑 AGV 故障）
- **考虑的扰动事件（两类）**：
  1. **New job arrivals（新工件到达）**：连续两次新工件到达的间隔服从指数分布 `exp(1/λ_NJA)`，均值 `λ_NJA`。参考 [48] 并考虑物流任务后（式 2）：
     `λ_NJA = JPT/(U · N_M) + JLT/(U · N_V)`
     其中 `JPT = JNO · TPT`（式 3），`JLT = (JNO + 1) · TLT`（式 4）。`JNO` 为工件平均工序数；`TPT` 为所有加工任务的平均耗时；`TLT` 为所有物流任务的平均耗时；`U` 为工厂利用率。
  2. **Machine breakdowns and repairs（机器故障与修复）**：连续两次机器故障的间隔服从指数分布 `exp(1/λ_MB)`，均值 `λ_MB`；故障机器修复时间服从 `exp(1/λ_MR)`，均值 `λ_MR`。二者关系（式 6）：`λ_MB = λ_MR/(Ag − λ_MR)`，`Ag` 为工厂故障水平。基于实际生产场景与既有研究，设 `λ_MR = 3 · TPT`。
- **工件交期**（式 5）：`DD_i = AT_i + DDT_i · ( Σ_{j=1..NOi} PT_i,j + (NO_i + 1) · TLT )`，其中 `DDT_i` 为工件 `J_i` 的交期松紧度，从均匀分布 `U[1, H_DDT]` 采样；`PT_i,j` 为工序 `O_i,j` 在所有机器候选上的平均加工时间。
- **原文自陈的研究空白（research gaps）**：
  1. 已有自组织调度研究存在，但**把 MADRL 与自组织调度结合以实现 agent 进一步演化的研究很少**。
  2. 既有研究聚焦于机器资源，**忽视了物流系统中 AGV 的存在**；同时生产与物流的耦合增加了建立 MAS 与协商机制的难度。
  3. 既有研究中，每个 DRL-based agent 的决策**仅基于 agent 自身的属性**，未考虑不同 agent 之间的差异，可能导致 DRL 训练陷入局部最优。

### 3. 它怎么做（MAS-DRL 自组织调度）

**方法全貌**：**多智能体系统（MAS）+ 多智能体深度强化学习（MADRL）的自组织调度方法**，用于智慧工厂中的生产—物流协同，训练算法为 **MAPPO**（Multi-Agent Proximal Policy Optimization）。三个设计支柱：① 部分去中心化控制（partially-decentralized control）的 SF-MAS；② 基于 **CNP（contract net protocol，合同网协议）**的自组织协商机制；③ 采用 **CTDE（centralized training and decentralized execution）**框架的 MADRL + 基于三个优先级的动作空间。

**MAS 建立（SF-MAS）**：

- 采用**部分去中心化控制**架构。原文说明三种控制架构（Table 1）：集中式控制（centralized）、完全去中心化控制（fully-decentralized）、部分去中心化控制（partially-decentralized）；**其中部分去中心化控制表现最好**，它结合集中式粗粒度控制与去中心化细粒度控制的混合结构，既整合自组织智能与整体优化，又同时保证灵活性与决策效率。
- **四类 agent**：
  1. **Job management agent（工件管理 agent）**：同步所有工件的数据，如每个工序的位置、完成率、加工机器。
  2. **Cloud agent（云 agent）**：同步并存储所有制造资源与工件的数据。基于内部数据与与其他资源的交互，云 agent 可为工件分配合适的机器与 AGV；当检测到车间异常时，云 agent 激活扰动事件响应机制。此外云 agent 基于记录的工件数据形成**任务池（task pool）**以触发决策与任务分配。
     - 任务池定义：存储待分配任务，分两类——① 若任务对应一道工序，则包含一个物流任务与一个加工任务；② 若任务是把工件运往成品仓库，则只包含一个物流任务。
     - 任务池添加：开始时把每个工件第一道工序的任务（第一个物流任务 + 第一个加工任务）加入任务池；车间运行中，**当一台机器开始加工某工序时，其下一个任务（物流任务 + 加工任务）被加入任务池**。大多数既有研究采用"机器完成一道工序加工后才释放其下一个任务"的方式；**本文的方式提前了任务释放时刻**，为 AGV 留出更多时间选择与准备任务 [37]；理想情况下某工序完成时 AGV 已到达其取件位置，从而减少工件等待时间。
     - 任务池删除：任务的 AGV 分配与机器分配完成后，从任务池中移除该任务。
  3. **Machine agent（机器 agent）**：每个机器 agent 对应一台机器实体并同步其生产数据。它有一个 **DNN**，能基于输入的生成状态数据给出适当输出，参与加工任务投标并最终选择合适的任务。**该 DNN 是通过 MADRL 训练得到的**。每个机器实体由操作平台、输入缓冲区、输出缓冲区组成。
  4. **AGV agent（AGV agent）**：每个 AGV agent 对应一台 AGV 实体并同步其生产数据。它有一个 **bidding engine（投标引擎）**，可提供相关生产属性数据参与物流任务投标。**AGV agent 的投标基于启发式规则**。每个 AGV agent 有一个任务列表，负责存储待执行的物流任务。
- 原文强调：SF-MAS 采用部分去中心化控制，兼具集中式与完全去中心化控制的优点，具有强优化能力与对扰动事件的快速响应能力；同时具有高度柔性，便于按同一协议增删改制造单元。

**执行流程（Algorithm 1）**：

```
1: Cloud agent receives customer order and forms a task pool.
2: while all orders are not finished do
3:    if cloud agent makes a judgment that the current moment is a decision point then
4:       Trigger a round of CNP-based self-organizing negotiation process, which includes the logistics task bidding and processing task bidding.
5:       Machines and AGVs execute their own running logic until the next decision point.
6:    end if
7: end while
8: When cloud agent determines that all orders are finished, the scheduling process ends.
```

- **决策点定义**：当任务池中存在**可执行任务**且车间中存在**空闲 AGV** 时，当前时刻即为决策点。"可执行任务"指机器候选正常的加工任务与 AGV 候选正常的物流任务。
- **运行逻辑**：机器空闲时采用 **FCFS** 从输入缓冲区取出已到达的工件并执行加工任务；空闲 AGV **收到物流任务后立即执行**。

**重要生产数据（Table 2）**：

- `JCR_i`：工件 `J_i` 的完成率
- `CTM_k`（式 7）：机器 `M_k` 的最短完成时间 = `RTIP_k + Σ_{p=1..NBk} BPT_k,p`，`RTIP_k` 为机器 `M_k` 在制任务的剩余时间，`BT_k,p` 为机器 `M_k` 输入缓冲区中第 p 个任务，`BPT_k,p` 为其加工时间
- `MATM^{TP_a}`（式 8）：任务 `TP_a` 的所有机器候选的最小可用时间 = `min(CTM_k, M_k ∈ M^{TP_a})`
- `EAT^{TP_a}`（式 9）：任务 `TP_a` 的可获得状态（earliest available time），`= Σ_{j=1..SIa} ( (PT^{TP_a}_j − HPT^{TP_a}_j) + (TT^{TP_a}_j − HTT^{TP_a}_j) )`，`SI_a` 为任务 `TP_a` 在其工件 `J^{TP_a}` 中的序列索引；`PT^{TP_a}_j` 为工件 `J^{TP_a}` 中第 j 个加工任务的耗时；`HPT^{TP_a}_j` 为该加工任务已加工的时间；`TT^{TP_a}_j` 为第 j 个物流任务的耗时；`HTT^{TP_a}_j` 为该物流任务已执行的时间
- `JRT^{TP_a}`（式 10）：工件剩余时间 = `Σ_{j=1..NOa}(PT^{TP_a}_j − HPT^{TP_a}_j) + Σ_{j=1..NOa+1}(TT^{TP_a}_j − HTT^{TP_a}_j)`
- `ETC^{TP_a}`（式 11）：时刻 t 的估计拖期成本 = `max(0, t + JRT^{TP_a} − DD^{TP_a}) · w^{TP_a}`

**CNP 自组织协商机制（Algorithm 2）**：

```
1: if the current moment is a decision point then
2:    Implement the heuristics-based logistics task bidding for AGV agents. (目的：合理分配物流任务)
3:    Determine all tasks that have been assigned to AGVs.
4:    Determine the machine assignment task set. (机器分配任务集 = 每个已完成 AGV 分配但尚未完成机器分配的任务)
5:    Determine the competition machine set. (竞争机器集 = 所有机器分配任务集中所有任务的机器候选的集合)
6:    Implement the MADRL-based processing task bidding for machine agents. (目的：为机器分配任务集中每个任务分配一台属于竞争机器集的机器)
7: end if
```

**Agents 分工**：物流任务投标基于**启发式**（计算复杂度低、物流决策性能优），加工任务投标基于 **MADRL**（加工作业对生产计划影响重大，MADRL 可通过训练挖掘生产状态与调度动作之间的映射关系）。整个协商机制分为 AGV agent 投标与 machine agent 投标，**具有层级特征**，使决策逻辑清晰。

**（1）启发式物流任务投标（Algorithm 3）**：

```
1: Cloud agent calculates the JRT^{TP_a} and MATM^{TP_a} of all tasks in task pool.
2: Cloud agent gets the normalized priority sequences NPS_JRT and NPS_MATM by (13).
3: Cloud agent obtains the priority of each task in task pool according to NPS_JRT + NPS_MATM.
4: Cloud agent sets the priority of each task that is not executable to infinitesimal.
5: Based on priority, cloud agent determines the sorted task sequence for task pool.
6: for each task in ST^{TP_a} = {ST^{TP_a}_l, l = 1, …, b} do
7:    Cloud agent calls for a bid to each idle AGV agent.
8:    Each idle AGV agent calculates the pickup time PUT_{l,a} for the task ST^{TP_a} by (14), and enters a bid to cloud agent.
9:    Cloud agent assigns task ST^{TP_a} to AGV V_l = argmin PUT_{l,a}, V_l ∈ IAS, where IAS is the set of idle AGVs. Then cloud agent sends a bidding result to this AGV agent V_l.
10:   This AGV agent V_l sends "accept" command, and signs a contract with cloud agent.
11:   The state of this AGV changes from idle to busy.
12:   if there are no idle AGVs or there are no executable tasks in the task pool then break
13: end for
```

- **任务排序（第 1–5 行）**：定义关于 `X_w` 的序列 `S(X_w) = [X_w, w = 1, …, N_X]`，其归一化序列（式 12）：
  `Norm[S(X_w)] = [ 1 − (X_w − min S(X_w))/(max S(X_w) − min S(X_w)), w = 1, …, N_X ]`
  由此定义两个序列 `S(JRT^{TP_a})` 与 `S(MATM^{TP_a})`，归一化优先级序列（式 13）：
  `NPS_JRT = Norm[S(JRT^{TP_a})]`，`NPS_MATM = Norm[S(MATM^{TP_a})]`
  最终优先级 = `NPS_JRT + NPS_MATM`，即 **LWKR + SMAT**。
- **AGV 分配（第 6–14 行）**：高优先级任务优先分配 AGV，原则是**优先选取件时间最短的 AGV**。AGV `V_l` 对任务 `ST^{TP_a}` 的取件时间（式 14）：
  `PUT_{l,a} = Δt(L_start, L_end)`，其中 `Δt(L_start, L_end)` 表示位置 `L_start` 与 `L_end` 之间的行驶时间；`L_start` 为 AGV `V_l` 在当前时刻的位置；`L_end` 为对应工件 `J^{TP_a}` 的位置。
- 原文说明：两个属性是经过精心选择的，使紧急任务优先使用 AGV 运输能力；"shortest pickup time"原则可减少工件等待 AGV 的时间。

**（2）MADRL 加工任务投标（Algorithm 4）**：

```
Input: Machine assignment task set MATS, Competition machine set CMS
Output: Machine assignment scheme of each task in MATS
1: Each machine agent calculates its observation o_{k,t}, ∀k.
2: DNN model in agent MA_k receives o_{k,t}, and outputs action a_{k,t}, ∀k.
3: for k = 1 to N_M do
4:    if machine MA_k is not in CMS then set action a_{k,t} as a do-nothing action
5: end for
6: Each machine agent sends its action to cloud agent.
7: Cloud agent calculates the production attribute priority.
8: Cloud agent calculates the operation processing time priority.
9: while MATS is not empty do
10:    Each machine agent in CMS calculates its CTM_k based on (7), and sends it to cloud agent.
11:    Cloud agent calculates the machine completion time priority.
12:    Cloud agent calculates the sum of three above priorities, and obtain the final priority of each task in MATS for each machine in CMS. Then, cloud agent sends priority data to each machine agent.
13:    In the interaction between machine agents, each task in MATS collects the priorities of all machines in CMS for itself, and selects the machine with the highest priority. (形成机器 agent 与任务之间的映射关系)
14:    In the interaction between machine agents, for multiple tasks with mapping, each machine agent in CMS selects only the task with the highest priority. (实现对部分机器的分配)
15:    Add each assigned task to the input buffer of corresponding machine.
16:    Remove the tasks assigned in this iteration from MATS.
17: end while
```

**Dec-POMDP 建模与 CTDE 框架（Fig. 5）**：

- 因 SF-MAS 中的**所有机器 agent 属于同质 agent（homogeneous agents）**，故采用 **CTDE** 框架：所有机器 agent **共享一个 actor 神经网络、一个 critic 神经网络、一个记忆缓冲区**。Actor 网络根据观测决定动作；critic 网络评估状态价值并引导策略更新。
- **网络结构**：每个网络使用 **MLP 与 RNN 的混合（a mixture of multilayer perceptron and recurrent neural network）**；RNN 采用 **GRU（gated recurrent unit，门控循环单元）** 以捕捉状态序列的内在联系。
- 训练完成后，机器 agent 可投入应用参与投标。

**状态空间（Section 4.3.1 + Table 3）**：

- **Actor 网络的观测（Table 3）**，分三部分：
  1. **Features of order `SF|Order`**：(1) 订单完成率；(2−3) 所有工件完成率的 mean / standard deviation。
  2. **Features of machine assignment task set MATS `SF|TP`**：(1) 任务数量；(2−3) 所有任务的 `TPT^{TP_a}` 的 mean / standard deviation；(4−5) 所有工件完成率的 mean / standard deviation；(6−7) 所有工件的 `JRT^{TP_a}` 的 mean / standard deviation；(8−9) 所有工件的 `EAT^{TP_a}` 的 mean / standard deviation。
  3. **Features of single machine `SF|Machine`**：(1) 利用率；(2) `CTM_k`；(3) MATS 中可执行任务的数量；(4) 所有可执行任务的 `ETC^{TP_a}` 的均值。
- **Critic 网络的状态**：采用 **"Featured-Pruned Agent-Specific Global State"** 方式 [26]。具体为：状态拼接所有机器 agent 的观测，并**去除重复特征**。
- **归一化**：基于随机生成的 actor 神经网络与多个不同规模的实例，机器 agent 在训练模式下随机探索获得大量状态空间记忆；然后计算每个特征的均值与标准差；最终在训练阶段的**每一步都做 Z-Score 归一化**，以缓解特征之间尺度差异的负面影响。

**动作空间（Section 4.3.2）——基于三个优先级**：

- 动作输出（式 15）：机器 agent `MA_k` 接收自身观测 `o_{k,t}` 并输出动作 `a_{k,t}`：
  `a_{k,t} = [v_ETC, v_JRT, v_EAT, v_TPT, v_HPT, v_MCT]`（6 维连续向量）
- **Production attribute priority（生产属性优先级）**（式 16）：每个任务 `MAT_f` 有属性 ① `ETC_f`；② `JRT_f`；③ `EAT_f`；④ `TPT_f`（分别为对应任务的 `ETC^{TP_a}`、`JRT^{TP_a}`、`EAT^{TP_a}`、`TPT^{TP_a}`）。
  `P_MATS = v_ETC · Norm[S(−ETC_f)] + v_JRT · Norm[S(JRT_f)] + v_EAT · Norm[S(EAT_f)] + v_TPT · Norm[S(TPT_f)]`
  其中 `P_MATS ∈ R^{1×N_MAT}`，表示 MATS 中每个任务的优先级。
- **Operation processing time priority（工序加工时间优先级）**（式 17）：定义任务 `MAT_f` 在机器候选 `M_k` 上的加工时间为 `PT_{f,k}`，则任务 `MAT_f` 在所有机器候选上的加工时间集合为 `S(PT_{f,k}) = [PT_{f,k}, k = 1, …, NMC_f]`，`NMC_f` 为机器候选数。任务 `MAT_f` 对所有机器候选的加工时间优先级集合：
  `PMC_f = Norm[S(PT_{f,k})]`，`PMC_f ∈ R^{1×NMC_f}`
  由此得到 `POPT_h`（竞争机器 `CM_h` 对 MATS 中所有任务的工序加工时间优先级），`POPT_h ∈ R^{1×NMAT}`；若任务 `MAT_f` 不能在机器 `CM_h` 上加工，则优先级设为**负无穷**。
- **Machine completion time priority（机器完成时间优先级，随迭代变化）**（式 18）：定义所有机器在 `CTM_h` 上的优先级集合 `PCTM = Norm[S(CTM_h)]`，`PCTM ∈ R^{1×NCM}`。从中取出 `CM_h` 的优先级并扩展为 `N_MAT` 维矩阵，即 `PCTM_h ∈ R^{1×NMAT}`。注意 `CTM_h` 随迭代次数增加而变化，因为每次迭代都有新任务被分配到机器 `CM_h` 的输入缓冲区（Algorithm 4 第 15 行）。
- **最终优先级（式 19）**：`FP_h = P_MATS + v_HPT · POPT_h + v_MCT · PCTM_h`，`FP_h ∈ R^{1×NMAT}`，表示竞争机器 `CM_h` 对 MATS 中所有任务的最终优先级集合。
- **示例（Fig. 6）**：设 MATS = [MAT1, MAT2, MAT3, MAT4]，CMS = [M1, M3, M4, M5, M7, M8]，机器 agent MA1 的动作 = [0.8, 0.4, 0.7, −0.2, 1.0, 1.4]，据此计算 M1 对各任务的三个优先级与最终优先级。
- **设计效果（原文）**：大多数既有方法只考虑机器自身与任务的属性，未考虑不同机器之间的属性差异 [3,24]。所提方法**把工序加工时间与机器完成时间在机器间的差异加入动作空间**，使每个机器 agent 了解自己在所有 agent 中的能力水平，从而使其投标价格更合理。

**奖励函数（Section 4.3.3）**：

- 工件 `J_i` 在时刻 t 的估计拖期 `ET_{t,i}`（式 20）：
  `ET_{t,i} = max(0, CT_i − DD_i)` 若工件 `J_i` 已完成；`= max(0, t + JRT_i − DD_i)` 若工件 `J_i` 未完成。`JRT_i` 为工件 `J_i` 的剩余时间（计算方式类似式 10）。
- 订单在时刻 t 的总估计拖期成本（式 21）：`TETC_t = Σ_{i=1..NJ} ( ET_{t,i} · w_i )`
- **机器 agent `MA_k` 的奖励（式 22）**：`r_{k,t} = TETC_t − TETC_{t+1}`
- **证明（式 23）**：所有机器 agent 的累计奖励
  `R = Σ_{k=1..NM} Σ_{t=1..Terminal} r_{k,t} = Σ_k (r_{k,1} + … + r_{k,Terminal}) = N_M · (TETC_1 − TETC_2 + … + TETC_Terminal − TETC_done) = N_M · (TETC_1 − TETC_done) = N_M · (0 − TTC) = −N_M · TTC`
  原文结论：式 (23) 证明机器 agents 能同步优化累计奖励与调度目标，最终获得具备高效调度能力的 DNN。

**MADRL 训练算法（Algorithm 5，MAPPO）**：

```
1: Initialize the actor neural network (θ) and critic neural network (ϕ).
2: Initialize the hyperparameters, the shared memory buffer, and the transition number TN = 0.
3: for epoch = 1 to L do
4:    Reset the factory environment using a training instance.
5:    while all jobs have not returned to the finished product area do
6:       At the current decision point, implement the logistics task bidding based on Algorithm 3.
7:       Determine the machine assignment task set MATS and competition machine set CMS.
8:       Implement the processing task bidding based on Algorithm 4.
9:       After completing the machine/AGV assignment of tasks, machines and AGVs execute their own running logic until the next decision point.
10:      Enter the next state, and calculate the reward based on (22).
11:      Put {[o, a, r, o'], k = 1, …, N_M} into the shared memory buffer.
12:      Update the transition number: TN = TN + 1.
13:      if one disturbance event occurs then activate the response mechanism.
14:   end while
15:   Form trajectory τ, and compute advantage estimate Â via GAE using PopArt.
16:   Compute reward-to-go R̂ on τ and normalize with PopArt.
17:   if TN > MUC (the memory update capacity) then
18:      Split trajectory into chunks of length L.
19:      Shuffle the order of all chunks and divide them into minibatch sets of data.
20:      Adam update θ on (24) with all sets of data.
21:      Adam update ϕ on (25) with all sets of data.
22:      Regenerate a new instance for training.
23:      Clear the shared memory buffer and set TN = 0.
24:   end if
25: end for
```

- 第 18–19 行：把序列记忆分成块（chunks），便于基于 RNN 挖掘隐含的时序信息。
- **Actor 损失（式 24，需最大化）**：
  `L(θ) = (1/BZ) · Σ_{i=1..BZ} ( min( r_i(θ)·A_i, clip(r_i(θ), 1−ϵ, 1+ϵ)·A_i ) + σ·S[π(o_i;θ)] )`
  其中 `BZ` 为 minibatch 大小；`r_i(θ) = π_θ(a_i|o_i)/π_{θ_old}(a_i|o_i)`；`A_i` 为 `Â` 中的优势估计；`ϵ` 为裁剪参数；`σ` 为熵系数；`S` 为策略熵；`o_i` 为观测。
- **Critic 损失（式 25，需最小化）**：
  `L(ϕ) = (1/BZ) · Σ_{i=1..BZ} ( max( L_H(v(s_i;ϕ) − R̂_i), L_H(clip(v(s_i;ϕ), v(s_i;ϕ_old)−ϵ, v(s_i;ϕ_old)+ϵ) − R̂_i) ) )`
  其中 `R̂_i` 为 `R̂` 中的折扣 reward-to-go；`s_i` 为全局状态与高斯噪声向量 `N(0, ε²)` 的拼接，**噪声变化周期为 100 个 episode**，这种方式可保持噪声的多样性并提升训练稳定性 [49]；`L_H()` 为 **Huber loss** [50]，它结合了平均绝对误差的鲁棒性与均方误差的稳定性。
- 第 22 行：**每个 epoch 使用随机生成的实例**以增强训练的泛化性。
- **探索机制**：训练过程中 actor 网络输出一个均值向量，基于该向量与固定标准差从**多元高斯分布**中采样得到动作；训练完成后，**直接把输出的均值向量作为动作**用于应用。该方式使机器 agent 能充分探索并获得良好学习效果。

**扰动事件响应机制（Section 4.4）**：核心思路是**把扰动事件对工厂的影响转化为生产数据的变化**，两类投标可基于新的生产数据给出自适应决策。

1. **New job arrival**：新工件到达时，云 agent 把其第一个任务（物流任务 + 加工任务）放入任务池；若当前时刻是决策点，云 agent 还会触发一轮自组织协商过程。该方式把新工件到达转化为任务池数据的变化。
2. **Machine breakdowns and repairs**：（1）机器故障时立即停止加工。正在加工的工序称为 **"disrupted in-process operation"**，故障机器缓冲区中的工序称为 **"disrupted buffer operations"**。（2）采用 **from scratch with the interrupt-repeat mode** 处理被中断的在制工序 [39]。（3）把被中断在制工序与缓冲区工序的任务放入任务池；若当前时刻是决策点，云 agent 触发一轮自组织协商过程。（4）**故障机器不具备投标能力**。（5）修复完成后机器恢复投标能力。

### 4. 它怎么验证

**实例来源与规模**：基于某工厂布局生成实例，每个实例都加入扰动因素以测试所提自组织调度方法的鲁棒性。

- 生产配置：
  1. 机器数 **12**；分为 **6 种类型，每类型 2 台**
  2. AGV 数 **10**
  3. 工件工序数 `randi[4, 8]`
  4. 加工任务耗时 `randi[7, 34]`（单位：min）；AGV 在不同位置间的行驶时间 `randi[15, 144]`（单位：second）
  5. 工件权重 ∈ `{1, 2, 4}`，对应比例为 20%、60%、20%
  6. 最大交期松紧度 `H_DDT ∈ {1.5, 2}`
- 扰动设置：
  1. 初始工件数 **20**；新工件数 `JN = randi[50, 200]`
  2. 工厂利用率 `U ∈ {0.9, 0.95}`
  3. 工厂故障水平 `Ag ∈ {0.025, 0.05}`
  - 符号 "randi" 表示整数均匀分布。
- 布局见 Fig. 1（smart factory layout）；行驶时间矩阵在补充材料中给出。

**对比基线（列全）**：

1. **7 条复合派工规则（composite PDRs）** `{CR_i, i = 1..7}`，每条包含一个工件排序规则 + 一个机器分配规则 + 一个 AGV 分配规则：
   1. SPT + SCTM + SPUT
   2. LPT + SCTM + SPUT
   3. EDD + SCTM + SPUT
   4. S/RT + SCTM + SPUT
   5. MOR + SCTM + SPUT
   6. MS + SCTM + SPUT
   7. AVPRO + SCTM + SPUT
   - 规则含义：SPT 选工序加工时间最短的工件优先；LPT 选工序加工时间最长的工件优先；EDD 选交期最早的工件；S/RT 选单位剩余时间松弛最小的工件；MOR 选剩余工序最多的工件；MS 选松弛最小的工件；AVPRO 选每道工序平均加工时间最短的工件；SCTM 选 `CTM_k` 最短的机器；SPUT 选取件时间最短的 AGV。
2. **Genetic Programming (GP)**：参照 [39]，每个个体采用**二叉树结构**；终端集合包含 5 个工件排序规则；函数集合为 `{+, −, ×, ÷, min, max}`；**机器分配规则固定为 SCTM，AGV 分配规则固定为 SPUT**；最终通过训练获得 **4 条高质量 GP 规则**用于对比。
3. **三个 DRL 方法**：
   - **DRL-Li [39]**：采用混合 DQN（hybrid DQN）学习 GP 规则的选择，其中 GP 规则来自 5.6 节。混合 DQN 是经典 DQN、double Q-learning、prioritized replay 与 soft target network update policy 的混合。
   - **DRL-Luo [52]**：采用 MAPPO 架构与**离散动作空间**。具体为训练三个 agent，分别学习选择工件排序规则、机器分配规则与 AGV 分配规则。
   - **DRL-Park [50]**：采用 MAPPO 架构与**连续动作空间**，工件排序 agent、机器分配 agent 与 AGV 分配 agent 进行协同训练；每个 agent 的动作解码通过计算欧氏距离实现。
4. **消融对照**：
   - **8 种物流任务投标启发式**对比（以 EDD 为基线）：涉及规则 EDD、BETC（选 `ETC^{TP_a}` 最大的任务）、LWKR（选 `JRT^{TP_a}` 最短的任务）、SEAT（选 `EAT^{TP_a}` 最短的任务）、SMAT（选 `MATM^{TP_a}` 最短的任务），以及它们的组合；本文最终采用 **LWKR + SMAT**。
   - **RNN 组件消融**：X = 带 RNN 的 MAPPO；Y = 不带 RNN 的 MAPPO。

**评估指标**：

- **`SP^X_Y`（式 27）**：`SP^X_Y = (TTC_Y − TTC_X)/TTC_X · 100%`，表示方法 X 相对方法 Y 的优越比例。
- **`SP_{−EDD}`（式 26）**：`SP_{−EDD} = (TTC_EDD − TTC_X)/TTC_EDD · 100%`，表示方法 X 相对 EDD 的优越比例，值越大表示方法 X 表现越好。
- 报告形式：`SP^X_Y` 的均值（Table 4/5/6 中每个数字为 10 个实例上 `SP^X_Y` 的均值）、箱线图（Fig. 10、Fig. 11 用 violin diagram）、raincloud plot（Fig. 12(e)）；另有 **Better / Even / Worse** 计数。
- 总拖期成本 TTC 与累计奖励 Cum_Rew 的收敛曲线（Fig. 7，横轴 1000 episodes，红色曲线为 5 个实例上 TTC 的均值对应左纵轴，蓝色曲线为 5 个实例上累计奖励的均值对应右纵轴，均配 95% 置信区间阴影）。

**实验环境（硬件/软件）**：Python 编程；服务器 Intel(R) Xeon(R) W-3365 CPU @ 2.70 GHz，125 GB RAM；超参数用 **random search** [51] 搜索。

**最优超参数（Section 5.2）**：

1. Actor 或 critic 的隐藏层结构：**{MLP(64), RNN(32)}**
2. 学习率：**5 × 10⁻⁵**
3. 记忆更新容量 MUC：**215**
4. 分块长度 L：**10**；minibatch 大小：**TN//L**；所有 transition 的重用次数：**6**
5. 裁剪参数 ϵ：**0.2**；熵系数 σ：**0.01**
6. 高斯噪声向量长度：**10**

**验证协议**：

- 训练中每个 epoch 含多个 episode，每个 episode 含多个 step；**每 2 个 epoch 保存一次训练好的 DNN**。
- 随机生成 **20 个实例**，分为 **4 组**，每组对应一个 problem setting，用保存的 DNN 在 4 个 problem setting 上做验证（Fig. 7）。
- **选取收敛阶段第 811 个 episode 的训练好的 DNN** 用于后续对比实验。
- 物流任务投标启发式选择：随机生成 **9 个 MAPPO 模型与 30 个实例**（30 个实例分 3 组，每组对应一个 problem setting）；8 种启发式对应 8 种物流任务投标过程；每个 MAPPO 模型固定加工任务投标过程。Fig. 8 子图 (a)–(i) 每个柱表示 10 个实例上 `SP_{−EDD}` 的均值；子图 (j) 汇总全部情况（**9 × 30 = 270**），红色数字为 270 个值的均值，蓝色数字为对应方法取得最佳表现的情况数。
- RNN 消融：分别做两次独立训练与验证，在随机生成的 **30 个不同规模实例**上测试。
- 与复合 PDR 对比：基于三参数 `{JN, H_DDT, U}` 随机生成 **16 组实例，每组含 10 个实例**（考虑新工件到达）；考虑新工件到达 + 机器故障时，基于两参数 `{JN, Ag}` 随机生成 **8 组实例，每组含 20 个实例**。
- 与 GP 对比：与 Table 4 相同，**16 组 × 10 实例**（新工件到达）；扰动下 **8 组 × 20 实例**（violin diagram）。
- 与 DRL 方法对比：**三种对比实验使用相同的实例**。

### 5. 核心结果（照抄原文数据）

**训练与验证（Fig. 7）**：4 个 problem setting 的子图中，**TTC 曲线均下降并收敛，Cum_Rew 曲线均上升并收敛**。原文结论：机器 agents 通过 MADRL 训练能获得良好的投标能力，从而实现合理分配。**选取收敛阶段第 811 个 episode 的 DNN** 用于对比实验。

**物流任务投标启发式选择（Fig. 8）**：原文结论："Fig. 8 shows that LWKR+SMAT has the best logistics task bidding ability compared with other heuristics. Therefore, this paper adopts LWKR+SMAT to implement the AGV allocation in logistics task bidding."

**RNN 组件消融（Fig. 9，30 个实例）**：**Method X（带 RNN）在 19 个实例上取得更好结果**；30 个 `SP^X_Y` 的**均值为 4.18%**。原文结论：RNN 组件能捕捉 DRL-based 方法中状态序列的内在联系，提升方法性能。

**与复合 PDR 比较（Table 4，16 组 × 10 实例 = 160 个用例，考虑新工件到达）**：

- 每个数字为 10 个实例上 `SP^X_Y` 的均值，**多数 `SP^X_Y` 超过 10%**（原文以粗体标记）。
- 典型值：`[50,80]/1.5/0.9` 组 CR1 9.52 / CR2 12.16 / CR3 4.26 / CR4 16.53 / CR5 14.79 / CR6 7.2 / CR7 10.38；`[80,120]/1.5/0.95` 组 CR4 达 **58.93**（全表最高）；`[120,160]/1.5/0.9` 组 CR4 30.46。
- **汇总计数（160 个用例）**：

| | CR1 | CR2 | CR3 | CR4 | CR5 | CR6 | CR7 |
|---|---|---|---|---|---|---|---|
| Better | 126 | 143 | 124 | 137 | 138 | 125 | 140 |
| Even | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Worse | 34 | 17 | 36 | 23 | 22 | 35 | 20 |

**与 GP 比较（Table 5，16 组 × 10 实例 = 160 个用例，考虑新工件到达）**：

- 每个数字为 10 个实例上 `SP^X_Y` 的均值，**多数 `SP^X_Y` 超过 5%**（原文以粗体标记）。
- 汇总计数：

| | GPRule1 | GPRule2 | GPRule3 | GPRule4 |
|---|---|---|---|---|
| Better | 126 | 123 | 116 | 117 |
| Even | 0 | 0 | 0 | 0 |
| Worse | 34 | 37 | 44 | 43 |

- 扰动下（Fig. 11，8 组 × 20 实例的 violin diagram）：**多数 `SP^X_Y` 超过 5%**，所提方法在两类扰动事件下均优于 GP。

**与三个 DRL 方法比较（Table 6，16 组 × 10 实例 = 160 个用例，考虑新工件到达）**：

- 每个数字为 10 个实例上 `SP^X_Y` 的均值，**多数 `SP^X_Y` 超过 5%**。
- 典型值：`[80,120]/1.5/0.9` 组 DRL-Li 11.53 / DRL-Luo 8.36 / DRL-Park 13.55；`[120,160]/1.5/0.95` 组 DRL-Park 达 **15.19**。
- 汇总计数：

| | DRL-Li | DRL-Luo | DRL-Park |
|---|---|---|---|
| Better | 111 | 107 | 136 |
| Even | 0 | 0 | 0 |
| Worse | 49 | 53 | 24 |

- 扰动下（Fig. 12(a)–(d) violin diagram，(e) raincloud plot，蓝色虚线连接三个均值）：**多数 `SP^X_Y` 超过 5%**，所提方法在新工件到达与机器故障下均优于其他 DRL 方法。

**对基线方法表现不佳的原因分析（原文 Section 5.7）**：

1. **DRL-Li**：采用基于 GP 规则的动作空间，属于离散动作空间，使 agent 无法通过探索获得所有可能的解，从而降低学习效率。
2. **DRL-Luo**：采用基于通用规则的动作空间，与 DRL-Li 类似无法覆盖整个解空间；此外，尽管 MAPPO 框架降低了 agent 的学习难度，但通用规则可能导致探索得到的解质量差，阻碍 agent 调度能力的演化。
3. **DRL-Park**：尽管通过端到端动作空间能探索所有可能的解，但 agent 容易在庞大的调度问题解空间中迷失，即难以探索到好的轨迹用于学习。

**摘要中的总括论断**："Experimental results show that compared with scheduling rules, genetic programming and three DRL methods, the proposed method achieves better scheduling performance through reasonable competition of heterogeneous resource agents, and can effectively handle new job arrivals and machine breakdowns."

### 6. 它自己承认的局限（原文逐字引用 + 中文翻译）

> **原文**："However, compared to other methods, the proposed method cannot achieve the best solution on all instances. The superiority and stability of algorithm deserve further exploration."
>
> **中文**：然而，与其他方法相比，所提方法并不能在所有实例上都取得最优解。算法的优越性与稳定性值得进一步探索。

> **原文**："The directions of future work are as follows. (1) Adopt more advanced DNN models to extract workshop states, such as GNN or transformer. (2) Consider more disturbance events and scheduling objectives. (3) More production features are worth studying, such as the addition of auxiliary resources and its impact on AGV transportation. (4) The collision management of AGV has a significant impact on transportation time. When the difference between transportation time and operation processing time is small, collision management will further play a decisive role in the production plan, which is worth studying."
>
> **中文**：未来工作的方向如下。(1) 采用更先进的 DNN 模型来提取车间状态，例如 GNN 或 transformer。(2) 考虑更多扰动事件与调度目标。(3) 更多生产特征值得研究，例如辅助资源的加入及其对 AGV 运输的影响。(4) AGV 的碰撞管理对运输时间有显著影响。当运输时间与工序加工时间之差较小时，碰撞管理将进一步在生产计划中起决定性作用，值得研究。

（另注：假设中原文已自陈 "The failure of AGVs is not considered."——不考虑 AGV 故障；以及 "The time consumption caused by the charging, collision avoidance, acceleration, deceleration of AGV is negligible."——AGV 充电、避碰、加速、减速造成的时间消耗可忽略。）

### 7. 有什么可以拿来用

- **开源代码**：无。文中未提供代码仓库链接。补充数据（含行驶时间矩阵，以及 5.5、5.6、5.7 节所有方法结果的细节）以 Supplementary data 形式随论文提供（doi:10.1016/j.jmsy.2025.01.004）。
- **公开数据集**：无公开数据集；实例由作者基于某工厂布局自建。
- **可借鉴的评测协议**：
  - **`SP^X_Y`（式 27）与汇总计数 Better/Even/Worse**：在 16 组 × 10 实例（=160 个用例）网格上报告均值，并以三档计数汇总（例如对 CR2 为 143 Better / 0 Even / 17 Worse）。这是把统计显著性以直观方式呈现的实用做法。
  - **三类基线递进式对比结构**：① 复合派工规则（7 条，含明确的 工件排序 + 机器分配 + AGV 分配 三件套结构）；② 遗传编程（GP，固定机器/AGV 规则、只进化工件排序规则，4 条高质量规则）；③ 三个 DRL 方法（分别代表 离散动作空间/混合 DQN、MAPPO+离散、MAPPO+连续/端到端）。这三层递进是"实时调度方法"论文的标准对比骨架，可直接复用。
  - **消融的两条主线**：① 物流任务投标启发式（8 种，以 EDD 为基线，用 9 个 MAPPO 模型 × 30 个实例 = 270 个用例汇总）；② RNN 组件（带/不带 RNN，30 个实例，报告均值 4.18%）。
  - **收敛阶段选取特定 episode 的模型固定用于全部对比**（本文取第 811 个 episode），避免"对每个基线挑最优模型"的隐性不公平。
  - **每 epoch 使用随机生成实例**以增强泛化（与 P2 一致）。
- **有价值的公式**：
  - **奖励函数（式 20–23）——最有借鉴价值**：`r_{k,t} = TETC_t − TETC_{t+1}`，其中 `TETC_t = Σ_i ET_{t,i}·w_i`，`ET_{t,i} = max(0, CT_i − DD_i)`（已完工）或 `max(0, t + JRT_i − DD_i)`（未完工）。**式 (23) 给出了严格的数学证明**：`R = N_M·(TETC_1 − TETC_done) = −N_M·TTC`，即"逐步估计拖期改善量之和"与"最终总拖期成本"严格等价（相差常数因子 `N_M`）。这是把稀疏的终局目标转化为稠密逐步奖励的一个可证明正确的模板，**可直接抄用**。
  - **动作空间 = 三个优先级的加权和（式 15–19）**：
    `FP_h = P_MATS + v_HPT·POPT_h + v_MCT·PCTM_h`，其中
    `P_MATS = v_ETC·Norm[S(−ETC_f)] + v_JRT·Norm[S(JRT_f)] + v_EAT·Norm[S(EAT_f)] + v_TPT·Norm[S(TPT_f)]`
    即**由 DRL 输出 6 个连续权重 `[v_ETC, v_JRT, v_EAT, v_TPT, v_HPT, v_MCT]`，再由这些权重组合若干归一化属性得到最终优先级**。关键创新在于把**机器之间的属性差异**（`POPT_h` 加工时间优先级、`PCTM_h` 完成时间优先级）显式纳入动作空间，使每个 agent 知道自己在所有 agent 中的相对能力水平——这是"同质 agent 共享网络"下避免局部最优的一个具体手段。
  - **CTDE + 同质 agent 共享网络**：所有机器 agent 共享一个 actor、一个 critic、一个记忆缓冲区；critic 状态用 **"Featured-Pruned Agent-Specific Global State"**（拼接所有 agent 观测并去重），可直接迁移到任何"资源同质、数量可变"的车间调度场景。
  - **状态设计（Table 3）**：三类特征（订单 / 任务集 MATS / 单机）+ 全程统计量（mean/std）。其中"**所有工件完成率的 mean/std**"、"**对所有可执行任务的 ETC 均值**"这类**集合级统计特征**是处理变规模实例的通用手法。
  - **CNP 协商机制的两段式设计**：AGV 投标用启发式（复杂度低、效果好）、机器投标用 MADRL（影响大、需要学习）——**按决策重要性与计算复杂度分工**，而非全部上 DRL。这是一个实用的工程折中。
  - **物流投标的 LWKR + SMAT 规则**（式 12、13）：`优先级 = Norm[S(JRT)] + Norm[S(MATM)]`，AGV 选取件时间最短者（式 14 `PUT_{l,a} = Δt(L_start, L_end)`）。归一化用 `1 − (X_w − min)/(max − min)` 的反向归一化，使"值越小优先级越高"。
  - **含物流因素的到达率（式 2）**：`λ_NJA = JPT/(U·N_M) + JLT/(U·N_V)`，其中 `JPT = JNO·TPT`、`JLT = (JNO+1)·TLT`——与 P2 的式 2 同源，都是把加工产能与物流产能并列的到达率建模。
  - **扰动水平参数（式 6）**：`λ_MB = λ_MR/(Ag − λ_MR)`，等价于 P2 的 `Ag = MTTR/(MTBF + MTTR)`，两篇的写法可互换使用。
  - **提前任务释放（early task release）**：**当一台机器开始加工某工序时（而非完成时），就释放其下一个任务**——为 AGV 预留选择与准备时间，理想情况下工序完成时 AGV 已到达取件位置。这是对 P2 "只剩一个已分配未完成任务时释放" 的一个变体，两篇都强调了它降低工件等待时间的作用。
  - **决策点定义**：任务池中存在可执行任务 且 车间中存在空闲 AGV。
  - **from scratch with the interrupt-repeat mode**：机器故障时被中断的在制工序从头重做——P2 与 P4 采用相同处理方式，可作为标准假设复用。

### cards_batchB — 三篇论文客观摘要卡片

> 说明：仅客观复述原文内容，不做与我方工作的比较或评价。
> 提取工具：PyMuPDF 1.28.0 全文文本提取（正文 + 公式 + 表格 + 伪代码）。
> batchB 覆盖：IJPR2025_HierarchicalMARL_DFJSPT.pdf；IJPR2025_LotStreaming_HFS_AGV_DRL.pdf；COR2025_EnergySaving_FJSP_MultiAGV_DRL.pdf

---

## 卡 1 — IJPR2025 层次化多智能体 DRL 求解带运输的动态柔性作业车间

### 1. 题录
- 标题：Hierarchical multi-agent deep reinforcement learning for dynamic flexible job-shop scheduling with transportation
- 作者与机构：Wenda Wang^a, Yi Zhang^a, Yong Wang^b, Ge Pan^a, Yiping Feng^a
  - a: College of Control Science and Engineering, Zhejiang University, Hangzhou, China
  - b: Key Laboratory of Discrete Industrial Internet of Things of Zhejiang Province, Hangzhou Dianzi University, Hangzhou, China
  - 通讯作者：Yiping Feng (ypfeng@zju.edu.cn)
- 期刊：International Journal of Production Research (IJPR)
- 年/卷期：2025，published online 03 Jun 2025（Received 31 May 2024; Accepted 9 May 2025）
- DOI：10.1080/00207543.2025.2511239
- 关键词：Flexible job-shop scheduling; hierarchical multi-agent scheduling; transportation; deep reinforcement learning; imitation learning
- 资助：Zhejiang Province "Vanguard and Leading Goose + X" Science and Technology Program (Grant No. 2025C1023)

### 2. 它解决什么问题
- 问题类型：DFJSP-T（Dynamic Flexible Job-shop Scheduling Problem with Transportation）——柔性作业车间 + 运输资源 + 动态不确定性的组合变体。
- 资源：n 台作业（job）、m 台机器（每道工序有兼容机器集合 Mij ⊆ M）、r 台运输机器人（transbot，本文亦作 transport robot）；车间含仓库（位置 0）与机器区（1..m）；每台 transbot 一次只能载一个工件。
- 目标：最小化最大完工时间 Cmax = max_{i=1..n} C_i。
- 动态/不确定性来源：(a) 机器与 transbot 有"质量"参数 q∈[0,1]，实际加工/运输时间围绕标称值波动；(b) 动态事件（job arrivals, machine breakdowns, urgent orders）在引言中被列为背景动因。
- 约束与假设：每台机器同一时刻只加工一个工件；不允许抢占；同一 job 内工序顺序固定、不同 job 之间无顺序约束；初始时刻机器与 job 均可用；不考虑 setup 时间；所有 job 同优先级；机器间缓冲区无限；每台 transbot 一次载一个 job。"infinite input/output buffer capacity"。
- 时间不确定性的建模（式 13）：
  p̃_ijk = p_ijk, with probability q_k；random([p_ijk, p_ijk/q_k]), with probability 1 − q_k（transbot 同理）。决策前只能观测 p_ijk 与 q_k，p̃_ijk 只有在所有 agent 完成动作后才确定。

### 3. 它怎么做
- 总体框架：将 DFJSP-T 建模为 MDP，决策空间分解为三级（3 个 agent，非按物理实体划分而是按决策逻辑划分）：
  1) job agent（高层）：选一个 job（及其下一道待排工序）；
  2) machine agent（中层）：为所选工序选机器；
  3) transbot agent（底层）：为"job–machine"运输任务选 transbot。
  每个 agent 有各自的 state/action 空间，只共享有限信息（全局 job 状态与进度）；上一级动作会即时更新下一级 agent 的状态（sequential, coordinated）。且"the processing task actually takes place only after the transportation task has been performed"。
- 算法：Multi-agent PPO（每个 agent 各自一套 Actor-Critic，结构相同、参数独立）+ 模仿学习（IL）损失。
- 网络结构：Actor 与 Critic 均为两层 MLP，每层 256 神经元；第一层接收状态拼接，ReLU；logits 经 action mask 后 Softmax；采样得动作。Critic 直接输出状态值。
- 状态定义（可变状态空间，上限由 N^max_jobs / N^max_machines / N^max_transbots 约束）：
  - job 状态（8 元素）：i（job 编号）；j（当前待排工序索引，范围 [1, h_i+1]）；E_i(j−1)（上一工序实际完工时间）；L_i（job 当前位置，0=仓库，1..m=机器区）；(j−1)/h_i×100%（排产进度）；Σ_{j'=0}^{j−1} p̃_ij'k（累计实际加工时间）；P_ij（期望加工时间，机器无关估计，式 (1)：P_ij = (1/|M_ij|)·Σ_{k∈M_ij} P_ijk）；Σ_{j'=j}^{h_i} P_ij'（估计剩余时间）。
  - machine 状态（6 元素）：k；q_k（质量估计 0~1）；N_k（累计加工工序数）；E_i'j'k（该机器最后一道工序实际完工时间）；P_{i*jk}（被选中 job 的对应工序在该机器上的期望加工时间）；Q_{i*k}（把被选 job 运到该机器的期望运输时间）。
  - transbot 状态（6 元素）：l；q_l；N_l；G_i'j'l（最后一次运输任务实际完工时间）；L_l（位置，0=仓库，0..m=机器区）；Q_{li*}（该 transbot 到达被选 job 的期望时间）。
- 动作空间：A_j={1..n}（job 选择）；A_k⊆{1..m}（机器选择，用 mask 限定为兼容机器 M_ij）；A_l={1..r}（transbot 选择，全部可行）。使用二值 action mask（式 10）：Mask_t(a)=1 若动作合法，否则 0；非法动作（已完工 job、不兼容机器、运输物流约束）被屏蔽。
- 奖励函数：先试稀疏奖励（每步 0，结束时单次 −Cmax），会收敛慢；改用密集奖励。单步奖励 r(t) = C(t−1) − C(t)（C(t) 为到第 t 步所有已排工序的最新实际完工时间；原文写"and C(t) is obviously equal to 0"，按上下文应为 C(0)=0）。累计回报（式 2）：
  R = Σ_{t=1}^{T} r(t) = C(0) − C(T) = −C(T) = −Cmax。
  该奖励被缩放到 [−10, 10]；三个 agent 共用同一全局奖励，各自动作执行后同步更新。
- PPO 目标（式 3–9）：J(θ)=E_{τ∼π_θ}[R(τ)]；总损失
  L(θ_Actor, θ_Critic) = Ê_t[ L_Actor(θ_Actor) − c_vf·L_Critic(θ_Critic) + c_entropy·H[π(·|s_t,θ_Actor)] − c_kl·D_KL[π_old(·|s_t,θ_Actor)‖π(·|s_t,θ_Actor)] ]；
  其中 L_Actor = Ê_t[min(r_t(θ)·Â_t, clip(r_t(θ),1−ε,1+ε)·Â_t)]，r_t(θ)=π(a_t|s_t,θ)/π_old(a_t|s_t,θ)；L_Critic = Ê_t[(V(s_t,θ_Critic) − V^target_t)^2]。
- 模仿学习损失（式 11–12）：L_IL = −E_{τ∼π_expert}[ Σ_{t=0}^{T} log π_θ(a_t|s_t) ]；总损失 L_Total = L(θ_Actor,θ_Critic) + α·L_IL。α 初始 10，逐步衰减到 0。
  专家策略选择：训练时每个 episode 开始时先用 8 条组合调度规则跑该实例，取 makespan 最小者作为该 episode 的引导策略（避免单一规则的偏差）。
- 8 条组合规则（job 规则 + machine 规则 + transbot 规则）：Rule1 EST+EET+EET；Rule2 EST+SPT+SPT；Rule3 MOPNR+EET+EET；Rule4 MOPNR+SPT+SPT；Rule5 SPT+EET+EET；Rule6 SPT+SPT+SPT；Rule7 MTWR+EET+EET；Rule8 MTWR+SPT+SPT。（EST=Earliest Start Time；MOPNR=Most Operations Number Remaining；SPT=Shortest Processing Time；MTWR=Most Total Work Remaining；EET=Earliest End Time）

### 4. 它怎么验证
- 实例来源与规模：
  - 自生成实例：n×m×r，n∈{10,20,30,50}，m∈{5,10,20}，r∈{3,5,10}，共 11 种规模（表 3/表 A1：10×5×3、20×5×3、30×5×3、50×5×3、10×10×5、20×10×5、30×10×5、50×10×5、10×20×10、20×20×10、30×20×10）；每种规模生成 5000 个训练实例；每规模取 100 个测试实例。参数生成：no-load 运输时间 ∼U[1,10]，loaded = 1.2×no-load；每 job 工序数 ∈[2,m]；标称加工时间 p_ij∈[1,100]；机器/transbot 质量 ∈[0.1,1]（1 位小数）；不可加工时 p_ijk=−1。
  - 公开基准：Ham (2020) 的第一个机器布局数据集 HUdata（四组 sdata/edata/rdata/vdata，各 40 个实例 la01–la40）；benchmark 中无动态因素，故机器与 transbot 质量参数全部设为 1。
  - 另一数据分布：按 Min Zhang et al. (2023) 的生成方法训练 9 种规模（10×5×2 到 40×10×3），加入 job 到达时间约束（到达前 mask 为非法），质量参数设为 1。
- 对比基线（列全）：
  - 8 条组合调度规则（Rule1–Rule8，上表）；
  - 无 IL 引导的 DRL 变体（"DRL"）；
  - 公开基准上：CP（Ham 2020 的约束规划）与 GNN_DRL（Min Zhang et al. 2023 的 GNN+DRL）；
  - 另一数据分布上：MWKR 规则（≈本文 Rule8）与 Min Zhang et al. (2023) 的第一个 AGV 分配策略结果。
- 评估指标：makespan（均值）、shifted geometric mean（SGM，式 14，s=100）、relative performance ratios（RPR，DRL+IL 归一为 1.000）、标准差（STD）、收敛曲线、Gantt 图、运行时间（每推理时间归一化）。
- 实验环境：RLlib + PyTorch；Linux 服务器，Tesla V100 GPU，56 个 Intel Xeon CPU 核；50 个 rollout worker（每 worker 1 个环境）；训练限时 24 h。MLP 结构见上。PPO 超参：γ=1，lr 1e−4 线性衰减到 1e−5，grad clip 0.5，Adam，λ=0.95，nsgd=3，mini-batch 9×n×m，c_kl=0.2，c_vf=1，c_entropy=0，ε=0.3，ε_vf=10，batch size 90×n×m×N_roll。

### 5. 核心结果（照抄原文数据）
- 表 3（不同规模 100 个实例的 SGM，RPR，STD）示例：
  - 10×5×3：DRL+IL SGM 642.46（RPR 1.000）、DRL 647.88（1.008）、Rule1 810.71（1.262）、Rule2 1187.15（1.848）、Rule7 859.22（1.337）。
  - 10×20×10：DRL+IL 1609.60（1.000）、DRL 1719.63（1.068）、Rule1 1976.19（1.228）。
  - 20×20×10：DRL+IL 1849.89（1.000）、DRL 2101.41（1.136）、Rule1 2166.46（1.171）。
  - 50×10×5：DRL+IL 2685.56（1.000）、DRL 2689.38（1.001）、Rule1 3128.67（1.165）。
- 相对提升（原文）："Compared to DRL without imitation learning guidance, the DRL+IL method achieves an average relative performance improvement of 4.4%, with a notable increase of 13.6% in large-scale problems. In comparison to dispatching rules, the DRL+IL method shows a minimum improvement of 4.7%, while the maximum improvement reaches 130.3%."（DRL 方法在所有规模上 STD 显著低于调度规则）
- 公开基准 HUdata（表 4）：
  - sdata：CP 2975.1 / GNN_DRL 2597.2 / DRL+IL 2449.95（Time: GNN_DRL 2.9 s, DRL+IL 1.2 s）
  - edata：2920.5 / 2561.9 / 2386.43
  - rdata：2398.9 / 2513.8 / 2210.05
  - vdata：1921.0 / 2524.4 / 1909.30
- 按 Zhang et al. (2023) 分布（表 5，100 实例，括号内为标准差）：9 组中 8 组本文 Cmax 更低；例如 10×5×2：247.91(17.93) vs MWKR(Rule8) 378.60(38.80)、GNN_DRL 286.32；40×10×3：1027.38(39.74) vs MWKR 1501.32(77.75)、GNN_DRL 1081.03；15×15×3 是唯一例外（本文 689.90 vs GNN_DRL 607.35）。
- 运行时间："in a test set sized 40 × 10 × 3, our method demonstrates a runtime reduction exceeding fivefold compared to that of Min. Zhang et al. (2023)"；"our method, with a time complexity of O(N_e), where N_e denotes the episode length"。

### 6. 它自己承认的局限（原文逐字引用 + 中译）
来源：第 6 节 Conclusion，第 21 页。
> "However, the DRL+IL method has inherent limitations. Its effectiveness partially relies on the quality of initial heuristic guidance, which, if suboptimal or biased, can lead to premature convergence to suboptimal solutions. Additionally, performance may degrade when models trained on one dataset are generalised to instances of different scales, due to the agents not encountering out-of-distribution data during training."

中译：然而，DRL+IL 方法存在固有局限。其效果部分依赖初始启发式引导的质量；若该引导次优或有偏，可能导致过早收敛到次优解。此外，当把在一个数据集上训练的模型泛化到不同规模的实例时，性能可能下降，因为智能体在训练中没有遇到过分布外数据。

来源：第 5.6 节，第 20 页（对 15×15×3 个例的说明）。
> "While our method demonstrates significant superiority in most cases, there are specific instances, such as the 15 × 15 × 3 problem, where it does not perform as well. The increased number of machines in the workshop leads to more complex dependencies between jobs and machines. Our current state representation, while effective in general, might lack the nuanced granularity captured by GNNs in certain instances. This can result in less accurate policy updates and suboptimal action selections in environments where understanding detailed dependencies is crucial. This is a direction for further research. For example, it may be beneficial to involve more correlated states between agents while designing the hierarchical structure."

中译：尽管我们的方法在多数情形下表现出明显优势，但在特定实例（如 15×15×3 问题）上表现不佳。车间机器数量增加导致 job 与机器之间的依赖关系更复杂。我们当前的状态表示总体上有效，但在某些实例中可能缺少 GNN 所捕捉到的细粒度信息，从而在需要理解细节依赖的环境中导致策略更新不够准确、动作选择次优。这是进一步研究的方向，例如在设计层次结构时引入更多 agent 间的相关状态。

来源：第 4.1 节（数据加工模型的边界）。
> "The specific design intricacies of the data processing model, while crucial, extend beyond the paper's scope."

中译：数据处理模型的具体设计细节虽然关键，但超出了本文范围。

### 7. 有什么可以拿来用
- 开源代码：https://github.com/ClouDaDaDa/DFJSPT_code （论文注 1 与 Data availability statement 均给出；论文未声明许可证类型，需自行到仓库确认）。
- 公开数据集：Ham (2020) HUdata（sdata/edata/rdata/vdata 四组，各 40 个实例 la01–la40），原文引用：Ham, Andy. 2020. "Transfer-Robot Task Scheduling in Flexible Job Shop." Journal of Intelligent Manufacturing 31 (7): 1783–1793. https://doi.org/10.1007/s10845-020-01537-6
- 可借鉴的评测协议：
  - SGM（shifted geometric mean，式 14，s=100）+ RPR（以本方法归一为 1.000）+ STD 三件套报告方式；
  - 运行时间按 episode 长度归一化，报告"每次推理时间"以证明与规模无关；
  - 训练实例生成器参数（n×m×r、运输时间 U[1,10]、loaded=1.2×no-load、p_ij∈[1,100]、quality∈[0.1,1]）可直接复刻；
  - 时间不确定性的两段式采样（式 13）与"决策前仅可观测 p_ijk 与 q_k"的信息结构。
- 有价值的公式：Cmax 目标与假设清单；三级状态向量定义（8/6/6 元素）；action mask 定义（式 10）；密集奖励 r(t)=C(t−1)−C(t) 与 R=−Cmax 的等价关系（式 2）；PPO 完整损失（式 4–9）；模仿损失 L_IL 与 α 退火（式 11–12）；质量-时间不确定性采样（式 13）。

---

## 卡 2 — IJPR2025 基于 DRL 的多目标批次流（lot-streaming）混合流水车间与 AGV 集成调度

### 1. 题录
- 标题：Integrated scheduling of multi-objective lot-streaming hybrid flowshop with AGV based on deep reinforcement learning
- 作者与机构：Hongtao Tang, Jiawei Huang, Chenhao Ren, Yiping Shao, Jiansha Lu —— College of Mechanical Engineering, Zhejiang University of Technology, Hangzhou, People's Republic of China（通讯：Hongtao Tang, tanght@zjut.edu.cn）
- 期刊：International Journal of Production Research (IJPR), 2025, VOL. 63, NO. 4, 1275–1303
- 年：Received 17 January 2024; Accepted 19 June 2024; Published online 09 Jul 2024（卷年为 2025）
- DOI：10.1080/00207543.2024.2373426
- 关键词：Lot-streaming hybrid flowshop scheduling; AGV; NSGA-II; integrated scheduling; deep reinforcement learning
- 资助：Natural Science Foundation of Zhejiang Province [LQ22E050017]

### 2. 它解决什么问题
- 问题类型：LSHFSP-AGV —— 三阶段 lot-streaming 混合流水车间（hybrid flowshop）与 AGV 物料搬运的集成调度（研究背景为某 PC 制造企业实际车间）。
- 资源：n 个订单（order），每订单含 N_i 个工件，可拆成 SN_i 个等量 sublot；三阶段，阶段 j 有 m_j 台机器（至少一个阶段含多台机器）；v 台 AGV，每台载重上限 20 个工件；车间布局已知。
- 目标（三目标，可能相互冲突）：
  - f1 = min(max C_ipjk)（最小最大完工时间）；
  - f2 = min Σ_j Σ_k (C_jk − S_jk − Σ_i Σ_p X_ipjk·t_ipjk)（最小机器总空闲时间）；
  - f3 = min Σ_i Σ_p Σ_f Σ_h (ET_ipfh − E'_Tipfh + LT_ipfh − L'_Tipfh)·V_r（最小 AGV 总运输距离）。
- 约束与假设（原文 10 条 + 式 8–17）：时刻 0 所有机器/sublot/AGV 可用；一台机器一次加工一个 sublot，一个 sublot 每阶段只在一台机器加工一次；不同订单工序间无顺序约束；加工一旦开始不可中断；每台机器缓冲区足够大；每个 sublot 同时只由一台 AGV 运输，每台 AGV 同时只运输一个 sublot；AGV 速度恒定且空载/负载速度相同；AGV 运输一旦开始不可中断、不考虑抢占；AGV 完成当前任务后可立即从当前位置开始下一任务；不考虑 AGV 路径冲突、充电与故障。
- 批策略：采用 equal sublots（非可混 sublot，non-mixable），sublot 批量固定 20（与 AGV 载重匹配）；SN_i = N_i/20（整除）或 N_i/20+1（不整除），最后一批为余数。文中用同一算例比较"不分批 / 等量分批 / 变量分批"：等量分批 makespan 7230.85、机器空闲 6816.0、AGV 距离 10712.47；不分批分别为 12242.02、17896.41、1246.4；变量分批分别为 12454.55、11700.95、21904.46。
- 静态调度（论文自认，见局限）。

### 3. 它怎么做
- 算法：NSGA2-MDDQN —— "improved multi-objective double-depth Q learning algorithm based on NSGA-II"，即用 NSGA-II 的进化策略为 MDDQN 生成/改进奖励信号，把 DQN 的利用与 NSGA-II 的探索结合；MDP 用四元组 (S, A, γ, R)（原文写"quaternion (S, A, γ, R)"，同时也提到 P）。
- 网络：MDDQN 使用两个深度神经网络（当前网络 + 目标网络）以降低 Q 值高估、提升稳定性；三个目标共享同一状态输入与同一特征提取网络，独立输出层与各自奖励函数分别计算 Q 值。原文未给出隐藏层维度/层数等具体结构参数。
- 状态（8 个特征，式 20–24）：
  - job 相关（4）：订单 sublot 平均完成率 CRJ_ave、工序平均完成率 CRO_ave、工序完成率标准差 CRO_std、当前最大完工时间 C_max；
  - machine 相关（3）：机器总空闲时间 FT_m、机器平均空闲率 U_ave、平均空闲率标准差 U_std；
  - AGV 相关（1）：AGV 总运输距离 S_v。
  - 其中 CRJ_i=OP_i/SN_i（OP_i 为已完成 sublot 数）；CRO_i=(Σ_p OA_ip·JN_ip)/(SN_i·a)（OA_ip 为 sublot p 已完成阶段数）；U_jk=(C_jk−S_jk−Σ_iΣ_p X_ipjk t_ipjk)/(C_jk−S_jk)。
- 动作空间：18 条组合调度规则 = 3（job）×2（machine）×3（AGV）：
  - job 规则 3 条：①选估计完工时间 ECT_ip 最大的 sublot（ECT_ip=argmax(C_ipJipk + t_ip + S_ip)，其中剩余加工时间 t_ip 取剩余阶段可用机器平均加工时间、剩余运输距离 S_ip 取剩余阶段机器间平均距离，式 25–27）；②选期望剩余加工时间最长者；③FIFO。
  - machine 规则 2 条：①选最早可用机器 M_jk=argmin(C_jk)；②选空闲时间最长的机器 M_jk=argmax(FT_jk)。
  - AGV 规则 3 条：①选最早可用 AGV V_h=argmin(L_h)；②在①基础上加入本任务空载+负载时间，选最快到达下一阶段机器的 AGV V_h=argmin(L_h + ET_ipfh − E'_Tipfh + LT_ipfh − L'_Tipfh)；③选累计运输距离最小（负载均衡）的 AGV V_h=argmax(S_h)。
- 奖励函数（核心创新，式 18 未编号但见 Algorithm 1–3）：对三个目标分别设计奖励；每步回溯 NSGA-II 与 MDDQN 各自 Pareto 解集中的"最小距离解"（minimum distance solution）得到逐步三目标值，取二者较小值作为奖励基准；当前状态值更小→+1，相同→0，更大→−1；三个奖励值求和作为最终奖励。第一轮迭代以 NSGA-II 结果作为奖励信号，后续迭代同时比较 NSGA-II 与 MDDQN 的最小距离解。
- NSGA-II 部分：实数编码 + 随机排序初始化（Bean 1994）；三条染色体（job 序列、机器分配、AGV 分配），长度均为 a×Σ_i SN_i；阶段 1 的 AGV 染色体全为 0（不需要运输）；单点交叉 + 单点变异（机器/AGV 染色体限制在各阶段范围内）；父子合并→解码→非支配排序→精英策略保留最优解。
- Q 值更新（式 33–34）：DDQN 形式
  y_t = r_{t+1} + γ·q(s_{t+1}, argmax_a q(s_{t+1},a;θ_t); θ'_t)。
- 参数：Taguchi 实验（5 水平）选优，指标为 HV；最终取 N=125、CR=0.3、MR=0.03、α=0.001、γ=0.8、ε=0.1。

### 4. 它怎么验证
- 实例来源与规模：全部随机生成，共 27 个算例（小/中/大），参数见表 4：订单数 n∈{5,10,15}；每订单工件数 N_i∈100–200；机器总数 m∈{12,24,36}（三阶段分为 4/5/3、8/10/6、12/15/9）；AGV 数 v∈{3,4,5}；阶段间距离 60，同阶段机器间距离 30；工序加工时间 1–50；单工件分拣时间 1–50。每个算例重复 20 次取平均。算例命名为 n#-m#-v#（原文示例文字与表格存在口径不一致，表头以 m1/m2/m3 三阶段机器数列出）。
- 对比基线（列全）：① 构成动作空间的 18 条组合调度规则；② NSGA-II（参数与 NSGA2-MDDQN 中 NSGA-II 部分一致）；③ MDDQN（参数与环境配置与 NSGA2-MDDQN 中的 MDDQN 部分一致，仅奖励函数不同）。
- 评估指标：三目标均值（makespan、机器总空闲时间 UT_m、AGV 总运输距离 S_v）、Spacing（式 35–36）、HV（式 37，先做 Z-score 归一化，式 38）、参数实验用 HV、敏感性分析（n、m、v 变化）、ANOVA/均值图、3D Pareto 图、20 次运行数据。
- 实验环境："The code is written in Python language, compiled and run using Python 3.7.6. The computer CPU is AMD Ryzen 5 5600U, the computer runs on 16G of RAM, and the graphics card is AMD Radeon (TM) Graphics."

### 5. 核心结果（照抄原文数据）
- 摘要与结论给出的一致数字："on the objective of minimising makespan, NSGA2-MDDQN reduces by an average of 23.17% compared to the composite scheduling rule, NSGA-II and MDDQN. And achieving an average reduction of 43.78% in machine idle time, and 9.12% in AGV transport distance."
- 与 18 条组合规则比较（27 算例、20 次运行）：makespan 上 20/27 个算例由 NSGA2-MDDQN 取得最优（Rule11 拿 4 个，Rule1 与 Rule12/Rule18 各 1 个）；机器总空闲时间 15/27 最优（Rule8 拿 8 个）；AGV 总距离 14/27 最优（Rule7 拿 5 个、Rule11 拿 4 个）。例如 15×36 组中 n=15、m=12/15/9、v=3 时 NSGA2-MDDQN makespan 21768 vs Rule1 30619 / Rule2 31953。
- 与 NSGA-II、MDDQN 比较：makespan 仅 1 个算例由 NSGA-II 取得最优，其余均为 NSGA2-MDDQN；机器空闲时间 2 个算例由 MDDQN 最优、其余 25 个为 NSGA2-MDDQN；AGV 总距离 1 个由 MDDQN 最优、其余为 NSGA2-MDDQN。（如 n=5、m=4/5/3、v=3：NSGA-II 8873 / MDDQN 7773 / NSGA2-MDDQN 7149）
- Spacing：27 个算例中 2 个在 MDDQN 上最小，其余 25 个在 NSGA2-MDDQN 上最小；HV：全部算例在 NSGA2-MDDQN 上取得最大值。
- 敏感性分析结论（原文）：订单数增加 → 三目标均增大；机器数增加 → makespan 总体下降、UT_m 总体上升、S_v 总体上升；AGV 数增加 → makespan 与 UT_m 总体下降、S_v 更平缓；m=12 时增加 AGV 对三目标影响较小。

### 6. 它自己承认的局限（原文逐字引用 + 中译）
来源：第 6 节 Concluding remarks，第 26–27 页。
> "First, the LSHFSP-AGV studied in this paper is static scheduling, but considering the frequent occurrence of dynamic events in the actual production system, in order to enhance the practicability and accuracy of the model, consideration should be given to introducing more variables and constraints that may be encountered in the actual production, such as random machine failures, maintenance schedules, job insertion, etc., to construct a production scheduling model that is closer to reality."

中译：首先，本文研究的 LSHFSP-AGV 是静态调度；考虑到实际生产系统中动态事件频繁发生，为提升模型的实用性与准确性，应考虑引入更多实际生产中可能遇到的变量与约束，如随机机器故障、维修计划、工件插入等，以构建更贴近现实的生产调度模型。

> "However, the proposed algorithm in this study is specifically for certain goals and scenarios, and this algorithm has not been validated in other extended problems for HFSP. Applying this algorithm to other problems requires designing the state space, action space, and reward function for different problems and adjusting the network structure."

中译：然而，本研究提出的算法针对特定目标与场景，尚未在 HFSP 的其他扩展问题上得到验证；将该算法应用于其他问题需要针对不同问题重新设计状态空间、动作空间与奖励函数，并调整网络结构。

> "In addition, this study does not consider the path conflict problem that may occur during AGV traveling, which may affect the delivery time of AGVs and the overall efficiency of the system. Future research can explore the path conflict problem in depth, study its impact on production performance, and explore ways to reduce the conflict in order to solve the production scheduling challenges in practical applications more comprehensively."

中译：此外，本研究未考虑 AGV 行驶中可能发生的路径冲突问题，这可能影响 AGV 配送时间与系统整体效率。未来研究可深入探讨路径冲突问题、研究其对生产性能的影响，并探索减少冲突的方法，以更全面地解决实际应用中的生产调度挑战。

其他可引用的数据可得性声明：
> "Data will be made available on request."（中译：数据可应要求提供。）

### 7. 有什么可以拿来用
- 开源代码：论文未提供代码仓库或链接。
- 公开数据集：论文未使用公开数据集，全部为随机生成实例（生成参数见表 4，可复刻）。
- 可借鉴的评测协议：
  - 三目标 + Spacing + HV（附 Z-score 归一化公式，式 38）的 Pareto 解集比较方式；
  - Taguchi 实验以 HV 为主效应指标做参数选优（N、CR、MR、α、γ、ε 五水平）；
  - 20 次重复运行取均值 + 单目标最优计数（"多少个算例上取得最优"）的呈现方式；
  - 用同一算例对比"不分批 / 等量分批 / 变量分批"的批策略消融。
- 有价值的公式：LSHFSP-AGV 的 MILP 目标与约束（式 1–19，含 4 个决策变量 X/Y/Z/αT）；8 个状态特征定义（式 20–24）；18 条组合规则（式 25–32）；DDQN 目标值（式 34）；Spacing 与 HV 定义（式 35–37）；三等量分批公式（式 18–19）。

---

## 卡 3 — COR2025 DRL 求解高效节能的多 AGV 柔性作业车间调度

### 1. 题录
- 标题：Deep reinforcement learning for solving efficient and energy-saving flexible job shop scheduling problem with multi-AGV
- 作者与机构：Weiyao Cheng^a, Chaoyong Zhang^b, Leilei Meng^a,*, Biao Zhang^a, Kaizhou Gao^c, Hongyan Sang^a
  - a: School of Computer Science, Liaocheng University, Liaocheng 252000, China
  - b: State Key Lab of Digital Manufacturing Equipment and Technology, Huazhong University of Science and Technology, Wuhan 430074, China
  - c: Institute of Systems Engineering, Macau University of Science and Technology, Taipa 999078, Macau
  - 通讯作者：Leilei Meng (mengleilei@hust.edu.cn)
- 期刊：Computers & Operations Research 181 (2025) 107087
- 年：Received 7 August 2024; Received in revised form 28 February 2025; Accepted 31 March 2025; Available online 4 April 2025
- DOI：10.1016/j.cor.2025.107087
- 关键词：Automatic guided vehicle; Flexible job shop scheduling problem; Deep reinforcement learning; Deep Q-network
- 资助：NSFC [52205529, 62303204]；Natural Science Foundation of Shandong Province [ZR2021QE195, ZR2021QF036] 等

### 2. 它解决什么问题
- 问题类型：FJSP-AGV（柔性作业车间 + 多 AGV 运输），静态调度。
- 资源：n 个 job（每个含 n_i 道工序，工序可在可选机器集 K_ij 中选机）、m 台机器、n_V 台 AGV；所有 job 与 AGV 初始位于装卸位（loading and unloading, LU）；AGV 运输分空载（no-load）与负载（load）两段。
- 三个优化任务：① 最小化 makespan；② 最小化总能耗 TEC；③ 同时最小化 makespan 与 TEC（求 Pareto 前沿）。
- 目标函数：Cmax = max_{i∈I} ( S_{i,ni} + Σ_{k∈K_i,ni} Σ_{v∈V} X_{i,ni,k,v}·pt_{i,ni,k} )（式 1）。
- 能耗模型（式 2–5）：TEC = PE + IE + TE
  - PE = Σ_v Σ_k Σ_i Σ_j P_{i,j,k}·pt_{i,j,k}·X_{i,j,k,v}（加工能耗）
  - IE = Σ_k ( L_k − Σ_v Σ_i Σ_j pt_{i,j,k} X_{i,j,k,v} ) · P^idle_k（空闲能耗）
  - TE = Σ_v Σ_v' Σ_i Σ_j Σ_k Σ_k' X_{i,j−1,k,v} X_{i,j,k',v'} Tr_{k,k'} (P^load_v − P^idle_v) + Σ_v LA_v P^idle_v（运输能耗）
- 约束与假设：时刻 0 所有 job/机器/AGV 可用；job 无需返回 LU；每台机器任一时刻至多加工一道工序；每台 AGV 任一时刻至多运输一道工序；同一 job 的工序按给定顺序加工；工序加工不可中断（不允许抢占）。
- 三个子问题需同时优化：机器选择、工序排序（含 AGV 上的顺序）、AGV 选择。

### 3. 它怎么做
- 算法：DQN（Mnih et al. 2015 的经典 DQN，含 target network + experience replay），训练一个调度智能体；状态→Q 网络→每个动作打分→ε-greedy 选动作→奖励→存入经验回放。
- 网络结构：5 层全连接，激活函数 ReLU（rectified linear unit）；隐藏层维度依次为 64、128、64、32；输入层维度 = 状态特征向量长度；输出层维度 = 动作集大小。
- 状态：12 个特征（式 6–17），分三类——
  - 机器（4）：机器平均利用率 MU_ave、机器利用率标准差 MU_std、机器平均完工时间 MCT_ave、机器完工时间标准差 MCT_std；
  - job（4）：工序平均完成率 CR_ave、工序完成率标准差 CR_std、job 平均完工时间 JCT_ave、job 完工时间标准差 JCT_std；
  - AGV（4）：AGV 平均利用率 VU_ave、AGV 利用率标准差 VU_std、AGV 平均完工时间 VCT_ave、AGV 完工时间标准差 VCT_std。
- 动作空间：16 条组合调度规则 = 4（工序排序）×2（机器选择）×2（AGV 选择）：
  - 工序排序 4 条：MOPNR（剩余工序数最多）、FIFO（最早到达）、LWKR（剩余加工时间最少）、MWKR（剩余加工时间最多）；
  - 机器选择 2 条：EET（最早可用机器）、SPT（加工时间最短的机器）；
  - AGV 选择 2 条：V1（选能最早到达工序完工机器的车辆）、V2（选最早可用的车辆）（Homayouni and Fontes et al., 2023）。
  - 编号约定：1=FIFO+SPT+V1，…，16=MWKR+EET+V2。
- 奖励函数（式 18–20）：
  - 单一 makespan 目标：R(s_t, s_{t+1}, a_t) = Cmax(s_t) − Cmax(s_{t+1})
  - 单一 TEC 目标：R(s_t, s_{t+1}, a_t) = TEC(s_t) − TEC(s_{t+1})
  - 同时优化（加权综合奖励）：R(s_t, s_{t+1}, a_t) = α(Cmax(s_t) − Cmax(s_{t+1})) + (1−α)(TEC(s_t) − TEC(s_{t+1}))，α∈[0,1]
  - 核心思想：奖励为状态 s_t 与 s_{t+1} 目标值之差，奖励越大对应目标值越小；改变 α（0 到 1，步长 0.1，共训练 11 个模型）可得到 Pareto 前沿解。
- 评估指标：RPI_i = (V_i − V_best)/V_best × 100（式 21）；同时优化任务另用 GD、IGD、HV 三个多目标指标；统计上采用 ANOVA（95% LSD 均值图）与配对 t 检验。
- 训练细节：训练步数 3000；Adam 优化器；ε 从 1.0 到 0.1；折扣因子 γ=0.99；学习率 1e−4；batch size 32；经验回放缓冲区 1000；目标网络更新频率 100。

### 4. 它怎么验证
- 实例来源与规模（共 98 个测试实例）：
  - 训练集与验证集：以公开基准的 8 种规模（5×6、5×7、6×7、7×7、8×7、9×8、11×8、12×8，形如"5×6"表示 5 jobs × 6 machines）各生成 10 个实例 → 训练/验证各 8 组 × 10 个；
  - 公开集（public set）：MFJS01–MFJS10，下载自 https://fastmanufacturingproject.wordpress.com/2019/04/11/fjspt-instances/ ；
  - 泛化集（generalized set）：8 个规模为原实例两倍的实例（10×6、10×7、12×7、14×7、16×7、18×8、22×8、24×8），用对应小规模模型求解以验证泛化性；
  - 所有实例 AGV 数均为 2；加工时间 ∈[80,100]，加工功率 ∈[4,10]，机器空闲功率 ∈{1,2,3}，AGV 空载功率 1、负载功率 2。
- 对比基线（列全）：
  - 16 条组合调度规则；
  - 精确方法：MILP（Homayouni and Fontes, 2021）；
  - 元启发式：DCGA（Han et al., 2024）、IGA（Meng et al., 2023b）；
  - 其他 DRL 方法：DDQN（Yuan et al., 2023b）；
  - 同时优化任务：DQNMA（Zhang et al., 2024，原文称为目前 FJSP-AGV 同时优化 makespan 与 TEC 的 state-of-the-art）与 NSGA-II（Deb et al., 2002）。
  - 公平性处理：DCGA、IGA、MILP 运行时间统一限制为 3 s（原文说明 3 s 满足实际生产要求；MILP 在 3 s 内无法得到可行解）；DQNMA 与 NSGA-II 同样限时 3 s。
- 实验环境：PyTorch 1.12、Python 3.9；Intel® Core™ i7-12700 + NVIDIA T600 台式机。

### 5. 核心结果（照抄原文数据）
- 验证集：makespan 的 ARPI = 0（对全部实例均取得最好解），配对 t 检验 p=0.004 / 0.003（vs MOPNR+EET+V1 / V2）；TEC 同样最优，p=0.006 / 0.016。
- 公开集 makespan（表 8，均值行）：Ours 962.9（RPI 27.48）vs DDQN 1014.1（35.78）vs DCGA 1619.9（105.22）vs IGA 1679.2（113.55）；"Best"（两种元启发式与精确方法所得最低值）均值 731.8。平均 CPU 时间：Ours 0.011 s，DDQN 0.012 s，DCGA/IGA/MILP 各 3 s（限时）；原文另述 DCGA/IGA 不限时平均 548 s、MILP 平均 6474 s。配对 t 检验：Ours vs DDQN p=0.039，vs DCGA p=0.004，vs IGA p=0.002。
- 公开集 TEC（表 12，均值行）：Ours 25207.14（RPI 0.85）vs DDQN 26936.58（7.56）vs DCGA 33884.98（37.25）vs IGA 36626.04（47.03）；配对 t 检验 p=0.043 / 0.000 / 0.000。单实例示例 MFJS01：Ours 12902.8 vs DDQN 13311.8 vs DCGA 15529.4 vs IGA 16992.8。
- 泛化集：makespan 10 个实例中 9 个最优（MOPNR+EET+V1 在 10×7 上最优）；TEC 全部实例最优。
- 同时优化（表 18，均值）：GD Ours 0 vs DQNMA 0.333 vs NSGA-II 0.399；IGD 0 vs 0.340 vs 0.471；HV 0.997 vs 0.386 vs 0.336；三个指标的配对 t 检验 p 均 ≤0.05（GD/IGD/HV 的 Ours vs DQNMA 为 0.000/0.001/0.000，vs NSGA-II 均 0.000）；Ours 平均 CPU 时间 0.127 s。
- 决策可解释性（MFJS01）：makespan=559 的动作序列 [9, 3, 13, 3, 11, 1, 3, 2, 13, 5, 3, 8, 10, 4, 14]；TEC=12902.8 的动作序列 [5, 12, 2, 2, 11, 3, 8, 14, 8, 8, 3, 5, 6, 9, 9]；Pareto 前沿构建中规则 1、3、4、9、11、12 占 73.3% 的选择。

### 6. 它自己承认的局限（原文逐字引用 + 中译）
本文没有专门的 Limitations 小节；最接近的自我陈述是第 6 节 "Conclusion and future research" 中的以下原文：
> "In future research, additional constraints on the FJSP-AGV will be considered, such as distributed manufacturing and dynamic events. In addition, we will integrate advanced techniques of large language model and heterogeneous graph neural networks into our algorithm."

中译：未来研究将考虑 FJSP-AGV 的更多约束，例如分布式制造与动态事件；此外，我们将把大语言模型与异构图神经网络等先进技术集成到算法中。

数据可得性声明原文：
> "Data will be made available on request."

中译：数据可应要求提供。（论文未提供代码仓库或链接。）

### 7. 有什么可以拿来用
- 开源代码：论文未提供代码链接（Data availability 仅为"应要求提供"）。
- 公开数据集：MFJS01–MFJS10（FJSP-T 实例，含 AGV），下载地址 https://fastmanufacturingproject.wordpress.com/2019/04/11/fjspt-instances/ ；训练/验证集按该基准的 8 种规模各生成 10 个实例（生成规则：加工时间 [80,100]、加工功率 [4,10]、空闲功率 {1,2,3}、AGV 空载功率 1、负载功率 2、AGV 数 2），可复刻。
- 可借鉴的评测协议：
  - RPI 指标（式 21）+ ANOVA（95% LSD 均值图）+ 配对 t 检验的统计流程；
  - 用"限时 3 s"与元启发式/精确方法做实时性对齐比较，并同时报告不限时的 CPU 时间；
  - 三套数据划分：验证集（同分布）、公开集（跨来源）、泛化集（2 倍规模，用小子规模模型直接求解）；
  - 多目标比较用 GD/IGD/HV + Pareto 前沿图 + 动作分布可解释性分析。
- 有价值的公式：Cmax（式 1）；三段式能耗模型 PE/IE/TE/TEC（式 2–5）；12 个状态特征（式 6–17）；三种奖励函数（式 18–20，其中加权综合奖励是获得 Pareto 前沿的低成本做法）；RPI（式 21）。

---

## 附：三篇论文的可复用资产速查

| 项 | 卡 1（IJPR 层次化 MARL） | 卡 2（IJPR LSHFSP-AGV） | 卡 3（COR FJSP-AGV 节能） |
|---|---|---|---|
| 代码 | GitHub: ClouDaDaDa/DFJSPT_code（许可未在论文声明） | 未提供 | 未提供 |
| 公开数据 | Ham (2020) HUdata（sdata/edata/rdata/vdata，各 40 实例） | 无（全随机生成，参数可复刻） | MFJS01–10（fastmanufacturingproject.wordpress.com） |
| 基线 | 8 条组合规则、无 IL 的 DRL、CP、GNN_DRL、MWKR | 18 条组合规则、NSGA-II、MDDQN | 16 条组合规则、DDQN、DCGA、IGA、MILP、DQNMA、NSGA-II |
| 指标 | makespan、SGM(s=100)、RPR、STD、运行时间/推理 | 三目标均值、Spacing、HV（Z-score） | RPI、GD/IGD/HV、ANOVA+配对 t 检验 |
| 算法骨架 | 3 层 multi-agent PPO + 8 规则最优者做 IL 引导 | MDDQN + NSGA-II 提供逐步奖励 | 单智能体 DQN + 12 状态特征 + 16 复合规则 |
| 状态数/动作数 | 8 / 6 / 6（三 agent）；动作 n、m、r | 8 个状态特征；18 动作 | 12 个状态特征；16 动作 |

---

# 附：2026-10 批量精读卡片（PLCSP+DRL 全覆盖，27 篇）

> 6 组并行精读：期刊新作 5 + 用户下载 4 + 其他期刊 3 + arXiv 15


## 【组 g1】

### G1 事实卡（5 篇精读）

> 生成日期：2026-09-30　来源 PDF：`D:\research\DeepReinforcementLearningScheduling\references\`
> 原则：只记事实，不做评价。英文原文引用一律逐字照抄（含原文错误/排版噪声），中文为直译。

---

## 卡片 1 — COR2026_ProductionLogistics_AMR.pdf

### 1. 题录
- **标题**：Production-logistics cooperative scheduling in a two-stage assembly flow-shop with deteriorating robotic arm: a problem-specific heuristic
- **作者/机构**：Wenyu Zhang, Zihao Luo, Shuai Zhang*（School of Information Technology and Artificial Intelligence, Zhejiang University of Finance and Economics, Hangzhou 310018, China）
- **期刊/年**：Computers & Operations Research 189 (2026) 107409（Received 3 June 2025; revised 22 January 2026; accepted 22 January 2026; available online 25 January 2026）
- **DOI**：10.1016/j.cor.2026.107409
- **关键词**：Two-stage assembly flow-shop scheduling; Material transportation; Autonomous mobile robots; Deterioration; Energy consumption

### 2. 问题设定
- **问题类型**：两阶段装配流水车间（two-stage assembly flow-shop, TAFSP），可表示为 `DFM→1`——第一阶段为并行专用流水线机器（production workshop），第二阶段为单台装配工位 AW（assembly workshop）。每条专用流水线配一台 picking AMR，构成一个 robotic cell (RC)。仓库 M0i 与输出缓冲 M(|Gi|+1)i 容量无限。
- **资源**：机器（每个 RC 有 |Gi| 台）+ **两类 AMR**（picking AMR：带机械臂，负责仓库→各机器→输出缓冲；fetching AMR：平板，负责输出缓冲→AW 往返）+ 装配工位。**还显式建模了 picking AMR 机械臂的退化（deterioration）**。
- **优化目标**：双目标 —— min Cmax（最后一件产品装配完成时间）与 min TEC（总能耗）。TEC = EC1(机器)+EC2(picking AMR)+EC3(fetching AMR)+EC4(AW)。
  - Eq.(8): `min f1 = TEC = EC1 + EC2 + EC3 + EC4`；Eq.(9): `min f2 = Cmax = max_{j∈J}{FTA_j}`
  - Eq.(1): `EC1 = Σ_i Σ_{g∈Gi\{0,|Gi|+1}} Pig·tig·|J|`（忽略空闲能耗）
  - Eq.(2): `EC^{U&L}_2 = Σ_i Σ_{p∈Pi} (P^{U/L}·tU_ip + P^{U/L}·tL_ip)`
  - Eq.(7): `EC4 = PA·tA·J`
- **关键约束/假设**（原文假设逐条）：
  1. "All machines are in fully functional and available state at the beginning."
  2. "Two or more components are not allowed to be processed simultaneously on the same machine."
  3. "Processing, and assembling cannot be interrupted once starting."
  4. "There is no buffer between any adjacent machines."（相邻机器间无缓冲）
  5. "Transportation cannot be interrupted once starting."
  6. "Charging of the AMRs is not considered."
- **退化建模（本文首创点）**：机械臂装卸时间随生产时间线性增长。
  - Eq.(33): `tU_ip ≥ tU + σ·ST_ip`
  - Eq.(34): `tL_ip ≥ tL + σ·(ST_ip + tU_ip + Σ_j Σ_{g∈Gi\{0}} Σ_{r∈R} d^{(g−1)g}_i / v_r · χ^{gp}_{ij} · α^{pr}_i)`
  - 实验取 `σ = 0.002`（"α is set to 0.002 based on practical experience"，原文符号混用 α/σ）。
  - 论证线性假设的两点理由：机械臂寿命早中期退化近似线性累积（Arts et al., 2025）；建模上线性便于求解（Cheng et al., 2004）。
- **AMR 速度**：离散档位 |R|=3，v={v1,v2,v3}={1,2,3}；`P^AMR_r = ψ1·v_r^2`（功率随速度平方增长，ψ1∈[1,3] 随机整数）；fetching AMR 单程时间 `tr = ψ2·v_r^{−1}`（ψ2∈[5,10]）。
- **活动序列可行性（Crama & van de Klundert, 1997）**：每个 RC 有 |J|·(|Gi|+1) 个活动 Ag_ij；可行序列的两条件被 Eq.(25)(26) 充分保证（Proposition 1 给了证明）。

### 3. 方法
- **算法类型**：**精确法 + 问题特定启发式，无神经网络、无强化学习**。
  - 精确：ε-constraint 法把 Cmax 移入约束（`Cmax ≤ ε`，Eq.(45)），用 GUROBI 解一系列单目标模型构造 Pareto 前沿。ε 步长 Eq.(46)：`λ = (f2^N − f2^I)/(q−1)`，`ε1 = f2^N`，`εh = ε_{h−1} − λ`。
  - 启发式：SRSOA（sequence reconstruction and speed optimization algorithm）；增强版 SRSOA II = SRSOA + Algorithm 4（改善 X1−ESR1）+ Algorithm 5（改善 Xq−ESR2）。
- **两条极端调度规则 ESR（生成 PF 两端点）**：
  - ESR1：从 M2i 向输出缓冲倒推搬运，直到 M1i 被装载，再前向搬运最早未达输出缓冲的零件 → 配最高速度 → 构成 X1（Cmax^low）。
  - ESR2：当前零件的仓库→机器→输出缓冲全程运完才启动下一个零件 → 配最低速度 → 构成 Xq（Cmax^high）。
- **两条结构化性质（含证明）**：
  - Property 2：若 picking AMR 空载到达时机器仍在加工，则在不改变 makespan 前提下降低空载运输速度可降低 TEC。
  - Property 3：若第 j 序零件尚未装载到输出缓冲而 fetching AMR 已完成第 j−1 次运输，则降低第 j−1 次运输的负载/空载速度可在不改变 makespan 前提下降低 TEC。
- **SRSOA 流程（Algorithm 2）**：先取 X1、Xq 两端点；`ω = (Cmax^high − Cmax^low)/(q−1)`；对 h=2..q−1，反复执行"扩展邻域搜索 + 速度优化"，仅当新解 makespan 更大才接受；连续 `Nexe = 50` 次不改进即入库。扩展邻域搜索改编自 Gultekin et al. (2021)（单 RC → 并行 RC）；候选数 `S = |J|·|G1|·|G2|·…·|G|I||`。
- **网络结构**：无（非学习类方法）。
- **状态定义**：无。解表示为 `X = (π, V)`：π 为 |I| 条活动序列，V 为 picking/fetching AMR 的负载与空载运输速度档位（Property 1）。
- **动作空间设计**：**不适用**（非 RL）。若类比：决策 = 活动序列重排 + 速度档位选择。

### 4. 实验
- **实例来源与规模**：因问题为新问题、无公开基准，**随机生成**。实例记法 `|I| × |J| × D`：|I|∈{2,3,4}（RC 数）、|J|∈{4,5,8,10,12}（每类零件数）、D∈{6,12}（相邻机器距离）；共 3×5×2 = **30 个实例，每个独立运行 20 次**。
- **参数**：|Gi| 从 [2,4] 随机整数；tA=30, tU=2, tL=2, tD=3；tig∈[25,50]；|R|=3, v={1,2,3}；ψ2∈[5,10]；Pig∈[4,8]；PA=6, P^{U/L}=2；ψ1∈[1,3]；σ=0.002。
- **基线（列全）**：因"there are no other optimization algorithms available for comparison"，只与 **ε-constraint 法（GUROBI）**、**SRSOA**、**SRSOA II** 三者互比；此外用 GUROBI 单目标最优解 X1−GUROBI / Xq−GUROBI 做模型校验。
- **指标**：HV（超体积）、IGD（反向世代距离）、CPU time；两目标归一化到 [0,1]，参考点 (1.2, 1.2)；PF_true 由所有方法所求解非支配排序得到。Pareto 解个数 q=5（案例研究 q=10）。
- **硬件/软件**：Python；GUROBI 11.0.1，单次 CPU 时间上限 1800 s；3.20 GHz AMD Ryzen 7 CPU，16 GB RAM，Windows 11 64-bit。
- **案例研究**：江西鑫通机械制造有限公司（Jiangxi Xintong Machinery Manufacturing Co., Ltd.）减震支架（shock-absorbing braces）再制造；RC1/RC2/RC3 分别处理方管/垫圈/板；|J|=10 与 |J|=15 两种规模。

### 5. 核心数字（照抄）
- HV：**ε-constraint 法在 30 个实例中 27 个取得最大 HV**；"SRSOA II also produces satisfactory solutions that are well-distributed, with HV values very close to those of the ε−constraint method."
- IGD：ε-法 IGD 不大于 SRSOA/SRSOA II；"SRSOA II achieves the same IGD values as the ε−constraint method for most of the test instances, meaning the solutions obtained by SRSOA II are not dominated by those obtained by the ε−constraint method."
- 代表行（表 4）：`2×12×6`：SRSOA HV 0.4919 / IGD 0.0786 / 124.30 s；SRSOA II 0.7274 / 0.0440 / 178.91 s；ε-法 0.7622 / 0.0324 / **9000.00 s**（超时）。`4×12×12`：SRSOA 0.5567 / 0.0255 / 884.26 s；SRSOA II 0.7260 / 0.0037 / 1013.28 s；ε-法 0.7736 / 0.0031 / 9000.00 s。`2×4×6`：SRSOA II 0.7432 / 0.0000 / 18.32 s 与 ε-法 0.7432 / 0.0000 / 4.12 s 持平。
- 模型校验（表 2）：X1−ESR1 的 Cmax 在全部实例上等于 X1−GUROBI（如 2×4×6：393.48）；Xq−ESR2 在 **30 个中 25 个**实例上 Cmax 与 TEC 与 Xq−GUROBI 完全相同（如 2×4×6：920.24 / 6575.26），在 5 个实例（2×10×6, 2×12×6, 3×12×6, 4×8×6, 4×12×6）被 Xq−GUROBI 支配。
- 退化影响对照（表 3）：把 σ 设为 0 重跑那 5 个实例后，Xq−ESR2 与 Xq−GUROBI 目标值完全一致（如 2×10×6：2401 / 22462；4×12×6：2651 / 42036）。
- 时间复杂度观察："for test instances whose number of RCs and number of components for each type are larger than 2 and 8 respectively, the ε−constraint method requires a significantly longer computation time."（大量实例触到 9000 s 上限）

### 6. 自认局限（原文逐字 + 中文翻译）
1. "In the subsequent experiments, the results show that the deterioration effect weakens the ability of the proposed algorithm to find optimal or near-optimal solutions. Although we added a new algorithm to counteract this impact, it has led to time consumption and failed to completely eliminate the impact. This demonstrates to managers the necessity of timely maintenance."
   → 后续实验表明，退化效应削弱了所提算法找到最优或近优解的能力。尽管我们增加了一个新算法来抵消该影响，但它带来了时间消耗，且未能完全消除这一影响。这向管理者说明及时维护的必要性。
2. "when the number of components reaches a certain quantity, the deterioration makes the ESR2 less effective, thus impacting the algorithm's performance. Even though the Algorithm 5 can alleviate this issue, it increases the run time of the SRSOA, and cannot guarantee the quality of the initial solution every time."
   → 当零件数达到一定规模时，退化使 ESR2 效率下降，从而影响算法性能。虽然 Algorithm 5 可缓解该问题，但它增加了 SRSOA 的运行时间，且不能保证每次初始解的质量。
3. 未来工作（原文）："This study has the potential for further expansion in the future, including: 1) consideration of differences in processing times for different components of the same type; 2) execution of appropriate maintenance activity to address the deterioration of industrial robots; and 3) exploration of more advanced algorithms to yield high-quality solutions that are closer to the ideal Pareto front, even for large-scale TAFSP-AMRs."
   → 1) 考虑同类型不同零件加工时间的差异；2) 通过适当的维护活动应对工业机器人退化；3) 探索更先进算法，使大规模 TAFSP-AMRs 也能得到更接近理想 Pareto 前沿的高质量解。

### 7. 可复用
- **开源代码**：无（原文未提供代码仓库）。
- **公开数据**："All Experiment data used in this study have been uploaded to the Figshare database (https://doi.org/10.6084/m9.figshare.28296515)."（实验数据公开，含 Data availability 与 Supporting information 两处声明）
- **可借鉴的评测协议**：
  - HV 定义 Eq.(47) 与归一化 Eq.(48)；参考点取 (1.2, 1.2)（归一化后最差值 ×1.2，出自 Ciavotta et al. 2013 / Minella et al. 2008）。
  - IGD 定义 Eq.(49)；PF_true 由所有方法解集合非支配排序近似生成（Yao et al., 2024 做法）。
  - **模型正确性验证套路**（源自 He et al., 2022）：① 用 GUROBI 优化单目标 → ② 得解及目标值 → ③ 把该解输入所设计算法 → 若目标值一致则模型正确。
- **可借鉴公式**：退化装卸时间约束 Eq.(33)(34)；四类能耗分解 Eq.(1)–(7)；ε 步长 Eq.(46)；邻域搜索需保持活动序列可行性的规则（Eq.(25)(26) 的充分条件）。

---

## 卡片 2 — CIE2026_WorkerAGV_Hierarchical_DFJSP.pdf

### 1. 题录
- **标题**：Hierarchical collaborative scheduling of workers and AGVs for digital twin-based distributed flexible job shop
- **作者/机构**：Minghai Yuan*, Qi Yu, Liang Zheng, Songwei Lu, Fengque Pei（College of Mechanical and Electrical Engineering, Hohai University, Changzhou, Jiangsu, China）
- **期刊/年**：Computers & Industrial Engineering 214 (2026) 111882（Received 13 May 2025; revised 27 November 2025; accepted 1 February 2026; online 5 February 2026）
- **DOI**：10.1016/j.cie.2026.111882
- **关键词**：Human-AGV collaboration; Distributed flexible job-shop scheduling; Hierarchical multi-agent; Deep Q network; Automated guided vehicle

### 2. 问题设定
- **问题类型**：**分布式柔性作业车间调度**（DFJSP）扩展为 DFJSP-WAC（with Worker-AGV Collaboration）。
- **资源**：多工厂（nf 个）+ 机器 + **AGV** + **工人**（工人既做监控 monitoring，也做运输 transportation，与 AGV 形成异构运输资源竞争）+ 数字孪生（digital twin）环境。决策含：工厂选择、机器选择、监控工人选择、运输资源（工人或 AGV）选择、作业选择。
- **优化目标**：min Cmax（Eq.(4) `Minimize Cmax`；Eq.(5) `Cmax ⩾ SO_ij + pt_ijfk − (1 − Y_ijfk)×L`）。实验只做单目标。
- **关键约束/假设（原文 8 条）**：
  1. "The assembly and unloading times of the jobs are not taken into account, and both workers and AGVs start at the loading/unloading station."
  2. "At any given time, a machine can only process only one job, and each job can only be processed on only one machine."
  3. "At any given time, a worker can monitor only one machine, and once monitoring starts, it cannot be interrupted."
  4. "Once processing or transport of a job has begun, it cannot be interrupted."
  5. "The learning and forgetting of workers are assumed to depend solely on monitoring times at the corresponding machine and interruption durations, with transportation time considered as part of the interruption."
  6. "Once a job is assigned to a factory, it remains in that factory and is not transferred."
  7. "Each factory is equipped with an identical number of workers and AGVs, and this number remains constant throughout the production process."
  8. "The number and layout of machines in each factory are identical, but their processing capabilities vary."
- **数字工人模型（digital worker model）**：
  - 静态能力：Eq.(1) `E = P_phys×W_phys + P_ment×W_ment + P_skill×W_skill`（权重和为 1）。
  - 动态学习-遗忘：Eq.(2) `P_ijk = P2_ijk[ζ + (1−ζ)r^{−α}] / E`，Eq.(3) `ζ = 1 − exp(−βt)`；实验取 α=0.1, β=0.005。
  - 模型复杂度：`O(nf × n²_o × (nf−r + nf−a + nf−m))`；"the problem is highly complex and can only be solved exactly for very small cases."

### 3. 方法
- **算法**：**HMADDQN**（hierarchical multi-agent double deep Q-network）。高层单智能体（全局优化 + 奖励评估）+ 低层 5 个多智能体（感知-执行层，各自基于本地状态决策）。低层智能体的奖励直接取高层智能体输出。
- **网络结构**：5 个低层智能体与高层智能体结构一致 —— **线性层 + 两个残差块（deep residual blocks）+ 全连接层**（用残差跳跃连接缓解梯度消失/过拟合）。隐层维度 64。非 GNN、非注意力。
- **状态定义（S = [S11, S12, S21, S22, S23, S31, S32, S41, S42]，共 9 维）**：
  - 机器：[S11,S12] = 各工厂机器利用率的平均偏差（mean − min）与工厂间标准差，Eq.(23)–(26)；
  - 作业：[S21,S22,S23] = 订单完成率、完成率的平均偏差、完成率标准差，Eq.(27)–(30)；
  - AGV：[S31,S32] = AGV 利用率平均偏差与标准差，Eq.(31)–(34)；
  - 工人：[S41,S42] = 工人利用率平均偏差与标准差，Eq.(35)–(38)。
  - 设计要点：用"均值−最小值偏差 + 跨厂标准差"使**输入维度不随工厂数变化**。
- **动作空间设计**：**选/加权规则（dispatching rules），不是选实体**。低层 5 个智能体各输出一条调度规则（表 4）：
  - Agent1 作业：SPTJ（下道工序加工时间最短）、FOPNR（剩余工序最少）、MOPNR（剩余工序最多）、EIJ（最早可用）；
  - Agent2 机器：LUM（利用率最低）、EIM（最早可用）、SPTM（加工时间最短）；
  - Agent3 工人：EIP（最早可用）、STTP（距加工机器最近）、SMTW（监控时间最短）；
  - Agent4 运输：EAT（最早可用）、LUT（利用率最低）；
  - Agent5 工厂：EAF（最早可用）、MaxJF（作业最多）、MinJF（作业最少）。
  - "The joint decision-making of five agents results in **216 possible actions** at each decision point and 216^N possible scheduling policies."
  - 探索：ε-greedy，Eq.(39)；ε 衰减 Eq.(40)：`ε = 0.995^e (ε≥0.01) else 0.01`（原文写法 `0.995e`/`0.995^e` 排版不清）。
- **奖励函数**：
  - 高层局部奖励 Eq.(41)：`rt = max_{i∈J}(CTJ_i(s_t)) − max_{i∈J}(CTJ_i(s_{t+1}))`（相邻状态最大完工时间之差，缓解稀疏奖励）；
  - 低层智能体直接把高层输出当作奖励。原文："The low-level agents adopt the output of the high-level agent as reward."

### 4. 实验
- **实例来源与规模（4 组实验）**：
  1. 小规模随机算例，与 **Gurobi** 对比（Case1–Case7）；
  2. 随机生成的变规模分布式车间（Instance1–15，规模 J×F×M×A×W，从 10×2×5×2×5 到 300×7×6×8×7，作业数 10→300、工厂 2→7）；
  3. 公开 FJSP-AGV 数据集（Deroussi & Norre, 2010）标准算例 fjsp1–fjsp10；
  4. 真实分布式车间生产数据集（Li et al., 2022，20 个不同规模案例），下载地址 `https://cuglirui.github.io/Dataset/DHFJSP.rar`。
- **基线（列全）**：
  - 组合调度规则：**CDR1** (MinJF+MOPNR+SPTM+EIV+EIP)、**CDR2** (MinJF+MOPNR+SPTM+LUV+EIP)、**CDR3** (MinJF+EIJ+LUM+EIV+EIP)、**CDR4** (MinJF+EIJ+EIM+EIV+EIP)；
  - **DDQN**、**DDQN-WAC**、**HMADDQN without WAC**（消融）、**Gurobi**；
  - 标准算例上另比：DQN (Luo, 2020)、PPO (Lei et al., 2022)、DDQN (Yuan et al., 2025)、HDPSO (Chen et al., 2022)、HGNN (Wang et al., 2024)、PPO (Zhao et al., 2025)、DCGA (Han et al., 2024)。
- **指标**：Cmax、Gap(%) (相对 HMA DDQN 的百分差)、决策时间(s)、Wilcoxon 符号秩检验 p 值、收敛曲线、工人任务分布热力图（负载均衡）。用 Taguchi 正交实验 + "smaller-the-better" S/N 比 Eq.(42) 调参。
- **硬件**：Python 3.7.0 + PyTorch；Intel Core i7-13790F，DDR5 6000 MHz 32 GB，NVIDIA GeForce RTX 4070。
- **超参**：隐层 64；遗忘因子 0.005；学习因子 0.1；学习率 5e-4；γ=0.96；batch 64；低/高层经验回放容量 10000；目标网络每 200 步更新；τ=0.18。训练从 10 作业 2 工厂渐进到 300 作业 7 工厂。

### 5. 核心数字（照抄）
- 摘要/结论："yielding more than a **35% reduction in average makespan** compared with non-human-AGV collaborative scheduling approaches under constrained transportation resources"；表 11 汇总 **Average = 35.30%**（HMADDQN without WAC gap 35.30%，HMA DDQN 为 0.00% 基准）。
- 实验 2 平均 Gap（表 11 Average 行）：CDR1 16.81、CDR2 42.69、CDR3 22.51、CDR4 10.17、DDQN−WAC 2.87、HMADDQN without WAC 35.30、HMA DDQN 0.00。
- 小规模（表 7）：HMADDQN 在 Case1/2/5/6/7 与 Gurobi 同解（54/53/65/132/86）；Case3 86 vs Gurobi 85；Case4 76 vs Gurobi 70（DDQN 为 88）。
- 实验 2 示例（表 9）：Instance15（300×7×6×8×7）HMADDQN 762.32、HMADDQN without WAC 762.44、DDQN-WAC 771.30、CDR4 787.47、CDR1 966.54、决策时间 17.78 s；Instance6 HMADDQN 481.45 vs DDQN-WAC 521.88。
- 标准算例（表 10）：HMADDQN 在多数算例达 Gurobi 已知最优（如 fjsp1 = 134 = Gurobi；fjsp2 = 114；fjsp5 = 94；fjsp7 = 108；fjsp9 = 144；fjsp10 = 174）；fjsp8 HMADDQN 184、Gurobi 178；fjsp3 128 vs Gurobi 120。
- Wilcoxon（表 12）：小规模 10J2F2A p=0.9441、40J3F2A p=0.1081（>0.05 不显著）；大规模 50J5F2A p=1.6×10⁻⁵、100J6F2A p=1.9×10⁻⁶、150J6F3A p=9.5×10⁻⁸。
- AGV 数量梯度实验：AGV 从 3 增至 8 时 makespan 下降变得边际化，说明人-AGV 协同可缓解运输任务拥堵。

### 6. 自认局限（原文逐字 + 中文翻译）
> 本文**没有独立的 Limitations 章节**，以下为最接近的原文表述（结论的 future research 段与数据可用性声明）。

1. "In future research, we aim to broaden the scope of the study to incorporate more complex factors that affect worker efficiency and further explore the collaborative mechanisms between workers and AGVs. Particular emphasis will be placed on achieving cross-workshop collaboration between workers and AGVs, exploring how to optimize resource allocation between personnel and equipment across different workshops, improve information flow efficiency, and ensure seamless collaboration between workers and AGVs to achieve human-AGV integration. Furthermore, this study will consider the overall workload of workers as a key optimization objective, aiming to provide more comprehensive and efficient decision support for the advancement of Industry 5.0."
   → 未来研究将纳入更多影响工人效率的复杂因素，并进一步探索工人与 AGV 的协同机制；重点是跨车间的人-AGV 协同、优化不同车间之间人员与设备的资源配置、提升信息流效率、实现人机无缝协同。此外将把工人总体工作量作为关键优化目标，为工业 5.0 提供更全面高效的决策支持。
2. "All the data used in this study are available from the corresponding author upon reasonable request and with the permission of all parties involved."（Data availability）
   → 本研究使用的全部数据需在合理请求并经相关各方许可后向通讯作者索取（即**未公开数据集**）。

### 7. 可复用
- **开源代码（URL）**：
  - MILP 模型代码：`https://github.com/fyqiqi/DFJSP-WAC-MILP-model`（"The specific MILP code has been open-sourced at https://github.com/fyqiqi/DFJSP-WAC-MILP-model."）
  - FJSP-AGV 的 Gurobi 实现：`https://github.com/fyqiqi/MILP-for-FJSP-AGV`
  - 原文未标注许可证（license）字样。
- **公开数据集**：
  - 分布式异构柔性作业车间数据集：`https://cuglirui.github.io/Dataset/DHFJSP.rar`（Li et al., 2022，20 个不同规模分布式车间案例）
  - FJSP-AGV 标准算例：Deroussi & Norre (2010)（fjsp1–fjsp10，用于表 10 对比）
- **可借鉴的评测协议/公式**：
  - 用"均值−最小值偏差 + 跨厂标准差"构造**与工厂数无关的定维状态表示**（Eq.(25)(26)(29)(30)(33)(34)(37)(38)）——可直接迁移到任何可变规模的多单元调度场景。
  - 工人学习-遗忘监控时间模型 Eq.(2)(3)，数字工人效率 Eq.(1)。
  - 分层用 DDQN（高层聚合 + 低层 5 智能体）与 ε 衰减 Eq.(40)、局部奖励 Eq.(41)。
  - 统计检验用 Wilcoxon 符号秩检验，报 mean±std 与 p 值（表 12）；调参用 Taguchi 正交表 + S/N 比 Eq.(42)。

---

## 卡片 3 — EJOR2026_JointScheduling_Production_MaterialHandling.pdf

### 1. 题录
- **标题**：A joint scheduling approach for production and material handling under customized manufacturing paradigm
- **作者/机构**：Yuxuan Dong ᵃ˒ᵇ, Shan Jiang ᶜ, Liping Zhou ᵃ˒ᵇ, Zhibin Jiang ᵃ˒ᵇ*, Yugang Yu ᵈ
  - a Sino-US Global Logistics Institute, Antai College of Economics and Management, Shanghai Jiao Tong University, 1954 Huashan Road, Shanghai 200030, China
  - b Data-Driven Management Decision Making Lab, Shanghai Jiao Tong University
  - c The Global Institute of Future Technology (GIFT), Shanghai Jiao Tong University
  - d Anhui Province Key Laboratory of Contemporary Logistics and Supply Chain & International Institute of Finance, School of Management, University of Science and Technology of China
- **期刊/年**：European Journal of Operational Research 332 (2026) 730–747（Discrete Optimization 栏目；Received 23 February 2025; Accepted 13 January 2026; Available online 15 January 2026）
- **DOI**：10.1016/j.ejor.2026.01.022
- **关键词**：Scheduling; Customized manufacturing; Flexible assembly line; Material handling robot; Lagrangian relaxation

### 2. 问题设定
- **问题类型**：**装配流水线 + 物料搬运机器人（MHR）联合调度**（JFMS, Joint FAL and MHR Scheduling）。生产子系统 = 柔性装配线 FAL（M 台串行工位，传送带输送 WIP）；物料子系统 = V 台电池驱动 MHR + 一个充电站。**不是** job shop，也不是经典 flow shop；MHR 按 point-to-point（P2P）投料，等价于"并行机 + 后续单机阶段"的混合流水车间变体（但前置关系由订单排序内生决定）。
- **资源**：装配工位（机械臂抓取+装配）、MHR（携带能力 C、电池容量 B）、充电站、传送带、仓库（投料点）。
- **优化目标**：min `Σ_{a∈A} t^f_a`（所有订单总完成时间；等价于最小平均流程时间，反映系统响应性）。
- **三类作业**：(1) feeding job（MHR 送一批零件到工位）；(2) assembly job（工位装配该批零件，三元组 (a,m,k)）；(3) charging job（MHR 到充电站充电，时长可变）。
- **关键约束/假设**：MHR 携带容量 C 与电池容量 B 有限；投料批次量 `K_{a,m} = ⌈D_a ω_{a,m}/C⌉`，单次最大运量 `Q_{a,m} = ⌊C/ω_{a,m}⌋`；P2P 投料（禁止混装，"mixed component transportation is disallowed"）；FAL 已平衡，节拍 T_CT（工位加工时长相同，且供料充足时每 T_CT 产出一件）；MHR 同质；**充电桩数量 = MHR 数量**；零件库存无限；电池消耗只取决于行驶时间（消耗率 δ），充电速率 Δ；订单初始全部到达。

### 3. 方法
- **算法类型**：**精确建模 + 拉格朗日松弛（Lagrangian relaxation）分解算法**，无神经网络、无 RL。
  - 先证明结构性质再做等价简化：**Proposition 1**（最优投料量闭式解：`q*_{a,m,k} = Q_{a,m}` for 1≤k≤K−1，`q*_{a,m,K} = D_a − (K−1)Q_{a,m}`）→ 投料量不再是决策变量；
  - **Lemma 1**：充电站最优位置 = 投料点（component loading point）；
  - **Proposition 2**（最优充电策略）：若 `b_{i_s} < δ(T^{VT}_{i_s,c} + 2T^{VT}_{c,i_{s+1}})` 则在两个 feeding job 之间充电，充电量 `r = δ(T^{VT}_{i_s,c} + 2T^{VT}_{c,i_{s+1}}) − b_{i_s}`，否则不充电。据此定义"**integrated feeding job**"，其加工时间 = `2T^{VT}_{c,i} + 2δT^{VT}_{c,i}/Δ`（与执行顺序无关）→ 物料子问题化为**并行机调度 PMSP**（原为带序相关准备时间的 PMSP-SDST 与 VRP 框架）。
- **模型与分解**：
  - [P] 目标 Eq.(22) `Z_P = min Σ_a t^f_a`，约束含生产约束 (1~5, 9~14)、物料约束 (15~21) 与**唯一耦合约束 (23)** `t^s_{a,m,k} ≥ t^d_{I(a,m,k)}`（装配开工不早于零件送达）；
  - **Proposition 3**：问题 [P] 是强 NP-hard（由 1|r_a|ΣC_a 归约）；
  - 对约束 (23) 施加乘子 λ 松弛得 [LR] Eq.(24)，分解为 **[LR-P]（生产子问题）** 与 **[LR-MH]（物料子问题）**；[LR-P] 由 **Proposition 4** 证明为 SWPT 规则最优（权重 `1 − Σ_m Σ_k λ_{a,m,k}`，加工时间 `(D_a+M−1)T_CT + T^{LC}_{1,M}`），多项式时间可解；[LR-MH] 为 PMSP（NP-hard）。
- **算法（JFMS，Algorithm 3）**：三分量 = ① 下界生成与提升（Proposition 4 解 [LR-P]，Proposition 5(i) 用 PMSP 下界 `T_LB(V, N_{j,1},…,N_{j,M}) − max_i{T^{VT}_{o,i}}` 抬升）；② 上界改进（Algorithm 1 可行解构造 + **Algorithm 2 局部搜索**：swap / insertion 邻域，启发自 Proposition 5(ii)"生产序列优先级高的订单，其 feeding job 在 MHR 排程中也倾向高优先级"）；③ 乘子更新（次梯度法 `λ ← max{λ + βg(λ), 0}`，步长 `β = (Z_P − Z_LD)/‖g(λ)‖²`）。大规模场景改用 **surrogate subgradient method**（不需要完全最优求解子问题）。
- **Proposition 6**：两个规则启发式即为最优的特例 —— (i) `T_CT ≥ T̄_CT`（节拍足够大）；(ii) 单工位且 ω 非降。
- **网络结构 / 状态定义 / 动作空间**：**不适用**（非学习方法）。

### 4. 实验
- **实例来源与规模**：**随机生成 + 真实车间尺寸**。车间 200 m × 60 m；投料点/充电站位于 (0,0)，两个工位位于 (200/3,30) 与 (400/3,30)；传送带速度 0.5 m/s；MHR 参数取真实数据（MiR1000 数据表：载重 1000 kg、电池 2 kWh、1 m/s 下理论续航 8 h；实用 SoC 10%–90%，有效容量 1.6 kWh ≈ 6.4 h = 23,040 s；充电速率 Δ=1.6/7200 kWh/s，2 h 充满）。
  - 20 个场景（表 2）：A ∈ {2,3,10,20} 订单数，M=2 或 8 工位，V ∈ {2,8,14,20} MHR 数，T_CT ∈ {3,8,13} 秒，|F| 为 feeding job 数区间；**每场景 100 个随机实例**；场景 1–6 小规模、7–12 中规模、13–20 大规模（T_CT=3）。
  - 随机：每订单产品数 D_a ~ U[40,60]；零件重量 ω ~ U[10,40]。
- **基线（列全，表 3）**：**CP**（CPLEX 12.9 解 [P]，15 min 时限）、**SD**（Sequential Decision，两阶段：先定生产顺序，再用 CPLEX 优化物料搬运）、**RULE**（SWPT + FCFF 规则启发式）、**LBL**（提升下界 Z_LD）、**UBNL**（Upper bound without local search）、**LBNL**（Lower bound without lift）、**LNS**（Emde & Schneider 2018 的邻域搜索元启发式）。
- **指标**：Best/Opt rate（100 个实例中取得最优/最佳解的比例）、mean gap from Best/Opt (%)、mean computational time (s)；另有 gap 定义 `G^{j'}_j = (Z_j − Z_{j'})/Z_j`。
- **硬件/软件**：Java 实现；Intel Xeon W-2255 @ 3.70 GHz；**15 分钟计算时限**；JFMS 终止条件 Z_LD 连续 10 次不改进；局部搜索 2000 次迭代；每实例跑 5 次。

### 5. 核心数字（照抄）
- 小规模（场景 1–6）："the JFMS algorithm can reach the same solution quality as the CPLEX solver (CP) **within one second**"（场景 1 的 JFMS Best/Opt rate 100%，平均 gap 0，平均时间 0.86 s；CP 平均 12.22 s）。
- 中规模（场景 7–12）："JFMS, in contrast, consistently achieves the best solutions in **at least 97%** of the instances"（场景 7/10/11 达 100%，9/12 达 97%，8 达 99%）；CPLEX 在 15 min 内无法获得高质量解，其 gap 甚至劣于 LNS 与 RULE（如场景 10：CP gap 44.6% vs SD 39.39%）。
- 大规模（场景 13–20）：JFMS 在全部场景 Best/Opt rate **100%**，平均 gap 0.30%–0.73%；LNS 在场景 13–20 的 Best/Opt rate 为 0。
- 下界紧度：`G^{JFMS}_{LBL}` "less than 10% for most instances"，场景 7–12 "below 5% in most instances"（摘要：gaps typically below 10% in most instances）。
- 联合 vs 顺序决策：小规模 CP 优于 SD 最高达 **25%**（场景 1，Fig.5a）；规模扩大后 **SD 在 80% 以上的实例中优于 CP**（场景 7–12）。
- 结构性质价值（表 5）：不用 Proposition 2 时，**即使极小规模（场景 1–3）CPLEX 也无法在 15 min 时限内求出最优解**（场景 1 仅 44/100 实例精确求解，均值 550.74 s）；用 Proposition 2 后 100/100 精确求解，均值 1.59 s；再用 Proposition 1 后进一步降至 **0.16 s**（场景 2：0.65→0.14 s；场景 3：0.16→0.03 s）。
- Proposition 5 价值：下界提升（Algorithm 2 第 14 步）有助于抬高下界；局部搜索最多把 Algorithm 1 构造的解改进 **up to 20% in certain extreme instances**；MHR 越少，上界改进越大。

### 6. 自认局限（原文逐字 + 中文翻译）
> 本文**无独立 Limitations 章节**，以下为最接近的原文表述。

1. "Additionally, in Scenarios 13∼20, the JFMS algorithm is extended from the subgradient method to the surrogate subgradient method framework. However, **the latter cannot guarantee the validity of the lower bound when subproblems are not solved exactly.** Therefore, the G^{JFMS}_{LBL} metric only considers the first 12 small-to-medium-scale scenarios."
   → 在场景 13~20 中，JFMS 从次梯度法扩展为代理次梯度法框架；但后者在子问题未精确求解时无法保证下界的有效性。因此 G^{JFMS}_{LBL} 指标只考虑前 12 个小/中规模场景。
2. "There are several potential avenues to explore in future studies. These include investigating methodologies for determining the optimal number of MHRs and designing efficient layouts for the shop floor. Additionally, the application of the joint scheduling approach to different types of smart manufacturing systems could be explored. Further research can also focus on considering dynamic environments and real-time scheduling to enhance the adaptability of the model. **Relaxing the assumption that mixed component transportation is disallowed** is also an interesting research direction. Alternative objective functions, such as makespan, can also be investigated. Furthermore, the exploration of multi-objective optimization methods to strike a balance between various performance metrics is worth investigating."
   → 未来可研究方向：最优 MHR 数量的确定方法与车间布局设计；将联合调度推广到其他类型智能制造系统；考虑动态环境与实时调度以提升适应性；**放松"禁止混装"假设**；改用其他目标（如 makespan）；多目标优化方法。
3. 建模简化假设（原文已自陈其边界）：P2P 投料 "implicitly entails the assumption that mixed component transportation is disallowed"；"In theory, allowing mixed transportation would enlarge the feasible solution space and yield schedules that are at least as good as those under the P2P strategy."

### 7. 可复用
- **开源代码**：`https://github.com/everdyx/numerical-results-for-JFMS`（脚注："The code can be found at https://github.com/everdyx/numerical-results-for-JFMS"）。原文未标注许可证。
- **公开数据集**：无自有公开数据集；MHR 参数来自公开产品数据表 `https://www.wiredworkers.io/wp-content/uploads/2019/11/mir1000-datasheet.pdf`（MiR1000）。
- **可借鉴的评测协议/公式**：
  - **基线设计范式（很值得复用）**：联合求解器 CP / 顺序两阶段 SD / 规则启发式 RULE / 下界 LBL / 无局部搜索上界 UBNL / 无提升下界 LBNL / 外部元启发式 LNS —— 覆盖"解质量 + 界紧度 + 消融"三类证据。
  - 三指标报告法：Best/Opt rate、mean gap from Best/Opt、mean computational time。
  - 消融方式：逐一去掉 Proposition 1/2 重解 MIP，用"时限内精确求解实例数 + 平均求解时间"量化结构性质的价值（表 5）。
  - 可移植公式：最优投料量闭式解（Proposition 1）；充电站选址引理（Lemma 1）；最优充电策略（Proposition 2）；integrated feeding job 处理时间 `2T^{VT}_{c,i} + 2δT^{VT}_{c,i}/Δ`；SWPT 权重 `1 − Σ_m Σ_k λ_{a,m,k}`（Proposition 4）；PMSP 抬升下界（Proposition 5(i)）。

---

## 卡片 4 — ESWA2026_HCHGNN_PPO.pdf

### 1. 题录
- **标题**：Hub-centric heterogeneous graph reinforcement learning for integrated automated guided vehicle dispatching and unrelated parallel machine scheduling with finite-buffer constraints
- **作者/机构**：Lei Yue ᵃ, Yuhao Zhu ᵃ, Dan Luo ᵇ, Leilei Meng ᶜ, Linshan Ding ᵃ*
  - a School of Mechanical and Electrical Engineering, Guangzhou University, Guangzhou 510006, China
  - b School of Business, Dalian University of Technology, Dalian 116024, China
  - c School of Computer Science, Liaocheng University, Liaocheng, Shandong 252000, China
- **期刊/年**：Expert Systems With Applications 334 (**2027**) 134345；Available online 14 September 2026（Received 3 April 2026; revised 16 July 2026; accepted 7 September 2026）——注意卷年标注为 2027，DOI 年份为 2026。
- **DOI**：10.1016/j.eswa.2026.134345
- **关键词**：Integrated scheduling; Heterogeneous graph neural networks; Deep reinforcement learning; Unrelated parallel machine scheduling; Finite-buffer constraints

### 2. 问题设定
- **问题类型**：**非相关并行机调度（unrelated parallel machines, UPM）+ AGV 派送 + 有限缓冲**的一体化事件驱动调度。不是 job shop、也不是 flow shop。
- **资源**：AGV 车队（NA 台，速度 ν_j，一次载一件）、非相关并行机（NM 台，加工时间 `p_kτ` 同时依赖机器与工件类型）、每台机器的等待区（FIFO）与**有限输出缓冲（容量 C_k=2）**、位于 p_H 的集中仓库、位于 p_D 的统一派送点。
- **优化目标**：min makespan = 最后一个工件送达仓库的完成时间（Eq.(1) `min C_max`, Eq.(2) `E^comp_w ≤ C_max`）。
- **关键约束/假设（A1–A9 逐条）**：
  - (A1) 所有订单在时刻 0 于统一派送点释放；(A2) 每台 AGV 一次最多载一件；(A3) AGV 行驶时间确定，由欧氏距离与恒速计算；(A4) AGV 之间行驶不互相干扰；(A5) 每台机器一次加工一件，等待区 FIFO；(A6) 加工时间确定，依赖机器与工件类型；(A7) 同一机器上相邻工件类型不同时产生类型切换准备时间 s_k；(A8) 仓库容量无限；(A9) 不允许抢占。
  - 有限缓冲阻塞：机器无法把完工工件释放到已满的输出缓冲（"blocked-release mechanism"），是死锁风险的主要来源；约束 (12) `b_k(t) ≤ C_k`，约束 (13) `b_k(S^proc_w) < C_k`。
- **术语区分**：dispatching = AGV 的实时事件驱动指派；scheduling = 工件到并行机的分配与加工排序。

### 3. 方法
- **算法类型**：**单智能体 DRL — PPO**（clip + GAE），配**去耦 actor–critic**（策略网络与价值网络**各自独立的 HCHGNN 编码器与优化器**）。事件驱动离散事件仿真器提供 MDP 动力学。
- **网络结构（HCHGNN 编码器）**，三个组件：
  1. **可学习特征标准化** Eq.(29)：`X̂ = γ ⊙ (X−μ)/(σ+ε) + β`；
  2. **类型特定投影** Eq.(30)–(33)：`h^{O,(0)}_i = W^O x̂^O_i` 等四类投影矩阵；
  3. **多层异构图注意力 + 类型池化**：注意力 Eq.(34) `e^{(l)}_{ij} = LeakyReLU(a_s^{⊤}W_s h_i + a_d^{⊤}W_d h_j + a_e^{⊤}W_e ê_ij)`；H 头拼接；残差 + LayerNorm 更新 Eq.(35)–(38)；层后按类型平均池化，`g = MLP(h̄_O ‖ h̄_A ‖ h̄_W ‖ h̄_B)` 作为价值网络全局表示。
  - **原文以符号 L（层数）与 H（头数）表示，正文与超参数表（表 6）均未给出数值。**
- **图结构（hub-centric heterogeneous graph）**：节点 4 类 —— 订单 O、AGV A、等待区 W、缓冲 B；边 3 类 —— E^{OA}（订单–AGV）、E^{AW}（AGV–等待区）、E^{AB}（AGV–缓冲）；**不设等待区–缓冲直接边**，因为二者交互由 AGV 中介。AGV 作为 hub 节点。
- **节点/边特征**：
  - 订单 Eq.(21)：`x^O_i = [onehot(τ_i,K) ‖ n^rem_i]`；
  - AGV Eq.(22)：`x^A_j = [onehot(σ_j,5) ‖ p^x_j ‖ p^y_j ‖ ν_j ‖ t^idle_j ‖ onehot(τ^c_j,K)]`；
  - 等待区 Eq.(23)：`x^W_k = [p^x_k ‖ p^y_k ‖ q_k ‖ I(q_k>0)]`；
  - 缓冲 Eq.(24)：`x^B_k = [p^x_k ‖ p^y_k ‖ b_k ‖ C_k ‖ I(b_k>0) ‖ I(b_k<C_k)]`；
  - 边 Eq.(25)–(28)：订单–AGV 用派送点距离；AGV–等待区用距离 + 潜在换型准备 `s_jk`；AGV–缓冲只用距离。
- **动作空间设计**：**选实体（AGV + 订单 + 机器）或（AGV + 机器）**，定维索引 + 二值可行性掩码。
  - `A = A_dispatch ∪ A_pickup`；`A_dispatch = {(j,i,k)}`，`A_pickup = {(j,k)}`；`|A| = N_A(N_O N_M + N_M)`；
  - 索引映射 `idx_dis(j,i,k) = (j−1)N_O N_M + (i−1)N_M + k`，`idx_pick(j,k) = N_A N_O N_M + (j−1)N_M + k`；
  - **优先级掩码 4 条规则**：① 非空闲 AGV 的动作全屏蔽；② 派送动作要求 `n^rem_i > 0` 且 `b_k < C_k`；③ 取件动作要求 `b_k > 0`；④ **只要有任一缓冲有待取完工工件，屏蔽所有派送动作**（缓冲清空优先，防阻塞/死锁传播）。
- **奖励函数（含公式）**：
  - 增量奖励：Eq.(39) `Φ_t = min( min_j max(t^idle_j, t), min_k max(t^idle_k, t) )`；`r_t = Φ_t − Φ_{t+1}`；
  - 望远镜求和性质：`Σ_{t=0}^{T−1} r_t = Φ_0 − Φ_T`，无折扣时最大化累积奖励等价于最小化终端 makespan；γ<1 时作为"makespan-oriented shaping signal"。
  - PPO：GAE Eq.(41) `Â_t = Σ_{l=0}^{T−t−1}(γλ)^l δ_{t+l}`，`δ_t = r_t + γV_φ(s_{t+1}) − V_φ(s_t)`；critic 用 MSE；策略用 clipped surrogate + 熵项。

### 4. 实验
- **实例来源与规模**：**自生成基准 + 3 个规模档**（表 3/5）。小：NM{6,9}, NA{5,9}, K{2,3}, 地图 60×100；中：NM{10,13}, NA{9,13}, K{4,6}, 80×100；大：NM{16,19}, NA{15,19}, K{7,10}, 100×100。每档 8 个实例共 **24 个算例**，工件总数 16→684。
  - 生成参数（表 4）：派送点 (0,25)，仓库 (0,75)（原文写作 "(0, 75)"），缓冲容量固定 2，准备时间 {0,150}，加工时间 {90,360}，AGV 速度 {1,1.5} m/s，订单数 {3,15}，每订单工件数 {5,40}；订单全部在 0 时刻释放（无随机到达、无在线插入）。
- **基线（列全）**：
  - 优先级派单规则 5 个：**FIFO, SPT, Nearest（最近派送）, SST（最短准备时间）, SQ（最短队列）**；
  - HGNN 增强的 DRL 3 个（**同一图构建、同一特征、同一掩码**，仅换 RL 算法）：**HGNN-A2C, HGNN-SAC, HGNN-DQN**；
  - 消融 3 个：**HCHGNN-PPO-NC**（去 hub-centric 拓扑）、**-NH**（去节点/边异质性）、**-NG**（用 MLP 替换图编码器）。
- **指标**：makespan C_max (s)、决策/推理时间 (s)、相对 gap Eq.(42) `Δgap = (1/|C|)Σ_c (C^variant_max(c) − C^full_max(c))/C^full_max(c) × 100%`、Wilcoxon 符号秩检验、Gantt 图。
- **硬件**：Intel Core i9-14900K (3.20 GHz)，128 GB RAM，NVIDIA GeForce RTX 4090，Windows 11，Python 3.12。
- **训练超参（表 6）**：1500 episodes；lr 1e-4；γ=0.99；GAE λ=0.95；clip 0.2；熵系数 0.001；价值损失系数 0.5；梯度裁剪 0.5；PPO 更新 epoch 4；minibatch 32；Adam。

### 5. 核心数字（照抄）
- 与规则对比：**HCHGNN-PPO 在全部 24 个实例上取得最优 C_max**。Case 1：1175.47 s vs FIFO 5908.10、SPT 3335.07、Nearest 4912.44、SST 3864.19、SQ 2519.83。Case 24：16335.88 vs FIFO 230166.43、SPT 144834.28、Nearest 205870.54、SST 191182.52、SQ 34315.58。
- 与 HGNN-DRL 对比：**23/24 个实例最优**；仅 Case 15 略输给 HGNN-A2C（5620.19 vs 5464.88）；HGNN-SAC 与 HGNN-DQN 在所有实例上均更差（如 Case 22：15245.90 vs SAC 19912.10、DQN 31151.58）。
- 消融（表 8）：**-NC** Better/Tie/Worse = 2/1/21，平均 gap **6.06%**（小 4.95%、中 3.57%、大 9.67%）；**-NH** 0/0/24，平均 gap **9.39%**（7.81/8.60/11.75）；**-NG** 0/0/24，平均 gap **24.68%**（16.12/21.15/**36.76**）。
- Wilcoxon（表 11）：vs -NC R⁺=278/R⁻=29, p=1.05×10⁻⁴；vs -NH 300/0, p=1.19×10⁻⁷；vs -NG 300/0, p=1.19×10⁻⁷；vs HGNN-A2C 285/15, p=2.98×10⁻⁶；vs HGNN-SAC、HGNN-DQN 均 300/0, p=1.19×10⁻⁷。
- 训练行为：makespan 在约前 300–400 episodes 明显下降，随后在窄区间稳定。

### 6. 自认局限（原文逐字 + 中文翻译）
1. "Nevertheless, the reward does not constitute a formal proof of global optimality, and it may still be imperfect in highly stochastic or adversarial environments. Its role is to reduce reward sparsity and provide makespan-consistent training guidance under the deterministic event-driven assumptions of this study."
   → 然而该奖励并不构成全局最优性的形式化证明，在高随机或对抗性环境中仍可能不完善。其作用是在本文确定性事件驱动假设下降低奖励稀疏性并提供与 makespan 一致的训练引导。
2. "Future research may extend this framework to more realistic production environments that incorporate stochastic processing times, dynamic order arrivals, multi-AGV collaboration, and online rescheduling. The integrated scheduling problem can also be extended to the multi-objective decision-making with criteria, such as energy consumption, tardiness, and throughput."
   → 未来研究可将该框架推广到更真实的生产环境，纳入随机加工时间、动态订单到达、多 AGV 协同与在线重调度；也可将该一体化调度问题扩展为考虑能耗、拖期、吞吐量等准则的多目标决策。
3. 当前基准生成的自陈边界："stochastic order arrivals and online order insertion are not included in the current benchmark generation."（当前基准生成不含随机订单到达与在线插入）
4. 计算代价自陈："Although the proposed framework introduces additional graph-encoding overhead, it enables structured state abstraction and coordinated policy learning in a problem setting where purely flat representations are insufficient."

### 7. 可复用
- **开源代码**："The data and code are available at **https://github.com/Linshan-Ding/HCHGNN_PPO_UPMSP_AGV**"（原文未标注许可证）。
- **公开数据集**：无（全部为自生成基准，生成参数见 Table 4，可完整复现）。
- **可借鉴的评测协议/公式**：
  - 三档规模 × 每档 8 实例 = 24 算例的**可复现生成协议**（表 3 规模范围 + 表 4 生成参数 + 表 5 逐算例配置）。
  - 消融三维度设计：拓扑（-NC）、异质性（-NH）、图编码器本身（-NG），并用 Better/Tie/Worse + 分规模 gap 汇总（表 8）。
  - 图编码器公平性对照：**同一图构建/特征/掩码，仅替换 RL 算法**（HGNN-A2C / SAC / DQN）——用于论证增益来自 PPO 而非编码器。
  - 死锁感知的优先级掩码规则（缓冲有待取件时禁用派送）可直接迁移。
  - 增量奖励 `r_t = Φ_t − Φ_{t+1}`（最小下一空闲时间的推进量），其无折扣望远镜等价于终端 makespan——可移植到任何事件驱动调度 MDP。
  - 复杂度报告：动作打分 `O(N_A(N_O N_M + N_M)d_e)`；HCHGNN 编码 `O(L(|E|d_e + |N|d_e²))`。

---

## 卡片 5 — SWEVO2026_HGA_MPPO_AGV_FJSP.pdf

### 1. 题录
- **标题**：HGA-MPPO: A unified heterogeneous graph attention and multi-policy PPO framework for AGV-assisted flexible job shop scheduling
- **作者/机构**：Xiaoyu Yang ᵃ, Yuyan Han ᵃ˒ᵇ*, Yuting Wang ᵃ, Yuhang Wang ᶜ, Leilei Meng ᵃ
  - a School of Computer Science, Liaocheng University, Liaocheng 252059, China
  - b The Research Center of System Science, Liaocheng University, Liaocheng 252059, China
  - c National Frontiers Science Center for Industrial Intelligence and Systems Optimization, Northeastern University, Shenyang 110819, China
- **期刊/年**：Swarm and Evolutionary Computation 102 (2026) 102331（Received 14 October 2025; revised 16 January 2026; accepted 10 February 2026; online 18 February 2026）
- **DOI**：10.1016/j.swevo.2026.102331
- **关键词**：Flexible job shop scheduling; Automated guided vehicle; Graph neural network; Deep reinforcement learning; Proximal policy optimization
- **命名注**：标题为 HGA-MPPO，正文中算法统称 **MPPPO**（multi-policy PPO）。

### 2. 问题设定
- **问题类型**：**柔性作业车间调度 + AGV（FJSP-AGVs）**；多 AGV 在 LU（装卸区）与机器之间运输作业。
- **资源**：机器（ϱ 台）、AGV（ζ 台）、LU 区；每个作业含多道工序，工序机器可选集 M_{i,j}。
- **优化目标**：min Cmax（Eq.(1) `Minimize C_max`）；DRL 侧另设复合奖励间接兼顾机器负载均衡与 AGV 利用率（但奖励权重不进入目标函数）。
- **关键约束/假设（原文 9 条）**：
  1. "Each operation can only be processed by a single machine without interruption."
  2. "Initially, both AGVs and machines are in an idle state."
  3. "Each machine has an infinite buffer to accommodate the finished jobs."（**缓冲无限**）
  4. "The speed of each AGV remains constant, and the delivery time depends on the actual distance between machines."
  5. "Each AGV can transport only one job at a time."
  6. "After completing a transport task, the AGV can park next to the current machine and wait for the next transport assignment."
  7. "If adjacent operations of the same job are processed on the same machine, they can be transported without using an AGV."
  8. "Breakdowns and collisions during AGV transport are disregarded."
  9. "No delays occur in AGV transportation."
- **MILP 模型**：Eq.(1) 目标；约束 (2) 每工序一台机器；(3)–(5) 机器排序无冲突；(6)–(8) 运输任务生成（同机相邻工序可免运输）；(9)–(16) AGV 分配与任务排序；(17)–(26) 时间耦合（运输时间、到达时间）；(27)–(29) 完工时间与 makespan 定义。
- **示例算例**：n=4, ϱ=5, ζ=2, n1=n2=n3=3, n4=2，可行解 Cmax = 21。

### 3. 方法
- **算法类型**：**端到端单智能体 DRL — MPPPO**（PPO + 保守双 Critic + 自适应裁剪），配异构图注意力编码器 HGAE-Net。三步行序决策：① 识别空闲 AGV 集；② 选待排工序；③ 为该工序分配空闲 AGV 与兼容机器。
- **网络结构**：
  - **HGAE-Net**（三层嵌入）：① 机器节点嵌入（多维图注意力 Eq.(36)–(40) + GraphSAGE 能力聚合 Eq.(41)(42) + 多尺度特征融合 Eq.(43) + 门控残差融合 Eq.(44)(45) → MLP）；② AGV 节点嵌入（同机制 Eq.(36)–(45)）；③ 工序节点嵌入（聚合前驱 O_{i,j−1}、后继 O_{i,j+1}、自身 O_{i,j}、机器嵌入 M'_{i,j}、AGV 嵌入 V'_{i,j}，五源门控融合 Eq.(46)(47)）；随后 **堆叠 L 层**并分别对工序/机器/AGV 做均值池化，拼接成 `h_t ∈ R^{3d_h}`。
  - 超参（表 7）：**L=2**，**d_h=128**，注意力头数 **H_{k−o}=8, H_k=4, H_{v−o}=6, H_v=3**。
  - Actor/Critic：**MLP 3 个隐层 × 64 节点，Tanh 激活**，参数独立（θ 与 φ）。
- **状态定义（异构图 H = (O, M, A, C, E)）**：
  - 节点特征：工序 `O_{i,j} ∈ R^8`（8 维见表 6：是否已排、可选机器数、估计/实际加工时间、同作业未排工序数、作业估计/实际完工时间、工序估计/实际开始时间、从前驱位置到当前机器的运输时间、可运输该工序的 AGV 数）；机器 `M_k ∈ R^3`（可分配工序数、最早可开工时间、利用率）；AGV `V_v ∈ R^3`（可分配工序数、最早可开工时间、利用率）。
  - 边特征：O–M 边 `λ_{i,j,k} ∈ R²`（前驱位置到当前机器的运输时间、该机器上的加工时间）；O–A 边 `λ_{i,j,v} ∈ R`（AGV 当前位置到作业位置的运输时间）。
  - 含 **virtual operation node**（连接所有作业的首工序，作为统一起点）。已排工序仅保留被选中的 O–M 与 O–A 边。
  - 动态估计（Eq.(30)(31)）：运输时间 `T_{i,j}(t)` 按前驱/当前工序是否已排分三种情形（实际值 / α×平均 / 平均）；开始时间 `S_{i,j}(t)` 同机用 `max(C_{i,j−1}, A_k)`，异机用 `max([max(t_ava + t_{loc}, C_{i,j−1}) + T_{i,j}(t)], A_k)`；未排工序加工时间用可选机器平均 `p_{i,j} = (1/|M_{i,j}|)Σ_k p_{i,j,k}`。
- **动作空间设计**：**选实体 —— 三元组 `(O_{i,j}, k, v)`**（工序、机器、空闲 AGV）。注意：即使相邻工序同机、无需运输，动作中仍须指定 AGV，以保持机-车耦合的统一表示。动作空间随排程推进动态缩减。**无掩码机制描述**（仅称"feasible triplet"）。
- **奖励函数（含公式）**：复合奖励 Eq.(32) `r_t = r^s_t + r^m_t + r^a_t`；
  - Eq.(33) `r^s_t = C_max(s_t) − C_max(s_{t+1})`（makespan 差分，基础项，来自 Huang et al. 2023）；
  - Eq.(34) `r^m_t = −α · Var(t)`（机器利用率方差惩罚）；
  - Eq.(35) `r^a_t = −β · idle(t)`（AGV 平均累积空闲时间惩罚）；
  - 权重：**α=0.15, β=0.1**。
- **MPPPO 机制**：
  - 保守双 Critic：Eq. 优势估计 `Â_t = R_t − min(V_φ(s_t), V_{φ'}(s_t))`；价值损失 `L^VF = ½[(V_φ−R_t)² + (V_{φ'}−R_t)²]`。
  - 自适应裁剪：`ε_new = clip(ε·exp(τ(KL − kl)), ε_min, ε_max)`，ε=0.2, ε_min=0.01, ε_max=0.3, kl=0.01, τ=1.5。
  - 训练：1000 epochs，1000 训练实例/epoch，batch B=20，γ=0.95，lr 2e-4，c_p=1.0, c_v=0.2, c_e=0.01。

### 4. 实验
- **实例来源与规模**：
  - **D1 验证集（自生成）**：三个规模 10×5×2、10×5×5、20×10×10（ϱ×n×ζ），各 100 个实例；另用 30×15×15（100 实例）测泛化（用 20×10×10 训练的模型）。生成参数（表 8）：n_i ~ U(0.8ϱ,1.2ϱ)，|M_{i,j}| ~ U(1,ϱ)，平均加工时间 ~ U(1,20)，实际加工时间 ~ U(0.8p̄,1.2p̄)，运输时间 ~ U(1,20)。
  - **D2 公开基准**：来源 `https://fastmanufacturingproject.wordpress.com/2019/04/11/fjspt-instances/`，共 **65 个实例**：Δ1 = 10 个小规模 [54, Deroussi & Norre 2010]；Δ2 = 28 个小规模 [55, Kumar et al. 2011]；Δ3 = 7 个大规模 [16, Homayouni et al. 2023]；Δ4 = 20 个（10 小 + 10 中）[56, Homayouni & Fontes 2021]。
- **基线（列全）**：
  - **3 条优先派工规则（PDR）+ 最早可用 AGV 规则**：**SPT、FIFO、MOR（most operations remaining）**；
  - **3 个先进 DRL 算法**：**DRL-S** [34, Wang et al. 2025, attention+GAT+MLP+PPO]、**HGS** [36, Moon et al. 2023, graph embedding]、**DRL-Z** [37, Zhang et al. 2023, HGNN 端到端]；
  - **DRL-B**（消融：把复合奖励换成 `r_t = C_max(s_t) − C_max(s_{t+1})`）；
  - **Gurobi 10.0.2**（MILP 正确性验证，时限 3600 s）。
- **指标**：Cmax、**RPI Eq.(48)** `RPI = ((C_max − C_best)/C_best) × 100%`、**winning rate**（取得最优结果的实例配置比例）、执行时间、Friedman 检验、Wilcoxon 符号秩检验（α=0.05）。
- **硬件/软件**：Intel Core i7-9700K @ 3.60 GHz，32 GB RAM，Windows 10 Home，Python 3.9 + PyCharm。

### 5. 核心数字（照抄）
- D1（表 12，平均 Cmax）：
  - 10×5×2：**MPPPO 405.94** vs SPT 638.71、FIFO 850.68、MOR 723.13、DRL-S 446.32、HGS 563.83、DRL-Z 536.92（提升 36.44%、52.28%、43.86%、12.95%、28.00%、24.39%）；RPI 0.0168。
  - 10×5×5：**180.77** vs 225.09 / 262.52 / 290.30 / 198.29 / 214.47 / 238.30（19.69%、31.14%、37.72%、8.83%、15.71%、24.14%）；RPI 0.0167。
  - 20×10×10：**390.38**（33.65%、39.51%、52.97%、4.59%、25.98%、38.23%）；RPI 0.0034。
  - 30×15×15：**607.89**（44.08%、41.21%、60.24%、11.45%、34.38%、36.64%）；RPI 0.0000。
- 胜率：10×5×2 与 10×5×5 分别 77.2% 与 81.2%（100 实例中 78 / 82 个最优）；20×10×10 与 30×15×15 分别 82.4% 与 **98%**（82 / 100 个最优）。
- D2（表 13，平均 Cmax / RPI）：Δ1 **147 / 0.0087**（vs SPT 256.60、FIFO 210.00、MOR 275.60、DRL-S 156.50、HGS 161.00、DRL-Z 159.80）；Δ2 **82.18 / 0.0216**；Δ3 **386.29 / 0.0000**（vs SPT 776.86、DRL-S 492.29）；Δ4 **622 / 0.0251**。D2 胜率 **65.7%**（PDR 2.8%、DRL-S 15.7%、HGS 8.9%、DRL-Z 7.1%）。
- 复合奖励消融（表 10，17 个实例）：MPPPO 均值 **120.59** vs DRL-B **125.24**，提升 **3.71%**；MPPPO 在 14/17 实例取得最低 Cmax/RPI，DRL-B 仅 5 个。
- MILP vs MPPPO（表 11）：小规模同解（Sfjs02 111、Sfjs06 324、Sfjs10 531）；中大规模 Gurobi 超时且 Gap 高：EXF74 (4×8×24) Gurobi 113 / gap 53.0973% / 3600 s vs MPPPO 102 / 1.93 s；MK01 Gurobi 176 / 69.8864% / 3600 s vs MPPPO 151 / 2.64 s；MK05（15×4×106）Gurobi 336 / 74.1583% / 3600 s vs MPPPO 318 / 2.98 s。
- Friedman 秩（表 14/15）：D1（100 实例）MPPPO 1.22 vs DRL-S 2.16、HGS 2.94、DRL-Z 3.68；D2（65 实例）MPPPO 1.52 vs DRL-S 2.28、HGS 2.92、DRL-Z 3.29。Wilcoxon：vs DRL-S / HGS / DRL-Z 在 RPI 与 Cmax 上 **p-value 均为 0.000**，全部 Reject H0。

### 6. 自认局限（原文逐字 + 中文翻译）
> 结论第 6 节含明确的 **Limitations** 段。

"**Limitations:** Despite its performance, the framework has some limitations. First, its adaptability under real-world dynamics—such as frequent job insertions and machine failures—needs further validation. Second, the method inherits DRL's challenges: high sample demand, long training time, and sensitivity to hyperparameters. The graph structure and multi-policy design also increase training cost. Although adaptive clipping improves stability, efficiency and robustness still require enhancement. As the problem scale grows, the enlarged action space and system coupling introduce more computational overhead and decision difficulty."
→ 局限：尽管性能良好，框架仍存在局限。第一，其在真实动态环境（如频繁插入作业、机器故障）下的适应性仍需进一步验证。第二，方法继承了 DRL 的固有问题：样本需求大、训练时间长、对超参数敏感；图结构与多策略设计还增加了训练成本。虽然自适应裁剪提升了稳定性，但效率与鲁棒性仍需增强。随着问题规模增大，扩大的动作空间与系统耦合带来更多计算开销和决策难度。

"**Future Work:** Future efforts will enhance the framework's ability to handle dynamic events and complex constraints. We will explore multi-objective scheduling, balancing traditional metrics with sustainability goals. On the algorithm side, we aim to design lightweight encoders and simplified policies, use transfer learning to reduce training time, and develop adaptive modules for tuning, reward shaping, and feature selection. For large-scale scenarios, distributed learning and agent communication will be introduced to enable scalable, efficient scheduling [60]."
→ 未来工作：增强对动态事件与复杂约束的处理能力；探索多目标调度（兼顾传统指标与可持续性目标）；算法侧设计轻量编码器与简化策略、用迁移学习减少训练时间、开发自适应调参/奖励塑形/特征选择模块；大规模场景引入分布式学习与智能体通信。

另有一处自陈边界（4.1 节奖励设计）："While the current design targets static environments and does not explicitly handle disruptions (e.g., machine failures), its modular structure allows future extension."（当前设计面向静态环境，未显式处理机器故障等扰动）。

### 7. 可复用
- **开源代码**：**无**。原文 "Data availability: Data will be made available on request."（数据按请求提供，未给仓库或许可证）。
- **公开数据集/基准**：
  - FJSP-T 基准集合：`https://fastmanufacturingproject.wordpress.com/2019/04/11/fjspt-instances/`（Δ1 Deroussi & Norre 2010；Δ2 Kumar et al. 2011；Δ3 Homayouni et al. 2023；Δ4 Homayouni & Fontes 2021，共 65 实例，含 FJSP1–10、EXF、MK、Sfjs 系列）。
  - D1 自生成协议（表 8 生成参数 + 表 7 训练超参）可完整复现。
- **可借鉴的评测协议/公式**：
  - **双数据集策略**：D1（自生成，分规模测泛化 10×5×2 → 30×15×15）+ D2（公开基准，测跨分布泛化），并用 Friedman + Wilcoxon 双检验支撑。
  - **RPI Eq.(48)** 与 **winning rate** 两个指标的联合使用（比单看平均 Cmax 更能反映稳定性）。
  - **复合奖励消融设计**：把 `r_t = r^s + r^m + r^a` 替换为纯 makespan 差分的 DRL-B 对照，量化负载均衡/AGV 空闲惩罚的增益。
  - 保守双 Critic `Â_t = R_t − min(V_φ, V_{φ'})` 与 KL 自适应裁剪 `ε_new = clip(ε·exp(τ(KL−kl)), ε_min, ε_max)`（可直接搬到 PPO 类调度框架）。
  - 状态估计处理"未排工序"的插值做法（运输时间/加工时间按可选集合取平均，Eq.(30) 与 `p_{i,j} = 1/|M_{i,j}| Σ p_{i,j,k}`），使异构图在部分排程状态下仍可编码。
  - 用时序 5 源门控融合（前驱/后继/自身/机器/AGV）构造工序嵌入，Eq.(46)(47)。

---

## 横向对照速查（仅事实罗列）

| 项 | COR2026 | CIE2026 | EJOR2026 | ESWA2026 | SWEVO2026 |
|---|---|---|---|---|---|
| 问题 | 两阶段装配流水车间 TAFSP-AMRs | 分布式柔性作业车间 DFJSP-WAC | 装配线 FAL + MHR 联合调度 | 非相关并行机 + AGV + 有限缓冲 | 柔性作业车间 + AGV (FJSP-AGVs) |
| 资源 | 机器 + 2 类 AMR + 机械臂退化 | 机器 + AGV + 工人（数字工人） | 工位 + MHR + 充电站 + 传送带 | AGV + 非相关并行机 + 有限输出缓冲 | 机器 + AGV |
| 目标 | 双目标 Cmax + TEC | Cmax（单） | Σ 订单完成时间（单） | Cmax（单） | Cmax（单，奖励含负载/空闲惩罚） |
| 方法 | ε-constraint(GUROBI) + 启发式 SRSOA/SRSOA II | HMADDQN（分层多智能体 DDQN + 残差网络） | MIP + 拉格朗日松弛（JFMS 算法） | HCHGNN + PPO（去耦 actor-critic） | HGAE-Net + MPPPO（PPO 变体） |
| 动作空间 | 不适用（非 RL） | **选规则**（5 智能体 × 调度规则，组合 216 个动作） | 不适用（非 RL） | **选实体**（AGV,订单,机器）/（AGV,机器），掩码 | **选实体**（工序,机器,AGV）三元组 |
| 网络 | 无 | 线性层 + 2 残差块 + FC，隐层 64 | 无 | 异构图注意力（hub-centric，4 类节点 3 类边），H/L 未给数值 | 异构图注意力 HGAE-Net，L=2, d_h=128, 头数 8/4/6/3 |
| 硬件 | Ryzen 7 3.2GHz, 16GB | i7-13790F, 32GB, RTX 4070 | Xeon W-2255 @3.7GHz | i9-14900K, 128GB, RTX 4090 | i7-9700K @3.6GHz, 32GB |
| 代码 | 无 | 2 个 GitHub 仓库 | 1 个 GitHub 仓库 | 1 个 GitHub 仓库 | 无 |


## 【组 g2】

### 事实卡 G2（4 篇）

> 仅记录原文事实，不做评价。原文英文逐字引用；中文为翻译。表格内容因 PDF 表格为图片未能提取时，以正文文字为准并标注。

---

## 1. Distributed Flexible Job Shop Scheduling With Heterogeneous Transportation Resources Constraints via Deep Reinforcement Learning and Graph Neural Network

### 1.1 题录
- 标题：Distributed Flexible Job Shop Scheduling With Heterogeneous Transportation Resources Constraints via Deep Reinforcement Learning and Graph Neural Network
- 作者：Kaikai Zhu, Xiaobin Li (Member, IEEE), Pei Jiang (Member, IEEE), Min Cheng, Yuanqing Wu (Senior Member, IEEE), Kaizhou Gao (Member, IEEE), Lei Ren (Member, IEEE)
- 机构：State Key Laboratory of Mechanical Transmission for Advanced Equipment, Chongqing University, Chongqing 400030, China（Zhu、Li、Jiang、Cheng）；School of Intelligent Systems Engineering, Sun Yat-sen University（吴元庆）；Department of Engineering Science, Macau University of Science and Technology（高凯洲）；School of Automation Science and Electrical Engineering, Beihang University / Zhongguancun Laboratory / State Key Laboratory of Intelligent Manufacturing System Technology（任磊）
- 期刊：IEEE Transactions on Systems, Man, and Cybernetics: Systems, VOL. 56, NO. 5, MAY 2026, pp. 3086–3098
- 年：2026（Received 13 November 2025; accepted 13 January 2026; Date of publication 29 January 2026; date of current version 16 April 2026）
- DOI：10.1109/TSMC.2026.3656196
- 基金：National Key R&D Program of China 2023YFB3308001；NSFC 52075060、52575561；Chongqing CSTB2024TIAD-STX0029
- 补充材料：有（Supplementary Materials，含符号表、PPO 可行性证明、超参敏感性、运行时间与 t 检验表）

### 1.2 问题设定
- 问题类型：**分布式柔性作业车间（DFJSP）的扩展**——首次提出 DFJSPHT（distributed flexible job shop scheduling problem with heterogeneous transportation resource constraints）。非 flow shop。
- 资源：n 个工件、m 台机器、y 个工厂、r 台厂内 AGV（IAGV）、v 台跨厂 AGV（CAGV）。实例规模记作 n × m × y × r × v。
- 优化目标：最小化 makespan（Cmax）。
- 分解为 4 个子问题：1) 为每道工序指定工厂；2) 指定机器；3) 为需运输的工序指定运输资源；4) 确定全部工序的加工顺序。
- 关键约束/假设（逐条）：
  1) "An operation can only be transported or processed after its previous operation is completed."
  2) "Each IAGV can perform all transportation tasks within its affiliated factory and can only transport one job at a time."
  3) "Each CAGV can perform all cross-factory transportation tasks and can only transport one job at a time."
  4) "All IAGVs have the same transportation time between two machines within their affiliated factories."
  5) "All CAGVs have the same transportation time between two machines in different factories."
  6) "All machines, jobs, IAGVs, and CAGVs are idle at time 0."
  7) "Transportation route conflicts and power replenishment are not taken into account."（不建模路径冲突与补能/充电）
  8) "The loading time, unloading time, and return time to the designated location of IAGVs and CAGVs are included in the predefined transportation time."
- 运输时间由预定义矩阵 D ∈ R^{(g+2)×(g+2)} 给出；M0、Mg+1 分别对应原料区与成品区。每个工件末道工序后加一个虚拟工序 Oic（仅由成品区"加工"）。

### 1.3 方法
- 总体：端到端 DRL；异构析取图（heterogeneous disjunctive graph）建模 + HGNN 特征提取 + PPO（actor–critic）训练；另加"资源释放策略"（resource release strategy）。
- 图模型：H = (O, M, A, C, G) —— 工序节点、机器节点（工厂信息嵌入机器节点）、AGV 节点（CAGV 工厂编号统一记为 0）、同工件工序间有向连接弧 C、O–M 与 O–A 析取弧集合 G。
- 网络结构：三阶段节点特征嵌入（GAT + MLP）：
  1) 工序节点先融合相邻 AGV 节点特征，机器节点再聚合对应工序节点嵌入（式(3)–(8)）；
  2) 工序节点再整合其关联机器节点嵌入，AGV 节点聚合相邻工序节点（式(9)–(10)）；
  3) 工序节点整合相邻工序、更新后的机器与 AGV 节点嵌入，经 MLP0–MLP5 + ELU 得到 8 维工序嵌入（式(11)）。
  注意力系数用 LeakyReLU + Softmax；聚合用 sigmoid。
- 状态定义（三类节点 + 两类弧特征）：
  - 工序节点 μij ∈ R^10：是否已排（1/0）；可用机器数 Nk(Oij)；Ji 未排工序数；加工时间 Ptijfk（未排时取估计平均）；运输开始时间 Tsij = max{EOij, EAfc}；运输结束时间 Teij = Tsij + Tc；加工开始时间 Stij = max{Teij, EMfk}；加工完成时间 Ctij = Stij + Ptijfk；指定工厂 Ff；指定机器 Mfk。估计平均运输时间见式(1)(2)。
  - 机器节点 Vfk ∈ R^6：可加工工序数、最早可用时间、利用率、负载、所属工厂 Ff、该厂负载。（start/end 节点：最早可用时间与负载为 0、利用率为 1、工厂为 −1）
  - AGV 节点 Yfc ∈ R^3：所属工厂 Ff（CAGV 为 0）、最早可用时间、利用率。
  - O–M 弧特征 = 加工时间 Pti j fk 与平均运输时间 Tc 之和；O–A 弧特征 = 该工序到各可用机器的平均运输时间。
- **动作空间设计（关键）**：**选实体** —— "The action space A(t) is composed of a set of mutually compatible action pairs, which represent all feasible O-M-A at decision point t." 即每个决策点从可行的（工序–机器–AGV）三元组中选一个；不可行动作由 mask 屏蔽。决策后动态删除冗余 O–M / O–A 弧以缩小动作空间。对不需要运输的工序仍选一台 AGV 以保持动作一致性，但不占用该 AGV。动作概率：
  P(at, st) = MLPw[μ'(L)ij ‖ V'(L)fk ‖ Y'(L)fc ‖ ht]（式(13)），
  π(at|st) = exp(P(at,st)) / Σ_{a't∈A(t)} exp(P(a't,st))（式(14)）。
- 全局表示 ht ∈ R^{3d} 由三类节点嵌入平均池化拼接（式(12)）。
- 资源释放策略：当 t 时刻任一批次都无可行 O-M-A 时，识别被占用资源类型（机器占用 / AGV 占用 / 两者），若当前工件的可用机器全被占用，则立即释放完成时间最早的机器并同步更新机器与工件状态；若释放后仍无可行 O-M-A，则恢复占用状态并推进时间 t。
- **奖励函数（逐字公式）**：r(st, at, st+1) = Cmax(st) − Cmax(st+1)。
- 训练：PPO actor–critic；共 1000 次迭代；每 2 次迭代更新网络参数；每 10 次迭代用 100 个实例的验证集评估；折扣因子 1；策略/值/熵损失权重 1、0.5、0.01；Adam，学习率 2×10⁻⁴；HGNN 层数 L=2，嵌入维度 8，工序节点三层 MLP 隐层 128；actor/critic 均为 3 个 64 维隐层。

### 1.4 实验
- 实例：
  - **DS1（合成）**：18 个实例（特定均匀分布生成），分 3 组；工件–机器组合 10×3、15×3、20×3、20×5、30×5、40×5；工厂–IAGV–CAGV 组合 2×2×2、2×3×2、3×3×3、3×4×3、4×4×4、4×5×4。训练实例动态生成（含 100 个验证实例）；每个规模抽样 100 个测试实例。
  - **DS2（公开基准改造）**：基于经典 FJSP 基准 la 实例（Hurink 的 rdata、edata、vdata [37]），la01–la40 各扩展为 G1–G3 三组，工厂数取 [2,3,4]、机器数取 [3,5,7]、IAGV 与 CAGV 数统一取 [2,3,4]。模型不做再训练或微调直接应用。
  - **大规模泛化测试**：9 组大实例，每组 100 个随机实例（共 900 个）；用 10×5×5×5×5、10×5×6×6×6、10×5×7×7×7 上训练的 3 个模型直接求解，且每个模型只用求解与其工厂数相同的实例。
- 基线：4 个经典优先级调度规则（PDR）：FIFO、SPT、LPT、MOR（剩余工序最多）；2 个 SOTA DRL 方法：HGNN（Song et al. [23]）与 RL（Wang et al. [25]），并把工厂级信息与异构 AGV 特征并入其网络以保证公平。所提方法记作 **RL-H**。
- 指标：Gap = (Cmax − Cb) / Cb × 100%（式(15)）；配对 t 检验（显著性阈值 0.05）；另报告平均运行时间（在补充材料）。
- 硬件：**正文未报告**（未给出 GPU/CPU 配置；仅报告训练时长）。

### 1.5 核心数字（照抄原文）
- 摘要："Comparative experiments are conducted on synthetic and benchmark instances demonstrate that the proposed method outperforms the classical priority scheduling rules and two popular DRL-based scheduling methods in solving DFJSPHT, with performance improvements exceeding 10% in most instances."
- 训练时长："In terms of training time, RL-H takes 0.63, 1.24, 1.34, 4.52, 5.12, and 5.56 h on 10 × 3 × 2 × 2 × 2, 10 × 3 × 3 × 3 × 3, 10 × 3 × 4 × 4 × 4, 20 × 5 × 2 × 3 × 2, 20 × 5 × 3 × 4 × 3 and 20 × 5 × 4 × 5 × 4, respectively."
- DS2（la 实例）："In most instances, the RL-H reduces the makespan by more than 15% compared to the MOR rule and by more than 5% compared to the comparison DRL methods."
- 显著性："The mean paired differences highlight that RL-H can always achieve smaller makespans, and all p values are far below the significance threshold (0.05)."
- 规模效应："the relative gap between RL-H and other comparison methods gradually narrows as the number of factories increases when solving the instances with three machines (m = 3)."
- 运行时间："although the RL-H is slightly higher than other comparison methods in terms of average running time, its impact on the scheduling efficiency of real distributed manufacturing systems is acceptable."
- 消融（释放策略）："RL-H achieved the best results in all instances. Second, RL-H1 demonstrates superior performance to both HGNN and RL in most instances."

### 1.6 自认局限（原文逐字 + 中译）
> "Due to space limitations, the average running time and results analysis of the RL-H and the comparison methods are provided in 'Section D of Supplementary Materials.'"
中译：受篇幅限制，RL-H 与对比方法的平均运行时间与结果分析放在补充材料 D 节。
> "In future research, we will focus on the following directions: 1) considering dynamic factors in real-world production, such as urgent orders, machine failures, and quality anomalies; 2) exploring new methods to enhance model generalization and solution performance based on multiagent technology; and 3) incorporating the differences in load weight, speed, and energy consumption of transportation resources."
中译：未来研究将聚焦以下方向：1) 考虑现实生产中的动态因素，如紧急订单、机器故障与质量异常；2) 探索基于多智能体技术的新方法以增强模型泛化与求解性能；3) 纳入运输资源在载重、速度与能耗方面的差异。
- 隐含局限（假设第 7 条）："Transportation route conflicts and power replenishment are not taken into account."（不考虑运输路径冲突与能量补充。）

### 1.7 可复用
- 公开数据集：DS2 基于公开 FJSP 基准 Hurink la 实例（rdata/edata/vdata，la01–la40），改造协议已写明（工厂数、机器数、AGV 数、均匀分布设置）→ 可复现的跨问题扩展协议。
- 公式：Gap = (Cmax − Cb)/Cb × 100%（式(15)）；奖励 r = Cmax(st) − Cmax(st+1)；估计平均运输时间式(1)(2)；三阶段 GAT 嵌入式(3)–(11)。
- 评测协议：配对 t 检验 + 100 实例/规模的采样测试 + 跨规模/跨分布泛化（训练集 DS1 → 测试集 DS2 零微调）。
- 开源代码：未提供。
- 其他：补充材料含超参敏感性分析与可行性证明。

---

## 2. Digital twin-driven deep reinforcement learning for real-time optimisation in dynamic AGV systems

### 2.1 题录
- 标题：Digital twin-driven deep reinforcement learning for real-time optimisation in dynamic AGV systems
- 作者：Donggun Lee (a,b), Yong-Shin Kang (a), Sang Do Noh (b)
- 机构：a) Industrial Intelligence Laboratory, Advanced Institute of Convergence Technology, Suwon 16229, Republic of Korea；b) Department of Industrial Engineering, Sungkyunkwan University, Suwon, Republic of Korea
- 期刊：International Journal of Production Research, 2026, VOL. 64, NO. 1, 106–124
- 年：2026（Received 18 January 2025; Accepted 26 July 2025; Published online: 25 Aug 2025）
- DOI：10.1080/00207543.2025.2543491；开放获取（CC BY 4.0）
- 基金：MOTIE / KIAT International Cooperative R&D program [No. P0022807]

### 2.2 问题设定
- 问题类型：**AGV 系统实时优化（非 job shop 排产）**——同时处理 (a) 实时路由问题（routing：将动态到达的作业 w1…wn 分配给 AGV 并决定执行顺序）与 (b) 实时路径规划问题（path planning：确定两点之间的具体节点路径，避碰、绕障）。
- 资源：k 台 AGV（案例中 4 台，各配充电站）、网格地图节点与有向路径、作业（works）；作业属性 wi = (ni, oi, τi, pi)：执行节点、工序、预计完成时间、优先级。
- 目标：最大化系统吞吐、最小化 makespan、优先级遵守、AGV 负载均衡、路径避碰与行驶时间最小化（多目标加权为单一奖励）。
- 关键约束/假设：
  - "Each AGV operates on a grid composed of nodes and the directed paths connecting them."
  - "All decision-making is centralised through an AGV control system (ACS), indicating a single-agent decision-making framework rather than a multi-agent one."
  - "Decision points arise at two critical moments: (1) When the initial work schedule is generated and routes are first assigned to AGVs, and (2) when dynamic changes (e.g. work schedule updates, obstacles, failures) necessitate real-time re-routing."
  - 路径候选集由 DT 仿真在每个决策点动态生成，"it does not assume full enumeration of all possible paths"。
  - 动态事件：随机新作业到达、设备故障、工作量增加。

### 2.3 方法
- 算法：DDQN（double DQN），Q 网络采用 LSTM-DQN 结构（Zhang et al. 2022）；ε-greedy 探索；DT 同时参与训练与推理（DT-driven DRL，区别于 DRL-only 与 DT-assisted DRL）。
- 架构：制造车间（MES + ACS）→ 信息管理模块（MariaDB 数据库、数字模型库、回放缓冲）→ DT 仿真模块（DT 基模型 + 仿真引擎 + 虚拟 AGV 控制器）→ DRL 智能体模块；数字模型与经验元组一一对应存储（DT repository）。
- 状态（式(6)(7)）：
  - s = (W, V, AGVb, AGVl, AGVs)
    - W = {w1,…,wn}：全部作业；wi = (ni, oi, τi, pi)
    - V ⊆ W：已完成作业集合（m 个）
    - AGVb = [b1,…,bk]：各 AGV 电量
    - AGVl = [l1,…,lk]：各 AGV 当前位置
    - AGVs = [s1,…,sk]：各 AGV 运行状态（空闲/工作/充电）
- **动作空间设计（关键）**：**选实体 + 选序列/路径的组合**。a = (AGVj, D, P)（式(8)）：
  - AGVj：被选中的 AGV，j ∈ {1,…,k}；
  - D = [wd1,…,wdq]：分配给 AGVj 的有序作业列表（从 W\V 中选）；
  - P = [p1,…,pq−1]：相邻作业之间的有序路径列表，每条路径 pr = [n(r)1,…,n(r)e] 为节点序列，起点为 wdr 的位置、终点为 wdr+1 的位置。
  - 动作空间不枚举全部路径，而是在每个决策点由 DT 仿真生成有限可行路径候选（反映交通、电量、障碍等实时工况）。
- **奖励函数（逐字公式）**：
  - fo = λsystem · Rsystem + λrouting · Rrouting + λpath · Rpath，λsystem + λrouting + λpath = 1（式(10)）
  - Rsystem = λThroughput · Throughput − λMakespan · Makespan，λThroughput + λMakespan = 1（式(11)）
  - Rrouting = −Ttotal + λPw · Pw − λσassign · σassign，λPw + λσassign = 1（式(12)），其中 Ttotal = Σ_{j=1}^{k} Tj
  - Pw = Σ_{wi∈V} pi / Ri（式(13)），pi 为优先级，Ri 为完成排名
  - σassign = sqrt( (1/k) Σ_{j=1}^{k} (Tj − T̄)² )（式(14)）
  - Rpath = Σ_{j=1}^{k} Σ_{r=1}^{qj−1} C(pr(j)) · Ttravel(pr(j))（式(15)）
  - C(pr(j)) = −1 若该路径发生碰撞，1 若无碰撞（式(16)）；碰撞定义为多台 AGV 同时占用同一节点
- 训练：TD 损失；目标网络每 τ 步同步；早停条件 ΔL < δ（式(5)）；在线网络选动作、目标网络评估（式17 的 Bellman 更新）；策略梯度损失 L = −log πt(a|st)·A(st,a)（式(3)），值损失为 TD 平方（式(4)）。
- 算法 1（训练）与算法 2（路由与 AGV 选择、路径规划）伪码已给出；算法 2 中对每个候选动作在 DT 中仿真并用 fo 评估，取 argmax Q。

### 2.4 实验
- 实例来源：**真实工业案例**——韩国某电动汽车电池模组生产现场；4 台 AGV（各有专用充电站）、6 个设施 + 2 个缓冲区、网格化地图；存在生产计划频繁变更、突发作业到达、设备故障等扰动。
- 场景设计（Table 3）：S1 正常（作业量 50，随机作业 0，故障 0）；S2 工作量增加（60/0/0）；S3 突发作业到达（50/10/0）；S4 设备故障（50/0/2）；S5 复合扰动（60/10/2）。所有方法在 S1 训练，其余场景**不重新训练**直接评测。
- 基线：DRL-only、DT-assisted DRL、所提 DT-driven DRL——三者都使用相同 DDQN + LSTM-DQN 架构。
- 指标：平均作业完成时间 (s)、平均 AGV 行驶时间 (s)、平均 AGV 空闲时间 (s)、系统吞吐 (works/h)；每个实验独立重复 10 次，报告 mean ± std。
- 实现/硬件：信息管理模块 MariaDB，接口 TCP/IP + Socket；DT 仿真 Siemens Plant Simulation 2404（SimTalk 2.0 + Python 3.9.13）；DRL 模块 PyCharm 2023.2、Python 3.9.13、PyTorch 2.0.1、CUDA 11.8（未给出具体 GPU 型号）。
- 超参（Table 5）：折扣因子 γ = 0.99；学习率 α = 0.001；探索概率 ε = 0.01~1.0；目标网络更新频率 50；FC 层数 2，每层 64 节点；LSTM 层数 1，单元数 128；激活函数 Linear。

### 2.5 核心数字（照抄原文）
- 摘要："Results demonstrate the proposed approach's capability to significantly enhance adaptability and efficiency in complex manufacturing settings."
- S5（最复杂场景）："the DT-driven method achieved an average throughput of 58.2 works/h, which represents an improvement of approximately 13.5% over DT-assisted DRL (51.3 works/h) and approximately 27.4% over DRL-only (45.7 works/h). Additionally, its average AGV idle time was reduced to 94.1 s, corresponding to a reduction of about 18.5% compared to DT-assisted DRL and about 37.6% compared to DRL-only"
- 权衡："in scenarios S2 and S3, the DT-driven method exhibited marginally higher average AGV travel time compared to the DT-assisted variant, indicating a trade-off where increased throughput may slightly extend travel paths."
- 碰撞："As a result, no collisions occurred during the experiments, and the reported performance metrics in Table 6 were obtained under strictly collision-free conditions."
- 收敛："DT-driven DRL exhibits the fastest convergence and highest reward stability ... In contrast, DRL-only suffers from high reward variance and slow convergence, while DT-assisted DRL stabilises earlier but plateaus at a lower reward level."
- Table 6 数值（mean ± std，节选）：S2：DRL-only 157.8±5.2 s / 846.7±18.9 / 134.5±10.2 / 47.2±1.8；DT-assisted 142.6±5.5 / 733.2±19.3 / 102.3±8.7 / 52.9±1.6；DT-driven 128.9±5.0 / 735.6±20.1 / 86.4±7.1 / 58.5±1.5。S5：161.2±6.0 / 860.1±23.7 / 150.7±11.7 / 45.7±2.2；146.3±5.6 / 782.5±22.9 / 115.4±9.0 / 51.3±1.8；132.5±5.1 / 729.3±18.7 / 94.1±7.1 / 58.2±1.4。

### 2.6 自认局限（原文逐字 + 中译）
> "From an operational perspective, the current framework relies on a centralised single-agent decision-making structure, which performed effectively in the experimental setting. However, as the number of AGVs and the complexity of operations increase, the state-action space expands rapidly, raising computational demands and potentially compromising real-time responsiveness. In addition, real-world production environments are prone to sensor errors, data delays, and communication failures, all of which can degrade system performance."
中译：从运行角度看，当前框架依赖集中式单智能体决策结构，在实验环境下表现良好；但随着 AGV 数量与作业复杂度增加，状态–动作空间迅速膨胀，推高计算需求并可能损害实时响应能力。此外，真实生产环境易出现传感器误差、数据延迟与通信故障，都会降低系统性能。
> "Another major consideration is the cost of establishing and maintaining both the digital twin and the DRL model. The framework depends on high-fidelity, real-time synchronised DTs, but creating and sustaining such systems is resource intensive. Frequent changes in production layouts and operations require continuous DT updates and regular DRL retraining, which increase the overall maintenance burden, especially for organisations lacking dedicated AI or simulation teams."
中译：另一个重要考量是数字孪生与 DRL 模型的建立与维护成本。框架依赖高保真、实时同步的 DT，而创建与维持此类系统资源消耗巨大。生产布局与作业的频繁变更要求持续更新 DT 并定期重训 DRL，增加了总体维护负担，对缺乏专职 AI 或仿真团队的组织尤为如此。
> "The generalisation of the framework to other industrial environments is also challenging. Each site has unique workflows and constraints, requiring customised DT modelling and DRL reconfiguration. Furthermore, the absence of standardised data formats and protocols limits scalability. Moving towards decentralised or multi-agent architectures could improve scalability but introduces new complexities, such as coordination and real-time conflict resolution."
中译：框架向其他工业环境的泛化同样困难。每个现场的工作流与约束各不相同，需要定制 DT 建模与 DRL 重配置；缺乏标准化数据格式与协议也限制了可扩展性。转向去中心化或多智能体架构可提升扩展性，但会引入协调与实时冲突消解等新复杂性。
- 数据可得性："The data supporting this study's findings are not publicly available due to confidentiality agreements and privacy considerations but are available from the corresponding author upon reasonable request."

### 2.7 可复用
- 开源代码：未提供。数据集：不公开（保密协议）。
- 可借鉴的评测协议：五场景扰动梯度设计（S1 基准 → S5 复合扰动）、"仅在 S1 训练 + 其余场景零重训评测"的泛化协议、每场景 10 次重复报告 mean±std、以及三类方法（DRL-only / DT-assisted / DT-driven）的同架构对照。
- 可借鉴公式：三层加权奖励分解 fo = λsystem·Rsystem + λrouting·Rrouting + λpath·Rpath；优先级项 Pw = Σ pi/Ri；负载不均衡惩罚 σassign 为 AGV 作业时间的标准差；碰撞惩罚 C(pr) = −1/1。
- 工程栈：Siemens Plant Simulation 2404 + SimTalk 2.0 + Python + MariaDB + TCP/IP Socket 的 DT-DRL 闭环实现路径。
- Table 1 汇总了 7 篇 AGV 实时优化文献的问题/算法/局限，可作为综述性引用来源。

---

## 3. A Multiagent Transformer-Based Algorithm for Multitask Dynamic Scheduling With Constrained Machines

### 3.1 题录
- 标题：A Multiagent Transformer-Based Algorithm for Multitask Dynamic Scheduling With Constrained Machines
- 作者：Yun Liu, Sri Srinivasa Raju Modampuri, Jiahao Fan (Member, IEEE), Yanan Sun (Senior Member, IEEE)
- 机构：College of Computer Science, Sichuan University, Chengdu 610065, China
- 期刊：IEEE Transactions on Cybernetics, VOL. 56, NO. 5, MAY 2026, pp. 2583–2596
- 年：2026（Received 30 April 2025; revised 19 July 2025, 14 November 2025, and 20 January 2026; accepted 12 February 2026; Date of publication 3 March 2026）
- DOI：10.1109/TCYB.2026.3665544

### 3.2 问题设定
- 问题类型：**动态柔性作业车间调度（DFJSS）的多任务扩展**——MT-DFJSS-CM（multitask DFJSS with constrained machines）。K 个 DFJSS 任务同时求解、共享受限机器。非 flow shop。
- 资源：任务 Ti 有 ni 个工件 Ji 与 mi 台机器 Mi；工件 Jij 有 eij 道工序；工序可在兼容机器集合 Mij,l 上加工且加工时间不同。**不涉及 AGV/运输资源**（对比第 1、4 篇）。
- 优化目标：多任务目标向量 min F(X) = (f1(x1), f2(x2), …, fK(xK))^T，本研究中 K=3 个任务分别对应：
  1) Mean-weighted-tardiness：WTmean = (1/ni) Σ wi j × max{0, Ci j − di j}
  2) Mean-weighted-flowtime：WFmean = (1/ni) Σ wi j × max{0, Ci j − ri j}
  3) Max-weighted-tardiness：WTmax = max wi j × max{0, Ci j − di j}
- 关键约束（式(1)）：(a) 解属于各自解空间；(b) 到达时间前不可开工；(c) 完成时间非负；(d) 每道工序必须在其兼容机器之一上加工；(e) 机器在某任务占用区间内保持被占用；(f) 工序先后约束；(g) 不同任务解空间不同；(h) 所有任务共享机器资源；(i) 共享机器在某时刻至多被一个任务占用。
- 耦合约束的后果（原文）："the joint solution space Ω is not a simple Cartesian product of all individual task solution spaces ∏K i=1 Ωi, but rather a strict subset"。
- 动态事件：新工件到达（Poisson 过程）与机器故障；发生前不可知。
- 求解范式："this article simulates the scheduling environment to enforce the above hard constraints, while the proposed algorithm focuses on learning high-quality decisions within the feasible space."

### 3.3 方法
- 算法：**MATS**（multiagent transformer-based scheduling），基于 MAT（Multi-Agent Transformer，Wen et al. NeurIPS 2022 [23]）扩展；把 MT-DFJSS-CM 建模为序贯多智能体决策（Markov game <O, A, R, P, N>）。
- 智能体划分：**K 个 routing agent（每个任务一个，负责机器分配）+ 1 个 sequencing agent（所有任务共用，负责机器等待队列内的工序排序）**，共 n = K+1 个智能体；与常见的并行决策不同，采用**序贯（自回归）决策**，每个智能体可观测前序智能体的动作。
- 网络：MATJP（MAT-based joint policy network）含三个模块：
  1) perception 模块：embedding 层 + 注意力 + 两个 MLP（第二个仅训练时估计状态值）+ 残差连接；
  2) reasoning 模块：embedding 层 + 两个注意力 + MLP + 残差，输入为已决策的嵌入动作 (a1t,…,a_{i−1}t)；
  3) task-based actor 模块：门控机制按 agent ID 选择任务专属 actor（Z 个 actor，每个含 3 个线性层）+ softmax + argmax，并施加任务专属 mask 保证动作可行。
- 观测（每类智能体 6 维）：
  - routing agent：or(t) = [CRJavg(t), CRJstd(t), TRavg(t), TRstd(t), Tde(t), Tda(t)]（平均工件完成率、完成率标准差、平均加工延误率、延误率标准差、估计拖期率、实际拖期率）
  - sequencing agent：os(t) = [Uavg(t), Ustd(t), Wavg(t), Wstd(t), duravg(t), durstd(t)]（机器平均利用率、利用率标准差、平均归一化机器负载、负载标准差、等待队列工序平均加工时间、其标准差）
  - 联合观测 Ot = [o1r(t), …, oKr(t), os(t)]
- **动作空间设计（关键）**：**规则选择（rule/heuristic selection）**。动作 = 从 9 个调度启发式中选一个（Table I；前 4 个用于机器分配，其余用于工序排序）。原文动机："when a large number of new jobs constantly arrive, the size of the action space can be greatly expanded. Such a dynamic and large action space results in high computation costs and introduces additional challenges for agent exploration. To address this issue, we select nine well-known scheduling heuristics to construct the action space, as detailed in Table I."（Table I 内容在 PDF 中为图片，未提取出具体 9 条规则名；正文另一处提到基线启发式包括 LWT、MOPNR、SPT、LPT、MWKR）
- **奖励函数（逐字公式）**：
  - 生产评估奖励 PAR：RP = Σ_{i=1}^{K} ( e^D − e^{−D} ) / ( e^D + e^{−D} )（式(2)），其中 D = max{0, fi(t) − fi(t−1)}，即 tanh(D)
  - 即时增量奖励 IIR：RI = e^{−ΔPT(t)}（式(3)），ΔPT(t) = −Σ_{i=1}^{K} Σ_{j=1}^{ni} wi j × ( OPTij(t) − OPTij(t−1) )
  - 排产结束奖励 ESR：RE = RWTmean + RWFmean + RWTmax（式(4)）；
    RWTmean = 1 若 WTmean = 0，−1 若 WTmean > 0（式(5)）；
    RWFmean = 1 若 WFmean_z < WFmean_{z−1}，0 若相等，−1 若更大（式(6)）；
    RWTmax = 1 若 WTmax = 0，−1 若 WTmax > 0（式(7)）
  - 总奖励：R = RP + RI（未排完）或 RP + RI + RE（排完）（式(8)）；默认权重 1:1:1。
- 训练：perception 模块用 empirical Bellman error 最小化；reasoning 与 task-based actor 模块用序贯策略更新（基于联合策略导出的局部优势函数）并最小化 clipping PPO 目标；文中给出单调改进定理（Theorem 1）及引理 1–3 的证明。
- 复杂度：MATS 为 O(EUBTZLN²)；启发式为 O(T log T)–O(T²)；EMTO-GP 为 O(GSnm)，TSIF-NSGA-III 为 O(TrGS²M)；PPO/HMAPPO/DMDDQN/AMDQN 为 O(EUBTLN²)。

### 3.4 实验
- 实例：仿真模型生成（遵循社区惯例）；设计 MTDFJS-CMR 场景，3 个任务共享同一利用率水平但目标不同（WTmean、WTmax、WFmean）；工件权重按 [51]：WTmean/WFmean 取 1、2、4 对应 20%、60%、20% 工件；WTmax 取 1、5、10。3 个利用率水平 0.95、0.85、0.75；**270 个实例**（3 场景 × 3 规模 × 30 seeds）；规模为 100×10、1000×10、5000×10（"100 × 10 indicates that 100 dynamically arriving jobs across all tasks need to be processed by ten machines"）。初始每任务 10 个工件，后续工件按 Poisson 过程到达；每工件工序数 1–10；每工序可选机器数 1–10；加工时间 1–99；交期 = 到达时间 + 1.5 × 总加工时间；机器故障随机发生，修复时间随机。
- 基线（14 个，3 类）：8 个手工启发式（机器分配：LWT、MOPNR；工序排序：SPT、LPT、MWKR、MOPNR 等）；2 个 SOTA 元启发式（TSIF-NSGA-III [55]、EMTO-GP [19]）；4 个 SOTA 学习型方法（PPO [56]、HMAPPO [10]、DMDDQN [54]、AMDQN [11]）。竞争方法的参数设置"follow their official repositories"。
- 指标：各任务目标的 mean 与 std；Wilcoxon rank sum test（显著性 0.05，"+" / "=" / "−"）；Friedman test（27 个用例）；平均训练与推理时间；箱线图（中位数、IQR、离群点）。
- 消融：模块共享（MATSs− 完全不共享 / MATSs+ 全部共享）；task-based actor 模块（MATS vs MAT）；5 种奖励配置（MATS1 仅 PAR；MATS2 仅 IIR；MATS3 PAR+ESR；MATS4 IIR+ESR；MATS5 PAR+IIR）；奖励权重 7 种配置（1:1:1、2:1:1、1:2:1、1:1:2、2:1:2、2:2:1、1:2:2）。
- 硬件/框架："MATS is implemented in PyTorch and is executed on a GPU with the model of Nvidia GeForce RTX 3090."
- 训练规模：MATS 在 100×10、利用率 0.95 的多任务实例上训练。

### 3.5 核心数字（照抄原文）
- 摘要："The proposed algorithm is evaluated against 14 state-of-the-art competitors on 270 instances with varying scales. The results confirm that the proposed algorithm outperforms all competitors on each instance."
- 消融（MATS vs MAT）："the mean values of MATS over those of MAT range from 0.19 to 1.2 for the task with the objective of WTmean, 28.45–106.91 in the task with the objective of WFmean, and 6.1–90.93 for the task with the objective of WTmax."（原文未说明该区间的单位/口径）
- 效率权衡："In terms of training time, MATS requires a longer training time than PPO, DMDDQN, and AMDQN, but less than HMAPPO and EMTO-GP. In terms of inference time, MATS is more time-consuming than most competitors, except for TSIF-NSGA-III."
- 奖励消融结论："IIR plays the most important role for the overall performance, while PAR and ESR serve as effective complements, with PAR contributing more significantly than ESR."
- 权重敏感性："the default one simply sums components by assigning equal weights, it can still achieve superior performance in most cases. Therefore, this experiment selects 1:1:1 as the default of MATS."
- 动态事件敏感性："the reward function exhibits consistent convergence trends under all settings. This suggests that the reward function demonstrates low sensitivity to the varied frequency of dynamic events."
- 模块共享结论："sharing either all or none of the parameters among agents in MATS is not optimal for dealing with multitask scenarios."
- 统计："The results indicate that MATS significantly outperforms its variants in most cases."（Wilcoxon rank sum + Friedman，27 个用例）

### 3.6 自认局限（原文逐字 + 中译）
> "Nevertheless, MATS primarily focused on heuristic selection while not explicitly considering the impact of alternative heuristic compositions in the action space, which may lead to suboptimal decisions. In the future, we will further investigate the relationship between different heuristic compositions and performance, which is expected to enhance the decision quality."
中译：尽管如此，MATS 主要关注启发式选择，而未显式考虑动作空间中备选启发式组合的影响，这可能导致次优决策。未来我们将进一步研究不同启发式组合与性能之间的关系，以期提升决策质量。
- 适用性说明（Discussion 中亦承认实验局限）："Although MATS is evaluated on simulated numerical data, its design still maintains better applicability and feasibility in real-world scheduling environments."（尽管 MATS 在仿真数值数据上评测，其设计在真实调度环境中仍保持较好的适用性与可行性。）

### 3.7 可复用
- 开源代码：未提供（但各基线"follow their official repositories"，即 TSIF-NSGA-III、EMTO-GP、PPO、HMAPPO、DMDDQN、AMDQN 均有官方仓库）。
- 公开数据集：无（实例由仿真模型生成）；但生成协议完整可复现：Poisson 到达、初始 10 工件/任务、工序数 1–10、可选机器 1–10、加工时间 1–99、交期 = 到达 + 1.5×总加工时间、利用率 0.95/0.85/0.75、工件权重 (1,2,4)/(1,5,10) 对应 20%/60%/20%。
- 评测协议：270 实例 × 30 seeds；Wilcoxon rank sum (0.05) + Friedman；mean±std 报表；训练/推理时间的双维度对比；三类基线（启发式/元启发式/学习型）覆盖单智能体与多智能体。
- 可借鉴公式：奖励三项 RP（tanh 形式）、RI = e^{−ΔPT}、RE 的 ±1 终局奖励；复杂度分析范式 O(EUBTZLN²)。
- 方法论参考：MAT（Wen et al., "Multi-agent reinforcement learning is a sequence modeling problem", NeurIPS 2022）作为基础架构。

---

## 4. Solving Collaborative Scheduling of Production and Logistics via Deep Reinforcement Learning: Considering Limited Transportation Resources and Charging Constraints

### 4.1 题录
- 标题（期刊版含冒号）：Solving Collaborative Scheduling of Production and Logistics via Deep Reinforcement Learning: Considering Limited Transportation Resources and Charging Constraints
- 作者：Xianping Huang, Yong Chen, Wenchao Yi (通讯), Zhi Pei, Ziwen Cheng
- 机构：School of Mechanical Engineering, Zhejiang University of Technology, Hangzhou 310012, China
- 期刊：Applied Sciences (MDPI), 2025, 15(13), 6995
- 年：2025（Received 21 May 2025; Revised 15 June 2025; Accepted 17 June 2025; Published 20 June 2025）
- DOI：10.3390/app15136995；开放获取（CC BY 4.0）
- 基金：NSFC 重点项目 W2411062

### 4.2 问题设定
- 问题类型：**柔性作业车间（FJSP）的生产–物流协同调度**，记为 FJSPLSP-LTCC（flexible job shop production-logistics collaborative scheduling problem with limited transportation resources and charging constraints）。非 flow shop、非分布式。
- 资源：n 个工件、m 台机器、x 台 AGV、y 个充电桩（charger）、装卸站（L/U）。AGV 三态：idle、unloaded running、loaded running；充电动作三类：no charging (NC)、opportunity charging (OC)、full charging (FC)；另有强制维护换电 MBC（mandatory maintenance battery change，电量低于阈值自动触发，不占用动作空间）。
- 优化目标：最小化 makespan（min Cmax = min(max_{i∈J} Ci)，式(2)）；仅单目标。
- 决策子问题：(1) 为每道工序选加工机器；(2) 确定各机器上工序加工顺序；(3) 为每道工序分配 AGV；(4) 为每台 AGV 安排充电动作。
- 关键约束（式(3)–(14)）：工序先后约束、加工唯一性、机器分配约束、运输唯一性、AGV 分配约束、完整运输时间约束（含充电与换电时间）、充电决策逻辑 da ∈ {0,1,2}、强制换电 ma、充电时间联动（Tcharge2 = (sfull − sa)/Pcharge）、AGV 电量更新规则、充电桩互斥约束。
- AGV "deterioration"（劣化）效应：电量下降导致行驶速度非线性下降，四段速度控制函数（式(1)，α = 0.5, β = 0.8, λ = 2.5）：
  - Saturation Zone (s ≥ srated)：v = 1
  - Decay Zone (scritical < s < srated)：v = α·s + β·e^{−λ(srated−s)}
  - Sustainment Zone (sdead ≤ s ≤ scritical)：v = vmin·tanh( (s−sdead)/(scritical−sdead) · π )
  - Failure Zone (s < sdead)：v = 0
- 假设 11 条，含："AGVs can only travel along predefined paths"；"AGV speed is updated prior to each movement based on the current battery level and remains constant during the trip"；"AGVs can only be charged prior to transporting jobs; after each transport task, a forced battery swap is evaluated"；"Each charging station can serve only one AGV at a time"；"Machine failures, job insertions, and AGV path conflicts are not considered."

### 4.3 方法
- 算法：**CRGPPO-TKL**（基于 PPO 的改进：Candidate Ratio Guided PPO + Target KL divergence 动态裁剪），Actor–Critic 双网络（Actor 3 层全连接 + Tanh + Softmax；Critic 共享 Actor 前两层输出状态价值）。PPO 原始 CRGPPO 由 Li and Tan [51] 提出（面向连续动作空间），本文将高斯噪声采样改为离散分布随机采样（πθ(a|s) = Categorical(p1,…,pAmax)，式(30)(31)），每步采 5 个候选动作。
- 动态裁剪机制（式(32)–(38)）：
  - raction = πθ(at|st)/πθold(at|st)（式(32)）；r(i)t(θ) = πθ(a(i)t|st)/πθold(a(i)t|st)（式(33)）
  - rcandidates = (1/N) Σ_{i=1}^{N} r(i)t(θ)（式(34)）
  - Iaction = I( (raction < 1−ε) ∨ (raction > 1+ε) )（式(35)）；Icandidates 同理（式(36)）
  - δKL = (Iaction + Icandidates)/2 − TKL（式(37)）
  - εnew = clip(εold · exp(η · δKL), εmin, εmax)（式(38)）
  - LCLIP(θ) = Et[min(rt(θ)At, clip(rt(θ), 1−ε, 1+ε)At)]（式(41)）；LActor = LCLIP + β·Et[H(πθ)]（式(42)）；LValue(φ) = Et[(Vφ(st) − Rt)²]（式(43)）
  - 优势估计用 GAE：δt = rt + γVφ(st+1) − Vφ(st)（式(39)）；AGAEt = δt + (γλ)δt+1 + …（式(40)）
- **状态（13 维，式(15)–(28)）**：机器平均利用率 URMave、机器利用率标准差 URMstd、工件平均完成率 CRJave、完成率标准差 CRJstd、未完工工件平均剩余加工时间 RTave、未完工工件剩余工序总数 ROtotal、AGV 平均当前电量 CPAave、电量标准差 CPAstd、AGV 平均利用率 URAave、利用率标准差 URAstd、AGV 平均充电次数 NCAave、充电次数标准差 NCAstd、电量低于 srated 的 AGV 总数 TABtotal。
- **动作空间设计（关键）**：**分层规则组合（rule combination），非选实体**。五层决策规则：
  1) 工件选择：MOR（Most Operations Remaining）、LOR（Least number of Operations Remaining）
  2) 机器选择：EAM（Earliest Available Machine）、LPT（Lowest Processing Time of Machines）
  3) AGV 选择：EAA（Earliest Available AGV）、HRP（Highest Remaining Power of AGV）
  4) 充电选择：NC、OC、FC（MBC 不入动作空间，自动执行）
  5) 充电桩选择：EAC（Earliest Available Charger）、STC（Shortest Transport Time of Chargers），仅在 OC/FC 时激活
  - 组合后共 **40 个复合调度动作**（Figure 4 示例：MOR+EAM+EAA+OC+EAC、LOR+EAM+HRP+NC、LOR+LPT+EAA+FC+STC 等）。
  - 空间压缩策略原文："this study adopts a structured decision-making process by partitioning the overall action space into staged and hierarchical local subspaces."
- **奖励函数（逐字公式）**：R(st, st+1, at) = Cmax(st) − Cmax(st+1)（式(29)），并采用奖励归一化（"a reward normalization strategy that dynamically tracks the observed changes in makespan and scales the reward into a bounded range"）。
- 训练细节：Step 1–9 流程 + Algorithm 1 伪码（候选动作生成 → 计算 raction 与 rcandidates → 计算指示函数 → 按 δKL 更新 ε → 更新 actor/critic）。

### 4.4 实验
- 实例来源：**公开基准 Brandimarte MK01–MK10**（共 10 个实例，[2]）；按规模分配到三种车间布局：Layout 1（6 机器、2 充电桩，用于 MK01/02/05/07）、Layout 2（10 机器、2 充电桩，MK03/04/08/09）、Layout 3（15 机器、3 充电桩，MK06/10）；充电桩:机器 ≈ 1:5~1:6。
- 参数：AGV 初始满电 43,200；充电功率 1440；载载功率 1200；空载功率 600；距离等于 AGV 运输时间（无量纲）；OC 充电时间设为 10；MBC 时间设为 60。
- 基线：(a) 40 条复合调度规则（两级层次：4 种机器–工件分配策略 × 10 种 AGV 充电策略）；(b) 两种 DRL 方法（DQN-based 与 PPO-based）；(c) 随机版本 CRGPPO-TKL（记作 RN，每层随机选动作）。每个测试实例 20 次独立运行。
- 指标：mean、std、ODR（Optimality Deviation Ratio）：ODR(si) = ( (si − sbest)/sbest ) × 100%（式(44)）；独立双样本 t 检验（显著性 0.05）；另报告收敛所需训练轮数、平均推理时间、复杂度。
- 超参（Table 5）：γ = 0.99；λGAE = 0.95；εclip = 0.2；εmin/εmax = 0.1/0.3；TKL = 0.3；αadjust = 0.05；βentropy = 0.01；Ncandidates = 5；Kepoch = 4；Bmin = 128；ηactor/ηcritic = 3×10⁻⁴ / 1×10⁻³；Ttrain = 500。
- 硬件：**笔记本 Intel Core i5-13500HX (2.5 GHz) + 16 GB DDR4 RAM**（未使用 GPU）；Python 3.9 + PyCharm 2023。

### 4.5 核心数字（照抄原文）
- 摘要："Experimental results demonstrate that the proposed method outperforms composite dispatching rules and mainstream DRL methods across multiple scheduling scenarios, achieving an average improvement of 8.2% and 10.5% in makespan, respectively."
- DRL 对比："CRGPPO-TKL significantly outperforms both the DQN-based and PPO-based methods across 10 MK benchmark instances. It achieves the best performance in 9 out of 10 instances, with MK06 being the only case where PPO slightly outperforms CRGPPO-TKL (ODR = 1.8%). As shown in Figure 9, on average, CRGPPO-TKL achieves an ODR of 0.18%, compared to 8.19% and 10.54% for DQN and PPO, respectively. Furthermore, CRGPPO-TKL exhibits a lower standard deviation of 11.9, in contrast to 10.1 for DQN and a much higher 140.7 for PPO"
- 布局层面（Table 9）："DQN and PPO record average ODRs of 8.5% and 13.3%, and standard deviations of 10.59 and 127.2, respectively."
- 复合规则对比："The most competitive composite actions were Actions 1, 2, 11, and 12, with average ODRs of 6.6%, 12.2%, 17.9%, and 19.8%, and variances of 2.26, 7.05, 13.64, and 13.28, respectively."；"the performance differences are statistically significant (p < 0.05) in all comparisons"
- 最优 AGV 数量："the AGV quantity in Layout 1 is configured as 3 units."；"the AGV quantity in Layout 2 is configured as 4 units."；"the AGV quantity in Layout 3 is configured as 6 units."
- AGV 数量影响："In Layout 1, when the number of AGVs increases from 1 to 6, the average makespan of test sets MK01, MK02, MK05, and MK07 decreases by 227.4, 271.2, 360.7, and 483.1, corresponding to reductions of 82.2%, 83.7%, 63.8%, and 70.7%, respectively."；Layout 2（AGV 2→7）降幅 52.8%、59.1%、38.8%、56.3%；Layout 3（AGV 3→8）MK06、MK10 降幅 65.1%、58.4%。
- 与随机版本对比："in the MK08 instance, RN records a mean completion time of 2951.2 with a standard deviation of 1315.7, whereas CRGPPO-TKL achieves a mean of 653.6 and a standard deviation of only 14"
- 动作分析："the agent strongly favors the NC action. OC occurs more frequently than FC, while FC actions are rarely chosen."；"the majority of job–machine combination decisions fall into the MOR + EAM and MOR + LPT categories, accounting for 55.29% and 42.44%, respectively."
- 敏感性："Increasing TKL from 0.1 to 0.3 resulted in a substantial rise in ODR—particularly in MK03, where ODR rose from 0% to 32.1%"；"smaller values of εclip (e.g. 0.1) slowed convergence and introduced greater variability, whereas larger values (e.g. 0.3) yielded better stability and convergence speed."

### 4.6 自认局限（原文逐字 + 中译）
> "Despite its strengths, the method has several limitations, including assumptions of deterministic parameters, simplified battery models, sensitivity to reward design, and the lack of real-world validation."
中译：尽管有上述优势，该方法仍存在若干局限，包括参数确定性假设、简化的电池模型、对奖励设计的敏感性，以及缺乏真实世界验证。
> "Despite these achievements, some limitations remain. The current model assumes deterministic AGV path planning and does not consider path conflicts or dynamic traffic congestion common in high-density shop floors. Additionally, real-world disturbances such as machine breakdowns, order insertions, and stochastic transport delays have not been incorporated into the scheduling framework. Future work will focus on modeling these disturbances and optimizing scheduling accordingly. More accurate nonlinear battery models, including temperature effects, will also be incorporated to improve simulation realism."
中译：尽管取得了这些成果，仍存在一些局限。当前模型假设 AGV 路径规划是确定性的，未考虑高密度车间常见的路径冲突或动态交通拥堵。此外，机器故障、订单插入、随机运输延误等现实扰动尚未纳入调度框架。未来工作将聚焦于对这些扰动建模并据此优化调度；同时将引入更精确的非线性电池模型（含温度效应）以提升仿真真实性。
> "Although comprehensive empirical modeling data are currently lacking, the incorporation of this effect enhances the practical applicability of the proposed scheduling model and reflects the actual operating characteristics of AGVs. Future research will focus on collecting empirical data to further validate and optimize the model."
中译：尽管目前缺乏完整的经验建模数据，引入该效应（电量–速度劣化效应）仍提升了所提调度模型的实用适用性，并反映了 AGV 的实际运行特征。未来研究将聚焦于收集经验数据以进一步验证与优化该模型。
- 假设第 11 条（隐含局限）："Machine failures, job insertions, and AGV path conflicts are not considered."

### 4.7 可复用
- 公开数据集：**Brandimarte MK01–MK10**，Data Availability Statement 给出地址："The data used in this article are available at GitHub—Adenacademic/Brandimarte-benchmark."
- 开源代码：未提供。
- 可复用评测协议：
  - ODR(si) = ((si − sbest)/sbest) × 100%（式(44)），配合独立双样本 t 检验（0.05）与 20 次独立运行；
  - 三布局（6/10/15 机器，2/2/3 充电桩）与"先定最优 AGV 数量、再固定配置做算法对比"的两阶段实验设计；
  - 复合规则基线构造方式：五层规则组合共 40 条 + 每实例 20 次运行 + ODR 均值/方差比较。
- 可复用公式：奖励 R = Cmax(st) − Cmax(st+1)；CRGPPO-TKL 动态裁剪式(32)–(38)；四段 AGV 速度–电量函数（式(1)，含 α/β/λ 参数）；AGV 电量更新式(13)；充电桩互斥式(14)；完整运输时间式(8)。
- 附录 A 提供全部 40 条复合规则的完整实验结果（可作基线复现参照）。


## 【组 g3】

### 事实卡 g3（3 篇）

> 仅记录原文事实，不做评价。原文引文用英文原样抄录，其后附中文翻译。

---

### 卡片 1

## 1. 题录

- **标题**：A heuristic-assisted deep reinforcement learning algorithm for flexible job shop scheduling with transport constraints
- **作者**：Xiaoting Dong^1,2,3^ · Guangxi Wan^1,2^ · Peng Zeng^1,2^
- **机构**：
  1. State Key Laboratory of Robotics, Shenyang Institute of Automation, Chinese Academy of Sciences, Shenyang 110016, China
  2. Key Laboratory of Networked Control Systems, Shenyang Institute of Automation, Chinese Academy of Sciences, Shenyang 110016, China
  3. University of Chinese Academy of Sciences, Beijing 100049, China
- **通讯**：Guangxi Wan (wanguangxi@sia.cn)；Peng Zeng (zp@sia.cn)
- **期刊**：Complex & Intelligent Systems (2025) 11:210
- **DOI**：https://doi.org/10.1007/s40747-025-01828-6
- **时间线**：Received 21 July 2024 / Accepted 28 February 2025 / Published online 17 March 2025
- **基金**：NSFC [U24A20277, 2467301, 2367301, 2267205]；Natural Science Foundation of Liaoning Province [2024-MSBA-83]；GZB20230805；State Key Laboratory of Robotics of China [2023-Z15]；Fundamental Research Project of SIA [2024JC1K10]
- **数据可得性**：Data available on request from the authors

## 2. 问题设定

- **问题类型**：Flexible Job Shop + AGV 协同调度，文中缩写为 **FJS-AGV**（经典 FJS 调度 + AGV 调度问题合并，作为一个统一问题被建模为 MDP）。
- **资源**：
  - n 个工件 J = {J1,…,Jn}；m 台机器 M = {M1,…,Mm}；v 台 AGV V = {V1,…,Vv}。
  - 每个工件 Ji 含 ni 道连续工序 Oi = {Oi1,…,Oini}，有前后约束；工序 Oij 可在合格机器子集 Mij ⊆ M 上加工。
  - AGV 负责机器间运输，**每台 AGV 可运任意类型工件，但一次只能运一个**（capacity to hold only one at a time）。
  - 车间含原材料库 RW、生产库 PW。
- **目标**：最小化 makespan：
  `Cmax = max{Eini}, i ∈ {1,…,n}`（Eq.1，Eini 为工件 Ji 的完工时间）。
- **决策内容**（原文四项）：(1) 把工序分配到兼容机器；(2) 确定每台机器上的工序加工顺序；(3) 把工序分配到合适 AGV；(4) 确定每台 AGV 上的运输顺序。
- **约束（公式）**：
  - 机器唯一分配：`Σ_{l=1..m} Yi,e,l = 1`（Eq.2）
  - AGV 唯一分配：`Σ_{r=1..v} YTi,e,r = 1`（Eq.3）
  - 运输结束：`ETi,e,r = STi,e,r + UTi,e,r + LTi,e,r`（Eq.4，UT=空载时间，LT=负载时间）
  - 加工开始：`Si,e,l = max{ETi,e,r, MRl}`（Eq.5，MRl 为机器释放时间）
  - 加工结束：`ei,e,l = Si,e,l + Pi,e,l`（Eq.6）
  - 机器能力：`Σ_{i,h} Σ_{e,f} Y^l_{i,e→h,f} * Yi,e,l(t) ≤ 1, l ∈ {1..m}`（Eq.7，任一时刻一台机器只加工一个工件）
  - AGV 能力：`Σ_{i,h} Σ_{e,f} YT^r_{i,e→h,f} * YTi,e,r(t) ≤ 1, r ∈ {1..v}`（Eq.8，任一时刻一台 AGV 只运输一个工件）
- **假设（原文 5 条）**：
  1. 所有工件、机器、AGV 在时刻 0 可用；
  2. 所有工件与 AGV 初始位于 RW；
  3. 每台机器有无限容量缓冲区（工件加工前/后在缓冲区等待 AGV）；
  4. 运输任务完成后 AGV 停靠在当前位置，等待下一任务触发；
  5. 若同一工件的相邻两道工序在同一台机器上完成，则**无需调动 AGV**运输。
- 原文示例实例 `fjsp3`（来自 Kumar et al. benchmark）：5 工件，工序数 {3,3,4,4,5}，每工序 3 台可选机器，2 台 AGV。

## 3. 方法

**算法**：HA-DQN（heuristic-assisted deep Q-network）。核心思想：**DQN 只负责"选哪道工序"，机器与 AGV 由启发式规则（ECT，Earliest Completion Time First）选**，以此降低动作空间维度。

**MDP**：五元组 (S, A, P, R, γ)。

### 状态（5 维特征，原文 "five state features are designed"）
1. 当前 makespan `CM(t) = max{Ei,e,l(t)}`（Eq.9）
2. 当前累计加工时间 `CPT(t) = Σ_{i=1..n} Σ_{e=1..ni} Σ_{l=1..m} Pi,e,l × Yi,e,l(t)`（Eq.10）
3. 平均完工百分比 `ACP(t) = (1/n) Σ_{i=1..n} CPi(t)`，`CPi(t) = cni(t)/ni`（Eq.11–13，cni(t) 为 t 时刻 Ji 已完成工序数）
4. 平均机器利用率 `AMU(t) = (1/m) Σ_{l=1..m} MUl(t)`，`MUl(t) = (Σ_i Σ_e Pi,e,l × Yi,e,l(t)) / CPT(t)`（Eq.14–16）
5. 平均 AGV 利用率 `AAU(t) = (1/v) Σ_{r=1..v} AUr(t)`，`AUr(t) = Σ_i Σ_e (UTi,e,r + LTi,e,r)×YTi,e,r(t) / Σ_r Σ_i Σ_e (UTi,e,r + LTi,e,r)×YTi,e,r(t)`（Eq.17–19）

### 动作空间设计
- 每个决策点的动作 = "一个调度行为"，引导 agent 选择下一道待加工工序。
- **共设计 9 种调度行为/规则作为动作空间**（Table 3）：SJPT（最短加工时间工件）、LJPT（最长加工时间工件）、SOPT（最短加工时间工序）、LOPT（最长加工时间工序）、SJRPT（最小剩余加工时间工件）、LJRPT（最大剩余加工时间工件）、EOST（最早可开工工序）、SMOPS（加工机器与其前驱相同的工序）、SOTT（最短运输时间工序）。
- 机器与 AGV 的选取**不进 DQN 动作空间**，由启发式完成：
  - 机器 `Mk = arg min_{k} ( RTMk(t) + LT_{PJi(t)→Mk} + Pi,e,k )`（Eq.23）
  - AGV `Vs = arg min_{s} ( RTVs(t) + UT_{PVs(t)→PJi(t)} )`（Eq.24）
  - Algorithm 1 / Algorithm 2：先以 25% 概率随机选（`α < 0.25`），否则选上述最小时间者（机器选 `STOi,e,k + Pi,e,k` 最小）。
- 原文表述："the agent's action is a triplet (Oi,e(t), Mk, Vs)"（若直接让 DQN 决策则动作空间爆炸，故用 ECT 消解）。

### 奖励函数（照抄）
```
r(t) = −(Cmax(t) − Cmax(t−1))        (Eq.20)
```
原文："When the discount factor γ = 1, the cumulative reward R = −(Σ_{t=0..T} Cmax(T) − Cmax(0)), where Cmax(0) = 0, so maximizing R and minimizing makespan is equivalent."

### 探索/利用
- ε-greedy（Eq.21）：`a(t) = argmax_a Q(st,at)` with prob. 1−ε；random with prob. ε。
- **自适应 ε**（Eq.22）：`ε(t) = εmax − learn_step(t)/learn_step_max, ε ≥ εmin; 否则 εmin`。

### 网络结构与训练
- **双网络结构**：OnlineNet 选动作，Target Net 评估动作（缓解 Q 值高估）。
- 结构：1 个输入层 + **3 个全连接隐藏层** + 1 个输出层；隐藏层用 ReLU。
- **输入层节点数 = 状态特征数（5）；输出层节点数 = 动作数（9）**。
- 训练：RMSprop 随机梯度法，误差 `E = (1/|A|) Σ_{i=1..|A|} (Q̂i − Qi)²`；目标网络**每 20 步更新**一次（`Q̂ = Q`）；采用经验回放（experience replay）。
- 超参数（Taguchi 正交表标定，180 组合，在 MFJST07 上标定）：**ε = 0.90, γ = 0.75, 学习率 l = 0.005, 目标网络更新阈值 λ = 20**；episode 数 L = 2000，memory size MEM = 300，batch size BS = 64。
- 解编码：解 = 工序向量 θ + 机器向量 ω + AGV 向量 α，各含 Σni 个元素；θi 为工件号，ωi/αi 为分配给工序 θi 的机器/AGV 号；解码按操作向量给出的顺序在最早就绪时间上安排运输+加工。

## 4. 实验

- **硬件/软件**：Python 3.7，Windows 10，Intel Core i7 3.0 GHz PC，10 GB RAM。（原文未报告 GPU）
- **算例（5 个 benchmark 数据集）**：
  1. **小规模**（Deroussi & Norre）：5–8 工件，13–20 工序，8 台机器；实例 fjsp1–fjsp10。
  2. **中规模**（Seyed Mahdi Homayouni）：5–12 工件，15–48 工序，6–8 台机器；MFJST01–MFJST10。
  3. **大规模**（Homayouni）：10–15 工件，100–225 工序，11–18 台机器；mt10ct1, mt10cc, mt10x, mt10xxxt, mt10xyz, setb4c9, setb4cc, setb4xyz, seti5c12, seti5cc, seti5xyz。
  4. **泛化测试**：Brandimarte MK01–MK10（经 Homayouni & Fontes 加入运输时间），10/15/20 工件，55–240 工序，4–15 台机器。
  5. **对比 DRL 方法**：MKT01–MKT10（数据来自文献[51]），AGV 数量服从 U(0.8m, 1.2m)。
  - 所有实例均为 **2 台 AGV**（除第 5 组外）；每个 benchmark 含 10 个实例，重复 10 次取最好值。
- **基线**：
  - 小规模：ILS[44]、SBA[47]、TS[48]、BRKGA[45]、GA[49]（+ LB 下界）。
  - 中/大规模：LAHC[45]（H=1000）、GA[49]；大规模另报 LAHC 求解时间。原文：LAHC 求解 seti5xyz 需 2.5 h 才得到可行解。
  - 消融：传统 DQN、PPO（二者同时选工序+机器+AGV），另报 Cov（变异系数）。
  - DRL 对比：PPO+GNNs（原为 FJSP 设计[52]，由[51]加 Nearest Vehicle Selection 扩展到 FJS-AGV）、HGS（Heterogeneous Graph Scheduler）。
- **指标**：RPD（Eq.25）`RPD = (C_HA-DQN_max − C_alg_max) / C_alg_max`；表中负号表示 HA-DQN 优于对方。另有 Cov（coefficient of variation）与求解时间 T(seconds)。

## 5. 核心数字（照抄原文）

- 摘要："the HA-DQN algorithm yields a significant **12.63%** reduction in makespan compared with that when traditional heuristics are employed"（大规模基准）。
- 大规模（Table 6）：mt10ct1 Cmax 992→969（T 382.7s→81.36s，RPD −0.0232）；**mt10xyz 934→816（−12.63%）**；**mt10xxxt 983→872（−11.29%）**；**seti5cc 1358→1209（−10.97%）**；seti5xyz 1347→1208（−10.31%，T 8905.7s→186.75s）。
- 小规模（Table 4）：HA-DQN 在 6 个实例取得最优、其余 4 个近优；与 ILS 的最大差距达 **25.33%**。
- 中规模（Table 5）：HA-DQN 在 MFJST01–05 取得最优；与 LAHC 的最大差距仅 **1.22%**；**MFJST08 相比 GA 节省 23.39% 的 Cmax**（GA 1214 → HA-DQN 930）。
- 泛化（Fig.11）：**MK07 完工时间比 LAHC 减少 14.15%**；最复杂的 MK10 训练后模型仅需 **95 s**，而 LAHC 需 **10768.30 s**。
- 消融（Table 7）：HA-DQN 平均 Cmax 在 MK06 比 DQN 低 **7.74%**，在 MK02 比 PPO 低 **4.78%**；MK10 上 HA-DQN 的 Cov = **0.0102**，DQN = 0.0319，PPO = 0.0234。
- 与 DRL 方法对比（Table 8）：与 PPO+GNNs 相比 **RPD 达 42.28%**；与 HGS 相比在 MKT07 上 **Cmax 降低 38.79%**（HGS 348 → HA-DQN 213，T 1.12s→1.07s）。

## 6. 自认局限（原文逐字 + 中文翻译）

> "Currently, our research considers only static scheduling when solving the FJS-AGV problem, which limits the application of the proposed HA-DQN algorithm in dynamic scheduling environments. In the future, we plan to expand our work on the FJS-AGV problem to incorporate complex constraints such as dynamic scheduling, multiobjective optimization, and energy-aware scheduling. Furthermore, we intend to explore the integration of DRL with effective evolutionary algorithms. Additionally, we will investigate learning strategies and neural network architectures to improve the performance of the proposed algorithm."

中文翻译：目前，我们在求解 FJS-AGV 问题时只考虑了静态调度，这限制了所提出的 HA-DQN 算法在动态调度环境中的应用。未来，我们计划把工作扩展到 FJS-AGV 问题中更复杂的约束，例如动态调度、多目标优化和能耗感知调度。此外，我们打算探索将 DRL 与有效的进化算法相结合。同时，我们将研究学习策略和神经网络架构以提升所提算法的性能。

（另一处原文对方法短板的表述："when solving large-scale FJS-AGV problems, it takes hours to search for the optimal solution"（指 LAHC 类方法）——不属自认局限，仅作背景。）

## 7. 可复用

- **代码**：无论文提供代码/仓库链接；数据 "available on request from the authors"。
- **数据集（可复现来源）**：
  - 小规模：Deroussi & Norre（文献[44]）；
  - 中/大规模 + MK 系列运输时间版本：Homayouni & Fontes（文献[45]、[10]、[15]）；
  - MKT01–MKT10：文献[51]；
  - 原始 MK：Brandimarte（文献[50]）。
- **评测协议**：RPD 公式（Eq.25，负值=本方更优）；每个实例重复 10 次；同时报告 makespan、求解时间、Cov。
- **可复用公式**：MDP 状态 5 特征（Eq.9–19）、奖励 `r(t)=−(Cmax(t)−Cmax(t−1))`（Eq.20）、自适应 ε（Eq.22）、ECT 机器/AGV 选择（Eq.23/24）、运输时间分解 `ET = ST + UT + LT`（Eq.4）、`ST = max{ET, MR}`（Eq.5）。
- **可复用启发式算子**：Algorithm 1（机器选择，含 25% 随机扰动）/ Algorithm 2（AGV 选择，含 25% 随机扰动）——可直接作为 DRL 的动作降维辅助模块。

---

### 卡片 2（重点）

## 1. 题录

- **标题**：Matrix manufacturing system layout and scheduling via graph neural network and multi-action deep reinforcement learning
- **作者**：Tong Zhu^a^, Xuemei Liu^a,\*^, Yanbin Yu^b^, Ling Fu^b^
- **机构**：a. School of Mechanical Engineering, Tongji University, Shanghai, 201804, China；b. Siemens Technology, Siemens Ltd., China
- **通讯**：liuxuemei@tongji.edu.cn (X. Liu)
- **期刊**：Journal of Manufacturing Systems 82 (2025) 239–253
- **DOI**：https://doi.org/10.1016/j.jmsy.2025.06.005
- **时间线**：Received 24 June 2024; Received in revised form 13 March 2025; Accepted 4 June 2025; Available online 19 June 2025
- **版权**：© 2025 The Society of Manufacturing Engineers. Published by Elsevier Ltd.
- **关键词**：Matrix manufacturing system; Layout and scheduling; Heterogeneous graph neural network; Multi-action deep reinforcement learning; Proximal policy optimization

## 2. 问题设定

- **问题类型**：**MMSLS**（Matrix Manufacturing System Layout and Scheduling）。原文定性："can be essentially regarded as an extension of the flexible job-shop scheduling problem (FJSP) that incorporates dynamic transportation delays"。
- **资源/要素**：n 个产品 P={P1,…,Pn}（产品 Pi 含工序 Oi={Oi1,…,Oij}，工序有先后约束）；m 台待布置工位 W={W1,…,Wm}；**m 个待分配位置 L={L1,…,Lm}**。工位与位置是**一一对应**（workstations are assigned to specific locations in a one-to-one correspondence）。
- **运输**：工序 Oij 的运输时间 rtijk 由"产品工序顺序 + 工位分配 + 位置分配"共同决定，即由工位间距离决定；`ptijk = pijk + rtijk`。原文假设 (5)："All products have the same transportation speed, and the transportation time is independent of the product, only depending on the transportation distance between the starting workstation and the destination workstation."
- **目标**：最小化 makespan，`Etmax = max_{i,j}{Etij}`。
- **约束**：工位排他性（每个工位一次只能加工一道工序、每道工序一次只能在一台工位上加工，非抢占 non-preemptive）、工艺路线优先级约束（process plan precedence）；耦合一阶：位置分配 × 工序-工位匹配 × 加工顺序 → 超多项式复杂度，NP-hard。
- **假设（原文 6 条）**：(1) 产品相互独立，所有产品与工位在时刻 0 可用；(2) 非抢占分配；(3) 所有产品的工序顺序预先给定；(4) 每工位一次一道工序、每工序一次一个工位；(5) 所有产品运输速度相同，运输时间只依赖起止工位间距离；(6) 不考虑突发事件与未知/不确定加工信息。

### 图表示（关键）
- 传统析取图 G = (O, C, D)：O = {Oij} ∪ {S, E}，C 为同一产品相邻工序的合取弧，D 为连接可由同一工位加工的一对工序的析取（无向）弧。
- **本文定义三节点异构图 `HG = (O, W, L, C, ξ)`**：在原析取图基础上加入工位节点集 W 与位置节点集 L；**析取弧集 D 改为 ξ，ξ 包含 operation-workstation (O-W) 弧与 workstation-location (W-L) 弧**。
- 原文理由："the traditional disjunctive graph G also struggles to express characteristics such as varying processing times for the same operation on different compatible workstations"——故同一工序在不同兼容工位上的加工时间 pijk 改由 **O-W 弧的特征**表示。

### 逐维节点/边特征定义（原文原文照抄）
> "This paper refers to the features of heterogeneous graph nodes and edges defined by [48,51] and defines the initial features of operation, workstation, location, and O-W arcs for MMSLS as follows:"

**Raw features of operation nodes（工序节点，6 项）**
- Status: expressed in binary, the value for Oij is 1 if it has been scheduled; otherwise, it is 0.（二值：已排产=1）
- Number of available workstations for Oij: |Nt(Oij)|.（Oij 的可用工位数）
- Processing time and transportation time: the sum of the processing time and transportation time of Oij.（加工时间 + 运输时间之和）
- Number of unscheduled operations in the product.（该产品中未排产的工序数）
- Process start time: the estimated or actual start time of Oij.（估计或实际的开工时间）
- Product completion time: the time to complete all operations of a product.（产品完工时间）

**Raw features of workstation nodes（工位节点，4 项）**
- Status: expressed in binary, the value for Wk is 1 if it has been allocated; otherwise, it is 0.（二值：已分配=1）
- Available time: the time when Wk is idle and can handle a new operation.（空闲可接新工序的时刻）
- Number of operations that workstation (Wk) can process: |Nt(Wk)|.（该工位能加工的工序数）
- Workstation utilization: ratio of Wk processing time to total production time.（工位利用率 = Wk 加工时间 / 总生产时间）

**Raw features of location nodes（位置节点，2 项）**
- Status: expressed in binary, the value for Ll is 1 if it has been allocated; otherwise, it is 0.（二值：已分配=1）
- Number of workstations that can be arranged per location: |Nt(Ll)|.（每个位置可布置的工位数）

**O-W arcs（O-W 弧，1 项）**
- The processing time of Oij on the Wk, without transportation time.（Oij 在 Wk 上的加工时间，**不含运输时间**）

**W-L 弧**：原文**未定义显式 W-L 弧特征**。Table 1 符号表中给出的是 `hij`（工序节点特征向量）、`mk/ml`（工位节点特征向量）、`nm/nl`（位置节点特征向量）、`μijk`（**O-W 弧的特征向量**）——没有 W-L 弧的特征向量符号。

**嵌入维度**：Table 3 给出 "Dimensions of operation nodes = 8 / Dimensions of workstation nodes = 8 / Dimensions of location nodes = 8"；正文亦称 "The final **8-dimensional** embedded feature ... denoted as n′l, m′k, h′ij"（**最终为 8 维嵌入**，原始维度论文未逐一给出）。

### 位置/布局信息如何进入网络（汇总）
1. **作为独立节点类型**：位置 L 是与工序 O、工位 W 并列的第三类节点，带 2 维原始特征（status、|Nt(Ll)|）。
2. **作为注意力计算的一端**：GAT 注意力系数显式含 W→L 与 L→L：
   - `e(Oij,Wk) = LeakyReLU[a^T (M_O h_ij ‖ M_W m_k)]`（Eq.1）
   - `e(Wl,Wk) = LeakyReLU[a^T (M_W m_l ‖ M_W m_k)]`（Eq.2）
   - `e(Wk,Ll) = LeakyReLU[a^T (M_W m_k ‖ M_L n_l)]`（Eq.3）
   - `e(Lm,Ll) = LeakyReLU[a^T (M_L n_m ‖ M_L n_l)]`（Eq.4）
   - softmax 归一化（Eq.5）；三套节点共享线性变换矩阵 M_O / M_W / M_L。
3. **三阶段嵌入的顺序**（决定位置信息何时进入）：第一阶段：工序节点 + 工位节点 → 位置节点（更新位置嵌入）；第二阶段：工序节点 + 位置节点 → 一阶相邻工位节点（更新工位嵌入）；第三阶段：位置节点 + 工位节点 + 工序节点 → 工序节点。
   - 位置节点嵌入：`m_ijk = σ[ Σ_{Oij∈Nt(Wk)} α(Oij,Wk)(h_ij + μ_ijk) + α(Wl,Wk) m_k ]`（Eq.6）；`n^t_l = σ[ Σ_{Wk∈Nt(Ll)} α(Wk,Ll) m_ijk + α(Lm,Ll) n_l ]`（Eq.7）
   - 工位节点嵌入：`m_ijk1 = σ[ Σ_{Oij∈Nt(Wk)} β(Oij,Wk)(h_ij + μ_ijk) ]`（Eq.8）；`m^t_k = σ[ Σ_{Ll∈Nt(Wk)} β(Ll,Wk) m_ijk1 + α(Wl,Wk) m_l ]`（Eq.9）
   - 工序节点嵌入（改用 MLP0…MLP5，各含 2 层 128 维隐层 + ELU）：
     `h^t_ij = MLP0( ELU[MLP1(h_i(j−1))] ‖ ELU[MLP2(h_ij)] ‖ ELU[MLP3(h_i(j+1))] ‖ ELU[MLP4( Σ_{Wk∈Nt(Oij)} m^t_k )] ‖ ELU[MLP5( Σ_{Ll∈Nt(Wk)} n^t_l )] )`（Eq.10）
   - 图级嵌入（均值池化）：`H^{W,O}_G = ( (1/|W|)Σ_{Wk∈W} m′_k ‖ (1/|O|)Σ_{Oij∈O} h′_ij )`（Eq.11）；`H^{L,W}_G = ( (1/|L|)Σ_{Ll∈L} n′_l ‖ (1/|W|)Σ_{Wk∈W} m′_k )`（Eq.12）
4. **运输时间经工序节点特征进入**：工序节点原始特征含 "processing time and transportation time (sum of the two)"，其中 `rt_ij` 是到下一可用工位的估计运输时间（由位置距离决定）；论文说明初始 `rt_ij` 由"工位号与位置号一致"（W1–L1, W2–L2, …, Wm–Lm）的状态导出。
5. **作为第二条动作子策略**：位置由 `π(a_{W-L}|s_t)` 直接选出（见下）。

### MDP 定义（六元组 (S, A, P, R, γ, π)）
- **State**：`s_t` 由 `s^{o-w}_t`（工序状态 + 工位状态）与 `s^{w-l}_t`（位置是否已分给工位）组成；`s_0` 为按分布随机生成的实例。
  - 开工时间规则：若是首工序 `Stij = Iwk(t)`；若前驱已排产且加工于 Wk′：`Stij = max{Et_i(j−1) + rt_ijk, Iwk(t)}`；否则 `Stij = max{St_i(j−1) + p_i(j−1) + rt_ij, Iwk(t)}`，其中 `rt_ijk = 0` 当相邻两工序选同一工位。
  - `p_ij = Σ_{Wk∈Wij} p_ijk / |Wij|`（平均加工时间）；`rt_ij = [ Σ_{Wk′∈Wi(j−1)} ( Σ_{Wk∈Wij} rt_ijk / |Wij| ) ] / |Wi(j−1)|`（估计运输时间）。
- **Action**：`A = A_{O−W} × A_{W−L}`，二维；先选 `a_{O−W}=(Oij,Wk)`（把工序与工位绑定，同时解决工序排序问题），再基于已选工位 Wk 计算所有可用位置 Ll 的优先指数并选 `a_{W−L}=(Wk,Ll)`。**所有 m 个 W-L 分配完成后停止位置动作**；一旦 Wk 被分配过位置，后续不再改派（"Wk follows its initially assigned location and is not reassigned"）。
- **Transition**：保留 `ξijk`（Oij-Wk）与 `ξkl`（Wk-Ll）两条析取弧，删除 Oij 的其余 O-W 弧以及 Wk、Ll 的其余 W-L 弧，完成析取图更新。
- **Reward**（照抄）：`Rt = Cmax(s_t) − Cmax(s_{t+1})`；`γ = 1`，故调度完成时累计回报 `G = Cmax(s_0) − Cmax`，且 `Gmax = Cmax(s_0) − min Cmax`。
- **Policy**：`π(a_{O−W}, a_{W−L}|s_t)` 由两个随机子策略 `π(a_{O−W}|s_t)` 与 `π(a_{W−L}|s_t)` 组成。
  - `P(a_{o−w}, s^{o−w}_t) = MLP_O[ h′_ij ‖ m′_k ‖ H^{W,O}_G ]`（Eq.13）
  - `P(a_{w−l}, s^{w−l}_t) = MLP_L[ m′_k ‖ n′_l ‖ H^{L,W}_G ]`（Eq.14）
  - 再经 softmax 得 π（Eq.15/16）；MLP_O 与 MLP_L 结构相同（L_H 层 d 维隐层 + tanh）但**不共享参数**。

### 训练
- 算法：**PPO**（actor-critic，policy gradient）；选择 policy-based 而非 value-based 的理由是状态空间连续、特征高维。
- 流程：共 **1000 次迭代**；每 10 次迭代在**含 100 个实例的验证集**上评估策略；**每 10 次迭代替换一次 20 个训练实例**。
- 超参数（Table 3）：并行实例数 20；迭代数 1000；工序/工位/位置节点嵌入维度均 8；MLP 隐层 128 维；MLP 隐层数 2；优化器 Adam；学习率 2×10⁻⁴；Clip rate 0.2；折扣因子 1；熵奖励系数 1×10⁻²；PPO optimization epochs 3；value loss 系数 0.5。（**HGNN 堆叠层数 L_H 在文中仅为符号，未给出具体数值**。）
- 测试策略：随机采样（MMSLS-S，N_s = 100 实例）与贪心（MMSLS-G）。

## 3. 实验

- **硬件/框架**："The algorithms and MADRL environment were developed using PyTorch. The hardware ... is the 13th-generation Intel Core i9-13900K CPU and NVIDIA RTX A5000 GPU."
- **随机实例（Table 2，8 种规模）**，`U(a,b)` 为 [a,b] 上均匀整数：
  | Size (n×m×m) | \|Oi\| | \|Wij\| | Pij | rtij |
  |---|---|---|---|---|
  | 10×9×9 | U(5,8) | U(2,6) | U(4,40) | [0,8] |
  | 15×9×9 | U(5,8) | U(2,6) | U(4,40) | [0,8] |
  | 15×15×15 | U(6,10) | U(2,10) | U(4,40) | [0,12] |
  | 20×15×15 | U(6,10) | U(2,10) | U(4,40) | [0,12] |
  | 30×20×20 | U(8,12) | U(2,14) | U(4,40) | [0,14] |
  | 40×20×20 | U(8,12) | U(2,14) | U(4,40) | [0,14] |
  | 60×40×40 | U(10,14) | U(2,28) | U(4,40) | [0,22] |
  | 100×40×40 | U(10,14) | U(2,28) | U(4,40) | [0,22] |
  - 训练用两个较小规模：**10×9×9 与 15×15×15**；10×9×9 训练的模型直接用于 15×9×9、20×15×15、30×20×20、40×20×20；15×15×15 训练的模型用于 60×40×40、100×40×40。
- **基线**：MMSLS-G（贪心）、DRL-R（随机策略 DRL）、4 条 PDR：FIFO、MOR（most operations remaining）、MWKR（most work remaining）、SPT（shortest processing time）。
- **公共基准**：
  - **Hurink**：从 Vdata_la 选 **40 个**基准实例（Vdata_la1–la40），按规模分 8 组、每组 5 个；为适配 MMSLS，**加入与工位等量的位置点并指定任意两点间距离**；对比 RGA（regular GA）、2SGA（two-stage GA）、MOR、UB（文献最优）。
  - **Behnke**：分 4 组（50×20、100×20、50×40、100×40），用 15×15×15 模型测试；对比 FIFO+EET、MOR+EET、MWKR+EET、UB (60 min)。
- **案例研究**：10 类产品 P、9 台工位 W、9 个位置 L；每台工位最多加工 4 种工序类型；Table 8 给出每个产品的工序顺序（工序类型 a–i），Table 9 给出 9×9 位置间运输时间矩阵。
- **指标**：平均 Cmax（100 个测试用例）、Gap（相对 MMSLS-S 的相对差）、Time（每用例平均计算时间）；公共基准用相对 UB 的偏差百分比。

## 4. 核心数字（照抄原文）

- **Table 5（随机实例，100 用例均值）**：
  - 10×9×9：MMSLS-S 222.2（Gap 0.00%）/ MMSLS-G 236.2（6.30%）/ DRL-R 258.5（16.34%）/ MOR 303.6（36.63%）/ FIFO 314.2（41.40%）/ MWKR 298.6（34.38%）/ SPT 486.2（118.81%）；时间 1.09 / 0.67 / 0.81 / 0.10 / 0.08 / 0.11 / 0.09 s。
  - 15×9×9：252.4 / 262.7 / 326.13（**29.2%**）/ 315.8（25.12%）/ 400.5（58.68%）/ 371.8（47.31%）/ 519.8（105.94%）。
  - 15×15×15：297.3 / 315.3 / 330.3 / 375.3 / 423.0 / 390.2 / 588.6。
  - 20×15×15：293.1 / 307.8 / 364.9 / 348.9 / 438.1 / 412.1 / 605.6。
  - 30×20×20：413.5 / 453.5 / 500.3 / 500.3 / 612.1 / 575.0 / 1367.5（SPT gap **230.71%**）。
  - 40×20×20：574.3 / 617.5 / 635.26 / 667.5 / 926.3 / 941.4 / 2489.7（SPT **333.52%**）。
  - 60×40×40：633.5 / 645.06 / 671.38 / 709.9 / 824.0 / 848.6 / 2856.3（SPT **350.88%**）；时间 9.54 / 7.87 / 8.57 / 4.15 / 4.57 / 4.26 / 3.89 s。
  - 100×40×40：823.64 / 848.04 / 947.8 / 962.52 / 1081.27 / 986.02 / 3602.02（SPT **337.33%**）；时间 14.05 / 13.63 / 18.65 / 6.35 / 5.86 / 5.22 / 6.89 s。
- **Table 6（Hurink Vdata_la，Ave.Gap 行）**：MMSLS 列记 "-"；**MMSS 1.53%、RGA 2.03%、2SGA 0.71%、MOR 5.98%、UB 0.00%**。（MMSLS 每行格式为 `值 [相对 MMSS 的百分比]`，MMSS/RGA/2SGA/MOR 为 `值 (相对 UB 的百分比)`；例：Vdata_la1 MMSLS 586 [2.09%]，MMSS 574 (0.70%)，RGA 577 (1.23%)，2SGA 572 (0.35%)，MOR 618 (8.42%)，UB 570*。）
- **Table 7（Behnke，Ave.Gap 行）**：MMSLS 列 "-"；**MMSS 15.88%、FIFO+EET 27.93%、MOR+EET 55.96%、MWKR+EET 67.85%、UB(60 min) 0.00%**。（例：实例 1 MMSLS 399 [22.39%]，MMSS 326 (25.87%)，FIFO+EET 355 (37.07%)，MOR+EET 492 (89.96%)，MWKR+EET 519 (100.39%)，UB 259*。）
- **其它照抄**：
  - 摘要："not only outperforms manually crafted heuristic scheduling rules in solution quality but also exceeds metaheuristic algorithms in computational velocity"。
  - 6.5 节："our proposed method shows a significant advantage in runtime, **solving each case in less than 1.5 s**"（对比 RGA/2SGA 每实例共 1600 次实验运行、每次 1–30 min）。
  - 结论 (1)："Especially in the 15 × 9 × 9 instance, the MMSLS outperforms DRL-R by more than **29.2 %**."
  - 结论 (3)："reducing makespan by over **12 %** compared to MMSS without considering layout when the number of workstations exceeds 15"。
  - 6.5 节末："in smaller-scale cases (e.g., 10×5, 15×5, 20×5), the results of the MMSLS scheme considering transportation time are very close to those of MMSS ... as the problem size increases, the gap ... becomes progressively larger, especially when the number of workstations exceeds 15, and **the average difference has exceeded 12 %**."

## 5. 自认局限（原文逐字 + 中文翻译）

> "Despite the promising results, this work has several limitations. Firstly, the framework lacks an optimized material transportation system, which is a key element of MMS scheduling. Future research should focus on incorporating a comprehensive transportation model to enhance material flow and overall scheduling performance. Additionally, real-world complexities, such as downtime, maintenance, and irregular order arrivals, were not fully addressed, necessitating further adaptation of the framework to meet practical MMS requirements. Finally, although the end-to-end DRL framework is adaptable, its high computational complexity, particularly for large-scale problems, may hinder practical deployment. Exploring efficiency improvements, such as parallel computing, remains a valuable direction for future research."

中文翻译：尽管结果喜人，本工作仍有若干局限。首先，该框架缺少一个经过优化的物料运输系统，而这是 MMS 调度的关键要素。未来研究应聚焦于纳入完整的运输模型，以改善物料流与整体调度性能。此外，现实世界的复杂性（如停机、维护、订单不规则到达）未被充分处理，需要进一步改造框架以满足实际 MMS 需求。最后，尽管该端到端 DRL 框架具有适应性，但其高计算复杂度（尤其在大规模问题上）可能阻碍实际部署。探索并行计算等效率改进手段仍是未来研究中有价值的方向。

（另一处原文相关表述（6.5 节）："Since MMSLS is being proposed for the first time and no prior studies have addressed this problem, we are unable to compare the algorithm's efficiency with any existing work."）

## 6. 可复用

- **代码/数据**：论文未提供代码仓库或数据下载地址（数据集为按 Table 2 分布随机生成；公共基准为 Hurink / Behnke）。
- **评测协议（可复现）**：
  - 随机实例：按 Table 2 的 8 种规模与分布生成，训练 10×9×9 与 15×15×15 → 零样本迁移到更大规模；每规模 100 个测试用例；测两种解码（贪心 MMSLS-G、采样 N_s=100 的 MMSLS-S）；报告平均 Cmax、相对 MMSLS-S 的 Gap、单例平均运行时间。
  - 公共基准改造方式：在 Hurink/Behnke 的 FJSP 实例上"加入与工位数等量的位置点，并指定每对位置点之间的距离"，即可把 FJSP 基准转成 MMSLS 基准。
- **可复用公式**：异构图定义 HG=(O,W,L,C,ξ)；GAT 注意力 4 式（Eq.1–4）；softmax 归一化（Eq.5）；三阶段节点嵌入（Eq.6–10）；图级均值池化嵌入（Eq.11–12）；双子策略动作概率（Eq.13–16）；奖励 `R_t = Cmax(s_t) − Cmax(s_{t+1})`、γ=1、`G_max = Cmax(s_0) − min Cmax`；开工时间递推 `Stij = max{Et_i(j−1) + rt_ijk, Iwk(t)}`。
- **可复用数据结构**：工序/工位/位置三类节点的原始特征清单（6 / 4 / 2 项）+ O-W 弧特征（1 项，不含运输时间的加工时间）——可直接作为 FJSP 类问题构图模板。
- **案例数据**（Table 8/9 可直接抄用）：10 产品 × 9 工位 × 9 位置的工序路线表；9×9 位置间运输时间矩阵（0/2/4/6/8，对称，L1=0、L2=2、L3=4、L4=2、L5=4、L6=6、L7=4、L8=6、L9=8 的第一行）。

---

### 卡片 3

## 1. 题录

- **标题**：A Digital Twin-driven Deep Reinforcement Learning Approach for Smart Workshop Scheduling
- **作者**：Jianlong Qian（单一作者："The sole author designed, analyzed, interpreted and prepared the manuscript."）
- **机构**：School of Mechanical Engineering, North China University of Water Resources and Electric Power, Zhengzhou 450045, China
- **通讯邮箱**：marks@88.com
- **期刊**：Journal of Engineering Research and Reports, Volume 28, Issue 3, Page 135–146, 2026; Article no.JERR.154509; ISSN: 2582-2926
- **DOI**：https://doi.org/10.9734/jerr/2026/v28i31826
- **时间线**：Received 21/12/2025；Accepted 11/03/2026；Published 17/03/2026
- **引用格式（原文给出）**：Qian, J. (2026). ... Journal of Engineering Research and Reports, 28(3), 135–146.
- **栏目**：Original Research Article；Open Peer Review History 链接：https://pr.sdiarticle5.com/review-history/154509

## 2. 问题设定

- **问题类型**：Job Shop Scheduling Problem（JSP），偏**动态调度（DJSSP）**；原文关键词 "job shop scheduling"。
- **资源**：机床/设备（Fig.4 中标注 M1…Mn 与工件 J1…Jj）；无 AGV、无机器人、无运输资源建模。
- **目标（多目标，原文列举）**："minimizing manufacturing time, maximizing resource utilization, or minimizing costs"；摘要另提 "maximizing resource utilization and minimizing production costs"；4.3 节又列 "minimizing completion time, cost and energy consumption"。摘要中量化报告的指标为 makespan、resource utilization、equipment utilization rate。
- **约束/扰动**："practical manufacturing challenges like machine breakdowns and urgent order insertions"（机器故障与紧急插单）；动态调度触发条件为设备可用性预测与扰动监测（Fig.4：Device availability prediction based on twin data / Disturbance monitoring based on virtual and real interaction → Rescheduling is triggered → 生成重调度方案 → 基于仿真的方案评估）。
- **数字孪生侧**：静态虚拟模型（Geometric dimensioning / Physical property / Behavioral model / Rule model）+ 动态生产数据（设备运行数据：开关机状态、机器效率与产能、运行速度、故障与停机次数；人员数据；生产计划数据）→ Fig.3。

## 3. 方法

- **算法**：**改进 DQN**——"an improved DQN algorithm (with dual networks and prioritized experience replay)"；原文亦称 "dual DQN, preferential experience playback"（双网络 + 优先经验回放）。
- **整体框架**：DT（高保真实时同步虚拟车间）为 DRL 提供"risk-free simulation bed"进行训练与验证；DRL agent 依据实时孪生数据做动态调度决策（"which device to assign a task to, or when to perform equipment maintenance"），并在线持续学习与调整孪生模型。
- **网络结构**：**未给出**具体层数/节点数/激活函数。仅表述："the algorithm will explore the use of advanced neural network architectures, such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs), to better handle temporal dependencies and spatial features in shop floor scheduling."
- **状态定义**：**未给出**。原文仅："This study will deeply analyze the dynamic characteristics and multi-objective requirements of shop floor scheduling, and **define** the state space, action space and reward function suitable for deep reinforcement learning applications."（即状态/动作/奖励被描述为"将要定义"，论文未列出具体定义或公式。）
- **动作空间设计**：**未给出**具体设计（无动作定义、无动作掩码/候选集描述）。
- **奖励函数**：**未给出**公式。仅："use reward shaping or Pareto optimization to deal with multi-objective problems"；"The improved DQN algorithm will use multiple value functions to estimate the long-term returns of different goals, and combine a trade-off mechanism to balance the conflicts and synergies between the goals."
- **给出的唯一算法伪码（Algorithm 1，基础 DQN，Mnih 式）**：
  1. Initialize playback memory sequence D with capacity N
  2. Initialize the randomly weighted action-value network Q with weights θ
  3. Initialize the randomly weighted action-value network Q̂ with weights θ⁻ = θ
  4. For episode = 1, M do
  5.   Initialize feature vectors ϕ1 = s1
  6.   For t = 1, T do
  7.     Choose a random action at with probability ε, Otherwise select at = argmax_a Q(ϕ(st), a; θ)
  8.     Execute at, Observe reward rt and the new status s_{t+1}
  9.     ϕ_{t+1} = s_{t+1}
  10.    Store the combination (ϕt, at, rt, ϕ_{t+1}) into D
  11.    A small batch of (ϕt, at, rt, ϕ_{j+1}) combinations is randomly selected from D
  12.    Compute yj = rj if episode stops at step j+1；otherwise rj + γ max_{a′} Q̂(ϕ_{j+1}, a′; θ⁻)
  13.    Update θ with a gradient descent method of error (yj − Q(ϕj, a; θ))²
  14.    Sync the network Q̂ = Q every C step
  15.   End For
  16. End For
- **数字孪生车间模型构建**：数据与模型融合驱动；关键技术在 "rapid analysis based on model reduction, virtual-real interaction, and modeling methods of mechanism and data fusion"。

## 4. 实验

- **实例与规模**：**未报告**。原文："simulation experiments are carried out on the standard workshop scheduling benchmark problem"——未给出基准名称、实例数、工件/机器规模。
- **基线**：Traditional Rules (FIFO) 与 Genetic Algorithm (GA)（见 Table 1）；另与 "Simple Dispatching Rules (SDR)" 相比（正文提到，表中未单列）。
- **指标**：Makespan (min)、Resource utilization (%)、Avg. waiting time (min)。
- **硬件**：**未报告**（无 CPU/GPU/内存/软件版本信息）。
- **实验表格（Table 1，全表照抄）**：

  | Scheduling method | Makespan (min) | Resource utilization | Avg. waiting time (min) |
  |---|---|---|---|
  | Traditional Rules (FIFO) | 485 | 74.2% | 42.5 |
  | Genetic Algorithm (GA) | 442 | 81.5% | 36.8 |
  | Proposed DT-DRL (DQN) | 414 | 86.5% | 29.4 |

（表题：Performance comparison of different scheduling algorithms in an intelligent workshop environment）

## 5. 核心数字（照抄原文）

- 摘要："achieving a **14.6% reduction in makespan** and a **12.3% improvement in resource utilization** compared to traditional rules"；"maintaining an **86.5% equipment utilization rate** under stochastic disturbances"。
- 正文（Table 1 后）："The proposed method outperforms traditional rules by **reducing the makespan by 14.6% and waiting time by 30.8%**."
- 结论："These innovations ... provide a measurable performance gain, specifically achieving an **average production efficiency increase of over 14%**."

## 6. 自认局限（原文逐字 + 中文翻译）

> "While the DT-DRL model significantly improves efficiency, practical risks such as data synchronization latency and model over-fitting must be addressed through edge computing and transfer learning. Future research will expand this framework to Multi-Agent Reinforcement Learning (MARL) for collaborative scheduling."

中文翻译：尽管 DT-DRL 模型显著提升了效率，但数据同步时延与模型过拟合等实际风险必须通过边缘计算与迁移学习来应对。未来研究将把该框架扩展到多智能体强化学习（MARL）以实现协同调度。

（该文仅此一处集中陈述局限；全文其余部分未设独立的 Limitations 章节。）

## 7. 可复用

- **代码**：未提供。
- **数据集**：未提供（未指明所"标准车间调度基准问题"的具体来源与文件）。
- **评测协议**：未给出协议细节（无实例生成方式、无重复次数、无随机种子、无评价公式）；仅 Table 1 的三行对比结果。
- **可复用内容**：
  - DQN 伪码（Algorithm 1，经典双网络 + 经验回放 + 周期性目标网络同步），可作为基础实现模板。
  - DT 增强动态调度流程图（Fig.4）的环节划分：孪生建模与校正 → 仿真与交互 → 基于孪生数据的设备可用性预测 → 扰动监测 → 触发重调度 → 生成重调度方案 → 基于仿真的方案评估。
  - DT 车间模型的要素清单：静态虚拟模型（几何尺寸、物理属性、行为模型、规则模型）+ 动态数据（设备运行数据 / 人员数据 / 生产计划数据）。
  - 无公式、无状态/动作/奖励定义可供直接复用。


## 【组 g4】

### 事实卡 G4 — 5 篇 AGV/调度论文精读

> 来源目录：`D:\research\DeepReinforcementLearningScheduling\references\`
> 提取工具：PyMuPDF 1.28.0（文本抽取自 PDF 全文，公式为 PDF 文本层直出，可能有排版失真）
> 所有引号内英文均为原文逐字（verbatim）；中文为翻译。未做评价。

---

## 1. `arXiv2604.24117_CoordinationGap_JSSP_Transport.pdf` ★重点

### 1.1 题录

| 项 | 内容 |
|---|---|
| 标题 | *An Analysis of the Coordination Gap between Joint and Modular Learning for Job Shop Scheduling with Transportation Resources* |
| 作者 | Moritz Link, Jonathan Hoss, Noah Klarmann |
| 单位 | Faculty of Management and Engineering, University of Applied Sciences, 83024 Rosenheim, Germany（通讯：moritz.link@th-rosenheim.de） |
| arXiv ID | arXiv:2604.24117v3 [cs.AI], 21 Sep 2026 |
| 年 | 2026 |
| **是否正式发表** | **已录用**。原文脚注："This paper has been accepted for presentation at the IEEE 22st International Conference on Automation Science and Engineering (CASE 2026)." |
| 资助 | Chips Joint Undertaking / Cynergy4MIE project (Grant Agreement No. 101140226) |
| 开源代码 | https://github.com/proto-lab-ro/jsspt-coordination-gap （原文："To facilitate reproducibility and provide the full hyperparameter configuration, the source code is available at ..."） |

### 1.2 问题设定

- **问题类型**：JSSPT（Job-Shop Scheduling Problem with Transportation Resources），即"机器 + AGV"联合调度。
- **资源**：
  - 作业 `J = {J1,...,Jn}`，机器 `M = {Ml, Mu, M1,...,Mm}`（`Ml` 上料机、`Mu` 卸料机，各含输入/输出缓冲区；输入缓冲区无限、保持分配顺序；输出缓冲区无限、无顺序约束，AGV 可任意顺序取件）。
  - 每个作业 `j` 有固定工艺路线：`m+1` 道工序 `O_{j,i}`（`1 ≤ i ≤ m+1`），每道工序在指定机器 `μ_{j,i}` 上加工，加工时间 `p_{j,i} > 0`；末道工序 `O_{j,m+1}` 表示释放到卸料机 `Mu`，`p_{j,m+1} = 0`。
  - AGV 集合 `V = {V1,...,Vk}`，机器间运输时间 `t(Mi, Mj)` 可非对称；所有 AGV 初始位于上料机。
- **目标**：最小化 makespan `min Cmax`，`Cmax = max_{j∈J} c_{j,m+1}`；`c_{j,i} = s_{j,i} + p_{j,i}`；运输完成时间 `c^T_{j,i} = s^T_{j,i} + t(μ_{j,i-1}, μ_{j,i})`。
- **约束**：
  - 工序顺序 `O_{j,1} → O_{j,2} → ... → O_{j,m+1}`；机器一次一道工序、不可抢占。
  - AGV 单位容量、运输过程不可抢占；`s^T_{j,i} ≥ c_{j,i-1}`（前道工序完成后才能运）；`s_{j,i} ≥ c^T_{j,i}`（运到后才能加工）。
  - 式(1)：`s^T_{j,i} ≥ max( c_{j,i-1}, idle_u + t(loc_u, μ_{j,i-1}) )`，其中 `idle_u` 为 AGV u 完成在途任务的时间，`loc_u` 为 u 变空闲时所在机器。

### 1.3 方法

**MDP 元组** `(N, S, A, P, R, γ)`。`N = 2`（多智能体设定）或 `N = 1`（单智能体训练）。

**状态**：全局状态 `S_t = (S^o_t, S^agv_t)`（作业调度器局部状态 + AGV 调度器局部状态）。

- `S^o_t`：析取图 `G = {V, E}`，顶点 `V = {O, M}`（工序节点 + 机器节点）；边分两类——析取边 `D`（工序间先后约束）、合取边 `C`（机器–工序指派）。同一作业的所有工序顺序相连，每道工序与其加工机器双向相连。
- 顶点公共特征：二值 `S(v)`（该顶点是否已调度）、二值 `T(v)`（是否为机器顶点）。
- 工序节点特征 `LB_{j,i}`（式 2）：
  ```
  LB_{j,i} = max_{s ∈ scheduled O_j, s<i} c_{j,s} + Σ_{k ∈ unscheduled O_j, k≤i} p_{j,k}
  ```
- 机器节点特征 `r_{v_m} = n_sched(v_m) / n`（式 3，n 为作业数）。
- `LB_{j,i}` 在全体工序顶点上归一化。
- `S^agv_t`：每台 AGV 六个标量（均归一化到 [0,1]，均以作业智能体已选定的工序 `O_{j,i+1}` 为条件）：
  - **EPUT**（最早取件时间）`EPUT_j = c_{j,i}`
  - **EST**（目标机器 `M_t` 最早可开工时间）`EST_m = max_{O_{p,l} ∈ S_{M_t}} { c_{p,l} }`
  - **ERT**（AGV u 下一个空闲时刻，即在途任务全部完成）
  - **TTS**（空驶到起点机器 `M_s` 的时间）`TTS_u = t(loc, M_s)`
  - **EAT**（最早到达）`EAT_u = ERT_u + TTS_u`
  - **EFT**（最早完成）`EFT_u = EAT_u + t(M_s, M_t)`
  - 缩放：`EAT/ERT/TTS/EFT` 相对其他 AGV 归一化；`EPUT_j` 相对其他在制工序的 ready time；`EST_m` 相对其他在制机器。

**动作空间（关键）**：`A = A_o × A_agv`，其中
- `A_o` = 决策时刻 t 所有未完成且可调度的工序集合，`a_o ∈ A_o` 为被选中的工序；
- `A_agv` = 对所选工序 `a_o` 而言"兼容的 AGV 集合"，`a_agv ∈ A_agv`。
- 联合动作 `a = a_o × a_agv`（两级/层次化分解：先选工序，再选 AGV）。论文明确称两者均为 **E2E（end-to-end）**：作业侧 "E2E: O"，AGV 侧 "E2E: AGV"。
- **动作掩码**：对无效工序施加 action masking（"To prevent incentivizing the agents to select infinitely invalid decisions, action masking is applied to invalid operations."）。

**策略**：`π_θ(a_t | S_t)` 分解为 `π(a_o | S^o_t)` 与 `π(a_agv | S^agv_t, a_o)`——AGV 策略以作业策略的动作 `a_o` 为条件。

**转移**：确定性，`P(s_{t+1}|s_t, a_t) = 1`；新状态不代表物理时间的离散推进，而是"满足 JSSPT 条件、可安排下一个工序–AGV 对"的下一个合法状态。

**奖励（抄公式，式 4）**：
```
R = { − Cmax / (LB · s),   if the schedule is complete
    { 0,                    otherwise
```
其中 `s` 为静态缩放因子（超参表 `s = 5`），`LB` 为"带运输时间的、所有作业在调度开始时的下界"。原文说明："While sparse, this reward structure ensures that the agents optimize for the global objective (Cmax) rather than local processing efficiencies, avoiding greedy but sub-optimal scheduling behaviors."

**网络结构**：
- 作业智能体：GIN（Graph Isomorphism Network）编码器（式 5）：
  ```
  h^{(l)}_v = MLP^{(l)}_{θ_l}( (1+ε^{(l)})·h^{(l-1)}_v + Σ_{u∈N(v)} h^{(l-1)}_u )
  ```
  `L = 2`，`ε = 0`，每层 2 层 MLP、隐维 64、ReLU；图嵌入 `h_G` = 全部节点末层嵌入的均值。
- 作业解码器：3 层 MLP + tanh（前两层输出 64，末层单 logit）；输入为 `h^{(L)}_v` 与 `h_G` 的拼接（局部+全局）；对全部工序节点输出 logit。
- AGV 智能体：3 层 MLP + tanh（前两层输出 16，末层 logit），作用于所有 AGV 嵌入。
- Critic：同样 GIN 编码器；解码器用 `h_G`（3 层 MLP + tanh，64→scalar）。原文解释只用 `h_G` 作全系统状态代理："The disjunctive graph embedding h_G serves as a sufficient proxy for the total system state, as it implicitly captures the temporal dynamics and completion times that already account for transportation delays."

**训练框架（本文核心）**：PPO，联合训练用 MAPPO。
- joint：两个调度器均为可学习智能体（`π(a_o|S^o)` = GNN，`π(a_agv|S^agv, a_o)` = MLP）；MAPPO 下"both agents undergo concurrent training with independent parameter updates while being optimized through a joint objective"。
- modular：作业与 AGV 策略学习拆成两次独立训练；为满足 MDP 定义（联合动作必须同时包含两个决策），把其中一个策略替换为基线 DR：
  - Job policy training：`π(a_o|S^o_t) = GNN` 且 `π(a_agv|S^agv_t, a_o) = DR`
  - AGV policy training：`π(a_o|S^o_t) = DR` 且 `π(a_agv|S^agv_t, a_o) = MLP`

**超参表（Table III）**：

| 类别 | 参数 | 值 |
|---|---|---|
| training | total frames | 4 × 10^6 |
| | optimizer | Adam |
| | learning rate | 0.0003（linear decay） |
| | instances per rollout | 4 |
| | number epochs | 1 |
| | scaling factor s | 5 |
| PPO | clipping ε | 0.2 |
| | discount factor γ | 0.999 |
| | GAE λ | 1.0 |
| | entropy coefficient | 0.01 |
| | critic (value) coefficient | 0.5 |

### 1.4 实验（★联合 vs 模块化对比的完整设计）

**（a）基线**：10 条作业选择 DR × 4 条 AGV 选择 DR = **40 个 DR 组合**。
- 作业 DR：SPT、SMPT、LPT、MWR、LWR、FDD/MWR、MOR、LOR、random、FCFS。
- AGV DR：random、SPUT（shortest pick-up time）、SCTA（shortest completion time of in-transport tasks）、SCPT（SCTA with pick-up time）。

**（b）模块化解法器集合**：`Sc = { J_i A_j | i ∈ {1..4}, j ∈ {1..10} }`，共 **40 个模块化解法器**；下标标示"训练时所用的 DR"。
- `i`（表头 "Ji DR"，取值为 SCTA / SCPT / random / SPUT，均为 **AGV 规则**）＝训练作业策略时在场的那条 AGV 规则；
- `j`（表头 "Aj DR"，取值为 FDD/MWR、MOR、LWR、SPT、MWR、random、FCFS、SMPT、LOR、LPT，均为**作业规则**）＝训练 AGV 策略时在场的那条作业规则。
- 联合解法器记为 **J0 A0**。原文："A modular solver is constructed by pairing a job scheduling agent and an AGV scheduling agent, with the training DRs as solver identifiers."
- Table II 原文（modular, id i,j）：
  | id | Ji DR | Aj DR |
  |---|---|---|
  | 1 | SCTA | FDD/MWR |
  | 2 | SCPT | MOR |
  | 3 | random | LWR |
  | 4 | SPUT | SPT |
  | 5–10 | — | MWR / random / FCFS / SMPT / LOR / LPT |

**（c）两个训练框架**："two distinct training frameworks are implemented within an identical environment to ensure consistent conditions"——同一环境、同一 PPO 底算法，仅训练模态不同。

**（d）指标**：
- RPI（式 6）：`RPI_i = − (C^i_max − C^y_max) / C^y_max × 100`（负号在原文中如此；数值为正表示相对基线改善）。
- WR（式 7）：`w(i,y) = 1 if C^i_max < C^y_max, else 0`，WR 为独立获胜的平均。
- **coordination gap 的定义**："The coordination gap denotes the performance difference between the joint and modular solvers, quantified by the RPI computed using C^i_max from the joint solver and C^y_max from the modular solver."

**（e）评测配置（Table IV）**：instance: jobs × machines × AGVs
- 15×10 → 18,15,12,9,6,3；10×10 → 12,10,8,6,4,2；12×12 → 14,12,10,7,5,2；14×14 → 17,14,11,8,6,3；20×5 → 24,20,16,12,8,4；5×10 → 6,5,4,3,2,1；15×15 → 18,15,12,9,6,3；30×10 → 36,30,24,18,12,6。
- 资源稀缺度 `ρ = k/n`（式 8，k 为 AGV 数，n 为作业数），`ρ ∈ [1/n, 1]`；评测取 `ρ ∈ {0.2, 0.4, 0.6, 0.8, 1.0, 1.2}`；**每个配置 100 个独立实例**。
- 时间主导指数 `τ*`（式 9–11）：`p' = (p_raw − T_min)/(T_max − T_min)`，`t' = (t_raw − T_min)/(T_max − T_min)`，`T_min = 1`，`T_max = 100`；`φ = p'/(p'+t')`；`τ* = 2φ − 1 = (p' − t')/(p' + t') ∈ [−1,1]`。
- 训练实例：加工时间与运输时间均 ~ DU(1,100)；`k ~ DU(3, n)`；尺寸集 {6×6, 10×10, 15×10, 20×5, 30×10}。
- 网格实验（τ* × ρ 分析）：加工/运输时间各自从 [1,100] 的 10 个连续区间（[1,10],[11,20],...,[91,100]）抽样，构成 100 个网格单元；每单元每配置 20 个随机实例；ρ、τ* 二值分组后共 **126 个观测** 进入 OLS。

**（f）结论：联合 vs 模块化**

1. **总体**：联合解法器 J0 A0 在学习类方法中稳定最强，"exceeding the best modular solver (J2 A1) by approximately 3% in RPI across all baselines"；模块化解法器性能波动大，"indicating sensitivity to the choice of heuristic pairings during independent training"。最好模块化 = **J2 A1**，最差 = **J3 A7**。
2. **相对 DR**：最优单条 DR 在学习型解法器之上；但 "Once the three strongest combinations (out of 40) are excluded, all solvers surpass the remaining DR baselines."（top1 DR: MOR & SCPT；top2: FDD/MWR & SCPT；top3: MOR & SCTA）
3. **随 ρ 变化**：均衡资源时联合优势最大；极端 ρ 下差距从 **3.6% 收窄到 0.5%**（ρ<0.4 严重运输瓶颈，或 ρ>0.8 加工受限）。
4. **四象限（Fig. 2 / 区域均值 RPI）**：
   - Underutilized transport（ρ<0.5, τ*>0）：均值 **3.8%**
   - Resource-saturated（ρ>0.5, τ*<0）：均值 **3.0%**
   - Process-constrained（ρ>0.5, τ*>0）：均值 **1.1%**
   - Transport-constrained（ρ<0.5, τ*<0）：均值 **0.8%**（若干网格为 0，一格略负）
   - 峰值 RPI 达到 **6.7%**（出现在 ρ≈1.2、τ*≈0.6 附近）。
5. **回归量化**：`BD = | −max(0, τ*) + (−ρ + 1) |`（式 12），`BM = (BD − 1)^2`；`JBN = τ* × ρ`，`ABN = (ρ − 1) × τ*`（式 13）。单特征 OLS 的 R²：BM 0.43、ABN 0.26、JBN 0.19（系数 BM +3.31、ABN −3.00、JBN −1.48）；双特征（ABN+JBN）R² = 0.62（系数 −3.94、−2.09）。
   - 全模型（Table V/VI）：R² = 0.64，adj. R² = 0.63，F = 71.34，Prob(F) = 1.03×10^−26，观测 126，cond no. 2.43；系数 const 2.16、**BM +0.32**、**JBN −0.80**、**ABN −0.90**；VIF 分别 1.97 / 1.72 / 1.75（均 < 5）；特征相关性 JBN–BM −0.42、JBN–ABN −0.26、ABN–BM −0.43。
   - 解读原文："The positive coefficient of the balance metric yields an increase in RPI with growing balance between tasks. ... the coordination gap is driven by the level of functional interdependence between scheduling tasks."

### 1.5 核心数字（照抄原文）

- "the joint solver (J0 A0) consistently achieves the strongest performance among learning-based methods, exceeding the best modular solver (J2 A1) by approximately 3% in RPI across all baselines."
- "the performance gap narrows from 3.6% to 0.5%."
- "J0 A0 demonstrates significant performance gains within the resource-saturated and underutilized transport regimes, with regime mean RPIs of 3.0% and 3.8%, respectively. ... with peak RPI values reaching up to 6.7%."
- "the mean RPIs for these bottleneck regimes decrease to 1.1% and 0.8%, respectively."
- "With an F-statistic of 71.34 (p < 0.001) ..."；"While the full model achieves the highest R2 (0.64), the marginal increase over the two-feature model is relatively modest."
- 网格图（Fig. 2）单元格数值样本：ρ=0.2/τ*=0.9 → 0.9；ρ=1.2/τ*=−0.2 → 0.3–1.5 区间；最大值 6.7 出现在 ρ=0.95–1.0、τ*=0.3–0.5 一列。

### 1.6 自认局限（原文逐字 + 翻译）

本文**没有独立的 Limitations 章节**；最接近的表述在结论末句（原文）：

> "This work provides a foundation for extending the MARL framework to industrial JSSPT scenarios. Future research will investigate flexible JSSPT with the inclusion of a machine-selection agent, as well as the integration of multiple objectives, including energy consumption, delay minimization, and cost efficiency."

翻译：本工作为把 MARL 框架扩展到工业 JSSPT 场景提供了基础。未来研究将考察引入**机器选择智能体**的柔性 JSSPT，以及**多目标**（能耗、延迟最小化、成本效率）的整合。

（另需注意其问题规模上的自设边界，原文 Table II 脚注式表述："The boundary condition ρ = 1 denotes the unconstrained transport limit, where each job can be theoretically serviced by a dedicated vehicle, thereby minimizing resource contention."；且模块化训练中"未被学习的那个策略被固定为 DR"，即模块化方案本身隐含"用规则代理另一方"的假设。）

### 1.7 可复用

- **开源代码**：https://github.com/proto-lab-ro/jsspt-coordination-gap （含完整超参配置）
- **数据集**：无公开数据集；训练/评测实例由随机分布生成（重复性由代码保证）。
- **评测协议**：RPI（式 6，含负号约定）+ WR（式 7）；ρ ∈ {0.2,...,1.2} 每点 100 实例；τ* 网格 10×10 bins、每格 20 实例；matched-instance 对比。
- **可复用公式**：`ρ = k/n`；`τ* = (p'−t')/(p'+t')`；`BD`/`BM`/`JBN`/`ABN` 回归特征；析取图 `LB` 下界特征；AGV 六元特征 (EPUT, EST, ERT, TTS, EAT, EFT)；稀疏奖励 `−Cmax/(LB·s)`；层次动作 `a_o × a_agv`。
- **方法论可复用点（与本方向直接相关）**：模块化 vs 联合的**同期同环境对照**设计 + "用 DR 替换其中一个策略以补全联合动作"这一让模块化训练可在同一 MDP 中运行的技巧；以及"用一个 (ρ, τ*) 二维参数空间刻画何时联合训练才值得"的实验模板。

---

## 2. `arXiv2608.09343_LLM_HeuristicDesign_SimulationTraces_AGV.pdf`

### 2.1 题录

| 项 | 内容 |
|---|---|
| 标题 | *LLM-Guided Heuristic Design from Simulation Traces: A Case Study in Dynamic Production and AGV Scheduling* |
| 作者 | Jinbo Li, Chuanhao Li（通讯：chuanhao-li@tsinghua.edu.cn） |
| 单位 | Department of Industrial Engineering, Tsinghua University |
| arXiv ID | arXiv:2608.09343v1 [cs.AI], 10 Aug 2026 |
| 年 | 2026 |
| **是否正式发表** | 未见发表信息；arXiv 预印本（版式为期刊投稿稿，含 Keywords 与 Appendix A/B，无期刊名/DOI） |
| 资助 | 未提及 |

### 2.2 问题设定

- **问题类型**：动态生产 + AGV 调度的**仿真优化（SBO）**；不训练神经网络，而是在**可执行策略代码空间**上做 LLM 引导的启发式设计。
- **仿真系统**：三条产线的离散事件仿真（DES），改造自 FreezoneX 2025 智能制造调度环境并加装事件级 trace 记录。含工位（Workstation A/B/C）、传送带（Conveyor AB/BC/CQ-main/CQ-upper/CQ-lower）、缓存区、质检站、原料库/成品库、AGV、动态订单、返工（rework）、充电、随机故障。
- **策略控制范围（= 决策空间）**：生产线选择、运输任务排序、AGV 指派、**主动充电**（+可选参数估计逻辑）。原文："Product routes and local workstation sequencing remain under simulator control."
- **目标**：最大化仿真总得分 `J(π) = E_ξ[S(π; ξ)]`（式 1），得分为 8 项 KPI 归一化后的固定加权和，落在 [0, 100]。
- **约束/动态要素**：动态订单到达、有限缓冲、可重入加工、质量导致的返工（首次检验 <80 分返工，<60 分报废；返工后仍 <80 报废）、充电约束、可选设备故障（故障间隔 `U(80,120)` 分钟，恢复 `U(20,60)` 分钟）。

### 2.3 方法

**总体循环（Algorithm 1）**：`Evaluate_R(π) = ( S̄_R(π), τ(π) )`（式 3）——均值分数用于选择，诊断 trace 用于修订。

- **式(2) 样本均值**：`S̄_R(π) = (1/R) Σ_{r=1}^{R} S_r(π)`
- **式(4) 修订**：`{π_{t,j}}^{k_t}_{j=1} ← Revise_t(π_t; S̄_t, τ_t)`
- **式(5) 诊断回放选择**：`r(π) ∈ arg min_{1≤r≤R} S_r(π)`（对**得分最低**的那次重复做回放，产生可查询 trace；该次得分不计入均值）
- 论文明确：**LLM 只在评估批次之间修订策略**；每次仿真运行内由固定策略决策（"LLM revision occurs between evaluation batches, while a fixed policy controls each simulation run."）

**Agent 结构**：1 个 **Manager Agent**（读当前策略、均值分数、保留的诊断证据与历史，提出多条互异的修订方向）+ 多个 **Editing Agent**（每个在隔离工作区、独立上下文中实现一条修订方向，产出可执行候选策略代码）。每轮候选数 `k_t` 由 manager 在建议区间内自选。

**状态/动作/奖励的对应关系**（本文无 MDP 表述，等价物如下）：
- "状态"=管理器可见的证据：分组 KPI 摘要（生产效率/质量与成本/AGV 效率三组）+ 从 trace 数据库查询的记录（订单事件、资源状态变化、队列与缓存状态、AGV 任务与充电事件、故障与恢复事件、时变指标值）。
- "动作"=对策略代码的编辑（策略由 `scheduler.py` + `param_estimator.py` 组成）。
- "奖励"=**总得分**（式见下），候选晋升采用 best-so-far 且必须严格优于当前 incumbent。

**总得分的 8 项指标与权重（Appendix A.3, Table 5）**：

| 维度（组权重） | 指标（总得分中权重） | 含义 |
|---|---|---|
| 生产效率 (40%) | On-time order completion (16%) | 早于 due date 完成的订单比例 |
| | Weighted production cycle (16%) | 合格品实际/理论生产时间比，按完成率惩罚 |
| | Equipment utilization (8%) | 设备工作时间/可用时间 |
| 质量与成本 (30%) | Quality pass rate (12%) | 合格品占比 |
| | Cost-benefit ratio (18%) | 基线成本/实际成本（材料、能源、维修、报废） |
| AGV 效率 (30%) | Charging-strategy efficiency (9%) | 主动充电事件占全部（主动+被迫）充电事件比 |
| | AGV energy efficiency (12%) | 单位充电时间完成的运输任务数 |
| | AGV utilization (9%) | 运输时间/AGV 可用时间（不含故障与充电） |

**LLM 骨干**：Gemini-3.1-Pro、GPT-5.5、GPT-5.4-mini、GLM-5（每个骨干独立重复 5 次完整优化）。

### 2.4 实验

- **实例/规模**：默认仿真时长 `T_sim = 500` 分钟，确定性动态订单到达 `Θ_dyn`（每 10 分钟一单）；另一设定 `T_sim = 3000` 分钟；再一设定把订单到达改为 `U(5,15)` 分钟。
- **评测协议**：每个候选 `R = 10` 次评分重复 + 1 次诊断回放（回放分数不计入均值）；候选评估用独立种子集，incumbent 保留自己的旧评估而非用候选种子重评；最终报告为各次完整运行最终 incumbent 的存储均值分数。匹配种子复核用 **100 个相同种子（123–222）**，双边 Wilcoxon 符号秩检验 + Holm 校正。
- **基线**：
  - 滚动 MILP（仅 AGV 运输任务；不含电池/充电约束；新任务或返工事件出现时重优化）；
  - 规则组合启发式（生产线选择 × 任务排序 × AGV 指派，共 **135 种组合**）；
  - 传统元启发式 GA / DE / PSO（在该组合空间上搜索）；
  - 注：**所有基线都无充电控制层**（原文强调这是差距的部分来源）。
- **指标**：8 项 KPI 加权总得分（0–100）。

**核心数字（照抄原文）**：
- 各骨干最终均值：**77.51（Gemini-3.1-Pro）**、76.13（GPT-5.5）、73.83（GLM-5）、**72.84（GPT-5.4-mini）**；best/worst 区间 Gemini 76.10–78.61、GPT-5.4-mini 70.94–75.80。
- "the best Gemini-3.1-Pro run reaching **78.61** in the default setting"；初始策略 **62.49** → 78.61，**improvement of approximately 25.8%**（10 轮迭代内）。
- 启发式/MILP/元启发式基线"reach the **low-60** score range"。
- 匹配种子：LLM 策略在**全部 100 个种子上**均胜出（`W− = 0`，渐近双边 p = 3.90×10⁻¹⁸，Holm 校正后 1.17×10⁻¹⁷，12 组比较全部同向）。
- 换设定（Table 1）：`T_sim = 3000`：MILP 50.03、Best heuristic 57.67、GA 63.06、DE 62.91、PSO 62.96、**Proposed 74.16 (72.37, 78.16)**；改变 `Θ_dyn`：MILP 62.71、Best heuristic 62.72、GA 63.42、DE 63.15、PSO 62.82、**Proposed 76.34 (74.71, 78.97)**。
- 迭代轨迹（Fig. 6）：62.49 → 70.28 → 71.63 → 72.08 → 75.00 → 75.98 → 76.82 → 77.18 → **78.61**（iteration 8）→ 78.61 → 78.61。
- 消融（Table 2）：
  - Gemini-3.1-Pro：原始 均值 77.51 / min 76.10 / max 78.61 / 平均迭代 9.2；单候选 73.65 / 62.36 / 78.56 / 5.8；无数据库 76.16 / 72.27 / 77.61 / 6.4。
  - GPT-5.4-mini：原始 72.84 / 70.94 / 75.80 / 8.0；单候选 66.22 / 62.29 / 72.07 / 4.6；无数据库 69.28 / 66.26 / 71.84 / 8.0。
- 故障场景（无再优化）："the optimized policy retains its advantage when random faults are introduced after optimization."
- 诊断驱动的具体改动（RQ2 案例）：初始 trace 显示 AGV 很少主动充电、被迫充电、且到达任务起点前有大量空驶；第一轮最优候选加入"就近任务偏好 + 低电量空闲 AGV 主动充电"，62.49 → 70.28；后续加重下游缓冲清空任务优先级；第五轮 75.98；后期发现过度偏向下游清空会饿死上游，再调权重，第 8 轮达 78.61。

### 2.5 自认局限（原文逐字 + 翻译）

原文 "Limitations. Three aspects of the work should be considered when interpreting the results:"（逐字）：

> "• **Use of optimization history.** The framework retains previous policies, feedback, and evaluation outcomes, but provides them mainly as contextual records. It does not convert this information into a structured memory of successful revisions, unsuccessful attempts, and their performance effects. The current experiments therefore do not establish how systematically reusing prior search experience would affect optimization."

翻译：优化历史的利用。框架保留了既往策略、反馈与评估结果，但主要作为上下文记录提供，未将其转化为"哪些修订成功、哪些失败及其性能影响"的结构化记忆。因此当前实验无法确定系统性复用先前搜索经验会如何影响优化。

> "• **Incumbent-centered search.** The best-so-far mechanism provides a simple and auditable sequence of policy revisions, with every promoted candidate subjected to execution checks and simulation evaluation. However, the search maintains a single incumbent rather than preserving several promising policy branches, which limits diversity when improvement slows."

翻译：以 incumbent 为中心的搜索。best-so-far 机制给出简单、可审计的策略修订序列，每个被提升的候选都经过执行检查与仿真评估；但搜索只保留单一 incumbent 而非多条有前景的策略分支，改进放缓时多样性受限。

> "• **Stochastic evaluation and trace selection.** Repeated replications reduce dependence on a single simulation realization, and the matched-seed re-evaluation independently checks the reported final comparisons. Candidate promotion nevertheless relies on finite-sample estimates obtained from independent seed sets. Moreover, selecting the lowest-scoring replication for diagnosis deliberately emphasizes observed failure modes, but that trace may not represent typical operating conditions. The current experiments do not isolate the effects of alternative promotion criteria or trace-selection strategies."

翻译：随机评估与 trace 选择。重复实验降低了对单次仿真实现的依赖，匹配种子复核独立校验了最终比较；但候选晋升仍依赖独立种子集上的有限样本估计。此外，选择**得分最低**的那次重复做诊断，刻意强调已观测到的失效模式，该 trace 可能不代表典型运行条件。当前实验未分离不同晋升准则或 trace 选择策略的影响。

另一处边界条件（Discussion）："Although these conditions may hold in manufacturing, warehouse logistics, port operations, intra-hospital logistics, and other discrete-event systems, the empirical evidence in this study comes from dynamic production and AGV scheduling. Operational use would additionally require a verified, validated, and calibrated simulator..."

### 2.6 可复用

- **开源代码**：论文本身未提供代码仓库；仿真环境改造自 https://github.com/supcon-international/25-AdventureX-SUPCON-Hackathon （原文脚注 1）。
- **数据集**：未提供；仿真参数在 Appendix A.4（Tables 6–9）。
- **评测协议（可复用性高）**：`R = 10` 次评分重复 + 1 次最低分回放；独立种子评估 + incumbent 存储分数；100 个匹配种子（123–222）上的配对 Wilcoxon 符号秩检验 + Holm 校正；minimum–maximum over 5 independent runs 的报法。
- **可复用公式**：式(1)–(5)；Table 5 的 8 项 KPI 权重表（可直接用于"生产+AGV"综合评分设计）；优先级的指数衰减奖励 `r_ex(f) = K·b^{Fmax−f}`，`b = (R_high/R_low)^{1/(f_low−f_high)}`，`K = R_high / b^{Fmax−f_high}`（式 45–48）。
- **方法论可复用点**：把"**事件级仿真 trace 作为 LLM 诊断输入**"与"聚合分数作为选择信号"分离的接口设计；诊断回放选最低分重复的做法；多候选并行 + best-so-far 的搜索骨架；消融的两个轴（并行候选数 / trace 数据库访问）。

---

## 3. `arXiv2601.04887_FMS_Intralogistics_AGV_ToolSharing.pdf`

### 3.1 题录

| 项 | 内容 |
|---|---|
| 标题 | *Flexible Manufacturing Systems Intralogistics: Dynamic Optimization of AGVs and Tool Sharing Using Coloured-Timed Petri Nets and Actor-Critic RL with Actions Masking* |
| 作者 | Sofiene Lassoued, Laxmikant Shrikant Baheti, Nathalie Weiß-Borkowski, Stefan Lier, Andreas Schwung |
| 单位 | South Westphalia University of Applied Sciences（Soest / Meschede, Germany） |
| arXiv ID | arXiv:2601.04887v1 [cs.AI], 8 Jan 2026 |
| 年 | 2026 |
| **是否正式发表** | 未见发表信息；arXiv 预印本（Elsevier 双栏投稿版式，含 Keywords，无期刊名/DOI） |
| 方法名 | **PetriRL for intralogistics** |

### 3.2 问题设定

- **问题类型**：FMS 内物流调度 = 经典 JSSP 扩展，**同时**加入 AGV 物料搬运 + **刀具共享（tool sharing）** + 刀具运输车。
- **资源**：
  - 机器 `M`、作业 `J_i`（工序序列 `O_ij`，每道工序需指定机器 `m`、刀具 `t`、加工时间 `p ∈ Z⁺`）；
  - AGV 集合 `A = {A1,...,Aq}`，运输时间矩阵 `D_AGV = [d^AGV_uv]`，考虑 deadheading（空驶）；
  - 刀具集合 `T`，每种刀具类型**只有一件**（shared resource）；刀具运输车 `TT = {TT1,...,TTs}`，时间矩阵 `D_TT = [d^TT_uv]`，同样考虑空驶；
  - 系统布局含机器、Tool Magazine、Load/Unload Station。
- **目标**：最小化 makespan（同时协调机器、AGV、刀具、运输车）。
- **约束（原文 6 条）**：① 机器一次一道工序、不可抢占；② AGV 数量有限，全部占用时等待；③ 每种刀具类型仅一件单件可用，冲突需靠调度避免；④ 刀具运输车数量有限；⑤ 工序先后约束（前道完成 且 物料与刀具均就位）；⑥ Deadheading 为非生产时间，计入总 makespan。
- **附加假设**：所有运输时间不能预先确定（依赖加工顺序），只有工序内的载货运输时间是固定的。

### 3.3 方法

- **建模**：Coloured-Timed Petri Net（CTPN）。Petri 网的可控变迁（controllable transitions）为 **job select** 与 **AGV select**，RL 智能体通过外部触发控制它们；Petri 网自动处理约束执行与 token 管理。原文举例："with 15 jobs and two AGVs, the agent has 17 possible actions"。
- **动作空间**：可控变迁集合（作业选择 + AGV 选择显式建模为 E2E 选择），**动态掩码**由 Petri 网的 **guard function** 给出——"generates a Boolean vector indicating enabled transitions based on token distribution"。
- **观测**：由 Petri 网 token 分布导出；在 marking 基础上增强为「token 特征（**剩余加工时间** + **color**）」。
- **奖励（抄公式，式 8）**：
  ```
  Reward = − ( Σ idle machines / total number of machines )
  ```
  原文动机："We penalize idle machines to address sparse rewards in larger instances, providing immediate feedback through machine state changes. This strategy penalizes delays in material and tool availability... By focusing on penalizing idleness instead of rewarding active use, the agent learns to complete jobs more efficiently."
  原文也说明用 makespan 作奖励的困难："using it as a reward poses challenges due to its sparsity, as feedback is only provided at the end of the episode... In these cases, the lengthy action sequences create a credit assignment problem."
- **算法**：**Maskable PPO**（Stable Baselines 3），Actor-Critic；Algorithm 1 = PPO + Petri 网 guard function 动作掩码（`G(a_t)` 为 guard 函数，`ρ_t = π_θ(a_t|s_t)/π_θold(a_t|s_t)`，clip ε，`L_VF`、`L_CLIP`、`L(θ)`）。
- **Lookahead（Model-Based RL 成分）**：当 AGV 缓冲区为空、无法确定下一个位置时，默认取 `D_AGV` 中的**最大运输时间**；改进做法是"创造一个环境孪生体（twin），模拟未来步骤"以预测将填入缓冲区的下一个 token，从而提前优化 AGV 位置（受 Dyna-Q 的 simulation 阶段启发）。两个场景：① 缓冲区已有 token → 用 `D_AGV` 直接算；② 缓冲区为空 → 用 twin 环境模拟未来。
- **实现**：PyTorch + Stable Baselines 3；NVIDIA Quadro RTX 5000；Windows 11。
- **训练规模**：示例（job set 1 layout 1，5 jobs × 4 machines × 2 AGVs）训练 3×10⁵ 步；大规模需 3e6–15e6 步（Fig. 5）；部署时间随规模 5–60 秒。

### 3.4 实验

**（a）新基准（本文贡献之一）**：
- 背景：既有研究多用 Bilge benchmark，但其"scenarios typically involve no more than eight jobs and six machines"。
- 新基准：**80 个实例、8 组、规模从 15×15×15 到 100×20×20（jobs × machines × tools）**，每组 10 个实例；用 **LCG（Linear Congruential Generator，a=16807, b=127773, c=2836, m=2³¹−1）** 生成随机值。
- 种子（Table 1）：Machine Allocation 398197754；Tool Allocation 170719940；Processing Times 840612802；Tools Transport Times 280219920；AGVs Transport Times 180119550。
- 声明提供 gym 兼容环境与实例生成器（"we propose a gym-compatible environment and an instance generator"）——但**文中未给出任何 URL**（全文 grep 无 http(s) 链接）。

**（b）基线**：
- **12 条启发式**：FIFO、SPS、LPS、SPTN、LPTN、MTWR、LTWR、LWT、SPT、LPT、SPSR、LPSR（作业选择）；AGV 选择两条规则——"Chooses the first AGV available" 与 "Chooses the AGV with least work done"。
- **元启发式 SOS（Symbiotic Organisms Search）**：复现自 Reddy et al.（唯一同时处理 AGV + tool sharing 的 JSSP 研究），先在原文小实例上验证复现保真度（Fig. 4），再用于大实例；大实例上 **SOS 搜索时间限制为 10 分钟（600 s）**。
- 公平性："the heuristics and metaheuristics interact with the same Petri net environment. All methods operate under identical constraints and optimization objectives. At each time step, the environment produces a Boolean list of enabled actions, determined by the guard functions. This list is shared with all the RL agent, the heuristic, and the metaheuristic methods"。

**（c）指标**：makespan（steps）+ 计算时间（s）。

**核心数字（照抄原文）**：
- 小实例 (Table 3, EX11–EX102, 5×4 至 8×4)：PetriRL 平均 **104.4** vs 启发式 96.6–133.8 区间（两种 AGV 规则下 Average 分别为 100.1 / 96.6 / 101.9 / ... / 127.0 等）；"our approach outperformed the tested heuristics with both proposed AGVs allocation strategies"。
- 计算时间："the average computation time for PetriRL was **0.24 seconds**, compared to **217 seconds** for SOS."（约三个数量级）
- 大实例（Table 4，PetriRL vs SOS，SOS 限时 600 s）：

| Instance | Size | AGV-only: PetriRL(时间) vs SOS → Gap | AGV+Tool: PetriRL(时间) vs SOS → Gap |
|---|---|---|---|
| sl00 | 15×15 | 1801 (5.50s) vs 2194 → 18% | 2639 (14.48s) vs 4057 → 35% |
| sl10 | 20×15 | 2057 (7.44s) vs 2483 → 17% | 3163 (17.29s) vs 5146 → 39% |
| sl20 | 20×20 | 2619 (11.54s) vs 3448 → 24% | 4148 (26.26s) vs 6179 → 33% |
| sl30 | 30×15 | 2436 (11.38s) vs 3250 → 25% | 5499 (28.94s) vs 7635 → 40% |
| sl40 | 30×20 | 3178 (17.64s) vs 4194 → 24% | 5989 (37.61s) vs 9065 → 34% |
| sl50 | 50×15 | 3478 (27.64s) vs 4703 → 26% | 7122 (48.50s) vs 12558 → 43% |
| sl60 | 50×20 | 4139 (37.65s) vs 5837 → 29% | 8755 (72.27s) vs 15765 → 44% |
| sl70 | 100×20 | 7623 (64.09s) vs 9738 → 22% | 18845 (144.12s) vs 30537 → 38% |

  原文总结："improvements ranging from **18% to 25%** in the AGVs-only implementation and from **35% to 40%** in the AGVs and tools-sharing implementation."（另有一处写 43%/44%）
- 泛化（Fig. 11）：单一通用 agent（仅在 sl60 = 50×20 上训练，测试另外 60 个实例）vs 6 个按尺寸专门训练的 agent（覆盖 7 组、70 个实例）："the general agent exhibited a slight decrease in performance with a **4% increase in makespan average** compared to specialized agents"。机制：用 100×20 的 Petri 网布局承载 15×15 实例，空槽位的 job selection 变迁被 guard 自动掩码，观测以 null 填充。
- 动态性（Fig. 12）：10 个分区 P1–P10 顺序注入，RL 无需重训即可适应；"SOS approach initially outperforms the RL-based PetriRL optimizer due to its exhaustive search capabilities within constrained search spaces"，但计入 rescheduling time 后 RL 综合更优。
- 消融（Table 5 / Fig. 13）：
  - Reward shaping（30×20 中等实例，machine utilization vs 稀疏 makespan）：**Average 3064.4（with）vs 3238.2（without）**；小实例 EX12–EX102：**Average 102（with lookahead）vs 112（without）**。
  - 动作掩码："the reward steadily increases and stabilizes after **200,000 steps** with masking. In contrast, without masking, the reward remains unstable and fails to reach the same level."
- 瓶颈观察（sl30 Gantt）："the tool transporters are the bottleneck in the system."（刀具运输车成为瓶颈）

### 3.5 自认局限（原文逐字 + 翻译）

本文**没有 Limitations 章节**；结论末段给出未来方向（逐字）：

> "It opens avenues to explore how the model can adjust to internal environmental changes, including unforeseen equipment breakdowns, scheduled maintenance, and strategic tool changes, particularly when machining advanced materials with high strength. The model can account for tool replacements based on wear patterns, enhancing system robustness and operational efficiency in real-world manufacturing environments."

翻译：（本研究）开启了探索该模型如何适应**内部环境变化**的路径，包括未预见的设备故障、计划性维护，以及策略性换刀——尤其是在加工高强度先进材料时。模型可以考虑基于磨损模式的刀具更换，从而提升真实制造环境中的系统鲁棒性与运行效率。

（另一处自认边界：泛化 agent 相对专门 agent 有 4% 的 makespan 劣化，原文："This highlights a trade-off between generalizability and optimal performance."）

### 3.6 可复用

- **开源代码**：**论文中未给出 URL**（声称提供 gym 兼容环境与实例生成器，全文无链接，无法从文中定位）。
- **数据集**：新基准 80 实例（15×15×15 → 100×20×20），由 LCG + 文中 5 个种子完全可复现；小实例沿用公开的 Bilge benchmark / Reddy et al. 的 EX 系列。
- **评测协议**：所有方法共用同一 Petri 网环境 + 同一 guard 产生的 enabled-action 布尔列表；SOS 大实例限时 600 s；makespan（steps）+ 计算时间（s）双指标。
- **可复用公式/机制**：奖励 `Reward = −(Σ idle machines / total machines)`；Petri 网 guard function → 布尔掩码向量 → Maskable PPO；Lookahead（缓冲区空时用 twin 环境模拟未来，否则退化为 max 运输时间的 worst-case 计划）；`D_AGV`、`D_TT` 矩阵 + deadheading 建模；LCG 实例生成器伪代码（Algorithm 2）。

---

## 4. `arXiv2609.19048_AutomatedWarehouse_LastMile_Integrated.pdf`

### 4.1 题录

| 项 | 内容 |
|---|---|
| 标题 | *Integrated Optimization of Automated Warehouse Operations and Last-Mile Transport for Differentiated On-Demand Delivery* |
| 作者 | Xiaozhu Sun（通讯）, Bilal Farooq |
| 单位 | Laboratory of Innovations in Transportation (LiTrans), Toronto Metropolitan University, Toronto, Canada |
| arXiv ID | arXiv:2609.19048v1 [cs.LG], 21 Jul 2026 |
| 年 | 2026 |
| **是否正式发表** | 未见发表信息；arXiv 预印本（Elsevier 作者–年份引用版式，无期刊名/DOI） |
| 资助 | Canada Research Chair (CRC-2021-00480)、NSERC Discovery (RGPIN-2020-04492) |
| 方法名 | 内部：**MORM-AGDQN**；外部：**MRMH-HCVRP** |

### 4.2 问题设定

- **问题类型**：**仓内 AGV 调度（含路径/避碰）+ 仓外 heterogeneous fleet 最后一公里配送**的一体化优化（两层耦合）。
- **仓内资源**：多层货架、**多容量 AGV**（`m_k` 为可载订单数）、工位、传送带、单/双巷道；SKU 按货格存储；WMS 无线调度。
- **仓外资源**：异构车队 `H`（按服务距离、容量、速度分组：scooter / motorcycle / van），客户集合 `C`，仓库为 depot 0。
- **目标（多目标，MORL）**：`min F_d(x,t) = (f_1(x,t),...,f_N(x,t))`（式 2）；每时间窗 `min F(x,t) = min(F_s(x) + F_d(x,t))`（式 3）。四类子目标：能耗、库存成本、外部影响、安全/稳定性。
  - 能耗（式 5）：`min f_En = Σ_k Σ_i Σ_j d^k_ij x^k_ij f(Q_k + w^k_ij)`，约束(6)–(10)：载重 ≤ 限重、最多 K 台 AGV、路径二进制、载件数 ≤ `m_k`、能耗 ≤ `E_k`。
  - 库存成本（式 11）：`f_In(x,t) = (f_abc(x), f_st(x,t))`，按 ABC 分类 + 等待时间。
- **约束（原文分类）**：机器/AGV/刀具类约束（本文为 AGV 载重、容量、电量、巷道单 AGV 冲突消解）；车辆侧容量、最大行驶距离、时间窗、截止期限；订单 `T^dead_u` 与优先级。
- **简化声明**：不含 EV 充电/车辆租赁，"we group fleets by service distance, capacity, and speed, deferring fully hybrid fleet scenarios to future research."

### 4.3 方法

**（a）仓内 MORM-AGDQN**（cooperative multi-agent MDP `M = (S, A, P, R)`）
- **状态（式 32–36）**：`s^in_t = (s^env_t, s^op_t, s^ord_t, s^ext_t)`
  - `s^env_t = (G, O_t, B)`：仓库布局图 + 静态障碍 + 动态障碍（各 AGV 实时位姿）
  - `s^op_{t,k} = (ℓ^k_t, θ^k_t, load^k_t, battery^k_t, status^k_t)`：位置、朝向、载件数、剩余电量、状态（idle/moving/picking/delivering/charging）
  - `s^ord_t = (ℓ^pick_j, ℓ^del_j, o_j, c_j)`：取货点、送货点、释放时间、订单类别
  - `s^ext_t = (f^Pareto_u, T^dead_u)`：客户 Pareto 前沿优先级 + 截止期
  - 张量化表示：`s_t ∈ R^{N×N×C}`，`C = 4` 通道：① 仓库环境（货架/充电站/AGV 位置）② 外部因素（订单优先级/交通密度图/距离变换）③ AGV 运行状态（位置/电量/载重）④ 订单属性（目标位置/截止期/优先级类）。
- **动作空间**：AGV 的 5 个二维平面动作——**前进、后退、左移、右移、停留**（"moving forward, backward, left, right, and staying. The AGVs can move vertically and horizontally simultaneously."）。
- **转移**：确定性 `P(s_{t+1}|s_t,a_t)=1`；`ℓ^k_{t+1} = ℓ^k_t + Δ(a^k_t)`，`θ^k_{t+1} = Θ(a^k_t)`（式 37）；完成卸货时记录 `T^fin_j = t`（式 38）。
- **奖励：奖励机（Reward Machines）**（式 39–44）
  - 4 台奖励机 `RM_En, RM_In, RM_Ex, RM_Sa`，对应能耗、库存成本、外部影响、安全/稳定；`R_{i,t} = (R^En_{i,t}, R^In_{i,t}, R^Ex_{i,t}, R^Sa_{i,t})`（式 42）。
  - 稀疏奖励（任务完成时触发）：`R_sparse(t) = w^sparse_Ex · r^sparse_Ex + w^sparse_In · r^sparse_In`（式 43）
  - 即时奖励：`R_instant(t) = w^instant_Sa · r^instant_Sa + w^instant_En · r^instant_En`（式 44）
  - 优先级奖励按 Pareto 前沿指数衰减（式 45–48）：`r_ex(f) = K · b^{Fmax−f}, b>1, K>0`；`b = (R_high/R_low)^{1/(f_low−f_high)}`；`K = R_high / b^{Fmax−f_high}`；示例 `f_high = 2 → R_high = 20`，`f_low = 22 → R_low = 0.1`。
  - **权重空间**：`Ω = {w | w = [w_1,...,w_k]^T, w_j ∈ [0,1]}`（式 49）。
- **网络**：多通道状态 → 3 个 3×3 卷积（Conv1 64 / Conv2 128 / Conv3 256 滤波器，ReLU + BatchNorm，其间 2×2 max-pool）→ flatten + 标量特征拼接 → 3 个全连接（FC1 1024 / FC2 512 / FC3 256）→ `q_t = f_CNN(S_t; θ_c)`；再接 **A\*-DQN**（A\* 提供引导奖励与动作偏置）。动作选择（式 61）：`argmax Q` 以概率 1−ε；`a*_t`（A\* 推荐动作）以概率 `ε_A*`；随机动作以概率 `ε − ε_A*`。
  - 损失（式 62）：`L(θ) = E[(r + γ max_{a'} Q(q',a';θ⁻) − Q(q,a;θ))²] + β·Ω(θ)`（Ω 惩罚偏离 A\* 策略）；Adam 更新（式 63）；`ε(t) = ε_min + (ε_max − ε_min)·e^{−t/τ}`（式 64）。
- **Algorithm 1/2**：MORM-MADQN，含 `RM-A*` 引导、`Q^RM_i(s,a) ← Q_i(s,a;θ_i) + γ_RM · R_RM(a, u⃗)`，经验元组 `(s,a,r,s',u,u')`。

**（b）仓外 MRMH-HCVRP**（Transformer 策略网络 `π_θ(a_t|s_t)`）
- **状态（式 50–54）**：`s^ex_t = (s^veh_t, s^cus_t, s^traf_t, s^seq_t)`
  - `s^veh_t = (ℓ^h_t, o^h_t, T^h_t, D^h_t, G^h_t, f^base_h, Q_h, Dmax_h, z_h)`（位置、剩余容量、累计时间/距离、部分路径、速度、容量、里程上限、单位距离成本）
  - `s^cus_t = (l_u, d^t_u, p_u, r_u, T^dead_u, T^ser_u, served_u)`
  - `s^traf_t = (R, β(t), γ(r), λ_hrt)`（区域划分：centre/urban/suburb）
  - `s^seq_t = (Q_t, Δt)`（仓库订单准备序列与更新窗口，新订单每 5 分钟追加）
- **动作空间**：每步选择一个 **(车辆 h, 客户 u)** 对，`a_t = (veh_h, y_u) ∈ A(s_t)`（含可行性掩码：`feasible_u ← (demand_u ≤ cap_h) ∧ (delivery time feasible)`，不可行则 mask 为 −∞）。
- **转移**：`ℓ^h_{t+1}=l_u`，`o^h_{t+1}=o^h_t−d^t_u`，`T^h_{t+1}=T^h_t+t_travel(ℓ^h_t,l_u)+T^ser_u`，`D^h_{t+1}=D^h_t+d(ℓ^h_t,l_u)`，`G^h_{t+1}=G^h_t∪{l_u}`；`d^{t+1}_u=0, served_u=1`（式 55）。
- **奖励（式 56–60）**：
  ```
  R^VRP_t = ( r^Pr_t, r^Tr_t, r^Con_t )
  r^Pr_t  = − ρ_p · (w_max − w_u)                      (57)
  r^Tr_t  = − γ · D_uv / v^base_h                      (58)
  r^Con_t = − κ · [ (1 − β_h(t)) + (1 − γ_h(r)) + λ_hrt ]   (59)
  R^total_t = r^Pr_t · r^Tr_t · r^Con_t                (60)
  ```
  原文说明乘积形式："The multiplicative formula for the reward ensures that all reward components must be satisfied simultaneously to obtain a higher total reward."
- **编码器（式 65–67）**：客户输入 `x̃_u = (s_u, d^u_{Q1},...,d^u_{Qm}, w^pr_u, τ_u, T^dead_u/T_max, T^ideal_u/T_max)`；N 层多头注意力 + FF + 残差 + BN（式 66：`Z_{l,y}=softmax(Q_{l,y}K^T_{l,y}/√d_k)V_{l,y}`）；三个专用注意力头对应运输时间、服务优先级、区域交通延误；路径特征嵌入 `C̃^h_t = [h^{g^h_0}_N, ..., h^{g^h_{t−1}}_N]`（式 67）；vehicle-selection decoder + node-selection decoder 双解码器。
- **外部训练（Algorithm 4）**：Actor-Critic，`r_t ← w_1·r_travel time + w_2·r_travel cost + w_3·r_priority`；`y ← r + γV_φ(s')`，`L_value=(V_φ(s)−y)²`，`L_policy=−log π_θ(a|s)·(y−V_φ(s))`。
- **两层耦合（4.3 节）**：用 **3D Pareto 前沿**把外部因素转成**基于时间的 AGV 取货顺序**，作为神经网络输入；线性目标（能耗、库存成本）用 LP 求解，非线性/非凸目标用 Pareto 优化。
- **工艺细节**：订单在仓内包装，处理时间按原文数据分为 5/10/15 分钟三档；系统每 5 分钟更新一次信息。

### 4.4 实验

- **数据集**：**Amazon delivery data**（Alianwar, 2024）作为基础数据；地理数据用 Google Maps 处理；交通模式基于该数据集历史数据；Python 离散事件仿真 + agent-based 建模。
- **三个场景**：① Baseline（正常订单量、可预测交通）；② Peak Season（订单激增 **2–5 倍**、复杂交通、多种车队规模）；③ Uncertainty（突发拥堵、紧急订单、特殊日）。
- **实例规模**：仓内 1 AGV/17 订单 到 4 AGV/120 订单；泛化实验用 17 订单（小）与 96 订单（大）训练，10,000 rounds；VRP 对比用 2 台车；案例详细展示 34 订单的 Gantt 与路径；外部训练 50 epochs。
- **基线**：
  - 仓内算法对比：PG、DQN、AGDQN、MO-AGDQN、MORM-AGDQN（1,000 training episodes）；
  - 路径方案对比：Time Sequence & Clustering、Shortest Path & Clustering、HCVRP、**MRMH-HCVRP**（Table 3）；
  - KPI 对比：Proposed vs **FIFO** vs **Historical Data**（Table 4）。
  - 防过拟合："we employed a rolling baseline strategy that updates the model only when it significantly outperforms historical bests on an independent test set (p < 0.05)."
- **指标**：total/vehicle distance (km)、time (hr)、rapid delivery rate、priority first ratio；仓内/运输/联合三层 KPI（fulfilment time、on-time departure rate、high-priority scheduling rate、on-time inventory pickup rate、average delivery time、on-time delivery rate、high-priority satisfaction rate、vehicle utilization、total travel distance、latest delivery time、high-priority service rate、average/longest/shortest completion time）。

**核心数字（照抄原文/表格）**：

Table 3（四种路径方法）：
| Metric | Time Sequence & Clustering | Shortest Path & Clustering | HCVRP | MRMH-HCVRP |
|---|---|---|---|---|
| Vehicle 1 距离(km)/时间(hr) | 170.46 / 2.1307 | 114.05 / 1.4256 | 97.32 / 1.2165 | 147.91 / 1.8489 |
| Vehicle 2 距离(km)/时间(hr) | 237.67 / 2.9709 | 165.38 / 2.0672 | 293.13 / 3.6642 | 124.76 / 1.5596 |
| Total 距离(km) | 408.13 | 279.43 | 390.45 | **272.67** |
| Total 时间(hr) | 5.1016 | 3.4928 | 4.8807 | **3.4085** |
| High-Priority Rapid Delivery Rate (%) | 85% | 92% | 38% | **92%** |
| Priority First Ratio (%) | 89.36% | 93.62% | 89.36% | **93.62%** |

Table 4（KPI 对比，Proposed / FIFO / Historical）：
| 视角 | KPI | Proposed | FIFO | Historical |
|---|---|---|---|---|
| In-Warehouse | Ave fulfilment time (mins) ↓ | 9.85 | 9.68 | 10.74 |
| | On-time departure rate ↑ | **100%** | 70.59% | 58.80% |
| | High-priority scheduling rate ↑ | 84.62% | 76.92% | 69.23% |
| | On-time inventory pickup rate ↑ | **100%** | 70.59% | 73.53% |
| Land Transport | Average delivery time (mins) ↓ | **58** | 82 | 124 |
| | On-time delivery rate ↑ | 97.06% | 79.41% | 76.47% |
| | High-priority satisfaction rate ↑ | **92.31%** | 61.54% | 53.85% |
| | Vehicle utilization rate ↑ | 85.00% | 75.50% | – |
| | Total travel distance (km) ↓ | **209** | 390 | – |
| | Latest delivery time (mins) ↓ | 132 | 217 | 265 |
| Joint | High-priority service rate ↑ | 78.11% | 46.95% | 44.96% |
| | Average order completion time ↓ | 66.38 | 91.59 | – |
| | Longest completion time ↓ | 142 | 227 | – |
| | Shortest completion time ↓ | 19 | 26 | – |

其他原文数字：
- Abstract："achieving **100% on-time delivery rate** for warehousing operations. After joint optimization, the average delivery time for the last mile was reduced by **29.3% to 53.2%**, the total transportation distance was reduced by **46.4%**, the high-priority service rate was increased to **over 92%**"。
- 正文："reduces average delivery time to 58 minutes and total distance to 209 km (a **46.4% saving versus FIFO**) ... a **27.5% reduction in average order completion time (66.38 minutes)**"。
- 异构车队训练收敛："reaching optimal solutions by iterations **44–49**"；成本改善："improved short-to-medium distance transport costs by **44.2% for scooters (24 orders)** and **42.6% for motorcycles (30 orders)**, while boosting long-distance van transport by **63.6% (48 orders)**"。
- 仓内算法："MORM-AGDQN converges smoothly and maintains a narrow reward fluctuation band after convergence, particularly after **episode 600**"；规模从 1 AGV/17 订单 到 4 AGV/120 订单，"larger scenarios (up to 4 AGVs, 120 orders) ultimately achieve much higher cumulative rewards"。

### 4.5 自认局限（原文逐字 + 翻译）

> "Despite these contributions, some limitations exist. Current reinforcement learning frameworks rely on regional traffic information and predefined vehicle characteristics (based on historical data), which may not adequately reflect the volatility of the real-time operations. Future research can extend this work by incorporating segmented real-time traffic information, stochastic demand modelling, and adaptive fleet control strategies. Furthermore, the integration of emerging paradigms, such as digital twins or advanced learning architectures, can further enhance scalability and flexibility. Given the NP-hard nature of this problem, exact optimization is computationally prohibitive for large-scale scenarios. While our RL framework provides efficient approximate solutions, future integration with quantum computing holds the potential to further improve solution quality and scalability beyond classical limits."

翻译：尽管有上述贡献，仍存在一些局限。当前的强化学习框架依赖**区域性交通信息**和（基于历史数据的）**预设车辆特性**，这可能不足以反映实时运行的波动性。未来研究可纳入**分段的实时交通信息、随机需求建模与自适应车队控制策略**。此外，引入数字孪生或先进学习架构等新范式，可进一步提升可扩展性与灵活性。鉴于该问题的 NP-hard 性质，精确优化在大规模场景下计算上不可行；我们的 RL 框架提供高效近似解，未来与量子计算的结合有望在经典极限之外进一步提升解质量与可扩展性。

另有一处主动声明的范围裁剪（3.2 节）："Rather than incorporating complex variables like EV charging or vehicle leasing, we group fleets by service distance, capacity, and speed (Meng et al., 2019), deferring fully hybrid fleet scenarios to future research."

### 4.6 可复用

- **开源代码**：**未提供代码仓库**；论文引用的公开资源为 Kaggle notebook `https://www.kaggle.com/code/`（正文未给完整 slug）。
- **数据集**：Amazon delivery data（Alianwar, 2024）；地理数据 Google Maps；路径可视化用 Graphhopper。
- **评测协议**：三场景（Baseline / Peak Season 2–5× 订单 / Uncertainty）；"rolling baseline strategy"（仅在独立测试集上显著优于历史最优才更新模型，p < 0.05）；Table 3（4 方法 × 2 车 × 距离/时间/两个优先率）与 Table 4（三层 KPI × 3 策略）可直接作为对比模板。
- **可复用公式**：式(1)–(3) 静态/动态/混合多目标；式(5) 能耗 `Σ d·x·f(Q+w)`；式(32)–(38) 仓内状态与转移；式(39)–(49) **奖励机 + 权重空间**；式(45)–(48) Pareto 前沿指数衰减奖励（含边界定标）；式(56)–(60) 外部**乘积型**奖励；式(61) A\* 引导的 ε-greedy；式(62)–(64) 带 A\* 正则的 DQN 损失与 ε 衰减；式(65)–(67) 多目标注意力编码器。
- **方法论可复用点**：**双层/双向协调**（外层 Pareto → 仓内取货时序 → 外层路由输入，再回流）；4 通道张量状态表示 `N×N×4`；A\* 既作路径规划又作动作偏置与正则项的做法。

---

## 5. `arXiv2604.15572_BiLevel_PrioritySorting_AGV_Scheduling.pdf`

### 5.1 题录

| 项 | 内容 |
|---|---|
| 标题 | *A bi-level priority sorting framework for flexible AGV service scheduling in smart warehouses* |
| 作者 | Xiaozhu Sun（通讯，xiaozhu.sun@torontomu.ca）, Bilal Farooq |
| 单位 | Laboratory of Innovations in Transportation (LiTrans), Toronto Metropolitan University, Ontario, Canada |
| arXiv ID | arXiv:2604.15572v1 [math.OC], 16 Apr 2026 |
| 年 | 2026 |
| **是否正式发表** | 未见发表信息；arXiv 预印本（Springer 期刊版式，含 Declarations/Funding/Data availability/Code availability，无期刊名/DOI）。Declarations 中含"During the peer-review process, the code is available from the corresponding author upon reasonable request."→ 表明处于投稿审稿状态 |
| 资助 | Canada Research Chair (CRC-2021-00480)、NSERC Discovery (RGPIN-2020-04492) |
| 方法名 | **PDSP-AGMADQN / DCSP-AGMADQN**（A\* 引导多智能体 DQN） |

### 5.2 问题设定

- **问题类型**：智能仓库 AGV 柔性服务调度（分拣/配送），建模为 **增强版 DARP（Dial-A-Ride Problem）** 的 many-to-one ride-matching：一台 AGV 可载多单，每单只分配给一台 AGV。
- **双层结构**：
  - **Level 1（外部/高层）**：按客户优先级类别动态调整实时调度参数（订单承诺时间、延迟容忍度）。
  - **Level 2（内部/低层）**：为每台 AGV 做实时路由优化，寻找满足高层优先级的最短可行路径并避碰。
- **资源**：多容量 AGV（每台最多 **4 件**，可调）、网格化仓库地图、传送带、分拣区；仓库按存储容量分小/中/大三种规模，布局分 A–F 区（F 区需求密度最高，紧邻传送带）。
- **目标**：`min F = w_t × E[Power] + (1 − w_t) × E[Time]`（式 1）——能耗货币成本与系统时间成本（行驶、等待、延迟）的加权和；`w_t` 随订单优先级、客户重要性、上下游供应链影响动态调整。
  - 功率（式 2–5）：`P^k_unit = δ1 × G_k + δ2`；`G_k = K×G_0 + Σ g_i y_ik + Σ q_ijk`；`Po_k = P^k_unit × L_k`；`L_k = Σ d_ij x_ijk`；`E[Power]` 经电价折算为货币。
  - 时间（式 6–9）：`C_Time = C_waiting + β × C_delay`；`T_waiting = Σ T_wi`，`T_wi = max(0, s_i − r_i)`；`T_delay = Σ |T_dd_i − T_wi|`。
  - 单位库存持有成本（式 12–13）：`UIHC(min) = γ × AverageAnnualInventoryCost / (365×24×60×PackageQuantity)`，`γ ≃ 25%`。
  - 延迟成本曲线（式 14a–d）：四类客户——**Expedite**（零容忍）、**Standard Urgency**、**Intangible**、**Fixed Date**，各带截止期 `T_dd_i`、升级率 `λ_1,λ_2,λ_3`、封顶损失 `CA ≥ CB ≥ CC ≥ CD`（式 24）。
- **约束（式 15–24）**：流守恒（15, 17）、订单唯一分配（16）、AGV 容量（18）、重量上限（19）、时间窗内完成（20）、`T_dd_i ≥ T_arrival_i`（21）、优先级越高截止期越紧（22）、`0 ≤ w_t ≤ 1`（23）、`CA ≥ CB ≥ CC ≥ CD`（24）。
- **附加假设**：AGV 恒速（安全与可预测）、装卸时间不计入运输时间、走廊仅容一车（冲突靠优先级规则消解，序号小的车优先，其他车停车等待）、AGV 必须完成当前 loop 才能接新任务。

### 5.3 方法

**（a）6 条排序规则（Section 4.2）**：4 条经典 —— **FCFS**（严格按到达时间）、**SPT**（到达时间 + 最短行驶距离）、**EDT**（Earliest Due Time，最近截止期优先）、**LDC**（Least Delay Cost，按预计延迟罚金）；2 条新提出 ——
- **PDSP**（Priority, Deadline, Shortest Path）：先按客户类别 A→D 排序，再按容忍窗口（deadline − arrival）排序；按 AGV 容量分组后用 A\* 求最小距离路线。
- **DCSP**（Delay Cost, Shortest Path）：**忽略客户类别**，纯粹按非线性延迟成本函数动态排序，再用 A\* 求最短路线。
- 原文对比："the PDSP rule integrates external priorities, such as customer class and deadlines, with internal route optimization to minimize the travel distance. In contrast, the DCSP rule omits customer classification and prioritizes tasks solely based on the delay cost before optimizing the route sequence."

**（b）状态定义（式 26–28）**：
```
s^t_i = ( p^t_i, v^t_i, B, S_t, G^t_i, T^t_i, A^t_{−i} )
```
- `p^t_i = (x^t_i, y^t_i)` 当前位置；`v^t_i = (v^t_x, v^t_y)` 速度；
- `B = {(x_k,y_k)}^K_{k=1}` 障碍与货架；`S_t = {(x^t_l,y^t_l)}^L_{l=1}`；
- `G^t_i = ( {g1,g2,g3,g4}_{Sub-goals}, g_pick_{Final goal} )`（式 27，多订单子目标 + 最终取货站目标）；
- `T^t_i = ( priority, deadline, cost, visited-subgoals )`（式 28，任务属性）；
- `A^t_{−i} = {p^t_j, v^t_j}_{j≠i}`（其他 AGV 状态）。

**（c）动作空间**：A\* 生成访问全部 m 个任务的最优路径 → `a_k ← ConvertPathToAction(path_k)`；否则 DQN 选 `argmax_a Q(s_t,a;θ)` 或随机合法动作序列。冲突靠"序号小者优先、其余停车等待"的规则消解。容量分组：每台 AGV 每循环分配 **4 个顺序任务**（FCFS 规则表述为 "assigning each AGV four sequential tasks per cycle"）。

**（d）奖励**：论文**未给出显式奖励公式**。DQN 更新（式 29–30）：
```
y_t = r_t                                  if s_{t+1} is terminal
    = r_t + γ max_{a'} Q_{θ−}(s_{t+1}, a') otherwise
L(θ) = [ r_t + γ · max_{a_{t+1}} Q(s_{t+1}, a_{t+1}) − Q(s_t, a_t) ]²
```
Algorithm 1 中为 "Execute joint action a_t, observe r_t, s_{t+1}, done"，奖励由环境给出（成本/延迟驱动），未展开定义。A\* 代价（式 25）：`f(n) = g(n) + h(n)`。

**（e）算法**：A\* 引导的 **多智能体 DQN（AGMADQN）**，扩展自 Luo et al. (2023)，加入动态优先级与容量约束。网络结构：**两个卷积层 + 两个线性层**，从仓库环境（障碍、货架、其他 AGV）、AGV 位置与运动信息中提特征；`rand() < ξ` 用 A\* 最优路径，否则 `rand() < ε` 随机、再否则 `argmax Q`；目标网络每 C 步同步；`ε ← max(ε_min, ε·ε_decay)`。

### 5.4 实验

- **数据集**：Kaggle 公开电商运输数据（EDA）—— `https://www.kaggle.com/datasets/prachi13/customer-analytics`，**超过 10,000 条**运输记录（ID 范围 [1, 10977]）。
- **数据映射**：客户**评分**（rating 1–5）→ 优先级：rating 1 = 最高外部优先级（A 类），rating 4–5 = 最低（D 类）；产品价格 96–310 美元 → 库存持有成本；产品重量 1001–7846 克 → AGV 载重动态。
- **AGV 参数**：多层料箱搬运车，载重 270 kg，自重约 450 kg，电池 42 Ah / 24 V，满载平均可运行 **5 h**，恒速 **1 m/s**。
- **场景规模**：**10 个场景**，订单量 **1,000–10,000**，AGV 数 **1–10**；仓库分小/中/大三种规模；延迟罚金**每 10 s** 动态更新；算法对比实验用 2 AGV + 200 动态订单 + 小仓库（1,000 episodes，最终性能看 5,000 episodes），运行测试用 20 动态订单 + 10,000 episodes。
- **基线**：FCFS、SPT、EDT、LDC（+ 两个新规则 PDSP/DCSP）；算法层对比 **PDSP/DCSP-AGMADQN、PDSP/DCSP-DQN、PDSP/DCSP-AC (Actor Critic)、PDSP/DCSP-A\***。
- **指标**：**12 个 KPI**（service：travel time、waiting time、operation time、delay cost、inventory cost、order-related energy；AGV efficiency：idle time、running time、AGV energy use；composite：total energy cost、total time cost、total system cost），单位 s / USD / Wh；另有 convergence speed、stability、rewards、robustness。

**核心数字（照抄原文）**：
- "the proposed framework equipped with both heuristic rules consistently reduces average order delay and total system costs by **over 50%** during peak demand periods. This is achieved while maintaining a **service level above 90%**."
- 行驶时间："FCFS achieves the shortest travel time in all scenarios, reducing transport time by **13.34%** on average"；PDSP/DCSP 因优先级与重调度反而最长。
- 等待时间："EDT exceeded the best methods by **16.46% to 61.33%**"；规模增大后平均等待时间在中小规模下降 **10%–20%**，大规模下降 **over 40%**。
- 延迟成本："PDSP and DCSP reducing them by **50% to 80%**, and up to **90%** in large-scale settings (e.g., 8,000 to 9,000 orders with 8 to 9 AGVs)."
- 空闲时间："SPT, DCSP, and PDSP reduce the idle time by **30% to 50%** via cycle-based order sorting."
- 系统成本："the PDSP or DCSP cut system costs by **50% to 90%** during high-load periods compared to other methods."；"The PDSP is the most cost-effective on small or medium scales, whereas the DCSP outperforms as the scale increases."
- 敏感性（Table 3/4）：订单量上升时"**The DCSP cost increase rates (67.56% and 64.24%) are higher than those of the PDSP (52.93% and 44.58%)**, showing that the PDSP has a more stable response to rising demand."；下单频率 0–8s → 0–2s 时"**the DCSP reduces the cost by 77% and the PDSP by 61%**"；延迟窗口 5 min → 24 h 时窗口越长延迟与系统成本越低，"the DCSP showing sharper sensitivity to time pressure. PDSP's increase is smoother"。
- 权重扫描：`w = 0.9` 偏能耗、`w = 0.1` 偏时间关键场景。
- 算法对比："PDSP/DCSP-AGMADQN consistently achieved higher cumulative rewards and maintained a stable growth pattern"；"the combination of a high cumulative reward, low final loss, and minimal performance variance positions the MADQN as the superior algorithm in this multi-agent context"。

### 5.5 自认局限（原文逐字 + 翻译）

> "Owing to the limitations of the computational time and scale of reinforcement learning, this study only discusses cases in small-scale scenarios; however, it is sufficient to demonstrate the superiority of our method. For currently common large-scale problems, such as large scenarios, a large number of orders, and high-density demand, this study recommends that users model and evaluate within the framework of PDSP/DCSP-A*."

翻译：受强化学习计算时间与规模的限制，本研究只讨论了**小规模场景**的算例；但这已足以展示本方法的优越性。对当前常见的大规模问题（大场景、大订单量、高密度需求），本研究建议用户在 **PDSP/DCSP-A\*** 框架内建模与评估。

未来工作（Same paragraph 之后）：

> "we will transform the fixed delay curve into a dynamic one influenced by external warehouse factors, enabling the system to truly achieve internal and external linkage and ensuring the effectiveness and timeliness of services; we will combine the actual transportation results of the last mile with the optimization of warehouse scheduling to form a closed loop for the automated system; and we will explore more AI or reinforcement learning methods to make the automated logistics system more intelligent and stable."

翻译：我们将把固定延迟曲线改为受仓库外部因素影响的动态曲线，使系统真正实现内外联动、确保服务的有效性与及时性；我们将把最后一公里的实际运输结果与仓库调度优化结合，形成自动化系统的闭环；并将探索更多 AI/强化学习方法，使自动化物流系统更智能、更稳定。

另有一处方法依赖（4.3.2）：算法取自 Luo et al. (2023) 的 AGMADQN，"used in this study is one of the best to date, to the best of our knowledge"。

### 5.6 可复用

- **开源代码**：**暂无**。Declarations："All custom code and algorithms supporting the findings of this study will be made publicly available upon acceptance of the manuscript. The code repository will be hosted on GitHub. During the peer-review process, the code is available from the corresponding author upon reasonable request."
- **数据集**：Kaggle `https://www.kaggle.com/datasets/prachi13/customer-analytics`（>10,000 条，ID [1,10977]）；辅助 notebook `https://www.kaggle.com/code/niteshyadav3103/eda-e-commerce-shipping-data`。
- **评测协议**：10 个场景（订单 1,000–10,000 × AGV 1–10）；小/中/大三种仓库规模；延迟罚金每 10 s 动态更新；12 KPI + 累积奖励/损失/方差四轴对比；敏感性四维扫描（订单量、车队规模、下单频率、延迟时间窗）。
- **可复用公式**：式(1) 加权多目标；式(2)–(5) 功率模型 `P_unit = δ1·G + δ2`（G 含自重与动态载重）；式(6)–(9) 时间成本与延迟定义；式(12)–(13) 单位库存持有成本（γ≈25% 年库存费用）；式(14a–d) 四类客户的**分段/指数延迟成本曲线**；式(24) `CA ≥ CB ≥ CC ≥ CD`；式(26)–(28) 状态定义（含 `visited-subgoals` 与 `A_{−i}`）；式(25) A\* `f = g + h`。
- **方法论可复用点**：**双层优先级排序**（外部客户类别/延迟容忍 → 内部 A\* 最短可行路径）；把"优先级规则"与"A\* + MADQN"做正交组合的消融式对比设计（PDSP/DCSP × {A\*, DQN, AC, MADQN}）；用公开电商数据集的**客户评分**直接映射为优先级类别的做法。

---

## 附：五篇横向速查表

| # | 问题 | 方法 | 决策主体 | 动作空间 | 奖励 | 基线 | 开源 |
|---|---|---|---|---|---|---|---|
| 1 | JSSPT（机器+AGV） | GIN + MLP 双智能体，PPO/MAPPO（joint）vs 独立训练+DR 代理（modular） | 2 个（作业、AGV） | 层次：工序 E2E × 兼容 AGV E2E | `−Cmax/(LB·s)`（稀疏，仅终局） | 40 条 DR 组合 + 40 个模块化组合 | ✅ github.com/proto-lab-ro/jsspt-coordination-gap |
| 2 | 动态生产+AGV 的 SBO | LLM Manager+Editor 在策略**代码**上做搜索；仿真 trace 诊断 | LLM agent（策略代码） | 改代码（产线选择/任务排序/AGV 指派/充电） | 8 项 KPI 加权总分 0–100 | 滚动 MILP、135 规则组合、GA/DE/PSO | ❌（仿真器复用 supcon hackathon 仓库） |
| 3 | FMS：JSSP + AGV + 刀具共享 | CTPN + Maskable PPO（Actor-Critic）+ Lookahead | 1 个（单智能体控制全部可控变迁） | Petri 网可控变迁（job select / AGV select），guard 动态掩码 | `− (Σ idle machines / total machines)` | 12 启发式 + SOS 元启发式 | ❌ 文中无 URL |
| 4 | 仓内 AGV + 最后一公里异构车队 | MORM-AGDQN（CNN + 奖励机 + A\* 引导 DQN）；MRMH-HCVRP（多头上注意力 Encoder-Decoder + Actor-Critic） | 仓内多智能体 + 外部单智能体 | 仓内：前/后/左/右/停；外部：(车辆, 客户) 对 | 仓内：4 台奖励机 + 稀疏/即时加权 + Pareto 指数衰减；外部：`r_Pr·r_Tr·r_Con` 乘积 | PG/DQN/AGDQN/MO-AGDQN；Time-Seq、Shortest-Path、HCVRP；FIFO、Historical | ❌ |
| 5 | 智能仓库 AGV 分拣（双层优先级） | 双层：优先级参数 + A\* 路由；PGSP/DCSP × AGMADQN | 多 AGV（MADQN） | A\* 路径 → 动作序列；或 DQN 选动作 | 未显式给出（环境给出 `r_t`），DQN 更新式 29–30 | FCFS/SPT/EDT/LDC + {A\*, DQN, AC, MADQN} | ❌（承诺录用后开源） |

**与本方向最接近的一篇**：第 1 篇（CASE 2026）——直接研究"机器 + AGV 联合学习 vs 模块化学习"，并提供可复现的开源代码与完整的 (ρ, τ\*) 敏感性实验协议。


## 【组 g5】

### 事实卡 · G5（5 篇）

> 只记录事实，不做评价。所有引用均为原文逐字。检索日期：2026-09-30。

---

## 1. arXiv2506.16795 — Robust Dynamic Material Handling via Adaptive Constrained Evolutionary Reinforcement Learning

### 1.1 题录

| 项 | 内容 |
|---|---|
| 标题 | Robust Dynamic Material Handling via Adaptive Constrained Evolutionary Reinforcement Learning |
| 作者 | Chengpeng Hu, Ziming Wang, Bo Yuan, Jialin Liu, Chengqi Zhang, Xin Yao |
| 单位 | Eindhoven University of Technology (C. Hu)；SUSTech (Z. Wang, B. Yuan)；Lingnan University (J. Liu, X. Yao)；Hong Kong Polytechnic University (C. Zhang) |
| arXiv ID | arXiv:2506.16795v1 [cs.NE] 20 Jun 2025 |
| 年 | 2025 |
| **是否已发表** | **已发表**：IEEE Transactions on Neural Networks and Learning Systems (TNNLS), Vol. 36, Issue 10, pp. 19327–19341, Oct. 2025. DOI: 10.1109/TNNLS.2025.3582299 |
| 佐证 | PDF 首页脚注原文："This work is accepted by IEEE Transactions on Neural Networks and Learning Systems" |

### 1.2 问题设定

- **问题类型**：Dynamic Material Handling (DMH)，多 AGV 实时任务分配。建模为 Constrained MDP (CMDP)，记为元组 ⟨S, A, R, C, P, γ⟩。
- **资源**：一队 AGV `V`（n 台），三种状态 Ψ = {I, W, B} = Idle / Working / Broken；站点图 G(L, T)，L 为站点集（取货点、交货点、停车点），T 为路径集；任务池（staging list）。
- **动态事件**：新任务动态到达（任务总数 m 事先未知）、AGV 故障（故障后原地修理 v.rp 时间，释放当前任务）。
- **目标**：最小化 makespan F_m(π)（Eq.1）。
- **约束**：
  - 累计约束：tardiness F_t(π) ≤ ξ（ξ = 50），即 Eq.(5) `J_C^{πθ} = E_{πθ}(F_t(πθ)) ≤ ξ`。
  - 瞬时约束：Eq.(6) `v.ψ = I`（只有 Idle 状态的车才能被指派任务）。
- **难点（原文自述）**：reward 与 constraint 双向稀疏——"A feedback is received only when all tasks are served, which leads to sparse reward."；且多实例训练下计算资源分配存在 trade-off。

### 1.3 方法

**算法**：ACERL (Adaptive Constrained Evolutionary Reinforcement Learning)。含三个部件：

1. **Adaptive Instance Sampler (AIS)** —— 自适应实例选择
2. **Intrinsic Stochastic Ranking (ISR) with rank-based fitness** —— 内在随机排序 + 基于秩的适应度
3. **Natural Evolution Strategies (NES)** —— 自然梯度上升更新策略，无需反向传播

**网络结构**：两个隐层全连接层 128 × 128（"The network structure is formed by two hidden fully connected layers 128 × 128."）。无 critic（gradient-free）。

**状态定义**（原文）："the state space, S, is encoded by task and AGV information, such as the remaining time before timeout and waiting time of unassigned tasks in the task pool at decision time t."

**动作空间设计**（混合动作空间，与同组论文 4 一脉相承）：

> "The action space, A = D × V_t, is a hybrid space combining AGVs information V_t at t and dispatching rules D; for instance, it involves deciding a specific dispatching rule to assign a task to a chosen AGV."

即：动作 = (选一条调度规则 d ∈ D) × (选一台 AGV v ∈ V_t)，规则再决定把哪个任务派给该车。D 含 FCFS / EDD / NVF / STD。

**奖励（原文公式，逐字）**：

> "To maintain consistency between the reward function and objective function, the policy receives only the negative number of the final makespan as its reward, i.e., −F_m, while receive zero at other time. Similarly, the penalty function, C, is constructed based on the tardiness, where the policy gets −F_t once all tasks are served [18]."

即稀疏奖励：episode 终止时 R = −F_m，其余时刻 R = 0；代价（constraint）同理仅在终止时给 −F_t。

**关键公式（逐字）**：

- Eq.(1) makespan：
  `F_m(π) = max_{v∈V^π} FT(u^v_{|HL(v)|}, v)`
- Eq.(2) tardiness：
  `F_t(π) = (1/m) Σ_{v∈V^π} Σ_{i=1}^{|HL(v)|} max{FT(u^v_i, v) − u^v_i.o − u^v_i.τ, 0}`
- Eq.(3)(4)(5) 约束优化目标：
  `max_θ J_R^{πθ} = E_{τ∼πθ}[Σ_{t=0}^∞ γ^t R(s_t,a_t,s_{t+1})]`，`s.t. J_C^π ≤ ξ`，`a_t is legal, ∀t = 0,1,2,...`
- Eq.(6) 反向距离度量（评估实例优势）：
  `u_η = (1/|B_R^η|) Σ_{i=1}^{|B_R^η|} ( max_{1≤k≤|B_R^η|} J_R^k − J_R^i ) / ( max_{1≤k≤|B_R^η|} J_R^k − min_{1≤k≤|B_R^η|} J_R^k )`
- Eq.(7) UCB 自适应实例采样：
  `UCB = u_i + α_u sqrt( log(Σ_j^K N_j) / N_i )`
- Eq.(8) 罚函数：
  `φ(π) = (max{0, J_C^π − ξ})²`
- Eq.(9) NES 梯度估计：
  `∇_ψ E_{θ∼p_ψ} F(θ) = E_{θ∼p_ψ} F(θ) ∇_ψ log(p_ψ(θ))`
- Eq.(10) `E_{θ∼p_ψ} F(θ) = E_{ϵ∼N(0,I)} F(θ + σϵ)`
- Eq.(11) vanilla estimator：`∇_θ^ϵ F(θ + σϵ) = (1/σ) E_{ϵ∼N(0,I)} [F(θ + σϵ) ϵ]`
- Eq.(12) antithetic estimator：`∇_θ^ϵ F(θ + σϵ) = (1/2σ) E_{ϵ∼N(0,I)} [(F(θ + σϵ) − F(θ − σϵ)) ϵ]`
- Eq.(14) 光滑化罚函数：`φ'(θ) = ρ ln(1 + e^{(g(θ)−ξ)/ρ})`
- Eq.(15) 随机排序下的松弛目标：`f_SR(θ) = p_f F(θ) − (1 − p_f) φ'(θ)`
- Eq.(17) Theorem 2 梯度估计偏差界：`‖∇_θ E_{ϵ∼N(0,I)} f_SR(θ + σϵ) − ∇f_SR(θ)‖ ≤ E`

**评估指标公式（逐字）**：

- Eq.(18) `M = (1/K) Σ_{j=1}^K (F_m^max − F_m^j) / (F_m^max − F_m^min)`
- Eq.(19) `C = (1/K) Σ_{j=1}^K (F_t^max − F_t^j) / (F_t^max − F_t^min)`
- Eq.(20) `P = (1/K) Σ_{j=1}^K 1{F_t^j < ξ}`

（M = 归一化 makespan 得分，C = 归一化 tardiness 得分，P = 约束满足百分比；三者均为越大越好。）

**超参数**：γ = 0.97；种群规模 λ = 256；代数 G = 128；约束阈值 ξ = 50；RCPOM 的初始乘子 λ = 0.001、学习率 0.0001；训练 1e6 steps；5 个随机种子；每个策略独立测 30 次。基线实现基于 Tianshou 框架。

### 1.4 实验

- **实例**：模拟器 DMH-GYM 及其公开实例。训练集 DMH-01 ~ DMH-08（"drawn from different distributions using a searching-based method"），测试集 DMH-09 ~ DMH-16（"generated by noising the training instances"）。
  - 额外：40 个加噪实例，扰动强度 DMH±0 / ±5 / ±10 / ±15 / ±20 / ±25 / ±30（扰动对象为每个任务的到达时间）。
  - 额外：leave-one-out 交叉验证（依次留出 DMH-01 ~ DMH-08）。
- **基线（5 组）**：
  1. CRL/MARL：MAPPO [33]、RCPOM [5]、固定拉格朗日乘子的 SAC（"LSAC" [56]）
  2. RL：SAC、PPO
  3. 上述五者 + 自适应实例采样器：AMAPPO / ARCPOM / ALSAC / ASAC / APPO
  4. 调度规则：FCFS、EDD、NVF、STD
  5. 随机策略：MIX（随机选规则）、Random（随机选任务）
- **指标**：归一化 makespan M、归一化 tardiness C、约束满足百分比 P；另报训练耗时与单次决策耗时。

### 1.5 核心数字（照抄原文）

**主结果（Table I 训练集 / Table II 测试集）**

| 算法 | 训练集 M/C(P) | 测试集 M/C(P) |
|---|---|---|
| ACERL | 0.90/0.90 (100%) | 0.89/0.92 (97%) |
| MAPPO | 0.72/0.72 (62%) | 0.70/0.73 (65%) |
| RCPOM | 0.71/0.75 (60%) | 0.71/0.76 (63%) |
| LSAC | 0.55/0.68 (56%) | 0.57/0.70 (56%) |
| SAC | 0.57/0.69 (57%) | 0.59/0.70 (58%) |
| PPO | 0.61/0.71 (58%) | 0.64/0.73 (59%) |
| FCFS | 0.25/0.45 (37%) | 0.27/0.45 (35%) |
| EDD | 0.52/0.85 (86%) | 0.48/0.74 (73%) |
| NVF | 0.56/0.63 (51%) | 0.59/0.66 (58%) |
| STD | 0.65/0.70 (53%) | 0.65/0.71 (55%) |
| MIX | 0.49/0.65 (54%) | 0.49/0.66 (54%) |
| Random | 0.02/0.01 (12%) | 0.02/0.03 (11%) |

原文："ACERL achieves 100% and 97% constraint satisfaction on training and test instances, respectively, which is much better than other algorithms. For example, RCPOM [5], a state-of-the-art CRL method, only gets 60% and 63% constraint satisfaction on training and test instances, respectively."

**单实例点值（Table I/II）**：ACERL 在 DMH-01 得 F_m/F_t = 1797.8/29.5；DMH-07 得 1929.6/9.5；DMH-09 得 1801.6/29.9；DMH-16 得 1857.0/25.8。

**耗时（原文）**：
- "The training time of ACERL is about 30 minutes, while other RL and CRL-based methods take more than 1 hour for training 1e6 steps."
- "Moreover, the computation time of ACERL for one single decision is about 2 ms, which meets the real-time requirement [6]."

**加噪实例（Table III，M/C(P)）**

| 算法 | ±0 | ±5 | ±10 | ±15 | ±20 | ±25 | ±30 |
|---|---|---|---|---|---|---|---|
| ACERL | 0.90/0.90 (100%) | 0.89/0.92 (97%) | 0.90/0.94 (80%) | 0.94/0.94 (95%) | 0.87/0.92 (75%) | 0.89/0.96 (84%) | 0.93/0.96 (93%) |
| RCPOM | 0.71/0.75 (60%) | 0.71/0.76 (63%) | 0.85/0.88 (64%) | 0.85/0.88 (82%) | 0.87/0.91 (66%) | 0.81/0.84 (60%) | 0.87/0.87 (70%) |
| NVF | 0.56/0.63 (51%) | 0.59/0.66 (58%) | 0.77/0.82 (61%) | 0.76/0.77 (65%) | 0.86/0.89 (77%) | 0.84/0.86 (62%) | 0.92/0.89 (81%) |

原文："Although RCPOM achieves a slightly higher M with 0.88 over ACERL with 0.87, its constraint satisfaction percentage P is rarely higher than ACERL's, i.e., 66%<75% in DMH±20."

**消融（Table V/VI，训练集 M/C(P)）**：ISR+AIS = 0.90/0.90 (100%)；ISR+Uniform = 0.88/0.89 (91%)；ISR+Random = 0.87/0.86 (94%)；CF+AIS = 0.88/0.93 (95%)；R+AIS = 0.89/0.91 (93%)；F+AIS = 0.88/0.86 (87%)。

**单实例训练消融（Table XIII）**：ISR+DMH-1 得 0.67/0.77 (63%)；ISR+DMH-7 最低 0.52/0.64 (52%)。

**复杂度（原文）**："the optimisation complexity scales linearly with the number of weights, i.e., O(|θ|). The expected covering time to ensure all weights dimensions are sufficiently perturbed is at least polynomial bound O(|θ| log |θ|). Given the population size λ, the space complexity and the communication complexity are both O(λ|θ|)"。

### 1.6 自认局限（原文逐字 + 中译）

> **L1.** "However, we also find that ACERL does not perform well on the left instance DMH-01 as expected, with Fm = 1914.1 and Ft = 54.0. At the same time, STD, a dispatching rule achieves the best makespan value of 1868.8 and EDD achieves the lowest tardiness Ft = 33.9. The phenomenon is attributed to the out-of-distribution. The diverse training instances are intentionally designed to facilitate a comprehensive evaluation, and thus instances may differ from each other significantly. It is not surprising to observe the limited performance of learning policies on DMH-01, which is separated from the training set as they have never encountered the instance before."

中译：然而我们也发现，ACERL 在留出的实例 DMH-01 上并未如预期表现良好，F_m = 1914.1、F_t = 54.0。同时，调度规则 STD 取得最佳 makespan 1868.8，EDD 取得最低 tardiness F_t = 33.9。该现象归因于分布外（out-of-distribution）。多样化的训练实例是为便于全面评估而刻意设计的，因此实例之间可能差异显著。学习类策略在 DMH-01 上表现有限并不意外，因为它被从训练集中分离，这些策略此前从未见过该实例。

> **L2.** "Although ACERL does not demonstrate superior performance in terms of makespan on the split instance, it still achieves the overall best performance."

中译：尽管 ACERL 在留出实例上的 makespan 未展现出优越性能，它仍取得了总体最佳表现。

> **L3.** "In future work, it is worth applying ACERL to more real-world problems such as vehicle routing and grid optimisation. It is also interesting to figure out delicate mechanisms for a better generalisation. Besides, theoretical aspect presents an intriguing avenue for future research, potentially bridging the gap between empirical success and theoretical understanding in constrained evolutionary reinforcement learning. Theoretical analysis will be a valuable yet challenging future work."

中译：未来工作值得将 ACERL 应用于更多现实问题，如车辆路径与电网优化。设计更精细的机制以获得更好泛化也很有意思。此外，理论层面是一条诱人的研究路径，可能弥合约束进化强化学习中经验成功与理论理解之间的鸿沟。理论分析将是宝贵但充满挑战的未来工作。

> **L4.** "A policy may perform well on the given instances, while failing on others, as it might not have been sufficiently trained on certain instances with limited computational budget."

中译：一个策略可能在给定实例上表现良好，却在其他实例上失败，因为受限于计算预算，它可能未在某些实例上得到充分训练。

> **L5.** （对基础方法 RCPOM 的批评）"However, RCPOM suffers from sparse feedback from both reward and constraint sides. And the robust performance across multiple DMH problem instances is not fully addressed."

中译：然而 RCPOM 在奖励与约束两侧都受稀疏反馈困扰，且跨多个 DMH 问题实例的鲁棒性能未被充分解决。

> **L6.** （对 ES 类方法的评述，间接自陈）"However, on the other hand, the inherent population and elite survival prevent it from training on multiple scenarios at the same time."

中译：但另一方面，固有的种群与精英保留机制使其无法同时面向多个场景训练。

> **说明**：该文**无独立 "Limitations" 章节**。以上为散见于第 V-D、V-E、VI 节及第 II-C 节的自认局限表述。

### 1.7 可复用

- **开源代码（RL 策略）**：https://github.com/HcPlu/ACERL —— 论文脚注原文："Code: https://github.com/HcPlu/ACERL"
- **开源模拟器 + 数据集**：https://github.com/HcPlu/DMH-GYM —— 论文脚注原文："3https://github.com/HcPlu/DMH-GYM"；"The experiments are conducted on publicly available DMH instances and simulator, DMH-GYM, provided in [5]."
- **依赖框架**：Tianshou，https://github.com/thu-ml/tianshou
- **数据集划分协议**：DMH-01~08 训练 / DMH-09~16 测试（测试集由训练集加噪生成）；40 个加噪实例按 ±10/±15/±20/±25/±30 分档；leave-one-out 交叉验证 8 折。
- **评测协议**：三指标 M / C / P（Eq.18-20），均为越大越好；每策略 5 seeds × 30 次独立试验；结果表用 "+/≈/−" 标注相对 ACERL 的统计显著性。
- **可复用公式**：makespan Eq.(1)、tardiness Eq.(2)、UCB 实例采样 Eq.(7)、随机排序目标 Eq.(15)、ES 反对称梯度估计 Eq.(12)、归一化三指标 Eq.(18)-(20)。
- **可复用机制**：AIS（UCB + 反向距离度量做课程/实例选择）；ISR（无需时域信息的 rank-based fitness，天然适配稀疏反馈）；小样本预算下的种群 ES 更新。

---

## 2. arXiv2402.08979 — Learning-enabled Flexible Job-shop Scheduling for Scalable Smart Manufacturing

### 2.1 题录

| 项 | 内容 |
|---|---|
| 标题 | Learning-enabled Flexible Job-shop Scheduling for Scalable Smart Manufacturing |
| 作者 | Sihoon Moon, Sanghoon Lee, Kyung-Joon Park（通信作者） |
| 单位 | Department of Electrical Engineering and Computer Science, DGIST, Daegu 42988, South Korea |
| arXiv ID | arXiv:2402.08979v1 [eess.SY] 14 Feb 2024 |
| 年 | 2024 |
| **是否已发表** | **已发表**：Journal of Manufacturing Systems (Elsevier), Vol. 77, pp. 356–367, December 2024. DOI: 10.1016/j.jmsy.2024.09.011 |
| 备注 | PDF 使用 IEEEtran 模板（页眉 "JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015"），实际发表于 Elsevier JMS |

### 2.2 问题设定

- **问题类型**：FJSPT —— Flexible Job-shop Scheduling Problem with Transportation constraints（柔性作业车间调度 + 运输约束）。
- **资源**：n 个工件 J = {J₁,...,Jₙ}；m 台机器 M = {M₁,...,Mₘ}；v 台 AGV V = {V₁,...,V_v}。
- **目标**：最小化 makespan，Eq.(1)：`C_max = max C_{in_i}, ∀i ∈ {1,...,n}`，其中 C_{in_i} 为工件 J_i 最后一道工序 O_{in_i} 的完工时间。
- **约束/机制**：
  - 每个工件 J_i 含 n_i 道有先后约束的工序 O_i = {O_{i1},...,O_{in_i}}；
  - 工序 O_ij 只能在兼容机器子集 M_ij ⊂ M 上加工，加工时间 T^p_{ijk} 随机器不同；
  - 运输时间 = off-load 时间 T^t_{iju} + on-load 时间 T^t_{kk'}；on-load 时间假设所有车相同；
  - 机器不抢占（preemption）：机器 Mk 被占用后从其他工序的邻居集中移除；车辆同理。
- **核心挑战（原文自述）**：scale generalization（规模泛化）与 end-to-end decision making（端到端决策，而非从预定义规则里选）。
  - "Conventional DRL-based schedulers are trained on specific-scale instances... their efficacy diminishes substantially for unseen large-scale instances"
  - "it is impractical and costly to constantly retrain the scheduler in response to scale changes"

### 2.3 方法

**算法**：HGS (Heterogeneous Graph Scheduler)，三个部件：

1. **异构图表征**：`H = (O ∪ M ∪ V, C, E_m ∪ E_v^off ∪ E_v^on)`
   - 相比传统 disjunctive graph，加入机器节点集 M、车辆节点集 V、兼容机器弧集 E_m、兼容车辆弧集 E_v；
   - 动态图：`H_t(O ∪ M ∪ V, C, E_{m_t} ∪ E_{v_t}^off ∪ E_v^on)`，选中动作 (O_ij, M_k, V_u) 后仅保留被选中的边，其余兼容边删除（含被抢占机器/车辆的边）；
   - E_m 元素 ε^m_{ijk} = O_ij 在 M_k 上的加工时间；E_v^off 元素 ε^v_{iju} = V_u 到 O_ij 工件的空载运输时间；E_v^on 元素 ε^v_{kk'} = 从 M_k 到 M_{k'} 的满载运输时间。
2. **结构感知异构编码器**：3 个子编码器 F_O / F_M / F_V（各自只聚合本类节点的邻居）+ 1 个全局编码器 F_G。每层含 HMHA（Heterogeneous Multi-Head Attention）+ AN（instance normalization）+ FF。
   - 关键设计动机：机器节点只看加工时间、车辆节点只看运输时间，降低编码复杂度、提升规模泛化。
3. **三阶段解码器**：按 O → M → V 顺序串行选择，构造复合动作 a_t = (O_ij, M_k, V_u)。

**网络结构（Sec. V-A2 逐字）**："The HGS model is constructed by stacking L = 2 encoding layers. The embedding dimension of nodes and edges, d_h and d_e, is set to 128 and 1, respectively. Additionally, we use d_z = 16 for calculating augmented compatibility and d_ff = 512 in the "Feed-forward" blocks. The (H)MHA blocks employed in both the encoder and decoder utilize Z = 8 attention heads. Each attention head processes query, key, and value as 8-dimensional vectors, denoted as d_q = d_k = d_v = 8. To optimize the model, we employ the Adam optimizer with a learning rate of 2 × 10⁻⁴ and use a batch size B = 50. In the training process, an epoch corresponds to the training of the model on E′ = 1,000 episodes. We train E = 1,000 epochs for given instance parameters n × m × v."

**状态定义（原始特征向量，逐字）**：

- 工序节点 μ_ij ∈ R⁷（7 维）：
  - Status：二值，若 O_ij 在 t 前已被调度为 1，否则 0
  - Number of neighboring machines: `|Nm_t(O_ij)|`
  - Number of neighboring vehicles: `|Nv_t(O_ij)|`
  - Processing time: 若已调度则 T^p_{ijk}，否则兼容机器平均 `T̄^p_ij = Σ_{M_k∈M_ij} T^p_{ijk} / |M_ij|`
  - Number of unscheduled operations in job J_i: `n_i − |F_t(J_i)|`
  - Job completion time: C_i，未完工时为估计值 `Ĉ_i = C_{ij'} + Σ_{O_ij∈J_i\F_t(J_i)} T̄^p_ij`
  - Start time: 已调度给实际值，否则估计 `T^s_ij = C_{ij'} + Σ_{z=j'+1}^{j-1} T̄^p_iz`（j' < j−1）；`T^s_ij = C_{ij'}`（j' = j−1）
- 机器节点 μ_k ∈ R⁴：Status（是否加工中）、`|N_t(M_k)|`、Available time（完成所有已分配工序、可接新工序的时间）、Utilization（机器使用时间与当前 t 之比）
- 车辆节点 μ_v ∈ R⁴：Status（是否运输中）、`|N_t(V_u)|`、Available time、Current location `L_t(V_u) ∈ M`
- 边特征：ν^m_{ijk} = T^p_{ijk}；ν^v_{iju} = T^t_{iju}；ν^v_{kk'} = T^t_{kk'}（均为标量）

**动作空间设计（复合动作，逐字）**：

> "To address FJSPT, we define a composite action, a_t = (O_ij, M_k, V_u), which constitutes operation selection, machine assignment and vehicle utilization. This implies that the operation O_ij is assigned to the machine M_k, with Vu transporting it. Specifically, action a_t ∈ A_t is to select a feasible operation-machine-vehicle pair. Feasible operation O_ij implies its immediate predecessor O_{i(j−1)} has been completed. A feasible machine is an idle one among the compatible machines, M_k ∈ M_ij. A feasible vehicle is an idle one among all vehicles at time t, V_u ∈ V_t."

三阶段选择概率（Eq.17-26）：

- 上下文节点：`h_c,t^(L) = [ h̄_t^(L) ‖ h_{g,t−1} ]`（Eq.17），图嵌入取所有节点嵌入均值
- 工序选择：`h_c^(L+1) = MHA_c^(L+1)({h_ij^(L) | O_ij ∈ O})`（Eq.18）
  `σ_cij^(L+1) = C · tanh( [h_c^(L+1)]ᵀ h_ij^(L) / sqrt(d_k) )`，若 O_ij 合法；否则 −∞（Eq.19，C = 10 用于裁剪）
  `Pr(O_ij|s_t) = e^{σ_cij^(L+1)} / Σ_{i'}Σ_{j'} e^{σ_ci'j'^(L+1)}`（Eq.20）
- 机器选择：`h_c^(L+2) = MHA_c^(L+2)({h_k^(L) + h_ijk^(L) | M_k ∈ M_ij, selected O_ij})`（Eq.21）
  `Pr(M_k|s_t, O_ij) = e^{σ_ck^(L+2)} / Σ_{k'} e^{σ_ck'^(L+2)}`（Eq.22）
- 车辆选择：`h_c^(L+3) = MHA_c^(L+3)({h_u^(L) + h_iju^(L) | V_u ∈ V, selected O_ij})`（Eq.23）
  `Pr(V_u|s_t, O_ij) = e^{σ_cu^(L+3)} / Σ_{u'} e^{σ_cu'^(L+3)}`（Eq.24）
- glimpse 更新：`h_{g,t} = h_ij^(L) + h_k^(L) + h_u^(L)`（Eq.25）
- 复合动作概率：`Pr(a_t|s_t) = Pr(O_ij|s_t) Pr(M_k|s_t, O_ij) Pr(V_u|s_t, O_ij)`（Eq.26）

**奖励（原文公式，逐字）**：

> "We construct the reward function as the difference between the makespan corresponding to s_t and s_{t+1}, r(s_t, a_t, s_{t+1}) = C_max(s_t) − C_max(s_{t+1})."

其中 C_max(s_t) 由未调度工序的估计完工下界定义：`C_LB(O_ij, s_t) = C_LB(O_{i(j−1)}, s_t) + T̄^p_ij`（若 O_{i(j−1)} 已调度则用实际完工时间 C_{i(j−1)} 更新），`C_max(s_t) = max_{i,j}{C_LB(O_ij, s_t)}`；终止态 `C_max(s_{|O|}) = C_max`。

原文："When the discount factor γ = 1, the cumulative reward is G = Σ_{t=0}^{|O|} r(s_t, a_t, s_{t+1}) = C_max(s_0) − C_max, where C_max(s_0) is a constant for a specific instance. Therefore, maximizing G is equivalent to minimizing the makespan."

**训练算法**：REINFORCE，梯度 `∇_θ J(θ) ← (1/B) Σ_{b=1}^B (G(τ^b) − G_base^b) ∇_θ log π_θ(τ^b)`（Eq.27），baseline 为 deterministic greedy rollout baseline；每 20 个 epoch 重新生成一批 B 个实例。

### 2.4 实验

- **实例生成（合成）**：工序数 `n_i ∼ U(0.8|M|, 1.2|M|)`；兼容机器数 `|M_ij| ∼ U(1, |M|)`；`T^p_{ijk} ∼ U(0.8 T̄^p_ij, 1.2 T̄^p_ij)`；`T^t_{kk'} ∼ U(0.8 T̄^t_{kk'}, 1.2 T̄^t_{kk'})`；`T̄^p_ij ∼ U(1,30)`，`T̄^t_{kk'} ∼ U(1,20)`。
  - 小规模：n×m×v = 5×3×3、10×3×6、10×6×3、10×6×6，工序数范围 [10, 70]，每尺寸生成 100 个实例取平均。
  - 大规模泛化：训练于 10×6×6，测试于 20×10×10、30×15×15、40×20×20、50×25×25，工序数范围 [160, 1000]。
- **公开基准**：Brandimarte 数据集（10 个实例 MKT01–MKT10，10/15/20 工件，55–240 道工序，4–15 台机器，机器间运输时间随机生成于 2–10）；车辆数 `v ∼ U(0.8m, 1.2m)`。DRL 方法统一用 10×6×6 上训练的模型。
- **基线**：SPT（最短加工时间优先）、LPT（最长加工时间优先）、FIFO（调度规则）；MatNet [8]（DRL，原只解 FJSP，作者加 NVS 最近车辆选择）；HGNN [7]（DRL，同样加 NVS）；IGA [28]（改进遗传算法，元启发式）。
- **指标**：makespan C_max、相对 gap `Gap = (C_max / C_max^BS − 1) × 100%`（Eq.28，C_max^BS 为该实例上所有方法的最优解）、运行时间（秒）。

### 2.5 核心数字（照抄原文）

**小规模实例（Table II，格式：C_max / Gap）**

| 规模 | SPT | LPT | FIFO | IGA | HGNN | MatNet | **HGS** |
|---|---|---|---|---|---|---|---|
| 5×3×3 | 102.75 / 12% | 107.95 / 26% | 105.3 / 23% | 109.95 / 28% | 105.95 / 24% | 96.55 / 13% | **85.65 / 0%** (0.2s) |
| 10×3×6 | 182.75 / 20% | 183.75 / 21% | 189.7 / 25% | **175.0 / 15%** | 183.55 / 21% | 170.55 / 12% | 152.05 / 0% (0.4s) |
| 10×6×3 | 422.15 / 37% | 446.75 / 45% | 420.2 / 37% | 332.9 / 8% | **307.45 / 0%** | 371.65 / 21% | 318.7 / 4% (1.33s) |
| 10×6×6 | 283.7 / 22% | 301.7 / 29% | 275.3 / 18% | 260.9 / 12% | 274.9 / 18% | 234.65 / 1% | **233.5 / 0%** (0.95s) |

原文："Especially, it can obtain up to 24% gap and 29% gap when compared with other DRL-based methods and dispatch rules, respectively."

**大规模泛化（Table III）**

| 规模 | SPT | LPT | FIFO | IGA | HGNN | MatNet | **HGS** |
|---|---|---|---|---|---|---|---|
| 20×10×10 | 632.6 / 24% | 669.85 / 31% | 642.15 / 25% | 692.75 / 35% | 635.85 / 24% | 555.05 / 8% | **511.7 / 0%** |
| 30×15×15 | 937.6 / 26% | 977.4 / 32% | 926.35 / 25% | 1108.45 / 49% | 904.65 / 22% | 800.05 / 8% | **742.0 / 0%** |
| 40×20×20 | 1206.05 / 25% | 1287.15 / 33% | 1219.05 / 26% | 1497.3 / 55% | 1181.9 / 22% | 1028.62 / 6% | **968.15 / 0%** |
| 50×25×25 | 1572.35 / 26% | 1660.05 / 33% | 1580.5 / 26% | 2009.1 / 61% | 1510.4 / 21% | 1337.5 / 7% | **1251.4 / 0%** (33.06s) |

原文："it achieves an increased gap ranging [24%, 28%] (from 24% in instance 20×10×10 to 28% in instance 50×25×25) for SPT, [30%, 36%] for LPT, [26%, 31%] for FIFO, [41%, 54%] for IGA and [21%, 25%] for HGNN."

**Brandimarte 基准（Table IV，10 实例平均 gap）**

| 方法 | SPT | LPT | FIFO | IGA | HGNN | MatNet | **HGS** |
|---|---|---|---|---|---|---|---|
| 平均 Gap | 28% | 35% | 23% | 12% | 14% | 10% | **3%** |

原文："As depicted in Table IV, the proposed method finds the best solutions for most instances, with the exception of only three instances (MKT01, 06 and 08). When comparing with DRL-based methods (HGNN and MatNet), the proposed method significantly enhances performance, with the average gap difference reaching up to 9%. Compared with the IGA method (average gap of 12%), HGS yields better results and is much less computationally expensive."

单实例：MKT09 上 MatNet 与 HGS 同为 0% gap（528 vs 529）；MKT06 上 HGS 217 (5%) 劣于 MatNet 206.5 (0%)；MKT08 上 HGS 812 (8%) 劣于 IGA 752 (0%)。

**消融（Fig.6）**：
- HGS (non-graph)：在训练规模 10×6×6 上平均 gap 29%，在未见规模 50×25×25 上 gap 增至 41%
- HGS (with self-attention)：训练规模上 gap 约 1%（部分测试甚至更优），未见规模 50×25×25 上 gap 增至 5%

**训练曲线（Fig.5）**：10×6×6 实例，10 次验证运行、每 100 episode 记录一次，makespan 从约 300 收敛至约 230–240 区间。

**IGA 耗时对比**：5×3×3 上 272.51s；10×3×6 上 569.8s；10×6×3 上 1005.5s；10×6×6 上 1005.91s；大规模实例普遍 1000s+（如 50×25×25 为 1186.63s），而 HGS 为 0.2–33.06s。

### 2.6 自认局限（原文逐字 + 中译）

> **说明**：该文**无独立 "Limitations" 章节，也无 "Future Work" 章节**。以下为原文中作者明确承认的不足与失败情形的表述。

> **L1.** "For 10×3×6 instance, the meta-heuristic method (IGA) yields the best solution. However, it is notable that this method necessitates extensive computational resources. Our proposed method, in contrast, obtains a near-best solution (with a 1% gap) while demanding a more reasonable computational cost."

中译：在 10×3×6 实例上，元启发式方法（IGA）取得最优解。但值得注意的是该方法需要大量计算资源。相比之下，我们的方法在更合理的计算成本下获得接近最优的解（1% gap）。

> **L2.** "In the case of 10×6×3 instance, which is characterized by an insufficient number of vehicles, other DRL-based methods such as MatNet and HGNN achieve the best solution while requiring relatively lower computation times However, these methods experience degraded performance in other instances."

中译：在车辆数量不足的 10×6×3 实例上，MatNet 与 HGNN 等其他 DRL 方法取得最优解，且所需计算时间相对更低。然而这些方法在其他实例上性能退化。

> **L3.** "As depicted in Table IV, the proposed method finds the best solutions for most instances, with the exception of only three instances (MKT01, 06 and 08)."

中译：如表 IV 所示，所提方法在多数实例上找到最优解，仅有三个实例（MKT01、06、08）例外。

> **L4.** "The one-to-all connection in HGS (self-attention) proves challenging to generalize as the graph size increases, due to the complex node relationships. These observations suggest that the sub-graph-based one-to-few connection in HGS offers superior generalization to unseen large-scale instances, attributable to fewer node connections and the inductive bias between node classes."

中译：随着图规模增大，HGS（自注意力）中的一对全连接被证明难以泛化，原因是节点关系复杂。这些观察表明，HGS 中基于子图的一对少连接对未见大规模实例具有更优的泛化能力，归因于更少的节点连接以及节点类别之间的归纳偏置。

> **L5.** （对既有 DRL 方法的问题陈述，HGS 声称解决之）"The main drawback of these representations is that the vector size is fixed, making them inflexible and unable to solve problems of varying sizes."

中译：这些（MLP/CNN）表征的主要缺点是向量尺寸固定，使其不灵活、无法求解不同规模的问题。

> **L6.** （关于 DRL 基线适配的说明，反映评测条件的局限）"MatNet [8]: A DRL-based algorithm for solving FJSPT has not been developed entirely, and this aims only to solve FJSP. In light of this, we augment these algorithms with a simple vehicle selection mechanism, namely the nearest vehicle selection (NVS) method, to make them applicable to FJSPT."

中译：MatNet [8]：完整求解 FJSPT 的 DRL 算法尚未被开发，该方法仅旨在求解 FJSP。鉴于此，我们为其增补一个简单的车辆选择机制，即最近车辆选择（NVS）方法，使其适用于 FJSPT。

### 2.7 可复用

- **开源代码**：**未提供**。全文无代码仓库链接，无 "Code available at" 声明。
- **数据集**：
  - 合成实例生成器参数完整公开（见 2.4 节），可在 `n_i ∼ U(0.8|M|, 1.2|M|)`、`|M_ij| ∼ U(1,|M|)`、`T^p ∼ U(0.8T̄^p, 1.2T̄^p)`、`T̄^p ∼ U(1,30)`、`T̄^t ∼ U(1,20)` 下复现；
  - 公开基准 Brandimarte MKT01–MKT10 的 FJSPT 改编版来自 Homayouni & Fontes, Journal of Global Optimization 79(2):463–502, 2021。
- **评测协议**：Gap 定义 `Gap = (C_max / C_max^BS − 1) × 100%`（Eq.28）；小规模每尺寸 100 个实例取平均；DRL 方法统一用 10×6×6 训练模型跨规模测试（这是该文泛化评测的关键协议）。
- **可复用公式/模块**：
  - 异构图表征 H = (O∪M∪V, C, E_m ∪ E_v^off ∪ E_v^on)，可直接用于任何 FJSPT/含运输的调度建模；
  - 复合动作三段式概率分解 Eq.(26)：`Pr(a_t|s_t) = Pr(O|s) Pr(M|s,O) Pr(V|s,O)`；
  - 差分类奖励 `r = C_max(s_t) − C_max(s_{t+1})`（用估计下界构造稠密信号，γ=1 时累计回报等价于 makespan 改进量）—— 这是一个可跨问题迁移的 reward 设计；
  - 带边属性的 HMHA，增广兼容度 Eq.(8)：`σ̃_xy = W_xy^{e2} · ReLU(W_xy^{e1}[σ_xy ‖ h_xy^{(l−1)}])`；
  - greedy rollout baseline + REINFORCE 训练流程（Alg.1）。
- **节点特征维度模板**：工序 7 维 / 机器 4 维 / 车辆 4 维（见 2.3 节），可直接复用作为 GNN 输入规范。

---

## 3. arXiv2506.13566 — A Production Scheduling Framework for Reinforcement Learning Under Real-World Constraints

### 3.1 题录

| 项 | 内容 |
|---|---|
| 标题 | A Production Scheduling Framework for Reinforcement Learning Under Real-World Constraints |
| 作者 | Jonathan Hoss, Felix Schelling, Noah Klarmann |
| 单位 | Faculty of Management and Engineering, Rosenheim Technical University of Applied Sciences, Germany（通信：jonathan.hoss@th-rosenheim.de） |
| arXiv ID | arXiv:2506.13566v2 [cs.LG] 17 Jun 2025 |
| 年 | 2025 |
| **是否已发表** | **已发表**：2025 IEEE 21st International Conference on Automation Science and Engineering (CASE), Los Angeles, CA, Aug 17–21, 2025, pp. 1736–1743. DOI: 10.1109/CASE58245.2025.11163982 |
| 资助 | "The work has been performed in the Cynergy4MIE project (GA. 101140226)"（EU Horizon Europe Chips JU） |

### 3.2 问题设定

- **问题类型**：**框架/环境论文**（不是新算法论文）。扩展经典 JSSP（`J||C_max`）到含真实车间约束的版本，并提供 RL 训练与评测的通用环境。
- **扩展的约束 β（三类）**：
  1. **Transport management**：运输时间（机器间移动延迟）、运输容量（单元数与能否多载）、运输故障、装卸/行驶/卸载期间的占用时间。
  2. **Buffer management**：缓冲区容量上限（不足会造成拥塞）、缓冲区取件顺序约束（固定顺序还是任意顺序）。
  3. **Machine setup and breakdown**：与工件类型序列相关的 setup 时间；按概率建模的机器故障。
  4. **Stochasticity**：运输时间与机器故障的概率建模，加工时间建模为随机变量。
- **目标 γ（多目标支持）**：
  - 主目标：makespan、最大延迟、总加权完工时间、总加权拖期、加权拖期工件数；
  - 次目标：机器运行的能耗、缓冲区占用（降低资金占用）、提前期、柔性、按机器重要性/成本的利用率均衡。
- **框架支持的三字段记号范围**：`α | β | γ`，经典 JSSP 为 `J||C_max`。

### 3.3 方法

**算法**：不含新 RL 算法。框架名 **JobShopLab**。核心是**模块化、无状态的离散事件仿真框架**，由互相作用的子状态机构成。

**架构四层**：

1. **State Machine**：中心离散仿真组件，管理生产环境状态；采用 plug-in 架构。
2. **Middleware**：含 observation factory / action factory / reward factory，定义 RL 智能体的观测、动作、奖励空间；使用 Gymnasium 的 Observation/Reward/Action 实现。
3. **DSL**：两段式 —— DSL for Problem Instances（Jobs / Machines / Transport Systems / States）与 DSL for Framework Configuration（declarative 定义观测、动作、奖励）。
4. **Visualization**：基于 three.js 的 3D 可视化，支持可回放的逐场景动画。

**内置 plug-in**（`P_i : S × E × Θ_i → S × Ξ_i`，Eq.12）：breakdown、setup time、stochasticity、consumption（跟踪含能耗的资源消耗）。

**关键形式化（逐字公式）**：

- Eq.(1) 工序先后约束：`O_{i,1} ≺ O_{i,2} ≺ ... ≺ O_{i,m}  ∀J_i ∈ J`
- Eq.(2) 经典 makespan：`C_max = max_{i,j}(C_{i,j} = Z_{i,j} + p_{i,j})`（Z_{i,j} 为工序 O_{i,j} 的开始时间）
- Eq.(3) 事件集：`E(t) = E_agent(t) ∪ E_auto(t)`
- Eq.(4) 原子转移函数：`T : S × E → S`
- Eq.(5) 优先级函数：`ρ : E → R`
- Eq.(6) 降序处理：`(e_1, e_2, ..., e_n) = sortDescending(E(t), ρ)`
- Eq.(7) 故障优先：`∀b ∈ B, ∀e ∈ E \ B, ρ(b) > ρ(e)`
- Eq.(8) 顺序转移：`s_{t,0} = s_t, s_{t,i} = T(s_{t,i−1}, e_i), i = 1,...,n`
- Eq.(9) 状态更新：`s_{t+1} = s_{t,n}`
- Eq.(10)-(11) 复合转移：`T̂(s_t, E(t)) := T(T(···T(T(s_t, e_1), e_2),···), e_n)`，`s_{t+1} = T̂(s_t, E(t))`
- Eq.(12) 插件：`P_i : S × E × Θ_i → S × Ξ_i`
- Eq.(13) 工厂映射：`(S, A) --Factories--> {Observation, Action, Reward}`

**状态定义**：无状态设计（"every state fully encodes the production system at a specific simulation time, independent of history, forming a Markov Decision Process where transitions depend solely on the current state and actions"）。系统状态由 Job / Machine / Transport / Buffer 四类子状态机的状态组合而成。机器/运输/缓冲的状态转移图：Machine = {idle, setup, processing, down}；Transport = {idle, move, pick, drop, wait, down}；Buffer = {empty, not empty, full}。

**动作空间设计（关键事实）**：

- 智能体扮演 **dispatching controller**："an RL agent functions as a dispatching controller. It receives an observation of the current state, selects actions from a predefined action space, and receives a reward based on the resulting transitions. The agent's actions include scheduling jobs on machines and dispatching transport units."
- 无效转移被拒绝并返回 error response（"Invalid transitions are rejected with an error response, guiding the agent toward feasible decisions."）。
- 框架提供 3 类工厂（初始版本）：
  - **binary action factory**：二值动作空间，决定"是否执行动作"；
  - **multidiscrete action factory**：多维离散动作空间，从若干候选中选择。
- **PoC 实验实际使用的动作空间**（逐字）："The agent's action space consists of a binary decision: whether to execute the next scheduling action or defer it."

**观测定义（PoC 实际使用）**（逐字）："The observation space is defined by seven features representing the current system state and the next available scheduling action."

**奖励**（框架提供 `minimize makespan reward factory`，PoC 使用最小奖励函数）："To ensure the agent learns scheduling strategies purely based on fundamental system dynamics, it is trained with a minimal reward function, a simplified action space, and a compact observation space, avoiding excessive domain-specific tuning." —— **原文未给出奖励的显式公式**。

**多目标机制**（原文）："This modular design inherently supports multi-objective optimization by enabling multiple, configurable reward signals within the framework."

**RL 实现**：Stable Baselines3 的 PPO。

### 3.4 实验

- **实例**：经典 JSSP 基准 —— ft06、ft10（Fischer & Thompson 1963）、la01–la24（Lawrence 1984）、ta01/ta02/ta41（Taillard 1993）。
  - 扩展实例命名加 `-t` 后缀：ft06-t, ft10-t, la01-t … la24-t。
- **基线**：PDR（Priority Dispatching Rules）—— SPT（Shortest Processing Time）与 MWKR（Most Work Remaining）。
  - 原文说明为何不用精确方法作基线："Exact methods such as Constraint Programming or Mixed-Integer Linear Programming struggle to efficiently solve large, complex instances within reasonable time frames, making them impractical for comparison against an agent trained in a real-world framework."
- **指标**：
  - 实验一（经典 JSSP）：`LB/makespan` 比率（LB = 已知下界），即越接近 100% 越好；
  - 实验二（含 buffer + transport 约束）：相对 SPT / MWKR 的 makespan 相对改进百分比（因真实约束下无已知下界）。
- **两个实验**：
  1. 经典 JSSP，验证框架实现与训练能力；
  2. 加入 buffer 与 transport 约束的扩展 JSSP，验证框架能表征真实复杂度。

### 3.5 核心数字（照抄原文）

**Fig.4（经典 JSSP，LB/makespan 比率，纵轴 60%–100%）**：涉及实例 la10、la16、ta01、ta02、ta41；三条曲线为 LB（100% 参照）、LB/Ours、LB/SPT、LB/MWKR。

原文结论（逐字）："Across all tested problems, the RL agent consistently achieves a makespan closer to the lower bound than the heuristic-based methods."

**Fig.5（含真实约束，相对改进，纵轴 5%–35%）**：横轴覆盖 ft06-t、ft10-t、la01-t 至 la24-t 共 26 个实例；两条序列为 Improvement over SPT、Improvement over MWKR。

原文结论（逐字）："The results demonstrate that, despite the added complexity, the RL agent consistently outperforms both PDRs, learning to adapt its scheduling decisions to account for transport and buffer constraints more effectively than rule-based methods."

**原文未在任何地方给出具体的 makespan 数值或改进百分比数值** —— 全部结果只以 Fig.4、Fig.5 的图形呈现，正文无表格、无点估计值。

### 3.6 自认局限（原文逐字 + 中译）

> **说明**：该文**无独立 "Limitations" 章节**。以下为原文中作者明确承认的局限与未覆盖范围的表述。

> **L1.** "Although our framework natively supports multi-objective optimization, this study primarily benchmarks performance on the makespan objective. This focus is motivated by the lack of widely accepted multi-objective benchmarks and standardized evaluation metrics for real-world JSSP. While classical single-objective benchmarks are well established, resources for multi-objective and real-world scenarios remain scarce. Developing such benchmarks and metrics is an important direction for future work, and our framework is designed to accommodate these evaluations as the field progresses."

中译：尽管我们的框架原生支持多目标优化，本研究主要以 makespan 目标进行基准评测。这一侧重源于缺乏被广泛接受的多目标基准以及面向真实 JSSP 的标准化评测指标。经典单目标基准已相当成熟，而多目标与真实场景的资源仍然稀缺。开发此类基准与指标是未来工作的重要方向，我们的框架已设计为可随领域进展容纳这些评测。

> **L2.** "Unlike the classical JSSP, where a known lower bound serves as a reference for optimality, the introduction of real-world constraints makes direct optimality comparisons impractical. Instead, the evaluation focuses on the relative performance improvements achieved by the RL agent over traditional heuristics."

中译：与以已知下界作为最优性参照的经典 JSSP 不同，引入真实约束后直接做最优性比较不切实际。因此评测聚焦于 RL 智能体相对传统启发式的性能改进。

> **L3.** "Exact methods such as Constraint Programming or Mixed-Integer Linear Programming struggle to efficiently solve large, complex instances within reasonable time frames, making them impractical for comparison against an agent trained in a real-world framework. By prioritizing makespan minimization in our experiments, we provide a clear and direct benchmark between the RL-based approach and established heuristics."

中译：约束规划或混合整数线性规划等精确方法难以在合理时间内高效求解大型复杂实例，使其不适于与在真实框架中训练的智能体作比较。通过在我们的实验中优先考虑 makespan 最小化，我们提供了 RL 方法与传统启发式之间清晰直接的基准对比。

> **L4.** "To ensure the agent learns scheduling strategies purely based on fundamental system dynamics, it is trained with a minimal reward function, a simplified action space, and a compact observation space, avoiding excessive domain-specific tuning."

中译：为确保智能体纯粹基于基本系统动力学学习调度策略，其训练使用最小化的奖励函数、简化的动作空间和紧凑的观测空间，避免了过度的领域特定调参。

> **L5.** "Future work will refine the framework by investigating diverse action spaces, observation structures, and reward functions to optimize RL-based scheduling. By adjusting environmental complexity, we aim to achieve the optimal balance for integrating RL agents into real-world production. Another direction involves integrating the framework with real-time production systems, enabling adaptive online scheduling under dynamic constraints. In addition, establishing standardized benchmarks and evaluation metrics for multi-objective scheduling remains a central goal. Leveraging the framework's extensibility, we seek to close the gap between academic research and industrial requirements. These developments will enhance robustness, scalability, and increase the framework's technology readiness level, supporting real-world deployment."

中译：未来工作将通过研究多样化动作空间、观测结构与奖励函数来改进框架，以优化基于 RL 的调度。通过调整环境复杂度，我们力求找到将 RL 智能体融入真实生产的最优平衡。另一方向是将框架与实时生产系统集成，实现动态约束下的自适应在线调度。此外，为多目标调度建立标准化基准与评测指标仍是核心目标。借助框架的可扩展性，我们试图弥合学术研究与工业需求之间的差距。这些发展将增强鲁棒性、可扩展性并提升框架的技术成熟度（TRL），支撑真实部署。

> **L6.** （背景性判断，构成动因）"Existing frameworks vary in complexity and often focus on simplified objectives rather than adopting a multi-objective perspective."

中译：现有框架在复杂度上各不相同，且往往聚焦于简化目标，而非采取多目标视角。

### 3.7 可复用

- **开源代码**：**https://github.com/proto-lab-ro/jobshoplab** —— 论文原文："We release JobShopLab as an open-source tool for both research and industrial applications, accessible at: https://github.com/proto-lab-ro/jobshoplab"。OpenAI Gym / Gymnasium 兼容，Python 3.12+，pip 可安装。
- **数据集**：使用公开经典基准（ft06/ft10、la01–la24、ta01/ta02/ta41），无新数据集。扩展实例（`*-t`）由框架配置生成，配置方法通过 DSL 公开。
- **评测协议**：
  - 经典 JSSP 用 `LB/makespan` 比率（有下界时）；
  - 含真实约束时用相对 PDR 的改进百分比（无下界时）；
  - 基准实例 ft / la / ta 三族，可与其他 JSSP 工作横向比较。
- **可复用公式/机制**：
  - 事件驱动无状态状态机形式化（Eq.3–11）：事件优先级函数 ρ、故障事件强制优先（Eq.7）、复合转移 T̂（Eq.10）—— 可直接用于任何含并发/异步事件的调度仿真；
  - 插件式约束注入接口 `P_i : S × E × Θ_i → S × Ξ_i`（Eq.12）—— 新增车间约束（能耗、缓冲区、故障）的标准化扩展点；
  - 三工厂（observation / action / reward factory）+ DSL 的配置化 RL 接口（Eq.13）—— 支持同一仿真下切换动作空间与奖励结构的可比实验设计。
- **工程组件**：three.js 3D 可视化与回放工具；Gymnasium 标准接口（可与 Stable Baselines3 等直接对接）。

---

## 4. arXiv2305.13824 — Constrained Reinforcement Learning for Dynamic Material Handling

### 4.1 题录

| 项 | 内容 |
|---|---|
| 标题 | Constrained Reinforcement Learning for Dynamic Material Handling |
| 作者 | Chengpeng Hu, Ziming Wang, Jialin Liu, Junyi Wen, Bifei Mao, Xin Yao |
| 单位 | RITAS & 广东省类脑智能计算重点实验室，南方科技大学（C. Hu, Z. Wang, J. Liu, X. Yao）；华为技术有限公司可信理论研究中心（J. Wen, B. Mao）；香港岭南大学（J. Liu, X. Yao） |
| arXiv ID | arXiv:2305.13824v1 [cs.LG] 23 May 2023 |
| 年 | 2023 |
| **是否已发表** | **已发表**：International Joint Conference on Neural Networks (IJCNN) 2023, pp. 1–9. IEEE Xplore 文档号 10191999 |
| 佐证 | 同组论文 arXiv2506.16795 的参考文献 [5] 原文："C. Hu, Z. Wang, J. Liu, J. Wen, B. Mao, and X. Yao, "Constrained reinforcement learning for dynamic material handling," in International Joint Conference on Neural Networks, 2023, pp. 1–9." |

### 4.2 问题设定

- **问题类型**：Dynamic Material Handling (DMH) —— 多 AGV 实时任务分配；建模为 Constrained MDP (CMDP)，元组 `(S, A, R, C, P, γ)`。
- **资源**：
  - 车间图 `G(L, T)`，L 为站点集（取货点 pickup points、交货点 delivery points、停车位 parking positions），T 为路径集；
  - 停车位集 K ⊂ L；A* 算法做路径规划，AGV 沿预设路径同速行驶；
  - 仓库（warehouse）只能作为交货点；任务起点只能是工作站，终点可以是工作站或仓库；
  - 一队 AGV `V`，`v = ⟨vl, rp, pl, ψ⟩`（速度、修理时间、停车位置、状态）；状态集 Ψ = {I, W, B} = Idle / Working / Broken。
- **动态事件**：新任务动态到达（未分配任务暂存于待分配列表 U，任务总数 m 事先未知）、AGV 故障（故障车原地停止并释放当前任务到任务池，修理 v.rp 后恢复可用）。
- **目标**：最小化 makespan `F_m(π)`（Eq.1）；**tardiness 作为约束而非目标**。
- **约束**：
  - 累计约束（Eq.5）：`J_C^{πθ} = E_{πθ}(F_t(πθ)) ≤ ε`，ε = 50；
  - 瞬时约束（Eq.6）：`v.ψ = I`（只有 Idle 车可被指派）。
- **简化假设**（原文自述）："Collisions and traffic congestion are not considered in this paper for simplification."；"The time cost of waiting to handle material is ignored at pickup points and delivery points for AGVs as they are often constant."
- **任务定义**：`u = ⟨s, e, τ, o⟩ ∈ U`，s 取货点、e 交货点、τ 到达时间、o 到期时间（expiry time）。

### 4.3 方法

**算法**：**RCPOM** = RCPO（Reward Constrained Policy Optimization，拉格朗日松弛型，Actor-Critic 框架） + **invalid action masking**（无效动作掩码），backbone 为 **SAC**（Soft Actor-Critic，最大熵 RL）。

**网络结构**（逐字）："The network structure is formed by two hidden fully connected layers 128 × 128."

**状态定义（逐字）**：

> "At each decision time t, we consider the current state of the whole system consisting of tasks and AGVs, denoted as S_t = ρ(U_t, V_t), where U_t is the set of unassigned tasks and V_t represents information of all AGVs at time t. We encode the information of the system at decision time t with a feature extract function ρ(U_t, V_t) as structure representation. Specifically, a state is mapped into a vector by ρ consisting of the number of unassigned tasks, tasks' remaining time, waiting time and distance between pickup and delivery point as well as statuses of vehicles and the minimal time from the current position to pickup point then delivery point."

即状态向量 ρ 的组成：未分配任务数、任务剩余时间、等待时间、取送点之间距离、车辆状态、从当前位置到取货点再到交货点的最短时间。

**动作空间设计（混合动作空间 D × V_t，含掩码压缩）**：

> "In our CMDP, making an action is to choose a suitable dispatching rule and a corresponding vehicle. A hybrid action space A_t = D × V_t is considered for the CMDP, where D and V_t are defined as dispatching rule space and AGV space, respectively. Four dispatching rules, including first come first served (FCFS), shortest travel distance (STD), earliest due date first (EDD) and nearest vehicle first (NVF), form the dispatching rule space D."

四条调度规则（原文公式）：

- FCFS: `arg min_{u∈U} u.τ`
- EDD: `arg min_{u∈U} (u.o − u.τ)`
- NVF: `arg min_{u∈U} d(v.c, u.s)`
- STD: `arg min_{u∈U} d(v.c, u.s) + d(u.s, u.e)`

（`v.s` 为 AGV 当前位置，`d(p, p')` 为 p 与 p' 之间的距离。原文中 `v.s` 与 `v.c` 混用，未统一。）

动作 `a_t = ⟨d_t, v_t⟩ ∈ A_t`；掩码后动作空间压缩为 `A_t = D × AV_t`（AV_t 为可用 AGV 集）。掩码实现（Alg.1）：把无效动作对应的 logits 替换为大负数，再 softmax 重采样。

> "It is notable that, to the best of our knowledge, no existing work has applied the masking technique to handle the instantaneous constraint in DMH."

**奖励（原文公式）**：Eq.(7) 稀疏奖励，仅终止时给：

```
R(s_t, a_t, s_{t+1}) = { −F_m(π),  if terminates
                       { 0,        otherwise
```

**拉格朗日松弛（原文公式）**：

- Eq.(8) 对偶形式：`min_λ max_θ [ R^π − λ(J_C^π − ε) ]`，其中 λ > 0
- Eq.(9) 奖励重塑：`R̂(s, a, s', λ) = R(s, a, s') − λ c(s, a, s')`
- Eq.(10) 乘子更新（比 actor/critic 更慢的时间尺度）：`λ = max{ λ + η(J_C^π − ε), 0 }`，η 为乘子学习率
- Eq.(11) SAC 最大熵目标：`J(π_θ) = E_{πθ}[ Σ_{t=0}^∞ γ^t ( R(s_t,a_t,s_{t+1}) + α H(π(·|s_t)) ) ]`，α 为温度参数

**不变奖励塑形（Invariant reward shaping）**，Eq.(12)：`R̃(s, a, s', λ) = β R(s, a, s') + b`（β > 0，b 为正常数）。

Remark 1（策略不变性）：`∀s ∈ S, V*(s) = arg max_π V^π(s) = arg max_π β V^π(s) + b/(1−γ)`。

**RCPOM 伪代码（Alg.2）要点**：初始化经验回放池 B_R、策略 π_θ、两个 critic Q̂_ψ1 与 Q̂_ψ2；每步先用 masking 得 a'_t（Alg.1），执行后存 ⟨s_t, a'_t, s_{t+1}, r_t, c_t⟩；critic 目标 `y ← r − λc + γ( min_{j=1,2} Q̂_ψj(s', ã') − α log π_θ(ã'|s') )`，`ã' ∼ π_θ(·|s')`；actor 用重参数化采样 `ã_θ` 更新 `min_{j=1,2} Q̂_ψj(s, ã_θ) − α log π_θ(ã_θ|s)`；每轮末尾按 Eq.(10) 更新 λ。

**超参数（逐字）**："The network structure is formed by two hidden fully connected layers 128 × 128. α is 0.1. γ is 0.97. The initial multiplier λ and its learning rate are set as 0.001 and 0.0001, respectively. The constraint threshold ε is set as 50. Reward shaping parameters β and b are set as 1 and 2000 in terms of dispatching rules, respectively. All learning agents are trained for 5e5 steps on an Intel Xeon Gold 6240 CPU and four TITAN RTX GPUs. The best policies during training are selected. All dispatching policies and learning agents are tested 30 times on the instances independently. Parameter values are either set following previous studies [26], [36] or arbitrarily chosen."

**实现框架**：Tianshou（https://github.com/thu-ml/tianshou）。

### 4.4 实验

- **模拟器与实例**：自研 DMH-GYM（OpenAI gym 兼容）。共 **16 个问题实例**：instance01–instance08 为**训练实例**，instance09–instance16 为**训练中未见过的实例**（"These unseen instances are generated by mutating the training instances."）。
  - 车间布局：工作站 st1–st8 + 仓库 + 车棚（carport，AGV 初始停车位）。所有 AGV 同速，A* 做路径规划。例：st8 (0,45) 与 st1 (20,70) 之间距离为 45（需经过拐角点 (0,70)）。
- **基线（五组 + 消融）**：
  1. CRL：IPO（Interior-point Policy Optimization）[34]
  2. RL：PPO、SAC
  3. 固定拉格朗日乘子的 SAC / PPO："L-SAC"、"L-PPO"
  4. 调度规则：FCFS、STD、EDD、NVF
  5. 消融：RCPOM-NS（去掉不变奖励塑形）
  6. 另加 Random agent 与 Hu et al. [6]（DQN + 重采样）
  - 所有学习型智能体均施加无效动作掩码；除 RCPOM-NS 外均施加奖励塑形。
- **指标**：平均 makespan `F_m`、平均 tardiness `F_t`、归一化指标 `M/C(P)`（平均归一化 makespan / tardiness / 约束满足率）、单次决策耗时。
- **评测协议**：每实例 30 次独立试验；结果表用 "+/≈/−" 标注相对 RCPOM 的统计显著性；表底给出 RCPOM 在 makespan 与 tardiness 上优于其他策略的个数。

### 4.5 核心数字（照抄原文）

**决策耗时（Table I，2000 次试验平均，单位 ms）**

| RCPOM | RCPOM-NS | IPO | L-SAC | L-PPO | SAC | PPO | FCFS | EDD | NVF | STD | Random | Hu et al. [6] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.143 | 2.112 | 2.038 | 2.74 | 2.514 | 2.79 | 2.626 | 0.0169 | 0.0199 | 0.0239 | 0.0354 | 0.0259 | Timeout |

**训练实例 M/C(P)（Table II 末列）**

| 算法 | M/C(P) |
|---|---|
| RCPOM | 0.97/0.89 (75%) |
| RCPOM-NS | 0.68/0.70 (56%) |
| IPO | 0.80/0.77 (61%) |
| L-SAC | 0.74/0.76 (58%) |
| L-PPO | 0.80/0.73 (53%) |
| SAC | 0.82/0.83 (62%) |
| PPO | 0.79/0.75 (61%) |
| FCFS | 0.31/0.48 (37%) |
| EDD | 0.66/0.91 (86%) |
| NVF | 0.71/0.67 (51%) |
| STD | 0.83/0.75 (53%) |
| Random | 0.02/0.00 (12%) |

**测试（未见）实例 M/C(P)（Table III 末列）**

| 算法 | M/C(P) |
|---|---|
| RCPOM | 0.95/0.90 (74%) |
| RCPOM-NS | 0.68/0.72 (55%) |
| IPO | 0.80/0.80 (61%) |
| L-SAC | 0.73/0.78 (58%) |
| L-PPO | 0.81/0.76 (57%) |
| SAC | 0.76/0.81 (62%) |
| PPO | 0.81/0.77 (62%) |
| FCFS | 0.32/0.48 (35%) |
| EDD | 0.59/0.79 (73%) |
| NVF | 0.73/0.71 (58%) |
| STD | 0.80/0.76 (55%) |
| Random | 0.03/0.02 (11%) |

**引用单点值（原文逐字）**：

- "For Instance01-Instance03, EDD gets the lowest tardiness 33.9, 29.6 and 26.5 compared with other policies, respectively."
- "For example, STD has the smallest makespan 1883.5 in Instance08 among all the policies."
- "In Instance01, the SAC agent gets the best makespan of 1798.4."
- "Although STD achieves 1883.5 on Instance08, RCPOM still gets a very closed makespan value of 1898.0."
- "Even though in Instance04, SAC gets the lowest tardiness value with 28.9, we consider RCPOM as the best agent since it gets the lowest makespan value of 1956.9, whose tardiness is under the constraint threshold, i.e., 40.5 < 50."
- "FCFS and EDD are time-related type dispatching rules... On Instance01, Instance04 and Instance08, SAC agent gets the lowest tardiness."
- "It is shown on Instance08 that NVF and STD are better in both makespan and tardiness."
- "RCPOM, the proposed method, achieves the best average makespan and is statistically better than almost other policies on Instance10-14."

**RCPOM 训练集单实例点值（Table II）**：Instance01 1840.0/56.7；Instance02 1908.4/48.2；Instance03 1914.8/45.0；Instance04 1956.9/40.5；Instance05 1852.5/21.3；Instance06 1911.1/41.0；Instance07 1927.0/8.4；Instance08 1898.0/21.0。
**RCPOM 测试集单实例点值（Table III）**：Instance09 1840.0/57.1；Instance10 1917.6/49.4；Instance11 1900.3/45.8；Instance12 1954.3/38.8；Instance13 1862.7/18.4；Instance14 1914.1/41.8；Instance15 1960.8/19.6；Instance16 1896.4/21.3。

### 4.6 自认局限（原文逐字 + 中译）

> **说明**：该文**无独立 "Limitations" 章节**。以下为原文中作者明确承认的局限、简化与不足的表述。

> **L1.** "Collisions and traffic congestion are not considered in this paper for simplification."

中译：为简化起见，本文不考虑碰撞与交通拥堵。

> **L2.** "The time cost of waiting to handle material is ignored at pickup points and delivery points for AGVs as they are often constant."

中译：AGV 在取货点与交货点等待处理物料的时间成本被忽略，因为它们通常是常数。

> **L3.** "As future work, we are interested in extending the problem by introducing more dynamic events and constraints that widely exist in real-world scenarios."

中译：作为未来工作，我们有意通过引入真实场景中广泛存在的更多动态事件与约束来扩展该问题。

> **L4.** "Parameter values are either set following previous studies [26], [36] or arbitrarily chosen."

中译：参数取值或沿用前人研究 [26], [36]，或为任意选定。

> **L5.** "RCPOM has a slight gap with EDD on tardiness but outperforms it much on makespan."

中译：RCPOM 在 tardiness 上与 EDD 存在小幅差距，但在 makespan 上大幅优于 EDD。

> **L6.** "SAC agent gets a high tardiness value on Instances indexed with 2, 3, 6, 10, 11 and 14. It is attributed to the single-attribute reward function since it only relies on the makespan and does not consider tardiness. A straightforward method to handle the cumulative constraint is augmenting the objective function. L-SAC and L-PPO reshape the reward function with a fixed Lagrangian multiplier, seeing Eq. (9). However, it is hard to decide the multiplier."

中译：SAC 智能体在编号为 2、3、6、10、11 和 14 的实例上 tardiness 偏高。这归因于其单属性奖励函数——它只依赖 makespan，不考虑 tardiness。处理累计约束的一个直接方法是增广目标函数。L-SAC 与 L-PPO 用固定拉格朗日乘子重塑奖励函数（见 Eq.9）。然而乘子难以确定。

> **L7.** （对基线 IPO 的评价，间接说明约束初始化前提的局限）"Although IPO is also a CRL algorithm, its assumption that the policy should satisfy constraints upon initialisation [37] limits its performance when solving the DMH problem."

中译：尽管 IPO 也是一种 CRL 算法，但其"策略须在初始化时即满足约束"的假设限制了它求解 DMH 问题时的表现。

> **L8.** （对基础方法 Hu et al. [6] 的批评，本工作正是为克服之）"However, when applying the method of [6] to our problem, the agent will keep being trapped in this step if the selected action is infeasible. Even though keeping resampling the action, the same feasible action will always be selected by this deterministic policy at a certain state."

中译：然而，将 [6] 的方法应用于我们的问题时，若所选动作不可行，智能体会一直困在该步骤中。即便不断重采样动作，在该确定策略下、在某个特定状态中总会选中同一个可行动作。

### 4.7 可复用

- **开源代码 + 模拟器 + 数据集**：**https://github.com/HcPlu/DMH-GYM** —— 论文原文脚注："Codes available at https://github.com/HcPlu/DMH-GYM"；"Extending OpenAI's gym, we develop an open-source simulator, named DMH-GYM, and provide a dataset comprised of diverse instances for researching the dynamic material handling problem."
  - 含 16 个问题实例（instance01–16，其中 09–16 由 01–08 变异生成），可直接用于跨方法对比。
- **依赖框架**：Tianshou，https://github.com/thu-ml/tianshou
- **评测协议**：三指标 M/C(P)（平均归一化 makespan / tardiness / 约束满足率）；每实例 30 次独立试验；"+/≈/−" 统计显著性标注；另报单次决策耗时（安全/实时性维度）。
- **可复用公式**：
  - makespan Eq.(1)：`F_m(π) = max_{v∈V^π} FT(u^v_{|HL(v)|}, v)`
  - tardiness Eq.(2)：`F_t(π) = (1/m) Σ_v Σ_i max(FT(u^v_i, v) − u^v_i.o − u^v_i.τ, 0)`
  - 拉格朗日对偶 Eq.(8)、奖励重塑 Eq.(9)、乘子在线更新 Eq.(10) —— 三者构成一个可直接复用的"累计约束"处理模块；
  - 不变奖励塑形 Eq.(12) + Remark 1 的策略不变性证明 —— 可安全用于任何稀疏负奖励场景；
  - 四条调度规则的显式定义（FCFS / EDD / NVF / STD）—— 可作为混合动作空间的规则子空间模板。
- **可复用机制**：
  - **invalid action masking** 处理瞬时约束（Alg.1）：单次采样即得合法动作，避免重采样死循环，且压缩动作空间、提升探索效率；
  - **混合动作空间 D × V_t**：同时决定"用哪条规则"与"派哪台车"，打破多车情形的平局（"which breaks the tie of the multiple vehicles case"）；
  - **CMDP 中"目标 vs 约束"的分离**：makespan 作目标、tardiness 作累计约束、车辆可用性作瞬时约束—— 一种把安全约束从奖励设计中剥离的范式。

---

## 5. arXiv2604.04753 — Toward Self-Organizing Production Logistics: A Multi-Agent Approach

### 5.1 题录

| 项 | 内容 |
|---|---|
| 标题 | Toward Self-Organizing Production Logistics: A Multi-Agent Approach |
| 作者 | Jan-Felix Klein (ORCID 0000-0002-3704-9567), Yongkuk Jeong (0000-0003-1878-773X), Erik Flores-García (0000-0003-0798-0753), Magnus Wiktorsson (0000-0001-7935-8811) |
| 单位 | KTH Royal Institute of Technology, 10044 Stockholm, Sweden（通信：jfklein@kth.se） |
| arXiv ID | arXiv:2604.04753v2 [eess.SY] 21 May 2026 |
| 年 | 2026 |
| **是否已发表** | **截至 2026-09-30 未正式发表**。arXiv 预印本；arXiv 列表页 Comments 字段注明"Submitted to IFIP International Conference..."（投往 IFIP APMS 2026）。 |
| 备注 | 早期版本标题为 "Toward Self-Organizing Production Logistics in Circular Factories: A Multi-Agent Approach"（见 Semantic Scholar 记录）。PDF 为 Springer LNCS 模板（含 "Disclosure of Interests" 段）。 |
| 资助 | Centre of Excellence in Production Research (XPRES)；German Research Foundation (DFG) - SFB 1574 – 471687386 |

### 5.2 问题设定

- **问题类型**：**概念/架构论文**（conceptual design study），**非算法论文**。研究问题是：
  > "How can multi-agent AI systems be designed to realize Self-Organizing Production Logistics (SOPL) and thereby improve responsiveness and resilience under disturbances?"
- **应用场景**：production logistics (PL)，以 **circular factory**（循环工厂）作为"代表性边界案例"（"Circular factories are therefore used in this paper as a representative boundary case in which these dynamics become especially visible."）。
- **资源/对象**：
  - 物理层资产组合：输送系统、自动化仓储、AMR、AGV、cobot、人形机器人、模块化物流单元与工艺模块；
  - 人类被显式建模为高能力、知识密集型资源；
  - 决策层：具身 agent（AMR、cobot、人形、机械臂）与非具身 agent（Task Planner、Knowledge、Digital Twin 等）。
- **目标（system-level performance objectives，5 项）**：Improve responsiveness；Improve resilience；Enable scalable adaptability；Enable continuous improvement；Safeguard core logistics performance。
- **设计需求（design requirements，5 项）**：Decentralized coordination；Modular assets；Semantic knowledge；Learning & adaptation；Human governance。
- **约束/边界**：循环生产带来构件质量与可用性高变异、回收时间不确定、跨代兼容性约束、批量趋近于 1；结构不确定性使长期全局最优规划日益失效。

### 5.3 方法

**方法论框架**：Design Science Research Methodology (DSRM, Peffers et al. [22])，六阶段：Identify Problem & Motivate → Define Objectives of a Solution → Design & Development → Demonstration → Evaluation → Communication。本文定位为"早期概念设计研究"，主要覆盖前三个阶段（见 5.6 L1）。

**需求推导透镜**：Self-Organizing Logistics (SOL) typology（Gerrits et al. [8]），四维度：system architecture、cooperativeness、autonomy、system features。作者补充三项 SOL 未充分覆盖的需求：modular and reconfigurable physical assets、shared semantic knowledge structures、human governance mechanisms。

**提出的架构（三层）**：

1. **Physical and Embodied Layer**（物理与具身层）
   - 异构资源组合：常规基础设施（输送、自动化仓储）+ 自主移动机器人 / AGV / cobot / 人形机器人 + 模块化物流单元与工艺模块 + 人类。
   - 核心假设："future production logistics will rely on a diverse portfolio of assets rather than a single dominant automation paradigm"；能力不集中于单一资源类型，而是由互补资产组合涌现。
   - 模块化的作用：资产可组合、重定位、改用途，在资源受限下实现 capability composition。
2. **Decision-Making Layer**（决策层）—— 分布式多 agent 系统
   - agent 定义（原文）："a computational entity situated within a production logistics environment and capable of autonomous action in pursuit of defined objectives [29]"。
   - **具身 agent（embodied）**：直接关联物理资源，决策与物理执行紧耦合；但耦合**不通过直接无限制访问物理系统**，而是经由一个 virtual representation 作为决策与执行之间的中间层（暴露状态与动作信息、保证安全受控交互）。
   - **非具身 agent（non-embodied）**：不直接执行物理动作，承担协调、推理、监控、仿真、编排；含任务分解、系统级协调、预测分析、共享服务交互。
   - 协调通过 shared message spaces、event streams、negotiation protocols 涌现。
   - **数字孪生**：作为基于模型的动态决策支持工具，使 agent 能在物理执行前预判后果、比较方案；架构假定一个专门的 **DT agent** 协调仿真任务、参数化场景、以适合去中心化决策的形式提供结果。
   - 架构图中的协作类型：Cooperation / Competition / Coopetition；结构：Centralized / Distributed / Hierarchical；策略：Rule-based / Role-based / Model-based。
   - agent 内部能力构成：Perception / Action / Interaction / Memory / Reasoning & Planning。
3. **Knowledge Layer**（知识层）
   - 基于本体（ontologies）刻画 products、processes、resources、capabilities、tasks、operations；
   - 本体实例化于共享知识图谱（knowledge graph），辅以结构化数据仓库；含 Domain Ontology / Core Ontology / Linked Open Data；
   - 编码 constraints、rules、safety requirements、escalation mechanisms 以治理和限制自主行为；
   - 存储执行日志与结果数据，支持协调策略、能力模型与系统知识的持续精化。

**无算法、无网络结构、无状态/动作/奖励定义** —— 该文不涉及 RL 形式化，不含 MDP 建模、动作空间设计或奖励函数。

**三阶段演示路线图（Demonstration Roadmap）**：

- **Phase I: Foundations** —— 受控实验室环境中实现基础 SOPL 架构；验证异构具身/非具身 agent（含 LLM 推理能力）的技术可行性，以及事件驱动机制与共享消息池下的分布式任务执行协调行为。
- **Phase II: Distributed Autonomy** —— 任务复杂度、运行真实性与 agent 数量提升；从结构化单任务流程走向动态多任务；agent 获得更丰富的任务/资源/能力语义描述，支持上下文感知推理、基于能力的任务分配、通过协商与冲突解决的去中心化协调；数字孪生作为协作决策支持工具集成。
- **Phase III: Intelligent Collective** —— 持续学习、大规模协调、更高真实变异下运行；agent 通过历史交互数据、数字孪生轨迹与运行反馈精化决策策略与协调策略，实现更主动的行为、职责的自主重组、与系统级生产目标的持续对齐。

**Phase I 演示器（KTH IPU Lab）**：

- 场景：order-driven kitting and supply —— 装配站请求一套组件套件（kit）。流程：从仓储取所需组件 → 订单拣选 → 运送至 kitting 站 → 在料箱（tote）中组装套件 → 交付至请求的装配站。
- 具身 agent：1 台 **AMR**（运输）、1 台 **picking cobot**（通过 scene understanding、target-kit planning、pick-and-place sequence planning 完成 kitting 子任务）。
- 人类 agent（3 名）：仓储区订单拣选 1 名、kitting 站料箱处理 1 名、装配站接收与确认 1 名。
- 非具身软件 agent：含 **supervisor agent**，被作为**实验设计参数**（"a supervisor agent is treated as an experimental design parameter"）——依配置可维持系统级任务进度视图、监控上报事件与扰动、协调恢复动作。
- 通信：事件驱动机制 + shared message pools，交换任务请求、状态变更、完成通知、扰动告警。
- 研究问题：
  - **RQ1**: "How do individual LLM-based agents react to disturbance alerts, and which initial decision strategies do they apply to restore task execution?"
  - **RQ2**: "How can human operators be integrated effectively into disturbance-handling processes?"
  - **RQ3**: "What role does a supervisor agent play in coordinating disturbance resolution across heterogeneous agents, and how does its involvement influence the quality of decision-making?"

**三个预定义扰动场景**（用于评测演示器的扰动处理能力）：

1. **资源类扰动**：AMR 在运输途中停止（"a resource-related disturbance is considered in which the AMR stops during a transport operation, interrupting the physical flow of materials between process stages"）。
2. **流程类扰动**：意外的新优先任务到达（"the arrival of an unexpected new priority task, requiring the involved agents to reassess the current execution sequence and respond to changing operational priorities"）。
3. **产品类扰动**：仓储区拣错组件（"a wrongly picked component at the storage area, creating a mismatch between the requested and the actually supplied material"）。

### 5.4 实验

**无定量实验。** 该文不含仿真实验、不含基线对比、不含数值指标。仅有：

- **Table 1 追溯矩阵（traceability matrix）**：5 个性能目标 × 5 个设计需求的定性贡献强度矩阵，符号含义为 `++` = strong contribution、`+` = relevant contribution、`o` = supporting contribution。原文明确："The strength indicators are qualitative and represent the relative conceptual importance of a design requirement for achieving a given performance objective."
- 该矩阵的两个结论（原文）：
  > "First, responsiveness and scalable adaptability are strongly enabled by decentralized coordination and modular assets, as both support local decision-making and structural reconfiguration close to execution. Second, safeguarding core logistics performance depends particularly on shared semantic knowledge and human governance, which ensure global coherence and prevent local autonomy from degrading system-wide objectives."

**Table 1 矩阵内容（转录）**：

| 性能目标 \ 设计需求 | Decentralized coordination | Modular assets | Semantic knowledge | Learning & adaptation | Human governance |
|---|---|---|---|---|---|
| Improve responsiveness | ++ | + | + | + | o |
| Improve resilience | + | ++ | ++ | + | + |
| Enable scalable adaptability | ++ | ++ | ++ | + | o |
| Enable continuous improvement | + | o | + | ++ | + |
| Safeguard core logistics performance | o | + | ++ | + | ++ |

**基础设施**：Phase 1 演示环境为 KTH 的 IPU Lab（Fig.4 示意）；车间布局基于为循环工厂环境开发的模块化厂内物流资源 [14]。

**评测协议**：**尚不存在**。原文明确将其列为未来工作（见 5.6 L2）。

### 5.5 核心数字

**该文无定量结果数字。**全文不含性能数值、成功率、对比实验数据或统计检验。唯一的结构化定量信息是：

- 架构分 **3 层**（physical/embodied、decision-making、knowledge）；
- 路线图分 **3 阶段**（Phase I Foundations / II Distributed Autonomy / III Intelligent Collective）；
- Phase I 演示器含 **3 个预定义扰动场景**、**3 个研究问题（RQ1–RQ3）**；
- Phase I 演示器的 agent 构成：**2 类具身 agent**（1 台 AMR + 1 台 picking cobot）+ **3 名人类 agent**（storage / kitting / assembly 各 1）+ 可选的非具身软件 agent（含 supervisor）；
- Table 1 为 **5×5 定性矩阵**；
- 引用的支撑数据（非本文实验）：World Robotics Report 2025（服务机器人）、LogisticsIQ AGV-AMR Market 2025。

### 5.6 自认局限（原文逐字 + 中译）

> **L1.** "The work is positioned as an early-stage conceptual design study and primarily addresses the problem identification, objective definition, and design & development phases of the DSRM cycle with initial prototypical demonstration activities considered in the roadmap."

中译：本工作被定位为一项早期概念设计研究，主要处理 DSRM 循环中的问题识别、目标定义与设计开发阶段，路线图中考虑了初步的原型演示活动。

> **L2.** "Future work will focus first on the systematic design and evaluation of the Phase I demonstrator. In particular, a design of experiments is needed to define which disturbance scenarios, system configurations, and performance indicators should be assessed in order to answer the demonstrator research questions. This includes evaluating how agents respond to disturbances, how human operators are integrated into recovery processes, and how the presence of a supervisor agent affects coordination quality."

中译：未来工作将首先聚焦于 Phase I 演示器的系统性设计与评估。特别是需要一套实验设计，以确定应评估哪些扰动场景、系统配置与性能指标，从而回答演示器的研究问题。这包括评估 agent 如何响应扰动、人类操作员如何被整合进恢复流程，以及 supervisor agent 的存在如何影响协调质量。

> **L3.** "A second priority is an industry study targeting disturbance-prone production-logistics scenarios in brownfield environments. The aim is to identify industrial cases that motivate SOPL, analyze the structural characteristics of existing systems, and derive requirements for their gradual transformation toward SOPL-compatible architectures."

中译：第二项优先事项是针对棕地环境中易受扰动影响的生产物流场景开展行业研究。目标是识别推动 SOPL 的工业案例，分析现有系统的结构特征，并推导其向 SOPL 兼容架构渐进转型的需求。

> **L4.** "Despite growing interest, practical realization remains limited due to challenges in interoperable communication, adaptive control mechanisms, and scalable coordination architectures [23]."

中译：尽管兴趣日增，但受制于互操作通信、自适应控制机制与可扩展协调架构方面的挑战，实际落地仍然有限。

> **L5.** "However, the SOL typology does not fully capture several requirements that are critical in production logistics contexts. These include modular and reconfigurable physical assets, shared semantic knowledge structures, and human governance mechanisms. These aspects are therefore introduced as complementary design requirements, extending the SOL perspective toward the specific needs of SOPL systems."

中译：然而，SOL 分类法并未充分涵盖生产物流情境中若干关键需求。这些包括模块化与可重构的物理资产、共享语义知识结构，以及人类治理机制。因此这些方面被作为补充性设计需求引入，将 SOL 视角扩展至 SOPL 系统的特定需要。

> **L6.** "The strength indicators are qualitative and represent the relative conceptual importance of a design requirement for achieving a given performance objective."

中译：强度指标是定性的，表示某一设计需求对达成某一给定性能目标的相对概念重要性。

> **L7.** "Inspired by biological systems such as flocks of birds or schools of fish [2], the concept of self-organization has been increasingly adopted in logistics and manufacturing research (SOL, SOMS) [7,23]." / "Decentralized control approaches improve adaptability by distributing decision-making to local entities, but may result in globally suboptimal behavior due to limited coordination and information availability [28]."

中译：受鸟群或鱼群等生物系统启发 [2]，自组织概念在物流与制造研究中被日益采纳（SOL、SOMS）[7,23]。／去中心化控制方法通过把决策分配到局部实体来提升适应性，但可能因协调与信息可得性受限而导致全局次优行为 [28]。

> **L8.** （对本工作性质的定位，间接自陈）"Overall, the paper contributes a conceptual foundation for the design, implementation, and experimental evaluation of SOPL systems."

中译：总体而言，本文为 SOPL 系统的设计、实现与实验评估贡献了一个概念基础。

### 5.7 可复用

- **开源代码**：**无**。
- **数据集**：**无**。
- **评测协议**：**无既定协议**。仅有可作为实验设计输入的候选要素：
  - 3 个预定义扰动场景（AMR 运输途中停止 / 意外新优先任务 / 拣错组件）—— 可直接作为故障注入用例；
  - 3 个研究问题（RQ1 个体 LLM agent 的扰动响应、RQ2 人类操作员整合、RQ3 supervisor agent 的作用）—— 可作为实验自变量/因变量设计框架；
  - supervisor agent 的有无被明确设为实验设计参数。
- **可复用的概念框架**：
  - **SOL typology**（Gerrits, Van Heeswijk, Mes, "Towards self-organizing logistics in transportation: a literature review and typology", International Transactions in Operational Research 31(3):1309–1374, 2024, DOI 10.1111/itor.13408）—— 四维度需求推导透镜；
  - **DSRM 六阶段流程**（Peffers et al., JMIS 24(3):45–77, 2007）—— 系统设计与评估的结构化研究流程；
  - **Table 1 追溯矩阵模板**（性能目标 × 设计需求 + 定性强度符号）—— 可迁移至其他系统的需求-目标对齐分析；
  - **三层参考架构**（physical/embodied + decision-making + knowledge）—— 可作为多 agent 生产物流系统的架构基线；
  - **agent 分类**（embodied vs non-embodied；具身 agent 经 virtual representation 与物理资源交互）—— 一种可直接借用的 agent 边界设计约定；
  - **三阶段路线图**（Foundations → Distributed Autonomy → Intelligent Collective）—— 可作为阶段性研究议程模板。
- **可复用的关联资源**（文中引用的自有/协作工作）：
  - Klein et al., "A Knowledge-Based Intralogistic System for a Circular Factory", Logistics Journal: Proceedings (21), 2025 —— 模块化厂内物流资源；
  - Hofmann et al., "The role of an ontology-based knowledge backbone in a circular factory", at - Automatisierungstechnik 72(9):875–883, 2024 —— 本体知识骨干；
  - Li et al., "A survey on LLM-based multi-agent systems: workflow, infrastructure, and challenges", Vicinagearth 1(1):9, 2024 —— 架构参考来源 [16,27]。


## 【组 g6】

### 事实卡 G6（5 篇）

> 来源目录：`D:\research\DeepReinforcementLearningScheduling\references\`
> 提取工具：Python + PyMuPDF 1.28.0
> 正式发表状态于 2026-09-30 核对各篇 arXiv abs 页面。
> 全部内容只记录事实，不作评价。原文引用用英文原样抄录，后附中文翻译。

---

## 卡片 1 — arXiv:2409.11820

### 1. 题录

| 项 | 内容 |
|---|---|
| 标题 | Optimizing Job Shop Scheduling in the Furniture Industry: A Reinforcement Learning Approach Considering Machine Setup, Batch Variability, and Intralogistics |
| 作者 | Malte Schneevogt, M.Sc.（Lead Author）；Dipl.-Ing (FH) Karsten Binninger, M.Sc.（Supervising Scientist）；Prof. Dr.-Ing. Noah Klarmann（Supervising Professor） |
| 单位 | Rosenheim Technical University of Applied Sciences（罗森海姆应用技术大学），Faculty of Wood Technology and Construction / Faculty of Management and Engineering |
| arXiv ID | arXiv:2409.11820v1 [cs.AI]（cross-list cs.LG, eess.SY） |
| 年 | 2024（论文标注日期 September 19, 2024；arXiv 提交 18 Sep 2024） |
| arXiv Comments | "18 pages, 8 pages" |
| 是否已正式发表 | **否**。arXiv 页面无 Journal reference 字段、无会议名、无出版方 DOI；仅有 arXiv 自颁 DOI `https://doi.org/10.48550/arXiv.2409.11820`（DataCite）。按 arXiv 预印本处理。 |

**论文性质提示**：全文为**概念/框架（concept）论文**，不含任何训练实验、基线对比或评测指标。作者在结论中明确"下一步是落实该概念"（见局限 b）。

### 2. 问题设定

**问题类型**：扩展的作业车间调度问题（JSSP），面向家具制造业的批量生产车间。作者自述是把通用 JSSP 模型向真实生产环境扩展的概念框架。

**资源**
- n 个工件 J = {J1, J2, ..., Jn}，m 台机器 M = {M1, M2, ..., Mm}，总工序数 O = n × m
- **缓冲区（buffer）**：每台机器前设一个，可容纳多个工件，无加工时间，有容量上限（单位可为体积 m³ 或托盘位）
- **运输（intralogistics）**：仅以"运输时间 t"的形式建模，**未建模 AGV 车辆实体**（无车辆分配、无路径冲突）
- **工件体积 V**：随每道工序变化（原木家具的毛坯体积可达成品的 5 倍："The volume of the raw material at the beginning of the process may be up to five times greater than that of the end product"）

**目标**：未给出单一数学目标函数。原文表述为 reward 需按行业目标定制，可含"low buffer levels, reducing make span times, or achieving other, industry-specific objectives"。

**约束**（原文第 9 页逐字列出 6 条）
1. "Machines are limited to processing one job at a time"
2. "The processing of an operation cannot be interrupted"
3. "There are no precedence constraints between operations of different jobs"
4. "Each job has a fixed machine sequence"
5. "Each operation requires a specific machine setup"
6. "Machines need to be set up accordingly before processing a job."

**对通用 JSSP 的 5 项扩展**（原文第 3.1 节）：① 机器换型时间（含对称/非对称/虚拟换型三种）；② 批量大小可变（quantity factor δ）；③ 运输时间（intralogistics）；④ 缓冲区容量；⑤ 交期（deadlines）。

### 3. 方法

**算法**：仅声明使用"DRL"，把 JSSP 建模为 MDP。正文提到 PPO 与 DQN，但**未指定最终算法**。原文："In order to identify the optimal policy, the application of PPO balances the trade-off between policy improvement and stability." 实现建议："The job shop environment can be implemented using toolkits such as OpenAI Gym and libraries such as Stable-Baselines3"。

**网络结构**：**未给出**（论文中无网络层数、维度、激活函数等信息）。

**状态定义**（3 个观测分量，原文 3.3.3 节）
1. **machine info**：3 × m 矩阵，每列一台机器。第 1 行 = 当前加工的工件；第 2 行 = 剩余加工时间；第 3 行 = 当前机器 setup
   - 示例：`(1)t0 = [0 0 0 | 0 0 0]`（Machine 1 | Machine 2 | Machine 3）
2. **job info**：2 × n 矩阵，每列一个工件。第 1 行 = 当前工件体积；第 2 行 = 距交期剩余时间
   - 示例：`(2)t0 = [30 10 20 ; 120 110 100]`
3. **buffer info**：b 维向量，b = 缓冲区数量，表示容量占用状态（可用单位或百分比）
   - 示例：`(3)t0 = [60 0 0]`

原文补充："If a machine has no job assigned to it, it is considered to be idling and available."

**动作空间设计**：**单一离散动作空间（single discrete action space）**。原文："The environment is controlled by a single discrete action space, through which the agent selects the appropriate jobs for processing on a given machine at each time step. The agent's choices are limited to the subset of available jobs and machines"。即：每步为某台机器选择一个工件；动作被掩码到"当前可用工件 × 可用机器"的子集内。

**时间推进**：事件驱动，时间步由"可执行工序的资格（eligibility）"决定。原文："At each time step, the agent identifies all eligible operations Oij, based on the predefined job order of the problem scenario." 示例轨迹：t0 分配 J3→M1（29 min）→ 跳至 t29 → 分配 J1→M1 与 J3→M2 → …

**转移概率**：原文只说"The transition probabilities are estimated based on the experiences of past actions and resulting observations in the training environment"，未给出显式模型。

**奖励（无公式）**：论文**未给出奖励公式**，只有定性描述。
- "it is necessary for the agent to be rewarded in a densely manner"（必须稠密奖励）
- 正奖励项：正确设置机器 setup；按工件正确工序顺序分配机器；加工前把工件存入缓冲区
- 负奖励项："provide the agent with a negative reward for unwanted actions, such as overfilling a buffer or failing to meet a job's deadline"
- 原文："Identifying the optimal reward function is an iterative process that must be conducted on an individual basis."

**折扣因子 γ**：无论文数值，只说需按应用设定，并强调"the application of PPO balances the trade-off between policy improvement and stability"。

**论文中唯一的公式（逐字抄录）**

式 (1)：
> total processing time of an operation = δ · d_ij
> with d_ij as the processing time of one single element of an operation

式 (2)：
> T_total = δ · d + t + s

式 (3)（算例）：
> T_total,O31 = δ · d + t + s
> = 400 · 0.0625 min + 0 min + 4 min
> = 29 min

其中 δ = 批量因子（quantity factor），d = 单件加工时间，t = 运输时间，s = 机器换型时间。

**两种集成策略**（第 4.1 节，非算法差异）
- **Episodic planning**（情节式/离线）：适合低自动化、低网络化工厂；无需与现有系统接口；以周或单个客户订单为一个 episode；通过 dashboard 输入订单（数量、交期）
- **Continuous planning**（连续式/在线）：适合高度自动化工厂；agent 位于 ERP 与 MES 之间（Figure 7 自动化金字塔），实时读取两系统数据，数据显著变化即触发重调度，结果写回系统；原文指出这要求"highly detailed customization"

### 4. 实验

**本论文没有任何实验。** 无训练曲线、无基线对比、无消融、无评测指标。

论文仅提供一个 **mock-up 教学算例**（第 3.3.1 节，Table 5 明确标注 "mock-up data"）：

- 3 个工件 × 3 台机器
- 机器序：J1 = {M1, M2, M3}；J2 = {M1, M3, M2}；J3 = {M1, M2, M3}
- 批量：J1 = 400 pcs；J2 = 100 pcs；J3 = 400 pcs
- 单件加工时间（min）：d11=0.04, d12=0.08, d13=0.0625, d21=0.12, d22=0.14, d23=0.13, d31=0.0625, d32=0.04, d33=0.0625
- 工件体积（m³）：V11=30, V12=20, V13=15, V21=10, V22=8, V23=5, V31=20, V32=15, V33=10
- 缓冲区容量：B1 = 60 m³, B2 = 43 m³, B3 = 30 m³
- 运输时间（min，Table 1）：M1↔M2 = 10，M1↔M3 = 15，M2↔M3 = 15
- 换型时间（min，Table 2 对称）：neutral↔各 setup = 4；setup↔setup = 8
- 换型时间（min，Table 3 非对称，Machine 2）：from neutral → s1/s2/s3 = 9/7/8；from s1 → neutral/s2/s3 = 5/10/13；from s2 → 5/8/6；from s3 → 5/8/7
- 换型时间（min，Table 4 含虚拟换型，Machine 3）：s1↔s2 双向 = 0（虚拟换型，无需物理调整）

**基线 / 指标**：均无。

### 5. 核心数字（照抄原文）

- 批量与工序时长算例："= 400 · 0.0625 min + 0 min + 4 min = 29 min"
- "J1 = 400 pcs；J2 = 100 pcs；J3 = 400 pcs"
- "B1 = 60 m3；B2 = 43 m3；B3 = 30 m3"
- 组合爆炸："The number of possible schedules in a JSSP is growing exponentially by (n!)m"
- 综述统计（Panzer & Bender 2022）："89% of the benchmarked implementations increase the scheduling performance"；"In the field of production scheduling, 67% of the reviewed papers applied value-based algorithms"
- 被引用的他人成绩（Tassel et al.）："their method finds solutions 11% better make-span than the best dispatching rule on Taillard's instances, 10% better than Han and Yang [6] and around 18% better than Zhang et al. [27]"
- 被引用的他人退化现象（Liu et al.）："With increasing sizes of the instances, the performance eventually declined."
- 体积变化量级："The volume of the raw material at the beginning of the process may be up to five times greater than that of the end product."

### 6. 自认局限（原文逐字 + 中文翻译）

**a) 奖励函数与缓冲区溢出**
> "A challenge remains in precisely defining a reward function that considers the proposed elements and the industry specific optimization goals. It remains unclear how to best prevent buffer overfill, particularly given that buffer levels are only checked at the time steps in episodic planning, rather than in between time steps. It remains to be investigated whether an overfilled buffer causes a deadlock of the system when a job cannot be moved to the next buffer because it is full."

中文翻译：如何精确定义一个既能涵盖所提要素、又能体现行业特定优化目标的奖励函数，仍是一项挑战。如何最好地防止缓冲区溢出仍不清楚，尤其是在情节式规划中缓冲区水平只在各时间步被检查、而非时间步之间被检查的情况下。缓冲区溢出是否会导致系统死锁（工件因下一缓冲区已满而无法移动），仍有待研究。

**b) 尚未实现（概念停留于纸面）**
> "The following step of this process is the realisation of the concept described above in order to examine, how an agent would deal with the increased level of complexity and increasing problem sizes."

中文翻译：本工作的下一步是落实上述概念，以考察 agent 如何应对更高的复杂度与更大的问题规模。

**c) agent 的定位是辅助人而非替代人**
> "It is also important to note that the scheduling agents described above have been designed to support human production planners, rather than creating artificial copies of them."

中文翻译：同样重要的是，上述调度 agent 的设计目的是辅助人类生产计划员，而非制造其人工复制品。

**d) 方案不可直接迁移**
> "the presented solutions thereby may not be readily transferable to all furniture production facilities, as each facility will require a solution that is precisely tailored to the factory and its individual scheduling goals."

中文翻译：因此所提出的方案未必能直接迁移到所有家具生产工厂，因为每家工厂都需要一个针对该厂及其自身调度目标精确定制的方案。

**e) Episodic planning 的局限**
> "In a real-world production environment, a number of factors may influence the production planning process, including unforeseen events such as machine breakdowns, delivery delays, or changes in customer requirements. As the episodic planning process is conducted in advance, it is not possible to incorporate unforeseen events into the scheduling process. The introduction of a new product to the portfolio necessitates the re-training of the agent. Even an agent with high generalization capabilities is unable to predict the correct order of operations for the components of a new furniture article. A trained agent is constrained to the specific set of jobs for which it was trained."

中文翻译：在真实生产环境中，诸多因素会影响生产计划流程，包括机器故障、交付延迟或客户需求变化等不可预见事件。由于情节式规划是提前进行的，无法把不可预见事件纳入调度过程。产品组合中引入新产品必然需要重新训练 agent。即使泛化能力很强的 agent，也无法预测新家具部件各构件的正确工序顺序。训练好的 agent 被限制在它所训练的那组特定工件上。

**f) Continuous planning 的局限**
> "optimized production planning frequently requires the simultaneous consideration of multiple objectives, including a minimized lead time, maximized machine utilization and minimized inventory costs. These objectives may, however, be in conflict with one another. Despite the agent's full integration into the production systems, the choice of the optimization goal remains the prerogative of an experienced production planner, who sets and adjusts the goals according to current needs. The interface communication between the agent, the ERP system and the MES requires a highly detailed customization, tailored to the specific needs and local conditions on site. As with the episodic approach, the continuous approach requires a retraining of the agent, when a new furniture article is entered into the system."

中文翻译：优化后的生产计划常常需要同时考虑多个目标，包括最小化提前期、最大化机器利用率、最小化库存成本。然而这些目标可能相互冲突。尽管 agent 已完全集成到生产系统中，优化目标的选择仍是有经验的生产计划员的特权，由其根据当前需要设定和调整目标。agent 与 ERP 系统、MES 之间的接口通信需要按现场具体需求与本地条件做高度细致的定制。与情节式方法一样，当系统中录入新的家具品种时，连续式方法也需要重新训练 agent。

**g) 概念前提假设**
> "The overall range of products doesn't change drastically after training, as this would require a re-training of the agent. Minor changes like 'color changes' would not disrupt the production process."

中文翻译：训练之后整体产品范围不会剧烈变化，否则需要重新训练 agent。像"颜色变化"这样的小改动不会打断生产过程。

### 7. 可复用

- **开源代码**：无。全文未提及代码仓库。
- **数据集**：无。Table 1–5 的 mock-up 数据（3×3 算例、运输时间表、对称/非对称/虚拟换型表）可作为最小可复现算例。
- **评测协议**：无。
- **可复用公式**
  - 式 (1) `δ · d_ij`：用批量因子把批量大小耦合进单工序加工时长
  - 式 (2) `T_total = δ · d + t + s`：把"批量加工 + 运输 + 换型"统一折算为一个工序的总占用时长
- **可复用时序/状态设计**
  - 状态三元组：`3×m 机器矩阵`（加工工件 / 剩余时间 / 当前 setup）+ `2×n 工件矩阵`（体积 / 交期余量）+ `b 维缓冲向量`
  - 用**体积**而非件数追踪工件在缓冲区的占位（适配体积剧烈变化的行业）
  - 事件驱动的可变时间步：由"可执行工序资格"决定跳步，而非固定时间片
- **可复用的集成分类**：episodic（离线、无接口、可单机跑）vs continuous（夹在 ERP 与 MES 之间、事件触发重调度）两种落地形态
- **可复用的实现清单**：论文第 4.2 节给出 10 步实施流程（接口梳理 → 目标定义 → 状态空间 → 动作空间 → 奖励/惩罚 → 训练 → 测试环境 → 部署与扩展 → 员工培训与变革管理 → 维护与持续改进）

---

## 卡片 2 — arXiv:2409.18742

### 1. 题录

| 项 | 内容 |
|---|---|
| 标题 | A History-Guided Regional Partitioning Evolutionary Optimization for Solving the Flexible Job Shop Problem with Limited Multi-load Automated Guided Vehicles |
| 作者 | Feige Liu（China University of Geosciences, Wuhan）、Chao Lu（通讯作者，China University of Geosciences, Wuhan）、Xin Li（The Education University of Hong Kong） |
| 基金 | National Natural Science Foundation of China, Grant Nos. 52175490 and 51805495 |
| arXiv ID | arXiv:2409.18742v1 [eess.SY] |
| 年 | 2024（提交 27 Sep 2024） |
| arXiv Comments | "14 pages" |
| 是否已正式发表 | **否**。arXiv 页面无 Journal reference、无会议名、无出版方 DOI；仅有 arXiv 自颁 DOI `https://doi.org/10.48550/arXiv.2409.18742`。按 arXiv 预印本处理。 |

**论文性质提示**：**这不是 DRL 论文**。全文为进化算法（evolutionary optimization / niching + 空间划分）方法，不含任何神经网络或强化学习组件。

### 2. 问题设定

**问题类型**：**FJSPMA** = Flexible Job Shop Problem with limited Multi-load AGVs（有限多载 AGV 的柔性作业车间调度问题）。原文定位："FJSPMA is a combination of FJSP and Vehicle Routing Problem (VRP)"，属 NP-hard。

**资源**
- n 个工件，每工件含 n_i 道工序；m 台机器
- v 台**多载 AGV**，载重上限为 C（每个工件尺寸为 1）
- 原材料仓库 MW（raw material warehouse）、成品仓库 PW（product warehouse）
- 每台机器前缓冲无限

**目标**：最小化最大完工时间 makespan（"The goal of FJSPMA is to minimize the maximum completion time"，模型写作 `Minimize C_max`）。

**约束**（原文第 III.A 节逐字 6 条）
1. "All jobs and AGVs are ready at the MW from time zero."
2. "The maximum load of each AGV during transportation shall not exceed the upper capacity of the AGV."
3. "It is assumed that the loading and unloading time of the AGV is taken into account in the processing time."
4. "The buffer of each machine is infinite, and if many jobs arrive in the buffer of a machine, the order of process on the machine is not constrained by the order in which the jobs arrive."
5. "When starting processing, jobs need to be transported from MW to the machine, and after completing processing, jobs need to be transported to PW."
6. "The path conflicts caused by AGVs transporting jobs are not considered."

**四个子问题**（原文逐字）
1. "Determine the processing order of all operations for all jobs."
2. "Select processing machines for each operation."
3. "Select a AGV machine for each operation"
4. "Determine the sequence of loading and unloading tasks of all jobs on each AGV."

**MILP 模型要点**（第 III.B 节，公式编号为原文编号）
- 目标 (1)：`C_max ≥ C_ij + p_ij, ∀O_ij`
- (2)：有运输任务的工序必有装卸两个任务，`Σ_t Σ_v X_ijtv = 2 * y_ij`
- (3)：多载 AGV 的核心约束——从任一运输任务起，连续 2C 个任务中装卸数量平衡且不超过容量 C
- (4)(5)：AGV 同一时刻只能执行一个运输任务
- (6)(7)：运输任务先后序的唯一性（`Y_ijt i'j't'v + Y_i'j't'ijtv = 1`）
- (8)(9)(10)：运输任务与加工任务的耦合，卸载任务开始时间受前道工序完工时间制约
- (11)(12)(13)：FJSP 常规约束（每工序选一台机器、同机工序不重叠、工序先后序）

### 3. 方法

**算法**：**HRPEO**（History-guided Regional Partitioning Evolutionary Optimization）。核心思想是把"求解多模态问题（multimodal optimization）"的 niching 技术移植到车间调度：用全局 k-d 树记录历史解空间的区域划分，按区域潜力选择个体进化，并持续探索未搜索区域。

**核心数据结构概念**（原文定义）
- **Region（区域）**：由问题编码指定的多维区间；解落入区域 ⟺ 各维取值都在对应维度的区间内
- **Cluster（簇）**：区域的集合，是更大的多维区间
- **Subpopulation（子种群）**：与 cluster 一一对应，其解全在该 cluster 范围内

**编码**：**三层编码 + 运输任务列表**（Fig 4）
- 第 1 层：全部工序的加工顺序（operation processing sequence）。约定 PW 中的工序加工时间为 0、加工机器标记为 −1
- 第 2 层：机器编码，与工序编码对应，指明该工序的加工机器
- 第 3 层：AGV 编码，指明该工序选用的 AGV 索引
- 运输任务列表：记录每台 AGV 上运输任务的顺序；任务号为 `+` 表示装载任务，`−` 表示卸载任务；**若同一工件的相邻两道工序在同一机器上加工，则后者的运输任务不计**

**解码**：用一个容量等于 AGV 载重上限的容器模拟。按 AGV 与工序的对应关系，把每台 AGV 的运输任务按工序加工顺序列出；每道工序有两个运输任务（装载 `+O_ij`、卸载 `−O_ij`）。装载任务则向容器存入 job block，卸载任务则取出。初始时若容器未满则执行装载，容器满则把其中工件逐个卸载。解码 5 步（原文 Step1–Step5）：取工序 O_ij → 处理装载任务 +O_ij（开始时间 = AGV 上一任务完成时间）→ 遍历 AGV 任务表处理 −O_ij（开始时间 = max(AGV 上一任务完成时间, 前道工序完工时间)）→ 确定工序开始加工时间（max(−O_ij 完成时间, 该机器上前序工序完工时间)）→ 遍历完毕输出最大完工时间。

**种群初始化**：基于分支定界（branch and bound）的**决策树**生成初始解（Fig 9）。根节点为起始节点，第二层为各工件的首道工序，第三层为父节点之后可选工序；机器与 AGV 用 **FCFS 启发式**确定。初始种群规模 `k·k1`，每个二级根节点子树最终只保留 k1 个分支，超出则按适应度剪枝。子种群规模同样是 `k·k1`，故种群规模随实例规模自适应增大。

**种子解识别（NBD 指标，式 (10)）**
```
NBD(x, P) = ∞,                      if {y ∈ P | y better than x} = ∅
          = min_{y ∈ {better}} |x − y|,  otherwise
```
NBD 值大于 `μ + σ·k` 的解被判为离群点（即种子解）；还需过滤相似解。原文还做了两个验证实验：随机生成 1000 个解的 NBD 直方图，以及 HRPEO 迭代过程中某 cluster 内解的 NBD 直方图。

**区域划分与聚类**
- 区域用**全局 k-d 树**记录整个迭代过程中的划分，避免每代重建
- 划分方法：找出所有种子解中方差最大的维度，沿该维度把空间二分成两个半区域（Fig 8）；当簇内任意两个种子解不在相邻区域时停止划分（Algorithm 2）
- 聚类：按区域平均适应度升序排序，递归寻找邻域区域合并成簇（Algorithm 3 / Algorithm 4）；聚类时保证任意两个种子解不同簇
- 区域的平均适应度 = 该区域解集适应度的均值，用作区域潜力的度量
- 需过滤"必定无解"的区域（如某维 High−Low = 0，或受工序顺序约束该区域解必定不可行）

**进化算子**
- **Exploitation**：父代来自同一 cluster。每簇选 `k·k1` 个更优个体构成子种群；若簇内解数 > k·k1 则取最优的 k·k1 个，若不足则全部复制并用簇范围内随机生成的解补齐
- **Exploration**：按各 cluster 适应度均值做轮盘赌选两个 cluster，各随机选一个解进化，共生成 `k2` 个解
- 算子：工序编码用 **POX 交叉与变异**；机器编码与 AGV 编码用 **PMX 交叉与变异**；机器编码变异后需修正（随机选一台适合该工序的机器）
- 所有进化产生的解都必须存入全局 k-d 树

**局部搜索（贪心）**：仅当个体适应度**小于**其所属区域的平均适应度时才执行（以控制贪心带来的时间开销）。
- 机器编码与 AGV 编码：顺序遍历，尝试所有可选机器与 AGV，变优则替换
- AGV 运输任务序列：基于装载/卸载节点的**层数变换**规则（Fig 10）。设 0 为起点、L0…LC 表示当前 AGV 上工件数；装载节点使层数 +1，卸载节点使层数 −1；因容量约束节点位置不能超过给定层数。容量为 3 时，只有 L2、L3 的装载节点可变异为卸载节点，L0、L1 的卸载节点可变异为装载节点；容量为 2 时只有 L2 的装载节点与 L0 的卸载节点可变换。**最后一个卸载节点必须位于 L0**。

### 4. 实验

**实例与规模**
- 采用两个 benchmark：**FJSPT** 与 **EX**，均来自 Deroussi & Norre (2010) "Simultaneous scheduling of machines and vehicles for the flexible job shop problem"
- 原文说明："the datasets do not consider the upper limit of the AGV capacity. Therefore, this paper sets the upper limit of the AGV capacity to **2 or 3**"
- 共 **38 个实例**（Table I 与 Table III 均有 38 行）
- 实现语言 Java；运行环境 Windows 11、20 GB RAM、3.20 GHz Intel(R) Core(TM) i5-10505

**终止准则**：CPU 时间，每算法运行时间 = `n × m × v × 10 ms`（n 工件数，m 机器数，v 多载 AGV 数）

**运行次数**：每算法每实例独立运行 **20 次**

**指标（式 (11)，原文）**
```
ARPD = (1/n) Σ_{i=1..n} (C_i − C_best) / C_best
```
其中 C_i 为当前方法第 i 次试验找到的解的 makespan，C_best 为该实例在所有试验中的当前最优 makespan，n 为试验次数。

**参数标定**：Taguchi 实验法。水平设定 `k1 ∈ {4,5,6,7,8}`、`k2 ∈ {8,9,10,11,12}`、`k ∈ {1.5, 2.0, 2.5, 3.0, 3.5}`。结论：k1 影响最大，其次 k，k2 影响最小。**最优组合 k1 = 6, k2 = 9, k = 3.5**。

**消融变体（4 个）**
- HRPEO1：移除 exploration
- HRPEO2：移除初始化策略，改用随机初始化
- HRPEO3：移除局部搜索策略
- HRPEO4：移除区域划分策略，改用 GA 框架 + 锦标赛选择，保留其他策略

**对比基线（6 个）**
| 基线 | 年份 | 参数设置（原文） |
|---|---|---|
| DHNDE | 2022 | NP = 0.9, CR = 0.3, k = 5, pmax = 0.1, pmin = 1.0e−5, FEs = 200, MaxFEs = 200 |
| DQNMMA（Yao et al., Swarm Evol. Comput. 2024） | 2024 | Np = 90, α = 0.001, γ = 0.9, ε = 0.6, … = 0.2 |
| EDA_ACO_LS（IEEE TASE 2024） | 2024 | n = 400, … = 0.1, PGmax = 0.8, PGmin = 0.4, … = 0.4 |
| EMOEA（Knowl.-Based Syst. 2022） | 2022 | N = 60, … = 0.8, … = 0.5, … = 0.2 |
| LMEO（IEEE TEVC 2023） | 2023 | N1 = 30, N2 = 15, N3 = 15 |
| PSOSA（2023） | 2023 | N1 = 100, N = 50, w = 1/(2 + ln2), c1 = 0.5 + ln2, c2 = 0.5 + ln2 |

**统计检验**：Friedman 方差分析，显著性水平 0.05（Table II 与 Table IV 给出 mean 与 p-value）。

### 5. 核心数字（照抄原文）

**Table II（HRPEO 与其变体的 Friedman 检验，mean / p-value）**
| 对比 | mean | p-value |
|---|---|---|
| HRPEO / HRPEO1 | 3.15582 | 0.016 |
| HRPEO / HRPEO2 | 3.69993 | 0.00216 |
| HRPEO / HRPEO3 | 10.48313 | 1.03266E-24 |
| HRPEO / HRPEO4 | 7.87142 | 3.50646E-14 |

**Table IV（HRPEO 与对比算法的 Friedman 检验）**
| 对比 | mean | p-value |
|---|---|---|
| DHNDE / HRPEO | 8.28351 | 2.51153E-15 |
| DQNMMA / HRPEO | 10.19509 | 4.3796E-23 |
| EDA_ACO_LS / HRPEO | 3.50456 | 0.0096 |
| EMOEA / HRPEO | 3.50456 | 0.0096 |
| LMEO / HRPEO | 5.52234 | 7.0248E-7 |
| PSOSA / HRPEO | 11.36328 | 1.33789E-28 |

**Table III / Table I 部分 ARPD 实例（逐字）**
- instance 1：DHNDE 9.862, DQNMMA 10.793, EDA_ACO_LS 4.207, EMOEA 4.207, LMEO 9.121, PSOSA 10.741, **HRPEO 1.431**
- instance 4：HRPEO **0.765**（全表最小之一）
- instance 9：DHNDE 13.207, DQNMMA 16.052, EDA_ACO_LS 5.190, EMOEA 5.190, LMEO 11.948, PSOSA 14.224, **HRPEO 2.776**
- instance 36：DHNDE 4.750, DQNMMA 4.982, EDA_ACO_LS 3.613, EMOEA 3.613, LMEO 2.452, PSOSA 6.274, **HRPEO 0.375**
- instance 20（消融表，HRPEO 最高值）：HRPEO 2.45679, HRPEO1 2.95062, HRPEO2 3.16049, HRPEO3 12.76543, HRPEO4 5.65432

**结论性数字**
- "HRPEO performs better than other algorithms on **38 instances**"
- 消融对比："HRPEO has achieved good results on **32 instances**"
- "In summary, the local search strategy and the region division strategy have made the main contribution to the improvement of the algorithm performance."
- 箱线图观察："the box of HRPEO is the flattest, the data distribution is denser, and the mean of its distribution is smaller than that of some other comparison algorithms"

### 6. 自认局限（原文逐字 + 中文翻译）

**a) 探索策略并非总是有效**
> "However, in the instance 7 and 14, it is easier to find a better solution without exploration strategy. This is because the solution in the exploration process may take more time to perform regional division, thereby reducing the search efficiency."

中文翻译：然而在实例 7 和 14 上，不使用探索策略反而更容易找到更好的解。这是因为探索过程中的解可能需要更多时间来做区域划分，从而降低了搜索效率。

**b) 初始化剪枝会丢弃有潜力的个体**
> "In the instance 2, 6, 31 and 33, the initialization method mentioned in this article will discard individuals with better potential during pruning, which causes the algorithm to fall into a local optimal state."

中文翻译：在实例 2、6、31 和 33 上，本文所述初始化方法在剪枝过程中会丢弃潜力更好的个体，导致算法陷入局部最优。

**c) 两项策略并非每次都奏效**
> "The initialization strategy and the exploration strategy have also improved the performance of the algorithm to a certain extent, but due to the influence of the instance, they do not work every time."

中文翻译：初始化策略与探索策略也在一定程度上提升了算法性能，但受实例影响，它们并非每次都奏效。

**d) 未来工作（即当前未覆盖的约束）**
> "In the future, we will continue to study the job shop scheduling problem of coupled processing and transportation. We can introduce more realistic constraints, such as path conflicts, AGV transportation speed and energy consumption. At the same time, applying multimodal problem-solving strategies to improve algorithms for solving shop scheduling problems still has significant room for improvement, such as in distinguishing individual characteristics, constructing subpopulations, and optimizing clustering methods."

中文翻译：未来我们将继续研究加工与运输耦合的车间调度问题。我们可以引入更真实的约束，如路径冲突、AGV 运输速度与能耗。同时，将多模态问题求解策略用于改进车间调度算法仍有很大提升空间，例如在区分个体特征、构建子种群以及优化聚类方法方面。

**e) 模型假设层面的局限（约束第 3、6 条逐字）**
> "It is assumed that the loading and unloading time of the AGV is taken into account in the processing time."
> "The path conflicts caused by AGVs transporting jobs are not considered."

中文翻译：假设 AGV 的装卸时间已计入加工时间中；不考虑 AGV 运输工件所导致的路径冲突。

**f) 基准数据本身不含 AGV 载重上限**
> "This paper uses two benchmarks: FJSPT and EX [36], but the datasets do not consider the upper limit of the AGV capacity. Therefore, this paper sets the upper limit of the AGV capacity to 2 or 3."

中文翻译：本文使用 FJSPT 与 EX 两个基准，但这些数据集不考虑 AGV 载重上限。因此本文自行把 AGV 载重上限设为 2 或 3。

### 7. 可复用

- **开源代码**：无。全文未提及代码仓库。
- **数据集**：**FJSPT 与 EX 基准集**，来源 Deroussi & Norre (2010), "Simultaneous scheduling of machines and vehicles for the flexible job shop problem"。注意：该基准原设计不含 AGV 载重上限，本文自行设定容量为 2 或 3——复现时须采用同一设定。
- **评测协议（可直接借用）**
  - 指标 **ARPD**（式 11）：`ARPD = (1/n) Σ (C_i − C_best)/C_best`
  - 每算法每实例独立运行 **20 次**
  - **终止准则用 CPU 时间**，且尺度随实例规模缩放：`n × m × v × 10 ms`
  - **Friedman 检验**，α = 0.05
  - 参数标定用 **Taguchi 正交实验**（本文最优 k1=6, k2=9, k=3.5）
- **公式**：MILP 模型 (1)–(13)（尤其 (3) 是多载 AGV 的容量平衡约束，可复用）；NBD niching 指标（式 10）；ARPD（式 11）
- **结构性可复用设计**
  - 三层编码 + 运输任务列表（`+` 装载 / `−` 卸载）的表示法
  - 容器式解码（容量 = AGV 载重上限）
  - 多载 AGV 装载/卸载节点**层数变换规则**：容量 C 下，只有位于较高层（如 C=3 时的 L2、L3）的装载节点能变异为卸载节点，位于低层（C=3 时的 L0、L1）的卸载节点能变异为装载节点；最后一个卸载节点必须落在 L0
  - 用**全局 k-d 树**持久记录解空间划分，避免每代重建
  - 局部搜索的**条件触发**（仅当个体劣于所属区域平均适应度时执行），以控制贪心的时间开销

---

## 卡片 3 — arXiv:2205.03294

### 1. 题录

| 项 | 内容 |
|---|---|
| 标题 | Vehicle management in a modular production context using Deep Q-Learning |
| 作者 | Lucain Pouget¹†、Timo Hasenbichler²†、Jakob Auer¹、Klaus Lichtenegger²、Andreas Windisch²,³,⁴,⁵,⁶（† 前两位为同等贡献） |
| 单位 | ¹Skalar Systems GmbH, Graz, Austria；²FH JOANNEUM – University of Applied Sciences, Data Science and Artificial Intelligence, Graz；³Know-Center GmbH, Graz；⁴Institute of Interactive Systems and Data Science, Graz University of Technology；⁵Physics Department, Washington University in St. Louis；⁶RL Community, AI Austria, Vienna |
| arXiv ID | arXiv:2205.03294v1 [cs.LG] |
| 年 | 2022（提交 6 May 2022） |
| arXiv Comments | 无（arXiv 页面未显示 Comments 字段） |
| 是否已正式发表 | **否**。arXiv 页面无 Journal reference、无出版方 DOI；仅有 arXiv 自颁 DOI `https://doi.org/10.48550/arXiv.2205.03294`。PDF 使用 "Springer Nature 2021 LATEX template" 排版（表明按 Springer 期刊格式准备），但 PDF 内亦无卷期/期刊名。按 arXiv 预印本处理。 |

**基金**：Austrian Research Promotion Agency FFG, Kleinprojekt Nr. 883243。原文注明："Parts of the findings of this article are also given in [18]."

### 2. 问题设定

**问题类型**：模块化生产（modular production）中的**车辆管理问题（Vehicle Management, VM）**——把工件在 Source、各工作站、Sink 之间运输的 AGV 调度。原文将其归为 JSSP："This task leads to the Job Shop Scheduling Problem (JSSP), which is – like the related Travelling Salesman Problem – a computationally 'hard' (NP-complete) problem of enormous practical importance"。原文还指出 VM 问题在 [6] 中是用 proximal policy optimization 解决的。

**资源**
- **nM 个模块化工作站**：每站 = 输入缓冲 IB + 生产单元 PU + 输出缓冲 OB；两个缓冲容量均为 nbuf，按 FIFO 队列运行
- **Source**（起点，相当于只有 OB 的站）、**Sink**（终点，相当于只有 IB 的站）
- **nV 台 AGV** V，速度 vV，**一次只能载 1 个工件**
- 转移时间 Ttransfer（工件在车辆与站之间、或站内两单元之间转移）
- **waypoint graph G**：加权多向图，节点为车辆可经过的路径点（带笛卡尔坐标 x_j, y_j），边为连接路径（单向或双向），边权为节点间欧氏距离；每个工作站缓冲绑定到图的一个节点，一个节点最多绑定一个缓冲
- **工件（parts）**：按 part type PT 定义；每种 PT_j 有 nO 道工序的序列，指明必须按序访问的工作站；序列总从 Source 开始、Sink 结束；同一工作站可在序列中重复出现；不同 PT 不必使用相同工作站
- 工件在 Source 的释放顺序：part type 上的均匀分布；source clock C_source 决定新工件何时进入系统

**目标**：最大化产率（"Our goal is to maximize the production throughput by assigning jobs to the AGVs"）。

**约束**
- 无死锁（"Jobs must be assigned in such a way as to ensure that there are no deadlocks"）
- 工件必须按各自 PT 的工序序列经过工作站
- 缓冲容量有限（FIFO）
- 每台 AGV 一次只载 1 个工件
- 原文明确排除的能力：卸货站是隐含的，不由 agent 选择（见动作空间与局限 d）

### 3. 方法

**算法**：**DQN 家族**。原文："we tested the vanilla DQN agent described in [13] and implemented some additional promising extensions"。测试的扩展包括：**Double DQN（DDQN）**、**noisy networks**、**dueling network 架构**、**优先经验回放（PER）**。原文："the most significant extension that showed the most robust performance during all tests is the prioritized experience replay (PER)"。最终用于对比的是 **DDQN**。

**网络结构**：2 隐层全连接网络，FC1 = 64 单元，FC2 = 32 单元（Table 1 "DDQN training hyperparameters"）。

**超参数（Table 1 逐字）**
| 参数 | 值 |
|---|---|
| Replay memory size | 1e5 |
| Batch size | 64 |
| Gamma | 0.99 |
| Learning rate | 0.001 |
| Target update rate | 24 |
| Update rate | 4 |
| Epsilon | 1.0 |
| Epsilon decay | 0.9995 |
| Epsilon min | 0.01 |
| NN nb layers | 2 |
| FC1 units | 64 |
| FC2 units | 32 |

**状态定义**：环境状态 S = 所有车辆状态 S_v 与所有站单元状态 S_unit 的拼接。

每辆车 V_i 的 S_vi 包含：
- **Action state**：当前动作的 1-hot 向量，取值集合 = {DRIVING, TRANSFERRING IN, TRANSFERRING OUT, WAITING FOR ORDER, WAITING TO DROPDOWN, WAITING TO PICKUP}
- **Carried part**：所载工件的表示（若为空则无）
- **Current order target**：1-hot 向量，表示 V_i 的驾驶目的地；未驾驶时为零向量
- **Last visited node**：1-hot 向量，表示 V_i 最后访问的节点

每个站单元 U_i 的 S_uniti 包含：
- **Action state**：1-hot 向量，取值集合 = {PROCESSING, TRANSFERRING IN, TRANSFERRING OUT, WAITING TO DROPDOWN, WAITING TO PICKUP}
- **Carried parts**：该单元内各工件的表示

每个工件（在站单元中或在 AGV 上）的表示包含：
- **Part type**：1-hot 向量
- **Part completion**：0~1 的浮点值。工件在 Source 时为 0，在 Sink 时为 1，中间每执行一次动作（驾驶或加工）线性增加
- **Part next station**：1-hot 向量，表示下一个需访问的站（加工完毕后为 Sink）。原文注："In principle, if only part type and part completion is provided, the Agent should be able to learn the next steps for a part by itself. However, we discovered by experience that providing redundant information helped to increase the overall performance."

**动作空间设计**：**离散动作空间**，集合 = 所有可**取件**的站（Source + 全部工作站）。原文："The set of all possible actions is equivalent to the set of stations where it is possible to pick-up a part (i.e., the Source and all the workstations). The Sink is not included in the set of actions, since it's only possible to drop off a part there."

外加一个 **"do nothing" 动作**，不分配任何任务。原文："In addition to this set of actions, we also allow the Agent to take a 'do nothing' action, for which no task is assigned."

**关键设计差异（与 [6] 相比）**：**卸货动作不建模**。原文："In contrast to [6], the set of all dropdown actions A_drop is not defined in our case, since the dropdown station is always implicit and only depends on the carried part."（作者自认这是局限，见局限 d）

**奖励（势能差分形式，逐字抄录）**

原文："we built a score function for each environment state S_t. From a state S_t and for an action A_t that resulted in the state S_{t+1}, we defined the reward R(S_t, A_t) as `Score(S_{t+1}) − Score(S_t)`."

Score 由 4 个分量组成：
- **Per-part reward S_pp**：每个到达 Sink 的完工工件给正奖励
- **Per-part completion reward S_pp%**：工件每次被加工给正奖励；工件完成度（0~1）定义为已完成加工步数占全部步数的百分比，Source 处为 0，Sink 处为 1
- **Per-assigned decision reward S_decisions**：每个被真正分配到 AGV 的决策给正奖励；**"do-nothing" 动作不给奖励**。原文："We noticed a slight increase in performance when forcing the Agent into being less lazy."
- **Per-second reward S_time**：每模拟一秒未发生死锁给正奖励

记 N_decisions(t) 为自仿真开始以来被分配的决策数，N_pp(t) 为完工工件总数，N_pp%(t) 为各工件完成度之和（实际中 N_pp%(t) ≥ N_pp(t)）。定义三个可配置值：E_pp（期望加工工件数）、E_decisions（期望分配决策数）、E_seconds（期望时长）。最终 Score 为 4 个分量的加权平均：

```
        ⎧ K [ (N_pp(t)+N_pp%(t))/E_pp + N_decisions(t)/E_decisions ],                     if deadlocked,
S(t) =  ⎨
        ⎩ K [ (N_pp(t)+N_pp%(t))/E_pp + N_decisions(t)/E_decisions + t/E_seconds ],        otherwise.
```
（式 (2)）

原文："In practice, we found that **K = 4000** worked best for us. E_pp, E_decisions and E_seconds are estimated based on the Cost Table Agent performances."

**Q 学习定义（式 (1) 逐字）**
> Q*(s, a) = E_{s'}[ r + γ max_{a'} Q*(s', a') | s, a ]

**环境与通信**
- **离散事件仿真（DES）**：Python + Simpy 库。原文解释选型："We chose this approach because of its efficiency as compared to continuous simulations."
- **OpenAI Gym 接口**：把 Simpy 仿真与决策过程解耦；Observation 是人类可读字典，Gym 负责序列化为向量
- **Agent 统一接口**（确定性 agent 与参数化 agent 共用）：`Act(Observation) → action`；`Step(Experience) → nothing`
- **Controller 通信协议**（8 步，原文）：① 环境初始化 → ② 仿真开始 → ③ 某 AGV 空闲（完成当前任务）时触发自定义 Simpy 事件 → ④ Controller 暂停仿真并由环境计算观测状态 → ⑤ 观测发给 Agent → ⑥ Agent 给出动作 → ⑦ Controller 反序列化动作并分配给等待的 AGV（**若多台 AGV 同时空闲，Controller 把任务分给能最快完成该动作的 AGV**，即最近的 AGV；然后立即再次调用 Agent 为其余空闲 AGV 指定任务，直到所有 AGV 都有任务）→ ⑧ 所有 AGV 都有任务后回到 ②

### 4. 实验

**实例与规模（7 类环境配置）**
| 配置 | 机器数 | AGV 数 | 特点 |
|---|---|---|---|
| Mayer-Classen-Endisch | 2 | 1 | 复现自 [6]，1 种工件类型，用于验证实现能重现 [6] 的结果 |
| 1-machine-big | 1 | 2 | waypoint graph 距离特殊：source/机器入口近、机器出口/sink 近，但 source/sink 与 机器入口/机器出口 相距很远。最优解是两台 AGV 各跑一条短循环 |
| 3-machines-loop | 3 | 1 / 2 / 3 | 工序序列含回路：Source → M1 → M2 → M1 → M3 → Sink。原文："This is a well known type of problem in modular production that often leads to deadlocks when using static algorithms." |
| 6-machines-grid（家族） | 2 / 4 / 6 在用 | 1 / 2 / 3 / 4 | 6 台机器按 2 行 3 列网格布置。2 机器场景：M1→M2 与 M2→M1；4 机器：M1→M3→M2→M4 与 M2→M4→M1→M3；6 机器：M1→M3→M5→M2→M4→M6 与 M2→M4→M6→M1→M3→M5。每场景 2 种工件类型 |

**基线（3 个确定性 agent）**
- **FIFO**：把等待最久的工作站输出缓冲中的工件（S_longest）分配给等待最久的 AGV（AGV_longest）
- **Nearest Neighbor（NN）**：选定 S_longest，对每台 AGV 估计到达该站出口的耗时（空闲 AGV 估计直接驶往的耗时；忙碌 AGV 估计完成当前任务后再到达的耗时），分配给耗时最短者；若该 AGV 已在忙则不分配
- **Cost Table**：NN 的扩展。构建 `[未激活 AGV 数 × 等待站数]` 的耗时矩阵（"costs"），用 SciPy 的 linear sum assignment 求解线性求和分配问题；每步只做一个分配，在解中所有 AGV 里选等待最久的那台对应的任务，然后立即重算 Cost Table 再求解，直到所有 AGV 都激活或无工件等待

**指标**：**throughput（parts per hour, pph）**；同时记录 12 小时总产量。

**方法学关键（评测协议）**：静态算法对输入节拍 C_source 高度敏感。原文："If C_source is too low, the AGVs will almost immediately overload the first machine with parts and run into a deadlock. On the contrary, if C_source is too high, the optimization problem doesn't make sense anymore because it is too easy to get an output clock C_sink equal to C_source."
- 因此先对每个静态 agent、每种环境配置用**二分搜索**求出 **C*_source**（不陷入死锁的最低输入节拍，取整到 1 秒），再跑 12 小时仿真
- **DRL agent 的 C_source 在训练与测试阶段都设为 0**。原文："It is assumed that it should learn how to avoid deadlocks by not overloading the first machine."

**仿真性能基准**：典型训练环境下，仿真 1 小时生产耗时 **360 ms ~ 2.6 s**（取决于环境复杂度）。测试机：Dell XPS 15，Intel core i7 2.60 Hz 单线程。原文说明不提供定量加速比："Lacking an obvious definition of complexity of the environment renders quantitative speed-up factors somewhat arbitrary. We thus refrain from providing quantitative data at this point"。

### 5. 核心数字（照抄原文）

**Mayer-Classen-Endisch（复现 [6]）**
- "The trained agent reached a throughput of **71.8 parts/hour** over 12 hours (**863 parts**). The maximum throughput in this case is **72 parts/hours (864 parts)**."
- "the baseline agents (FIFO, Nearest Neighbor and Cost Table) have also been able to achieve the same performance (**862 parts** produced). The main benefit of the DDQN agent is its robustness to the source clock."
- 学到的最优策略：AGV 按固定顺序循环 Source → M1input → M1output → M2input → M2output → Sink → Source → …

**1-machine-big（2 AGV）**
- FIFO **23.6 pph**（总产 284 件）、NN **74.9 pph**（899 件）、Cost Table **143.3 pph**（1720 件）、**DDQN 143.4 pph（1721 件）**；平衡态最优吞吐 **144 pph**
- 原文："It is also worth mentioning that the Nearest Neighbor approach gets intermediate results (74.9 parts/hour) which confirm that all static approaches are not equivalent even in small scenarios."
- DDQN 与 Cost Table 学到同一策略：一台 AGV 在 Source 与机器入口间循环，另一台在机器出口与 Sink 间循环

**3-machines-loop**
- 1 AGV：FIFO 23.4、NN 23.4、CT 26.9、DDQN 26.3（pph）；总产 281 / 281 / 326 / 317
- 2 AGV：FIFO 44.2、NN 44.8、CT 43.7、DDQN 45.4（pph）；总产 531 / 538 / 525 / 547
- 3 AGV：FIFO 63.8、NN 62.8、CT 65.0、DDQN 60.2（pph）；总产 768 / 755 / 782 / 725
- 原文："Cost Table is slightly better than DDQN with 1 AGV (26.9 vs 26.3 pph), slightly worse with 2 AGVs (43.7 vs 45.4 pph) and significantly better with 3 AGVs (65.0 vs 60.2 pph)."

**6-machines-grid / 2-machines 场景**
- 1 AGV：FIFO 115.8、NN 115.8、CT 123.1、DDQN 115.2（pph）
- 2 AGV：FIFO 143.7、NN 143.8、CT 143.8、DDQN 139.2
- 3/4 AGV：所有 agent 都达 144 pph 上限（DDQN 143.8/143.8）

**6-machines-grid / 4-machines 场景**
- 1 AGV：CT 95.7 vs DDQN 88.8（原文："Cost Table performs better than other approaches with 1 AGV (96 pph vs 89)"）
- 2 AGV：DDQN 128.9，其余均 ~143.4
- 3 AGV：DDQN 138.9；4 AGV：DDQN 134.1
- 原文："The best results are achieved with 3 AGVs (139 pph), a better performance than when a 4th AGV was available (134 pph)."

**6-machines-grid / 6-machines 场景**
- 原文逐字："even if the DDQN performs worse than Cost Table in all cases (**69 vs 74 pph with 1 AGV, 79 vs 128 pph with 2, 117 vs 143 with 3 and 124 vs 143 with 4**), it is still able to run the 12 hours without deadlocks."
- 对应总产：DDQN 828 / 951 / 1413 / 1491；FIFO 812 / 1485 / 1722 / 1723

**最优 source clock 表（Table 2，秒）部分值（原文表格，按列语义重排）**
- Mayer-Classen-Endisch（1 AGV）：FIFO/NN/CT/DDQN 均为 50
- 1-machine-big：FIFO 83、NN 48、CT 11、DDQN 0
- 3-machines-loop：FIFO 153/81/56；NN 153/80/57；CT 120/82/55；DDQN 0/0/0
- 6-machines-grid 系列：DDQN 全为 0；FIFO/NN/CT 从约 20~53（1 AGV）降至 25~29（多 AGV）

### 6. 自认局限（原文逐字 + 中文翻译）

**a) 观测信息不完整**
> "Our final observation state does not provide all the information needed by the agent to take the optimal decision. In particular, with our approach, the DDQN agent can't take advantage of the exact position of the AGVs and the transportation times. Our efforts to include this information in the observation state have not yet been successful. However, this information is available and used by the Cost Table agent, which can explain its better performances."

中文翻译：我们最终的观测状态并未提供 agent 做出最优决策所需的全部信息。特别地，在我们的方法中，DDQN agent 无法利用 AGV 的精确位置与运输时间。我们试图把该信息纳入观测状态的努力尚未成功。然而，Cost Table agent 可用并使用了这些信息，这可以解释其更优的表现。

**b) 对奖励信号（尤其死锁惩罚）过于敏感，导致策略过度保守**
> "Another learning of our study is the sensitivity of the DDQN agent to the reward signal, especially to the deadlock punishment signal. In the end, our agent learned how to avoid running into a deadlock but at the cost of been too protective in some cases (i.e., it is more risky and not enough rewarded to start processing a new part as compared to wait for the production plant to be less full). This balance has been tough to configure and is still not entirely satisfying."

中文翻译：我们研究的另一项经验是 DDQN agent 对奖励信号、尤其对死锁惩罚信号的敏感性。最终我们的 agent 学会了避免陷入死锁，但代价是在某些情况下过于保守（即与等待产线不那么满相比，开始加工新工件风险更高、回报不足）。这一平衡很难配置，至今仍不完全令人满意。

**c) 多 AGV 管理能力受限，动作集过窄**
> "The study has shown the limitation of our approach when it comes to managing several AGVs at the same time. We believe that there is room for improvement in the communication protocol between the Agent and the AGVs. Our intuition is that the current set of actions the Agent can take is too limited. We had difficulties into making the Agent choose both the Station and the AGV that must perform the action."

中文翻译：本研究显示了我们的方法在同时管理多台 AGV 时的局限。我们认为 Agent 与 AGV 之间的通信协议仍有改进空间。我们的直觉是当前 Agent 可采取的动作集合过于受限。我们难以让 Agent 同时选择工作站和执行该动作的 AGV。

**d) 卸货动作未建模**
> "In contrast to [6], the set of all dropdown actions A drop is not defined in our case, since the dropdown station is always implicit and only depends on the carried part. This is a limitation in the case of a production plant where the same operation can be performed by 2 different workstations, but it is beyond the scope of this paper."

中文翻译：与 [6] 不同，我们这里没有定义全部卸货动作集合 A_drop，因为卸货站总是隐含的且只取决于所载工件。当同一道工序可由 2 个不同工作站完成时，这是一个局限，但已超出本文范围。

**e) 论文定位为初步可行性探索**
> "all these avenues are left unaddressed in this study, as the purpose of this paper is to establish a first, preliminary exploration of the feasibility of Deep-Q based DRL approaches in modular production environments."

中文翻译：本研究未涉及所有这些方向，因为本文的目的是对基于 Deep-Q 的 DRL 方法在模块化生产环境中的可行性做一次初步探索。

**f) 探索中未被处理的议题（结论原文）**
> "Beyond the scope of this paper, there exist various avenues for further exploration. For instance, we hypothesize that a DRL approach increases stability in the system. This is suggested by the fact the DDQN agent is robust against the source clock, while static agents need additional precautions."
> "Also, the issue of dead-lock avoidance could be investigated more thoroughly."

中文翻译：在本文范围之外还有多种可进一步探索的路径。例如我们假设 DRL 方法能提升系统稳定性——DDQN agent 对 source clock 具有鲁棒性、而静态 agent 需要额外预防措施，这一点支持了该假设。此外，死锁避免这一问题也可以被更彻底地研究。

**g) 仿真加速比无法定量**
> "Lacking an obvious definition of complexity of the environment renders quantitative speed-up factors somewhat arbitrary. We thus refrain from providing quantitative data at this point"

中文翻译：由于缺乏对"环境复杂度"的明确定义，定量给出加速比会显得随意。因此我们在此暂不提供定量数据。

### 7. 可复用

- **开源代码**：无。全文未提及代码仓库。
- **数据集**：无外部数据集。7 类环境配置描述完整；其中 Mayer-Classen-Endisch 复现自 [6]（Mayer, Classen, Endisch），其余 6 类为自建。
- **评测协议（可直接借用）**
  - 对 deadlock-prone 场景下的静态基线，先用**二分搜索**求各自的不死锁最小输入节拍 C*_source（取整到 1 秒），再在同一时长的仿真中对比——这解决了"基线因参数不当而死锁"造成的对比不公平
  - DRL agent 的 C_source 训练与测试均设为 0，把"避免过载"交由 agent 自学
  - 统一对比时长 12 小时生产；主指标为 parts per hour
- **公式**
  - 式 (1) `Q*(s,a) = E_{s'}[r + γ max_{a'} Q*(s',a') | s,a]`
  - 式 (2) Score 函数（含 deadlock 分支）与 4 分量设计；`K = 4000`；权重 E_pp / E_decisions / E_seconds 由 Cost Table agent 的表现估计
  - **奖励即势能差**：`R(S_t, A_t) = Score(S_{t+1}) − Score(S_t)` —— 这是把多目标分项拼成一个标量 Score 再差分的可复用手法
- **工程可复用组件**
  - **Simpy 离散事件仿真 + OpenAI Gym 接口 + 统一 Agent 接口（Act / Step）+ Controller 通信协议** 的整套骨架（事件驱动、仿真暂停/恢复、观测序列化）
  - 多 AGV 同时空闲时的分配规则："把任务分给能最快完成该动作的 AGV（即离得最近的）"，然后立即再次调用 Agent 处理其余空闲 AGV
  - 观测中对工件编码使用**冗余特征**（part type + completion + next station）以提升性能

---

## 卡片 4 — arXiv:2511.07071（博士论文）

> **说明：本篇系博士论文（Dissertation），全文 173 页。按要求仅读摘要（Abstract / Kurzfassung）、方法章（Chapter 3 Methodology，第 73–102 页）、结果章结论（4.4，第 143–144 页）、讨论章（Chapter 5，第 145–156 页）与结论章（Chapter 6，第 154–156 页）；未逐页通读。数据取自上述范围内的正文与表格。**

### 1. 题录

| 项 | 内容 |
|---|---|
| 标题 | Multi-Agent Reinforcement Learning for Deadlock Handling among Autonomous Mobile Robots |
| 作者 | Marcel Müller, M.Sc.（单一作者） |
| 学位 | Dissertation zur Erlangung des akademischen Grades Doktoringenieur (Dr.-Ing.)，即工学博士论文 |
| 授予单位 | Otto-von-Guericke-Universität Magdeburg，Fakultät für Maschinenbau（马格德堡奥托·冯·格里克大学机械工程学院） |
| 评审 | Prof. Dr.-Ing. Hartmut Zadek；Prof. Dr.-Ing. Ernesto William De Luca |
| 答辩日期 | Promotionskolloquium am 27. Oktober 2025（2025 年 10 月 27 日） |
| arXiv ID | arXiv:2511.07071v1 [cs.MA]（cross-list cs.RO） |
| 年 | 2025（提交 10 Nov 2025） |
| arXiv Comments | "for associated repositories, see [https URL]"（含两个 GitHub 链接） |
| 是否已正式发表 | **否**（作为期刊/会议论文而言）。arXiv 页面无 Journal reference、无出版方 DOI；仅有 arXiv 自颁 DOI `https://doi.org/10.48550/arXiv.2511.07071`。作为**学位论文**已通过答辩并被大学接受。 |

### 2. 问题设定

**问题类型**：**死锁可发生的多智能体路径规划（deadlock-capable MAPF）**，应用于依赖 AMR 的内部物流（intralogistics）系统。原文摘要："This dissertation explores the application of multi-agent reinforcement learning (MARL) for handling deadlocks in intralogistics systems that rely on autonomous mobile robots (AMRs)."

**研究缺口与动机**（原文摘要）："Existing approaches often neglect deadlock handling in the planning phase and rely on rigid control rules that cannot adapt to dynamic operational conditions."

**资源**
- 网格化环境中的 AMR（agent）群体；障碍/墙体格、自由通行格
- 充电桩（外部仿真用例中）
- 参考模型覆盖三类场景：冲突场景、仓库布局、生产物流
- 外部仿真：Tecnomatix Plant Simulation 中的轨道（track）系统，工件经源 A/B → 工位 A/B → sink

**目标**
- 物流侧：把"韧性（resilience）"作为性能、质量、成本之外的**第四物流目标**，并提出把死锁处理前置到规划阶段
- RL 侧：最大化 episode 回报
- 评测指标：成功率、时间步数、计算时间

**约束**
- 碰撞禁止：顶点冲突（多智能体同格同时）与边冲突（两智能体交换位置）均不允许
- 部分可观测：CTDE 与 DTE 模式下观测被限制在传感器范围内；仅 CTE 可见完整网格
- 死锁：论文的核心处理对象
- 智能体被阻挡时可能保持不动（导致"不作为"惩罚）

**6 个研究问题（原文逐字）**
- "RQ 1.1: Is deadlock handling significant?"
- "RQ 1.2: How to integrate RL into the planning process?"
- "RQ 2.1: What do reference scenarios for deadlock-capable MAPF problems look like?"
- "RQ 2.2: When and in which configurations is multi-agent RL effective for MAPF with deadlocks?"
- "RQ 3.1: Does the deadlock handling strategy matter?"
- "RQ 3.2: Is centralized training and decentralized execution the best approach in deadlock-capable MAPF problems?"

（对应三类研究：Fundamental research（RQ1.x）、Transfer research（RQ2.x）、Application research（RQ3.x），见 Figure 2.23）

### 3. 方法

**算法**：**PPO** 与 **IMPALA**，在三种训练/执行模式下对比：
- **CTE**（Centralized Training, Centralized Execution，集中训练集中执行）
- **CTDE**（Centralized Training, Decentralized Execution，集中训练分散执行）
- **DTE**（Decentralized Training, Decentralized Execution，分散训练分散执行）

**网络结构**：全连接前馈网络。外部仿真用例的具体结构："The ANN consists of two hidden layers, each containing 256 neurons, with the hyperbolic tangent (tanh) activation function applied to each layer. This architecture is designed to support simultaneous operation of one to four agents in the environment."

**环境实现**："The reference models are developed in Python and conform to the Gymnasium standard, ensuring compatibility with popular RL libraries such as Stable-Baselines3 and RLlib." 多智能体场景继承 Ray RLlib 的 `MultiAgentEnv`，单智能体场景继承 Gymnasium 的 `gym.Env`。网格用 2D NumPy 数组表示。

**参考模型（3 类共 7 个，加变体）**

| 用例 | 参考模型 | 结构 |
|---|---|---|
| 冲突场景 | 1.1 | 双智能体，含短侧道供避让；变体：基本、不利起终点、侧道位置变化 |
| | 1.2 | 无侧道的窄走廊，agent 须事先决定是否进入或退回；变体：基本、起终点与智能体数、走廊长度 |
| | 1.3 | 中间竖墙形成单一小通道，通道前后有充足空间；变体：基本、起终点与智能体数、区域与通道位置 |
| | 1.4 | 四向交叉口，4 个智能体必经路口；变体：基本、起终点与智能体数、路径长度 |
| 仓库 | 2.1 | 多区块仓库布局：竖向双车道巷道 + 横向单车道巷道；变体 2.1b：去掉竖向车道、增加智能体数，形成死胡同 |
| | 2.2 | 鱼骨（fishbone）布局：斜向巷道与中央垂直巷道相交 |
| 生产物流 | 3.1 | 单/双车道混合 + 死胡同 + 固定目标位（加工单元/缓冲/存储区）；变体：基本、短目标巷道。**改编自真实装配供料布局** |

设计推导遵循"从抽象到真实、从简单到复杂"两个维度；两类死锁原型被识别为：(i) 一维受限通道内的相互阻塞（如单车道走廊的正面相遇），(ii) 交叉口或缓冲区的空间争夺造成的循环依赖。参考模型**不含随机生成的环境**。

**状态定义（Table 4.2，网格参考模型）**
| 模式 | 观测空间 |
|---|---|
| CTE | 完整网格：空格、障碍、智能体位置、目标 |
| CTDE & DTE | 受传感器范围限制的部分网格视图 + 该智能体的当前位置 + 该智能体的目标 |

**动作空间设计（Table 4.2，网格参考模型）——离散/多离散**
| 模式 | 动作空间 |
|---|---|
| CTE | **MultiDiscrete([5] * n)**（n = 智能体数），每个智能体 5 个动作：0: No-op, 1: move up, 2: Move right, 3: move down, 4: move left |
| CTDE & DTE | **Discrete(5)**：0: No-op, 1: move up, 2: Move right, 3: move down, 4: move left |

**外部仿真用例的状态与动作（Table 4.11）**
- **观测空间**
  - 通用：被考虑的 AGV/agent 的 ID；工位 A 和 B 的状态；源 A 和 B 缓冲中的物品数
  - 每个 agent：agent i 的 x、y 位置；agent i 的当前速度；agent i 的目标；agent i 的空载状态；agent i 到目标的剩余路径长度
- **动作空间（驾驶行为）——离散 3 个**：**0 (hold), 1 (forwards), 2 (backwards)**

**奖励 A：网格参考模型（逐字抄录）**

式 (4.20)：
> r_i(t) = α · J_i(t) + β · K(t) + γ · C_i(t),   ∀ i ∈ I, t ∈ T.

式 (4.21) 常数赋值：
> α = 0.5, β = 1, γ = −1.

三个分量：
1. "**First-time goal reach reward (α · J_i(t))**: An agent receives a reward of +0.5 only when it reaches the goal for the first time."
2. "**All agents at goal reward (β · K(t))**: If all agents are at the goal at the same time t, each agent receives an additional reward of +1, and the episode terminates."
3. "**Collision penalty (γ · C_i(t))**: An agent i incurs a penalty of −1 if it is involved in a collision at time t."

碰撞判定（式 4.18 / 4.19）：`pos_i(t+1) = pos_j(t)`（试图进入他者占据的格）或 `pos_i(t+1) = pos_j(t+1)`（同时进入同一格）。原文注明位置互换的情形被包含在第一个条件中。

episode 终止条件（式 4.22 / 4.23）：`K(t) = 1` 时终止；或 `t = T_max = 100` 时终止。

总回报（式 4.24）：`R_i = Σ_{t=1}^{T_end} r_i(t), ∀ i ∈ I.`

**奖励 B：外部仿真用例（逐字抄录）**

式 (4.38)：
> r_i(t) = α · D_i(t) + β · P^pickup_i(t) + γ · P^place_i(t) + δ · Σ_{s∈S} W_s(t) + ε · Col_i(t).

常数（原文）：
- "α = 1: reward weight for delivering a product"
- "β = 0.1: reward weight for picking up an item"
- "γ = 0.1: reward weight for placing an item into the correct processing unit"
- "δ = −0.01: penalty weight per station not working"
- "ε = −1: penalty weight for collisions"

五个分量（原文）：
1. "**Delivering product (α · D_i(t))**: agents receive a reward of +1 for delivering a finished product to the sink."
2. "**Picking up item (β · P^pickup_i(t))**: agents receive a reward of +0.1 for picking up an item from a source."
3. "**Placing in unit (γ · P^place_i(t))**: agents receive a reward of +0.1 for placing an item in the correct processing unit."
4. "**Stations not working penalty (δ · Σ_{s∈S} W_s(t))**: agents incur a penalty of −0.01 for each station that is not working at time t ∈ T."
5. "**Collision penalty (ε · Col_i(t))**: agents incur a penalty of −1 if they are involved in a collision at time t ∈ T. The penalty applies to the oncoming AGV."

总回报（式 4.39）：`R_i = Σ_{t=0}^{T_max} r_i(t), ∀ i.` episode 在达到最大时间步 T_max 时结束。

**对比基线（传统 MAPF 算法，第 3.4 节）**
- **MA-A\***（Multi-agent A* variant）：分散式（decentralized execution）范式，agent 立即开始移动并在遇到他人时动态调整路径；不预计算完整路径，与 RL 的局部决策范式一致。使用优先队列（open set）+ visited 集合，每一步评估所有 agent 的所有可能移动（假设同时移动），检查节点冲突与边冲突；代价函数整合路径长度（g cost）与启发式估计（f cost）
- **CBS**（Conflict-based Search）：集中式（centralized planning）范式，假设掌握全部已规划路径，通过引入约束并重算受影响路径来消解冲突。两层结构：高位 CBS（识别并消解冲突，冲突分为顶点冲突与边冲突；每个子节点引入一条约束——禁止某 agent 在某时刻占据某格，或禁止某条边转移）+ 低位 A*（在约束下为单个 agent 计算路径，使用曼哈顿距离启发式）
- 选这两个的理由（原文）："These two algorithms are chosen because they represent opposite ends of the planning spectrum: reactive, decentralized decision-making (MA-A*) versus preplanned, centralized conflict resolution (CBS)."

**程序模型（RQ 1.2 的答案）**：把 RL 集成到既有物流规划流程，分四阶段——RL 问题建模（对应需求分析与系统建模）、模型选择（对应设计与仿真）、算法选择（对应优化）、部署（对应执行与监控，含实时监控与持续学习）。同时把"韧性"作为第四物流目标，其子层级为 Availability、Recovery、Robustness、Failure Management（Figure 3.6 把 Gudehus 的物流目标三角模型扩展为四边模型）。

### 4. 实验

**硬件与软件环境**："All RL experiments are conducted on a system equipped with an Intel Core i7-10700K central processing unit (CPU) with 3.80 GHz, utilizing 11 CPUs. Of these, 10 CPUs are allocated to the learning environment, each managing 2 environments, while one CPU is reserved for the execution of the algorithm. A single NVIDIA GeForce RTX 3080 GPU is available for accelerating computations where applicable. The system provides 16 GB of RAM."

外部仿真环境："The simulation model uses Tecnomatix Plant Simulation version 2201"，脚本语言 SimTalk 2.0，通过 COM 接口的 "StepPython" 函数与 Python 脚本同步（每 1 秒仿真时间暂停一次，Python 脚本监听事件控制器并在每次暂停后恢复仿真，从而定义 RL 循环中的单个时间步）；Python 侧使用 **Ray 2.35.0 + RLlib**，自定义环境遵循 Gymnasium 标准。

**超参数搜索**
- 网格参考模型：PPO 用随机搜索。**每个变体分配固定算力预算：参考模型 1.1 为 1 小时，参考模型 2.1 与 3.1 为 4 小时**；只对基本变体（1.1、2.1、3.1）搜索，结果外推到其他变体（如 1.1 的参数用于 1.3）；每个算法 9 组超参搜索实验。搜索范围（Table 4.3）：训练 batch size 4000（固定）、minibatch 4000（固定）、γ = 0.99（固定）、GAE λ = 0.95（固定）、KL 初始系数 0.2（固定）、KL 目标值 0.01（固定）、epochs 5–14、clipping ε ∈ {0.05, 0.1, 0.2, 0.3}、学习率 α ∈ {0.0001, 0.0003, 0.0005, 0.001}、熵系数 ∈ {0, 0.001, 0.01}、隐藏层维度 ∈ {[32,32], [64,64], [256,256]}
- 外部仿真：**贝叶斯优化**，UCB 采集函数，探索参数 κ 与最小改进阈值 ξ。Run 1：225 样本 + 50 随机搜索样本，minibatch 512，epochs 20，clipping [0.1, 0.3]，lr [5e-6, 0.003]，KL 初始 [0.3, 1]，KL 目标 [0.003, 0.03]，γ [0.8, 0.9997]，GAE λ [0.9, 1]，值函数系数 [0.5, 1]，熵系数 [0, 0.01]，κ = 2.5，ξ = 0.0。Run 2：43 样本，lr 范围收窄为 [5e-6, 0.001]，熵系数 [0.001, 0.01]，κ = 0.5；初始点取 Run 1 中 100 个评测 episode 平均回报最高的超参组合（clipping 0.1749, lr 0.000179, KL init 0.7191, KL target 0.007212, γ 0.9462, λ 0.9156, VF coef 0.9331, entropy 0.009507）；使用 iteration stopper（500 trials）与 trial plateau stopper（100 次迭代内 episode 平均回报标准差低于 0.2 即停）

**三阶段实验设计**
- Phase 1–2（网格参考模型）：全因子式参数研究（PPO 超参、熵系数、折扣因子、网络规模）；在参考模型 1.1–3.1 上与 MA-A*、CBS 对比，指标为**成功率**与**时间步数**，用热力图分析智能体访问分布；原文说明指标取舍："We omit the classical path-length metric... the analysis focuses on success rate and timesteps because success rate is [more informative in deadlock-capable settings]"
- Phase 3（泛化与规模）：变化智能体数量、在未见环境上评测
- Phase 4（外部仿真）：1~4 个 agent，训练 2500 次迭代，每配置 3 次独立运行以评估方差；训练用 2 个 worker node（各 1 个 CPU），训练中并行跑独立评测 episode

**第三章（前置研究）的实验规模**
- 第一系列：252 组实验 = Prevention 63（Layouts 1,5,6 × AGV 数 3–9 × 巷道长 15/30/45 m）+ Avoidance 126（Layouts 1–6）+ Detection and recovery 63（Layouts 1,3,4）；每组仿真 8 小时 × 10 次运行
- 第二系列：144 组实验（巷道数 4/6/8/10，AGV 固定 7 台），每组 10 次观测
- 中断（disruption）实验（Table 3.4）：AGV 数 5–10（6 档）× 死锁处理策略 3 种 × 可用率 91%/95%/99%（3 档）× MTTR 5/30/120 min（3 档）；仿真 8 小时 × 10 次运行
- 仿真软件：Plant Simulation 15.1.1，方法用 SimTalk 2.0 编写

### 5. 核心数字（照抄原文）

**参考模型上的成功率（网格环境）**
- 仓库 2.1（block 布局）："MA-A* achieves a success rate of **100 %**, with PPO and IMPALA reaching **96 %** and **90 %**, respectively."
- 仓库 2.1（dead ends 布局）与 2.2（fishbone）："tested as a baseline and consistently achieve a success rate of **0 %** across all layouts"；"CBS improves on this, achieving a success rate of **33 %**"；"MA-A* achieves a success rate of **44 %**, while CBS improves to **80 %**, and PPO and IMPALA maintain high performance with success rates of [高位]"
- 生产物流 3.1："MA-A* achieves a success rate of **34 %**"；"CBS, with a success rate of **77 %**, performs significantly better than MA-A* in terms of [timesteps]"

**泛化**
- "Generalization of PPO and IMPALA on different layouts for 1,000 evaluation [episodes]" 对应原文数字："length of **34.84** compared to **75.89** in DTE and **100** in CTE"

**可扩展性（第 4.4 节结论，逐字）**
- "Scalability remains limited. When the number of agents rises from two to four, the mean reward falls by **18 %** and its standard deviation doubles, indicating that current methods struggle to coordinate in dense settings."
- "In the four-agent case, the mean reward stabilizes with limited improvement, and individual episodes achieve rewards as low as **−26** due to the increased likelihood of collisions."
- "With two agents, the steepest learning curve is observed, as agents learn to balance collision avoidance with productive throughput and the hyperparameter are tuned on this configuration."
- "An exception is the single-agent setting, which shows the highest variance. In this case, sparse rewards and limited exploration capabilities hinder stable policy learning, resulting in diverging outcomes across runs."

**外部仿真（贝叶斯优化）**
- 最优超参的泛化性数字："The best configuration achieves a mean reward of **−16.9**; the worst reaches **−26.7**. Many configurations lead to policies where agents remain stationary to avoid collisions, which triggers penalties for inactivity. This behavior results in mean rewards around **−20**."
- "A clipping parameter near **0.17**, KL divergence initialization above **0.7**, and a discount factor γ close to **0.95** are consistent among the best-performing configurations."
- "Better performance correlates with lower learning rates and reduced λ."
- 效果："Performance decreases with the number of agents, reflecting the higher coordination demands"；"more agents... leads to penalties from non-operational stations and lower mean rewards. The standard deviation of the mean reward increases with the number of agents."

**第三章（死锁处理策略的重要性）**
- "over an eight-hour simulation period, **hundreds of deadlocks** are recovered, with an average detour of approximately **20 meters** for each recovered deadlock, leading to significant cumulative detours."
- "For the prevention strategy, throughput significantly decreases once the number of AGVs exceeds approximately **5-7**."
- 中断实验："under conditions of high AGV availability with **99 %** and a low MTTR of **5 minutes**, the avoidance strategy maximized system throughput, achieving up to **120 units per hour with seven AGVs**. The prevention strategy lagged, unable to reach this throughput even with 10 AGVs."
- "The avoidance strategy consistently yielded the best throughput per hour across various disruption scenarios."
- "M¨uller et al. (2020) demonstrate that **no single strategy dominates** in deadlock handling. Instead, the effectiveness of a strategy hinges on the specific warehouse layout and operational parameters."
- "layouts with unidirectional storage aisles, which easily prevent deadlocks, perform worse compared to layouts with bidirectional aisles where deadlock avoidance and detection and recovery strategies are necessary"

**RQ 结论（原文判断）**
- RQ 3.1："The choice of the correct deadlock handling strategy is important and **answers the RQ 3.1 positively**."
- RQ 1.1："The analysis affirms also the general importance of deadlock handling in intralogistics systems, **addressing RQ 1.1 positively**."
- RQ 2.2（MARL 何时有效）："CTDE achieves the most consistent performance across different reference scenarios."；"PPO outperforms IMPALA across most scenarios, particularly when paired with CTDE."；"MARL's effectiveness diminishes in simpler configurations where deterministic algorithms can perform efficiently with lower computational overhead."
- RQ 3.2："CTDE is a highly effective approach" 但 "the question of whether CTDE is universally the 'best' approach remains open and context-dependent."
- RQ 3.1 补充："MARL can learn and apply typical deadlock handling strategies, such as avoidance or recovery"；"In the current configurations, MARL learns avoidance strategies in environments where agents have sufficient information and deterministic conditions, especially in CTE... In scenarios with limited information or higher uncertainty, MARL tend to adopt detection and recovery strategies."

### 6. 自认局限（原文逐字 + 中文翻译）

**a) 规模扩展受限（第 4.4 节）**
> "Scalability remains limited. When the number of agents rises from two to four, the mean reward falls by 18 % and its standard deviation doubles, indicating that current methods struggle to coordinate in dense settings. Future work should measure how training time and resource usage grow with agent count and how to handle larger AMR fleets with MARL."

中文翻译：可扩展性仍然有限。当智能体数量从 2 增加到 4 时，平均回报下降 18%，标准差翻倍，表明现有方法在密集场景中难以协调。未来工作应测量训练时间与资源占用如何随智能体数量增长，以及如何用 MARL 处理更大规模的 AMR 车队。

**b) 计算开销、泛化与实时性（讨论 5.1，RQ 2.2）**
> "Despite its strengths, MARL has limitations. Training MARL models can be computationally expensive, particularly in large-scale systems. Additionally, MARL policies may require fine-tuning to generalize effectively to new environments. While CTDE offers a balance between centralized coordination and decentralized execution, its reliance on centralized training poses challenges for real-time applications where communication delays or failures can occur."

中文翻译：尽管有优势，MARL 仍有局限。MARL 模型的训练计算开销可能很高，在大规模系统中尤其如此。此外，MARL 策略可能需要微调才能有效泛化到新环境。虽然 CTDE 在集中式协调与分散式执行之间取得平衡，但其对集中式训练的依赖给可能出现通信延迟或故障的实时应用带来挑战。

**c) 结论章识别的挑战**
> "Despite these contributions, the thesis identifies challenges. MARL solutions require computational resources and technical expertise. There are also further studies and improvements necessary to make MARL work in more complex scenarios with a higher number of agents."

中文翻译：尽管有上述贡献，本论文也识别出挑战。MARL 方案需要计算资源与技术专长。要让 MARL 在智能体数量更多的复杂场景中奏效，还需进一步研究与改进。

**d) 参考模型依赖网格（RQ 2.1 讨论）**
> "One potential weakness is their reliance on grid-based environments, which, while widely used in research, may oversimplify real-world systems. For instance, continuous-space environments with more complex movement dynamics might not be fully captured within the constraints of a grid. Another challenge is balancing realism with computational efficiency. Highly detailed scenarios might offer greater fidelity but could also impose significant computational costs, limiting their practical applicability in iterative testing processes."

中文翻译：一个潜在弱点是参考场景依赖网格化环境；网格虽在研究中被广泛使用，却可能过度简化真实系统。例如，具有更复杂运动动力学的连续空间环境可能无法在网格约束内被充分刻画。另一个挑战是平衡真实性与计算效率。高度细致的场景可能带来更高保真度，但也可能造成显著计算开销，限制其在迭代测试流程中的实际可用性。

**e) CTDE 是否普遍最优仍未决（RQ 3.2）**
> "Nevertheless, the question of whether CTDE is universally the 'best' approach remains open and context-dependent, as the effectiveness of any training and execution mode depends on the specific characteristics of the problem and the operational requirements."

中文翻译：然而，CTDE 是否普遍是"最优"方法这一问题仍然悬而未决，且取决于具体情境，因为任何训练与执行模式的有效性都取决于问题的具体特征与运营需求。

**f) CTE 的扩展性缺陷**
> "CTE performs consistently well in smaller settings with a limited number of agents but fails to scale effectively in larger and more stochastic environments."

中文翻译：CTE 在智能体数量有限的较小场景中表现稳定，但在更大、更随机的环境中无法有效扩展。

**g) 真实场景迁移仍是开放挑战**
> "while this thesis demonstrates effectiveness in controlled environments, transfer to real-world operations, where partial observability, delayed actuation, and stochastic disturbances are present, remains an open challenge."

中文翻译：虽然本论文在受控环境中证明了有效性，但迁移到存在部分可观测、执行延迟与随机扰动的真实运营场景仍是未解挑战。

**h) 死锁可被忽略的边界条件尚未刻画**
> "While the presented analysis in Section 3.1 provides qualitative guidance on when deadlocks must be considered, an exact characterization of the boundary conditions under which deadlocks can be ignored remains an open question."

中文翻译：虽然 3.1 节的分析对"何时必须考虑死锁"提供了定性指导，但对"死锁在何种边界条件下可以被忽略"的精确刻画仍是开放问题。

**i) MARL 在简单场景中优势消失**
> "MARL's effectiveness diminishes in simpler configurations where deterministic algorithms can perform efficiently with lower computational overhead. In structured environments with minimal overlap of agent paths and predictable paths, classical algorithms such as CBS or MA-A* achieve comparable or better results without the need for extensive training."

中文翻译：在确定性算法能以更低计算开销高效运行的较简单配置中，MARL 的有效性会减弱。在智能体路径重叠少、路径可预测的结构化环境中，CBS 或 MA-A* 等经典算法无需大量训练即可取得相当或更好的结果。

**j) 图论/环境层面的未来改进项（Chapter 6，逐字）**
> "Future research should refine MARL-based solutions to better reflect real-world scenarios. For example, improving simulation environments by using JAX or C-based frameworks can enhance computational efficiency and throughput. Adapting action and observation spaces to align with robotic driving behavior, such as continuous speed and turning, and incorporating obstacles that obstruct the line of sight would make the solutions more realistic. Exploring dynamic obstacles and adding unexpected failures of the AMRs would provide a robust test of MARL policies under more challenging conditions."

中文翻译：未来研究应改进基于 MARL 的方案以更好反映真实场景。例如，用 JAX 或基于 C 的框架改进仿真环境可提升计算效率与吞吐。把动作与观测空间调整为贴合机器人驾驶行为（如连续的速度与转向），并加入遮挡视线的障碍物，将使方案更真实。探索动态障碍物并加入 AMR 的意外故障，将能在更具挑战性的条件下对 MARL 策略做稳健测试。

### 7. 可复用

- **开源代码：有（两个仓库，arXiv Comments 与正文均给出）**
  - 参考模型 / 学习环境：`https://github.com/Nerozud/dl_reference_models`
  - 外部 Plant Simulation + RLlib 用例：`https://github.com/Nerozud/FTS_simpel`
    （正文 4.3.3 原文："The simulation model and source code used for this use case are available in a public repository maintained by the author: https://github.com/Nerozud/FTS_simpel."）
- **数据集**：无外部数据集。**7 个自建参考模型**（1.1、1.2、1.3、1.4、2.1/2.1b、2.2、3.1）本身就是可复现基准，遵循 Gymnasium 标准（多智能体用 RLlib `MultiAgentEnv`，单智能体用 `gym.Env`），可直接被 Stable-Baselines3 / RLlib 加载。
- **评测协议（可直接借用）**
  - 三阶段协议：① 固定算力预算的超参搜索（参考模型 1.1 给 1 小时、2.1/3.1 各给 4 小时）→ ② 与 MA-A*、CBS 对比成功率与时间步数（放弃传统 path-length 指标）→ ③ 变智能体数 + 未见布局测泛化与规模
  - 外部仿真侧：贝叶斯优化（UCB 采集函数，κ、ξ 参数化）做超参搜索，含 iteration stopper 与 trial plateau stopper；训练 2500 迭代 × 每配置 3 次独立运行以评估方差
  - 空白/基线选取原则：选 MA-A*（分散式反应式）与 CBS（集中式预规划）作为"规划谱系两端"的代表
- **公式**
  - 碰撞判定：(4.17)、(4.18)、(4.19)
  - 网格奖励：(4.20)、(4.21)、(4.24)（α=0.5, β=1, γ=−1；T_max = 100）
  - 外部仿真奖励：(4.38)、(4.39)（α=1, β=0.1, γ=0.1, δ=−0.01, ε=−1）
- **可复用的表格/框架**
  - **Table 4.2**（观测空间 / 动作空间 / 奖励 三栏对照，按 CTE vs CTDE&DTE 分支）与 **Table 4.11**（外部仿真用例的同款对照）——可直接作为同类工作的论文表述模板
  - **Table 5.1**：传统物流规划 vs RL 韧性规划在"死锁处理 / 规划目标 / 设计哲学 / 灵活性 / 可扩展性 / 仿真的用途"六个维度的对照
  - 把**韧性（resilience）**作为第四物流目标的四边模型（Performance / Quality / Costs / Resilience），韧性下分 Availability、Recovery、Robustness、Failure Management
  - 多载/多模式 RL 对比的三分类：CTE / CTDE / DTE（可作为 MAPF 类论文的标准对照设置）

---

## 卡片 5 — arXiv:2502.16079

### 1. 题录

| 项 | 内容 |
|---|---|
| 标题 | Together We Rise: Optimizing Real-Time Multi-Robot Task Allocation using Coordinated Heterogeneous Plays |
| 作者 | Aritra Pal、Anandsingh Chauhan、Mayank Baranwal（均为 TCS Research, Mumbai, India） |
| arXiv ID | arXiv:2502.16079v1 [cs.RO] |
| 年 | 2025（提交 22 Feb 2025） |
| 会议 | **AAMAS 2025（AAAI Track）**，第 24 届 International Conference on Autonomous Agents and Multiagent Systems，2025 年 5 月 19–23 日，美国 Detroit, Michigan，共 9 页 |
| arXiv Comments | "Accepted to AAMAS 2025 (AAAI Track)" |
| 是否已正式发表 | **是**。PDF 页脚逐字："Proc. of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025), Y. Vorobeychik, S. Das, A. Nowé (eds.), May 19 – 23, 2025, Detroit, Michigan, USA. © 2025 International Foundation for Autonomous Agents and Multiagent Systems (www.ifaamas.org)."，采用 CC BY 4.0 许可。arXiv 页面另有 arXiv 自颁 DOI `10.48550/arXiv.2502.16079`。 |

### 2. 问题设定

**问题类型**：**实时多机器人任务分配（multi-robot task allocation, MRTA）**，动态仓储环境；论文自述与 multi-agent pickup and delivery (MAPD) 问题相关，但聚焦**在线（online）**任务分配而非离线 MAPD。原文："our approach emphasizes learning-based methods for online task allocation"。

**资源**
- **机器人集合 R**：多台机器人，建模为 2D 双积分器系统（连续运动，非网格）
- **任务**：动态实时生成，每个任务有起始位置（origin）与结束位置（destination），以及到达时间（generation time / timestamp）
- **任务缓冲**：任务生成后立即存入主任务缓冲（buffer），采 FIFO；Planner 只能看到一个**有限长度的前瞻队列（look-ahead queue, LA）**
- **充电桩**：机器人 SOC 低于阈值时须前往最近的可用充电桩充电
- **Planner（任务选择 agent）+ Executor（机器人分配 agent）+ Navigator（LQR-APF 导航器）**三层结构

**目标**（原文摘要）："The objective is to minimize both the total travel distance of robots and delays in task completion, while also considering practical constraints such as battery management and collision avoidance."

**约束**
- **电池管理**：SOC 低于 30% 必须停靠充电；充电速率 = 放电速率的 16 倍
- **碰撞避免**：机器人间保持最小允许距离 d_min
- **机器人动力学**：双积分器模型，控制量直接作用于加速度，须满足执行器约束
- **任务时效**：任务不应长时间无人处理（由奖励第二项 TTGT 约束）
- **LA 队列长度固定**；当 LA 中的任务被分配后，从缓冲补入新任务；若无新任务可补，则**把 LA 中停留最久的任务复制一份**以维持队列长度（从而提高它被 Planner 选中的概率）；若 LA 与缓冲均空则等待新任务出现

**机器人分配的时间点特点**（原文）："the Executor does not wait for robots to become available before making robot allocations, as the state information includes time markers indicating when robots will be free." 即允许把任务预分配给当前仍忙碌的机器人。

### 3. 方法

**算法**：**MRTAgent** —— self-play 启发的**双层 RL 框架**，两个独立的 RL agent：
- **Planner（任务选择 agent）**：从 LA 队列中挑选任务
- **Executor（机器人分配 agent）**：为被选中的任务分配机器人
- 二者均用 **PPO**（Proximal Policy Optimization）
- 导航用 **LQR（线性二次调节器）+ APF（人工势场）**，非学习式

**自博弈训练机制（原文）**："Both the agents are trained using a self-play inspired strategy, where one agent is actively trained while the other operates in evaluation mode, alternating every 40 episodes." 训练细节："Initially, the planner undergoes training for 40 episodes, while the executor remains in evaluation mode. After every 40 episodes, the roles of the planner and executor are reversed... This cycle is repeated 24 times, leading to a total of 960 training episodes."

**网络结构（三段式，原文 Figure 2）**
- **第一段 · 特征提取**
  - 机器人属性嵌入：4 层线性层，维度 `[4, 16, 16, 1]`
  - 任务属性嵌入：4 层线性层，维度 `[6, 16, 16, 1]`
  - 公式：`E^P_i = W^P2 * ReLU(W^P1 * F^P_i)`；`E'^R_j = W'^R2 * ReLU(W'^R1 * F^R_j)`
- **第二段 · 拼接与注意力式汇聚**
  - 机器人与任务的嵌入拼接后，通过一层线性层：48 输入神经元 → 8 输出神经元，ReLU 激活
  - 公式：`a^P_i = Sigmoid(W^P4 * Tanh(W^P3 * E^P_i))`；`a'^R_j = Sigmoid(W'^R4 * Tanh(W'^R3 * E'^R_j))`
  - 策略表示：
    - `π_planner = ( Σ_j E^R_j * a^R_j, Σ_i E^P_i * a^P_i, E^P_i )`
    - `π_executor = ( Σ_j E'^R_j * a'^R_j, Σ_i E'^P_i * a'^P_i, E'^R_j )`
  - （该结构原文注明 "inspired from [1]"，即 Agrawal et al. 的 RTAW）
- **第三段 · 输出层**
  - 线性层：8 输入神经元 → 1 输出神经元
  - `TaskAllocated = Categorical(W2 * ReLU(W1 * π_planner))`
  - `RobotAllocated = Categorical(W'2 * ReLU(W'1 * π_executor))`

**状态定义（Planner 与 Executor 共享同一状态构成）**

状态 `s_t ∈ S` 在时刻 t 包含：
- 任务相关（来自 LA 队列 P，每任务 **6 维**）：
  - (a) 任务起点坐标 `{o_i}`
  - (b) 任务终点坐标 `{d_i}`
  - (c) 任务起点到终点的欧氏距离 `{k_i}`
  - (d) 任务在 LA 队列中出现的时间戳 `{l_i}`
- 机器人相关（集合 R，每机器人 **4 维**）：
  - (e) 机器人坐标 `{p_j}`
  - (f) 机器人可用性及预计完成当前任务的时间 `{r_j}`
  - (g) 机器人电量百分比 `{c_j}`

**动作空间设计**：**分类（离散）动作空间**，由 Categorical 分布采样，分两级
- **Planner 的动作**：从 LA 队列中选出 1 个任务（`TaskAllocated`）
- **Executor 的动作**：为选中的任务选出 1 台机器人（`RobotAllocated`）
- 环境实际执行的动作 = "被选中的任务 + 其对应机器人"这一对。原文："Thus, the actions executed in the environment consist of the selected task and its corresponding robot pair."
- 第三级不是 RL 动作：**Navigation** 由 LQR-APF 控制器直接生成连续控制量

**奖励（逐字抄公式）**

原文："The step reward attributed to a task-action pair encompasses two distinct components. The initial component is calculated based on the time it takes for the robot to travel from its current position to the task's starting point, termed **travel time to origin (TRTO)**. The second component is the time gap between task arrival in the LA and the robot's initiation of execution, denoted as the **total time gap for the task (TTGT)**."

符号：`(x_oi, y_oi)` 与 `(x_di, y_di)` 为第 i 个任务的起点与终点坐标；`(x_rj, y_rj)` 为机器人 j 的当前位置（随机器人空闲、执行任务或充电而变化）；`t_stamp_i` 为任务 i 进入 LA 的时刻；`t_exec_i` 为任务 i 开始执行的时刻；`allotT` 为被选中任务的索引，`selR` 为被分配到的机器人；`d[(x_a,y_a), (x_b,y_b)]` 为两点距离。

式 (1)：
```
R_step = − d[(x_{r,selR}, y_{r,selR}), (x_{o,allotT}, y_{o,allotT})]
         − α * (t_{exec,allotT} − t_{stamp,allotT})
```
原文对 α 的说明："The coefficient α represents positive constant. The first term in (1) corresponds to TRTO, while the second term is associated with TTGT. This reflects the principle that tasks should not remain unattended for too long."

（注意：奖励为两项惩罚之和，无正项；因此 Cost = −R_step 越低越好，论文表格标题即为 "Cost (×10³) evaluation ... (the lower the better)"。）

**PPO 训练细节（Algorithm 1 逐字）**
```
Initialize Planner & Executor policy parameters θ^P_0, θ^E_0 and value function parameters φ^P_0, φ^E_0.
for itr = 0,1,2,...K1 do
  for episodes k = 0,1,2,...K2 do
    Step t=0
    State s_t := {(o_i,d_i,k_i,l_i) ∀i∈P, (p_j,r_j,c_j) ∀j∈R}
    Each robot j∈R is executing a task (o_j,d_j)
    while True do
      Update p_j ∀j∈R using LQR with APF
      if p_j == d_j then
        if c_j < threshold for some j∈R then
          Robot j navigates to nearest available charging dock for recharging.
        end if
        if itr is even then
          run Planner policy π^P_k = π(θ^P_k) in the environment to select a task i∈P.
          Executor policy takes state s_t as input and allots a robot j∈R.
        else
          Planner policy takes state s_t as input and selects a task i∈P.
          run Executor policy π^E_k = π(θ^E_k) to allocate a robot j∈R.
        end if
      end if
      t ← t+1
    end while
    if itr is even then
      Collect set of trajectories D^P_k
      Compute rewards-to-go R̂^P_t
      Compute advantage estimates Â^P_t based on the current value function V_{φ^P_k}
      Update policy by maximizing PPO-Clip obj.:
        θ^P_{k+1} = arg max_θ (1/|D^P_k|T) Σ_{τ∈D^P_k} Σ_{t=0}^{T}
                    min( (π_θ(a_t|s_t)/π_{θ^P_k}(a_t|s_t)) A^{π_{θ^P_k}}(s_t,a_t), g(ε, A^{π_{θ^P_k}}(s_t,a_t)) )
        via stochastic gradient ascent with Adam;
      Fit value function by regression on MSE:
        φ^P_{k+1} = arg min_φ (1/|D^P_k|T) Σ_{τ∈D^P_k} Σ_{t=0}^{T} ( V_φ(s_t) − R̂^P_t )²
        via gradient descent
    else
      Update θ^E_k & φ^E_k as above
    end if
  end for
end for
```

**导航：LQR + 人工势场 APF（逐字抄公式）**
- 机器人 i 的状态 `p_i(t) = (x_i(t), y_i(t))`（位置）与 `v_i(t)`（速度）；运动方程：`ṗ_i(t) = v_i(t)`，`ṿ_i(t) = u_i(t)`，`u_i(t) = (u_ix(t), u_iy(t))` 为控制输入（直接影响加速度）。完整状态 `z_i(t) = (p_i(t), v_i(t))`
- 多机器人系统（N 台）的增广动力学：`ż(t) = A_multi z(t) + B_multi u(t)`，其中
  `z(t) = [p_1(t); v_1(t); …; p_N(t); v_N(t)]`，`u(t) = [u_1(t); …; u_N(t)]`
  `A_multi = I_N ⊗ [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]`，`B_multi = I_N ⊗ [0 0; 0 0; 1 0; 0 1]`
  （`I_N` 为 N 阶单位阵，⊗ 为 Kronecker 积）
- LQR 代价函数：`J = ∫_0^∞ ( z(t)^T Q z(t) + u(t)^T R u(t) ) dt`，Q、R 为权重矩阵
- 最优控制：`u(t) = −K(z(t) − z_des)`，`K = R^{−1} B_multi^T P`，z_des 为目标位置与速度
- P 为连续时间代数 Riccati 方程（CARE）的解：`A_multi^T P + P A_multi − P B_multi R^{−1} B_multi^T P + Q = 0`
- APF 斥力势（机器人 i 与 j 之间）：
  `U_rep,ij(z_i,z_j) = ½ k_rep (1/d_ij − 1/d_min)²  if d_ij < d_min;  0  if d_ij ≥ d_min`
  其中 k_rep 为正的常数，`d_ij = ‖p_i − p_j‖` 为欧氏距离，d_min 为机器人间最小允许距离
- 最终控制输入：`u_i(t) = −K_i(z_i(t) − z_des,i) − ∇_{p_i} U_i`

**训练超参数（原文）**："The planner and executor agents are implemented using the PyTorch library in Python 3.8, with an Adam optimizer, a discount factor of 0.99, a lambda value of 0.95, a learning rate of 0.0003, an entropy coefficient of 0.001, a value function coefficient of 0.0002, and a batch size of 32. The policy networks for both the planner and executor are trained using the cross-entropy loss function, while the value networks are fine-tuned using the mean squared error loss metric." 训练在 **PFRL 框架**中执行。

### 4. 实验

**实例与规模（合成数据生成协议）**
- 原文说明数据来源："Due to the absence of publicly accessible real-world datasets for similar problem scenarios, synthetic data has been employed to evaluate our MRTAgent framework."
- 空间：方形 2D **连续空间** `[0, 64] × [0, 64]`；任务起点、终点与机器人位置均限制在该范围内
- 每个 episode 跨越 τ 个时间单位
- 任务集中度按一天中的高峰时段变化，用**不同均值与标准差的高斯分布**生成
- 两种配置：**正态分布任务到达时间** 与 **均匀分布任务到达时间**。每 episode 500 个任务。例如正态情形下任务生成时间服从 **N(600, 50)**（600 为平均任务生成时间）
- **训练设定**：10 台机器人；**LA 窗口长度固定为 5**；充电阈值 30%；充电速率 = 放电速率的 16 倍
- **训练规模**：960 个训练 episode，每 episode **505 个任务**，10 台机器人，LA = 5；用 **4 个不同随机种子**取平均学习曲线

**基线（2 个，作者自建，因无既有工作可对比）**
- 原文："To the best of our knowledge, no existing work in the literature concurrently addresses multiple aspects of MRTA simultaneously. In light of the absence of established approaches, we propose two suitable baselines."
- **BFO（Brute-force optimal）**：对 LA 内所有任务-机器人对做穷举，用标准欧氏距离计算执行所需时长，选择使时长最小的任务-机器人对。原文："While brute-force optimal approach represents a locally optimal solution, the exhaustive evaluation significantly amplifies the run-time, posing practical challenges." 并称 "BFO is an optimal task allocation and assignment approach given the current state of the LA received in an online fashion."
- **FIFO**：双层决策——先选最早进入 LA 队列的任务，再分配能最早完成它的机器人。原文："Due to its simplicity, the FIFO approach requires the least execution time among all the considered approaches."

**指标**：**Cost (×10³)**（越低越好），并拆分为两个分量：
- **Avg TRTO**（travel time to origin，机器人当前位置到任务起点的行驶时间）
- **Avg TTGT**（total time gap for the task，任务进入 LA 到机器人开始执行的时间差）

**评测维度（泛化性）**
- **分布漂移**：用高斯数据训练的模型在 (a) 同分布的 N(600, 50) 与 (b) 完全不同的 U(0, 1000) 上评测；用均匀数据训练的模型在 (a) 同分布 U(0, 1000) 与 (b) 完全不同的 N(600, 50) 上评测
- **机器人数量变化**：训练用 10 台；另**重训 Executor** 到 30 台（Planner 保持 evaluation 模式）；再把训练好的 30 台配置用于只有 25 台可用的场景（通过给多出的 5 台机器人的 `r_j` 赋大值，使其不被选中，**无需重训**），以模拟机器人故障
- **任务数量变化**：训练用 505 任务/episode，测试用 **2005** 任务/episode，不重训

### 5. 核心数字（照抄原文）

**Table 1：Cost（×10³），LA = 5, 10 robots, 505 tasks/episode（低者优）**

> 原表结构说明：列为两组评测数据分布（"Uniform distribution data" / "Gaussian distribution data"），每组下含 MRTAgent / BFO / FIFO 三列；行按"Similar / Totally Different"（评测数据是否与训练分布一致）分两个区块，每区块 5 行。PDF 转文本后表头与行标错位，下表按上述结构还原；"Similar / Totally Different" 与"训练分布 ↔ 评测分布"的具体对应关系（例如 Uniform 列在 Totally Different 区块下对应"用高斯训练、在均匀数据上评测"还是反之）从抽出的文本中无法唯一确定，引用时请以原文 Table 1 为准。

| 测试数据分布 | 与训练分布关系 | MRTAgent | BFO | FIFO |
|---|---|---|---|---|
| Uniform | Similar（同分布） | **13.49 ± 0.44** | 16.03 | 16.69 |
| Uniform | Similar（第 2 行） | 14.85 ± 0.15 | 17.72 | 18.35 |
| Uniform | Similar（第 3 行） | 13.72 ± 0.16 | 15.75 | 16.49 |
| Uniform | Similar（第 4 行） | 14.83 ± 0.26 | 16.97 | 17.21 |
| Uniform | Similar（第 5 行） | 14.47 ± 0.18 | 15.95 | 16.40 |
| **Uniform** | **Totally Different** | **18.89 ± 0.12** | 19.06 | 19.43 |
| Uniform | Totally Different（第 2 行） | 19.50 ± 0.67 | 20.15 | 20.56 |
| Uniform | Totally Different（第 3 行） | 18.88 ± 0.13 | 19.04 | 19.73 |
| Uniform | Totally Different（第 4 行） | 18.77 ± 0.24 | 19.58 | 19.65 |
| Uniform | Totally Different（第 5 行） | 18.53 ± 0.42 | 20.43 | 21.03 |
| Gaussian | Similar（同分布） | **18.23 ± 0.47** | 18.60 | 19.08 |
| Gaussian | Similar（第 2 行） | 18.66 ± 0.06 | 19.06 | 19.43 |
| Gaussian | Similar（第 3 行） | 19.10 ± 0.01 | 19.77 | 20.41 |
| Gaussian | Similar（第 4 行） | 18.40 ± 0.01 | 19.83 | 20.67 |
| Gaussian | Similar（第 5 行） | 18.67 ± 0.09 | 19.04 | 19.73 |
| **Gaussian** | **Totally Different** | **15.84 ± 0.56** | 17.72 | 18.35 |
| Gaussian | Totally Different（第 2 行） | 14.84 ± 0.44 | 17.25 | 17.83 |
| Gaussian | Totally Different（第 3 行） | 13.98 ± 0.43 | 15.75 | 16.49 |
| Gaussian | Totally Different（第 4 行） | 14.37 ± 0.38 | 16.92 | 17.31 |
| Gaussian | Totally Different（第 5 行） | 14.82 ± 0.38 | 15.95 | 16.40 |

（说明：原表未标注行号；上述 5 行即区块内的 5 次独立评测运行。）

**Table 2：Cost（×10³）分量对比（Gaussian 分布数据，LA = 5, 10 robots, 505 tasks/episode）**

| 行 | Avg TRTO MRTAgent | BFO | FIFO | Avg TTGT MRTAgent | BFO | FIFO |
|---|---|---|---|---|---|---|
| 1 | **8.44** | 8.48 | 8.63 | **11.38** | 11.68 | 11.93 |
| 2 | **8.03** | 8.59 | 8.93 | **11.09** | 11.18 | 11.48 |
| 3 | **7.59** | 8.62 | 9.11 | **10.79** | 11.20 | 11.56 |
| 4 | **7.70** | 8.08 | 8.30 | **10.90** | 10.96 | 11.43 |
| 5 | **8.20** | 8.49 | 9.22 | **10.70** | 10.81 | 11.25 |

原文解释："MRTAgent achieves value, in the TRTO component lesser than the brute-force optimal approach signifying its ability to minimize travel distance for the tasks in the look-ahead queue, as well as in the TTGT component because the brute-force optimal approach does not prioritize minimizing delays for tasks already in the look-ahead."

**Table 3：Cost（×10³），Gaussian 分布数据，LA = 5，**30 robots**，505 tasks/episode**

| 行 | MRTAgent | BFO | FIFO |
|---|---|---|---|
| 1 | **8.41** | 8.64 | 8.81 |
| 2 | **8.93** | 9.27 | 9.63 |
| 3 | **8.19** | 8.34 | 9.05 |
| 4 | **7.49** | 7.63 | 8.53 |
| 5 | **8.19** | 8.43 | 9.61 |

**Table 4：Cost（×10³），Gaussian 分布数据，LA = 5，**25 robots**（用 30 台训练、屏蔽 5 台，不重训），505 tasks/episode**

| 行 | MRTAgent | BFO | FIFO |
|---|---|---|---|
| 1 | **9.25** | 9.56 | 9.76 |
| 2 | **10.05** | 10.37 | 10.44 |
| 3 | **9.34** | 9.57 | 9.92 |
| 4 | **8.95** | 9.42 | 9.69 |
| 5 | **8.70** | 9.13 | 9.32 |

**Table 5：Cost（×10³），Gaussian 分布数据，LA = 5，10 robots，**2005 tasks/episode**（任务数增至 4 倍）**

| 行 | MRTAgent | BFO | FIFO |
|---|---|---|---|
| 1 | **223.44** | 237.2 | 279.38 |
| 2 | **215.81** | 229.54 | 274.59 |
| 3 | **218.44** | 230.08 | 277.88 |
| 4 | **217.31** | 232.45 | 276.56 |
| 5 | **214.67** | 227.45 | 274.31 |

**文字结论数字（逐字）**
- "The results clearly demonstrate MRTAgent's consistent outperformance over the brute-force optimal and FIFO-based methods across almost all test scenarios."
- "The MRTAgent framework outperforms the baselines in all instances, and for a completely different dataset, it performs significantly better compared to the baselines."
- "MRTAgent consistently outperforms the baseline methods in all scenarios."（30 机器人）
- "MRTAgent consistently outperforms the baselines across a variable number of tasks without requiring retraining."（2005 任务）
- "As expected, performing the same tasks with more robots incurs lower costs, as the TRTO and TTGT components reduce significantly."
- 为何能超过 BFO（原文解释）："The reason why MRTAgent is able to outperform it is due to the fact that MRTAgent exploits the underlying distribution defining task generation to plan for tasks to appear in future despite it having access to the same causal information as the BFO."

**训练规模数字**
- "This cycle is repeated 24 times, leading to a total of 960 training episodes."
- "Each episode consists of 505 tasks, with 10 robots in the environment and a LA length of 5."
- "This average is derived from four independent runs with different random seeds."
- 充电："The robots' charging threshold is set at 30%, meaning robots with a SOC below 30% must dock for recharging before resuming task execution. The steady charging rate is calibrated to be 16× the discharging rate."

### 6. 自认局限（原文逐字 + 中文翻译）

论文在 Conclusion and Future Work 中以编号列表形式列出 4 条：

**a) 单任务假设**
> "1. Single-Task Assumption: The algorithm currently assumes that robots are engaged in one task at a time, potentially limiting its applicability in scenarios where multitasking is prevalent."

中文翻译：1. 单任务假设：算法目前假设机器人一次只执行一个任务，这可能限制其在多任务普遍存在的场景中的适用性。

**b) 突发故障无应对**
> "2. No Contingency for Sudden Breakdowns: The model does not account for the sudden breakdown of a robot during operation. Once a task is allocated, our algorithm lacks the capability to reconsider or revert the decision in the event of an unforeseen robot malfunction."

中文翻译：2. 突发故障无应对：模型未考虑机器人在运行中突然故障。任务一旦分配，若发生未预料的机器人故障，我们的算法缺乏重新考虑或撤回该决策的能力。

**c) 同构机器人假设**
> "3. Homogeneous Robots: We have made the simplifying assumption that all robots in the system are homogeneous, sharing identical values of characteristics such as velocity, acceleration, and load capacity. This assumption may not reflect the diversity present in real-world robot fleets."

中文翻译：3. 同构机器人：我们做了简化假设，即系统中所有机器人是同构的，速度、加速度与负载能力等特性取值完全相同。该假设可能无法反映真实机器人车队中的多样性。

**d) 装卸时间可忽略假设**
> "4. Negligible Load/Unload Time Assumption: The algorithm assumes negligible time for loading/unloading operations after a robot reaches the task origin/destination. In reality, this may not hold true, and accounting for realistic loading and unloading times is a consideration for future enhancements."

中文翻译：4. 装卸时间可忽略假设：算法假设机器人到达任务起点/终点后的装卸操作耗时可以忽略。在现实中这未必成立，把真实的装卸时间纳入考虑是未来改进的一个方向。

**e) 数据层面（实验节）**
> "Due to the absence of publicly accessible real-world datasets for similar problem scenarios, synthetic data has been employed to evaluate our MRTAgent framework."

中文翻译：由于缺乏可公开获取的、类似问题场景的真实数据集，我们采用合成数据来评估 MRTAgent 框架。

**f) 未来工作（原文）**
> "Future work will focus on enabling robots to handle multiple tasks simultaneously, incorporating load/unload times, integrating heterogeneous robots, and implementing learning-based navigation, all of which will enhance the algorithm's effectiveness and applicability in diverse, practical scenarios."

中文翻译：未来工作将聚焦于让机器人能同时处理多个任务、纳入装卸时间、集成异构机器人以及实现基于学习的导航，这些都将提升算法在多样实际场景中的有效性与适用性。

（另，基线自建本身含一层说明："To the best of our knowledge, no existing work in the literature concurrently addresses multiple aspects of MRTA simultaneously. In light of the absence of established approaches, we propose two suitable baselines." —— 即无可直接对标的既有方法。）

### 7. 可复用

- **开源代码**：**无**。论文未提及代码仓库 URL（AAMAS 2025 正式论文，9 页）。
- **数据集**：无公开数据集。但**合成数据生成协议完整可复现**：
  - 2D 连续方形空间 `[0, 64] × [0, 64]`，任务起点、终点与机器人位置均在该范围内
  - 任务到达时间两种分布：高斯 `N(600, 50)` 与均匀 `U(0, 1000)`
  - 任务集中度用高斯分布的均值/标准差变化模拟一天中的高峰时段
  - 每 episode 任务数：505（训练与主评测）、2005（任务数泛化测试）
  - 机器人数据：10（训练）/ 25 / 30
- **评测协议（可直接借用）**
  - 主指标 **Cost (×10³)**，并**拆分为 Avg TRTO 与 Avg TTGT 两个分量分别对比**（Table 2），以说明增益来自哪一目标
  - **三轴泛化测试**：① 分布漂移（同分布 vs 完全不同的另一分布，双向验证）；② 机器人数量变化（10 / 30 / 用 30 训练但仅 25 可用，借"给多余机器人的 `r_j` 赋大值"屏蔽而不重训）；③ 任务数量变化（505 → 2005，不重训）
  - 学习曲线用**多个随机种子（4 个）取平均**
  - 基线选择：当文献无同设定工作时，用 **BFO（当前 LA 下的穷举最优，作为上界参照）+ FIFO（最低计算开销，作为下界参照）** 构成对照
- **公式**
  - 奖励式 (1)：`R_step = − d[robot→origin] − α (t_exec − t_stamp)`（TRTO + TTGT 双惩罚）
  - PPO-Clip 目标与 MSE 值函数回归（Algorithm 1 内，逐字可抄）
  - LQR：代价 `J = ∫(z^T Q z + u^T R u)dt`；控制律 `u(t) = −K(z(t)−z_des)`；增益 `K = R^{−1}B^T P`；CARE 方程；多机器人增广矩阵 `A_multi = I_N ⊗ …`、`B_multi = I_N ⊗ …`
  - APF 斥力势 `U_rep,ij = ½k_rep(1/d_ij − 1/d_min)²`（d_ij < d_min 时）
  - 最终控制律 `u_i(t) = −K_i(z_i(t) − z_des,i) − ∇_{p_i}U_i`
- **可复用架构设计**
  - **双层解耦（Planner / Executor）**：使 Executor 可单独升级或重训而不动 Planner；原文："The separation of Tasks (Planner) and Robot (Executor) agents in our framework enables scalability with varying task and robot counts. This setup also allows upgrading the Executor without re-training the entire system."
  - **用 `r_j` 赋大值来屏蔽故障/不可用机器人**，实现无需重训的机队规模适配
  - **自博弈轮换训练**（每 40 episode 交替训练一个 agent、另一个保持 evaluation）以协调两个异质 agent
  - **LA 队列的复制补位机制**：当缓冲无新任务时，把 LA 中停留最久的任务复制一份以维持队列长度
  - 若同一任务被分配给仍忙碌的机器人，其"预计完成当前任务时间 `r_j`"使 Executor 无需等待机器人空闲即可决策

---

## 附：五篇横向速查（只列事实）

| 项 | ① 2409.11820 家具JSSP | ② 2409.18742 HRPEO | ③ 2205.03294 模块化VM | ④ 2511.07071 博士论文 | ⑤ 2502.16079 MRTAgent |
|---|---|---|---|---|---|
| 方法族 | DRL（概念，未实现） | 进化算法（非 RL） | DQN / DDQN + PER | MARL（PPO / IMPALA） | 双层 PPO + LQR-APF |
| 问题 | 扩展 JSSP（换型+批量+运输+缓冲+交期） | FJSPMA（多载 AGV） | 模块化生产车辆管理（JSSP 类） | deadlock-capable MAPF | 实时 MRTA（在线） |
| 动作空间 | 单一离散（机器→工件） | 非 RL：三层编码 + 运输任务列表 | 离散（可选件站集合 + do-nothing） | 网格：Discrete(5) / MultiDiscrete([5]*n)；外部仿真：Discrete(3) | 两层 Categorical（任务 × 机器人） |
| 状态 | 3×m 机器 / 2×n 工件 / b 维缓冲 | 三层编码 + ±运输任务列表 | 车辆状态 + 单元状态 + 工件三特征（含冗余） | CTE 全网格；CTDE/DTE 部分网格+自身位置与目标 | 任务 6 维 + 机器人 4 维 |
| 奖励 | 无公式（仅定性） | 无（进化算法用适应度=makespan） | Score 差分，式(2)，K=4000 | 式(4.20)/(4.38) | 式(1)，TRTO + α·TTGT |
| 实验 | **无**（仅 mock-up 算例） | 38 实例，6 基线，ARPD，Friedman | 7 配置，3 静态基线，12h 吞吐 | 7 参考模型 + 外部仿真，MA-A*/CBS 基线 | 合成数据，BFO/FIFO 基线 |
| 开源 | 无 | 无 | 无 | **有（2 个 GitHub 仓库）** | 无 |
| 正式发表 | 否（arXiv） | 否（arXiv） | 否（arXiv；用 Springer 模板） | 否（arXiv；学位论文已答辩） | **是（AAMAS 2025, AAAI Track）** |

---

*生成工具：Python 3.12 + PyMuPDF 1.28.0；中间文本文件位于 `C:\Users\evari\AppData\Local\Temp\g6_cards\`，用毕即删。*

