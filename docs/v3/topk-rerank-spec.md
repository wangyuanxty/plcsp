# top-K 候选重排 —— 设计 spec（v3）

> **状态**：草案，待实现（2026-10-18）。
> **范围**：只改 `plcsp_v3/` 的下层动作空间与训练接口。
> **不改**：`env/des.py` · `env/rule_views.py` · `env/snapshot.py` 的动力学与回调契约。
> **读数依据**：`repro/v3_tie_frequency_probe.py`、`repro/v3_topk_coverage_probe.py`。

---

## 0. 一句话

**MA 仍选规则；规则输出候选集 top-K；PA/LA/RA 三个打分头在候选集内排序。**
规则超参数（PA 的 `λ`、LA 的 `λ`、RA 的 `A`）全部退出策略路径。

---

## 1. 为什么改

| 探针 | 读数 | 含义 |
|---|---|---|
| 平局频率 | PA `EAJ` 平局占 **44–87%**，平均平局集 **2.2–3.6**；LA `MINCOST` **31–41%**，平均 **1.5–1.7** | 规则筛完仍有东西可学 |
| top-K 覆盖 | PA `K=16` 覆盖 **100%**；LA `K=6` 覆盖 **100%** | 16 / 6 足以覆盖单点最优候选 |

旧架构让下层出 1–3 个连续超参数。一个标量很难表达“在 16 个候选里怎么排”。
新架构把“筛”和“排”分开：

```
MA ──选规则──► 规则 ──筛出 top-K──► 打分头 ──在 K 内排序──► 动作
```

---

## 2. top-K 的构造

### 2.1 常量

| 常量 | 值 | 位置 |
|---|---|---|
| `K_PA` | **16** | `hrl/candidates.py` |
| `K_LA` | **6** | `hrl/candidates.py` |
| `TOPK_SEED` | **6000** | `hrl/candidates.py`（独立子种子） |
| `RULE_EMB_DIM` | **8** | `hrl/agents.py` |
| `MIN_Z_NORM_K` | **3** | `hrl/agents.py` |

**严格 K**：返回的候选数 `K_eff = min(K, 合法候选数)`，**永不超过 K**。

### 2.2 PA 候选

输入：`view: PAView`、`pool`、复合规则下标 `r`、`K=16`、`rng`。

1. 拆规则：`jr = PA_JOB_RULES[r // 3]`，`mr = PA_MACHINE_RULES[r % 3]`。
2. 枚举全部 `(工件, 机台)` 对：

```python
js = pa_job_score(jr, view, j, (0.0, 0.0))
ms = pa_machine_score(mr, view, (m, t), (0.0,))
row = (js, ms, pi, mi, j, oi, m)
```

3. 按 `(js, ms)` 升序分组。`(js, ms)` 完全相等的行属于**同一个平局组**。
4. 组内按 `(pi, mi)` 排序（只用于确定性，不代表偏好）。
5. 取前 `K` 行。若第 `K-1` 行与第 `K` 行**同组**（平局组跨过截断线）：
   - 该组已有 `first` 行在 K 内；
   - 还差 `need = K - first` 行；
   - 从该组剩余成员中**均匀无放回抽 `need` 行**，用 `rng`；
   - **不按 `pi/mi` 截断**——那是任意标签偏置。
6. 返回 `(pi, j, oi, m)` 与特征，顺序即候选顺序。

**`rng` 的来源**（策略层，不进 `env/`）：

```python
rng = np.random.default_rng([seed_chain, TOPK_SEED, kind_code, decision_index])
```

- `kind_code`：PA=0，LA=1，RA=2。
- `decision_index`：本局该 kind 的第几个决策（0 起）。
- **每次决策重建**，不跨决策推进。故它与回放路径、采样路径无关，只由 `(seed_chain, kind, index)` 决定。
- ⚠️ 候选集在**边界平局组**内是随机的。这是**有意**的：规则平局本身在原文里就是随机（`apply_pa_rule` 的 `tie_rng`），按 `pi/mi` 截断会制造和 LA §4.28 同型的任意偏置。

### 2.3 LA 候选

1. 对每个车 `a ∈ cand_v`：`s = la_score(LA_RULE_NAMES[r], view, a, (0.0,))`（元组）。
2. 按 `s` 升序分组；组内按 `a` 排序。
3. 截断规则与 PA 相同：**严格 K；边界平局组内均匀抽**。
4. `K=6`；当车队 ≤6 台时 `K_eff = n_agv`（所有车）。

### 2.4 RA 候选（**不做 top-K**）

RA 是二元决策。规则只做**候选过滤**，不引入阈值：

| 规则 | 允许的动作集 |
|---|---|
| `AGE` | `{NOW, DEFER}` |
| `QUEUE_EMPTY` | `in_q_len == 0` ⟹ `{NOW, DEFER}`；否则 `{DEFER}` |
| `OLDEST` | 本机役龄 ≥ 其他所有机台 ⟹ `{NOW, DEFER}`；否则 `{DEFER}` |

- 逾期的强制保养**不产生决策**（`des._pm_after_op` 不变）。
- 阈值 `A` 不再存在。打分头从 `pm_used_min / pm_interval` 自己学。
- 允许集大小 ≤ 2；大小为 1 时动作被强制，`logp = 0`。

---

## 3. 三个打分头

### 3.1 共同结构

```
一个 agent 一个打分头（不是一条规则一个）。
规则身份通过 embedding 进入，不靠硬路由。
```

| 头 | 输入 | 隐藏 | 输出 |
|---|---|---|---|
| `score_pa` | `[z_M[m] ‖ z_B[j] ‖ f_pair(9) ‖ e_rule(8)]` | 128 | **每候选 1 个 logit**，形状 `(K,)` |
| `score_la` | `[z_V[a] ‖ la_ctx(4) ‖ f_la(4) ‖ e_rule(8)]` | 64 | `(K,)` |
| `score_ra` | `[z_M[m] ‖ f_ra(5) ‖ e_rule(8)]` | 64 | **2 个 logit**（NOW, DEFER） |

`z` = 共享编码器的逐 token 输出 `(N, d_model)`。行号：

| 段 | 行 |
|---|---|
| M | `0 .. n_m−1` |
| B | `n_m .. n_m+n_jobs−1` |
| V | `n_m+n_jobs .. n_m+n_jobs+n_agv−1` |
| G | 末行 |

故 PA 候选 `(j, m)` 取 `z[n_m+j]` 与 `z[m]`；LA 取 `z[n_m+n_jobs+a]`；RA 取 `z[m]`。

**规则 embedding**：PA `Embedding(12, 8)`，LA `Embedding(3, 8)`，RA `Embedding(3, 8)`。由 MA 选中的规则号索引。

### 3.2 候选特征（全部按实例静态量归一）

`m_ref` = `NormContext.m_ref`；`total_work_min` = `NormContext.total_work_min`。

**PA `f_pair`（9 维）**，来自 `rules.pa_job_terms` / `pa_machine_terms`：

| # | 特征 | 归一 | 裁 |
|---|---|---|---|
| 0 | `apt` | `/ m_ref` | `[0, 4]` |
| 1 | `rtj` | `/ m_ref` | `[0, 4]` |
| 2 | `wait` | `/ m_ref` | `[0, 4]` |
| 3 | `due`（`due − now`） | `/ m_ref` | `[−4, 4]` |
| 4 | `ready_delay` | `/ m_ref` | `[0, 4]` |
| 5 | `pt` | `/ m_ref` | `[0, 4]` |
| 6 | `avail_delay` | `/ m_ref` | `[0, 4]` |
| 7 | `util` | 无 | `[0, 4]` |
| 8 | `pt_over_den` | 无 | `[0, 4]` |

**LA `f_la`（4 维）**，来自 `rules.la_terms`：

| # | 特征 | 归一 | 裁 |
|---|---|---|---|
| 0 | `cost` | `/ m_ref` | `[0, 4]` |
| 1 | `util` | 无 | `[0, 4]` |
| 2 | `cost_over_den` | 无 | `[0, 4]` |
| 3 | `arrive_delay` | `/ m_ref` | `[0, 4]` |

**RA `f_ra`（5 维）**：

| # | 特征 | 归一 | 裁 |
|---|---|---|---|
| 0 | `pm_used_min` | `/ pm_interval` | `[0, 1.5]` |
| 1 | `in_q_len` | `/ max_queued` | `[0, 1]` |
| 2 | `is_oldest` | 0 / 1 | — |
| 3 | `remaining_min` | `/ total_work_min` | `[0, 1]` |
| 4 | `backlog_min` | `/ total_work_min` | `[0, 1]` |

⚠️ **不输入候选名次**。名次是规则和平局抽签的产物，不是偏好。头可以从特征里重建规则分（例如 SPT 看 `apt`），不需要名次。

### 3.3 组内 z 化

PA / LA：

```python
s = head(x)                                  # (K,)
if K_eff >= MIN_Z_NORM_K:                    # 3
    var = s[mask].var(unbiased=False)
    l = (s - s[mask].mean()) / (sqrt(var + 1e-6) + 1e-6)
else:
    l = s                                    # K_eff == 2：保留原始尺度
l = l.masked_fill(~mask, -inf)
```

⚠️ **`sqrt(var + ε)`，不是 `sqrt(var) + ε`**：候选分数全相等时 `var = 0`，
`sqrt` 在 0 点的导数是无穷 ⟹ `0 × inf = nan`，整条反传被污染
（LA 在 t=0 常见：所有车同构、特征相同、打分逐位相等）。

- `K_eff == 1`：跳过 z 化（`std` 无定义）。
- `K_eff == 2`：**不 z 化**——二元时 z 化会把 logit 差钉死成 ±1，抹掉“有多确定”。
- RA：**不 z 化**，原始 2 路 logit，再按允许集掩码。

⚠️ **z 化是单调仿射变换**，**不改变排序**。评测取 argmax，故 z 化只影响训练时的采样分布，不影响确定性策略的选法。

### 3.4 采样与 log-prob

| 情形 | 动作 | `logp` |
|---|---|---|
| `K_eff == 1` | 直接取唯一候选，**不抽签** | `0.0` |
| `K_eff >= 2`，`greedy=True` | `argmax`；**分数完全并列时用本节 `rng` 均匀抽** | `log_softmax(l)[a]` |
| `K_eff >= 2`，`greedy=False` | `multinomial(softmax(l), generator=self.gen)` | `log_softmax(l)[a]` |

- **`K_eff == 1` 不推进 `self.gen`**。否则“多一个强制决策”会把后续采样流整体移位。
- greedy 的分数并列抽签用 §2 的逐决策 `rng`，**不用车号 / 工件号 / 机台号**。
- `logp` 一律在 **K 个候选内**归一化：`Σ_{i∈C} exp(logp_i) = 1`。
- RA 的 `mask` 只有 1 个允许动作时同上（`logp=0`，不抽签）。

---

## 4. 候选集大小 1 的处理

| 决策 | 何时为 1 | 处理 |
|---|---|---|
| PA | 池只有 1 工件，或其 `alts` 只有 1 台 | 强制；`logp=0`；不抽签 |
| LA | 车队只有 1 台 | 强制；`logp=0`；不抽签 |
| RA | 规则过滤掉 NOW | 强制 DEFER；`logp=0`；不抽签 |

- `Decision.cand` 仍记录（供事后诊断），但训练损失里这一条贡献 `Â · 0 = 0`。
- `_one_logp` 对 `n_valid == 1` 返回 `torch.zeros((), requires_grad=True)`，不建分布。
- **候选集为空**（`n_valid == 0`）：显式 `ValueError`。PA 的空池、LA 的空车队、RA 的空允许集都是 bug，不静默兜底。

---

## 5. 训练接口

### 5.1 不变的部分

- **MA**：三个分类头（12 / 3 / 3）、`RuleSelectHeads`、`ma_logp_at`、`phase_a` 的 G 组采样——全部不变。
- **优势**：`group_advantage` 的组内 z 化 `(R − μ)/(σ+ε)` 不变。
- **损失形状**：`loss = −Σ_t Â_t · logp_t`，无 ratio、无 clip、无 entropy。
- **两阶段次序**：A 先 B 后不变。

### 5.2 变的部分

```python
# logp 重算（带梯度）
z = encoder(tok, seg)                    # 一次批量前向
l = score_head(z, cand_rows, cand_feats, rule, mask)
logp = log_softmax(l)[act]
```

| 项 | 旧 | 新 |
|---|---|---|
| 下层动作 | 连续 `u ∈ R^3 / R^1 / R^1` | 候选集里的下标 `a ∈ [0, K_eff)` |
| 下层 logp | `log N(u; μ, σ)` | `log_softmax(l)[a]` |
| `Decision.param_u` | 必填（下层） | **删除** |
| `Decision.cand` | — | 新增（见 §5.3） |
| `Decision.act` | — | 新增（回放用的具体动作） |

### 5.3 `Decision` 记录

```python
class CandRec(NamedTuple):
    rows_a: np.ndarray          # (K,) int32 —— PA: 作业行；LA: 车行；RA: (1,) 机台行
    rows_b: np.ndarray | None   # (K,) int32 —— 仅 PA：机台行
    feats: np.ndarray           # PA (K,9) / LA (K,4) / RA (1,5) float32
    mask: np.ndarray            # PA/LA (K,) bool；RA (2,) bool
    act: int                    # 选中下标；RA 为 0/1

class Decision(NamedTuple):
    kind: str
    tok: np.ndarray
    seg: tuple[int, ...]
    rule_pa: int
    rule_la: int
    rule_ra: int
    logp: float
    cand: CandRec | None = None
    act: tuple[int, ...] | None = None   # PA: (pool_idx, machine) / LA: (agv,) / RA: (code,) / MA: None
    la_ctx: np.ndarray | None = None
```

- `tok` 已存整张 token 表 ⟹ 重算时重新过编码器，按 `rows_a/rows_b` gather。
- `feats` 很小（K×9），可直接存。
- `rule_pa/la/ra` 已存规则号 ⟹ 规则 embedding 用它。

### 5.4 批量重算

- 编码器**一次前向**处理整批 `tok`（与今日的 `batched_forward` 相同）。
- 按 kind 分组，把候选补齐到 `K_max`：
  - `rows_a (B,K)`、`rows_b (B,K)`、`feats (B,K,Dc)`、`mask (B,K)`；
  - 打分头一次算出 `(B,K)`；
  - 在 `mask` 内做 z 化；`log_softmax`；按 `act` gather。
- RA 单独：`rows_a (B,1)`、`feats (B,1,5)`、`mask (B,2)`，输出 `(B,2)`。

### 5.5 梯度边界

| 阶段 | 允许有梯度 | 验收 |
|---|---|---|
| A | `select`（+ 共享编码器） | 三个 score head 的 `grad_norm == 0` |
| B | `score_pa` / `score_la` / `score_ra`（+ 编码器） | `select` 的 `grad_norm == 0` |

`hrl/train.py::_grad_norm` 与 `hrl/flat.py::grad_norm` 的 lower 元组改成三个 score head。

---

## 6. 接线点

| 文件 | 改动 |
|---|---|
| **`hrl/candidates.py`（新）** | `K_PA` · `K_LA` · `TOPK_SEED` · `pa_candidates` · `la_candidates` · `ra_candidates` · `PACand` / `LACand` |
| `hrl/rules.py` | **不改**。`apply_pa_rule` / `la_rule_policy` / `pm_rule_policy` 与 `λ` 参数保留，只给 CDR 基线与旧诊断脚本用 |
| `hrl/agents.py` | 删 `RuleParamHeads` 在策略中的使用；加 `CandidateScoreHead` ×3；`PolicyOut` 删 `param_mu`、加 `z`；加 `score_pa/score_la/score_ra`；`forward` 只出规则 logits + `z` |
| `hrl/rollout.py` | `Decision` 换字段；`policy_pa/policy_l/policy_m` 改为“建候选 → 打分 → 采样/argmax”；删 `_sample_param` |
| `hrl/replay.py` | `ScriptedPolicy` 直接用 `d.act` 回放，不再 `decode_param` / `pm_rule_policy`；加动作合法性校验 |
| `hrl/train.py` | `_one_logp` 改候选 logp；`batched_forward` 返回 `z`；`_grad_norm` 换 lower 元组 |
| `hrl/flat.py` | 仅 `grad_norm` 的 heads 元组 |
| `tests/test_agents.py` | 删 `param_mu` 断言；加 score head 宽度 / z 化断言 |
| `tests/test_rollout.py` | 删 `param_u`；加“重算 logp == 采样 logp”“K=1 强制” |
| `tests/test_train.py` | 梯度边界；批量 logp 形状 |
| `tests/test_candidates.py`（新） | 严格 K、边界平局抽样、规则平局集、RA 允许集 |

**`env/des.py` 一行不改**：回调契约仍是
`policy_pa(view, pool, tie_rng) -> (pool_idx, machine)`、
`policy_l(view, cand_v) -> agv`、
`policy_m(snap, mach, cands) -> action`。

---

## 7. 验收

### 7.1 功能

1. 对全部 12 条 PA 规则、随机视图：`len(cands) <= 16`；合法对总数 ≤16 时**全部**出现且不重复。
2. 边界平局抽样：构造一个跨 K 线的 3 人平局组，跑 100 个 seed；每个成员被排除的频率约 1/3，**与 `pi/mi` 无关**。
3. 对全部 3 条 LA 规则：`len(cands) <= 6`；`n_agv <= 6` 时等于全部车。
4. RA 允许集：`QUEUE_EMPTY` 队列非空 ⟹ 只有 `DEFER`；`OLDEST` 非最旧 ⟹ 只有 `DEFER`；`AGE` ⟹ 两个都在。
5. 重算 logp 与采样 logp 逐位一致（容差 `1e-5`）。
6. `K_eff == 1`：动作被强制、`logp == 0`、`self.gen` 状态不变（用后续一次自由决策复现来验证）。
7. greedy 在分数并列时用逐决策 `rng` 打破，**不按实体号**。

### 7.2 门禁

- `plcsp/tests/` · `plcsp_v2/tests/` · `plcsp_v3/tests/` 全绿。
- `ruff check plcsp/ plcsp_v2/ plcsp_v3/` 干净。

### 7.3 读数（实现后必做）

1. **重跑覆盖率探针**，用新的边界截断规则：
   - `TOPK_PA` 至少覆盖 `EAJ+SMPT` 与 `EDD+SMPT`（EDD 平局最大）；
   - 若 PA 覆盖率从 100% 明显下降，**如实报数**，不改 K（K 由用户定）。
2. **重跑平局频率探针**（可选，用于记录 K_eff 分布）。
3. **重新训练 v3 多实例主实验**（≥3 seed × 200 step）。
4. 报告：MA 规则分布、候选集大小分布、强制决策占比、PA/LA 动作分布、MS / TC / 能耗 / 拖期、对 27 条 CDR 与 HV。

---

## 8. 不做

- 不改 `env/des.py` 的动力学、事件次序、平局随机源。
- 不改 MA 的三个分类头与 `phase_a`。
- 不加 PPO ratio / clip / entropy。
- 不加规则超参数（`λ` / `A` 不再由策略输出）。
- 不加几何感知注意力；机台坐标与 `dist_lu` 已在 M 段特征里。
- 不删 `rules.py` 的 `λ` 机制——CDR 基线与旧诊断脚本仍用它。
- 不做旧 ckpt 兼容：`HierarchicalPolicy` 的参数集变了，旧 ckpt **不可加载**。

---

## 9. 风险与登记

| 风险 | 说明 | 处置 |
|---|---|---|
| EDD 的大平局集 | 平局均值 9–16、最大 44；K=16 会切掉同分候选 | 随机边界截断（无偏）；实现后重跑 EDD 覆盖率并报数 |
| 覆盖率 ≠ 全局最优 | 探针只做**单点偏离**，不是全局最优 | 论文里只写“候选覆盖”，不写“最优” |
| 共享打分头 vs 每规则一头 | 本 spec 选共享 + 规则 embedding | 若训练不稳，每规则一头是后备方案；不在本次范围 |
| RA 语义变化 | `A` 删除后，`AGE/QUEUE_EMPTY/OLDEST` 只做候选过滤，阈值由头学 | 在进展记录里写明；基线 `pm_rule_policy` 不变 |
| 旧读数作废 | 动作空间与 RNG 流都变了 | 全部 v3 读数重跑；不引用旧数 |
| z 化后的采样平坦 | logit 标准差被钉成 1，采样分布不会很尖 | 评测取 argmax，排序不受影响；训练探索更充分 |
