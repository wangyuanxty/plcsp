# P0 地基 实施计划（git 建仓 · 包改名 · 死代码归档）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把项目置于版本控制之下，把包名从说谎的 `geosched` 改为 `plcsp`，并把三条已砍路线的代码归档/删除——全程保持回归测试全绿。

**Architecture:** 三阶段单向推进，**顺序不可交换**：先建仓并提交现状快照（保住历史），再改名（纯机械替换，可验证），最后删代码（此时历史已在 git 里，删除是可回溯的）。每一阶段结束都有一次提交，失败可就地回滚到上一阶段。

**Tech Stack:** git 2.x ｜ Python 3.12.4（`D:/anaconda/python.exe`）｜ pytest 7.4.4 ｜ Windows 11 / PowerShell

**Spec:** `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（§7 模块划分与代码改动、§8 测试策略）

## Global Constraints

- **项目根** = `D:esearch\DeepReinforcementLearningScheduling`。**根目录不留临时产物**（`*.log` / `*.jsonl` / `figs/` / 提取文本 / 渲染图一律进系统临时目录，用完即删）——正式文档放哪里不受限。
- **Python 解释器**一律用 `D:/anaconda/python.exe`（Anaconda base；**非** `python`）。**CPU-only，torch 2.14.0+cpu**。
- **回归测试 `plcsp/tests/test_instances.py` 必须始终全绿（42 项）**——它是 bug#12 的门禁，任何阶段都不得为通过而修改其断言。
- **不删历史证据**：所有被砍路线的脚本与其产出 `*.jsonl` **一律归档，不删除**（它们是负例证据，见 spec §2）。
- **`env/layout.py` 与 `env/corridors.py` 不得删除或移出**（仿真器核心，见 spec §7 的"✅ 已确认砍不得"）。
- **`des.py::run_gated` 不得删除**（它是 L 层事件驱动决策接口，被全部存活训练器与评估路径使用；**只有 `batch_cap` 形参和波次注入两行该删**）。
- **提交信息格式**：`<type>: <description>`，type ∈ {feat, fix, refactor, docs, test, chore, perf, ci}。**不添加任何署名/生成标识**。

## Review Focus

以下五类是本计划的测试覆盖不到、但最容易出事的：

1. **`.gitignore` 漏项 → `third_party/` 被吞进仓库**。该目录 **1.5 GB、33 个嵌套 `.git`**；漏掉会让首次 `git add -A` 卡死或把仓库撑爆。期望行为：`git status` 里**不出现** `third_party/`、`references/`、`checkpoints/`。
2. **改名后遗留的字符串引用**（docstring / 日志前缀 / 文档正文里的 `geosched`）。它不影响运行，**所以测试抓不到**，但会让文档说谎。期望行为：除历史文档外，全库无 `geosched` 字样。
3. **`python -m` 路径引用不在 `.py` 的 import 语句里**，常规 import 替换会漏掉。期望行为：所有 `python -m geosched.X` 均已改为 `plcsp.X`。
4. **删除 `standard_tree.py` 后，归档脚本的导入路径失效**。期望行为：归档目录内的脚本**可保留失效导入**（已归档、不再运行），但**活代码不得有任何指向归档模块的 import**。
5. **删 batch 头导致旧 checkpoint 无法加载**。这是**预期结果**（spec §12 第 9 项已确认 19 个旧 `.pt` 全部作废），但 `runner.py::resume_training` 会随之报错。期望行为：该函数**明确报错并给出可读信息**，而不是静默产出错误结果。

---

### Task 1: git 建仓 + `.gitignore` + 现状快照

**Files:**
- Create: `.gitignore`（项目根，**本计划唯一允许新增的根级文件**）

**Interfaces:**
- Consumes: 无
- Produces: 一个已初始化并含一次提交的 git 仓库；后续 Task 2/3 的 `git mv` / `git rm` 均依赖它

- [ ] **Step 1: 确认仓库尚未存在**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git rev-parse --is-inside-work-tree
```
Expected: `fatal: not a git repository` — **若输出 `true`，停止并报告**（说明已有仓库，本计划的建仓步骤需重新评估）。

- [ ] **Step 2: 写 `.gitignore`**

创建 `D:\research\DeepReinforcementLearningScheduling\.gitignore`：

```gitignore
# ── 外部资源：体积 + 版权 + 嵌套 .git（33 个克隆各有自己的 .git）──
third_party/
references/
checkpoints/

# ── 生成物 ──
geosched/figs/
plcsp/figs/
__pycache__/
*.pyc
*.pyo

# ── 运行日志（可重生成）──
*_run.log
*_run*.log

# ── 编辑器/系统 ──
.vscode/
.idea/
Thumbs.db
desktop.ini
```

> 注：`geosched/figs/` 与 `plcsp/figs/` **两条都写**——Task 2 改名后前者失效、后者生效，冗余一条保证两个阶段都干净。

- [ ] **Step 3: 验证忽略规则生效（在提交前！）**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git init -q && git status --porcelain | grep -E "third_party|references|checkpoints" | head
```
Expected: **无输出**（三目录均被忽略）。

若**有输出**：`.gitignore` 未生效，**不要继续**——检查文件名拼写与位置。

- [ ] **Step 4: 统计将入库的文件数与体积**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git add -A && git status --porcelain | wc -l && du -sh .git
```
Expected: 文件数在 **数十到数百** 量级（不是数千）；`.git` 体积 **远小于 100 MB**（spec §7 估算忽略后约 2 MB 内容 + git 开销）。

**若文件数上千或 `.git` 超过 200 MB → 立即 `git rm -r --cached .` 并检查 `.gitignore`**。

- [ ] **Step 5: 首次提交（现状快照）**

```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git commit -q -m "chore: 初始提交——P0 地基前的完整快照（含待归档的三条已砍路线）"
```

> 提交信息**不含任何署名或生成标识**。

- [ ] **Step 6: 验证快照完整**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git log --oneline && echo "---" && git ls-files | grep -c "" && echo "---" && git ls-files | grep -E "^geosched/(env|nn|algo|tests)/" | head -20
```
Expected:
- `git log` 显示 **1 条**提交
- `git ls-files` 计数 **> 50**
- 第三段列出 `geosched/env/des.py`、`geosched/nn/encoder.py`、`geosched/algo/standard_tree.py`、`geosched/tests/test_instances.py` 等——**证明待归档的代码也进了快照**（这是"提交后删"的前提）

---

### Task 2: 包改名 `geosched` → `plcsp`

**Files:**
- Rename: `geosched/` → `plcsp/`（整目录，用 `git mv` 保历史）
- Modify: `plcsp/**/*.py` 中所有 `geosched` 字样（import + docstring + 日志）
- Modify: `docs/**/*.md` 中所有 `geosched` 字样（历史文档除外，见 Step 4）

**Interfaces:**
- Consumes: Task 1 的 git 仓库
- Produces: 包 `plcsp`，其 `load_mk()` / `parse_fjs_text()` / `rollout()` / `SimConfig` 等公开名**保持不变**（仅包路径变化）

- [ ] **Step 1: 改名前记录基线（测试必须全绿才动）**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && D:/anaconda/python.exe -m pytest geosched/tests/ -q --no-header 2>&1 | tail -3
```
Expected: `42 passed`。**若非 42 passed，停止并报告**——基线不干净时改名会掩盖问题。

- [ ] **Step 2: 用 git mv 改名目录**

```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git mv geosched plcsp && git status --porcelain | head -5
```
Expected: 输出一批 `R  geosched/... -> plcsp/...`（R = renamed），**证明 git 识别为重命名而非删除+新增**。

- [ ] **Step 3: 替换包内所有 `geosched` 引用**

```bash
cd "D:/research/DeepReinforcementLearningScheduling" && grep -rl "geosched" --include="*.py" plcsp/ | while read f; do sed -i 's/geosched/plcsp/g' "$f"; done && grep -rn "geosched" --include="*.py" plcsp/ | head
```
Expected: 第二条命令**无输出**（包内已无 `geosched`）。

> 这同时覆盖了 `from geosched.x import`、`python -m geosched.x`（写在 docstring 里的用法说明）、以及日志前缀——**Review Focus 第 2、3 条**。

- [ ] **Step 4: 替换文档引用（排除历史纪录）**

```bash
cd "D:/research/DeepReinforcementLearningScheduling" && grep -rl "geosched" docs/ | grep -v "progress-log.md" | grep -v "superpowers/specs/" | while read f; do sed -i 's/geosched/plcsp/g' "$f"; done && grep -rn "geosched" docs/ | grep -v "progress-log.md" | grep -v "superpowers/specs/" | head
```
Expected: 第二条命令**无输出**。

> **`progress-log.md` 与 spec 刻意排除**：它们是**历史记录**，其中"包改名 `sagrpo → geosched`"是**当时的事实**，改了反而篡改历史（spec §7 也以 `geosched/` 记录改动前状态）。这是 Review Focus 第 2 条的**有意例外**。

- [ ] **Step 5: 验证包可导入**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && D:/anaconda/python.exe -c "import plcsp; print(plcsp.__file__)"
```
Expected: 打印 `...\plcsp\__init__.py`。

**若报 `ModuleNotFoundError: No module named 'plcsp'`**：检查是否仍在 `geosched/` 下执行（旧目录可能因 `__pycache__` 残留仍可导入）。

- [ ] **Step 6: 跑改名后的回归测试**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header 2>&1 | tail -3
```
Expected: **`42 passed`**（与 Step 1 基线一致）。

- [ ] **Step 7: 验证 CLI 入口仍可用**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m9_due_calib --seeds 2 --mode tau 2>&1 | tail -5
```
Expected: 打印 MK01 的 τ 扫描表（不报 `No module named`）。

- [ ] **Step 8: 提交改名**

```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git add -A && git commit -q -m "refactor: 包名 geosched -> plcsp（geo 已随几何线砍除，原名失真）"
```

---

### Task 3: 归档/删除三条已砍路线的代码

**Files:**
- Create: `plcsp/archive/`（归档目录）
- Move: `plcsp/algo/standard_tree.py` 及 6 个导入它的脚本、`m5_*` 系列脚本与全部 `*.jsonl` → `plcsp/archive/`
- Modify: `plcsp/nn/encoder.py`（删 `GeomBias` + `dist`/`conf` 参数）
- Modify: `plcsp/nn/features.py`、`plcsp/nn/state_emb.py`（删几何上游）
- Modify: `plcsp/algo/policy.py`（删 batch 头）
- Modify: `plcsp/env/des.py`（删 `batch_cap` 形参 + 波次注入两行；**保留 `run_gated`**）
- Modify: `plcsp/algo/runner.py`（`resume_training` 对旧 ckpt 明确报错）

**Interfaces:**
- Consumes: Task 2 的 `plcsp` 包
- Produces: 活代码中不再存在 SA-GRPO 树 / 几何 / 分批三线；`run_gated`、`rollout`、`SimWorld`、`load_mk`、`PolicyNet`（两头）保留可用

> ⚠️ **本任务只做"删除与归档"，不做任何重写**。编码器改成标准 Transformer、`PolicyNet` 接口重设计等**属于计划 3（P2 骨架）**，此处不得顺手改。

- [ ] **Step 1: 建归档目录并搬入 SA-GRPO 树及其全部导入者**

```bash
cd "D:/research/DeepReinforcementLearningScheduling" && mkdir -p plcsp/archive && git mv plcsp/algo/standard_tree.py plcsp/archive/ && for f in m6_flat m6_std_tree m6_opt m6_sensor m7_mechanism m8_coevo; do git mv plcsp/$f.py plcsp/archive/ 2>/dev/null; done && ls plcsp/archive/
```
Expected: `ls` 列出 `standard_tree.py` + 6 个 `m*.py`。

> 搬这 6 个脚本是**必须的**：它们在模块级 `import standard_tree`，不搬则导入即崩（spec §7 影响面调查）。

- [ ] **Step 2: 搬入 `m5_*` 系列脚本与全部实验数据**

```bash
cd "D:/research/DeepReinforcementLearningScheduling" && for f in plcsp/m5_*.py plcsp/m5_*.jsonl plcsp/m6_*.jsonl plcsp/m7_*.jsonl plcsp/m8_*.jsonl; do [ -e "$f" ] && git mv "$f" plcsp/archive/ 2>/dev/null; done; ls plcsp/archive/ | wc -l
```
Expected: 归档目录文件数 **≥ 20**。

- [ ] **Step 3: 验证活代码中无指向归档模块的 import**（Review Focus 第 4 条）

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && grep -rn "standard_tree" --include="*.py" plcsp/ | grep -v "^plcsp/archive/"
```
Expected: **无输出**。

若有输出 → 仍有活代码依赖已归档模块，**必须一并归档或改接**。

- [ ] **Step 4: 从 `encoder.py` 删除几何偏置**

打开 `plcsp/nn/encoder.py`，删除：
- `class GeomBias`（整个类）
- `LayoutEncoder.__init__` 的 `w_d` / `w_c` 形参与其赋值
- `LayoutEncoder.__init__` 中的 `self.bias = GeomBias(w_d, w_c)`
- `LayoutEncoder.forward` 的 `dist` / `conf` 形参、其 dtype 守卫与 `bias` 计算
- `DualAxisLayer.forward` 与 `_axis_attn` 的 `bias` 形参，以及 `scores = scores + bias.unsqueeze(1)` 一行

删除后的三个签名应为（**照此改，不要自行发挥**）：

```python
def _axis_attn(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, N, _ = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q = q.view(B, N, self.h, self.dh).transpose(1, 2)
    k = k.view(B, N, self.h, self.dh).transpose(1, 2)
    v = v.view(B, N, self.h, self.dh).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
    scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)     # 轴掩码保留
    w = F.softmax(scores, dim=-1)
    out = (w @ v).transpose(1, 2).reshape(B, N, self.d)
    return self.oproj(out)

def forward(self, x: torch.Tensor, prod_m: torch.Tensor, logi_m: torch.Tensor) -> torch.Tensor:
    x = x + self.ln1(self._axis_attn(x, prod_m))
    x = x + self.ln2(self._axis_attn(x, logi_m))
    return x + self.ffn(x)
```

```python
def forward(self, tok_feat: torch.Tensor, seg: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """tok_feat: (B,N,F)；seg=(nm,nb,nv)。返回 (token 嵌入 (B,N,d), 全局上下文 (B,d))。"""
    B, N, _ = tok_feat.shape
    x = self.embed(tok_feat)
    if self.mask == "full":
        pm = torch.ones(N, N, dtype=torch.bool, device=x.device).unsqueeze(0).expand(B, -1, -1)
        lm = pm.clone()
    else:
        pm = block_mask(N, seg, "prod", x.device).unsqueeze(0).expand(B, -1, -1)
        lm = block_mask(N, seg, "logi", x.device).unsqueeze(0).expand(B, -1, -1)
    for layer in self.layers:
        x = layer(x, pm, lm)
    x = self.ln(x)
    return x, x.mean(dim=1)
```

`LayoutEncoder.__init__` 的签名改为：
```python
def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 8,
             feat_dim: int = 10, mask: str = "axial"):
```
（删去 `w_d` / `w_c`，其余不变。）

> ⚠️ **不要**改 `block_mask`、`DualAxisLayer` 的轴掩码逻辑、也不要把 `LayoutEncoder` 改名为 Transformer——**接口重设计属计划 3**。

- [ ] **Step 5: 删除 `features.py` 与 `state_emb.py` 中的几何上游**

在 `plcsp/nn/features.py` 删除：`geometry_features()`、`conf_sim()`、`pca_axes()`、`project_to_principal()`、`GeometryFeatures`、`token_feature_dim`，以及 `from .layout import ...`（若仅几何用）。

在 `plcsp/nn/state_emb.py` 删除：`EncState` 的 `dist` / `conf` 字段、其构造段，以及 `encode_state` 中对 `features.geometry_features` / `features.conf_sim` 的调用。

- [ ] **Step 6: 验证 encoder 与 state_emb 自洽**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && D:/anaconda/python.exe -c "
import torch
from plcsp.nn.encoder import LayoutEncoder
enc = LayoutEncoder()                      # 默认 feat_dim=10
tok = torch.randn(2, 9, 10)
emb, ctx = enc(tok, (3, 3, 3))
print('emb', tuple(emb.shape), 'ctx', tuple(ctx.shape))
"
```
Expected: `emb (2, 9, 128) ctx (2, 128)`。

- [ ] **Step 7: 从 `policy.py` 删除 batch 头**

在 `plcsp/algo/policy.py` 删除：`n_modes` 形参、`self.b_head`、`batch_logits()`。

- [ ] **Step 8: 从 `des.py` 删除分批门控（保留 `run_gated`）**

在 `plcsp/env/des.py` 删除：
- `run()` 签名的 `batch_cap` 形参与其 docstring 行
- `run()` 中首批波次注入两行（`first = jkeys[:batch_cap] ...` / `inject_q = ...`），改为全部作业一次注入
- `run_gated()` 签名的 `batch_cap` 形参与其波次注入
- `run_gated(policy_l=None)` 委托分支中对 `batch_cap` 的透传

**保留不动**：`run_gated` 本体、`policy_l`、`dec_log`、`ZoneManager`、`SimWorld.run` 的其余逻辑。

- [ ] **Step 9: 验证 `run_gated` 仍可用**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && D:/anaconda/python.exe -c "
from plcsp.env.instances import load_mk
from plcsp.env.des import rollout, SimConfig
inst = load_mk('mk01')
r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
print({k: r[k] for k in ('makespan','jobs_done','horizon_hit')})
print('zone_wait', r['zone_wait'])
"
```
Expected: `jobs_done: 10`、`horizon_hit: False`、`zone_wait` 含 `n`/`total`/`max`。

- [ ] **Step 10: 让 `resume_training` 对旧 checkpoint 明确报错**（Review Focus 第 5 条）

在 `plcsp/algo/runner.py::resume_training` 的 `torch.load` / `load_state_dict` 外层捕获 `RuntimeError`，转为可读错误：

```python
try:
    policy.load_state_dict(ckpt["policy"])
except RuntimeError as e:
    raise RuntimeError(
        "checkpoint 与当前 PolicyNet 不兼容（P0 已删除 batch 头 b_head.*）。"
        "checkpoints/ 下 19 个旧 .pt 全部作废——它们训练于 bug#12 的错误实例，"
        "且含已删除的 batch 头。请勿复用。"
    ) from e
```

- [ ] **Step 11: 跑回归测试**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header 2>&1 | tail -3
```
Expected: **`42 passed`**（`test_instances.py` 不依赖被删模块，应不受影响）。

- [ ] **Step 12: 全库语法自检**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && D:/anaconda/python.exe -m compileall -q plcsp/ 2>&1 | tail -5
```
Expected: 无 `SyntaxError`。（归档目录内的失效导入**不算错**——`compileall` 只查语法。）

- [ ] **Step 13: 提交归档与删除**

```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git add -A && git commit -q -m "refactor: 归档三条已砍路线（SA-GRPO 树/几何/分批），保留负例证据"
```

---

### Task 4: P0 验收

**Files:**
- Modify: `docs/INDEX.md`（登记 P0 完成与包名变更）

**Interfaces:**
- Consumes: Task 1–3 的全部产出
- Produces: 一个可交付给计划 2（P1 环境）的干净基线

- [ ] **Step 1: 端到端烟测**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -c "
from plcsp.env.instances import load_mk
from plcsp.env.des import rollout, SimConfig
for name in ('mk01','mk07','mk10'):
    inst = load_mk(name)
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    print(name, 'jobs', r['jobs_done'], '/', inst.n_jobs, 'mks', round(r['makespan'],1), 'horizon_hit', r['horizon_hit'])
"
```
Expected: 三实例均 `jobs N / N`、`horizon_hit False`。

- [ ] **Step 2: 确认 git 历史可回溯（本次工作的核心目的）**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git log --oneline && echo "--- 归档文件在历史中可见 ---" && git log --oneline --diff-filter=D --name-only | grep -c "standard_tree" 
```
Expected: **3 条**提交；第二条命令输出 **≥ 1**（证明 `standard_tree.py` 的删除记录在历史里，可随时 `git show` 取回）。

- [ ] **Step 3: 在 `docs/INDEX.md` 登记变更**

在 `docs/INDEX.md` 的 "## 6. 环境" 之前插入一节：

```markdown
## 5.5 P0 地基（2026-10-02 完成）

- **版本控制**：已 `git init` 并建仓。`.gitignore` 排除 `third_party/`（1.5 G / 33 嵌套 .git）、`references/`（410 M / 103 PDF）、`checkpoints/`（66 M）。**入库内容约 2 MB**。
- **包名**：`geosched` → **`plcsp`**（原名的 `geo` 随几何线砍除而失真）。
- **归档**：`plcsp/archive/` —— SA-GRPO 树（`standard_tree.py`）+ 6 个导入它的脚本 + `m5_*` 系列 + 全部 `*.jsonl`。**历史证据保留，不进论文**（spec §2）。
- **删除**：编码器 `GeomBias` 与 `dist`/`conf` 参数、`features.py`/`state_emb.py` 的几何上游、`PolicyNet` 的 batch 头、`des.py` 的 `batch_cap` 门控（**`run_gated` 保留**）。
- **旧 checkpoint**：`checkpoints/` 下 19 个 `.pt` **全部作废**（含已删的 `b_head.*`，且训练于 bug#12 的错误实例）。
- **回归门禁**：`plcsp/tests/test_instances.py` **42 项全绿**。
```

- [ ] **Step 4: 提交验收记录**

```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git add docs/INDEX.md && git commit -q -m "docs: 登记 P0 地基完成（建仓/改名/归档）"
```

- [ ] **Step 5: 最终确认**

Run:
```bash
cd "D:/research/DeepReinforcementLearningScheduling" && git status --porcelain && echo "--- 工作区应干净 ---" && git log --oneline | head -5
```
Expected: `git status` **无输出**；`git log` 显示 **4 条**提交。

---

## 完成后的状态

- 仓库受控、历史可回溯、工作区干净
- 包名 `plcsp`，回归测试 42 项全绿
- 三条已砍路线归档在 `plcsp/archive/`，活代码零依赖
- **可交付给计划 2（P1 环境）**

## 计划 2 / 3 的前置阻塞（不在本计划范围）

| 阻塞 | 说明 |
|---|---|
| ✅ **网格布局已定案（spec §3.2 已补）** | 二维网格 + zone=通道节点 + 撤 line/U/island。**连带须复测**：现有拥堵数据（32% 等待占比）与非单调机制均在 line 上测得，换网格后须**重新标定 `n_agv` 并复现机制** |
| `method-design.md` / `feasibility-report.md` 重写 | 见 spec §12 第 7 项 |
| §9 参数—出处表落到"来源 or assumed" | 见 spec §12 第 4 项 |
| 基线 4 仓库试跑 | 见 spec §12 第 3 项；P5 可并行 |
