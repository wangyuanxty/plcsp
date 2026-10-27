# `plcsp_v3/repro/` —— **Shi 2025 (NHDRL) 的复现与诊断脚本**

## 这个目录为什么存在

复现工作曾经全部放在系统临时目录（`%TEMP%`）。**那是错的** ——
`CLAUDE.md` 那条「临时产物放系统临时目录」管的是**日志 / 图 / 提取文本**，
**不含研究代码**。研究代码要可复算、可复核、要入库。

后果当时很实：`docs/v3/progress-log.md` **点名引用**了这些脚本作为登记读数的依据，
**而它们不在仓库里 ⟹ 读数无法复算**（违反本仓「读数可复算」的规程）。
**2026-10-12 移入本目录，并把脚本里硬编码的 `%TEMP%` 路径拆掉。**

⚠️ **本目录是复现/诊断用的脚本，不是 `plcsp_v3` 包的一部分**（不在 `tests/` 门禁里）。
它们的读数由 `docs/v3/progress-log.md` §37 各小节负责登记。

---

## 脚本 → 它支撑的登记读数

| 脚本 | 支撑哪一条 | 一句话 |
|---|---|---|
| `shi2025_min_pa.py` | §37.21 | **复现的最小版本**：只留 PA、只留 MS、奖励 `−d_ms`。逐层加的起点 |
| `v3_nhdrl.py` | §37.1–37.20 | **主复现**：OA/PA/LA + PPO。多个 `V3_*` 环境开关 |
| `v3_nhdrl_eval.py` | §37.2 / 37.12 | 与基线同协议评测（20 场景 → HV/SP） |
| `v3_baseline_mo.py` | §37.2 | 基线：RN + 27 条 CDR → Pareto → HV/SP |
| `v3_reward_probe.py` | §37.4 | `corr(ΣR, MS)`：奖励方向与分辨力 |
| `v3_zero_reward.py` | §37.14 / 37.19 | 逐决策点看 OA 选哪支、拿多少奖励 |
| `v3_oa_branch.py` | §37.19 | **强制钉死一支**，量两支的 ΣR 与 makespan |
| `v3_calib.py` | §37.20 | 两支的典型单步尺度 `S_ms` / `S_tc` |
| `v3_tc_step.py` | §37.16 | TC 支逐步增量有没有信号 |
| `v3_policy_hist.py` | §37.7 | argmax 直方图 + logit 极差 / softmax 熵 |
| `v3_action_resolution.py` | §37.9 | 决策点上 9 条规则给出几种不同 `(件,机)` |
| `v3_ready_at.py` | §37.10 | `ready_at` 的分辨力（旧口径 85.1% 等于 `now`） |
| `v3_eaj_score.py` | §37.10 / 37.11 | `EAJ` 打分的有效候选数（2.05 → 4.98） |
| `v3_dms_side.py` | §37.15 | **v3 侧**的同一个 `ms_committed` 量（跨版本共性对照） |
| `v3_ft06.py` | §37.18 | 本仓仿真跑 ft06（零运输），对 JobShopLab 仲裁 |
| `jsl_arbiter.py` | §37.18 | **用 JobShopLab 跑标准实例**（须用它自己的 venv） |
| `jsl_zero.py` | §37.18 | 把 JobShopLab 实例的物流矩阵清零（测试用改写） |
| `v3_seeds.py` | §36 | v3 主线的训练探测（regenerative regularization 等） |
| `v3_min2.py` / `v3_statevar.py` / `v3_bandit.py` / `v3_la_bandit.py` | §36 及更早 | 更早一轮的诊断探针 |

---

## 怎么跑

**本仓解释器**（CPU-only，门禁同款）：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:/anaconda/python.exe plcsp_v3/repro/v3_reward_probe.py
```

**`jsl_arbiter.py` / `jsl_zero.py` 例外** —— 它们要 **JobShopLab 自己的 venv**，
且 **cwd 必须是 `jobshoplab/`**（配置与实例都是相对路径）：

```powershell
cd D:/research/DeepReinforcementLearningScheduling/jobshoplab
$env:PYTHONIOENCODING='utf-8'
../jobshoplab-venv/Scripts/python.exe ../plcsp_v3/repro/jsl_arbiter.py ft06
```

⚠️ `jobshoplab/` 与 `jobshoplab-venv/` 在 `.gitignore` 里 ⟹ **换机器时要重新拉**。

---

## 已知的口径陷阱（踩过的，写下来）

1. **`v3_nhdrl.py` 的 `V3_*` 只剩运行参数**（`V3_A`/`V3_B`/`V3_STEPS`/`V3_CHAIN`/`V3_CKPT`），
   其余一律**按原文写死**。⚠️ **2026-10-12 重写**：此前长出 12 个开关，一半是为**已被证伪**
   的变体加的（架构开关、三种奖励归一化、`cal` 的标定常数）。**证伪的东西不再留开关** ——
   留着的开关下一次还会误导人。结论在 `progress-log` §37。
2. **`v3_nhdrl_eval.py` 的参考点**：必须与基线用**同一个**（新 `ready_at` 下是 `(353.6, 108.1)`）。
3. **跨版本比一个量之前，先确认两边的「切点」是同一个**（§37.15.4 的教训）。
4. **测「学习回路通不通」要用 `shi2025_min_pa.py`，不是 `v3_nhdrl.py`** ——
   最小版（只留 PA、奖励 = `−makespan`）一层判据不过，就说明回路有问题，
   此时调三智能体版是白费（§37.23）。

⚠️ **一处已知的复算缺口（如实标注）**：§37.14 / §37.16 里 `raw` 与 `cal` 两个奖励口径的读数
**无法从清理后的代码复算** —— 那两个变体已按「证伪的不留开关」删掉。
它们的结论是**「试过、不成立」**（登记在 §37.19–37.20），**不是论文要用的读数**。
