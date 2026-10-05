# -*- coding: utf-8 -*-
"""A 阶段（加权标量化）的正式训练脚本（P2 Task 8）——MK01 上的 300 步级运行**由本脚本承担**。

    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m13_train_a --inst mk01 --steps 300 --G 8

**为什么是脚本而不是测试**：MK01 上 `joint_chain_step(G=8)` 实测 **18.2 s/步**（Task 7）
⇒ 300 步 ≈ 1.5 h，**不能进单元测试**。验收测试只跑 30 步 × G=4 的小预算
（`plcsp/tests/test_end_to_end_a.py`）。本脚本要能**后台跑**、**断点续**（`--resume`，
ckpt 每 `--save-every` 步落盘）。

⚠️ **`--seed` 的语义（评审 I-3 修后：可复现）**——`--seed` 同时锁定三条随机来源：

1. **网络初始化**：`main()` 入口 `torch.manual_seed(args.seed)`（在 `build_training_setup`
   构造 `PolicyNet` **之前**）；
2. **仿真扰动流**：第 s 步第 g 条链的 `seed_chain = (args.seed + s) * SEED_STRIDE + g`
   （`runner.py` 传 `seed0+s`；步长常量在 `group_rel.py`）；
3. **动作采样**：`joint_chain_step` 用 `torch.Generator(device=策略设备).manual_seed(seed0+s)`
   采样并透传给 `roll_chain`——**不再走全局 torch RNG**。旧实现（Task 7）走
   `torch.multinomial` 的全局流，`--seed` 锁不住动作 ⇒ 多进程各跑各的、且**不可复现**，
   "同 seed 对照 / 种子矩阵"两件事都不成立。

故**同 seed 同调用序列 ⇒ 逐位可复现**，跨进程亦然（CPU 上 torch 的 RNG 由 seed 完全确定；
SimPy + 每条链独立的 numpy 流亦然）。**两个例外**：
① `--resume` 续跑——ckpt 不存优化器状态、`--lr` 以当次为准，故"续跑段"与"一次跑完"不同
（种子对照请用同一起点、同一预算）；
② **跨设备**（`--device cpu` vs `cuda`）——动作采样流跟随策略设备，CPU 与 CUDA 的
`torch.Generator` 是两条不同的流 ⟹ 同 seed 采出**不同**的链。种子对照必须在**同一设备**上做。

⚠️ **`--device`（2026-10-04，GPU 批次）默认 `cpu`**：CUDA 只影响**算在哪**，不影响任何公式。
本仓测试环境是 CPU-only torch（`cuda.is_available()=False`）；GPU 训练用
`D:/anaconda/envs/py312/python.exe`（torch 2.13.0+cu126，RTX 4060 8 GB）。本机实测（MK01、
`route_k=2`、G=1/G=4、预热后中位 3 次）：批重算的批量是 Σn_g（数百到数千），GPU 在这一情形
有优势；但**在线前向是 batch 1，GPU eager 在那一情形更慢**（kernel 启动开销，见
`progress-log` §36.3）⟹ "整策略上 GPU" 只与 CPU 批量化打平（G=1 1.68 vs 1.55 s/步），
**不是最优**；把两段分设备的"分工"方案（roll 在 CPU、重算在 GPU）实测约 **2.6×**（G=1 1.17、
G=4 5.00 s/步，基准 3.00/12.75），**本批未实现为开关**（需两模型或逐步搬模型的机制，
如实声明）。本节数字与 `progress-log` §36 的三段成本表可互相对照。

输出**一律落 `run_dir`（默认 = **仓库根**的 `checkpoints/a_<inst>`，锚 `__file__` 而非 CWD，
故在包目录里执行也不会建出包内 `checkpoints/`；该目录已被 `.gitignore` 排除），不落包目录**：

- `run_dir/metrics.ndjson`——每步一行 `{"step","r","t",…诊断}`（增量追加，可断外部分析）；
  `--eval-every` 打开时另有 `{"step","eval":{"makespan_mean","makespan_std","rule_makespan"}}`；
- `run_dir/ckpt.pt`——每 `--save-every` 步（含最后一步）覆写。

⚠️ `--lr` 只在**第一次** `joint_chain_step` 时生效（Adam 由该函数惰性创建，之后忽略新 lr）；
`--resume` 不恢复优化器状态（ckpt 只存模型权重与步号），续跑会以**当次 `--lr`** 重建 Adam。

⚠️ **`--route-k`（路线头 R，2026-10-04 恢复）默认 1 = 关闭**：`_drive` 恒走最短路，训练步与
既有读数**逐位相同**（黄金摘要见 `tests/test_route_choice.py`）。传 2 启用——每次行驶在 2 条
候选路径里由策略选（① 拥堵必须开，否则入口显式报错）。它同时透传进**评估**：训练开、
评估关 = 用另一个策略评估，且**静默**（同 `constraints` 的 R2 理由）。

⚠️ **`--prefs`（A 主对比的单目标配置，2026-10-05）默认 `None` = 用 f^ref 派生的 `w`（逐位不变）**：
给定时**取代** `w`（不是相乘），表达 spec §6.1 第 ② 层的"N 组权重"——用户裁定 N=3，即
one-hot `(1,0,0)` / `(0,1,0)` / `(0,0,1)` = 纯 makespan / 纯 energy / 纯 TWT 的单目标 GRPO。
组内 z 化把总尺度消掉 ⟹ one-hot 的原始量纲（makespan ~10²、energy ~10¹、TWT ~10¹）不进优势，
prefs 只改目标的相对权重。⚠️ 报告档 A（⑧ 交期关）的 TWT ≡ 0 ⟹ 纯 TWT 配置的奖励恒 0、
优势恒 0（**空转**）——开跑前先看 2 步的 `r_std`/`grad_norm`，退化就跑不出有意义的数。

⚠️ `--resume` 只在 `run_dir/ckpt.pt` **已存在**时生效（`runner.resume_training` 的既有语义）：
没有 ckpt 就**从头跑**，而 `metrics.ndjson` 是**追加**打开的——此时文件里会出现**重复的 step 号**
（前一段是废弃的尝试）。按 step 取最新一行即可；要干净曲线就换 `--run-dir` 或先删 run_dir。
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev

import torch

from .algo.group_rel import ADV_MODES, SEED_STRIDE, check_prefs, roll_chain
from .algo.policy import PolicyNet
from .algo.runner import run_training
from .algo.setup import build_setup
from .env.constraints import ABLATION_GROUPS, ConstraintConfig
from .env.des import (CHARGE_CAND_SKIP, PM_CAND_NOW, SimConfig, compute_due_dates,
                      rollout, weighted_tardiness)
from .env.instances import load_mk
from .env.reward import ReferenceObjectives, reward_weights
from .env.t3_budget import (BUDGET_RATIO, T3_CONSTRAINTS, T3_ETA, T3Budget,
                            T3Lagrangian, action_usage, check_influenceable)
from .nn.encoder import LayoutEncoder

# 评估种子起点。⚠️ **必须避开训练用过的扰动流**（评审 F1 修的就是这里）：`joint_chain_step`
# 第 s 步第 g 条链用 `seed_chain = (seed0+s)*SEED_STRIDE + g`（`runner.py` 传 `seed0+s`），
# 故训练流落在 `[seed0*1000, (seed0+steps)*1000)`。**旧值 10_000 与"第 10 步"的训练流正面
# 相撞**（G=8 时 10000..10007 全被第 10 步用过 ⇒ 5 个评估流全是训练流），评估集与训练集不
# 分离会把论文 setup 节那句话写错。取 **10**6**（本常量唯一定义处，测试从本模块导入）。
# ⚠️ 安全条件是 **`seed0 + steps ≤ 1000`**，**不是** `steps < 1000`（评审 I-2：训练流随
# `--seed` 整体平移；`--seed 700 --steps 400` 这种组合旧条件说"安全"（400<1000），实际
# 训练流最高到 1099000+G-1，正面撞上评估流——同一个 bug 在 seed 维度复发）。
# 该条件由 `assert_eval_seed_isolated()` 在入口**硬查**，不静默。
EVAL_SEED_BASE = 10 ** 6


def assert_eval_seed_isolated(seed0: int, steps: int) -> None:
    """训练扰动流与评估扰动流不得相交（="评估集与训练集分离"成立的前提）。

    训练流 = `(seed0+s)*SEED_STRIDE + g`（s < steps，g < G）⇒ 上界 `< (seed0+steps)*1000`；
    评估流自 `EVAL_SEED_BASE = 10**6` 起。条件：`(seed0 + steps) * SEED_STRIDE ≤ EVAL_SEED_BASE`
    （G ≤ SEED_STRIDE 时充分）。不满足则**显式报错**——`--seed 700 --steps 301` 正是旧注释
    `steps < 1000` 放过的反例（700+300 = 1000 恰在边界上，再多一步就越界）。
    """
    if (seed0 + steps) * SEED_STRIDE > EVAL_SEED_BASE:
        raise ValueError(
            f"--seed {seed0} 配 --steps {steps}：训练扰动流最高到 "
            f"{(seed0 + steps) * SEED_STRIDE - 1}，会与评估流 [{EVAL_SEED_BASE}, +∞) 相交"
            f"——评估集与训练集不再分离。请满足 seed0 + steps ≤ "
            f"{EVAL_SEED_BASE // SEED_STRIDE}（减小 --seed 或 --steps）。")


def build_training_setup(inst_name: str, cfg: SimConfig | None = None,
                         constraints: ConstraintConfig | None = None):
    """(inst, layout, dm, cfg, ctx, policy, constraints)——在共享 `algo.setup.build_setup`
    之上再补两件事：加载实例、构造策略网络。环境三件套的口径（布局 seed=0 / 真实参考
    makespan / `cfg` 的几何+车队参数 / `constraints` 同源）**全部**由 `build_setup` 定死
    （评审 F4 收敛）。

    ⚠️ **`constraints` 原样返回**（R2）：`ctx` 是按它建的（③/⑧ 的特征静默读 `ctx.constraints`），
    调用方拿到的就是"ctx 是按哪组约束建的"那一份，**不需要凭记忆把同一份再传给
    `joint_chain_step`**——`joint_chain_step` 入口还会校验两者同源，不同源直接报错。
    旧状（R2 前）：形参存在但 `main` 从不传、也不进 `step_kwargs` ⟹ 钩子在邀请
    "ctx 按 A 组约束建、训练跑全开"的静默错配（F2 要消灭的正是这一类）。`None` = 十约束全开。

    ⚠️ 布局 seed 必须为 0：奖励权重取自 `ReferenceObjectives.of`（固定用 seed_layout=0 的参考
    运行），而特征归一化的 `m_ref` 按**该布局**的 seed 取——两者同源才有一致的归一化刻度
    （非 0 种子会被 `joint_chain_step` 入口拒绝，`build_setup` 的默认值即 0）。
    """
    inst = load_mk(inst_name)
    c = cfg or SimConfig()
    cons = constraints or ConstraintConfig()    # None = 十约束全开（与 SimWorld 的语义一致）
    lay, dm, ctx = build_setup(inst, c, constraints=cons)
    pol = PolicyNet(enc=LayoutEncoder())        # 车队规模由 seg 定，网络无 n_agv 形参（M-4）
    return inst, lay, dm, c, ctx, pol, cons


def _make_eval_fn(inst, lay, dm, cfg, ctx, seeds: int, rule: float,
                  constraints: ConstraintConfig | None = None, route_k: int = 1,
                  route_zones: bool = False, geom_bias: bool = False,
                  pm_head: bool = False, charge_head: bool = False,
                  batch_head: bool = False, t3: bool = False):
    """评估回调：argmax 策略在 `seeds` 个扰动种子上的 makespan（spec §5.3.4 约定 3：J>1 只评估）。

    ⚠️ `constraints` 必须与训练同一份（R2）：评估跑的是训练后的策略，动力学口径不一致
    （如训练 ③ 关、评估全开）会让读数对不上训练环境，且**静默**。
    ⚠️ `route_k` 同理（R 头恢复后）：训练开路线头（`route_k=2`）而评估关着 = 用**另一个策略**
    评估（恒走最短路的那一个），读数与训练不对应，且**静默**——故与 `constraints` 一样必须透传。
    ⚠️ **`route_zones` / `geom_bias` / `pm_head` / `charge_head` / `batch_head` 同型**
    （2026-10-05 全开配置接线；`batch_head` 随 ⑩ 拼批头加入）：
    它们每一个都改变**策略看到的输入或动作空间**——训练开、评估关 = 用**另一个策略**评估。
    `cfg` 里的 `agv_failover` / `machine_age_failure` 是**动力学**开关，随 `cfg` 一并到达，同理必须同源。
    **本函数的每个开关都要与 `main` 传给 `joint_chain_step` 的那一份逐字相同。**
    ⚠️ **`t3`**：T3 **不改动力学、也不改策略输入**（罚项只在训练的优势上），故评估不需要它；
    但 T3 的上界判据（设计 §4.2"预算过紧会把动作推成单一取值"）要在**训练后的策略**上看——
    故 `t3=True` 时额外报两个动作使用率（⑪ 充电 / ⑫ 保养频率），供退化守卫读数。

    ⚠️ **两栏规则基线（2026-10-05，A 主对比）**：
    - `rule_makespan` = 调用方传入的 `rule`，锚在**十约束全开**的参考运行（`des.rollout` 的
      默认口径）。它是 `f^ref` / `m_ref` 的锚，**不随 constraints 漂**（刻意的，勿改）。
    - `rule_makespan_same_constraints` = **同约束集**的规则 rollout（`constraints=` 与训练
      同一份）。约束组消融（B）里 policy 跑的是"关掉几条"的问题，拿"全开"的规则当基线
      是**易问题比难问题**（`progress-log.md` §52.2 第 2 条）——本栏补上同口径的对照。
      报告档 B（全开）两栏相等；报告档 A（全关）两栏差很大（mk01：117.8 vs 73.0）。
    ⚠️ 两栏都**必须**报：删掉 `rule_makespan` 会切断 f^ref 的锚，删掉同约束栏则消融对照失真。
    """
    # 同约束集规则的惰性缓存：一次 rollout、全部评估轮次复用（规则是确定性的，不需重算）。
    rule_same: list[float] = []

    def _rule_same_constraints() -> float:
        """同约束集的规则基线 makespan（`rollout`，seed_chain=0，**与训练同一份 constraints**）。

        ⚠️ 与 `rule` 形参（全约束锚）是两个不同的量：只在报告档 B（全开）下两者相等。
        用 `seed_chain=0` 与 `rule` 同一条随机链——两栏的差只来自约束集。
        """
        if not rule_same:
            rule_same.append(float(rollout(inst, seed_chain=0, cfg=cfg,
                                           constraints=constraints)["makespan"]))
        return rule_same[0]

    def eval_fn(policy) -> dict:
        ms, jobs, hits = [], [], []
        eng, twt = [], []
        charge_use, pm_now = [], []
        # ⚠️ **另两项目标（energy / TWT）2026-10-05 补记**：此前评估**只记 makespan**，而
        #   ① `experiment-plan.md` §H 要求附表报**三个分量**；
        #   ② 实验 E（三目标 Pareto 前沿）**必须**有三者，否则根本画不出前沿；
        #   ③ **机制可能只在 energy/TWT 上有贡献**——⑪ 充电头 / ⑫ 维护头最像"拿 makespan
        #      换能耗/交期"的机制，只报 makespan 会把它们**误判成"没用"**。
        # ⚠️ TWT 口径与 `ReferenceObjectives.of` **逐字同源**：同一 TF/RDD 交期 + **等权**
        #   （`SimWorld._tardy` 就是等权，见 `des.py`）。两边各写一份口径就会**静默错位**。
        # **纯记录，不改任何语义**（不参与优势、不参与早停）。
        # `cfg or SimConfig()`：与 `ReferenceObjectives.of` 的 None 语义一致（None = 默认 cfg）。
        _c = cfg or SimConfig()
        due = compute_due_dates(inst, _c.tau, _c.due_range)
        eq_w = dict.fromkeys(range(inst.n_jobs), 1.0)
        for s in range(seeds):
            dec, met = roll_chain(inst, lay, dm, cfg, policy, seed=EVAL_SEED_BASE + s,
                                  ctx=ctx, sample=False, constraints=constraints,
                                  route_k=route_k, route_zones=route_zones,
                                  geom_bias=geom_bias, pm_head=pm_head,
                                  charge_head=charge_head, batch_head=batch_head)
            ms.append(float(met["makespan"]))
            eng.append(float(met["energy"]))
            twt.append(float(weighted_tardiness(met["completes"], due, eq_w)))
            # ⚠️ 完成度必须一起报：未跑完的 episode 的 makespan 是**部分完工的最大值**
            #（`max(completes)`），单看它会读出"小得多的 makespan"这种假改进。
            jobs.append(int(met["jobs_done"]))
            hits.append(bool(met["horizon_hit"]))
            if t3:      # 退化守卫（上界）的动作分布读数：只在 T3 打开时报
                charge_use.append(action_usage(dec, "C",
                                               lambda a: a != CHARGE_CAND_SKIP))
                pm_now.append(action_usage(dec, "M", lambda a: a == PM_CAND_NOW))
        out = {"makespan_mean": mean(ms), "makespan_std": pstdev(ms) if len(ms) > 1 else 0.0,
               "rule_makespan": rule,
               # 同约束集的规则基线（A 主对比；见 docstring）：约束组消融的 policy 跑的是
               # 另一个（更易/更难）的问题，只有这一栏才是它的同口径对照。
               "rule_makespan_same_constraints": _rule_same_constraints(),
               # 逐评估种子的**原始值**（2026-10-05 期①消融）：种子散度是消融差异的判据，
               # 只留均值/标准差看不出单个种子的离群。**纯记录，不改任何语义**。
               "makespan_per_seed": ms,
               # 另两项目标（同一条纪律：**纯记录，不改任何语义**）。逐种子一起留——
               # 消融的比较要用**配对**检验（同一批 eval 种子），只有均值做不了。
               "energy_mean": mean(eng), "energy_per_seed": eng,
               "twt_mean": mean(twt), "twt_per_seed": twt,
               "jobs_done_min": min(jobs), "horizon_hit_frac": sum(hits) / len(hits)}
        if t3:
            # ⑪ 的"充电动作使用率"与 ⑫ 的"主动保养占比"——上界退化守卫的直接读数
            #（`t3_budget.usage_is_degenerate` 是判据；这里只报分布）
            out["charge_action_usage"] = mean(charge_use)
            out["pm_now_rate"] = mean(pm_now)
        return out
    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="A（加权标量化）联合链 GRPO 训练")
    ap.add_argument("--inst", default="mk01", help="实例名（mk01..mk10）")
    # ⚠️ 组名里有以 `-` 开头的（`-物流`/`-生产`/`-信息`）——**argparse 会把 `-物流` 当选项**，
    #    故这类值必须写成 `--constraints=-物流`（等号形式）。help 里写明。
    ap.add_argument("--constraints", default="Full", choices=tuple(ABLATION_GROUPS),
                    help="约束组（消融 B 的那 5 组，定义在 env/constraints.py 的 ABLATION_GROUPS）："
                         f"{tuple(ABLATION_GROUPS)}。默认 Full（十约束全开）。"
                         "⚠️ 以 `-` 开头的组名必须用等号形式：`--constraints=-物流`。"
                         "它与 ctx（特征归一化）、训练、评估**三处同源**——"
                         "`build_training_setup` 原样返回，调用方不需要另传。")
    ap.add_argument("--adv-mode", default="scalar", choices=ADV_MODES,
                    help="优势口径（O1 + 消融 D）："
                         "`scalar`（默认，历史口径，逐位兼容）＝先加权求和再组内 z 化；"
                         "`per_objective`（O1）＝每目标各自组内 z 化再按 w 合成；"
                         "`reinforce`（消融 D 第二臂）＝**只减组内均值、不除以组内标准差** —— "
                         "证「组内相对」的那一半（std 归一化）是不是必要的；"
                         "`raw`（消融 D 第三臂）＝**A = 原始回报 r**（不减均值、不除 std）—— "
                         "证另一半（减不减组内均值）是不是必要的。"
                         "⚠️ 它**不进 `_make_eval_fn`**：评估只跑 argmax、不算优势。")
    # A 主对比的"单目标 GRPO × N 组权重"（spec §6.1 的第 ② 层；用户 2026-10-05 裁定 N=3）。
    # ⚠️ 默认 None ⟹ 走 f^ref 派生的 w，**逐位等于既有读数**（`_advantages`/`scalar_reward`
    #    的 prefs 缺省路径与今日同一表达式）。
    ap.add_argument("--prefs", type=float, nargs=3, default=None, metavar=("P1", "P2", "P3"),
                    help="固定偏好权重（三个浮点，顺序 = makespan/energy/TWT）：给定时"
                         "**取代** f^ref 派生的 w（**不是相乘**）。单目标配置用 one-hot，如 "
                         "`--prefs 1 0 0` = 纯 makespan、`--prefs 0 1 0` = 纯 energy、"
                         "`--prefs 0 0 1` = 纯 TWT。默认 None = 用 w（逐位不变）。"
                         "⚠️ 组内 z 化把总尺度消掉 ⟹ one-hot 的原始量纲不进优势；"
                         "⚠️ T3 开着时 λ 的有效强度随 σ(rewards) 变（见 group_rel 的 docstring）。")
    ap.add_argument("--steps", type=int, default=300, help="训练步数（MK01/G=8 约 18 s/步）")
    ap.add_argument("--G", type=int, default=8, help="组大小（J=1，预算全给 G）")
    ap.add_argument("--lr", type=float, default=3e-4, help="Adam 学习率（仅首步生效）")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子：同时锁定网络初始化/仿真扰动流/动作采样（见模块 docstring）")
    ap.add_argument("--run-dir", default=None,
                    help="默认 <仓库根>/checkpoints/a_<inst>（锚 __file__，不是 CWD）")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=0, help="每 N 步评估一次；0 = 不评估")
    ap.add_argument("--eval-seeds", type=int, default=5, help="评估的扰动种子数")
    ap.add_argument("--resume", action="store_true", help="从 run_dir/ckpt.pt 续跑")
    ap.add_argument("--route-k", type=int, default=1,
                    help="路线头（R）候选条数：1 = 关闭（默认，逐位等于既有读数）；"
                         "2 = 每次行驶在 2 条候选里选（① 拥堵必须开）")
    ap.add_argument("--pm-head", action="store_true",
                    help="⑫ 维护头（M）：默认关（规则自动保养，逐位等于既有读数）；"
                         "开启后机台在两件之间由策略选 {现在保养, 不保养}（要求 ⑫ 维护开启；"
                         "MK01 默认 pm_interval=120 从不逾期，机制验证请配短间隔验证档）")
    # ── 全开配置接线（2026-10-05）：以下五个开关都是**默认关 ⟹ 逐位不变** ──
    # ⚠️ 它们每一个都改变**策略看到的输入或动作空间**，所以训练与评估**必须逐字同源**
    #（`_make_eval_fn` 的 docstring 已写）——训练开、评估关 = 用另一个策略评估。
    ap.add_argument("--route-zones", action="store_true",
                    help="R2 区段 token（①）：默认关。开启后**区段**成为一段新 token，"
                         "R 头逐候选读该路径途经区段的聚合——上下文相关的排序才成立。"
                         "⚠️ 只在 `--route-k > 1` 时有意义，且开启后 token 行数与链长都变"
                         "（MK01：20→33 行、单步 ×1.44，见 progress-log §41）")
    ap.add_argument("--geom-bias", action="store_true",
                    help="几何/度量偏置（①）：默认关。开启后把 `-W·d(i,j)/bbox` 加到注意力"
                         "分数上（W 固定，不新增可学参数），距离取自与 `_seg_min` 同一张矩阵。"
                         "**只作「输入」用途**——不得据此主张跨布局泛化（§6.3 的限定）")
    ap.add_argument("--charge-head", action="store_true",
                    help="⑪ 充电头（C）：默认关。开启后 AGV 待命时由策略选 {不去充, 各桩}"
                         "（要求 ⑪ 充电开启）")
    ap.add_argument("--multi-drop", action="store_true",
                    help="⑩ multi-drop 行程模型（一趟 = 一个取货点 + 多个卸货点）；"
                         "⚠️ 模型变更：打开后与旧配置读数不可比")
    ap.add_argument("--batch-head", action="store_true",
                    help="⑩ 拼批头（在预构造的批次候选里选；需 --multi-drop）；默认关")
    ap.add_argument("--agv-failover", action="store_true",
                    help="⑨ 故障 failover：默认关（在途任务滞留在车上）。开启后停机期间把"
                         "车上/队列里的任务退回、交**未停机**的别的车。**改动力学**，"
                         "makespan 会变（高频验证档实测 139.09→128.40，见 progress-log §43）")
    ap.add_argument("--machine-age-failure", action="store_true",
                    help="③ 役龄故障率：默认关（`fail_rate` 常数、无记忆）。开启后故障率随"
                         "`pm_clock`（主轴工时、保养归零）按 Weibull 递增风险上升 ⟹ ③ 有记忆、"
                         "⑫ 多一重收益。**改动力学**；Weibull 形状参数标 assumed、引文待核"
                         "（见 progress-log §44）")
    # ── T3 拉格朗日（2026-10-05；默认关 ⟹ 逐位不变）──
    ap.add_argument("--t3", action="store_true",
                    help="T3 拉格朗日：奖励加罚项 r' = r − Σλᵢâᵢ，λ 对偶上升（无 critic）。"
                         "默认关（优势路径一字不改，逐位等于既有读数）。需要配套的**动作头**"
                         "（⑫/③ 要 --pm-head（③ 还要 --machine-age-failure）、⑪ 要 --charge-head）"
                         "——没有对应动作的约束不可控，入口显式报错（同 ④⑨ 不纳入的理由）。"
                         "λ 由训练环跨步持有、初值 0；`--resume` 不恢复 λ（见 runner）")
    ap.add_argument("--t3-constraints", default="all",
                    help='T3 的约束子集（逗号分隔）：all（默认，六条全上：'
                         f'{",".join(T3_CONSTRAINTS)}）或子集如 "congestion,finite_buffer,setup_time"')
    ap.add_argument("--t3-ratio", type=float, default=BUDGET_RATIO,
                    help=f"预算比例 b = ratio × a^ref（默认 {BUDGET_RATIO}；a^ref 是冻结表里的"
                         "参考激活量，见 env/t3_budget.py）。⚠️ 双侧判据的旋钮：太松 ⟹ 机制空转、"
                         "太紧 ⟹ 把动作咬死（上界守卫盯动作分布）")
    ap.add_argument("--t3-eta", type=float, default=T3_ETA,
                    help=f"对偶上升步长 η（默认 {T3_ETA}）：λ ← clip(λ + η(â − b), 0, λ_max)；"
                         "⚠️ 待扫（太大震荡、太小到不了预算），默认值只是可用起点")
    # ⚠️ 本节唯一**不改数值语义**的开关（它只改"分几次算"）：故不进 `_make_eval_fn`
    #    （评估走 `roll_chain(sample=False)`，**不做 logp 重算** ⟹ 与它无关）。
    ap.add_argument("--recompute-chunk", type=int, default=0, metavar="N",
                    help="重算的**分段+梯度检查点**（2026-10-05，mk10 显存闸）：0 = 关（默认，"
                         "一次整批前向，逐位不变）。**批大到被显存挡住时**才需要它——例如 mk10 "
                         "全开配置（`route_k=2`+R2）批 ≈ 8 链 × 440 决策 ≈ 3520，实测峰值 49.58 GB"
                         "（8 GB 卡）。⚠️ 分段**必须配检查点**才降峰值（单分段不降：图与整批相同）。"
                         "实测 mk10 全开配置：**128 → 峰值 1.32 GB、步时 126 s**（未分段 49.58 GB / "
                         "171 s——省显存与提速同时发生，因为未分段那一路在撞分配器重试）。"
                         "但**显存够用时它更慢**（反向要按段重算）：**它是换显存的手段，不是提速的**")
    ap.add_argument("--pm-interval", type=float, default=None, metavar="MIN",
                    help="⑫ 的保养间隔 [主轴分钟]（**机制验证档**）。默认 None = SimConfig 的 "
                         "120——MK01 每机负载 ~25.5 主轴分钟 ⟹ **从不逾期**，⑫ 的被迫激活量"
                         "（`pm_events_forced`）恒 0、T3 的 ⑫ 罚项不被激活（这是 T3 标定用"
                         "短间隔验证档的理由，见 env/t3_budget.py）。改它 = 改动力学，读数与默认配置不可比")
    ap.add_argument("--agv-battery-kwh", type=float, default=None, metavar="KWH",
                    help="把车队电池全换成该容量 [kWh]（**机制验证档**，同 test_charge_head 的"
                         "小电池口径：0.10 + `battery_low=0`）。默认 None = 布局默认 2–4 kWh——"
                         "一个 episode 放不空 ⟹ ⑪ 的耗尽激活量（`agv_dry_events`）恒 0，"
                         "T3 的 ⑪ 罚项无从生效。**改布局 ⟹ 读数与默认配置不可比**（电池是布局属性）")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="策略所在设备：默认 cpu（本仓测试环境是 CPU-only torch）。"
                         "cuda = 整步（在线前向 + 批重算）都在 GPU 上——重算的批大小是 "
                         "Σn_g（数百到数千），GPU 在这一情形对 CPU 有优势；但**在线前向是 "
                         "batch 1**，GPU eager 在那一情形反而慢（kernel 启动开销，"
                         "见 progress-log §36.3）⟹ 本开关是「整策略」粒度，"
                         "不是最优分工；分工方案（roll 在 CPU、重算在 GPU）见 §36")
    ap.add_argument("--parallel", action="store_true",
                    help="链级多进程（2026-10-04 并行批次）：G 条链铺到 worker 进程，worker "
                         "只跑 CPU（仿真 + 在线前向），主进程继续用 --device 做重算/反向。"
                         "默认关（原串行路径，读数逐位不变）。⚠️ 开启后采样流改为**逐链"
                         "独立**（否则消费次序不确定）⟹ 与串行配置同 seed 的数值不同"
                         "（第九次读数作废，见 progress-log §39）")
    ap.add_argument("--worker-device", default="cpu", choices=("cpu", "cuda"),
                    help="并行配置 worker 的策略设备（2026-10-04 worker 设备批次）："
                         "cpu = 默认配置（worker 直接读共享内存镜像，每步零参数 IPC）；"
                         "cuda = worker 在自己的进程里建 CUDA 上下文 + GPU 副本，在线前向"
                         "走 CUDA 图（8 个上下文要显存，失败显式报错、不退回 CPU worker）")
    ap.add_argument("--workers", type=int, default=None,
                    help="并行配置的 worker 进程数（默认 min(核数, G)）")
    args = ap.parse_args()
    # `--prefs`（A 主对比的单目标配置）：默认 None ⟹ 用 f^ref 派生的 w，逐位不变。
    # ⚠️ 开工前就校验（守卫在 `group_rel.check_prefs`，与 `joint_chain_step` 入口同一份）——
    #    非法 prefs 的代价不该是"跑完 20 分钟参考运行才报错"。
    prefs = None if args.prefs is None else check_prefs(args.prefs)

    # 默认落**仓库根**的 checkpoints/（锚 `__file__`，不是 CWD）——否则在包目录里执行会建出
    # `plcsp/checkpoints/`，正好破坏本文件 docstring 里"不落包目录"的保证（评审 Minor）。
    run_dir = Path(args.run_dir) if args.run_dir else (
        Path(__file__).resolve().parents[1] / "checkpoints" / f"a_{args.inst}")
    if args.resume and not (run_dir / "ckpt.pt").exists():
        print(f"[m13] ⚠️ --resume 但 {run_dir / 'ckpt.pt'} 不存在：将从 step 0 重跑，且 "
              "metrics.ndjson 是**追加**模式 ⇒ 会出现重复 step 号（按 step 取最新一行）。",
              flush=True)
    if args.eval_every > 0:                  # 只有真会用评估流时才查（不开评估 = 该条件空真）
        assert_eval_seed_isolated(args.seed, args.steps)
    # ⚠️ 必须在 build_training_setup **之前**：`PolicyNet` 的初始化吃全局 torch RNG。
    #    动作采样自评审 I-3 起由 `seed0+s` 派生的 `torch.Generator` 负责，与本流互不干扰。
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "[m13] --device cuda 但当前解释器的 torch 不可用 CUDA（cuda.is_available()=False）。"
            "本仓测试环境（D:/anaconda/python.exe）是 CPU-only 构建；GPU 训练请用 "
            "D:/anaconda/envs/py312/python.exe（torch 2.13.0+cu126）。")
    # ⚠️ ③/⑨ 的开关是 **`SimConfig` 级**（改动力学），必须在 `build_training_setup`
    #    **之前**建进 `cfg`——这样 `ctx`（特征归一化）与 `cfg`（仿真动力学）同源，
    #    正是 `build_setup` 的既定纪律（F4 收敛）。其余四个开关是**链级**（改输入/动作空间），
    #    走下面的 `step_kwargs` 与 `eval_fn`。
    cfg = SimConfig(agv_failover=args.agv_failover,
                    machine_age_failure=args.machine_age_failure,
                    multi_drop=args.multi_drop)
    # ⚠️ 小电池验证档（可选）：电池是**布局**属性、`battery_low` 是 **cfg** 属性，两者必须一起改
    #    （同 `test_charge_head` 的口径：只改电池不改 `battery_low`，规则配置会在低电就补电、
    #    **到不了耗尽**）。`battery_low` 必须在 `build_training_setup` **之前**进 cfg——
    #    否则 `ctx` 与 `ReferenceObjectives.of` 会按两份 cfg 取参考运行（m_ref 与 f^ref
    #    不同源，静默错位）。车队的电池替换在布局采样之后做（布局是 `build_training_setup` 产的）。
    if args.agv_battery_kwh is not None:
        cfg = replace(cfg, battery_low=0.0)
    if args.pm_interval is not None:
        cfg = replace(cfg, pm_interval=args.pm_interval)
    inst, lay, dm, cfg, ctx, pol, constraints = build_training_setup(
        args.inst, cfg=cfg, constraints=ABLATION_GROUPS[args.constraints])
    if args.agv_battery_kwh is not None:
        from .env.layout import AgvSpec
        for i, a in enumerate(lay.agvs):
            lay.agvs[i] = AgvSpec(id=a.id, speed_factor=a.speed_factor,
                                  capacity=a.capacity, battery_kwh=args.agv_battery_kwh)
    # ⚠️ 设备在**建好策略之后**统一搬（`--device cuda` 时整步在 GPU 上：在线前向经
    #    `forward_enc`、批重算经 `_decision_logp_terms`、动作采样流经 `policy.device` 三处
    #    全部跟随参数设备，没有任何一处硬编码 cpu）。
    pol = pol.to(args.device)
    ref = ReferenceObjectives.of(inst, cfg)     # ref 进训练入口（评审 F5：w 由 ref 派生）
    w = reward_weights(ref.as_tuple())          # 仅为日志打印
    # ⚠️ rule 与 f^ref 都锚在**全约束**参考运行（`reference_run` 的既定语义，des.py），
    #    不随 constraints 走；随 constraints 走的是训练/评估的动力学（ctx + step_kwargs + eval_fn）。
    rule = float(rollout(inst, seed_chain=0, cfg=cfg)["makespan"])   # = M_ref（同一运行）
    print(f"[m13] inst={args.inst} 作业{inst.n_jobs}×机台{inst.n_machines} "
          f"车队{cfg.n_agv}｜steps={args.steps} G={args.G} lr={args.lr} seed={args.seed}"
          f"｜route_k={args.route_k}｜pm_head={args.pm_head}｜device={pol.device}"
          f"｜parallel={args.parallel}(workers={args.workers or 'auto'},"
          f"worker_device={args.worker_device})")
    # 全开配置的开关逐个打印——**训练与评估是否同源**只看这一行就能核（见 `_make_eval_fn`）
    print(f"[m13] 开关（训练=评估，同源）：route_zones={args.route_zones} "
          f"geom_bias={args.geom_bias} pm_head={args.pm_head} "
          f"charge_head={args.charge_head} batch_head={args.batch_head} "
          f"multi_drop={args.multi_drop} agv_failover={args.agv_failover} "
          f"machine_age_failure={args.machine_age_failure} t3={args.t3}")
    # 这两项不进 `_make_eval_fn`（评估只跑 argmax）：约束组经 `constraints=` 三处同源；
    # 优势口径只在训练侧用。**单列一行**，好让日志一眼看出这一跑是哪一组消融。
    print(f"[m13] 约束组={args.constraints}｜优势口径={args.adv_mode}"
          f"（这两项不在上面的同源清单里——评估不跑优势，约束组由 build_training_setup 三处同源）")
    if args.agv_battery_kwh is not None:
        print(f"[m13] ⚠️ 小电池验证档：车队电池全换 {args.agv_battery_kwh} kWh、battery_low=0 "
              f"（电池是**布局**属性 ⟹ 改动力学，与默认配置读数不可比；⑪ 这才可能跑到耗尽）")
    if args.pm_interval is not None:
        print(f"[m13] ⚠️ 短保养间隔验证档：pm_interval={args.pm_interval}（改动力学；⑫ 的被迫激活量"
              f" `pm_events_forced` 这才可能非零）")
    print(f"[m13] 权重 w={tuple(round(x, 4) for x in w)}（f^ref={ref.as_tuple()}）")
    print(f"[m13] prefs={prefs}（None = 用 w；给定时**取代** w——A 主对比的单目标配置："
          f"one-hot (1,0,0)/(0,1,0)/(0,0,1) = 纯 makespan/energy/TWT）")
    # T3：预算表查不到实例 ⟹ `T3Budget` 显式报错（未标定实例不得静默无罚项）。
    # ⚠️ 可控性守卫在**开工前**查（同一个函数也守在 `joint_chain_step` 入口）——
    #    否则要跑完参考运行才发现"⑫ 罚了但策略没有保养动作"。
    t3_state = None
    if args.t3:
        keep = (tuple(T3_CONSTRAINTS) if args.t3_constraints == "all" else
                tuple(s.strip() for s in args.t3_constraints.split(",") if s.strip()))
        check_influenceable(keep, args.pm_head, args.charge_head, cfg.machine_age_failure)
        budget = T3Budget(inst, keep=keep, ratio=args.t3_ratio)
        t3_state = T3Lagrangian(budget, eta=args.t3_eta)
        print(f"[m13] T3 开：keep={budget.keep}｜{budget.describe()}")
        print(f"[m13] T3 λ0=0、η={t3_state.eta}、λ_max={t3_state.lam_max}；λ 由训练环跨步持有"
              f"（`--resume` 不恢复 λ）；退化守卫读数走 --eval-every 的 charge_action_usage /"
              f" pm_now_rate", flush=True)
    print(f"[m13] 规则基线 makespan={rule:.1f}｜run_dir={run_dir.resolve()}")
    print(f"[m13] seed={args.seed} 锁定「初始化 + 仿真流 + 动作采样」：同 seed 可逐位复现"
          "（同设备跨进程亦然；例外：--resume 续跑，以及跨设备——CPU/CUDA 的动作采样流不同，"
          "见模块 docstring）", flush=True)

    run_training(pol, inst, steps=args.steps,
                 step_kwargs=dict(layout=lay, dm=dm, cfg=cfg, ctx=ctx, ref=ref,
                                  constraints=constraints,   # R2：与 ctx 同一份（入口校验同源）
                                  G=args.G, lr=args.lr, route_k=args.route_k,
                                  route_zones=args.route_zones, geom_bias=args.geom_bias,
                                  pm_head=args.pm_head, charge_head=args.charge_head,
                                  batch_head=args.batch_head,
                                  recompute_chunk=args.recompute_chunk,
                                  adv_mode=args.adv_mode,
                                  prefs=prefs),
                 seed0=args.seed, run_dir=str(run_dir), save_every=args.save_every,
                 resume=args.resume,
                 parallel=args.parallel, n_workers=args.workers,
                 worker_device=args.worker_device,
                 eval_fn=(_make_eval_fn(inst, lay, dm, cfg, ctx, args.eval_seeds, rule,
                                        constraints=constraints, route_k=args.route_k,
                                        route_zones=args.route_zones,
                                        geom_bias=args.geom_bias,
                                        pm_head=args.pm_head,
                                        charge_head=args.charge_head,
                                        batch_head=args.batch_head,
                                        t3=args.t3)
                          if args.eval_every > 0 else None),
                 eval_every=(args.eval_every or None),
                 t3=t3_state)
    print(f"[m13] 完成：{run_dir / 'metrics.ndjson'}（每步一行）｜{run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
