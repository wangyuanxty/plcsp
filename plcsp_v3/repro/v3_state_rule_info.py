# -*- coding: utf-8 -*-
"""**状态特征里到底有没有“该选哪条规则”的信息**？

══ 为什么 ══

MA 的训练失败可能有两种完全不同的原因：

| 原因 | 判据 |
|---|---|
| **信息在，优化没找到** | 从状态能**预测**最好规则 |
| **信息不在** | 从状态**预测不了**；再训也没用 |

本脚本用**监督学习**回答，不训练任何 RL。

══ 测法 ══

1. 每个训练实例跑一条 base 链（greedy · exact_rule · 保养关 · PA 固定 EAJ+SMPT）；
2. 取每个顶层点 `Z_j` 的**状态特征**（M/B/V/G 四段 token 均值，共 31 维）；
3. 在 `Z_j` 把 PA 规则固定成第 `r` 条、**跑到局末** → `MS(Z_j, r)`；
4. **标签** `y = argmin_r MS(Z_j, r)`（终局，不是段回报）；
5. 用状态特征做监督分类（交叉验证），报：
   - top-1 准确率 vs **多数类基线**；
   - **后悔值**：`MS(预测规则) − MS(最好规则)`。

⚠️ **两个判据缺一不可**：
准确率高但后悔值大 ⟹ 分类器只在平局上猜对。
后悔值小但准确率低 ⟹ 几条规则等效，**这本身是好消息**（选错不疼）。

⚠️ **本测法的局限**：状态轨迹来自**一条固定规则**（EAJ+SMPT）。
分类器看到的是这条轨迹上的状态，不是任意策略的状态分布。

══ 用法 ══

`D:/anaconda/python.exe -m plcsp_v3.repro.v3_state_rule_info`
环境变量：`V3_INFO_NINST`（默认 6）· `V3_INFO_NPA`（默认 9）· `V3_INFO_FOLDS`（默认 5）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plcsp_v3.env.constraints import ConstraintConfig        # noqa: E402
from plcsp_v3.env.instances import gen_mk_like               # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy            # noqa: E402
from plcsp_v3.hrl.replay import run_branch                    # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy                     # noqa: E402
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES    # noqa: E402
from plcsp_v3.hrl.train import _prefix_to, base_chain, rules_dict  # noqa: E402
from plcsp_v3.nn.features import SEG_SLICE                     # noqa: E402
from plcsp_v3.q1_flat import build                             # noqa: E402

NINST = int(os.environ.get("V3_INFO_NINST", "6"))
NPA = int(os.environ.get("V3_INFO_NPA", "9"))
FOLDS = int(os.environ.get("V3_INFO_FOLDS", "5"))
DT = 20
LA = LA_RULE_NAMES.index("LATEST_IN_TIME")
BASE_PA = PA_RULE_NAMES.index("EAJ+SMPT")


def _pool(tok: np.ndarray, seg: tuple[int, ...]) -> np.ndarray:
    """M/B/V/G 四段均值 → 31 维（10+8+7+6）。"""
    out, r = [], 0
    for key, n in zip(("M", "B", "V", "G"), seg):
        block = tok[r:r + int(n), SEG_SLICE[key]]
        out.append(block.mean(axis=0))
        r += int(n)
    return np.concatenate(out).astype(np.float64)


def _branch_ms(net, sp, base, j: int, rule: tuple[int, int, int]) -> float:
    """从 `Z_j` 起把规则固定成 `rule`，跑到局末 → 最终 makespan。"""
    prefix, _ = _prefix_to(base.rec.decisions, j)
    inner = NetPolicy(net, sp.inst, sp.layout, sp.ctx, rules=rules_dict(rule),
                      greedy=True, seed=0, dt_events=DT, freeze_rules=True,
                      seed_chain=base.seed_chain, exact_rule=True)
    m, _ = run_branch(sp.world(), prefix, inner, switch_after_ma=j,
                      stop_after_ma=0)
    return float(m["makespan"])


def _collect():
    torch.manual_seed(0)
    net = HierarchicalPolicy().eval()      # 权重无关：两条路径都走 exact_rule
    cons = ConstraintConfig(maintenance=False)
    X, Y, MS = [], [], []
    for i in range(NINST):
        sp, _ = build(seed=0, inst_obj=gen_mk_like(seed=1000 + i), tau=2.0,
                      pm_interval=30, target_failures=4, n_agv=3,
                      finite_buffer=True, layout_seed=100 + i,
                      reward_norm="minmax", sim_overrides={"due_range": 0.6},
                      constraints=cons)
        base = base_chain(net, sp, greedy=True, seed=0, dt_events=DT,
                          seed_chain=1, fixed_rules={"PA": BASE_PA, "LA": LA},
                          exact_rule=True)
        ma = [d for d in base.rec.decisions if d.kind == "MA"]
        for j, d in enumerate(ma):
            ms = np.array([_branch_ms(net, sp, base, j, (k, LA, 0))
                           for k in range(NPA)])
            X.append(_pool(d.tok, d.seg))
            Y.append(int(np.argmin(ms)))
            MS.append(ms)
        print(f"  inst {i} · {sp.inst.n_jobs}×{sp.inst.n_machines} · "
              f"顶层点 {len(ma)}", flush=True)
    return np.asarray(X), np.asarray(Y), np.asarray(MS)


def main() -> None:
    print(f"**状态 → 最好规则** · {NINST} 个训练实例 · PA 前 {NPA} 条"
          f" · LA={LA_RULE_NAMES[LA]} · **终局标签**\n", flush=True)
    X, Y, MS = _collect()
    n, d = X.shape
    print(f"\n  样本 {n} · 特征 {d} 维 · 类别 {NPA}", flush=True)
    cnt = np.bincount(Y, minlength=NPA)
    print("  最好规则的分布：" + "　".join(
        f"{PA_RULE_NAMES[k]}×{int(c)}" for k, c in enumerate(cnt) if c), flush=True)
    maj = int(cnt.max())
    best_rule = int(np.argmax(cnt))
    spread = float((MS.max(axis=1) - MS.min(axis=1)).mean())
    print(f"  多数类 = {PA_RULE_NAMES[best_rule]}（{maj}/{n} = {maj / n:.1%}）",
          flush=True)
    print(f"  同一状态上**不同规则的 MS 极差**（均值）：{spread:.1f} min",
          flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    regret_maj = float((MS[:, best_rule] - MS.min(axis=1)).mean())
    for name, clf in (
        ("logistic", make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=3000, C=1.0))),
        ("mlp", make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(64,), max_iter=1500,
                                            random_state=0))),
    ):
        cv = StratifiedKFold(FOLDS, shuffle=True, random_state=0)
        pred = cross_val_predict(clf, X, Y, cv=cv)
        acc = float((pred == Y).mean())
        regret = float((MS[np.arange(n), pred] - MS.min(axis=1)).mean())
        print(f"\n  [{name}] {FOLDS} 折交叉验证", flush=True)
        print(f"    准确率 {acc:.1%}　多数类基线 {maj / n:.1%}", flush=True)
        print(f"    后悔值 {regret:.2f} min　"
              f"（总选多数类 {regret_maj:.2f} min）", flush=True)

    print("\n  判据：", flush=True)
    print("    · 准确率 ≈ 多数类、后悔值 ≈ 总选多数类 ⟹ **状态里没有规则选择的信息**",
          flush=True)
    print("    · 准确率明显高、后悔值明显低 ⟹ 信息在，是 RL 优化的问题", flush=True)


if __name__ == "__main__":
    main()
