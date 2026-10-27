# -*- coding: utf-8 -*-
"""**状态特征能否预测「MS 最好的 PA 规则」** —— 监督探针。

对 50 个随机规模实例：
1. 跑 9 条 PA 规则（LA 固定 MINCOST），标出 MS 最好的一条；
2. 取**初始 PA 特征**（24 维，用训练 ckpt 的 μ/σ 标准化）；
3. 训练简单分类器，5 折交叉验证。

判据：
- 分类器明显高于多数类基线 ⟹ **特征够用**，RL 的信用分配是问题；
- 接近基线 ⟹ **特征不够**，实例/布局/特征口径是问题。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import (POOL_STRIDE, PT_REDRAW_SEED,  # noqa: E402
                                        make_shi_instance)
from plcsp_v3.hrl.rules import PA_RULE_NAMES, apply_pa_rule, la_rule_policy  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

CKPT = os.environ.get("BP_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "nhdrl3_iso_vary.pt"))
N_INST = int(os.environ.get("BP_NINST", "50"))
JOBS_LO = int(os.environ.get("BP_JOBS_LO", "10"))
JOBS_HI = int(os.environ.get("BP_JOBS_HI", "40"))


def main() -> None:
    _obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    feat_stats = _obj.get("feat_stats") if isinstance(_obj, dict) else None
    if feat_stats is None:
        raise SystemExit("⚠️ ckpt 里没有 feat_stats")
    mu, sd = feat_stats["pa"]

    X, y, sizes = [], [], []
    rng = np.random.default_rng([20261018, 7])
    for i in range(N_INST):
        jobs = int(rng.integers(JOBS_LO, JOBS_HI + 1))
        inst = make_shi_instance(
            jobs, seed=POOL_STRIDE * (5000 + i),
            rot_seed=PT_REDRAW_SEED + 5000 + i,
            alts_dist="uniform", alts_min=N.ALTS_MIN, alts_max=N.ALTS_MAX,
            pt_dist="uniform", pt_lo=2, pt_hi=10)
        spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N.NAGV,
                        finite_buffer=True, layout_seed=0, reward_norm="none",
                        tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
                        layout_scheme="random", **dims_from_shop_len(200.0, inst.n_machines, layout_cols=N.LAYOUT_COLS),
                        layout_cols=N.LAYOUT_COLS, layout_rows=N.LAYOUT_ROWS,
                        shi2025=N.SHI)
        nf = NHFeatures(inst, spec.layout,
                        speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
        f0 = np.asarray(nf.pa(spec.world().snapshot(), 0.0, (5.0, 5.0)),
                        dtype=float)
        X.append(((f0 - mu) / sd).astype(np.float32))
        ms = []
        for k in range(9):
            m = spec.world().run_gated(
                seed_chain=100_000,
                policy_pa=lambda v, p, tr=None, _k=k: apply_pa_rule(_k, v, p, tr),
                policy_l=la_rule_policy(0), dt_events=0)
            ms.append(float(m["makespan"]))
        y.append(int(np.argmin(ms)))
        sizes.append(jobs)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_INST} 实例完成", flush=True)

    X = np.asarray(X)
    y = np.asarray(y)
    print(f"\n**{N_INST} 实例 · 规模 {JOBS_LO}–{JOBS_HI}**\n")
    print("  标签分布：" + "　".join(
        f"{PA_RULE_NAMES[k]}:{v}" for k, v in Counter(y).most_common()))
    base = Counter(y).most_common(1)[0][1] / len(y)
    print(f"  多数类基线：{base:.1%}\n")

    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except Exception as exc:     # noqa: BLE001
        raise SystemExit(f"⚠️ sklearn 不可用：{exc}") from exc

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    models = {
        "多数类": DummyClassifier(strategy="most_frequent"),
        "逻辑回归": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000)),
        "决策树 d=3": DecisionTreeClassifier(max_depth=3, random_state=0),
        "3-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
    }
    print(f"  {'模型':<12}{'5 折准确率':>12}")
    print("  " + "-" * 26)
    for name, model in models.items():
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        print(f"  {name:<12}{acc:>12.1%}")
    print("\n  ⚠️ 明显高于多数类 ⟹ 特征能预测最好规则 ⟹ RL 的信用分配是问题。")
    print("  ⚠️ 接近多数类 ⟹ 特征不足以预测 ⟹ 实例/布局/特征口径是问题。")


if __name__ == "__main__":
    main()
