# -*- coding: utf-8 -*-
"""PA 切换是否真的随状态？

对 span=3 ckpt 的 264 个 MA 决策，比较 PA 选择 `EAJ+EAM` vs `EAJ+SMPT` 时的状态特征：

- 每个特征的均值差、Cohen's d、置换检验 p、AUC；
- 5 折逻辑回归准确率（若装了 sklearn）。

若 p 普遍很大、AUC≈0.5、分类准确率≈0.5 ⟹ 50/50 切换是噪声。
若若干特征 p 很小、AUC 明显 >0.5 ⟹ 切换随状态。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.hrl.agents import HierarchicalPolicy  # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy  # noqa: E402
from plcsp_v3.hrl.rules import PA_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

CKPT = os.environ.get(
    "V3_DIAG_CKPT",
    os.path.join(os.environ.get("TEMP", ""), "v3_span3_200_seed0.pt"))
MK_NAMES = tuple(f"mk{i:02d}" for i in range(1, 11))
N_SC = 2
SC0 = 1
N_AGV = 3
DT = 20
A_RULE = PA_RULE_NAMES.index("EAJ+EAM")
B_RULE = PA_RULE_NAMES.index("EAJ+SMPT")


def _specs():
    out = []
    for i, name in enumerate(MK_NAMES):
        sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=200 + i, reward_norm="minmax")
        out.append((name, sp))
    return out


def _feats(tok: np.ndarray, seg: tuple[int, ...]) -> dict[str, float]:
    n_m, n_j, n_v = int(seg[0]), int(seg[1]), int(seg[2])
    M = tok[:n_m]
    B = tok[n_m:n_m + n_j]
    V = tok[n_m + n_j:n_m + n_j + n_v]
    G = tok[-1]
    unf = B[B[:, 3] < 0.5] if n_j else B
    if not len(unf):
        unf = np.zeros((1, B.shape[1]), dtype=B.dtype)
    return {
        "due_mean": float(unf[:, 2].mean()),
        "due_min": float(unf[:, 2].min()),
        "due_std": float(unf[:, 2].std()),
        "rem_mean": float(unf[:, 1].mean()),
        "prog_mean": float(unf[:, 0].mean()),
        "m_inq_mean": float(M[:, 1].mean()),
        "m_inq_max": float(M[:, 1].max()),
        "m_busy_frac": float(M[:, 3].mean()),
        "m_rem_mean": float(M[:, 4].mean()),
        "agv_idle_frac": float((V[:, 0] + V[:, 1]).mean()),
        "agv_queued_mean": float(V[:, 6].mean()),
        "g_time": float(G[0]),
        "g_done": float(G[1]),
        "g_inflight": float(G[2]),
        "g_mct_mean": float(G[3]),
    }


def _collect(net, specs):
    rows = []
    for _name, sp in specs:
        for k in range(N_SC):
            p = NetPolicy(net, sp.inst, sp.layout, sp.ctx, greedy=True,
                          seed=0, dt_events=DT, seed_chain=SC0 + k)
            sp.world().run_gated(
                seed_chain=SC0 + k, policy_pa=p.policy_pa,
                policy_l=p.policy_l, policy_m=p.policy_m,
                policy_ma=p.policy_ma, dt_events=DT)
            for d in p.rec.decisions:
                if d.kind != "MA":
                    continue
                rows.append((_feats(d.tok, d.seg), int(d.rule_pa)))
    return rows


def _perm_p(a, b, rng, n=2000):
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)


def _auc(a, b):
    # P(a > b) + 0.5 P(a == b)
    return float((a[:, None] > b[None, :]).mean()
                 + 0.5 * (a[:, None] == b[None, :]).mean())


def main() -> None:
    if not os.path.exists(CKPT):
        raise SystemExit(f"⚠️ 找不到 ckpt {CKPT}")
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = HierarchicalPolicy()
    net.load_state_dict(obj["state_dict"])
    net.eval()
    rows = _collect(net, _specs())
    A = [f for f, r in rows if r == A_RULE]
    B = [f for f, r in rows if r == B_RULE]
    print(f"ckpt = {CKPT}", flush=True)
    print(f"MA 决策 = {len(rows)}；{PA_RULE_NAMES[A_RULE]} = {len(A)}；"
          f"{PA_RULE_NAMES[B_RULE]} = {len(B)}\n", flush=True)

    keys = list(A[0])
    rng = np.random.default_rng(20261019)
    print(f"  {'feature':<18}{'EAJ+EAM':>12}{'EAJ+SMPT':>12}"
          f"{'d':>8}{'p':>8}{'AUC':>7}", flush=True)
    print("  " + "-" * 65, flush=True)
    X = np.asarray([[f[k] for k in keys] for f in A + B], dtype=float)
    y = np.asarray([1] * len(A) + [0] * len(B))
    for k in keys:
        a = np.asarray([f[k] for f in A], dtype=float)
        b = np.asarray([f[k] for f in B], dtype=float)
        sd = np.sqrt(((len(a) - 1) * a.var(ddof=1)
                      + (len(b) - 1) * b.var(ddof=1))
                     / max(len(a) + len(b) - 2, 1))
        d = (a.mean() - b.mean()) / max(sd, 1e-12)
        print(f"  {k:<18}{a.mean():>12.4f}{b.mean():>12.4f}"
              f"{d:>8.2f}{_perm_p(a, b, rng):>8.3f}{_auc(a, b):>7.3f}",
              flush=True)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        acc = cross_val_score(clf, X, y, cv=5).mean()
        print(f"\n  5 折逻辑回归准确率: {acc:.3f}（基线 {max(len(A), len(B)) / len(rows):.3f}）",
              flush=True)
    except Exception as ex:                      # sklearn 未装 / 其他
        print(f"\n  （跳过逻辑回归：{type(ex).__name__}: {ex}）", flush=True)


if __name__ == "__main__":
    main()
