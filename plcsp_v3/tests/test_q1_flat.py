# -*- coding: utf-8 -*-
"""`evaluate` 的**跨场景**口径（`q1_flat.py`）—— 修 `docs/v3/progress-log.md` §15.2。"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from plcsp_v3.env import layout as layout_mod                        # noqa: E402
from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v3.env.des import SimConfig, reference_makespan            # noqa: E402
from plcsp_v3.env.instances import load_mk                            # noqa: E402
from plcsp_v3.env.reward import ReferenceObjectives, reward_weights   # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy                    # noqa: E402
from plcsp_v3.hrl.train import Spec                                   # noqa: E402
from plcsp_v3.nn.features import norm_context                         # noqa: E402
from plcsp_v3.q1_flat import evaluate                                 # noqa: E402

DT = 50          # ⚠️ 只为测试提速（设计值是 20）


def _spec():
    torch.manual_seed(0)
    inst = load_mk("mk01")
    cfg = SimConfig(tau=1.0)
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg)
    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    return Spec(inst, lay, g, dm, cfg, ctx, w), HierarchicalPolicy()


@pytest.mark.slow
def test_evaluate_spreads_the_n_chains_across_scenarios():
    """⚠️ **`n` 条链要跑 `n` 个场景** —— 否则 `makespan_se` 恒为 0。

    **修之前**（`docs/v3/progress-log.md` §15.2）：`flat_rollout(n, seed=…)` 把
    `seed*1000+i` 只喂给 `NetPolicy.gen`，而 **greedy 档下 `gen` 一次都不用**
    （`_categorical` 走 argmax、`_sample_param` 走 `u = μ`）
    ⟹ **`n` 条链逐位相同**：`se` 恒为 0（**假的**），评测**白花 `n` 倍时间**。

    **修之后**：第 `i` 条链用 `seed_chain = seed + i`（不同的机障/故障实现）⟹ `se` 含**场景方差**。

    ⚠️ `spread_chains=False` **退回旧口径** —— **v2 的历史读数就是这么跑的**
    （三目标**均值可比**、`se` 不可比）。这条同时把那件事钉住。
    """
    spec, net = _spec()
    new = evaluate(net, spec, n=3, seed=0, dt_events=DT)
    assert new["makespan_se"] > 0.0, (
        f"跨场景后标准误不该还是 0（实测 makespan={new['makespan']}）——"
        "三个场景给出逐位相同的读数？")
    old = evaluate(net, spec, n=3, seed=0, dt_events=DT, spread_chains=False)
    assert old["makespan_se"] == 0.0, (
        "旧口径下 `n` 条链应当**逐位相同**（`se` 恒 0）—— 这条钉住历史口径，"
        "若它变了说明别处动了确定性")
