# -*- coding: utf-8 -*-
"""两阶段交替训练（步 5b）的机制验收。

⚠️ 这一步**慢**（重放前缀 × 组大小）—— 测试用 `dt_events=50` 把顶层点压到 2–3 个，
**不是**设计值 §10② 的 20。设计值的代价见模块末的实测记录。

⚠️ **2026-10-10：⑪ 充电机制整条删除** ⟹ 本文件里所有
`HierarchicalPolicy(n_chargers=len(...chargers))` 一律改成 `HierarchicalPolicy()`
（`Layout.chargers` 与 `n_chargers` 参数都不存在了）。**测试本身一条没删**。
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plcsp_v3.env import layout as layout_mod                        # noqa: E402
from plcsp_v3.env.constraints import ConstraintConfig                 # noqa: E402
from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v3.env.des import SimConfig, reference_makespan            # noqa: E402
from plcsp_v3.env.instances import load_mk                            # noqa: E402
from plcsp_v3.env.reward import EQUAL_WEIGHTS, objective_vector  # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy                    # noqa: E402
from plcsp_v3.hrl.pool import SegmentPool                             # noqa: E402
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES          # noqa: E402
from plcsp_v3.hrl.train import (Spec, base_chain, group_advantage,    # noqa: E402
                                ma_logp_at, segment_from, train_step)
from plcsp_v3.nn.features import norm_context                         # noqa: E402

DT = 50          # ⚠️ 只为测试提速；设计值见 §10②（20）


def _spec(tau: float = 1.0, constraints: ConstraintConfig | None = None):
    """⚠️ **必须给一个收紧的 `tau`**：MK01 默认参考解 **TWT=0**，而 `reward_weights`
    对 0 分量**显式报错**（"TWT=0 的实例须先调 τ"）。不静默补假权重。

    `constraints` 必须**同时**给 `norm_context` 与 `Spec` —— 两份不同源就是假信号。
    """
    torch.manual_seed(0)
    inst = load_mk("mk01")
    cfg = SimConfig(tau=tau)
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg,
                       constraints=constraints)
    # ⚠️ `1/f^ref` 已删（2026-10-11）⟹ 用等权 + 一个固定的 `bounds`
    # （测试只关心「接口通不通」，不关心具体数值）
    w = EQUAL_WEIGHTS
    bounds = ((60.0, 4.0, 0.0), (140.0, 8.0, 300.0))
    net = HierarchicalPolicy()
    return Spec(inst, lay, g, dm, cfg, ctx, w, constraints=constraints,
                bounds=bounds), net


@pytest.mark.slow
def test_seed_chain_changes_the_world_not_just_the_sampling_noise():
    """⚠️ **`seed_chain` 是「场景」那条随机流**（机障 ③ / AGV 故障 ⑨ 的实现）。

    `docs/v3/progress-log.md` §15.3：它此前**写死 0** ⟹ 所有链共享同一个世界 ——
    `--seeds 0 1 2` 只在**采样噪声**上不同，标准误**不含场景方差**（与 §130 同一类错）。

    判据：同一个策略（`greedy=True` ⟹ 确定性）在两个 `seed_chain` 上跑出**不同**的读数。
    """
    spec, net = _spec()
    a = base_chain(net, spec, greedy=True, seed=0, dt_events=50, seed_chain=0)
    b = base_chain(net, spec, greedy=True, seed=0, dt_events=50, seed_chain=1)
    fa, fb = a.metrics, b.metrics                     # type: ignore[attr-defined]
    assert objective_vector(fa) != objective_vector(fb), \
        "两个 `seed_chain` 给出逐位相同的三目标 ⟹ 它根本没接上"
    # 同 `seed_chain` 必须逐位可复现（否则是别处漏了确定性）
    c = base_chain(net, spec, greedy=True, seed=0, dt_events=50, seed_chain=1)
    assert c.metrics["makespan"] == fb["makespan"]    # type: ignore[attr-defined]


@pytest.mark.unit
def test_group_advantage_sums_to_zero_and_is_scale_invariant():
    """组内 z 化（§4.1）：**和恒为 0**（这是"组内只比相对好坏"的代数形式）、对整体平移不变。"""
    a = group_advantage([1.0, 2.0, 3.0, 4.0])
    assert abs(float(a.sum())) < 1e-9
    b = group_advantage([101.0, 102.0, 103.0, 104.0])
    assert a == pytest.approx(b)
    assert group_advantage([5.0, 5.0, 5.0]).tolist() == [0.0, 0.0, 0.0]   # σ=0 ⟹ 全 0


@pytest.mark.slow
def test_gradients_reach_only_the_layer_being_trained():
    """⚠️ **本步的关键验收**：**阶段 B 只动下层、阶段 A 只动顶层**（设计 §6）。

    ⚠️ **v3 把 v2 那条"阶段 A 空转"的断言反过来了。**
    v2 里 `B` 零初始化 ⟹ `g` 不影响行为 ⟹ G 个目标的段回报**全相同** ⟹ `σ_G = 0`
    ⟹ **阶段 A 的 loss 与梯度恒为 0**（§89.2 实测的空转）。
    **v3 删了 `B`**：选不同规则 ⟹ **走不同的 head** ⟹ 行为结构不同 ⟹
    **`σ_G > 0` 从第 0 步起就成立**（设计 §6.0 ③ / §8）⟹ **本阶段第一次就有信号**。
    **这条断言就是"v3 治好了 v2 的空转"的直接证据。**

    ⚠️ **但阶段 B 的空转风险没消** —— 它换了一个来源：v2 的规则是**离散**的
    （两条分支撞上同一条规则的概率 ~1/12），**v3 的下层是连续参数，微小的 `λ` 差异
    可能不改变 `argmin`** ⟹ M 条**各不相同**的参数序列给出**逐位相同**的段回报。

    ⚠️ **不是"链全同"** —— 实测 `种链 = M/M`（8/8），见 `docs/v3/progress-log.md` §12。
    本档实测：

    | `M` | 各顶层点的 `σ_M` | 非零 |
    |---|---|---|
    | **2** | `[0, 0, 0, 0]` | **0 / 4** |
    | **4** | `[0, 1.850, 0, 0]` | 1 / 4 |
    | **8** | `[0, 1.830, 0, 0]` | 1 / 4 |

    （MK01 · 随机初始化 · `dt_events=50` · seed 0 · 4 个顶层点。）
    **`M` 从 4 加到 8 毫无改善** ⟹ 不是"样本不够"，是"某些规则下噪声不影响 `argmin`"。
    **这是一条要带进实验的开放线索**（见 `docs/v3/progress-log.md`），不是本测试的结论。
    """
    spec, net = _spec()
    # ⚠️ **`M=4` 不是随手取的** —— `M=2` 时本档实测 **4 个顶层点的 `σ_M` 全是 0**
    # （见下表）⟹ 阶段 B 的 loss 恒 0 ⟹ `lower` 也测不出来，这条断言会假红。
    out = train_step(net, spec, None, M=4, G=2, seed=0, dt_events=DT)
    b, a = out["B"]["grad_norm"], out["A"]["grad_norm"]
    assert b["upper"] == 0.0, f"阶段 B 动了顶层（应为 0）：{b}"
    assert b["lower"] > 0.0, f"阶段 B 没动下层：{b}"
    assert out["B"]["loss"] != 0.0
    # 阶段 A：G 组规则**行为不同** ⟹ 段回报有离散度 ⟹ 优势非零 ⟹ 顶层拿到梯度
    std = out["A"]["seg_std"]
    assert max(std) > 0.0, f"G 组规则的段回报全相同 ⟹ σ_G=0 ⟹ 顶层拿不到梯度（v3 不该这样）：{std}"
    assert a["upper"] > 0.0, f"阶段 A 没动顶层：{a}"
    assert a["lower"] == 0.0, f"阶段 A 动了下层（该层被冻结）：{a}"
    assert out["A"]["loss"] != 0.0


@pytest.mark.slow
def test_per_head_groups_vary_one_head_at_a_time():
    """**逐头分组**：每个顶层点三组，成员数 12/3/3；只更新 `select`。"""
    spec, net = _spec()
    out = train_step(net, spec, None, M=2, G=2, seed=0, dt_events=DT,
                     per_head=True, span=1, ma_only=True)
    a = out["A"]
    assert a["kind"] == "A_per_head"
    assert a["n_top"] > 0
    assert len(a["groups"]) == 3 * a["n_top"]
    for g in a["groups"]:
        want = {"PA": 12, "LA": 3, "RA": 3}[g["head"]]
        assert len(g["adv"]) == want == len(g["rewards"]) == len(g["rule_ids"])
    assert a["grad_norm"]["upper"] > 0.0
    assert a["grad_norm"]["lower"] == 0.0


@pytest.mark.slow
def test_per_head_skips_ra_when_maintenance_is_off():
    """保养关掉时**不建 RA 组**：没有 RA 决策点，变 RA 规则不会改变读数。"""
    spec, net = _spec(constraints=ConstraintConfig(maintenance=False))
    out = train_step(net, spec, None, M=2, G=2, seed=0, dt_events=DT,
                     per_head=True, span=1, ma_only=True)
    a = out["A"]
    assert a["heads"] == ["PA", "LA"]
    assert {g["head"] for g in a["groups"]} == {"PA", "LA"}
    assert len(a["groups"]) == 2 * a["n_top"]
    assert a["grad_norm"]["upper"] > 0.0


@pytest.mark.slow
def test_fixed_rules_and_pa_limit_are_respected():
    """单训 PA 档：LA 固定、RA 关、PA 只在前 9 条里采。"""
    spec, net = _spec(constraints=ConstraintConfig(maintenance=False))
    la = LA_RULE_NAMES.index("LATEST_IN_TIME")
    p = base_chain(net, spec, greedy=True, dt_events=DT,
                   fixed_rules={"LA": la, "RA": 0}, pa_limit=9)
    ma = [d for d in p.rec.decisions if d.kind == "MA"]
    assert ma, "一条 MA 决策都没有"
    assert all(d.rule_pa < 9 for d in ma), "PA 采到了第 9 条以后"
    assert all(d.rule_la == la for d in ma), "LA 没固定住"
    assert all(d.rule_ra == 0 for d in ma), "RA 没固定住"


@pytest.mark.slow
def test_per_head_pa_only_uses_nine_rules():
    """只训 PA 头：组里只有 PA，成员数 = 9。"""
    spec, net = _spec(constraints=ConstraintConfig(maintenance=False))
    la = LA_RULE_NAMES.index("LATEST_IN_TIME")
    out = train_step(net, spec, None, M=2, G=2, seed=0, dt_events=DT,
                     per_head=True, ma_only=True, only_head="PA",
                     fixed_rules={"LA": la, "RA": 0}, pa_limit=9)
    a = out["A"]
    assert a["heads"] == ["PA"]
    assert {g["head"] for g in a["groups"]} == {"PA"}
    assert all(len(g["adv"]) == 9 for g in a["groups"])
    assert a["grad_norm"]["upper"] > 0.0


@pytest.mark.slow
def test_per_head_la_only_uses_three_rules():
    """只训 LA 头：PA 固定为最好 CDR 的 PA，组里只有 LA，成员数 = 3。"""
    spec, net = _spec(constraints=ConstraintConfig(maintenance=False))
    pa = PA_RULE_NAMES.index("EAJ+SMPT")
    out = train_step(net, spec, None, M=2, G=2, seed=0, dt_events=DT,
                     per_head=True, ma_only=True, only_head="LA",
                     fixed_rules={"PA": pa, "RA": 0})
    a = out["A"]
    assert a["heads"] == ["LA"]
    assert {g["head"] for g in a["groups"]} == {"LA"}
    assert all(len(g["adv"]) == 3 for g in a["groups"])
    assert a["grad_norm"]["upper"] > 0.0


@pytest.mark.slow
def test_per_head_pa_and_la_together():
    """PA 9 + LA 3 一起逐头分组：两组各自更新，不共用优势。"""
    spec, net = _spec(constraints=ConstraintConfig(maintenance=False))
    out = train_step(net, spec, None, M=2, G=2, seed=0, dt_events=DT,
                     per_head=True, ma_only=True, only_head=("PA", "LA"),
                     fixed_rules={"RA": 0}, pa_limit=9)
    a = out["A"]
    assert a["heads"] == ["PA", "LA"]
    assert {g["head"] for g in a["groups"]} == {"PA", "LA"}
    for g in a["groups"]:
        want = 9 if g["head"] == "PA" else 3
        assert len(g["adv"]) == want
    assert a["grad_norm"]["upper"] > 0.0


@pytest.mark.slow
def test_sequential_ma_conditions_la_on_pa_and_trains():
    """顺序 MA：LA 的 logits 随 PA 变；一步训练有梯度。"""
    spec, _ = _spec(constraints=ConstraintConfig(maintenance=False))
    torch.manual_seed(0)
    net = HierarchicalPolicy(seq_ma=True)
    out = train_step(net, spec, None, M=2, G=2, seed=0, dt_events=DT,
                     ma_only=True, span=1, fixed_rules={"RA": 0}, pa_limit=9)
    a = out["A"]
    assert np.isfinite(a["loss"])
    assert a["grad_norm"]["upper"] > 0.0
    base = base_chain(net, spec, greedy=True, dt_events=DT,
                      fixed_rules={"RA": 0}, pa_limit=9)
    d = next(d for d in base.rec.decisions if d.kind == "MA")
    t = torch.as_tensor(d.tok).unsqueeze(0)
    o1 = net(t, d.seg, rules={"PA": torch.tensor([3]), "LA": torch.tensor([0]),
                              "RA": torch.tensor([0])}, hard=True)
    o2 = net(t, d.seg, rules={"PA": torch.tensor([6]), "LA": torch.tensor([0]),
                              "RA": torch.tensor([0])}, hard=True)
    assert not torch.allclose(o1.rule_logits["LA"], o2.rule_logits["LA"])


@pytest.mark.slow
def test_factorial_group_covers_all_pa_la_pairs():
    """全因子组：每个顶层点 9×3=27 条，边际长度 9/3。"""
    spec, _ = _spec(constraints=ConstraintConfig(maintenance=False))
    torch.manual_seed(0)
    net = HierarchicalPolicy(seq_ma=True)
    out = train_step(net, spec, None, M=2, G=2, seed=0, dt_events=DT,
                     per_head=True, ma_only=True, factorial=True,
                     only_head=("PA", "LA"),
                     fixed_rules={"RA": 0}, pa_limit=9)
    a = out["A"]
    assert a["kind"] == "A_factorial"
    assert a["n_pa"] == 9 and a["n_la"] == 3
    assert len(a["matrices"]) == a["n_top"]
    for mat in a["matrices"]:
        assert len(mat) == 9 and len(mat[0]) == 3
    assert len(a["pa_adv"][0]) == 9
    assert len(a["la_adv"][0]) == 3
    assert a["grad_norm"]["upper"] > 0.0


@pytest.mark.slow
def test_segment_pool_matches_the_sequential_path():
    """⚠️ **并行跑段 = 只改怎么算，不改算什么**（设计 §105 的优化 #1）。

    `train_step` 每步要做 `n_top × (M+G)` 条段（mk01 96 条、mk10 400 条），**互相独立**。
    把它们放进进程池是**唯一不动设计参数**就能拿量级提升的一条 ——
    但必须证明它**不改读数**，否则所有既有结论作废。

    ⚠️ 容差不是逐位：并行档每进程 **1 线程**，浮点归约次序与父进程不同 ⟹ 末位可差。
    """
    spec, net = _spec()
    torch.manual_seed(0)
    base_net = HierarchicalPolicy()
    n_seq = copy.deepcopy(base_net)
    seq = train_step(n_seq, spec, None, M=2, G=2, seed=0, dt_events=DT)
    n_par = copy.deepcopy(base_net)
    with SegmentPool(n_par, spec, n_workers=3, dt_events=DT) as pool:
        par = train_step(n_par, spec, None, M=2, G=2, seed=0, dt_events=DT, pool=pool)
    for ph in ("B", "A"):
        assert seq[ph]["n_top"] == par[ph]["n_top"], "顶层点数不同 ⟹ 段的结构就不同了"
        assert seq[ph]["loss"] == pytest.approx(par[ph]["loss"], rel=1e-4, abs=1e-4), \
            f"阶段 {ph} 的 loss 不同：{seq[ph]['loss']} vs {par[ph]['loss']}"


@pytest.mark.unit
def test_segment_pool_refuses_one_worker():
    """⚠️ **先查后跑**：1 个进程没有意义，直接报错。"""
    spec, _ = _spec()
    net = HierarchicalPolicy()
    with pytest.raises(ValueError, match="n_workers"):
        SegmentPool(net, spec, n_workers=1, dt_events=DT)


@pytest.mark.slow
def test_ma_logp_is_a_sum_of_three_independent_heads():
    """⚠️ **顶层是三个独立分类头，不是一个联合分类**（设计 §3 / §4.6）。

    同一顶层点上，`ma_logp_at` 给出的每个数必须等于**三个 `log_softmax` 之和**
    （PA 的 12 路 + LA 的 3 路 + RA 的 3 路）；且**各不相同**
    （否则"每个成员配自己的动作"就退化成同一个数 ⟹ 组内 z 化后 `Σ Â·logp ≡ 0`）。

    ⚠️ **v2 这条测的是"64 档"（`g` 的三维分档）**；v3 的顶层动作是**规则组合**，
    而 `ma_logp_at` 也**不再扫全空间**（`12×3×3 = 108` 组，没必要）——
    它只为**调用方给的那几组**各算一个（设计 §6 的阶段 A 只用得到自己采的 G 组）。
    """
    spec, net = _spec()
    base = base_chain(net, spec, greedy=True, seed=0, dt_events=DT)
    d = next(x for x in base.rec.decisions if x.kind == "MA")
    rules = [(0, 0, 0), (5, 1, 2), (11, 2, 2)]
    lp = ma_logp_at(net, d, rules)
    assert lp.shape == (len(rules),), "返回长度应等于传入的规则组数"
    vals = {round(float(x.detach()), 9) for x in lp}      # `.detach()`：这里只看数值
    assert len(vals) > 1, "三组规则的 logp 全相同 ⟹ 分头没起作用"
    # 手算第 0 组：(0, 0, 0)
    t = torch.as_tensor(d.tok).unsqueeze(0)
    with torch.no_grad():
        o = net(t, d.seg, hard=True)
    manual = sum(torch.log_softmax(o.rule_logits[k][0], dim=-1)[r]
                 for k, r in zip(("PA", "LA", "RA"), rules[0]))
    assert float(lp[0].detach()) == pytest.approx(float(manual.detach()), abs=1e-6)


@pytest.mark.slow
def test_segment_from_returns_the_branchs_own_decision_chain():
    """⚠️ **分支的 loss 必须用分支自己的决策链**，不是 base 链的。

    规则变了 ⟹ 参数变 ⟹ 动作变 ⟹ 状态变 ⟹ 决策条数与内容都不同。用 base 的链算 loss 是**错配**。
    """
    spec, net = _spec()
    base = base_chain(net, spec, greedy=True, seed=0, dt_events=DT)
    rules = [(0, 0, 0), (11, 2, 2)]
    br0 = segment_from(net, spec, base, 0, rule=rules[0], seed=0, greedy=True, dt_events=DT)
    br1 = segment_from(net, spec, base, 0, rule=rules[1], seed=0, greedy=True, dt_events=DT)
    for br, rule in ((br0, rules[0]), (br1, rules[1])):
        assert br.decisions, "分支必须记下自己的决策链"
        no_ma = [d.kind for d in br.decisions if d.kind == "MA"]
        assert not no_ma, "分支里规则被**冻结** ⟹ 不该有顶层决策进链（前缀由脚本回放，也不进链）"
        # ⚠️ 下层的每一条都必须带**候选集** —— 少了它，这条分支的 logp 重算不了
        assert all(d.cand is not None for d in br.decisions), "下层决策缺候选集"
        # 冻结 ⟹ 全程是那一条规则
        assert {(d.rule_pa, d.rule_la, d.rule_ra) for d in br.decisions} == {rule}
    # ⚠️ **v2 在这里断言"两条分支逐位相同（`B=0`）"—— v3 没有 `B`，那条不成立**：
    # 选不同规则 ⟹ 走**不同的 head** ⟹ 参数不同 ⟹ 行为**可以**不同。
    # 但"可以不同"不等于"必然不同"（λ 的差异可能小到不改变 `argmin`，或两条规则的
    # `argmin` 恰好一致）⟹ **这里不断言相等关系**，只钉住上面那些结构性事实。
