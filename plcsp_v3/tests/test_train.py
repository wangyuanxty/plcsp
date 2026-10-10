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

import pytest

torch = pytest.importorskip("torch")

from plcsp_v3.env import layout as layout_mod                        # noqa: E402
from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v3.env.des import SimConfig, reference_makespan            # noqa: E402
from plcsp_v3.env.instances import load_mk                            # noqa: E402
from plcsp_v3.env.reward import ReferenceObjectives, reward_weights   # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy                    # noqa: E402
from plcsp_v3.hrl.pool import SegmentPool                             # noqa: E402
from plcsp_v3.hrl.train import (Spec, base_chain, group_advantage,    # noqa: E402
                                ma_logp_at, segment_from, train_step)
from plcsp_v3.nn.features import norm_context                         # noqa: E402

DT = 50          # ⚠️ 只为测试提速；设计值见 §10②（20）


def _spec(tau: float = 1.0):
    """⚠️ **必须给一个收紧的 `tau`**：MK01 默认参考解 **TWT=0**，而 `reward_weights`
    对 0 分量**显式报错**（"TWT=0 的实例须先调 τ"）。不静默补假权重。"""
    torch.manual_seed(0)
    inst = load_mk("mk01")
    cfg = SimConfig(tau=tau)
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg)
    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    net = HierarchicalPolicy()
    return Spec(inst, lay, g, dm, cfg, ctx, w), net


@pytest.mark.unit
def test_reward_weights_refuse_a_zero_component():
    """⚠️ 参考目标**任一分量为 0** ⟹ `reward_weights` 必须**显式报错**。

    **这不是障碍，是纪律**：0 分量会让逆量纲权重变成 `1/0`。语料里写明的处置是
    "先调 τ"，**不是**给一个假权重。

    ⚠️ **2026-10-09 改**：本测试原来拿 **MK01 的默认参考解**当"0 分量"的例子
    （表里 τ=2.75 时 TWT 恰好 = 0）。**而那正是被修掉的毛病** ——
    `due_dates.TF_RDD` 的 mk01/mk04/mk07 三个 τ **都恰好在悬崖上**，
    参考解没有拖期 ⟹ 那三个实例跑不起来。修完 τ（mk01 → 2.00 等）后，
    MK01 的默认参考解 TWT = 35.6 ≠ 0，**旧的断言不再成立**。
    **⟹ 改成直接构造 0 分量** —— 这样测的是 `reward_weights` 的行为，
    不再依赖任何一个实例的标定值（标定再变，本测试不受影响）。
    """
    with pytest.raises(ValueError, match="TWT=0"):
        reward_weights((75.9, 4.96, 0.0))
    with pytest.raises(ValueError, match="参考目标值必须为正"):
        reward_weights((0.0, 4.96, 240.0))        # 分量 0 的另一种位置
    # 正分量照常可用
    w = reward_weights((75.9, 4.96, 240.0))
    assert abs(sum(w) - 1.0) < 1e-9


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
    可能不改变 `argmin`** ⟹ 两条分支逐位跑出同一条链。本档实测：

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
        # ⚠️ 下层的每一条都必须带 `param_u` —— 少了它，这条分支的 logp 重算不了
        assert all(d.param_u is not None for d in br.decisions), "下层决策缺 `param_u`"
        # 冻结 ⟹ 全程是那一条规则
        assert {(d.rule_pa, d.rule_la, d.rule_ra) for d in br.decisions} == {rule}
    # ⚠️ **v2 在这里断言"两条分支逐位相同（`B=0`）"—— v3 没有 `B`，那条不成立**：
    # 选不同规则 ⟹ 走**不同的 head** ⟹ 参数不同 ⟹ 行为**可以**不同。
    # 但"可以不同"不等于"必然不同"（λ 的差异可能小到不改变 `argmin`，或两条规则的
    # `argmin` 恰好一致）⟹ **这里不断言相等关系**，只钉住上面那些结构性事实。
