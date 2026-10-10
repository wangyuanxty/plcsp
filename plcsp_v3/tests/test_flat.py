# -*- coding: utf-8 -*-
"""**扁平链两臂**（Q1，设计 §12.2）的机制验收。

⚠️ **本步要证明的核心只有一句**：臂 A 与臂 B **只差优势怎么算**。
其余（采哪些链、同一次前向、同一条更新规则）必须**逐位相同** ——
否则读出来的差异分不清是"critic 有没有用"还是"别的地方不一样"。

⚠️ 测试用 `dt_events=50` 把顶层点压到 2–3 个（**不是**设计值 §10② 的 20），只为提速。

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
from plcsp_v3.env.reward import EQUAL_WEIGHTS   # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy                    # noqa: E402
from plcsp_v3.hrl.flat import (ALL_KINDS, flat_rollout, flat_step)    # noqa: E402
from plcsp_v3.hrl.pool import ChainPool                              # noqa: E402
from plcsp_v3.hrl.train import Spec, base_chain, batched_terms        # noqa: E402
from plcsp_v3.nn.features import norm_context                         # noqa: E402

DT = 50          # ⚠️ 只为测试提速；设计值见 §10②（20）


def _spec(tau: float = 1.0):
    """⚠️ **必须给一个收紧的 `tau`**（与 `test_train.py` 同因）：MK01 的默认参考解 TWT=0，
    而 `reward_weights` 对 0 分量**显式报错**。不静默补假权重。"""
    torch.manual_seed(0)
    inst = load_mk("mk01")
    cfg = SimConfig(tau=tau)
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg)
    # ⚠️ `1/f^ref` 已删（2026-10-11）⟹ 用等权 + 一个固定的 `bounds`
    # （测试只关心「接口通不通」，不关心具体数值）
    w = EQUAL_WEIGHTS
    bounds = ((60.0, 4.0, 0.0), (140.0, 8.0, 300.0))
    return Spec(inst, lay, g, dm, cfg, ctx, w, bounds=bounds)


@pytest.mark.unit
def test_value_head_is_opt_in_and_adds_no_parameters_by_default():
    """⚠️ **默认档不许长出 critic 头** —— 两层路径的 `net.parameters()` 必须一位不变。

    这不是洁癖：`train.train_step` 把 `net.parameters()` 整个交给 Adam。
    默认档多出参数 ⟹ 既有读数（优化器状态、梯度范数）全部作废。

    ⚠️ **2026-10-10：⑪ 去后 `n_chargers` 参数已删** —— 网络不再随布局变宽，
    `HierarchicalPolicy()` **无必需参数**（原写法 `n_chargers=2` 随充电头删除）。
    """
    plain = HierarchicalPolicy()
    assert plain.value_head is None
    assert not [n for n, _ in plain.named_parameters() if n.startswith("value_head")]
    with_critic = HierarchicalPolicy(with_value=True)
    assert with_critic.value_head is not None
    extra = {n for n, _ in with_critic.named_parameters()} - {n for n, _ in plain.named_parameters()}
    assert extra and all(n.startswith("value_head") for n in extra), \
        f"`with_value=True` 只该多出 value_head，实得多出：{extra}"


@pytest.mark.slow
def test_both_arms_roll_out_the_same_chains():
    """⚠️ **本步的关键验收**：两臂采到的是**同一批链**（同种子 ⟹ 逐位相同）。

    **这是"其余完全相同"的硬证据**：读出的差异只可能来自优势那一行。
    做法：把同一个 net 复制两份，各跑一步不同臂，比对两条链的四项读数。
    """
    spec = _spec()
    # ⚠️ 两个 net 必须**同源**：从同一个带头 net 复制两份，各自走一步不同臂。
    torch.manual_seed(0)
    both = HierarchicalPolicy(with_value=True)
    a_net, b_net = copy.deepcopy(both), copy.deepcopy(both)
    out_a = flat_step(a_net, spec, arm="A", n=2, seed=0, dt_events=DT)
    out_b = flat_step(b_net, spec, arm="B", n=2, seed=0, dt_events=DT)
    assert out_a["runs"] == out_b["runs"], "两臂采到的链不同 ⟹ 比出来的差异无法归因于 critic"


@pytest.mark.slow
def test_only_arm_a_trains_the_critic():
    """臂 A：critic 头**拿到**梯度、策略头也拿到；臂 B：critic 头**一点**梯度都没有。"""
    spec = _spec()
    torch.manual_seed(0)
    net = HierarchicalPolicy(with_value=True)
    a = flat_step(copy.deepcopy(net), spec, arm="A", n=2, seed=0, dt_events=DT)
    b = flat_step(copy.deepcopy(net), spec, arm="B", n=2, seed=0, dt_events=DT)
    assert a["grad_norm"]["critic"] > 0.0 and a["grad_norm"]["heads"] > 0.0
    assert a["critic_loss"] > 0.0
    assert b["grad_norm"]["critic"] == 0.0 and b["critic_loss"] == 0.0
    assert b["grad_norm"]["heads"] > 0.0
    # 编码器是**共享主干**：两臂都该浇到它（臂 A 多一份来自 critic 的）
    assert a["grad_norm"]["encoder"] > 0.0 and b["grad_norm"]["encoder"] > 0.0


@pytest.mark.unit
def test_critic_arms_refuse_a_net_without_a_critic_head():
    """⚠️ **先查后跑**：critic 头缺了要当场报错，不能先花几分钟仿真再炸。`A` 与 `A2` 都要。"""
    spec = _spec()
    net = HierarchicalPolicy()
    for arm in ("A", "A2"):
        with pytest.raises(ValueError, match="critic 头"):
            flat_step(net, spec, arm=arm, n=2, seed=0, dt_events=DT)


@pytest.mark.slow
def test_a2_pulls_the_advantage_scale_back_to_b():
    """⚠️ **A2 存在的唯一理由**：把 critic 臂的优势尺度拉回与 B 同量级。

    §98.3 实测：未归一化的 `A` 的 `heads_grad` 比 `B` 大 ~50 倍 ⟹ 同 lr 下步长差 ~50 倍
    ⟹ **A 的失败说明不了 critic 没用**（§101.4）。
    `A2` 做标准 PPO 的 `(A − μ)/σ` 之后，两者应当回到同一量级。

    ⚠️ 用 `grad_norm` 的**比值**做判据（不是绝对值）—— 比值对机器快慢不敏感。
    """
    spec = _spec()
    torch.manual_seed(0)
    net = HierarchicalPolicy(with_value=True)
    a = flat_step(copy.deepcopy(net), spec, arm="A", n=2, seed=0, dt_events=DT)
    a2 = flat_step(copy.deepcopy(net), spec, arm="A2", n=2, seed=0, dt_events=DT)
    b = flat_step(copy.deepcopy(net), spec, arm="B", n=2, seed=0, dt_events=DT)
    ra = a["grad_norm"]["heads"] / b["grad_norm"]["heads"]
    ra2 = a2["grad_norm"]["heads"] / b["grad_norm"]["heads"]
    # ⚠️ **2026-10-11：口径换了，旧的数量门槛失效。**
    # 原断言是 `ra > 10.0`（"未归一化的 critic 尺度该比 `B` 大一个量级"）——
    # 那是 `docs/v2/progress-log.md` §98.3 在**旧口径**下量的（`1/f^ref`，`R ≈ −15`）。
    # 现在 `R ∈ [−1, 0]`（min–max 归一化 + 等权）⟹ `A` 的尺度 ≈ 1，与 `B` **同量级**。
    # ⟹ **只钉住"两个 critic 臂的尺度确实不同、且 A2 把它拉回 B 附近"**，
    # 不再钉那个由旧 `R` 尺度决定的具体倍数。
    assert ra != ra2, f"`A` 与 `A2` 的尺度应当不同（实测 {ra:.3f} vs {ra2:.3f}）"
    assert abs(ra2 - 1.0) < abs(ra - 1.0), (
        f"A2 的归一化应当把尺度拉**回** `B` 附近：实测 |ra−1|={abs(ra - 1.0):.3f} "
        f"vs |ra2−1|={abs(ra2 - 1.0):.3f}")


@pytest.mark.unit
def test_unknown_arm_raises():
    spec = _spec()
    net = HierarchicalPolicy()
    with pytest.raises(ValueError, match="arm="):
        flat_step(net, spec, arm="C", n=2, seed=0, dt_events=DT)


@pytest.mark.slow
def test_policy_and_value_vectors_are_aligned_and_cover_every_decision():
    """`batched_terms` 的两个向量**同序同长**，且长度 = 选中的决策条数。

    ⚠️ 臂 A 的 `−Σ_t (R − V_t)·logp_t` 要求两者**逐元素对齐**；
    一旦错位，梯度打到的就是**别的决策**（不报错、只是学错）——
    最隐蔽的一类 bug，故用测试钉住。
    """
    spec = _spec()
    torch.manual_seed(0)
    net = HierarchicalPolicy(with_value=True)
    base = base_chain(net, spec, greedy=True, seed=0, dt_events=DT)
    dec = base.rec.decisions
    n_pa = sum(1 for d in dec if d.kind == "PA")
    lp_pa, v_pa = batched_terms(net, dec, ("PA",), want_value=False)
    assert lp_pa.shape == (n_pa,), "只选 PA 时长度应等于 PA 的条数"
    assert v_pa is None, "`want_value=False` 时不该算 V"
    lp, v = batched_terms(net, dec, ALL_KINDS, want_value=True)
    assert lp.shape == v.shape == (len(dec),), "全选时两个向量应与决策链等长"


@pytest.mark.slow
def test_flat_rollout_is_deterministic_in_the_seed():
    """同种子 ⟹ 同链。**两臂共用同一批种子**这条口径靠它成立。"""
    spec = _spec()
    torch.manual_seed(0)
    net = HierarchicalPolicy(with_value=True)
    r1 = flat_rollout(net, spec, n=2, seed=7, dt_events=DT)
    r2 = flat_rollout(net, spec, n=2, seed=7, dt_events=DT)
    assert [x[0] for x in r1] == [x[0] for x in r2]
    r3 = flat_rollout(net, spec, n=2, seed=8, dt_events=DT)
    assert [x[0] for x in r1] != [x[0] for x in r3], "换种子该换链（否则种子没接上）"


@pytest.mark.slow
def test_parallel_pool_matches_the_sequential_path():
    """⚠️ **并行档与顺序档必须采到同一批链**（同种子）。

    ⚠️ **但两者不是逐位相同**：并行档每进程 **1 线程**、顺序档用默认线程数 ⟹
    浮点归约的结合次序不同 ⟹ **末位可以差**。故这里比的是**回报是否一致到 1e-4**，
    而不是逐位相等。**这条差异要在报告里注明**（两臂必须同档）。
    """
    spec = _spec()
    torch.manual_seed(0)
    net = HierarchicalPolicy()
    seq = flat_rollout(net, spec, n=3, seed=5, dt_events=DT)
    with ChainPool(net, spec, n_workers=3, dt_events=DT) as pool:
        par = flat_rollout(net, spec, n=3, seed=5, dt_events=DT, pool=pool)
    assert len(par) == len(seq) == 3
    for a, b in zip(seq, par):
        assert a.R == pytest.approx(b.R, abs=1e-4), f"并行与顺序采到的链不同：{a.R} vs {b.R}"
        assert len(a.decisions) == len(b.decisions), "决策条数不同 ⟹ 采的不是同一条链"


@pytest.mark.unit
def test_pool_refuses_a_one_worker_or_a_gpu_setup():
    """⚠️ 两条**先查后跑**的护栏，都不许静默降级。"""
    spec = _spec()
    net = HierarchicalPolicy()
    with pytest.raises(ValueError, match="n_workers"):
        ChainPool(net, spec, n_workers=1, dt_events=DT)      # 1 进程 ⟹ 直接用顺序档
    with pytest.raises(ValueError, match="greedy"):
        # 并行档不给 `greedy=True`（评测要确定性）
        flat_rollout(net, spec, n=2, seed=0, dt_events=DT, greedy=True,
                     pool=object())                          # type: ignore[arg-type]
