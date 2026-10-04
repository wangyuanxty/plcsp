"""链级多进程池的测试（2026-10-04 并行批次，`plcsp/algo/chain_pool.py`）。

判据分三层：
1. **序列化**：`Decision` 的全部字段都能 pickle 往返（硬要求 7——不许丢字段）；
2. **口径**：`parallel` 默认 `False`；`parallel=True, pool=None` 显式报错
   （硬要求 5——不许静默退回串行）；worker 数量非法时建池即报错；
3. **逐位等价**：worker 跑出来的链与"主进程里逐链独立采样流"的参照**逐位相同**
   （隔离变量：两边用同一条逐链流 `seed*SEED_STRIDE+g`，只差"在哪个进程跑"——
   这正是把采样流口径变化（第九次读数作废）与并行实现本身分开的那次对照）。

⚠️ 池用 spawn 起真实子进程，故本文件的 integration 用例比普通单测慢（约数秒）。
"""
from __future__ import annotations

import inspect
import pickle

import numpy as np
import pytest
import torch

from plcsp.algo.chain_pool import ChainWorkerPool
from plcsp.algo.group_rel import SEED_STRIDE, Decision, joint_chain_step, roll_chain
from plcsp.algo.policy import PolicyNet
from plcsp.algo.setup import build_setup
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig
from plcsp.env.instances import gen_random
from plcsp.env.reward import ReferenceObjectives
from plcsp.nn.encoder import LayoutEncoder


@pytest.fixture(scope="module")
def tiny():
    """小合成实例上的整套环境（跑得快，但链上有几十个决策，判据不失意义）。

    ⚠️ 交期 `(τ, R)` **显式传**：合成实例不在 `due_dates` 的标定表里（⑧ 口径是外生量，
    表只覆盖 mk01–mk10），不传会在参考运行处显式报错。测试只要求训练/仿真两侧同一份 cfg，
    故取一组合法值即可。
    """
    inst = gen_random(3, 2, seed=0)
    cfg = SimConfig(tau=0.9, due_range=0.5)
    lay, dm, ctx = build_setup(inst, cfg)
    pol = PolicyNet(enc=LayoutEncoder())
    ref = ReferenceObjectives.of(inst, cfg)
    return inst, lay, dm, cfg, ctx, pol, ref


@pytest.fixture(scope="module")
def pool(tiny):
    _inst, _lay, _dm, _cfg, _ctx, pol, _ref = tiny
    p = ChainWorkerPool(pol, n_workers=2)
    try:
        yield p
    finally:
        p.close()


@pytest.mark.unit
def test_decision_pickle_roundtrip_keeps_all_fields():
    """决策里的字段都是 CPU numpy/标量——pickle 往返后逐字段相等（不许丢字段）。"""
    d = Decision(kind="L", tok=np.arange(20 * 11, dtype=np.float32).reshape(20, 11),
                 seg=(2, 3, 1, 1), feat=np.ones(4, dtype=np.float32),
                 cand_feat=np.full((3, 1), 0.5, dtype=np.float32), cand=(0, 1, 2),
                 action=2, tok_idx=np.asarray([5, 6, 7]), logp=-1.25, mach=None, agv=1)
    e = pickle.loads(pickle.dumps(d))
    assert (e.kind, e.seg, e.cand, e.action, e.logp, e.mach, e.agv) == \
           (d.kind, d.seg, d.cand, d.action, d.logp, d.mach, d.agv)
    for field in ("tok", "feat", "cand_feat", "tok_idx"):
        assert np.array_equal(getattr(e, field), getattr(d, field)), field


@pytest.mark.unit
def test_parallel_defaults_to_off():
    """硬要求 1：默认档必须是原串行路径（逐位不变）——默认值就是那条防线。"""
    sig = inspect.signature(joint_chain_step)
    assert sig.parameters["parallel"].default is False
    assert sig.parameters["pool"].default is None


@pytest.mark.unit
def test_parallel_without_pool_raises():
    """硬要求 5：`parallel=True` 而没给池 —— 显式报错，不静默退回串行。"""
    with pytest.raises(ValueError, match="pool=None"):
        joint_chain_step(object(), None, None, None, None, None, None,
                         seed=0, G=1, parallel=True)


@pytest.mark.unit
def test_pool_rejects_bad_worker_count(tiny):
    _inst, _lay, _dm, _cfg, _ctx, pol, _ref = tiny
    with pytest.raises(ValueError, match="n_workers"):
        ChainWorkerPool(pol, n_workers=0)


@pytest.mark.integration
def test_pool_starts_the_requested_number_of_workers(tiny, pool):
    """硬要求 5/6：池起来了、数量对（启动探针在 `__init__` 里核对过 pid）。"""
    assert pool.n_workers == 2
    assert pool.startup_s is not None and pool.startup_s > 0.0
    # 硬要求 3：镜像参数必须在**共享内存**里（worker 直接读，每步零参数 IPC）。
    assert all(p.is_shared() for p in pool.mirror.parameters()), "镜像参数不在共享内存"


def _per_chain_reference(inst, lay, dm, cfg, ctx, pol, seed, G):
    """主进程参照：逐链独立采样流（与并行档同一条流），只差"在哪个进程跑"。"""
    out = []
    for g in range(G):
        gen = torch.Generator(device="cpu").manual_seed(seed * SEED_STRIDE + g)
        out.append(roll_chain(inst, lay, dm, cfg, pol, seed * SEED_STRIDE + g, ctx,
                              sample=True, generator=gen))
    return out


def _assert_chains_equal(refs, got):
    for g, ((dec_r, met_r), (dec_p, met_p)) in enumerate(zip(refs, got)):
        assert met_p == met_r, f"链 {g} 的指标不同——仿真/在线前向没在同一口径上"
        assert len(dec_p) == len(dec_r) > 10, f"链 {g} 决策数不同或太短"
        for a, b in zip(dec_r, dec_p):
            assert (a.kind, a.cand, a.action, a.logp, a.seg, a.mach, a.agv) == \
                   (b.kind, b.cand, b.action, b.logp, b.seg, b.mach, b.agv)
            for field in ("tok", "feat", "cand_feat", "tok_idx"):
                assert np.array_equal(getattr(a, field), getattr(b, field)), \
                    f"链 {g} 的 {field} 不同"


@pytest.mark.integration
def test_parallel_matches_per_chain_stream_reference(tiny, pool):
    """隔离对照：**同一条逐链采样流**下，worker 的结果与主进程参照逐位相同。

    串行档（`joint_chain_step(parallel=False)`）G 条链**共用一条**流，数值本就与并行档
    不同（第九次读数作废，见模块 docstring）——故这里的参照是"主进程里逐链独立流"，
    它只差"在哪个进程跑"这一个变量。相同即证明 IPC 没丢字段、worker 的 CPU 路径与主进程
    一致。**跑两轮**（第二轮前扰动参数并 `sync_policy`）——证明主进程的参数刷新真的到了
    worker（否则第二轮仍会按旧参数算，逐位判据当场红）。
    """
    inst, lay, dm, cfg, ctx, pol, _ref = tiny
    seed, G = 3, 2
    for rnd in range(2):
        if rnd == 1:                      # 扰动参数：镜像不同步就会露馅
            with torch.no_grad():
                for p in pol.parameters():
                    p.add_(0.01)
        refs = _per_chain_reference(inst, lay, dm, cfg, ctx, pol, seed, G)
        pool.sync_policy(pol)             # 主进程 → 共享内存镜像（原地 copy_）
        got = pool.run_chains(seed, G, inst=inst, layout=lay, dm=dm, cfg=cfg, ctx=ctx)
        assert len(got) == G, "返回条数不等于 G"
        _assert_chains_equal(refs, got)


@pytest.mark.integration
def test_parallel_joint_step_updates_parameters(tiny, pool):
    """整步（并行）：worker 出链 → 主进程重算/反向/Adam——参数必须真的被更新。"""
    inst, lay, dm, cfg, ctx, pol, ref = tiny
    before = [p.detach().clone() for p in pol.parameters()]
    r, diag = joint_chain_step(pol, inst, lay, dm, cfg, ctx, ref, seed=11, G=2,
                               lr=1e-3, parallel=True, pool=pool)
    assert np.isfinite(r)
    for key in ("loss", "ratio", "clipped_frac", "grad_norm", "r_mean", "r_std", "A_std"):
        assert key in diag, f"诊断缺 {key}"
    assert any(not torch.equal(a, b) for a, b in zip(before, pol.parameters())), \
        "并行档跑完一步后参数未更新——主进程的反向/优化器步没接上"


@pytest.mark.integration
def test_run_training_owns_and_closes_the_pool(tiny, tmp_path):
    """runner 的并行入口：建池 → 每步透传 (parallel, pool) → finally 关池。"""
    from plcsp.algo.runner import run_training

    inst, lay, dm, cfg, ctx, pol, ref = tiny
    out = run_training(pol, inst, steps=1,
                       step_kwargs=dict(layout=lay, dm=dm, cfg=cfg, ctx=ctx, ref=ref, G=2),
                       seed0=0, run_dir=str(tmp_path / "run"), save_every=1,
                       parallel=True, n_workers=2)
    assert out["step"] == 1
    assert (tmp_path / "run" / "metrics.ndjson").exists(), "并行档没走通 runner 的落盘路径"


@pytest.mark.integration
def test_pool_failure_is_explicit_and_then_pool_is_closed(tiny):
    """硬要求 5：worker 抛异常 → 显式 RuntimeError（不退回串行），且池被终止不再可用。"""
    inst, lay, dm, cfg, ctx, pol, _ref = tiny
    p = ChainWorkerPool(pol, n_workers=1)
    try:
        # route_k=2 而 ① 拥堵关：`roll_chain` 在 worker 里显式报错（入口守卫）。
        with pytest.raises(RuntimeError, match="不退回串行"):
            p.run_chains(0, 1, inst=inst, layout=lay, dm=dm, cfg=cfg, ctx=ctx,
                         constraints=ConstraintConfig().with_off("congestion"), route_k=2)
        with pytest.raises(RuntimeError, match="已关闭/终止"):
            p.run_chains(0, 1, inst=inst, layout=lay, dm=dm, cfg=cfg, ctx=ctx)
    finally:
        p.close()
    p.close()          # 清理路径幂等
