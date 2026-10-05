"""HGS 对位基线的测试（spec §6.1 的 ④ 层）。

钉住五件事：

1. **超参逐字**——原文 §V-A2 的 `L=2 / d_h=128 / d_e=1 / d_z=16 / d_ff=512 / Z=8 / d_v=8`；
2. **环境可行**——随机策略排出无冲突调度（机台不重叠、全部工序排完、时间单调）；
3. **合成实例生成器**符合原文 §V-A1 的区间；
4. **三段分布**归一化、掩码位概率为 0；
5. **奖励 = makespan 下界差**（原文 §IV-B4），且回报望远镜求和 = `Cmax_lb(s_0) − makespan`。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from plcsp.baselines.hgs.env import FjsptEnv
from plcsp.baselines.hgs.model import HgsNet
from plcsp.baselines.hgs.train import gen_fjspt, rollout


def _tiny_env(seed: int = 0) -> FjsptEnv:
    inst = gen_fjspt(3, 3, 2, np.random.default_rng(seed))
    return FjsptEnv(inst, 2)


def _random_policy(env: FjsptEnv, rng: np.random.Generator) -> None:
    while not env.finished:
        acts = [(j, o, env.idle_machines(j, o), env.idle_vehicles())
                for j, o in env.eligible()]
        acts = [(j, o, km, vs) for j, o, km, vs in acts if km and vs]
        if not acts:
            env.advance()
            continue
        j, o, km, vs = acts[0]
        env.step(j, o, int(rng.choice(km)), int(rng.choice(vs)))


@pytest.mark.unit
def test_network_hyperparameters_are_verbatim_from_the_paper():
    """原文 §V-A2 的逐字参数——改动即偏离方法（本基线存在的意义就是逐字）。"""
    net = HgsNet()
    assert net.cfg == {"dh": 128, "de": 1, "dz": 16, "dff": 512,
                       "n_heads": 8, "dv": 8, "layers": 2}
    assert len(net.layers) == 2, "L = 2 编码层"


@pytest.mark.unit
def test_env_random_policy_produces_a_feasible_schedule():
    """随机可行策略：全部工序排完、机台无重叠、后继工序晚于前驱、makespan > 0。"""
    env = _tiny_env()
    rng = np.random.default_rng(1)
    _random_policy(env, rng)
    assert env.finished and env.n_done == env.total_ops
    assert env.makespan > 0.0
    busy: dict[int, list[tuple[float, float]]] = {}
    for j, job in enumerate(env.jobs):
        for o in range(len(job)):
            k, s, e = env.op_machine[j][o], env.start[j][o], env.complete[j][o]
            assert e > s, "完工必须晚于开工"
            if o > 0:
                assert s >= env.complete[j][o - 1] - 1e-9, "工序开始早于前驱完工"
            for s2, e2 in busy.get(k, []):
                assert s >= e2 - 1e-9 or s2 >= e - 1e-9, f"机台 {k} 上的加工区间重叠"
            busy.setdefault(k, []).append((s, e))


@pytest.mark.unit
def test_env_rejects_infeasible_actions():
    """非可行动作必须显式报错（机台不兼容 / 车辆非空闲 / 工序前驱未完工）。"""
    env = _tiny_env(seed=2)
    j, o = env.eligible()[0]
    km = env.idle_machines(j, o)
    bad_k = next((k for k, _ in env.jobs[j][o] if k not in km), None)
    if bad_k is not None:
        with pytest.raises(ValueError):
            env.step(j, o, bad_k, env.idle_vehicles()[0])
    with pytest.raises(ValueError):
        env.step(j, o + 1 if o + 1 < len(env.jobs[j]) else j, km[0], env.idle_vehicles()[0])
    env.step(j, o, km[0], env.idle_vehicles()[0])
    with pytest.raises(ValueError):
        env.step(j, o, km[0], env.idle_vehicles()[0])      # 已排的工序不能再排


@pytest.mark.unit
def test_synthetic_generator_follows_the_paper_bands():
    """原文 §V-A1：每作业工序数 ∈ [0.8m, 1.2m]；加工时间落在 0.8–1.2×该工序均值内。"""
    m = 5
    rng = np.random.default_rng(3)
    for _ in range(5):
        inst = gen_fjspt(4, m, 3, rng)
        for job in inst.jobs:
            assert 0.8 * m - 1 <= len(job) <= 1.2 * m + 1
            for alts in job:
                assert 1 <= len(alts) <= m
                ts = [t for _, t in alts]
                assert max(ts) <= 30.0 * 1.2 + 1e-9 and min(ts) >= 1.0 * 0.8 - 1e-9
        T = inst.trans_time_full
        assert T.shape == (m + 1, m + 1) and np.allclose(np.diag(T), 0.0)


@pytest.mark.unit
def test_action_distributions_are_normalised_and_masked():
    """三段分布各自归一化；不可行的机台/车辆概率为 0。"""
    env = _tiny_env(seed=4)
    net = HgsNet()
    torch.manual_seed(0)
    g = env.graph()
    with torch.no_grad():
        h_op, h_mach, h_veh, h_edge = net.encode(g)
        act = net.decode(h_op, h_mach, h_veh, h_edge, g, torch.zeros(net.cfg["dh"]))
    assert 0 <= act["op_i"] < g.n_ops and g.eligible[act["op_i"]], "选中的工序不可行"
    assert 0 <= act["mach_k"] < g.n_machines and g.compat[act["op_i"], act["mach_k"]]
    assert 0 <= act["veh_u"] < g.n_agv and g.idle_veh[act["veh_u"]]
    assert torch.isfinite(act["logp_op"] + act["logp_mach"] + act["logp_veh"])


@pytest.mark.unit
def test_reward_telescopes_to_the_makespan_bound():
    """回报之和 = `Cmax_lb(s_0) − makespan`（原文 §IV-B4 的望远镜求和）。"""
    env = _tiny_env(seed=5)
    net = HgsNet()
    torch.manual_seed(0)
    lb0 = env.cmax_lb()
    with torch.no_grad():
        out = rollout(env, net, greedy=True)
    assert out["return"] == pytest.approx(lb0 - out["makespan"], abs=1e-6)


@pytest.mark.unit
def test_greedy_rollout_is_deterministic_and_training_moves_weights():
    """贪心 rollout 可复现；一次梯度步必须改变权重（训练链路是活的）。"""
    inst = gen_fjspt(3, 3, 2, np.random.default_rng(6))
    net = HgsNet()
    with torch.no_grad():
        a = rollout(FjsptEnv(inst, 2), net, greedy=True)
        b = rollout(FjsptEnv(inst, 2), net, greedy=True)
    assert a["makespan"] == b["makespan"], "同一网络同一实例的贪心 rollout 不稳定"
    assert a["logp_sum"] == 0.0, "贪心路径不该累计 logp"

    before = [p.detach().clone() for p in net.parameters()]
    torch.manual_seed(0)
    out = rollout(FjsptEnv(inst, 2), net, greedy=False)
    loss = -(out["return"]) * out["logp_sum"]
    loss.backward()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(p.grad, alpha=-1e-4)
    assert any(not torch.equal(x, y) for x, y in zip(before, net.parameters())), \
        "一次梯度步后权重没变——训练链路是死的"
