# -*- coding: utf-8 -*-
"""**强制 OA 走某一支**，看回报差多少 —— 回答「它为什么 100% 选 TC」。

══ 结构问题 ══

`Rollout._close` 把**同一个** `r` 发给三个缓冲，而 `r` 的算式由 `oa_a` 选：

    r = RewardFn(d_ms, tc_step, oa_ms=(oa_a == 0), prefs)

⟹ **OA 选哪支，就决定了它自己收到什么奖励。这是自指结构。**

判据：
- 强制 MS 的 `ΣR` > 强制 TC 的 `ΣR` ⟹ **OA 选 TC 是学错了**（它在最小化自己的回报）。
- 反之 ⟹ OA 没学错，**是奖励口径本身把 TC 变成了更划算的一支** —— 那是**口径错**，不是学习错。
- 同时看两支的**终局 makespan** —— 奖励高的一支**未必**调度好。

⚠️ `tc_step` 已按 `UC_TRANSPORT` 换回分钟（见 `tc_cum` 的注释），故两支同量纲、可比。
"""
import os
import sys

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))   # 同目录（脚本已入库）
from v3_nhdrl import NHDRL, Rollout, RewardFn  # noqa: E402


class ForcedOA(Rollout):
    """把 OA 钉死在某一支上；PA / LA 照常由网络出。"""

    def __init__(self, *a, force: int | None = None, **k) -> None:
        super().__init__(*a, **k)
        self.force = force

    def _pick(self, key, f):
        if key != "oa" or self.force is None:
            return super()._pick(key, f)
        head = self.net.heads()[key]
        ft = torch.as_tensor(f)[None]
        with torch.no_grad():
            pr = torch.softmax(head.actor(ft)[0], -1)
            a = int(self.force)
            lp = float(torch.log(pr[a].clamp_min(1e-12)))
            v = float(head.value(ft)[0])
        return a, lp, v


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_sc = int(os.environ.get("V3_N_SC", "8"))
    ck = os.environ.get("V3_CKPT", "")
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    net = NHDRL()
    if ck and os.path.exists(ck):
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        print(f"已载入 {ck}")
    net.eval()

    print(f"\n{'支':<8}{'ΣR 均值':>12}{'makespan 均值':>15}{'MS 最小':>10}")
    print("  " + "-" * 45)
    for force, nm in ((0, "MS"), (1, "TC"), (None, "网络自选")):
        rew = RewardFn()
        Rs, MKs = [], []
        for i in range(n_sc):
            r = ForcedOA(net, spec, nf, rew, torch.Generator().manual_seed(0),
                         greedy=True, force=force)
            m = spec.world().run_gated(seed_chain=100_000 + 13 * i,
                                       policy_pa=r.policy_pa, policy_l=r.policy_l,
                                       dt_events=0)
            Rs.append(sum(float(row[4]) for row in r.buf["oa"]))
            MKs.append(float(m["makespan"]))
        print(f"  {nm:<8}{np.mean(Rs):>12.2f}{np.mean(MKs):>15.2f}{min(MKs):>10.1f}")

    print("\n  ⟹ 判据：若「MS 的 ΣR」>「TC 的 ΣR」，则 OA 选 TC 是**学错了**；")
    print("     反之则是**奖励口径**把 TC 变成更划算的一支（口径错，不是学习错）。")


if __name__ == "__main__":
    main()
