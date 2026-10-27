# -*- coding: utf-8 -*-
"""**NHDRL 评测** —— 与基线同协议：20 个场景 → (MS, TC) → Pareto 解集 → SP / HV。

⚠️ **与基线的口径必须逐项一致**（否则比的是口径不是方法）：
- 同一批 20 个 `seed_chain`
- 同一实例 `Exp(a/b)`
- 同一参考点（本脚本**沿用基线脚本算出的那个**，由命令行传入）
"""
import os
import sys

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.mo_metrics import hypervolume_2d, pareto_front, spacing  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))   # 同目录（脚本已入库）
from v3_nhdrl import NHDRL, Rollout, RewardFn, tc_of  # noqa: E402


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_runs = int(os.environ.get("V3_NRUNS", "20"))
    ck = os.environ.get("V3_CKPT", "")
    ref = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else None

    torch.manual_seed(0)
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    from plcsp_v3.env.nhdrl_features import NHFeatures
    nf = NHFeatures(inst, spec.layout)
    net = NHDRL()
    if ck and os.path.exists(ck):
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        print(f"已载入 {ck}", flush=True)
    else:
        print("⚠️ 未载入权重 —— 评的是**随机初始化**的 NHDRL（对照用）", flush=True)
    net.eval()

    scs = [100_000 + 13 * i for i in range(n_runs)]
    P = []
    for sc in scs:
        r = Rollout(net, spec, nf, RewardFn(),
                    torch.Generator().manual_seed(0), greedy=True)
        m = spec.world().run_gated(seed_chain=sc, policy_pa=r.policy_pa,
                                   policy_l=r.policy_l, dt_events=0)
        P.append((float(m["makespan"]), tc_of(m, N_AGV_SHI)))

    F = pareto_front(P)
    print(f"\nNHDRL · Exp({a}/{b}) · {n_runs} 局")
    print(f"  全部点 MS {min(p[0] for p in P):.1f}–{max(p[0] for p in P):.1f}"
          f"　TC {min(p[1] for p in P):.1f}–{max(p[1] for p in P):.1f}")
    print(f"  前沿点数 {len(F)}"
          f"　MS {F[:, 0].min():.1f}–{F[:, 0].max():.1f}"
          f"　TC {F[:, 1].min():.1f}–{F[:, 1].max():.1f}")
    print(f"  SP = {spacing(F):.4f}")
    if ref is not None:
        print(f"  HV = {hypervolume_2d(F, ref):.1f}　（参考点 {ref}）")


if __name__ == "__main__":
    main()


def normalized_hv(P: list, ref: tuple[float, float]) -> float:
    """**归一化后的 HV** —— 落在 [0,1]，与原文的 HV 同量纲。

    做法：先按参考点把两个目标各自除到 (1,1)=参考点，再算 HV 相对 (1,1)。
    原文的 HV ∈ [0.22, 0.82]（见 `citation-cards.md:186`），说明他们也是这么归一的。
    """
    A = np.asarray(P, dtype=float)
    An = np.stack([A[:, 0] / ref[0], A[:, 1] / ref[1]], axis=1)
    Fn = pareto_front([tuple(x) for x in An])
    return hypervolume_2d(Fn, (1.0, 1.0))
