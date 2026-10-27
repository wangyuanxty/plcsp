# -*- coding: utf-8 -*-
"""**MS / TC 两支奖励的探针** —— 查 OA 为什么锁死 TC。

逐决策点记录：
```
d_ms    = CM(t) − CM(t−1)        （CM = max(mct_assigned)）
tc_step = TC(t) − TC(t−1)
oa_ms   = OA 选了 MS 支？
x_ms    = 用当前运行极值算的 MS 支归一化得分
x_tc    = 同一时刻 TC 支的归一化得分
```
判据：
- `d_ms == 0` 的占比；
- `x_ms` 与 `x_tc` 的均值；
- `x_ms > x_tc` 的占比（OA 若只看即时奖励，应选 MS 的比例）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

CKPT = os.environ.get("RP_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "nhdrl3_vary.pt"))
N_EP = int(os.environ.get("RP_EP", "5"))


class LogReward(N.RewardFn):
    """在 `RewardFn` 外面包一层，记录两支的即时归一化得分（**更新极值之前**）。"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple] = []

    def __call__(self, d_ms, tc_step, oa_ms, prefs):
        ms_span = self.ms_hi - self.ms_lo
        tc_span = self.tc_hi - self.tc_lo
        x_ms = (self.ms_hi - d_ms) / ms_span if ms_span > 1e-9 else np.nan
        x_tc = (self.tc_hi - tc_step) / tc_span if tc_span > 1e-9 else np.nan
        r = super().__call__(d_ms, tc_step, oa_ms, prefs)
        self.rows.append((float(d_ms), float(tc_step), bool(oa_ms),
                          float(x_ms), float(x_tc), float(r),
                          float(ms_span), float(tc_span)))
        return r


def main() -> None:
    if not os.path.exists(CKPT):
        raise SystemExit(f"⚠️ 找不到 ckpt `{CKPT}`")
    net = N.NHDRL()
    _obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(_obj, dict) and "state_dict" in _obj:
        net.load_state_dict(_obj["state_dict"])
        feat_stats = _obj.get("feat_stats")
    else:
        net.load_state_dict(_obj)
        feat_stats = None
    net.eval()

    inst = eval_instance(140, 16, 20, **N._inst_kw())
    spec = build(seed=0, pm_interval=30, target_failures=4, n_agv=N.NAGV,
                 finite_buffer=True, layout_seed=0, reward_norm="none",
                 tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(200.0, inst.n_machines, layout_cols=N.LAYOUT_COLS),
                 layout_cols=N.LAYOUT_COLS, layout_rows=N.LAYOUT_ROWS,
                 shi2025=N.SHI)[0]
    nf = NHFeatures(inst, spec.layout,
                    speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
    rew = LogReward()
    for ep in range(N_EP):
        r = N.Rollout(net, spec, nf, rew,
                      torch.Generator().manual_seed(ep), eps=0.0,
                      feat_stats=feat_stats)
        w = spec.world()
        m = w.run_gated(seed_chain=9_000_000 + ep, policy_pa=r.policy_pa,
                        policy_l=r.policy_l, dt_events=0)
        r.finish(w.snapshot(), float(m["makespan"]))
    rows = rew.rows
    n = len(rows)
    d_ms = np.array([x[0] for x in rows])
    tc = np.array([x[1] for x in rows])
    oa = np.array([x[2] for x in rows])
    x_ms = np.array([x[3] for x in rows])
    x_tc = np.array([x[4] for x in rows])
    print(f"**MS/TC 奖励探针** · {N_EP} 局 · {n} 个决策点\n", flush=True)
    print(f"  d_ms：均值 {np.nanmean(d_ms):.3f}　中位 {np.nanmedian(d_ms):.3f}"
          f"　= 0 占比 {np.mean(np.abs(d_ms) < 1e-9):.1%}"
          f"　< 0.1 占比 {np.mean(np.abs(d_ms) < 0.1):.1%}")
    print(f"  tc_step：均值 {np.nanmean(tc):.3f}　中位 {np.nanmedian(tc):.3f}"
          f"　= 0 占比 {np.mean(np.abs(tc) < 1e-9):.1%}")
    print(f"\n  OA 选择：MS {np.mean(oa):.1%}　TC {1 - np.mean(oa):.1%}")
    print(f"  x_ms（含首步 NaN 前）：均值 {np.nanmean(x_ms):.3f}"
          f"　中位 {np.nanmedian(x_ms):.3f}　有效 {np.isfinite(x_ms).sum()}/{n}")
    print(f"  x_tc：均值 {np.nanmean(x_tc):.3f}"
          f"　中位 {np.nanmedian(x_tc):.3f}　有效 {np.isfinite(x_tc).sum()}/{n}")
    both = np.isfinite(x_ms) & np.isfinite(x_tc)
    print(f"\n  两支都有效：{both.sum()}/{n}")
    if both.sum():
        print(f"  x_ms > x_tc 的占比：{np.mean(x_ms[both] > x_tc[both]):.1%}"
              f"（OA 若只看即时奖励，应选 MS）")
        print(f"  两支差（x_ms − x_tc）：均值 {np.mean(x_ms[both] - x_tc[both]):+.3f}"
              f"　中位 {np.median(x_ms[both] - x_tc[both]):+.3f}")
    print("\n  按 OA 选择拆：")
    for flag, name in ((True, "MS"), (False, "TC")):
        m_ = oa == flag
        if m_.sum():
            print(f"    {name}：{m_.sum():>5} 点　"
                  f"x_ms 均值 {np.nanmean(x_ms[m_]):.3f}　"
                  f"x_tc 均值 {np.nanmean(x_tc[m_]):.3f}　"
                  f"实得奖励均值 {np.mean([rows[i][5] for i in np.where(m_)[0]]):.3f}")


if __name__ == "__main__":
    main()
