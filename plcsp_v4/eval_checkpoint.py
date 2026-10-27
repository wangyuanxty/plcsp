# -*- coding: utf-8 -*-
"""评测**一个** checkpoint：100 个官方测试实例 × N 次重复 → mean / SGM。

用法：

```
D:/anaconda/envs/dfjspt/python.exe -m plcsp_v4.eval_checkpoint \
    --ckpt plcsp_v4/training_results/J10_M5_T3/<run>/checkpoint_000500
```

口径（**三个数都报，不再只报一个**）：

- 每实例跑 `--repeats` 次，同时算 **min / mean / median**；
- 每个统计量各自汇总成 **SGM(s=100)**；
- `--greedy` 时策略取 argmax（上游是 `explore=True`，动作也在采样）。

⚠️ 上游 **DRL** 口径是 `np.min`（20 次取最小），上游 **规则** 口径是 `np.mean`
（10 次取均值）—— 两边不对称。min 是“20 次里最好的一次”，不是期望性能。
- 与论文 **642.46 / 发布权重** 比：用 `min`（their protocol）；
- 与 **规则基线** 或我们自己的方法比：两边都用 `mean`，且重复次数相同。
见 `docs/v3/progress-log.md` §38.26。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint 目录（含 policies/）")
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--shift", type=float, default=100.0)
    ap.add_argument("--save", default="",
                    help="把逐实例值存成 .npz（默认存到 <ckpt>/eval_values.npz）")
    ap.add_argument("--greedy", action="store_true",
                    help="策略取 argmax（默认 explore=True，与上游一致）")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)

    # ⚠️ 旧 checkpoint（third_party 下）的 pickle 里模块名是 `DFJSPT`。
    #    这里把它别名到 `plcsp_v4`，两类 checkpoint 都能加载。
    import importlib

    import plcsp_v4
    sys.modules.setdefault("DFJSPT", plcsp_v4)
    for _sub in ("dfjspt_params", "dfjspt_env", "dfjspt_env_for_benchmark",
                 "dfjspt_env_for_imitation", "dfjspt_agent_model",
                 "env_for_rule", "dfjspt_data", "dfjspt_rule"):
        sys.modules[f"DFJSPT.{_sub}"] = importlib.import_module(
            f"plcsp_v4.{_sub}")

    import ray
    from ray.rllib import Policy
    from ray.rllib.models import ModelCatalog

    from .dfjspt_agent_model import (JobActionMaskModel, MachineActionMaskModel,
                                     TransbotActionMaskModel)
    from .dfjspt_env import DfjsptMaEnv

    ckpt = os.path.abspath(args.ckpt)
    if not os.path.isdir(os.path.join(ckpt, "policies")):
        raise FileNotFoundError(f"{ckpt} 不像 RLlib checkpoint（缺 policies/）")

    ray.init(local_mode=False, include_dashboard=False, log_to_driver=False)
    ModelCatalog.register_custom_model("job_agent_model", JobActionMaskModel)
    ModelCatalog.register_custom_model("machine_agent_model", MachineActionMaskModel)
    ModelCatalog.register_custom_model("transbot_agent_model", TransbotActionMaskModel)

    policies = {
        k: Policy.from_checkpoint(os.path.join(ckpt, "policies", f"policy_agent{i}"))
        for i, k in enumerate(("agent0", "agent1", "agent2"))
    }
    env = DfjsptMaEnv({"train_or_eval_or_test": "test"})

    values = []
    for test_id in range(args.instances):
        runs = []
        for _ in range(args.repeats):
            obs, info = env.reset(options={"instance_id": test_id})
            done = False
            stage = next(iter(info["agent0"].values()), None)
            while not done:
                if stage == 0:
                    a = {"agent0": policies["agent0"].compute_single_action(
                        obs=obs["agent0"], explore=not args.greedy)[0]}
                    obs, _r, term, _tr, info = env.step(a)
                    stage = next(iter(info["agent1"].values()), None)
                elif stage == 1:
                    a = {"agent1": policies["agent1"].compute_single_action(
                        obs=obs["agent1"], explore=not args.greedy)[0]}
                    obs, _r, term, _tr, info = env.step(a)
                    stage = next(iter(info["agent2"].values()), None)
                else:
                    a = {"agent2": policies["agent2"].compute_single_action(
                        obs=obs["agent2"], explore=not args.greedy)[0]}
                    obs, _r, term, _tr, info = env.step(a)
                    stage = next(iter(info["agent0"].values()), None)
                    done = term["__all__"]
            runs.append(float(env.final_makespan))
        values.append(runs)
        if (test_id + 1) % 20 == 0:
            print(f"  {test_id + 1}/{args.instances}", flush=True)

    raw = np.asarray(values, dtype=float)          # (n_instances, repeats)
    values_min = raw.min(axis=1)
    values_mean = raw.mean(axis=1)
    values_median = np.median(raw, axis=1)

    def sgm(v):
        return float(np.exp(np.log(v + args.shift).mean()) - args.shift)

    sgm_min, sgm_mean, sgm_median = sgm(values_min), sgm(values_mean), sgm(values_median)
    out = args.save or os.path.join(ckpt, "eval_values.npz")
    np.savez(out, values=values_min, values_min=values_min,
             values_mean=values_mean, values_median=values_median,
             values_all=raw, repeats=args.repeats, shift=args.shift,
             ckpt=ckpt, greedy=bool(args.greedy),
             sgm_min=sgm_min, sgm_mean=sgm_mean, sgm_median=sgm_median,
             sgm=sgm_min)
    print(f"\nckpt: {ckpt}")
    print(f"n={len(raw)}  repeats={args.repeats}  greedy={bool(args.greedy)}")
    print(f"  SGM(s={args.shift:g}):  min={sgm_min:.2f}  "
          f"mean={sgm_mean:.2f}  median={sgm_median:.2f}")
    print(f"  arithmetic:           min={values_min.mean():.2f}  "
          f"mean={values_mean.mean():.2f}  median={values_median.mean():.2f}")
    print(f"saved per-instance values -> {out}")
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
