# -*- coding: utf-8 -*-
"""**仲裁器**：用 JobShopLab（独立开源实现）跑标准 JSSP 实例，取 makespan。

跑法照 `jobshoplab/utils/dipatching_benchmark.py`：
    env.reset(); while not env.done: env.step(1 if strategy(env) else 0)
    makespan = env.state.state.time.time

⚠️ **必须用 `jobshoplab-venv/Scripts/python.exe`**（只有它装了 jobshoplab）。
⚠️ **cwd 必须是 `jobshoplab/`**（配置与实例都是相对路径）。
"""
import os
import sys
from pathlib import Path

from jobshoplab import JobShopLabEnv
from jobshoplab.compiler import Compiler
from jobshoplab.compiler.repos import DslRepository
from jobshoplab.utils.load_config import load_config
from jobshoplab.utils.state_machine_utils import dispatch_rules_uitls as du


def fifo(env):
    return True


def spt(env):
    tr = env.state.possible_transitions[0]
    if tr.component_id.startswith("m"):
        best = du.get_machine_prebuffer_jobs_by_processing_time(
            env.instance, env.state.state, mode="SPT")
        return best[tr.component_id] == tr.job_id
    return True


def mwkr(env):
    tr = env.state.possible_transitions[0]
    if tr.component_id.startswith("m"):
        best = du.get_machine_jobs_by_remaining_processing_time(
            env.instance, env.state.state, mode="srpt")
        return best[tr.component_id] == tr.job_id
    return True


STRATS = {"fifo": fifo, "spt": spt, "mwkr": mwkr}


def run(inst_name: str, config) -> None:
    inst_path = (Path(inst_name) if ("/" in inst_name or "\\" in inst_name)
                 else Path(os.path.join("data/jssp_instances/transport", inst_name + ".yaml")))
    repo = DslRepository(inst_path, loglevel="error", config=config)
    compiler = Compiler(config, loglevel="error", repo=repo)
    out = {}
    for name, strat in STRATS.items():
        env = JobShopLabEnv(config=config, compiler=compiler)
        env.reset()
        n = 0
        while not env.done:
            env.step(1 if strat(env) else 0)
            n += 1
            if n > 200000:
                raise RuntimeError("不收敛")
        out[name] = env.state.state.time.time
        if name == "fifo":
            inst = env.instance
            print(f"\n{'=' * 60}\n实例 {inst_name} · {getattr(inst, 'description', '?')}")
            for attr in ("instance", "logistics", "transport"):
                v = getattr(inst, attr, None)
                print(f"  {attr}: {type(v).__name__}"
                      f"{'' if v is None else ' · ' + repr(v)[:300]}")
    print("\n  ⟹ makespan：", {k: out[k] for k in STRATS})
    return out


if __name__ == "__main__":
    cfg = load_config(config_path="data/config/default_config.yaml")
    for nm in (sys.argv[1:] or ["ft06"]):
        run(nm, cfg)
