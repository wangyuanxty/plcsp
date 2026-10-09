# -*- coding: utf-8 -*-
"""**Q1 跑数器**（设计 §12.2–§12.4）——扁平链两臂的读数。

```
python -m plcsp_v2.q1_flat --arms A B --seeds 0 1 2 --steps 200
```

**问的是**：*Learning Without Critics?*（arXiv:2511.03527）称
*"learned critics remain essential for long-horizon tasks"* —— **而调度是长时程。**
臂 A 有 critic（`R − V(s_t)`），臂 B 是现行 GRPO 口径（组内 z 化）。**其余完全相同。**

**测三样**（§12.4）：**训练曲线方差 · 收敛速度 · 扰动（③⑨）下的最终业绩。**
⚠️ **必须多种子** —— 单种子说明不了稳定性，而"稳定性"正是这里要测的东西。

══ 输出 ══

JSONL，每行一条记录（`kind` 区分）：
- `train`：每步的 `step / r_mean / r_std / loss / heads_grad / critic_grad / enc_grad`；
- `eval`：每 `--eval-every` 步的 **greedy 评测**（`makespan / energy / twt` 的均值与标准误）；
- `final`：跑完后的**扰动评测**（⑨ 按 `scaled_agv_mtbf` 激活到每局 ~4 次故障）。

⚠️ **默认写到系统临时目录**（项目硬规则 1：`*.jsonl` 不进项目根）。`--out` 可覆盖。

══ 一条必须写明的口径 ══

**两臂的优势尺度不同，这是构造性的**：臂 B 的 `z(R)` 天生均值 0、标准差 1；
臂 A 的 `R − V(s_t)` 在 `V` 还没学好时**约等于 `R`（≈ −40）**，比 z 值大 ~25 倍。
**不改。** 理由：Adam 的更新是 `lr·m̂/(√v̂+ε)`，**对整体的尺度因子近似不变**，
故两臂的**更新幅度可比**；而尺度差本身就是"学出来的基线 vs 组内基线"的一部分。
⚠️ **这条要在论文里写明**，别让审稿人以为两个梯度范数差 50 倍是我漏了归一化。
（实测 `n=8`：臂 B `heads_grad≈199`，臂 A `heads_grad≈10527`。）
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .env import layout as layout_mod
from .env.constraints import ConstraintConfig
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .env.des import SimConfig, reference_makespan, scaled_agv_mtbf
from .env.instances import load_mk
from .env.reward import ReferenceObjectives, reward_weights
from .hrl.agents import HierarchicalPolicy
from .hrl.flat import flat_rollout, flat_step
from .hrl.pool import ChainPool
from .hrl.train import Spec
from .nn.features import norm_context

DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "q1_flat.jsonl")

TARGET_FAILURES = 4.0
"""⑨ 的**激活口径**（`scaled_agv_mtbf`）：每局期望故障次数。

⚠️ 为什么必须激活：`SimConfig.agv_mtbf` 默认 480 min，而 MK01 一局只有 ~100–170 min
⟹ **默认档下 ⑨ 几乎不发生**，"故障下稳不稳"就测了个空。"""


def build(seed: int, *, tau: float | None = None, disturb: bool = False,
          n_agv: int = 3, with_value: bool = False, device: str = "cpu",
          instance: str = "mk01", pm_interval: float | None = None,
          target_failures: float | None = None, agv_n_failures: int = 0,
          finite_buffer: bool = True, constraints: ConstraintConfig | None = None,
          sim_overrides: dict | None = None):
    """→ `(spec, net)`。**两臂走同一个 `build`** ⟹ 实例/布局/编码器/约束档全同。

    `tau=None`（默认）⟹ **查逐实例标定表**（`due_dates.TF_RDD`，2026-10-09 起 10/10 可用）。
    ⚠️ Q1/D5 早先的轮次用的是**显式 `tau=1.0`**（不查表）—— 两者奖励权重不同，**不可混比**。

    `disturb=True`：⑨ 按 `TARGET_FAILURES` 激活（只用于**最终评测**，不用于训练）。
    `device`：`"cpu"`（默认）/ `"cuda"`。⚠️ **两臂必须同设备**。

    ══ 两条"让约束真的触发"的开关（见 progress-log §110–§112）══
    - `pm_interval`：⑫ 保养间隔（主轴分钟）。默认 120；**MK01 一局只有 ~76 min ⟹ 永不触发**，
      **dev 档用 30**。
    - `target_failures`：⑨ 按**每局期望故障次数**反推 `agv_mtbf`（`scaled_agv_mtbf`）。
      **对位论文 Shi 2025 就是按次数注入（每实例 1–8 次），不设 MTBF** ⟹ 比固定 480 min 有依据。
    ⚠️ **2026-10-10：⑪ 充电机制删除 ⟹ `soc0` 开关一并删**（车队无电量）。
    """
    torch.manual_seed(seed)
    inst = load_mk(instance)
    cfg = SimConfig(tau=tau, n_agv=n_agv)
    # ⚠️ `SimConfig` / `AgvSpec` 都是**普通 dataclass** ⟹ 用 `dataclasses.replace`，
    # 不用 `_replace`（那是 NamedTuple 的方法，写错是 AttributeError 而不是静默）。
    mtbf = scaled_agv_mtbf(inst, TARGET_FAILURES) if disturb else None
    if target_failures is not None:
        mtbf = scaled_agv_mtbf(inst, target_failures)
    over = {}
    # ⚠️ `sim_overrides`（2026-10-10 加）：**机制消融**用的通用 `SimConfig` 覆盖口
    # （如 `{"agv_failover": False}` / `{"machine_age_failure": False}`）。
    # **并进 `over` ⟹ 参考解与奖励权重也用它**（它们必须与仿真的 cfg 同源，否则归一化错位）。
    # 未知键由 `dataclasses.replace` 当场 `TypeError` —— **不静默忽略**。
    if sim_overrides:
        over.update(sim_overrides)
    if mtbf is not None:
        over["agv_mtbf"] = mtbf
    if pm_interval is not None:
        over["pm_interval"] = float(pm_interval)
    if agv_n_failures:
        # ⚠️ **按次数注入**（口径照 Shi et al. 2025）—— 比 `target_failures`（只定期望）更接近对位论文。
        # ⚠️ **抽签上界必须在这里算**：`_build_entities` 自己算会递归（参考解也走那条路）。
        # 而且用**不含注入**的 cfg 算 —— 否则"注入改动力学 ⟹ 改 T ⟹ 改注入"自指。
        over["agv_fail_horizon"] = float(
            reference_makespan(inst, dataclasses.replace(cfg, **over) if over else cfg))
        over["agv_n_failures"] = int(agv_n_failures)
    if over:
        cfg = dataclasses.replace(cfg, **over)
    lay = layout_mod.sample_layout(inst.n_machines, seed=seed, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    # ② 有限缓冲：**约束对象要同时给仿真与特征层**（两份不同源 = 假信号）。
    # ⚠️ **2026-10-10：② 的默认值翻成 `True`**（用户裁定「在报告档打开」）⟹
    # 这里**恒传一个真对象**，不再用 `None`。`None` 的语义是"全开"，
    # 而"全开"现在**恰好等于** `ConstraintConfig()` —— 那就直接给它。
    #
    # ⚠️ `constraints=`（2026-10-10 加）：给**整个**约束对象，供**约束组消融**用
    # （`ABLATION_GROUPS` 的五个组）。给了它就忽略 `finite_buffer`。
    cons = (constraints if constraints is not None
            else ConstraintConfig(finite_buffer=bool(finite_buffer)))
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg, constraints=cons)
    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    # ⚠️ **2026-10-10：⑪ 去后构造网络不再要 `n_chargers`**（充电头是唯一随布局变宽的头）。
    net = HierarchicalPolicy(with_value=with_value)
    # ⚠️ **设备跟随策略**（`rollout.net_device`）：搬一次，之后 rollout/重算自己按参数所在设备走。
    return Spec(inst, lay, g, dm, cfg, ctx, w, cons), net.to(device)


def evaluate(net: HierarchicalPolicy, spec: Spec, *, n: int, seed: int,
             dt_events: int) -> dict:
    """**greedy** 评测 `n` 条链 → 三目标的均值与标准误（越小越好）。"""
    prev = net.training
    net.eval()                                   # 关 dropout/BN（若有）——评测要确定性
    try:
        runs = flat_rollout(net, spec, n=n, seed=seed, dt_events=dt_events, greedy=True)
    finally:
        net.train(prev)
    f = np.array([x.f for x in runs], dtype=np.float64)
    return {"n": n,
            "makespan": float(f[:, 0].mean()), "makespan_se": float(f[:, 0].std(ddof=1)
                                                                   / np.sqrt(n)) if n > 1 else 0.0,
            "energy": float(f[:, 1].mean()),
            "twt": float(f[:, 2].mean()),
            "R": float(np.mean([x.R for x in runs]))}


CONSTRAINT_KEYS = ("fail_events", "tardy_twt", "agv_fail_events",
                   "buffer_block_min", "pm_events")
"""**五条约束**的触发计数（③机障 · ⑧拖期 · ⑨AGV故障 · ②有限缓冲 · ⑫保养）。

⚠️ **2026-10-10：⑪ 充电删除 ⟹ `charge_events` 移出本表**，`buffer_block_min` 移入 ——
本表的条目**与 `ConstraintConfig` 的五个开关一一对应**，不是历史编号的搬运。
"""


def constraint_counts(spec: Spec) -> dict:
    """**规则档**（不给任何策略）跑一局，回报五条约束各自的触发计数。

    ⚠️ **必须用规则档**（§115.3）：策略档下 `pm_events` 数的是
    **"策略多爱用这个动作"**（随机初始化的网会乱保养，实测某布局 67 次），
    与**"约束是否 binding"是两回事**。规则档才是"约束按设计触发"的口径。
    （⑪ 充电曾是本条最刺眼的例子；它已整条删除。）

    ⚠️ 它**另外跑一局仿真**（不带策略 ⟹ 很便宜，§99 实测 ~0.02–2 s）。
    为什么值得：**"跑通了但约束一次都没触发"是看不出来的** —— 那正是这几轮反复踩的坑
    （§110：MK01 上 ⑨⑪⑫ 全不触发，而所有测试都是绿的）。
    """
    m = spec.world().run_gated(seed_chain=0)
    return {k: float(m.get(k, 0)) for k in CONSTRAINT_KEYS}


def run_one(arm: str, seed: int, *, steps: int, n: int, dt_events: int,
            eval_every: int, eval_n: int, rec, device: str = "cpu",
            workers: int = 0, save_dir: str = "", env: dict | None = None) -> None:
    """一条 `(臂, 种子)` 的完整训练 + 评测。`rec` = 写一行的回调。

    `workers > 1` ⟹ 多进程并行采链（`hrl/pool.py`，**只支持 CPU**）。
    ⚠️ **并行档与单进程档不是逐位相同**（每进程 1 线程 ⟹ 浮点归约次序不同）⟹
    **同一次对比里两臂必须同档**，且报出的数字要注明档位。
    """
    env = dict(env or {})
    spec, net = build(seed, with_value=(arm in ("A", "A2")), device=device, **env)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    rec({"kind": "start", "arm": arm, "seed": seed, "steps": steps, "n": n,
         "dt_events": dt_events, "device": device, "workers": workers,
         "n_params": sum(p.numel() for p in net.parameters())})
    base = evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)
    rec({"kind": "eval", "arm": arm, "seed": seed, "step": 0, **base})

    pool = None
    if workers > 1:
        if device != "cpu":
            raise ValueError("并行档只跑 CPU（一个 GPU 上开多上下文只会互抢）——"
                             "`--workers` 与 `--device cuda` 二选一")
        pool = ChainPool(net, spec, n_workers=workers, dt_events=dt_events)
    t0 = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            out = flat_step(net, spec, optim, arm=arm, n=n, seed=seed * 10000 + step,
                            dt_events=dt_events, pool=pool)
            gn = out["grad_norm"]
            rs = [r["R"] for r in out["runs"]]
            # ⚠️ `flat_step` 返回的 loss 是**整组 8 条链的和**（真实被优化的量）。这里**除以 `n`**
            # 再记，否则两个数会读错：
            #   · 臂 A 的 `loss ≈ −9.4e4` 看着吓人，其实是 8 条链之和（每条 ≈ −1.2e4）；
            #   · 臂 B 的 `loss ≈ −12.6` **不是"没学"** —— 组内 z 化使 `Σ_i z_i ≡ 0`，
            #     8 条链的贡献**互相抵消**。**这才是 GRPO 的正常形状**，别当成 bug。
            rec({"kind": "train", "arm": arm, "seed": seed, "step": step,
                 "r_mean": float(np.mean(rs)),
                 "r_std": float(np.std(rs, ddof=1)) if n > 1 else 0.0,
                 "loss_per_chain": out["loss"] / n,
                 "policy_loss_per_chain": out["policy_loss"] / n,
                 "critic_loss_per_chain": out["critic_loss"] / n,
                 # ⚠️ `critic_loss_per_chain` ≈ `E[(V−R)²]`：初值 ≈ R² ≈ 1300 ⟹ 说明 **`V` 从 0 附近起步**
                 # （critic 头是默认小初值）。随训练下降才说明 critic 在学。
                 "heads_grad": gn["heads"], "critic_grad": gn["critic"],
                 "enc_grad": gn["encoder"],
                 "makespan": float(np.mean([r["makespan"] for r in out["runs"]])),
                 "elapsed_s": time.perf_counter() - t0})
            if eval_every and step % eval_every == 0:
                ev = evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)
                rec({"kind": "eval", "arm": arm, "seed": seed, "step": step, **ev})
    finally:
        if pool is not None:
            pool.close()

    # 最终：**扰动**（⑨ 激活）下的业绩
    dspec, _ = build(seed, disturb=True, with_value=(arm in ("A", "A2")), **env)
    dist = evaluate(net, dspec, n=eval_n, seed=1, dt_events=dt_events)
    rec({"kind": "final", "arm": arm, "seed": seed, "total_s": time.perf_counter() - t0,
         "clean": evaluate(net, spec, n=eval_n, seed=1, dt_events=dt_events),
         "disturbed": dist, "agv_mtbf": dspec.cfg.agv_mtbf,
         # ⚠️ **规则档**下的五条约束触发数（§115.3）—— 让"跑通了但约束没触发"当场可见
         "constraints": constraint_counts(spec)})

    if save_dir:
        # ⚠️ **必须存盘**：不然训好的策略丢了 ⟹ 后面要"读出 `B` 矩阵"（v2 实验计划 D6）、
        # 要"同策略在扰动环境里重评"（H 敏感性）全都做不了。§103.5 就是这么发现缺口的。
        # 连同**重建网络要的全部元信息**一起存（`with_value` / 配置），
        # 否则拿到 `.pt` 也重建不出同构的网。
        p = Path(save_dir) / f"{arm}_s{seed}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "arm": arm, "seed": seed, "steps": steps, "n": n,
                    "dt_events": dt_events, "device_trained_on": device,
                    "with_value": arm in ("A", "A2"),
                    "tau": spec.cfg.tau}, p)
        rec({"kind": "saved", "arm": arm, "seed": seed, "path": str(p)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Q1：扁平链两臂（设计 §12.2–§12.4）")
    ap.add_argument("--arms", nargs="+", default=["A", "B"], choices=["A", "A2", "B"],
                    help="A=critic(原始优势) · A2=critic+优势归一化 · B=组内 z 化")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8, help="组大小（设计 §12.2 的 G）")
    ap.add_argument("--dt-events", type=int, default=20, help="顶层定时器步长（§10②）")
    ap.add_argument("--eval-every", type=int, default=25, help="0 = 不中途评测")
    ap.add_argument("--eval-n", type=int, default=3, help="评测链数（greedy）")
    ap.add_argument("--device", default="cpu", help="cpu / cuda（⚠️ 必须用 GPU 版解释器才认 cuda）")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch 线程数（默认 1）。⚠️ 实测 §99：24 线程比 1 线程慢 2.7×")
    ap.add_argument("--workers", type=int, default=0,
                    help="多进程并行采链的进程数（0 = 顺序档）。仅 CPU；每进程 1 线程")
    ap.add_argument("--save-dir", default="",
                    help="存训练好的策略（.pt）；空 = 不存。⚠️ 不存就无法做 D6/H")
    # ── ⚠️ 让约束真的触发（§110–§112）──
    ap.add_argument("--instance", default="mk01", help="mk01–mk10")
    ap.add_argument("--tau", type=float, default=None,
                    help="交期 τ；None = 查逐实例标定表（10/10 可用）")
    ap.add_argument("--pm-interval", type=float, default=None,
                    help="⑫ 保养间隔（主轴分钟）。默认 120；dev 档用 30（MK01 一局只 76 min）")
    ap.add_argument("--no-finite-buffer", dest="finite_buffer", action="store_false",
                    help="② **关掉**有限缓冲（**默认是开的**，见 progress-log §118）。"
                         "关 ⟹ 缓冲无界，退回 2026-10-10 之前的动力学（消融用）")
    ap.add_argument("--agv-n-failures", type=int, default=0,
                    help="⑨ 按次数注入（0 = 关，用 agv_mtbf 泊松流）。照 Shi 2025 口径")
    ap.add_argument("--target-failures", type=float, default=None,
                    help="⑨ 按「每局期望故障次数」反推 agv_mtbf（对位论文 Shi 2025 同口径）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    if a.threads:
        # ⚠️ **只在跑数器里设**，不进库：改线程数会改浮点归约的结合次序 ⟹ 末位可能不同，
        # 而门禁与既有读数依赖 CPU 默认档**逐位不变**。实测（§99）：24 线程 5.72 s，
        # 8 线程 2.70 s —— 小张量被线程调度拖垮，不是算力不够。
        torch.set_num_threads(a.threads)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        # ⚠️ **快速失败**：CPU-only 的 torch 上写 `--device cuda` 静默退回 CPU
        # ⟹ 你以为在用 GPU，实际跑了 10 小时 CPU。（本仓禁用静默降级。）
        print(f"✗ 这个解释器没有 CUDA（torch {torch.__version__}）。"
              "GPU 请用 D:/anaconda/envs/py312/python.exe", file=sys.stderr)
        return 2

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ **断点续跑**：跳过已经写出 `final` 的 `(臂, 种子)`。
    # 为什么需要：一次全量是小时级，而这台笔记本目前**电源功率不足（电池 1%）**——
    # 跑到一半掉电会整个丢失，跑数器本身不写中间检查点。
    done: set[tuple[str, int]] = set()
    if out.exists():
        for ln in out.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "final":
                done.add((str(d["arm"]), int(d["seed"])))
        if done:
            print(f"续跑：已完成 {sorted(done)}，跳过", file=sys.stderr)
    with out.open("a", encoding="utf-8") as fh:
        def rec(d: dict) -> None:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps(d, ensure_ascii=False), flush=True)

        for seed in a.seeds:
            for arm in a.arms:
                if (arm, seed) in done:
                    continue
                run_one(arm, seed, steps=a.steps, n=a.n, dt_events=a.dt_events,
                        eval_every=a.eval_every, eval_n=a.eval_n, rec=rec,
                        device=a.device, workers=a.workers, save_dir=a.save_dir,
                        env={"instance": a.instance, "tau": a.tau,
                             "pm_interval": a.pm_interval,
                             "target_failures": a.target_failures,
                             "agv_n_failures": a.agv_n_failures,
                             "finite_buffer": a.finite_buffer})
    print(f"\n写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
