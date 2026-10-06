# -*- coding: utf-8 -*-
"""能耗**分项审计**（2026-10-06）：期① 各 run 的 `能耗 / makespan` 比值谱，按口径分解。

为什么要本脚本
--------------
实测 `energy / makespan` 在 makespan 70→105 的跨度上只在 **0.0772–0.0784** 浮动（±1.5%），
实验 E 的 Pareto 前沿因此退化成两个点（`progress-log.md` §52.9.3）。一个初判是：
能耗里**随 makespan 线性增长**的部分（机床空载 + AGV 待机 + 车间固定）占比太大。
本脚本把该判断变成可复核的数字——按三种**总量口径**（外加一个反事实）各算一次
`能耗 / makespan`，看哪个口径的比值谱最宽；再逐分项看**谁与 makespan 同向**。

口径
----
| 口径 | 含什么 | 由分项拼法 |
|---|---|---|
| **总能耗**（现奖励口径） | 全含 | `met["energy"]` |
| **总@待机**（反事实，诊断） | 同总能耗，但机床待机功率换成引证源的 *Standby Power* | `_corrected_total` |
| **净能耗 A** | 机床(加工+换型) + AGV(载货+空驶) | `machine proc + setup + agv empty + loaded` |
| **净能耗 B** | 机床加工 + AGV 载货 | `machine proc + agv loaded` |
| **逐分项** | 七项各自 / makespan | 见 `COMPONENTS` |

同时导出**机台时间利用率** `U = Σ 加工分钟 ÷ (机台数 × makespan)` 与逐分项分钟
（`machine_states_min`）——用于回答"是不是实例太小/利用率太低"。

口径纪律（三条）
----------------
1. **不重训**——从 run 目录的 `ckpt.pt` 载最终策略、argmax 评估（`sample=False`）。
2. **口径从该 run 的日志解析**（复用 `m17_reeval.parse_run_spec` / `build_env`）——
   不手写开关表；解析不出即报错，不猜。
3. **评估种子与训练同批**（`EVAL_SEED_BASE + s`，默认 5）——与 `m13._make_eval_fn`
   的评估循环**逐字同参**（本脚本只多取 `energy_breakdown`，评估逻辑仍是 `roll_chain`）。

⚠️ 完成度守卫：`horizon_hit` 的 episode 里 makespan 是**部分完工的最大值**，能耗也按更短的
makespan 计 ⟹ 比值无意义。凡有掐表的 run **不进汇总**（单独列表报告），与 `m17_reeval`
"未跑满不评估"同一纪律。

⚠️ **`f-retrain-*` 必须排除**（`--exclude f-retrain-`）：F 两跑训练在**扰动环境**里，而扰动
**不在 `m13` 日志里** ⟹ 本脚本（与 `m17_reeval`）会按**未扰动**口径重建环境，对它们的策略
是错口径（`experiment-plan.md` §9.9 已写明"`m17_reeval` 不得用于它们"）。

跑法::

    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m18_energy_audit \
        --roots D:/Temp/phase1,D:/Temp/phase1c --exclude f-retrain- --out D:/Temp/energy-audit.json
    # 规模对照（纯 rollout，不训练）：mk01/mk07/mk10 同一套分项口径
    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m18_energy_audit \
        --instances mk01,mk07,mk10 --out D:/Temp/energy-scale.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .algo.group_rel import roll_chain
from .energy import machine_params_for
from .env.des import reference_makespan, rollout
from .env.instances import load_mk
from .m13_train_a import EVAL_SEED_BASE
from .m17_reeval import (COMPLETE_MIN_LINES, DEFAULT_EVAL_SEEDS, build_env, count_lines,
                         parse_run_spec)

# 分项名 → 取数函数（键名与 `_energy_report` 的导出键一致；**不重命名**，便于对拍）
COMPONENTS = ("machine_proc", "machine_idle", "machine_setup",
              "agv_idle", "agv_empty", "agv_loaded", "shop")

# ⚠️ **反事实口径**（诊断用，**不是**本仓模型）：把机床"待机"功率从现用的**空载功率**
# （GFJSPT-MMRS 的 P^u：M1–M4 0.74 / M5–M6 0.24 / M7–M10 0.16 kW）换成该文 **Table 9 的
# Standby Power**（M1–M4 0.54 / M5–M6 0.08 / M7–M10 0.08 kW）。原文里 P^u 是**加工中**的
# 空转功率（P^u = P_idle + P_sp + P_f，见原文 Fig.1/Fig.2 与 §5.6 的换算），
# 与 E_idle（Eq.7–8）用的 P_idle **不是同一个量**。本表只用于量化"参数换成待机值会怎样"，
# **不改 `energy.py` 的任何冻结值**（`met["energy"]` 逐位不变由测试钉死）。
CORRECTED_STANDBY_KW = {"M1_M4": 0.54, "M5_M6": 0.08, "M7_M10": 0.08}


def _corrected_total(met: dict) -> float:
    """反事实总能耗：只换机床待机功率（用 `machine_states_min` 重算），其余项不动。

    ⚠️ **换型功率跟着换**：本仓假设"换型功率 = 待机功率"（`energy.py` 的 assumed 1）⟹
    待机值一变，换型项也按同一比例变，否则反事实内部不自洽。
    """
    bd = met["energy_breakdown"]
    mins = bd["machine_states_min"]
    n_m = len(mins["proc"])
    old_idle = bd["machine_states_kwh"]["idle"]
    old_setup = bd["machine_states_kwh"]["setup"]
    idle_new = setup_new = 0.0
    for m in range(n_m):
        kw = CORRECTED_STANDBY_KW[machine_params_for(m, n_m)["tier"]]
        idle_new += kw * mins["idle"][m] / 60.0
        setup_new += kw * mins["setup"][m] / 60.0
    return float(met["energy"] - old_idle - old_setup + idle_new + setup_new)


def components_of(met: dict) -> dict[str, float]:
    """一次 episode 的 `metrics` → 七个分项 [kWh] + 四个口径 [kWh] + 利用率（**纯读取**）。"""
    bd = met["energy_breakdown"]
    ms_ = bd["machine_states_kwh"]
    ag_ = bd["agv_states_kwh"]
    parts = {"machine_proc": ms_["proc"], "machine_idle": ms_["idle"],
             "machine_setup": ms_["setup"],
             "agv_idle": ag_["idle"], "agv_empty": ag_["empty"],
             "agv_loaded": ag_["loaded"],
             "shop": bd["shop_kwh"]}
    parts["total"] = float(met["energy"])
    parts["total_standby_fix"] = _corrected_total(met)
    parts["net_a"] = (ms_["proc"] + ms_["setup"] + ag_["empty"] + ag_["loaded"])
    parts["net_b"] = ms_["proc"] + ag_["loaded"]
    # 机台时间利用率 U = Σ 加工分钟 ÷ (机台数 × makespan)——"空载占比"的直接读数
    mins = bd["machine_states_min"]
    n_m = len(mins["proc"])
    parts["n_machines"] = float(n_m)
    parts["utilization"] = (sum(mins["proc"]) / (n_m * met["makespan"])
                            if met["makespan"] else 0.0)
    parts["idle_time_share"] = (sum(mins["idle"]) / (n_m * met["makespan"])
                                if met["makespan"] else 0.0)
    return parts


def eval_run(rd: Path, log_path: Path, seeds: int, device: str = "cpu") -> dict:
    """载 `ckpt.pt` → 逐评估种子跑 argmax → 逐 episode 取分项。

    评估调用与 `m13_train_a._make_eval_fn` 的循环**逐字同参**（同 `EVAL_SEED_BASE`、
    `sample=False`、同 `constraints`、同链级开关）；只多取一个 `energy_breakdown`。
    `t3` 不进 `roll_chain`（它只改训练优势，见 `m17_reeval` 模块 docstring 的"`t3`"条）。
    """
    spec = parse_run_spec(log_path.read_text(encoding="utf-8"))
    inst, lay, dm, cfg, ctx, pol, cons = build_env(spec)
    ck = torch.load(rd / "ckpt.pt", map_location="cpu", weights_only=False)
    pol.load_state_dict(ck["model"])
    pol.to(device)
    pol.eval()
    kwargs = {k: v for k, v in spec.chain_kwargs().items() if k != "t3"}
    episodes = []
    for s in range(seeds):
        _, met = roll_chain(inst, lay, dm, cfg, pol, seed=EVAL_SEED_BASE + s,
                            ctx=ctx, sample=False, constraints=cons, **kwargs)
        row = components_of(met)
        row["makespan"] = float(met["makespan"])
        row["horizon_hit"] = bool(met["horizon_hit"])
        row["jobs_done"] = int(met["jobs_done"])
        episodes.append(row)
    return {"run_dir": str(rd), "spec": spec.record(), "ckpt_step": int(ck.get("step", -1)),
            "n_jobs": int(inst.n_jobs), "rule_makespan": float(reference_makespan(inst, cfg)),
            "episodes": episodes}


def eval_rollout_instance(name: str, seed_chains: tuple[int, ...], ) -> dict:
    """纯 rollout（规则档，**不训练**）跑一个实例——用于 mk07/mk10 的规模对照。

    口径与期① 训练 run 的几何口径一致（`SimConfig()` 默认动力学、`constraints=None` 全开、
    同 `seed_layout=0` 的布局）；`seed_chain` 取多个 ⇒ 有种子内极差可看。
    """
    from .env.des import SimConfig

    inst = load_mk(name)
    cfg = SimConfig()
    episodes = []
    for sc in seed_chains:
        met = rollout(inst, seed_chain=sc, cfg=cfg)
        row = components_of(met)
        row["makespan"] = float(met["makespan"])
        row["horizon_hit"] = bool(met["horizon_hit"])
        row["jobs_done"] = int(met["jobs_done"])
        episodes.append(row)
    return {"run_dir": f"rollout:{name}", "spec": {"kind": "rollout", "inst": name,
                                                   "seed_chains": list(seed_chains)},
            "ckpt_step": -1, "n_jobs": int(inst.n_jobs),
            "rule_makespan": float(episodes[0]["makespan"]), "episodes": episodes}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson r（n<3 或零方差时返回 nan——不硬算）。"""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _rank(xs: list[float]) -> list[float]:
    """平均秩（并列取均值）——Spearman 用。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman 秩相关（n<3 时 nan）。比 Pearson 更贴"排序是否被目标改变"这个问题。"""
    if len(xs) < 3:
        return float("nan")
    return _pearson(_rank(xs), _rank(ys))


CALIBERS = ("total", "total_standby_fix", "net_a", "net_b") + COMPONENTS


def summarize(records: list[dict]) -> dict:
    """跨 run 汇总：每个口径的**比值谱**（min/mean/max/展宽%）+ 与 makespan 的 Pearson r。

    ⚠️ 只收**全部 episode 都跑完**的 run（`horizon_hit` 一个都不许有）——掐表的比值无意义。
    逐 run 先对 5 个评估种子取均值，再在 run 之间比较（run 是独立单位，种子是重复测量）。
    """
    ok = [r for r in records if all(not e["horizon_hit"] for e in r["episodes"])]
    out = {"n_runs": len(ok), "n_runs_excluded_horizon_hit": len(records) - len(ok),
           "calibers": {}}
    ms = [_mean([e["makespan"] for e in r["episodes"]]) for r in ok]
    for cal in CALIBERS:
        vals = [_mean([e[cal] for e in r["episodes"]]) for r in ok]
        ratios = [v / m for v, m in zip(vals, ms)]
        rmin, rmax = min(ratios), max(ratios)
        rmean = _mean(ratios)
        out["calibers"][cal] = {
            "ratio_min": rmin, "ratio_mean": rmean, "ratio_max": rmax,
            "spread_pct": (rmax - rmin) / rmean * 100.0 if rmean else float("nan"),
            "pearson_r_with_makespan": _pearson(ms, vals),
            "spearman_r_with_makespan": _spearman(ms, vals),
        }
    return out


def format_tables(records: list[dict], summary: dict) -> str:
    """逐 run 表（比值）+ 汇总表（比值谱 + r）。"""
    lines = []
    head = f"{'run':<20}{'ms':>9}{'U':>7}{'总':>9}{'总@待机':>9}{'净A':>9}{'净B':>9}" + "".join(
        f"{c:>9}" for c in COMPONENTS)
    lines.append("逐 run：能耗口径 / makespan（逐 run 对 5 个评估种子取均值；单位 kWh/min；"
                 "U = 机台时间利用率）")
    lines.append(head)
    for r in records:
        eps = r["episodes"]
        if any(e["horizon_hit"] for e in eps):
            lines.append(f"{r['arm']:<20}{'—— 掐表（horizon_hit），不进汇总 ——':>40}")
            continue
        ms = _mean([e["makespan"] for e in eps])
        u = _mean([e["utilization"] for e in eps])
        cells = [_mean([e[c] for e in eps]) / ms for c in CALIBERS]
        lines.append(f"{r['arm']:<20}{ms:>9.2f}{u:>7.3f}" + "".join(f"{v:>9.5f}" for v in cells))
    lines.append("")
    lines.append("汇总（只含全部 episode 跑完的 run）：比值谱 + 与 makespan 的相关（Pearson / Spearman）")
    lines.append(f"{'口径':<14}{'min':>10}{'mean':>10}{'max':>10}{'展宽%':>9}{'r':>8}{'rho':>8}")
    for cal in CALIBERS:
        s = summary["calibers"][cal]
        lines.append(f"{cal:<14}{s['ratio_min']:>10.5f}{s['ratio_mean']:>10.5f}"
                     f"{s['ratio_max']:>10.5f}{s['spread_pct']:>9.2f}"
                     f"{s['pearson_r_with_makespan']:>8.3f}"
                     f"{s['spearman_r_with_makespan']:>8.3f}")
    lines.append(f"（n={summary['n_runs']} 个 run；掐表被排除 "
                 f"{summary['n_runs_excluded_horizon_hit']} 个）")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="能耗分项审计：四种口径的 能耗/makespan 比值谱")
    ap.add_argument("--roots", default="", help="逗号分隔的批次根目录（各含 <arm>/ 与 logs/<arm>.log）")
    ap.add_argument("--instances", default="",
                    help="逗号分隔的实例名（mk01,…）：纯 rollout（规则档、不训练）跑规模对照")
    ap.add_argument("--rollout-seeds", default="1,2,3,4,5",
                    help="rollout 模式用的 seed_chain（逗号分隔）")
    ap.add_argument("--arms", default="", help="只跑这些 run（逗号分隔）；空 = 自动发现全部")
    ap.add_argument("--exclude", default="", help="排除这些**前缀**（逗号分隔）；"
                    "F 的扰动重训跑必须排除：扰动不在日志里，重建出的是错口径（见模块 docstring）")
    ap.add_argument("--seeds", type=int, default=DEFAULT_EVAL_SEEDS,
                    help=f"评估种子数（默认 {DEFAULT_EVAL_SEEDS}，与期① 批次同批）")
    ap.add_argument("--min-lines", type=int, default=COMPLETE_MIN_LINES,
                    help=f"完成判据：metrics.ndjson 行数下限（默认 {COMPLETE_MIN_LINES}）")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--out", required=True, help="JSON 输出路径")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.arms.split(",") if s.strip()}
    banned = tuple(s.strip() for s in args.exclude.split(",") if s.strip())
    records, skipped = [], []
    if not args.roots and not args.instances:
        ap.error("--roots 与 --instances 至少要给一个")
    for name in [s.strip() for s in args.instances.split(",") if s.strip()]:
        rec = eval_rollout_instance(name, tuple(int(s) for s in args.rollout_seeds.split(",")))
        rec["arm"] = f"rollout-{name}"
        rec["root"] = "rollout"
        records.append(rec)
        print(f"[m18] rollout-{name:<12} ms={_mean([e['makespan'] for e in rec['episodes']]):.2f}"
              f"  total={_mean([e['total'] for e in rec['episodes']]):.3f}"
              f"  U={_mean([e['utilization'] for e in rec['episodes']]):.3f}", flush=True)
    for root in [Path(p) for p in args.roots.split(",") if p.strip()]:
        names = sorted(p.name for p in root.iterdir()
                       if p.is_dir() and (p / "ckpt.pt").exists())
        for name in names:
            if wanted and name not in wanted:
                continue
            if banned and name.startswith(banned):
                skipped.append({"arm": name, "reason": "excluded（口径重建不了，见 docstring）"})
                continue
            rd = root / name
            log_path = root / "logs" / f"{name}.log"
            if not log_path.exists():
                skipped.append({"arm": name, "reason": f"no log: {log_path}"})
                continue
            n_lines = count_lines(rd / "metrics.ndjson") if (rd / "metrics.ndjson").exists() else 0
            if n_lines < args.min_lines:
                skipped.append({"arm": name,
                                "reason": f"incomplete: {n_lines} < {args.min_lines}"})
                continue
            rec = eval_run(rd, log_path, args.seeds, args.device)
            rec["arm"] = name
            rec["root"] = str(root)
            records.append(rec)
            flags = ("hit" if any(e["horizon_hit"] for e in rec["episodes"]) else "ok")
            print(f"[m18] {name:<20} ms={_mean([e['makespan'] for e in rec['episodes']]):.2f}"
                  f"  total={_mean([e['total'] for e in rec['episodes']]):.3f}  {flags}",
                  flush=True)

    summary = summarize(records)
    print("\n" + format_tables(records, summary))
    if skipped:
        print("\n跳过（未跑满/缺日志；不部分评估）：")
        for s in skipped:
            print(f"  {s['arm']}: {s['reason']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"eval_seed_base": EVAL_SEED_BASE, "seeds": args.seeds,
         "min_lines": args.min_lines, "banned_prefixes": list(banned),
         "summary": summary, "runs": records, "skipped": skipped}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[m18] JSON 已写入 {out.resolve()}")


if __name__ == "__main__":
    main()
