# -*- coding: utf-8 -*-
"""事后重评（**不训练**）：从 run 目录的 `ckpt.pt` 重算三目标——makespan / energy / TWT。

为什么有本脚本
--------------
`m13_train_a._make_eval_fn` 在 2026-10-05 之前**只记 makespan**。期① 的 13 个消融 run
（MK01、300 步、seed 0）**不会再跑第二遍**（一个 run 60–77 min），`metrics.ndjson` 里的内联
评估只有 makespan ⟹ energy 与 TWT 的消融对照只能**事后**从最终策略重算。本脚本做这件事。

口径纪律（三条；缺一条就会**静默**错位）
----------------------------------------
1. **开关从该 run 的日志解析，不手写开关表。** 手写的表会与实跑漂开，且漂开时零报错。
   解析两行：`[m13] 开关（训练=评估，同源）：…` 与 `[m13] 约束组=…｜优势口径=…`。
   另需第一行的 `inst=` / `route_k=` / `steps=` / `seed=`——`route_k` 改的是**策略的动作
   空间**，漏了就是用另一个策略评估。解析不出（缺行、缺键、组名不在 `ABLATION_GROUPS`）
   一律**显式报错**，不猜默认值。
2. **复用 `m13_train_a._make_eval_fn`**（DRY）。另写一份评估逻辑 ⟹ 两侧口径静默分叉，
   而那正是本脚本要消灭的风险。
3. **评估种子 = `EVAL_SEED_BASE + s`、`sample=False`（argmax）、默认 `seeds=5`**——与训练
   时同一批种子。逐种子原始值一并带出（配对检验要用；只有均值做不了）。

已知边界（引用本脚本的读数时必须一并声明）
------------------------------------------
- **设备**：训练 run 跑在 CUDA（`--device cuda`）上，本脚本**默认**按门禁口径跑在 **CPU**。
  `sample=False` 不消费动作采样流，故不存在"两条流"问题；但 CPU 与 CUDA 的浮点内核
  （以及 CUDA 图快路）可能给出不同结果，**个别决策的 argmax 可能翻转** ⟹ 重评值与训练
  内联评估**不保证逐位相同**。记录里带 `inline_eval`（该 run 的 `metrics.ndjson` 最后一次内联
  评估）+ `makespan_delta_vs_inline`：口径错位通常给**大**差；设备浮点差多数给 **0**
  （实测 13 个 run 里 **12 个逐位相同**），但**确定性 run**例外（见下条）。
  ⚠️ **确定性的 run 要把差读对**：约束全关时仿真的随机源全没了（`c-none` 的 5 个种子给出
  同一个值），此时一次 argmax 翻转会**整体平移**该 run 读数（实测 −2.45 / −4.7%），
  而**不是**小幅噪声。要与内联评估逐位一致，用 `--device cuda` 重跑（需 GPU 解释器）。
- **`t3`**：T3 的罚项只加在训练的优势上，**不改动力学、不改策略输入** ⟹ 重评不需要它；
  但开着时评估回调会多报两个动作使用率（⑪/⑫ 的退化守卫读数），故照解析值透传。
- **`--eval-seeds` 不在日志里**（m13 不打印它）。本脚本默认 5 = 期① 批次的公共值
  （`docs/experiment-plan.md` §9.7）。换过 `--eval-seeds` 的 run 必须显式传 `--seeds`。
- **完成判据**：`metrics.ndjson` 行数 ≥ `--min-lines`（默认 **310** = 300 步 + 10 次评估）。
  ⚠️ 增量追加的文件**不许读最后一行 eval 判进度**——中间值会误导；行数才是判据。
  批次模式下未跑满的 run**跳过并标注**（不猜、不部分评估）。

跑法::

    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m17_reeval \
        --root D:/Temp/phase1 --arms full,c-none --out D:/Temp/phase1/reeval.json
    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m17_reeval \
        --run-dir D:/Temp/phase1/full --out D:/Temp/phase1/reeval-full.json

⚠️ 本脚本按"一次一个 run"**串行**跑（同一进程、顺序循环）：机器上若有训练 run 在跑，
再叠并发会让两边的墙钟读数都不可采信。
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .env.constraints import ABLATION_GROUPS
from .env.des import SimConfig, reference_makespan
from .env.layout import AgvSpec
from .m13_train_a import (EVAL_SEED_BASE, _make_eval_fn, build_training_setup)

# 期① 批次的公共 `--eval-seeds`（`run_phase1.sh` / `run_phase1b.sh` 的 COMMON）。
# ⚠️ 它**不在日志里**（m13 不打印）——本默认值只在"跑的就是期① 批次"时成立。
DEFAULT_EVAL_SEEDS = 5

# 完成判据：300 步（每步一行）+ 10 次评估（`--eval-every 30`）== 310 行。
COMPLETE_MIN_LINES = 310

# ── 日志锚点（对齐 `m13_train_a.main` 的 print；正则兼容旧词「小电池档」的历史日志）──
_SWITCH_MARKER = "[m13] 开关（训练=评估，同源）："
_GROUP_MARKER = "[m13] 约束组="
_META_MARKER = "[m13] inst="
_BATTERY_MARKER = r"(?:小电池档|小电池验证档)：车队电池全换"
_PM_MARKER = r"(?:短保养间隔档|短保养间隔验证档)：pm_interval="

# 开关行的九个键（顺序 = `m13.main` 的打印顺序）。**一个都不能少**：每一个都改变策略
# 看到的输入或动作空间（`multi_drop` / `agv_failover` / `machine_age_failure` 改动力学，
# 随 cfg 到达）；漏一个 ⟹ 用另一个策略评估，且不报错。
_SWITCH_KEYS = ("route_zones", "geom_bias", "pm_head", "charge_head", "batch_head",
                "multi_drop", "agv_failover", "machine_age_failure", "t3")


@dataclass(frozen=True)
class RunSpec:
    """一个 run 的**实跑口径**——全部字段来自该 run 的日志，没有一个来自默认值。"""
    inst: str
    steps: int
    seed: int
    route_k: int
    constraint_group: str
    adv_mode: str
    route_zones: bool
    geom_bias: bool
    pm_head: bool
    charge_head: bool
    batch_head: bool
    multi_drop: bool
    agv_failover: bool
    machine_age_failure: bool
    t3: bool
    # 机制验证档的两个可选覆盖（改动力学/改布局；日志里有才解析出来）：
    agv_battery_kwh: float | None = None
    pm_interval: float | None = None

    def cfg(self) -> SimConfig:
        """`SimConfig`——与 `m13.main` 的构造顺序逐字一致。

        ⚠️ 顺序有语义：③⑨⑩ 的开关与 `battery_low` / `pm_interval` 必须在
        `build_training_setup` **之前**进 cfg（`ctx` 的归一化标度与仿真动力学同源）。
        """
        c = SimConfig(agv_failover=self.agv_failover,
                      machine_age_failure=self.machine_age_failure,
                      multi_drop=self.multi_drop)
        if self.agv_battery_kwh is not None:
            c = replace(c, battery_low=0.0)
        if self.pm_interval is not None:
            c = replace(c, pm_interval=self.pm_interval)
        return c

    def chain_kwargs(self) -> dict:
        """链级开关——`_make_eval_fn` 与 `roll_chain` 的那一份（**逐字相同**才叫同源）。"""
        return {"route_k": self.route_k, "route_zones": self.route_zones,
                "geom_bias": self.geom_bias, "pm_head": self.pm_head,
                "charge_head": self.charge_head, "batch_head": self.batch_head,
                "t3": self.t3}

    def record(self) -> dict:
        """进 JSON 的口径块（可复核：读数与它跑在什么口径上必须同框）。"""
        return {"inst": self.inst, "steps": self.steps, "seed": self.seed,
                "constraint_group": self.constraint_group, "adv_mode": self.adv_mode,
                "agv_battery_kwh": self.agv_battery_kwh, "pm_interval": self.pm_interval,
                **self.chain_kwargs()}


def _find_line(text: str, marker: str, what: str) -> str:
    """含 `marker` 的第一行；找不到即显式报错（不猜默认值）。"""
    for line in text.splitlines():
        if marker in line:
            return line
    raise ValueError(
        f"日志里找不到{what}（判据：含 {marker!r} 的行）。"
        "本脚本的口径**只从日志来**——解析不出时必须报错，不得静默退回默认值"
        "（默认值与实跑漂开时零报错，读数会静默错误）。")


def parse_switches(line: str) -> dict:
    """`[m13] 开关（训练=评估，同源）：key=Bool …` → `{key: bool}`。

    九个键**全都要在**：少一个就报错（`m13` 打印时一个不少，缺了说明日志被截断或格式变了）。
    """
    out: dict[str, bool] = {}
    for key in _SWITCH_KEYS:
        m = re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}=(True|False)", line)
        if m is None:
            raise ValueError(f"开关行里没有 {key}=True/False（日志格式变了或行被截断）：{line!r}")
        out[key] = m.group(1) == "True"
    return out


def parse_constraint_group(line: str) -> tuple[str, str]:
    """`[m13] 约束组=Full｜优势口径=scalar（…）` → `("Full", "scalar")`。

    ⚠️ 组名必须能在 `ABLATION_GROUPS` 里查到：查不到 = 这一个 run 的约束口径无法重建 ⟹ 报错。
    ⚠️ `优势口径` 只记录（评估不跑优势、不建优势），但它是该 run 身份的一部分。
    """
    m = re.search(r"\[m13\] 约束组=(\S+?)｜优势口径=([A-Za-z_]+)", line)
    if m is None:
        raise ValueError(f"约束组行的格式不符（期待 `约束组=<名>｜优势口径=<名>`）：{line!r}")
    group, adv = m.group(1), m.group(2)
    if group not in ABLATION_GROUPS:
        raise ValueError(
            f"日志里的约束组 {group!r} 不在 ABLATION_GROUPS 里（已知：{tuple(ABLATION_GROUPS)}）"
            "——无法重建这一个 run 的约束口径。")
    return group, adv


def parse_run_spec(text: str) -> RunSpec:
    """整份日志 → `RunSpec`。缺任一必需项即显式报错。"""
    switches = parse_switches(_find_line(text, _SWITCH_MARKER, "开关行"))
    group, adv = parse_constraint_group(_find_line(text, _GROUP_MARKER, "约束组行"))
    meta = _find_line(text, _META_MARKER, "首行（inst/steps/seed/route_k）")
    m_inst = re.search(r"(?<![A-Za-z0-9_])inst=(\S+)", meta)
    m_steps = re.search(r"(?<![A-Za-z0-9_])steps=(\d+)", meta)
    m_seed = re.search(r"(?<![A-Za-z0-9_])seed=(-?\d+)", meta)
    m_rk = re.search(r"(?<![A-Za-z0-9_])route_k=(\d+)", meta)
    if not all((m_inst, m_steps, m_seed, m_rk)):
        raise ValueError(f"首行缺 inst=/steps=/seed=/route_k= 之一：{meta!r}")
    m_bat = re.search(rf"{_BATTERY_MARKER} ([\d.]+) kWh", text)
    m_pm = re.search(rf"{_PM_MARKER}([\d.]+)", text)
    return RunSpec(inst=m_inst.group(1), steps=int(m_steps.group(1)),
                   seed=int(m_seed.group(1)), route_k=int(m_rk.group(1)),
                   constraint_group=group, adv_mode=adv,
                   agv_battery_kwh=float(m_bat.group(1)) if m_bat else None,
                   pm_interval=float(m_pm.group(1)) if m_pm else None, **switches)


def build_env(spec: RunSpec):
    """`(inst, lay, dm, cfg, ctx, pol, cons)`——与 `m13.main` 的构造顺序逐字一致。

    ⚠️ 车队电池的替换在**布局之后**做（电池是布局属性）；`battery_low` 在 `cfg` 里
    （两者必须一起改，见 `m13.main` 的小电池验证档注释）。
    """
    inst, lay, dm, cfg, ctx, pol, cons = build_training_setup(
        spec.inst, cfg=spec.cfg(), constraints=ABLATION_GROUPS[spec.constraint_group])
    if spec.agv_battery_kwh is not None:
        for i, a in enumerate(lay.agvs):
            lay.agvs[i] = AgvSpec(id=a.id, speed_factor=a.speed_factor,
                                  capacity=a.capacity, battery_kwh=spec.agv_battery_kwh)
    return inst, lay, dm, cfg, ctx, pol, cons


def make_eval_fn(spec: RunSpec, inst, lay, dm, cfg, ctx, cons, seeds: int, rule: float):
    """按该 run 解析出的口径建评估回调——**唯一**构造点（`reeval_arm` 与测试共用）。

    ⚠️ 这里就是"复用而不是另写一份"的落点：本函数只做**转调**，评估逻辑仍在
    `m13_train_a._make_eval_fn` 里。另写一份 ⟹ 两边口径静默分叉（本脚本要消灭的风险）。
    """
    return _make_eval_fn(inst, lay, dm, cfg, ctx, seeds, rule, constraints=cons,
                         **spec.chain_kwargs())


def count_lines(path: Path) -> int:
    """`metrics.ndjson` 的行数——**完成判据只看它**（增量追加，末行 eval 是中间值）。"""
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def last_inline_eval(path: Path) -> dict | None:
    """`metrics.ndjson` 里最后一次内联评估（`{"step","eval"}` 行）；没有则 `None`。"""
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval" in rec:
                last = {"step": rec["step"], **rec["eval"]}
    return last


def reeval_arm(run_dir: Path, log_path: Path,
               seeds: int = DEFAULT_EVAL_SEEDS, device: str = "cpu") -> dict:
    """重评一个 run：解析日志 → 建环境 → 载 ckpt → 调 `m13._make_eval_fn` → 记录。

    返回的记录含**口径块**（`spec`）、**逐种子原始值**（`metrics`）与该 run 的**内联评估快照**
    （`inline_eval`，用于判断重评与训练内联评估差多少）。

    `device` 默认 `"cpu"`（门禁解释器的唯一选项）。训练 run 跑在 CUDA 上 ⟹ 要与内联评估
    逐位一致时用 `"cuda"` + GPU 解释器（见模块 docstring 的"设备"条）。
    """
    spec = parse_run_spec(log_path.read_text(encoding="utf-8"))
    ckpt_path = run_dir / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 不存在——这一个 run 没有可重评的最终策略。")
    inst, lay, dm, cfg, ctx, pol, cons = build_env(spec)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol.load_state_dict(ck["model"])
    pol.to(device)
    pol.eval()      # 无 Dropout/BatchNorm（只有 LayerNorm）⟹ 模式对数值无影响；语义记号而已
    # `rule` 与 m13 的 `rollout(inst, seed_chain=0, cfg=cfg)["makespan"]` 同一次运行：
    # `reference_makespan` 就是 `reference_run` 的薄封装，而 `build_env` 已经跑过并缓存它 ⟹ 零成本。
    rule = reference_makespan(inst, cfg)
    fn = make_eval_fn(spec, inst, lay, dm, cfg, ctx, cons, seeds, rule)
    metrics = fn(pol)
    metrics_path = run_dir / "metrics.ndjson"
    inline = last_inline_eval(metrics_path) if metrics_path.exists() else None
    delta = (None if not inline else
             metrics["makespan_mean"] - float(inline["makespan_mean"]))
    return {"run_dir": str(run_dir), "log": str(log_path),
            "ckpt_step": int(ck.get("step", -1)), "n_jobs": int(inst.n_jobs),
            "n_lines": count_lines(metrics_path) if metrics_path.exists() else 0,
            "eval_seeds": [EVAL_SEED_BASE + s for s in range(seeds)],
            "device": device,
            "spec": spec.record(), "metrics": metrics,
            "inline_eval": inline, "makespan_delta_vs_inline": delta}


def _fmt_seed_values(values: list[float]) -> str:
    return "[" + " ".join(f"{v:.2f}" for v in values) + "]"


def format_table(records: list[dict]) -> str:
    """逐个 run 一行：三目标的均值 + **逐种子原始值**（配对检验用）。"""
    out = []
    for r in records:
        m = r["metrics"]
        out.append(
            f"{r['arm']:<15} {m['makespan_mean']:>11.2f}±{m['makespan_std']:<6.2f}"
            f" {m['energy_mean']:>8.3f}"
            f" {m['twt_mean']:>9.3f}"
            f" │ ms{_fmt_seed_values(m['makespan_per_seed'])}"
            f" en{_fmt_seed_values(m['energy_per_seed'])}"
            f" twt{_fmt_seed_values(m['twt_per_seed'])}")
    return "\n".join(out)


def _discover_arms(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "ckpt.pt").exists())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="事后重评：从 run 目录的 ckpt 重算 makespan / energy / TWT（不训练）")
    ap.add_argument("--run-dir", default=None, help="单 run 模式：run 目录（内含 ckpt.pt）")
    ap.add_argument("--root", default=None, help="批次模式：根目录（含 <arm>/ 与 logs/<arm>.log）")
    ap.add_argument("--arms", default="", help="批次模式的 run 名（逗号分隔）；空 = 自动发现全部")
    ap.add_argument("--log", default=None, help="覆盖日志路径（默认 <run_dir 父>/logs/<run 名>.log）")
    ap.add_argument("--out", required=True, help="JSON 输出路径")
    ap.add_argument("--seeds", type=int, default=DEFAULT_EVAL_SEEDS,
                    help=f"评估种子数（默认 {DEFAULT_EVAL_SEEDS} = 期① 批次的 --eval-seeds；"
                         "⚠️ 它不在日志里，换过的 run 必须显式传）")
    ap.add_argument("--min-lines", type=int, default=COMPLETE_MIN_LINES,
                    help=f"完成判据：metrics.ndjson 行数下限（默认 {COMPLETE_MIN_LINES} = "
                         "300 步 + 10 次评估）。未达标者**跳过并标注**")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="评估跑在哪个设备（默认 cpu = 门禁解释器）。训练内联评估跑在 "
                         "`--device` 指定的那一台上；要与它逐位一致（尤其确定性的 run，见模块 "
                         "docstring）就用同一设备——cuda 需 GPU 解释器")
    args = ap.parse_args()
    if bool(args.run_dir) == bool(args.root):
        ap.error("--run-dir 与 --root 必须二选一")

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
        root = run_dirs[0].parent
    else:
        root = Path(args.root)
        names = ([s.strip() for s in args.arms.split(",") if s.strip()]
                 if args.arms else _discover_arms(root))
        run_dirs = [root / n for n in names]

    print(f"[m17] 评估口径：EVAL_SEED_BASE={EVAL_SEED_BASE}、sample=False（argmax）、"
          f"seeds={args.seeds}、device={args.device}｜完成判据：metrics.ndjson 行数 ≥ "
          f"{args.min_lines}")
    print("[m17] 口径来源 = 各 run 日志（不手写开关表）；评估逻辑 = 复用 m13._make_eval_fn"
          "（DRY，两侧不会漂）")
    print("[m17] ⚠️ 训练 run 跑在 CUDA 上：argmax 不消费采样流，但 CPU 与 CUDA 的浮点内核不同，"
          "个别决策可能翻转 ⟹ 与内联评估不保证逐位相同（见下方的 delta 列）；"
          "确定性的 run（约束全关）会把一次翻转整体平移，用 --device cuda 复现", flush=True)

    records, skipped = [], []
    for rd in run_dirs:
        arm = rd.name
        log_path = Path(args.log) if args.log else root / "logs" / f"{arm}.log"
        metrics_path = rd / "metrics.ndjson"
        if not rd.exists() or not (rd / "ckpt.pt").exists():
            print(f"[m17] 跳过 {arm}：{rd} 下没有 ckpt.pt", flush=True)
            skipped.append({"arm": arm, "reason": "no ckpt.pt"})
            continue
        if not log_path.exists():
            print(f"[m17] 跳过 {arm}：日志 {log_path} 不存在（口径解析不出来就不猜）", flush=True)
            skipped.append({"arm": arm, "reason": f"no log: {log_path}"})
            continue
        n_lines = count_lines(metrics_path) if metrics_path.exists() else 0
        if n_lines < args.min_lines:
            print(f"[m17] 跳过 {arm}：metrics.ndjson 只有 {n_lines} 行（< {args.min_lines}）"
                  "——这一个 run 还没跑满，不部分评估", flush=True)
            skipped.append({"arm": arm, "reason": f"incomplete: {n_lines} < {args.min_lines}",
                            "n_lines": n_lines})
            continue
        rec = reeval_arm(rd, log_path, seeds=args.seeds, device=args.device)
        rec["arm"] = arm
        records.append(rec)
        d = rec["makespan_delta_vs_inline"]
        print(f"[m17] {arm} 完成：ckpt step={rec['ckpt_step']}｜"
              f"ms={rec['metrics']['makespan_mean']:.2f}"
              f"（内联 {rec['inline_eval']['makespan_mean']:.2f}，差 {d:+.4f}）", flush=True)

    if records:
        print("\n逐个 run 一行（三目标均值 + 逐种子原始值）；"
              f"逐种子顺序 = 评估种子 {records[0]['eval_seeds']}（配对检验用）")
        print(format_table(records))
        print("\nrun            规则基线   ckpt步  完成/hit")
        for r in records:
            m = r["metrics"]
            print(f"{r['arm']:<15}{r['metrics']['rule_makespan']:>8.2f}{r['ckpt_step']:>8}"
                  f"   {int(m['jobs_done_min'])}/{r['n_jobs']}  hit={m['horizon_hit_frac']:.2f}")
    if skipped:
        print("\n跳过的 run（未跑满或缺失；**没有**部分评估）：")
        for s in skipped:
            print(f"  {s['arm']}: {s['reason']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"eval_seed_base": EVAL_SEED_BASE, "seeds": args.seeds,
         "min_lines": args.min_lines, "arms": records, "skipped": skipped},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[m17] JSON 已写入 {out.resolve()}")


if __name__ == "__main__":
    main()
