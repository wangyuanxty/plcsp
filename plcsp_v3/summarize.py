# -*- coding: utf-8 -*-
"""**跑数器输出汇总** —— 把 `q1_flat` / `d5_nested` / `b_ablation` / `c_mechanism` 的 JSONL
读成一张表。

```
python -m plcsp_v3.summarize q1.jsonl d5.jsonl b.jsonl c.jsonl
python -m plcsp_v3.summarize --final-only q1.jsonl        # 只看 final
```

══ ⚠️ 本工具**只做算术，不做判断** ══

它把 `final` 记录里的 `clean` / `disturbed` 三目标按臂与种子**平均**，报出均值与标准误。
**它不报显著性、不报"谁赢了"** —— 那要看实验设计（`experiment-plan.md` §K 的"赢"的定义），
而 §K **目前还缺**（原文自己写着"这一节现在缺，必须补"）。

⚠️ **n 很小的时候标准误几乎和均值一样大**，而本工具**照样把它印出来** ——
**看到 `±` 比均值还大就该知道这一格说明不了问题。** 这是刻意的：不藏。

══ 用法要点 ══

- **多种子**：`--seeds` 不指定就全收。**同一个臂只有 1 个种子时会明确警告**
  （单种子说明不了稳定性，而稳定性正是 Q1/D5 要测的东西）。
- **约束探针**：`final` 记录里的 `constraints` 会一并汇总 ——
  **"跑通了但约束没触发"必须当场看得见**（§110/§115.3 的教训）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OBJ = ("makespan", "energy", "twt")


def _load(paths: list[str]) -> dict[str, list[dict]]:
    """`{文件名: [记录, ...]}`。坏行跳过并计数（不静默丢）。"""
    out: dict[str, list[dict]] = {}
    for p in paths:
        rows, bad = [], 0
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                bad += 1
        if bad:
            print(f"⚠️ {p}: 跳过 {bad} 行坏 JSON", file=sys.stderr)
        out[Path(p).name] = rows
    return out


def _mean_se(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), float("nan")     # ⚠️ 单个种子给不出标准误 —— 报 NaN，不报 0
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))


def summarize(path: str, rows: list[dict], *, final_only: bool = False) -> None:
    finals = [r for r in rows if r.get("kind") == "final"]
    if not finals:
        print(f"\n═══ {path} ═══  ⚠️ 没有 `final` 记录（还在跑 / 中途挂了）")
        n_train = sum(1 for r in rows if r.get("kind") == "train")
        if rows:
            last = [r for r in rows if r.get("kind") == "train"]
            if last:
                print(f"    最新一步：{last[-1]['arm']} seed{last[-1]['seed']} "
                      f"step{last[-1]['step']}（共 {n_train} 条 train 记录）")
        return

    print(f"\n═══ {path} ═══  {len(finals)} 条 final")
    arms: dict[str, list[dict]] = {}
    for r in finals:
        arms.setdefault(str(r["arm"]), []).append(r)

    for arm, rs in arms.items():
        seeds = sorted(int(r["seed"]) for r in rs)
        if len(seeds) == 1:
            print(f"  ⚠️ **臂 `{arm}` 只有 1 个种子** —— 单种子说明不了稳定性"
                  f"（而稳定性正是 Q1/D5 要测的东西）")
        line = [f"  {arm:<12} n={len(rs)} seed={seeds}  "]
        for key in ("clean", "disturbed"):
            vals = {o: [float(r[key][o]) for r in rs] for o in OBJ}
            m, se = _mean_se(vals["makespan"])
            line.append(f"{key}: ms={m:7.2f}±{se:5.2f}")
        # ⚠️ 奖励 `R` **越大越好**（三目标越小越好）—— 标注写清，别让读的人反着看
        mR, seR = _mean_se([float(r["clean"]["R"]) for r in rs])
        print("".join(line) + f"  R={mR:8.2f}±{seR:5.2f}（R 越大越好）")

    # 约束探针：**"跑通了但约束没触发"必须当场可见**
    keys = sorted({k for r in finals for k in (r.get("constraints") or {})})
    if keys:
        print("  ── 规则驱动约束探针（同一臂内各种子应相同，不同才要查）──")
        for arm, rs in arms.items():
            cs = [r.get("constraints") or {} for r in rs]
            spread = {k: (min(c.get(k, 0) for c in cs), max(c.get(k, 0) for c in cs))
                      for k in keys}
            txt = "  ".join(f"{k}={spread[k][0]:g}" if spread[k][0] == spread[k][1]
                            else f"{k}={spread[k][0]:g}..{spread[k][1]:g}⚠️" for k in keys)
            print(f"    {arm:<12} {txt}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="跑数器输出汇总（只做算术）")
    ap.add_argument("jsonl", nargs="+", help="一个或多个跑数器输出")
    ap.add_argument("--final-only", action="store_true",
                    help="只报 final（默认也是只报 final；此开关保留给将来的曲线汇总）")
    a = ap.parse_args(argv)
    for path, rows in _load(a.jsonl).items():
        summarize(path, rows, final_only=a.final_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
