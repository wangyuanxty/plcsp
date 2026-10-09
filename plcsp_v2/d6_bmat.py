# -*- coding: utf-8 -*-
"""**D6 · `B` 矩阵读出**（`docs/v2/experiment-plan.md` §D 的 D6）—— **不是消融，是"读出来给人看"**。

```
python -m plcsp_v2.d6_bmat --ckpt <Full_s0.pt> --top 4
```

══ 要证的主张 ══

设计 §1/§3：`logits(第 i 个动作) = base_i(Z) + Σ_j B[i, j] · g_j`。
**若 `B` 学到东西，它的第 `j` 列应当读得懂**：""第 `j` 个目标把哪些动作往上抬"。

| 列 | 目标 | **预期**抬高的动作 |
|---|---|---|
| 0 | `θ_wip`（在制品） | 倾向于"少投料 / 先清活"的规则 —— 如 `SPT*` 系（短工序先行） |
| 1 | `θ_otr`（交期达成） | **`EDD*` 系**（交期最紧先行）—— 这是最该看得出的一列 |
| 2 | `θ_agv`（AGV 周转） | 倾向于少等待的匹配 |

**⚠️ 上面那张表是"预期"，不是"结果"。** 本跑数器**只报实测**，
**并显式对照预期**：对上了就写对上，对不上就写对不上 —— **不挑好看的那几列展示**。

══ ⚠️ `B = 0` 时本跑数器会明确报"没有可解释的内容" ══

设计 §95：`B` **零初始化**。只训了几步的 ckpt，`B` 会几乎是 0 ⟹ **读数无意义**。
本跑数器**先报 `‖B‖` 的量级**，并在它过小时**明确警告**（不静默给出一张全是 0 的表）。

⚠️ 默认写到系统临时目录（硬规则 1）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from .hrl.agents import HierarchicalPolicy
from .hrl.rules import LA_RULE_NAMES, PA_JOB_RULES, PA_MACHINE_RULES

DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "d6_bmat.jsonl")

G_NAMES = ("θ_wip 在制品", "θ_otr 交期达成", "θ_agv AGV 周转")
PARAM_NAMES = ("PA", "LA", "RA_maint")
"""`b_matrices()` 的三个键（⑪ 去后由四个变三个）。"""


def _pa_rule_names() -> list[str]:
    """12 条复合规则的名字，**顺序照 `apply_pa_rule` 的 `idx // len(PA_MACHINE_RULES)`**。

    ⚠️ **不得自己另排一遍**：顺序与 `rules.py` 的解析口径必须一致，
    否则读出来的"哪条规则被抬高"会张冠李戴。
    """
    return [f"{j}+{m}" for j in PA_JOB_RULES for m in PA_MACHINE_RULES]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D6：B 矩阵读出（可解释性凭据）")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=4, help="每列展示前几名动作")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        def rec(d: dict) -> None:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()

        for path in a.ckpt:
            ck = torch.load(path, weights_only=False)
            net = HierarchicalPolicy(with_value=ck["with_value"])
            net.load_state_dict(ck["state_dict"])
            bm = net.b_matrices()
            names = {"PA": _pa_rule_names(), "LA": list(LA_RULE_NAMES),
                     "RA_maint": ["生产", "保养"]}
            print(f"\n=== {Path(path).name}（arm={ck['arm']} seed={ck['seed']} "
                  f"steps={ck['steps']}）===", flush=True)

            payload = {"kind": "bmat", "ckpt": Path(path).name, "arm": ck["arm"],
                       "train_seed": ck["seed"], "steps": ck["steps"], "heads": {}}
            any_meaningful = False
            for head in PARAM_NAMES:
                B = bm[head].detach().cpu().numpy()      # (n_actions, 3)
                norm = float(abs(B).max())
                print(f"\n  ── {head} ──  ‖B‖_max = {norm:.4f}"
                      f"{'   ⚠️ **接近 0 ⟹ 此表无意义**（B 零初始化，训练不足）' if norm < 1e-3 else ''}",
                      flush=True)
                head_rows = []
                for j in range(B.shape[1]):
                    col = B[:, j]
                    order = col.argsort()[::-1][:a.top]
                    rows = [{"action": names[head][i],
                             "bias": round(float(col[i]), 4)} for i in order]
                    head_rows.append({"g": G_NAMES[j], "top": rows})
                    print(f"    {G_NAMES[j]:<16} ↑ " +
                          "  ".join(f"{r['action']}({r['bias']:+.4f})" for r in rows),
                          flush=True)
                if norm >= 1e-3:
                    any_meaningful = True
                payload["heads"][head] = {"norm_max": norm, "columns": head_rows}

            # ⚠️ **明确报"有没有可读的东西"** —— 不静默给出一张全 0 的表。
            payload["meaningful"] = bool(any_meaningful)
            if not any_meaningful:
                print("\n  ⚠️⚠️ **三个头的 `B` 都接近 0 ⟹ 本次读出没有可解释的内容。**\n"
                      "      设计 §95：`B` 零初始化，故「`g` 影响动作」是**学出来的**，"
                      "不是初始就有的。\n"
                      "      **训练步数不够的 ckpt 上跑本跑数器，得到的一定是这个结论 "
                      "—— 那不是发现，是没训够。**", flush=True)
            rec(payload)

    print(f"写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
