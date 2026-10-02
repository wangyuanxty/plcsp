# -*- coding: utf-8 -*-
"""M8-Phase B：**对抗臂（adv）** —— 布局-策略共演化的第五臂：生成器找**最难**布局。

设计（与 m8_coevo.py 的 co/rs/dr/fixed 四臂并列，四臂行为不动）：
- 生成器奖励（本臂唯一差异）：
      R_adv = −regret(layout)
      regret = mks_rule(layout) − mks_policy(layout)     （同一布局、同一评估协议）
  **为什么不是纯 −策略性能**：纯 −性能会让对手收敛到"无解布局"（谁都做不好，例如极窄通道
  把所有人拖垮）——那是退化难例，对策略改进无信息。regret 在"所有人都失败"处 → 0，
  **自动排除退化难例**：只有"规则能做、策略做不好"的布局才被奖励（= 策略的真实短板）。
  参考基线 = 规则调度 load_min（无学习参数）。
  **符号约定（务必看清）**：makespan 越小越好，按规格字面 regret = mks_rule − mks_policy，
  故 **regret 为负 = 策略不如规则 = 该布局对策略难**（与规格里"regret 为正 = 策略难"的
  文字表述差一个符号——那句话把"性能"当成了越大越好的质量分）。等价的无歧义量 =
  **R_adv = −regret = mks_policy − mks_rule，为正 = 难**，也正是生成器**最大化**的目标；
  CMA 最小化 −R_adv = regret ⇒ 朝"策略比规则差"的布局搜索（= 找最难，方向正确）。
  jsonl 里 `regret` 字段 = 规格字面值（负=难），`R`/`adv_best_R` = mks_policy − mks_rule
  （正=难）。
- 三件套（防退化）：
  ① 可行性约束：复用 m8_coevo.BOUNDS（含现实下界 通道≥1.2 / 间距≥0.8）——候选与均值均投影回界内；
  ② regret 奖励（上式，抗退化难例）；
  ③ 混合采样：内层训练的布局 = 50% 生成器当前分布（cma.mean + σ·N(0,I)）+ 50% 均匀域采样
     （单臂内同时保留"对手分布"与"全域覆盖"，防对手把策略拖进单一角落）。
- 交替结构（与 co 臂一致，双时间尺度）：每轮 = 内层策略训练 STEPS 步（混合采样）→
  外层 CMA 一代（K=pop 候选，用刚训练的策略评估，最小化 −R_adv = regret）→ ROUNDS 轮。
- 终评（30 种子，同 m8_coevo 协议 EVAL_SEEDS=1..30）：
  (a) 难例集 = adv 臂 CMA **终态**候选的 top-N（按 R_adv）——对抗找到的最难布局；
  (b) 均匀随机布局集（与 m8_coevo 用同一 rng(7) 序列 → 跨文件可比）——
  三臂策略 × 两集 × 30 种子，指标 = makespan 均值。另附规则基线在两集上的均值
  （非退化证据：难例集若规则也崩 = 退化难例，regret 口径本应已排除）。
输出：plcsp/m8_adv.jsonl（每轮一行 + 终评行）。
运行：python -m plcsp.m8_adv --rounds 6 --steps 45 --pop 16 --arms adv,dr,fixed

参考基线的口径说明（诚实注记）：仓库历史锚 "load_min 457.1"（docs/progress-log.md §一）
产生于旧仿真修订（bug#8b horizon 修复前），当前仿真下任何 load_min 变体都不复现该绝对值
（实测 370–424），故本模块**按规则语义重新实现** load_min 并以**同协议同轮**为参照——
regret 只用参考项与策略项的**差**，绝对值不可比历史锚不影响对照有效性。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .env.instances import load_mk
from .env.layout import sample_layout
from .env.des import SimWorld, SimConfig
from .nn.state_emb import encode_state
from .nn.encoder import LayoutEncoder
from .algo.policy import PolicyNet
from .m8_coevo import (NAGV, BOUNDS, X0, EVAL_SEEDS, SepCMA,
                       decode, build, train_on, eval_on, run_arm)

OUT = "plcsp/m8_adv.jsonl"
MIX_P = 0.5          # ③ 混合采样：生成器当前分布占比（其余=均匀域采样）
HARD_TOP = 8         # 难例集大小（= m8_coevo 均匀集大小，两集可比）
UNI_N = 8            # 均匀随机布局集大小（同 m8_coevo）
CAND_SEEDS = (1, 11, 21)     # 候选评估协议（同 m8_coevo.search）


# ---------------- 参考基线：规则调度 load_min ----------------

def load_min_policy_l(feat: np.ndarray) -> int:
    """规则 L 层：选负载最小的 AGV（负载 = 累计派单 + 在途任务数）。

    特征布局（des.SimWorld._transporter）：[3]=(派单0−派单1)/10，[6:6+n_agv]=在途队列长/5。
    本模块 n_agv=2（m8_coevo.NAGV）→ 两车负载差 = 派单差 + 在途差，符号即可判定。
    纯实时状态规则、无学习参数 → regret 的参考项与臂无关。
    """
    f = np.asarray(feat, dtype=float)
    load_diff = f[3] * 10.0 + (f[6] - f[7]) * 5.0
    return 0 if load_diff <= 0 else 1


def eval_rule(inst, p: dict, seeds=CAND_SEEDS) -> float:
    """规则调度（load_min）在同一布局、同一协议下的 makespan（regret 的参考项）。

    计划层 = 规则默认（每工序最短加工候选，= des.py 缺省口径，与历史规则基线一致）；
    L 层 = load_min 实时门控。无策略依赖 → 结果按布局缓存复用（跨臂/跨轮安全）。
    """
    key = (p["layout_type"], round(p["aisle_width"], 6), round(p["machine_gap"], 6),
           int(p["n_zones"]), tuple(seeds))
    if key in _RULE_CACHE:
        return _RULE_CACHE[key]
    lay, dm = build(inst, p)
    w = SimWorld(inst, lay, dm, SimConfig(n_agv=NAGV, n_zones=p["n_zones"],
                                          aisle_width=lay.aisle_width))
    ms = [w.run_gated(seed_chain=s, op_choices=None, policy_l=load_min_policy_l)["makespan"]
          for s in seeds]
    val = float(np.mean(ms))
    _RULE_CACHE[key] = val
    return val


_RULE_CACHE: dict[tuple, float] = {}


def regret_of(pol, inst, p: dict, seeds=CAND_SEEDS) -> tuple[float, float, float]:
    """(regret, mks_policy, mks_rule)；regret 按规格字面 = mks_rule − mks_policy。

    符号：makespan 越小越好 ⇒ **regret < 0 = 策略不如规则 = 该布局对策略难**；
    R_adv = −regret = mks_policy − mks_rule（>0 = 难）才是生成器最大化的目标。
    """
    mks_pol = eval_on(pol, inst, p, seeds=seeds)
    mks_rule = eval_rule(inst, p, seeds=seeds)
    return mks_rule - mks_pol, mks_pol, mks_rule


# ---------------- adv 臂 ----------------

def policy_new(feat_dim: int) -> PolicyNet:
    """新策略（与 m8_coevo.run_arm 同构、同初始化种子口径）。"""
    torch.manual_seed(0)
    return PolicyNet(enc=LayoutEncoder(feat_dim=feat_dim), n_agv=NAGV, n_feat_l=6 + NAGV + 2)


def run_arm_adv(inst, feat_dim: int, rounds: int, steps: int, pop: int,
                f, t0: float) -> tuple[PolicyNet, dict | None, list[dict], list[dict]]:
    """adv 臂：生成器最小化 regret（=最大化 R_adv）。返回 (策略, 历史最难布局, 终态候选, 全历史候选)。

    三件套落地：BOUNDS 投影（候选/均值）＋ regret 奖励 ＋ 混合采样（MIX_P）。
    与 co 臂唯一的结构差异 = 奖励（co=−(mks/mks0)−λ·cost，本臂=−regret）与内层采样。
    """
    pol = policy_new(feat_dim)
    rng = np.random.default_rng(1234)                     # 与 run_arm 同种子口径
    cma = SepCMA(X0, 0.5, popsize=pop, seed=0)
    gstep = 0
    p_default = decode(X0)
    last_recs: list[dict] = []
    pool: list[dict] = []                                 # 全历史候选（终态候选不足时补位）
    best_hard, best_R = None, -1e9

    for r in range(rounds):
        row = {"arm": "adv", "round": r + 1, "elapsed": round(time.time() - t0, 1)}
        for _ in range(steps):                            # 内层：混合采样训练
            if rng.random() < MIX_P:                      # ③a 生成器当前分布
                x = cma.mean + cma.sigma * rng.standard_normal(len(X0))
            else:                                         # ③b 均匀域采样
                x = BOUNDS[:, 0] + rng.random(len(X0)) * (BOUNDS[:, 1] - BOUNDS[:, 0])
            train_on(pol, inst, decode(x), seed=gstep)
            gstep += 1
        X = np.clip(cma.ask(), BOUNDS[:, 0], BOUNDS[:, 1])   # ① 可行性：候选投影（同 co 臂）
        recs = []
        for x in X:
            p = decode(x)
            reg, mks_pol, mks_rule = regret_of(pol, inst, p, seeds=CAND_SEEDS)
            recs.append({"p": p, "mks": round(mks_pol, 2), "mks_rule": round(mks_rule, 2),
                         "regret": round(reg, 2), "R": round(-reg, 4)})
        cma.tell(np.array([-c["R"] for c in recs]))       # 最小化 −R_adv = regret
        bi = int(np.argmax([c["R"] for c in recs]))       # 本轮最难布局（R_adv 最大 = gap 最大）
        if recs[bi]["R"] > best_R:
            best_R, best_hard = recs[bi]["R"], recs[bi]["p"]
        last_recs, pool = recs, pool + recs
        regs = [c["regret"] for c in recs]
        row.update({"cma_mean": [round(float(v), 3) for v in cma.mean],
                    "adv_best_R": recs[bi]["R"], "adv_best": recs[bi]["p"],
                    "adv_best_mks": recs[bi]["mks"], "adv_best_rule_mks": recs[bi]["mks_rule"],
                    "adv_best_regret": recs[bi]["regret"],
                    "adv_best_gap": round(-recs[bi]["regret"], 2),      # 正 = 策略比规则差 = 难
                    "regret_mean": round(float(np.mean(regs)), 2),
                    "n_hard": int(np.sum(np.array(regs) < 0)),          # 策略不如规则的候选数
                    "best_ever_R": round(best_R, 4), "best_ever": best_hard})
        row["policy_on_default"] = round(eval_on(pol, inst, p_default, seeds=CAND_SEEDS), 2)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
    return pol, best_hard, last_recs, pool


def hard_set(last_recs: list[dict], pool: list[dict], n: int = HARD_TOP) -> list[dict]:
    """难例集 = CMA 终态候选按 R_adv 取 top-N；pop<n 时用全历史候选补位。"""
    top = sorted(last_recs, key=lambda c: -c["R"])[:n]
    if len(top) < n:
        seen = {tuple(sorted(c["p"].items())) for c in top}
        for c in sorted(pool, key=lambda c: -c["R"]):
            k = tuple(sorted(c["p"].items()))
            if k in seen:
                continue
            seen.add(k)
            top.append(c)
            if len(top) >= n:
                break
    return top


# ---------------- 终评（a 难例集 / b 均匀集，30 种子） ----------------

def final_eval(inst, policies: dict, hard_cfgs: list[dict], uni_cfgs: list[dict],
               f, t0: float) -> None:
    """三臂策略 × {default, hard0..7, uni0..7} × 30 种子；另报规则基线两集均值（非退化证据）。"""
    cfgs = {"default": decode(X0)}
    cfgs.update({f"hard{i}": c for i, c in enumerate(hard_cfgs)})
    cfgs.update({f"uni{i}": c for i, c in enumerate(uni_cfgs)})
    hk = [f"hard{i}" for i in range(len(hard_cfgs))]
    uk = [f"uni{i}" for i in range(len(uni_cfgs))]
    rule = {k: round(eval_rule(inst, p, seeds=EVAL_SEEDS), 2) for k, p in cfgs.items()}
    row_rule = {"arm": "rule_load_min", "final": True, "kind": "reference",
                "eval": rule,
                "hard_mean": round(float(np.mean([rule[k] for k in hk])), 2),
                "uni_mean": round(float(np.mean([rule[k] for k in uk])), 2),
                "elapsed": round(time.time() - t0, 1)}
    print(json.dumps(row_rule, ensure_ascii=False), flush=True)
    f.write(json.dumps(row_rule, ensure_ascii=False) + "\n")
    for arm, pol in policies.items():
        ev = {k: round(eval_on(pol, inst, p), 2) for k, p in cfgs.items()}   # 30 种子
        hard_mean = float(np.mean([ev[k] for k in hk]))
        uni_mean = float(np.mean([ev[k] for k in uk]))
        gap = float(np.mean([ev[k] - rule[k] for k in hk]))                 # 难例集平均 gap
        row = {"arm": arm, "final": True, "eval": ev,
               "hard_mean": round(hard_mean, 2), "uni_mean": round(uni_mean, 2),
               "rule_hard_mean": row_rule["hard_mean"], "rule_uni_mean": row_rule["uni_mean"],
               "hard_gap": round(gap, 2),     # mks_policy − mks_rule：>0 = 该臂策略在难例集上不如规则
               "own": round(hard_mean, 2),    # 端到端：自有对抗设计 × 自有策略
               "elapsed": round(time.time() - t0, 1)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--arms", type=str, default="adv,dr,fixed")
    ap.add_argument("--out", type=str, default=OUT)
    args = ap.parse_args()

    inst = load_mk("mk01")
    es_ref = encode_state(inst, sample_layout(inst.n_machines, "line", 1), n_agv=NAGV)
    feat_dim = es_ref.tok_feat.shape[-1]
    t0 = time.time()

    rng_e = np.random.default_rng(7)                      # 同 m8_coevo：均匀集逐位一致
    uni_cfgs = [decode(BOUNDS[:, 0] + rng_e.random(len(X0)) * (BOUNDS[:, 1] - BOUNDS[:, 0]))
                for _ in range(UNI_N)]

    policies: dict[str, PolicyNet] = {}
    hard_cfgs: list[dict] = [decode(X0)]
    with open(args.out, "a", encoding="utf-8") as f:
        for arm in args.arms.split(","):
            if arm == "adv":
                pol, _best, last_recs, pool = run_arm_adv(inst, feat_dim, args.rounds,
                                                          args.steps, args.pop, f, t0)
                top = hard_set(last_recs, pool)
                hard_cfgs = [c["p"] for c in top]
                row_hs = {"arm": "adv", "hard_set": hard_cfgs,
                          "hard_set_gap": [round(-c["regret"], 2) for c in top],
                          "hard_set_regret": [c["regret"] for c in top],
                          "elapsed": round(time.time() - t0, 1)}
                print(json.dumps(row_hs, ensure_ascii=False), flush=True)
                f.write(json.dumps(row_hs, ensure_ascii=False) + "\n")
                f.flush()
            else:
                pol, _phi = run_arm(arm, inst, feat_dim, args.rounds, args.steps,
                                    args.pop, f, t0)
            policies[arm] = pol
        final_eval(inst, policies, hard_cfgs, uni_cfgs, f, t0)
    print("[m8adv] done")


if __name__ == "__main__":
    main()
