# -*- coding: utf-8 -*-
"""M8-Phase A：布局-策略共演化（浅生成器 sep-CMA-ES + 交替训练）—— 论文"0 空白"主线首版。

设计（会话记录 Phase A 定稿）：
- 生成器 φ = 4 维参数向量 [布局类型, 通道宽, 机台间距, 区段数(拥塞档)]；n_agv 固定=2
  （L 头维度随 n_agv 变，车队规模共设计留 Phase C——已在记录注明）。
- sep-CMA-ES（对角协方差，pop 候选/代）优化；CO 奖励：
    R_co = −(mks/mks0) − λ·(cost/cost0)
  mks0 = 同轮同策略下默认配置性能（轮内归一 → 候选可比）；cost = aisle + 0.3·zones
  （占地面积+基建代理，已标注近似）；λ = 0.5。
- 交替（双时间尺度）：内层策略在 φ 当前高斯分布上训练 STEPS 步（flat GRPO 引擎，
  layout_obj 外部构图）→ 外层 CMA 一代（用刚训练的策略评估候选）→ ROUNDS 轮。
- 四臂对照（同总步数/同种子序列/同 K 次候选评估预算）：**co（共演化 CMA）**/ **rs（随机搜索+固定策略训练
  = 杀手对照："共演化 vs 随手搜"）**/ dr（均匀 DR）/ fixed（固定默认配置）。
- Phase A2（真权衡）：通道 <1.5m → AGV 限速（des.SimConfig.eff_speed，下限 0.4×）；通道≥1.2、间距≥0.8
  现实下界 → 设计最优解回到内部（不再退化到角点）。
- 终评：三臂策略 × {默认, 8 个固定均匀随机配置, CO 终态推荐} × 30 种子
  （greedy 计划 + L=argmax，与主实验同协议）。
输出：geosched/m8_coevo.jsonl。运行：python -m geosched.m8_coevo --rounds 6 --steps 60
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
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .nn.state_emb import encode_state
from .nn.encoder import LayoutEncoder
from .algo.policy import PolicyNet
from .algo.group_rel import greedy_plan
from .algo.standard_tree import standard_tree_step

NAGV = 2
G, J, JL, LR, ENT = 4, 2, 8, 1e-4, 0.03
LAM = 0.5                                   # 成本权重（CO 奖励）
BOUNDS = np.array([[0.0, 2.999], [1.2, 3.0], [0.8, 3.0], [1.0, 6.0]])   # Phase A2：现实下界
                                                                        # （通道≥1.2 过车 / 间距≥0.8 检修）
X0 = np.array([0.0, 1.5, 1.0, 3.0])         # 默认设计：line / aisle1.5 / gap1.0 / 3 区段
LAYOUT_TYPES = ("line", "U", "island")
SEED_LAYOUT = 1
EVAL_SEEDS = tuple(range(1, 31))
OUT = "geosched/m8_coevo.jsonl"


# ---------------- 参数空间 ----------------

def decode(x: np.ndarray) -> dict:
    x = np.clip(np.asarray(x, float), BOUNDS[:, 0], BOUNDS[:, 1])
    return {"layout_type": LAYOUT_TYPES[int(x[0])], "aisle_width": float(x[1]),
            "machine_gap": float(x[2]), "n_zones": int(round(x[3]))}


def cost_of(p: dict) -> float:
    return p["aisle_width"] + 0.3 * p["n_zones"]        # 代理：占地 + 基建（近似，论文标注）


def build(inst, p: dict, seed_layout: int = SEED_LAYOUT):
    lay = sample_layout(inst.n_machines, p["layout_type"], seed_layout,
                        machine_gap=p["machine_gap"], aisle_width=p["aisle_width"])
    dm = dock_distance_matrix(build_corridor_graph(lay))
    return lay, dm


# ---------------- 策略训练 / 评估 ----------------

def train_on(pol, inst, p: dict, seed: int) -> float:
    lay, _ = build(inst, p)
    es = encode_state(inst, lay, n_agv=NAGV)
    r, _ = standard_tree_step(pol, inst, seed=seed, G=G, J=J, J_L=JL, lr=LR, n_agv=NAGV,
                              enc_state=es, ent_beta=ENT, norm="flat",
                              layout_obj=lay, n_zones=p["n_zones"])
    return r


def eval_on(pol, inst, p: dict, seeds=EVAL_SEEDS) -> float:
    lay, dm = build(inst, p)
    es = encode_state(inst, lay, n_agv=NAGV)
    tok = pol.encode_state(es)
    pl = greedy_plan(inst, pol, 10.0, tok_emb=tok)
    w = SimWorld(inst, lay, dm, SimConfig(n_agv=NAGV, n_zones=p["n_zones"],
                                          aisle_width=lay.aisle_width))   # A2：窄通道限速入仿真

    def lg(feat_np):
        lx = pol.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
        return int(torch.argmax(lx).item())

    ms = [w.run_gated(seed_chain=s, op_choices=pl, policy_l=lg)["makespan"] for s in seeds]
    return float(np.mean(ms))


# ---------------- sep-CMA-ES（对角协方差，最小化约定） ----------------

class SepCMA:
    def __init__(self, x0: np.ndarray, sigma0: float, popsize: int = 16, seed: int = 0):
        self.n = len(x0)
        self.mean = np.array(x0, float)
        self.sigma = float(sigma0)
        self.C = np.ones(self.n)
        self.pc, self.ps = np.zeros(self.n), np.zeros(self.n)
        self.gen = 0
        self.lam = popsize
        self.mu = popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2.0 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) /
                       ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))
        self.rng = np.random.default_rng(seed)
        self._y = None

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        self._y = z * np.sqrt(self.C)
        return self.mean + self.sigma * self._y

    def tell(self, fitness: np.ndarray) -> None:
        idx = np.argsort(fitness)                      # 升序=最小化，前 mu 优
        yw = (self.w[:, None] * self._y[idx[:self.mu]]).sum(0)
        self.mean = np.clip(self.mean + self.sigma * yw, BOUNDS[:, 0], BOUNDS[:, 1])
                                                       # A2 修正：均值投影回界内（原缺陷=
                                                       # 均值漂出界 → 设计恒被 clamp 在下界）
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (yw / np.sqrt(self.C)))
        hsig = (np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        Cmu = (self.w[:, None] * (self._y[idx[:self.mu]] ** 2)).sum(0)
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * self.pc ** 2 + self.cmu * Cmu
        self.sigma *= float(np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)))
        self.gen += 1


# ---------------- 实验臂 ----------------

def run_arm(arm: str, inst, feat_dim: int, rounds: int, steps: int, pop: int,
            f, t0: float):
    """单臂：co=共演化（CMA）｜rs=随机搜索（同预算，杀手对照）｜dr=均匀 DR｜fixed=固定默认。"""
    torch.manual_seed(0)
    pol = PolicyNet(enc=LayoutEncoder(feat_dim=feat_dim), n_agv=NAGV, n_feat_l=6 + NAGV + 2)
    rng = np.random.default_rng(1234)
    cma = SepCMA(X0, 0.5, popsize=pop, seed=0) if arm == "co" else None
    gstep = 0
    p_default = decode(X0)
    best_p, best_R = None, -1e9

    def search(pol_) -> list[dict]:
        """候选评估（K=pop 个）：co→CMA 采样；rs→均匀采样。R_co 同式。"""
        if arm == "co":
            X = np.clip(cma.ask(), BOUNDS[:, 0], BOUNDS[:, 1])   # A2：候选投影（防越界重复设计）
        else:
            X = BOUNDS[:, 0] + rng.random((pop, len(X0))) * (BOUNDS[:, 1] - BOUNDS[:, 0])
        mks0 = eval_on(pol_, inst, p_default, seeds=(1, 11, 21))
        recs = []
        for x in X:
            p = decode(x)
            mks = eval_on(pol_, inst, p, seeds=(1, 11, 21))
            R = -(mks / mks0) - LAM * (cost_of(p) / cost_of(p_default))
            recs.append({"p": p, "mks": round(mks, 2), "R": round(R, 4)})
        return recs

    for r in range(rounds):
        row = {"arm": arm, "round": r + 1, "elapsed": round(time.time() - t0, 1)}
        if arm == "rs":                       # 随机搜索先行（同预算 K 次评估），训在历史最优上
            recs = search(pol)
            bi = int(np.argmax([c["R"] for c in recs]))
            if recs[bi]["R"] > best_R:
                best_R, best_p = recs[bi]["R"], recs[bi]["p"]
            row.update({"rs_best_R": best_R, "rs_best": best_p})
        for _ in range(steps):
            if arm == "co":
                x = cma.mean + cma.sigma * rng.standard_normal(len(X0))
                p = decode(x)
            elif arm == "rs":
                p = best_p if best_p is not None else p_default
            elif arm == "dr":
                p = decode(BOUNDS[:, 0] + rng.random(len(X0)) * (BOUNDS[:, 1] - BOUNDS[:, 0]))
            else:
                p = p_default
            train_on(pol, inst, p, seed=gstep)
            gstep += 1
        if arm == "co":                       # 共演化：训练后更新生成器（交替/双时间尺度）
            recs = search(pol)
            cma.tell(np.array([-c["R"] for c in recs]))
            bi = int(np.argmax([c["R"] for c in recs]))
            if recs[bi]["R"] > best_R:        # A2b 修正：与 rs 同口径跟踪历史最优
                best_R, best_p = recs[bi]["R"], recs[bi]["p"]   # （原缺陷：co 终设计取 CMA 均值
            row.update({"cma_mean": [round(float(v), 3) for v in cma.mean],   # → 卡在下界，
                        "cma_best_R": recs[bi]["R"], "cma_best": recs[bi]["p"],  #  与 rs 的 best-ever
                        "cma_best_mks": recs[bi]["mks"],                          #  不可比）
                        "best_ever_R": best_R, "best_ever": best_p})
        row["policy_on_default"] = round(eval_on(pol, inst, p_default, seeds=(1, 11, 21)), 2)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
    if best_p is not None:                # co / rs 同口径：终设计 = 历史最优（best-ever）
        return pol, best_p
    return pol, p_default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=45)      # A2：4 臂同预算（6×45=270 步/臂）
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--arms", type=str, default="co,rs,dr,fixed")
    args = ap.parse_args()

    inst = load_mk("mk01")
    es_ref = encode_state(inst, sample_layout(inst.n_machines, "line", 1), n_agv=NAGV)
    feat_dim = es_ref.tok_feat.shape[-1]
    t0 = time.time()

    rng_e = np.random.default_rng(7)                    # 终评用固定均匀配置集
    uni_cfgs = [decode(BOUNDS[:, 0] + rng_e.random(len(X0)) * (BOUNDS[:, 1] - BOUNDS[:, 0]))
                for _ in range(8)]
    policies, finals = {}, {}
    with open(OUT, "a", encoding="utf-8") as f:
        for arm in args.arms.split(","):
            pol, phi = run_arm(arm, inst, feat_dim, args.rounds, args.steps, args.pop, f, t0)
            policies[arm] = pol
            finals[arm] = phi                            # 各臂终态设计（dict）
        cfgs = {"default": decode(X0),
                **{f"{a}_final": finals[a] for a in ("co", "rs") if a in finals},
                **{f"uni{i}": c for i, c in enumerate(uni_cfgs)}}
        for arm, pol in policies.items():
            ev = {k: round(eval_on(pol, inst, p), 2) for k, p in cfgs.items()}
            row = {"arm": arm, "final": True, "eval": ev,
                   "own": ev.get(f"{arm}_final", ev["default"]),   # 端到端：自有设计 × 自有策略
                   "uni_mean": round(float(np.mean([ev[f"uni{i}"] for i in range(8)])), 2),
                   "elapsed": round(time.time() - t0, 1)}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    print("[m8coevo] done")


if __name__ == "__main__":
    main()
