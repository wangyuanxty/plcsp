# -*- coding: utf-8 -*-
"""**多种子**：现状（A：池化 + 全部 bias）自己的离散度有多大？

设定与 `v3_ladder.py` 的 K=1 档逐项相同：**12 条 PA 规则**、规则在每个 PA 决策点重采、
整局回报、只训 `net.select.heads["PA"]`、`lr=1e-3`。

只变一个东西：**网络的随机初始化种子**（`torch.manual_seed(seed)`）。

**若种子之间的结果本身就散得很开 ⟹ 我之前所有单次对比都没有信息量。**
"""
import sys

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from plcsp_v3.env.reward import objective_vector, scalar_reward  # noqa: E402
from plcsp_v3.hrl.rules import PA_RULE_NAMES, apply_pa_rule  # noqa: E402
from plcsp_v3.hrl.rules import la_rule_policy, pm_rule_policy  # noqa: E402
from plcsp_v3.nn.state_emb import build_tok  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

LA_FIX, RA_FIX, THR = 0, 0, 30.0
import os  # noqa: E402
N_CHAIN = int(os.environ.get("V3_CHAIN", "8"))
# ⚠️ **默认 `1e-4` —— 仓里训整个网络时的档**（`a_main.py` / `q1_flat.py`）。
# 冻编码器那版用 `1e-3`（只动 780 个参数）。
LR = float(os.environ.get("V3_LR", "1e-4"))
STEPS = int(os.environ.get("V3_STEPS", "16"))
FREEZE = os.environ.get("V3_FREEZE", "0") == "1"
LAM_D = float(os.environ.get("V3_LAM_D", "0"))
LAM_R = float(os.environ.get("V3_LAM_R", "0"))   # regenerative regularization 权重（0 = 关）
BN = os.environ.get("V3_BN", "0") == "1"          # 打分头入口加 BatchNorm1d（治逐维方差不平衡）
BNALL = os.environ.get("V3_BNALL", "0") == "1"    # **把编码器里的 LayerNorm 全换成 BatchNorm**
# 目标权重：`""` = 默认（`build(reward_norm="minmax")` 给的等权）。
# `"ms"` = **单目标纯 makespan**（`prefs=(1,0,0)` ⟹ `scalar_reward` 里的 `prefs` **取代** `w`）。
PREFS = os.environ.get("V3_PREFS", "")
_PREFS_T = {"ms": (1.0, 0.0, 0.0), "energy": (0.0, 1.0, 0.0),
            "twt": (0.0, 0.0, 1.0)}.get(PREFS)


class SegBN(nn.Module):
    """在 `(B, N, d)` 上把 **d 当通道**做 BatchNorm。

    ⚠️ **不能直接把 `(B,N,d)` 喂给 `BatchNorm1d`** —— 它会把张量当成 `(N, C, L)`
    解释，通道数取成 `N`（token 数）而不是 `d` ⟹ **归一化的是错的那根轴**，且通道数
    随实例规模变。故这里显式转置到 `(B, d, N)` 再转回。
    """

    def __init__(self, d: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


def swap_ln_to_bn(module: nn.Module) -> int:
    """把 `module` 子树里**所有** `LayerNorm` 就地换成 `SegBN`。返回换掉几个。"""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, SegBN(int(child.normalized_shape[0])))
            n += 1
        else:
            n += swap_ln_to_bn(child)
    return n
# ⚠️ **多个评测场景**（此前只有 100_000 一个）—— 见 `eval_pol` 的 docstring。
EVAL_W = [100_000 + 13 * i for i in range(8)]


class PA:
    def __init__(self, net, spec, gen):
        self.net, self.spec, self.gen = net, spec, gen


    def policy_pa(self, view, pool, tie_rng=None):
        tok, seg = build_tok(view.snap, self.spec.inst, self.spec.layout, self.spec.ctx)
        with torch.no_grad():
            lg = self.net(torch.as_tensor(tok)[None], seg,
                          hard=True).rule_logits["PA"][0]
            a = int(torch.multinomial(torch.softmax(lg, -1), 1, generator=self.gen))
        self.rec.append((tok, tuple(seg), a, None))
        return apply_pa_rule(a, view, pool, tie_rng)


def chain(net, spec, seed, sc):
    # ⚠️ **rollout 必须 eval 态**：在线路径逐状态前向（批=1），
    # 若让 BatchNorm 在 train 态算批统计，批=1 会直接报错。评测/采样用 running 统计。
    net.eval()
    p = PA(net, spec, torch.Generator().manual_seed(seed))
    p.rec = []
    m = spec.world().run_gated(seed_chain=sc, policy_pa=p.policy_pa,
                               policy_l=la_rule_policy(LA_FIX),
                               policy_m=pm_rule_policy(RA_FIX, THR), dt_events=0)
    f = objective_vector(m)
    return (scalar_reward(f, spec.w, prefs=_PREFS_T, bounds=spec.bounds),
            p.rec, float(f[0]))


def eval_pol(net, spec, scs):
    """在**多个场景**上评测：`ms` 取**均值**，规则分布取**并集**。

    ⚠️ **单场景评测是个大噪声源** —— 12×8 矩阵里 `EAJ+EAM` 跨场景从 63.2 到 90.4。
    此前只用 `100_000` 一个场景，导致我把 1.6σ 的差当成结论（2026-10-12 第四次的同类错误）。
    """
    net.eval()                                   # 逐状态前向，必须 eval 态
    mss: list = []
    ens: list = []
    tws: list = []
    scs_out: list = []
    seen: list = []
    for sc in scs:
        def policy_pa(view, pool, tie_rng=None, _seen=seen):
            tok, seg = build_tok(view.snap, spec.inst, spec.layout, spec.ctx)
            with torch.no_grad():
                lg = net(torch.as_tensor(tok)[None], seg, hard=True).rule_logits["PA"][0]
            a = int(lg.argmax())
            _seen.append(a)
            return apply_pa_rule(a, view, pool, tie_rng)

        m = spec.world().run_gated(seed_chain=sc, policy_pa=policy_pa,
                                   policy_l=la_rule_policy(LA_FIX),
                                   policy_m=pm_rule_policy(RA_FIX, THR), dt_events=0)
        fo = objective_vector(m)
        mss.append(float(fo[0]))
        ens.append(float(fo[1]))
        tws.append(float(fo[2]))
        scs_out.append(scalar_reward(fo, spec.w, prefs=_PREFS_T, bounds=spec.bounds))
    u, c = np.unique(np.asarray(seen), return_counts=True)
    # ⚠️ **全部列出来，不许只取前两名** —— 前一版 `[:2]` 让我把「≥3 种」误读成「2 种」。
    order = np.argsort(-c)
    allr = [f"{PA_RULE_NAMES[int(u[i])]}:{int(c[i])}" for i in order]
    # **三项都返回** —— 只报 ms 会让人（和我）跨口径下结论。
    return len(u), allr, (float(np.mean(mss)), float(np.mean(ens)),
                          float(np.mean(tws)), float(np.mean(scs_out)))


def eff_rank(net, spec) -> float:
    """训练后 `phi` 的**有效秩**（SIGReg 那篇的判据）。"""
    toks = []

    def pp(view, pool, tie_rng=None):
        tok, _ = build_tok(view.snap, spec.inst, spec.layout, spec.ctx)
        toks.append(tok)
        return apply_pa_rule(11, view, pool, tie_rng)

    for sc in range(100_000, 100_012):
        spec.world().run_gated(seed_chain=sc, policy_pa=pp,
                               policy_l=la_rule_policy(0),
                               policy_m=pm_rule_policy(0, 30.0), dt_events=0)
    seg = (int(spec.inst.n_machines), int(spec.inst.n_jobs), int(spec.ctx.n_agv), 1)
    with torch.no_grad():
        _z, pooled = net.enc(torch.as_tensor(np.stack(toks)), seg)
        H = net.select.trunk(pooled).numpy()
    Hc = H - H.mean(0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(Hc.T @ Hc / max(len(H) - 1, 1)), 0, None)
    p_ = ev / max(ev.sum(), 1e-12)
    p_ = p_[p_ > 0]
    return float(np.exp(-(p_ * np.log(p_)).sum()))


def run_seed(seed: int) -> tuple:
    torch.manual_seed(seed)
    spec, net = build(seed=seed, pm_interval=30, target_failures=4,
                      finite_buffer=True, layout_seed=0, reward_norm="minmax",
                      )
    if FREEZE:
        # **只冻 LA / RA 两个头**（它们不用 —— LA/RA 走固定规则），
        # **编码器 + trunk + PA 头全部训**。⟹ 这才是「单 PA」而不是「冻掉整个编码器」。
        for h in ("LA", "RA"):
            for q in net.select.heads[h].parameters():
                q.requires_grad_(False)
    if BNALL:
        # ══ **把编码器里的 LayerNorm 全换成 BatchNorm** ══
        # 三处：`AttnLayer.ln1` / `AttnLayer.ln2` / `LayoutEncoder.ln`。
        # ⚠️ 与打分头入口那个 BN **互斥** —— 本档只做这一件事。
        n_sw = swap_ln_to_bn(net.enc)
        print(f"    seed {seed:>2}：编码器里换掉 {n_sw} 个 LayerNorm → BatchNorm", flush=True)
    elif BN:
        # ══ 治「逐维方差不平衡」══（§36.4：`pooled` 第 1 维占 81.7% 的方差，
        # 少数维的尺度压倒其余维 ⟹ 打分头实际只用 1–2 维。纯前向实测：真头 argmax 2 → 10。）
        # 插在 **trunk 入口**：`nn.Sequential(bn, *原 trunk)` ⟹ **所有**走 `net(...)` 的路径
        # （在线采样、评测）自动带上，不必改三处。训练态用批统计、评测态用 running 统计。
        bn0 = nn.BatchNorm1d(net.select.trunk[0].in_features)
        net.select.trunk = nn.Sequential(bn0, *net.select.trunk)
    params = [p for p in net.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=LR)
    # ══ **regenerative regularization** ══
    # Kumar, Marklund, van Roy & Machado, *Maintaining Plasticity in Continual
    # Learning via Regenerative Regularization*, NeurIPS 2023：
    #
    #     L += (λ/2) · Σ_i ‖θ_i − θ_i⁰‖²
    #
    # 注意**不是**「拉向 0」：拉向 **θ⁰（初始值）**。θ⁰ 本身就是一个函数正常的网，
    # 所以这个惩罚不改变「网络该输出什么」，只限制**权重离初始化能走多远** ——
    # 走得越远，表示越容易塌到少数几维（我们量到 `φ` 有效秩 2.22/64）。
    # arXiv 2405.19153 §5.4 把它列为 on-policy RL 里有效的五种之一；
    # §5.5 又指出它**与 LayerNorm 叠加**时训练侧和泛化侧同时改善（我们已有 pre-LN）。
    theta0 = [p.detach().clone() for p in params]
    _ = 0.0                       # 正则项的上一档读数（仅诊断）
    k0, top0, o0 = eval_pol(net, spec, EVAL_W)          # o = (ms, energy, twt, 标量)
    for step in range(1, STEPS + 1):
        # ⚠️ **一组内 8 条链跑同一个场景**（仓里的口径：
        # 对 n 条链是同一个值）。**场景多样性在步与步之间**。
        # 此前写成 （i 变）⟹ 组内比的是**场景差异**，不是动作差异。
        sc = 9_000_000 + step
        runs = [chain(net, spec, step * 100 + i, sc) for i in range(N_CHAIN)]
        raw = np.array([r[0] for r in runs])       # 组内 8 条链的整局回报
        r_mean, r_std = float(raw.mean()), float(raw.std())
        adv = (raw - raw.mean()) / (raw.std() + 1e-6)
        net.train()                              # 训练态：BatchNorm 用整批的统计
        optim.zero_grad(set_to_none=True)
        # ⚠️ **必须把整批决策堆成一次前向**（此前逐决策点前向，批 = 1）。
        # 两个理由：① `BatchNorm` 在训练态下批=1 会直接报错（`_verify_batch_size`）；
        # ② 仓里的重算路径本来就是批前向（`encoder.py` docstring：231 条批一次 ≈402 ms
        #    vs 逐条 ≈1.38 s，**~3.4×**）。这里顺带对齐它。
        toks_all: list = []
        acts_all: list = []
        adv_all: list = []
        for a, (_r, rec, _ms) in zip(adv, runs):
            for tok, sg, act, _ag in rec:        # `_ag`：聚合旁路已摘，占位保持元组形状
                toks_all.append(tok)
                acts_all.append(int(act))
                adv_all.append(float(a))
        X = torch.as_tensor(np.stack(toks_all))
        seg_b = runs[0][1][0][1]                 # `seg` 是实例级常量（本档处处相同）
        assert all(r[1][0][1] == seg_b for r in runs), "seg 在决策点之间变了——批前向不成立"
        _z, pooled = net.enc(X, seg_b)
        lg = net.select(pooled)["PA"]                        # (n, 12)
        logp = torch.log_softmax(lg, dim=-1)
        idx = torch.as_tensor(acts_all)
        loss = -(torch.as_tensor(np.asarray(adv_all)) * logp[torch.arange(len(idx)), idx]).mean()
        _ = float(loss)          # 策略梯度项，仅看量级                     # 策略梯度那一项（看两个损失的量级比）
        reg = torch.zeros(())
        if LAM_R:
            # `Σ‖θ − θ⁰‖²`，**对每个参数张量分别求和**再相加（与 Kumar 2023 一致）。
            reg = sum(((p - p0) ** 2).sum() for p, p0 in zip(params, theta0))
            loss = loss + 0.5 * LAM_R * reg
        _ = float(reg)            # 正则项，仅看量级
        loss.backward()
        optim.step()
        every = max(1, STEPS // 10)
        if step % every == 0 or step == 1:
            _k, _top, _o = eval_pol(net, spec, EVAL_W)
            print(f"    seed {seed:>2} step {step:>4}/{STEPS}  "
                  f"评测 ms={_o[0]:7.2f} 能耗={_o[1]:6.3f} TWT={_o[2]:7.2f} "
                  f"标量={_o[3]:+.4f}  训练r={r_mean:+.4f}±{r_std:.4f}  "
                  f"**{_k} 种**  {'  '.join(_top)}",
                  flush=True)
    k1, top1, o1 = eval_pol(net, spec, EVAL_W)
    er = eff_rank(net, spec)
    # `V3_CKPT="路径{seed}.pt"` ⟹ 存权重，供事后分析（本会话此前没存，只能重训）。
    _ck = os.environ.get("V3_CKPT", "")
    if _ck:
        torch.save(net.state_dict(), _ck.format(seed=seed))
        print(f"    seed {seed}：权重已存 {_ck.format(seed=seed)}", flush=True)
    return (k0, o0, k1, o1, top1, er)


def main() -> None:
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4]
    print(f"12 条 PA 规则 · 每个决策点重采 · 整局回报 · "
          f"**{('只冻 LA/RA 两个头；编码器 + trunk + PA 头全训' if FREEZE else '训整个网络（不冻结）')}** · "
          f"lr={LR} · {STEPS} 步 · {N_CHAIN} 条链 · "
          f"regenerative regularization λ={LAM_R} · "
          f"{'**编码器 LayerNorm 全换 BatchNorm**' if BNALL else ('打分头入口 BatchNorm1d' if BN else '无 BN 改动')}"
          f" · 目标={PREFS or '等权三项'}")
    print(f"{'seed':>5} {'init 种数':>9} {'init ms':>8} | {'训后 种数':>9} {'训后 ms':>8} "
          f"{'能耗':>7} {'TWT':>8} {'标量':>9}  前矛")
    print("-" * 96)
    out = []
    for s in seeds:
        k0, o0, k1, o1, top1, er = run_seed(s)
        out.append((k1, o1))
        print(f"{s:>5} {k0:>9} {o0[0]:>8.2f} | {k1:>9} {o1[0]:>8.2f} "
              f"{o1[1]:>7.3f} {o1[2]:>8.2f} {o1[3]:>+9.4f}  "
              f"phi 有效秩={er:.2f}/64  {top1}", flush=True)
    kk = np.array([x[0] for x in out])
    mm = np.array([x[1][0] for x in out])
    ss = np.array([x[1][3] for x in out])
    print("-" * 96)
    print(f"  训后 ms：均值 {mm.mean():.2f}　最小 {mm.min():.2f}　最大 {mm.max():.2f}　"
          f"标准差 {mm.std():.2f}")
    print(f"  训后标量：均值 {ss.mean():+.4f}　最小 {ss.min():+.4f}　最大 {ss.max():+.4f}　"
          f"标准差 {ss.std():.4f}")
    print(f"  训后 argmax 种数：{dict(zip(*np.unique(kk, return_counts=True)))}")


if __name__ == "__main__":
    main()
