"""M3a 冒烟：S 层组训练器（三种层内基线 × J=1 退化验证 + J=2 分支）。

运行：python -m plcsp.m3_smoke
"""
import numpy as np
import torch

from .env.instances import gen_random
from .algo.policy import PolicyNet
from .algo.group_rel import train_step, tree_step, local_tree_step


def main() -> None:
    inst = gen_random(4, 3, seed=0)          # 4 jobs × 3 machines 小实例（调试快）
    for mode in ("z", "mean", "loo"):
        pol = PolicyNet()
        rs = [round(train_step(pol, inst, seed=s, G=8, J=1, mode=mode)[0], 1) for s in range(5)]
        print(f"[m3a] mode={mode:5s} 5 步组均值轨迹: {rs}")
    pol = PolicyNet()
    r = train_step(pol, inst, seed=0, G=4, J=2)
    print(f"[m3a] J=2 分支: r_mean={r[0]:.1f} diag={ {k: round(v, 3) for k, v in r[1].items() if isinstance(v, float)} }")
    # M3b-①：两层树（B 层批模式 × S 层计划；分层归因）
    pol2 = PolicyNet()
    for s in range(4):
        t = tree_step(pol2, inst, seed=s, G=4, J=1, cap_opts=[None, 2, 1])
        print(f"[m3b] tree s={s} r_mean={t[0]:.1f} pB={t[1]['pB']} A_std_B={t[1]['A_std_B']:.3f}")
    # M3b-③：批量重用（K-epoch）+ 裁剪代理
    torch.manual_seed(0); np.random.seed(0)
    pA = PolicyNet(); base = train_step(pA, inst, seed=0, G=4, J=1, mode="z")
    torch.manual_seed(0); np.random.seed(0)
    pB_ = PolicyNet(); clip1 = train_step(pB_, inst, seed=0, G=4, J=1, mode="z", epochs=1, clip_eps=0.2)
    dmax = max(float((pa - pb).abs().max().detach()) for pa, pb in zip(pA.parameters(), pB_.parameters()))
    print(f"[m3b-③] ratio=1 时 clip 梯度=纯REINFORCE: maxΔparam={dmax:.2e}（应为 0）")
    torch.manual_seed(0); np.random.seed(0)
    pC = PolicyNet(); k3 = train_step(pC, inst, seed=0, G=4, J=1, mode="z", epochs=3, clip_eps=0.2)
    print(f"[m3b-③] K=3 clip: r_mean={k3[0]:.1f} loss={k3[1]['loss']:.4f} ratio={k3[1]['ratio']:.3f}")
    t2 = tree_step(pB_, inst, seed=1, G=4, J=1, cap_opts=[None, 2, 1], epochs=2, clip_eps=0.2)
    print(f"[m3b-③] tree K=2 clip: pB={t2[1]['pB']} ratio={t2[1]['ratio']:.3f} A_std_B={t2[1]['A_std_B']:.3f}")
    # M3c：M2 编码器接入（双轴轴向注意力 token 路径；布局=line seed 1，与 rollout_evaluate 默认一致）
    from .env.layout import sample_layout
    from .nn.state_emb import encode_state
    from .nn.encoder import LayoutEncoder
    lay = sample_layout(inst.n_machines, "line", seed=1)
    es = encode_state(inst, lay, n_agv=2)
    print(f"[m3c] enc N={es.tok_feat.shape[1]} seg={es.seg} F={es.tok_feat.shape[-1]}")
    torch.manual_seed(0); np.random.seed(0)
    polE = PolicyNet(enc=LayoutEncoder(feat_dim=es.tok_feat.shape[-1]))
    rs = [round(train_step(polE, inst, seed=s, G=4, J=1, mode="z", enc_state=es)[0], 1)
          for s in range(3)]
    print(f"[m3c] token 路径 3 步 r_mean: {rs} （params={sum(p.numel() for p in polE.parameters())/1e6:.2f}M）")
    # M3b-②：决策点级局部树（单步枚举；同一决策点 (0,0) 重复 6 次 → 看 p 向 vB 更优候选漂移）
    torch.manual_seed(0); np.random.seed(0)
    polL = PolicyNet()
    for s in range(6):
        t_l = local_tree_step(polL, inst, seed=s, J=2, mode="z", dec_idx=(0, 0))
        d = t_l[1]
        print(f"[m3b-②] s={s} dec={d['dec']} vB={[round(x, 1) for x in d['vB']]} "
              f"p={d['p']} A_std={d['A_std']:.3f}")
    # M3b-② × M3c：编码器 token 路径（候选机台 token 不同 → 几何可分 → p 应漂移）
    torch.manual_seed(0); np.random.seed(0)
    polLE = PolicyNet(enc=LayoutEncoder(feat_dim=es.tok_feat.shape[-1]))
    for s in range(6):
        t_l = local_tree_step(polLE, inst, seed=s, J=2, mode="z", dec_idx=(0, 0), enc_state=es)
        d = t_l[1]
        print(f"[m3b-②]enc s={s} vB={[round(x, 1) for x in d['vB']]} p={d['p']}")


if __name__ == "__main__":
    main()
