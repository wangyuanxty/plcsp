"""M2 冒烟：几何特征 + 双轴编码器（形状/反向/参数/CUDA）。失败时打印全链形状。

运行：python -m geosched.m2_smoke
"""
import traceback

import torch

from .env.layout import sample_layout
from .nn.features import geometry_features, conf_sim
from .nn.encoder import LayoutEncoder, block_mask, GeomBias


def main() -> None:
    lay = sample_layout(6, "island", seed=1)
    gf = geometry_features(lay)
    N, B, F, nm, nb, nv = 20, 2, 10, 10, 6, 4
    enc = LayoutEncoder(feat_dim=F)
    x = torch.randn(B, N, F)
    dist = (torch.rand(N, N) + torch.rand(N, N).T) / 2
    conf = torch.tensor(conf_sim(torch.rand(N).numpy()))
    try:
        # —— 分步：与 forward 一致，每步打印形状 ——
        bi = GeomBias()(dist, conf).float().unsqueeze(0)   # 等价 forward 守卫：(N,N)→(1,N,N)
        print("bias", tuple(bi.shape))
        pm = block_mask(N, (nm, nb, nv), "prod").expand(B, -1, -1)
        lm = block_mask(N, (nm, nb, nv), "logi").expand(B, -1, -1)
        print("masks", tuple(pm.shape), tuple(lm.shape))
        hi = enc.embed(x)
        for i, layer in enumerate(enc.layers):
            hi = layer(hi, pm, lm, bi)
            if i in (0, 7):
                print(f"layer{i} out", tuple(hi.shape))
        hi = enc.ln(hi)
        print("tok", tuple(hi.shape), "ctx", tuple(hi.mean(1).shape))
        (hi.mean(1)).sum().backward()
        n = sum(p.numel() for p in enc.parameters())
        print(f"{n/1e6:.2f}M params | cuda {torch.cuda.is_available()} | M2-SMOKE-OK")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
