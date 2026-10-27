# -*- coding: utf-8 -*-
"""**地板测试：这段 PPO 能学会一个 9 臂老虎机吗？**

══ 为什么要有这个 ══

一整天的失败，我一直在怀疑**世界**（奖励口径、环境、实例、判据……）。
**这个脚本把世界拿掉**：不碰仿真、不碰调度、不碰特征 —— 只剩 `gae` / `update` / `Head`
三段代码，喂一个**平凡到不可能失败**的问题。

- 9 个动作，**动作 0** 有 0.8 概率给 +1，其余动作给 0 的概率 0.8。
- 每"局"只有 1 步（无折扣、无时序）⟹ **没有任何信用分配问题。**
- 只要 PPO 是活的，它必须收敛到动作 0。

**判据：200 步内，动作 0 的占比 > 0.9。**
**不过 ⟹ 病在 `gae`/`update`/`Head`，与仿真无关，而且在这个规模上能一眼看穿。**

⚠️ 用的是 `plcsp_v3/repro/shi2025_min_pa.py` **同一个** `gae` 与 `update`（原样 import）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 同目录

import numpy as np  # noqa: E402
import torch  # noqa: E402

import shi2025_min_pa as M  # noqa: E402

BEST, N_ACT, P, BATCH, STEPS = 0, 9, 0.8, 64, int(os.environ.get("FLOOR_STEPS", "200"))
F_DIM = 4
CTX = os.environ.get("FLOOR_CTX", "0") == "1"   # 1 ⟹ 有状态：**最优动作随状态变**
N_ST = 5


def main() -> None:
    torch.manual_seed(0)
    net = M.Head(F_DIM)          # 同一个 Head 类（构造参数是 f_dim）
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    print(f"9 臂老虎机（{'**有状态**：最优动作随状态变' if CTX else '无状态'}）"
          f" · 选对 p={P} 给 +1，选错 p={1 - P:.1f} 给 +1")
    print(f"每局 1 步 · batch {BATCH} · {STEPS} 步 · lr 1e-3 · **无仿真、无特征、无时序**\n")

    for step in range(1, STEPS + 1):
        chains, acts = [], []
        for _ in range(BATCH):
            f = (np.array([rng.integers(0, N_ACT)] + list(rng.normal(0, 1, F_DIM - 1)),
                          dtype=np.float32) if CTX
                 else rng.normal(0, 1, F_DIM).astype(np.float32))
            a, lp, v = net_pick(net, f)
            ok = (a == best_of(f))
            r = 1.0 if rng.random() < (P if ok else 1 - P) else 0.0
            chains.append([(f, a, lp, v, r)])
            acts.append(ok)
        M.update(net, chains, opt)
        if step % max(1, STEPS // 5) == 0 or step == 1:
            share = float(np.mean(acts))
            print(f"  step {step:>4}/{STEPS}　**选对动作的比例 {share:6.1%}**", flush=True)

    net.eval()
    got = []
    for _ in range(2000):
        f = (np.array([rng.integers(0, N_ACT)] + list(rng.normal(0, 1, F_DIM - 1)),
                      dtype=np.float32) if CTX
             else rng.normal(0, 1, F_DIM).astype(np.float32))
        with torch.no_grad():
            got.append(int(net.pi(torch.as_tensor(f)[None]).argmax()) == best_of(f))
    share = float(np.mean(got))
    print(f"\n  greedy 评测 2000 次：**选对的比例 = {share:.1%}**"
          f"　（{'有状态' if CTX else '无状态'}档）")
    print(f"  ⟹ 判据 > 90%　**{'✅ 通过 —— PPO 是活的' if share > 0.9 else '❌ 不过 —— PPO 坏了'}**")


def best_of(x: np.ndarray) -> int:
    """有状态档：最优动作 = 状态编号本身。

    ⚠️ **状态必须是**均匀**的编号**（2026-10-12）：起初用 `x[0] ~ N(0,1)` 再取整，
    结果 `P(x[0] < 1) = 84%` ⟹ **最优动作 84% 的时候是同一个** ⟹ 学个常数就打平，
    根本测不出状态依赖（那档实测 84.4%，与CDR无异）。
    """
    return int(x[0]) if CTX else BEST


def net_pick(net, f):
    ft = torch.as_tensor(f)[None]
    with torch.no_grad():
        d = net.dist(ft)
        a = int(d.sample())
        return a, float(d.log_prob(torch.tensor(a))), float(net.value(ft)[0])


if __name__ == "__main__":
    main()
