# -*- coding: utf-8 -*-
"""**台阶**：从 1 步的无时序老虎机，一格一格加到调度任务的形状。

══ 为什么要台阶 ══

单步台子（`v3_bandit_floor.py`）已确立一件事：**输入标准化 55% → 78%**。
但把标准化接回调度任务后**没有重现**（三条判据仍全 ❌）。

⟹ **要么调度侧还有别的东西，要么单步台子与调度不是同一个问题。**
⟹ 办法：**一格一格加**，加到哪一格掉下来，那一格就是病灶。

══ 台阶（`LADDER_STEPS` 控）══

| 格 | 局长度 | 奖励 | 新增的东西 |
|---|---|---|---|
| 1 | **1 步** | 每步 | ——（= 已测的单步台子，**78%**） |
| 2 | 4 步 | 每步 | **时序开始** |
| 3 | 16 步 | 每步 | 更长的信用分配 |
| 4 | 123 步 | 每步 | **与调度同长** |

每局：状态 `s` **一局固定**（与调度里"一个场景"对应），正确动作恒为 `s`。
回报：每步「选对给 1、选错给 0」——**就是调度里"这一步做得好不好"**。
算法：与调度同一个 `gae` / `update` / `Head`。**上限恒为 100%。**
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 同目录

import numpy as np  # noqa: E402
import torch  # noqa: E402

import shi2025_min_pa as M  # noqa: E402


N_ST, N_ACT, P, STEPS = 9, 9, 0.8, 1500
BATCH = int(os.environ.get("LADDER_BATCH", "16"))
F_DIM = N_ST          # ⚠️ **one-hot**，不是 1 维标量（见下）


def enc(s: int) -> np.ndarray:
    """状态 → **one-hot**。

    ⚠️ **2026-10-12 修**：此前用 `[(s − 4)/2.6]` 一维标量 —— 那让网络在**一条 1 维轴上切 9 段**。
    实测（`textbook.py`，教科书 REINFORCE、30 行、不共享本仓任何代码）：

    | 状态编码 | 结果 |
    |---|---|
    | 1 维标量 `s/4−1` | **6/9 = 66.7%** |
    | **one-hot 9 维** | **9/9 = 100%** |

    ⟹ **台子此前所有读数（40%/55%/66%/77%）都是"1 维标量编码 9 个离散状态"的病，
    不是 RL 的，也不是本仓实现的。**
    """
    v = np.zeros(F_DIM, dtype=np.float32)
    v[s] = 1.0
    return v


def main() -> None:
    n_step = int(os.environ.get("LADDER_STEPS", "1"))
    torch.manual_seed(0)
    net = M.Head(F_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    norm = "逐链" if os.environ.get("LADDER_NORM") == "per" else "全批"
    print(f"台阶 · 局长度 **{n_step} 步** · 优势归一化 **{norm}** · "
          f"9 状态 × 9 动作 · 上限 100% · batch {BATCH} × {STEPS} 轮\n")

    # ⚠️ `LADDER_STATE=fixed`（默认）状态一局固定；`=changing` **每步换一个状态**
    # —— 后者才是调度的形状（决策点的状态每步都在变）。
    changing = os.environ.get("LADDER_STATE") == "changing"
    # ⚠️ `LADDER_REWARD=terminal` ⟹ **整局一次性给 `Σ_t 1[a_t==s_t]`**，逐步全 0。
    # 与逐步口径**总回报完全相同**，只差「信用怎么摊到每一步」——**这才是时序信用分配**。
    terminal = os.environ.get("LADDER_REWARD") == "terminal"
    for it in range(1, STEPS + 1):
        chains = []
        for _ in range(BATCH):
            s = int(rng.integers(0, N_ST))
            buf, tot = [], 0.0
            for _t in range(n_step):
                if changing:
                    s = int(rng.integers(0, N_ST))     # 每步换
                f = enc(s)
                ft = torch.as_tensor(f[None], dtype=torch.float32)
                with torch.no_grad():
                    d = net.dist(ft)
                    a = int(d.sample())
                    lp = float(d.log_prob(torch.tensor(a)))
                    v = float(net.value(ft)[0])
                hit = 1.0 if rng.random() < (P if a == s else 1 - P) else 0.0
                r = 0.0 if terminal else hit      # 终局口径：逐步全 0
                buf.append((f, a, lp, v, r))
                if terminal:
                    tot += hit
            if terminal and buf:                 # 局末一次性补给**最后一步**
                f_, a_, lp_, v_, _ = buf[-1]
                buf[-1] = (f_, a_, lp_, v_, tot)
            chains.append(buf)
        M.update(net, chains, opt)
        if it % max(1, STEPS // 5) == 0 or it == 1:
            print(f"  iter {it:>4}/{STEPS}", flush=True)

    rng2 = np.random.default_rng(1)
    ok = 0
    N = 9000
    with torch.no_grad():
        for _ in range(N):
            s = int(rng2.integers(0, N_ST))
            f = enc(s)
            ok += int(net.pi(torch.as_tensor(f[None], dtype=torch.float32)).argmax()) == s
    print(f"\n  ⟹ **{ok / N:.1%}**　（上限 100%）")


if __name__ == "__main__":
    main()
