"""plcsp：生产-物流协同调度（PLCSP）的深度强化学习研究包。

问题 = 两环节联动：排产（FJSP）→ 物流（真 AGV 派车）。
骨架 = Transformer 编码器 + 组内相对 RL（GRPO）。
权威设计见 `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`。
"""
from __future__ import annotations

import torch as _torch

# ⚠️ **torch 线程数钉成 1**——放在包入口，任何 `import plcsp` 都生效。
#
# 为什么：本项目的张量极小（MK01 只有 20 个 token × F_MAX=11），每次前向摊到约 35 个
# `nn.Linear`。torch 默认按核数开线程（本机 32 核 → 24 线程），**线程同步开销压过计算本身**。
# 实测（MK01、`joint_chain_step`、预热后取中位）：
#
#   | 线程 | G=1 | G=4 |
#   |---|---|---|
#   | 1  | **1.34 s/步** | **5.53 s/步** |
#   | 24 | 3.68 s/步 | 14.83 s/步 |
#
# 即**单线程快 2.7 倍**。更要紧的是**波动**：24 线程下同一步实测过
# 8.85 / 31.95 / 34.36 / 34.88 s（约 4 倍），单线程是 1.34/1.34/1.34/1.34。
#
# ⚠️ 这不只是快慢：本项目在 `progress-log` 里记**墙钟数字**（如消融表 546 s/组）。
#    线程数不钉死，那些数字**不可复现**，报出去站不住。
#
# ⚠️ 放在包入口而非各脚本 `main()` 里，是因为本项目反复出现"新建入口忘了继承既有规则"
#    （见 `progress-log` §20.6 的种子流事故）。守卫见 `tests/test_thread_budget.py`。
#    代价：`import plcsp` 会连带加载 torch（约 1–2 s）——本包所有路径本就依赖它，可接受。
_torch.set_num_threads(1)
