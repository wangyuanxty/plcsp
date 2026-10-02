"""plcsp：生产-物流协同调度（PLCSP）的深度强化学习研究包。

问题 = 两环节联动：排产（FJSP）→ 物流（真 AGV 派车）。
骨架 = Transformer 编码器 + 组内相对 RL（GRPO）。
权威设计见 `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`。
"""
