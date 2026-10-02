"""plcsp.algo：策略与学习算法。

- `policy.py`     —— PolicyNet：机台选择（S 层）+ AGV 派车（L 层）两头
- `group_rel.py`  —— 组内相对 RL（GRPO 族）训练器：train_step / local_tree_step 等
- `runner.py`     —— 训练循环、保存与恢复

⚠️ 分批（B 层）与 SA-GRPO 树式分层**已砍除**（`progress-log.md` §12.5）。
"""
