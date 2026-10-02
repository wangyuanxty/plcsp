"""plcsp.algo：策略与学习算法。

- `policy.py`     —— PolicyNet：机台选择（S 层）+ AGV 派车（L 层）两头
- `group_rel.py`  —— 联合链组内相对 RL（GRPO 族）训练器：`roll_chain` / `chain_logp` /
  `joint_chain_step`（P2 Task 7；旧的 `train_step` / `l_seq_step*` / `local_l_step` 已删）
- `runner.py`     —— 训练循环、保存与恢复

⚠️ 分批（B 层）与 SA-GRPO 树式分层**已砍除**（`progress-log.md` §12.5）。
"""
