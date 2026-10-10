# -*- coding: utf-8 -*-
"""**实验 A 的跑数器**（`a_main.py`）的判据。

⚠️ **跑数器本身不测端到端**（那要真跑 200 步）—— 同 `q1_flat.py` / `b_ablation.py` 的惯例。
这里只钉两件**写错了就会让读数错**的事：

1. `stage_b_idle` 的计数（实验 A 的"空转"读数靠它）；
2. **扁平臂用哪个优势口径** —— 那是"两臂只差分层"这个前提的守门人。
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from plcsp_v3 import a_main                                       # noqa: E402


@pytest.mark.unit
def test_stage_b_idle_counts_groups_whose_advantages_are_all_zero():
    """**空转 = 该顶层点的 M 条分支优势全为 0**（= 组内段回报全相同 ⟹ `Â ≡ 0`）。

    ⚠️ 判据是"**整组**全 0"，不是"有 0"。组内 z 化后 `Σ Â ≡ 0`，
    所以**任何**一组里都必然有正有负；**全 0 只可能是"组内回报全相同"**。
    """
    segs = [{"j": 0, "adv": 0.0}, {"j": 0, "adv": 0.0},     # 空转
            {"j": 1, "adv": 1.0}, {"j": 1, "adv": -1.0},    # 正常
            {"j": 2, "adv": 0.0}, {"j": 2, "adv": 0.5}]     # 正常（只一个有 0）
    assert a_main.stage_b_idle(segs) == (1, 3)
    assert a_main.stage_b_idle([]) == (0, 0)


@pytest.mark.unit
def test_flat_arm_uses_group_zscoring_not_a_critic():
    """⚠️ **扁平臂必须用 `B`（组内 z 化），不是 `A`/`A2`（critic）**。

    实验 A 的问的是"**分层有没有用**"。两臂只该差**这一个**变量。
    `B` 的优势是组内 z 化，与两层两个阶段用的 `train.group_advantage` **同源**；
    换成 `A` 会引入**第二个**变量（有没有 critic）⟹ 读出的差异分不清是哪个起的作用。

    （这正是 `hrl/flat.py` 口径 ②"两臂的更新规则逐字相同"的延伸。）
    """
    assert a_main.FLAT_ARM_INTERNAL == "B"
    assert a_main.ARMS == ("flat", "hrl")
