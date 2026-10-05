"""① 期消融用的两个开关（2026-10-05）。

## 背景

`experiment-plan.md` §0.1 把实验分了三期，① 期在 MK01 上跑全部实验。核对下来缺两个开关：

1. **约束组**（消融 B 的 5 组）——`ABLATION_GROUPS` 早就在 `env/constraints.py` 里，
   但 `m13` 只走默认（全开），**CLI 没暴露**；
2. **优势口径 `reinforce`**（消融 D 的对照）——`adv_mode` 只有 `scalar` / `per_objective`。

## 本文件钉什么

- **约束组的组名与 CLI 的 choices 一致**（含那三个以 `-` 开头的组名——它们在 argparse 里
  是**陷阱**：`--constraints -物流` 会被当成选项，必须写 `--constraints=-物流`）。
- **`reinforce` 只改一个变量**：把"除以组内标准差"去掉，**其余一律不动**。
  这是消融 D 能成立的前提——若连均值也不减，梯度尺度会绑到 r 的量纲上，
  测出来的是"两件事叠加"，不是"std 归一化有没有用"。
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from plcsp import m13_train_a as m13
from plcsp.algo.group_rel import ADV_MODES, _advantages
from plcsp.env.constraints import ABLATION_GROUPS
from plcsp.env.reward import scalar_reward


def test_cli_exposes_every_ablation_group(capsys):
    """消融 B 的 5 个组名必须都在 CLI 的 choices 里。"""
    with pytest.raises(SystemExit):
        sys.argv = ["m13_train_a.py", "--help"]
        m13.main()
    out = capsys.readouterr().out
    for g in ABLATION_GROUPS:
        assert g in out, f"CLI 的 --constraints 少了组名 {g!r}"


def test_cli_exposes_every_adv_mode(capsys):
    """消融 D 要用 `reinforce`，它必须在 CLI 的 choices 里。"""
    with pytest.raises(SystemExit):
        sys.argv = ["m13_train_a.py", "--help"]
        m13.main()
    out = capsys.readouterr().out
    for m in ADV_MODES:
        assert m in out, f"CLI 的 --adv-mode 少了 {m!r}"


def test_dash_prefixed_group_needs_the_equals_form():
    """⚠️ **以 `-` 开头的组名在 argparse 里是陷阱**：裸写会被当成选项。

    这条钉住"必须用等号形式"这件事本身——否则 ① 期的脚本会**在跑到第 2 组时静默失败**
    （或报一个看不出原因的 argparse 错）。
    """
    dash_groups = [g for g in ABLATION_GROUPS if g.startswith("-")]
    assert dash_groups, "前提：ABLATION_GROUPS 里确实有以 - 开头的组名（否则本测试没意义）"
    with pytest.raises(SystemExit):
        sys.argv = ["m13_train_a.py", "--constraints", dash_groups[0]]   # ← 裸写，应当失败
        m13.main()


def test_reinforce_only_removes_the_std_division():
    """⭐ **消融 D 的核心判据**：`reinforce` 与 `scalar` 只差"除不除 std"。

    造一组 r（mean ≠ 0、std ≠ 1），则 `scalar` 的 A 恰是 `reinforce` 的 A 除以 std。
    **差出别的东西就说明改了不止一个变量。**
    """
    f = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
    w = (1 / 3, 1 / 3, 1 / 3)

    a_sc = _advantages(f, w, "scalar").numpy()
    a_rf = _advantages(f, w, "reinforce").numpy()

    r = np.asarray([scalar_reward(tuple(o), w) for o in f])
    std = r.std()
    assert not np.isclose(std, 1.0), "前提：std ≠ 1，否则两种口径恒等、测不出差别"
    assert not np.isclose(r.mean(), 0.0), "前提：mean ≠ 0，否则「减不减均值」也测不出"

    # reinforce = 只减均值
    assert np.allclose(a_rf, r - r.mean(), atol=1e-6), "reinforce 不是 r − mean"
    # scalar = (r − mean)/(std + eps) ⟹ 两者恰好差一个 std
    assert np.allclose(a_sc, a_rf / (std + 1e-9), atol=1e-6), \
        "scalar 与 reinforce 的差不止一个 std——说明这两个口径改了不止一个变量"


def test_reinforce_is_on_the_whitelist():
    """白名单守卫必须认它（写错名不得静默回退到别的口径）。"""
    assert "reinforce" in ADV_MODES
    with pytest.raises(ValueError, match="adv_mode"):
        _advantages(np.ones((3, 3)), (1 / 3, 1 / 3, 1 / 3), "reinforc")   # 拼错
