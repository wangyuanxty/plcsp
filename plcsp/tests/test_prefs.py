"""`--prefs`（A 主对比的单目标配置，2026-10-05）——固定偏好权重**取代** `w`。

## 为什么单独立一个测试文件

`spec §6.1` 的第 ② 层是"单目标 GRPO × N 组权重"（用户裁定 N=3 = 三个 one-hot）。
`reward.scalar_reward` 早就预留了 `prefs` 形参、`m13` 却**没有 CLI 入口** ⟹ 这一层
此前**表达不出来**。本文件钉住新入口的四条契约：

1. **默认关 ⟹ 逐位不变**：`prefs=None` 走 f^ref 派生的 `w`，表达式与今日一字不差
   （整步的逐位不变由 `test_joint_chain.test_scalar_adv_mode_regression_pin` 的
   捕获摘要兜底——那条比"两次调用相等"强）。
2. **取代而不是相乘**：`Σ prefsᵢ(−fᵢ)`，与把 prefs 当 `w` 传入**完全相等**。
3. **one-hot = 单目标**：`A = z(−fᵢ)`，且原始量纲被组内 z 化消掉（尺度不进优势）。
4. **全 0 / 负数 / 长度不对 ⟹ 显式报错**（全 0 是"静默空转"，跑满 300 步一步不学）。
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from plcsp import m13_train_a as m13
from plcsp.algo.group_rel import ADV_MODES, _advantages, check_prefs, joint_chain_step
from plcsp.algo.policy import PolicyNet
from plcsp.algo.setup import build_setup
from plcsp.env.des import SimConfig
from plcsp.env.instances import load_mk
from plcsp.env.reward import ReferenceObjectives, scalar_reward
from plcsp.nn.encoder import LayoutEncoder

# 造一组"三个目标尺度差很大"的原始目标值（越小越好）——one-hot 的判别力全靠它：
# makespan ~10²、energy ~10¹、TWT ~10⁰。
_F = np.array([[31.0, 2.5, 0.0], [47.0, 3.25, 6.0], [58.0, 1.75, 12.0], [39.0, 4.0, 3.0]])
_W = (0.2, 0.5, 0.3)


def test_default_prefs_is_bitwise_unchanged():
    """`prefs=None` ⟹ 与不传 prefs 的表达式**逐位相同**（不是近似）。

    这是"默认关 ⟹ 既有读数逐位不变"的本层证据；整步的证据在
    `test_joint_chain.test_scalar_adv_mode_regression_pin`（捕获摘要）。
    """
    f = (tuple(_F[0]))
    assert scalar_reward(f, _W) == scalar_reward(f, _W, None)
    for mode in ADV_MODES:
        a = _advantages(_F, _W, mode)
        b = _advantages(_F, _W, mode, prefs=None)
        assert torch.equal(a, b), f"{mode}：prefs=None 与本日表达式不再逐位相同"


def test_prefs_replaces_w_not_multiplies():
    """prefs 给定时**取代** `w`：`Σ prefsᵢ(−fᵢ)`——不是 `wᵢ·prefsᵢ` 的乘积。"""
    p = (1.0, 0.0, 0.0)
    f = tuple(_F[1])
    assert scalar_reward(f, _W, p) == scalar_reward(f, p)
    # 相乘会得到 w₁·p₁ = 0.2 —— 与取代口径（1.0）差一个量级，本断言当场分开
    assert scalar_reward(f, _W, p) != scalar_reward(f, tuple(w * q for w, q in zip(_W, p)))
    for mode in ("scalar", "reinforce"):
        a = _advantages(_F, _W, mode, prefs=p)
        b = _advantages(_F, p, mode)
        assert torch.equal(a, b), f"{mode}：prefs 与「把 prefs 当 w 传」不等价 ⟹ 不是取代"


def test_one_hot_prefs_gives_the_single_objective_advantage():
    """one-hot `eᵢ` ⟹ `A = z(−fᵢ)`（逐元素），能量纲被 z 化消掉。"""
    for i in range(3):
        p = tuple(1.0 if k == i else 0.0 for k in range(3))
        got = _advantages(_F, _W, "scalar", prefs=p).numpy()
        neg = -_F[:, i]
        want = (neg - neg.mean()) / (neg.std() + 1e-9)
        assert np.allclose(got, want, rtol=0, atol=1e-6), (
            f"one-hot {p} 的优势不是 z(−f_{i})——单目标配置的含义被改了")


def test_one_hot_scale_is_cancelled_by_the_group_z_score():
    """同一目标**换单位**（×1000）不改变优势——尺度被 z 化消掉，prefs 只改相对权重。"""
    a = _advantages(_F, _W, "scalar", prefs=(1.0, 0.0, 0.0))
    scaled = _F.copy()
    scaled[:, 0] *= 1000.0
    b = _advantages(scaled, _W, "scalar", prefs=(1.0, 0.0, 0.0))
    assert torch.allclose(a, b, atol=1e-5), "量纲进了优势 ⟹ z 化没消掉尺度"


def test_prefs_guard_rejects_bad_values():
    """长度 / 负值 / 非有限 / 全 0 一律显式报错；合法输入原样返回（转 float）。"""
    for bad in [(1.0, 1.0), (1.0, 1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
                (float("nan"), 0.0, 1.0), (0.0, 0.0, 0.0)]:
        with pytest.raises(ValueError):
            check_prefs(bad)
    assert check_prefs((1, 0, 0)) == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="prefs"):
        _advantages(_F, _W, "scalar", prefs=(0.0, 0.0, 0.0))


def _fresh_policy() -> PolicyNet:
    """同一起点的策略——同 seed 下两条链必须逐位同源，否则下面的对照不成立。"""
    torch.manual_seed(1234)
    return PolicyNet(enc=LayoutEncoder())


def _one_step(prefs):
    """在 **MK01** 上跑一步（G=2）：返回 diag。每次同一初始化 ⟹ 链同源。

    ⚠️ 用 mk01 而不是 `gen_random`：交期标定表是**逐实例名**的，合成实例查不到 (τ, R) 会
    显式报错（`due_dates.due_dates_for`）；且 G=2 一步实测 ~1–3 s，三个 one-hot 可接受。
    """
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    ref = ReferenceObjectives.of(inst, cfg)
    _, diag = joint_chain_step(_fresh_policy(), inst, lay, dm, cfg, ctx, ref,
                               seed=0, G=2, prefs=prefs)
    return diag


@pytest.mark.slow
def test_joint_step_rewards_follow_prefs():
    """整步证据：三个 one-hot 与默认配置的 `r_mean` **两两不同**。

    ⚠️ 这是"prefs 真的接进了训练环（含 T3 路径共用的 `rewards`）"的绊线：
    只要哪一处漏传 prefs，同 seed 同初始化下四条链逐位同源 ⟹ 四个 `r_mean` 会**全部相等**，
    本测试当场红。相比"断言 r_mean < −10"这类量级假设，两两不同不依赖任何实例性质。
    """
    r_default = _one_step(None)["r_mean"]
    r_ms = _one_step((1.0, 0.0, 0.0))["r_mean"]
    r_en = _one_step((0.0, 1.0, 0.0))["r_mean"]
    vals = [round(r_default, 9), round(r_ms, 9), round(r_en, 9)]
    assert len(set(vals)) == 3, (
        f"default/纯 makespan/纯 energy 的 r_mean 出现了相等值 {vals}——"
        "prefs 没被接进奖励（或哪一份分支漏传），单目标配置跑出来的是同一个东西")


def test_cli_exposes_prefs_and_rejects_all_zero(capsys):
    """`--prefs` 必须在 CLI 里（A 第 ② 层只能靠它表达）；全 0 在**开工前**报错。"""
    with pytest.raises(SystemExit):
        sys.argv = ["m13_train_a.py", "--help"]
        m13.main()
    out = capsys.readouterr().out
    assert "--prefs" in out, "CLI 少了 --prefs——单目标配置表达不出来"

    sys.argv = ["m13_train_a.py", "--prefs", "0", "0", "0"]
    with pytest.raises(ValueError, match="prefs"):
        m13.main()          # 守卫在 argparse 之后、任何重活之前
