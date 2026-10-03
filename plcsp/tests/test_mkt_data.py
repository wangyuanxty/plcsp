"""MKT 布局数据的测试（P4-A Task 1）。

数据来源与口径见 `plcsp/data/mkt/README.md`；存疑点见 `progress-log §19.7d`。
"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.mkt import MKT_LAYOUT_DIR, load_mkt_layout

NEEDED = (4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 18)


@pytest.mark.unit
def test_all_layout_files_present():
    """12 个机台数的布局矩阵必须齐——少了任何一个，对应规模的实例都跑不了。"""
    missing = [m for m in NEEDED if not (MKT_LAYOUT_DIR / f"{m}_machine_layout.txt").exists()]
    assert not missing, f"缺布局文件：{missing}（目录 {MKT_LAYOUT_DIR}）"


@pytest.mark.unit
@pytest.mark.parametrize("m", NEEDED)
def test_layout_shape_is_m_plus_1_square(m: int):
    """原始矩阵是 (m+1)×(m+1)：**第 0 行/列是装卸站（LU）**（§19.7d）。"""
    raw = load_mkt_layout(m, drop_lu=False)
    assert raw.shape == (m + 1, m + 1), f"{m} 机布局维度应为 {m+1}，实得 {raw.shape}"


@pytest.mark.unit
@pytest.mark.parametrize("m", [6, 10, 15])
def test_drop_lu_gives_m_by_m_with_zero_diagonal(m: int):
    """⚠️ Review Focus #2：丢 LU 后必须是 m×m 且**对角为 0**（同机台间无行程）。"""
    t = load_mkt_layout(m, drop_lu=True)
    assert t.shape == (m, m)
    assert np.allclose(np.diag(t), 0.0), f"{m} 机布局丢 LU 后对角非 0：{np.diag(t)}"


@pytest.mark.unit
def test_matrix_is_asymmetric_and_crop_direction_is_pinned():
    """⚠️ Review Focus #2：行程时间矩阵**非对称**，且**裁剪方向**必须被直接钉住。

    ⚠️ 判据说明（这条是我先写错、后又改回来的）：`§19.7d` 记「10 机布局 **M[1][2]=3** 而
    **M[2][1]=12**」——**它是对的**，用的是 **1 基机台号**：M[1][2] = 丢 LU 后 0 基的 `t[0][1]`。
    我起初按 0 基读成 `t[1][2]`(2.5)/`t[2][1]`(4.5)，误判为"二手读数错"，实际是我索引错。

    ⭐ **为什么要钉具体值，而不只是"非对称"**：`t.shape`、对角、`not allclose(t, t.T)` 这三条
    **对两种错误实现同样成立**——验证过：
    - 裁错方向 `raw[:m, :m]`（该丢后一行一列，却丢了前一行一列）：形状同、对角同为 0、同样非对称；
    - 转置 `raw[1:, 1:].T`：形状同、对角不变、`t[1][2] != t[2][1]` 照样成立。

    只有把**具体方向上的具体值**钉死才能杀掉这两者。故下面同时钉 `t[0][1]` 与 `t[1][0]`。
    """
    t = load_mkt_layout(10, drop_lu=True)
    assert t[0][1] == pytest.approx(3.0), "10 机布局 t[0,1] 实测为 3——裁错方向会变成 2"
    assert t[1][0] == pytest.approx(12.0), "10 机布局 t[1,0] 实测为 12——裁错方向会变成 11"
    assert not np.allclose(t, t.T), "行程时间矩阵应非对称——疑似被对称化了"


@pytest.mark.unit
def test_values_are_measured_not_assumed():
    """⚠️ Review Focus #4：值域**实测**后钉住，不得照抄文献的「随机 2–10」。

    §19.7d 的侦查报「6 机到 17、15 机到 15」——本机复测**相符**（见 README 实测值域表）。
    12 个布局合起来的实测值域是 **1–17**，且含 **.5 的半整数**（2.5/4.5/…/10.5），
    与文献正文的「随机 2–10（整数）」不符（§19.7d 存疑点①）。
    """
    for m, expected_max in ((6, 17), (15, 15)):
        t = load_mkt_layout(m, drop_lu=True)
        got = float(t.max())
        assert got == pytest.approx(expected_max, abs=0.0), (
            f"{m} 机布局实测最大值 {got}，与 §19.7d 记录的 {expected_max} 不符——"
            "若数据源换了版本，请更新 progress-log 而不是改这个断言")


@pytest.mark.unit
def test_data_quirks_are_pinned_not_assumed():
    """⚠️ 两个**实测得到的数据异常**必须被钉住，不得靠"应该不会吧"的假设过日子：

    1. **m=17 的对角线非零**：`t[5][5]=4`——上游文件里第 6 台机到它自己有 4 分钟行程。
       这是唯一一个对角非零的布局（其余 11 个对角全 0）。
    2. **m=4 的矩阵恰好对称**：它是全部 12 个里唯一对称的一个
       ——故"非对称"只能钉在 m=10 上，不能对 12 个规模一刀切。

    两处都已在 `plcsp/data/mkt/README.md` 如实记录。
    """
    diag17 = np.diag(load_mkt_layout(17, drop_lu=True))
    assert diag17[5] == pytest.approx(4.0), (
        f"17 机布局对角第 6 项实测 {diag17[5]}，README 记的是 4.0")
    assert np.count_nonzero(diag17) == 1, f"17 机布局只有一处对角非零，实得 {np.count_nonzero(diag17)} 处"

    t4 = load_mkt_layout(4, drop_lu=True)
    assert np.allclose(t4, t4.T), "4 机布局实测对称——若不再对称，README 要改"

    # 其余 10 个规模：对角为 0（m=17 是唯一例外，已在上面钉住）
    odd = [m for m in NEEDED if m != 17 and not np.allclose(np.diag(load_mkt_layout(m)), 0.0)]
    assert not odd, f"除 17 外还有规模对角非零：{odd}"


@pytest.mark.unit
def test_missing_machine_count_raises():
    """⚠️ Review Focus #6：取不到的机台数必须**显式报错**，不得静默回退。"""
    with pytest.raises((FileNotFoundError, ValueError)):
        load_mkt_layout(7)          # 7 不在 4..18 的已发布集合里
