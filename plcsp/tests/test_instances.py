"""实例加载回归测试（bug#12 门禁）。

背景（2026-10-02 发现）：
1. `parse_fjs_text` 只读 .fjs 表头的两个 token，**不消费第三个**（该 token 是
   "每作业平均工序数"，Mk01=2 / Mk02=3.5 / Mk03=3 …）→ 全部错位，Mk02 直接崩。
2. `load_mk` 读的是第三方转换的 `.npy`，其 shape 与官方实例不符
   （Mk03: 10 vs 官方 8 机；Mk05: 9 vs 4；Mk08/09: 14 vs 10），
   且内容被转置——**MK01 逐工序对比 54/55 不符**。

本测试锁定不变量：**加载出的实例必须等于官方 Brandimarte 实例**。

官方维度（双重核对：.fjs 表头 + 已发表表格）：
    实例   jobs  machines  总工序
    Mk01    10      6        55
    Mk02    10      6        58
    Mk03    15      8       150
    Mk04    15      8        90
    Mk05    15      4       106
    Mk06    10     15       150
    Mk07    20      5       100
    Mk08    20     10       225
    Mk09    20     10       240
    Mk10    20     15       240
"""
from __future__ import annotations

from pathlib import Path

import pytest

from plcsp.env.instances import load_mk, parse_fjs_text

# 官方规格：(n_jobs, n_machines, 总工序数)
OFFICIAL: dict[str, tuple[int, int, int]] = {
    "mk01": (10, 6, 55),
    "mk02": (10, 6, 58),
    "mk03": (15, 8, 150),
    "mk04": (15, 8, 90),
    "mk05": (15, 4, 106),
    "mk06": (10, 15, 150),
    "mk07": (20, 5, 100),
    "mk08": (20, 10, 225),
    "mk09": (20, 10, 240),
    "mk10": (20, 15, 240),
}

FJS_DIR = (
    Path(__file__).resolve().parents[2]
    / "third_party" / "fjsp-gnnrl" / "evaluations" / "standard"
    / "brandimarte" / "brandimarte_dataset"
)


def _require_fjs(name: str) -> Path:
    p = FJS_DIR / f"{name.capitalize()}.fjs"
    if not p.exists():
        pytest.fail(f"官方实例文件缺失：{p}（多实例复现的硬依赖）")
    return p


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(OFFICIAL))
def test_instance_matches_official_dimensions(name: str) -> None:
    """每个实例的 作业数 / 机器数 / 总工序数 必须等于官方表。"""
    # Arrange
    expected_jobs, expected_machines, expected_ops = OFFICIAL[name]

    # Act
    inst = load_mk(name)

    # Assert
    assert inst.n_jobs == expected_jobs, f"{name} 作业数不符"
    assert inst.n_machines == expected_machines, f"{name} 机器数不符"
    assert sum(len(job) for job in inst.jobs) == expected_ops, f"{name} 总工序数不符"


@pytest.mark.unit
def test_mk01_first_operations_exact() -> None:
    """MK01 前几道工序的候选机台与工时逐位核对——防"维度对但内容被转置"。

    数据源：Mk01.fjs 原文
        job0: 6 工序，首工序 "2  1 5 3 4"  => 2 个候选：(机1,5),(机3,4)（1 基）=> (0,5.0),(2,4.0)
    """
    # Arrange
    inst = load_mk("mk01")

    # Act
    op0 = sorted(inst.jobs[0][0])
    op1 = sorted(inst.jobs[0][1])

    # Assert
    assert op0 == [(0, 5.0), (2, 4.0)], f"job0/op0 不符：{op0}"
    assert op1 == [(1, 1.0), (2, 5.0), (4, 3.0)], f"job0/op1 不符：{op1}"


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(OFFICIAL))
def test_parse_fjs_consumes_third_header_token(name: str) -> None:
    """**bug#12 的直接回归**：把 .fjs 原文件整份喂给 parse_fjs_text 必须正确。

    表头形如 `10  6  2`（第三个数=每作业平均工序数），解析器必须消费它，
    否则后续 token 全部错位（Mk02 的第三个数是 `3.5`，错位后直接崩）。
    """
    # Arrange
    raw_text = _require_fjs(name).read_text()
    expected_jobs, expected_machines, expected_ops = OFFICIAL[name]

    # Act
    n, m, jobs = parse_fjs_text(raw_text)

    # Assert
    assert (n, m) == (expected_jobs, expected_machines), f"{name} 维度不符"
    assert sum(len(j) for j in jobs) == expected_ops, f"{name} 总工序数不符"


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(OFFICIAL))
def test_parsed_jobs_per_job_matches_official(name: str) -> None:
    """逐作业工序数：.fjs 解析结果与 load_mk 必须一致（两路径同源）。"""
    # Arrange
    raw_text = _require_fjs(name).read_text()

    # Act
    _, _, jobs_raw = parse_fjs_text(raw_text)
    inst = load_mk(name)

    # Assert
    assert [len(j) for j in jobs_raw] == [len(j) for j in inst.jobs]


@pytest.mark.unit
def test_mk02_header_third_token_is_fractional() -> None:
    """Mk02 表头第三个 token 是 `3.5`——必须被当作"平均工序数"跳过，而不是当整数读。"""
    # Arrange
    raw_text = _require_fjs("mk02").read_text()
    assert raw_text.split()[2] == "3.5", "Mk02 表头第三 token 应为 3.5（本测试前提）"

    # Act
    n, m, jobs = parse_fjs_text(raw_text)

    # Assert
    assert (n, m) == (10, 6)
    assert sum(len(j) for j in jobs) == 58


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(OFFICIAL))
def test_no_degenerate_operations(name: str) -> None:
    """sanity：每道工序至少一个候选、工时为正、机台号在范围内。"""
    # Arrange
    inst = load_mk(name)

    # Act / Assert
    for j, job in enumerate(inst.jobs):
        for k, alts in enumerate(job):
            assert alts, f"{name} job{j} op{k} 无候选机台"
            for mach, t in alts:
                assert 0 <= mach < inst.n_machines, f"{name} job{j} op{k} 机台越界：{mach}"
                assert t > 0, f"{name} job{j} op{k} 工时非正：{t}"
