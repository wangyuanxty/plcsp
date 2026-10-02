"""一致性回归测试——锁定 P0 整支评审的三项 Important 发现。

对应评审发现：
1. **包 docstring 仍描述已砍路线**（SA-GRPO / 几何不变特征 / 双轴）——spec §7 与 §12#6
   逐字要求"改名时一并重写"，计划只做了机械改名，语义未改。
2. **docs/INDEX.md 自称唯一入口，却没指向权威 spec**，且仍把已作废的 `method-design.md`
   标为"写论文的主依据"。
3. **唯一的回归门禁在新克隆上必然全红**——`test_instances.py` 依赖被 `.gitignore` 排除的
   1.5 GB `third_party/`；而所需的 10 个官方 `.fjs` 合计仅约 21 KB。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
BRANDIMARTE_FIXTURES = Path(__file__).resolve().parents[1] / "data" / "brandimarte"

# P0 明确砍除的概念——不得再被**肯定地**描述为包的能力。
# 注意 "双轴" **不在**此列：轴向分解是当前编码器的真实架构，P0 只删了几何偏置，
# 按计划保留轴掩码逻辑（P2 才改标准 Transformer）。
RETIRED_TERMS = ("SA-GRPO", "sagrpo", "几何不变")
# 同一行若含下列标记，视为"说明其已废弃"的正当留痕（防止有人把它们加回来）
RETIREMENT_MARKERS = ("已砍", "砍除", "已删", "已废弃", "已作废", "不再", "勿依据")


@pytest.mark.unit
@pytest.mark.parametrize("modname", ["plcsp", "plcsp.algo", "plcsp.nn", "plcsp.env"])
def test_package_docstring_free_of_retired_terms(modname: str) -> None:
    """包 docstring 不得再把已砍路线**说成自己的能力**（`help()` / IDE 悬浮会骗人）。

    注意：**允许**"X 已砍除"这类留痕——它防止后来者把砍掉的东西加回来。故判据是
    "同句内出现已砍术语 **且** 无废弃标记"才算违规。
    """
    # Arrange
    doc = importlib.import_module(modname).__doc__ or ""

    # Act
    offenders = [
        line.strip() for line in doc.splitlines()
        if any(t.lower() in line.lower() for t in RETIRED_TERMS)
        and not any(m in line for m in RETIREMENT_MARKERS)
    ]

    # Assert
    assert not offenders, f"{modname} 的 docstring 仍在肯定地描述已砍路线：{offenders}"


@pytest.mark.unit
def test_index_points_to_authoritative_spec() -> None:
    """INDEX.md 是"唯一入口"，其文档地图必须指向权威 spec。"""
    # Arrange
    text = (DOCS / "INDEX.md").read_text(encoding="utf-8")

    # Act / Assert
    assert "2026-10-02-plcsp-rebuild-design.md" in text, "文档地图未登记权威 spec"
    assert "已取代" in text or "已作废" in text, "未标出被取代的旧设计文档"


@pytest.mark.unit
def test_brandimarte_fixtures_present_and_complete() -> None:
    """10 个官方 .fjs 必须随包入库——否则新克隆上 42 项回归门禁全红。"""
    # Arrange / Act
    files = sorted(BRANDIMARTE_FIXTURES.glob("Mk*.fjs"))

    # Assert
    assert len(files) == 10, f"夹具不全：找到 {len(files)} 个，应为 10 个（{BRANDIMARTE_FIXTURES}）"
    for i in range(1, 11):
        p = BRANDIMARTE_FIXTURES / f"Mk{i:02d}.fjs"
        assert p.exists(), f"缺 {p.name}"
        assert p.stat().st_size > 100, f"{p.name} 体积异常（{p.stat().st_size} B）"


@pytest.mark.unit
def test_load_mk_defaults_to_bundled_fixtures() -> None:
    """`load_mk()` 默认应读随包夹具，而非被 ignore 的 third_party/。"""
    # Arrange / Act
    from plcsp.env.instances import load_mk

    inst = load_mk("mk01")

    # Assert
    assert "third_party" not in inst.source, f"仍依赖 third_party：{inst.source}"
    assert (inst.n_jobs, inst.n_machines) == (10, 6)
