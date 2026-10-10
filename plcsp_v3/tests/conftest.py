"""pytest 配置（与 `plcsp/tests/conftest.py` 同形；放在包内，不在项目根新增文件）。"""
from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "slow: 慢测试（完整训练等）")
