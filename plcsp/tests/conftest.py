"""pytest 配置（放在包内，避免在项目根新增文件——见项目文档约定）。"""
from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "slow: 慢测试（完整训练等）")
