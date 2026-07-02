"""pytest 全局 fixtures / 配置。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# 确保 /media/data/venv 默认 venv 优先
_VENV_BIN = Path("/media/data/venv/bin")
if _VENV_BIN.exists() and str(_VENV_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"


# 在导入 app 之前注入必需的 env vars（Pydantic 启动校验）
os.environ.setdefault("UPSTREAM_OPENAI_URL", "http://localhost:4000")
os.environ.setdefault("UPSTREAM_API_KEY", "sk-test")
os.environ.setdefault("UPSTREAM_MODEL", "gpt-4o-test")
os.environ.setdefault("PROXY_API_KEY", "test-proxy-key")


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """每个测试前清空 lru_cache（如有）。当前 settings 无缓存，留作扩展。"""
    yield
