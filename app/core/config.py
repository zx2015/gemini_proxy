"""Pydantic Settings 配置管理。

设计原则：
- 单一 Settings 实例，全局通过 `settings` 访问。
- 必填项在启动期校验失败即抛出，避免运行期才发现。
- 与 .env 文件协同；未知变量被忽略（`extra="ignore"`），便于将来扩展。
"""
from __future__ import annotations

from typing import Set

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 可重试的 HTTP 状态码（与 functional_requirements.md §2.8.2 对齐）
RETRYABLE_STATUS_CODES: Set[int] = {408, 429, 500, 502, 503, 504}


class Settings(BaseSettings):
    """网关运行配置。"""

    # ---- 上游 OpenAI 兼容服务 ----
    upstream_openai_url: str = Field(
        ...,
        description="上游 OpenAI 兼容服务地址（不含 /v1 后缀），如 http://localhost:4000",
    )
    upstream_api_key: str = Field(
        ...,
        description="注入到上游 Authorization: Bearer 的凭据",
    )
    upstream_model: str = Field(
        ...,
        description=(
            "强制使用的上游模型名（如 gpt-4o）。"
            "客户端传入的 model 字段被完全忽略，永远使用此值。"
        ),
    )

    # ---- 网关入站鉴权 ----
    proxy_api_key: str = Field(
        ...,
        description="客户端访问本网关的凭据",
    )

    # ---- 监听 ----
    host: str = "0.0.0.0"
    port: int = 8000

    # ---- 日志 ----
    log_level: str = "INFO"

    # ---- HTTP 行为 ----
    request_timeout: float = Field(
        600.0,
        description="上游单次请求超时（秒）",
    )
    retry_max_attempts: int = Field(
        3,
        ge=1,
        description="上游非流式请求最大尝试次数（含首次）",
    )
    retry_base_delay: float = Field(
        1.0,
        ge=0.0,
        description="重试退避基数（秒）。实际延迟 = base * 2^(attempt-1)",
    )

    # ---- 模型发现 ----
    model_discovery_cache_ttl: int = Field(
        300,
        ge=0,
        description="上游 /v1/models 缓存秒数；0 表示不缓存",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        """统一为大写，并校验合法值。"""
        v_norm = v.upper()
        if v_norm not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"非法 LOG_LEVEL: {v}")
        return v_norm


# 全局单例
settings = Settings()
