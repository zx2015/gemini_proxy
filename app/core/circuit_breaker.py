"""熔断器（Circuit Breaker）——保护上游 LiteLLM 不被级联打垮。

三态状态机（参考 cc-switch circuit_breaker.rs 模式）：
  CLOSED   → 正常放行；连续失败 >= failure_threshold 进入 OPEN
  OPEN     → 拒绝所有请求，直接返回 503；recovery_timeout 秒后进入 HALF_OPEN
  HALF_OPEN → 放行最多 half_open_max_calls 个探针请求；
              探针成功回 CLOSED；探针失败回 OPEN

计入失败的情况：上游 5xx、网络错误（httpx.RequestError）、流式首字节/idle 超时。
不计入失败：4xx 业务错误（如 400/401/422）——这些是请求本身的问题，不是上游不可用。
"""
from __future__ import annotations

import asyncio
import enum
import time
from typing import Optional

from app.core.logging import logger


class CircuitState(enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpen(Exception):
    """上游熔断器处于 OPEN/HALF_OPEN 状态，本次请求被直接拒绝。"""


class CircuitBreaker:
    """异步安全的三态熔断器。"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def before_call(self) -> None:
        """在发起上游请求前调用。OPEN 状态且尚未到恢复时间则抛 CircuitBreakerOpen。"""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - (self._last_failure_time or 0.0)
                if elapsed >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.warning(
                        f"[CircuitBreaker] OPEN → HALF_OPEN "
                        f"(elapsed={elapsed:.0f}s >= recovery={self._recovery_timeout}s)"
                    )
                else:
                    remaining = self._recovery_timeout - elapsed
                    raise CircuitBreakerOpen(
                        f"Circuit breaker OPEN; retry after {remaining:.0f}s"
                    )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    raise CircuitBreakerOpen(
                        "Circuit breaker HALF_OPEN; probe already in flight"
                    )
                self._half_open_calls += 1

    async def record_success(self) -> None:
        """上游请求成功后调用——重置计数，HALF_OPEN 回 CLOSED。"""
        async with self._lock:
            prev_failures = self._failure_count
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.warning("[CircuitBreaker] HALF_OPEN → CLOSED (probe succeeded)")
            elif prev_failures > 0:
                logger.info(f"[CircuitBreaker] failure count reset (was {prev_failures})")

    async def record_failure(self) -> None:
        """上游可重试失败后调用——累计计数，超阈值后打开熔断。"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                logger.warning(
                    f"[CircuitBreaker] HALF_OPEN → OPEN (probe failed; "
                    f"total_failures={self._failure_count})"
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(
                    f"[CircuitBreaker] CLOSED → OPEN "
                    f"(failures={self._failure_count} >= threshold={self._failure_threshold})"
                )

    def status_dict(self) -> dict:
        """返回当前熔断器状态摘要（用于 /health/upstream 端点）。"""
        elapsed = (
            time.monotonic() - self._last_failure_time
            if self._last_failure_time
            else None
        )
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
            "seconds_since_last_failure": round(elapsed, 1) if elapsed is not None else None,
            "recovery_timeout": self._recovery_timeout,
        }


# ---- 全局单例（lazy init，避免 import-time 副作用）----

_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """返回全局熔断器单例，首次调用时从 settings 读取配置初始化。"""
    global _circuit_breaker
    if _circuit_breaker is None:
        from app.core.config import settings

        _circuit_breaker = CircuitBreaker(
            failure_threshold=settings.cb_failure_threshold,
            recovery_timeout=settings.cb_recovery_timeout,
            half_open_max_calls=settings.cb_half_open_max_calls,
        )
    return _circuit_breaker
