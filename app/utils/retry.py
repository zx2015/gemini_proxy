"""上游重试装饰器（仅非流式请求）。

按 functional_requirements.md §2.8 落地：
  - 仅对 5xx / 429 / 网络超时 / 连接错误 重试
  - 不对 4xx 业务错误重试
  - 指数退避 1s/2s/4s（默认 base=1.0）
  - 默认最大 3 次尝试（含首次）
  - 流式请求**不**走此装饰器

实现说明：
  - 选用 `tenacity` 实现重试循环，配置清晰、易测。
  - 关键：4xx 业务错误需要先 raise 出来再被 tenacity 识别为不可重试——
    这里使用一个包装函数 `_raise_if_non_retryable_status`，
    在 `client.post` 后若拿到 4xx 立即 raise `httpx.HTTPStatusError`，
    tenacity 通过 `retry_if_exception` 排除。
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import RETRYABLE_STATUS_CODES, settings
from app.core.logging import logger


T = TypeVar("T")


# 视为可重试的异常类型：
#   - httpx.RequestError 的子类：连接错误、读超时、写超时、连接池超时、协议错误
#   - HTTPStatusError 包装为可重试状态时（由 _raise_if_retryable_status 抛出）
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.NetworkError,
)


class UpstreamRetryableError(httpx.HTTPStatusError):
    """对可重试的 HTTP 状态码（5xx / 429 / 408）抛出的标记异常。

    继承 HTTPStatusError 以便上层捕获时仍能拿到 status_code。
    """


def _raise_if_retryable_status(response: httpx.Response) -> None:
    """检查上游响应状态码，若可重试则抛出 UpstreamRetryableError。"""
    if response.status_code in RETRYABLE_STATUS_CODES:
        raise UpstreamRetryableError(
            f"Upstream returned retryable status {response.status_code}",
            request=response.request,
            response=response,
        )
    # 4xx 业务错误 → raise_for_status() 标准路径
    response.raise_for_status()


def get_retry_decorator() -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """返回 tenacity 装饰器实例。

    使用方式：
        @get_retry_decorator()
        async def call_upstream(...): ...
    """
    max_attempts = settings.retry_max_attempts

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(
                    multiplier=settings.retry_base_delay,
                    exp_base=2,
                    min=settings.retry_base_delay,
                    max=settings.retry_base_delay * (2 ** (max_attempts - 1)),
                ),
                retry=retry_if_exception_type(
                    RETRYABLE_EXCEPTIONS + (UpstreamRetryableError,)
                ),
                reraise=True,
            ):
                with attempt:
                    attempt_number = attempt.retry_state.attempt_number
                    try:
                        return await fn(*args, **kwargs)
                    except UpstreamRetryableError as e:
                        logger.warning(
                            f"Upstream retryable error "
                            f"(attempt {attempt_number}/{max_attempts}, "
                            f"status={e.response.status_code}), "
                            f"will retry"
                        )
                        raise
                    except RETRYABLE_EXCEPTIONS as e:
                        logger.warning(
                            f"Upstream network error "
                            f"(attempt {attempt_number}/{max_attempts}, "
                            f"type={type(e).__name__}), will retry"
                        )
                        raise
            # reraise=True 时，重试耗尽会直接抛出最后一次异常；
            # 不可重试的异常（4xx 业务错误）也会直接抛出。

        return wrapper

    return decorator


def is_retryable_status(status_code: int) -> bool:
    """工具函数：判断 HTTP 状态码是否可重试。"""
    return status_code in RETRYABLE_STATUS_CODES
