"""错误归一化：将 httpx 异常 / 业务错误 → Gemini `error` 包装。

按 functional_requirements.md §2.5 + transformer.md §3 落地。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from app.core.logging import logger
from app.utils.retry import UpstreamRetryableError


# HTTP 状态码 → Gemini `status` 枚举（参考 GCP API errors）
_HTTP_TO_GEMINI_STATUS: Dict[int, str] = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    408: "DEADLINE_EXCEEDED",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    502: "INTERNAL",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}


def map_http_status_to_gemini_status(http_status: int) -> str:
    """将 HTTP 状态码映射为 Gemini 状态枚举。未知状态码 → UNKNOWN。"""
    return _HTTP_TO_GEMINI_STATUS.get(http_status, "UNKNOWN")


def build_gemini_error(
    code: int,
    message: str,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """构造 Gemini `error` 包装对象。"""
    if status is None:
        status = map_http_status_to_gemini_status(code)
    return {
        "error": {
            "code": code,
            "message": message,
            "status": status,
        }
    }


def _extract_upstream_message(exc: httpx.HTTPStatusError) -> str:
    """从上游错误响应中提取人类可读 message。"""
    resp = exc.response
    try:
        data = resp.json()
        # 优先取 OpenAI 风格 error.message，其次任意字符串字段
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and "message" in err:
                return str(err["message"])
            if "message" in data:
                return str(data["message"])
            if "detail" in data:
                return str(data["detail"])
        return resp.text or f"Upstream returned status {resp.status_code}"
    except Exception:
        return resp.text or f"Upstream returned status {resp.status_code}"


def handle_upstream_error(exc: BaseException) -> HTTPException:
    """将上游异常转换为 FastAPI HTTPException，detail 字段为 Gemini 错误对象。

    用途：在路由层捕获后 raise 出去，由 FastAPI 序列化为 JSON 响应。
    """
    # 1. 可重试状态码耗尽（来自 retry 装饰器）
    if isinstance(exc, UpstreamRetryableError):
        return HTTPException(
            status_code=exc.response.status_code,
            detail=build_gemini_error(
                code=exc.response.status_code,
                message=_extract_upstream_message(exc),
            ),
        )

    # 2. 上游 HTTP 4xx/5xx 错误（不重试的 4xx 业务错误）
    if isinstance(exc, httpx.HTTPStatusError):
        return HTTPException(
            status_code=exc.response.status_code,
            detail=build_gemini_error(
                code=exc.response.status_code,
                message=_extract_upstream_message(exc),
            ),
        )

    # 3. 上游网络错误（连接、超时、协议错误）—— 重试已耗尽仍失败
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError)):
        logger.error(f"Upstream network error (exhausted retries): {exc}")
        return HTTPException(
            status_code=502,
            detail=build_gemini_error(
                code=502,
                message=f"Upstream network error: {exc}",
                status="UNAVAILABLE",
            ),
        )
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        logger.error(f"Upstream timeout (exhausted retries): {exc}")
        return HTTPException(
            status_code=504,
            detail=build_gemini_error(
                code=504,
                message=f"Upstream timeout: {exc}",
                status="DEADLINE_EXCEEDED",
            ),
        )
    if isinstance(exc, httpx.RequestError):
        logger.error(f"Upstream request error: {exc}")
        return HTTPException(
            status_code=502,
            detail=build_gemini_error(
                code=502,
                message=f"Upstream request error: {exc}",
                status="UNAVAILABLE",
            ),
        )

    # 4. 未分类异常
    logger.exception(f"Internal proxy error: {exc}")
    return HTTPException(
        status_code=500,
        detail=build_gemini_error(
            code=500,
            message=f"Internal proxy error: {exc}",
            status="INTERNAL",
        ),
    )
