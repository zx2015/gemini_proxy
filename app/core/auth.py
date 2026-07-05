"""多模式 API Key 鉴权。

按 functional_requirements.md §2.1 要求，兼容以下三种凭据传递方式：
  1. HTTP Header `x-goog-api-key: <key>`（Google GenAI SDK 默认）
  2. HTTP Header `Authorization: Bearer <key>`（OpenAI 风格）
  3. URL Query 参数 `?key=<key>`（旧版 GenAI SDK 行为）

任一位置凭据匹配 `settings.proxy_api_key` 即视为通过。
"""
from __future__ import annotations
import hmac

from typing import Optional

from fastapi import HTTPException, Request, status

from app.core.config import settings
from app.core.logging import logger


def _extract_credentials(request: Request) -> Optional[str]:
    """按优先级从三个位置提取凭据。"""
    # 1. x-goog-api-key header
    key = request.headers.get("x-goog-api-key")
    if key:
        return key.strip()

    # 2. Authorization: Bearer
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()

    # 3. ?key= query param
    key = request.query_params.get("key")
    if key:
        return key.strip()

    return None


async def verify_api_key(request: Request) -> str:
    """FastAPI Depends 入口；验证失败抛 401。"""
    client_ip = request.client.host if request.client else "<unknown>"
    token = _extract_credentials(request)

    if not token:
        logger.warning(f"Auth failed (missing credentials) from {client_ip} {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via x-goog-api-key header, "
                   "Authorization: Bearer header, or ?key= query param.",
        )

    if not hmac.compare_digest(token, settings.proxy_api_key):
        logger.warning(f"Auth failed (invalid key) from {client_ip} {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return token
