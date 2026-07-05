"""Gemini 协议路由层。

按 docs/design/architecture.md §4 + functional_requirements.md §2 落地。

路由：
  - POST /v1beta/models/{model}:generateContent   (非流式，带重试)
  - POST /v1beta/models/{model}:streamGenerateContent (流式，不重试)
  - POST /v1beta/models/{model}:countTokens       (token 计数)
  - GET  /v1beta/models                           (模型列表)
  - GET  /v1beta/models/{model}                   (单模型详情)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.auth import verify_api_key
from app.core.config import settings
from app.core.logging import logger
from app.services.discovery import model_discovery
from app.services.stream.processor import StreamProcessor
from app.services.transformer.from_openai import response_transformer
from app.services.transformer.to_openai import request_transformer
from app.utils.error_handler import build_gemini_error, handle_upstream_error, map_http_status_to_gemini_status
from app.utils.retry import (
    UpstreamRetryableError,
    _raise_if_retryable_status,
    get_retry_decorator,
)


router = APIRouter()

# 模块级共享 AsyncClient（连接池复用，延迟初始化）
_upstream_client: Optional[httpx.AsyncClient] = None


def _get_upstream_client() -> httpx.AsyncClient:
    """返回模块级共享的 AsyncClient，首次调用时创建。"""
    global _upstream_client
    if _upstream_client is None or _upstream_client.is_closed:
        _upstream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _upstream_client


async def close_upstream_client() -> None:
    """关闭共享 AsyncClient（在 lifespan 关闭阶段调用）。"""
    global _upstream_client
    if _upstream_client and not _upstream_client.is_closed:
        await _upstream_client.aclose()
    _upstream_client = None


# ====================================================================
# 1. 非流式 generateContent（带重试）
# ====================================================================

@router.post(
    "/v1beta/models/{model:path}:generateContent",
    dependencies=[Depends(verify_api_key)],
)
async def generate_content(
    request: Request,
    model: str = Path(..., description="Gemini 模型名（仅用于日志，不影响出站）"),
) -> JSONResponse:
    """非流式生成。带指数退避重试。"""
    body = await request.json()
    inbound_model = model

    # 1. 转换请求体
    openai_body = request_transformer.transform(body, stream=False, inbound_model=inbound_model)

    # 2. 构造上游请求（带重试装饰器）
    upstream_url = f"{settings.upstream_openai_url.rstrip('/')}/v1/chat/completions"
    upstream_headers = {
        "Authorization": f"Bearer {settings.upstream_api_key}",
        "Content-Type": "application/json",
    }


    @get_retry_decorator()
    async def _call_upstream() -> Dict[str, Any]:
        client = _get_upstream_client()
        resp = await client.post(upstream_url, json=openai_body, headers=upstream_headers)
        _raise_if_retryable_status(resp)  # 5xx/429 → 抛 UpstreamRetryableError
        return resp.json()

    try:
        openai_resp = await _call_upstream()
    except (httpx.HTTPStatusError, UpstreamRetryableError) as e:
        # 4xx 业务错误 或 重试耗尽后的 5xx → 归一化返回
        raise handle_upstream_error(e)
    except (httpx.RequestError,) as e:
        raise handle_upstream_error(e)
    except Exception as e:
        logger.exception(f"Internal error in generateContent: {e}")
        raise HTTPException(
            status_code=500,
            detail=build_gemini_error(500, f"Internal proxy error: {e}"),
        )

    # 3. 转换响应
    gemini_resp = response_transformer.transform(openai_resp)
    return JSONResponse(content=gemini_resp)


# ====================================================================
# 2. 流式 streamGenerateContent（不重试）
# ====================================================================

@router.post(
    "/v1beta/models/{model:path}:streamGenerateContent",
    dependencies=[Depends(verify_api_key)],
)
async def stream_generate_content(
    request: Request,
    model: str = Path(..., description="Gemini 模型名（仅用于日志，不影响出站）"),
) -> StreamingResponse:
    """流式生成。不重试（流式重试会让客户端拿到重复 chunk）。"""
    body = await request.json()
    inbound_model = model

    # 1. 转换请求体
    openai_body = request_transformer.transform(body, stream=True, inbound_model=inbound_model)
    logger.info(f"streamGenerateContent: outbound request to OpenAI model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')}")
    logger.debug(f"streamGenerateContent: outbound request body={openai_body}")

    upstream_url = f"{settings.upstream_openai_url.rstrip('/')}/v1/chat/completions"
    upstream_headers = {
        "Authorization": f"Bearer {settings.upstream_api_key}",
        "Content-Type": "application/json",
    }


    async def event_generator():
        client = _get_upstream_client()
        try:
            async with client.stream(
                "POST", upstream_url, json=openai_body, headers=upstream_headers
            ) as resp:
                # 5xx / 429 → 不重试，直接下发 SSE 错误帧
                if resp.status_code >= 400:
                    err_body = await resp.aread()
                    try:
                        err_json = json.loads(err_body)
                        msg = (
                            err_json.get("error", {}).get("message")
                            or err_json.get("message")
                            or str(err_body.decode("utf-8", errors="ignore"))
                        )
                    except Exception:
                        msg = err_body.decode("utf-8", errors="ignore")
                    err_frame = {
                        "error": {
                            "code": resp.status_code,
                            "message": msg,
                            "status": map_http_status_to_gemini_status(resp.status_code),
                        }
                    }
                    yield ("data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n").encode("utf-8")
                    return

                processor = StreamProcessor()
                async for chunk in processor.process(resp.aiter_lines()):
                    yield chunk
        except asyncio.CancelledError:
            logger.info("Client disconnected (streamGenerateContent).")
            raise
        except httpx.RequestError as e:
            logger.error(f"Upstream stream error: {e}")
            err_frame = {
                "error": {
                    "code": 502,
                    "message": f"Upstream stream error: {e}",
                    "status": "UNAVAILABLE",
                }
            }
            yield ("data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n").encode("utf-8")



    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",  # Gemini SDK 期望 SSE 格式
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


# ====================================================================
# 3. countTokens（启发式估算 / 尝试转发）
# ====================================================================

@router.post(
    "/v1beta/models/{model:path}:countTokens",
    dependencies=[Depends(verify_api_key)],
)
async def count_tokens(
    request: Request,
    model: str = Path(...),
) -> JSONResponse:
    """Token 计数。

    v0.1.0 策略：启发式估算（OpenAI 无标准计数端点）。
    字符数 / 3 + 20，与 claude_proxy 同款（参考 docs/requirements/functional_requirements.md §2.4）。
    """
    body = await request.json()

    full_text = ""
    sys = body.get("systemInstruction")
    if isinstance(sys, dict):
        full_text += "".join(
            str(p.get("text", "")) for p in sys.get("parts", []) if isinstance(p, dict)
        )
    for msg in body.get("contents", []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("parts", [])
        for p in content:
            if isinstance(p, dict) and "text" in p:
                full_text += str(p["text"])
    # tools 序列化入估算
    full_text += json.dumps(body.get("tools", []), ensure_ascii=False)
    full_text += json.dumps(body.get("generationConfig", {}), ensure_ascii=False)

    estimated = (len(full_text) // 3) + 20
    return JSONResponse(content={"totalTokens": estimated})


# ====================================================================
# 4. 模型列表 / 单模型详情
# ====================================================================

@router.get(
    "/v1beta/models",
    dependencies=[Depends(verify_api_key)],
)
async def list_models() -> JSONResponse:
    """从上游 /v1/models 拉取并转换为 Gemini `models[]` 格式。"""
    models = await model_discovery.get_gemini_models()
    return JSONResponse(content={"models": models})


@router.get(
    "/v1beta/models/{model:path}",
    dependencies=[Depends(verify_api_key)],
)
async def get_model(model: str = Path(...)) -> JSONResponse:
    """单模型详情。"""
    models = await model_discovery.get_gemini_models()
    target = f"models/{model}" if not model.startswith("models/") else model
    for m in models:
        if m.get("name") == target:
            return JSONResponse(content=m)
    raise HTTPException(
        status_code=404,
        detail=build_gemini_error(404, f"Model {model} not found", status="NOT_FOUND"),
    )
