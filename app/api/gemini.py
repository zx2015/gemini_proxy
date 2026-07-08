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
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
import re
import uuid
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.auth import verify_api_key
from app.core.circuit_breaker import CircuitBreakerOpen, get_circuit_breaker
from app.core.config import settings
from app.core.logging import logger, logger_debug
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


def _mask_query_key(url_str: str) -> str:
    """掩码隐藏 URL 中的敏感 ?key=..."""
    return re.sub(r'key=[^&]+', 'key=***', url_str)


def _truncate_log_data(data: Any, is_history: bool = False) -> Any:
    """递归遍历请求/响应结构，对普通消息进行智能裁剪（历史项完全省略），保障工具调用指令完整，但截断工具超长返回结果。"""
    if isinstance(data, dict):
        # 1. 模型返回的工具调用指令 (functionCall / tool_calls) 绝对保护不裁剪
        if "functionCall" in data or "tool_calls" in data:
            return data

        # 2. 专门处理 Gemini contents 数组和 OpenAI messages 数组，识别历史项并传递 is_history
        if "contents" in data and isinstance(data["contents"], list):
            copied = data.copy()
            contents = data["contents"]
            copied["contents"] = [
                _truncate_log_data(item, is_history=(idx < len(contents) - 1))
                for idx, item in enumerate(contents)
            ]
            for k, v in copied.items():
                if k != "contents":
                    copied[k] = _truncate_log_data(v, is_history)
            return copied

        if "messages" in data and isinstance(data["messages"], list):
            copied = data.copy()
            messages = data["messages"]
            copied["messages"] = [
                _truncate_log_data(item, is_history=(idx < len(messages) - 1))
                for idx, item in enumerate(messages)
            ]
            for k, v in copied.items():
                if k != "messages":
                    copied[k] = _truncate_log_data(v, is_history)
            return copied

        # 3. 多模态 Base64 原始数据直接大幅度裁剪
        if "inline_data" in data:
            inline = data["inline_data"]
            if isinstance(inline, dict) and "data" in inline:
                copied = inline.copy()
                d_str = str(copied["data"])
                if len(d_str) > 50:
                    copied["data"] = f"{d_str[:20]}... [BASE64_DATA_TRUNCATED_{len(d_str)}B] ...{d_str[-20:]}"
                return {"inline_data": copied}

        # 4. 处理 Gemini 格式的工具返回结果 (functionResponse) — 缩紧至进行 300 字符截断
        if "functionResponse" in data:
            fr = data["functionResponse"]
            if isinstance(fr, dict) and "response" in fr:
                copied_fr = fr.copy()
                resp_val = copied_fr["response"]
                resp_str = json.dumps(resp_val, ensure_ascii=False) if isinstance(resp_val, (dict, list)) else str(resp_val)
                if len(resp_str) > 300:
                    # 中段裁剪
                    truncated_str = f"{resp_str[:150]} ... [TOOL_RESP_TRUNCATED {len(resp_str) - 300} CHARS] ... {resp_str[-150:]}"
                    copied_fr["response"] = truncated_str
                return {"functionResponse": copied_fr}

        # 5. 处理 OpenAI 格式的 role="tool" 消息 — 缩紧至进行 300 字符截断
        if data.get("role") == "tool" and "content" in data:
            copied_tool = data.copy()
            content_val = str(copied_tool["content"])
            if len(content_val) > 300:
                copied_tool["content"] = f"{content_val[:150]} ... [TOOL_RESP_TRUNCATED {len(content_val) - 300} CHARS] ... {content_val[-150:]}"
            return copied_tool

        # 6. 递归处理其它常规字段
        return {k: _truncate_log_data(v, is_history) for k, v in data.items()}

    elif isinstance(data, list):
        return [_truncate_log_data(item, is_history) for item in data]

    elif isinstance(data, str):
        if is_history:
            # 历史普通文本消息：完全省略以防 O(N^2) 日志膨胀
            if len(data) > 50:
                return f"[OMITTED_HISTORICAL_TEXT_LEN_{len(data)}]"
            return data
        else:
            # 最新普通消息：限制 400 字符裁剪
            if len(data) > 400:
                return f"{data[:150]} ... [TEXT_TRUNCATED {len(data) - 300} CHARS] ... {data[-150:]}"
            return data

    return data


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
    request_id = uuid.uuid4().hex[:8]
    start_time = time.monotonic()
    body = await request.json()
    inbound_model = model

    if settings.debug_log_enabled:
        masked_url = _mask_query_key(str(request.url))
        inbound_max_tokens = (body.get("generationConfig") or {}).get("maxOutputTokens")
        logger_debug.info(
            f"[{request_id}] [INBOUND_REQUEST] URL: {masked_url} "
            f"inbound_model={inbound_model!r} maxOutputTokens={inbound_max_tokens}\n"
            f"Body: {json.dumps(_truncate_log_data(body), ensure_ascii=False)}"
        )

    # 1. 转换请求体
    openai_body = request_transformer.transform(body, stream=False, inbound_model=inbound_model)

    if settings.debug_log_enabled:
        out_max_tokens = openai_body.get("max_tokens")
        out_model = openai_body.get("model")
        logger_debug.info(
            f"[{request_id}] [OUTBOUND_REQUEST] URL: {settings.upstream_openai_url.rstrip('/')}/v1/chat/completions "
            f"model={out_model!r} max_tokens={out_max_tokens}\n"
            f"Body: {json.dumps(_truncate_log_data(openai_body), ensure_ascii=False)}"
        )

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

    cb = get_circuit_breaker()
    try:
        await cb.before_call()
    except CircuitBreakerOpen as e:
        logger.warning(f"[{request_id}] Circuit breaker rejected request: {e}")
        raise HTTPException(
            status_code=503,
            detail=build_gemini_error(503, f"Service temporarily unavailable: {e}"),
        )

    try:
        openai_resp = await _call_upstream()
        await cb.record_success()
        if settings.debug_log_enabled:
            logger_debug.info(
                f"[{request_id}] [UPSTREAM_RESPONSE] Body: {json.dumps(_truncate_log_data(openai_resp), ensure_ascii=False)}"
            )
    except (httpx.HTTPStatusError, UpstreamRetryableError) as e:
        await cb.record_failure()
        if settings.debug_log_enabled:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger_debug.info(
                f"[{request_id}] [REQUEST_SUMMARY] FAILED retryable/status error "
                f"model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')} "
                f"duration_ms={duration_ms} error={e}"
            )
        # 4xx 业务错误 或 重试耗尽后的 5xx → 归一化返回
        raise handle_upstream_error(e)
    except (httpx.RequestError,) as e:
        await cb.record_failure()
        if settings.debug_log_enabled:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger_debug.info(
                f"[{request_id}] [REQUEST_SUMMARY] FAILED request error "
                f"model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')} "
                f"duration_ms={duration_ms} error={e}"
            )
        raise handle_upstream_error(e)
    except Exception as e:
        if settings.debug_log_enabled:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger_debug.info(
                f"[{request_id}] [REQUEST_SUMMARY] FAILED unexpected error "
                f"model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')} "
                f"duration_ms={duration_ms} error={e}"
            )
        logger.exception(f"Internal error in generateContent: {e}")
        raise HTTPException(
            status_code=500,
            detail=build_gemini_error(500, f"Internal proxy error: {e}"),
        )

    # 3. 转换响应
    gemini_resp = response_transformer.transform(openai_resp)
    if settings.debug_log_enabled:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        usage = openai_resp.get("usage") or {}
        finish_reason = ((openai_resp.get("choices") or [{}])[0]).get("finish_reason")
        logger_debug.info(
            f"[{request_id}] [REQUEST_SUMMARY] OK "
            f"model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')} "
            f"finish_reason={finish_reason!r} "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"total_tokens={usage.get('total_tokens')} "
            f"duration_ms={duration_ms}"
        )
        logger_debug.info(
            f"[{request_id}] [OUTBOUND_RESPONSE] Body: {json.dumps(_truncate_log_data(gemini_resp), ensure_ascii=False)}"
        )
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
    request_id = uuid.uuid4().hex[:8]
    start_time = time.monotonic()
    body = await request.json()
    inbound_model = model

    if settings.debug_log_enabled:
        masked_url = _mask_query_key(str(request.url))
        inbound_max_tokens = (body.get("generationConfig") or {}).get("maxOutputTokens")
        logger_debug.info(
            f"[{request_id}] [INBOUND_REQUEST] URL: {masked_url} "
            f"inbound_model={inbound_model!r} maxOutputTokens={inbound_max_tokens}\n"
            f"Body: {json.dumps(_truncate_log_data(body), ensure_ascii=False)}"
        )

    # 1. 转换请求体
    openai_body = request_transformer.transform(body, stream=True, inbound_model=inbound_model)
    logger.info(f"streamGenerateContent: outbound request to OpenAI model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')}")

    if settings.debug_log_enabled:
        out_max_tokens = openai_body.get("max_tokens")
        out_model = openai_body.get("model")
        logger_debug.info(
            f"[{request_id}] [OUTBOUND_REQUEST] URL: {settings.upstream_openai_url.rstrip('/')}/v1/chat/completions "
            f"model={out_model!r} max_tokens={out_max_tokens}\n"
            f"Body: {json.dumps(_truncate_log_data(openai_body), ensure_ascii=False)}"
        )

    upstream_url = f"{settings.upstream_openai_url.rstrip('/')}/v1/chat/completions"
    upstream_headers = {
        "Authorization": f"Bearer {settings.upstream_api_key}",
        "Content-Type": "application/json",
    }

    # 流式超时配置（参考 cc-switch 的三段式超时）
    _first_byte_timeout = settings.stream_first_byte_timeout or None
    _idle_timeout = settings.stream_idle_timeout or None

    async def log_aiter_lines(lines_iterator):
        """转发 SSE 行并记录 compact 调试日志（每块仅提取 delta/finish_reason，节省存储）。"""
        chunk_count = 0
        async for line in lines_iterator:
            if settings.debug_log_enabled:
                line_str = line.decode('utf-8', errors='ignore') if isinstance(line, bytes) else str(line)
                if line_str.startswith("data: "):
                    data_part = line_str[6:].strip()
                    if data_part == "[DONE]":
                        logger_debug.info(f"[{request_id}] [UPSTREAM_SSE] #{chunk_count} [DONE]")
                    else:
                        try:
                            data_json = json.loads(data_part)
                            chunk_count += 1
                            choice = (data_json.get("choices") or [{}])[0]
                            delta = choice.get("delta") or {}
                            finish_reason = choice.get("finish_reason")
                            usage = data_json.get("usage")
                            parts = [f"#{chunk_count}"]
                            if delta.get("content") is not None:
                                txt = delta["content"]
                                parts.append(f"content={txt[:60]!r}" if len(txt) > 60 else f"content={txt!r}")
                            if delta.get("reasoning_content"):
                                parts.append(f"reasoning_len={len(delta['reasoning_content'])}")
                            if delta.get("tool_calls"):
                                parts.append(f"tool_calls_delta={json.dumps(delta['tool_calls'], ensure_ascii=False)[:120]}")
                            if finish_reason:
                                parts.append(f"finish_reason={finish_reason!r}")
                            if usage:
                                parts.append(f"usage={usage}")
                            logger_debug.info(f"[{request_id}] [UPSTREAM_SSE] {' '.join(parts)}")
                        except Exception:
                            logger_debug.info(f"[{request_id}] [UPSTREAM_SSE] raw: {line_str[:200]}")
                else:
                    if line_str.strip():
                        logger_debug.info(f"[{request_id}] [UPSTREAM_SSE] non-data: {line_str[:200]}")
            yield line

    async def _aiter_with_timeout(raw_aiter):
        """为 aiter_lines 添加首字节超时和 idle 超时，超时时 yield 一个错误帧并终止。"""
        is_first = True
        it = raw_aiter.__aiter__()
        while True:
            timeout_secs = _first_byte_timeout if is_first else _idle_timeout
            try:
                if timeout_secs:
                    line = await asyncio.wait_for(it.__anext__(), timeout=timeout_secs)
                else:
                    line = await it.__anext__()
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                timeout_type = "首字节" if is_first else "idle"
                duration_ms = int((time.monotonic() - start_time) * 1000)
                logger.error(
                    f"[{request_id}] 流式响应{timeout_type}超时 ({timeout_secs}s) "
                    f"model={openai_body.get('model')!r} duration_ms={duration_ms}"
                )
                if settings.debug_log_enabled:
                    logger_debug.info(
                        f"[{request_id}] [REQUEST_SUMMARY] TIMEOUT ({timeout_type}) "
                        f"model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')} "
                        f"timeout_secs={timeout_secs} duration_ms={duration_ms}"
                    )
                # 向 Gemini CLI 下发超时错误帧
                err_frame = {
                    "error": {
                        "code": 504,
                        "message": f"Upstream stream {timeout_type} timeout after {timeout_secs}s",
                        "status": "DEADLINE_EXCEEDED",
                    }
                }
                yield ("data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n").encode("utf-8")
                return
            is_first = False
            yield line

    async def event_generator():
        cb = get_circuit_breaker()
        try:
            await cb.before_call()
        except CircuitBreakerOpen as _cb_err:
            logger.warning(f"[{request_id}] Circuit breaker rejected stream request: {_cb_err}")
            err_frame = {
                "error": {
                    "code": 503,
                    "message": f"Service temporarily unavailable: {_cb_err}",
                    "status": "UNAVAILABLE",
                }
            }
            yield ("data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n").encode("utf-8")
            return

        client = _get_upstream_client()
        chunk_out = 0
        try:
            async with client.stream(
                "POST", upstream_url, json=openai_body, headers=upstream_headers
            ) as resp:
                # 5xx / 429 → 不重试，直接下发 SSE 错误帧
                if resp.status_code >= 400:
                    err_body = await resp.aread()
                    if settings.debug_log_enabled:
                        duration_ms = int((time.monotonic() - start_time) * 1000)
                        logger_debug.info(
                            f"[{request_id}] [REQUEST_SUMMARY] FAILED HTTP {resp.status_code} "
                            f"model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')} "
                            f"duration_ms={duration_ms} body={err_body.decode('utf-8', errors='ignore')[:500]}"
                        )
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
                    out_chunk = ("data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n").encode("utf-8")
                    if settings.debug_log_enabled:
                        logger_debug.info(f"[{request_id}] [OUTBOUND_RESPONSE] error chunk: {out_chunk.decode('utf-8', errors='ignore')[:300]}")
                    yield out_chunk
                    return

                processor = StreamProcessor()
                async for chunk in processor.process(log_aiter_lines(_aiter_with_timeout(resp.aiter_lines()))):
                    chunk_out += 1
                    if settings.debug_log_enabled:
                        chunk_str = chunk.decode('utf-8', errors='ignore') if isinstance(chunk, bytes) else str(chunk)
                        data_part = chunk_str[6:].strip() if chunk_str.startswith("data: ") else chunk_str
                        try:
                            data_json = json.loads(data_part)
                            candidate = (data_json.get("candidates") or [{}])[0]
                            parts_list = (candidate.get("content") or {}).get("parts") or []
                            finish = candidate.get("finishReason")
                            usage = data_json.get("usageMetadata")
                            summary_parts = [f"out#{chunk_out}"]
                            for p in parts_list:
                                if p.get("thought"):
                                    summary_parts.append(f"thought_len={len(p.get('text',''))}")
                                elif "text" in p:
                                    txt = p["text"]
                                    summary_parts.append(f"text={txt[:60]!r}" if len(txt) > 60 else f"text={txt!r}")
                                elif "functionCall" in p:
                                    summary_parts.append(f"functionCall={p['functionCall'].get('name')!r}")
                            if finish:
                                summary_parts.append(f"finishReason={finish!r}")
                            if usage:
                                summary_parts.append(f"usage={usage}")
                            logger_debug.info(f"[{request_id}] [OUTBOUND_SSE] {' '.join(summary_parts)}")
                        except Exception:
                            logger_debug.info(f"[{request_id}] [OUTBOUND_SSE] raw: {chunk_str[:200]}")
                    yield chunk

                await cb.record_success()
                if settings.debug_log_enabled:
                    duration_ms = int((time.monotonic() - start_time) * 1000)
                    logger_debug.info(
                        f"[{request_id}] [REQUEST_SUMMARY] OK (stream complete) "
                        f"model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')} "
                        f"usage={processor._last_usage} finish_reason={processor._last_finish_reason!r} "
                        f"outbound_chunks={chunk_out} duration_ms={duration_ms}"
                    )

        except asyncio.CancelledError:
            logger.info("Client disconnected (streamGenerateContent).")
            if settings.debug_log_enabled:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                logger_debug.info(
                    f"[{request_id}] [REQUEST_SUMMARY] CANCELLED (client disconnected) "
                    f"model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')} "
                    f"outbound_chunks={chunk_out} duration_ms={duration_ms}"
                )
            raise
        except httpx.RequestError as e:
            await cb.record_failure()
            logger.error(f"Upstream stream error: {e}")
            if settings.debug_log_enabled:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                logger_debug.info(
                    f"[{request_id}] [REQUEST_SUMMARY] FAILED request error "
                    f"model={openai_body.get('model')!r} max_tokens={openai_body.get('max_tokens')} "
                    f"duration_ms={duration_ms} error={e}"
                )
            err_frame = {
                "error": {
                    "code": 502,
                    "message": f"Upstream stream error: {e}",
                    "status": "UNAVAILABLE",
                }
            }
            out_chunk = ("data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n").encode("utf-8")
            if settings.debug_log_enabled:
                logger_debug.info(f"[{request_id}] [OUTBOUND_RESPONSE] request error chunk: {out_chunk.decode('utf-8', errors='ignore')[:300]}")
            yield out_chunk

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
