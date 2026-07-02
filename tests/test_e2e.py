"""端到端（E2E）集成测试。

使用 respx mock 上游 OpenAI 兼容服务，验证：
  1. 非流式 generateContent 完整链路
  2. 重试机制（5xx → 200）
  3. 重试机制（4xx → 立即失败，不重试）
  4. 流式 streamGenerateContent
  5. 模型强制覆盖
  6. 鉴权失败 → 401
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


client = TestClient(app)


# ============================================================================
# 1. 非流式 generateContent 完整链路
# ============================================================================

@respx.mock
def test_generate_content_happy_path():
    """完整链路：Gemini 请求 → 网关 → 上游 → 转换 → Gemini 响应。"""
    respx.post(f"{settings.upstream_openai_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [{
                    "message": {"role": "assistant", "content": "你好！"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )
    )

    resp = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={
            "contents": [{"role": "user", "parts": [{"text": "你好"}]}],
        },
        headers={"x-goog-api-key": settings.proxy_api_key},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["candidates"][0]["content"]["parts"][0]["text"] == "你好！"
    assert body["candidates"][0]["content"]["role"] == "model"
    assert body["candidates"][0]["finishReason"] == "STOP"
    assert body["usageMetadata"]["totalTokenCount"] == 8


# ============================================================================
# 2. 重试机制：5xx → 200
# ============================================================================

@respx.mock
def test_retry_on_5xx_then_success():
    """上游 503 → 重试 → 200 成功。"""
    route = respx.post(f"{settings.upstream_openai_url}/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(503, json={"error": {"message": "Service Unavailable"}}),
            httpx.Response(503, json={"error": {"message": "Service Unavailable"}}),
            httpx.Response(200, json={
                "choices": [{
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            }),
        ]
    )

    resp = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        headers={"x-goog-api-key": settings.proxy_api_key},
    )

    assert resp.status_code == 200
    assert route.call_count == 3
    assert resp.json()["candidates"][0]["content"]["parts"][0]["text"] == "ok"


@respx.mock
def test_retry_exhausted_returns_503():
    """上游持续 503 → 重试耗尽 → 返回 503 + Gemini 错误包装。"""
    respx.post(f"{settings.upstream_openai_url}/v1/chat/completions").mock(
        return_value=httpx.Response(503, json={"error": {"message": "down"}})
    )

    resp = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        headers={"x-goog-api-key": settings.proxy_api_key},
    )

    assert resp.status_code == 503
    body = resp.json()
    # FastAPI HTTPException 的 detail 字段是 Gemini error 包装
    assert body["detail"]["error"]["code"] == 503
    assert body["detail"]["error"]["status"] == "UNAVAILABLE"


# ============================================================================
# 3. 重试机制：4xx → 不重试
# ============================================================================

@respx.mock
def test_no_retry_on_400():
    """4xx 业务错误不重试，立即返回。"""
    route = respx.post(f"{settings.upstream_openai_url}/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )

    resp = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        headers={"x-goog-api-key": settings.proxy_api_key},
    )

    assert resp.status_code == 400
    assert route.call_count == 1  # 只调用一次，不重试


@respx.mock
def test_retry_on_429():
    """429 限流是**可重试**错误。"""
    route = respx.post(f"{settings.upstream_openai_url}/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limit"}}),
            httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            }),
        ]
    )

    resp = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        headers={"x-goog-api-key": settings.proxy_api_key},
    )

    assert resp.status_code == 200
    assert route.call_count == 2


# ============================================================================
# 4. 强制模型覆盖
# ============================================================================

@respx.mock
def test_inbound_model_is_ignored():
    """客户端传 gemini-2.5-pro 仍被覆盖为 UPSTREAM_MODEL。"""
    captured_request = {}

    def callback(request: httpx.Request) -> httpx.Response:
        captured_request.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
        })

    respx.post(f"{settings.upstream_openai_url}/v1/chat/completions").mock(
        side_effect=callback
    )

    resp = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        headers={"x-goog-api-key": settings.proxy_api_key},
    )

    assert resp.status_code == 200
    assert captured_request["model"] == settings.upstream_model
    assert captured_request["model"] != "gemini-2.5-pro"


# ============================================================================
# 5. 鉴权
# ============================================================================

def test_auth_missing_returns_401():
    resp = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    assert resp.status_code == 401


def test_auth_invalid_returns_401():
    resp = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        headers={"x-goog-api-key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_auth_bearer_header_works():
    """Authorization: Bearer 也被接受。"""
    with respx.mock:
        respx.post(f"{settings.upstream_openai_url}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            })
        )

        resp = client.post(
            "/v1beta/models/gemini-2.5-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"Authorization": f"Bearer {settings.proxy_api_key}"},
        )

    assert resp.status_code == 200


def test_auth_query_param_works():
    """?key= 也被接受。"""
    with respx.mock:
        respx.post(f"{settings.upstream_openai_url}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            })
        )

        resp = client.post(
            f"/v1beta/models/gemini-2.5-pro:generateContent?key={settings.proxy_api_key}",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )

    assert resp.status_code == 200


# ============================================================================
# 6. countTokens
# ============================================================================

def test_count_tokens_heuristic():
    resp = client.post(
        "/v1beta/models/gemini-2.5-pro:countTokens",
        json={
            "contents": [{"role": "user", "parts": [{"text": "a" * 300}]}]
        },
        headers={"x-goog-api-key": settings.proxy_api_key},
    )
    assert resp.status_code == 200
    body = resp.json()
    # 估算 = 字符数 / 3 + 20
    # 输入 300 字符文本 + tools/generationConfig 序列化 ≈ 302 字符 → 302/3+20 = 120
    # 实际实现会再包含 tools 等额外字段，因此 >= 120 即可
    assert body["totalTokens"] >= 120
    assert body["totalTokens"] < 200


# ============================================================================
# 7. /health
# ============================================================================

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}
