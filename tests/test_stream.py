"""流式处理器单测（respx mock 上游 OpenAI SSE）。

按 docs/design/stream_handler.md §8 矩阵 + @google/genai SDK 期望。

关键约束（2026-07-02 实测发现）：
  - 输出必须是 SSE 格式：`data: [{...}]\\n\\n`
  - data: 后必须是 **JSON 数组**（即使只有 1 个元素）—— SDK 期望
  - 不能用 `[{...},\\n{...}\\n]` 这种"裸数组流"，否则 SDK 报
    "Incomplete JSON segment at the end"
  - 末尾不要再发 `data: [DONE]`（@google/genai 不需要，由服务端关闭连接触发）
"""
from __future__ import annotations

import json
from typing import AsyncGenerator, List

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.stream.processor import StreamProcessor


client = TestClient(app)


# ============================================================================
# Helper：构造 SSE 行流（模拟 OpenAI 上游）
# ============================================================================

async def _sse_lines(chunks: List[dict]) -> AsyncGenerator[str, None]:
    """将 OpenAI chunk 列表转为 SSE 行迭代器（含 [DONE]）。"""
    for c in chunks:
        yield f"data: {json.dumps(c)}\n\n"
    yield "data: [DONE]\n\n"


def _parse_sse_body(body: bytes) -> List[dict]:
    """解析 SSE 响应体：每行 `data: {...}` 提取为单个 dict。

    输入示例：
        b'data: {"a":1}\\n\\ndata: {"b":2}\\n\\n'
    输出：
        [{"a":1}, {"b":2}]
    """
    frames: List[dict] = []
    text = body.decode("utf-8")
    # 按 SSE 事件分割（每个事件以 \n\n 结束）
    for event in text.split("\n\n"):
        event = event.strip()
        if not event:
            continue
        for line in event.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError as e:
                pytest.fail(f"SSE frame not valid JSON: {payload!r} ({e})")
            # 必须是对象
            assert isinstance(obj, dict), f"data: 后必须是 JSON 对象，实际是 {type(obj)}"
            frames.append(obj)
    return frames


# ============================================================================
# 测试用例
# ============================================================================

@pytest.mark.asyncio
async def test_pure_text_stream():
    """纯文本流：3 个 text delta → 逐帧输出 + finishReason=stop。"""
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": []}}]},
        {"choices": [{"index": 0, "delta": {"content": "你"}}]},
        {"choices": [{"index": 0, "delta": {"content": "好"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    proc = StreamProcessor()
    out: List[bytes] = []
    async for b in proc.process(_sse_lines(chunks)):
        out.append(b)
    frames = _parse_sse_body(b"".join(out))

    # 文本逐帧输出（2 个 text 帧） + 1 个 final 帧
    assert len(frames) >= 2
    # 拼接所有文本 parts 应等于 "你好"
    texts = [
        p["text"]
        for f in frames
        for c in f.get("candidates", [])
        for p in c.get("content", {}).get("parts", [])
        if "text" in p and not p.get("thought")
    ]
    assert "".join(texts) == "你好"
    # final frame 包含 finishReason
    assert frames[-1]["candidates"][0].get("finishReason") == "STOP"


@pytest.mark.asyncio
async def test_stream_with_tool_calls():
    """工具调用：多 chunk 增量聚合 → 收尾一次性输出 functionCall。"""
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_abc", "function": {"name": "get_", "arguments": ""}}
        ]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"name": "weather", "arguments": "{\"loc"}}
        ]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "ation\":\"SF\"}"}}
        ]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    proc = StreamProcessor()
    out: List[bytes] = []
    async for b in proc.process(_sse_lines(chunks)):
        out.append(b)
    frames = _parse_sse_body(b"".join(out))

    # 1 个 tool_call 帧 + 1 个 final 帧
    tool_frames = [f for f in frames if any("functionCall" in p for p in f.get("candidates", [{}])[0].get("content", {}).get("parts", []))]
    assert len(tool_frames) == 1
    fc = tool_frames[0]["candidates"][0]["content"]["parts"][0]["functionCall"]
    assert fc["id"] == "call_abc"
    assert fc["name"] == "get_weather"
    assert fc["args"] == {"location": "SF"}


@pytest.mark.asyncio
async def test_stream_with_usage_tail():
    """usage 末帧（choices 为空）→ usageMetadata 进入 final 帧。"""
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
    ]
    proc = StreamProcessor()
    out: List[bytes] = []
    async for b in proc.process(_sse_lines(chunks)):
        out.append(b)
    frames = _parse_sse_body(b"".join(out))

    final = frames[-1]
    assert final.get("usageMetadata", {}).get("totalTokenCount") == 12


@pytest.mark.asyncio
async def test_stream_with_no_chunks():
    """上游零 chunk（极端情况）→ 仍输出 final 帧。"""
    chunks = []
    proc = StreamProcessor()
    out: List[bytes] = []
    async for b in proc.process(_sse_lines(chunks)):
        out.append(b)
    frames = _parse_sse_body(b"".join(out))

    # 仅 1 个 final 帧（默认 finishReason=STOP）
    assert len(frames) == 1
    assert frames[0]["candidates"][0]["content"]["role"] == "model"
    assert frames[0]["candidates"][0]["content"]["parts"] == []
    assert frames[0]["candidates"][0].get("finishReason") == "STOP"


@pytest.mark.asyncio
async def test_stream_with_upstream_error_frame():
    """上游流式错误（LiteLLM 偶发）→ 转 Gemini error 帧。"""
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "partial"}}]},
        {"error": {"message": "Timeout on reading data from socket", "type": "None", "code": "500"}},
    ]
    proc = StreamProcessor()
    out: List[bytes] = []
    async for b in proc.process(_sse_lines(chunks)):
        out.append(b)
    frames = _parse_sse_body(b"".join(out))

    err_frames = [f for f in frames if "error" in f]
    assert len(err_frames) == 1
    assert "Timeout" in err_frames[0]["error"]["message"]


@pytest.mark.asyncio
async def test_output_is_sse_format():
    """关键约束：每行必须是 `data: {...}\\n\\n` 格式。

    验证：
      1. 整体以 `\\n\\n` 结束事件
      2. 每个 data: 后跟 JSON 对象（不能用数组包）
      3. 多个事件间用 `\\n\\n` 分隔
    """
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "a"}}]},
        {"choices": [{"index": 0, "delta": {"content": "b"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    proc = StreamProcessor()
    out: List[bytes] = []
    async for b in proc.process(_sse_lines(chunks)):
        out.append(b)
    body = b"".join(out).decode("utf-8")

    # 1. 每个事件必须以 \\n\\n 结束
    events = [e for e in body.split("\n\n") if e.strip()]
    # 增量输出：2 个 text 帧（"a", "b"）+ 1 个 final 帧 = 3 个事件
    assert len(events) >= 2

    # 2. 每个事件必须以 `data: {` 开头（对象，不是数组）
    for ev in events:
        assert ev.strip().startswith("data: {"), f"Event must start with 'data: {{', got: {ev!r}"
        payload = ev.strip()[6:]  # 去掉 "data: "
        obj = json.loads(payload)
        assert isinstance(obj, dict)


@pytest.mark.asyncio
async def test_first_chunk_with_empty_tool_calls_does_not_break():
    """首 chunk 含 `tool_calls: []`（LiteLLM/部分 SDK 行为）→ 累积逻辑不应误入。"""
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": []}}]},
        {"choices": [{"index": 0, "delta": {"content": "ok"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    proc = StreamProcessor()
    out: List[bytes] = []
    async for b in proc.process(_sse_lines(chunks)):
        out.append(b)
    frames = _parse_sse_body(b"".join(out))

    for f in frames:
        parts = f.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for p in parts:
            assert "functionCall" not in p


# ============================================================================
# E2E 集成（用 respx 模拟上游）
# ============================================================================

@respx.mock
def test_e2e_stream_full_flow():
    """路由层 + respx 模拟上游 OpenAI 流式端到端。"""
    sse_chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
        {"choices": [{"index": 0, "delta": {"content": " world"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
    ]
    sse_body = "".join(f"data: {json.dumps(c)}\n\n" for c in sse_chunks) + "data: [DONE]\n\n"

    respx.post(f"{settings.upstream_openai_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode("utf-8"),
        )
    )

    with client.stream(
        "POST",
        "/v1beta/models/gemini-2.5-pro:streamGenerateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        headers={"x-goog-api-key": settings.proxy_api_key},
    ) as resp:
        assert resp.status_code == 200
        # Content-Type 必须是 text/event-stream
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = b"".join(resp.iter_bytes())

    frames = _parse_sse_body(body)
    # 合并后：1 个 text 帧 + 1 个 final 帧
    assert len(frames) >= 2

    # 拼接所有 text 应为 "Hello world"
    texts = []
    for f in frames:
        for c in f.get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "text" in p:
                    texts.append(p["text"])
    assert "".join(texts) == "Hello world"

    # 最后一帧含 finishReason=STOP + usageMetadata
    last = frames[-1]
    assert last["candidates"][0].get("finishReason") == "STOP"
    assert last.get("usageMetadata", {}).get("totalTokenCount") == 7
