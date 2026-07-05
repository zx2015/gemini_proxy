"""<think> 标签解析与清洗回归测试。"""
from __future__ import annotations

import json
from typing import AsyncGenerator, List

import pytest

from app.services.stream.processor import StreamProcessor
from app.services.transformer.from_openai import response_transformer
from app.utils.thinking import parse_thinking_segments, strip_thinking


async def _sse_lines(chunks: List[dict]) -> AsyncGenerator[str, None]:
    for c in chunks:
        yield f"data: {json.dumps(c)}\n\n"
    yield "data: [DONE]\n\n"


def _parse_sse_body(body: bytes) -> List[dict]:
    frames: List[dict] = []
    for event in body.decode("utf-8").split("\n\n"):
        event = event.strip()
        if not event:
            continue
        if event.startswith("data: "):
            frames.append(json.loads(event[6:].strip()))
    return frames


class TestThinkingStripper:
    def test_strip_full_think_block(self):
        assert strip_thinking("<think>内部推理</think>正式回答") == "正式回答"

    def test_strip_multiline_think_block(self):
        text = "<think>line1\nline2</think>\n\n答案"
        assert strip_thinking(text) == "答案"

    def test_strip_unclosed_think_block(self):
        # 标签不在开头，作为普通文本保留
        assert strip_thinking("前缀<think>未闭合推理") == "前缀<think>未闭合推理"

    def test_strip_stray_close_tag(self):
        # 标签不在开头，作为普通文本保留
        assert strip_thinking("残留推理</think>正式回答") == "残留推理</think>正式回答"

    def test_keep_plain_text(self):
        assert strip_thinking("正常回答") == "正常回答"

    def test_parse_thinking_segments_basic(self):
        segments = parse_thinking_segments("<think>内部推理</think>\n\n正式回答")
        assert segments == [
            {"thought": True, "text": "内部推理"},
            {"text": "正式回答"},
        ]

    def test_parse_thinking_segments_direct_reasoning(self):
        segments = parse_thinking_segments("最终答案", "思考过程")
        assert segments == [
            {"thought": True, "text": "思考过程"},
            {"text": "最终答案"},
        ]


class TestNonStreamThinkingCleanup:
    def test_non_stream_preserves_think_block(self):
        openai_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<think>我先想一下</think>\n\n最终答案",
                },
                "finish_reason": "stop",
            }]
        }
        out = response_transformer.transform(openai_resp)
        parts = out["candidates"][0]["content"]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"thought": True, "text": "我先想一下"}
        assert parts[1] == {"text": "最终答案"}


class TestStreamThinkingCleanup:
    @pytest.mark.asyncio
    async def test_stream_preserves_think_across_chunks(self):
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "<think>我"}}]},
            {"choices": [{"index": 0, "delta": {"content": "正在推"}}]},
            {"choices": [{"index": 0, "delta": {"content": "理</think>\n\n"}}]},
            {"choices": [{"index": 0, "delta": {"content": "最终答案"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        proc = StreamProcessor()
        out: List[bytes] = []
        async for b in proc.process(_sse_lines(chunks)):
            out.append(b)
        frames = _parse_sse_body(b"".join(out))
        
        parts = []
        for f in frames:
            for c in f.get("candidates", []):
                for p in c.get("content", {}).get("parts", []):
                    parts.append(p)
                    
        assert len(parts) == 2
        assert parts[0] == {"thought": True, "text": "我正在推理"}
        assert parts[1] == {"text": "最终答案"}

    @pytest.mark.asyncio
    async def test_stream_supports_direct_reasoning_content(self):
        chunks = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "思考1"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "思考2"}}]},
            {"choices": [{"index": 0, "delta": {"content": "最终"}}]},
            {"choices": [{"index": 0, "delta": {"content": "结果"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        proc = StreamProcessor()
        out: List[bytes] = []
        async for b in proc.process(_sse_lines(chunks)):
            out.append(b)
        frames = _parse_sse_body(b"".join(out))

        all_parts = [
            p
            for f in frames
            for c in f.get("candidates", [])
            for p in c.get("content", {}).get("parts", [])
        ]
        thought_parts = [p for p in all_parts if p.get("thought")]
        text_parts = [p for p in all_parts if not p.get("thought") and "text" in p]

        # reasoning 作为 thought 帧在 content 之前输出
        assert thought_parts == [{"thought": True, "text": "思考1思考2"}]
        assert "".join(p["text"] for p in text_parts) == "最终结果"

        # 验证 thought 帧先于第一个 text 帧出现
        first_thought_idx = next(
            i for i, f in enumerate(frames)
            if any(p.get("thought") for c in f.get("candidates", []) for p in c.get("content", {}).get("parts", []))
        )
        first_text_idx = next(
            i for i, f in enumerate(frames)
            if any(not p.get("thought") and "text" in p for c in f.get("candidates", []) for p in c.get("content", {}).get("parts", []))
        )
        assert first_thought_idx < first_text_idx, "thought 帧应早于 text 帧出现"

    @pytest.mark.asyncio
    async def test_stream_interleaved_reasoning_and_content(self):
        # 极个别模型会交替推送 reasoning_content 和 content。
        # 每段推理都应排在其后续正文之前，而非全部堆到收尾。
        chunks = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "思考A"}}]},
            {"choices": [{"index": 0, "delta": {"content": "答案A"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "思考B"}}]},
            {"choices": [{"index": 0, "delta": {"content": "答案B"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        proc = StreamProcessor()
        out: List[bytes] = []
        async for b in proc.process(_sse_lines(chunks)):
            out.append(b)
        frames = _parse_sse_body(b"".join(out))

        ordered = [
            ("thought" if p.get("thought") else "text", p.get("text", ""))
            for f in frames
            for c in f.get("candidates", [])
            for p in c.get("content", {}).get("parts", [])
            if "text" in p
        ]
        # 顺序必须是：思考A → 答案A → 思考B → 答案B
        assert ordered == [
            ("thought", "思考A"),
            ("text", "答案A"),
            ("thought", "思考B"),
            ("text", "答案B"),
        ]

    def test_thinking_tag_variants(self):
        # 测试各类变体标签是否能正确被解析
        for tag in ["thinking", "reflection", "reasoning", "antml:thinking"]:
            text = f"<{tag}>内部推理</{tag}>\n\n最终结果"
            segments = parse_thinking_segments(text)
            assert segments == [
                {"thought": True, "text": "内部推理"},
                {"text": "最终结果"},
            ]

    def test_request_transformer_thought_part(self):
        # 测试 Gemini 的 thought=True 属性在请求时是否被还原为 <think> 标签
        from app.services.transformer.to_openai import request_transformer
        gemini = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"text": "我的思考过程", "thought": True},
                        {"text": "最终答案"},
                    ],
                }
            ]
        }
        out = request_transformer.transform(gemini)
        messages = out["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "<think>我的思考过程</think>\n最终答案"

    def test_thinking_tag_case_insensitive(self):
        # 大小写混合匹配测试
        text = "<Think>我的推理</THINK>我的最终回答"
        segments = parse_thinking_segments(text)
        assert segments == [
            {"thought": True, "text": "我的推理"},
            {"text": "我的最终回答"},
        ]

    def test_non_thinking_html_tag_safety(self):
        # 测试带有正常网页标签的普通回答（非推理标签），确保不被剥离
        text = "请参考以下代码：\n<script>console.log('hi');</script>\n谢谢！"
        segments = parse_thinking_segments(text)
        assert len(segments) == 1
        assert segments[0] == {"text": text}

    def test_parse_multiple_think_blocks(self):
        # 文本中间及多个成对思考块应被正确拆分，顺序保留
        text = "开头正文<think>推理一</think>中间正文<think>推理二</think>结尾正文"
        segments = parse_thinking_segments(text)
        assert segments == [
            {"text": "开头正文"},
            {"thought": True, "text": "推理一"},
            {"text": "中间正文"},
            {"thought": True, "text": "推理二"},
            {"text": "结尾正文"},
        ]

    def test_parse_think_block_not_at_start(self):
        # 思考块出现在正文中间（非开头）也应被解析
        text = "先给个结论。<think>其实我在权衡</think>"
        segments = parse_thinking_segments(text)
        assert segments == [
            {"text": "先给个结论。"},
            {"thought": True, "text": "其实我在权衡"},
        ]

    def test_reasoning_content_priority(self):
        # 测试 reasoning_content 和 <think> 同时出现的优先级
        openai_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "真正的推理过程",
                    "content": "<think>冗余推理</think>真正的最终答案",
                },
                "finish_reason": "stop",
            }]
        }
        out = response_transformer.transform(openai_resp)
        parts = out["candidates"][0]["content"]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"thought": True, "text": "真正的推理过程"}
        assert parts[1] == {"text": "<think>冗余推理</think>真正的最终答案"}

    @pytest.mark.asyncio
    async def test_stream_reasoning_and_content_same_chunk(self):
        # 测试流式单个 chunk 同时携带 reasoning_content 和 content 时的解析
        chunks = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "推理段", "content": "回答段"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        proc = StreamProcessor()
        out: List[bytes] = []
        async for b in proc.process(_sse_lines(chunks)):
            out.append(b)
        frames = _parse_sse_body(b"".join(out))

        all_parts = [
            p
            for f in frames
            for c in f.get("candidates", [])
            for p in c.get("content", {}).get("parts", [])
        ]
        thought_parts = [p for p in all_parts if p.get("thought")]
        text_parts = [p for p in all_parts if not p.get("thought") and "text" in p]

        assert thought_parts == [{"thought": True, "text": "推理段"}]
        assert "".join(p["text"] for p in text_parts) == "回答段"

        # 即使在同一 chunk 内，thought 帧也应出现在 text 帧之前
        first_thought_idx = next(
            i for i, f in enumerate(frames)
            if any(p.get("thought") for c in f.get("candidates", []) for p in c.get("content", {}).get("parts", []))
        )
        first_text_idx = next(
            i for i, f in enumerate(frames)
            if any(not p.get("thought") and "text" in p for c in f.get("candidates", []) for p in c.get("content", {}).get("parts", []))
        )
        assert first_thought_idx < first_text_idx


