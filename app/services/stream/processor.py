"""流式响应处理器：OpenAI SSE → Gemini JSON 数组流。

按 docs/design/stream_handler.md 规范实现。

关键设计：
  - 默认输出数组流格式：[\\n{...},\\n{...}\\n]
  - 不重试（流式重试会让客户端拿到重复 chunk）
  - tool_calls 多 chunk 增量 → 在收尾时一次性输出 functionCall
  - 数组流首帧输出 `[`，中间帧以 `,\\n` 分隔，末帧 `]`
"""
from __future__ import annotations

import json
from typing import AsyncGenerator, Dict, Any, Optional

import httpx

from app.core.logging import logger
from app.services.transformer import fields
from app.utils.thinking import parse_thinking_segments, strip_thinking


class StreamProcessor:
    """将 OpenAI SSE 行流转换为 Gemini JSON 数组流。

    使用方式：
        processor = StreamProcessor()
        async for chunk in processor.process(openai_response_aiter_lines):
            yield chunk
    """

    def __init__(self) -> None:
        # OpenAI tool_calls 增量聚合：{index: {"id":..., "name":..., "arguments":...}}
        self._tool_calls_acc: Dict[int, Dict[str, Optional[str]]] = {}
        # 累积的最后一个 usage
        self._last_usage: Optional[Dict[str, Any]] = None
        # 累积的最后一个 finish_reason
        self._last_finish_reason: Optional[str] = None
        # 是否已输出首帧的 '['
        self._bracket_emitted = False
        # 是否已输出过至少一个数据帧（用于决定分隔符）
        self._has_emitted_data = False
        # 是否已发送终止信号（流结束）
        self._closed = False
        # 累积文本 delta，用于清理 <think>...</think>（可能跨 chunk）
        self._text_acc: list[str] = []
        # 累积推理文本 delta（如果是直接返回的 reasoning_content）
        self._reasoning_acc: list[str] = []

    async def process(
        self,
        line_iter,
    ) -> AsyncGenerator[bytes, None]:
        """处理 OpenAI SSE 行流，yield Gemini SSE 帧（bytes）。

        输出格式（@google/genai SDK 期望的 SSE）：
            data: [{...}]\\n\\n
            data: [{...}]\\n\\n
            data: [{...}]\\n\\n

        关键：每个 data: 字段后是一个 **JSON 数组**（即使只有 1 个元素）。
        不能输出原始的 JSON 数组流 "[\\n{...}\\n]\\n"，否则 SDK 会报
        "Incomplete JSON segment at the end"。

        Args:
            line_iter: 来自 httpx response.aiter_lines() 的异步行迭代器。
        """
        try:
            frames_buffer: list[Dict[str, Any]] = []

            async for raw_line in line_iter:
                line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue

                payload = line[6:].strip()
                if not payload:
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as e:
                    logger.warning(f"StreamProcessor: JSON decode failed, skip: {e}")
                    continue

                # 上游流式错误（LiteLLM 偶发）→ 转 error 帧
                if "error" in chunk and "choices" not in chunk:
                    logger.warning(f"StreamProcessor: upstream stream error: {chunk['error']}")
                    frames_buffer.append({
                        "error": {
                            "code": 500,
                            "message": str(chunk["error"].get("message", "Upstream stream error")),
                            "status": "INTERNAL",
                        }
                    })
                    self._stream_error = True
                    break

                frame = self._build_gemini_frame(chunk)
                if frame is not None:
                    frames_buffer.append(frame)

            # 收尾：文本及推理内容解析后输出（保留 <think> 推理内容为 Gemini thought=True）
            full_text = "".join(self._text_acc)
            full_reasoning = "".join(self._reasoning_acc)
            parsed_parts = parse_thinking_segments(full_text, full_reasoning if full_reasoning else None)
            logger.info(f"StreamProcessor: end of stream text_len={len(full_text)} reasoning_len={len(full_reasoning)} parsed_parts_count={len(parsed_parts)}")
            logger.debug(f"StreamProcessor: end of stream parsed_parts={parsed_parts}")
            if parsed_parts:
                frames_buffer.append({
                    "candidates": [{
                        "content": {
                            "role": fields.ROLE_MODEL,
                            "parts": parsed_parts,
                        },
                        "index": 0,
                    }],
                })

            # 收尾：tool_calls 一次性输出
            tool_frame = self._build_tool_call_frame()
            if tool_frame is not None:
                frames_buffer.append(tool_frame)

            # 收尾：finishReason + usageMetadata
            # 错误路径用 OTHER，正常路径用 STOP 作默认值
            default_finish = "OTHER" if getattr(self, "_stream_error", False) else "STOP"
            final_frame = self._build_final_frame(default_finish=default_finish)
            if final_frame is not None:
                frames_buffer.append(final_frame)

            # 统一以 SSE data: 格式输出（每帧一个元素包成数组）
            for frame in frames_buffer:
                yield self._format_sse_frame(frame)

            self._closed = True

        except Exception as e:
            logger.error(f"StreamProcessor: unhandled error: {e}")
            if not self._closed:
                try:
                    err_frame = {
                        "error": {
                            "code": 500,
                            "message": f"Stream processing error: {e}",
                            "status": "INTERNAL",
                        }
                    }
                    yield self._format_sse_frame(err_frame)
                except Exception:
                    pass
            raise

    # ====================================================================
    # 内部：帧构造
    # ====================================================================

    def _build_gemini_frame(self, openai_chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从单个 OpenAI chunk 构造 Gemini 帧（仅 text delta / usage 累积 / finish_reason 记录）。"""
        # ---- usage 可能出现在 choices 为空但带 usage 字段的尾部 chunk ----
        if openai_chunk.get("usage"):
            self._last_usage = openai_chunk["usage"]

        choices = openai_chunk.get("choices")
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})

        # ---- 记录 finish_reason ----
        if choice.get("finish_reason"):
            self._last_finish_reason = choice["finish_reason"]

        # ---- 累积 tool_calls（不在本帧输出） ----
        for tc_delta in delta.get("tool_calls", []) or []:
            idx = tc_delta.get("index", 0)
            slot = self._tool_calls_acc.setdefault(idx, {
                "id": None, "name": "", "arguments": "",
            })
            if tc_delta.get("id"):
                slot["id"] = tc_delta["id"]
            fn = tc_delta.get("function", {}) or {}
            if fn.get("name"):
                slot["name"] = (slot["name"] or "") + fn["name"]
            if fn.get("arguments"):
                slot["arguments"] = (slot["arguments"] or "") + fn["arguments"]

        # ---- 文本增量 ----
        text = delta.get("content")
        if text:
            # 注意：MiniMax/M3 等模型会输出 <think>...</think>，且标签可能跨 chunk。
            # 为避免泄漏思考链，流式路径先累积文本，收尾时统一解析后输出。
            self._text_acc.append(text)

        reasoning = delta.get("reasoning_content")
        if reasoning:
            self._reasoning_acc.append(reasoning)
        return None

    def _build_tool_call_frame(self) -> Optional[Dict[str, Any]]:
        """收尾时一次性输出累积的 tool_calls。"""
        if not self._tool_calls_acc:
            return None

        parts: list[Dict[str, Any]] = []
        for idx in sorted(self._tool_calls_acc.keys()):
            slot = self._tool_calls_acc[idx]
            name = slot.get("name") or ""
            try:
                args = json.loads(slot.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = slot.get("arguments") or {}
            parts.append({
                "functionCall": {
                    "id": slot.get("id") or f"call_{idx}",
                    "name": name,
                    "args": args,
                }
            })

        return {
            "candidates": [{
                "content": {
                    "role": fields.ROLE_MODEL,
                    "parts": parts,
                },
                "index": 0,
            }],
        }

    def _build_final_frame(self, default_finish: str = "OTHER") -> Optional[Dict[str, Any]]:
        """收尾：finishReason + usageMetadata。

        Args:
            default_finish: 当上游未发送 finish_reason 时的默认值。
                            路由层传 "STOP"，错误路径传 "OTHER"。
        """
        candidate: Dict[str, Any] = {
            "content": {"role": fields.ROLE_MODEL, "parts": []},
            "index": 0,
        }
        frame: Dict[str, Any] = {"candidates": [candidate]}

        if self._last_finish_reason is not None:
            candidate["finishReason"] = fields.map_finish_reason(
                self._last_finish_reason,
                has_tool_calls=bool(self._tool_calls_acc),
            )
        else:
            # 默认值：保证 Gemini 客户端能识别流已结束
            candidate["finishReason"] = default_finish

        logger.info(f"StreamProcessor: stream complete last_finish_reason={self._last_finish_reason} default_finish={default_finish}")
        logger.debug(f"StreamProcessor: final_frame={frame}")
        if self._last_usage:
            frame["usageMetadata"] = {
                "promptTokenCount": self._last_usage.get("prompt_tokens", 0),
                "candidatesTokenCount": self._last_usage.get("completion_tokens", 0),
                "totalTokenCount": self._last_usage.get("total_tokens", 0),
            }

        return frame

    def _format_sse_frame(self, frame: Dict[str, Any]) -> bytes:
        """将 Gemini 帧格式化为标准的 SSE `data:` 帧。

        格式：data: {"candidates": [...]}\\n\\n
        - 必须是 JSON 对象（@google/genai SDK 期望，不能用数组包）
        - 末尾必须有 \\n\\n（SSE 事件分隔符）
        """
        body = json.dumps(frame, ensure_ascii=False).encode("utf-8")
        self._has_emitted_data = True
        return b"data: " + body + b"\n\n"
