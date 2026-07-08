"""流式响应处理器：OpenAI SSE → Gemini SSE 帧流（状态机增量输出）。

按 docs/design/stream_handler.md 规范实现。

关键设计（v0.2.0 起）：
  - 文本 delta 在非思考模式下**立即**逐帧输出，不再全量缓冲，保证流式延迟最优。
  - 思考内容检测（inline think tags）：
      detecting  → 探测文本开头是否有 <think> 等标签（仅在流的最开始阶段）
      in_think   → 已进入思考块，缓冲直到闭标签
      streaming  → 普通文本，每个 delta 立即输出
  - reasoning_content（DeepSeek 风格）：
      全量缓冲；当第一个 content delta 到来时先输出 thought 帧再输出文本帧，
      保证推理内容始终出现在回答内容之前。
  - tool_calls 多 chunk 增量 → 在收尾时一次性输出 functionCall
"""
from __future__ import annotations

import json
import re
from typing import AsyncGenerator, Dict, Any, List, Optional, Set

from app.core.config import settings
from app.core.logging import logger
from app.services.transformer import fields
from app.utils.minimax_tool_markup import (
    contains_minimax_markup,
    consume_minimax_markup_chunks,
)


# 支持的思考开标签正则（必须出现在文本开头）
_OPEN_TAGS_RE = re.compile(
    r"^<(think|thinking|reflection|reasoning|antml:thinking)\b[^>]*>",
    re.IGNORECASE,
)

# 探测阶段最大缓冲字符数（超过此长度仍无法匹配则视为普通文本）
_THINK_DETECT_CHARS = 30


class StreamProcessor:
    """将 OpenAI SSE 行流转换为 Gemini SSE 帧流（增量输出）。

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
        # 是否已发送终止信号（流结束）
        self._closed = False
        # 流式错误标志
        self._stream_error = False

        # ---- 思考标签状态机（处理 inline <think>...</think>） ----
        # detecting → streaming  （文本不以 <think> 开头）
        # detecting → in_think   （文本以 <think> 开头）
        # in_think  → streaming  （找到对应的 </think>）
        self._think_state: str = "detecting"
        self._think_tag_name: str = ""   # 当前标签名，用于构造闭标签正则
        self._think_buffer: str = ""     # IN_THINK 阶段的缓冲
        self._detect_buffer: str = ""    # DETECTING 阶段的缓冲

        # ---- reasoning_content（DeepSeek 风格）----
        # 累积尚未输出的 reasoning_content；一旦有正文 content 到来即刷新为 thought 帧。
        # 采用「渐进式刷新」而非一次性锁：支持极个别模型交替推送 reasoning/content 时，
        # 也能保证每段思考都排在其后续正文之前。
        self._reasoning_acc: List[str] = []
        # ---- MiniMax tool_call 文本泄漏恢复（防御性）----
        self._minimax_model_hint: bool = "minimax" in settings.upstream_model.lower()
        self._minimax_markup_buffer: str = ""
        self._recovered_tool_call_indexes: Set[int] = set()
        self._saw_structured_tool_calls: bool = False

    async def process(
        self,
        line_iter,
    ) -> AsyncGenerator[bytes, None]:
        """处理 OpenAI SSE 行流，yield Gemini SSE 帧（bytes）。

        文本 delta 立即输出；思考内容、tool_calls 和 finishReason 在合适时机输出。

        Args:
            line_iter: 来自 httpx response.aiter_lines() 的异步行迭代器。
        """
        try:
            async for raw_line in line_iter:
                line = (
                    raw_line.strip()
                    if isinstance(raw_line, str)
                    else raw_line.decode("utf-8", errors="ignore").strip()
                )
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

                # 上游流式错误（LiteLLM 偶发）→ 立即输出 error 帧并终止
                if "error" in chunk and "choices" not in chunk:
                    logger.warning(f"StreamProcessor: upstream stream error: {chunk['error']}")
                    self._stream_error = True
                    yield self._format_sse_frame({
                        "error": {
                            "code": 500,
                            "message": str(chunk["error"].get("message", "Upstream stream error")),
                            "status": "INTERNAL",
                        }
                    })
                    break

                # ---- usage（可能出现在无 choices 的尾部 chunk）----
                if chunk.get("usage"):
                    self._last_usage = chunk["usage"]

                choices = chunk.get("choices")
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})

                if choice.get("finish_reason"):
                    self._last_finish_reason = choice["finish_reason"]

                # ---- 累积 tool_calls（收尾时一次性输出）----
                for tc_delta in delta.get("tool_calls", []) or []:
                    # 一旦观察到标准结构化 tool_calls，优先信任它并丢弃先前的恢复结果，避免重复。
                    if not self._saw_structured_tool_calls:
                        self._saw_structured_tool_calls = True
                        for ridx in list(self._recovered_tool_call_indexes):
                            self._tool_calls_acc.pop(ridx, None)
                        self._recovered_tool_call_indexes.clear()
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

                # ---- reasoning_content（DeepSeek 风格，全量缓冲）----
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    self._reasoning_acc.append(reasoning)

                # ---- 文本 delta：状态机增量处理 ----
                text = delta.get("content")
                if text:
                    # MiniMax 防御恢复：上游若将 tool_call 泄漏在 content 文本中，
                    # 则提取成结构化调用并从文本中清理。
                    if settings.minimax_tool_markup_recovery and (
                        self._minimax_model_hint
                        or self._minimax_markup_buffer
                        or contains_minimax_markup(text)
                    ):
                        clean_text, recovered_calls, new_buffer = consume_minimax_markup_chunks(
                            self._minimax_markup_buffer,
                            text,
                        )
                        self._minimax_markup_buffer = new_buffer
                        if recovered_calls and not self._saw_structured_tool_calls:
                            self._append_recovered_tool_calls(recovered_calls)
                        text = clean_text

                    # 渐进式刷新：只要此前累积了 reasoning_content，在正文到来前
                    # 先打包输出一次 thought 帧并清空，保证「先思考、后回答」的顺序。
                    # 对交替推送 reasoning/content 的模型同样成立。
                    if text and self._reasoning_acc:
                        full_reasoning = "".join(self._reasoning_acc)
                        self._reasoning_acc = []
                        logger.info(f"StreamProcessor: emitting reasoning thought len={len(full_reasoning)}")
                        yield self._format_sse_frame(
                            self._make_thought_frame_dict(full_reasoning)
                        )
                    # 通过状态机处理文本
                    if text:
                        for frame_dict in self._process_text_delta(text):
                            yield self._format_sse_frame(frame_dict)

            # ==================================================================
            # 收尾：按顺序输出残余内容
            # ==================================================================

            # 1. 探测缓冲未消费（流结束时仍处于 detecting 阶段，未见 <think>）→ 普通文本
            if self._minimax_markup_buffer:
                # 若末尾仍有未闭合 MiniMax 标记，按普通文本回传，避免内容丢失。
                for frame_dict in self._process_text_delta(self._minimax_markup_buffer):
                    yield self._format_sse_frame(frame_dict)
                self._minimax_markup_buffer = ""

            if self._detect_buffer:
                yield self._format_sse_frame(
                    self._make_text_frame_dict(self._detect_buffer)
                )
                self._detect_buffer = ""

            # 2. 思考缓冲未闭合（收到了 <think> 但没找到 </think>）→ 作为 thought 输出
            if self._think_buffer:
                yield self._format_sse_frame(
                    self._make_thought_frame_dict(self._think_buffer)
                )
                self._think_buffer = ""

            # 3. 未刷新的 reasoning_content：整个流只有推理没有 content，
            #    或最后一段推理之后再无正文（收尾兜底）。
            if self._reasoning_acc:
                full_reasoning = "".join(self._reasoning_acc)
                logger.info(f"StreamProcessor: emitting tail reasoning thought len={len(full_reasoning)}")
                yield self._format_sse_frame(
                    self._make_thought_frame_dict(full_reasoning)
                )

            # 4. tool_calls 一次性输出
            tool_frame = self._build_tool_call_frame()
            if tool_frame is not None:
                yield self._format_sse_frame(tool_frame)

            # 5. finishReason + usageMetadata（最后一帧）
            default_finish = "OTHER" if self._stream_error else "STOP"
            final_frame = self._build_final_frame(default_finish=default_finish)
            yield self._format_sse_frame(final_frame)

            self._closed = True

        except Exception as e:
            logger.error(f"StreamProcessor: unhandled error: {e}")
            if not self._closed:
                try:
                    yield self._format_sse_frame({
                        "error": {
                            "code": 500,
                            "message": f"Stream processing error: {e}",
                            "status": "INTERNAL",
                        }
                    })
                except Exception:
                    pass
            raise

    # ====================================================================
    # 文本状态机
    # ====================================================================

    def _process_text_delta(self, text: str) -> List[Dict[str, Any]]:
        """将文本 delta 通过状态机处理，返回 0 或多个 Gemini 帧 dict。"""
        if self._think_state == "streaming":
            return [self._make_text_frame_dict(text)]
        if self._think_state == "in_think":
            return self._handle_in_think(text)
        # detecting
        return self._handle_detecting(text)

    def _handle_detecting(self, text: str) -> List[Dict[str, Any]]:
        """探测阶段：判断文本是否以思考开标签开头。"""
        self._detect_buffer += text
        stripped = self._detect_buffer.lstrip()

        if not stripped:
            return []

        # 首字符不是 '<' → 确定非思考标签，立即切换到 streaming 并输出
        if stripped[0] != "<":
            buf = self._detect_buffer
            self._detect_buffer = ""
            self._think_state = "streaming"
            return [self._make_text_frame_dict(buf)]

        # 尝试匹配完整的思考开标签
        match = _OPEN_TAGS_RE.match(stripped)
        if match:
            self._think_state = "in_think"
            self._think_tag_name = match.group(1)
            remaining = stripped[match.end():]
            self._detect_buffer = ""
            self._think_buffer = remaining
            # remaining 中可能已经含有闭标签
            return self._handle_in_think("")

        # 以 '<' 开头但还不足以判断 → 继续缓冲，直到超过阈值
        if len(stripped) > _THINK_DETECT_CHARS:
            buf = self._detect_buffer
            self._detect_buffer = ""
            self._think_state = "streaming"
            return [self._make_text_frame_dict(buf)]

        return []

    def _handle_in_think(self, text: str) -> List[Dict[str, Any]]:
        """思考块内：缓冲直到找到对应的闭标签。"""
        if text:
            self._think_buffer += text

        close_re = re.compile(
            rf"</{re.escape(self._think_tag_name)}\s*>",
            re.IGNORECASE,
        )
        m = close_re.search(self._think_buffer)
        if m is None:
            return []  # 还没找到闭标签，继续缓冲

        thought_text = self._think_buffer[:m.start()]
        after_text = self._think_buffer[m.end():].lstrip("\n\r \t")
        self._think_state = "streaming"
        self._think_buffer = ""
        self._think_tag_name = ""

        frames: List[Dict[str, Any]] = []
        if thought_text:
            frames.append(self._make_thought_frame_dict(thought_text))
        if after_text:
            frames.append(self._make_text_frame_dict(after_text))
        return frames

    # ====================================================================
    # 帧构造工具
    # ====================================================================

    @staticmethod
    def _make_text_frame_dict(text: str) -> Dict[str, Any]:
        return {
            "candidates": [{
                "content": {
                    "role": fields.ROLE_MODEL,
                    "parts": [{"text": text}],
                },
                "index": 0,
            }],
        }

    @staticmethod
    def _make_thought_frame_dict(thought: str) -> Dict[str, Any]:
        return {
            "candidates": [{
                "content": {
                    "role": fields.ROLE_MODEL,
                    "parts": [{"thought": True, "text": thought}],
                },
                "index": 0,
            }],
        }

    def _build_tool_call_frame(self) -> Optional[Dict[str, Any]]:
        """收尾时一次性输出累积的 tool_calls。"""
        if not self._tool_calls_acc:
            return None

        parts: List[Dict[str, Any]] = []
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

    def _append_recovered_tool_calls(self, recovered_calls: List[Dict[str, Any]]) -> None:
        """将恢复出的 MiniMax 文本 tool_call 追加到内部聚合槽。"""
        next_idx = (max(self._tool_calls_acc.keys()) + 1) if self._tool_calls_acc else 0
        for offset, call in enumerate(recovered_calls):
            idx = next_idx + offset
            args = call.get("args", {})
            if isinstance(args, str):
                args_payload = args
            else:
                args_payload = json.dumps(args, ensure_ascii=False)
            self._tool_calls_acc[idx] = {
                "id": f"call_recovered_{idx}",
                "name": str(call.get("name", "")),
                "arguments": args_payload,
            }
            self._recovered_tool_call_indexes.add(idx)

    def _build_final_frame(self, default_finish: str = "OTHER") -> Dict[str, Any]:
        """收尾帧：finishReason + usageMetadata。"""
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
            candidate["finishReason"] = default_finish

        logger.info(
            f"StreamProcessor: stream complete "
            f"last_finish_reason={self._last_finish_reason} "
            f"default_finish={default_finish}"
        )
        if self._last_usage:
            details = self._last_usage.get("completion_tokens_details") or {}
            reasoning_tokens = int(details.get("reasoning_tokens") or 0)
            completion_tokens = self._last_usage.get("completion_tokens", 0)
            usage_meta: Dict[str, Any] = {
                "promptTokenCount": self._last_usage.get("prompt_tokens", 0),
                "candidatesTokenCount": completion_tokens - reasoning_tokens,
                "totalTokenCount": self._last_usage.get("total_tokens", 0),
            }
            if reasoning_tokens > 0:
                usage_meta["thoughtsTokenCount"] = reasoning_tokens
            frame["usageMetadata"] = usage_meta

        return frame

    def _format_sse_frame(self, frame: Dict[str, Any]) -> bytes:
        """将 Gemini 帧格式化为标准的 SSE `data:` 帧。

        格式：data: {...}\\n\\n
        - 必须是 JSON 对象（@google/genai SDK 期望，不能用数组包）
        - 末尾必须有 \\n\\n（SSE 事件分隔符）
        """
        body = json.dumps(frame, ensure_ascii=False).encode("utf-8")
        return b"data: " + body + b"\n\n"
