"""OpenAI 响应 → Gemini 响应 转换器。

按 docs/design/transformer.md §2 规范实现。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.services.transformer import fields
from app.utils.minimax_tool_markup import (
    contains_minimax_markup,
    recover_minimax_tool_calls_from_text,
)
from app.utils.thinking import parse_thinking_segments


class ResponseTransformer:
    """OpenAI chat/completions 响应 → Gemini generateContent 响应 转换器。"""

    def transform(self, openai_resp: Dict[str, Any]) -> Dict[str, Any]:
        """主入口。返回 Gemini `generateContent` 响应 JSON 字典。"""
        # choices 缺失 / 为空 → 返回空 candidates + error 包装
        choices = openai_resp.get("choices")
        if not choices:
            return {
                "candidates": [],
                "error": {
                    "code": 502,
                    "message": "Upstream returned no choices",
                    "status": "UNAVAILABLE",
                },
            }

        # 只取第一条 choice（Gemini 无多候选语义）
        choice = choices[0]
        msg = choice.get("message", {})
        response_model = str(openai_resp.get("model") or "")

        parts: List[Dict[str, Any]] = []
        tool_call_ids: List[str] = []
        recovered_calls: List[Dict[str, Any]] = []
        model_is_minimax = "minimax" in response_model.lower()
        should_recover_minimax = settings.minimax_tool_markup_recovery and model_is_minimax

        # ---- 文本 ----
        content = msg.get("content")
        reasoning_content = msg.get("reasoning_content")
        if content or reasoning_content:
            if isinstance(content, str) or reasoning_content:
                clean_content = content or ""
                if (
                    settings.minimax_tool_markup_recovery
                    and isinstance(content, str)
                    and (should_recover_minimax or contains_minimax_markup(content))
                ):
                    clean_content, recovered_calls = recover_minimax_tool_calls_from_text(content)
                parts.extend(parse_thinking_segments(clean_content, reasoning_content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item_text = item.get("text", "")
                        if not isinstance(item_text, str):
                            continue
                        if (
                            settings.minimax_tool_markup_recovery
                            and (should_recover_minimax or contains_minimax_markup(item_text))
                        ):
                            clean_text, item_calls = recover_minimax_tool_calls_from_text(item_text)
                            recovered_calls.extend(item_calls)
                            parts.extend(parse_thinking_segments(clean_text))
                        else:
                            parts.extend(parse_thinking_segments(item_text))

        # ---- 工具调用 ----
        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            try:
                args = json.loads(fn.get("arguments", "{}")) if fn.get("arguments") else {}
            except (json.JSONDecodeError, TypeError):
                args = fn.get("arguments", "")
            tc_id = tc.get("id") or f"call_{len(tool_call_ids)}"
            tool_call_ids.append(tc_id)
            parts.append({
                "functionCall": {
                    "id": tc_id,
                    "name": fn.get("name", ""),
                    "args": args,
                }
            })
        # MiniMax 特殊兜底：当上游未提供结构化 tool_calls 时，尝试从 content 恢复。
        if not tool_call_ids:
            for idx, call in enumerate(recovered_calls):
                tc_id = f"call_recovered_{idx}"
                tool_call_ids.append(tc_id)
                parts.append({
                    "functionCall": {
                        "id": tc_id,
                        "name": call.get("name", ""),
                        "args": call.get("args", {}),
                    }
                })

        # ---- finishReason ----
        has_tool_calls = bool(msg.get("tool_calls")) or bool(recovered_calls)
        finish_reason = fields.map_finish_reason(
            choice.get("finish_reason"),
            has_tool_calls=has_tool_calls,
        )

        # ---- 组装 candidates ----
        candidate: Dict[str, Any] = {
            "content": {
                "role": fields.ROLE_MODEL,
                "parts": parts,
            },
            "finishReason": finish_reason,
            "index": 0,
        }
        gemini_resp: Dict[str, Any] = {
            "candidates": [candidate],
        }

        # ---- usageMetadata ----
        usage = openai_resp.get("usage")
        if usage:
            gemini_resp["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            }

        return gemini_resp


# 全局单例
response_transformer = ResponseTransformer()
