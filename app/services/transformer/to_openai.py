"""Gemini 请求体 → OpenAI 请求体 转换器。

按 docs/design/transformer.md 规范实现。

关键约束（v0.1.0-r1）：
  - model 字段**强制覆盖**：永远使用 settings.upstream_model，忽略入站任何位置传入的 model。
  - 详细字段映射见 transformer.md §1.2。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.services.transformer import fields


class RequestTransformer:
    """Gemini 请求 → OpenAI 请求 转换器。"""

    def transform(
        self,
        gemini_req: Dict[str, Any],
        *,
        stream: bool = False,
        inbound_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """主入口。

        Args:
            gemini_req: Gemini generateContent 请求体（已 JSON 解析）
            stream: 是否流式（写入 OpenAI 请求的 `stream` 字段）
            inbound_model: 客户端传入的 model 名（仅用于日志，不影响出站）

        Returns:
            可直接 JSON 序列化的 OpenAI chat/completions 请求体
        """
        # ---- 1. model 强制覆盖 ----
        outbound_model = settings.upstream_model
        if inbound_model is not None:
            truncated = inbound_model[:64]
            logger.info(
                f"[inbound] model={truncated!r} → upstream_model={outbound_model!r}"
            )

        openai_req: Dict[str, Any] = {
            "model": outbound_model,
            "messages": [],
            "stream": stream,
        }

        # ---- 2. systemInstruction → 首条 system 消息 ----
        sys_inst = gemini_req.get("systemInstruction")
        if sys_inst:
            parts = sys_inst.get("parts", [])
            sys_text = fields.extract_text_from_parts(parts)
            if sys_text:
                openai_req["messages"].append({
                    "role": fields.ROLE_SYSTEM,
                    "content": sys_text,
                })

        # ---- 3. contents → messages ----
        # 维护跨 contents 的 tool_call_id 映射：
        #   - name -> 最近一次出现的 tool_call_id（多轮会刷新）
        #   - ids  -> 按出现顺序的所有 id（用于 functionResponse 缺 id 时按顺序回退）
        last_tool_call_ids: Dict[str, str] = {}
        ordered_tool_call_ids: List[str] = []

        for item in gemini_req.get("contents", []):
            if not isinstance(item, dict):
                continue
            role = item.get("role", "")
            parts = item.get("parts", []) or []
            openai_role = fields.map_gemini_role_to_openai(role)
            if openai_role is None:
                continue

            # ---- 3.1 role="function"（Gemini 旧 SDK 残留）：整条就是工具回传 ----
            if openai_role == fields.ROLE_TOOL:
                tool_msgs = self._extract_tool_messages_from_parts(
                    parts, last_tool_call_ids, ordered_tool_call_ids
                )
                openai_req["messages"].extend(tool_msgs)
                continue

            # ---- 3.2 抽取该 part 中的 functionCall → assistant.tool_calls ----
            text_chunks, content_items, tool_calls = self._partition_parts(parts)

            # ---- 3.3 抽取该 part 中的 functionResponse → 紧跟在本 assistant 后的 role=tool 消息 ----
            # 这是关键：gemini-cli 把 functionResponse 放在 role="user" 的 parts 内。
            tool_msgs_inline = self._extract_tool_messages_from_parts(
                parts, last_tool_call_ids, ordered_tool_call_ids, only_function_response=True
            )

            # ---- 3.4 构造普通消息 ----
            if text_chunks or content_items or tool_calls:
                msg: Dict[str, Any] = {"role": openai_role}
                if text_chunks and not content_items and not tool_calls:
                    msg["content"] = "\n".join(text_chunks)
                elif content_items:
                    if text_chunks:
                        content_items.append({
                            "type": "text",
                            "text": "\n".join(text_chunks),
                        })
                    msg["content"] = content_items
                elif tool_calls:
                    msg["content"] = None  # OpenAI 允许 tool_calls 时 content 为 None
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                    # 记录 tool_call_id，便于后续 functionResponse 关联
                    for tc in tool_calls:
                        if tc.get("id"):
                            last_tool_call_ids[tc["function"]["name"]] = tc["id"]
                            ordered_tool_call_ids.append(tc["id"])
                openai_req["messages"].append(msg)

            # ---- 3.5 把工具回传插在本 assistant/tool_calls 之后（OpenAI 强约束） ----
            if tool_msgs_inline:
                openai_req["messages"].extend(tool_msgs_inline)

        # ---- 4. generationConfig ----
        gen_config = gemini_req.get("generationConfig", {})
        openai_req.update(fields.map_generation_config(gen_config))

        # ---- 5. tools ----
        if gemini_req.get("tools"):
            openai_req["tools"] = self._flatten_function_declarations(gemini_req["tools"])
        # tool_choice 只在同时有 tools 与 toolConfig 时才有意义
        if openai_req.get("tools") and gemini_req.get("toolConfig"):
            choice = self._map_tool_choice(gemini_req["toolConfig"])
            if choice is not None:
                openai_req["tool_choice"] = choice

        # ---- 6. json_object 兼容性补齐 ----
        # 许多上游模型要求：启用 response_format=json_object 时，
        # 提示（system/user）中必须显式出现 "json" 一词，否则直接报错。
        # 若客户端未在提示里带上该词，则在 system 消息中补一句说明。
        self._ensure_json_hint(openai_req)

        return openai_req

    @staticmethod
    def _ensure_json_hint(openai_req: Dict[str, Any]) -> None:
        rf = openai_req.get("response_format")
        if not (isinstance(rf, dict) and rf.get("type") == "json_object"):
            return

        def _contains_json(content: Any) -> bool:
            if isinstance(content, str):
                return "json" in content.lower()
            if isinstance(content, list):
                for it in content:
                    if isinstance(it, dict) and "json" in str(it.get("text", "")).lower():
                        return True
            return False

        messages = openai_req.get("messages", [])
        if any(_contains_json(m.get("content")) for m in messages):
            return

        hint = "Please respond with a valid JSON object."
        for m in messages:
            if m.get("role") == fields.ROLE_SYSTEM and isinstance(m.get("content"), str):
                m["content"] = f"{m['content']}\n\n{hint}"
                return
        # 没有可复用的 system 消息 → 在最前插入一条
        messages.insert(0, {"role": fields.ROLE_SYSTEM, "content": hint})

    # ====================================================================
    # 内部工具
    # ====================================================================

    def _partition_parts(
        self,
        parts: List[Dict[str, Any]],
    ) -> tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """将 Gemini parts 拆分为 (text_chunks, content_items, tool_calls)。

        - text_chunks: 纯文本累积（用于字符串型 content）。
        - content_items: 多元素 content（用于包含图片等多模态场景）。
        - tool_calls: OpenAI 风格的 tool_calls 数组。

        注意：本方法**忽略** `functionResponse`（由 `_extract_tool_messages_from_parts`
        单独处理）。这样可以让普通消息和工具回传在同一 contents 节点中并行处理。
        """
        text_chunks: List[str] = []
        content_items: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []

        for p in parts:
            if not isinstance(p, dict):
                continue

            # ---- 工具回传：本方法跳过 ----
            if p.get("functionResponse"):
                continue

            p_type = p.get("type") or p.get("part_type")

            # ---- 文本 ----
            if p.get("text") is not None or p_type == "text":
                if p.get("thought"):
                    text_chunks.append(f"<think>{p.get('text', '')}</think>")
                else:
                    text_chunks.append(str(p.get("text", "")))
                continue

            # ---- 多模态 inline_data ----
            if "inline_data" in p or p_type == "inline_data":
                inline = p.get("inline_data", {})
                mime = inline.get("mime_type", "image/jpeg")
                data = inline.get("data", "")
                data_url = f"data:{mime};base64,{data}"
                if text_chunks:
                    content_items.append({
                        "type": "text",
                        "text": "\n".join(text_chunks),
                    })
                    text_chunks = []
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
                continue

            # ---- file_data (Gemini 1.5+ URI 形式) → 转 image_url ----
            if "file_data" in p or p_type == "file_data":
                fd = p.get("file_data", {})
                file_uri = fd.get("file_uri", "")
                mime = fd.get("mime_type", "image/jpeg")
                url = file_uri  # data: URL 直传；其它 URL 也直传
                if text_chunks:
                    content_items.append({
                        "type": "text",
                        "text": "\n".join(text_chunks),
                    })
                    text_chunks = []
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
                continue

            # ---- functionCall → tool_calls ----
            if "functionCall" in p or p_type == "function_call":
                fc = p.get("functionCall", {})
                if text_chunks:
                    content_items.append({
                        "type": "text",
                        "text": "\n".join(text_chunks),
                    })
                    text_chunks = []
                # 兼容两种 ID 字段（顶层 id 或 functionCall.id）
                fc_id = p.get("id") or fc.get("id")
                tool_calls.append({
                    "id": fc_id,
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                    },
                })
                continue

        return text_chunks, content_items, tool_calls

    def _extract_tool_messages_from_parts(
        self,
        parts: List[Dict[str, Any]],
        last_tool_call_ids: Dict[str, str],
        ordered_tool_call_ids: List[str],
        only_function_response: bool = False,
    ) -> List[Dict[str, Any]]:
        """把 parts 内的 functionResponse 转成 OpenAI `role="tool"` 消息列表。

        Args:
            parts: Gemini parts 数组
            last_tool_call_ids: name -> 最近一次 tool_call_id
            ordered_tool_call_ids: 跨 contents 出现过的 id 序列（用于顺序回退）
            only_function_response: 若 True，仅处理 functionResponse 类型的 part；
                                    旧 path 仍兼容老 parts 结构
        """
        msgs: List[Dict[str, Any]] = []
        pending_ids: List[str] = list(ordered_tool_call_ids)

        for p in parts:
            if not isinstance(p, dict):
                continue
            fr = p.get("functionResponse")
            if not fr:
                continue

            name = fr.get("name", "")
            response = fr.get("response", {})
            # tool_call_id 三级回退：
            # 1) 该 part 自带的 id（顶层或 fr 内部）
            part_id = (
                p.get("id")
                or fr.get("id")
            )
            # 2) 历史中按 name 关联
            # 3) 按出现顺序的最近 id
            tool_call_id = (
                part_id
                or last_tool_call_ids.get(name)
                or self._pop_sequential_id(pending_ids)
                or f"call_{name}"  # 最后兜底
            )

            # content 优先使用 response 字段；否则用 string
            content_str = (
                json.dumps(response, ensure_ascii=False)
                if isinstance(response, (dict, list))
                else str(response)
            )

            msgs.append({
                "role": fields.ROLE_TOOL,
                "tool_call_id": tool_call_id,
                "content": content_str,
            })
        return msgs

    @staticmethod
    def _pop_sequential_id(pending_ids: List[str]) -> Optional[str]:
        """从顺序队列中弹出一个尚未使用的 id（用于多函数并发的兜底）。"""
        while pending_ids:
            return pending_ids.pop(0)
        return None

    def _flatten_function_declarations(
        self,
        gemini_tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """将 Gemini tools（含 functionDeclarations 嵌套）展平为 OpenAI tools[]。"""
        out: List[Dict[str, Any]] = []
        for t in gemini_tools:
            if not isinstance(t, dict):
                continue
            decls = t.get("functionDeclarations", [])
            for d in decls:
                if not isinstance(d, dict):
                    continue
                out.append({
                    "type": "function",
                    "function": {
                        "name": d.get("name", ""),
                        "description": d.get("description", ""),
                        "parameters": d.get(
                            "parametersJsonSchema",
                            d.get("parameters", {"type": "object", "properties": {}}),
                        ),
                    },
                })
        return out

    def _map_tool_choice(
        self,
        tool_config: Dict[str, Any],
    ) -> Optional[Any]:
        """将 Gemini toolConfig 转 OpenAI tool_choice。

        AUTO  → "auto"
        ANY   → "required"
        NONE  → "none"
        """
        fcc = tool_config.get("functionCallingConfig", {})
        mode = fcc.get("mode", "AUTO")
        choice = fields.map_tool_config_mode(mode)
        if choice is None:
            return None
        # 单一函数限制：allowedFunctionNames 长度=1 → tool_choice={type:function, function:{name}}
        allowed = fcc.get("allowedFunctionNames")
        if isinstance(allowed, list) and len(allowed) == 1 and choice != "none":
            return {
                "type": "function",
                "function": {"name": allowed[0]},
            }
        return choice


# 全局单例
request_transformer = RequestTransformer()
