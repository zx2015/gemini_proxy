"""Thinking / reasoning 标签解析与清洗工具。

支持提取 <think>, <thinking>, <reflection>, <reasoning>, <antml:thinking> 等推理链并转换为 Gemini parts 结构：
[{"thought": True, "text": "推理文本"}, {"text": "回答文本"}]
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# 支持匹配各种推理标签的开标签和闭标签
_OPEN_TAGS_RE = re.compile(r"^<(think|thinking|reflection|reasoning|antml:thinking)\b[^>]*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """仅当文本以支持的推理开标签开头时，移除文本中的推理块，否则原样返回。"""
    if not text:
        return text

    parts = parse_thinking_segments(text)
    # 筛选出没有 thought=True 的部分拼合成普通文本
    normal_parts = [p["text"] for p in parts if not p.get("thought")]
    if normal_parts:
        return "".join(normal_parts)
    # 如果全都是 thought 且原本包含标签，则说明被全部清洗了
    stripped_text = text.lstrip()
    if _OPEN_TAGS_RE.match(stripped_text):
        return ""
    return text


def parse_thinking_segments(
    text: Optional[str],
    reasoning_content: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """将文本中的各种推理标签内容提取，转换为 Gemini 规范的 parts 结构。"""
    parts = []

    # 1. 如果上游直接返回了独立的 reasoning_content (例如 DeepSeek)
    if reasoning_content:
        parts.append({"thought": True, "text": reasoning_content})
        if text:
            parts.append({"text": text})
        return parts

    if not text:
        return []

    # 2. 仅当文本以支持的推理标签开头时才解析为推理块，避免误伤正文中的字面讨论
    stripped_text = text.lstrip()
    match_open = _OPEN_TAGS_RE.match(stripped_text)
    if match_open:
        tag_name = match_open.group(1)
        open_tag_len = match_open.end()
        # 匹配对应的闭标签
        close_tag_re = re.compile(rf"</{re.escape(tag_name)}\s*>", re.IGNORECASE)
        close_tag_match = close_tag_re.search(stripped_text)
        if close_tag_match:
            close_start = close_tag_match.start()
            close_end = close_tag_match.end()
            
            thought_val = stripped_text[open_tag_len:close_start]
            normal_val = stripped_text[close_end:]
            
            if thought_val:
                parts.append({"thought": True, "text": thought_val})
            if normal_val:
                normal_val = normal_val.lstrip("\n\r \t")
                if normal_val:
                    parts.append({"text": normal_val})
        else:
            # 未闭合的标签，整个剩下的部分都是 thought
            thought_val = stripped_text[open_tag_len:]
            if thought_val:
                parts.append({"thought": True, "text": thought_val})
        return parts

    # 否则，不以推理开标签开头，整个文本作为普通文本返回
    parts.append({"text": text})
    return parts
