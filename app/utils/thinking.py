"""Thinking / reasoning 标签解析与清洗工具。

支持提取 <think>, <thinking>, <reflection>, <reasoning>, <antml:thinking> 等推理链并转换为 Gemini parts 结构：
[{"thought": True, "text": "推理文本"}, {"text": "回答文本"}]
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# 支持匹配各种推理标签的开标签（仅用于判断文本是否以推理标签开头）
_OPEN_TAGS_RE = re.compile(r"^<(think|thinking|reflection|reasoning|antml:thinking)\b[^>]*>", re.IGNORECASE)

# 支持匹配文本任意位置出现的成对推理标签块（用于非流式完整文本的多块解析）
_THINK_BLOCK_RE = re.compile(
    r"<(think|thinking|reflection|reasoning|antml:thinking)\b[^>]*>(.*?)</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


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
    """将文本中的各种推理标签内容提取，转换为 Gemini 规范的 parts 结构。

    行为：
      1. 若上游返回独立的 reasoning_content（如 DeepSeek），优先作为首个 thought 段。
      2. 解析文本中**任意位置**出现的成对推理标签块（支持多个块），
         块内容作为 thought 段，块外文本作为普通 text 段，顺序保留。
      3. 若文本以推理开标签开头但未闭合（流式残留），剩余部分整体作为 thought。
      4. 无任何推理标签时，整段作为普通 text 返回。
    """
    parts: List[Dict[str, Any]] = []

    # 1. 独立 reasoning_content：作为首个 thought 段；
    #    此时文本内容不再二次解析思考标签，保持字面（避免与已提供的推理内容重复）。
    if reasoning_content:
        parts.append({"thought": True, "text": reasoning_content})
        if text:
            parts.append({"text": text})
        return parts

    if not text:
        return parts

    # 2. 解析任意位置的成对推理标签块
    last_end = 0
    matched_any = False
    for m in _THINK_BLOCK_RE.finditer(text):
        matched_any = True
        # 块前的普通文本
        before = text[last_end:m.start()]
        if before.strip():
            parts.append({"text": before})
        # 块内思考内容
        thought_val = m.group(2)
        if thought_val.strip():
            parts.append({"thought": True, "text": thought_val})
        last_end = m.end()

    if matched_any:
        # 最后一个块之后的普通文本
        tail = text[last_end:]
        if tail.strip():
            parts.append({"text": tail.lstrip("\n\r \t")})
        return parts

    # 3. 未闭合的开标签（仅当出现在文本开头时按 thought 处理）
    stripped_text = text.lstrip()
    match_open = _OPEN_TAGS_RE.match(stripped_text)
    if match_open:
        thought_val = stripped_text[match_open.end():]
        if thought_val:
            parts.append({"thought": True, "text": thought_val})
        return parts

    # 4. 普通文本
    parts.append({"text": text})
    return parts
