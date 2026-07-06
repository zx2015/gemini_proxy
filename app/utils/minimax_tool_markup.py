"""MiniMax M3 文本内嵌 tool_call 标记的解析与清理工具。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


NAMESPACE_TOKEN = "]<]minimax[>["
TOOL_CALL_START = f"{NAMESPACE_TOKEN}<tool_call>"
TOOL_CALL_END = f"{NAMESPACE_TOKEN}</tool_call>"

_NS_RE = re.escape(NAMESPACE_TOKEN)
_INVOKE_RE = re.compile(
    rf"{_NS_RE}<invoke\s+name=\"([^\"]+)\">(.*?){_NS_RE}</invoke>",
    re.DOTALL,
)
_OPEN_TAG_RE = re.compile(rf"{_NS_RE}<([^/!?\s>]+)>")


def contains_minimax_markup(text: str) -> bool:
    """判断文本中是否出现 MiniMax 命名空间标记。"""
    return NAMESPACE_TOKEN in text or "<tool_call>" in text or "<invoke name=" in text


def parse_minimax_tool_call_block(block: str) -> List[Dict[str, Any]]:
    """解析完整的 MiniMax tool_call 块为通用函数调用列表。"""
    calls: List[Dict[str, Any]] = []
    for name, args_block in _INVOKE_RE.findall(block):
        args: Dict[str, Any] = {}
        for key, inner in _iter_keyed_tags(args_block):
            args[key] = _parse_xml_value(inner)
        calls.append({"name": name, "args": args})
    return calls


def recover_minimax_tool_calls_from_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """从完整文本中恢复 MiniMax tool_call，并返回清理后的文本。"""
    clean_parts: List[str] = []
    recovered_calls: List[Dict[str, Any]] = []
    cursor = 0
    text_len = len(text)

    while cursor < text_len:
        start = text.find(TOOL_CALL_START, cursor)
        if start == -1:
            clean_parts.append(text[cursor:])
            break
        clean_parts.append(text[cursor:start])
        end = text.find(TOOL_CALL_END, start + len(TOOL_CALL_START))
        if end == -1:
            # 非流式场景下无收尾，按普通文本保留，避免误删。
            clean_parts.append(text[start:])
            break
        block_end = end + len(TOOL_CALL_END)
        block = text[start:block_end]
        recovered_calls.extend(parse_minimax_tool_call_block(block))
        cursor = block_end

    return "".join(clean_parts), recovered_calls


def consume_minimax_markup_chunks(
    existing_buffer: str,
    incoming_chunk: str,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """流式场景：消费一段文本，返回 clean_text / recovered_calls / new_buffer。"""
    source = existing_buffer + incoming_chunk
    clean_parts: List[str] = []
    recovered_calls: List[Dict[str, Any]] = []
    cursor = 0
    source_len = len(source)

    while cursor < source_len:
        start = source.find(TOOL_CALL_START, cursor)
        if start == -1:
            clean_parts.append(source[cursor:])
            return "".join(clean_parts), recovered_calls, ""
        clean_parts.append(source[cursor:start])
        end = source.find(TOOL_CALL_END, start + len(TOOL_CALL_START))
        if end == -1:
            # 末尾是未闭合标记，留给后续 chunk 继续拼接。
            return "".join(clean_parts), recovered_calls, source[start:]
        block_end = end + len(TOOL_CALL_END)
        block = source[start:block_end]
        recovered_calls.extend(parse_minimax_tool_call_block(block))
        cursor = block_end

    return "".join(clean_parts), recovered_calls, ""


def _iter_keyed_tags(content: str):
    cursor = 0
    total = len(content)
    while cursor < total:
        match = _OPEN_TAG_RE.search(content, cursor)
        if match is None:
            return
        tag_name = match.group(1)
        open_marker = f"{NAMESPACE_TOKEN}<{tag_name}>"
        close_marker = f"{NAMESPACE_TOKEN}</{tag_name}>"

        depth = 1
        scan = match.end()
        while depth > 0 and scan < total:
            next_open = content.find(open_marker, scan)
            next_close = content.find(close_marker, scan)
            if next_close == -1:
                return
            if next_open != -1 and next_open < next_close:
                depth += 1
                scan = next_open + len(open_marker)
            else:
                depth -= 1
                scan = next_close + len(close_marker)

        inner = content[match.end(): scan - len(close_marker)]
        yield tag_name, inner
        cursor = scan


def _parse_xml_value(content: str) -> Any:
    value = content.strip()
    if not value:
        return ""
    if value.startswith(f"{NAMESPACE_TOKEN}<item>"):
        return [_parse_xml_value(inner) for inner in _iter_tagged_items(value)]
    if value.startswith(f"{NAMESPACE_TOKEN}<"):
        nested: Dict[str, Any] = {}
        for key, inner in _iter_keyed_tags(value):
            nested[key] = _parse_xml_value(inner)
        if nested:
            return nested
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _iter_tagged_items(content: str):
    pattern = re.compile(
        rf"{_NS_RE}<item>(.*?){_NS_RE}</item>",
        re.DOTALL,
    )
    for match in pattern.finditer(content):
        yield match.group(1)
