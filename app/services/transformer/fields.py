"""字段映射常量与工具函数。

集中存放 Gemini ↔ OpenAI 字段映射规则，避免在请求/响应两侧散落硬编码。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---- Role 映射 ----

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_TOOL = "tool"
ROLE_MODEL = "model"
ROLE_FUNCTION = "function"


def map_gemini_role_to_openai(gemini_role: str) -> Optional[str]:
    """Gemini role → OpenAI role。

    "user"   → "user"
    "model"  → "assistant"
    "function" → "tool"（需要配合 tool_call_id，单独处理）
    "system" → "system"
    其他 → None（跳过）
    """
    if gemini_role == ROLE_USER:
        return ROLE_USER
    if gemini_role == ROLE_MODEL:
        return ROLE_ASSISTANT
    if gemini_role == ROLE_SYSTEM:
        return ROLE_SYSTEM
    if gemini_role == ROLE_FUNCTION:
        return ROLE_TOOL
    return None


# ---- finish_reason 映射 ----

FINISH_REASON_TO_GEMINI: Dict[str, str] = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",          # 工具调用也算正常结束（含 functionCall）
    "function_call": "STOP",        # 旧版 OpenAI 字段
    "content_filter": "SAFETY",
}


def map_finish_reason(openai_reason: Optional[str], has_tool_calls: bool = False) -> str:
    """OpenAI finish_reason → Gemini finishReason。

    Args:
        openai_reason: 上游返回的 finish_reason 字符串
        has_tool_calls: 响应中是否含 tool_calls（用于兜底推理）
    """
    if openai_reason and openai_reason in FINISH_REASON_TO_GEMINI:
        return FINISH_REASON_TO_GEMINI[openai_reason]
    # 兜底：上游缺失 finish_reason 但有 tool_calls
    if has_tool_calls:
        return "STOP"
    return "OTHER"


# ---- toolConfig.mode → tool_choice ----

TOOL_CONFIG_MODE_TO_CHOICE: Dict[str, str] = {
    "AUTO": "auto",
    "ANY": "required",
    "NONE": "none",
}


def map_tool_config_mode(mode: str) -> Optional[str]:
    """Gemini toolConfig.functionCallingConfig.mode → OpenAI tool_choice。"""
    return TOOL_CONFIG_MODE_TO_CHOICE.get(mode)


# ---- responseMimeType → response_format.type ----

MIME_TO_RESPONSE_FORMAT: Dict[str, str] = {
    "application/json": "json",
    "text/plain": "text",
}


def map_response_mime_type(mime: str) -> Optional[Dict[str, str]]:
    """Gemini generationConfig.responseMimeType → OpenAI response_format。"""
    rt = MIME_TO_RESPONSE_FORMAT.get(mime)
    if rt is None:
        return None
    return {"type": rt}


# ---- generationConfig 字段映射 ----

GENERATION_CONFIG_MAP: Dict[str, str] = {
    "temperature": "temperature",
    "maxOutputTokens": "max_tokens",
    "topP": "top_p",
    "candidateCount": "n",
    "stopSequences": "stop",
}


def map_generation_config(gen_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """将 Gemini generationConfig 字段按映射表转写到 OpenAI 请求顶层。

    topK 等不存在的字段被丢弃（按 transformer.md §1.2）。
    """
    if not gen_config:
        return {"max_tokens": 32768}
    out: Dict[str, Any] = {}
    for g_field, o_field in GENERATION_CONFIG_MAP.items():
        if g_field in gen_config:
            out[o_field] = gen_config[g_field]
    
    # 默认给一个充足的 max_tokens，避免上游使用极小的缺省值截断输出
    if "max_tokens" not in out:
        out["max_tokens"] = 32768

    # responseMimeType 单独处理
    if "responseMimeType" in gen_config:
        rf = map_response_mime_type(gen_config["responseMimeType"])
        if rf:
            out["response_format"] = rf
    # 丢弃字段：topK / presencePenalty / frequencyPenalty / seed（OpenAI 有 seed 但语义略不同）
    return out


# ---- Part 提取 ----

def extract_text_from_parts(parts: List[Dict[str, Any]]) -> str:
    """从 parts 数组中拼接所有 text 字段。

    兼容两种格式：
      1. Gemini 官方：`{"text": "..."}` （无 type 字段）
      2. 部分 SDK：`{"type": "text", "text": "..."}`
    """
    chunks: List[str] = []
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        # 优先按 "text" 字段识别（Gemini 官方格式）
        if "text" in p and p.get("type", "text") == "text":
            chunks.append(str(p["text"]))
        # 退化：仅当 type 显式为 "text" 时才采用
        elif p.get("type") == "text" and "text" in p:
            chunks.append(str(p["text"]))
    return "\n".join(chunks)
