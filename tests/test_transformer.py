"""转换器单元测试。

覆盖 transformer.md §1 + §2 的字段映射矩阵 + §1.9 强制模型覆盖矩阵。
"""
from __future__ import annotations

import pytest

from app.services.transformer.from_openai import response_transformer
from app.services.transformer.to_openai import request_transformer


# ============================================================================
# 强制模型覆盖矩阵（v0.1.0-r1）
# ============================================================================

class TestForcedModelOverride:
    """验证 §1.9 强制模型覆盖：忽略入站 model。"""

    def test_override_with_gemini_model_name(self):
        gemini = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        out = request_transformer.transform(gemini, inbound_model="gemini-2.5-pro")
        # 强制覆盖为 settings.upstream_model
        from app.core.config import settings
        assert out["model"] == settings.upstream_model
        assert out["model"] != "gemini-2.5-pro"

    def test_override_with_non_gemini_name(self):
        gemini = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        out = request_transformer.transform(gemini, inbound_model="claude-3-5-sonnet")
        from app.core.config import settings
        assert out["model"] == settings.upstream_model

    def test_override_without_inbound_model(self):
        gemini = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        out = request_transformer.transform(gemini, inbound_model=None)
        from app.core.config import settings
        assert out["model"] == settings.upstream_model

    def test_override_with_empty_string(self):
        gemini = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        out = request_transformer.transform(gemini, inbound_model="")
        from app.core.config import settings
        assert out["model"] == settings.upstream_model

    def test_override_with_very_long_string(self):
        long_name = "x" * 5000
        gemini = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        out = request_transformer.transform(gemini, inbound_model=long_name)
        from app.core.config import settings
        assert out["model"] == settings.upstream_model

    def test_no_model_mapping_artifact(self):
        """验证响应中不含任何 MODEL_MAPPING 之类的痕迹。"""
        gemini = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        out = request_transformer.transform(gemini, inbound_model="gemini-2.5-pro")
        # 不应有任何旧字段
        assert "model_mapping" not in out
        assert "mapped_model" not in out


# ============================================================================
# 请求转换矩阵（transformer.md §1.2 - §1.6）
# ============================================================================

class TestRequestTransformation:
    """请求体转换核心矩阵。"""

    def test_basic_user_text(self):
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "你好"}]}]
        }
        out = request_transformer.transform(gemini)
        assert out["messages"] == [{"role": "user", "content": "你好"}]
        assert out["stream"] is False

    def test_with_system_instruction(self):
        gemini = {
            "systemInstruction": {"parts": [{"text": "你是助手"}]},
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        out = request_transformer.transform(gemini)
        assert out["messages"][0] == {"role": "system", "content": "你是助手"}
        assert out["messages"][1]["role"] == "user"

    def test_model_role_to_assistant(self):
        gemini = {
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "hello"}]},
            ]
        }
        out = request_transformer.transform(gemini)
        roles = [m["role"] for m in out["messages"]]
        assert roles == ["user", "assistant"]
        assert out["messages"][1]["content"] == "hello"

    def test_generation_config_mapping(self):
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024,
                "topP": 0.9,
                "candidateCount": 1,
            },
        }
        out = request_transformer.transform(gemini)
        assert out["temperature"] == 0.7
        assert out["max_tokens"] == 1024
        assert out["top_p"] == 0.9
        assert out["n"] == 1

    def test_generation_config_default_max_tokens(self):
        """若请求中没有 maxOutputTokens，不向 OpenAI 请求中传递 max_tokens 字段。"""
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        out = request_transformer.transform(gemini)
        assert "max_tokens" not in out

    def test_topK_is_dropped(self):
        """OpenAI 无 topK，对应字段应被丢弃。"""
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"topK": 40},
        }
        out = request_transformer.transform(gemini)
        assert "top_k" not in out
        assert "topK" not in out

    def test_response_mime_type_json(self):
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        out = request_transformer.transform(gemini)
        assert out["response_format"] == {"type": "json_object"}

    def test_json_object_injects_hint_when_missing(self):
        # 消息中不含 "json" 时，应补一条 system 提示以兼容上游校验
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "给我数据"}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        out = request_transformer.transform(gemini)
        assert out["response_format"] == {"type": "json_object"}
        joined = " ".join(
            m["content"] for m in out["messages"] if isinstance(m.get("content"), str)
        )
        assert "json" in joined.lower()
        assert out["messages"][0]["role"] == "system"

    def test_json_object_appends_to_existing_system(self):
        gemini = {
            "systemInstruction": {"parts": [{"text": "你是助手"}]},
            "contents": [{"role": "user", "parts": [{"text": "给我数据"}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        out = request_transformer.transform(gemini)
        sys_msgs = [m for m in out["messages"] if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert "你是助手" in sys_msgs[0]["content"]
        assert "json" in sys_msgs[0]["content"].lower()

    def test_json_object_no_duplicate_hint_when_present(self):
        # 用户提示已含 "json" 时，不应再注入额外 system 消息
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "return json please"}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        out = request_transformer.transform(gemini)
        assert all(m["role"] != "system" for m in out["messages"])

    def test_function_declarations_flattened(self):
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "天气？"}]}],
            "tools": [{
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "获取天气",
                        "parametersJsonSchema": {"type": "object", "properties": {"loc": {"type": "string"}}},
                    },
                    {
                        "name": "get_time",
                        "description": "获取时间",
                    },
                ],
            }],
        }
        out = request_transformer.transform(gemini)
        assert len(out["tools"]) == 2
        assert out["tools"][0]["type"] == "function"
        assert out["tools"][0]["function"]["name"] == "get_weather"
        assert "parameters" in out["tools"][0]["function"]
        assert out["tools"][1]["function"]["name"] == "get_time"

    def test_tool_config_any_to_required(self):
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "tools": [{"functionDeclarations": [{"name": "f", "description": "x"}]}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
        }
        out = request_transformer.transform(gemini)
        assert out["tool_choice"] == "required"

    def test_tool_config_none_to_none(self):
        gemini = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "tools": [{"functionDeclarations": [{"name": "f", "description": "x"}]}],
            "toolConfig": {"functionCallingConfig": {"mode": "NONE"}},
        }
        out = request_transformer.transform(gemini)
        assert out["tool_choice"] == "none"

    def test_inline_data_to_image_url(self):
        gemini = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "看图："},
                    {"inline_data": {"mime_type": "image/png", "data": "BASE64DATA"}},
                ],
            }],
        }
        out = request_transformer.transform(gemini)
        msg = out["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0] == {"type": "text", "text": "看图："}
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"]["url"] == "data:image/png;base64,BASE64DATA"

    def test_inline_data_trailing_text_preserves_order(self):
        """图片后面跟着文字时，顺序应保留（图→文），不能被提前到图片前。"""
        gemini = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": "BASE64DATA"}},
                    {"text": "这张图里有什么？"},
                ],
            }],
        }
        out = request_transformer.transform(gemini)
        msg = out["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "image_url"
        assert msg["content"][1] == {"type": "text", "text": "这张图里有什么？"}

    def test_function_call_in_parts(self):
        gemini = {
            "contents": [{
                "role": "model",
                "parts": [{
                    "functionCall": {"name": "get_weather", "args": {"loc": "SF"}}
                }],
            }],
        }
        out = request_transformer.transform(gemini)
        msg = out["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_user_role_function_response_becomes_tool_message(self):
        """gemini-cli 真实行为：role=user 的 parts 内携带 functionResponse。"""
        gemini = {
            "contents": [
                {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "list_files",
                            "args": {"path": "."},
                            "id": "call_list_1",
                        }
                    }],
                },
                {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": "list_files",
                            "response": {"result": "a.py\nb.py"},
                            "id": "call_list_1",
                        }
                    }],
                },
            ],
        }
        out = request_transformer.transform(gemini)
        assert out["messages"][0]["role"] == "assistant"
        assert out["messages"][0]["tool_calls"][0]["id"] == "call_list_1"
        assert out["messages"][1] == {
            "role": "tool",
            "tool_call_id": "call_list_1",
            "content": '{"result": "a.py\\nb.py"}',
        }

    def test_user_role_function_response_uses_previous_function_call_id_by_name(self):
        """functionResponse 无 id 时，按 functionCall.name 关联上一次 tool_call_id。"""
        gemini = {
            "contents": [
                {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "read_file",
                            "args": {"file_path": "README.md"},
                            "id": "call_read_1",
                        }
                    }],
                },
                {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": "read_file",
                            "response": {"content": "hello"},
                        }
                    }],
                },
            ],
        }
        out = request_transformer.transform(gemini)
        tool_msg = out["messages"][1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_read_1"
        assert tool_msg["content"] == '{"content": "hello"}'

    def test_function_response_fallback_id_when_no_prior_call(self):
        """没有历史 functionCall 时，兜底为 call_<name>，避免丢弃工具结果。"""
        gemini = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": "unknown_tool",
                        "response": {"ok": True},
                    }
                }],
            }],
        }
        out = request_transformer.transform(gemini)
        assert out["messages"] == [{
            "role": "tool",
            "tool_call_id": "call_unknown_tool",
            "content": '{"ok": true}',
        }]


# ============================================================================
# 响应转换矩阵（transformer.md §2.2 - §2.4）
# ============================================================================

class TestResponseTransformation:
    """响应体转换核心矩阵。"""

    def test_basic_text_response(self):
        openai = {
            "id": "chatcmpl-1",
            "choices": [{
                "message": {"role": "assistant", "content": "你好"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        out = response_transformer.transform(openai)
        assert out["candidates"][0]["content"]["role"] == "model"
        assert out["candidates"][0]["content"]["parts"][0]["text"] == "你好"
        assert out["candidates"][0]["finishReason"] == "STOP"
        assert out["usageMetadata"]["promptTokenCount"] == 10
        assert out["usageMetadata"]["candidatesTokenCount"] == 5
        assert out["usageMetadata"]["totalTokenCount"] == 15

    def test_finish_reason_length(self):
        openai = {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}]}
        out = response_transformer.transform(openai)
        assert out["candidates"][0]["finishReason"] == "MAX_TOKENS"

    def test_finish_reason_content_filter(self):
        openai = {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "content_filter"}]}
        out = response_transformer.transform(openai)
        assert out["candidates"][0]["finishReason"] == "SAFETY"

    def test_tool_calls_become_functionCall(self):
        openai = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"loc":"SF"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        out = response_transformer.transform(openai)
        parts = out["candidates"][0]["content"]["parts"]
        # 应有 1 个 functionCall 部分
        assert any("functionCall" in p for p in parts)
        fc = next(p["functionCall"] for p in parts if "functionCall" in p)
        assert fc["id"] == "call_abc"
        assert fc["name"] == "get_weather"
        assert fc["args"] == {"loc": "SF"}
        # finish_reason="tool_calls" → finishReason="STOP"
        assert out["candidates"][0]["finishReason"] == "STOP"

    def test_tool_calls_invalid_json_args(self):
        """arguments 不是合法 JSON 时降级为字符串。"""
        openai = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_abc",
                        "function": {"name": "f", "arguments": "not-json"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        out = response_transformer.transform(openai)
        fc = out["candidates"][0]["content"]["parts"][0]["functionCall"]
        assert fc["args"] == "not-json"

    def test_empty_choices(self):
        openai = {"choices": []}
        out = response_transformer.transform(openai)
        assert out["candidates"] == []
        assert "error" in out

    def test_missing_usage(self):
        openai = {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]}
        out = response_transformer.transform(openai)
        assert "usageMetadata" not in out

    def test_minimax_markup_recovered_to_function_call(self):
        openai = {
            "model": "MiniMax/MiniMax-M3",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "]<]minimax[>[<tool_call>]"
                        "<]minimax[>[<invoke name=\"run_shell_command\">]"
                        "<]minimax[>[<command>pwd]<]minimax[>[</command>]"
                        "<]minimax[>[</invoke>]"
                        "<]minimax[>[</tool_call>"
                    ),
                    "tool_calls": None,
                },
                "finish_reason": "tool_calls",
            }],
        }
        out = response_transformer.transform(openai)
        parts = out["candidates"][0]["content"]["parts"]
        assert all("text" not in p for p in parts)
        fc = next(p["functionCall"] for p in parts if "functionCall" in p)
        assert fc["name"] == "run_shell_command"
        assert fc["args"] == {"command": "pwd"}
