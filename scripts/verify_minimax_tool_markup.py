#!/usr/bin/env python3
"""验证 MiniMax 上游是否把工具调用泄漏到 content 文本里。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
import os


MINIMAX_MARKUP_RE = re.compile(
    r"\]<\]minimax\[>\[|<tool_call>|</tool_call>|<invoke\s+name=",
    re.IGNORECASE,
)


@dataclass
class ProbeResult:
    ok: bool
    status_code: int
    has_tool_calls_field: bool
    has_minimax_markup_in_content: bool
    finish_reason: Optional[str]
    content_preview: str
    raw_response: Dict[str, Any]


@dataclass
class StreamProbeResult:
    ok: bool
    status_code: int
    finish_reason: Optional[str]
    has_tool_calls_delta: bool
    has_minimax_markup_in_delta_content: bool
    collected_delta_content: str


def _build_chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    path = urlparse(base_url).path.rstrip("/")
    if path.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _make_payload(model: str, tool_choice: Any) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "请调用 run_shell_command 工具执行命令 `echo minimax_probe`。"
                    "不要直接回答，必须通过工具调用完成。"
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "description": "Execute shell command and return output.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute.",
                            }
                        },
                        "required": ["command"],
                    },
                },
            }
        ],
        "tool_choice": tool_choice,
        "temperature": 0,
        "stream": False,
    }


def _extract_result(resp_json: Dict[str, Any], status_code: int) -> ProbeResult:
    choices = resp_json.get("choices") or []
    msg = choices[0].get("message", {}) if choices else {}
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")

    if isinstance(content, list):
        content_text = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        content_text = content or ""

    has_markup = bool(MINIMAX_MARKUP_RE.search(content_text))
    has_tool_calls_field = bool(tool_calls)
    finish_reason = choices[0].get("finish_reason") if choices else None

    return ProbeResult(
        ok=True,
        status_code=status_code,
        has_tool_calls_field=has_tool_calls_field,
        has_minimax_markup_in_content=has_markup,
        finish_reason=finish_reason,
        content_preview=content_text[:500],
        raw_response=resp_json,
    )


def run_probe(timeout: float) -> ProbeResult:
    load_dotenv()
    upstream_url = os.getenv("UPSTREAM_OPENAI_URL", "").strip()
    upstream_key = os.getenv("UPSTREAM_API_KEY", "").strip()
    upstream_model = os.getenv("UPSTREAM_MODEL", "").strip()

    missing = [
        k
        for k, v in [
            ("UPSTREAM_OPENAI_URL", upstream_url),
            ("UPSTREAM_API_KEY", upstream_key),
            ("UPSTREAM_MODEL", upstream_model),
        ]
        if not v
    ]
    if missing:
        raise RuntimeError(f".env 缺少必要变量: {', '.join(missing)}")

    url = _build_chat_completions_url(upstream_url)
    headers = {
        "Authorization": f"Bearer {upstream_key}",
        "Content-Type": "application/json",
    }

    payloads: List[Dict[str, Any]] = [
        _make_payload(upstream_model, "required"),
        _make_payload(
            upstream_model,
            {"type": "function", "function": {"name": "run_shell_command"}},
        ),
    ]

    last_error: Optional[str] = None
    with httpx.Client(timeout=timeout) as client:
        for payload in payloads:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                last_error = f"{resp.status_code}: {resp.text[:500]}"
                continue
            data = resp.json()
            return _extract_result(data, resp.status_code)

    raise RuntimeError(f"上游请求失败: {last_error or 'unknown error'}")


def run_stream_probe(timeout: float) -> StreamProbeResult:
    load_dotenv()
    upstream_url = os.getenv("UPSTREAM_OPENAI_URL", "").strip()
    upstream_key = os.getenv("UPSTREAM_API_KEY", "").strip()
    upstream_model = os.getenv("UPSTREAM_MODEL", "").strip()
    url = _build_chat_completions_url(upstream_url)
    headers = {
        "Authorization": f"Bearer {upstream_key}",
        "Content-Type": "application/json",
    }
    payload = _make_payload(upstream_model, "required")
    payload["stream"] = True

    collected: List[str] = []
    has_tool_calls_delta = False
    finish_reason = None

    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"流式上游请求失败: {resp.status_code}: {resp.text[:500]}")
            for raw in resp.iter_lines():
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                if delta.get("tool_calls"):
                    has_tool_calls_delta = True
                text = delta.get("content")
                if text:
                    collected.append(text)
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]

    content_text = "".join(collected)
    return StreamProbeResult(
        ok=True,
        status_code=200,
        finish_reason=finish_reason,
        has_tool_calls_delta=has_tool_calls_delta,
        has_minimax_markup_in_delta_content=bool(MINIMAX_MARKUP_RE.search(content_text)),
        collected_delta_content=content_text[:500],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe MiniMax tool-call behavior via upstream OpenAI-compatible API."
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print raw JSON response (may be long).",
    )
    args = parser.parse_args()

    try:
        result = run_probe(timeout=args.timeout)
        stream_result = run_stream_probe(timeout=args.timeout)
    except Exception as exc:  # noqa: BLE001 - probe script needs full error context
        print(f"[probe] failed: {exc}", file=sys.stderr)
        return 2

    print(f"[probe] status_code={result.status_code}")
    print(f"[probe] finish_reason={result.finish_reason}")
    print(f"[probe] has_tool_calls_field={result.has_tool_calls_field}")
    print(f"[probe] has_minimax_markup_in_content={result.has_minimax_markup_in_content}")
    print("[probe] content_preview:")
    print(result.content_preview or "<empty>")
    print("\n[stream_probe] status_code=200")
    print(f"[stream_probe] finish_reason={stream_result.finish_reason}")
    print(f"[stream_probe] has_tool_calls_delta={stream_result.has_tool_calls_delta}")
    print(
        "[stream_probe] has_minimax_markup_in_delta_content="
        f"{stream_result.has_minimax_markup_in_delta_content}"
    )
    print("[stream_probe] content_preview:")
    print(stream_result.collected_delta_content or "<empty>")

    if args.print_raw:
        print("\n[probe] raw_response:")
        print(json.dumps(result.raw_response, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
