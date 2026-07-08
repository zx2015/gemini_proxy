"""新功能单元测试：thoughtsTokenCount、thinkingConfig、circuit_breaker。"""
from __future__ import annotations

import asyncio
import pytest

from app.services.transformer.from_openai import ResponseTransformer
from app.services.transformer.fields import map_generation_config
from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState


# ============================================================================
# 1. thoughtsTokenCount mapping
# ============================================================================

class TestThoughtsTokenCount:
    def _make_resp(self, completion_tokens: int, reasoning_tokens: int, total_tokens: int):
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": completion_tokens,
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
                "total_tokens": total_tokens,
            },
            "model": "test-model",
        }

    def test_no_reasoning_tokens(self):
        resp = ResponseTransformer().transform(self._make_resp(20, 0, 30))
        usage = resp["usageMetadata"]
        assert usage["promptTokenCount"] == 10
        assert usage["candidatesTokenCount"] == 20
        assert usage["totalTokenCount"] == 30
        assert "thoughtsTokenCount" not in usage

    def test_with_reasoning_tokens(self):
        resp = ResponseTransformer().transform(self._make_resp(50, 30, 90))
        usage = resp["usageMetadata"]
        assert usage["promptTokenCount"] == 10
        assert usage["candidatesTokenCount"] == 20   # 50 - 30
        assert usage["thoughtsTokenCount"] == 30
        assert usage["totalTokenCount"] == 90

    def test_missing_details_key(self):
        """completion_tokens_details 缺失时不应崩溃。"""
        openai_resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "model": "m",
        }
        resp = ResponseTransformer().transform(openai_resp)
        usage = resp["usageMetadata"]
        assert usage["candidatesTokenCount"] == 10
        assert "thoughtsTokenCount" not in usage

    def test_none_reasoning_tokens(self):
        """reasoning_tokens 为 None 时视为 0。"""
        openai_resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": None},
                "total_tokens": 15,
            },
            "model": "m",
        }
        resp = ResponseTransformer().transform(openai_resp)
        usage = resp["usageMetadata"]
        assert usage["candidatesTokenCount"] == 10
        assert "thoughtsTokenCount" not in usage


# ============================================================================
# 2. thinkingConfig → extra_body.thinking
# ============================================================================

class TestThinkingConfig:
    def test_thinking_budget_enabled(self):
        gen_config = {"thinkingConfig": {"thinkingBudget": 8192}}
        out = map_generation_config(gen_config)
        assert out["extra_body"]["thinking"] == {"type": "enabled", "budget_tokens": 8192}

    def test_thinking_budget_disabled(self):
        gen_config = {"thinkingConfig": {"thinkingBudget": 0}}
        out = map_generation_config(gen_config)
        assert out["extra_body"]["thinking"] == {"type": "disabled"}

    def test_no_thinking_config(self):
        gen_config = {"temperature": 0.7}
        out = map_generation_config(gen_config)
        assert "extra_body" not in out

    def test_thinking_config_combined_with_other_fields(self):
        gen_config = {
            "temperature": 1.0,
            "maxOutputTokens": 4096,
            "thinkingConfig": {"thinkingBudget": 1024},
        }
        out = map_generation_config(gen_config)
        assert out["temperature"] == 1.0
        assert out["max_tokens"] == 4096
        assert out["extra_body"]["thinking"]["budget_tokens"] == 1024

    def test_thinking_config_no_budget(self):
        """thinkingConfig 存在但没有 thinkingBudget 字段时不写 extra_body。"""
        gen_config = {"thinkingConfig": {}}
        out = map_generation_config(gen_config)
        assert "extra_body" not in out


# ============================================================================
# 3. CircuitBreaker state machine
# ============================================================================

class TestCircuitBreaker:
    def _cb(self, threshold=3, recovery=60.0):
        return CircuitBreaker(failure_threshold=threshold, recovery_timeout=recovery)

    def test_initial_state_closed(self):
        cb = self._cb()
        assert cb.state == CircuitState.CLOSED

    def test_allows_calls_when_closed(self):
        cb = self._cb()
        asyncio.get_event_loop().run_until_complete(cb.before_call())  # should not raise

    def test_opens_after_threshold_failures(self):
        cb = self._cb(threshold=3)
        loop = asyncio.get_event_loop()
        for _ in range(3):
            loop.run_until_complete(cb.record_failure())
        assert cb.state == CircuitState.OPEN

    def test_rejects_when_open(self):
        cb = self._cb(threshold=1)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cb.record_failure())
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpen):
            loop.run_until_complete(cb.before_call())

    def test_success_resets_failure_count(self):
        cb = self._cb(threshold=5)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cb.record_failure())
        loop.run_until_complete(cb.record_failure())
        loop.run_until_complete(cb.record_success())
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_recovery(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cb.record_failure())
        assert cb.state == CircuitState.OPEN
        import time; time.sleep(0.05)
        # Next before_call should transition OPEN → HALF_OPEN
        loop.run_until_complete(cb.before_call())
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cb.record_failure())
        import time; time.sleep(0.05)
        loop.run_until_complete(cb.before_call())  # OPEN → HALF_OPEN
        loop.run_until_complete(cb.record_success())
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cb.record_failure())
        import time; time.sleep(0.05)
        loop.run_until_complete(cb.before_call())  # OPEN → HALF_OPEN
        loop.run_until_complete(cb.record_failure())
        assert cb.state == CircuitState.OPEN

    def test_half_open_rejects_second_probe(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=1)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cb.record_failure())
        import time; time.sleep(0.05)
        loop.run_until_complete(cb.before_call())  # first probe allowed
        with pytest.raises(CircuitBreakerOpen):
            loop.run_until_complete(cb.before_call())  # second probe rejected

    def test_status_dict(self):
        cb = self._cb(threshold=5, recovery=30.0)
        status = cb.status_dict()
        assert status["state"] == "CLOSED"
        assert status["failure_threshold"] == 5
        assert status["recovery_timeout"] == 30.0
        assert status["failure_count"] == 0
