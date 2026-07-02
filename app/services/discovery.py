"""模型发现服务：从上游 /v1/models 拉取并转换为 Gemini models[] 格式。

按 functional_requirements.md §2.4 落地。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings
from app.core.logging import logger


class ModelDiscoveryService:
    """缓存上游 OpenAI /v1/models，按需转换为 Gemini 格式。"""

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: List[Dict[str, Any]] = []
        self._cache_time: float = 0.0
        self._gemini_cache: List[Dict[str, Any]] = []

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.upstream_openai_url,
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def get_gemini_models(self) -> List[Dict[str, Any]]:
        """返回 Gemini `models[]` 格式的模型列表（带缓存）。"""
        now = time.time()
        ttl = settings.model_discovery_cache_ttl
        if self._gemini_cache and ttl > 0 and (now - self._cache_time) < ttl:
            return self._gemini_cache

        try:
            client = await self._ensure_client()
            resp = await client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {settings.upstream_api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            upstream_models = data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to fetch models from upstream: {e}")
            # 兜底：返回至少包含 UPSTREAM_MODEL 的列表
            upstream_models = [{"id": settings.upstream_model, "object": "model"}]

        self._cache = upstream_models
        self._gemini_cache = [self._to_gemini_model(m) for m in upstream_models]
        self._cache_time = now
        return self._gemini_cache

    @staticmethod
    def _to_gemini_model(openai_model: Dict[str, Any]) -> Dict[str, Any]:
        """将 OpenAI 单个 model 对象转 Gemini 格式。"""
        model_id = openai_model.get("id", "")
        return {
            "name": f"models/{model_id}" if not model_id.startswith("models/") else model_id,
            "baseModelId": model_id,
            "version": "001",
            "displayName": model_id,
            "description": f"Proxied to {model_id}",
            "supportedGenerationMethods": [
                "generateContent",
                "streamGenerateContent",
                "countTokens",
            ],
        }


# 全局单例
model_discovery = ModelDiscoveryService()
