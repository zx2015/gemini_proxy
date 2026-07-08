"""FastAPI 应用入口。

按 docs/design/architecture.md §5 落地。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api import gemini
from app.api.gemini import close_upstream_client
from app.core.circuit_breaker import get_circuit_breaker
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.services.discovery import model_discovery


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动/关闭 model_discovery 客户端。"""
    logger.info(
        f"gemini_proxy starting: upstream={settings.upstream_openai_url} "
        f"model={settings.upstream_model} listen={settings.host}:{settings.port}"
    )
    # 预初始化熔断器（避免第一个请求时延迟初始化竞态）
    get_circuit_breaker()
    try:
        yield
    finally:
        await model_discovery.close()
        await close_upstream_client()
        logger.info("gemini_proxy shutdown complete.")


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(
        title="Gemini Proxy",
        description="A protocol adapter between Gemini SDK and OpenAI-Compatible upstream.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check() -> dict:
        return {"status": "healthy"}

    @app.get("/health/upstream")
    async def health_upstream() -> JSONResponse:
        """探活上游 LiteLLM：发一个 GET /v1/models 请求，返回连通状态与熔断器信息。"""
        cb = get_circuit_breaker()
        cb_status = cb.status_dict()
        upstream_url = f"{settings.upstream_openai_url.rstrip('/')}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    upstream_url,
                    headers={"Authorization": f"Bearer {settings.upstream_api_key}"},
                )
            if resp.status_code < 400:
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "ok",
                        "upstream": settings.upstream_openai_url,
                        "upstream_http_status": resp.status_code,
                        "circuit_breaker": cb_status,
                    },
                )
            return JSONResponse(
                status_code=502,
                content={
                    "status": "error",
                    "upstream": settings.upstream_openai_url,
                    "upstream_http_status": resp.status_code,
                    "circuit_breaker": cb_status,
                },
            )
        except Exception as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "status": "error",
                    "upstream": settings.upstream_openai_url,
                    "detail": str(exc),
                    "circuit_breaker": cb_status,
                },
            )

    @app.get("/")
    async def root() -> dict:
        return {
            "message": "gemini_proxy is running",
            "upstream": settings.upstream_openai_url,
            "upstream_model": settings.upstream_model,
        }

    app.include_router(gemini.router)
    return app


app = create_app()


def main() -> None:
    """CLI 入口：python -m app.main"""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,  # 生产环境关 reload，省 CPU
        access_log=True,
    )


if __name__ == "__main__":
    main()



def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(
        title="Gemini Proxy",
        description="A protocol adapter between Gemini SDK and OpenAI-Compatible upstream.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check() -> dict:
        return {"status": "healthy"}

    @app.get("/")
    async def root() -> dict:
        return {
            "message": "gemini_proxy is running",
            "upstream": settings.upstream_openai_url,
            "upstream_model": settings.upstream_model,
        }

    app.include_router(gemini.router)
    return app


app = create_app()


def main() -> None:
    """CLI 入口：python -m app.main"""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,  # 生产环境关 reload，省 CPU
        access_log=True,
    )


if __name__ == "__main__":
    main()
