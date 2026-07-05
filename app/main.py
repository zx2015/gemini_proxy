"""FastAPI 应用入口。

按 docs/design/architecture.md §5 落地。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api import gemini
from app.api.gemini import close_upstream_client
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
