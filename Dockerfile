# =====================================================================
# gemini_proxy — Dockerfile
#
# 修订记录（内容递增）：
# - v0.1.0: 基础镜像 (python:3.12-slim + curl)
# - v0.1.0-r1: 叠加多阶段构建、tini 信号处理、非 root 用户、显式 EXPOSE、构建参数
# =====================================================================

# ---- 语法版本 ----
# syntax=docker/dockerfile:1.7

# =====================================================================
# Stage 1: builder —— 解析依赖，产出可复用的 wheels
# =====================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# 仅安装构建期工具，不进入最终镜像
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements 利用 Docker 缓存
COPY requirements.txt .

# 产出 wheels 到 /wheels/
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# =====================================================================
# Stage 2: runtime —— 最小化生产镜像
# =====================================================================
FROM python:3.12-slim AS runtime

# ---- 构建参数：可用 docker build --build-arg 覆盖 ----
ARG PYTHON_VERSION=3.12
ARG APP_USER=app
ARG APP_UID=1000
ARG APP_GID=1000

# ---- 系统层依赖（仅运行时需要） ----
# tini: 正确处理 PID 1 信号（SIGTERM/SIGINT），确保 graceful shutdown
# curl: HEALTHCHECK 探针
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ---- 创建非 root 用户 ----
RUN groupadd --system --gid ${APP_GID} ${APP_USER} \
    && useradd  --system --uid ${APP_UID} --gid ${APP_GID} \
                --no-create-home --shell /usr/sbin/nologin ${APP_USER}

# ---- 安装 Python 依赖（仅从 builder 阶段复制 wheels，不重新下载） ----
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels /root/.cache

# ---- 工作目录与代码 ----
WORKDIR /app
COPY --chown=${APP_USER}:${APP_USER} app/ ./app/

# ---- 切换到非 root 用户 ----
USER ${APP_USER}

# ---- 暴露端口（文档化；运行时仍由 -p 或 compose ports 控制） ----
EXPOSE 8000

# ---- 环境变量默认值（可在 docker run -e 或 .env 覆盖） ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO

# ---- 健康检查（被 docker-compose 与 k8s 共用） ----
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# ---- 入口点：tini 作为 PID 1 处理信号 ----
ENTRYPOINT ["/usr/bin/tini", "--"]

# ---- 默认命令 ----
CMD ["python", "-m", "app.main"]
