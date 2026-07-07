"""统一日志配置。

设计原则：
- 单次 setup_logging() 调用初始化根 logger 与 uvicorn 接入。
- 默认关闭 /health 访问日志（防止 K8s/Docker 探针刷屏）。
- 使用 ISO 8601 时间戳便于跨时区排查。
"""
from __future__ import annotations

import logging
import sys
from logging.config import dictConfig

from app.core.config import settings


_HEALTH_PATH = "/health"


class HealthCheckFilter(logging.Filter):
    """过滤掉 /health 端点的访问日志。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return _HEALTH_PATH not in record.getMessage()
        except Exception:
            return True


def setup_logging() -> None:
    """根据 settings.log_level 初始化日志。"""
    log_level = settings.log_level

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": (
                    "%(asctime)s.%(msecs)03d %(levelname)-7s "
                    "[%(name)s] %(message)s"
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "access": {
                "format": (
                    "%(asctime)s.%(msecs)03d %(levelname)-7s "
                    "[%(name)s] %(message)s"
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
                "level": log_level,
            },
            "access": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "access",
                "level": log_level,
            },
        },
        "loggers": {
            "": {  # root
                "handlers": ["stdout"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["stdout"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "WARNING",  # 访问日志默认偏静默
                "propagate": False,
            },
        },
    }

    dictConfig(config)

    # 安装 /health 过滤器
    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    # 初始化详细调试日志（隔离写入物理文件）
    if settings.debug_log_enabled:
        import os
        from logging.handlers import RotatingFileHandler
        
        log_path = settings.debug_log_path
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        debug_logger = logging.getLogger("gemini_debug")
        debug_logger.setLevel(logging.DEBUG)
        
        # 避免在多次 setup_logging 调用时重复挂载 Handler
        if not debug_logger.handlers:
            # 20MB * 3 文件轮转
            handler = RotatingFileHandler(
                log_path,
                maxBytes=20 * 1024 * 1024,
                backupCount=2,
                encoding="utf-8"
            )
            formatter = logging.Formatter(
                fmt="%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S"
            )
            handler.setFormatter(formatter)
            debug_logger.addHandler(handler)
            debug_logger.propagate = False


# 业务代码统一通过此 logger 输出
logger = logging.getLogger("gemini_proxy")
logger_debug = logging.getLogger("gemini_debug")
