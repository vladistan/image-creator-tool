"""Observability setup for image-creator-tool.

Configures structured logging (structlog) and error tracking (Sentry).
Both functions are idempotent and safe to call multiple times.
"""

from __future__ import annotations

import logging
import os

import sentry_sdk
import structlog

from image_creator_tool import __version__
from image_creator_tool.config import load_settings

_logging_configured = False
_sentry_configured = False


def setup_logging() -> None:
    """Configure structlog with console renderer for stderr output.

    Idempotent — subsequent calls are no-ops.
    """
    global _logging_configured  # noqa: PLW0603
    if _logging_configured:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=None),
        cache_logger_on_first_use=True,
    )
    _logging_configured = True


def setup_sentry(environment: str = "production") -> None:
    """Initialize Sentry SDK if DSN is configured. No-op otherwise.

    Resolution order: SENTRY_DSN env > IMAGE_CREATOR_SENTRY_DSN env > config.toml sentry_dsn.
    Idempotent — subsequent calls are no-ops.
    """
    global _sentry_configured  # noqa: PLW0603
    if _sentry_configured:
        return

    dsn = os.environ.get("SENTRY_DSN", "") or os.environ.get("IMAGE_CREATOR_SENTRY_DSN", "")
    if not dsn:
        dsn = load_settings().sentry_dsn
    if not dsn:
        return

    sentry_sdk.init(
        dsn=dsn,
        release=__version__,
        traces_sample_rate=1.0,
        send_default_pii=False,
        attach_stacktrace=True,
        environment=environment,
    )
    _sentry_configured = True
