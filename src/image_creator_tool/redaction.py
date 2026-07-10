"""Redact provider API key values from user-facing error output.

Provider secrets can surface in a raw exception string on the error path — an
upstream response body echoing the request, or a network error carrying the
request URL / Authorization header. This module masks every configured key value
(from the environment and from the TOML config profiles) before such a string
reaches the console.
"""

from __future__ import annotations

import os

from image_creator_tool.config import _load_profiles_raw

_REDACTED = "***REDACTED***"

# Env var names of the provider keys whose values must never reach the console.
_SECRET_ENV_VARS: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    "DEEPINFRA_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
)


def _env_secrets() -> list[str]:
    """Collect non-empty provider key values set in the environment."""
    return [value for name in _SECRET_ENV_VARS if (value := os.environ.get(name, ""))]


def _config_secrets() -> list[str]:
    """Collect ``api_key`` values from every ``[profile.*]`` section of the config.

    Redaction must never itself raise or leak config internals, so any failure to
    read the config is swallowed and yields no secrets (env redaction still applies).
    """
    try:
        profiles = _load_profiles_raw()
    except Exception:  # safety wrapper: never crash the error handler on a config-load failure
        return []
    return [
        key
        for profile_data in profiles.values()
        if isinstance(profile_data, dict)
        and isinstance((key := profile_data.get("api_key")), str)
        and key
    ]


def _redact(value: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret in value:
            value = value.replace(secret, _REDACTED)
    return value


def sanitize_error(message: str) -> str:
    """Return ``message`` with every configured provider key value masked.

    Secrets are sorted longest-first so an embedded key is replaced before any
    value it is a substring of, avoiding partial leaks.
    """
    secrets = sorted(
        set(_env_secrets()) | set(_config_secrets()), key=len, reverse=True
    )
    return _redact(message, secrets)
