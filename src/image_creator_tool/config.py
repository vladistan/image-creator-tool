"""Configuration management for image-creator-tool.

Uses pydantic-settings with TOML file support and profile overlays.
Profiles are [profile.*] sections in config.toml that override base settings.

Precedence: env vars > active profile > base config > defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_DIR = Path.home() / ".config" / "image-creator-tool"
CONFIG_FILE = CONFIG_DIR / "config.toml"
DATA_DIR = Path(__file__).resolve().parent / "data"


class ImageCreatorSettings(BaseSettings):
    """Application settings with env var and TOML file support."""

    model_config = SettingsConfigDict(
        env_prefix="IMAGE_CREATOR_",
        extra="ignore",
    )

    default_provider: str = "gemini"
    default_model: str = ""
    default_platform: str = ""
    default_project: str = ""
    default_preset: str = ""
    default_profile: str = ""
    output_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "image-creator-tool" / "outputs"
    )
    sentry_dsn: str = ""
    gcp_project: str = ""
    gcp_region: str = "us-central1"
    provider_preference: list[str] = [
        "vertex", "deepinfra", "openrouter", "openai", "bedrock", "gemini"
    ]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Configure settings sources: env > TOML > defaults."""
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if CONFIG_FILE.exists():
            sources.append(
                TomlConfigSettingsSource(settings_cls, toml_file=CONFIG_FILE)
            )
        sources.append(file_secret_settings)
        return tuple(sources)


def load_settings() -> ImageCreatorSettings:
    """Load settings from TOML config and environment variables."""
    return ImageCreatorSettings()


def _load_profiles_raw() -> dict[str, dict[str, Any]]:
    """Load raw profile sections from config.toml.

    Returns dict mapping profile name -> profile settings dict.
    """
    if not CONFIG_FILE.exists():
        return {}
    with CONFIG_FILE.open("rb") as f:
        data = tomllib.load(f)
    profiles = data.get("profile", {})
    if not isinstance(profiles, dict):
        return {}
    return profiles


def list_profiles() -> list[str]:
    """Return sorted list of available profile names."""
    return sorted(_load_profiles_raw().keys())


def build_provider_kwargs(prov_name: str, config: dict[str, Any]) -> dict[str, str]:
    """Build provider constructor kwargs for `prov_name` from config + matching profile.

    Overlays settings from the first [profile.*] section whose `provider` matches
    `prov_name` onto a copy of `config`, then extracts the provider-specific
    constructor arguments (GCP project/region, AWS profile/region, api_key).
    """
    merged = config.copy()
    for pdata in _load_profiles_raw().values():
        if pdata.get("provider") == prov_name:
            for k, v in pdata.items():
                if v and k != "provider":
                    merged[k] = str(v) if not isinstance(v, str) else v
            break

    kwargs: dict[str, str] = {}
    if prov_name == "vertex":
        if "gcp_project" in merged:
            kwargs["project"] = merged["gcp_project"]
        if "gcp_region" in merged:
            kwargs["region"] = merged["gcp_region"]
    elif prov_name == "bedrock":
        if "aws_profile" in merged:
            kwargs["aws_profile"] = merged["aws_profile"]
        if "aws_region" in merged:
            kwargs["aws_region"] = merged["aws_region"]
    if "api_key" in merged:
        kwargs["api_key"] = merged["api_key"]
    return kwargs


def get_config_dict(profile_name: str | None = None) -> dict[str, Any]:
    """Load settings with optional profile overlay applied.

    Resolution order:
    1. Base settings (env vars > TOML top-level > defaults)
    2. Profile overlay (if profile_name specified or default_profile set)

    Empty string values are excluded so generation.py's `or` fallback logic works.
    """
    settings = load_settings()

    # Determine active profile
    active_profile = profile_name or settings.default_profile

    # Base config
    raw: dict[str, Any] = {
        "default_provider": settings.default_provider,
        "default_model": settings.default_model,
        "default_platform": settings.default_platform,
        "default_project": settings.default_project,
        "default_preset": settings.default_preset,
        "output_dir": str(settings.output_dir),
        "gcp_project": settings.gcp_project,
        "gcp_region": settings.gcp_region,
    }

    # Apply profile overlay
    # Profile keys can use short names (provider, model) which map to full keys
    _profile_key_map = {
        "provider": "default_provider",
        "model": "default_model",
        "platform": "default_platform",
        "project": "default_project",
        "preset": "default_preset",
    }
    if active_profile:
        profiles = _load_profiles_raw()
        if active_profile in profiles:
            profile_data = profiles[active_profile]
            for key, value in profile_data.items():
                if not value:
                    continue
                # Map short profile keys to full config keys
                mapped_key = _profile_key_map.get(key, key)
                raw[mapped_key] = str(value) if not isinstance(value, str) else value

    # Filter empty values
    return {k: v for k, v in raw.items() if v}
