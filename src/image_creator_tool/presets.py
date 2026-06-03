"""Preset and platform data loading with user-override merge.

Bundled data (from package data/) serves as baseline. User overrides
at ~/.config/image-creator-tool/{presets,platforms}.yaml extend or replace
bundled entries via shallow merge (top-level key wins).
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

import yaml

_USER_CONFIG_DIR = Path.home() / ".config" / "image-creator-tool"


def _load_bundled_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML file from the bundled data/ directory."""
    data_files = importlib.resources.files("image_creator_tool.data")
    resource = data_files.joinpath(filename)
    content = resource.read_text(encoding="utf-8")
    return yaml.safe_load(content) or {}


def _load_user_yaml(filename: str) -> dict[str, Any]:
    """Load a user-override YAML file if it exists."""
    user_file = _USER_CONFIG_DIR / filename
    if not user_file.exists():
        return {}
    with user_file.open() as f:
        return yaml.safe_load(f) or {}


def _validate_preset(name: str, entry: dict[str, Any]) -> bool:
    """Validate a preset entry has required fields."""
    if "description" not in entry or "prompt" not in entry:
        return False
    return "{subject}" in entry["prompt"]


def _validate_platform(name: str, entry: dict[str, Any]) -> bool:
    """Validate a platform entry has required fields."""
    if "description" not in entry:
        return False
    return isinstance(entry.get("width"), int) and isinstance(entry.get("height"), int)


def load_presets() -> dict[str, Any]:
    """Load style presets with user-override merge and validation.

    Bundled presets from data/presets.yaml are extended/overridden by
    user presets at ~/.config/image-creator-tool/presets.yaml.
    Invalid entries are silently dropped.
    """
    bundled = _load_bundled_yaml("presets.yaml")
    user = _load_user_yaml("presets.yaml")
    merged = {**bundled, **user}
    return {k: v for k, v in merged.items() if _validate_preset(k, v)}


def load_platforms() -> dict[str, Any]:
    """Load platform presets with user-override merge and validation.

    Bundled platforms from data/platforms.yaml are extended/overridden by
    user platforms at ~/.config/image-creator-tool/platforms.yaml.
    Invalid entries are silently dropped.
    """
    bundled = _load_bundled_yaml("platforms.yaml")
    user = _load_user_yaml("platforms.yaml")
    merged = {**bundled, **user}
    return {k: v for k, v in merged.items() if _validate_platform(k, v)}
