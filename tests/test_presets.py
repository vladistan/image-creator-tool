"""Tests for preset/platform loading and merge logic."""

from unittest.mock import patch

from image_creator_tool.presets import (
    _validate_platform,
    _validate_preset,
    load_platforms,
    load_presets,
)


def test_validate_preset_valid():
    entry = {"description": "Test", "prompt": "draw {subject} nicely"}
    assert _validate_preset("test", entry) is True


def test_validate_preset_missing_description():
    entry = {"prompt": "draw {subject}"}
    assert _validate_preset("test", entry) is False


def test_validate_preset_missing_prompt():
    entry = {"description": "Test"}
    assert _validate_preset("test", entry) is False


def test_validate_preset_missing_subject_placeholder():
    entry = {"description": "Test", "prompt": "draw something nicely"}
    assert _validate_preset("test", entry) is False


def test_validate_platform_valid():
    entry = {"description": "Test", "width": 1280, "height": 720}
    assert _validate_platform("test", entry) is True


def test_validate_platform_missing_description():
    entry = {"width": 1280, "height": 720}
    assert _validate_platform("test", entry) is False


def test_validate_platform_missing_width():
    entry = {"description": "Test", "height": 720}
    assert _validate_platform("test", entry) is False


def test_validate_platform_string_dimensions():
    entry = {"description": "Test", "width": "1280", "height": "720"}
    assert _validate_platform("test", entry) is False


def test_load_presets_bundled():
    presets = load_presets()
    assert len(presets) >= 8
    assert "editorial" in presets
    assert "blueprint" in presets


def test_load_presets_all_have_subject_placeholder():
    presets = load_presets()
    for name, entry in presets.items():
        assert "{subject}" in entry["prompt"], f"Preset '{name}' missing {{subject}}"


@patch("image_creator_tool.presets._load_user_yaml")
def test_load_presets_user_override_replaces_entry(mock_user):
    mock_user.return_value = {
        "editorial": {"description": "Custom", "prompt": "custom {subject}"}
    }
    presets = load_presets()
    assert presets["editorial"]["prompt"] == "custom {subject}"


@patch("image_creator_tool.presets._load_user_yaml")
def test_load_presets_user_adds_new_entry(mock_user):
    mock_user.return_value = {
        "my-style": {"description": "My style", "prompt": "{subject} my way"}
    }
    presets = load_presets()
    assert "my-style" in presets


@patch("image_creator_tool.presets._load_user_yaml")
def test_load_presets_invalid_user_entry_dropped(mock_user):
    mock_user.return_value = {
        "bad": {"description": "No placeholder", "prompt": "no subject here"}
    }
    presets = load_presets()
    assert "bad" not in presets


def test_load_platforms_bundled():
    platforms = load_platforms()
    assert len(platforms) >= 8
    assert "youtube" in platforms
    assert platforms["youtube"]["width"] == 1280


@patch("image_creator_tool.presets._load_user_yaml")
def test_load_platforms_user_override(mock_user):
    mock_user.return_value = {
        "custom": {"description": "Custom", "width": 500, "height": 500}
    }
    platforms = load_platforms()
    assert "custom" in platforms
