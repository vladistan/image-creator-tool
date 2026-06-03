"""Tests for generation utilities: slugify, compose_prompt, resolve_output_path."""

import re
from pathlib import Path

import pytest

from image_creator_tool.errors import PermanentAPIError
from image_creator_tool.generation import compose_prompt
from image_creator_tool.history import resolve_output_path, slugify


def test_slugify_basic_text():
    assert slugify("Hello World") == "hello-world"


def test_slugify_special_characters():
    assert slugify("a robot & a cat!") == "a-robot-a-cat"


def test_slugify_max_length():
    result = slugify("a" * 100, max_len=40)
    assert len(result) <= 40


def test_slugify_empty_string():
    assert slugify("") == "image"


def test_slugify_only_special_chars():
    assert slugify("!@#$%") == "image"


def test_compose_prompt_no_preset_returns_subject():
    assert compose_prompt("a robot", None, {}) == "a robot"


def test_compose_prompt_preset_applies_template():
    presets = {"editorial": {"prompt": "styled {subject} art"}}
    result = compose_prompt("a cat", "editorial", presets)
    assert result == "styled a cat art"


def test_compose_prompt_unknown_preset_exits():
    with pytest.raises(PermanentAPIError):
        compose_prompt("a cat", "nonexistent", {})


def test_compose_prompt_subject_placeholder_replaced():
    presets = {"test": {"prompt": "{subject} in space"}}
    result = compose_prompt("astronaut", "test", presets)
    assert "astronaut" in result
    assert "{subject}" not in result


def test_resolve_output_path_explicit(tmp_path):
    explicit = str(tmp_path / "out.png")
    result = resolve_output_path(explicit, "subject", None, {})
    assert result == Path(explicit).resolve()


def test_resolve_output_path_auto_generated(tmp_path):
    config = {"output_dir": str(tmp_path)}
    result = resolve_output_path(None, "hello world", None, config)
    assert result.parent == tmp_path
    assert "hello-world" in result.name
    assert result.suffix == ".png"


def test_resolve_output_path_project_subdirectory(tmp_path):
    config = {"output_dir": str(tmp_path)}
    result = resolve_output_path(None, "test", "myproject", config)
    assert "myproject" in str(result.parent)


def test_resolve_output_path_timestamp_in_filename(tmp_path):
    config = {"output_dir": str(tmp_path)}
    result = resolve_output_path(None, "test", None, config)
    # Should have YYYYMMDD-HHMMSS pattern
    assert re.search(r"\d{8}-\d{6}", result.name)
