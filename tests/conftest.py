"""Shared test fixtures for image-creator-tool."""


import pytest
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Prevent tests from reading user's real config file."""
    fake_config = tmp_path / "config.toml"
    monkeypatch.setattr("image_creator_tool.config.CONFIG_FILE", fake_config)


@pytest.fixture
def runner() -> CliRunner:
    """Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_presets(tmp_path):
    """Create a sample presets YAML file and return its path."""
    content = """
test-preset:
  description: "Test preset for unit tests"
  prompt: "test style {subject} with flair"
invalid-preset:
  description: "Missing subject placeholder"
  prompt: "no placeholder here"
"""
    path = tmp_path / "presets.yaml"
    path.write_text(content)
    return path


@pytest.fixture
def sample_platforms(tmp_path):
    """Create a sample platforms YAML file and return its path."""
    content = """
test-platform:
  description: "Test platform"
  width: 800
  height: 600
invalid-platform:
  description: "Missing dimensions"
"""
    path = tmp_path / "platforms.yaml"
    path.write_text(content)
    return path
