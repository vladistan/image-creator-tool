"""Regression tests for output_dir tilde expansion in settings.

A config/env override like `output_dir = "~/.local/share/..."` must be expanded
at load time; otherwise index and provenance scans hit a literal `~/...` path
that does not exist and silently find nothing.
"""

from pathlib import Path

from image_creator_tool.config import ImageCreatorSettings, reference_images_dir


def test_output_dir_tilde_is_expanded():
    settings = ImageCreatorSettings(output_dir=Path("~/foo/bar"))
    assert "~" not in str(settings.output_dir)
    assert settings.output_dir == Path.home() / "foo" / "bar"


def test_output_dir_absolute_path_unchanged(tmp_path):
    settings = ImageCreatorSettings(output_dir=tmp_path)
    assert settings.output_dir == tmp_path


def test_default_output_dir_is_absolute():
    settings = ImageCreatorSettings()
    assert settings.output_dir.is_absolute()
    assert "~" not in str(settings.output_dir)


def test_ref_images_dir_defaults_under_output_dir(tmp_path):
    settings = ImageCreatorSettings(output_dir=tmp_path)
    assert settings.ref_images_dir == tmp_path / "ref-images"
    assert reference_images_dir(settings) == tmp_path / "ref-images"


def test_ref_images_dir_explicit_override_is_kept(tmp_path):
    custom = tmp_path / "elsewhere" / "refs"
    settings = ImageCreatorSettings(output_dir=tmp_path, ref_images_dir=custom)
    assert settings.ref_images_dir == custom
    assert reference_images_dir(settings) == custom


def test_ref_images_dir_tilde_is_expanded():
    settings = ImageCreatorSettings(ref_images_dir=Path("~/refs"))
    assert "~" not in str(settings.ref_images_dir)
