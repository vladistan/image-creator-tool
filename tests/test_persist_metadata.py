"""Regression tests for per-image persistence metadata (sidecar + provenance).

Guards that the plain .json sidecar records the provider and that the raw
subject is threaded into the PROV record alongside the composed prompt.
"""

import json
from types import SimpleNamespace

from image_creator_tool import generation_core
from image_creator_tool.generation_core import GenerationResult
from image_creator_tool.provenance import load_record, sidecar_path_for


def _result(path):
    return GenerationResult(
        output_path=path,
        prompt="photograph of a red barn, heavy film grain, cinematic",
        model="krea/Krea-2-Turbo",
        preset="grain",
        platform="slides",
        edit_source=None,
        timestamp="2026-07-08T12:00:00",
        duration_s=1.5,
    )


def _args():
    return SimpleNamespace(prompt="a red barn", no_metadata=False)


def test_sidecar_records_provider(tmp_path):
    image = tmp_path / "barn.png"
    image.write_bytes(b"data")
    provider = SimpleNamespace(name="huggingface")

    generation_core._persist_generation(
        [_result(image)], _args(), provider, None, None, None, None, None
    )

    sidecar = json.loads((tmp_path / "barn.json").read_text())
    assert sidecar["provider"] == "huggingface"


def test_provenance_records_subject_and_composed_prompt(tmp_path):
    image = tmp_path / "barn.png"
    image.write_bytes(b"data")
    provider = SimpleNamespace(name="huggingface")

    generation_core._persist_generation(
        [_result(image)], _args(), provider, None, None, None, None, None
    )

    record = load_record(sidecar_path_for(image))
    assert record.subject == "a red barn"
    assert record.prompt == "photograph of a red barn, heavy film grain, cinematic"
    assert record.provider == "huggingface"
