"""Tests for the provenance domain layer (Phase 4: PROV Provenance Tracking).

Covers the ProvenanceRecord schema and PROV serialization (4.1), the sidecar
writer (4.2), and best-effort EXIF/tEXt metadata embedding (4.3).
"""

import json

import piexif
import pytest
from PIL import Image
from prov.model import ProvDocument

from image_creator_tool import provenance
from image_creator_tool.errors import PermanentAPIError
from image_creator_tool.provenance import (
    ProvenanceRecord,
    embed_exif_metadata,
    load_record,
    sidecar_path_for,
    write_provenance_sidecar,
)


def _record(**overrides) -> ProvenanceRecord:
    base = {
        "prompt": "a red barn at sunset",
        "model": "gemini-3.1-flash-image-preview",
        "provider": "gemini",
        "output_path": "/tmp/out/barn.png",
        "timestamp": "2026-07-07T12:00:00",
        "seed": 42,
        "parameters": {"size": "1024x1024", "aspect": "1:1"},
    }
    base.update(overrides)
    return ProvenanceRecord(**base)


# --- Step 4.1: Provenance record schema + serialization ----------------------


def test_record_captures_required_fields():
    rec = _record()
    assert rec.prompt == "a red barn at sunset"
    assert rec.model == "gemini-3.1-flash-image-preview"
    assert rec.provider == "gemini"
    assert rec.seed == 42
    assert rec.parameters == {"size": "1024x1024", "aspect": "1:1"}
    assert rec.output_path == "/tmp/out/barn.png"
    assert rec.timestamp == "2026-07-07T12:00:00"


def test_subject_defaults_empty_and_is_omitted_from_prov():
    rec = _record()
    assert rec.subject == ""
    raw = json.loads(rec.to_prov_json())
    entity_attrs = next(iter(raw["entity"].values()))
    assert "imgc:subject" not in entity_attrs


def test_origin_defaults_generated_and_is_omitted_from_prov():
    rec = _record()
    assert rec.origin == "generated"
    assert rec.sources == []
    raw = json.loads(rec.to_prov_json())
    entity_attrs = next(iter(raw["entity"].values()))
    assert "imgc:origin" not in entity_attrs
    assert "imgc:sources" not in entity_attrs


def test_imported_origin_and_sources_round_trip(tmp_path):
    rec = _record(origin="imported", sources=["https://ex/cat.png", "/local/cat.png"])
    image = tmp_path / "cat.png"
    image.write_bytes(b"data")
    sidecar = write_provenance_sidecar(rec, image, overwrite=True)
    loaded = load_record(sidecar)
    assert loaded.origin == "imported"
    assert loaded.sources == ["https://ex/cat.png", "/local/cat.png"]


def test_subject_recorded_and_round_trips(tmp_path):
    rec = _record(
        subject="a red barn",
        prompt="photograph of a red barn, heavy film grain, cinematic",
    )
    raw = json.loads(rec.to_prov_json())
    entity_attrs = next(iter(raw["entity"].values()))
    assert entity_attrs["imgc:subject"] == "a red barn"
    assert entity_attrs["imgc:prompt"] == "photograph of a red barn, heavy film grain, cinematic"

    image = tmp_path / "barn.png"
    image.write_bytes(b"data")
    sidecar = write_provenance_sidecar(rec, image, overwrite=True)
    loaded = load_record(sidecar)
    assert loaded.subject == "a red barn"
    assert loaded.prompt == "photograph of a red barn, heavy film grain, cinematic"


def test_to_prov_json_is_valid_prov_json():
    rec = _record()
    doc_json = rec.to_prov_json()
    # Round-trips through the prov library => structurally valid PROV-JSON.
    doc = ProvDocument.deserialize(content=doc_json, format="json")
    records = list(doc.get_records())
    assert records  # entity + activity + agent + relations

    raw = json.loads(doc_json)
    entity_attrs = next(iter(raw["entity"].values()))
    assert entity_attrs["imgc:prompt"] == "a red barn at sunset"
    assert entity_attrs["imgc:provider"] == "gemini"


def test_to_prov_n_is_nonempty_notation():
    rec = _record()
    provn = rec.to_prov_n()
    assert provn.startswith("document")
    assert "imgc" in provn
    assert "gemini" in provn


def test_identical_inputs_different_timestamps_are_distinct_entities():
    a = _record(timestamp="2026-07-07T12:00:00")
    b = _record(timestamp="2026-07-07T12:05:00")
    assert a.entity_id != b.entity_id


def test_seed_none_omitted_from_record():
    rec = _record(seed=None)
    raw = json.loads(rec.to_prov_json())
    entity_attrs = next(iter(raw["entity"].values()))
    assert "imgc:seed" not in entity_attrs


# --- Step 4.2: Sidecar file writer -------------------------------------------


def test_sidecar_path_is_prov_json_next_to_image():
    assert sidecar_path_for("/tmp/out/barn.png").name == "barn.prov.json"
    assert sidecar_path_for("/tmp/out/barn.jpg").name == "barn.prov.json"


def test_write_sidecar_creates_valid_prov_json(tmp_path):
    image = tmp_path / "barn.png"
    image.write_bytes(b"not-a-real-image")
    rec = _record(output_path=str(image))

    sidecar = write_provenance_sidecar(rec, image)

    assert sidecar == tmp_path / "barn.prov.json"
    assert sidecar.is_file()
    ProvDocument.deserialize(content=sidecar.read_text(), format="json")  # parses


def test_write_sidecar_refuses_overwrite_by_default(tmp_path):
    image = tmp_path / "barn.png"
    rec = _record(output_path=str(image))
    write_provenance_sidecar(rec, image)

    with pytest.raises(PermanentAPIError):
        write_provenance_sidecar(rec, image)


def test_write_sidecar_overwrite_flag_replaces(tmp_path):
    image = tmp_path / "barn.png"
    rec = _record(output_path=str(image))
    write_provenance_sidecar(rec, image)
    # Overwrite with a different prompt and confirm the new content lands.
    rec2 = _record(output_path=str(image), prompt="a blue lake")
    sidecar = write_provenance_sidecar(rec2, image, overwrite=True)
    assert load_record(sidecar).prompt == "a blue lake"


def test_load_record_round_trips(tmp_path):
    image = tmp_path / "barn.png"
    rec = _record(output_path=str(image))
    sidecar = write_provenance_sidecar(rec, image)

    loaded = load_record(sidecar)
    assert loaded.prompt == rec.prompt
    assert loaded.model == rec.model
    assert loaded.provider == rec.provider
    assert loaded.seed == 42
    assert loaded.parameters == {"size": "1024x1024", "aspect": "1:1"}


def test_load_record_missing_sidecar_raises(tmp_path):
    with pytest.raises(PermanentAPIError):
        load_record(tmp_path / "nope.prov.json")


# --- Step 4.3: EXIF / tEXt metadata embedding --------------------------------


def test_embed_png_writes_text_chunks(tmp_path):
    image = tmp_path / "barn.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
    rec = _record(output_path=str(image))

    assert embed_exif_metadata(image, rec) is True

    reopened = Image.open(image)
    assert reopened.text["Description"] == rec.prompt
    assert reopened.text["Software"] == "gemini/gemini-3.1-flash-image-preview"


def test_embed_jpeg_writes_exif(tmp_path):
    image = tmp_path / "barn.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image, "JPEG")
    rec = _record(output_path=str(image))

    assert embed_exif_metadata(image, rec) is True

    exif = piexif.load(str(image))
    assert exif["0th"][piexif.ImageIFD.ImageDescription].decode() == rec.prompt
    assert exif["0th"][piexif.ImageIFD.Software].decode() == "gemini/gemini-3.1-flash-image-preview"


def test_embed_jpeg_graceful_noop_when_piexif_missing(tmp_path, monkeypatch):
    image = tmp_path / "barn.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image, "JPEG")
    rec = _record(output_path=str(image))
    monkeypatch.setattr(provenance, "_load_piexif", lambda: None)

    # No exception, returns False (embedding skipped).
    assert embed_exif_metadata(image, rec) is False


def test_embed_unsupported_format_is_noop(tmp_path):
    image = tmp_path / "barn.webp"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image, "WEBP")
    rec = _record(output_path=str(image))
    assert embed_exif_metadata(image, rec) is False
