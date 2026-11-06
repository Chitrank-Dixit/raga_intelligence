"""Compatibility and validation tests for the versioned dataset metadata contract."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest
from raga_intelligence.features.cache import compute_manifest_content_fingerprint


def minimal_item(**updates) -> DatasetItem:
    values = {
        "recording_id": "rec-1",
        "source": "fixture",
        "file_path": "rec-1.wav",
        "raga": "Bhairav",
        "duration_sec": 1.0,
        "sample_rate": 24000,
    }
    values.update(updates)
    return DatasetItem(**values)


def test_legacy_manifest_json_loads_without_claiming_license_or_acceptance(tmp_path):
    manifest_path = tmp_path / "legacy.json"
    manifest_path.write_text(json.dumps([{
        "recording_id": "legacy-1", "source": "old-source", "file_path": "legacy.wav",
        "raga": "Bhairav", "tradition": "Hindustani", "performer": "Artist name",
        "performance_id": "video-1", "duration_sec": 30.0, "sample_rate": 24000,
        "split": "train", "metadata": {},
    }]), encoding="utf-8")

    manifest = DatasetManifest.load_json(manifest_path)
    item = manifest.items[0]

    assert item.license is None
    assert item.rights_status == "unknown"
    assert item.record_status == "legacy"
    assert item.performer_id is None
    assert item.performance_identity_source == "unknown"
    assert manifest.validate_splits()["performer_isolation_status"] == "not_verifiable"


def test_new_versioned_metadata_serializes_and_round_trips(tmp_path):
    item = minimal_item(
        raw_raga_label="Bhairav",
        performer="Artist display name",
        performer_id="artist-42",
        performance_id="manual-performance-1",
        performance_identity_source="manual_review",
        source_recording_id="source-7",
        source_url="https://example.org/recording",
        license="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution="Fixture creator",
        rights_status="verified",
        annotation_status="reviewed",
        annotation_confidence=0.75,
        annotation_notes="Reviewed by test",
        record_status="accepted",
        metadata_schema_version="1.1",
        audio_sha256="A" * 64,
    )
    path = tmp_path / "new.json"
    DatasetManifest(items=[item]).save_json(path)

    loaded = DatasetManifest.load_json(path).items[0]

    assert loaded.model_dump() == item.model_dump()
    assert loaded.audio_sha256 == "a" * 64


def test_missing_license_remains_unknown_and_explicit_cc0_is_preserved():
    assert minimal_item().license is None
    synthetic_like = minimal_item(license="CC0-1.0", rights_status="verified")
    assert synthetic_like.license == "CC0-1.0"
    assert synthetic_like.rights_status == "verified"


@pytest.mark.parametrize("field,value", [
    ("rights_status", "approved"),
    ("annotation_status", "gold"),
    ("record_status", "published"),
    ("performance_identity_source", "filename"),
])
def test_controlled_status_fields_reject_unrecognized_values(field, value):
    with pytest.raises(ValidationError):
        minimal_item(**{field: value})


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_annotation_confidence_rejects_values_outside_unit_interval(confidence):
    with pytest.raises(ValidationError):
        minimal_item(annotation_confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_annotation_confidence_accepts_boundaries(confidence):
    assert minimal_item(annotation_confidence=confidence).annotation_confidence == confidence


def test_identity_source_requires_a_performance_id_and_never_infers_performer_id():
    with pytest.raises(ValidationError, match="performance_id is required"):
        minimal_item(performance_identity_source="filename_pattern")
    item = minimal_item(performer="Display Name")
    assert item.performer_id is None


@pytest.mark.parametrize("updates", [
    {"recording_id": "  "},
    {"file_path": ""},
    {"duration_sec": 0.0},
    {"sample_rate": 0},
    {"tonic_hz": 0.0},
    {"audio_sha256": "not-a-hash"},
])
def test_invalid_recording_and_audio_metadata_is_rejected(updates):
    with pytest.raises(ValidationError):
        minimal_item(**updates)


@pytest.mark.parametrize("updates", [
    {"recording_id": "  "},
    {"file_path": ""},
    {"duration_sec": 0.0},
    {"sample_rate": 0},
    {"tonic_hz": 0.0},
    {"audio_sha256": "not-a-hash"},
])
def test_invalid_recording_and_audio_metadata_is_rejected(updates):
    with pytest.raises(ValidationError):
        minimal_item(**updates)


def test_raw_noncanonical_label_is_preserved_without_canonicalization():
    item = minimal_item(raga="Unmapped Source Label", raw_raga_label="Unmapped Source Label")
    assert item.raga == "Unmapped Source Label"
    assert item.raw_raga_label == "Unmapped Source Label"


def test_historical_fingerprint_policy_is_stable_and_v2_tracks_new_fields():
    item = minimal_item()
    manifest = DatasetManifest(items=[item])
    historical_payload = json.dumps({
        "recording_id": item.recording_id,
        "split": item.split,
        "raga": item.raga,
        "tonic_hz": item.tonic_hz,
        "performer": item.performer,
        "performance_id": item.performance_id,
        "duration_sec": item.duration_sec,
        "sample_rate": item.sample_rate,
        "source": item.source,
        "tradition": item.tradition,
        "license": item.license,
        "metadata": item.metadata,
    }, sort_keys=True, separators=(",", ":"))
    expected_v1 = hashlib.sha256(historical_payload.encode()).hexdigest()

    assert manifest.compute_metadata_fingerprint() == expected_v1
    initial_v2 = manifest.compute_metadata_fingerprint_v2()
    updated = DatasetManifest(items=[item.model_copy(update={"rights_status": "pending"})])
    assert updated.compute_metadata_fingerprint_v2() != initial_v2
    assert updated.compute_metadata_fingerprint() == expected_v1


def test_historical_v1_fingerprint_for_legacy_serialized_license_is_unchanged():
    # Old model_dump() outputs stored this default explicitly; the v1 projection
    # remains stable for those serialized records while new missing values stay None.
    item = minimal_item(
        recording_id="rec-legacy", file_path="clip.wav", license="CC BY-NC-SA 4.0"
    )
    assert DatasetManifest(items=[item]).compute_metadata_fingerprint() == (
        "1807250cf4308769a838a214543c7d73838badd81923667a45e5e7103b24e207"
    )


def test_dataset_fingerprint_v2_uses_expanded_metadata_policy(tmp_path):
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"audio content")
    item = minimal_item(file_path=str(audio))
    manifest = DatasetManifest(items=[item])
    manifest.populate_audio_hashes()

    historical = manifest.compute_dataset_fingerprint()
    expanded = manifest.compute_dataset_fingerprint_v2()
    changed = DatasetManifest(items=[item.model_copy(update={"rights_status": "pending"})])

    assert expanded != historical
    assert changed.compute_dataset_fingerprint_v2() != expanded


def test_legacy_cache_fingerprint_handles_missing_source_metadata(tmp_path):
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"first")
    second_audio.write_bytes(b"second")
    items = [
        minimal_item(
            recording_id="duplicate", file_path=str(first_audio), source=None,
            audio_sha256=hashlib.sha256(b"first").hexdigest(),
        ),
        minimal_item(
            recording_id="duplicate", file_path=str(second_audio), source="known source",
            audio_sha256=hashlib.sha256(b"second").hexdigest(),
        ),
    ]

    assert len(compute_manifest_content_fingerprint(items)) == 64


def test_hindustani_adapter_records_identity_origin_and_does_not_assume_license(tmp_path):
    audio = tmp_path / "train_[video123]_chunk0.wav"
    audio.write_bytes(b"fixture")
    (tmp_path / "metadata.csv").write_text(
        "file_name,label,tonic_hz\ntrain_[video123]_chunk0.wav,Bhairav,220\n",
        encoding="utf-8",
    )
    from raga_intelligence.datasets.hindustani_dataset import HindustaniDatasetAdapter

    item = HindustaniDatasetAdapter(dataset_dir=tmp_path).build_manifest().items[0]

    assert item.raw_raga_label == "Bhairav"
    assert item.performance_identity_source == "filename_pattern"
    assert item.performer_id is None
    assert item.license is None
    assert item.rights_status == "unknown"


def test_hindustani_adapter_preserves_explicit_manual_performance_and_provenance(tmp_path):
    audio = tmp_path / "train_[video123]_chunk0.wav"
    audio.write_bytes(b"fixture")
    (tmp_path / "metadata.csv").write_text(
        "file_name,label,tonic_hz,performance_id,performance_identity_source,"
        "performer,performer_id,license,license_url,rights_status,source_url\n"
        "train_[video123]_chunk0.wav,Bhairav,220,reviewed-perf,manual_review,"
        "Artist,artist-9,CC0-1.0,https://example.org/license,verified,https://example.org/source\n",
        encoding="utf-8",
    )
    from raga_intelligence.datasets.hindustani_dataset import HindustaniDatasetAdapter

    item = HindustaniDatasetAdapter(dataset_dir=tmp_path).build_manifest().items[0]

    assert item.performance_id == "reviewed-perf"
    assert item.performance_identity_source == "manual_review"
    assert item.performer == "Artist"
    assert item.performer_id == "artist-9"
    assert item.license == "CC0-1.0"
    assert item.rights_status == "verified"
    assert item.source_url == "https://example.org/source"


def test_hindustani_adapter_can_preserve_noncanonical_raw_labels_without_mapping(tmp_path):
    audio = tmp_path / "train_[video123]_chunk0.wav"
    audio.write_bytes(b"fixture")
    (tmp_path / "metadata.csv").write_text(
        f"file_name,label,tonic_hz\n{audio.name},Unmapped Raga,220\n", encoding="utf-8"
    )
    from raga_intelligence.datasets.hindustani_dataset import HindustaniDatasetAdapter

    default_manifest = HindustaniDatasetAdapter(dataset_dir=tmp_path).build_manifest()
    inclusive_manifest = HindustaniDatasetAdapter(
        dataset_dir=tmp_path, include_noncanonical_ragas=True
    ).build_manifest()

    assert default_manifest.items == []
    assert inclusive_manifest.items[0].raga == "Unmapped Raga"
    assert inclusive_manifest.items[0].raw_raga_label == "Unmapped Raga"
