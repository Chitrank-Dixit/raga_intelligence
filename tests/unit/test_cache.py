"""
Unit tests for FeatureCacheManager and manifest fingerprinting.
Verifies cache hits, parameter invalidation, audio modification invalidation,
corruption handling, and atomic writes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest
from raga_intelligence.features.cache import (
    FeatureCacheManager,
    compute_file_sha256,
    compute_manifest_content_fingerprint,
    validate_cache_manifest_provenance,
)
from raga_intelligence.features.schema import FeatureSchema


@pytest.fixture
def temp_audio_file(tmp_path: Path) -> Path:
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
    return audio_file


@pytest.fixture
def sample_dataset_items(temp_audio_file: Path) -> list[DatasetItem]:
    items = [
        DatasetItem(
            recording_id="rec_001",
            raga="Yaman",
            split="train",
            file_path=str(temp_audio_file),
            duration_sec=10.0,
            sample_rate=24000,
            source="saraga",
            tonic_hz=220.0,
            metadata={"performer": "Artist A"},
        ),
        DatasetItem(
            recording_id="rec_002",
            raga="Bhairav",
            split="test",
            file_path=str(temp_audio_file),
            duration_sec=10.0,
            sample_rate=24000,
            source="saraga",
            tonic_hz=138.59,
            metadata={"performer": "Artist B"},
        ),
    ]
    for item in items:
        item.audio_sha256 = compute_file_sha256(item.file_path)
    return items


def test_fingerprint_deterministic_and_matches(sample_dataset_items: list[DatasetItem]) -> None:
    fp1 = compute_manifest_content_fingerprint(sample_dataset_items)
    fp2 = compute_manifest_content_fingerprint(sample_dataset_items)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256


def test_fingerprint_invalidated_by_file_modification(
    sample_dataset_items: list[DatasetItem],
    temp_audio_file: Path,
) -> None:
    fp_orig = compute_manifest_content_fingerprint(sample_dataset_items)

    # Modify audio file content (and size/mtime)
    time.sleep(0.01)
    temp_audio_file.write_bytes(b"RIFF" + b"\x00" * 200)

    fp_modified = compute_manifest_content_fingerprint(sample_dataset_items)
    assert fp_orig != fp_modified


def test_cache_hit_with_identical_inputs(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    arrays = {
        "X_train": np.ones((5, 10), dtype=np.float32),
        "y_train": np.array(["Yaman"] * 5),
    }
    metadata = {
        "manifest_hash": "abc123456",
        "pitch_tracker": "pyin",
        "model_revision": "main",
    }

    # Save
    FeatureCacheManager.save_atomic_cache(cache_path, arrays, metadata)
    assert cache_path.exists()

    # Load with identical metadata
    loaded = FeatureCacheManager.load_validated_cache(cache_path, expected_metadata=metadata)
    assert loaded is not None
    assert np.array_equal(loaded["X_train"], arrays["X_train"])
    assert np.array_equal(loaded["y_train"], arrays["y_train"])


def test_cache_miss_on_metadata_change(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    arrays = {"X": np.ones((2, 2))}
    metadata = {
        "manifest_hash": "hash_v1",
        "pitch_tracker": "pyin",
        "model_revision": "main",
    }

    FeatureCacheManager.save_atomic_cache(cache_path, arrays, metadata)

    # Change pitch tracker
    mismatched_meta = dict(metadata)
    mismatched_meta["pitch_tracker"] = "crepe"
    loaded = FeatureCacheManager.load_validated_cache(cache_path, expected_metadata=mismatched_meta)
    assert loaded is None

    # Change model revision
    mismatched_meta2 = dict(metadata)
    mismatched_meta2["model_revision"] = "v2"
    loaded2 = FeatureCacheManager.load_validated_cache(cache_path, expected_metadata=mismatched_meta2)
    assert loaded2 is None


def test_cache_miss_on_force_refresh(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    arrays = {"X": np.ones((2, 2))}
    metadata = {"manifest_hash": "hash_v1"}

    FeatureCacheManager.save_atomic_cache(cache_path, arrays, metadata)

    loaded = FeatureCacheManager.load_validated_cache(cache_path, expected_metadata=metadata, force_refresh=True)
    assert loaded is None


def test_cache_recovery_from_corrupted_file(tmp_path: Path) -> None:
    cache_path = tmp_path / "corrupted_cache.npz"
    # Write garbage bytes
    cache_path.write_bytes(b"NOT_A_VALID_ZIP_OR_NPZ_FILE_DATA_1234567890")

    metadata = {"manifest_hash": "hash_v1"}
    # Should not raise exception; should return None and log warning
    loaded = FeatureCacheManager.load_validated_cache(cache_path, expected_metadata=metadata)
    assert loaded is None


def test_cache_invalidation_when_metadata_section_missing(tmp_path: Path) -> None:
    cache_path = tmp_path / "no_meta.npz"
    # Save standard numpy archive without metadata
    np.savez(cache_path, X=np.ones((2, 2)))

    metadata = {"manifest_hash": "hash_v1"}
    loaded = FeatureCacheManager.load_validated_cache(cache_path, expected_metadata=metadata)
    assert loaded is None


def test_atomic_write_leaves_no_orphaned_temp_files(tmp_path: Path) -> None:
    cache_path = tmp_path / "subdir" / "atomic_cache.npz"
    arrays = {"feat": np.zeros((3, 3))}
    metadata = {"test": 123}

    FeatureCacheManager.save_atomic_cache(cache_path, arrays, metadata)
    assert cache_path.exists()

    # Verify no .tmp.npz files remain in parent dir
    tmp_files = list(cache_path.parent.glob("*.tmp.npz"))
    assert len(tmp_files) == 0


def test_compute_file_sha256_deterministic(tmp_path: Path) -> None:
    test_file = tmp_path / "cache_data.npz"
    test_file.write_bytes(b"deterministic_cache_payload_bytes_12345")
    h1 = compute_file_sha256(test_file)
    h2 = compute_file_sha256(test_file)
    assert h1 == h2
    assert len(h1) == 64


def test_replacing_cache_contents_changes_file_sha256(tmp_path: Path) -> None:
    cache_path = tmp_path / "feat_cache.npz"
    arrays_v1 = {"X": np.ones((5, 10), dtype=np.float32)}
    meta_v1 = {"version": "1.0"}
    FeatureCacheManager.save_atomic_cache(cache_path, arrays_v1, meta_v1)
    sha_orig = compute_file_sha256(cache_path)

    # Replace cache contents
    arrays_v2 = {"X": np.zeros((5, 10), dtype=np.float32)}
    meta_v2 = {"version": "2.0"}
    FeatureCacheManager.save_atomic_cache(cache_path, arrays_v2, meta_v2)
    sha_replaced = compute_file_sha256(cache_path)

    assert sha_orig != sha_replaced


def test_validate_cache_manifest_provenance_records_paths_and_hashes(tmp_path: Path) -> None:
    cache_path = tmp_path / "validated_cache.npz"
    manifest = DatasetManifest(items=[
        DatasetItem(
            recording_id="rec_A",
            source="test",
            file_path=str(tmp_path / "a.wav"),
            raga="Bhairav",
            split="train",
            duration_sec=10.0,
            sample_rate=24000,
        ),
        DatasetItem(
            recording_id="rec_B",
            source="test",
            file_path=str(tmp_path / "b.wav"),
            raga="Bhairav",
            split="test",
            duration_sec=10.0,
            sample_rate=24000,
        ),
    ])

    Path(manifest.items[0].file_path).write_bytes(b"audio-a")
    Path(manifest.items[1].file_path).write_bytes(b"audio-b")
    manifest.populate_audio_hashes()

    expected_fp = manifest.compute_dataset_fingerprint()
    arrays = {
        "X_train": np.ones((1, 35)),
        "y_train": np.array(["Bhairav"]),
        "recording_ids_train": np.array(["rec_A"]),
        "splits_train": np.array(["train"]),
        "X_test": np.ones((1, 35)),
        "y_test": np.array(["Bhairav"]),
        "recording_ids_test": np.array(["rec_B"]),
        "splits_test": np.array(["test"]),
    }
    meta = {
        "dataset_fingerprint": expected_fp,
        "fingerprint_policy": "audio_sha256_v1",
        "feature_schema_version": FeatureSchema.SCHEMA_VERSION,
    }
    FeatureCacheManager.save_atomic_cache(cache_path, arrays, meta)
    expected_sha = compute_file_sha256(cache_path)

    with np.load(cache_path, allow_pickle=True) as cache_data:
        report = validate_cache_manifest_provenance(
            manifest=manifest,
            feat_cache_data=cache_data,
            feat_cache_path=cache_path,
        )

    assert report.is_valid is True
    assert report.feat_cache_path == str(cache_path)
    assert report.feat_cache_sha256 == expected_sha

