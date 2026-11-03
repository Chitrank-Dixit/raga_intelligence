"""
Unit tests for dataset fingerprinting, cache-manifest provenance auditing,
multi-raga performer partitioning, and incomplete metadata diagnostics.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List

import numpy as np
import pytest

from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest
from raga_intelligence.datasets.partitioning import (
    GroupAwarePartitioner,
    GroupingDiagnostic,
    InfeasiblePartitionError,
)
from raga_intelligence.features.cache import (
    CacheValidationReport,
    FeatureCacheManager,
    compute_manifest_content_fingerprint,
    validate_cache_manifest_provenance,
)
from raga_intelligence.features.schema import FeatureSchema


def create_mock_manifest(audio_dir: Path) -> DatasetManifest:
    """Builds a test manifest with 2 ragas, multiple performers, and explicit splits."""
    manifest = DatasetManifest()
    # Raga Bhairav: 2 performers
    for i in range(4):
        manifest.add_item(DatasetItem(
            recording_id=f"bhairav_clip_{i}",
            source="Test",
            file_path=str(audio_dir / f"bhairav_{i}.wav"),
            raga="Bhairav",
            performer="Perf_1" if i < 2 else "Perf_2",
            performance_id="Perf_1" if i < 2 else "Perf_2",
            duration_sec=30.0,
            sample_rate=24000,
            split="train" if i < 3 else "test",
        ))
    # Raga Yaman: 2 performers
    for i in range(4):
        manifest.add_item(DatasetItem(
            recording_id=f"yaman_clip_{i}",
            source="Test",
            file_path=str(audio_dir / f"yaman_{i}.wav"),
            raga="Yaman",
            performer="Perf_3" if i < 2 else "Perf_4",
            performance_id="Perf_3" if i < 2 else "Perf_4",
            duration_sec=30.0,
            sample_rate=24000,
            split="train" if i < 3 else "test",
        ))
    audio_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(manifest.items):
        Path(item.file_path).write_bytes(f"audio-{index}".encode())
    manifest.populate_audio_hashes()
    return manifest


def test_dataset_fingerprint_deterministic_and_sensitive(tmp_path: Path):
    manifest1 = create_mock_manifest(tmp_path / "first")
    manifest2 = create_mock_manifest(tmp_path / "second")

    fp1 = manifest1.compute_dataset_fingerprint()
    fp2 = manifest2.compute_dataset_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64

    # Tampering with a label alters the fingerprint
    manifest2.items[0].raga = "Bhoopali"
    assert manifest2.compute_dataset_fingerprint() != fp1

    # Tampering with a split alters the fingerprint
    manifest3 = create_mock_manifest(tmp_path / "third")
    manifest3.items[0].split = "test"
    assert manifest3.compute_dataset_fingerprint() != fp1


def test_multi_raga_performer_assigned_to_single_split():
    """
    If a performer performs in both Bhairav and Yaman, all recordings of that
    performer must be assigned to strictly ONE split, preventing cross-split leakage.
    """
    manifest = DatasetManifest()
    # Performer MultiPerf sings in both Bhairav and Yaman
    manifest.add_item(DatasetItem(
        recording_id="rec_bhairav_multi",
        source="Test",
        file_path="/dummy/1.wav",
        raga="Bhairav",
        performer="MultiPerf",
        performance_id="Perf_A",
        duration_sec=10.0,
        sample_rate=24000,
    ))
    manifest.add_item(DatasetItem(
        recording_id="rec_yaman_multi",
        source="Test",
        file_path="/dummy/2.wav",
        raga="Yaman",
        performer="MultiPerf",
        performance_id="Perf_B",
        duration_sec=10.0,
        sample_rate=24000,
    ))
    # Other performers
    for raga in ["Bhairav", "Yaman"]:
        for p in range(3):
            manifest.add_item(DatasetItem(
                recording_id=f"rec_{raga}_{p}",
                source="Test",
                file_path=f"/dummy/{raga}_{p}.wav",
                raga=raga,
                performer=f"Solo_{raga}_{p}",
                performance_id=f"Solo_{raga}_{p}",
                duration_sec=10.0,
                sample_rate=24000,
            ))

    # Audit grouping
    diag = GroupAwarePartitioner.audit_grouping(manifest, group_col="performer")
    assert "MultiPerf" in diag.multi_raga_groups
    assert set(diag.multi_raga_groups["MultiPerf"]) == {"Bhairav", "Yaman"}

    # Partition
    train_m, val_m, test_m = GroupAwarePartitioner.partition(
        manifest,
        group_col="performer",
        test_ratio=0.3,
        val_ratio=0.0,
        seed=42,
    )

    # Check which split MultiPerf went into
    multi_splits = set()
    for item in train_m.items:
        if item.performer == "MultiPerf":
            multi_splits.add("train")
    for item in test_m.items:
        if item.performer == "MultiPerf":
            multi_splits.add("test")

    # MultiPerf must strictly belong to ONE split, never both
    assert len(multi_splits) == 1


def test_incomplete_performer_metadata_diagnostics():
    """Recordings with unknown/missing performer must be flagged as unverifiable isolation."""
    manifest = DatasetManifest()
    manifest.add_item(DatasetItem(
        recording_id="rec_known",
        source="Test",
        file_path="/dummy/known.wav",
        raga="Bhairav",
        performer="KnownArtist",
        performance_id="Perf_1",
        duration_sec=10.0,
        sample_rate=24000,
        split="train",
    ))
    manifest.add_item(DatasetItem(
        recording_id="rec_unknown",
        source="Test",
        file_path="/dummy/unknown.wav",
        raga="Bhairav",
        performer="Unknown",
        performance_id=None,
        duration_sec=10.0,
        sample_rate=24000,
        split="train",
    ))

    validation = manifest.validate_splits()
    assert validation["performer_isolation_status"] == "not_verifiable"
    assert validation["performance_isolation_status"] == "not_verifiable"
    assert validation["recording_isolation_status"] == "verified"
    assert validation["valid_no_performer_leakage"] is False
    assert validation["valid_no_performance_leakage"] is False

    diag = GroupAwarePartitioner.audit_grouping(manifest, group_col="performer")
    assert diag.unknown_group_items_count == 1
    assert diag.is_isolation_verifiable is False


def test_cache_manifest_provenance_success_and_failure_modes(tmp_path: Path):
    manifest = create_mock_manifest(tmp_path / "audio")
    train_items = [it for it in manifest.items if it.split == "train"]
    test_items = [it for it in manifest.items if it.split == "test"]

    dataset_fp = manifest.compute_dataset_fingerprint()
    content_hash = compute_manifest_content_fingerprint(manifest.items)

    feat_arrays = {
        "X_train": np.ones((len(train_items), 35), dtype=np.float32),
        "y_train": np.array([it.raga for it in train_items]),
        "recording_ids_train": np.array([it.recording_id for it in train_items]),
        "splits_train": np.array(["train"] * len(train_items)),
        "X_test": np.ones((len(test_items), 35), dtype=np.float32),
        "y_test": np.array([it.raga for it in test_items]),
        "recording_ids_test": np.array([it.recording_id for it in test_items]),
        "splits_test": np.array(["test"] * len(test_items)),
    }
    feat_meta = {
        "dataset_fingerprint": dataset_fp,
        "fingerprint_policy": "audio_sha256_v1",
        "feature_schema_version": FeatureSchema.SCHEMA_VERSION,
        "manifest_hash": content_hash,
    }
    feat_arrays["metadata"] = feat_meta

    # 1. Valid caches pass validation
    report = validate_cache_manifest_provenance(manifest, feat_arrays, raise_on_error=True)
    assert report.is_valid is True
    assert report.n_train == len(train_items)
    assert report.n_test == len(test_items)

    # 2. Missing metadata fails explicitly
    corrupt_feat = {k: v for k, v in feat_arrays.items() if k != "metadata"}
    with pytest.raises(ValueError, match="missing 'metadata' section"):
        validate_cache_manifest_provenance(manifest, corrupt_feat, raise_on_error=True)

    # 3. Fingerprint mismatch fails explicitly
    mismatch_meta = dict(feat_meta)
    mismatch_meta["dataset_fingerprint"] = "0000000000000000000000000000000000000000000000000000000000000000"
    mismatch_cache = dict(feat_arrays)
    mismatch_cache["metadata"] = mismatch_meta
    with pytest.raises(ValueError, match="dataset fingerprint mismatch"):
        validate_cache_manifest_provenance(manifest, mismatch_cache, raise_on_error=True)

    # 4. Schema version mismatch fails explicitly
    bad_schema_meta = dict(feat_meta)
    bad_schema_meta["feature_schema_version"] = "99.0.0"
    bad_schema_cache = dict(feat_arrays)
    bad_schema_cache["metadata"] = bad_schema_meta
    with pytest.raises(ValueError, match="schema version mismatch"):
        validate_cache_manifest_provenance(manifest, bad_schema_cache, raise_on_error=True)

    bad_policy_cache = dict(feat_arrays)
    bad_policy_cache["metadata"] = {**feat_meta, "fingerprint_policy": "metadata_only_v1"}
    with pytest.raises(ValueError, match="fingerprint policy"):
        validate_cache_manifest_provenance(manifest, bad_policy_cache, raise_on_error=True)

    # 5. Swapped or misaligned recording IDs fail explicitly
    swapped_ids_cache = dict(feat_arrays)
    swapped_ids_cache["metadata"] = feat_meta
    swapped_ids_cache["recording_ids_train"] = np.array(list(reversed(feat_arrays["recording_ids_train"])))
    with pytest.raises(ValueError, match="train recording IDs do not match manifest"):
        validate_cache_manifest_provenance(manifest, swapped_ids_cache, raise_on_error=True)

    # 6. Misaligned MERT cache fails explicitly
    valid_feat = dict(feat_arrays)
    valid_feat["metadata"] = feat_meta

    mert_arrays = {
        "E_train": np.ones((len(train_items), 1536), dtype=np.float32),
        "y_train": np.array([it.raga for it in train_items]),
        "recording_ids_train": np.array([it.recording_id for it in train_items]),
        "splits_train": np.array(["train"] * len(train_items)),
        "E_test": np.ones((len(test_items), 1536), dtype=np.float32),
        "y_test": np.array([it.raga for it in test_items]),
        "recording_ids_test": np.array([it.recording_id for it in test_items]),
        "splits_test": np.array(["test"] * len(test_items)),
        "metadata": {
            "dataset_fingerprint": dataset_fp,
            "fingerprint_policy": "audio_sha256_v1",
            "feature_schema_version": FeatureSchema.SCHEMA_VERSION,
        },
    }
    # Validate with MERT succeeds
    rep_with_mert = validate_cache_manifest_provenance(manifest, valid_feat, mert_arrays, raise_on_error=True)
    assert rep_with_mert.is_valid is True

    # Alter MERT labels
    bad_mert = dict(mert_arrays)
    bad_mert["y_test"] = np.array(["WrongLabel"] * len(test_items))
    with pytest.raises(ValueError, match="MERT cache test labels do not match"):
        validate_cache_manifest_provenance(manifest, valid_feat, bad_mert, raise_on_error=True)


def test_metadata_vs_audio_content_fingerprint_distinction(tmp_path: Path):
    """
    Demonstrates and verifies the architectural distinction:
    - Metadata fingerprint guarantees label, tonic, performer, and split consistency.
    - Audio-content fingerprint guarantees raw audio bytes have not been altered or corrupted.
    """
    audio1 = tmp_path / "rec1.wav"
    audio2 = tmp_path / "rec2.wav"
    audio1.write_bytes(b"RIFF" + b"\x00" * 50)
    audio2.write_bytes(b"RIFF" + b"\x01" * 50)

    manifest = DatasetManifest(items=[
        DatasetItem(
            recording_id="rec_1",
            source="Test",
            file_path=str(audio1),
            raga="Bhairav",
            split="train",
            performer="Artist_A",
            duration_sec=10.0,
            sample_rate=24000,
            tonic_hz=220.0,
        ),
        DatasetItem(
            recording_id="rec_2",
            source="Test",
            file_path=str(audio2),
            raga="Yaman",
            split="test",
            performer="Artist_B",
            duration_sec=10.0,
            sample_rate=24000,
            tonic_hz=140.0,
        ),
    ])

    manifest.populate_audio_hashes()

    meta_fp_before = manifest.compute_metadata_fingerprint()
    audio_fp_before = manifest.compute_audio_content_fingerprint()
    dataset_fp_before = manifest.compute_dataset_fingerprint()

    # Modify raw audio file on disk without changing any manifest metadata
    audio1.write_bytes(b"RIFF" + b"\xFF" * 100)

    # 1. Metadata fingerprint MUST remain invariant (metadata unchanged)
    meta_fp_after = manifest.compute_metadata_fingerprint()
    assert meta_fp_before == meta_fp_after

    # 2. Audio content fingerprint MUST change to reflect modified bytes
    audio_fp_after = manifest.compute_audio_content_fingerprint(force_recompute=True)
    assert audio_fp_before != audio_fp_after

    # 3. Overall dataset fingerprint MUST change to invalidate downstream caches
    dataset_fp_after = manifest.compute_dataset_fingerprint(force_recompute=True)
    assert dataset_fp_before != dataset_fp_after


def test_audio_fingerprint_fails_closed_and_is_path_independent(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "clip.wav").write_bytes(b"identical bytes")
    (right / "clip.wav").write_bytes(b"identical bytes")

    def manifest_at(path: Path, populate_hash: bool) -> DatasetManifest:
        manifest = DatasetManifest(items=[DatasetItem(
            recording_id="clip", source="test", file_path=str(path), raga="Bhairav",
            duration_sec=1.0, sample_rate=24000,
        )])
        if populate_hash:
            manifest.populate_audio_hashes()
        return manifest

    with pytest.raises(ValueError, match="Missing or invalid audio SHA-256"):
        manifest_at(left / "clip.wav", False).compute_dataset_fingerprint()
    assert (
        manifest_at(left / "clip.wav", True).compute_dataset_fingerprint()
        == manifest_at(right / "clip.wav", True).compute_dataset_fingerprint()
    )
    left_items = manifest_at(left / "clip.wav", True).items
    right_items = manifest_at(right / "clip.wav", True).items
    assert compute_manifest_content_fingerprint(left_items) == compute_manifest_content_fingerprint(right_items)


def test_audio_fingerprint_fails_for_missing_or_unreadable_audio(tmp_path: Path):
    missing = DatasetManifest(items=[DatasetItem(
        recording_id="missing", source="test", file_path=str(tmp_path / "missing.wav"),
        raga="Bhairav", duration_sec=1.0, sample_rate=24000, audio_sha256="0" * 64,
    )])
    with pytest.raises(OSError, match="Could not compute audio SHA-256"):
        missing.compute_dataset_fingerprint()

    unreadable_path = tmp_path / "unreadable.wav"
    unreadable_path.write_bytes(b"audio")
    unreadable = DatasetManifest(items=[DatasetItem(
        recording_id="unreadable", source="test", file_path=str(unreadable_path),
        raga="Bhairav", duration_sec=1.0, sample_rate=24000, audio_sha256="0" * 64,
    )])
    import builtins
    original_open = builtins.open

    def deny(path, *args, **kwargs):
        if Path(path) == unreadable_path:
            raise PermissionError("denied")
        return original_open(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(builtins, "open", deny)
        with pytest.raises(OSError, match="Could not compute audio SHA-256"):
            unreadable.compute_dataset_fingerprint()


def test_metadata_fingerprint_has_explicit_distinct_semantics(tmp_path: Path):
    manifest = create_mock_manifest(tmp_path / "audio")
    assert manifest.compute_metadata_fingerprint() != manifest.compute_dataset_fingerprint()


def test_interconnected_multi_raga_performer_cycle_impossible_coverage():
    """
    Tests edge case: A 3-cycle of performers linking 3 ragas:
      P1 -> {Bhairav, Yaman}
      P2 -> {Yaman, Bhoopali}
      P3 -> {Bhoopali, Bhairav}
    Because each performer must be isolated to strictly ONE split, any 2-partition of {P1, P2, P3}
    leaves at least one raga completely missing from one of the splits.
    The partitioner must:
    1. Identify the entangled component.
    2. Explicitly flag the impossible class coverage.
    3. Raise InfeasiblePartitionError if strict_coverage=True.
    """
    manifest = DatasetManifest()
    manifest.add_item(DatasetItem(
        recording_id="rec_1", source="Test", file_path="/f/1.wav",
        raga="Bhairav", performer="P1", performance_id="P1", duration_sec=10.0, sample_rate=24000,
    ))
    manifest.add_item(DatasetItem(
        recording_id="rec_2", source="Test", file_path="/f/2.wav",
        raga="Yaman", performer="P1", performance_id="P1", duration_sec=10.0, sample_rate=24000,
    ))
    manifest.add_item(DatasetItem(
        recording_id="rec_3", source="Test", file_path="/f/3.wav",
        raga="Yaman", performer="P2", performance_id="P2", duration_sec=10.0, sample_rate=24000,
    ))
    manifest.add_item(DatasetItem(
        recording_id="rec_4", source="Test", file_path="/f/4.wav",
        raga="Bhoopali", performer="P2", performance_id="P2", duration_sec=10.0, sample_rate=24000,
    ))
    manifest.add_item(DatasetItem(
        recording_id="rec_5", source="Test", file_path="/f/5.wav",
        raga="Bhoopali", performer="P3", performance_id="P3", duration_sec=10.0, sample_rate=24000,
    ))
    manifest.add_item(DatasetItem(
        recording_id="rec_6", source="Test", file_path="/f/6.wav",
        raga="Bhairav", performer="P3", performance_id="P3", duration_sec=10.0, sample_rate=24000,
    ))

    # Audit detects entangled component
    diag = GroupAwarePartitioner.audit_grouping(manifest, group_col="performer")
    assert len(diag.entangled_components) == 1
    assert diag.entangled_components[0]["num_ragas"] == 3
    assert diag.entangled_components[0]["num_groups"] == 3
    assert diag.is_fully_partitionable is False

    # Lenient partitioning notes limitation explicitly
    split_m, diag2 = GroupAwarePartitioner.create_group_aware_split(
        manifest, group_col="performer", strict_coverage=False
    )
    assert len(diag2.impossible_coverage_ragas) > 0
    assert any("Impossible class coverage" in lim for lim in diag2.limitations)

    # Strict partitioning raises InfeasiblePartitionError
    with pytest.raises(InfeasiblePartitionError, match="Cannot construct valid group-aware partition"):
        GroupAwarePartitioner.create_group_aware_split(
            manifest, group_col="performer", strict_coverage=True
        )


def test_global_split_consistency_across_seeds():
    """
    Tests that multi-seed split generation preserves global group isolation
    with 0 recording, performance, and performer leakage across all seeds.
    """
    manifest = DatasetManifest()
    # 4 performers, each singing 2 ragas
    for p_idx in range(4):
        p_name = f"Performer_{p_idx}"
        for r_name in ["Bhairav", "Yaman"]:
            manifest.add_item(DatasetItem(
                recording_id=f"{p_name}_{r_name}",
                source="Test",
                file_path=f"/dummy/{p_name}_{r_name}.wav",
                raga=r_name,
                performer=p_name,
                performance_id=p_name,
                duration_sec=10.0,
                sample_rate=24000,
            ))

    seeds = [101, 202, 303, 404]
    multi_splits = GroupAwarePartitioner.generate_multi_seed_splits(
        manifest, seeds=seeds, group_col="performer"
    )

    for s in seeds:
        split_manifest = multi_splits[s]
        audit = split_manifest.validate_splits()
        assert audit["valid_no_recording_leakage"] is True
        assert audit["valid_no_performer_leakage"] is True

