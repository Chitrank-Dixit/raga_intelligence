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
from raga_intelligence.datasets.partitioning import GroupAwarePartitioner, GroupingDiagnostic
from raga_intelligence.features.cache import (
    CacheValidationReport,
    FeatureCacheManager,
    compute_manifest_content_fingerprint,
    validate_cache_manifest_provenance,
)
from raga_intelligence.features.schema import FeatureSchema


def create_mock_manifest() -> DatasetManifest:
    """Builds a test manifest with 2 ragas, multiple performers, and explicit splits."""
    manifest = DatasetManifest()
    # Raga Bhairav: 2 performers
    for i in range(4):
        manifest.add_item(DatasetItem(
            recording_id=f"bhairav_clip_{i}",
            source="Test",
            file_path=f"/fake/bhairav_{i}.wav",
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
            file_path=f"/fake/yaman_{i}.wav",
            raga="Yaman",
            performer="Perf_3" if i < 2 else "Perf_4",
            performance_id="Perf_3" if i < 2 else "Perf_4",
            duration_sec=30.0,
            sample_rate=24000,
            split="train" if i < 3 else "test",
        ))
    return manifest


def test_dataset_fingerprint_deterministic_and_sensitive():
    manifest1 = create_mock_manifest()
    manifest2 = create_mock_manifest()

    fp1 = manifest1.compute_dataset_fingerprint()
    fp2 = manifest2.compute_dataset_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64

    # Tampering with a label alters the fingerprint
    manifest2.items[0].raga = "Bhoopali"
    assert manifest2.compute_dataset_fingerprint() != fp1

    # Tampering with a split alters the fingerprint
    manifest3 = create_mock_manifest()
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
    assert validation["performer_isolation_status"] == "unverifiable_incomplete_metadata"
    assert validation["performance_isolation_status"] == "unverifiable_incomplete_metadata"

    diag = GroupAwarePartitioner.audit_grouping(manifest, group_col="performer")
    assert diag.unknown_group_items_count == 1
    assert diag.is_isolation_verifiable is False


def test_cache_manifest_provenance_success_and_failure_modes(tmp_path: Path):
    manifest = create_mock_manifest()
    train_items = [it for it in manifest.items if it.split == "train"]
    test_items = [it for it in manifest.items if it.split == "test"]

    dataset_fp = manifest.compute_dataset_fingerprint()
    content_hash = compute_manifest_content_fingerprint(manifest.items, include_file_stats=False)

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
