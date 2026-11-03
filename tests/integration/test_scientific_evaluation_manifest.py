import hashlib
import re
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_scientific_evaluation import (
    build_evaluation_manifest,
    evaluation_mode_banner,
    format_leakage_audit,
    validate_scientific_partition,
)
import run_scientific_evaluation as scientific_evaluation
from raga_intelligence.config import RAGA_LIST
from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest
from raga_intelligence.datasets.partitioning import InfeasiblePartitionError
from raga_intelligence.features.cache import compute_manifest_content_fingerprint


def make_partition_manifest(*, missing_performance_ids=False, performance_leak=False):
    items = []
    for raga in RAGA_LIST:
        for group in range(2):
            if missing_performance_ids:
                performance_id = None
            elif performance_leak and raga == RAGA_LIST[0]:
                performance_id = f"{raga}_performance_0"
            else:
                performance_id = f"{raga}_performance_{group}"
            items.append(DatasetItem(
                recording_id=f"{raga}_{group}",
                source="Test",
                file_path=f"{raga}_{group}.wav",
                raga=raga,
                performer=None,
                performance_id=performance_id,
                duration_sec=10.0,
                sample_rate=24000,
                split="train" if group == 0 else "test",
            ))
    return DatasetManifest(items=items)


def test_evaluation_manifest_hashes_audio_and_invalidates_fingerprint(tmp_path):
    audio_path = tmp_path / "train_[performer1]_chunk0.wav"
    audio_path.write_bytes(b"original audio content")
    pd.DataFrame(
        [{"file_name": audio_path.name, "label": "Bhairav", "tonic_hz": 220.0}]
    ).to_csv(tmp_path / "metadata.csv", index=False)

    manifest = build_evaluation_manifest(tmp_path)
    item = manifest.items[0]
    original_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    original_fingerprint = manifest.compute_dataset_fingerprint()
    original_cache_fingerprint = compute_manifest_content_fingerprint(manifest.items)

    assert item.audio_sha256 == original_hash
    assert re.fullmatch(r"[0-9a-f]{64}", item.audio_sha256)

    audio_path.write_bytes(b"modified audio content")
    modified = build_evaluation_manifest(tmp_path)
    assert modified.items[0].audio_sha256 != original_hash
    assert modified.compute_dataset_fingerprint() != original_fingerprint
    assert compute_manifest_content_fingerprint(modified.items) != original_cache_fingerprint


def test_evaluation_manifest_fails_when_audio_cannot_be_read(tmp_path, monkeypatch):
    audio_path = tmp_path / "train_[performer1]_chunk0.wav"
    audio_path.write_bytes(b"audio")
    pd.DataFrame(
        [{"file_name": audio_path.name, "label": "Bhairav", "tonic_hz": 220.0}]
    ).to_csv(tmp_path / "metadata.csv", index=False)

    real_open = open

    def deny_audio(path, *args, **kwargs):
        if str(path) == str(audio_path):
            raise PermissionError("denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", deny_audio)
    with pytest.raises(OSError, match="Could not hash audio content"):
        build_evaluation_manifest(tmp_path)


def test_report_audit_does_not_claim_unverified_performer_isolation():
    lines = format_leakage_audit({
        "recording_isolation_status": "verified",
        "performance_isolation_status": "verified",
        "performer_isolation_status": "not_verifiable",
    })
    assert "Performer Isolation: `not_verifiable`" in lines[2]
    assert all("0.0%" not in line and "zero" not in line.lower() for line in lines)


def test_dirty_development_report_is_marked_non_canonical():
    assert "NON-CANONICAL DEVELOPMENT RUN" in evaluation_mode_banner(False)
    assert evaluation_mode_banner(True) == ""


def test_infeasible_partition_is_rejected_before_evaluation_cache_or_models(
    tmp_path, monkeypatch
):
    manifest = DatasetManifest(items=[
        DatasetItem(
            recording_id="only_performance",
            source="Test",
            file_path="unread_audio.wav",
            raga="Bhairav",
            performer=None,
            performance_id="single_performance",
            duration_sec=10.0,
            sample_rate=24000,
            split="train",
        )
    ])
    monkeypatch.setattr(
        scientific_evaluation,
        "assert_clean_source_tree",
        lambda **kwargs: ("a" * 40, True),
    )
    monkeypatch.setattr(
        scientific_evaluation, "build_evaluation_manifest", lambda dataset_dir: manifest
    )
    monkeypatch.setattr(
        scientific_evaluation.np,
        "load",
        lambda *args, **kwargs: pytest.fail("infeasible partition reached cache loading"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_scientific_evaluation.py", "--dataset-dir", str(tmp_path)],
    )

    with pytest.raises(InfeasiblePartitionError, match="requires every canonical raga"):
        scientific_evaluation.main()


def test_scientific_partition_requires_every_raga_in_train_and_test():
    manifest = DatasetManifest(items=[
        DatasetItem(
            recording_id=f"{raga}_{group}",
            source="Test",
            file_path=f"{raga}_{group}.wav",
            raga=raga,
            performer=f"artist_{raga}_{group}",
            performance_id=f"performance_{raga}_{group}",
            duration_sec=10.0,
            sample_rate=24000,
            split="train",
        )
        for raga in RAGA_LIST
        for group in range(2)
    ])

    with pytest.raises(InfeasiblePartitionError, match="missing coverage"):
        validate_scientific_partition(manifest)


def test_complete_performance_isolation_allows_unknown_performer_metadata():
    manifest = make_partition_manifest()

    diagnostic, audit = validate_scientific_partition(manifest)

    assert diagnostic.is_isolation_verifiable is True
    assert audit["recording_isolation_status"] == "verified"
    assert audit["performance_isolation_status"] == "verified"
    assert audit["performer_isolation_status"] == "not_verifiable"


def test_scientific_partition_rejects_performance_leakage():
    manifest = make_partition_manifest(performance_leak=True)

    with pytest.raises(ValueError, match="performance isolation status exactly 'verified'.*failed"):
        validate_scientific_partition(manifest)


def test_scientific_partition_rejects_unknown_performance_status(monkeypatch):
    manifest = make_partition_manifest()
    original_validate = DatasetManifest.validate_splits

    def unknown_performance_status(self):
        result = original_validate(self)
        if self is manifest:
            result["performance_isolation_status"] = "unknown"
        return result

    monkeypatch.setattr(DatasetManifest, "validate_splits", unknown_performance_status)
    with pytest.raises(ValueError, match="status exactly 'verified'.*unknown"):
        validate_scientific_partition(manifest)


def test_missing_performance_metadata_stops_before_all_evaluation_work(tmp_path, monkeypatch):
    manifest = make_partition_manifest(missing_performance_ids=True)
    cache_loader = Mock(side_effect=AssertionError("cache loading must not run"))
    model_initializer = Mock(side_effect=AssertionError("model initialization must not run"))
    ablation_initializer = Mock(side_effect=AssertionError("ablation must not run"))
    error_analysis = Mock(side_effect=AssertionError("evaluation must not run"))

    monkeypatch.setattr(
        scientific_evaluation,
        "assert_clean_source_tree",
        lambda **kwargs: ("a" * 40, True),
    )
    monkeypatch.setattr(
        scientific_evaluation, "build_evaluation_manifest", lambda dataset_dir: manifest
    )
    monkeypatch.setattr(scientific_evaluation.np, "load", cache_loader)
    monkeypatch.setattr(scientific_evaluation, "ModelBenchmarkRunner", model_initializer)
    monkeypatch.setattr(scientific_evaluation, "FeatureAblationStudy", ablation_initializer)
    monkeypatch.setattr(scientific_evaluation, "analyze_classification_errors", error_analysis)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_scientific_evaluation.py", "--dataset-dir", str(tmp_path)],
    )

    with pytest.raises(InfeasiblePartitionError, match="performance_id for every recording"):
        scientific_evaluation.main()

    cache_loader.assert_not_called()
    model_initializer.assert_not_called()
    ablation_initializer.assert_not_called()
    error_analysis.assert_not_called()
