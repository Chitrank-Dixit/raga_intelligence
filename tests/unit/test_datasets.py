"""
Unit tests for dataset manifests, split integrity, and synthetic generation.
"""

from pathlib import Path
import pytest

from raga_intelligence.config import RAGA_LIST
from raga_intelligence.datasets.adapters import SyntheticDatasetAdapter
from raga_intelligence.datasets.hindustani_dataset import HindustaniDatasetAdapter
from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest


def test_manifest_leakage_detection():
    manifest = DatasetManifest()
    manifest.add_item(
        DatasetItem(
            recording_id="rec_01",
            source="Test",
            file_path="dummy1.wav",
            raga="Bhairav",
            performer="Artist A",
            duration_sec=10.0,
            sample_rate=24000,
            split="train",
        )
    )
    manifest.add_item(
        DatasetItem(
            recording_id="rec_02",
            source="Test",
            file_path="dummy2.wav",
            raga="Bhairav",
            performer="Artist B",
            duration_sec=10.0,
            sample_rate=24000,
            split="test",
        )
    )

    validation = manifest.validate_splits()
    assert validation["valid_no_recording_leakage"] is True
    assert len(validation["leaked_recordings"]) == 0

    # Introduce intentional leakage
    manifest.add_item(
        DatasetItem(
            recording_id="rec_01",
            source="Test",
            file_path="dummy1.wav",
            raga="Bhairav",
            performer="Artist A",
            duration_sec=10.0,
            sample_rate=24000,
            split="test",
        )
    )
    leaked_validation = manifest.validate_splits()
    assert leaked_validation["valid_no_recording_leakage"] is False
    assert "rec_01" in leaked_validation["leaked_recordings"]


def test_synthetic_dataset_adapter(tmp_path: Path):
    adapter = SyntheticDatasetAdapter(
        output_dir=tmp_path / "synthetic",
        recordings_per_raga=2,
        duration_sec=1.5,
        sample_rate=24000,
    )
    manifest = adapter.build_manifest()
    assert len(manifest.items) == len(RAGA_LIST) * 2

    # Check files exist on disk
    for item in manifest.items:
        assert Path(item.file_path).exists()
        assert item.raga in RAGA_LIST

    validation = manifest.validate_splits()
    assert validation["valid_no_recording_leakage"] is True


def test_hindustani_video_id_is_not_performer_identity(tmp_path: Path):
    audio_name = "train_[video123]_chunk0.wav"
    (tmp_path / audio_name).write_bytes(b"audio")
    (tmp_path / "metadata.csv").write_text(
        f"file_name,label,tonic_hz\n{audio_name},Bhairav,220\n", encoding="utf-8"
    )
    manifest = HindustaniDatasetAdapter(dataset_dir=tmp_path).build_manifest()
    item = manifest.items[0]
    assert item.performance_id == "video123"
    assert item.performer is None
    assert manifest.validate_splits()["performer_isolation_status"] == "not_verifiable"


def test_hindustani_artist_metadata_is_kept_separate_from_video_id(tmp_path: Path):
    audio_name = "train_[video123]_chunk0.wav"
    (tmp_path / audio_name).write_bytes(b"audio")
    (tmp_path / "metadata.csv").write_text(
        f"file_name,label,tonic_hz,artist\n{audio_name},Bhairav,220,Artist Name\n",
        encoding="utf-8",
    )
    item = HindustaniDatasetAdapter(dataset_dir=tmp_path).build_manifest().items[0]
    assert item.performer == "Artist Name"
    assert item.performance_id == "video123"


def test_manifest_performer_and_performance_leakage_detection():
    manifest = DatasetManifest()
    manifest.add_item(
        DatasetItem(
            recording_id="rec_01",
            source="Test",
            file_path="dummy1.wav",
            raga="Bhairav",
            performer="Ustad A",
            performance_id="perf_01",
            duration_sec=10.0,
            sample_rate=24000,
            split="train",
        )
    )
    manifest.add_item(
        DatasetItem(
            recording_id="rec_02",
            source="Test",
            file_path="dummy2.wav",
            raga="Yaman",
            performer="Pandit B",
            performance_id="perf_02",
            duration_sec=10.0,
            sample_rate=24000,
            split="test",
        )
    )

    val = manifest.validate_splits()
    assert val["valid_no_recording_leakage"] is True
    assert val["valid_no_performance_leakage"] is True
    assert val["valid_no_performer_leakage"] is True
    assert val["performer_metadata_complete"] is True
    assert val["performer_isolation_status"] == "verified"
    assert val["performance_isolation_status"] == "verified"

    # Deliberate performer leakage: Ustad A in test split with new performance
    manifest.add_item(
        DatasetItem(
            recording_id="rec_03",
            source="Test",
            file_path="dummy3.wav",
            raga="Bhairav",
            performer="Ustad A",
            performance_id="perf_03",
            duration_sec=10.0,
            sample_rate=24000,
            split="test",
        )
    )
    val2 = manifest.validate_splits()
    assert val2["valid_no_performer_leakage"] is False
    assert val2["performer_isolation_status"] == "failed"
    assert "Ustad A" in val2["leaked_performers"]

    # Deliberate performance leakage: perf_01 in test split with new performer
    manifest2 = DatasetManifest()
    manifest2.add_item(
        DatasetItem(
            recording_id="rec_10",
            source="Test",
            file_path="dummy10.wav",
            raga="Bhairav",
            performer="Artist X",
            performance_id="concert_alpha",
            duration_sec=10.0,
            sample_rate=24000,
            split="train",
        )
    )
    manifest2.add_item(
        DatasetItem(
            recording_id="rec_11",
            source="Test",
            file_path="dummy11.wav",
            raga="Bhairav",
            performer="Artist Y",
            performance_id="concert_alpha",
            duration_sec=10.0,
            sample_rate=24000,
            split="test",
        )
    )
    val3 = manifest2.validate_splits()
    assert val3["valid_no_performance_leakage"] is False
    assert val3["performance_isolation_status"] == "failed"
    assert "concert_alpha" in val3["leaked_performances"]


def test_incomplete_metadata_cannot_pass_legacy_leakage_booleans():
    manifest = DatasetManifest(items=[
        DatasetItem(
            recording_id="rec_unknown",
            source="Test",
            file_path="unknown.wav",
            raga="Bhairav",
            performer=None,
            performance_id=None,
            duration_sec=10.0,
            sample_rate=24000,
            split="train",
        )
    ])

    audit = manifest.validate_splits()
    assert audit["performer_isolation_status"] == "not_verifiable"
    assert audit["performance_isolation_status"] == "not_verifiable"
    assert audit["valid_no_performer_leakage"] is False
    assert audit["valid_no_performance_leakage"] is False


def test_split_summary_and_class_distribution():
    manifest = DatasetManifest()
    manifest.add_item(
        DatasetItem(
            recording_id="rec_01",
            source="Test",
            file_path="dummy1.wav",
            raga="Bhairav",
            performer="Artist 1",
            performance_id="p1",
            duration_sec=5.0,
            sample_rate=24000,
            split="train",
        )
    )
    manifest.add_item(
        DatasetItem(
            recording_id="rec_02",
            source="Test",
            file_path="dummy2.wav",
            raga="Bhairav",
            performer="Artist 2",
            performance_id="p2",
            duration_sec=5.0,
            sample_rate=24000,
            split="test",
        )
    )
    manifest.add_item(
        DatasetItem(
            recording_id="rec_03",
            source="Test",
            file_path="dummy3.wav",
            raga="Yaman",
            performer="Artist 3",
            performance_id="p3",
            duration_sec=5.0,
            sample_rate=24000,
            split="val",
        )
    )

    val = manifest.validate_splits()
    summary = val["split_summary"]
    assert summary["recordings_count"] == {"train": 1, "val": 1, "test": 1, "total": 3}
    assert summary["performers_count"] == {"train": 1, "val": 1, "test": 1, "total": 3}
    assert summary["performances_count"] == {"train": 1, "val": 1, "test": 1, "total": 3}
    assert "Bhairav" in summary["raga_distribution"]
    assert summary["raga_distribution"]["Bhairav"]["train"] == 1
    assert summary["raga_distribution"]["Bhairav"]["test"] == 1
    assert summary["raga_distribution"]["Yaman"]["val"] == 1


def test_saraga_deterministic_group_partition():
    from raga_intelligence.datasets.saraga import SaragaHindustaniAdapter

    adapter = SaragaHindustaniAdapter(dataset_dir="dummy", seed=42)

    # Create synthetic item pool simulating Saraga metadata with multi-track performers
    items = []
    ragas = ["Bhairav", "Yaman", "Bhoopali", "Kafi"]
    performers = [f"Artist_{i}" for i in range(8)]

    for idx, performer in enumerate(performers):
        for track_num in range(3):
            items.append(
                DatasetItem(
                    recording_id=f"rec_{idx}_{track_num}",
                    source="SaragaTest",
                    file_path=f"path_{idx}_{track_num}.wav",
                    raga=ragas[(idx + track_num) % len(ragas)],
                    performer=performer,
                    performance_id=f"perf_{idx}",
                    duration_sec=10.0,
                    sample_rate=24000,
                    split="train",
                )
            )

    # 1. Verify partition respects performer and performance boundaries
    partitioned1 = adapter._partition_items(list(items))
    m1 = DatasetManifest(items=partitioned1)
    val1 = m1.validate_splits()
    assert val1["valid_no_recording_leakage"] is True
    assert val1["valid_no_performance_leakage"] is True
    assert val1["valid_no_performer_leakage"] is True

    # 2. Verify deterministic output for same seed
    adapter_same = SaragaHindustaniAdapter(dataset_dir="dummy", seed=42)
    partitioned2 = adapter_same._partition_items(list(items))
    assert [it.split for it in partitioned1] == [it.split for it in partitioned2]


def test_training_fails_on_leakage():
    from raga_intelligence.training.trainer import _assert_no_split_leakage

    manifest = DatasetManifest()
    manifest.add_item(
        DatasetItem(
            recording_id="rec_01",
            source="Test",
            file_path="dummy1.wav",
            raga="Bhairav",
            performer="Shared Performer",
            duration_sec=10.0,
            sample_rate=24000,
            split="train",
        )
    )
    manifest.add_item(
        DatasetItem(
            recording_id="rec_02",
            source="Test",
            file_path="dummy2.wav",
            raga="Bhairav",
            performer="Shared Performer",
            duration_sec=10.0,
            sample_rate=24000,
            split="test",
        )
    )

    with pytest.raises(ValueError, match="Fatal dataset leakage"):
        _assert_no_split_leakage(manifest)


def test_hindustani_dataset_adapter():
    data_dir = Path("./data/hindustani_dataset")
    if not data_dir.exists():
        pytest.skip(f"Dataset directory '{data_dir}' is not mounted in current environment.")

    from raga_intelligence.datasets.hindustani_dataset import HindustaniDatasetAdapter
    adapter = HindustaniDatasetAdapter(dataset_dir=data_dir)
    manifest = adapter.build_manifest()
    assert len(manifest.items) == 390
    validation = manifest.validate_splits()
    assert validation["valid_no_recording_leakage"] is True
    assert validation["valid_no_performance_leakage"] is True
    assert validation["valid_no_performer_leakage"] is False
    assert validation["performer_isolation_status"] == "not_verifiable"
    assert validation["train_recordings_count"] == 360
    assert validation["test_recordings_count"] == 30

