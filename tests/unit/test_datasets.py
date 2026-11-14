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
    assert all(item.license == "CC0-1.0" for item in manifest.items)
    assert all(item.rights_status == "verified" for item in manifest.items)
    assert all(item.performer_id for item in manifest.items)


def test_hindustani_video_id_is_not_performer_identity(tmp_path: Path):
    audio_name = "train_[video123]_chunk0.wav"
    (tmp_path / audio_name).write_bytes(b"audio")
    (tmp_path / "metadata.csv").write_text(
        f"file_name,label,tonic_hz\n{audio_name},Bhairav,220\n", encoding="utf-8"
    )
    manifest = HindustaniDatasetAdapter(dataset_dir=tmp_path).build_manifest()
    item = manifest.items[0]
    assert item.performance_id == "video123"
    assert item.performance_identity_source == "filename_pattern"
    assert item.performer is None
    assert item.performer_id is None
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
    assert item.performer_id is None
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
            performer_id="artist-a",
            performance_id="perf_01",
            performance_identity_source="manual_review",
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
            performer_id="artist-b",
            performance_id="perf_02",
            performance_identity_source="manual_review",
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
            performer_id="artist-a",
            performance_id="perf_03",
            duration_sec=10.0,
            sample_rate=24000,
            split="test",
        )
    )
    val2 = manifest.validate_splits()
    assert val2["valid_no_performer_leakage"] is False
    assert val2["performer_isolation_status"] == "failed"
    assert "artist-a" in val2["leaked_performers"]

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


def test_recording_fallback_is_not_verified_performance_identity():
    manifest = DatasetManifest(items=[
        DatasetItem(
            recording_id="recording-1", source="SaragaTest", file_path="one.wav",
            raga="Bhairav", performance_id="recording-1", performance_identity_source="unknown",
            duration_sec=1.0, sample_rate=24000, split="train",
        ),
        DatasetItem(
            recording_id="recording-2", source="SaragaTest", file_path="two.wav",
            raga="Yaman", performance_id="recording-2", performance_identity_source="unknown",
            duration_sec=1.0, sample_rate=24000, split="test",
        ),
    ])

    validation = manifest.validate_splits()

    assert validation["performance_metadata_complete"] is True
    assert validation["performance_grouping_isolation_status"] == "verified"
    assert validation["performance_identity_verified"] is False
    assert validation["performance_isolation_status"] == "not_verifiable"
    assert validation["valid_no_performance_leakage"] is False


def test_reviewed_performance_metadata_can_be_verified():
    manifest = DatasetManifest(items=[
        DatasetItem(
            recording_id="recording-1", source="fixture", file_path="one.wav",
            raga="Bhairav", performance_id="performance-1",
            performance_identity_source="manual_review", duration_sec=1.0,
            sample_rate=24000, split="train",
        ),
        DatasetItem(
            recording_id="recording-2", source="fixture", file_path="two.wav",
            raga="Yaman", performance_id="performance-2",
            performance_identity_source="manual_review", duration_sec=1.0,
            sample_rate=24000, split="test",
        ),
    ])

    validation = manifest.validate_splits()

    assert validation["performance_identity_verified"] is True
    assert validation["performance_grouping_isolation_status"] == "verified"
    assert validation["performance_isolation_status"] == "verified"


def test_split_summary_and_class_distribution():
    manifest = DatasetManifest()
    manifest.add_item(
        DatasetItem(
            recording_id="rec_01",
            source="Test",
            file_path="dummy1.wav",
            raga="Bhairav",
            performer="Artist 1",
            performer_id="artist-1",
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
            performer_id="artist-2",
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
            performer_id="artist-3",
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
                    performer_id=f"artist-{idx}",
                    performance_id=f"perf_{idx}",
                    performance_identity_source="manual_review",
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


def test_saraga_partition_uses_stable_performer_ids_not_display_names():
    from raga_intelligence.datasets.saraga import SaragaHindustaniAdapter

    adapter = SaragaHindustaniAdapter(dataset_dir="dummy", seed=42)
    items = [
        DatasetItem(
            recording_id=f"rec-{index}", source="SaragaTest", file_path=f"{index}.wav",
            raga="Bhairav", performer="Same display name", performer_id=f"artist-{index}",
            performance_id=f"performance-{index}", duration_sec=1.0, sample_rate=24000,
        )
        for index in range(30)
    ]

    partitioned = adapter._partition_items(items)

    assert len({item.split for item in partitioned}) > 1


def test_saraga_missing_performer_id_keeps_performance_grouping():
    from raga_intelligence.datasets.saraga import SaragaHindustaniAdapter

    adapter = SaragaHindustaniAdapter(dataset_dir="dummy", seed=42)
    items = [
        DatasetItem(
            recording_id=f"rec-{index}", source="SaragaTest", file_path=f"{index}.wav",
            raga="Bhairav", performer="Same display name", performance_id="shared-performance",
            duration_sec=1.0, sample_rate=24000,
        )
        for index in range(4)
    ]

    partitioned = adapter._partition_items(items)

    assert len({item.split for item in partitioned}) == 1


def test_saraga_performer_identity_requires_explicit_stable_source_id():
    from raga_intelligence.datasets.saraga import _extract_performer_identity

    display_name, performer_id = _extract_performer_identity(
        {"lead_artist": {"name": "Known Artist"}}
    )
    assert display_name == "Known Artist"
    assert performer_id is None

    display_name, performer_id = _extract_performer_identity(
        {"lead_artist": {"name": "Known Artist", "id": "source-artist-42"}}
    )
    assert display_name == "Known Artist"
    assert performer_id == "source-artist-42"


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"performance_id": "explicit-performance"}, ("explicit-performance", "metadata")),
        (
            {"performance": {"id": "performance-42", "title": "Evening Concert"}},
            ("performance-42", "metadata"),
        ),
        (
            {"concert": {"id": "concert-9", "title": "Live at Hall"}},
            ("concert-9", "metadata"),
        ),
        ({"performance": {"title": "Evening Concert"}}, ("recording-1", "unknown")),
        ({"performance_id": {"title": "Not an ID"}}, ("recording-1", "unknown")),
        ({"concert": {"title": "Live at Hall"}}, ("recording-1", "unknown")),
        ({"album": {"id": "album-id", "title": "Shared Album"}}, ("recording-1", "unknown")),
        ({"release": {"id": "release-id", "title": "Shared Release"}}, ("recording-1", "unknown")),
        ({"work": {"id": "work-id", "title": "Composition"}}, ("recording-1", "unknown")),
        ({"composition": {"title": "Composition"}}, ("recording-1", "unknown")),
        ({"concert": "Concert title"}, ("recording-1", "unknown")),
        ({}, ("recording-1", "unknown")),
    ],
)
def test_saraga_performance_identity_uses_only_explicit_performance_ids(metadata, expected):
    from raga_intelligence.datasets.saraga import _extract_performance_identity

    assert _extract_performance_identity(metadata, "recording-1") == expected


def test_saraga_performance_object_id_wins_and_album_titles_do_not_group():
    from raga_intelligence.datasets.saraga import _extract_performance_identity

    first = _extract_performance_identity(
        {"performance": {"id": "performance-1", "title": "Same Album Tour"}, "album": "Shared"},
        "recording-1",
    )
    second = _extract_performance_identity(
        {"performance": {"id": "performance-2", "title": "Same Album Tour"}, "album": "Shared"},
        "recording-2",
    )

    assert first == ("performance-1", "metadata")
    assert second == ("performance-2", "metadata")
    assert first[0] != second[0]


def test_saraga_adapter_keeps_titles_as_descriptions_and_uses_recording_fallback(tmp_path):
    import json

    from raga_intelligence.datasets.saraga import SaragaHindustaniAdapter

    records = {
        "title-only": {
            "raag": "Bhairav",
            "performance": {"title": "Evening Concert"},
            "album": {"title": "Shared Album"},
            "lead_artist": {"name": "Display Only Artist"},
            "duration": 1.0,
            "sample_rate": 24000,
        },
        "explicit-id": {
            "raag": "Yaman",
            "concert": {"id": "concert-81", "title": "Morning Concert"},
            "release": {"title": "Shared Release"},
            "duration": 1.0,
            "sample_rate": 24000,
        },
    }
    records.update({
        f"album-only-{index}": {
            "raag": "Bhairav",
            "album": {"title": "One Shared Album"},
            "duration": 1.0,
            "sample_rate": 24000,
        }
        for index in range(8)
    })
    for recording_id, metadata in records.items():
        (tmp_path / f"{recording_id}.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (tmp_path / f"{recording_id}.wav").write_bytes(b"fixture audio")

    manifest = SaragaHindustaniAdapter(dataset_dir=tmp_path, seed=42).build_manifest()
    by_recording = {item.recording_id: item for item in manifest.items}

    assert by_recording["title-only"].performance_id == "title-only"
    assert by_recording["title-only"].performance_identity_source == "unknown"
    assert by_recording["title-only"].performer == "Display Only Artist"
    assert by_recording["title-only"].performer_id is None
    assert by_recording["explicit-id"].performance_id == "concert-81"
    assert by_recording["explicit-id"].performance_identity_source == "metadata"
    album_only = [item for item in manifest.items if item.recording_id.startswith("album-only-")]
    assert len({item.split for item in album_only}) > 1
    validation = manifest.validate_splits()
    assert validation["performance_isolation_status"] == "not_verifiable"


def test_saraga_fallback_identity_source_survives_manifest_json_roundtrip():
    from raga_intelligence.datasets.saraga import _extract_performance_identity

    fallback_id, identity_source = _extract_performance_identity({"album": "Shared"}, "recording-1")
    manifest = DatasetManifest(items=[
        DatasetItem(
            recording_id="recording-1", source="SaragaTest", file_path="one.wav",
            raga="Bhairav", performance_id=fallback_id,
            performance_identity_source=identity_source, duration_sec=1.0, sample_rate=24000,
        )
    ])

    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())

    assert restored.items[0].performance_id == "recording-1"
    assert restored.items[0].performance_identity_source == "unknown"
    assert restored.validate_splits()["performance_isolation_status"] == "not_verifiable"


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
            performer_id="shared-performer",
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
            performer_id="shared-performer",
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
    assert validation["valid_no_performance_leakage"] is False
    assert validation["performance_isolation_status"] == "not_verifiable"
    assert validation["valid_no_performer_leakage"] is False
    assert validation["performer_isolation_status"] == "not_verifiable"
    assert validation["train_recordings_count"] == 360
    assert validation["test_recordings_count"] == 30

