"""
Unit tests for dataset manifests, split integrity, and synthetic generation.
"""

from pathlib import Path
import pytest

from raga_intelligence.config import RAGA_LIST
from raga_intelligence.datasets.adapters import SyntheticDatasetAdapter
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

