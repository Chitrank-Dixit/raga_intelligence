import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_scientific_evaluation import (
    build_evaluation_manifest,
    evaluation_mode_banner,
    format_leakage_audit,
)
from raga_intelligence.features.cache import compute_manifest_content_fingerprint


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
