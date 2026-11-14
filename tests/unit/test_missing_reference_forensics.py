import csv
import json
import os
from pathlib import Path

import pytest

from raga_intelligence.datasets import audit as dataset_audit
from raga_intelligence.datasets.missing_reference_forensics import (
    analyze_missing_references,
    render_forensics_markdown,
    write_forensics_reports,
)


FIELDS = [
    "recording_id",
    "file_name",
    "label",
    "split",
    "source",
    "source_recording_id",
    "performance_id",
    "performer",
    "performer_id",
]


def write_manifest(dataset: Path, rows: list[dict[str, str]]) -> Path:
    dataset.mkdir(parents=True, exist_ok=True)
    path = dataset / "metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def missing_row(path: str, **overrides: str) -> dict[str, str]:
    row = {
        "recording_id": "recording_1",
        "file_name": path,
        "label": "Bhairav",
        "split": "train",
        "source": "",
        "source_recording_id": "",
        "performance_id": "",
        "performer": "",
        "performer_id": "",
    }
    row.update(overrides)
    return row


def analyze(dataset: Path, search_root: Path | None = None) -> dict:
    return analyze_missing_references(dataset, search_root=search_root or dataset.parent)


def test_missing_relative_reference_is_unresolved_with_original_path(tmp_path):
    dataset = tmp_path / "dataset"
    original = "Raga/missing track.wav"
    write_manifest(dataset, [missing_row(original)])

    report = analyze(dataset, tmp_path)

    record = report["records"][0]
    assert record["original_reference_path"] == original
    assert record["normalized_reference_path"] == original
    assert record["resolved_expected_path"] == original
    assert record["forensic_classification"] == "unresolved"
    assert record["local_candidates"] == []
    assert report["summary"]["unresolved_references"] == 1


def test_absolute_missing_reference_is_redacted_and_not_resolved(tmp_path):
    dataset = tmp_path / "dataset"
    absolute = tmp_path / "outside" / "private track.wav"
    write_manifest(dataset, [missing_row(str(absolute))])

    report = analyze(dataset, tmp_path)
    record = report["records"][0]

    assert record["reference_path_type"] == "absolute"
    assert record["absolute_reference_redacted"] is True
    assert record["original_reference_path"] == "<absolute path redacted>"
    assert record["reference_path_sha256"]
    assert str(tmp_path) not in json.dumps(record)
    assert record["path_resolution_error"] == "audio path escapes dataset root"


def test_local_file_uri_is_redacted(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row("file:///private/audio/track.wav")])

    report = analyze(dataset, tmp_path)
    record = report["records"][0]

    assert record["reference_path_type"] == "absolute"
    assert record["original_reference_path"] == "<absolute path redacted>"
    assert "file:///private/audio" not in json.dumps(record)


def test_backslash_path_is_normalized_without_changing_original(tmp_path):
    dataset = tmp_path / "dataset"
    backslash = chr(92)
    write_manifest(dataset, [missing_row("Raga" + backslash + "missing.wav")])

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["original_reference_path"] == "Raga" + backslash + "missing.wav"
    assert record["normalized_reference_path"] == "Raga/missing.wav"
    assert record["resolved_expected_path"] == "Raga/missing.wav"


def test_exact_filename_elsewhere_is_only_an_alternate_candidate(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row("Raga/track.wav")])
    candidate = tmp_path / "archive" / "track.wav"
    candidate.parent.mkdir()
    candidate.write_bytes(b"not analyzed as the missing audio")

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["forensic_classification"] == "alternate_local_location_candidate"
    assert record["local_candidates"] == [{
        "path": "archive/track.wav",
        "evidence": ["exact_filename_match"],
        "identity_established": False,
    }]
    assert "does not establish" in record["unresolved_questions"][-1]


def test_different_extension_same_stem_is_similarity_only(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row("track.mp3")])
    (tmp_path / "track.wav").write_bytes(b"candidate bytes")

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["forensic_classification"] == "alternate_local_location_candidate"
    assert record["local_candidates"][0]["evidence"] == ["filename_similarity_only"]
    assert record["local_candidates"][0]["identity_established"] is False


def test_normalized_path_candidate_is_classified_as_convention_mismatch(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row("Raga/sub folder/track.wav")])
    candidate = tmp_path / "archive" / "Raga" / "sub folder" / "track.wav"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate")

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["forensic_classification"] == "path_convention_mismatch"
    assert "normalized_path_match" in record["local_candidates"][0]["evidence"]


def test_case_only_path_candidate_is_platform_sensitive(tmp_path):
    if os.name == "nt":
        pytest.skip("Windows resolves path components case-insensitively")
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row("raga/track.wav")])
    candidate = dataset / "Raga" / "track.wav"
    candidate.parent.mkdir()
    candidate.write_bytes(b"candidate")

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["forensic_classification"] == "path_convention_mismatch"
    assert "case_only_path_match" in record["local_candidates"][0]["evidence"]


def test_source_id_is_evidence_but_does_not_establish_identity(tmp_path):
    dataset = tmp_path / "dataset"
    row = missing_row("missing.wav", source="collection_x", source_recording_id="SRC-007")
    write_manifest(dataset, [row])
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "copy_SRC-007.wav").write_bytes(b"candidate")

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["forensic_classification"] == "alternate_local_location_candidate"
    assert "explicit_source_identifier_match" in record["local_candidates"][0]["evidence"]
    assert record["local_candidates"][0]["identity_established"] is False


def test_missing_source_metadata_remains_unresolved(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row("missing.wav")])

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["source"] is None
    assert record["source_recording_id"] is None
    assert record["forensic_classification"] == "unresolved"
    assert "Source metadata is absent." in record["unresolved_questions"]


def test_candidate_search_excludes_python_environment_audio(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row("missing.wav")])
    environment_audio = tmp_path / ".venv-linux" / "share" / "example.wav"
    environment_audio.parent.mkdir(parents=True)
    environment_audio.write_bytes(b"not a dataset candidate")

    report = analyze(dataset, tmp_path)

    assert report["candidate_analysis"]["local_audio_file_count"] == 0
    assert report["candidate_analysis"]["filesystem_scan_incomplete_paths"] == []
    assert report["records"][0]["forensic_classification"] == "unresolved"


def test_source_metadata_does_not_claim_upstream_availability(tmp_path):
    dataset = tmp_path / "dataset"
    row = missing_row("missing.wav", source="source_collection", source_recording_id="item-42")
    write_manifest(dataset, [row])

    report = analyze(dataset, tmp_path)
    record = report["records"][0]

    assert record["forensic_classification"] == "source_metadata_present_audio_unlocated"
    assert record["evidence_strength"] == "manifest_source_metadata"
    definition = report["classification_definitions"][record["forensic_classification"]]
    assert "Upstream source availability has not been verified" in definition
    assert "source_record_available_audio_missing" not in json.dumps(report)


def test_analysis_does_not_run_full_audio_audit_or_decode_present_audio(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [
        missing_row("present.wav", recording_id="present"),
        missing_row("missing.wav", recording_id="missing"),
    ])
    (dataset / "present.wav").write_bytes(b"present audio bytes")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("full audio audit or audio read was invoked")

    monkeypatch.setattr(dataset_audit, "audit_dataset", unexpected)
    monkeypatch.setattr(dataset_audit.librosa, "load", unexpected)
    monkeypatch.setattr(dataset_audit, "_sha256_file", unexpected)

    report = analyze(dataset, tmp_path)

    assert report["summary"]["present_references"] == 1
    assert report["summary"]["missing_references"] == 1


def test_distinct_redacted_absolute_paths_have_distinct_grouping_keys(tmp_path):
    dataset = tmp_path / "dataset"
    first = str(tmp_path / "private-a" / "track.wav")
    second = str(tmp_path / "private-b" / "track.wav")
    write_manifest(dataset, [missing_row(first), missing_row(second)])

    report = analyze(dataset, tmp_path)

    assert {row["original_reference_path"] for row in report["records"]} == {
        "<absolute path redacted>"
    }
    assert report["summary"]["unique_missing_paths"] == 2
    assert len({row["missing_path_fingerprint"] for row in report["records"]}) == 2


def test_normalized_equivalent_paths_share_missing_path_group(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [
        missing_row("  Raga\\missing.wav  ", recording_id="first"),
        missing_row("Raga/missing.wav", recording_id="second"),
    ])

    report = analyze(dataset, tmp_path)

    assert report["summary"]["unique_missing_paths"] == 1
    assert report["records"][0]["missing_path_fingerprint"] == report["records"][1]["missing_path_fingerprint"]


@pytest.mark.parametrize(
    ("reference", "path_type", "expected_normalized"),
    [
        ("  Raga\\track.wav  ", "relative", "Raga/track.wav"),
        (r"C:\\private\\track.wav", "absolute", None),
        ("file:///private/audio/track.wav", "absolute", None),
        ("Raga//nested/../track [ü].wav", "relative", "Raga/nested/../track [ü].wav"),
        ("", "missing", None),
    ],
)
def test_path_normalization_edge_cases(tmp_path, reference, path_type, expected_normalized):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row(reference)])

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["reference_path_type"] == path_type
    assert record["normalized_reference_path"] == expected_normalized
    if path_type == "absolute":
        assert record["original_reference_path"] == "<absolute path redacted>"
    if path_type == "missing":
        assert record["path_resolution_error"] == "missing audio path"


def test_report_discloses_search_exclusions_and_bounds_unresolved_claim(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row("missing.wav")])
    report = analyze(dataset, tmp_path)
    markdown = render_forensics_markdown(report)

    exclusions = report["candidate_analysis"]["excluded_directory_names"]
    assert {".git", ".venv", ".cache", "build", "reports"}.issubset(exclusions)
    assert "excluded locations and upstream availability were not checked" in report["classification_definitions"]["unresolved"]
    assert "Excluded directory names" in markdown
    assert "does not decode, load, or hash" in " ".join(report["limitations"]).lower()


def test_same_source_id_different_chunk_is_only_possible_segment(tmp_path):
    dataset = tmp_path / "dataset"
    row = missing_row("Raga/train_[source42]_chunk2.mp3")
    write_manifest(dataset, [row])
    sibling = dataset / "Raga" / "train_[source42]_chunk1.mp3"
    sibling.parent.mkdir()
    sibling.write_bytes(b"sibling candidate")

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["forensic_classification"] == "possible_segment_reference"
    assert "same_filename_source_id_different_chunk" in record["local_candidates"][0]["evidence"]
    assert record["local_candidates"][0]["identity_established"] is False


def test_shared_candidate_is_flagged_for_multiple_missing_references(tmp_path):
    dataset = tmp_path / "dataset"
    rows = [
        missing_row("track.wav", recording_id="recording_a"),
        missing_row("folder/track.wav", recording_id="recording_b"),
    ]
    write_manifest(dataset, rows)
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "track.wav").write_bytes(b"candidate")

    records = analyze(dataset, tmp_path)["records"]

    assert [record["forensic_classification"] for record in records] == [
        "duplicate_reference_candidate",
        "duplicate_reference_candidate",
    ]
    assert records[0]["local_candidates"][0]["path"] == records[1]["local_candidates"][0]["path"]


def test_missing_path_and_unusual_characters_are_preserved(tmp_path):
    dataset = tmp_path / "dataset"
    original = "Rāga one/track (live) #1.mp3"
    write_manifest(dataset, [missing_row(original)])

    record = analyze(dataset, tmp_path)["records"][0]

    assert record["original_reference_path"] == original
    assert record["normalized_reference_path"] == original
    assert record["forensic_classification"] == "unresolved"


def test_report_structure_and_analysis_are_deterministic(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, [missing_row("missing.wav")])

    first = analyze(dataset, tmp_path)
    second = analyze(dataset, tmp_path)

    assert first["records"] == second["records"]
    assert first["classification_counts"] == second["classification_counts"]
    assert first["summary"] == second["summary"]
    assert "## Record-level results" in render_forensics_markdown(first)
    json.dumps(first, ensure_ascii=False)
    json_path, markdown_path = write_forensics_reports(first, tmp_path / "reports")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["records"] == first["records"]
    assert markdown_path.is_file()


def test_analysis_does_not_modify_manifest_or_candidate_audio(tmp_path):
    dataset = tmp_path / "dataset"
    metadata = write_manifest(dataset, [missing_row("missing.wav")])
    candidate = tmp_path / "archive" / "missing.wav"
    candidate.parent.mkdir()
    candidate.write_bytes(b"unchanged candidate bytes")
    manifest_before = metadata.read_bytes()
    candidate_before = candidate.read_bytes()

    analyze(dataset, tmp_path)

    assert metadata.read_bytes() == manifest_before
    assert candidate.read_bytes() == candidate_before
