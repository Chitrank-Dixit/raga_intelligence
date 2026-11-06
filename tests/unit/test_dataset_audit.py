import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from raga_intelligence.config import RAGA_LIST
from raga_intelligence.datasets.audit import audit_dataset, render_audit_markdown
from scripts.audit_dataset import main


CSV_FIELDS = [
    "recording_id", "file_name", "label", "tonic_hz", "performance_id", "performer",
    "source", "source_url", "license", "attribution", "split",
]


def write_audio(path: Path, value: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(4000, value, dtype=np.float32), 8000, format="WAV")


def write_manifest(root: Path, rows: list[dict], fields: list[str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def valid_rows(root: Path) -> list[dict]:
    rows = []
    for raga_index, raga in enumerate(RAGA_LIST):
        for split, suffix in (("train", "a"), ("test", "b")):
            name = f"{split}_[{raga.replace(' ', '_')}_{suffix}].wav"
            write_audio(root / name, 0.05 + (raga_index * 2 + (suffix == "b")) * 0.02)
            rows.append({
                "recording_id": f"{raga}_{suffix}", "file_name": name, "label": raga,
                "tonic_hz": "220", "performance_id": f"performance_{raga}_{suffix}",
                "performer": f"artist_{raga}_{suffix}", "source": "fixture",
                "source_url": "https://example.invalid/source", "license": "CC0-1.0",
                "attribution": "Fixture author", "split": split,
            })
    return rows


def test_fully_valid_dataset_has_expected_inventory_and_readiness(tmp_path):
    write_manifest(tmp_path, valid_rows(tmp_path))

    result = audit_dataset(tmp_path)

    assert result["summary"]["total_manifest_records"] == 20
    assert result["summary"]["valid_records"] == 20
    assert result["summary"]["physical_audio_files_present"] == 20
    assert result["evaluation_readiness"]["scientific_evaluation_ready"] is True
    assert result["evaluation_readiness"]["performance_isolation_status"] == "verified"
    assert result["evaluation_readiness"]["performer_isolation_status"] == "verified"
    assert result["annotation_quality"]["unknown_raga_labels"] == []


def test_noncanonical_missing_rows_do_not_expand_canonical_evaluation_scope(tmp_path):
    rows = valid_rows(tmp_path)
    rows.append({
        "recording_id": "legacy_absent", "file_name": "Legacy/legacy_absent.mp3",
        "label": "Noncanonical Raga", "tonic_hz": "220", "performance_id": "legacy_perf",
        "performer": "unknown", "source": "legacy", "source_url": "", "license": "",
        "attribution": "", "split": "train",
    })
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["evaluation_readiness"]["evaluation_scope_record_count"] == 20
    assert result["evaluation_readiness"]["scientific_evaluation_ready"] is True
    assert result["dataset_has_critical_errors"] is True


def test_label_completeness_distinguishes_present_noncanonical_labels(tmp_path):
    rows = valid_rows(tmp_path)
    rows.extend([
        {
            "recording_id": "noncanonical", "file_name": "noncanonical.wav",
            "label": "Invented Raga", "tonic_hz": "220", "performance_id": "extra_perf",
            "performer": "unknown", "source": "fixture", "source_url": "",
            "license": "", "attribution": "", "split": "train",
        },
        {
            "recording_id": "missing_label", "file_name": "missing_label.wav",
            "label": "", "tonic_hz": "220", "performance_id": "missing_perf",
            "performer": "unknown", "source": "fixture", "source_url": "",
            "license": "", "attribution": "", "split": "train",
        },
    ])
    write_audio(tmp_path / "noncanonical.wav")
    write_audio(tmp_path / "missing_label.wav")
    write_manifest(tmp_path, rows)

    completeness = audit_dataset(tmp_path)["annotation_quality"]

    assert completeness["metadata_completeness"] == {
        "present": 21, "missing": 1, "total": 22, "fraction": 21 / 22,
    }
    assert completeness["canonical_label_coverage"] == {
        "present": 20, "missing": 2, "total": 22, "fraction": 20 / 22,
    }


def test_missing_audio_file_is_reported_and_has_no_audio_fingerprint(tmp_path):
    rows = valid_rows(tmp_path)
    (tmp_path / rows[0]["file_name"]).unlink()
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["summary"]["physical_audio_files_missing"] == 1
    assert result["audio_content_fingerprint"] is None
    assert any(issue["code"] == "missing_audio" for issue in result["issues"])


def test_unreadable_audio_is_reported_as_invalid(tmp_path):
    rows = valid_rows(tmp_path)
    (tmp_path / rows[0]["file_name"]).write_bytes(b"not audio")
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["summary"]["unreadable_audio_files"] == 1
    assert result["audio_integrity"]["unreadable_files"][0]["recording_id"] == rows[0]["recording_id"]
    assert str(tmp_path) not in result["audio_integrity"]["unreadable_files"][0]["error"]


def test_duplicate_recording_ids_are_reported(tmp_path):
    rows = valid_rows(tmp_path)
    rows[1]["recording_id"] = rows[0]["recording_id"]
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["summary"]["duplicate_recording_id_count"] == 1
    assert rows[0]["recording_id"] in result["identity_coverage"]["recording"]["duplicate_recording_ids"]


def test_explicit_unknown_recording_id_is_not_replaced_from_filename(tmp_path):
    rows = valid_rows(tmp_path)
    rows[0]["recording_id"] = "unknown"
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    first = result["records"][0]
    assert first["recording_identity_source"] == "missing"
    assert first["recording_id"].startswith("__missing_recording_id_row_")
    assert result["evaluation_readiness"]["missing_recording_id_count"] == 1
    assert result["summary"]["unique_recording_ids"] == 19


def test_canonical_record_without_valid_split_blocks_readiness(tmp_path):
    rows = valid_rows(tmp_path)
    rows.append({
        "recording_id": "unassigned", "file_name": "unassigned.wav", "label": "Bhairav",
        "tonic_hz": "220", "performance_id": "unassigned_performance",
        "performer": "unassigned_performer", "source": "fixture", "source_url": "",
        "license": "CC0-1.0", "attribution": "", "split": "",
    })
    write_audio(tmp_path / "unassigned.wav")
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    readiness = result["evaluation_readiness"]
    assert readiness["missing_split_assignment_count"] == 1
    assert readiness["recording_isolation_status"] == "not_verifiable"
    assert readiness["performance_isolation_status"] == "not_verifiable"
    assert readiness["scientific_evaluation_ready"] is False
    assert any(issue["code"] == "missing_evaluation_split" for issue in result["issues"])


def test_exact_duplicate_audio_is_found_across_recording_ids(tmp_path):
    rows = valid_rows(tmp_path)
    shutil.copyfile(tmp_path / rows[0]["file_name"], tmp_path / rows[1]["file_name"])
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["summary"]["exact_duplicate_audio_groups"] == 1
    digest = hashlib.sha256((tmp_path / rows[0]["file_name"]).read_bytes()).hexdigest()
    assert len(result["audio_integrity"]["duplicate_content_groups"][digest]) == 2


def test_missing_performance_and_performer_identity_remain_unknown(tmp_path):
    rows = valid_rows(tmp_path)
    for row in rows:
        row["performance_id"] = "unknown"
        row["performer"] = " "
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["identity_coverage"]["performance"]["missing_count"] == 20
    assert result["identity_coverage"]["performer"]["missing_count"] == 20
    assert result["evaluation_readiness"]["performance_isolation_status"] == "not_verifiable"
    assert result["evaluation_readiness"]["performer_isolation_status"] == "not_verifiable"


def test_filename_performance_rule_is_disclosed_and_never_populates_performer(tmp_path):
    rows = valid_rows(tmp_path)
    for row in rows:
        row.pop("performance_id")
        row["performer"] = "unknown"
    write_manifest(tmp_path, rows, [field for field in CSV_FIELDS if field != "performance_id"])

    result = audit_dataset(tmp_path)

    assert result["identity_coverage"]["performance"]["missing_count"] == 0
    assert result["identity_coverage"]["performance"]["identity_source_counts"] == {"filename_pattern": 20}
    assert result["identity_coverage"]["performer"]["missing_count"] == 20
    assert all(record["performer_id"] is None for record in result["records"])


def test_performance_leakage_across_splits_is_reported(tmp_path):
    rows = valid_rows(tmp_path)
    rows[0]["performance_id"] = rows[1]["performance_id"]
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["evaluation_readiness"]["performance_isolation_status"] == "failed"
    assert any(issue["code"] == "performance_leakage" for issue in result["issues"])


def test_performer_leakage_across_splits_is_reported(tmp_path):
    rows = valid_rows(tmp_path)
    rows[0]["performer"] = rows[1]["performer"]
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["evaluation_readiness"]["performer_isolation_status"] == "failed"
    assert any(issue["code"] == "performer_leakage" for issue in result["issues"])


def test_missing_and_unknown_raga_labels_are_distinguished(tmp_path):
    rows = valid_rows(tmp_path)
    rows[0]["label"] = ""
    rows[1]["label"] = "Invented Raga"
    rows[1]["tonic_hz"] = "-1"
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["annotation_quality"]["missing_raga_labels"] == 1
    assert result["annotation_quality"]["unknown_raga_labels"] == ["Invented Raga"]
    assert any(issue["code"] == "missing_raga_labels" for issue in result["issues"])
    assert any(issue["code"] == "invalid_metadata" for issue in result["issues"])
    assert any(issue["code"] == "noncanonical_raga_labels" for issue in result["issues"])


def test_nonfinite_tonic_and_invalid_split_are_marked_invalid(tmp_path):
    rows = valid_rows(tmp_path)
    rows[0]["tonic_hz"] = "NaN"
    rows[1]["split"] = "development"
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["summary"]["missing_or_invalid_metadata_records"] == 2
    assert "tonic_hz" in result["records"][0]["invalid_metadata"]
    assert "split" in result["records"][1]["invalid_metadata"]


def test_incomplete_license_and_provenance_are_flagged(tmp_path):
    rows = valid_rows(tmp_path)
    for row in rows:
        row["license"] = ""
        row["source_url"] = ""
        row["attribution"] = ""
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["provenance"]["license_completeness"]["missing"] == 20
    assert result["provenance"]["source_url_completeness"]["missing"] == 20
    assert result["provenance"]["attribution_completeness"]["missing"] == 20
    assert len(result["provenance"]["rights_uncertain_recording_ids"]) == 20
    assert result["provenance"]["lifecycle_classification_available"] is False
    assert result["provenance"]["legacy_record_count"] is None
    assert result["provenance"]["unclassified_lifecycle_record_count"] == 20


def test_markdown_report_explains_unavailable_lifecycle_classification(tmp_path):
    write_manifest(tmp_path, valid_rows(tmp_path))
    result = audit_dataset(tmp_path)

    report = render_audit_markdown(result)

    assert "Record lifecycle classification: unavailable" in report
    assert result["provenance"]["note"] in report


def test_markdown_report_shows_lifecycle_counts_when_status_field_exists(tmp_path):
    rows = valid_rows(tmp_path)
    for row in rows:
        row["record_status"] = "legacy"
    rows[0]["record_status"] = "accepted"
    write_manifest(tmp_path, rows, [*CSV_FIELDS, "record_status"])
    result = audit_dataset(tmp_path)

    report = render_audit_markdown(result)

    assert "Legacy records: 19" in report
    assert "Newly accepted records: 1" in report
    assert "Records without a lifecycle status: 0" in report


def test_local_paths_in_source_urls_are_redacted(tmp_path):
    rows = valid_rows(tmp_path)
    rows[0]["source"] = "/var/lib/private/source"
    rows[0]["source_url"] = "/home/private/audio.mp3"
    rows[0]["attribution"] = "/mnt/user/private/credit.txt"
    write_manifest(tmp_path, rows)

    result = audit_dataset(tmp_path)

    assert result["records"][0]["provenance"]["source_url"] == "<local-path>"
    assert result["records"][0]["provenance"]["attribution"] == "<local-path>"
    assert result["records"][0]["source"] == "<local-path>"
    assert str(tmp_path) not in json.dumps(result)


def test_empty_dataset_is_reported_and_fingerprints_are_not_fabricated(tmp_path):
    write_manifest(tmp_path, [])

    result = audit_dataset(tmp_path)

    assert result["summary"]["total_manifest_records"] == 0
    assert result["audio_content_fingerprint"] is None
    assert any(issue["code"] == "empty_dataset" for issue in result["issues"])


def test_report_schema_and_fingerprints_are_reproducible(tmp_path):
    write_manifest(tmp_path, valid_rows(tmp_path))

    first = audit_dataset(tmp_path)
    second = audit_dataset(tmp_path)

    assert first["manifest_fingerprint"] == second["manifest_fingerprint"]
    assert first["audio_content_fingerprint"] == second["audio_content_fingerprint"]
    assert first["manifest_fingerprint_policy"] == "manifest_metadata_v1"
    assert first["audio_content_fingerprint_policy"] == "relative_path_sha256_v1"
    assert first["schema_version"] == "1.0"
    report = render_audit_markdown(first)
    assert "Evaluation readiness" in report
    assert "Raga label metadata present: 20/20 (100.0%)" in report
    assert "Labels in canonical vocabulary: 20/20 (100.0%)" in report
    assert str(tmp_path) not in report
    assert json.loads(json.dumps(first))["summary"]["total_manifest_records"] == 20


def test_fingerprints_do_not_depend_on_absolute_dataset_root(tmp_path):
    first_root = tmp_path / "one" / "same_dataset"
    second_root = tmp_path / "elsewhere" / "same_dataset"
    rows = valid_rows(first_root)
    second_root.mkdir(parents=True)
    for row in rows:
        shutil.copyfile(first_root / row["file_name"], second_root / row["file_name"])
    write_manifest(first_root, rows)
    write_manifest(second_root, rows)

    first = audit_dataset(first_root)
    second = audit_dataset(second_root)

    assert first["manifest_fingerprint"] == second["manifest_fingerprint"]
    assert first["audio_content_fingerprint"] == second["audio_content_fingerprint"]


def test_manifest_fingerprint_does_not_depend_on_csv_row_order(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    rows = valid_rows(first_root)
    second_root.mkdir()
    for row in rows:
        shutil.copyfile(first_root / row["file_name"], second_root / row["file_name"])
    write_manifest(first_root, rows)
    write_manifest(second_root, list(reversed(rows)))

    assert audit_dataset(first_root)["manifest_fingerprint"] == audit_dataset(second_root)["manifest_fingerprint"]


def test_cli_writes_both_reports_and_uses_error_exit_code(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, valid_rows(dataset))
    output = tmp_path / "reports"

    assert main(["--dataset-dir", str(dataset), "--output-dir", str(output)]) == 0
    assert (output / "dataset_audit.json").is_file()
    assert (output / "dataset_audit.md").is_file()

    invalid_root = tmp_path / "invalid"
    write_manifest(invalid_root, valid_rows(invalid_root)[:1])
    assert main([
        "--dataset-dir", str(invalid_root), "--output-dir", str(output / "invalid")
    ]) == 1


def test_cli_returns_clear_error_for_missing_dataset(tmp_path, capsys):
    result = main(["--dataset-dir", str(tmp_path / "does-not-exist"), "--output-dir", str(tmp_path / "out")])

    assert result == 2
    assert "Dataset directory does not exist" in capsys.readouterr().err


def test_cli_rejects_report_output_inside_dataset_root(tmp_path, capsys):
    write_manifest(tmp_path, [])

    result = main(["--dataset-dir", str(tmp_path), "--output-dir", str(tmp_path / "audit")])

    assert result == 2
    assert "outside the dataset root" in capsys.readouterr().err


def test_audit_never_modifies_audio_or_manifest(tmp_path):
    rows = valid_rows(tmp_path)
    metadata_path = write_manifest(tmp_path, rows)
    before_manifest = metadata_path.read_bytes()
    audio_hashes = {
        row["file_name"]: hashlib.sha256((tmp_path / row["file_name"]).read_bytes()).hexdigest()
        for row in rows
    }

    audit_dataset(tmp_path)

    assert metadata_path.read_bytes() == before_manifest
    assert audio_hashes == {
        row["file_name"]: hashlib.sha256((tmp_path / row["file_name"]).read_bytes()).hexdigest()
        for row in rows
    }
