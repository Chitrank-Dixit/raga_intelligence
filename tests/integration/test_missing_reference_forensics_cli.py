import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def test_forensics_cli_writes_reports_without_mutating_manifest(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = dataset / "metadata.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["recording_id", "file_name", "label", "split"],
        )
        writer.writeheader()
        writer.writerow({
            "recording_id": "missing_recording",
            "file_name": "Raga/missing.wav",
            "label": "Bhairav",
            "split": "train",
        })
    manifest_before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "reports"
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "analyze_missing_audio_references.py"
    )
    environment = os.environ.copy()
    environment["DEBUG"] = "false"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dataset-dir",
            str(dataset),
            "--search-root",
            str(tmp_path),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "1 missing reference(s)" in result.stdout
    json_report = output / "d2_1_missing_reference_forensics.json"
    markdown_report = output / "d2_1_missing_reference_forensics.md"
    assert json_report.is_file()
    assert markdown_report.is_file()
    report = json.loads(json_report.read_text(encoding="utf-8"))
    assert report["summary"]["manifest_records_examined"] == 1
    assert report["summary"]["missing_references"] == 1
    assert report["records"][0]["forensic_classification"] == "unresolved"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == manifest_before


def test_forensics_cli_rejects_report_output_inside_dataset(tmp_path, capsys):
    from scripts.analyze_missing_audio_references import main

    dataset = tmp_path / "dataset"
    dataset.mkdir()

    status = main([
        "--dataset-dir", str(dataset),
        "--search-root", str(tmp_path),
        "--output-dir", str(dataset / "reports"),
    ])

    assert status == 2
    assert "outside the dataset root" in capsys.readouterr().err


def test_forensics_cli_reports_unexpected_analysis_error(tmp_path, capsys, monkeypatch):
    from scripts import analyze_missing_audio_references as cli

    def fail(*_args, **_kwargs):
        raise RuntimeError("candidate scan failed")

    monkeypatch.setattr(cli, "analyze_missing_references", fail)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    status = cli.main([
        "--dataset-dir", str(dataset),
        "--search-root", str(tmp_path),
        "--output-dir", str(tmp_path / "reports"),
    ])

    captured = capsys.readouterr()
    assert status == 1
    assert "forensic analysis or report generation failed (RuntimeError)" in captured.err
    assert "candidate scan failed" in captured.err
    assert "Forensics complete" not in captured.out
