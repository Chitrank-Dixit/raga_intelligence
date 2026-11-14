import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def test_audit_cli_reads_dataset_writes_reports_and_preserves_inputs(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    audio_path = dataset / "train_[performance_one].wav"
    sf.write(audio_path, np.zeros(8000, dtype=np.float32), 8000)
    metadata_path = dataset / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["file_name", "label", "tonic_hz", "performer", "split"],
        )
        writer.writeheader()
        writer.writerow({
            "file_name": audio_path.name,
            "label": "Bhairav",
            "tonic_hz": "220",
            "performer": "unknown",
            "split": "train",
        })

    metadata_before = metadata_path.read_bytes()
    audio_before = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    output_dir = tmp_path / "reports"
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "audit_dataset.py"
    environment = os.environ.copy()
    environment["DEBUG"] = "false"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 1
    assert "1 error(s)" in result.stdout
    json_report = output_dir / "dataset_audit.json"
    markdown_report = output_dir / "dataset_audit.md"
    assert json_report.is_file()
    assert markdown_report.is_file()
    report = json.loads(json_report.read_text(encoding="utf-8"))
    assert report["summary"]["total_manifest_records"] == 1
    assert report["audio_integrity"]["records"][0]["relative_path"] == audio_path.name
    assert metadata_path.read_bytes() == metadata_before
    assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == audio_before
