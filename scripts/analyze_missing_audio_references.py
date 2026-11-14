"""Forensically analyze missing dataset audio references without changing data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raga_intelligence.config import settings
from raga_intelligence.datasets.missing_reference_forensics import (
    MissingReferenceForensicsError,
    analyze_missing_references,
    write_forensics_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=settings.data_dir / "hindustani_dataset",
        help="Dataset root (defaults to DATA_DIR/hindustani_dataset from project settings).",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Metadata CSV path, absolute or relative to the dataset root (default: metadata.csv).",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=settings.project_root,
        help="Local root to search for candidate audio files (default: project root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dataset_audit"),
        help="Directory for local JSON and Markdown forensic reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset_root = args.dataset_dir.expanduser().resolve()
        output_root = args.output_dir.expanduser().resolve()
        output_root.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        print("ERROR: report output directory must be outside the dataset root.", file=sys.stderr)
        return 2

    try:
        report = analyze_missing_references(
            args.dataset_dir,
            metadata_file=args.metadata_file,
            search_root=args.search_root,
        )
        json_path, markdown_path = write_forensics_reports(report, args.output_dir)
    except MissingReferenceForensicsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(
        "Forensics complete: "
        f"{summary['manifest_records_examined']} record(s), "
        f"{summary['missing_references']} missing reference(s), "
        f"{summary['unresolved_references']} unresolved."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
