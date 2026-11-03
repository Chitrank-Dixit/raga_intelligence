"""Run a read-only quality and evaluation-readiness audit of an audio dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raga_intelligence.config import settings
from raga_intelligence.datasets.audit import DatasetAuditError, audit_dataset, write_audit_reports


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
        "--output-dir",
        type=Path,
        default=Path("reports/dataset_audit"),
        help="Directory for dataset_audit.json and dataset_audit.md.",
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
        audit = audit_dataset(args.dataset_dir, args.metadata_file)
    except DatasetAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    json_path, markdown_path = write_audit_reports(audit, args.output_dir)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    errors = sum(issue["severity"] == "ERROR" for issue in audit["issues"])
    warnings = sum(issue["severity"] == "WARNING" for issue in audit["issues"])
    print(f"Audit complete: {errors} error(s), {warnings} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
