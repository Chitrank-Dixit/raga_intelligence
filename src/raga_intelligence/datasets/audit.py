"""Read-only quality audit for CSV-backed audio datasets."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import librosa
import pandas as pd

from raga_intelligence.config import CANONICAL_RAGAS
from raga_intelligence.datasets.hindustani_dataset import CANONICAL_NAME_MAP

IDENTITY_UNKNOWN = {"", "unknown", "none", "null"}
PERFORMANCE_COLUMNS = ("performance_id", "concert_id")
AMBIGUOUS_PERFORMANCE_METADATA_COLUMNS = (
    "performance", "source_id", "video_id", "concert", "album", "release", "work", "composition"
)
PERFORMER_COLUMNS = ("performer_id", "artist_id")
PERFORMER_DISPLAY_COLUMNS = ("performer", "artist", "lead_artist")
SOURCE_COLUMNS = ("source", "dataset_source", "source_collection", "collection")
SOURCE_URL_COLUMNS = ("source_url", "url", "origin_url")
LICENSE_COLUMNS = ("license", "licence", "rights")
ATTRIBUTION_COLUMNS = ("attribution", "credit", "citation")
CONFIDENCE_COLUMNS = ("annotation_confidence", "confidence")
REVIEW_COLUMNS = ("review_status", "annotation_status", "reviewed")
RIGHTS_UNCERTAIN = {"unknown", "pending", "restricted", "unclear", "", "none", "null"}
_FILENAME_PERFORMANCE_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+)\]")
PERFORMANCE_IDENTITY_RELIABILITY = {
    "manual_review": "independently_reviewed",
    "metadata": "source_metadata_not_independently_reviewed",
    "filename_pattern": "inferred_from_filename",
    "unknown": "unknown",
    "missing": "unknown",
}


class DatasetAuditError(ValueError):
    """Raised when an audit cannot load its requested dataset manifest."""


def _identity(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return None if normalized.lower() in IDENTITY_UNKNOWN else normalized


def _first_value(row: dict[str, Any], columns: tuple[str, ...]) -> str | None:
    normalized = {key.strip().lower(): value for key, value in row.items()}
    for column in columns:
        if column in normalized:
            value = _identity(normalized[column])
            if value is not None:
                return value
    return None


def _first_controlled_value(row: dict[str, Any], columns: tuple[str, ...]) -> str | None:
    """Read status fields without collapsing literal `unknown` into a missing value."""
    normalized = {key.strip().lower(): value for key, value in row.items()}
    for column in columns:
        if column not in normalized:
            continue
        value = normalized[column]
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _manifest_audio_reference(row: dict[str, Any]) -> Any:
    """Return the audio reference using the audit's established column precedence."""
    return row.get("file_name") or row.get("file_path") or row.get("audio_path") or row.get("path")


def _canonical_raga(value: Any) -> str | None:
    label = _identity(value)
    if label is None:
        return None
    if label in CANONICAL_RAGAS:
        return label
    return CANONICAL_NAME_MAP.get(label)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=3
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_source_tree_state() -> tuple[bool | None, str | None]:
    """Capture source changes affecting audit semantics, excluding generated reports."""
    roots = ("scripts/", "src/", "tests/", "configs/", "docs/")
    files = {"README.md", ".gitignore", "pyproject.toml", "uv.lock"}
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        digest = hashlib.sha256()
        relevant = []
        for line in sorted(result.stdout.splitlines()):
            rel_path = line[3:]
            if not rel_path.startswith(roots) and rel_path not in files:
                continue
            relevant.append(line)
            digest.update(line.encode("utf-8", errors="replace"))
            candidate = Path(rel_path)
            if candidate.is_file():
                digest.update(candidate.read_bytes())
        return not relevant, digest.hexdigest()
    except (OSError, subprocess.SubprocessError):
        return None, None


def _split_for(row: dict[str, Any], relative_path: str) -> tuple[str | None, str]:
    split = _first_value(row, ("split", "partition", "subset"))
    if split:
        normalized = split.lower()
        if normalized == "validation":
            normalized = "val"
        return (normalized if normalized in {"train", "val", "test"} else split), "metadata"
    path_lower = relative_path.lower()
    if "test_" in path_lower:
        return "test", "filename_pattern"
    if "train_" in path_lower:
        return "train", "filename_pattern"
    return None, "missing"


def _relative_path(root: Path, value: Any) -> tuple[str | None, Path | None, str | None]:
    name = _identity(value)
    if name is None:
        return None, None, "missing audio path"
    if name.lower().startswith("file://"):
        return None, None, "file URI is not a dataset-relative audio path"
    if PureWindowsPath(name).is_absolute() and not Path(name).is_absolute():
        return None, None, "audio path escapes dataset root"
    candidate = (root / name.replace("\\", "/")).resolve()
    try:
        relative = candidate.relative_to(root.resolve()).as_posix()
    except ValueError:
        return None, None, "audio path escapes dataset root"
    return relative, candidate, None


def _audit_one_audio(path: Path, root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "exists": path.is_file(),
        "readable": False,
        "decoded": False,
        "duration_sec": None,
        "sample_rate": None,
        "format": path.suffix.lstrip(".").lower() or None,
        "audio_sha256": None,
        "decode_error": None,
        "hash_error": None,
    }
    if not info["exists"]:
        info["hash_error"] = "File does not exist."
        info["decode_error"] = "File does not exist."
        return info

    try:
        info["audio_sha256"] = _sha256_file(path)
        info["readable"] = True
    except OSError as exc:
        info["hash_error"] = _relative_error(exc, path, root)

    try:
        samples, sample_rate = librosa.load(path, sr=None, mono=False, dtype="float32")
        info["decoded"] = True
        info["sample_rate"] = int(sample_rate)
        info["duration_sec"] = float(samples.shape[-1] / sample_rate)
    except Exception as exc:  # decoder backends raise several library-specific exceptions
        info["decode_error"] = _relative_error(exc, path, root)
    return info


def _relative_error(exc: Exception, path: Path, root: Path) -> str:
    message = str(exc)
    relative = path.relative_to(root).as_posix()
    for absolute in (str(path), path.as_posix()):
        message = message.replace(absolute, relative)
    return f"{type(exc).__name__}: {message}"


def _safe_source_reference(value: str | None) -> str | None:
    if value and (
        value.lower().startswith("file://")
        or value.startswith(("/", "\\"))
        or re.match(r"^[a-zA-Z]:[\\/]", value)
    ):
        return "<local-path>"
    return value


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_dataset_manifest(
    dataset_dir: str | Path,
    metadata_file: str | Path | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    """Load a CSV manifest using the shared dataset-root and metadata path rules."""
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise DatasetAuditError(
            f"Dataset directory does not exist or is not a directory: {dataset_dir}"
        )
    csv_path = Path(metadata_file) if metadata_file else root / "metadata.csv"
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.is_file():
        raise DatasetAuditError(f"Dataset metadata CSV was not found: {csv_path}")
    try:
        frame = pd.read_csv(csv_path, dtype=object, keep_default_na=False)
    except Exception as exc:
        raise DatasetAuditError(
            f"Could not load dataset metadata CSV {csv_path}: {exc}"
        ) from exc
    return root, csv_path, frame


def audit_dataset(
    dataset_dir: str | Path,
    metadata_file: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a CSV dataset without changing any source, annotation, or cache files."""
    root, csv_path, frame = load_dataset_manifest(dataset_dir, metadata_file)

    rows = frame.to_dict(orient="records")
    manifest_columns = {str(column).strip().lower() for column in frame.columns}
    has_performance_column = any(column in manifest_columns for column in PERFORMANCE_COLUMNS)
    has_recording_id_column = "recording_id" in manifest_columns
    acceptance_column = next(
        (column for column in ("record_status", "acceptance_status", "dataset_status") if column in manifest_columns),
        None,
    )
    record_details: list[dict[str, Any]] = []
    duplicate_record_ids: dict[str, list[int]] = defaultdict(list)
    recording_to_ids: dict[str, list[str]] = defaultdict(list)
    performance_to_rows: dict[str, list[int]] = defaultdict(list)
    performer_to_rows: dict[str, list[int]] = defaultdict(list)
    performance_ragas: dict[str, set[str]] = defaultdict(set)
    performer_ragas: dict[str, set[str]] = defaultdict(set)
    performance_splits: dict[str, set[str]] = defaultdict(set)
    performer_splits: dict[str, set[str]] = defaultdict(set)
    raga_counts: Counter[str] = Counter({raga: 0 for raga in CANONICAL_RAGAS})
    raw_label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    raw_manifest_rows: list[dict[str, Any]] = []
    all_audio_hashable = bool(rows)

    for index, raw in enumerate(rows, start=2):
        row = {str(k).strip().lower(): v for k, v in raw.items()}
        file_value = _manifest_audio_reference(row)
        relative_path, audio_path, path_error = _relative_path(root, file_value)
        recording_id = _identity(row.get("recording_id"))
        recording_identity_source = "metadata" if recording_id else "missing"
        if recording_id is None and not has_recording_id_column and relative_path:
            recording_id = Path(relative_path).stem
            recording_identity_source = "filename"
        if recording_id is None:
            recording_id = f"__missing_recording_id_row_{index}"
        if recording_identity_source != "missing":
            duplicate_record_ids[recording_id].append(index)

        performance_id = _first_value(row, PERFORMANCE_COLUMNS)
        performance_origin = "metadata" if performance_id else "missing"
        if performance_id is None and not has_performance_column and relative_path:
            match = _FILENAME_PERFORMANCE_PATTERN.search(relative_path)
            if match and _identity(match.group(1)):
                performance_id = _identity(match.group(1))
                performance_origin = "filename_pattern"
        performer_id = _first_value(row, PERFORMER_COLUMNS)
        performer_display_name = _first_value(row, PERFORMER_DISPLAY_COLUMNS)
        label = _identity(row.get("label") or row.get("raga"))
        canonical_raga = _canonical_raga(label)
        source = _safe_source_reference(_first_value(row, SOURCE_COLUMNS))
        split, split_origin = _split_for(row, relative_path or "")
        if source:
            source_counts[source] += 1
        else:
            source_counts["<missing>"] += 1
        split_counts[split or "<missing>"] += 1
        if canonical_raga:
            raga_counts[canonical_raga] += 1
        raw_label_counts[label or "<missing>"] += 1

        audio = _audit_one_audio(audio_path, root) if audio_path else {
            "exists": False, "readable": False, "decoded": False, "duration_sec": None,
            "sample_rate": None, "format": None, "audio_sha256": None,
            "decode_error": None, "hash_error": path_error or "missing audio path",
        }
        audio["relative_path"] = relative_path
        if not audio["audio_sha256"]:
            all_audio_hashable = False

        missing_metadata = []
        invalid_metadata = []
        if not label:
            missing_metadata.append("raga label")
        if not recording_id or recording_id.startswith("__missing_recording_id_row_"):
            missing_metadata.append("recording ID")
        if not split:
            missing_metadata.append("split assignment")
        tonic = row.get("tonic_hz")
        if tonic not in (None, ""):
            try:
                tonic_value = float(tonic)
                if not math.isfinite(tonic_value) or tonic_value <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                invalid_metadata.append("tonic_hz")
        if split and split not in {"train", "val", "test"}:
            invalid_metadata.append("split")
        if missing_metadata:
            issue_counts["missing_metadata"] += 1
        if invalid_metadata:
            issue_counts["invalid_metadata"] += 1
        if not audio["exists"]:
            issue_counts["missing_audio"] += 1
        elif not audio["decoded"]:
            issue_counts["unreadable_audio"] += 1
        if not performance_id:
            issue_counts["missing_performance_id"] += 1
        if not performer_id:
            issue_counts["missing_performer_id"] += 1

        if performance_id:
            performance_to_rows[performance_id].append(index)
            if canonical_raga:
                performance_ragas[performance_id].add(canonical_raga)
            if split:
                performance_splits[performance_id].add(split)
        if performer_id:
            performer_to_rows[performer_id].append(index)
            if canonical_raga:
                performer_ragas[performer_id].add(canonical_raga)
            if split:
                performer_splits[performer_id].add(split)
        if audio["audio_sha256"]:
            recording_to_ids[audio["audio_sha256"]].append(recording_id)

        license_value = _first_value(row, LICENSE_COLUMNS)
        rights_value = _first_controlled_value(row, ("rights_status", "license_status"))
        performance_identity_source_claim = _first_controlled_value(
            row, ("performance_identity_source",)
        )
        if performance_identity_source_claim is not None:
            performance_identity_source_claim = performance_identity_source_claim.strip().lower()
        if performance_origin == "filename_pattern":
            # A source claim cannot upgrade a filename-derived group key to a
            # reviewed or source-metadata performance identity.
            performance_identity_source = "filename_pattern"
        elif performance_origin == "metadata":
            performance_identity_source = performance_identity_source_claim or "metadata"
        else:
            performance_identity_source = "unknown"
        if (
            performance_identity_source_claim == "manual_review"
            and performance_origin != "metadata"
        ):
            issue_counts["unapplied_performance_review_claim"] += 1
        ambiguous_performance_metadata = {
            field: _safe_source_reference(str(row[field]).strip())
            for field in AMBIGUOUS_PERFORMANCE_METADATA_COLUMNS
            if field in row and row[field] is not None and str(row[field]).strip()
        }
        detail = {
            "metadata_row": index,
            "recording_id": recording_id,
            "recording_identity_source": recording_identity_source,
            "relative_path": relative_path,
            "raw_raga_label": label,
            "canonical_raga": canonical_raga,
            "raw_tonic_hz": _identity(row.get("tonic_hz")),
            "source": source,
            "split": split,
            "split_identity_source": split_origin,
            "performance_id": performance_id,
            "performance_identity_source": performance_identity_source,
            "performance_identity_source_claim": performance_identity_source_claim,
            "performance_identity_reliability": PERFORMANCE_IDENTITY_RELIABILITY.get(
                performance_identity_source, "unknown"
            ),
            "ambiguous_performance_metadata": ambiguous_performance_metadata,
            "performer_id": performer_id,
            "performer_display_name": performer_display_name,
            "audio": audio,
            "missing_metadata": missing_metadata,
            "invalid_metadata": invalid_metadata,
            "provenance": {
                "source_recording_id": _first_value(row, ("source_recording_id",)),
                "source_url": _safe_source_reference(_first_value(row, SOURCE_URL_COLUMNS)),
                "license": _safe_source_reference(license_value),
                "license_url": _safe_source_reference(_first_value(row, ("license_url",))),
                "attribution": _safe_source_reference(_first_value(row, ATTRIBUTION_COLUMNS)),
                "rights_status": _safe_source_reference(rights_value),
            },
            "annotation": {
                "confidence": _first_value(row, CONFIDENCE_COLUMNS),
                "review_status": _first_controlled_value(row, REVIEW_COLUMNS),
                "status": _first_controlled_value(row, ("annotation_status",)),
                "notes": _first_value(row, ("annotation_notes", "review_notes")),
            },
            "record_lifecycle_status": (
                _first_controlled_value(row, (acceptance_column,)) if acceptance_column else None
            ),
            "metadata_schema_version": _first_controlled_value(row, ("metadata_schema_version",)),
        }
        record_details.append(detail)
        raw_manifest_rows.append({key: value for key, value in detail.items() if key != "audio"})

    duplicate_ids = {key: value for key, value in duplicate_record_ids.items() if len(value) > 1}
    duplicate_audio = {
        digest: values for digest, values in recording_to_ids.items() if len(values) > 1
    }
    recording_splits: dict[str, set[str]] = defaultdict(set)
    for record in record_details:
        if record["split"]:
            recording_splits[record["recording_id"]].add(record["split"])
    recording_overlap = {key: sorted(value) for key, value in recording_splits.items() if len(value) > 1}
    performance_overlap = {key: sorted(value) for key, value in performance_splits.items() if len(value) > 1}
    performer_overlap = {key: sorted(value) for key, value in performer_splits.items() if len(value) > 1}

    canonical_ragas = sorted(CANONICAL_RAGAS)
    class_coverage = {
        split: {
            raga: sum(1 for rec in record_details if rec["split"] == split and rec["canonical_raga"] == raga)
            for raga in canonical_ragas
        }
        for split in ("train", "val", "test")
    }
    missing_class_coverage = {
        split: [raga for raga in canonical_ragas if class_coverage[split][raga] == 0]
        for split in ("train", "test")
    }
    missing_class_coverage = {split: ragas for split, ragas in missing_class_coverage.items() if ragas}
    canonical_records = [rec for rec in record_details if rec["canonical_raga"]]
    canonical_performance_splits: dict[str, set[str]] = defaultdict(set)
    canonical_performer_splits: dict[str, set[str]] = defaultdict(set)
    canonical_recording_splits: dict[str, set[str]] = defaultdict(set)
    for record in canonical_records:
        split = record["split"]
        if not split:
            continue
        canonical_recording_splits[record["recording_id"]].add(split)
        if record["performance_id"]:
            canonical_performance_splits[record["performance_id"]].add(split)
        if record["performer_id"]:
            canonical_performer_splits[record["performer_id"]].add(split)
    evaluation_recording_overlap = {key: sorted(value) for key, value in canonical_recording_splits.items() if len(value) > 1}
    evaluation_performance_overlap = {key: sorted(value) for key, value in canonical_performance_splits.items() if len(value) > 1}
    evaluation_performer_overlap = {key: sorted(value) for key, value in canonical_performer_splits.items() if len(value) > 1}
    evaluation_missing_performance = sum(not rec["performance_id"] for rec in canonical_records)
    evaluation_missing_performer = sum(not rec["performer_id"] for rec in canonical_records)
    evaluation_missing_recording = sum(rec["recording_identity_source"] == "missing" for rec in canonical_records)
    evaluation_missing_split = sum(rec["split"] not in {"train", "val", "test"} for rec in canonical_records)
    evaluation_invalid_audio = sum(not rec["audio"]["exists"] or not rec["audio"]["decoded"] for rec in canonical_records)
    evaluation_duplicate_record_ids = {
        identity: row_numbers for identity, row_numbers in duplicate_ids.items()
        if any(rec["recording_id"] == identity and rec["canonical_raga"] for rec in record_details)
    }
    evaluation_performance_groups = {
        split: len({rec["performance_id"] for rec in canonical_records if rec["split"] == split and rec["performance_id"]})
        for split in ("train", "val", "test")
    }
    evaluation_performance_sources = Counter(
        rec["performance_identity_source"] if rec["performance_id"] else "missing"
        for rec in canonical_records
    )
    performance_identity_review_complete = bool(canonical_records) and all(
        rec["performance_id"] and rec["performance_identity_source"] == "manual_review"
        for rec in canonical_records
    )
    performance_grouping_verifiable = (
        bool(canonical_records)
        and evaluation_missing_performance == 0
        and evaluation_missing_split == 0
    )
    performance_identity_verifiable = (
        performance_identity_review_complete and evaluation_missing_split == 0
    )
    performance_grouping_status = (
        "failed" if evaluation_performance_overlap
        else "not_verifiable" if not performance_grouping_verifiable
        else "verified"
    )
    performance_identity_status = (
        "failed" if evaluation_performance_overlap
        else "verified" if performance_identity_verifiable
        else "not_verifiable"
    )

    source_completeness = _completeness(record_details, lambda rec: rec["source"])
    source_url_completeness = _completeness(record_details, lambda rec: rec["provenance"]["source_url"])
    license_completeness = _completeness(record_details, lambda rec: rec["provenance"]["license"])
    license_evidence_completeness = _completeness(record_details, lambda rec: rec["provenance"]["license_url"])
    attribution_completeness = _completeness(record_details, lambda rec: rec["provenance"]["attribution"])
    rights_uncertain = [
        rec["recording_id"] for rec in record_details
        if (rec["provenance"]["rights_status"] or "").strip().lower() in RIGHTS_UNCERTAIN
    ]
    metadata_columns = set(frame.columns.str.strip().str.lower())
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    physical_audio_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in audio_extensions
    )
    referenced_audio_paths = {rec["relative_path"] for rec in record_details if rec["relative_path"]}
    missing_recording_id_count = sum(rec["recording_identity_source"] == "missing" for rec in record_details)
    annotations = {
        "confidence_field_present": bool(metadata_columns.intersection(CONFIDENCE_COLUMNS)),
        "review_status_field_present": bool(metadata_columns.intersection(REVIEW_COLUMNS)),
        "missing_confidence_count": sum(not rec["annotation"]["confidence"] for rec in record_details)
        if metadata_columns.intersection(CONFIDENCE_COLUMNS) else None,
        "missing_review_status_count": sum(not rec["annotation"]["review_status"] for rec in record_details)
        if metadata_columns.intersection(REVIEW_COLUMNS) else None,
        "missing_tonic_count": sum(rec["raw_tonic_hz"] is None for rec in record_details),
        "invalid_tonic_count": sum("tonic_hz" in rec["invalid_metadata"] for rec in record_details),
    }

    issues: list[dict[str, str]] = []
    if duplicate_ids:
        issues.append({"severity": "ERROR", "code": "duplicate_recording_ids", "message": f"{len(duplicate_ids)} recording ID(s) occur more than once."})
    if issue_counts["missing_audio"]:
        issues.append({"severity": "ERROR", "code": "missing_audio", "message": f"{issue_counts['missing_audio']} referenced audio file(s) are missing or have invalid paths."})
    if issue_counts["unreadable_audio"]:
        issues.append({"severity": "ERROR", "code": "unreadable_audio", "message": f"{issue_counts['unreadable_audio']} audio file(s) could not be decoded."})
    if issue_counts["invalid_metadata"]:
        issues.append({"severity": "ERROR", "code": "invalid_metadata", "message": f"{issue_counts['invalid_metadata']} record(s) contain invalid labels or annotation metadata."})
    if sum(not rec["raw_raga_label"] for rec in record_details):
        issues.append({"severity": "ERROR", "code": "missing_raga_labels", "message": f"{sum(not rec['raw_raga_label'] for rec in record_details)} record(s) have no raga label."})
    if performance_overlap:
        issues.append({"severity": "ERROR", "code": "performance_leakage", "message": f"{len(performance_overlap)} performance ID(s) overlap dataset splits."})
    if performer_overlap:
        issues.append({"severity": "ERROR", "code": "performer_leakage", "message": f"{len(performer_overlap)} performer ID(s) overlap dataset splits."})
    unknown_labels = sorted({rec["raw_raga_label"] for rec in record_details if rec["raw_raga_label"] and not rec["canonical_raga"]})
    if unknown_labels:
        issues.append({"severity": "WARNING", "code": "noncanonical_raga_labels", "message": f"{len(unknown_labels)} raw raga label(s) are outside the configured canonical vocabulary; they are reported without being remapped."})
    if missing_class_coverage:
        issues.append({"severity": "ERROR", "code": "incomplete_class_coverage", "message": f"Canonical raga coverage is incomplete in train/test: {missing_class_coverage}."})
    if evaluation_missing_split:
        issues.append({"severity": "ERROR", "code": "missing_evaluation_split", "message": f"{evaluation_missing_split} canonical evaluation record(s) have no valid train/validation/test assignment."})
    if missing_recording_id_count:
        issues.append({"severity": "ERROR", "code": "missing_recording_ids", "message": f"{missing_recording_id_count} record(s) have no recording ID or usable filename."})
    if not record_details:
        issues.append({"severity": "ERROR", "code": "empty_dataset", "message": "The manifest contains no records."})
    if issue_counts["missing_metadata"]:
        issues.append({"severity": "WARNING", "code": "missing_metadata", "message": f"{issue_counts['missing_metadata']} record(s) have required annotation/identity fields missing."})
    if issue_counts["missing_performance_id"]:
        issues.append({"severity": "ERROR", "code": "performance_isolation_unverifiable", "message": f"{issue_counts['missing_performance_id']} record(s) lack a performance ID; scientific evaluation requires complete performance identities."})
    elif canonical_records and not performance_identity_review_complete:
        issues.append({
            "severity": "WARNING",
            "code": "performance_identity_not_independently_verified",
            "message": "Performance IDs do not have manual-review evidence; non-overlapping grouping IDs alone do not verify performance identity.",
        })
    if issue_counts["unapplied_performance_review_claim"]:
        issues.append({
            "severity": "WARNING",
            "code": "performance_review_claim_without_explicit_id",
            "message": (
                f"{issue_counts['unapplied_performance_review_claim']} record(s) claim manual review "
                "without an accepted performance_id or concert_id; the claim was not applied."
            ),
        })
    if issue_counts["missing_performer_id"]:
        issues.append({"severity": "WARNING", "code": "performer_isolation_unverifiable", "message": f"{issue_counts['missing_performer_id']} record(s) lack performer metadata; performer-held-out generalization cannot be claimed."})
    if duplicate_audio:
        issues.append({"severity": "WARNING", "code": "duplicate_audio_content", "message": f"{len(duplicate_audio)} SHA-256 group(s) contain identical audio bytes."})
    if rights_uncertain:
        issues.append({"severity": "WARNING", "code": "uncertain_rights", "message": f"{len(rights_uncertain)} record(s) have unknown, pending, or restricted rights status."})
    for field_name, completeness in (
        ("source", source_completeness),
        ("source URL", source_url_completeness),
        ("license", license_completeness),
        ("attribution", attribution_completeness),
    ):
        if completeness["missing"]:
            issues.append({
                "severity": "WARNING",
                "code": f"incomplete_{field_name.lower().replace(' ', '_')}",
                "message": f"{completeness['missing']} record(s) lack {field_name} information.",
            })

    manifest_fingerprint = _hash_json(sorted(
        ({key: value for key, value in row.items() if key != "metadata_row"} for row in raw_manifest_rows),
        key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ))
    audio_fingerprint = _hash_json(sorted((rec["relative_path"], rec["audio"]["audio_sha256"]) for rec in record_details)) if all_audio_hashable else None
    audio_unavailable_reason = None if audio_fingerprint else "At least one referenced audio file is missing or could not be hashed."
    source_tree_clean, source_tree_fingerprint = _git_source_tree_state()
    return {
        "schema_version": "1.0",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_commit(),
        "repository_source_tree_clean": source_tree_clean,
        "repository_source_tree_fingerprint": source_tree_fingerprint,
        "dataset_root_identifier": root.name,
        "manifest_path": csv_path.name,
        "manifest_fingerprint": manifest_fingerprint,
        "manifest_fingerprint_policy": "manifest_metadata_v1",
        "audio_content_fingerprint": audio_fingerprint,
        "audio_content_fingerprint_policy": "relative_path_sha256_v1" if audio_fingerprint else None,
        "audio_fingerprint_unavailable_reason": audio_unavailable_reason,
        "summary": {
            "total_manifest_records": len(record_details),
            "unique_recording_ids": len(duplicate_record_ids),
            "duplicate_recording_id_count": sum(len(rows) - 1 for rows in duplicate_ids.values()),
            "valid_records": sum(not rec["missing_metadata"] and not rec["invalid_metadata"] and rec["audio"]["decoded"] for rec in record_details),
            "missing_or_invalid_metadata_records": sum(bool(rec["missing_metadata"] or rec["invalid_metadata"]) for rec in record_details),
            "physical_audio_files_present": len({rec["relative_path"] for rec in record_details if rec["relative_path"] and rec["audio"]["exists"]}),
            "physical_audio_files_missing": len({rec["relative_path"] for rec in record_details if rec["relative_path"] and not rec["audio"]["exists"]}) + sum(not rec["relative_path"] for rec in record_details),
            "physical_audio_files_on_disk": len(physical_audio_files),
            "unreferenced_physical_audio_files": sorted(set(physical_audio_files) - referenced_audio_paths),
            "unreadable_audio_files": sum(rec["audio"]["exists"] and not rec["audio"]["decoded"] for rec in record_details),
            "exact_duplicate_audio_groups": len(duplicate_audio),
        },
        "recordings_per_raga": dict(sorted(raga_counts.items())),
        "records_per_raw_raga_label": dict(sorted(raw_label_counts.items())),
        "records_per_source": dict(sorted(source_counts.items())),
        "recordings_per_split": dict(sorted(split_counts.items())),
        "identity_coverage": {
            "recording": {
                "missing_count": missing_recording_id_count,
                "unique_recording_ids": len(duplicate_record_ids),
                "duplicate_recording_ids": duplicate_ids,
            },
            "performance": {
                "missing_count": issue_counts["missing_performance_id"],
                "unique_performances": len(performance_to_rows),
                "identity_source_counts": dict(Counter(rec["performance_identity_source"] for rec in record_details)),
                "identity_source_reliability": dict(PERFORMANCE_IDENTITY_RELIABILITY),
                "reliability_counts": dict(Counter(rec["performance_identity_reliability"] for rec in record_details)),
                "recordings_per_performance": {key: len(value) for key, value in sorted(performance_to_rows.items())},
                "raga_coverage_per_performance": {key: sorted(value) for key, value in sorted(performance_ragas.items())},
                "split_overlap": performance_overlap,
            },
            "performer": {
                "missing_count": issue_counts["missing_performer_id"],
                "unique_performers": len(performer_to_rows),
                "display_name_coverage": _completeness(record_details, lambda rec: rec["performer_display_name"]),
                "performer_id_coverage": _completeness(record_details, lambda rec: rec["performer_id"]),
                "recordings_per_performer": {key: len(value) for key, value in sorted(performer_to_rows.items())},
                "performances_per_performer": _performances_per_performer(record_details),
                "raga_coverage_per_performer": {key: sorted(value) for key, value in sorted(performer_ragas.items())},
                "split_overlap": performer_overlap,
            },
        },
        "audio_integrity": {
            "decode_library": f"librosa {librosa.__version__}",
            "unreadable_files": [
                {"recording_id": rec["recording_id"], "relative_path": rec["relative_path"], "error": rec["audio"]["decode_error"]}
                for rec in record_details if rec["audio"]["exists"] and not rec["audio"]["decoded"]
            ],
            "records": [
                {"recording_id": rec["recording_id"], **rec["audio"]} for rec in record_details
            ],
            "duplicate_content_groups": duplicate_audio,
        },
        "annotation_quality": {
            "missing_raga_labels": sum(not rec["raw_raga_label"] for rec in record_details),
            "unknown_raga_labels": unknown_labels,
            "class_distribution": dict(sorted(raga_counts.items())),
            "raw_label_distribution": dict(sorted(raw_label_counts.items())),
            # Label presence and canonical-vocabulary membership are distinct:
            # an out-of-vocabulary annotation is present metadata, but not part
            # of the canonical evaluation label set.
            "metadata_completeness": _completeness(record_details, lambda rec: rec["raw_raga_label"]),
            "canonical_label_coverage": _completeness(record_details, lambda rec: rec["canonical_raga"]),
            "annotation_status_distribution": _value_distribution(
                record_details,
                lambda rec: rec["annotation"]["status"] or rec["annotation"]["review_status"],
            ),
            **annotations,
        },
        "schema_coverage": {
            "metadata_schema_version_distribution": _value_distribution(
                record_details, lambda rec: rec["metadata_schema_version"]
            ),
            "record_lifecycle_status_distribution": _value_distribution(
                record_details, lambda rec: rec["record_lifecycle_status"]
            ),
            "performance_identity_source_distribution": _value_distribution(
                record_details, lambda rec: rec["performance_identity_source"]
            ),
        },
        "provenance": {
            "source_completeness": source_completeness,
            "source_url_completeness": source_url_completeness,
            "license_completeness": license_completeness,
            "license_evidence_completeness": license_evidence_completeness,
            "attribution_completeness": attribution_completeness,
            "rights_status_distribution": _value_distribution(
                record_details, lambda rec: rec["provenance"]["rights_status"]
            ),
            "rights_uncertain_recording_ids": rights_uncertain,
            "record_lifecycle_field": acceptance_column,
            "lifecycle_classification_available": acceptance_column is not None,
            "legacy_record_count": sum(
                (rec.get("record_lifecycle_status") or "").strip().lower() == "legacy"
                for rec in record_details
            ) if acceptance_column else None,
            "newly_accepted_record_count": sum(
                (rec.get("record_lifecycle_status") or "").strip().lower() in {"accepted", "newly_accepted"}
                for rec in record_details
            ) if acceptance_column else None,
            "unclassified_lifecycle_record_count": sum(not rec.get("record_lifecycle_status") for rec in record_details),
            "note": "Access to a recording does not establish permission to use it; records were audited, not excluded. Lifecycle counts are unavailable when no status field exists.",
        },
        "evaluation_readiness": {
            "evaluation_scope": "records with labels in the configured canonical raga vocabulary",
            "evaluation_scope_record_count": len(canonical_records),
            "recording_isolation_status": "failed" if evaluation_recording_overlap else "not_verifiable" if evaluation_missing_recording or evaluation_missing_split else "verified",
            "recording_split_overlap": evaluation_recording_overlap,
            "performance_isolation_status": performance_identity_status,
            "performance_isolation_verifiable": performance_identity_verifiable,
            "performance_identity_verification_status": performance_identity_status,
            "performance_identity_verification_verifiable": performance_identity_verifiable,
            "performance_grouping_split_isolation_status": performance_grouping_status,
            "performance_grouping_split_isolation_verifiable": performance_grouping_verifiable,
            "performance_identity_source_counts": dict(evaluation_performance_sources),
            "performer_isolation_status": "failed" if evaluation_performer_overlap else "not_verifiable" if evaluation_missing_performer or evaluation_missing_split else "verified",
            "performer_isolation_verifiable": evaluation_missing_performer == 0 and evaluation_missing_split == 0,
            "missing_performance_id_count": evaluation_missing_performance,
            "missing_recording_id_count": evaluation_missing_recording,
            "missing_split_assignment_count": evaluation_missing_split,
            "unavailable_or_unreadable_audio_count": evaluation_invalid_audio,
            "duplicate_recording_ids": evaluation_duplicate_record_ids,
            "raga_class_coverage_by_split": class_coverage,
            "independent_performance_groups_by_split": evaluation_performance_groups,
            "scientific_evaluation_ready": (
                bool(canonical_records)
                and evaluation_invalid_audio == 0
                and evaluation_missing_recording == 0
                and evaluation_missing_split == 0
                and evaluation_missing_performance == 0
                and performance_identity_status == "verified"
                and not evaluation_duplicate_record_ids
                and not evaluation_performance_overlap
                and not evaluation_recording_overlap
                and all(class_coverage[split][raga] > 0 for split in ("train", "test") for raga in canonical_ragas)
            ),
        },
        "dataset_has_critical_errors": any(issue["severity"] == "ERROR" for issue in issues),
        "issues": issues,
        "records": record_details,
    }


def _completeness(records: list[dict[str, Any]], get_value) -> dict[str, int | float]:
    present = sum(bool(get_value(record)) for record in records)
    total = len(records)
    return {"present": present, "missing": total - present, "total": total, "fraction": present / total if total else 0.0}


def _value_distribution(records: list[dict[str, Any]], get_value) -> dict[str, int]:
    def category(record: dict[str, Any]) -> str:
        value = get_value(record)
        if value is None or pd.isna(value):
            return "<missing>"
        normalized = str(value).strip()
        return normalized or "<missing>"

    values = Counter(category(record) for record in records)
    return dict(sorted(values.items()))


def _performances_per_performer(records: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, set[str]] = defaultdict(set)
    for record in records:
        performer = record["performer_id"]
        performance = record["performance_id"]
        if performer and performance:
            result[performer].add(performance)
    return {key: len(value) for key, value in sorted(result.items())}


def render_audit_markdown(audit: dict[str, Any]) -> str:
    """Render the stable audit structure as a human-readable Markdown report."""
    summary = audit["summary"]
    readiness = audit["evaluation_readiness"]
    performer_id_coverage = audit["identity_coverage"]["performer"]["performer_id_coverage"]
    label_completeness = audit["annotation_quality"]["metadata_completeness"]
    canonical_label_coverage = audit["annotation_quality"]["canonical_label_coverage"]
    lines = [
        "# Dataset Quality Audit",
        "",
        f"- **Audit timestamp (UTC):** `{audit['audit_timestamp']}`",
        f"- **Repository commit:** `{audit['repository_commit'] or 'unavailable'}`",
        f"- **Relevant source tree clean:** `{audit['repository_source_tree_clean']}`",
        f"- **Dataset root identifier:** `{audit['dataset_root_identifier']}`",
        f"- **Manifest:** `{audit['manifest_path']}`",
        f"- **Manifest fingerprint:** `{audit['manifest_fingerprint']}`",
        f"- **Manifest fingerprint policy:** `{audit['manifest_fingerprint_policy']}`",
        f"- **Audio content fingerprint:** `{audit['audio_content_fingerprint'] or 'unavailable'}`",
    ]
    if audit["audio_fingerprint_unavailable_reason"]:
        lines.append(f"- **Audio fingerprint reason:** {audit['audio_fingerprint_unavailable_reason']}")
    lines.extend([
        "",
        "## Inventory",
        "",
        f"- Manifest records: {summary['total_manifest_records']}",
        f"- Unique recording IDs: {summary['unique_recording_ids']}",
        f"- Duplicate recording IDs: {summary['duplicate_recording_id_count']}",
        f"- Valid records: {summary['valid_records']}",
        f"- Records with missing / invalid metadata: {summary['missing_or_invalid_metadata_records']}",
        f"- Audio files present / missing: {summary['physical_audio_files_present']} / {summary['physical_audio_files_missing']}",
        f"- Audio files physically present under dataset root: {summary['physical_audio_files_on_disk']}",
        f"- Unreferenced audio files: {len(summary['unreferenced_physical_audio_files'])}",
        f"- Unreadable audio: {summary['unreadable_audio_files']}",
        f"- Exact duplicate audio groups: {summary['exact_duplicate_audio_groups']}",
        "",
        "## Class, source, and split counts",
        "",
        _markdown_table("Raga", audit["recordings_per_raga"]),
        "",
        _markdown_table("Source", audit["records_per_source"]),
        "",
        _markdown_table("Split", audit["recordings_per_split"]),
        "",
        "## Evaluation readiness",
        "",
        f"- Recording isolation: **{readiness['recording_isolation_status']}**",
        f"- Performance isolation: **{readiness['performance_isolation_status']}** (verifiable: {readiness['performance_isolation_verifiable']})",
        f"- Performance grouping split isolation: **{readiness['performance_grouping_split_isolation_status']}**",
        f"- Performance identity sources: `{readiness['performance_identity_source_counts']}`",
        f"- Identity source reliability: `{audit['identity_coverage']['performance']['identity_source_reliability']}`",
        f"- Performer isolation: **{readiness['performer_isolation_status']}** (verifiable: {readiness['performer_isolation_verifiable']})",
        f"- Performer ID coverage: {performer_id_coverage['present']}/{performer_id_coverage['total']}",
        f"- Evaluation scope records / unavailable audio: {readiness['evaluation_scope_record_count']} / {readiness['unavailable_or_unreadable_audio_count']}",
        f"- Canonical records without split assignment: {readiness['missing_split_assignment_count']}",
        f"- Canonical evaluation scope ready: **{readiness['scientific_evaluation_ready']}**",
        f"- Dataset has critical errors: **{audit['dataset_has_critical_errors']}**",
        "",
        "### Canonical raga coverage",
        "",
        _nested_table(audit["annotation_quality"]["class_distribution"], readiness["raga_class_coverage_by_split"]),
        "",
        "### Raw source raga labels",
        "",
        _markdown_table("Raw raga label", audit["records_per_raw_raga_label"]),
        "",
        f"- Raga label metadata present: {label_completeness['present']}/{label_completeness['total']} "
        f"({label_completeness['fraction']:.1%})",
        f"- Labels in canonical vocabulary: {canonical_label_coverage['present']}/"
        f"{canonical_label_coverage['total']} ({canonical_label_coverage['fraction']:.1%})",
        f"- Missing annotation confidence: `{audit['annotation_quality']['missing_confidence_count']}`",
        f"- Missing review status: `{audit['annotation_quality']['missing_review_status_count']}`",
        f"- Missing tonic annotation: `{audit['annotation_quality']['missing_tonic_count']}`",
        f"- Unknown raw labels: `{len(audit['annotation_quality']['unknown_raga_labels'])}`",
        "",
        "### Findings",
        "",
    ])
    lines.extend(f"- **{issue['severity']} `{issue['code']}`:** {issue['message']}" for issue in audit["issues"])
    if not audit["issues"]:
        lines.append("- No issues detected.")
    lines.extend(["", "## Metadata contract coverage", ""])
    for name, distribution in audit["schema_coverage"].items():
        lines.append(f"- {name.replace('_', ' ').title()}: `{distribution}`")
    lines.append(
        f"- Annotation status distribution: `{audit['annotation_quality']['annotation_status_distribution']}`"
    )
    lines.append(
        f"- Rights status distribution: `{audit['provenance']['rights_status_distribution']}`"
    )
    lines.extend(["", "## Provenance completeness", ""])
    for name, metrics in audit["provenance"].items():
        if isinstance(metrics, dict) and "fraction" in metrics:
            lines.append(f"- {name.replace('_', ' ').title()}: {metrics['present']}/{metrics['total']} ({metrics['fraction']:.1%})")
    provenance = audit["provenance"]
    if provenance["lifecycle_classification_available"]:
        lines.extend([
            f"- Legacy records: {provenance['legacy_record_count']}",
            f"- Newly accepted records: {provenance['newly_accepted_record_count']}",
            f"- Records without a lifecycle status: {provenance['unclassified_lifecycle_record_count']}",
        ])
    else:
        lines.append("- Record lifecycle classification: unavailable (no lifecycle status field was present).")
    lines.append(f"- {provenance['note']}")
    lines.extend(["", "Audit is read-only; source records and audio files are not modified.", ""])
    return "\n".join(lines)


def _markdown_table(first_column: str, values: dict[str, Any]) -> str:
    rows = [f"| {first_column} | Records |", "|---|---:|"]
    rows.extend(f"| {key} | {value} |" for key, value in sorted(values.items()))
    return "\n".join(rows)


def _nested_table(class_distribution: dict[str, int], by_split: dict[str, dict[str, int]]) -> str:
    rows = ["| Raga | Total | Train | Validation | Test |", "|---|---:|---:|---:|---:|"]
    for raga, total in sorted(class_distribution.items()):
        if raga in CANONICAL_RAGAS:
            rows.append(f"| {raga} | {total} | {by_split['train'][raga]} | {by_split['val'][raga]} | {by_split['test'][raga]} |")
    return "\n".join(rows)


def write_audit_reports(audit: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports without touching dataset files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "dataset_audit.json"
    markdown_path = directory / "dataset_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_audit_markdown(audit), encoding="utf-8")
    return json_path, markdown_path
