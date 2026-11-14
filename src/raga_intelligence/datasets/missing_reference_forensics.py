"""Read-only forensic analysis of missing CSV audio references."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from raga_intelligence.datasets.audit import (
    PERFORMANCE_COLUMNS,
    PERFORMER_COLUMNS,
    PERFORMER_DISPLAY_COLUMNS,
    SOURCE_COLUMNS,
    DatasetAuditError,
    _FILENAME_PERFORMANCE_PATTERN,
    _first_controlled_value,
    _first_value,
    _git_commit,
    _git_source_tree_state,
    _hash_json,
    _identity,
    _manifest_audio_reference,
    _relative_path,
    _safe_source_reference,
    load_dataset_manifest,
)

_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma"}
_EXCLUDED_DIRECTORIES = {
    ".cache",
    ".coverage",
    ".env",
    ".git",
    ".mypy_cache",
    ".nox",
    ".tox",
    ".venv",
    ".venv-linux",
    ".pytest_cache",
    ".ruff_cache",
    ".task_pytest_tmp",
    "__pycache__",
    "build",
    "cache",
    "dist",
    "env",
    "generated",
    "htmlcov",
    "node_modules",
    "reports",
    "site-packages",
    "target",
    "venv",
}
_CLASSIFICATIONS = (
    "alternate_local_location_candidate",
    "duplicate_reference_candidate",
    "path_convention_mismatch",
    "possible_segment_reference",
    "source_metadata_present_audio_unlocated",
    "unresolved",
)
_SEGMENT_PATTERN = re.compile(
    r"^(?P<prefix>.*\[(?P<identity>[^\]]+)\]_chunk)(?P<index>\d+)(?P<extension>\.[^.]+)$",
    re.IGNORECASE,
)
_BRACKETED_ID_PATTERN = re.compile(r"\[([^\]]+)\]")
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
_ABSOLUTE_WINDOWS_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")


class MissingReferenceForensicsError(ValueError):
    """Raised when forensic analysis cannot load its manifest or roots."""


def _is_absolute_reference(value: str) -> bool:
    value = value.strip()
    return (
        value.lower().startswith("file://")
        or Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or bool(_ABSOLUTE_WINDOWS_PATTERN.match(value))
    )


def _normalized_reference(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    return path.as_posix()


def _relative_report_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _audio_files(search_root: Path) -> tuple[list[Path], list[str]]:
    found: list[Path] = []
    failures: list[str] = []

    def onerror(error: OSError) -> None:
        filename = getattr(error, "filename", None)
        if filename:
            try:
                relative = Path(filename).resolve().relative_to(search_root).as_posix()
            except (OSError, ValueError):
                relative = Path(filename).name
        else:
            relative = "<unknown>"
        failures.append(relative)

    for current, dirs, files in os.walk(
        search_root, topdown=True, followlinks=False, onerror=onerror
    ):
        dirs[:] = sorted(name for name in dirs if name not in _EXCLUDED_DIRECTORIES)
        current_path = Path(current)
        for filename in sorted(files):
            path = current_path / filename
            if (
                path.suffix.lower() in _AUDIO_EXTENSIONS
                and path.is_file()
                and not path.is_symlink()
            ):
                found.append(path)
    return (
        sorted(found, key=lambda path: path.relative_to(search_root).as_posix()),
        sorted(failures),
    )


def _tokens(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(value)]


def _has_token_sequence(value: str, expected: list[str]) -> bool:
    tokens = _tokens(value)
    return bool(expected) and any(
        tokens[index : index + len(expected)] == expected
        for index in range(len(tokens) - len(expected) + 1)
    )


def _safe_reference(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if _is_absolute_reference(value):
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
        return "<absolute path redacted>", digest
    return value, None


def _missing_path_fingerprint(
    normalized_reference: str | None,
    original_reference: str | None,
    absolute_path_hash: str | None,
) -> str:
    if normalized_reference is not None:
        identity = ["relative", normalized_reference]
    elif absolute_path_hash is not None:
        normalized_absolute = _normalized_reference(original_reference or "")
        if normalized_absolute.lower().startswith("file://"):
            parsed = urlsplit(normalized_absolute)
            normalized_absolute = f"file://{parsed.netloc}{parsed.path}"
        identity = ["absolute", normalized_absolute]
    else:
        identity = ["missing", ""]
    serialized = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _candidate_evidence(
    reference: str,
    normalized_path: str | None,
    source_recording_id: str | None,
    candidates: list[Path],
    search_root: Path,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    if not normalized_path:
        reference = reference.strip()
        if reference.lower().startswith("file://"):
            basename = PurePosixPath(urlsplit(reference).path).name
        else:
            basename = (
                PureWindowsPath(reference).name
                if _is_absolute_reference(reference)
                else Path(reference.replace("\\", "/")).name
            )
    else:
        basename = PurePosixPath(normalized_path).name
    stem = Path(basename).stem
    exact_name = [candidate for candidate in candidates if candidate.name == basename]
    casefold_name = [
        candidate
        for candidate in candidates
        if candidate.name.casefold() == basename.casefold()
    ]
    same_stem = [
        candidate for candidate in candidates if candidate.stem.casefold() == stem.casefold()
    ]

    normalized_parts = PurePosixPath(normalized_path).parts if normalized_path else ()
    source_tokens = _tokens(source_recording_id) if source_recording_id else []
    matches: dict[Path, set[str]] = defaultdict(set)

    for candidate in candidates:
        try:
            under_dataset = candidate.relative_to(dataset_root).as_posix()
        except ValueError:
            under_dataset = None
        candidate_rel = candidate.relative_to(search_root).as_posix()
        if normalized_parts and under_dataset:
            candidate_parts = PurePosixPath(under_dataset).parts
            if (
                len(candidate_parts) >= len(normalized_parts)
                and candidate_parts[-len(normalized_parts) :] == normalized_parts
            ):
                matches[candidate].add("normalized_path_match")
            elif len(candidate_parts) >= len(normalized_parts) and tuple(
                part.casefold() for part in candidate_parts[-len(normalized_parts) :]
            ) == tuple(part.casefold() for part in normalized_parts):
                matches[candidate].add("case_only_path_match")
        elif normalized_parts:
            candidate_parts = PurePosixPath(candidate_rel).parts
            if (
                len(candidate_parts) > len(normalized_parts)
                and candidate_parts[-len(normalized_parts) :] == normalized_parts
            ):
                matches[candidate].add("normalized_path_match")

        if candidate in exact_name:
            matches[candidate].add("exact_filename_match")
        elif candidate in casefold_name:
            matches[candidate].add("casefold_filename_match")
        if candidate in same_stem and candidate.name != basename:
            matches[candidate].add("filename_similarity_only")
        if _has_token_sequence(candidate.name, source_tokens):
            matches[candidate].add("explicit_source_identifier_match")

    segment = _SEGMENT_PATTERN.match(basename)
    if segment:
        wanted_identity = segment.group("identity").casefold()
        for candidate in candidates:
            candidate_segment = _SEGMENT_PATTERN.match(candidate.name)
            if (
                candidate_segment
                and candidate_segment.group("identity").casefold() == wanted_identity
                and candidate_segment.group("index") != segment.group("index")
            ):
                matches[candidate].add("same_filename_source_id_different_chunk")

    return [
        {
            "path": _relative_report_path(candidate, search_root),
            "evidence": sorted(evidence),
            "identity_established": False,
        }
        for candidate, evidence in sorted(
            matches.items(), key=lambda entry: entry[0].relative_to(search_root).as_posix()
        )
    ]


def _classify(record: dict[str, Any]) -> str:
    evidence = {item for candidate in record["local_candidates"] for item in candidate["evidence"]}
    if "case_only_path_match" in evidence:
        return "path_convention_mismatch"
    if "normalized_path_match" in evidence:
        return "path_convention_mismatch"
    if "same_filename_source_id_different_chunk" in evidence:
        return "possible_segment_reference"
    if record["source"] and record["source_recording_id"] and not record["local_candidates"]:
        return "source_metadata_present_audio_unlocated"
    if record["local_candidates"]:
        return "alternate_local_location_candidate"
    return "unresolved"


def analyze_missing_references(
    dataset_dir: str | Path,
    metadata_file: str | Path | None = None,
    search_root: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze every missing manifest reference using the existing audit path rules.

    Candidate file matches are evidence for review, never verified recording identity.
    The analysis reads files only and does not alter the dataset or metadata.
    """
    root = Path(dataset_dir).expanduser().resolve()
    search = Path(search_root).expanduser().resolve() if search_root else root.parent
    if not root.is_dir():
        raise MissingReferenceForensicsError(f"Dataset directory does not exist: {dataset_dir}")
    if not search.is_dir():
        raise MissingReferenceForensicsError(f"Search root does not exist: {search_root}")

    try:
        root, csv_path, frame = load_dataset_manifest(root, metadata_file)
    except DatasetAuditError as exc:
        raise MissingReferenceForensicsError(
            str(exc)
        ) from exc

    audio_files, scan_failures = _audio_files(search)
    records: list[dict[str, Any]] = []
    present_references = 0
    present_filename_groups: set[str] = set()
    columns = {str(column).strip().lower() for column in frame.columns}
    has_recording_id_column = "recording_id" in columns
    has_performance_column = any(column in columns for column in PERFORMANCE_COLUMNS)
    for row_number, raw in enumerate(frame.to_dict(orient="records"), start=2):
        row = {str(key).strip().lower(): value for key, value in raw.items()}
        reference_value = _manifest_audio_reference(row)
        original = str(reference_value) if reference_value is not None else None
        normalized, expected_path, path_error = _relative_path(root, original)
        reference_exists = expected_path is not None and expected_path.is_file()

        recording_id = _identity(row.get("recording_id"))
        if recording_id is None and not has_recording_id_column and normalized:
            recording_id = Path(normalized).stem
        if recording_id is None:
            recording_id = f"__missing_recording_id_row_{row_number}"

        performance_id = _first_value(row, PERFORMANCE_COLUMNS)
        performance_origin = "metadata" if performance_id else "missing"
        if performance_id is None and not has_performance_column and normalized:
            match = _FILENAME_PERFORMANCE_PATTERN.search(normalized)
            if match and _identity(match.group(1)):
                performance_id = _identity(match.group(1))
                performance_origin = "filename_pattern"
        source_claim = _first_controlled_value(row, ("performance_identity_source",))
        if source_claim is not None:
            source_claim = source_claim.strip().lower()
        performance_identity_source = (
            "filename_pattern"
            if performance_origin == "filename_pattern"
            else (source_claim or "metadata")
            if performance_origin == "metadata"
            else "unknown"
        )
        raw_source = _first_value(row, SOURCE_COLUMNS)
        source = _safe_source_reference(raw_source)
        raw_source_recording_id = _first_value(row, ("source_recording_id",))
        source_recording_id = _safe_source_reference(raw_source_recording_id)
        source_id_for_matching = (
            raw_source_recording_id
            if source_recording_id == raw_source_recording_id
            else None
        )
        performer_id = _first_value(row, PERFORMER_COLUMNS)
        performer_display_name = _first_value(row, PERFORMER_DISPLAY_COLUMNS)
        raw_raga_label = _identity(row.get("label") or row.get("raga"))

        if reference_exists:
            present_references += 1
            if performance_identity_source == "filename_pattern" and performance_id:
                present_filename_groups.add(performance_id)
            continue

        safe_original, reference_hash = _safe_reference(original)
        normalized_reference = (
            _normalized_reference(original)
            if original and not _is_absolute_reference(original)
            else None
        )
        local_candidates = _candidate_evidence(
            original or "",
            normalized,
            source_id_for_matching,
            audio_files,
            search,
            root,
        )
        relative_expected = None
        if expected_path is not None:
            try:
                relative_expected = expected_path.relative_to(root).as_posix()
            except ValueError:
                relative_expected = None
        record = {
            "metadata_row": row_number,
            "recording_id": recording_id,
            "original_reference_path": safe_original,
            "absolute_reference_redacted": reference_hash is not None,
            "reference_path_sha256": reference_hash,
            "normalized_reference_path": normalized_reference,
            "missing_path_fingerprint": _missing_path_fingerprint(
                normalized_reference, original, reference_hash
            ),
            "resolved_expected_path": relative_expected,
            "reference_path_type": (
                "absolute"
                if original and _is_absolute_reference(original)
                else "relative"
                if original
                else "missing"
            ),
            "path_resolution_error": path_error,
            "reference_exists": reference_exists,
            "source": source,
            "source_recording_id": source_recording_id,
            "performance_id": performance_id,
            "performance_identity_source": performance_identity_source,
            "raw_raga_label": raw_raga_label,
            "performer_display_name": performer_display_name,
            "performer_id": performer_id,
            "local_candidates": local_candidates,
            "forensic_classification": None,
            "evidence_strength": (
                "weak" if local_candidates else
                "manifest_source_metadata" if source and source_recording_id else
                "none"
            ),
            "unresolved_questions": [],
        }
        record["forensic_classification"] = _classify(record)
        if not source:
            record["unresolved_questions"].append("Source metadata is absent.")
        if not source_recording_id:
            record["unresolved_questions"].append("Source recording identifier is absent.")
        if not local_candidates:
            record["unresolved_questions"].append(
                "No matching local audio candidate was found by exact path, filename, "
                "source-ID, or chunk-sibling checks."
            )
        if local_candidates:
            record["unresolved_questions"].append(
                "Candidate path/name evidence does not establish that the file is the "
                "referenced recording."
            )
        if path_error:
            record["unresolved_questions"].append(
                "The existing audit path resolver rejected this reference."
            )
        records.append(record)

    candidate_reference_counts: Counter[str] = Counter(
        candidate["path"] for record in records for candidate in record["local_candidates"]
    )
    for record in records:
        if any(candidate_reference_counts[item["path"]] > 1 for item in record["local_candidates"]):
            record["forensic_classification"] = "duplicate_reference_candidate"

    records.sort(key=lambda record: (record["metadata_row"], record["recording_id"]))
    observed_classifications = Counter(record["forensic_classification"] for record in records)
    classification_counts = {name: observed_classifications[name] for name in _CLASSIFICATIONS}
    source_classifications: dict[str, Counter[str]] = defaultdict(Counter)
    path_prefixes: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    path_kinds: Counter[str] = Counter()
    filename_patterns: Counter[str] = Counter()
    explicit_source_id_count = 0
    candidate_count = 0
    candidate_evidence_counts: Counter[str] = Counter()
    unresolved_count = 0
    for record in records:
        source_classifications[record["source"] or "<missing>"][
            record["forensic_classification"]
        ] += 1
        path_kinds[record["reference_path_type"]] += 1
        relative = record["normalized_reference_path"]
        if relative:
            parts = PurePosixPath(relative).parts
            path_prefixes[parts[0] if len(parts) > 1 else "<root>"] += 1
            extensions[PurePosixPath(relative).suffix.lower() or "<no extension>"] += 1
            filename = PurePosixPath(relative).name.casefold()
            patterns = []
            if "_chunk" in filename:
                patterns.append("chunk_suffix")
            if _BRACKETED_ID_PATTERN.search(filename):
                patterns.append("bracketed_identifier")
            if filename.startswith("train_"):
                patterns.append("train_prefix")
            if filename.startswith("test_"):
                patterns.append("test_prefix")
            filename_patterns.update(patterns or ["other"])
        explicit_source_id_count += bool(record["source_recording_id"])
        unresolved_count += record["forensic_classification"] == "unresolved"
        for candidate in record["local_candidates"]:
            candidate_count += 1
            candidate_evidence_counts.update(candidate["evidence"])

    missing_filename_groups = {
        record["performance_id"]
        for record in records
        if record["performance_identity_source"] == "filename_pattern"
        and record["performance_id"]
    }
    normalized_manifest_rows = [
        {
            str(key).strip().lower(): None if value is None else str(value)
            for key, value in raw.items()
        }
        for raw in frame.to_dict(orient="records")
    ]
    manifest_fingerprint = _hash_json(normalized_manifest_rows)
    source_tree_clean, source_tree_fingerprint = _git_source_tree_state()
    try:
        manifest_path = csv_path.resolve().relative_to(search).as_posix()
    except ValueError:
        manifest_path = csv_path.name

    return {
        "schema_version": "1.0",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_commit(),
        "repository_source_tree_clean": source_tree_clean,
        "repository_source_tree_fingerprint": source_tree_fingerprint,
        "dataset_root_identifier": root.name,
        "search_root_identifier": search.name,
        "manifest_path": manifest_path,
        "manifest_fingerprint": manifest_fingerprint,
        "manifest_fingerprint_policy": "normalized_manifest_csv_rows_v1",
        "path_resolution_policy": "datasets.audit._relative_path",
        "summary": {
            "manifest_records_examined": len(frame.index),
            "missing_references": len(records),
            "present_references": present_references,
            "references_analyzed": len(records),
            "unresolved_references": unresolved_count,
            "unique_missing_paths": len(
                {record["missing_path_fingerprint"] for record in records}
            ),
            "local_audio_candidates_found": candidate_count,
            "unique_local_candidate_files": len(candidate_reference_counts),
        },
        "classification_counts": dict(sorted(classification_counts.items())),
        "classification_percentages": {
            key: (count / len(records) * 100 if records else 0.0)
            for key, count in sorted(classification_counts.items())
        },
        "classifications_by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_classifications.items())
        },
        "path_analysis": {
            "common_missing_path_prefixes": dict(path_prefixes.most_common()),
            "common_filename_patterns": dict(filename_patterns.most_common()),
            "extension_distribution": dict(extensions.most_common()),
            "relative_versus_absolute": dict(sorted(path_kinds.items())),
            "path_convention_anomaly_count": classification_counts.get(
                "path_convention_mismatch", 0
            ),
            "repository_path_convention": {
                "audio_column_precedence": ["file_name", "file_path", "audio_path", "path"],
                "relative_paths_resolved_under_dataset_root": True,
                "backslashes_normalized_to_forward_slashes": True,
                "paths_escaping_dataset_root_rejected": True,
                "adapter_filename_grouping": (
                    "Bracketed filename identifiers may be used as inferred performance "
                    "grouping keys; this does not establish source recording identity."
                ),
            },
        },
        "candidate_analysis": {
            "search_root_identifier": search.name,
            "search_scope": "Recursive search for audio-file extensions below the configured search root.",
            "excluded_directory_names": sorted(_EXCLUDED_DIRECTORIES),
            "symlink_directories_followed": False,
            "symlink_audio_files_indexed": False,
            "locations_outside_search_root_searched": False,
            "local_audio_file_count": len(audio_files),
            "candidate_match_count": candidate_count,
            "unique_candidate_file_count": len(candidate_reference_counts),
            "candidate_identity_unverified_count": candidate_count,
            "candidates_referenced_by_multiple_missing_rows": sum(
                count > 1 for count in candidate_reference_counts.values()
            ),
            "candidate_evidence_counts": dict(sorted(candidate_evidence_counts.items())),
            "identity_verified_from_candidate_match": False,
            "filesystem_scan_complete": not scan_failures,
            "filesystem_scan_incomplete_paths": scan_failures,
        },
        "source_analysis": {
            "missing_reference_counts_by_source": {
                source: sum(counts.values())
                for source, counts in sorted(source_classifications.items())
            },
            "references_with_source_recording_id": explicit_source_id_count,
            "references_without_source_recording_id": len(records) - explicit_source_id_count,
            "missing_records_with_filename_group_key": sum(
                record["performance_identity_source"] == "filename_pattern"
                for record in records
            ),
            "unique_missing_filename_group_keys": len(missing_filename_groups),
            "unique_present_filename_group_keys": len(present_filename_groups),
            "filename_group_key_overlap_with_present_records": len(
                missing_filename_groups.intersection(present_filename_groups)
            ),
            "source_path_conventions_documented_for_missing_rows": [],
            "note": (
                "Filename-derived performance grouping keys are not source recording "
                "IDs. No source mapping or source-specific path convention is inferred "
                "when the manifest lacks record-level evidence."
            ),
        },
        "classification_definitions": {
            "path_convention_mismatch": (
                "A local file matches the expected path after a case-only or normalized "
                "path comparison; location convention is a candidate explanation, not "
                "identity proof."
            ),
            "alternate_local_location_candidate": (
                "A local file has an exact filename, same stem, source-ID token, or "
                "path-suffix match outside the expected location; identity is unverified."
            ),
            "source_metadata_present_audio_unlocated": (
                "Source and source_recording_id metadata are present in the manifest, "
                "but the referenced audio was not located in the searched local "
                "filesystem. Upstream source availability has not been verified."
            ),
            "possible_segment_reference": (
                "A different chunk filename with the same bracketed filename identifier "
                "exists locally; segment relation is only possible."
            ),
            "duplicate_reference_candidate": (
                "The same local candidate is suggested for multiple missing manifest "
                "rows; each mapping requires review."
            ),
            "unresolved": (
                "No supported local explanation was found within the searched locations; "
                "excluded locations and upstream availability were not checked."
            ),
        },
        "evidence_level_definitions": {
            "exact_path_match": (
                "The expected path exists exactly; such rows are present and excluded "
                "from missing-reference records."
            ),
            "exact_filename_match": (
                "Same basename exists locally at another path; does not establish asset identity."
            ),
            "normalized_path_match": (
                "Same normalized path components are found in another local location; "
                "candidate only."
            ),
            "explicit_source_identifier_match": (
                "An explicit source_recording_id token occurs in a local filename; "
                "semantics and identity remain unverified."
            ),
            "same_filename_source_id_different_chunk": (
                "A filename-derived bracket ID and different chunk number match; only "
                "a possible segment relationship."
            ),
            "filename_similarity_only": (
                "Same stem with different extension/name details; weak candidate evidence only."
            ),
        },
        "limitations": [
            "No external datasets or restricted sources were accessed.",
        "The forensic analysis does not decode, load, or hash any audio file.",
            "No candidate file is decoded or treated as the missing asset; path and name "
            "evidence cannot prove byte identity.",
            "The repository search excludes the directory names listed in "
            "candidate_analysis.excluded_directory_names. Symlink directories and "
            "files are skipped; locations outside the configured search root are not searched.",
            "Missing source mapping, acquisition history, and upstream path documentation "
            "cannot be reconstructed from filenames alone.",
            "Absolute manifest paths are redacted; a SHA-256 of the original path value "
            "is retained for record correlation.",
        ],
        "records": records,
    }


def render_forensics_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable summary; the JSON carries all row-level evidence."""
    summary = report["summary"]
    lines = [
        "# Missing Audio Reference Forensics — D2.1 Phase B1",
        "",
        f"Audit run: {report['audit_timestamp']} (UTC)",
        "",
        "> Read-only analysis. Candidate paths are evidence for review and do not "
        "establish recording identity.",
        "",
        "## A. Executive summary",
        "",
        "| Measure | Count |",
        "|---|---:|",
        f"| Manifest records examined | {summary['manifest_records_examined']} |",
        f"| Missing references | {summary['missing_references']} |",
        f"| Present references | {summary['present_references']} |",
        f"| Missing references analyzed | {summary['references_analyzed']} |",
        f"| Unresolved references | {summary['unresolved_references']} |",
        "| Local candidate matches (candidate-record pairs) | "
        f"{summary['local_audio_candidates_found']} |",
        f"| Unique local candidate files | {summary['unique_local_candidate_files']} |",
        "",
        f"Dataset root: `{report['dataset_root_identifier']}`  ",
        f"Manifest: `{report['manifest_path']}`  ",
        f"Manifest fingerprint: `{report['manifest_fingerprint']}`  ",
        f"Repository commit: `{report['repository_commit']}`",
        f"Source tree clean: `{report['repository_source_tree_clean']}`  ",
        f"Source tree fingerprint: `{report['repository_source_tree_fingerprint']}`",
        f"Manifest fingerprint policy: `{report['manifest_fingerprint_policy']}`",
        "",
        "## B. Classification summary",
        "",
        "| Classification | Count | Percent |",
        "|---|---:|---:|",
    ]
    for name, count in report["classification_counts"].items():
        lines.append(f"| `{name}` | {count} | {report['classification_percentages'][name]:.2f}% |")
    lines.extend(["", "### By source", "", "| Source | Classification counts |", "|---|---|"])
    for source, counts in report["classifications_by_source"].items():
        lines.append(f"| `{source}` | `{json.dumps(counts, sort_keys=True)}` |")
    lines.extend([
        "",
        "## C. Path analysis",
        "",
        "- Relative/absolute/missing references: "
        f"`{report['path_analysis']['relative_versus_absolute']}`",
        f"- Extensions: `{report['path_analysis']['extension_distribution']}`",
        f"- Common directory prefixes: `{report['path_analysis']['common_missing_path_prefixes']}`",
        f"- Filename patterns: `{report['path_analysis']['common_filename_patterns']}`",
        "- Path-convention anomaly classifications: "
        f"{report['path_analysis']['path_convention_anomaly_count']}",
        f"- Existing path rules: `{report['path_analysis']['repository_path_convention']}`",
        "",
        "## D. Local candidate analysis",
        "",
        f"- Search scope: {report['candidate_analysis']['search_scope']}",
        f"- Local audio files searched: {report['candidate_analysis']['local_audio_file_count']}",
        "- Excluded directory names: "
        f"`{report['candidate_analysis']['excluded_directory_names']}`",
        "- Symlink directories/files followed/indexed: "
        f"{report['candidate_analysis']['symlink_directories_followed']}/"
        f"{report['candidate_analysis']['symlink_audio_files_indexed']}",
        "- Locations outside the search root were searched: "
        f"{report['candidate_analysis']['locations_outside_search_root_searched']}",
        "- Filesystem scan complete: "
        f"{report['candidate_analysis']['filesystem_scan_complete']}; incomplete paths: "
        f"{report['candidate_analysis']['filesystem_scan_incomplete_paths']}",
        "- Candidate evidence counts: "
        f"`{report['candidate_analysis']['candidate_evidence_counts']}`",
        "- Candidate files suggested by multiple rows: "
        f"{report['candidate_analysis']['candidates_referenced_by_multiple_missing_rows']}",
        "- No candidate match is treated as verified identity or exact audio duplication.",
        "- Complete record-level candidates, evidence, and limitations are in the "
        "JSON `records` array.",
        "",
        "### Classification meanings",
        "",
    ])
    for classification, definition in report["classification_definitions"].items():
        lines.append(f"- `{classification}`: {definition}")
    lines.extend([
        "",
        "## E. Source analysis",
        "",
        "- Missing references with an explicit source recording ID: "
        f"{report['source_analysis']['references_with_source_recording_id']}",
        "- Missing references without one: "
        f"{report['source_analysis']['references_without_source_recording_id']}",
        "- Missing rows with filename-derived grouping keys: "
        f"{report['source_analysis']['missing_records_with_filename_group_key']} rows, "
        f"{report['source_analysis']['unique_missing_filename_group_keys']} unique keys",
        "- Unique filename-derived grouping keys for present audio: "
        f"{report['source_analysis']['unique_present_filename_group_keys']}",
        "- Missing/present filename grouping key overlap: "
        f"{report['source_analysis']['filename_group_key_overlap_with_present_records']}",
        "- Source mapping/path conventions documented for these rows: "
        f"{len(report['source_analysis']['source_path_conventions_documented_for_missing_rows'])}",
        "- Source distributions are grouped from manifest fields only; no source is inferred.",
        "",
        "## F. Limitations",
        "",
    ])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend([
        "",
        "## G. Recommended next actions",
        "",
        "1. Compare unresolved references with upstream acquisition records or "
        "source-maintained manifests; do not download or repair files during this phase.",
        "2. Obtain source documentation for path conventions and the meaning of "
        "bracketed identifiers and chunk numbering.",
        "3. Manually review candidate path/source-ID relationships before recording "
        "any inventory classification beyond `unresolved`.",
        "4. Keep any future recovery or metadata changes in a separately authorized "
        "phase with provenance and content-hash verification.",
        "",
        "## Record-level results",
        "",
        "The JSON report contains one result per missing manifest reference. Absolute "
        "local path literals are redacted; relative manifest path values are preserved exactly.",
        "",
    ])
    return "\n".join(lines)


def write_forensics_reports(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "d2_1_missing_reference_forensics.json"
    markdown_path = output / "d2_1_missing_reference_forensics.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_forensics_markdown(report), encoding="utf-8")
    return json_path, markdown_path
