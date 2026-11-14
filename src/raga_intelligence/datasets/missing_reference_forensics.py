"""Read-only forensic analysis of missing CSV audio references."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from raga_intelligence.datasets.audit import _relative_path, audit_dataset

_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma"}
_EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    ".venv-linux",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".task_pytest_tmp",
    "node_modules",
    "reports",
}
_CLASSIFICATIONS = (
    "alternate_local_location_candidate",
    "duplicate_reference_candidate",
    "path_convention_mismatch",
    "possible_segment_reference",
    "source_record_available_audio_missing",
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


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=3
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _is_absolute_reference(value: str) -> bool:
    return (
        value.lower().startswith("file://")
        or Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or bool(_ABSOLUTE_WINDOWS_PATTERN.match(value))
    )


def _normalized_reference(value: str) -> str:
    normalized = value.replace("\\", "/")
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


def _raw_audio_value(row: dict[str, str]) -> str | None:
    normalized = {key.strip().lower(): value for key, value in row.items()}
    for name in ("file_name", "file_path", "audio_path", "path"):
        value = normalized.get(name)
        if value is not None and value.strip():
            return value
    return None


def _safe_reference(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if _is_absolute_reference(value):
        import hashlib

        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
        return "<absolute path redacted>", digest
    return value, None


def _record_source(row: dict[str, str]) -> tuple[str | None, str | None]:
    normalized = {key.strip().lower(): value for key, value in row.items()}

    def first(names: tuple[str, ...]) -> str | None:
        for name in names:
            value = normalized.get(name, "").strip()
            if value and value.casefold() not in {"unknown", "none", "null"}:
                return value
        return None

    source = first(("source", "dataset_source", "source_collection", "collection"))
    source_recording_id = first(("source_recording_id",))
    safe_source, _ = _safe_reference(source)
    safe_source_recording_id, _ = _safe_reference(source_recording_id)
    return safe_source, safe_source_recording_id


def _candidate_evidence(
    reference: str,
    normalized_path: str | None,
    source_recording_id: str | None,
    candidates: list[Path],
    search_root: Path,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    if not normalized_path:
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
        return "source_record_available_audio_missing"
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

    audit = audit_dataset(root, metadata_file)
    csv_path = Path(metadata_file) if metadata_file else root / "metadata.csv"
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MissingReferenceForensicsError(
            f"Could not reread metadata CSV: {type(exc).__name__}"
        ) from exc

    audio_files, scan_failures = _audio_files(search)
    records: list[dict[str, Any]] = []
    for audited in audit["records"]:
        if audited["audio"]["exists"]:
            continue
        row_index = int(audited["metadata_row"]) - 2
        raw_row = rows[row_index] if 0 <= row_index < len(rows) else {}
        original = _raw_audio_value(raw_row)
        normalized, expected_path, path_error = _relative_path(root, original)
        source, source_recording_id = _record_source(raw_row)
        safe_original, reference_hash = _safe_reference(original)
        normalized_reference = (
            _normalized_reference(original)
            if original and not _is_absolute_reference(original)
            else None
        )
        local_candidates = _candidate_evidence(
            original or "",
            normalized,
            source_recording_id,
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
            "metadata_row": audited["metadata_row"],
            "recording_id": audited["recording_id"],
            "original_reference_path": safe_original,
            "absolute_reference_redacted": reference_hash is not None,
            "reference_path_sha256": reference_hash,
            "normalized_reference_path": normalized_reference,
            "resolved_expected_path": relative_expected,
            "reference_path_type": (
                "absolute"
                if original and _is_absolute_reference(original)
                else "relative"
                if original
                else "missing"
            ),
            "path_resolution_error": path_error,
            "reference_exists": False,
            "source": source,
            "source_recording_id": source_recording_id,
            "performance_id": audited["performance_id"],
            "performance_identity_source": audited["performance_identity_source"],
            "raw_raga_label": audited["raw_raga_label"],
            "performer_display_name": audited["performer_display_name"],
            "performer_id": audited["performer_id"],
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

    present_references = [item for item in audit["records"] if item["audio"]["exists"]]
    missing_filename_groups = {
        record["performance_id"]
        for record in records
        if record["performance_identity_source"] == "filename_pattern"
        and record["performance_id"]
    }
    present_filename_groups = {
        record["performance_id"]
        for record in present_references
        if record["performance_identity_source"] == "filename_pattern"
        and record["performance_id"]
    }
    metadata_path = Path(metadata_file) if metadata_file else root / "metadata.csv"
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    try:
        manifest_path = metadata_path.resolve().relative_to(search).as_posix()
    except ValueError:
        manifest_path = metadata_path.name

    return {
        "schema_version": "1.0",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_commit(),
        "repository_source_tree_clean": audit["repository_source_tree_clean"],
        "repository_source_tree_fingerprint": audit["repository_source_tree_fingerprint"],
        "dataset_root_identifier": root.name,
        "search_root_identifier": search.name,
        "manifest_path": manifest_path,
        "manifest_fingerprint": audit["manifest_fingerprint"],
        "path_resolution_policy": "datasets.audit._relative_path",
        "summary": {
            "manifest_records_examined": len(audit["records"]),
            "missing_references": len(records),
            "present_references": len(present_references),
            "references_analyzed": len(records),
            "unresolved_references": unresolved_count,
            "unique_missing_paths": len(
                {
                    record["normalized_reference_path"] or record["original_reference_path"]
                    for record in records
                }
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
            "local_audio_file_count": len(audio_files),
            "candidate_match_count": candidate_count,
            "unique_candidate_file_count": len(candidate_reference_counts),
            "candidate_identity_unverified_count": candidate_count,
            "candidates_referenced_by_multiple_missing_rows": sum(
                count > 1 for count in candidate_reference_counts.values()
            ),
            "candidate_evidence_counts": dict(sorted(candidate_evidence_counts.items())),
            "identity_verified_from_candidate_match": False,
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
            "source_record_available_audio_missing": (
                "Both source and source_recording_id are explicit, but no matching "
                "local audio candidate was found."
            ),
            "possible_segment_reference": (
                "A different chunk filename with the same bracketed filename identifier "
                "exists locally; segment relation is only possible."
            ),
            "duplicate_reference_candidate": (
                "The same local candidate is suggested for multiple missing manifest "
                "rows; each mapping requires review."
            ),
            "unresolved": "No supported local explanation was found.",
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
            "No candidate file is decoded or treated as the missing asset; path and name "
            "evidence cannot prove byte identity.",
            "The repository search excludes Git metadata, Python environments/caches, "
            "and generated report directories.",
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
        f"- Local audio files searched: {report['candidate_analysis']['local_audio_file_count']}",
        "- Candidate evidence counts: "
        f"`{report['candidate_analysis']['candidate_evidence_counts']}`",
        "- Candidate files suggested by multiple rows: "
        f"{report['candidate_analysis']['candidates_referenced_by_multiple_missing_rows']}",
        "- No candidate match is treated as verified identity or exact audio duplication.",
        "- Complete record-level candidates, evidence, and limitations are in the "
        "JSON `records` array.",
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
