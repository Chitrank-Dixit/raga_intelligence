# Dataset quality audit

Run `python scripts/audit_dataset.py` to audit the configured
`DATA_DIR/hindustani_dataset` dataset. Pass `--dataset-dir` to inspect another
CSV-backed dataset, `--metadata-file` to select its manifest, and `--output-dir`
to choose where JSON and Markdown reports are written. Report output must be
outside the dataset root. By default, JSON and Markdown reports are generated
locally under `reports/dataset_audit/`.
These outputs are ignored by Git because they contain machine- and dataset-specific
audit results; keep the implementation and documentation under version control.

The audit reads each manifest row, hashes audio in chunks, and decodes one audio
file at a time with Librosa. It does not edit audio, metadata, caches, or
experiment records. It reports raw labels as well as labels mapped to the
configured canonical raga vocabulary. A filename-derived performance identity
is used only when the CSV has no performance identity column and the existing
adapter's bracketed ID convention matches; its origin is marked in JSON. A
present but empty or unknown identity field remains unknown. Performer IDs are
read only from explicit `performer_id` or `artist_id` columns; display names are
reported separately and never treated as performer identities.

Only `performance_id` and the `concert_id` documented in
[`dataset_metadata_contract.md`](dataset_metadata_contract.md) are read as
explicit performance identity fields. `performance`, `source_id`, `video_id`,
album/release/work, and composition fields are retained as ambiguous source
metadata in each record but are never promoted to performance IDs. A
`manual_review` source claim is applied only when an accepted explicit
performance identifier is present; filename grouping and ambiguous fields
cannot upgrade it.

Missing referenced files, unreadable audio, duplicate recording IDs, invalid
annotations, leakage, and incomplete canonical train/test coverage are errors
and give the CLI a nonzero status. Missing performer or rights/provenance fields
are warnings. Audio fingerprints are unavailable if any referenced file cannot
be hashed. Reports omit absolute dataset paths and include relative audio paths.
Evaluation readiness is calculated for rows with labels in the configured
canonical raga vocabulary; dataset-wide integrity findings remain separately
reported and can still make the CLI return nonzero.

The metadata fingerprint covers normalized manifest facts and relative paths.
The audio content fingerprint is a separate hash over relative paths and file
SHA-256 values and is unavailable unless every referenced file can be hashed.

The audit separates split grouping from identity verification. Filename-derived
IDs remain useful for checking whether a grouping key overlaps splits, but are
not independently verified performance identities. Source metadata is identified
as source-declared rather than independently reviewed; only records marked with
`manual_review` provide the audit's independent identity evidence. JSON and
Markdown output expose reliability levels and keep canonical evaluation readiness
separate from dataset-wide integrity errors.

The audit also reports schema-version, lifecycle, annotation, rights, and
performance-identity-source distributions when those columns exist. Missing
schema and lifecycle values remain visible as missing; auditing does not assign
legacy/accepted or rights statuses to the CSV records. See
[`dataset_metadata_contract.md`](dataset_metadata_contract.md) for the shared
`DatasetItem` contract and the explicit metadata fingerprint v2 policy.

## Missing-reference forensics

Run `python scripts/analyze_missing_audio_references.py` to investigate each
missing reference in the configured `DATA_DIR/hindustani_dataset` manifest.
Use `--dataset-dir`, `--metadata-file`, and `--search-root` to select the
dataset, CSV, and local filesystem tree to inspect. By default, local candidate
search is limited to the project root and excludes Git metadata, Python
environments/caches, pytest scratch directories, and generated report
directories. `--output-dir` defaults to the ignored `reports/dataset_audit/`
directory and must remain outside the dataset root.

The command reuses the audit's path-resolution behavior and reports one JSON
result per missing manifest row, plus aggregate Markdown. Classifications such
as `alternate_local_location_candidate` and `possible_segment_reference` are
evidence labels for human review; neither a similar filename nor a shared
filename-derived ID proves recording identity. Evidence is recorded separately
from classification. The report distinguishes exact filename, normalized path,
explicit source-ID, chunk-sibling, and filename-similarity evidence. An
unresolved result is retained when no local explanation is supported. Absolute
manifest paths are redacted in reports. Repository commit, source-tree
cleanliness, and the source-tree fingerprint are included for reproducibility.
The forensic `manifest_fingerprint` is explicitly identified as
`normalized_manifest_csv_rows_v1`; it is a fingerprint of normalized CSV rows,
not the full audio audit's metadata fingerprint.

The local search recursively indexes `.aac`, `.flac`, `.m4a`, `.mp3`, `.ogg`,
`.wav`, and `.wma` filenames under `--search-root`. It does not open, hash, or
decode candidate audio. It skips directories named `.cache`, `.coverage`,
`.env`, `.git`, `.mypy_cache`, `.nox`, `.tox`, `.venv`, `.venv-linux`,
`.pytest_cache`, `.ruff_cache`, `.task_pytest_tmp`, `__pycache__`, `build`,
`cache`, `dist`, `env`, `generated`, `htmlcov`, `node_modules`, `reports`,
`site-packages`, `target`, or `venv`. It does not follow symlink directories,
index symlink files, or search outside the configured root. The JSON and Markdown
reports list these exclusions and disclose incomplete filesystem scans. An
unresolved result means no explanation was found in the accessible searched
locations; excluded folders, other local roots, and upstream availability remain
unverified.

Classification meanings:

- `path_convention_mismatch`: a local file matches the expected path after path
  normalization or case folding; this is a location candidate, not identity proof.
- `alternate_local_location_candidate`: a filename, path suffix, source-ID token,
  or same-stem match exists elsewhere locally; the relationship is unverified.
- `source_metadata_present_audio_unlocated`: source and source-recording ID
  metadata are present, but audio was not located under the searched root.
  Upstream source-record availability has **not** been verified.
- `possible_segment_reference`: a local file has the same bracketed filename
  group key with a different chunk number; segmentation remains a possibility.
- `duplicate_reference_candidate`: one local candidate is suggested for more
  than one missing manifest row; each relationship requires review.
- `unresolved`: no supported explanation was found in the searched locations.

Candidate source-ID matches use exact token sequences from explicit
`source_recording_id` metadata. Filename-derived grouping IDs are considered
only for the possible-chunk check. Neither establishes recording or performance
identity; candidate records always set `identity_established` to `false`.
Relative path values are trimmed and separator-normalized for resolution while
the original relative value is retained. Absolute paths and `file://` references
are redacted and represented by deterministic path fingerprints. Empty paths
remain missing; unsupported absolute paths are not treated as relative paths.

This analysis only reads the manifest and local filesystem. It does not decode,
copy, recover, or modify audio; edit metadata; access external sources; or infer
licenses, identities, or rights. It does not change the existing audit's
duplicate-content definition.
