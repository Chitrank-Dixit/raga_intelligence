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
