# Dataset metadata contract

`DatasetItem` is the shared, versioned record used by dataset adapters and
training manifests. New fields are optional or have conservative legacy
defaults so existing JSON manifests continue to load. The default
`metadata_schema_version` is `1.0`; new producers should set the schema version
they emit. Old manifests without an explicit `record_status` are classified as
`legacy`, and missing rights metadata remains `rights_status="unknown"`.

The Hindustani adapter keeps its established canonical-only default so existing
training and benchmark partitions do not change. Callers that need to inspect
every source label can set `include_noncanonical_ragas=True`; unmapped labels
are then retained verbatim in `raga` and `raw_raga_label` without canonical
conversion.

## Field contract

| Field | Type / default | Meaning |
|---|---|---|
| `recording_id` | required non-empty string | Stable recording identity. |
| `file_path` | required non-empty string | Path to the recording. |
| `audio_sha256` | optional 64-digit hex string | Content digest; normalized to lowercase. |
| `raga` | required non-empty string | Label consumed by current training code; unknown labels remain raw. |
| `raw_raga_label` | optional string | Source annotation before canonical mapping. |
| `tradition` | `Hindustani` | Existing tradition behavior. |
| `tonic_hz` | optional positive finite number | Annotated tonic frequency. |
| `duration_sec` | required positive finite number | Recording duration. |
| `sample_rate` | required positive integer | Audio sample rate. |
| `split` | `train` | Existing `train`, `val`, or `test` assignment. |
| `performer` | optional string | Display name only. |
| `performer_id` | optional string | Stable identity supplied by source or curation. |
| `performance_id` | optional string | Performance grouping identity. |
| `performance_identity_source` | `unknown` | `metadata`, `filename_pattern`, `manual_review`, or `unknown`. A source does not imply verification. |
| `source` | optional string, absent by default | Source collection or dataset name when known. |
| `source_recording_id` | optional string | Original source identifier. |
| `source_url` | optional string | Original recording or dataset location. |
| `license` / `license_url` | optional strings, both absent by default | Explicit license information and its evidence. |
| `attribution` | optional string | Required credit when known. |
| `rights_status` | `unknown` | `verified`, `pending`, `restricted`, or `unknown`. |
| `annotation_status` | absent by default | `pending`, `reviewed`, `accepted`, or `rejected`. |
| `annotation_confidence` | absent by default | Optional finite number in [0, 1]. |
| `annotation_notes` | optional string | Review or annotation notes. |
| `record_status` | `legacy` | `legacy`, `candidate`, `accepted`, `quarantined`, or `rejected`. |
| `metadata_schema_version` | `1.0` | Version of the emitted metadata schema. |

## Identity and annotations

- `recording_id` identifies a recording; `file_path` locates its audio;
  `audio_sha256` is an optional content digest.
- `raga` is the adapter's label used by current training code. `raw_raga_label`
  preserves the input annotation. Adapters do not map unknown labels to a
  canonical raga.
- `performer` is a display name. `performer_id` is the independently supplied
  stable identity. No ID is inferred from a display name.
- `performance_id` is the performance grouping value;
  `performance_identity_source` says whether it came from metadata, the known
  filename pattern, manual review, or an unknown origin. Only an independently
  reviewed identity (`manual_review`) is treated as verified by split validation;
  other IDs can still serve as grouping keys while verification remains
  `not_verifiable`.
- The Saraga adapter accepts top-level `performance_id`/`concert_id`, or explicit
  `performance_id`, `concert_id`, or context-scoped `id` values inside `performance`
  or `concert` objects. It ignores titles/names and does not use `release`, `album`,
  `work`, or composition metadata as performance identity. When no accepted ID is
  present, the recording ID is a grouping fallback with source `unknown`.
- Annotation status is optional and, when present, is `pending`, `reviewed`,
  `accepted`, or `rejected`. Confidence is optional and constrained to [0, 1].

## Provenance and lifecycle

`source`, `source_recording_id`, `source_url`, `license`, `license_url`, and
`attribution` describe provenance evidence. Missing license values remain
`None`; `rights_status` defaults to `unknown`. The status values are `verified`,
`pending`, `restricted`, and `unknown`. A license string alone is not marked
verified unless the adapter has documented evidence.

`record_status` is one of `legacy`, `candidate`, `accepted`, `quarantined`, or
`rejected`. Legacy JSON without the field receives the conservative `legacy`
default. Audits never modify or reclassify source records.

## Fingerprints and compatibility

`compute_metadata_fingerprint()` and `compute_dataset_fingerprint()` retain
their historical v1 field projection and audio policy for existing caches and
experiment provenance. Expanded metadata is fingerprinted explicitly with
`compute_metadata_fingerprint_v2()` or `compute_dataset_fingerprint_v2()`;
these use the `audio_sha256_metadata_v2` policy. Callers must record which
policy they use and must not compare fingerprints from different policies as
if they were equivalent.

The CSV dataset audit reads new fields when present and reports missing values
as missing; it does not populate defaults into or rewrite the source CSV.
Performer isolation is based only on `performer_id`, never a display name.
