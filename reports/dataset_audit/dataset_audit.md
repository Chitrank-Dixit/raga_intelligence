# Dataset Quality Audit

- **Audit timestamp (UTC):** `2026-11-03T21:54:58.607658+00:00`
- **Repository commit:** `76d60d96e97194a83657caae7250494ac03f4122`
- **Relevant source tree clean:** `False`
- **Dataset root identifier:** `hindustani_dataset`
- **Manifest:** `metadata.csv`
- **Manifest fingerprint:** `6ae7dae1d56176054b640f6453cc3e0b5bcc4acceafbcebf5cb6667af78add66`
- **Manifest fingerprint policy:** `manifest_metadata_v1`
- **Audio content fingerprint:** `unavailable`
- **Audio fingerprint reason:** At least one referenced audio file is missing or could not be hashed.

## Inventory

- Manifest records: 1960
- Unique recording IDs: 1960
- Duplicate recording IDs: 0
- Valid records: 390
- Records with missing / invalid metadata: 0
- Audio files present / missing: 390 / 1570
- Audio files physically present under dataset root: 390
- Unreferenced audio files: 0
- Unreadable audio: 0
- Exact duplicate audio groups: 0

## Class, source, and split counts

| Raga | Records |
|---|---:|
| Bageshree | 43 |
| Bhairav | 33 |
| Bhairavi | 23 |
| Bhoopali | 38 |
| Darbari Kanada | 33 |
| Deshkar | 58 |
| Kafi | 38 |
| Malkauns | 53 |
| Todi | 33 |
| Yaman | 38 |

| Source | Records |
|---|---:|
| <missing> | 1960 |

| Split | Records |
|---|---:|
| test | 150 |
| train | 1810 |

## Evaluation readiness

- Recording isolation: **verified**
- Performance isolation: **verified** (verifiable: True)
- Performer isolation: **not_verifiable** (verifiable: False)
- Performance identity sources: `{'filename_pattern': 1960}`
- Evaluation scope records / unavailable audio: 390 / 0
- Canonical records without split assignment: 0
- Canonical evaluation scope ready: **True**
- Dataset has critical errors: **True**

### Canonical raga coverage

| Raga | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Bageshree | 43 | 40 | 0 | 3 |
| Bhairav | 33 | 30 | 0 | 3 |
| Bhairavi | 23 | 20 | 0 | 3 |
| Bhoopali | 38 | 35 | 0 | 3 |
| Darbari Kanada | 33 | 30 | 0 | 3 |
| Deshkar | 58 | 55 | 0 | 3 |
| Kafi | 38 | 35 | 0 | 3 |
| Malkauns | 53 | 50 | 0 | 3 |
| Todi | 33 | 30 | 0 | 3 |
| Yaman | 38 | 35 | 0 | 3 |
- Raga label metadata present: 1960/1960 (100.0%)
- Labels in canonical vocabulary: 390/1960 (19.9%)
- Missing annotation confidence: `None`
- Missing review status: `None`
- Missing tonic annotation: `0`
- Unknown raw labels: `40`

### Findings

- **ERROR `missing_audio`:** 1570 referenced audio file(s) are missing or have invalid paths.
- **WARNING `noncanonical_raga_labels`:** 40 raw raga label(s) are outside the configured canonical vocabulary; they are reported without being remapped.
- **WARNING `performer_isolation_unverifiable`:** 1960 record(s) lack performer metadata; performer-held-out generalization cannot be claimed.
- **WARNING `uncertain_rights`:** 1960 record(s) have unknown, pending, or restricted rights status.
- **WARNING `incomplete_source`:** 1960 record(s) lack source information.
- **WARNING `incomplete_source_url`:** 1960 record(s) lack source URL information.
- **WARNING `incomplete_license`:** 1960 record(s) lack license information.
- **WARNING `incomplete_attribution`:** 1960 record(s) lack attribution information.

## Provenance completeness

- Source Completeness: 0/1960 (0.0%)
- Source Url Completeness: 0/1960 (0.0%)
- License Completeness: 0/1960 (0.0%)
- Attribution Completeness: 0/1960 (0.0%)
- Record lifecycle classification: unavailable (no lifecycle status field was present).
- Access to a recording does not establish permission to use it; records were audited, not excluded. Lifecycle counts are unavailable when no status field exists.

Audit is read-only; source records and audio files are not modified.
