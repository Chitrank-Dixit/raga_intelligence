"""
Dataset manifest schema, validation, and split integrity verification.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


def _known_identity(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() not in {"", "unknown", "none", "null"}


def _isolation_status(leakage_found: bool, metadata_complete: bool) -> str:
    if leakage_found:
        return "failed"
    return "verified" if metadata_complete else "not_verifiable"


def _canonical_item_key(item: "DatasetItem") -> str:
    return json.dumps(
        {
            "recording_id": item.recording_id,
            "split": item.split,
            "source": item.source,
            "performance_id": item.performance_id,
            "raga": item.raga,
            "tonic_hz": item.tonic_hz,
            "performer": item.performer,
            "duration_sec": item.duration_sec,
            "sample_rate": item.sample_rate,
            "tradition": item.tradition,
            "license": item.license,
            "metadata": item.metadata,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _performance_key(item: "DatasetItem") -> str:
    return item.performance_id if _known_identity(item.performance_id) else item.recording_id


class DatasetItem(BaseModel):
    """Versioned metadata for one recording; absent provenance remains unknown."""
    recording_id: str
    source: Optional[str] = None
    file_path: str
    raga: str
    tradition: str = "Hindustani"
    performer: Optional[str] = None
    performer_id: Optional[str] = None
    performance_id: Optional[str] = None
    performance_identity_source: Literal[
        "metadata", "filename_pattern", "manual_review", "unknown"
    ] = "unknown"
    duration_sec: float = Field(gt=0.0, allow_inf_nan=False)
    sample_rate: int = Field(gt=0)
    tonic_hz: Optional[float] = Field(default=None, gt=0.0, allow_inf_nan=False)
    split: Literal["train", "val", "test"] = "train"
    # No default license: older records without an explicit value are unknown.
    license: Optional[str] = None
    source_recording_id: Optional[str] = None
    source_url: Optional[str] = None
    license_url: Optional[str] = None
    attribution: Optional[str] = None
    rights_status: Literal["verified", "pending", "restricted", "unknown"] = "unknown"
    raw_raga_label: Optional[str] = None
    annotation_status: Optional[Literal["pending", "reviewed", "accepted", "rejected"]] = None
    annotation_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    annotation_notes: Optional[str] = None
    record_status: Literal[
        "legacy", "candidate", "accepted", "quarantined", "rejected"
    ] = "legacy"
    metadata_schema_version: str = "1.0"
    metadata: Dict[str, str] = Field(default_factory=dict)
    audio_sha256: Optional[str] = None

    @field_validator("recording_id", "file_path", "raga", "metadata_schema_version")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value

    @field_validator("audio_sha256")
    @classmethod
    def validate_audio_sha256(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            import re

            if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise ValueError("audio_sha256 must be a 64-character hexadecimal SHA-256")
        return value.lower() if value else value

    @model_validator(mode="after")
    def validate_performance_identity_source(self) -> "DatasetItem":
        if self.performance_identity_source != "unknown" and not _known_identity(self.performance_id):
            raise ValueError("performance_id is required when its identity source is known")
        return self


class DatasetManifest(BaseModel):
    """A collection of validated DatasetItems."""
    items: List[DatasetItem] = Field(default_factory=list)

    def add_item(self, item: DatasetItem) -> None:
        self.items.append(item)

    def validate_splits(self) -> Dict[str, bool | List[str] | int | Dict]:
        """
        Verifies that no recording, performance, or performer crosses dataset split boundaries.
        Returns detailed leakage diagnostics and a split distribution summary.
        """
        train_items = [it for it in self.items if it.split == "train"]
        val_items = [it for it in self.items if it.split == "val"]
        test_items = [it for it in self.items if it.split == "test"]

        # 1. Recording-level checks
        train_ids = {it.recording_id for it in train_items}
        val_ids = {it.recording_id for it in val_items}
        test_ids = {it.recording_id for it in test_items}
        rec_leakage = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)

        # 2. Performance-level checks
        train_perfs = {_performance_key(it) for it in train_items}
        val_perfs = {_performance_key(it) for it in val_items}
        test_perfs = {_performance_key(it) for it in test_items}
        perf_leakage = (train_perfs & val_perfs) | (train_perfs & test_perfs) | (val_perfs & test_perfs)

        # 3. Performer-level checks
        train_performers = {it.performer_id for it in train_items if _known_identity(it.performer_id)}
        val_performers = {it.performer_id for it in val_items if _known_identity(it.performer_id)}
        test_performers = {it.performer_id for it in test_items if _known_identity(it.performer_id)}
        performer_leakage = (train_performers & val_performers) | (train_performers & test_performers) | (val_performers & test_performers)

        # 4. Metadata completeness check
        unknown_performer_count = sum(1 for it in self.items if not _known_identity(it.performer_id))
        unknown_performance_count = sum(1 for it in self.items if not _known_identity(it.performance_id))
        performer_metadata_complete = (unknown_performer_count == 0)
        performance_metadata_complete = (unknown_performance_count == 0)

        split_summary = {
            "recordings_count": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
                "total": len(self.items),
            },
            "performances_count": {
                "train": len(train_perfs),
                "val": len(val_perfs),
                "test": len(test_perfs),
                "total": len(train_perfs | val_perfs | test_perfs),
            },
            "performers_count": {
                "train": len(train_performers),
                "val": len(val_performers),
                "test": len(test_performers),
                "total": len(train_performers | val_performers | test_performers),
            },
            "raga_distribution": self.get_class_counts(),
            "performer_metadata_complete": performer_metadata_complete,
            "performance_metadata_complete": performance_metadata_complete,
            "unknown_performer_count": unknown_performer_count,
            "unknown_performance_count": unknown_performance_count,
        }

        # 5. Isolation Status Assessment (distinguishing verified isolation from unverifiable incomplete metadata)
        recording_metadata_complete = all(_known_identity(it.recording_id) for it in self.items)
        recording_isolation_status = _isolation_status(bool(rec_leakage), recording_metadata_complete)
        performer_isolation_status = _isolation_status(bool(performer_leakage), performer_metadata_complete)
        performance_isolation_status = _isolation_status(bool(perf_leakage), performance_metadata_complete)

        return {
            "valid_no_recording_leakage": len(rec_leakage) == 0,
            "valid_no_performance_leakage": performance_isolation_status == "verified",
            "valid_no_performer_leakage": performer_isolation_status == "verified",
            "recording_isolation_status": recording_isolation_status,
            "is_performer_isolation_verified": performer_isolation_status == "verified",
            "is_performance_isolation_verified": performance_isolation_status == "verified",
            "performer_isolation_status": performer_isolation_status,
            "performance_isolation_status": performance_isolation_status,
            "leaked_recordings": sorted(list(rec_leakage)),
            "leaked_performances": sorted(list(perf_leakage)),
            "leaked_performers": sorted(list(performer_leakage)),
            "performer_metadata_complete": performer_metadata_complete,
            "performance_metadata_complete": performance_metadata_complete,
            "unknown_performer_count": unknown_performer_count,
            "unknown_performance_count": unknown_performance_count,
            "train_recordings_count": len(train_ids),
            "val_recordings_count": len(val_ids),
            "test_recordings_count": len(test_ids),
            "performer_overlap_in_test": sorted(list(train_performers & test_performers)),
            "split_summary": split_summary,
        }

    def compute_metadata_fingerprint(self) -> str:
        """
        Computes a stable, deterministic SHA-256 fingerprint strictly over dataset metadata.
        
        Guarantees that semantic annotations have not drifted:
        recording ID, split assignment, canonical raga, tonic (Sa), duration,
        sample rate, performer, and performance identifiers.
        
        Does not require audio files to be present on disk.
        """
        import hashlib
        sorted_items = sorted(self.items, key=_canonical_item_key)
        descriptors = [
            json.dumps(
                {
                    "recording_id": it.recording_id,
                    "split": it.split,
                    "raga": it.raga,
                    "tonic_hz": it.tonic_hz,
                    "performer": it.performer,
                    "performance_id": it.performance_id,
                    "duration_sec": it.duration_sec,
                    "sample_rate": it.sample_rate,
                    "source": it.source,
                    "tradition": it.tradition,
                    "license": it.license,
                    "metadata": it.metadata,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for it in sorted_items
        ]
        return hashlib.sha256("|".join(descriptors).encode("utf-8")).hexdigest()

    def compute_metadata_fingerprint_v2(self) -> str:
        """Fingerprint the expanded metadata contract under an explicit new policy."""
        descriptors = sorted(
            json.dumps(
                {
                    "recording_id": item.recording_id,
                    "split": item.split,
                    "raga": item.raga,
                    "raw_raga_label": item.raw_raga_label,
                    "tonic_hz": item.tonic_hz,
                    "tradition": item.tradition,
                    "performer": item.performer,
                    "performer_id": item.performer_id,
                    "performance_id": item.performance_id,
                    "performance_identity_source": item.performance_identity_source,
                    "duration_sec": item.duration_sec,
                    "sample_rate": item.sample_rate,
                    "source": item.source,
                    "source_recording_id": item.source_recording_id,
                    "source_url": item.source_url,
                    "license": item.license,
                    "license_url": item.license_url,
                    "attribution": item.attribution,
                    "rights_status": item.rights_status,
                    "annotation_status": item.annotation_status,
                    "annotation_confidence": item.annotation_confidence,
                    "annotation_notes": item.annotation_notes,
                    "record_status": item.record_status,
                    "metadata_schema_version": item.metadata_schema_version,
                    "metadata": item.metadata,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in self.items
        )
        return hashlib.sha256("|".join(descriptors).encode("utf-8")).hexdigest()

    def compute_dataset_fingerprint_v2(self) -> str:
        """Content fingerprint using expanded metadata and audio SHA-256 (policy v2)."""
        self._require_audio_hashes()
        self.populate_audio_hashes(force_recompute=True)
        payload = json.dumps(
            {
                "fingerprint_policy": "audio_sha256_metadata_v2",
                "metadata_fingerprint": self.compute_metadata_fingerprint_v2(),
                "audio_content_fingerprint": self._audio_content_fingerprint_from_hashes(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def populate_audio_hashes(self, force_recompute: bool = False) -> None:
        """
        Computes and records SHA-256 content hashes for all physical audio files in the manifest.
        Mutates items in-place with their computed audio_sha256.
        """
        for it in self.items:
            p = Path(it.file_path)
            try:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                it.audio_sha256 = h.hexdigest()
            except OSError as exc:
                raise OSError(f"Could not compute audio SHA-256 for {p}: {exc}") from exc

    def compute_audio_content_fingerprint(self, force_recompute: bool = True) -> str:
        """Return a fail-closed fingerprint of every audio file's current bytes."""
        import hashlib

        self._require_audio_hashes()
        self.populate_audio_hashes(force_recompute=True)
        return self._audio_content_fingerprint_from_hashes()

    def _require_audio_hashes(self) -> None:
        import re

        missing = [
            item.file_path
            for item in self.items
            if not item.audio_sha256 or not re.fullmatch(r"[0-9a-f]{64}", item.audio_sha256)
        ]
        if missing:
            raise ValueError(
                f"Missing or invalid audio SHA-256 for {len(missing)} recording(s): {missing[:5]}"
            )

    def _audio_content_fingerprint_from_hashes(self) -> str:
        import hashlib

        audio_descriptors = [
            {"recording_id": item.recording_id, "audio_sha256": item.audio_sha256}
            for item in sorted(self.items, key=_canonical_item_key)
        ]
        payload = json.dumps(audio_descriptors, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compute_dataset_fingerprint(
        self,
        force_recompute: bool = False,
    ) -> str:
        """
        Computes a stable, deterministic cryptographic fingerprint of all dataset items.
        
        Guarantees:
        1. Metadata consistency (labels, splits, tonics, performers, durations).
        2. Audio content integrity (audio file byte hashes when audio_sha256 is present).
        """
        import hashlib

        self._require_audio_hashes()
        self.populate_audio_hashes(force_recompute=True)
        payload = json.dumps(
            {
                "fingerprint_policy": "audio_sha256_v1",
                "metadata_fingerprint": self.compute_metadata_fingerprint(),
                "audio_content_fingerprint": self._audio_content_fingerprint_from_hashes(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_class_counts(self) -> Dict[str, Dict[str, int]]:
        """Returns per-class item counts grouped by split."""
        summary: Dict[str, Dict[str, int]] = {}
        for it in self.items:
            if it.raga not in summary:
                summary[it.raga] = {"train": 0, "val": 0, "test": 0, "total": 0}
            summary[it.raga][it.split] += 1
            summary[it.raga]["total"] += 1
        return summary

    def to_pandas(self) -> pd.DataFrame:
        data = [it.model_dump() for it in self.items]
        return pd.DataFrame(data)

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump([it.model_dump() for it in self.items], f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> DatasetManifest:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = [DatasetItem(**row) for row in data]
        return cls(items=items)
