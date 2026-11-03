"""
Dataset manifest schema, validation, and split integrity verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    """Metadata record for a single audio recording."""
    recording_id: str
    source: str
    file_path: str
    raga: str
    tradition: str = "Hindustani"
    performer: str = "Unknown"
    performance_id: Optional[str] = None
    duration_sec: float = Field(ge=0.0)
    sample_rate: int = Field(gt=0)
    tonic_hz: Optional[float] = None
    split: Literal["train", "val", "test"] = "train"
    license: str = "CC BY-NC-SA 4.0"
    metadata: Dict[str, str] = Field(default_factory=dict)


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
        train_perfs = {it.performance_id or it.recording_id for it in train_items}
        val_perfs = {it.performance_id or it.recording_id for it in val_items}
        test_perfs = {it.performance_id or it.recording_id for it in test_items}
        perf_leakage = (train_perfs & val_perfs) | (train_perfs & test_perfs) | (val_perfs & test_perfs)

        # 3. Performer-level checks
        train_performers = {it.performer for it in train_items if it.performer not in ("Unknown", "unknown", None, "")}
        val_performers = {it.performer for it in val_items if it.performer not in ("Unknown", "unknown", None, "")}
        test_performers = {it.performer for it in test_items if it.performer not in ("Unknown", "unknown", None, "")}
        performer_leakage = (train_performers & val_performers) | (train_performers & test_performers) | (val_performers & test_performers)

        # 4. Metadata completeness check
        unknown_performer_count = sum(1 for it in self.items if it.performer in ("Unknown", "unknown", None, ""))
        unknown_performance_count = sum(1 for it in self.items if not it.performance_id)
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
        if len(performer_leakage) > 0:
            performer_isolation_status = "leaked"
        elif performer_metadata_complete:
            performer_isolation_status = "verified_isolated"
        else:
            performer_isolation_status = "unverifiable_incomplete_metadata"

        if len(perf_leakage) > 0:
            performance_isolation_status = "leaked"
        elif performance_metadata_complete:
            performance_isolation_status = "verified_isolated"
        else:
            performance_isolation_status = "unverifiable_incomplete_metadata"

        return {
            "valid_no_recording_leakage": len(rec_leakage) == 0,
            "valid_no_performance_leakage": len(perf_leakage) == 0,
            "valid_no_performer_leakage": len(performer_leakage) == 0,
            "is_performer_isolation_verified": performer_isolation_status == "verified_isolated",
            "is_performance_isolation_verified": performance_isolation_status == "verified_isolated",
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

    def compute_dataset_fingerprint(self) -> str:
        """
        Computes a stable, deterministic SHA256 fingerprint of all dataset items.
        
        Includes recording ID, split assignment, canonical raga, tonic, duration,
        sample rate, performer, and performance identifiers.
        """
        import hashlib
        sorted_items = sorted(self.items, key=lambda x: (x.recording_id, x.split, x.file_path))
        descriptors = [
            f"{it.recording_id}:{it.split}:{it.raga}:{it.tonic_hz}:{it.performer}:{it.performance_id}:{it.duration_sec}:{it.sample_rate}"
            for it in sorted_items
        ]
        return hashlib.sha256("|".join(descriptors).encode("utf-8")).hexdigest()

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

