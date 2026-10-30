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

    def validate_splits(self) -> Dict[str, bool | List[str]]:
        """
        Verifies that no recording or performance is shared across splits (preventing data leakage).
        Also inspects performer overlap across splits.
        """
        train_ids = {it.recording_id for it in self.items if it.split == "train"}
        val_ids = {it.recording_id for it in self.items if it.split == "val"}
        test_ids = {it.recording_id for it in self.items if it.split == "test"}

        rec_leakage = list((train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids))

        train_performers = {it.performer for it in self.items if it.split == "train" and it.performer != "Unknown"}
        test_performers = {it.performer for it in self.items if it.split == "test" and it.performer != "Unknown"}
        performer_overlap = list(train_performers & test_performers)

        return {
            "valid_no_recording_leakage": len(rec_leakage) == 0,
            "leaked_recordings": rec_leakage,
            "train_recordings_count": len(train_ids),
            "val_recordings_count": len(val_ids),
            "test_recordings_count": len(test_ids),
            "performer_overlap_in_test": performer_overlap,
        }

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

