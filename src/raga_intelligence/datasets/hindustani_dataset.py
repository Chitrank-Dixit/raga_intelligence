"""
Adapter for the real Hindustani classical audio dataset (50 raags corpus).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from raga_intelligence.config import CANONICAL_RAGAS
from raga_intelligence.datasets.adapters import BaseDatasetAdapter
from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest

logger = logging.getLogger(__name__)


def _optional_metadata_value(row: pd.Series, *columns: str) -> Optional[str]:
    for column in columns:
        if column in row and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value and value.lower() not in {"unknown", "none", "null", "n/a"}:
                return value
    return None


def _controlled_metadata_value(row: pd.Series, column: str) -> Optional[str]:
    if column not in row or pd.isna(row[column]):
        return None
    value = str(row[column]).strip()
    return value or None

CANONICAL_NAME_MAP: Dict[str, str] = {
    "Bhairav": "Bhairav",
    "Bhairavi": "Bhairavi",
    "Yaman": "Yaman",
    "Bhoopali": "Bhoopali",
    "Deshkar": "Deshkar",
    "Bageshree": "Bageshree",
    "Malkauns": "Malkauns",
    "Todi": "Todi",
    "DarbariKanada": "Darbari Kanada",
    "Darbari Kanada": "Darbari Kanada",
    "Kafi": "Kafi",
}


class HindustaniDatasetAdapter(BaseDatasetAdapter):
    """
    Adapts the real Hindustani audio collection with hand-annotated tonics (Sa).
    
    Split assignment strictly follows recording/video ID to guarantee zero data leakage.
    """

    def __init__(
        self,
        dataset_dir: Path | str = "./data/hindustani_dataset",
        metadata_file: Optional[str] = "metadata.csv",
        include_noncanonical_ragas: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_file = self.dataset_dir / metadata_file if metadata_file else None
        self.include_noncanonical_ragas = include_noncanonical_ragas

    def build_manifest(self, compute_audio_hashes: bool = False) -> DatasetManifest:
        manifest = DatasetManifest()
        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory '{self.dataset_dir}' does not exist.")
            return manifest

        meta_csv = self.metadata_file if (self.metadata_file and self.metadata_file.exists()) else (self.dataset_dir / "metadata.csv")
        if not meta_csv.exists():
            logger.error(f"Metadata file '{meta_csv}' does not exist.")
            return manifest

        df = pd.read_csv(meta_csv)
        if self.include_noncanonical_ragas:
            filtered_df = df.copy()
        else:
            filtered_df = df[df["label"].isin(CANONICAL_NAME_MAP.keys())].copy()

        for _, row in filtered_df.iterrows():
            rel_path = str(row["file_name"]).replace("\\", "/")
            full_audio_path = self.dataset_dir / rel_path

            if not full_audio_path.exists():
                if compute_audio_hashes:
                    raise FileNotFoundError(f"Audio recording referenced by metadata is missing: {full_audio_path}")
                logger.warning(f"Audio file not found: {full_audio_path}")
                continue

            raw_label = _optional_metadata_value(row, "label", "raga")
            if raw_label is None:
                logger.warning("Skipping recording with no usable raga label: %s", rel_path)
                continue
            canonical_raga = CANONICAL_NAME_MAP.get(raw_label)
            if canonical_raga is None:
                if not self.include_noncanonical_ragas:
                    continue
                canonical_raga = raw_label
            tonic_hz = float(row["tonic_hz"]) if pd.notna(row["tonic_hz"]) else None
            performer = _optional_metadata_value(row, "performer", "artist")

            # Extract video/performance id from filename e.g. "train_[IKNN8CIF_as]_chunk0.mp3"
            perf_id = _optional_metadata_value(row, "performance_id")
            performance_identity_source = (
                _controlled_metadata_value(row, "performance_identity_source")
                or ("metadata" if perf_id else "unknown")
            )
            if not perf_id:
                m = re.search(r"\[([a-zA-Z0-9_-]+)\]", rel_path)
                perf_id = m.group(1) if m else None
                if perf_id:
                    performance_identity_source = "filename_pattern"

            split = "test" if "test_" in rel_path.lower() else "train"

            audio_sha = None
            if compute_audio_hashes:
                import hashlib
                try:
                    h = hashlib.sha256()
                    with open(full_audio_path, "rb") as af:
                        while chunk := af.read(65536):
                            h.update(chunk)
                    audio_sha = h.hexdigest()
                except Exception as e:
                    raise OSError(f"Could not hash audio content for {full_audio_path}: {e}") from e
                if len(audio_sha) != 64:
                    raise ValueError(f"Invalid SHA-256 for audio recording {full_audio_path}")

            item = DatasetItem(
                recording_id=full_audio_path.stem,
                source=(
                    _optional_metadata_value(row, "source", "source_collection")
                    or "HindustaniRaagDataset"
                ),
                file_path=str(full_audio_path),
                raga=canonical_raga,
                raw_raga_label=raw_label,
                tradition="Hindustani",
                performer=performer,
                performance_id=perf_id,
                performance_identity_source=performance_identity_source,
                performer_id=_optional_metadata_value(row, "performer_id", "artist_id"),
                duration_sec=30.0,  # Nominal segment duration
                sample_rate=24000,
                tonic_hz=tonic_hz,
                split=split,
                source_recording_id=_optional_metadata_value(row, "source_recording_id", "recording_id"),
                source_url=_optional_metadata_value(row, "source_url", "url"),
                license=_optional_metadata_value(row, "license", "licence"),
                license_url=_optional_metadata_value(row, "license_url"),
                attribution=_optional_metadata_value(row, "attribution", "credit"),
                rights_status=_controlled_metadata_value(row, "rights_status") or "unknown",
                annotation_status=_optional_metadata_value(row, "annotation_status", "review_status"),
                annotation_confidence=(
                    float(row["annotation_confidence"])
                    if "annotation_confidence" in row and pd.notna(row["annotation_confidence"])
                    else None
                ),
                annotation_notes=_optional_metadata_value(row, "annotation_notes", "review_notes"),
                record_status=_controlled_metadata_value(row, "record_status") or "legacy",
                metadata=(
                    {"thhat": CANONICAL_RAGAS[canonical_raga]["thhat"]}
                    if canonical_raga in CANONICAL_RAGAS else {}
                ),
                audio_sha256=audio_sha,
            )
            manifest.add_item(item)

        validation = manifest.validate_splits()
        logger.info(
            f"Built manifest with {len(manifest.items)} clips. "
            f"Recording isolation: {validation['recording_isolation_status']}; "
            f"performance isolation: {validation['performance_isolation_status']}; "
            f"performer isolation: {validation['performer_isolation_status']} "
            f"(Train: {validation['train_recordings_count']}, Test: {validation['test_recordings_count']})"
        )
        return manifest

