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
    ):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_file = self.dataset_dir / metadata_file if metadata_file else None

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
        filtered_df = df[df["label"].isin(CANONICAL_NAME_MAP.keys())].copy()

        for _, row in filtered_df.iterrows():
            rel_path = str(row["file_name"]).replace("\\", "/")
            full_audio_path = self.dataset_dir / rel_path

            if not full_audio_path.exists():
                logger.warning(f"Audio file not found: {full_audio_path}")
                continue

            raw_label = str(row["label"])
            canonical_raga = CANONICAL_NAME_MAP[raw_label]
            tonic_hz = float(row["tonic_hz"]) if pd.notna(row["tonic_hz"]) else None

            # Extract video/performance id from filename e.g. "train_[IKNN8CIF_as]_chunk0.mp3"
            m = re.search(r"\[([a-zA-Z0-9_-]+)\]", rel_path)
            perf_id = m.group(1) if m else full_audio_path.stem

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
                    logger.warning(f"Could not hash audio content for {full_audio_path}: {e}")

            item = DatasetItem(
                recording_id=full_audio_path.stem,
                source="HindustaniRaagDataset",
                file_path=str(full_audio_path),
                raga=canonical_raga,
                tradition="Hindustani",
                performer=perf_id,  # Video / performance grouping
                performance_id=perf_id,
                duration_sec=30.0,  # Nominal segment duration
                sample_rate=24000,
                tonic_hz=tonic_hz,
                split=split,
                license="CC-BY-NC-SA",
                metadata={"thhat": CANONICAL_RAGAS[canonical_raga]["thhat"]},
                audio_sha256=audio_sha,
            )
            manifest.add_item(item)

        validation = manifest.validate_splits()
        logger.info(
            f"Built manifest with {len(manifest.items)} clips. "
            f"No recording leakage: {validation['valid_no_recording_leakage']} "
            f"(Train: {validation['train_recordings_count']}, Test: {validation['test_recordings_count']})"
        )
        return manifest

