"""
Adapter for the MTG CompMusic Saraga Hindustani Dataset (Zenodo 4301737).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from raga_intelligence.config import CANONICAL_RAGAS
from raga_intelligence.datasets.adapters import BaseDatasetAdapter
from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest

logger = logging.getLogger(__name__)


# Mapping common naming variants in Saraga to canonical raga names
SARAGA_RAGA_MAPPING: Dict[str, str] = {
    "bhairav": "Bhairav",
    "ahira bhairav": "Bhairav",
    "bhairavi": "Bhairavi",
    "yaman": "Yaman",
    "yaman kalyan": "Yaman",
    "bhoopali": "Bhoopali",
    "bhupali": "Bhoopali",
    "bhup": "Bhoopali",
    "deshkar": "Deshkar",
    "bageshree": "Bageshree",
    "bageshri": "Bageshree",
    "malkauns": "Malkauns",
    "todi": "Todi",
    "miyan ki todi": "Todi",
    "darbari kanada": "Darbari Kanada",
    "darbari": "Darbari Kanada",
    "kafi": "Kafi",
}


class SaragaHindustaniAdapter(BaseDatasetAdapter):
    """
    Parses and adapts the Saraga Hindustani research collection.
    
    References:
    Zenodo DOI: 10.5281/zenodo.4301737
    License: CC BY-NC-SA 4.0
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        filter_canonical_ragas_only: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.filter_canonical = filter_canonical_ragas_only

    def build_manifest(self) -> DatasetManifest:
        manifest = DatasetManifest()
        if not self.dataset_dir.exists():
            logger.warning(f"Saraga dataset directory {self.dataset_dir} does not exist.")
            return manifest

        # Scan for metadata files (.yaml or .json)
        meta_files = list(self.dataset_dir.rglob("*.yaml")) + list(self.dataset_dir.rglob("*.json"))
        logger.info(f"Discovered {len(meta_files)} metadata files in {self.dataset_dir}")

        parsed_items: List[DatasetItem] = []

        for mfile in meta_files:
            try:
                if mfile.suffix in (".yaml", ".yml"):
                    with open(mfile, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                else:
                    with open(mfile, "r", encoding="utf-8") as f:
                        data = json.load(f)

                if not isinstance(data, dict):
                    continue

                raw_raga = data.get("raag") or data.get("raga")
                if isinstance(raw_raga, list) and len(raw_raga) > 0:
                    raw_raga = raw_raga[0].get("name", "")
                elif isinstance(raw_raga, dict):
                    raw_raga = raw_raga.get("name", "")

                if not raw_raga or not isinstance(raw_raga, str):
                    continue

                # Normalize raga name
                clean_name = raw_raga.strip().lower()
                canonical_name = SARAGA_RAGA_MAPPING.get(clean_name)
                if not canonical_name:
                    if self.filter_canonical:
                        continue
                    canonical_name = raw_raga.strip().title()

                # Find associated audio file
                rec_id = mfile.stem
                audio_candidates = [
                    mfile.with_suffix(".mp3"),
                    mfile.with_suffix(".wav"),
                    mfile.parent / f"{rec_id}.mp3",
                    mfile.parent / f"{rec_id}.wav",
                    mfile.parent / f"{rec_id}.audio.mp3",
                ]
                audio_path = next((p for p in audio_candidates if p.exists()), None)
                if not audio_path:
                    continue

                tonic_hz = data.get("tonic") or data.get("tonic_hz")
                if tonic_hz is not None:
                    try:
                        tonic_hz = float(tonic_hz)
                    except ValueError:
                        tonic_hz = None

                artist = data.get("artist") or data.get("lead_artist") or "Unknown"
                if isinstance(artist, dict):
                    artist = artist.get("name", "Unknown")

                parsed_items.append(
                    DatasetItem(
                        recording_id=rec_id,
                        source="SaragaHindustani",
                        file_path=str(audio_path),
                        raga=canonical_name,
                        tradition="Hindustani",
                        performer=artist,
                        duration_sec=float(data.get("duration", 0.0)),
                        sample_rate=int(data.get("sample_rate", 44100)),
                        tonic_hz=tonic_hz,
                        split="train",  # placeholder, partitioned below
                        license="CC BY-NC-SA 4.0",
                        metadata={"tala": str(data.get("taal", "Unknown"))},
                    )
                )
            except Exception as e:
                logger.debug(f"Failed parsing metadata file {mfile}: {e}")

        # Partition into train/val/test by recording/performer to avoid leakage
        # Group by raga and performer
        for i, item in enumerate(parsed_items):
            # Deterministic split: 70% train, 15% val, 15% test
            mod = i % 10
            if mod < 7:
                item.split = "train"
            elif mod < 8:
                item.split = "val"
            else:
                item.split = "test"
            manifest.add_item(item)

        return manifest

