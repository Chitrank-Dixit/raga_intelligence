"""
Adapter for the MTG CompMusic Saraga Hindustani Dataset (Zenodo 4301737).
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
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
        seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.filter_canonical = filter_canonical_ragas_only
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def _partition_items(self, parsed_items: List[DatasetItem]) -> List[DatasetItem]:
        """
        Deterministically partitions items into train/val/test using group-based splitting
        by stable performer ID (or performance/recording ID fallback). Keeps each known
        grouping key within one split; it cannot verify identity when only a fallback exists.
        Display names are never used as stable IDs.
        """
        if not parsed_items:
            return []

        groups: Dict[str, List[DatasetItem]] = defaultdict(list)
        missing_performer_id_count = 0

        for item in parsed_items:
            if item.performer_id and item.performer_id.strip().lower() not in {
                "unknown", "none", "null"
            }:
                group_key = f"performer::{item.performer_id}"
            elif item.performance_id:
                group_key = f"performance::{item.performance_id}"
                missing_performer_id_count += 1
            else:
                group_key = f"recording::{item.recording_id}"
                missing_performer_id_count += 1
            groups[group_key].append(item)

        if missing_performer_id_count > 0:
            logger.warning(
                f"Stable performer IDs unavailable for {missing_performer_id_count}/{len(parsed_items)} items. "
                "Fell back to performance/recording group key for those items."
            )

        # Deterministically order groups using seed and group content
        def group_sort_key(g_key: str):
            h = hashlib.sha256(f"{self.seed}::{g_key}".encode("utf-8")).hexdigest()
            return (-len(groups[g_key]), h)

        sorted_group_keys = sorted(groups.keys(), key=group_sort_key)

        # Compute target proportions
        total_items = len(parsed_items)
        split_names = ["train", "val", "test"]
        split_ratios = {"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio}

        raga_totals: Dict[str, int] = defaultdict(int)
        for item in parsed_items:
            raga_totals[item.raga] += 1

        target_totals = {s: total_items * split_ratios[s] for s in split_names}
        target_raga = {s: {r: raga_totals[r] * split_ratios[s] for r in raga_totals} for s in split_names}

        curr_totals = {s: 0 for s in split_names}
        curr_raga: Dict[str, Dict[str, int]] = {s: defaultdict(int) for s in split_names}

        for g_key in sorted_group_keys:
            group_items = groups[g_key]
            g_raga_counts: Dict[str, int] = defaultdict(int)
            for it in group_items:
                g_raga_counts[it.raga] += 1

            best_split = "train"
            best_score = -float("inf")

            for s in split_names:
                score = 0.0
                for r, count in g_raga_counts.items():
                    deficit = target_raga[s][r] - curr_raga[s][r]
                    score += deficit * count
                capacity_deficit = target_totals[s] - curr_totals[s]
                score += 0.5 * capacity_deficit

                if score > best_score:
                    best_score = score
                    best_split = s

            for it in group_items:
                it.split = best_split
                curr_totals[best_split] += 1
                curr_raga[best_split][it.raga] += 1

        return parsed_items

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

                artist, performer_id = _extract_performer_identity(data)

                perf_id, performance_identity_source = _extract_performance_identity(data, rec_id)

                parsed_items.append(
                    DatasetItem(
                        recording_id=rec_id,
                        source="SaragaHindustani",
                        file_path=str(audio_path),
                        raga=canonical_name,
                        raw_raga_label=raw_raga,
                        tradition="Hindustani",
                        performer=(
                            artist
                            if artist and artist.strip().lower() not in {"unknown", "none", "null"}
                            else None
                        ),
                        performer_id=performer_id,
                        performance_id=perf_id,
                        performance_identity_source=performance_identity_source,
                        source_recording_id=rec_id,
                        duration_sec=float(data.get("duration", 0.0)),
                        sample_rate=int(data.get("sample_rate", 44100)),
                        tonic_hz=tonic_hz,
                        split="train",  # placeholder, partitioned below
                        license="CC BY-NC-SA 4.0",
                        source_url="https://doi.org/10.5281/zenodo.4301737",
                        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
                        attribution="Music Technology Group, Universitat Pompeu Fabra",
                        rights_status="verified",
                        metadata={"tala": str(data.get("taal", "Unknown"))},
                    )
                )
            except Exception as e:
                logger.debug(f"Failed parsing metadata file {mfile}: {e}")

        # Deterministic grouping partition; identity verification remains source-dependent.
        partitioned_items = self._partition_items(parsed_items)
        for item in partitioned_items:
            manifest.add_item(item)

        return manifest


def _extract_performer_identity(data: dict) -> tuple[Optional[str], Optional[str]]:
    """Return display name and an explicitly supplied stable performer ID, if present."""
    top_level_identity = _known_identifier(data.get("performer_id") or data.get("artist_id"))
    for key in ("lead_artist", "artist"):
        value = data.get(key)
        if isinstance(value, dict):
            name = value.get("name")
            identity = value.get("performer_id") or value.get("artist_id") or value.get("id")
            return _known_text(name), _known_identifier(identity) or top_level_identity
        if value is not None:
            return _known_text(value), top_level_identity

    artists = data.get("artists")
    if isinstance(artists, list):
        for value in artists:
            if isinstance(value, dict) and (value.get("role") == "lead" or "name" in value):
                name = value.get("name")
                identity = value.get("performer_id") or value.get("artist_id") or value.get("id")
                return _known_text(name), _known_identifier(identity) or top_level_identity
            if isinstance(value, str):
                return _known_text(value), top_level_identity

    return _known_text(data.get("performer")), top_level_identity


def _known_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text.lower() not in {"", "unknown", "none", "null"} else None


def _known_identifier(value: object) -> Optional[str]:
    if isinstance(value, (dict, list, tuple, set)):
        return None
    return _known_text(value)


def _extract_performance_identity(data: dict, recording_id: str) -> tuple[str, str]:
    """Read only explicit IDs in performance/concert fields; otherwise use an unknown recording fallback.

    Accepted source fields are top-level ``performance_id``/``concert_id`` and
    ``id``/those explicit ID keys nested in a ``performance`` or ``concert``
    object. Titles and names, including release/album/work/composition metadata,
    are descriptive values and never become performance identities.
    """
    for field in ("performance_id", "concert_id"):
        identity = _known_identifier(data.get(field))
        if identity:
            return identity, "metadata"

    for field in ("performance", "concert"):
        value = data.get(field)
        if not isinstance(value, dict):
            continue
        for id_field in ("performance_id", "concert_id", "id"):
            identity = _known_identifier(value.get(id_field))
            if identity:
                return identity, "metadata"

    return recording_id, "unknown"

