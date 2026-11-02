"""
Reliable, atomic feature and embedding caching with cryptographic fingerprint validation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from raga_intelligence.datasets.manifest import DatasetItem

logger = logging.getLogger(__name__)


def compute_manifest_content_fingerprint(
    items: List[DatasetItem],
    include_file_stats: bool = True,
) -> str:
    """
    Computes a deterministic cryptographic fingerprint of a dataset manifest.

    Incorporates:
      - Recording ID, split, canonical raga, tonic.
      - File path, file size, and modification time (mtime) if file exists.
    Any alteration of audio content or split assignment will invalidate this fingerprint.
    """
    sorted_items = sorted(items, key=lambda x: (x.recording_id, x.split, x.file_path))
    descriptors: List[str] = []

    for it in sorted_items:
        file_stat_desc = "missing"
        p = Path(it.file_path)
        if include_file_stats and p.exists():
            try:
                st = p.stat()
                file_stat_desc = f"sz:{st.st_size}:mt:{int(st.st_mtime)}"
            except Exception:
                file_stat_desc = "unreadable"

        desc = f"{it.recording_id}:{it.split}:{it.raga}:{it.tonic_hz}:{it.file_path}:{file_stat_desc}"
        descriptors.append(desc)

    combined = "|".join(descriptors)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class FeatureCacheManager:
    """
    Manages loading and atomic persistence of feature and embedding arrays
    with strict metadata validation and corruption recovery.
    """

    @staticmethod
    def load_validated_cache(
        cache_path: Path | str,
        expected_metadata: Dict[str, Any],
        force_refresh: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Loads cached numpy archive if and only if all metadata fingerprint fields match.

        Returns None if cache is missing, corrupted, invalidated, or force_refresh is True.
        """
        p = Path(cache_path)
        if not p.exists() or force_refresh:
            if force_refresh and p.exists():
                logger.info(f"Force refresh requested: bypassing existing cache at {p}")
            return None

        try:
            data = np.load(p, allow_pickle=True)
            if "metadata" not in data:
                logger.warning(f"Cache invalidated for {p}: missing metadata section. Re-extracting.")
                return None

            cached_meta = data["metadata"].item() if hasattr(data["metadata"], "item") else dict(data["metadata"])
            for key, expected_val in expected_metadata.items():
                cached_val = cached_meta.get(key)
                if cached_val != expected_val:
                    logger.info(
                        f"Cache invalidated for {p}: parameter '{key}' changed "
                        f"(cached: {cached_val} != expected: {expected_val}). Re-extracting."
                    )
                    return None

            logger.info(f"Validated cache hit at {p} (fingerprint: {expected_metadata.get('manifest_hash', '')[:12]}...)")
            # Convert NpzFile to dict of arrays
            result = {k: data[k] for k in data.files}
            return result

        except Exception as e:
            logger.warning(f"Corrupted or unreadable cache file at {p} ({e}). Safely regenerating cache.")
            return None

    @staticmethod
    def save_atomic_cache(
        cache_path: Path | str,
        arrays: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Atomically saves feature arrays and metadata dictionary using a temporary file.
        Guarantees that partially written or interrupted files will not corrupt the cache.
        """
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_name(f"{p.stem}_{os.getpid()}_{hashlib.md5(p.name.encode()).hexdigest()[:8]}.tmp.npz")

        try:
            np.savez_compressed(
                tmp_p,
                **arrays,
                metadata=metadata,
            )
            # Atomic file replacement
            os.replace(tmp_p, p)
            logger.info(f"Atomically cached {len(arrays)} arrays and metadata to {p}")
        except Exception as e:
            if tmp_p.exists():
                try:
                    tmp_p.unlink()
                except Exception:
                    pass
            logger.error(f"Failed to write atomic cache to {p}: {e}")
            raise

