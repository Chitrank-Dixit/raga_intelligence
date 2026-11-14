"""
Reliable, atomic feature and embedding caching with cryptographic fingerprint validation
and strict provenance auditing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest
from raga_intelligence.features.schema import FeatureSchema

logger = logging.getLogger(__name__)


@dataclass
class CacheValidationReport:
    """Provenance audit report verifying feature/embedding caches against a dataset manifest."""
    is_valid: bool
    dataset_fingerprint: str
    feature_schema_version: str
    n_train: int
    n_test: int
    n_val: int
    recording_ids_train: List[str]
    recording_ids_test: List[str]
    recording_ids_val: List[str]
    errors: List[str] = field(default_factory=list)
    feat_cache_path: Optional[str] = None
    feat_cache_sha256: Optional[str] = None
    mert_cache_path: Optional[str] = None
    mert_cache_sha256: Optional[str] = None


def compute_file_sha256(path: Path | str, chunk_size: int = 65536) -> str:
    """
    Computes a deterministic cryptographic SHA-256 hex digest of a file's raw bytes.
    Used for tamper-evident provenance tracking of exact cache and audio files.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found for SHA-256 computation: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest_content_fingerprint(
    items: List[DatasetItem],
) -> str:
    """
    Computes a deterministic cryptographic fingerprint of a dataset manifest.

    Requires a valid audio SHA-256 for every item and excludes machine-specific paths.
    """
    sorted_items = sorted(items, key=lambda x: (x.recording_id, x.split, x.source or "", x.performance_id or ""))
    descriptors: List[str] = []

    for it in sorted_items:
        if not it.audio_sha256 or not re.fullmatch(r"[0-9a-f]{64}", it.audio_sha256):
            raise ValueError(f"Missing or invalid audio SHA-256 for recording {it.recording_id}")
        if not Path(it.file_path).is_file():
            raise FileNotFoundError(f"Audio file missing for recording {it.recording_id}: {it.file_path}")
        actual_sha = compute_file_sha256(it.file_path)
        if not re.fullmatch(r"[0-9a-f]{64}", actual_sha):
            raise ValueError(f"Could not compute a valid audio SHA-256 for recording {it.recording_id}")
        it.audio_sha256 = actual_sha
        desc = json.dumps(
            {
                "recording_id": it.recording_id,
                "source": it.source,
                "split": it.split,
                "raga": it.raga,
                "tonic_hz": it.tonic_hz,
                "performer": it.performer,
                "performance_id": it.performance_id,
                "duration_sec": it.duration_sec,
                "sample_rate": it.sample_rate,
                "tradition": it.tradition,
                "license": it.license,
                "metadata": it.metadata,
                "audio_sha256": actual_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        descriptors.append(desc)

    combined = "audio_sha256_v1|" + "|".join(sorted(descriptors))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _extract_meta_dict(cache_data: Dict[str, Any] | np.lib.npyio.NpzFile) -> Optional[Dict[str, Any]]:
    """Helper to safely extract the metadata dictionary from a numpy cache object."""
    if "metadata" not in cache_data:
        return None
    raw = cache_data["metadata"]
    if hasattr(raw, "item") and raw.shape == ():
        val = raw.item()
        return dict(val) if isinstance(val, dict) else None
    if isinstance(raw, dict):
        return raw
    return None


def validate_cache_manifest_provenance(
    manifest: DatasetManifest,
    feat_cache_data: Dict[str, Any] | np.lib.npyio.NpzFile,
    mert_cache_data: Optional[Dict[str, Any] | np.lib.npyio.NpzFile] = None,
    raise_on_error: bool = True,
    feat_cache_path: Optional[Path | str] = None,
    mert_cache_path: Optional[Path | str] = None,
) -> CacheValidationReport:
    """
    Validates exact provenance, ordering, labels, splits, and fingerprints between
    a DatasetManifest and cached feature/embedding arrays.

    Fails explicitly if:
    - Metadata is missing or unreadable
    - Dataset fingerprint does not match the freshly computed manifest fingerprint
    - Feature schema version is missing or mismatched
    - Recording IDs or splits are missing
    - Recording IDs or labels do not match manifest split items in exact order
    - Handcrafted and MERT caches differ in recording IDs, labels, or split assignments
    """
    errors: List[str] = []
    expected_fp = manifest.compute_dataset_fingerprint(force_recompute=True)

    # 1. Inspect handcrafted cache metadata
    feat_meta = _extract_meta_dict(feat_cache_data)
    if feat_meta is None:
        errors.append("Handcrafted feature cache is missing 'metadata' section; provenance cannot be verified")
        feat_fp = None
        feat_schema = None
    else:
        feat_fp = feat_meta.get("dataset_fingerprint")
        feat_policy = feat_meta.get("fingerprint_policy")
        feat_schema = feat_meta.get("feature_schema_version")
        if feat_policy != "audio_sha256_v1":
            errors.append("Handcrafted feature cache uses a missing or unsupported fingerprint policy")
        if feat_fp != expected_fp:
            errors.append(
                f"Handcrafted feature cache dataset fingerprint mismatch "
                f"(cached: {feat_fp} != manifest: {expected_fp})"
            )
        if not feat_schema:
            errors.append("Handcrafted feature cache is missing 'feature_schema_version' in metadata")
        elif feat_schema != FeatureSchema.SCHEMA_VERSION:
            errors.append(
                f"Handcrafted feature cache schema version mismatch: {feat_schema} != {FeatureSchema.SCHEMA_VERSION}"
            )

    # 2. Extract manifest split items
    manifest_train = [it for it in manifest.items if it.split == "train"]
    manifest_test = [it for it in manifest.items if it.split == "test"]
    manifest_val = [it for it in manifest.items if it.split == "val"]

    manifest_train_ids = [it.recording_id for it in manifest_train]
    manifest_test_ids = [it.recording_id for it in manifest_test]
    manifest_val_ids = [it.recording_id for it in manifest_val]

    manifest_train_labels = [it.raga for it in manifest_train]
    manifest_test_labels = [it.raga for it in manifest_test]

    # 3. Check handcrafted recording IDs
    if "recording_ids_train" in feat_cache_data:
        feat_train_ids = [str(x) for x in feat_cache_data["recording_ids_train"]]
    elif "ids_train" in feat_cache_data:
        feat_train_ids = [str(x) for x in feat_cache_data["ids_train"]]
    else:
        errors.append("Handcrafted feature cache missing train recording IDs ('recording_ids_train')")
        feat_train_ids = []

    if "recording_ids_test" in feat_cache_data:
        feat_test_ids = [str(x) for x in feat_cache_data["recording_ids_test"]]
    elif "ids_test" in feat_cache_data:
        feat_test_ids = [str(x) for x in feat_cache_data["ids_test"]]
    else:
        errors.append("Handcrafted feature cache missing test recording IDs ('recording_ids_test')")
        feat_test_ids = []

    if feat_train_ids and feat_train_ids != manifest_train_ids:
        errors.append(
            f"Handcrafted cache train recording IDs do not match manifest train split exactly "
            f"(count: {len(feat_train_ids)} vs {len(manifest_train_ids)})"
        )
    if feat_test_ids and feat_test_ids != manifest_test_ids:
        errors.append(
            f"Handcrafted cache test recording IDs do not match manifest test split exactly "
            f"(count: {len(feat_test_ids)} vs {len(manifest_test_ids)})"
        )

    # 4. Check handcrafted labels
    if "y_train" in feat_cache_data:
        feat_y_train = [str(x) for x in feat_cache_data["y_train"]]
        if feat_train_ids and feat_y_train != manifest_train_labels:
            errors.append("Handcrafted cache train labels do not match manifest train labels exactly")
    else:
        errors.append("Handcrafted cache missing 'y_train'")

    if "y_test" in feat_cache_data:
        feat_y_test = [str(x) for x in feat_cache_data["y_test"]]
        if feat_test_ids and feat_y_test != manifest_test_labels:
            errors.append("Handcrafted cache test labels do not match manifest test labels exactly")
    else:
        errors.append("Handcrafted cache missing 'y_test'")

    # 5. Check splits arrays in handcrafted cache
    if "splits_train" in feat_cache_data:
        splits_train = [str(x) for x in feat_cache_data["splits_train"]]
        if any(s != "train" for s in splits_train):
            errors.append("Handcrafted cache 'splits_train' contains non-train assignments")
    else:
        errors.append("Handcrafted cache missing 'splits_train' array")

    if "splits_test" in feat_cache_data:
        splits_test = [str(x) for x in feat_cache_data["splits_test"]]
        if any(s != "test" for s in splits_test):
            errors.append("Handcrafted cache 'splits_test' contains non-test assignments")
    else:
        errors.append("Handcrafted cache missing 'splits_test' array")

    # 6. Validate MERT cache if supplied
    if mert_cache_data is not None:
        mert_meta = _extract_meta_dict(mert_cache_data)
        if mert_meta is None:
            errors.append("MERT cache is missing 'metadata' section; provenance cannot be verified")
        else:
            mert_fp = mert_meta.get("dataset_fingerprint")
            mert_policy = mert_meta.get("fingerprint_policy")
            if mert_policy != "audio_sha256_v1":
                errors.append("MERT cache uses a missing or unsupported fingerprint policy")
            if mert_fp != expected_fp:
                errors.append(
                    f"MERT cache dataset fingerprint mismatch (cached: {mert_fp} != manifest: {expected_fp})"
                )

        if "recording_ids_train" in mert_cache_data:
            mert_train_ids = [str(x) for x in mert_cache_data["recording_ids_train"]]
        elif "ids_train" in mert_cache_data:
            mert_train_ids = [str(x) for x in mert_cache_data["ids_train"]]
        else:
            errors.append("MERT cache missing train recording IDs ('recording_ids_train')")
            mert_train_ids = []

        if "recording_ids_test" in mert_cache_data:
            mert_test_ids = [str(x) for x in mert_cache_data["recording_ids_test"]]
        elif "ids_test" in mert_cache_data:
            mert_test_ids = [str(x) for x in mert_cache_data["ids_test"]]
        else:
            errors.append("MERT cache missing test recording IDs ('recording_ids_test')")
            mert_test_ids = []

        if mert_train_ids and mert_train_ids != feat_train_ids:
            errors.append("MERT cache train recording IDs do not match handcrafted feature cache")
        if mert_test_ids and mert_test_ids != feat_test_ids:
            errors.append("MERT cache test recording IDs do not match handcrafted feature cache")

        if "y_train" in mert_cache_data:
            mert_y_train = [str(x) for x in mert_cache_data["y_train"]]
            if feat_train_ids and mert_y_train != [str(x) for x in feat_cache_data.get("y_train", [])]:
                errors.append("MERT cache train labels do not match handcrafted cache train labels")
        else:
            errors.append("MERT cache missing 'y_train'")

        if "y_test" in mert_cache_data:
            mert_y_test = [str(x) for x in mert_cache_data["y_test"]]
            if feat_test_ids and mert_y_test != [str(x) for x in feat_cache_data.get("y_test", [])]:
                errors.append("MERT cache test labels do not match handcrafted cache test labels")
        else:
            errors.append("MERT cache missing 'y_test'")

        if "splits_train" in mert_cache_data:
            m_splits_train = [str(x) for x in mert_cache_data["splits_train"]]
            if any(s != "train" for s in m_splits_train):
                errors.append("MERT cache 'splits_train' contains non-train assignments")
        else:
            errors.append("MERT cache missing 'splits_train' array")

        if "splits_test" in mert_cache_data:
            m_splits_test = [str(x) for x in mert_cache_data["splits_test"]]
            if any(s != "test" for s in m_splits_test):
                errors.append("MERT cache 'splits_test' contains non-test assignments")
        else:
            errors.append("MERT cache missing 'splits_test' array")

    # Compute file SHA-256 for physical cache files if paths are provided
    feat_sha: Optional[str] = None
    if feat_cache_path is not None:
        p_feat = Path(feat_cache_path)
        if p_feat.exists():
            feat_sha = compute_file_sha256(p_feat)

    mert_sha: Optional[str] = None
    if mert_cache_path is not None:
        p_mert = Path(mert_cache_path)
        if p_mert.exists():
            mert_sha = compute_file_sha256(p_mert)

    is_valid = len(errors) == 0
    report = CacheValidationReport(
        is_valid=is_valid,
        dataset_fingerprint=expected_fp,
        feature_schema_version=feat_schema or "unknown",
        n_train=len(feat_train_ids),
        n_test=len(feat_test_ids),
        n_val=len(manifest_val_ids),
        recording_ids_train=feat_train_ids,
        recording_ids_test=feat_test_ids,
        recording_ids_val=manifest_val_ids,
        errors=errors,
        feat_cache_path=str(feat_cache_path) if feat_cache_path else None,
        feat_cache_sha256=feat_sha,
        mert_cache_path=str(mert_cache_path) if mert_cache_path else None,
        mert_cache_sha256=mert_sha,
    )

    if not is_valid and raise_on_error:
        error_msg = f"Dataset cache provenance validation failed ({len(errors)} violations):\n  - " + "\n  - ".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    return report


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
