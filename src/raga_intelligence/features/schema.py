"""
Centralized musical feature schema, block definitions, and dimension validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatureBlock:
    """Metadata describing a contiguous slice in the feature representation."""
    name: str
    start_idx: int
    end_idx: int
    description: str

    @property
    def dimension(self) -> int:
        return self.end_idx - self.start_idx

    @property
    def slice(self) -> slice:
        return slice(self.start_idx, self.end_idx)


class FeatureSchema:
    """
    Formal musical feature schema version 1.0.0.
    Authoritative reference for feature extraction, caching, and ablation slicing.
    """
    VERSION: str = "1.0.0"
    SCHEMA_VERSION: str = "1.0.0"

    BLOCKS: Dict[str, FeatureBlock] = {
        "pcd": FeatureBlock(
            name="pcd",
            start_idx=0,
            end_idx=12,
            description="12-bin tonic-relative pitch class distribution",
        ),
        "chroma": FeatureBlock(
            name="chroma",
            start_idx=12,
            end_idx=24,
            description="12-bin tonic-aligned Chroma CQT energy",
        ),
        "pitch_combined": FeatureBlock(
            name="pitch_combined",
            start_idx=0,
            end_idx=24,
            description="Combined 24-dim pitch class distribution and aligned chroma",
        ),
        "spectral": FeatureBlock(
            name="spectral",
            start_idx=24,
            end_idx=34,
            description="10-dim spectral descriptors: centroid, rolloff, flatness, contrast",
        ),
        "voiced": FeatureBlock(
            name="voiced",
            start_idx=34,
            end_idx=35,
            description="1-dim proportion of voiced frames",
        ),
        "spectral_and_voiced": FeatureBlock(
            name="spectral_and_voiced",
            start_idx=24,
            end_idx=35,
            description="Combined 11-dim spectral statistics and voiced frame ratio",
        ),
        "all_handcrafted_35": FeatureBlock(
            name="all_handcrafted_35",
            start_idx=0,
            end_idx=35,
            description="Full 35-dimensional baseline handcrafted representation",
        ),
        "melodic": FeatureBlock(
            name="melodic",
            start_idx=35,
            end_idx=75,
            description="40-dim sequential melodic motif intervals and transition stats",
        ),
        "all_handcrafted_75": FeatureBlock(
            name="all_handcrafted_75",
            start_idx=0,
            end_idx=75,
            description="Full 75-dimensional baseline + sequential melodic representation",
        ),
        "mert_1536": FeatureBlock(
            name="mert_1536",
            start_idx=0,
            end_idx=1536,
            description="1536-dim pooled self-supervised MERT embeddings",
        ),
        "hybrid_all": FeatureBlock(
            name="hybrid_all",
            start_idx=0,
            end_idx=1571,
            description="1571-dim concatenated handcrafted 35-dim and MERT 1536-dim vectors",
        ),
    }

    @classmethod
    def has_block(cls, name: str) -> bool:
        return name in cls.BLOCKS

    @classmethod
    def get_block(cls, name: str) -> FeatureBlock:
        if name not in cls.BLOCKS:
            raise KeyError(f"Unknown feature block '{name}'. Available: {list(cls.BLOCKS.keys())}")
        return cls.BLOCKS[name]

    @classmethod
    def get_slice(cls, name: str) -> slice:
        return cls.get_block(name).slice

    @classmethod
    def validate_array(
        cls,
        X: np.ndarray,
        expected_dim: Optional[int] = None,
        array_name: str = "feature array",
        expected_feature_type: Optional[str] = None,
    ) -> bool:
        """Validates that a feature array is 2D, finite, non-empty, and matches expected dimensions."""
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{array_name} must be a numpy ndarray, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"{array_name} must be a 2D matrix (samples, features), got ndim={X.ndim} with shape {X.shape}")
        if len(X) == 0:
            raise ValueError(f"{array_name} must contain at least 1 sample, got shape {X.shape}")

        if expected_feature_type is not None:
            if expected_feature_type == "mert_1536":
                expected_dim = 1536
            elif expected_feature_type in cls.BLOCKS:
                expected_dim = cls.BLOCKS[expected_feature_type].dimension
            else:
                raise ValueError(f"Unknown expected_feature_type '{expected_feature_type}'")
        elif expected_dim is None:
            if X.shape[1] not in (35, 75):
                raise ValueError(
                    f"Feature vector dimension {X.shape[1]} does not match supported schemas (35 or 75)."
                )

        if expected_dim is not None and X.shape[1] != expected_dim:
            if expected_feature_type == "mert_1536":
                raise ValueError(f"Expected MERT embedding dimension 1536, got {X.shape[1]}")
            raise ValueError(
                f"{array_name} dimension mismatch: expected {expected_dim} features, but got {X.shape[1]}."
            )
        if not np.all(np.isfinite(X)):
            nan_count = int(np.sum(np.isnan(X)))
            inf_count = int(np.sum(np.isinf(X)))
            raise ValueError(f"{array_name} contains non-finite values (NaN: {nan_count}, Inf: {inf_count}).")
        return True
