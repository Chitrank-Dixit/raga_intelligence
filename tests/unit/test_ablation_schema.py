"""
Unit tests for FeatureAblationStudy, FeatureSchema integration, weighted-F1 correctness,
and split identity persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from raga_intelligence.config import RAGA_LIST
from raga_intelligence.evaluation.ablation import FeatureAblationStudy
from raga_intelligence.features.schema import FeatureBlock, FeatureSchema


def test_schema_blocks_and_validation():
    """Verify FeatureSchema block structure and validation methods."""
    assert FeatureSchema.VERSION == "1.0.0"
    assert FeatureSchema.SCHEMA_VERSION == "1.0.0"

    pcd_block = FeatureSchema.get_block("pcd")
    assert pcd_block.start_idx == 0
    assert pcd_block.end_idx == 12
    assert pcd_block.dimension == 12

    chroma_block = FeatureSchema.get_block("chroma")
    assert chroma_block.start_idx == 12
    assert chroma_block.end_idx == 24

    handcrafted_block = FeatureSchema.get_block("all_handcrafted_35")
    assert handcrafted_block.dimension == 35

    # Validation succeeds for 35 or 75 dims
    valid_35 = np.zeros((10, 35), dtype=np.float32)
    valid_75 = np.zeros((10, 75), dtype=np.float32)
    valid_1536 = np.zeros((10, 1536), dtype=np.float32)

    assert FeatureSchema.validate_array(valid_35) is True
    assert FeatureSchema.validate_array(valid_75) is True
    assert FeatureSchema.validate_array(valid_1536, expected_feature_type="mert_1536") is True

    # Invalid dimension raises ValueError
    invalid_40 = np.zeros((10, 40), dtype=np.float32)
    with pytest.raises(ValueError, match="Feature vector dimension 40 does not match"):
        FeatureSchema.validate_array(invalid_40)

    # 1D array raises ValueError
    with pytest.raises(ValueError, match="must be a 2D matrix"):
        FeatureSchema.validate_array(np.zeros(35, dtype=np.float32))


def test_ablation_weighted_f1_correctness(tmp_path: Path):
    """
    Verify that weighted-F1 properly weights class F1 scores by their true class support,
    and is mathematically distinct from unweighted macro-F1 under class imbalance.
    """
    n_train = 40
    n_test = 20
    rng = np.random.default_rng(42)

    X_train = rng.standard_normal((n_train, 35)).astype(np.float32)
    X_test = rng.standard_normal((n_test, 35)).astype(np.float32)

    # Strongly imbalanced test labels: 18 Bhairav, 2 Yaman
    y_train = np.array(["Bhairav"] * 30 + ["Yaman"] * 10)
    y_test = np.array(["Bhairav"] * 18 + ["Yaman"] * 2)

    study = FeatureAblationStudy(
        class_names=["Bhairav", "Yaman"],
        classifier_type="logistic_regression",
        random_seed=42,
    )
    json_path = tmp_path / "ablation_imbalance.json"
    results = study.run_study(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_json_path=json_path,
        recording_ids_train=[f"tr_{i}" for i in range(n_train)],
        recording_ids_test=[f"te_{i}" for i in range(n_test)],
        dataset_fingerprint="mock_fingerprint_abc",
    )

    ref = results["all_handcrafted_35"]
    assert ref.weighted_f1 != ref.macro_f1, (
        f"Weighted-F1 ({ref.weighted_f1}) should differ from Macro-F1 ({ref.macro_f1}) under severe class imbalance."
    )

    # Verify JSON persistence of split identity and fingerprint
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_fingerprint"] == "mock_fingerprint_abc"
    assert data["feature_schema_version"] == FeatureSchema.SCHEMA_VERSION
    assert data["split_identity"]["n_train"] == n_train
    assert data["split_identity"]["n_test"] == n_test
    assert len(data["split_identity"]["recording_ids_train"]) == n_train
    assert len(data["split_identity"]["recording_ids_test"]) == n_test


def test_ablation_dimension_validation():
    """Verify that FeatureAblationStudy rejects invalid array dimensions and sample count mismatches."""
    study = FeatureAblationStudy(class_names=RAGA_LIST, random_seed=42)

    # Wrong feature dimension (e.g. 30 instead of 35 or 75)
    bad_x_train = np.zeros((20, 30), dtype=np.float32)
    valid_x_test = np.zeros((10, 35), dtype=np.float32)
    y_train = np.array(["Bhairav"] * 20)
    y_test = np.array(["Bhairav"] * 10)

    with pytest.raises(ValueError, match="Feature vector dimension 30 does not match"):
        study.run_study(
            X_train=bad_x_train,
            y_train=y_train,
            X_test=valid_x_test,
            y_test=y_test,
        )

    # Sample count mismatch between X and y
    valid_x_train = np.zeros((20, 35), dtype=np.float32)
    bad_y_train = np.array(["Bhairav"] * 15)  # 15 vs 20
    with pytest.raises(ValueError, match="Dimension mismatch: X_train has 20 samples, y_train has 15"):
        study.run_study(
            X_train=valid_x_train,
            y_train=bad_y_train,
            X_test=valid_x_test,
            y_test=y_test,
        )

    # MERT dimension mismatch
    bad_e_train = np.zeros((20, 512), dtype=np.float32)  # 512 instead of 1536
    valid_e_test = np.zeros((10, 1536), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected MERT embedding dimension 1536, got 512"):
        study.run_study(
            X_train=valid_x_train,
            y_train=y_train,
            X_test=valid_x_test,
            y_test=y_test,
            E_train=bad_e_train,
            E_test=valid_e_test,
        )
