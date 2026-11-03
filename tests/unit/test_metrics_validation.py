"""
Unit tests for evaluation metrics validation, strict grouped bootstrap requirements,
and probability matrix verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from raga_intelligence.evaluation.metrics import (
    EvaluationMetrics,
    compute_calibration_error,
    compute_classification_metrics,
    compute_grouped_bootstrap_confidence_intervals,
    compute_multiclass_brier_score,
    compute_multiclass_log_loss,
)


def test_grouped_bootstrap_strictly_requires_groups():
    """Grouped bootstrap must fail explicitly if groups is None; no silent fallback."""
    class_names = ["Bhairav", "Yaman"]
    y_true = ["Bhairav", "Bhairav", "Yaman", "Yaman"]
    y_pred = ["Bhairav", "Yaman", "Yaman", "Bhairav"]

    # compute_grouped_bootstrap_confidence_intervals
    with pytest.raises(ValueError, match="Group identifiers.*are strictly required"):
        compute_grouped_bootstrap_confidence_intervals(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            groups=None,  # type: ignore
        )

    # compute_classification_metrics with compute_ci=True and groups=None
    with pytest.raises(ValueError, match="Group identifiers.*are strictly required"):
        compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            compute_ci=True,
            groups=None,
        )


def test_grouped_bootstrap_rejects_single_group():
    """Single-group data cannot be resampled via cluster bootstrap; must fail explicitly."""
    class_names = ["Bhairav", "Yaman"]
    y_true = ["Bhairav", "Bhairav", "Yaman", "Yaman"]
    y_pred = ["Bhairav", "Yaman", "Yaman", "Bhairav"]
    groups = ["SinglePerf", "SinglePerf", "SinglePerf", "SinglePerf"]

    with pytest.raises(ValueError, match="requires at least 2 independent groups"):
        compute_grouped_bootstrap_confidence_intervals(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            groups=groups,
        )


def test_grouped_bootstrap_validates_dimensions():
    """Input length mismatches must raise ValueError."""
    class_names = ["Bhairav", "Yaman"]
    y_true = ["Bhairav", "Bhairav", "Yaman", "Yaman"]
    y_pred_short = ["Bhairav", "Yaman"]
    groups = ["g1", "g1", "g2", "g2"]

    with pytest.raises(ValueError, match="dimension mismatch"):
        compute_grouped_bootstrap_confidence_intervals(
            y_true=y_true,
            y_pred=y_pred_short,
            class_names=class_names,
            groups=groups,
        )

    groups_short = ["g1", "g2"]
    with pytest.raises(ValueError, match="dimension mismatch"):
        compute_grouped_bootstrap_confidence_intervals(
            y_true=y_true,
            y_pred=y_true,
            class_names=class_names,
            groups=groups_short,
        )


def test_probability_matrix_validation():
    """Validation of probability matrix shape, column count, finite values, and non-negativity."""
    class_names = ["Bhairav", "Yaman", "Kafi"]
    y_true = ["Bhairav", "Yaman"]

    # 1. Not 2D
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        compute_multiclass_brier_score(y_true, np.array([0.5, 0.5]), class_names)

    # 2. Sample count mismatch
    with pytest.raises(ValueError, match="sample count"):
        compute_multiclass_brier_score(y_true, np.ones((3, 3)), class_names)

    # 3. Class count mismatch (e.g. 2 columns instead of 3 class_names)
    with pytest.raises(ValueError, match="class dimension"):
        compute_multiclass_brier_score(y_true, np.ones((2, 2)), class_names)

    # 4. Non-finite values (NaN / Inf)
    nan_probs = np.array([[0.5, 0.5, np.nan], [0.3, 0.3, 0.4]])
    with pytest.raises(ValueError, match="non-finite values"):
        compute_multiclass_log_loss(y_true, nan_probs, class_names)

    inf_probs = np.array([[0.5, 0.5, np.inf], [0.3, 0.3, 0.4]])
    with pytest.raises(ValueError, match="non-finite values"):
        compute_calibration_error(y_true, inf_probs, class_names)

    # 5. Negative probabilities
    neg_probs = np.array([[-0.1, 0.6, 0.5], [0.2, 0.4, 0.4]])
    with pytest.raises(ValueError, match="negative"):
        compute_multiclass_brier_score(y_true, neg_probs, class_names)


def test_grouped_bootstrap_reports_n_groups():
    """CI result dictionaries and EvaluationMetrics must report n_groups."""
    class_names = ["Bhairav", "Yaman"]
    y_true = ["Bhairav", "Bhairav", "Yaman", "Yaman"]
    y_pred = ["Bhairav", "Bhairav", "Yaman", "Yaman"]
    groups = ["perf_1", "perf_1", "perf_2", "perf_3"]

    ci_results = compute_grouped_bootstrap_confidence_intervals(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        groups=groups,
        n_bootstraps=50,
        seed=42,
    )

    assert "accuracy" in ci_results
    assert ci_results["accuracy"]["n_groups"] == 3.0

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        groups=groups,
        compute_ci=True,
        n_bootstraps=50,
        seed=42,
    )
    assert metrics.n_groups == 3
