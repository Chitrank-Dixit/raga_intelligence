"""
Comprehensive evaluation metrics and calibration assessment for raga classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@dataclass
class EvaluationMetrics:
    """Consolidated performance evaluation metrics."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]
    confusion_matrix: List[List[int]]
    class_names: List[str]
    expected_calibration_error: Optional[float] = None


def compute_classification_metrics(
    y_true: List[str] | np.ndarray,
    y_pred: List[str] | np.ndarray,
    class_names: List[str],
    y_probs: Optional[np.ndarray] = None,
) -> EvaluationMetrics:
    """Computes comprehensive classification metrics and confusion matrix."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=class_names,
        zero_division=0,
    )

    macro_f1 = float(np.mean(f1))
    total_samples = np.sum(support)
    weighted_f1 = float(np.sum(f1 * support) / total_samples) if total_samples > 0 else 0.0

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=class_names)

    ece: Optional[float] = None
    if y_probs is not None:
        ece = compute_calibration_error(y_true_arr, y_probs, class_names)

    return EvaluationMetrics(
        accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class_precision={c: round(float(p), 4) for c, p in zip(class_names, precision)},
        per_class_recall={c: round(float(r), 4) for c, r in zip(class_names, recall)},
        per_class_f1={c: round(float(f), 4) for c, f in zip(class_names, f1)},
        per_class_support={c: int(s) for c, s in zip(class_names, support)},
        confusion_matrix=cm.tolist(),
        class_names=class_names,
        expected_calibration_error=round(ece, 4) if ece is not None else None,
    )


def compute_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-class predictions.
    
    ECE partitions predictions into n_bins based on confidence (max softmax probability)
    and computes the weighted difference between average confidence and true accuracy.
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    true_indices = np.array([class_to_idx.get(name, -1) for name in y_true])

    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == true_indices).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        in_bin = (confidences >= bins[i]) & (confidences < bins[i + 1])
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)

