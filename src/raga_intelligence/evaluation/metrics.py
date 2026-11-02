"""
Comprehensive evaluation metrics, calibration assessment, and grouped bootstrap confidence intervals
for raga classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@dataclass
class EvaluationMetrics:
    """Consolidated performance evaluation metrics including calibration and statistical uncertainty."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]
    confusion_matrix: List[List[int]]
    class_names: List[str]
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    weighted_precision: float = 0.0
    weighted_recall: float = 0.0
    multiclass_log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    expected_calibration_error: Optional[float] = None
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = field(default=None)


def compute_multiclass_brier_score(
    y_true: Sequence[str] | np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
) -> float:
    """
    Computes the multi-class Brier score:
    BS = (1/N) * sum_{i=1}^N sum_{c=1}^K (p_{ic} - y_{ic})^2
    
    Lower is better (0.0 is perfect calibration and sharpness).
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    n_samples, n_classes = y_probs.shape
    if n_samples == 0:
        return 0.0

    y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i, label in enumerate(y_true):
        idx = class_to_idx.get(label, -1)
        if 0 <= idx < n_classes:
            y_one_hot[i, idx] = 1.0

    brier = float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))
    return round(brier, 4)


def compute_multiclass_log_loss(
    y_true: Sequence[str] | np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    eps: float = 1e-15,
) -> float:
    """
    Computes multi-class cross-entropy log loss with numerical clipping:
    Loss = - (1/N) * sum_{i=1}^N sum_{c=1}^K y_{ic} * log(p_{ic})
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    n_samples, n_classes = y_probs.shape
    if n_samples == 0:
        return 0.0

    # Clip probabilities to prevent log(0)
    clipped_probs = np.clip(y_probs, eps, 1.0 - eps)
    # Renormalize to ensure sum to 1
    clipped_probs = clipped_probs / np.sum(clipped_probs, axis=1, keepdims=True)

    y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i, label in enumerate(y_true):
        idx = class_to_idx.get(label, -1)
        if 0 <= idx < n_classes:
            y_one_hot[i, idx] = 1.0

    log_loss = -float(np.mean(np.sum(y_one_hot * np.log(clipped_probs), axis=1)))
    return round(log_loss, 4)


def compute_calibration_error(
    y_true: Sequence[str] | np.ndarray,
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


def compute_grouped_bootstrap_confidence_intervals(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    class_names: List[str],
    y_probs: Optional[np.ndarray] = None,
    groups: Optional[Sequence[str] | np.ndarray] = None,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes non-parametric cluster/grouped bootstrap confidence intervals.
    
    Resampling occurs at the cluster/group level (e.g. performance_id or recording_id)
    to respect the intra-cluster correlation structure of musical recordings and avoid
    treating non-independent chunks as i.i.d. observations.
    
    Returns empirical confidence bounds [alpha/2, 1 - alpha/2], bootstrap mean, and standard error.
    """
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_samples = len(y_true_arr)

    if n_samples == 0:
        return {}

    if groups is None:
        # Fall back to standard sample-level bootstrap if no grouping metadata exists
        group_keys = np.arange(n_samples)
        group_to_indices = {i: [i] for i in range(n_samples)}
        unique_groups = group_keys
    else:
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        group_to_indices = {g: np.where(groups_arr == g)[0] for g in unique_groups}

    n_groups = len(unique_groups)
    if n_groups == 0:
        return {}

    boot_metrics: Dict[str, List[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
    }
    if y_probs is not None:
        boot_metrics["brier_score"] = []
        boot_metrics["log_loss"] = []
        boot_metrics["ece"] = []

    for _ in range(n_bootstraps):
        # Draw clusters with replacement
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        # Assemble item indices from the sampled clusters
        sampled_idx_list = [idx for g in sampled_groups for idx in group_to_indices[g]]
        if not sampled_idx_list:
            continue
        sampled_indices = np.array(sampled_idx_list, dtype=np.int64)

        b_true = y_true_arr[sampled_indices]
        b_pred = y_pred_arr[sampled_indices]

        # Accuracy
        b_acc = float(accuracy_score(b_true, b_pred))
        boot_metrics["accuracy"].append(b_acc)

        # F1 scores
        _, _, b_f1, b_support = precision_recall_fscore_support(
            b_true, b_pred, labels=class_names, zero_division=0
        )
        b_macro_f1 = float(np.mean(b_f1))
        b_tot = np.sum(b_support)
        b_weighted_f1 = float(np.sum(b_f1 * b_support) / b_tot) if b_tot > 0 else 0.0

        boot_metrics["macro_f1"].append(b_macro_f1)
        boot_metrics["weighted_f1"].append(b_weighted_f1)

        if y_probs is not None:
            b_probs = y_probs[sampled_indices]
            boot_metrics["brier_score"].append(compute_multiclass_brier_score(b_true, b_probs, class_names))
            boot_metrics["log_loss"].append(compute_multiclass_log_loss(b_true, b_probs, class_names))
            boot_metrics["ece"].append(compute_calibration_error(b_true, b_probs, class_names))

    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0

    ci_results: Dict[str, Dict[str, float]] = {}
    for metric_name, vals in boot_metrics.items():
        if not vals:
            continue
        v_arr = np.asarray(vals)
        ci_lower = float(np.percentile(v_arr, lower_p))
        ci_upper = float(np.percentile(v_arr, upper_p))
        mean_val = float(np.mean(v_arr))
        std_val = float(np.std(v_arr))

        ci_results[metric_name] = {
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
        }

    return ci_results


def compute_classification_metrics(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    class_names: List[str],
    y_probs: Optional[np.ndarray] = None,
    groups: Optional[Sequence[str] | np.ndarray] = None,
    compute_ci: bool = False,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> EvaluationMetrics:
    """Computes comprehensive classification metrics, calibration measures, and confidence intervals."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=class_names,
        zero_division=0,
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    total_samples = np.sum(support)
    weighted_precision = float(np.sum(precision * support) / total_samples) if total_samples > 0 else 0.0
    weighted_recall = float(np.sum(recall * support) / total_samples) if total_samples > 0 else 0.0
    weighted_f1 = float(np.sum(f1 * support) / total_samples) if total_samples > 0 else 0.0

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=class_names)

    ece: Optional[float] = None
    brier: Optional[float] = None
    log_loss: Optional[float] = None

    if y_probs is not None:
        ece = compute_calibration_error(y_true_arr, y_probs, class_names)
        brier = compute_multiclass_brier_score(y_true_arr, y_probs, class_names)
        log_loss = compute_multiclass_log_loss(y_true_arr, y_probs, class_names)

    ci_dict: Optional[Dict[str, Dict[str, float]]] = None
    if compute_ci:
        ci_dict = compute_grouped_bootstrap_confidence_intervals(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            class_names=class_names,
            y_probs=y_probs,
            groups=groups,
            n_bootstraps=n_bootstraps,
            alpha=alpha,
            seed=seed,
        )

    return EvaluationMetrics(
        accuracy=round(acc, 4),
        macro_precision=round(macro_precision, 4),
        macro_recall=round(macro_recall, 4),
        macro_f1=round(macro_f1, 4),
        weighted_precision=round(weighted_precision, 4),
        weighted_recall=round(weighted_recall, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class_precision={c: round(float(p), 4) for c, p in zip(class_names, precision)},
        per_class_recall={c: round(float(r), 4) for c, r in zip(class_names, recall)},
        per_class_f1={c: round(float(f), 4) for c, f in zip(class_names, f1)},
        per_class_support={c: int(s) for c, s in zip(class_names, support)},
        confusion_matrix=cm.tolist(),
        class_names=class_names,
        multiclass_log_loss=log_loss,
        brier_score=brier,
        expected_calibration_error=round(ece, 4) if ece is not None else None,
        confidence_intervals=ci_dict,
    )
