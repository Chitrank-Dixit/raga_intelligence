"""
Report generation, markdown summary formatting, and evaluation presentation.
"""

from __future__ import annotations

from typing import Optional

from raga_intelligence.evaluation.metrics import EvaluationMetrics


def generate_markdown_report(
    metrics: EvaluationMetrics,
    experiment_name: str,
    dataset_name: str = "Saraga Hindustani / Authentic Evaluation",
    notes: Optional[str] = None,
) -> str:
    """Formats EvaluationMetrics into an evidence-driven GitHub markdown report."""
    lines = [
        f"# Evaluation Report: {experiment_name}",
        f"**Dataset**: {dataset_name}",
        "",
        "## Overall Summary",
        f"- **Macro-F1 Score**: `{metrics.macro_f1:.4f}`",
        f"- **Weighted-F1 Score**: `{metrics.weighted_f1:.4f}`",
        f"- **Macro Precision**: `{metrics.macro_precision:.4f}`",
        f"- **Macro Recall**: `{metrics.macro_recall:.4f}`",
        f"- **Accuracy (Supplementary)**: `{metrics.accuracy:.4f}`",
    ]

    if metrics.multiclass_log_loss is not None:
        lines.append(f"- **Multiclass Log Loss**: `{metrics.multiclass_log_loss:.4f}`")

    if metrics.brier_score is not None:
        lines.append(f"- **Brier Score**: `{metrics.brier_score:.4f}`")

    if metrics.expected_calibration_error is not None:
        lines.append(f"- **Expected Calibration Error (ECE)**: `{metrics.expected_calibration_error:.4f}`")

    if metrics.n_groups is not None:
        lines.append(f"- **Independent Resampling Groups**: `{metrics.n_groups}`")

    if metrics.confidence_intervals:
        lines.extend([
            "",
            "### Statistical Confidence Intervals (95% Grouped Bootstrap)",
            "| Metric | Bootstrap Mean | Std Err | 95% CI Lower | 95% CI Upper |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for metric_key, ci_stats in metrics.confidence_intervals.items():
            lines.append(
                f"| **{metric_key}** | `{ci_stats.get('mean', 0.0):.4f}` | "
                f"`{ci_stats.get('std', 0.0):.4f}` | "
                f"`{ci_stats.get('ci_lower', 0.0):.4f}` | "
                f"`{ci_stats.get('ci_upper', 0.0):.4f}` |"
            )

    lines.extend([
        "",
        "## Per-Raga Detailed Metrics",
        "| Raga | Precision | Recall | F1-Score | Support |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for raga in metrics.class_names:
        p = metrics.per_class_precision.get(raga, 0.0)
        r = metrics.per_class_recall.get(raga, 0.0)
        f = metrics.per_class_f1.get(raga, 0.0)
        s = metrics.per_class_support.get(raga, 0)
        lines.append(f"| **{raga}** | {p:.4f} | {r:.4f} | {f:.4f} | {s} |")

    lines.extend([
        "",
        "## Confusion Matrix",
        "Rows represent Ground Truth, Columns represent Model Predictions.",
        "",
    ])

    # Header
    col_headers = [c[:6] for c in metrics.class_names]
    lines.append("| True \\ Pred | " + " | ".join(col_headers) + " |")
    lines.append("| :--- | " + " | ".join([":---:" for _ in col_headers]) + " |")

    for i, raga in enumerate(metrics.class_names):
        row_vals = [str(metrics.confusion_matrix[i][j]) for j in range(len(metrics.class_names))]
        lines.append(f"| **{raga}** | " + " | ".join(row_vals) + " |")

    if notes:
        lines.extend([
            "",
            "## Experimental Observations & Limitations",
            notes,
        ])

    return "\n".join(lines)
