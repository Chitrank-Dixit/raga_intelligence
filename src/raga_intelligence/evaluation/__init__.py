"""
Evaluation metrics, uncertainty calibration, error analysis, ablation studies, and reporting utilities.
"""

from raga_intelligence.evaluation.ablation import FeatureAblationStudy, AblationResult
from raga_intelligence.evaluation.benchmark import ModelBenchmarkRunner, AggregatedModelStats
from raga_intelligence.evaluation.error_analysis import (
    ErrorAnalysisReport,
    analyze_classification_errors,
    format_error_analysis_markdown,
)
from raga_intelligence.evaluation.metrics import (
    EvaluationMetrics,
    compute_calibration_error,
    compute_classification_metrics,
    compute_grouped_bootstrap_confidence_intervals,
    compute_multiclass_brier_score,
    compute_multiclass_log_loss,
)
from raga_intelligence.evaluation.reporting import generate_markdown_report
from raga_intelligence.evaluation.tracker import ExperimentRecord, ExperimentTracker

__all__ = [
    "EvaluationMetrics",
    "compute_classification_metrics",
    "compute_calibration_error",
    "compute_multiclass_brier_score",
    "compute_multiclass_log_loss",
    "compute_grouped_bootstrap_confidence_intervals",
    "generate_markdown_report",
    "FeatureAblationStudy",
    "AblationResult",
    "ModelBenchmarkRunner",
    "AggregatedModelStats",
    "ErrorAnalysisReport",
    "analyze_classification_errors",
    "format_error_analysis_markdown",
    "ExperimentRecord",
    "ExperimentTracker",
]
