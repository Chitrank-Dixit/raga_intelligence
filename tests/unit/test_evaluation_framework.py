"""
Unit tests for Phase 2.1 Scientific Evaluation Framework:
- Expanded metrics (Log loss, Brier score, ECE)
- Grouped bootstrap confidence intervals
- Group-aware partitioning and performer isolation
- Error analysis and musicological hypotheses
- Feature ablation study
- Experiment tracking and reproducibility
"""

import json
from pathlib import Path

import numpy as np
import pytest

from raga_intelligence.config import RAGA_LIST
from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest
from raga_intelligence.datasets.partitioning import GroupAwarePartitioner
from raga_intelligence.evaluation.ablation import FeatureAblationStudy
from raga_intelligence.evaluation.benchmark import ModelBenchmarkRunner
from raga_intelligence.evaluation.error_analysis import (
    analyze_classification_errors,
    format_error_analysis_markdown,
    get_musical_hypothesis,
)
from raga_intelligence.evaluation.metrics import (
    compute_calibration_error,
    compute_classification_metrics,
    compute_grouped_bootstrap_confidence_intervals,
    compute_multiclass_brier_score,
    compute_multiclass_log_loss,
)
from raga_intelligence.evaluation.reporting import generate_markdown_report
from raga_intelligence.evaluation.tracker import ExperimentTracker, get_environment_metadata


def test_multiclass_brier_score():
    class_names = ["Bhairav", "Yaman", "Bhoopali"]
    y_true = ["Bhairav", "Yaman", "Bhoopali"]
    # Perfect probabilities
    probs_perfect = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    score_perfect = compute_multiclass_brier_score(y_true, probs_perfect, class_names)
    assert score_perfect == 0.0

    # Completely wrong probabilities
    probs_wrong = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    score_wrong = compute_multiclass_brier_score(y_true, probs_wrong, class_names)
    assert score_wrong == 2.0


def test_multiclass_log_loss():
    class_names = ["Bhairav", "Yaman"]
    y_true = ["Bhairav", "Yaman"]
    # Near perfect probabilities
    probs_good = np.array([
        [0.99, 0.01],
        [0.02, 0.98],
    ])
    loss_good = compute_multiclass_log_loss(y_true, probs_good, class_names)
    assert 0.0 < loss_good < 0.05

    # Uniform probabilities
    probs_uniform = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    loss_uniform = compute_multiclass_log_loss(y_true, probs_uniform, class_names)
    assert pytest.approx(loss_uniform, abs=0.01) == float(np.log(2.0))


def test_grouped_bootstrap_confidence_intervals():
    class_names = ["Bhairav", "Yaman"]
    # 6 samples belonging to 2 performances (3 samples each)
    y_true = np.array(["Bhairav", "Bhairav", "Bhairav", "Yaman", "Yaman", "Yaman"])
    y_pred = np.array(["Bhairav", "Bhairav", "Yaman", "Yaman", "Yaman", "Yaman"])
    groups = np.array(["perf_1", "perf_1", "perf_1", "perf_2", "perf_2", "perf_2"])

    ci = compute_grouped_bootstrap_confidence_intervals(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        groups=groups,
        n_bootstraps=100,
        alpha=0.05,
        seed=42,
    )

    assert "accuracy" in ci
    assert "macro_f1" in ci
    assert "weighted_f1" in ci
    assert 0.0 <= ci["accuracy"]["ci_lower"] <= ci["accuracy"]["ci_upper"] <= 1.0
    assert 0.0 <= ci["accuracy"]["mean"] <= 1.0
    assert ci["accuracy"]["std"] >= 0.0


def test_expanded_classification_metrics_and_reporting():
    class_names = ["Bhairav", "Yaman", "Kafi"]
    y_true = ["Bhairav", "Yaman", "Kafi", "Bhairav"]
    y_pred = ["Bhairav", "Yaman", "Bhairav", "Bhairav"]
    y_probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.6, 0.2, 0.2],
        [0.7, 0.2, 0.1],
    ])
    groups = ["perf_A", "perf_B", "perf_C", "perf_A"]

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        y_probs=y_probs,
        groups=groups,
        compute_ci=True,
        n_bootstraps=50,
        seed=42,
    )

    assert metrics.accuracy == 0.75
    assert metrics.macro_precision > 0.0
    assert metrics.macro_recall > 0.0
    assert metrics.multiclass_log_loss is not None
    assert metrics.brier_score is not None
    assert metrics.expected_calibration_error is not None
    assert metrics.confidence_intervals is not None
    assert "accuracy" in metrics.confidence_intervals

    report_md = generate_markdown_report(metrics, experiment_name="Test Experiment")
    assert "Statistical Confidence Intervals" in report_md
    assert "Multiclass Log Loss" in report_md
    assert "Brier Score" in report_md


def test_group_aware_partitioner():
    # Construct synthetic manifest with 3 ragas and multiple performances
    manifest = DatasetManifest()
    for raga in ["Bhairav", "Yaman", "Kafi"]:
        for perf_idx in range(4):
            perf_id = f"{raga}_perf_{perf_idx}"
            for chunk in range(3):
                item = DatasetItem(
                    recording_id=f"{perf_id}_chunk_{chunk}",
                    source="Test",
                    file_path=f"/dummy/{perf_id}_chunk_{chunk}.wav",
                    raga=raga,
                    performer=perf_id,
                    performance_id=perf_id,
                    duration_sec=10.0,
                    sample_rate=24000,
                    split="train",
                )
                manifest.add_item(item)

    # 1. Audit
    diag = GroupAwarePartitioner.audit_grouping(manifest, group_col="performance_id")
    assert diag.is_fully_partitionable is True
    assert diag.raga_group_counts["Bhairav"] == 4

    # 2. Split with zero leakage
    split_manifest, _ = GroupAwarePartitioner.create_group_aware_split(
        manifest=manifest,
        group_col="performance_id",
        test_ratio=0.25,
        seed=42,
    )
    val = split_manifest.validate_splits()
    assert val["valid_no_recording_leakage"] is True
    assert val["valid_no_performance_leakage"] is True

    # 3. Multi-seed consistency
    multi_splits = GroupAwarePartitioner.generate_multi_seed_splits(
        manifest=manifest,
        seeds=[42, 123, 999],
    )
    assert len(multi_splits) == 3
    for s, m in multi_splits.items():
        v = m.validate_splits()
        assert v["valid_no_performance_leakage"] is True


def test_group_aware_partitioner_reports_single_group_limitation():
    manifest = DatasetManifest()
    # Raga A has only 1 performance
    for chunk in range(3):
        manifest.add_item(DatasetItem(
            recording_id=f"solo_chunk_{chunk}",
            source="Test",
            file_path=f"/dummy/solo_chunk_{chunk}.wav",
            raga="Bhairav",
            performer="Performer_A",
            performance_id="Perf_1",
            duration_sec=10.0,
            sample_rate=24000,
        ))

    diag = GroupAwarePartitioner.audit_grouping(manifest, group_col="performance_id")
    assert diag.is_fully_partitionable is False
    assert "Bhairav" in diag.ragas_with_insufficient_groups
    assert len(diag.limitations) > 0


def test_error_analysis_and_musicological_hypotheses():
    class_names = ["Bhoopali", "Deshkar", "Bhairav", "Todi"]
    y_true = ["Bhoopali", "Bhoopali", "Bhairav", "Todi", "Deshkar"]
    y_pred = ["Deshkar", "Deshkar", "Todi", "Todi", "Deshkar"]
    voiced = [0.8, 0.4, 0.3, 0.9, 0.85]
    sf = [0.01, 0.05, 0.08, 0.02, 0.01]
    tonic_err = [10.0, 60.0, 75.0, 5.0, 12.0]

    report = analyze_classification_errors(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        voiced_ratios=voiced,
        spectral_flatness=sf,
        tonic_errors_cents=tonic_err,
    )

    assert report.total_samples == 5
    assert report.total_errors == 3
    assert len(report.top_confusion_pairs) >= 1
    # Check top confusion pair
    top_pair = report.top_confusion_pairs[0]
    assert top_pair.true_raga == "Bhoopali" and top_pair.pred_raga == "Deshkar"
    assert "Audav" in top_pair.musical_hypothesis

    # Check asymmetry
    assert len(report.directional_asymmetries) > 0

    # Check quality and tonic correlation
    assert report.quality_correlation is not None
    assert report.tonic_correlation is not None
    assert report.tonic_correlation["large_tonic_error_count"] == 2

    # Check formatting
    md = format_error_analysis_markdown(report)
    assert "Automated Error Analysis" in md
    assert "Top Frequently Confused Raga Pairs" in md
    assert "Hypothesis:" in md


def test_feature_ablation_study(tmp_path: Path):
    rng = np.random.default_rng(42)
    n_train = 40
    n_test = 10
    n_classes = len(RAGA_LIST)

    # 35-dim feature array
    X_train = rng.standard_normal((n_train, 35)).astype(np.float32)
    X_test = rng.standard_normal((n_test, 35)).astype(np.float32)
    # MERT 1536-dim embeddings
    E_train = rng.standard_normal((n_train, 1536)).astype(np.float32)
    E_test = rng.standard_normal((n_test, 1536)).astype(np.float32)

    y_train = np.array([RAGA_LIST[i % n_classes] for i in range(n_train)])
    y_test = np.array([RAGA_LIST[i % n_classes] for i in range(n_test)])

    study = FeatureAblationStudy(classifier_type="logistic_regression", random_seed=42)
    json_path = tmp_path / "ablation_test.json"

    results = study.run_study(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        E_train=E_train,
        E_test=E_test,
        output_json_path=json_path,
    )

    assert "all_handcrafted_35" in results
    assert "pcd" in results
    assert "chroma" in results
    assert "spectral" in results
    assert "voiced" in results
    assert "mert_embeddings" in results
    assert "hybrid_all" in results
    assert json_path.exists()

    md_table = study.format_ablation_markdown_table(results)
    assert "| Feature Group |" in md_table
    assert "mert_embeddings" in md_table


def test_experiment_tracker(tmp_path: Path):
    tracker = ExperimentTracker(experiment_dir=tmp_path / "experiments")
    assert len(tracker.list_runs()) == 0

    record = tracker.log_run(
        experiment_name="Unit Test Experiment",
        seed=42,
        model_name="Baseline SVM",
        model_architecture="SVC(kernel=rbf)",
        hyperparameters={"C": 10.0},
        feature_config={"dims": 35},
        metrics={"accuracy": 0.85, "macro_f1": 0.84, "multiclass_log_loss": 0.45},
        notes="Testing JSON tracker",
    )

    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == record.run_id
    assert runs[0].metrics["accuracy"] == 0.85

    fetched = tracker.get_run(record.run_id)
    assert fetched is not None
    assert fetched.model_name == "Baseline SVM"

    summary_md = tracker.to_markdown_summary()
    assert "Baseline SVM" in summary_md
    assert "0.8500" in summary_md


def test_environment_metadata():
    meta = get_environment_metadata()
    assert "python_version" in meta
    assert "torch_version" in meta
    assert "librosa_version" in meta
    assert "sklearn_version" in meta
    assert "transformers_version" in meta
    assert "os" in meta

