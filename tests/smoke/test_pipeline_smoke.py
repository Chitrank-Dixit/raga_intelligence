"""
End-to-end smoke tests for complete pipeline validation without external downloads.
"""

from pathlib import Path
import pytest
import torch

from raga_intelligence.config import RAGA_LIST, settings
from raga_intelligence.datasets.adapters import SyntheticDatasetAdapter
from raga_intelligence.inference.pipeline import InferencePipeline
from raga_intelligence.training.trainer import train_baseline_pipeline, train_hybrid_pipeline


def test_end_to_end_pipeline_smoke(tmp_path: Path):
    # 1. Generate synthetic dataset
    adapter = SyntheticDatasetAdapter(
        output_dir=tmp_path / "smoke_data",
        recordings_per_raga=2,
        duration_sec=2.0,
    )
    manifest = adapter.build_manifest()
    assert len(manifest.items) == len(RAGA_LIST) * 2

    # 2. Train baseline model
    model_save_path = tmp_path / "baseline_model.joblib"
    clf, metrics = train_baseline_pipeline(
        manifest=manifest,
        classifier_type="svm",
        save_path=model_save_path,
    )
    assert clf.is_fitted
    assert metrics.macro_f1 >= 0.0
    assert len(metrics.class_names) == 10

    # 3. Train hybrid model
    hybrid_save_path = tmp_path / "hybrid_model.pt"
    hybrid_clf, hybrid_metrics = train_hybrid_pipeline(
        manifest=manifest,
        save_path=hybrid_save_path,
        epochs=3,
    )
    assert hybrid_metrics.macro_f1 >= 0.0

    # 4. Run inference pipeline
    pipe = InferencePipeline(baseline_classifier_path=model_save_path)
    sample_file = manifest.items[0].file_path
    result = pipe.analyze_audio(sample_file, top_k=3)

    assert result.duration_sec > 1.5
    assert len(result.ranked_matches) == 3
    assert result.abstention_triggered is False
    assert result.caveat_statement != ""


def test_cpu_tensor_smoke():
    dev = settings.get_torch_device()
    x = torch.ones((100, 100), device=dev)
    y = torch.matmul(x, x)
    assert y.shape == (100, 100)
    assert torch.isclose(y[0, 0], torch.tensor(100.0, device=dev))

