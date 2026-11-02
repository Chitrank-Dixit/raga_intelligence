"""
Unit tests for baseline classifiers, neural heads, and similarity ranking.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from raga_intelligence.config import RAGA_LIST
from raga_intelligence.models.baseline import BaselineRagaClassifier
from raga_intelligence.models.classifier import MERTClassifierHead
from raga_intelligence.models.hybrid import HybridRagaClassifier
from raga_intelligence.similarity.engine import MelodicSimilarityEngine
from raga_intelligence.similarity.prototypes import RagaPrototypeRegistry


def test_baseline_classifier_fit_predict(tmp_path: Path):
    n_samples = 40
    n_features = 35
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = [RAGA_LIST[i % len(RAGA_LIST)] for i in range(n_samples)]

    clf = BaselineRagaClassifier(classifier_type="svm", class_names=RAGA_LIST)
    clf.fit(X, y)
    assert clf.is_fitted

    preds = clf.predict(X[:5])
    assert len(preds) == 5
    for p in preds:
        assert p in RAGA_LIST

    probs = clf.predict_proba(X[:5])
    assert probs.shape == (5, len(RAGA_LIST))
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # Save and reload
    ckpt_path = tmp_path / "baseline.joblib"
    clf.save(ckpt_path)
    loaded_clf = BaselineRagaClassifier.load(ckpt_path)
    assert loaded_clf.is_fitted
    loaded_preds = loaded_clf.predict(X[:5])
    assert loaded_preds == preds


def test_mert_classifier_head(tmp_path: Path):
    input_dim = 1536
    batch_size = 4
    num_classes = len(RAGA_LIST)

    model = MERTClassifierHead(input_dim=input_dim, num_classes=num_classes)
    model.eval()
    x = torch.randn(batch_size, input_dim)

    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch_size, num_classes)

    probs = model.predict_proba(x)
    assert probs.shape == (batch_size, num_classes)
    assert torch.allclose(torch.sum(probs, dim=-1), torch.ones(batch_size))

    ckpt_path = tmp_path / "mert_head.pt"
    model.save_checkpoint(ckpt_path)
    loaded = MERTClassifierHead.load_checkpoint(ckpt_path)
    loaded.eval()
    with torch.no_grad():
        loaded_logits = loaded(x)
    assert torch.allclose(logits, loaded_logits)


def test_hybrid_classifier():
    musical_dim = 35
    mert_dim = 1536
    batch_size = 4

    model = HybridRagaClassifier(musical_dim=musical_dim, mert_dim=mert_dim)
    m = torch.randn(batch_size, musical_dim)
    e = torch.randn(batch_size, mert_dim)

    logits = model(m, e)
    assert logits.shape == (batch_size, len(RAGA_LIST))

    probs = model.predict_proba(m, e)
    assert probs.shape == (batch_size, len(RAGA_LIST))


def test_melodic_similarity_prototypes():
    registry = RagaPrototypeRegistry()
    bhoopali = registry.get("Bhoopali")
    assert bhoopali is not None
    assert "G" in bhoopali.allowed_swaras
    assert "D" in bhoopali.allowed_swaras
    # Bhoopali is audav (5 notes): S R G P D (omits M and N)
    assert "M" not in bhoopali.allowed_swaras
    assert "N" not in bhoopali.allowed_swaras
    # Verify empirical PCD presence and normalization
    assert bhoopali.empirical_pcd is not None
    assert np.isclose(np.sum(bhoopali.effective_pcd), 1.0)


def test_baseline_classifier_missing_classes_mapping():
    """
    Prompt 10: Verify probability column mapping when some classes are absent during training.
    """
    rng = np.random.default_rng(42)
    # Train with only 3 classes out of 10: Yaman (idx 2), Bhairav (idx 0), Darbari Kanada (idx 8)
    active_ragas = ["Bhairav", "Yaman", "Darbari Kanada"]
    n_samples = 30
    X = rng.standard_normal((n_samples, 35)).astype(np.float32)
    y = [active_ragas[i % len(active_ragas)] for i in range(n_samples)]

    clf = BaselineRagaClassifier(classifier_type="logistic_regression", class_names=RAGA_LIST)
    clf.fit(X, y)

    # Underlying estimator only saw 3 classes
    assert len(clf.pipeline.named_steps["classifier"].classes_) == 3

    # predict_proba should return full 10-dimensional probabilities in RAGA_LIST order
    probs = clf.predict_proba(X[:5])
    assert probs.shape == (5, len(RAGA_LIST))

    # All inactive ragas must have probability 0.0
    for idx, raga_name in enumerate(RAGA_LIST):
        if raga_name not in active_ragas:
            assert np.all(probs[:, idx] == 0.0), f"Raga {raga_name} should have 0.0 probability"

    # Active ragas should sum to 1.0
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # predict() must return only active ragas and strictly match argmax of predict_proba()
    preds = clf.predict(X[:5])
    expected_preds = [RAGA_LIST[idx] for idx in np.argmax(probs, axis=1)]
    assert preds == expected_preds
    for p in preds:
        assert p in active_ragas


def test_baseline_classifier_non_contiguous_integer_labels():
    """
    Prompt 10: Verify training with non-contiguous integer class labels.
    """
    rng = np.random.default_rng(42)
    # Non-contiguous integer indices: 1, 4, 9
    y_indices = np.array([1, 4, 9, 1, 4, 9, 1, 4, 9], dtype=np.int64)
    X = rng.standard_normal((len(y_indices), 35)).astype(np.float32)

    clf = BaselineRagaClassifier(classifier_type="logistic_regression", class_names=RAGA_LIST)
    clf.fit(X, y_indices)

    probs = clf.predict_proba(X)
    assert probs.shape == (len(y_indices), len(RAGA_LIST))

    # Columns 1, 4, 9 should contain probabilities; others should be 0.0
    present_cols = {1, 4, 9}
    for col in range(len(RAGA_LIST)):
        if col not in present_cols:
            assert np.all(probs[:, col] == 0.0)

    preds = clf.predict(X)
    for p in preds:
        assert p in [RAGA_LIST[1], RAGA_LIST[4], RAGA_LIST[9]]


def test_baseline_classifier_rejects_unknown_raga():
    X = np.ones((2, 35))
    y = ["UnknownRaga123", "Yaman"]
    clf = BaselineRagaClassifier(classifier_type="svm", class_names=RAGA_LIST)
    with pytest.raises(ValueError, match="Unknown raga label"):
        clf.fit(X, y)

