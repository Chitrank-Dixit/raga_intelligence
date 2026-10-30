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
