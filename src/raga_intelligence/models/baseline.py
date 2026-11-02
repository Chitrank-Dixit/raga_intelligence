"""
Baseline conventional machine learning classifiers for raga recognition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from raga_intelligence.config import RAGA_LIST

logger = logging.getLogger(__name__)


class BaselineRagaClassifier:
    """
    Conventional ML baseline for raga recognition trained on explicit musical features
    (tonic-relative pitch class distribution, aligned chroma, spectral moments).
    """

    def __init__(
        self,
        classifier_type: Literal["svm", "logistic_regression", "random_forest"] = "svm",
        class_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.classifier_type = classifier_type
        self.class_names = class_names or RAGA_LIST
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}
        self.random_state = random_state

        if classifier_type == "svm":
            estimator = SVC(
                kernel="rbf",
                C=2.0,
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )
        elif classifier_type == "logistic_regression":
            estimator = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )
        elif classifier_type == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ])
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray | List[str]) -> BaselineRagaClassifier:
        """Trains the pipeline on feature matrix X and target labels y."""
        X = np.asarray(X)
        if isinstance(y, list) or (isinstance(y, np.ndarray) and len(y) > 0 and isinstance(y[0], str)):
            y_indices = []
            for name in y:
                if name not in self.class_to_idx:
                    raise ValueError(f"Unknown raga label '{name}'. Allowed classes: {self.class_names}")
                y_indices.append(self.class_to_idx[name])
            y_indices = np.array(y_indices, dtype=np.int64)
        else:
            y_indices = np.asarray(y, dtype=np.int64)

        self.pipeline.fit(X, y_indices)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes class probabilities for feature matrix X mapped strictly to canonical class_names.

        Returns:
            np.ndarray of shape (N, len(self.class_names)) where column j corresponds
            to self.class_names[j]. For any class not present during training, its
            probability column is explicitly assigned 0.0.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        raw_probs = self.pipeline.predict_proba(X)
        n_samples = X.shape[0]
        n_classes = len(self.class_names)

        # Scikit-learn estimator stores fitted classes in .classes_
        estimator = self.pipeline.named_steps["classifier"]
        fitted_classes = estimator.classes_

        canonical_probs = np.zeros((n_samples, n_classes), dtype=np.float32)

        for col_idx, fitted_cls in enumerate(fitted_classes):
            if isinstance(fitted_cls, (int, np.integer)):
                global_idx = int(fitted_cls)
            elif isinstance(fitted_cls, str):
                global_idx = self.class_to_idx.get(fitted_cls, -1)
            else:
                global_idx = int(fitted_cls)

            if 0 <= global_idx < n_classes:
                canonical_probs[:, global_idx] = raw_probs[:, col_idx]

        return canonical_probs

    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predicts raga names for feature matrix X.
        Guarantees strict agreement with predict_proba() and canonical class_names.
        """
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return [self.idx_to_class[idx] for idx in pred_indices]

    def save(self, filepath: Path | str) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "pipeline": self.pipeline,
            "classifier_type": self.classifier_type,
            "class_names": self.class_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(checkpoint, p)
        logger.info(f"Saved baseline classifier to {p}")

    @classmethod
    def load(cls, filepath: Path | str) -> BaselineRagaClassifier:
        p = Path(filepath)
        checkpoint = joblib.load(p)
        instance = cls(
            classifier_type=checkpoint["classifier_type"],
            class_names=checkpoint["class_names"],
        )
        instance.pipeline = checkpoint["pipeline"]
        instance.is_fitted = checkpoint["is_fitted"]
        return instance

