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
        if isinstance(y, list) or (isinstance(y, np.ndarray) and isinstance(y[0], str)):
            y_indices = np.array([self.class_to_idx[name] for name in y])
        else:
            y_indices = np.asarray(y)

        self.pipeline.fit(X, y_indices)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes class probabilities for feature matrix X."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba.")
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> List[str]:
        """Predicts raga names for feature matrix X."""
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

