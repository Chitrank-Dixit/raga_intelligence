"""
Feature ablation framework for evaluating individual and combined feature group contributions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from raga_intelligence.config import RAGA_LIST
from raga_intelligence.evaluation.metrics import (
    compute_multiclass_brier_score,
    compute_multiclass_log_loss,
)
from raga_intelligence.features.schema import FeatureBlock, FeatureSchema

logger = logging.getLogger(__name__)

# Canonical feature group evaluation order derived from FeatureSchema
CANONICAL_ABLATION_GROUPS: List[str] = [
    "pcd",
    "chroma",
    "pitch_combined",
    "spectral",
    "voiced",
    "spectral_and_voiced",
    "all_handcrafted_35",
    "melodic",
]


@dataclass
class AblationResult:
    """Performance metrics and relative impact of a specific feature configuration."""
    group_name: str
    feature_dimension: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    multiclass_log_loss: Optional[float]
    brier_score: Optional[float]
    delta_macro_f1: float  # Difference relative to all_handcrafted_35
    delta_accuracy: float


class FeatureAblationStudy:
    """
    Evaluates individual and combined feature groups on strictly identical data splits
    using a shared FeatureSchema and explicit array validation.
    """

    def __init__(
        self,
        class_names: List[str] = RAGA_LIST,
        classifier_type: str = "svm",
        random_seed: int = 42,
    ):
        self.class_names = class_names
        self.classifier_type = classifier_type
        self.random_seed = random_seed

    def _build_classifier(self) -> Any:
        if self.classifier_type == "logistic_regression":
            clf = LogisticRegression(max_iter=1000, random_state=self.random_seed)
        else:
            clf = SVC(kernel="rbf", C=10.0, probability=True, random_state=self.random_seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def run_study(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        E_train: Optional[np.ndarray] = None,
        E_test: Optional[np.ndarray] = None,
        output_json_path: Optional[Path | str] = None,
        recording_ids_train: Optional[Sequence[str]] = None,
        recording_ids_test: Optional[Sequence[str]] = None,
        dataset_fingerprint: Optional[str] = None,
    ) -> Dict[str, AblationResult]:
        """
        Runs comprehensive ablation experiments across all feature groups and saves results to JSON.
        Validates feature schema dimensions and ensures strictly identical recording splits.
        """
        # 1. Validate feature array dimensions using FeatureSchema
        FeatureSchema.validate_array(X_train)
        FeatureSchema.validate_array(X_test)

        n_train = len(X_train)
        n_test = len(X_test)

        if len(y_train) != n_train:
            raise ValueError(f"Dimension mismatch: X_train has {n_train} samples, y_train has {len(y_train)}")
        if len(y_test) != n_test:
            raise ValueError(f"Dimension mismatch: X_test has {n_test} samples, y_test has {len(y_test)}")

        if recording_ids_train is not None and len(recording_ids_train) != n_train:
            raise ValueError(
                f"Dimension mismatch: recording_ids_train has {len(recording_ids_train)} IDs, expected {n_train}"
            )
        if recording_ids_test is not None and len(recording_ids_test) != n_test:
            raise ValueError(
                f"Dimension mismatch: recording_ids_test has {len(recording_ids_test)} IDs, expected {n_test}"
            )

        if E_train is not None and E_test is not None:
            FeatureSchema.validate_array(E_train, expected_feature_type="mert_1536")
            FeatureSchema.validate_array(E_test, expected_feature_type="mert_1536")
            if len(E_train) != n_train:
                raise ValueError(f"Dimension mismatch: E_train has {len(E_train)} samples, expected {n_train}")
            if len(E_test) != n_test:
                raise ValueError(f"Dimension mismatch: E_test has {len(E_test)} samples, expected {n_test}")

        results: Dict[str, AblationResult] = {}

        # 2. Establish reference performance using all_handcrafted_35
        ref_block = FeatureSchema.get_block("all_handcrafted_35")
        ref_x_train = X_train[:, ref_block.slice]
        ref_x_test = X_test[:, ref_block.slice]

        ref_pipe = self._build_classifier()
        ref_pipe.fit(ref_x_train, y_train)
        ref_preds = ref_pipe.predict(ref_x_test)
        ref_probs = ref_pipe.predict_proba(ref_x_test)

        ref_acc = float(accuracy_score(y_test, ref_preds))
        _, _, ref_f1, ref_supp = precision_recall_fscore_support(
            y_test, ref_preds, labels=self.class_names, zero_division=0
        )
        ref_macro_f1 = float(np.mean(ref_f1))
        ref_tot = np.sum(ref_supp)
        ref_weighted_f1 = float(np.sum(ref_f1 * ref_supp) / ref_tot) if ref_tot > 0 else 0.0

        ref_log_loss = compute_multiclass_log_loss(y_test, ref_probs, self.class_names)
        ref_brier = compute_multiclass_brier_score(y_test, ref_probs, self.class_names)

        results["all_handcrafted_35"] = AblationResult(
            group_name="all_handcrafted_35",
            feature_dimension=ref_x_train.shape[1],
            accuracy=round(ref_acc, 4),
            macro_f1=round(ref_macro_f1, 4),
            weighted_f1=round(ref_weighted_f1, 4),
            multiclass_log_loss=ref_log_loss,
            brier_score=ref_brier,
            delta_macro_f1=0.0,
            delta_accuracy=0.0,
        )

        # 3. Evaluate each handcrafted feature block in FeatureSchema
        for group_name in CANONICAL_ABLATION_GROUPS:
            if group_name == "all_handcrafted_35":
                continue
            if not FeatureSchema.has_block(group_name):
                continue

            block = FeatureSchema.get_block(group_name)
            if block.end_idx > X_train.shape[1]:
                # Skip if feature array does not contain this block (e.g. melodic)
                continue

            sub_x_train = X_train[:, block.slice]
            sub_x_test = X_test[:, block.slice]

            pipe = self._build_classifier()
            pipe.fit(sub_x_train, y_train)
            preds = pipe.predict(sub_x_test)
            probs = pipe.predict_proba(sub_x_test)

            acc = float(accuracy_score(y_test, preds))
            _, _, f1_arr, supp = precision_recall_fscore_support(
                y_test, preds, labels=self.class_names, zero_division=0
            )
            macro_f1 = float(np.mean(f1_arr))
            tot = np.sum(supp)
            weighted_f1 = float(np.sum(f1_arr * supp) / tot) if tot > 0 else 0.0

            log_loss = compute_multiclass_log_loss(y_test, probs, self.class_names)
            brier = compute_multiclass_brier_score(y_test, probs, self.class_names)

            results[group_name] = AblationResult(
                group_name=group_name,
                feature_dimension=sub_x_train.shape[1],
                accuracy=round(acc, 4),
                macro_f1=round(macro_f1, 4),
                weighted_f1=round(weighted_f1, 4),
                multiclass_log_loss=log_loss,
                brier_score=brier,
                delta_macro_f1=round(macro_f1 - ref_macro_f1, 4),
                delta_accuracy=round(acc - ref_acc, 4),
            )

        # 4. Evaluate MERT embeddings alone if provided
        if E_train is not None and E_test is not None:
            pipe_mert = self._build_classifier()
            pipe_mert.fit(E_train, y_train)
            mert_preds = pipe_mert.predict(E_test)
            mert_probs = pipe_mert.predict_proba(E_test)

            m_acc = float(accuracy_score(y_test, mert_preds))
            _, _, m_f1, m_supp = precision_recall_fscore_support(
                y_test, mert_preds, labels=self.class_names, zero_division=0
            )
            m_macro_f1 = float(np.mean(m_f1))
            m_tot = np.sum(m_supp)
            m_weighted_f1 = float(np.sum(m_f1 * m_supp) / m_tot) if m_tot > 0 else 0.0

            results["mert_embeddings"] = AblationResult(
                group_name="mert_embeddings",
                feature_dimension=E_train.shape[1],
                accuracy=round(m_acc, 4),
                macro_f1=round(m_macro_f1, 4),
                weighted_f1=round(m_weighted_f1, 4),
                multiclass_log_loss=compute_multiclass_log_loss(y_test, mert_probs, self.class_names),
                brier_score=compute_multiclass_brier_score(y_test, mert_probs, self.class_names),
                delta_macro_f1=round(m_macro_f1 - ref_macro_f1, 4),
                delta_accuracy=round(m_acc - ref_acc, 4),
            )

            # 5. Evaluate Hybrid combination (Handcrafted 35-dim + MERT 1536-dim)
            H_train = np.hstack([ref_x_train, E_train])
            H_test = np.hstack([ref_x_test, E_test])

            pipe_hybrid = self._build_classifier()
            pipe_hybrid.fit(H_train, y_train)
            hyb_preds = pipe_hybrid.predict(H_test)
            hyb_probs = pipe_hybrid.predict_proba(H_test)

            h_acc = float(accuracy_score(y_test, hyb_preds))
            _, _, h_f1, h_supp = precision_recall_fscore_support(
                y_test, hyb_preds, labels=self.class_names, zero_division=0
            )
            h_macro_f1 = float(np.mean(h_f1))
            h_tot = np.sum(h_supp)
            h_weighted_f1 = float(np.sum(h_f1 * h_supp) / h_tot) if h_tot > 0 else 0.0

            results["hybrid_all"] = AblationResult(
                group_name="hybrid_all",
                feature_dimension=H_train.shape[1],
                accuracy=round(h_acc, 4),
                macro_f1=round(h_macro_f1, 4),
                weighted_f1=round(h_weighted_f1, 4),
                multiclass_log_loss=compute_multiclass_log_loss(y_test, hyb_probs, self.class_names),
                brier_score=compute_multiclass_brier_score(y_test, hyb_probs, self.class_names),
                delta_macro_f1=round(h_macro_f1 - ref_macro_f1, 4),
                delta_accuracy=round(h_acc - ref_acc, 4),
            )

        if output_json_path:
            p = Path(output_json_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {
                "dataset_fingerprint": dataset_fingerprint,
                "feature_schema_version": FeatureSchema.SCHEMA_VERSION,
                "split_identity": {
                    "n_train": n_train,
                    "n_test": n_test,
                    "recording_ids_train": list(recording_ids_train) if recording_ids_train is not None else None,
                    "recording_ids_test": list(recording_ids_test) if recording_ids_test is not None else None,
                },
                "results": {k: asdict(v) for k, v in results.items()},
            }
            # Top-level results preservation for backward-compatibility
            for k, v in results.items():
                payload[k] = asdict(v)

            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        return results

    @staticmethod
    def format_ablation_markdown_table(results: Dict[str, AblationResult]) -> str:
        """Formats ablation results into a clean markdown comparison table."""
        lines = [
            "| Feature Group | Dims | Accuracy | Macro-F1 | Weighted-F1 | Delta Macro-F1 | Log Loss | Brier Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for name, r in results.items():
            delta_str = f"{r.delta_macro_f1:+.4f}" if r.delta_macro_f1 != 0.0 else "0.0000 (ref)"
            ll_str = f"{r.multiclass_log_loss:.4f}" if r.multiclass_log_loss is not None else "N/A"
            br_str = f"{r.brier_score:.4f}" if r.brier_score is not None else "N/A"
            lines.append(
                f"| **{name}** | {r.feature_dimension} | `{r.accuracy:.4f}` | `{r.macro_f1:.4f}` | "
                f"`{r.weighted_f1:.4f}` | `{delta_str}` | `{ll_str}` | `{br_str}` |"
            )
        return "\n".join(lines)
