"""
Consolidated multi-model and multi-seed benchmarking framework.
Evaluates Handcrafted Baselines (SVM, RF, LR), Frozen MERT, and Hybrid architectures
on strictly identical held-out test partitions.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from raga_intelligence.config import RAGA_LIST, settings
from raga_intelligence.evaluation.metrics import (
    EvaluationMetrics,
    compute_classification_metrics,
)
from raga_intelligence.models.classifier import MERTClassifierHead
from raga_intelligence.models.hybrid import HybridRagaClassifier

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationResult:
    """Individual seed evaluation results including test predictions and probabilities."""
    seed: int
    metrics: EvaluationMetrics
    y_pred: List[str]
    y_probs: np.ndarray

    @property
    def accuracy(self) -> float:
        return self.metrics.accuracy

    @property
    def macro_f1(self) -> float:
        return self.metrics.macro_f1

    @property
    def weighted_f1(self) -> float:
        return self.metrics.weighted_f1

    @property
    def macro_precision(self) -> float:
        return self.metrics.macro_precision

    @property
    def macro_recall(self) -> float:
        return self.metrics.macro_recall

    @property
    def multiclass_log_loss(self) -> Optional[float]:
        return self.metrics.multiclass_log_loss

    @property
    def brier_score(self) -> Optional[float]:
        return self.metrics.brier_score

    @property
    def expected_calibration_error(self) -> Optional[float]:
        return self.metrics.expected_calibration_error

    @property
    def confidence_intervals(self) -> Optional[Dict[str, Dict[str, float]]]:
        return self.metrics.confidence_intervals

    @property
    def n_groups(self) -> Optional[int]:
        return self.metrics.n_groups


@dataclass
class AggregatedModelStats:
    """Consolidated performance statistics across multiple random seeds."""
    model_name: str
    num_seeds: int
    seeds: List[int]
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    weighted_f1_mean: float
    weighted_f1_std: float
    macro_precision_mean: float
    macro_recall_mean: float
    log_loss_mean: Optional[float]
    brier_score_mean: Optional[float]
    ece_mean: Optional[float]
    representative_metrics: EvaluationMetrics  # Metrics from primary seed (seeds[0])
    primary_seed: int = 42


class ModelBenchmarkRunner:
    """
    Executes controlled, reproducible model comparisons across multiple random seeds.
    Returns explicit test predictions and probabilities to eliminate redundant retraining.
    """

    def __init__(
        self,
        class_names: List[str] = RAGA_LIST,
        device: Optional[torch.device] = None,
    ):
        self.class_names = class_names
        self.device = device or settings.get_torch_device()
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

    def evaluate_handcrafted_baseline(
        self,
        classifier_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_groups: Optional[Sequence[str]] = None,
        seeds: Sequence[int] = (42, 123, 999),
    ) -> Tuple[AggregatedModelStats, Dict[int, ModelEvaluationResult]]:
        """Evaluates a handcrafted baseline (SVM, RF, LR) across multiple random seeds."""
        if not seeds:
            raise ValueError("At least one random seed must be provided for evaluation.")

        primary_seed = seeds[0]
        seed_results: Dict[int, ModelEvaluationResult] = {}

        for seed in seeds:
            if classifier_type == "svm":
                clf = SVC(kernel="rbf", C=10.0, probability=True, random_state=seed)
            elif classifier_type == "random_forest":
                clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed)
            elif classifier_type == "logistic_regression":
                clf = LogisticRegression(max_iter=1000, random_state=seed)
            else:
                raise ValueError(f"Unknown classifier type: {classifier_type}")

            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(X_train, y_train)

            y_pred = list(pipe.predict(X_test))
            y_probs = pipe.predict_proba(X_test)

            m = compute_classification_metrics(
                y_true=y_test,
                y_pred=y_pred,
                class_names=self.class_names,
                y_probs=y_probs,
                groups=test_groups,
                compute_ci=True,
                seed=seed,
            )
            seed_results[seed] = ModelEvaluationResult(
                seed=seed,
                metrics=m,
                y_pred=y_pred,
                y_probs=y_probs,
            )

        agg = self._aggregate_seed_metrics(
            model_name=f"Handcrafted ({classifier_type.upper()})",
            seed_results=seed_results,
            primary_seed=primary_seed,
        )
        return agg, seed_results

    def evaluate_mert_head(
        self,
        E_train: np.ndarray,
        y_train: np.ndarray,
        E_test: np.ndarray,
        y_test: np.ndarray,
        test_groups: Optional[Sequence[str]] = None,
        seeds: Sequence[int] = (42, 123, 999),
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 16,
    ) -> Tuple[AggregatedModelStats, Dict[int, ModelEvaluationResult]]:
        """Evaluates frozen MERT embeddings + neural classification head across multiple random seeds."""
        if not seeds:
            raise ValueError("At least one random seed must be provided for evaluation.")

        primary_seed = seeds[0]
        seed_results: Dict[int, ModelEvaluationResult] = {}
        y_train_idx = torch.tensor([self.class_to_idx[c] for c in y_train], dtype=torch.long)
        E_train_t = torch.tensor(E_train, dtype=torch.float32)
        E_test_t = torch.tensor(E_test, dtype=torch.float32)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = MERTClassifierHead(
                input_dim=E_train.shape[1],
                num_classes=len(self.class_names),
            ).to(self.device)

            train_ds = TensorDataset(E_train_t, y_train_idx)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

            model.train()
            for _ in range(epochs):
                for b_x, b_y in train_loader:
                    b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                    optimizer.zero_grad()
                    logits = model(b_x)
                    loss = criterion(logits, b_y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                test_logits = model(E_test_t.to(self.device))
                probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
                pred_indices = np.argmax(probs, axis=1)
                y_pred = [self.class_names[i] for i in pred_indices]

            m = compute_classification_metrics(
                y_true=y_test,
                y_pred=y_pred,
                class_names=self.class_names,
                y_probs=probs,
                groups=test_groups,
                compute_ci=True,
                seed=seed,
            )
            seed_results[seed] = ModelEvaluationResult(
                seed=seed,
                metrics=m,
                y_pred=y_pred,
                y_probs=probs,
            )

        agg = self._aggregate_seed_metrics(
            model_name="Frozen MERT (95M)",
            seed_results=seed_results,
            primary_seed=primary_seed,
        )
        return agg, seed_results

    def evaluate_hybrid_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        E_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        E_test: np.ndarray,
        test_groups: Optional[Sequence[str]] = None,
        seeds: Sequence[int] = (42, 123, 999),
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 16,
    ) -> Tuple[AggregatedModelStats, Dict[int, ModelEvaluationResult]]:
        """Evaluates late-fusion Hybrid architecture across multiple random seeds."""
        if not seeds:
            raise ValueError("At least one random seed must be provided for evaluation.")

        primary_seed = seeds[0]
        seed_results: Dict[int, ModelEvaluationResult] = {}
        y_train_idx = torch.tensor([self.class_to_idx[c] for c in y_train], dtype=torch.long)

        M_train_t = torch.tensor(X_train, dtype=torch.float32)
        M_test_t = torch.tensor(X_test, dtype=torch.float32)
        E_train_t = torch.tensor(E_train, dtype=torch.float32)
        E_test_t = torch.tensor(E_test, dtype=torch.float32)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = HybridRagaClassifier(
                musical_dim=X_train.shape[1],
                mert_dim=E_train.shape[1],
                num_classes=len(self.class_names),
            ).to(self.device)

            train_ds = TensorDataset(M_train_t, E_train_t, y_train_idx)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

            model.train()
            for _ in range(epochs):
                for m_b, e_b, y_b in train_loader:
                    m_b, e_b, y_b = m_b.to(self.device), e_b.to(self.device), y_b.to(self.device)
                    optimizer.zero_grad()
                    logits = model(m_b, e_b)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                test_logits = model(M_test_t.to(self.device), E_test_t.to(self.device))
                probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
                pred_indices = np.argmax(probs, axis=1)
                y_pred = [self.class_names[i] for i in pred_indices]

            m = compute_classification_metrics(
                y_true=y_test,
                y_pred=y_pred,
                class_names=self.class_names,
                y_probs=probs,
                groups=test_groups,
                compute_ci=True,
                seed=seed,
            )
            seed_results[seed] = ModelEvaluationResult(
                seed=seed,
                metrics=m,
                y_pred=y_pred,
                y_probs=probs,
            )

        agg = self._aggregate_seed_metrics(
            model_name="Hybrid (Pitch + MERT)",
            seed_results=seed_results,
            primary_seed=primary_seed,
        )
        return agg, seed_results

    def _aggregate_seed_metrics(
        self,
        model_name: str,
        seed_results: Dict[int, ModelEvaluationResult | EvaluationMetrics],
        primary_seed: int,
    ) -> AggregatedModelStats:
        """Aggregates metrics across seeds into mean and sample standard deviation."""
        seeds = list(seed_results.keys())

        def _get_metric(res: ModelEvaluationResult | EvaluationMetrics) -> EvaluationMetrics:
            return res.metrics if isinstance(res, ModelEvaluationResult) else res

        accs = [_get_metric(res).accuracy for res in seed_results.values()]
        mf1s = [_get_metric(res).macro_f1 for res in seed_results.values()]
        wf1s = [_get_metric(res).weighted_f1 for res in seed_results.values()]
        m_precs = [_get_metric(res).macro_precision for res in seed_results.values()]
        m_recs = [_get_metric(res).macro_recall for res in seed_results.values()]

        lls = [_get_metric(res).multiclass_log_loss for res in seed_results.values() if _get_metric(res).multiclass_log_loss is not None]
        briers = [_get_metric(res).brier_score for res in seed_results.values() if _get_metric(res).brier_score is not None]
        eces = [_get_metric(res).expected_calibration_error for res in seed_results.values() if _get_metric(res).expected_calibration_error is not None]

        rep_res = seed_results[primary_seed]
        rep_metrics = rep_res.metrics if isinstance(rep_res, ModelEvaluationResult) else rep_res

        return AggregatedModelStats(
            model_name=model_name,
            num_seeds=len(seeds),
            seeds=seeds,
            accuracy_mean=round(float(np.mean(accs)), 4),
            accuracy_std=round(float(np.std(accs)), 4),
            macro_f1_mean=round(float(np.mean(mf1s)), 4),
            macro_f1_std=round(float(np.std(mf1s)), 4),
            weighted_f1_mean=round(float(np.mean(wf1s)), 4),
            weighted_f1_std=round(float(np.std(wf1s)), 4),
            macro_precision_mean=round(float(np.mean(m_precs)), 4),
            macro_recall_mean=round(float(np.mean(m_recs)), 4),
            log_loss_mean=round(float(np.mean(lls)), 4) if lls else None,
            brier_score_mean=round(float(np.mean(briers)), 4) if briers else None,
            ece_mean=round(float(np.mean(eces)), 4) if eces else None,
            representative_metrics=rep_metrics,
            primary_seed=primary_seed,
        )

    @staticmethod
    def format_comparison_markdown_table(aggregated_models: List[AggregatedModelStats]) -> str:
        """Formats the consolidated benchmark into a publication-ready markdown comparison table."""
        lines = [
            "| Model Architecture | Macro-F1 (Mean ± Std) | Accuracy (Mean ± Std) | Weighted-F1 | Log Loss | Brier Score | ECE |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for a in aggregated_models:
            mf1_str = f"`{a.macro_f1_mean:.4f} ± {a.macro_f1_std:.4f}`"
            acc_str = f"`{a.accuracy_mean:.4f} ± {a.accuracy_std:.4f}`"
            wf1_str = f"`{a.weighted_f1_mean:.4f}`"
            ll_str = f"`{a.log_loss_mean:.4f}`" if a.log_loss_mean is not None else "N/A"
            br_str = f"`{a.brier_score_mean:.4f}`" if a.brier_score_mean is not None else "N/A"
            ece_str = f"`{a.ece_mean:.4f}`" if a.ece_mean is not None else "N/A"
            lines.append(
                f"| **{a.model_name}** | {mf1_str} | {acc_str} | {wf1_str} | {ll_str} | {br_str} | {ece_str} |"
            )
        return "\n".join(lines)
