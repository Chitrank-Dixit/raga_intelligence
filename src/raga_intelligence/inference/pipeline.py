"""
End-to-end inference pipeline supporting classical classification and cross-genre similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import numpy as np

from raga_intelligence.audio.loader import AudioFileError, AudioSignal
from raga_intelligence.audio.preprocessor import AudioPreprocessor
from raga_intelligence.audio.segmentation import AudioSegment
from raga_intelligence.config import RAGA_LIST, settings
from raga_intelligence.features.extractor import BaselineFeatureExtractor
from raga_intelligence.models.baseline import BaselineRagaClassifier
from raga_intelligence.similarity.engine import AnalysisReport, MelodicSimilarityEngine, RagaSimilarityMatch

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Consolidated inference outcome delivered to API and UI."""
    duration_sec: float
    estimated_tonic_hz: float
    voiced_ratio: float
    detected_dominant_swaras: List[str]
    ranked_matches: List[RagaSimilarityMatch]
    strict_classification_winner: Optional[str]
    strict_classification_confidence: Optional[float]
    abstention_triggered: bool
    abstention_reason: Optional[str]
    caveat_statement: str


class InferencePipeline:
    """
    Main inference interface orchestrating decoding, preprocessing, feature extraction,
    and raga intelligence matching.
    """

    def __init__(
        self,
        baseline_classifier_path: Optional[Path | str] = None,
        similarity_engine: Optional[MelodicSimilarityEngine] = None,
    ):
        self.preprocessor = AudioPreprocessor()
        self.similarity_engine = similarity_engine or MelodicSimilarityEngine()
        self.extractor = BaselineFeatureExtractor()

        self.classifier: Optional[BaselineRagaClassifier] = None
        target_ckpt = baseline_classifier_path
        if target_ckpt is None:
            default_ckpt = Path("./artifacts/checkpoints/baseline_svm.joblib")
            if default_ckpt.exists():
                target_ckpt = default_ckpt

        if target_ckpt and Path(target_ckpt).exists():
            try:
                self.classifier = BaselineRagaClassifier.load(target_ckpt)
                logger.info(f"Loaded classifier from {target_ckpt}")
            except Exception as e:
                logger.warning(f"Could not load baseline classifier: {e}")

    def is_ready(self) -> tuple[bool, str, list[str]]:
        """
        Validates pipeline readiness.
        Returns:
            (ready_bool, active_backend_name, list_of_loaded_models)
        """
        loaded_models = []
        if self.similarity_engine and len(self.similarity_engine.registry.all_prototypes()) > 0:
            loaded_models.append("MelodicSimilarityEngine (PCD + Swara Templates)")
        else:
            return False, "none", []

        if self.extractor is not None:
            loaded_models.append("BaselineFeatureExtractor")
        else:
            return False, "none", []

        active_backend = "melodic_similarity_prototypes"
        if self.classifier is not None and getattr(self.classifier, "is_fitted", False):
            loaded_models.append(f"BaselineRagaClassifier ({self.classifier.classifier_type})")
            active_backend = f"baseline_{self.classifier.classifier_type}"

        return True, active_backend, loaded_models

    def analyze_audio(
        self,
        audio_source: Union[str, Path, BinaryIO, bytes],
        ground_truth_tonic_hz: Optional[float] = None,
        top_k: int = 5,
    ) -> AnalysisResult:
        """
        Executes complete audio analysis pipeline.
        """
        # 1. Preprocess audio into signal and 5-second overlapping segments
        signal, segments = self.preprocessor.process(audio_source, segment=True)

        # 2. Run melodic similarity analysis
        report: AnalysisReport = self.similarity_engine.analyze(
            signal=signal,
            segments=segments,
            ground_truth_tonic_hz=ground_truth_tonic_hz,
            top_k=top_k,
        )

        # 3. Optional strict classification via trained baseline model
        winner = None
        confidence = None
        if self.classifier is not None and not report.abstention_triggered:
            try:
                feat = self.extractor.extract(signal, ground_truth_tonic_hz=report.estimated_tonic_hz)
                probs = self.classifier.predict_proba(feat.feature_vector.reshape(1, -1))[0]
                best_idx = int(np.argmax(probs))
                winner = self.classifier.idx_to_class[best_idx]
                confidence = round(float(probs[best_idx]), 4)
            except Exception as e:
                logger.debug(f"Classifier prediction bypassed: {e}")

        return AnalysisResult(
            duration_sec=report.input_duration_sec,
            estimated_tonic_hz=report.estimated_tonic_hz,
            voiced_ratio=report.voiced_ratio,
            detected_dominant_swaras=report.detected_dominant_swaras,
            ranked_matches=report.matches,
            strict_classification_winner=winner,
            strict_classification_confidence=confidence,
            abstention_triggered=report.abstention_triggered,
            abstention_reason=report.abstention_reason,
            caveat_statement=report.caveat_statement,
        )

