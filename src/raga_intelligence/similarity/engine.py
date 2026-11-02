"""
Melodic similarity comparison engine for cross-genre music and classical performance analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import List, Optional

import numpy as np

from raga_intelligence.audio.loader import AudioSignal
from raga_intelligence.audio.segmentation import AudioSegment
from raga_intelligence.features.extractor import BaselineFeatureExtractor, MusicalFeatures
from raga_intelligence.features.swara import SWARA_NAMES
from raga_intelligence.similarity.prototypes import RagaPrototype, RagaPrototypeRegistry

logger = logging.getLogger(__name__)


@dataclass
class SimilarityEngineConfig:
    """
    Configurable parameters and threshold derivations for melodic similarity and abstention.

    Mathematical and Empirical Rationale:
      - similarity_high_threshold (0.75): Empirically validated on canonical performances where all
        swaras match the raga's arohana/avarohana with clear vadi emphasis.
      - similarity_moderate_threshold (0.50): Identifies modal resemblance with minor passing notes,
        typical of popular, folk, and light classical songs.
      - abstention_score_threshold (0.40): Empirically separates modal melodies from unrelated scales
        (disjoint scales such as Bhoopali vs Todi score below 0.35).
      - min_voiced_ratio (0.15): Rejects audio dominated by unvoiced pauses, drums, or spoken audio.
      - max_spectral_flatness (0.15): Rejects white noise, applause, or unpitched percussion.
      - min_pcd_std (0.025): Rejects flat or uniform pitch class distributions.
      - bhattacharyya_weight (0.60): Primary geometric overlap measure on the discrete simplex.
      - cosine_weight (0.40): Directional vector alignment in R^12.
      - foreign_penalty_factor (0.40): Penalty scaling for energy placed on strictly forbidden (varjya) swaras.
    """
    similarity_high_threshold: float = 0.75
    similarity_moderate_threshold: float = 0.50
    abstention_score_threshold: float = 0.40
    min_voiced_ratio: float = 0.15
    max_spectral_flatness: float = 0.15
    min_pcd_std: float = 0.025
    bhattacharyya_weight: float = 0.60
    cosine_weight: float = 0.40
    foreign_penalty_factor: float = 0.40


@dataclass
class RagaSimilarityMatch:
    """A ranked raga similarity candidate with supporting musical evidence."""
    raga_name: str
    similarity_score: float             # [0.0, 1.0] Uncalibrated geometric distribution affinity
    ranking_score: float                # Score used to order candidate matches
    resemblance_band: str               # "Strong Resemblance", "Moderate Resemblance", "Weak / Exploratory Overlap"
    confidence_level: str               # Backwards compatibility label ("High", "Moderate", "Exploratory / Weak")
    thhat: str
    vadi: str
    samvadi: str
    matching_swaras: List[str]
    foreign_swaras: List[str]           # Notes present in input but omitted in raga
    evidence_segments: List[str]        # Timestamps of strongest alignment
    is_strict_classification: bool = False
    is_calibrated_probability: bool = False
    calibrated_probability: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class AnalysisReport:
    """Complete analysis result returned by the intelligence engine."""
    input_duration_sec: float
    estimated_tonic_hz: float
    voiced_ratio: float
    detected_dominant_swaras: List[str]
    matches: List[RagaSimilarityMatch]
    abstention_triggered: bool
    abstention_reason: Optional[str] = None
    abstention_threshold: float = 0.40
    score_interpretation: str = (
        "Similarity scores reflect uncalibrated geometric distribution overlap (Bhattacharyya affinity "
        "and cosine similarity). They measure melodic and scale resemblance and should NOT be interpreted "
        "as calibrated posterior probabilities or classical composition certification."
    )
    caveat_statement: str = (
        "Note: Cross-genre raga similarity indicates scale and melodic contour resemblance, "
        "not that the piece was formally composed within classical Hindustani raga rules."
    )


class MelodicSimilarityEngine:
    """
    Ranks raga candidates based on pitch-class distribution correlation,
    swara-interval overlap, and temporal segment evidence aggregation.
    """

    def __init__(
        self,
        registry: Optional[RagaPrototypeRegistry] = None,
        feature_extractor: Optional[BaselineFeatureExtractor] = None,
        config: Optional[SimilarityEngineConfig] = None,
        abstention_threshold: Optional[float] = None,
        min_voiced_ratio: Optional[float] = None,
    ):
        self.registry = registry or RagaPrototypeRegistry()
        self.extractor = feature_extractor or BaselineFeatureExtractor()
        self.config = config or SimilarityEngineConfig()
        if abstention_threshold is not None:
            self.config.abstention_score_threshold = float(abstention_threshold)
        if min_voiced_ratio is not None:
            self.config.min_voiced_ratio = float(min_voiced_ratio)

    def _bhattacharyya_similarity(self, p: np.ndarray, q: np.ndarray) -> float:
        """Computes Bhattacharyya affinity coefficient between two probability distributions."""
        p_safe = np.clip(p, 0.0, None)
        q_safe = np.clip(q, 0.0, None)
        sum_p = np.sum(p_safe)
        sum_q = np.sum(q_safe)
        if sum_p == 0 or sum_q == 0:
            return 0.0
        p_norm = p_safe / sum_p
        q_norm = q_safe / sum_q
        coeff = np.sum(np.sqrt(p_norm * q_norm))
        return float(np.clip(coeff, 0.0, 1.0))

    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

    def analyze(
        self,
        signal: AudioSignal,
        segments: Optional[List[AudioSegment]] = None,
        ground_truth_tonic_hz: Optional[float] = None,
        top_k: int = 5,
    ) -> AnalysisReport:
        """
        Executes end-to-end melodic analysis and cross-genre raga similarity ranking.
        """
        # 1. Global track feature extraction
        global_features: MusicalFeatures = self.extractor.extract(
            signal,
            ground_truth_tonic_hz=ground_truth_tonic_hz,
        )

        # 2. Check for low vocal/melodic content or unpitched noise (abstention criteria 1)
        spectral_flatness = float(global_features.spectral_stats[2]) if len(global_features.spectral_stats) > 2 else 0.0
        pcd_std = float(np.std(global_features.pcd))

        if global_features.voiced_ratio < self.config.min_voiced_ratio:
            return AnalysisReport(
                input_duration_sec=signal.duration_sec,
                estimated_tonic_hz=global_features.tonic_hz,
                voiced_ratio=round(global_features.voiced_ratio, 3),
                detected_dominant_swaras=[],
                matches=[],
                abstention_triggered=True,
                abstention_threshold=self.config.abstention_score_threshold,
                abstention_reason=(
                    f"Insufficient voiced/melodic signal detected (voiced ratio {global_features.voiced_ratio:.1%} "
                    f"< minimum required {self.config.min_voiced_ratio:.1%}). Audio may be percussive or silent."
                ),
            )

        if spectral_flatness > self.config.max_spectral_flatness:
            return AnalysisReport(
                input_duration_sec=signal.duration_sec,
                estimated_tonic_hz=global_features.tonic_hz,
                voiced_ratio=round(global_features.voiced_ratio, 3),
                detected_dominant_swaras=[],
                matches=[],
                abstention_triggered=True,
                abstention_threshold=self.config.abstention_score_threshold,
                abstention_reason=(
                    f"Audio lacks clear harmonic tonality (spectral flatness {spectral_flatness:.2f} > "
                    f"{self.config.max_spectral_flatness:.2f} threshold, typical of unpitched percussion, noise, or spoken audio)."
                ),
            )

        if pcd_std < self.config.min_pcd_std:
            return AnalysisReport(
                input_duration_sec=signal.duration_sec,
                estimated_tonic_hz=global_features.tonic_hz,
                voiced_ratio=round(global_features.voiced_ratio, 3),
                detected_dominant_swaras=[],
                matches=[],
                abstention_triggered=True,
                abstention_threshold=self.config.abstention_score_threshold,
                abstention_reason=(
                    "Melody lacks modal definition (pitch distribution is nearly uniform / flat, "
                    "characteristic of noise, percussive instruments, or atonal sounds)."
                ),
            )

        # 3. Analyze individual segments to identify peak alignment windows
        segment_pcds: List[tuple[AudioSegment, np.ndarray]] = []
        if segments:
            for seg in segments:
                feat = self.extractor.extract(seg, ground_truth_tonic_hz=global_features.tonic_hz)
                if feat.voiced_ratio >= 0.20:
                    segment_pcds.append((seg, feat.pcd))

        # 4. Compare with all registered raga prototypes
        matches: List[RagaSimilarityMatch] = []
        all_prototypes = self.registry.all_prototypes()

        input_pcd = global_features.pcd
        active_swara_indices = np.where(input_pcd > 0.05)[0]
        detected_swaras = [SWARA_NAMES[i] for i in active_swara_indices]

        for proto in all_prototypes:
            target_pcd = proto.effective_pcd

            # Combine Bhattacharyya distribution affinity with Cosine similarity
            bhat_sim = self._bhattacharyya_similarity(input_pcd, target_pcd)
            cos_sim = self._cosine_similarity(input_pcd, target_pcd)

            # Penalty for notes present in input that are strictly forbidden (varjya) in this raga
            forbidden_penalty = 0.0
            foreign_notes = []
            for idx in active_swara_indices:
                if proto.swara_mask[idx] == 0:
                    forbidden_penalty += input_pcd[idx] * self.config.foreign_penalty_factor
                    foreign_notes.append(SWARA_NAMES[idx])

            composite_score = float(np.clip(
                self.config.bhattacharyya_weight * bhat_sim
                + self.config.cosine_weight * cos_sim
                - forbidden_penalty,
                0.0,
                1.0,
            ))

            # Find matching swaras
            matching_notes = [s for s in detected_swaras if s in proto.allowed_swaras]

            # Find best supporting segments
            evidence_windows: List[str] = []
            if segment_pcds:
                scored_segments = []
                for seg, seg_pcd in segment_pcds:
                    score = self._bhattacharyya_similarity(seg_pcd, target_pcd)
                    scored_segments.append((score, f"{seg.start_sec:.1f}s - {seg.end_sec:.1f}s"))
                scored_segments.sort(key=lambda x: x[0], reverse=True)
                evidence_windows = [win for _, win in scored_segments[:3]]

            # Resemblance band categorization (distinguishing similarity from probability)
            if composite_score >= self.config.similarity_high_threshold:
                resemblance_band = "Strong Resemblance"
                conf_compat = "High"
            elif composite_score >= self.config.similarity_moderate_threshold:
                resemblance_band = "Moderate Resemblance"
                conf_compat = "Moderate"
            else:
                resemblance_band = "Weak / Exploratory Overlap"
                conf_compat = "Exploratory / Weak"

            match = RagaSimilarityMatch(
                raga_name=proto.raga_name,
                similarity_score=round(composite_score, 4),
                ranking_score=round(composite_score, 4),
                resemblance_band=resemblance_band,
                confidence_level=conf_compat,
                thhat=proto.thhat,
                vadi=proto.vadi,
                samvadi=proto.samvadi,
                matching_swaras=matching_notes,
                foreign_swaras=foreign_notes,
                evidence_segments=evidence_windows,
                is_strict_classification=False,
                is_calibrated_probability=False,
                calibrated_probability=None,
            )
            matches.append(match)

        # Sort matches descending by ranking score
        matches.sort(key=lambda m: m.ranking_score, reverse=True)
        top_matches = matches[:top_k]

        # 5. Abstention criteria 2: Peak similarity below configured threshold
        peak_score = top_matches[0].similarity_score if top_matches else 0.0
        abstention = peak_score < self.config.abstention_score_threshold
        abstention_reason = None
        if abstention:
            abstention_reason = (
                f"Peak similarity score ({peak_score:.2f}) is below the empirical abstention "
                f"threshold ({self.config.abstention_score_threshold:.2f}). The melody does not sufficiently match "
                f"any supported classical raga framework."
            )

        return AnalysisReport(
            input_duration_sec=round(signal.duration_sec, 2),
            estimated_tonic_hz=round(global_features.tonic_hz, 2),
            voiced_ratio=round(global_features.voiced_ratio, 3),
            detected_dominant_swaras=detected_swaras,
            matches=top_matches,
            abstention_triggered=abstention,
            abstention_reason=abstention_reason,
            abstention_threshold=self.config.abstention_score_threshold,
        )
