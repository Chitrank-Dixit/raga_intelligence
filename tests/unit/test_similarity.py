"""
Unit tests for the MelodicSimilarityEngine: score semantics, threshold validation, and calibrated abstention.
"""

from pathlib import Path
import numpy as np
import pytest

from raga_intelligence.audio.loader import AudioSignal
from raga_intelligence.config import CANONICAL_RAGAS
from raga_intelligence.similarity.engine import (
    MelodicSimilarityEngine,
    SimilarityEngineConfig,
    AnalysisReport,
)


def _synthesize_modal_audio(raga_name: str, duration_sec: float = 4.0, sr: int = 24000, tonic_hz: float = 146.83) -> AudioSignal:
    """Helper to synthesize authentic modal audio following raga pitch sets."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sig = np.zeros_like(t)

    # Subtle Tanpura drone
    pa_ratio = 2.0 ** (7.0 / 12.0)
    sig += 0.08 * np.sin(2 * np.pi * tonic_hz * t)
    sig += 0.05 * np.sin(2 * np.pi * tonic_hz * pa_ratio * t)

    # Notes progression
    swara_map = {
        "S": 0, "r": 1, "R": 2, "g": 3, "G": 4, "M": 5, "M'": 6,
        "P": 7, "d": 8, "D": 9, "n": 10, "N": 11,
    }
    arohana = [s for s in CANONICAL_RAGAS[raga_name]["arohana"].replace("'", "").split() if s in swara_map]
    allowed_intervals = [swara_map[s] for s in arohana]

    chunk = len(t) // len(allowed_intervals)
    for i, semitone in enumerate(allowed_intervals):
        freq = tonic_hz * (2.0 ** (semitone / 12.0))
        idx_start = i * chunk
        idx_end = (i + 1) * chunk
        t_sub = t[idx_start:idx_end]
        env = np.sin(np.pi * np.linspace(0, 1, len(t_sub))) ** 0.5
        sig[idx_start:idx_end] += 0.6 * np.sin(2 * np.pi * freq * t_sub) * env

    return AudioSignal(samples=sig.astype(np.float32), sample_rate=sr, duration_sec=duration_sec, channels=1)


def test_similarity_known_positive_pairs():
    """Verify that modal audio achieves strong resemblance against its own prototype."""
    engine = MelodicSimilarityEngine()

    # 1. Bhoopali modal audio
    bhoopali_audio = _synthesize_modal_audio("Bhoopali")
    report = engine.analyze(bhoopali_audio, ground_truth_tonic_hz=146.83, top_k=3)

    assert not report.abstention_triggered
    top_match = report.matches[0]
    assert top_match.raga_name == "Bhoopali"
    assert top_match.similarity_score >= 0.75
    assert top_match.resemblance_band == "Strong Resemblance"
    # Must clearly indicate uncalibrated geometric affinity, not calibrated probability
    assert top_match.is_calibrated_probability is False
    assert top_match.calibrated_probability is None

    # 2. Malkauns modal audio
    malkauns_audio = _synthesize_modal_audio("Malkauns")
    report_m = engine.analyze(malkauns_audio, ground_truth_tonic_hz=146.83, top_k=3)
    assert report_m.matches[0].raga_name == "Malkauns"
    assert report_m.matches[0].similarity_score >= 0.75


def test_similarity_known_dissimilar_pairs():
    """Verify that audio with strictly disjoint pitch sets receives low affinity and foreign penalty."""
    engine = MelodicSimilarityEngine()
    # Bhoopali (S R G P D) has no Komal notes; Todi is dominated by Komal Re, Komal Ga, Komal Dha, Tivra Ma
    bhoopali_audio = _synthesize_modal_audio("Bhoopali")
    report = engine.analyze(bhoopali_audio, ground_truth_tonic_hz=146.83, top_k=10)

    # Find Todi match
    todi_match = next(m for m in report.matches if m.raga_name == "Todi")
    # Dissimilar raga must score in weak resemblance band (< 0.50) and have foreign swaras
    assert todi_match.similarity_score < 0.50
    assert todi_match.resemblance_band == "Weak / Exploratory Overlap"
    assert todi_match.similarity_score < report.matches[0].similarity_score - 0.25
    assert len(todi_match.foreign_swaras) > 0


def test_similarity_abstention_on_unpitched_noise():
    """Verify that white noise triggers immediate abstention."""
    engine = MelodicSimilarityEngine()
    sr = 24000
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(sr * 2) * 0.1).astype(np.float32)
    signal = AudioSignal(samples=noise, sample_rate=sr, duration_sec=2.0, channels=1)

    report = engine.analyze(signal)
    assert report.abstention_triggered is True
    assert (
        "spectral flatness" in report.abstention_reason.lower()
        or "insufficient voiced/melodic" in report.abstention_reason.lower()
    )
    assert len(report.matches) == 0


def test_similarity_abstention_on_silent_unvoiced():
    """Verify that silence or low voiced ratio triggers abstention."""
    engine = MelodicSimilarityEngine()
    sr = 24000
    silence = np.zeros(sr * 2, dtype=np.float32)
    signal = AudioSignal(samples=silence, sample_rate=sr, duration_sec=2.0, channels=1)

    report = engine.analyze(signal)
    assert report.abstention_triggered is True
    assert "insufficient voiced/melodic" in report.abstention_reason.lower()


def test_similarity_boundary_threshold_behavior():
    """Verify that configurable abstention thresholds behave strictly as expected at boundaries."""
    # Configure custom strict threshold of 0.85
    strict_config = SimilarityEngineConfig(abstention_score_threshold=0.85)
    strict_engine = MelodicSimilarityEngine(config=strict_config)

    bhoopali_audio = _synthesize_modal_audio("Bhoopali")
    report = strict_engine.analyze(bhoopali_audio, ground_truth_tonic_hz=146.83)

    peak_score = report.matches[0].similarity_score
    if peak_score < 0.85:
        assert report.abstention_triggered is True
        assert "below the empirical abstention threshold (0.85)" in report.abstention_reason
    else:
        assert report.abstention_triggered is False


def test_score_semantics_and_interpretation_documentation():
    """Verify score interpretation metadata is attached and uncalibrated nature is explicit."""
    engine = MelodicSimilarityEngine()
    audio = _synthesize_modal_audio("Yaman")
    report = engine.analyze(audio, ground_truth_tonic_hz=146.83)

    assert "uncalibrated geometric distribution overlap" in report.score_interpretation.lower()
    assert "not be interpreted as calibrated posterior probabilities" in report.score_interpretation.lower()
    for m in report.matches:
        assert m.ranking_score == m.similarity_score
        assert m.resemblance_band in ["Strong Resemblance", "Moderate Resemblance", "Weak / Exploratory Overlap"]
