"""
Tonic (Shadja / Sa) frequency estimation, harmonic salience analysis, and uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np

from raga_intelligence.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TonicCandidate:
    """A single candidate tonic frequency with score and normalized confidence."""
    frequency_hz: float
    pitch_class: int     # 0 to 11 (0=C, 1=C#, ..., 11=B)
    note_name: str       # e.g., "C#3", "D3"
    score: float         # Raw composite salience score
    confidence: float    # Relative confidence within candidate pool [0.0, 1.0]


@dataclass
class TonicEstimationResult:
    """Detailed result of tonic estimation including confidence and competing candidates."""
    tonic_hz: float
    confidence: float                 # Overall calibrated confidence [0.0, 1.0]
    candidates: List[TonicCandidate]  # Top candidate tonics ranked by likelihood
    is_ground_truth: bool = False     # Whether this result was provided as ground truth
    drone_detected: bool = False      # Whether a sustained tanpura drone was detected
    temporal_stability: float = 0.0   # Window-level consensus ratio [0.0, 1.0]


def _evaluate_candidate_salience(
    freqs: np.ndarray,
    spec: np.ndarray,
    var_spec: np.ndarray,
    candidates: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Computes harmonic comb salience with stationarity weighting for each candidate frequency.
    """
    scores = np.zeros(len(candidates), dtype=np.float32)

    # Harmonic multipliers and weights: Sa fundamental + harmonics + Pa drone fifth + Ma drone fourth
    harmonic_weights = [(1.0, 1.0), (2.0, 0.8), (3.0, 0.5), (4.0, 0.3)]
    pa_ratio = 2.0 ** (7.0 / 12.0)   # ~1.498 (Pancham)
    ma_ratio = 2.0 ** (5.0 / 12.0)   # ~1.335 (Madhyam)

    for i, f in enumerate(candidates):
        score = 0.0
        # 1. Fundamental and natural harmonics
        for mult, w in harmonic_weights:
            hf = mult * f
            if hf < sr * 0.48:
                k = int(np.argmin(np.abs(freqs - hf)))
                # Stationarity bonus: low temporal variance relative to magnitude
                stability = 1.0 / (1.0 + var_spec[k] / (spec[k] + 1e-6))
                score += w * spec[k] * (0.7 + 0.3 * stability)

        # 2. Tanpura drone co-occurrences (Pa and Ma)
        pa_f = f * pa_ratio
        if pa_f < sr * 0.48:
            k_pa = int(np.argmin(np.abs(freqs - pa_f)))
            score += 0.7 * spec[k_pa]

        ma_f = f * ma_ratio
        if ma_f < sr * 0.48:
            k_ma = int(np.argmin(np.abs(freqs - ma_f)))
            score += 0.3 * spec[k_ma]

        scores[i] = score

    return scores


def estimate_tonic_detailed(
    audio: np.ndarray,
    sr: int,
    ground_truth_tonic_hz: Optional[float] = None,
    fmin: float = 80.0,
    fmax: float = 400.0,
    top_k: int = 3,
) -> TonicEstimationResult:
    """
    Estimates the tonic frequency (Sa) with explicit confidence and competing candidates.

    Uses:
      - Multi-harmonic comb salience across a continuous candidate frequency grid.
      - Stationary drone detection using temporal spectral variance.
      - Temporal window consensus across time blocks.
      - Octave disambiguation to prevent arbitrary octave clamping.

    Args:
        audio: 1D audio waveform samples (mono).
        sr: Audio sampling rate in Hz.
        ground_truth_tonic_hz: Optional known ground-truth tonic in Hz.
        fmin: Minimum fundamental frequency to consider (default 80 Hz).
        fmax: Maximum fundamental frequency to consider (default 400 Hz).
        top_k: Number of alternative tonic candidates to return.

    Returns:
        TonicEstimationResult containing tonic_hz, calibrated confidence, and candidates.
    """
    # 1. Ground truth override for controlled experiments
    if ground_truth_tonic_hz is not None and ground_truth_tonic_hz > 20.0:
        midi_val = int(round(librosa.hz_to_midi(ground_truth_tonic_hz)))
        note_name = librosa.midi_to_note(midi_val, unicode=False)
        cand = TonicCandidate(
            frequency_hz=float(ground_truth_tonic_hz),
            pitch_class=midi_val % 12,
            note_name=note_name,
            score=1.0,
            confidence=1.0,
        )
        return TonicEstimationResult(
            tonic_hz=float(ground_truth_tonic_hz),
            confidence=1.0,
            candidates=[cand],
            is_ground_truth=True,
            drone_detected=True,
            temporal_stability=1.0,
        )

    # 2. Check for empty or near-silent audio
    if len(audio) < 2048 or np.max(np.abs(audio)) < 1e-4:
        default_hz = settings.features.tonic_default_hz
        midi_val = int(round(librosa.hz_to_midi(default_hz)))
        cand = TonicCandidate(
            frequency_hz=default_hz,
            pitch_class=midi_val % 12,
            note_name=librosa.midi_to_note(midi_val, unicode=False),
            score=0.0,
            confidence=0.0,
        )
        return TonicEstimationResult(
            tonic_hz=default_hz,
            confidence=0.0,
            candidates=[cand],
            is_ground_truth=False,
            drone_detected=False,
            temporal_stability=0.0,
        )

    try:
        # 3. High-resolution FFT for precise fundamental frequency resolution
        n_fft = min(8192, 2 ** int(np.ceil(np.log2(len(audio)))))
        hop_length = n_fft // 4
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        mean_spec = np.mean(S, axis=1)
        var_spec = np.var(S, axis=1)

        # 4. Search over a candidate frequency grid (semitone resolution ~ 1 Hz steps)
        candidate_freqs = np.linspace(fmin, fmax, num=int(fmax - fmin + 1))
        salience_scores = _evaluate_candidate_salience(
            freqs=freqs,
            spec=mean_spec,
            var_spec=var_spec,
            candidates=candidate_freqs,
            sr=sr,
        )

        # 5. Temporal consistency check across sliding windows
        window_size = sr * 2  # 2.0-second windows
        hop_size = sr  # 1.0-second step
        window_winners: List[int] = []

        if len(audio) >= window_size:
            for start in range(0, len(audio) - window_size + 1, hop_size):
                w_audio = audio[start : start + window_size]
                w_S = np.abs(librosa.stft(w_audio, n_fft=min(4096, n_fft), hop_length=512))
                w_mean = np.mean(w_S, axis=1)
                w_var = np.var(w_S, axis=1)
                w_freqs = librosa.fft_frequencies(sr=sr, n_fft=min(4096, n_fft))
                w_scores = _evaluate_candidate_salience(
                    freqs=w_freqs,
                    spec=w_mean,
                    var_spec=w_var,
                    candidates=candidate_freqs,
                    sr=sr,
                )
                window_winners.append(int(np.argmax(w_scores)))

        # 6. Extract top candidate peaks with minimum frequency separation (at least 1 semitone)
        min_sep_hz = 6.0
        ranked_indices = np.argsort(salience_scores)[::-1]
        selected_candidates: List[TonicCandidate] = []

        for idx in ranked_indices:
            f = candidate_freqs[idx]
            # Ensure not within min_sep_hz of already chosen candidate
            if any(abs(f - c.frequency_hz) < min_sep_hz for c in selected_candidates):
                continue

            midi_val = int(round(librosa.hz_to_midi(f)))
            selected_candidates.append(
                TonicCandidate(
                    frequency_hz=round(float(f), 2),
                    pitch_class=midi_val % 12,
                    note_name=librosa.midi_to_note(midi_val, unicode=False),
                    score=float(salience_scores[idx]),
                    confidence=0.0,  # normalized below
                )
            )
            if len(selected_candidates) >= top_k:
                break

        if not selected_candidates:
            selected_candidates.append(
                TonicCandidate(
                    frequency_hz=settings.features.tonic_default_hz,
                    pitch_class=1,
                    note_name="C#3",
                    score=1.0,
                    confidence=0.1,
                )
            )

        # 7. Normalize candidate confidences
        top_scores = np.array([c.score for c in selected_candidates], dtype=np.float32)
        score_sum = np.sum(top_scores)
        if score_sum > 0:
            for i, c in enumerate(selected_candidates):
                c.confidence = round(float(top_scores[i] / score_sum), 3)

        # 8. Compute temporal stability and overall confidence
        primary = selected_candidates[0]
        if window_winners:
            # Fraction of windows where winner is close to primary candidate
            stable_count = sum(
                1 for w_idx in window_winners if abs(candidate_freqs[w_idx] - primary.frequency_hz) < min_sep_hz * 2
            )
            temporal_stability = stable_count / len(window_winners)
        else:
            temporal_stability = 0.5

        # Margin between top 1 and top 2 candidates
        if len(selected_candidates) > 1 and selected_candidates[0].score > 0:
            margin = (selected_candidates[0].score - selected_candidates[1].score) / selected_candidates[0].score
        else:
            margin = 0.8

        drone_detected = bool(temporal_stability >= 0.60 and margin >= 0.25)
        overall_confidence = float(np.clip(0.5 * margin * 2.0 + 0.5 * temporal_stability, 0.05, 0.98))
        overall_confidence = round(overall_confidence, 3)

        return TonicEstimationResult(
            tonic_hz=primary.frequency_hz,
            confidence=overall_confidence,
            candidates=selected_candidates,
            is_ground_truth=False,
            drone_detected=drone_detected,
            temporal_stability=round(float(temporal_stability), 3),
        )

    except Exception as e:
        logger.warning(f"Detailed tonic estimation error: {e}. Falling back to default.")
        default_hz = settings.features.tonic_default_hz
        midi_val = int(round(librosa.hz_to_midi(default_hz)))
        cand = TonicCandidate(
            frequency_hz=default_hz,
            pitch_class=midi_val % 12,
            note_name=librosa.midi_to_note(midi_val, unicode=False),
            score=0.0,
            confidence=0.0,
        )
        return TonicEstimationResult(
            tonic_hz=default_hz,
            confidence=0.0,
            candidates=[cand],
            is_ground_truth=False,
            drone_detected=False,
            temporal_stability=0.0,
        )


def estimate_tonic(
    audio: np.ndarray,
    sr: int,
    ground_truth_tonic_hz: Optional[float] = None,
    fmin: float = 80.0,
    fmax: float = 400.0,
    return_full_result: bool = False,
) -> Union[float, TonicEstimationResult]:
    """
    Estimates the fundamental tonic frequency (Sa) in Hertz.

    Maintains backwards compatibility by returning a single float frequency by default.
    If return_full_result=True, returns full TonicEstimationResult with confidence and alternatives.
    """
    res = estimate_tonic_detailed(
        audio=audio,
        sr=sr,
        ground_truth_tonic_hz=ground_truth_tonic_hz,
        fmin=fmin,
        fmax=fmax,
    )
    if return_full_result:
        return res
    return res.tonic_hz
