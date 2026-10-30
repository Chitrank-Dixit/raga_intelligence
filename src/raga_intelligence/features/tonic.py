"""
Tonic (Shadja / Sa) frequency estimation and normalization.
"""

from __future__ import annotations

import logging
from typing import Optional

import librosa
import numpy as np

from raga_intelligence.config import settings

logger = logging.getLogger(__name__)


def estimate_tonic(
    audio: np.ndarray,
    sr: int,
    ground_truth_tonic_hz: Optional[float] = None,
    fmin: float = 100.0,
    fmax: float = 300.0,
) -> float:
    """
    Estimates the fundamental tonic frequency (Sa) in Hertz.
    
    If ground_truth_tonic_hz is provided and positive, it is returned directly.
    Otherwise, computes a pitch salience histogram using short-time harmonic analysis
    focused on vocal/drone fundamental ranges (100Hz to 300Hz).
    """
    if ground_truth_tonic_hz is not None and ground_truth_tonic_hz > 20.0:
        return float(ground_truth_tonic_hz)

    try:
        # Compute chroma feature to find primary pitch class
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=sr,
            hop_length=512,
            fmin=librosa.note_to_hz("C2"),
            n_chroma=12,
        )
        chroma_mean = np.mean(chroma, axis=1)

        # Indian music drone typically sounds (Sa, Pa) or (Sa, Ma).
        # Pa is 7 semitones above Sa, Ma is 5 semitones above Sa.
        scores = np.zeros(12)
        for sa_idx in range(12):
            pa_idx = (sa_idx + 7) % 12
            ma_idx = (sa_idx + 5) % 12
            # Sa + Pa weight is strongest indicator of Indian classical drone
            scores[sa_idx] = chroma_mean[sa_idx] + 0.7 * chroma_mean[pa_idx] + 0.3 * chroma_mean[ma_idx]

        best_pitch_class = int(np.argmax(scores))

        # Map best pitch class to typical vocal octave (Octave 3: ~130Hz - 260Hz)
        # C3 is ~130.81 Hz
        base_midi = 48 + best_pitch_class  # MIDI 48 = C3
        estimated_hz = float(librosa.midi_to_hz(base_midi))

        # Clamp within expected bounds
        if estimated_hz < fmin:
            estimated_hz *= 2.0
        elif estimated_hz > fmax:
            estimated_hz /= 2.0

        return round(estimated_hz, 2)

    except Exception as e:
        logger.warning(f"Tonic estimation encountered error: {e}. Using default fallback.")
        return settings.features.tonic_default_hz

