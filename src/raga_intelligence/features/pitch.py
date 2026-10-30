"""
Pitch contour extraction and tonic-relative pitch class distribution (PCD).
"""

from __future__ import annotations

import logging
from typing import Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def extract_pitch_contour(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 1200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts fundamental frequency (F0) contour using parabolic-interpolated YIN or PYIN.
    
    Returns:
        f0: Array of estimated fundamental frequencies in Hz (NaN or 0.0 for unvoiced).
        voiced_flag: Boolean mask indicating voiced frames.
    """
    try:
        f0 = librosa.yin(
            y=audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
        )
        # Threshold unvoiced/noisy frames
        voiced_flag = (f0 >= fmin) & (f0 <= fmax) & ~np.isnan(f0)
        return f0, voiced_flag
    except Exception as e:
        logger.warning(f"YIN pitch extraction failed ({e}), returning empty pitch track.")
        frames = len(audio) // hop_length + 1
        return np.zeros(frames), np.zeros(frames, dtype=bool)


def compute_tonic_relative_pcd(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    tonic_hz: float,
    num_bins: int = 12,
) -> np.ndarray:
    """
    Computes the 12-bin tonic-relative Pitch Class Distribution (PCD).
    
    Each voiced frame's frequency f is converted to cents relative to tonic:
        cents = 1200 * log2(f / tonic_hz) mod 1200
    And histogrammed into 12 semitone bins (100 cents each) centered on swara intervals:
        Bin 0: Sa (0 cents)
        Bin 1: Komal Re (100 cents)
        ...
        Bin 11: Shuddha Ni (1100 cents)
        
    Normalized so the sum equals 1.0 (or zero if no voiced frames).
    """
    valid_f0 = f0[voiced_flag]
    if len(valid_f0) == 0 or tonic_hz <= 0:
        return np.zeros(num_bins, dtype=np.float32)

    # Convert Hz to cents relative to tonic
    cents = 1200.0 * np.log2(valid_f0 / tonic_hz)
    cents_mod = np.mod(cents, 1200.0)

    # Bin into 12 semitones: centered around 0, 100, 200, ...
    # Bin i covers [i*100 - 50, i*100 + 50)
    shifted = np.mod(cents_mod + 50.0, 1200.0)
    bin_indices = np.floor(shifted / (1200.0 / num_bins)).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    hist, _ = np.histogram(bin_indices, bins=np.arange(num_bins + 1))
    hist = hist.astype(np.float32)

    total = np.sum(hist)
    if total > 0:
        hist /= total

    return hist

