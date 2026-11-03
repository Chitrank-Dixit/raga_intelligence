"""
Pitch contour extraction, voiced-frame detection, and tonic-relative pitch class distribution (PCD).
"""

from __future__ import annotations

import logging
from typing import Literal, Tuple, Union

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def extract_pitch_contour(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 1200.0,
    method: Literal["pyin", "yin"] = "pyin",
    voicing_threshold: float = 0.4,
    min_energy_threshold: float = 1e-4,
    return_voicing_probs: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extracts fundamental frequency (F0) contour and robustly separates pitch estimation
    from voiced/unvoiced classification.

    Supports:
      - 'pyin': Probabilistic YIN with HMM Viterbi tracking and continuous voicing probabilities.
      - 'yin': Standard YIN with RMS energy gating.

    Args:
        audio: 1D audio waveform samples (mono).
        sr: Sampling rate in Hz.
        hop_length: Number of samples between successive frames.
        fmin: Minimum fundamental frequency to track (Hz).
        fmax: Maximum fundamental frequency to track (Hz).
        method: Pitch tracker ('pyin' or 'yin').
        voicing_threshold: Minimum probability threshold to declare a frame voiced.
        min_energy_threshold: Minimum frame RMS energy below which frames are forced unvoiced.
        return_voicing_probs: Whether to return the 3rd tuple element containing continuous voicing probabilities.

    Returns:
        If return_voicing_probs is False:
            (f0, voiced_flag)
        If return_voicing_probs is True:
            (f0, voiced_flag, voiced_probs)
    """
    if len(audio) < hop_length:
        frames = 1
        f0 = np.zeros(frames, dtype=np.float32)
        voiced = np.zeros(frames, dtype=bool)
        probs = np.zeros(frames, dtype=np.float32)
        return (f0, voiced, probs) if return_voicing_probs else (f0, voiced)

    # 1. Compute frame-level RMS energy to reliably detect silence / low-energy regions
    try:
        rms = librosa.feature.rms(y=audio, hop_length=hop_length).squeeze(0)
    except Exception:
        rms = np.zeros(max(1, len(audio) // hop_length), dtype=np.float32)

    # 2. Extract pitch contour via chosen method
    if method == "pyin":
        try:
            f0_raw, _, voiced_probs_raw = librosa.pyin(
                y=audio,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                hop_length=hop_length,
            )
            if voiced_probs_raw is None:
                voiced_probs = np.zeros(len(f0_raw), dtype=np.float32)
            else:
                voiced_probs = np.nan_to_num(voiced_probs_raw, nan=0.0).astype(np.float32)

            f0 = np.nan_to_num(f0_raw, nan=0.0).astype(np.float32)

            # Energy gating: suppress voicing on silent or near-silent frames
            min_len = min(len(voiced_probs), len(rms))
            voiced_probs[:min_len] = np.where(
                rms[:min_len] < min_energy_threshold,
                0.0,
                voiced_probs[:min_len],
            )

            # Voiced classification: require confidence threshold AND frequency bounds
            voiced_flag = (
                (voiced_probs >= voicing_threshold)
                & (f0 >= fmin)
                & (f0 <= fmax)
                & ~np.isnan(f0_raw)
            )

        except Exception as e:
            logger.warning(f"pYIN pitch extraction failed ({e}), falling back to empty pitch track.")
            frames = max(1, len(audio) // hop_length)
            f0 = np.zeros(frames, dtype=np.float32)
            voiced_flag = np.zeros(frames, dtype=bool)
            voiced_probs = np.zeros(frames, dtype=np.float32)

    else:  # 'yin'
        try:
            f0_raw = librosa.yin(
                y=audio,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                hop_length=hop_length,
            )
            f0 = np.nan_to_num(f0_raw, nan=0.0).astype(np.float32)
            min_len = min(len(f0), len(rms))

            # YIN does not provide true posterior probabilities; apply energy and range gating
            in_range = (f0 >= fmin) & (f0 <= fmax) & ~np.isnan(f0_raw)
            voiced_flag = np.zeros(len(f0), dtype=bool)
            voiced_flag[:min_len] = in_range[:min_len] & (rms[:min_len] >= min_energy_threshold)

            # Heuristic pseudo-probability
            voiced_probs = np.where(voiced_flag, 0.8, 0.0).astype(np.float32)

        except Exception as e:
            logger.warning(f"YIN pitch extraction failed ({e}), falling back to empty pitch track.")
            frames = max(1, len(audio) // hop_length)
            f0 = np.zeros(frames, dtype=np.float32)
            voiced_flag = np.zeros(frames, dtype=bool)
            voiced_probs = np.zeros(frames, dtype=np.float32)

    # 3. Spectral consistency check: suppress phantom subharmonic tracking for out-of-band signals
    try:
        n_fft = min(2048, 2 ** int(np.ceil(np.log2(max(hop_length * 2, 256)))))
        stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        min_frames = min(len(voiced_flag), stft.shape[1])

        for t in range(min_frames):
            if not voiced_flag[t]:
                continue
            peak_idx = int(np.argmax(stft[:, t]))
            peak_freq = freqs[peak_idx]
            # If the dominant spectral peak is above fmax
            if peak_freq > fmax * 1.05:
                cand_f0 = f0[t]
                cand_idx = int(np.argmin(np.abs(freqs - cand_f0)))
                w_start = max(0, cand_idx - 2)
                w_end = min(len(freqs), cand_idx + 3)
                cand_energy = float(np.max(stft[w_start:w_end, t]))
                peak_energy = float(stft[peak_idx, t])
                if cand_energy < 0.05 * peak_energy:
                    voiced_flag[t] = False
                    voiced_probs[t] = 0.0
    except Exception as e:
        logger.debug(f"Spectral consistency check skipped: {e}")

    if return_voicing_probs:
        return f0, voiced_flag, voiced_probs
    return f0, voiced_flag


def compute_tonic_relative_pcd(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    tonic_hz: float,
    num_bins: int = 12,
    weights: np.ndarray | None = None,
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

    Args:
        f0: Array of estimated fundamental frequencies in Hz.
        voiced_flag: Boolean mask indicating voiced frames.
        tonic_hz: Fundamental frequency of Shadja (Sa) in Hz.
        num_bins: Number of semitone bins (default 12).
        weights: Optional weighting per frame (e.g., voicing probabilities).

    Returns:
        Normalized 12-bin histogram summing to 1.0 (or all zeros if unvoiced).
    """
    valid_mask = voiced_flag & (f0 > 0.0)
    valid_f0 = f0[valid_mask]

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

    frame_weights = None
    if weights is not None:
        frame_weights = weights[valid_mask]

    hist, _ = np.histogram(bin_indices, bins=np.arange(num_bins + 1), weights=frame_weights)
    hist = hist.astype(np.float32)

    total = np.sum(hist)
    if total > 0:
        hist /= total

    return hist
