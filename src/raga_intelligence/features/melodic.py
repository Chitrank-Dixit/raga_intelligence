"""
Melodic and sequential feature extraction for Indian classical raga analysis.

Extracts melodic interval distributions, ascending/descending motion dynamics,
phrase-level cadence statistics, and microtonal ornamentation (meend / gamak / andolan).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MelodicSequentialFeatures:
    """Detailed sequential and melodic dynamics features."""
    interval_histogram: np.ndarray        # 13-dim: magnitude distribution of intervals (0 to 12 semitones)
    swara_dwell_ratios: np.ndarray        # 12-dim: relative sustained duration on each swara (0=Sa .. 11=Ni)
    direction_stats: np.ndarray           # 6-dim: ascending, descending, stationary ratios and mean velocities
    ornamentation_stats: np.ndarray       # 5-dim: pitch stability, vibrato extent, glissando (meend) dynamics
    phrase_stats: np.ndarray              # 4-dim: phrase rate, duration, range, cadence alignment
    feature_vector: np.ndarray            # 40-dim concatenated normalized feature vector


class MelodicFeatureExtractor:
    """
    Extracts sequential melodic features from fundamental frequency contours.
    All features are normalized and invariant to overall track duration.
    """

    def __init__(
        self,
        hop_length: int = 512,
        sr: int = 24000,
        cents_per_semitone: float = 100.0,
        stationary_threshold_cents: float = 25.0,
        glissando_min_rate: float = 80.0,   # cents / sec
        glissando_max_rate: float = 600.0,  # cents / sec
    ):
        self.hop_length = hop_length
        self.sr = sr
        self.dt = hop_length / float(sr)
        self.cents_per_semitone = cents_per_semitone
        self.stationary_threshold = stationary_threshold_cents
        self.glissando_min_rate = glissando_min_rate
        self.glissando_max_rate = glissando_max_rate

    def extract(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        tonic_hz: float,
    ) -> MelodicSequentialFeatures:
        """
        Extracts 40-dimensional melodic and sequential feature vector.

        Handles empty or unvoiced audio gracefully by returning zero-filled arrays.
        """
        if len(f0) == 0 or np.sum(voiced_flag) < 2 or tonic_hz <= 0.0:
            return self._empty_features()

        # 1. Convert valid voiced frames to cents relative to tonic
        valid_indices = np.where(voiced_flag & (f0 > 0.0))[0]
        if len(valid_indices) < 2:
            return self._empty_features()

        valid_f0 = f0[valid_indices]
        cents = 1200.0 * np.log2(valid_f0 / tonic_hz)

        # 2. Compute inter-frame pitch changes (cents)
        # Find consecutive voiced frames
        consecutive_mask = (np.diff(valid_indices) == 1)
        if np.sum(consecutive_mask) < 1:
            return self._empty_features()

        diff_cents = np.diff(cents)[consecutive_mask]
        abs_diff_cents = np.abs(diff_cents)
        velocity_cents_sec = abs_diff_cents / self.dt

        # 3. Melodic Interval Histogram (13 bins: 0 to 12 semitones)
        # Map step sizes to semitone bins: 0 = stationary (< 50 cents), 1 = ~100 cents, etc.
        semitone_steps = np.clip(np.round(abs_diff_cents / self.cents_per_semitone).astype(int), 0, 12)
        interval_counts, _ = np.histogram(semitone_steps, bins=np.arange(14))
        interval_hist = interval_counts.astype(np.float32)
        total_intervals = np.sum(interval_hist)
        if total_intervals > 0:
            interval_hist /= total_intervals

        # 4. Swara Dwell Ratios (12 bins: Sa to Ni)
        cents_mod = np.mod(cents + 50.0, 1200.0)
        swara_bins = np.clip(np.floor(cents_mod / 100.0).astype(int), 0, 11)
        swara_counts, _ = np.histogram(swara_bins, bins=np.arange(13))
        swara_dwell = swara_counts.astype(np.float32)
        total_dwell = np.sum(swara_dwell)
        if total_dwell > 0:
            swara_dwell /= total_dwell

        # 5. Ascending vs. Descending Movement Patterns (6 dims)
        asc_mask = diff_cents > self.stationary_threshold
        desc_mask = diff_cents < -self.stationary_threshold
        stat_mask = np.abs(diff_cents) <= self.stationary_threshold

        n_diff = len(diff_cents)
        asc_ratio = float(np.sum(asc_mask)) / n_diff
        desc_ratio = float(np.sum(desc_mask)) / n_diff
        stat_ratio = float(np.sum(stat_mask)) / n_diff

        mean_asc_cents = float(np.mean(diff_cents[asc_mask])) if np.sum(asc_mask) > 0 else 0.0
        mean_desc_cents = float(np.abs(np.mean(diff_cents[desc_mask]))) if np.sum(desc_mask) > 0 else 0.0
        motion_imbalance = (asc_ratio - desc_ratio) / (asc_ratio + desc_ratio + 1e-6)

        direction_stats = np.array([
            asc_ratio,
            desc_ratio,
            stat_ratio,
            mean_asc_cents / 100.0,   # normalized to semitones
            mean_desc_cents / 100.0,  # normalized to semitones
            motion_imbalance,
        ], dtype=np.float32)

        # 6. Ornamentation & Microtonal Inflection (Meend / Gamak) Statistics (5 dims)
        # Local detrended variance (meend / vibrato extent)
        if len(cents) >= 5:
            # Windowed median filter approximation for local baseline
            local_median = np.convolve(cents, np.ones(5) / 5.0, mode="same")
            detrended_cents = np.abs(cents - local_median)
            pitch_variance = float(np.std(detrended_cents))
            vibrato_extent = float(np.percentile(detrended_cents, 90) - np.percentile(detrended_cents, 10))
        else:
            pitch_variance = 0.0
            vibrato_extent = 0.0

        meend_frames = (velocity_cents_sec >= self.glissando_min_rate) & (velocity_cents_sec <= self.glissando_max_rate)
        jump_frames = velocity_cents_sec > self.glissando_max_rate

        mean_glide_rate = float(np.mean(velocity_cents_sec)) / 1000.0  # normalized (kCents/s)
        glide_fraction = float(np.sum(meend_frames)) / n_diff
        jump_fraction = float(np.sum(jump_frames)) / n_diff

        ornamentation_stats = np.array([
            np.clip(pitch_variance / 100.0, 0.0, 5.0),     # semitones std
            np.clip(vibrato_extent / 100.0, 0.0, 5.0),     # semitones 10-90 range
            np.clip(mean_glide_rate, 0.0, 5.0),
            glide_fraction,
            jump_fraction,
        ], dtype=np.float32)

        # 7. Phrase Segmentation & Cadence Alignment (4 dims)
        # Phrases are continuous voiced segments separated by pause > 0.20s (consecutive_mask break)
        pause_indices = np.where(~consecutive_mask)[0]
        phrase_starts = np.insert(pause_indices + 1, 0, 0)
        phrase_ends = np.append(pause_indices, len(valid_indices) - 1)

        phrase_durations: list[float] = []
        phrase_ranges: list[float] = []
        cadence_hits = 0

        for p_start, p_end in zip(phrase_starts, phrase_ends):
            if p_end <= p_start:
                continue
            p_cents = cents[p_start : p_end + 1]
            p_dur = (valid_indices[p_end] - valid_indices[p_start]) * self.dt
            phrase_durations.append(p_dur)
            phrase_ranges.append(float(np.max(p_cents) - np.min(p_cents)))

            # Cadence note: ending pitch within +/- 50 cents of Sa (0) or Pa (700)
            end_note_mod = np.mod(p_cents[-1], 1200.0)
            if end_note_mod < 50.0 or end_note_mod > 1150.0 or abs(end_note_mod - 700.0) < 50.0:
                cadence_hits += 1

        total_audio_sec = len(f0) * self.dt
        n_phrases = len(phrase_durations)
        phrases_per_min = (n_phrases / max(total_audio_sec, 1.0)) * 60.0
        mean_dur = float(np.mean(phrase_durations)) if phrase_durations else 0.0
        mean_range = float(np.mean(phrase_ranges)) / 100.0 if phrase_ranges else 0.0
        cadence_alignment = (cadence_hits / max(n_phrases, 1))

        phrase_stats = np.array([
            np.clip(phrases_per_min / 60.0, 0.0, 2.0),
            np.clip(mean_dur / 5.0, 0.0, 2.0),
            np.clip(mean_range / 12.0, 0.0, 2.0),
            cadence_alignment,
        ], dtype=np.float32)

        # Concatenate 13 + 12 + 6 + 5 + 4 = 40 dimensions
        feature_vector = np.concatenate([
            interval_hist,
            swara_dwell,
            direction_stats,
            ornamentation_stats,
            phrase_stats,
        ]).astype(np.float32)

        return MelodicSequentialFeatures(
            interval_histogram=interval_hist,
            swara_dwell_ratios=swara_dwell,
            direction_stats=direction_stats,
            ornamentation_stats=ornamentation_stats,
            phrase_stats=phrase_stats,
            feature_vector=feature_vector,
        )

    def _empty_features(self) -> MelodicSequentialFeatures:
        """Returns zero-valued features for unvoiced or missing pitch contours."""
        interval_hist = np.zeros(13, dtype=np.float32)
        swara_dwell = np.zeros(12, dtype=np.float32)
        direction_stats = np.zeros(6, dtype=np.float32)
        ornamentation_stats = np.zeros(5, dtype=np.float32)
        phrase_stats = np.zeros(4, dtype=np.float32)
        feature_vector = np.zeros(40, dtype=np.float32)

        return MelodicSequentialFeatures(
            interval_histogram=interval_hist,
            swara_dwell_ratios=swara_dwell,
            direction_stats=direction_stats,
            ornamentation_stats=ornamentation_stats,
            phrase_stats=phrase_stats,
            feature_vector=feature_vector,
        )

