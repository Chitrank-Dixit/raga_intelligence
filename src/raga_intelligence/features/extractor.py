"""
Comprehensive musical feature extractor combining pitch class distributions and spectral features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from raga_intelligence.audio.loader import AudioSignal
from raga_intelligence.audio.segmentation import AudioSegment
from raga_intelligence.config import FeatureConfig, settings
from raga_intelligence.features.melodic import MelodicFeatureExtractor, MelodicSequentialFeatures
from raga_intelligence.features.pitch import compute_tonic_relative_pcd, extract_pitch_contour
from raga_intelligence.features.swara import SWARA_NAMES
from raga_intelligence.features.tonic import estimate_tonic


@dataclass
class MusicalFeatures:
    """Extracted musical feature representation for an audio segment or track."""
    tonic_hz: float
    pcd: np.ndarray             # 12-dim tonic-relative pitch class distribution
    chroma: np.ndarray          # 12-dim mean tonic-aligned chroma
    spectral_stats: np.ndarray  # Spectral centroid, rolloff, contrast summary
    dominant_swaras: list[str]  # Top swaras active in the excerpt
    voiced_ratio: float         # Proportion of voiced frames
    feature_vector: np.ndarray  # Concatenated 1D representation for scikit-learn models
    tonic_confidence: float = 1.0
    tonic_candidates: list[float] = None
    melodic_features: Optional[MelodicSequentialFeatures] = None


class BaselineFeatureExtractor:
    """Extracts explicit musical and spectral features for raga classification."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or settings.features
        self.melodic_extractor = MelodicFeatureExtractor(
            hop_length=self.config.hop_length,
            sr=24000,
        )

    def extract(
        self,
        audio_or_segment: AudioSignal | AudioSegment | np.ndarray,
        sr: int = 24000,
        ground_truth_tonic_hz: Optional[float] = None,
    ) -> MusicalFeatures:
        if isinstance(audio_or_segment, (AudioSignal, AudioSegment)):
            samples = audio_or_segment.samples
            sr = audio_or_segment.sample_rate
        else:
            samples = audio_or_segment

        # 1. Estimate or verify tonic with uncertainty quantification
        tonic_res = estimate_tonic(
            audio=samples,
            sr=sr,
            ground_truth_tonic_hz=ground_truth_tonic_hz,
            return_full_result=True,
        )
        tonic_hz = tonic_res.tonic_hz
        tonic_confidence = tonic_res.confidence
        tonic_candidates = [c.frequency_hz for c in tonic_res.candidates]

        # 2. Extract pitch contour using probabilistic voicing
        f0, voiced, probs = extract_pitch_contour(
            audio=samples,
            sr=sr,
            hop_length=self.config.hop_length,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            method=getattr(self.config, "pitch_tracker", "pyin"),
            voicing_threshold=getattr(self.config, "pitch_confidence_threshold", 0.4),
            min_energy_threshold=getattr(self.config, "min_energy_threshold", 1e-4),
            return_voicing_probs=True,
        )
        voiced_ratio = float(np.mean(voiced)) if len(voiced) > 0 else 0.0

        # 3. Compute 12-bin tonic-relative Pitch Class Distribution (PCD) with probability weighting
        pcd = compute_tonic_relative_pcd(f0, voiced, tonic_hz, num_bins=12, weights=probs)

        # 4. Compute Chroma CQT and align cyclically with estimated tonic
        try:
            chroma = librosa.feature.chroma_cqt(
                y=samples,
                sr=sr,
                hop_length=self.config.hop_length,
                n_chroma=12,
            )
            chroma_mean = np.mean(chroma, axis=1)

            # Roll chroma vector so bin 0 corresponds to the estimated tonic note
            tonic_note_idx = int(round(librosa.hz_to_midi(tonic_hz))) % 12
            chroma_aligned = np.roll(chroma_mean, -tonic_note_idx)
            chroma_norm = chroma_aligned / (np.linalg.norm(chroma_aligned) + 1e-8)
        except Exception:
            chroma_norm = np.zeros(12, dtype=np.float32)

        # 5. Spectral descriptors (centroid, contrast, flatness)
        try:
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr)) / (sr / 2.0)
            spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=samples, sr=sr)) / (sr / 2.0)
            spec_flatness = np.mean(librosa.feature.spectral_flatness(y=samples))
            spec_contrast = np.mean(librosa.feature.spectral_contrast(y=samples, sr=sr), axis=1)
            spectral_stats = np.concatenate([
                np.array([spec_cent, spec_rolloff, spec_flatness], dtype=np.float32),
                spec_contrast.astype(np.float32),
            ])
        except Exception:
            spectral_stats = np.zeros(10, dtype=np.float32)

        # 6. Dominant swaras
        top_indices = np.argsort(pcd)[::-1][:4]
        dominant_swaras = [SWARA_NAMES[i] for i in top_indices if pcd[i] > 0.05]

        # 7. Concatenated feature vector (12 PCD + 12 Chroma + 10 Spectral + 1 voiced ratio = 35 dims)
        combined_features = np.concatenate([
            pcd,
            chroma_norm,
            spectral_stats,
            np.array([voiced_ratio], dtype=np.float32),
        ])

        # 8. Optional sequential melodic features (40 dims)
        mel_feats: Optional[MelodicSequentialFeatures] = None
        if getattr(self.config, "include_melodic_features", False):
            mel_feats = self.melodic_extractor.extract(
                f0=f0,
                voiced_flag=voiced,
                tonic_hz=tonic_hz,
            )
            combined_features = np.concatenate([combined_features, mel_feats.feature_vector])

        return MusicalFeatures(
            tonic_hz=tonic_hz,
            pcd=pcd,
            chroma=chroma_norm,
            spectral_stats=spectral_stats,
            dominant_swaras=dominant_swaras,
            voiced_ratio=voiced_ratio,
            feature_vector=combined_features,
            tonic_confidence=tonic_confidence,
            tonic_candidates=tonic_candidates,
            melodic_features=mel_feats,
        )

