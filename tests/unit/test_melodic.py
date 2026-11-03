"""
Unit tests for sequential melodic feature extraction and raga grammar dynamics.
"""

import numpy as np
import pytest

from raga_intelligence.config import FeatureConfig
from raga_intelligence.features.extractor import BaselineFeatureExtractor
from raga_intelligence.features.melodic import (
    MelodicFeatureExtractor,
    MelodicSequentialFeatures,
)


def test_melodic_feature_dimensions_and_stability():
    extractor = MelodicFeatureExtractor(hop_length=512, sr=24000)

    # 1. Test empty / unvoiced contour
    empty_f0 = np.zeros(100, dtype=np.float32)
    empty_voiced = np.zeros(100, dtype=bool)
    empty_feats = extractor.extract(empty_f0, empty_voiced, tonic_hz=146.83)

    assert isinstance(empty_feats, MelodicSequentialFeatures)
    assert len(empty_feats.interval_histogram) == 13
    assert len(empty_feats.swara_dwell_ratios) == 12
    assert len(empty_feats.direction_stats) == 6
    assert len(empty_feats.ornamentation_stats) == 5
    assert len(empty_feats.phrase_stats) == 4
    assert len(empty_feats.feature_vector) == 40
    assert not np.any(np.isnan(empty_feats.feature_vector))
    assert not np.any(np.isinf(empty_feats.feature_vector))
    assert np.all(empty_feats.feature_vector == 0.0)


def test_known_synthetic_ascending_sequence():
    """Verify that an ascending melodic phrase produces positive ascending statistics."""
    extractor = MelodicFeatureExtractor(hop_length=512, sr=24000)
    tonic_hz = 146.83  # D3

    # Construct Bhoopali ascending phrase: S -> R -> G -> P -> D -> S'
    # Semitones: [0, 2, 4, 7, 9, 12]
    swara_semitones = [0, 2, 4, 7, 9, 12]
    frames_per_note = 20
    f0_list = []
    for s in swara_semitones:
        hz = tonic_hz * (2.0 ** (s / 12.0))
        f0_list.extend([hz] * frames_per_note)

    f0 = np.array(f0_list, dtype=np.float32)
    voiced = np.ones(len(f0), dtype=bool)

    feats = extractor.extract(f0=f0, voiced_flag=voiced, tonic_hz=tonic_hz)

    assert len(feats.feature_vector) == 40
    assert not np.any(np.isnan(feats.feature_vector))

    # Direction stats: asc_ratio > desc_ratio, motion imbalance > 0
    asc_ratio = feats.direction_stats[0]
    desc_ratio = feats.direction_stats[1]
    motion_imbalance = feats.direction_stats[5]

    assert asc_ratio > desc_ratio
    assert motion_imbalance > 0.0

    # Dwell ratios: Sa (0), Re (2), Ga (4), Pa (7), Dha (9) must be active
    dwell = feats.swara_dwell_ratios
    assert dwell[0] > 0.10  # Sa
    assert dwell[2] > 0.10  # Re
    assert dwell[4] > 0.10  # Ga
    assert dwell[7] > 0.10  # Pa
    assert dwell[9] > 0.10  # Dha
    # Unused notes: Komal Re (1), Komal Ga (3), Ma (5), Tivra Ma (6), Ni (11) must be zero
    assert dwell[1] == 0.0
    assert dwell[3] == 0.0
    assert dwell[5] == 0.0


def test_melodic_duration_invariance():
    """Verify that normalized sequential distributions are invariant to duration."""
    extractor = MelodicFeatureExtractor(hop_length=512, sr=24000)
    tonic_hz = 138.59

    # Pattern: 10 frames Sa, 10 frames Pa
    pattern = [tonic_hz] * 10 + [tonic_hz * 1.5] * 10
    f0_short = np.array(pattern * 2, dtype=np.float32)
    f0_long = np.array(pattern * 6, dtype=np.float32)

    feats_short = extractor.extract(f0_short, np.ones(len(f0_short), dtype=bool), tonic_hz)
    feats_long = extractor.extract(f0_long, np.ones(len(f0_long), dtype=bool), tonic_hz)

    # Interval histograms and swara dwell must be virtually identical within small sample boundary effects
    assert np.allclose(feats_short.interval_histogram, feats_long.interval_histogram, atol=0.03)
    assert np.allclose(feats_short.swara_dwell_ratios, feats_long.swara_dwell_ratios, atol=0.03)


def test_glissando_and_meend_dynamics():
    """Verify detection of continuous microtonal glides (meend)."""
    extractor = MelodicFeatureExtractor(hop_length=512, sr=24000)
    tonic_hz = 146.83
    dt = 512 / 24000.0  # ~0.0213 s per frame

    # Synthesize smooth glide over 1 second from Sa to Ga (400 cents over 47 frames)
    # Velocity ~ 400 cents / 1.0 s = 400 cents/s (within meend range [80, 600])
    n_frames = 50
    cents_curve = np.linspace(0.0, 400.0, n_frames)
    f0 = (tonic_hz * (2.0 ** (cents_curve / 1200.0))).astype(np.float32)
    voiced = np.ones(len(f0), dtype=bool)

    feats = extractor.extract(f0=f0, voiced_flag=voiced, tonic_hz=tonic_hz)

    # Ornamentation stats: index 3 is glide_fraction
    glide_fraction = feats.ornamentation_stats[3]
    assert glide_fraction > 0.80


def test_baseline_extractor_integration_and_backward_compatibility():
    """Verify that BaselineFeatureExtractor preserves 35 dims by default and 75 dims when enabled."""
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = (0.7 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    # 1. Default config: baseline 35-dim feature vector
    default_extractor = BaselineFeatureExtractor()
    feat_default = default_extractor.extract(sine, sr=sr, ground_truth_tonic_hz=220.0)
    assert len(feat_default.feature_vector) == 35
    assert feat_default.melodic_features is None

    # 2. Config with melodic features enabled: 35 + 40 = 75 dims
    extended_config = FeatureConfig(include_melodic_features=True)
    extended_extractor = BaselineFeatureExtractor(config=extended_config)
    feat_extended = extended_extractor.extract(sine, sr=sr, ground_truth_tonic_hz=220.0)
    assert len(feat_extended.feature_vector) == 75
    assert feat_extended.melodic_features is not None
    assert len(feat_extended.melodic_features.feature_vector) == 40
