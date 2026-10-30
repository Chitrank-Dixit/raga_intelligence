"""
Unit tests for pitch, swara, tonic, and baseline feature extraction.
"""

import numpy as np
import pytest

from raga_intelligence.features.extractor import BaselineFeatureExtractor, MusicalFeatures
from raga_intelligence.features.pitch import compute_tonic_relative_pcd
from raga_intelligence.features.swara import cents_to_swara, SWARA_SEMITONES
from raga_intelligence.features.tonic import estimate_tonic


def test_cents_to_swara_mapping():
    # Sa = 0 cents
    swara, dev = cents_to_swara(0.0)
    assert swara == "S"
    assert abs(dev) < 1.0

    # Komal Re = 100 cents
    swara, dev = cents_to_swara(100.0)
    assert swara == "r"

    # Shuddha Re = 200 cents
    swara, dev = cents_to_swara(202.0)
    assert swara == "R"
    assert abs(dev - 2.0) < 0.2

    # Pancham = 700 cents
    swara, dev = cents_to_swara(700.0)
    assert swara == "P"


def test_compute_tonic_relative_pcd():
    tonic_hz = 138.59  # C#3
    # Synthetic pitch track on Sa and Pa
    f0 = np.array([tonic_hz, tonic_hz * 1.5, tonic_hz, tonic_hz * 1.5])
    voiced = np.array([True, True, True, True])

    pcd = compute_tonic_relative_pcd(f0, voiced, tonic_hz, num_bins=12)
    assert len(pcd) == 12
    assert np.isclose(np.sum(pcd), 1.0)
    # Bin 0 (Sa) and Bin 7 (Pa) must have highest energy
    assert pcd[0] > 0.4
    assert pcd[7] > 0.4


def test_baseline_feature_extractor():
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    # 220Hz sine wave (A3)
    audio = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    extractor = BaselineFeatureExtractor()
    feat = extractor.extract(audio, sr=sr, ground_truth_tonic_hz=220.0)

    assert isinstance(feat, MusicalFeatures)
    assert feat.tonic_hz == 220.0
    assert len(feat.pcd) == 12
    assert len(feat.chroma) == 12
    assert len(feat.feature_vector) == 35  # 12 pcd + 12 chroma + 10 spectral + 1 voiced
    assert feat.voiced_ratio > 0.5

