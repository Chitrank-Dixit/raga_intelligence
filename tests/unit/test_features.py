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


def test_pitch_tracking_known_sine_waves():
    from raga_intelligence.features.pitch import extract_pitch_contour
    sr = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. 220 Hz sine wave (A3)
    sig_220 = (0.7 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    f0, voiced, probs = extract_pitch_contour(sig_220, sr=sr, fmin=65.0, fmax=800.0, return_voicing_probs=True)
    assert np.mean(voiced) > 0.85
    voiced_f0 = f0[voiced]
    assert len(voiced_f0) > 0
    assert abs(np.median(voiced_f0) - 220.0) < 3.0
    assert np.mean(probs[voiced]) > 0.70

    # 2. 440 Hz sine wave (A4)
    sig_440 = (0.7 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    f0_440, voiced_440 = extract_pitch_contour(sig_440, sr=sr, fmin=65.0, fmax=800.0)
    assert np.mean(voiced_440) > 0.85
    assert abs(np.median(f0_440[voiced_440]) - 440.0) < 3.0


def test_pitch_tracking_silence():
    from raga_intelligence.features.pitch import extract_pitch_contour
    sr = 24000
    silence = np.zeros(sr, dtype=np.float32)
    f0, voiced, probs = extract_pitch_contour(silence, sr=sr, return_voicing_probs=True)

    assert np.sum(voiced) == 0
    assert np.all(probs == 0.0)
    assert np.mean(voiced) == 0.0


def test_pitch_tracking_white_noise():
    from raga_intelligence.features.pitch import extract_pitch_contour
    sr = 24000
    rng = np.random.default_rng(1234)
    noise = (rng.standard_normal(sr) * 0.1).astype(np.float32)

    f0, voiced, probs = extract_pitch_contour(
        noise,
        sr=sr,
        method="pyin",
        voicing_threshold=0.4,
        return_voicing_probs=True,
    )
    # PYIN with probability thresholding must reject unvoiced noise
    assert np.mean(voiced) < 0.10
    assert np.mean(probs) < 0.15


def test_pitch_tracking_unvoiced_percussive():
    from raga_intelligence.features.pitch import extract_pitch_contour
    sr = 24000
    # Create isolated impulses (clicks) surrounded by silence
    percussive = np.zeros(sr, dtype=np.float32)
    click_indices = np.arange(0, sr, sr // 8)
    percussive[click_indices] = 1.0
    percussive[click_indices + 1] = -0.8

    f0, voiced = extract_pitch_contour(percussive, sr=sr, method="pyin")
    assert np.mean(voiced) < 0.15


def test_pitch_tracking_changing_pitch():
    from raga_intelligence.features.pitch import extract_pitch_contour
    sr = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Chirp linearly ascending from 200 Hz to 400 Hz
    phase = 2 * np.pi * (200.0 * t + 100.0 * (t ** 2))
    chirp = (0.7 * np.sin(phase)).astype(np.float32)

    f0, voiced = extract_pitch_contour(chirp, sr=sr, fmin=65.0, fmax=800.0, method="pyin")
    assert np.mean(voiced) > 0.70
    valid_f0 = f0[voiced]
    # Beginning frames should be lower in frequency than ending frames
    start_pitch = np.mean(valid_f0[:5])
    end_pitch = np.mean(valid_f0[-5:])
    assert end_pitch > start_pitch + 100.0


def test_pitch_tracking_out_of_range():
    from raga_intelligence.features.pitch import extract_pitch_contour
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)

    # 40 Hz tone (below fmin=100.0)
    sig_low = (0.7 * np.sin(2 * np.pi * 40.0 * t)).astype(np.float32)
    f0_low, voiced_low = extract_pitch_contour(sig_low, sr=sr, fmin=100.0, fmax=500.0)
    assert np.mean(voiced_low) < 0.10

    # 1500 Hz tone (above fmax=800.0)
    sig_high = (0.7 * np.sin(2 * np.pi * 1500.0 * t)).astype(np.float32)
    f0_high, voiced_high = extract_pitch_contour(sig_high, sr=sr, fmin=100.0, fmax=800.0)
    assert np.mean(voiced_high) < 0.10


def test_pitch_tracking_yin_method_comparison():
    from raga_intelligence.features.pitch import extract_pitch_contour
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = (0.7 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)

    # YIN mode
    f0_yin, voiced_yin = extract_pitch_contour(sine, sr=sr, method="yin")
    assert np.mean(voiced_yin) > 0.80
    assert abs(np.median(f0_yin[voiced_yin]) - 330.0) < 5.0

