"""
Unit tests for tonic (Sa) estimation, harmonic salience, and uncertainty quantification.
"""

import numpy as np
import pytest

from raga_intelligence.features.tonic import (
    estimate_tonic,
    estimate_tonic_detailed,
    TonicEstimationResult,
)


def _synthesize_tanpura(
    sa_hz: float,
    duration_sec: float = 2.5,
    sr: int = 24000,
    pa_weight: float = 0.5,
) -> np.ndarray:
    """Helper to synthesize authentic tanpura harmonic drone."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    pa_hz = sa_hz * (2.0 ** (7.0 / 12.0))  # Pancham ~ 1.498

    # Drone strings: 1st Pa, 2nd & 3rd Middle Sa, 4th Lower Sa
    drone = np.zeros_like(t)
    drone += 0.35 * np.sin(2 * np.pi * sa_hz * t)           # Lower Sa fundamental
    drone += 0.25 * np.sin(2 * np.pi * 2.0 * sa_hz * t)     # Middle Sa (octave)
    drone += 0.15 * np.sin(2 * np.pi * 3.0 * sa_hz * t)     # 3rd harmonic
    drone += 0.10 * np.sin(2 * np.pi * 4.0 * sa_hz * t)     # 4th harmonic
    drone += pa_weight * np.sin(2 * np.pi * pa_hz * t)      # Pa fundamental
    drone += (pa_weight * 0.4) * np.sin(2 * np.pi * 2.0 * pa_hz * t)  # Pa octave

    # Soft envelope
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.5
    return (drone * env).astype(np.float32)


def test_synthetic_drone_known_tonic():
    sr = 24000
    expected_sa = 146.83  # D3
    drone = _synthesize_tanpura(sa_hz=expected_sa, duration_sec=3.0, sr=sr)

    res = estimate_tonic(drone, sr=sr, return_full_result=True)
    assert isinstance(res, TonicEstimationResult)
    assert abs(res.tonic_hz - expected_sa) < 3.0
    assert res.confidence > 0.60
    assert res.drone_detected is True
    assert len(res.candidates) >= 1
    assert res.candidates[0].note_name.startswith("D")


def test_tonic_pa_emphasis_does_not_usurp_sa():
    """Verify that a loud Pa string does not fool the estimator into declaring Pa as Sa."""
    sr = 24000
    sa_hz = 138.59  # C#3
    # Pa string has equal amplitude to Sa fundamental
    drone = _synthesize_tanpura(sa_hz=sa_hz, duration_sec=3.0, sr=sr, pa_weight=0.35)

    res = estimate_tonic(drone, sr=sr, return_full_result=True)
    # Must correctly identify C# (138.59 Hz) as Sa, not G# (207.65 Hz)
    assert abs(res.tonic_hz - sa_hz) < 3.0
    assert res.candidates[0].note_name.startswith("C#") or res.candidates[0].note_name.startswith("Db")


def test_tonic_with_accompaniment():
    """Verify that rhythmic/percussive accompaniment does not derail tonic tracking."""
    sr = 24000
    sa_hz = 220.0  # A3 (Female vocal Sa)
    drone = _synthesize_tanpura(sa_hz=sa_hz, duration_sec=3.0, sr=sr)

    # Add percussive clicks and broadband bursts to simulate tabla
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    percussion = np.zeros_like(t)
    click_steps = np.arange(0, len(t), sr // 4)
    percussion[click_steps] = 0.4
    percussion[click_steps + 1] = -0.3

    audio = drone + percussion
    res = estimate_tonic(audio, sr=sr, return_full_result=True)
    assert abs(res.tonic_hz - sa_hz) < 3.0
    assert res.confidence > 0.50


def test_tonic_changing_melodic_emphasis():
    """Verify that melodic variations over a drone do not alter the estimated tonic."""
    sr = 24000
    sa_hz = 146.83
    drone = _synthesize_tanpura(sa_hz=sa_hz, duration_sec=3.0, sr=sr)

    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    # Melodic notes moving: Komal Re (155 Hz), Gandhar (185 Hz), Dhaivat (220 Hz)
    melody = np.zeros_like(t)
    chunk = len(t) // 3
    melody[:chunk] = 0.3 * np.sin(2 * np.pi * 155.56 * t[:chunk])
    melody[chunk : 2 * chunk] = 0.3 * np.sin(2 * np.pi * 185.0 * t[chunk : 2 * chunk])
    melody[2 * chunk :] = 0.3 * np.sin(2 * np.pi * 220.0 * t[2 * chunk :])

    audio = drone + melody
    res = estimate_tonic(audio, sr=sr, return_full_result=True)
    assert abs(res.tonic_hz - sa_hz) < 4.0


def test_tonic_ambiguous_evidence():
    """Verify that unpitched noise or chromatic chaos exposes low confidence."""
    sr = 24000
    rng = np.random.default_rng(999)
    noise = (rng.standard_normal(int(sr * 2.0)) * 0.1).astype(np.float32)

    res = estimate_tonic(noise, sr=sr, return_full_result=True)
    assert res.confidence < 0.45
    assert res.drone_detected is False
    assert len(res.candidates) >= 2


def test_tonic_ground_truth_override():
    """Verify that ground truth is returned with 100% confidence without computation."""
    sr = 24000
    dummy_audio = np.zeros(sr, dtype=np.float32)

    res = estimate_tonic(dummy_audio, sr=sr, ground_truth_tonic_hz=220.0, return_full_result=True)
    assert res.tonic_hz == 220.0
    assert res.is_ground_truth is True
    assert res.confidence == 1.0

    # Test float return for backwards compatibility
    val = estimate_tonic(dummy_audio, sr=sr, ground_truth_tonic_hz=138.59)
    assert isinstance(val, float)
    assert val == 138.59

