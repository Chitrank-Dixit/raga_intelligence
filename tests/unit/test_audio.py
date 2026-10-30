"""
Unit tests for audio loading, decoding, and segmentation.
"""

import numpy as np
import pytest
import soundfile as sf
from io import BytesIO

from raga_intelligence.audio.loader import AudioFileError, AudioSignal, load_audio
from raga_intelligence.audio.segmentation import segment_audio


def create_synthetic_wav_bytes(duration_sec: float = 2.0, sr: int = 24000, freq: float = 440.0) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    buf = BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def test_load_audio_from_bytes():
    wav_bytes = create_synthetic_wav_bytes(duration_sec=3.0, sr=24000)
    sig = load_audio(wav_bytes, target_sr=24000)
    assert isinstance(sig, AudioSignal)
    assert sig.sample_rate == 24000
    assert abs(sig.duration_sec - 3.0) < 0.05
    assert sig.channels == 1
    assert sig.samples.ndim == 1


def test_load_audio_resampling():
    wav_bytes = create_synthetic_wav_bytes(duration_sec=2.0, sr=44100)
    sig = load_audio(wav_bytes, target_sr=24000)
    assert sig.sample_rate == 24000
    assert abs(len(sig.samples) - 48000) <= 2


def test_load_audio_duration_constraints():
    short_bytes = create_synthetic_wav_bytes(duration_sec=0.2, sr=24000)
    with pytest.raises(AudioFileError, match="shorter than minimum"):
        load_audio(short_bytes, min_duration_sec=1.0)


def test_segment_audio():
    wav_bytes = create_synthetic_wav_bytes(duration_sec=8.0, sr=24000)
    sig = load_audio(wav_bytes, target_sr=24000)
    segments = segment_audio(sig, segment_duration_sec=3.0, hop_duration_sec=2.0)
    assert len(segments) >= 3
    for seg in segments:
        assert len(seg.samples) == int(3.0 * 24000)
        assert seg.sample_rate == 24000

