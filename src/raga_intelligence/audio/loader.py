"""
Audio decoding, normalization, and validation routines.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Union

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioFileError(Exception):
    """Raised when an audio file cannot be decoded or fails validation."""
    pass


@dataclass
class AudioSignal:
    """Represents a loaded, validated audio signal."""
    samples: np.ndarray  # 1D float32 array
    sample_rate: int
    duration_sec: float
    channels: int
    source_path: Optional[str] = None

    def __post_init__(self):
        if self.samples.ndim != 1:
            raise ValueError(f"Expected 1D mono audio array, got shape {self.samples.shape}")
        if self.duration_sec <= 0:
            raise ValueError(f"Invalid non-positive duration: {self.duration_sec}s")


def load_audio(
    source: Union[str, Path, BinaryIO, bytes],
    target_sr: int = 24000,
    mono: bool = True,
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 600.0,
    normalize_peak: bool = True,
) -> AudioSignal:
    """
    Decodes audio from a file path, file-like object, or byte stream.
    
    Converts to mono float32 and resamples to target_sr.
    Safe against corrupted files and validates duration constraints.
    """
    try:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
            source_path = None
        elif isinstance(source, (str, Path)):
            source_path = str(source)
            if not Path(source_path).exists():
                raise AudioFileError(f"Audio file not found: {source_path}")
        else:
            source_path = getattr(source, "name", None)

        # First attempt fast loading with soundfile
        try:
            samples, orig_sr = sf.read(source, dtype="float32")
        except Exception as sf_err:
            logger.debug(f"Soundfile failed to load ({sf_err}), trying librosa fallback.")
            if isinstance(source, io.BytesIO):
                source.seek(0)
            samples, orig_sr = librosa.load(source, sr=None, mono=False, dtype=np.float32)

        # Handle multichannel
        if samples.ndim > 1:
            channels = samples.shape[1] if samples.shape[0] > samples.shape[1] else samples.shape[0]
            if mono:
                # Average across channels
                if samples.shape[0] < samples.shape[1]:
                    samples = np.mean(samples, axis=0)
                else:
                    samples = np.mean(samples, axis=1)
                channels = 1
        else:
            channels = 1

        # Resample if needed
        if orig_sr != target_sr:
            samples = librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
            curr_sr = target_sr
        else:
            curr_sr = orig_sr

        duration = len(samples) / float(curr_sr)

        if duration < min_duration_sec:
            raise AudioFileError(
                f"Audio duration ({duration:.2f}s) is shorter than minimum required ({min_duration_sec}s)"
            )
        if duration > max_duration_sec:
            logger.warning(
                f"Audio duration ({duration:.2f}s) exceeds max limit ({max_duration_sec}s). Truncating."
            )
            samples = samples[: int(max_duration_sec * curr_sr)]
            duration = max_duration_sec

        # Peak normalization (-1.0 to +1.0)
        if normalize_peak:
            max_val = np.max(np.abs(samples))
            if max_val > 1e-6:
                samples = samples / max_val * 0.95

        return AudioSignal(
            samples=samples.astype(np.float32),
            sample_rate=curr_sr,
            duration_sec=duration,
            channels=channels,
            source_path=source_path,
        )

    except AudioFileError:
        raise
    except Exception as e:
        raise AudioFileError(f"Failed to decode audio: {str(e)}") from e

