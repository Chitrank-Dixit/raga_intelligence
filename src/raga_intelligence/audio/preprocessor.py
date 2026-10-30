"""
High-level audio preprocessing pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO, List, Union

from raga_intelligence.audio.loader import AudioSignal, load_audio
from raga_intelligence.audio.segmentation import AudioSegment, segment_audio
from raga_intelligence.config import AudioConfig, settings

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Preprocesses raw audio files or streams into uniform analysis segments."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or settings.audio

    def process(
        self,
        source: Union[str, Path, BinaryIO, bytes],
        segment: bool = True,
    ) -> tuple[AudioSignal, List[AudioSegment]]:
        """
        Loads, resamples, normalizes audio, and segments it into windows.
        """
        signal = load_audio(
            source=source,
            target_sr=self.config.sample_rate,
            mono=self.config.mono,
            min_duration_sec=self.config.min_duration_sec,
            max_duration_sec=self.config.max_duration_sec,
            normalize_peak=self.config.normalize_loudness,
        )

        segments: List[AudioSegment] = []
        if segment:
            segments = segment_audio(
                signal=signal,
                segment_duration_sec=self.config.segment_duration_sec,
                hop_duration_sec=self.config.segment_hop_sec,
            )

        return signal, segments

