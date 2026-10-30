"""
Audio segmentation into fixed-length analysis windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from raga_intelligence.audio.loader import AudioSignal


@dataclass
class AudioSegment:
    """A sliced window of audio with timestamp metadata."""
    samples: np.ndarray
    sample_rate: int
    start_sec: float
    end_sec: float
    segment_idx: int
    parent_path: str | None = None

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def segment_audio(
    signal: AudioSignal,
    segment_duration_sec: float = 5.0,
    hop_duration_sec: float = 2.5,
    min_final_segment_sec: float = 2.0,
) -> List[AudioSegment]:
    """
    Slices an AudioSignal into uniform temporal windows with optional overlap.
    
    If the final residual segment is >= min_final_segment_sec, it is zero-padded
    to segment_duration_sec so all segments maintain equal length.
    """
    sr = signal.sample_rate
    seg_samples = int(segment_duration_sec * sr)
    hop_samples = int(hop_duration_sec * sr)
    total_samples = len(signal.samples)

    if total_samples <= seg_samples:
        # Pad short audio up to seg_samples
        padded = np.zeros(seg_samples, dtype=np.float32)
        padded[:total_samples] = signal.samples
        return [
            AudioSegment(
                samples=padded,
                sample_rate=sr,
                start_sec=0.0,
                end_sec=signal.duration_sec,
                segment_idx=0,
                parent_path=signal.source_path,
            )
        ]

    segments: List[AudioSegment] = []
    idx = 0
    start_idx = 0

    while start_idx < total_samples:
        end_idx = start_idx + seg_samples
        if end_idx <= total_samples:
            chunk = signal.samples[start_idx:end_idx]
            segments.append(
                AudioSegment(
                    samples=chunk,
                    sample_rate=sr,
                    start_sec=round(start_idx / sr, 3),
                    end_sec=round(end_idx / sr, 3),
                    segment_idx=idx,
                    parent_path=signal.source_path,
                )
            )
            idx += 1
            start_idx += hop_samples
        else:
            # Trailing segment
            remaining_samples = total_samples - start_idx
            remaining_sec = remaining_samples / sr
            if remaining_sec >= min_final_segment_sec:
                chunk = np.zeros(seg_samples, dtype=np.float32)
                chunk[:remaining_samples] = signal.samples[start_idx:]
                segments.append(
                    AudioSegment(
                        samples=chunk,
                        sample_rate=sr,
                        start_sec=round(start_idx / sr, 3),
                        end_sec=round(total_samples / sr, 3),
                        segment_idx=idx,
                        parent_path=signal.source_path,
                    )
                )
            break

    return segments

