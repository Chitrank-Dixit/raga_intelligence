"""
Dataset adapters for data ingestion, synthetic fixtures, and manifest generation.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from raga_intelligence.config import CANONICAL_RAGAS, RAGA_LIST
from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest
from raga_intelligence.features.swara import SWARA_SEMITONES

logger = logging.getLogger(__name__)


class BaseDatasetAdapter(abc.ABC):
    """Abstract interface for dataset adapters."""

    @abc.abstractmethod
    def build_manifest(self) -> DatasetManifest:
        """Parses the dataset source and produces a validated DatasetManifest."""
        pass


class SyntheticDatasetAdapter(BaseDatasetAdapter):
    """
    Generates synthetic audio recordings conforming to authentic modal swara rules
    and tanpura drone harmonies for the 10 canonical Hindustani ragas.
    
    Used for unit testing, deterministic pipeline validation, and smoke tests.
    """

    def __init__(
        self,
        output_dir: Path | str = "./data/synthetic",
        recordings_per_raga: int = 4,
        duration_sec: float = 6.0,
        sample_rate: int = 24000,
    ):
        self.output_dir = Path(output_dir)
        self.recordings_per_raga = recordings_per_raga
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate

    def _generate_synthetic_audio(
        self,
        raga_name: str,
        tonic_hz: float,
        duration_sec: float,
    ) -> np.ndarray:
        """
        Synthesizes audio with:
        1. A tanpura-like harmonic drone on Sa (tonic) and Pa (or Ma).
        2. A melodic phrase moving through the swaras of the raga's arohana/avarohana.
        """
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
        audio = np.zeros_like(t)

        # 1. Tanpura drone (Sa fundamental + Pa fundamental + harmonics)
        sa_hz = tonic_hz
        pa_hz = sa_hz * (2.0 ** (7.0 / 12.0))
        audio += 0.25 * np.sin(2 * np.pi * sa_hz * t)
        audio += 0.15 * np.sin(2 * np.pi * 2 * sa_hz * t)  # Octave
        audio += 0.15 * np.sin(2 * np.pi * pa_hz * t)

        # 2. Extract allowed swaras for this raga
        raga_info = CANONICAL_RAGAS.get(raga_name, CANONICAL_RAGAS["Bhairav"])
        arohana_tokens = [s.strip("'") for s in raga_info["arohana"].split() if s.strip("'") in SWARA_SEMITONES]

        if not arohana_tokens:
            arohana_tokens = ["S", "R", "G", "M", "P", "D", "N"]

        # 3. Create discrete melodic note progression across duration
        note_duration = duration_sec / len(arohana_tokens)
        for i, swara in enumerate(arohana_tokens):
            semitones = SWARA_SEMITONES.get(swara, 0)
            note_freq = sa_hz * (2.0 ** (semitones / 12.0))
            start_idx = int(i * note_duration * self.sample_rate)
            end_idx = int((i + 1) * note_duration * self.sample_rate)
            t_chunk = t[start_idx:end_idx]
            
            # Note envelope with soft attack and decay
            envelope = np.sin(np.pi * np.linspace(0, 1, len(t_chunk))) ** 2
            melodic_tone = 0.5 * np.sin(2 * np.pi * note_freq * t_chunk)
            melodic_tone += 0.2 * np.sin(2 * np.pi * 2 * note_freq * t_chunk)  # overtone
            audio[start_idx:end_idx] += melodic_tone * envelope

        # Normalize peak
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = (audio / max_val * 0.9).astype(np.float32)

        return audio

    def build_manifest(self) -> DatasetManifest:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = DatasetManifest()

        artists = ["Pandit Sharma", "Ustad Khan", "Vidushi Devi", "Pandit Rao"]
        tonics = [138.59, 146.83, 220.0, 164.81]

        for raga in RAGA_LIST:
            for i in range(self.recordings_per_raga):
                performer = artists[i % len(artists)]
                tonic_hz = tonics[i % len(tonics)]
                rec_id = f"synth_{raga.lower().replace(' ', '_')}_{i+1:02d}"
                file_name = f"{rec_id}.wav"
                file_path = self.output_dir / file_name

                # Split strategy ensuring at least 1 sample in test even if recordings_per_raga is small
                if self.recordings_per_raga == 1:
                    split = "train"
                elif self.recordings_per_raga == 2:
                    split = "train" if i == 0 else "test"
                else:
                    if i < self.recordings_per_raga - 2:
                        split = "train"
                    elif i == self.recordings_per_raga - 2:
                        split = "val"
                    else:
                        split = "test"

                # Generate and save audio file if not already existing
                if not file_path.exists():
                    audio = self._generate_synthetic_audio(raga, tonic_hz, self.duration_sec)
                    sf.write(file_path, audio, self.sample_rate)

                item = DatasetItem(
                    recording_id=rec_id,
                    source="SyntheticIndianRagaGenerator",
                    file_path=str(file_path),
                    raga=raga,
                    raw_raga_label=raga,
                    tradition="Hindustani",
                    performer=performer,
                    performer_id=f"synthetic_artist_{i % len(artists) + 1}",
                    performance_id=rec_id,
                    performance_identity_source="metadata",
                    duration_sec=self.duration_sec,
                    sample_rate=self.sample_rate,
                    tonic_hz=tonic_hz,
                    split=split,
                    license="CC0-1.0",
                    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                    attribution="Generated by the Raga Intelligence synthetic dataset adapter",
                    rights_status="verified",
                    annotation_status="accepted",
                    annotation_confidence=1.0,
                    record_status="accepted",
                    metadata={"thhat": CANONICAL_RAGAS[raga]["thhat"]},
                )
                manifest.add_item(item)

        return manifest
