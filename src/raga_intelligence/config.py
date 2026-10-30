"""
Configuration settings for Indian Raga Intelligence.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# Canonical 10 Hindustani Raga Definitions
CANONICAL_RAGAS: Dict[str, Dict[str, str]] = {
    "Bhairav": {
        "thhat": "Bhairav",
        "arohana": "S r G M P d N S'",
        "avarohana": "S' N d P M G r S",
        "vadi": "d",
        "samvadi": "r",
        "time": "Dawn",
    },
    "Bhairavi": {
        "thhat": "Bhairavi",
        "arohana": "S r g M P d n S'",
        "avarohana": "S' n d P M g r S",
        "vadi": "M",
        "samvadi": "S",
        "time": "Morning/Concluding",
    },
    "Yaman": {
        "thhat": "Kalyan",
        "arohana": "'N R G M' D N S'",
        "avarohana": "S' N D P M' G R S",
        "vadi": "G",
        "samvadi": "N",
        "time": "Evening",
    },
    "Bhoopali": {
        "thhat": "Kalyan",
        "arohana": "S R G P D S'",
        "avarohana": "S' D P G R S",
        "vadi": "G",
        "samvadi": "D",
        "time": "First quarter of night",
    },
    "Deshkar": {
        "thhat": "Bilawal",
        "arohana": "S R G P D S'",
        "avarohana": "S' D P G R S",
        "vadi": "D",
        "samvadi": "G",
        "time": "Morning",
    },
    "Bageshree": {
        "thhat": "Kafi",
        "arohana": "S g M D n S'",
        "avarohana": "S' n D M g R S",
        "vadi": "M",
        "samvadi": "S",
        "time": "Late night",
    },
    "Malkauns": {
        "thhat": "Bhairavi",
        "arohana": "S g M d n S'",
        "avarohana": "S' n d M g S",
        "vadi": "M",
        "samvadi": "S",
        "time": "Midnight",
    },
    "Todi": {
        "thhat": "Todi",
        "arohana": "S r g M' P d N S'",
        "avarohana": "S' N d P M' g r S",
        "vadi": "d",
        "samvadi": "g",
        "time": "Late morning",
    },
    "Darbari Kanada": {
        "thhat": "Asavari",
        "arohana": "S R g M P d n S'",
        "avarohana": "S' d n P M g M R S",
        "vadi": "R",
        "samvadi": "P",
        "time": "Midnight",
    },
    "Kafi": {
        "thhat": "Kafi",
        "arohana": "S R g M P D n S'",
        "avarohana": "S' n D P M g R S",
        "vadi": "P",
        "samvadi": "S",
        "time": "Late evening",
    },
}

RAGA_LIST: List[str] = list(CANONICAL_RAGAS.keys())


class AudioConfig(BaseModel):
    """Audio loading, resampling, and windowing settings."""
    sample_rate: int = 24000  # MERT native sample rate
    mono: bool = True
    segment_duration_sec: float = 5.0
    segment_hop_sec: float = 2.5
    min_duration_sec: float = 1.0
    max_duration_sec: float = 600.0  # 10 minutes max for single upload
    normalize_loudness: bool = True
    target_lufs: float = -20.0


class FeatureConfig(BaseModel):
    """Pitch and spectral feature extraction parameters."""
    n_fft: int = 2048
    hop_length: int = 512
    n_chroma: int = 12
    n_mels: int = 128
    fmin: float = 55.0  # ~A1
    fmax: float = 2000.0  # upper vocal range
    pitch_confidence_threshold: float = 0.6
    tonic_default_hz: float = 138.59  # C#3 (common Hindustani male tonic), or 220.0 (female tonic A3)


class ModelConfig(BaseModel):
    """Machine learning model settings."""
    mert_model_name: str = "m-a-p/MERT-v1-95M"
    mert_output_dim: int = 768
    num_classes: int = len(RAGA_LIST)
    dropout: float = 0.3
    hidden_dim: int = 256
    pooling_type: Literal["mean", "max", "mean_max_concat"] = "mean_max_concat"
    freeze_encoder: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    num_epochs: int = 15


class Settings(BaseSettings):
    """Global application settings with environment variable overrides."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = Field(default="development", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    debug: bool = Field(default=False, alias="DEBUG")

    # Hardware & Compute
    device_preference: str = Field(default="auto", alias="DEVICE")  # auto, cuda, cpu
    cuda_visible_devices: Optional[str] = Field(default="0", alias="CUDA_VISIBLE_DEVICES")

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    artifacts_dir: Path = Field(default=Path("./artifacts"), alias="ARTIFACTS_DIR")
    configs_dir: Path = Field(default=Path("./configs"), alias="CONFIGS_DIR")

    # Domain Subconfigs
    audio: AudioConfig = Field(default_factory=AudioConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    def get_torch_device(self) -> torch.device:
        """
        Determines the safe execution device with verified tensor execution.
        Falls back to CPU if CUDA is requested but incompatible (e.g., sm_120 without compiled kernels).
        """
        pref = self.device_preference.lower()
        if pref == "cpu":
            return torch.device("cpu")

        if torch.cuda.is_available() and pref in ("auto", "cuda"):
            try:
                test_dev = torch.device("cuda:0")
                x = torch.ones((10, 10), device=test_dev, dtype=torch.float32)
                _ = torch.matmul(x, x)
                return test_dev
            except Exception as e:
                logger.warning(
                    f"CUDA available but tensor execution failed ({e}). Falling back to CPU."
                )
                return torch.device("cpu")

        return torch.device("cpu")


# Global singleton instance
settings = Settings()
