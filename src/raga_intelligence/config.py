"""
Configuration settings for Indian Raga Intelligence.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from pydantic import BaseModel, Field, field_validator, model_validator
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
    pitch_tracker: Literal["pyin", "yin"] = "pyin"
    pitch_confidence_threshold: float = 0.4
    min_energy_threshold: float = 1e-4
    tonic_default_hz: float = 138.59  # C#3 (common Hindustani male tonic), or 220.0 (female tonic A3)
    include_melodic_features: bool = False  # Set True to append 40-dim sequential features


class ModelConfig(BaseModel):
    """Machine learning model settings."""
    mert_model_name: str = "m-a-p/MERT-v1-95M"
    mert_model_revision: str = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"  # Pinned immutable commit hash
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
    min_tonic_hz: float = Field(default=50.0, alias="MIN_TONIC_HZ", description="Minimum physically plausible singing tonic frequency in Hz")
    max_tonic_hz: float = Field(default=1000.0, alias="MAX_TONIC_HZ", description="Maximum physically plausible singing tonic frequency in Hz")

    # CORS Configuration
    cors_allowed_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        alias="CORS_ALLOWED_ORIGINS",
    )
    cors_allow_credentials: bool = Field(default=False, alias="CORS_ALLOW_CREDENTIALS")
    cors_allowed_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "OPTIONS"],
        alias="CORS_ALLOWED_METHODS",
    )
    cors_allowed_headers: List[str] = Field(
        default_factory=lambda: ["Content-Type", "Accept", "Authorization", "X-Request-ID"],
        alias="CORS_ALLOWED_HEADERS",
    )

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def validate_cors_security(self) -> Settings:
        if self.cors_allow_credentials and "*" in self.cors_allowed_origins:
            raise ValueError(
                "CORS security violation: allow_credentials cannot be True when wildcard ('*') origin is configured."
            )
        if self.env.lower() == "production":
            if not self.cors_allowed_origins:
                raise ValueError("Production configuration requires non-empty CORS_ALLOWED_ORIGINS.")
            if "*" in self.cors_allowed_origins:
                raise ValueError("Production CORS violation: wildcard origin ('*') is strictly forbidden in production.")
        return self

    def get_torch_device(self) -> torch.device:
        """
        Determines the safe execution device with verified tensor execution.
        Falls back to CPU if CUDA is requested but incompatible (e.g., sm_120 without compiled kernels).
        """
        device, _ = verify_execution_device(self.device_preference)
        return device


def verify_execution_device(device_preference: str = "auto") -> tuple[torch.device, dict[str, Any]]:
    """
    Formally verifies compute device availability and actual tensor execution.
    Logs hardware capabilities, architecture, and whether explicit CPU fallback occurred.
    """
    pref = device_preference.lower()
    diag: dict[str, Any] = {
        "requested_preference": pref,
        "cuda_is_available": torch.cuda.is_available(),
        "device_count": 0,
        "device_name": None,
        "compute_capability": None,
        "tensor_execution_verified": False,
        "fallback_to_cpu": False,
        "selected_device": "cpu",
        "diagnostic_reason": "",
    }

    if pref == "cpu":
        diag["diagnostic_reason"] = "Explicit CPU preference configured by operator."
        return torch.device("cpu"), diag

    if torch.cuda.is_available() and pref in ("auto", "cuda"):
        diag["device_count"] = torch.cuda.device_count()
        diag["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        diag["compute_capability"] = f"sm_{cap[0]}{cap[1]}"

        try:
            test_dev = torch.device("cuda:0")
            # Verify actual tensor allocation and matrix kernel execution
            x = torch.ones((32, 32), device=test_dev, dtype=torch.float32)
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            _ = float(y[0, 0].item())
            diag["tensor_execution_verified"] = True
            diag["selected_device"] = "cuda:0"
            diag["diagnostic_reason"] = (
                f"Verified CUDA GPU execution on {diag['device_name']} ({diag['compute_capability']})."
            )
            logger.info(diag["diagnostic_reason"])
            return test_dev, diag

        except Exception as e:
            diag["fallback_to_cpu"] = True
            diag["selected_device"] = "cpu"
            diag["diagnostic_reason"] = (
                f"CUDA GPU detected ({diag['device_name']} {diag['compute_capability']}), "
                f"but kernel execution failed ({e}). Explicitly falling back to CPU."
            )
            logger.warning(diag["diagnostic_reason"])
            return torch.device("cpu"), diag

    diag["selected_device"] = "cpu"
    diag["diagnostic_reason"] = "No CUDA GPU detected or supported by host environment."
    return torch.device("cpu"), diag


# Global singleton instance
settings = Settings()
