"""
Pydantic API request and response schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    environment: str
    device: str
    gpu_available: bool


class ReadyResponse(BaseModel):
    ready: bool = True
    active_inference_backend: str = "melodic_similarity_prototypes"
    models_loaded: List[str]
    supported_ragas_count: int
    details: Optional[str] = None


class RagaDetail(BaseModel):
    name: str
    thhat: str
    vadi: str
    samvadi: str
    time: str
    arohana: str
    avarohana: str


class ModelsInfoResponse(BaseModel):
    available_encoders: List[str]
    supported_ragas: List[RagaDetail]
    default_sample_rate: int
    segment_duration_sec: float


class MatchItemSchema(BaseModel):
    raga_name: str
    similarity_score: float
    confidence_level: str
    resemblance_band: Optional[str] = None
    is_calibrated_probability: bool = False
    calibrated_probability: Optional[float] = None
    thhat: str
    vadi: str
    samvadi: str
    matching_swaras: List[str]
    foreign_swaras: List[str]
    evidence_segments: List[str]
    is_strict_classification: bool = False
    notes: Optional[str] = None


class AnalysisResponse(BaseModel):
    status: str = "success"
    duration_sec: float
    estimated_tonic_hz: float
    voiced_ratio: float
    detected_dominant_swaras: List[str]
    ranked_matches: List[MatchItemSchema]
    strict_classification_winner: Optional[str] = None
    strict_classification_confidence: Optional[float] = None
    abstention_triggered: bool = False
    abstention_reason: Optional[str] = None
    caveat_statement: str

