"""
FastAPI route definitions for health, models info, and audio analysis.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from raga_intelligence import __version__
from raga_intelligence.api.schemas import (
    AnalysisResponse,
    HealthResponse,
    MatchItemSchema,
    ModelsInfoResponse,
    RagaDetail,
    ReadyResponse,
)
from raga_intelligence.audio.loader import AudioFileError
from raga_intelligence.config import CANONICAL_RAGAS, RAGA_LIST, settings
from raga_intelligence.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Global pipeline instance initialized on app startup
pipeline: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    global pipeline
    if pipeline is None:
        pipeline = InferencePipeline()
    return pipeline


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    device = str(settings.get_torch_device())
    return HealthResponse(
        status="healthy",
        version=__version__,
        environment=settings.env,
        device=device,
        gpu_available="cuda" in device,
    )


@router.get("/ready", response_model=ReadyResponse, tags=["System"])
async def readiness_check() -> ReadyResponse:
    pipe = get_pipeline()
    models = ["MelodicSimilarityEngine", "BaselineFeatureExtractor"]
    if pipe.classifier is not None:
        models.append(f"BaselineRagaClassifier ({pipe.classifier.classifier_type})")

    return ReadyResponse(
        ready=True,
        models_loaded=models,
        supported_ragas_count=len(RAGA_LIST),
    )


@router.get("/models", response_model=ModelsInfoResponse, tags=["Models"])
async def get_models_info() -> ModelsInfoResponse:
    ragas_details = [
        RagaDetail(
            name=name,
            thhat=info["thhat"],
            vadi=info["vadi"],
            samvadi=info["samvadi"],
            time=info["time"],
            arohana=info["arohana"],
            avarohana=info["avarohana"],
        )
        for name, info in CANONICAL_RAGAS.items()
    ]

    return ModelsInfoResponse(
        available_encoders=["m-a-p/MERT-v1-95M", "TonicRelativePCD", "AlignedChromaCQT"],
        supported_ragas=ragas_details,
        default_sample_rate=settings.audio.sample_rate,
        segment_duration_sec=settings.audio.segment_duration_sec,
    )


@router.post("/analyze", response_model=AnalysisResponse, tags=["Inference"])
async def analyze_audio_endpoint(
    audio_file: UploadFile = File(..., description="Audio recording (WAV, MP3, FLAC, OGG)"),
    tonic_hz: Optional[float] = Form(None, description="Optional known tonic (Sa) in Hz"),
    top_k: int = Form(5, ge=1, le=10, description="Number of candidate ragas to return"),
) -> AnalysisResponse:
    # 1. Validate file extension / mime
    filename = audio_file.filename or "uploaded_audio"
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{file_ext}'. Allowed formats: {', '.join(allowed_extensions)}",
        )

    # 2. Read bytes with size protection (50 MB max)
    max_bytes = 50 * 1024 * 1024
    content = await audio_file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum allowed limit (50 MB).",
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 3. Execute inference
    pipe = get_pipeline()
    try:
        result = pipe.analyze_audio(
            audio_source=content,
            ground_truth_tonic_hz=tonic_hz,
            top_k=top_k,
        )
    except AudioFileError as afe:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio processing error: {str(afe)}",
        )
    except Exception as e:
        logger.exception("Unexpected error during inference.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(e)}",
        )

    # 4. Map to response schema
    matches_schema = [
        MatchItemSchema(
            raga_name=m.raga_name,
            similarity_score=m.similarity_score,
            confidence_level=m.confidence_level,
            thhat=m.thhat,
            vadi=m.vadi,
            samvadi=m.samvadi,
            matching_swaras=m.matching_swaras,
            foreign_swaras=m.foreign_swaras,
            evidence_segments=m.evidence_segments,
            is_strict_classification=m.is_strict_classification,
            notes=m.notes,
        )
        for m in result.ranked_matches
    ]

    return AnalysisResponse(
        status="success",
        duration_sec=result.duration_sec,
        estimated_tonic_hz=result.estimated_tonic_hz,
        voiced_ratio=result.voiced_ratio,
        detected_dominant_swaras=result.detected_dominant_swaras,
        ranked_matches=matches_schema,
        strict_classification_winner=result.strict_classification_winner,
        strict_classification_confidence=result.strict_classification_confidence,
        abstention_triggered=result.abstention_triggered,
        abstention_reason=result.abstention_reason,
        caveat_statement=result.caveat_statement,
    )

