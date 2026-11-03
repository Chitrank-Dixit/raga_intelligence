"""
FastAPI route definitions for health, models info, and audio analysis.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status

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


class InferenceCapacityLimiter:
    """
    Manages concurrent expensive audio inference load with immediate saturation
    detection and predictable HTTP 503 backpressure.
    """
    def __init__(self, max_concurrent: int = 4, retry_after_sec: int = 2):
        self.max_concurrent = max_concurrent
        self.retry_after_sec = retry_after_sec
        self.active_count = 0

    def try_acquire(self) -> bool:
        if self.active_count >= self.max_concurrent:
            return False
        self.active_count += 1
        return True

    def release(self) -> None:
        if self.active_count > 0:
            self.active_count -= 1

    def configure(self, max_concurrent: int, retry_after_sec: Optional[int] = None) -> None:
        self.max_concurrent = max_concurrent
        if retry_after_sec is not None:
            self.retry_after_sec = retry_after_sec

    def reset(self) -> None:
        self.active_count = 0


inference_limiter = InferenceCapacityLimiter(
    max_concurrent=settings.max_concurrent_inferences,
    retry_after_sec=settings.inference_retry_after_sec,
)


def get_pipeline() -> InferencePipeline:
    global pipeline
    if pipeline is None:
        pipeline = InferencePipeline()
    return pipeline


def validate_audio_header(header: bytes) -> bool:
    """
    Validates audio magic bytes / header signature to ensure file content matches
    supported audio container formats (WAV, MP3, FLAC, OGG, M4A).
    """
    if len(header) < 4:
        return False
    # WAV: "RIFF....WAVE"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
        return True
    # FLAC: "fLaC"
    if header.startswith(b"fLaC"):
        return True
    # OGG: "OggS"
    if header.startswith(b"OggS"):
        return True
    # MP3 ID3 tag header: "ID3"
    if header.startswith(b"ID3"):
        return True
    # MP3 MPEG audio frame sync: 11 bits set (0xFF followed by 0b111x_xxxx)
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True
    # M4A / MP4: "ftyp" at offset 4
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return True
    return False


def sanitize_error_detail(msg: str, temp_file_path: Optional[Path] = None) -> str:
    """Strips internal filesystem paths from error messages before sending to client."""
    if temp_file_path:
        msg = msg.replace(str(temp_file_path), "<uploaded_audio>")
        msg = msg.replace(str(temp_file_path.resolve()), "<uploaded_audio>")
        msg = msg.replace(temp_file_path.name, "<uploaded_audio>")
    # Remove generic absolute paths (Windows C:\... or Unix /...)
    msg = re.sub(r"[a-zA-Z]:\\[^\s:;,]+", "<path>", msg)
    msg = re.sub(r"/(?:[a-zA-Z0-9_.-]+/)+[a-zA-Z0-9_.-]+", "<path>", msg)
    return msg


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
async def readiness_check(response: Response) -> ReadyResponse:
    pipe = get_pipeline()
    is_ready, active_backend, models_loaded = pipe.is_ready()

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            ready=False,
            active_inference_backend=active_backend,
            models_loaded=models_loaded,
            supported_ragas_count=len(RAGA_LIST),
            details="Pipeline unready: required inference models not loaded",
        )

    return ReadyResponse(
        ready=True,
        active_inference_backend=active_backend,
        models_loaded=models_loaded,
        supported_ragas_count=len(RAGA_LIST),
        details="Inference pipeline initialized and ready for requests",
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

    pipe = get_pipeline()
    active_encoders = ["TonicRelativePCD", "AlignedChromaCQT"]
    if hasattr(pipe, "mert_extractor") and getattr(pipe, "mert_extractor") is not None:
        active_encoders.append("m-a-p/MERT-v1-95M")

    return ModelsInfoResponse(
        available_encoders=active_encoders,
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
    # 1. Enforce concurrency limits and predictable backpressure
    if not inference_limiter.try_acquire():
        logger.warning(
            f"Inference capacity saturated ({inference_limiter.active_count}/{inference_limiter.max_concurrent}). "
            f"Rejecting request with HTTP 503 and Retry-After: {inference_limiter.retry_after_sec}."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server inference capacity is currently saturated. Please retry shortly.",
            headers={"Retry-After": str(inference_limiter.retry_after_sec)},
        )

    capacity_acquired = True
    temp_file: Optional[tempfile._TemporaryFileWrapper] = None
    temp_path: Optional[Path] = None

    try:
        # 2. Validate parameters & file extension / mime
        if tonic_hz is not None:
            import math
            if math.isnan(tonic_hz) or math.isinf(tonic_hz) or tonic_hz < settings.min_tonic_hz or tonic_hz > settings.max_tonic_hz:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Tonic frequency ({tonic_hz} Hz) is invalid or outside the permitted safe range "
                        f"[{settings.min_tonic_hz}, {settings.max_tonic_hz}] Hz."
                    ),
                )

        filename = audio_file.filename or "uploaded_audio"
        allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format '{file_ext}'. Allowed formats: {', '.join(allowed_extensions)}",
            )

        # 3. Bounded streaming read into temporary file with 50 MB limit & format verification
        max_bytes = 50 * 1024 * 1024  # 50 MB
        chunk_size = 64 * 1024  # 64 KB
        total_bytes = 0
        header_bytes = bytearray()
        header_validated = False

        temp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
        temp_path = Path(temp_file.name)

        try:
            while True:
                chunk = await audio_file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                        if hasattr(status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE")
                        else 413,
                        detail="File size exceeds maximum allowed limit (50 MB).",
                    )
                if not header_validated:
                    header_bytes.extend(chunk[: 32 - len(header_bytes)])
                    if len(header_bytes) >= 12:
                        if not validate_audio_header(bytes(header_bytes)):
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="File content does not match a supported audio format.",
                            )
                        header_validated = True

                temp_file.write(chunk)
        finally:
            if temp_file is not None and not temp_file.closed:
                try:
                    temp_file.close()
                except Exception:
                    pass

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )
        if not header_validated:
            if not validate_audio_header(bytes(header_bytes)):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File content does not match a supported audio format.",
                )

        # 4. Execute inference using temp_path
        pipe = get_pipeline()
        try:
            result = pipe.analyze_audio(
                audio_source=temp_path,
                ground_truth_tonic_hz=tonic_hz,
                top_k=top_k,
            )
        except AudioFileError as afe:
            clean_err = sanitize_error_detail(str(afe), temp_path)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Audio processing error: {clean_err}",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error during inference.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference failure encountered during analysis.",
            )

    except asyncio.CancelledError:
        logger.info("Inference request cancelled. Resources will be cleaned up.")
        raise
    finally:
        if temp_file is not None and not temp_file.closed:
            try:
                temp_file.close()
            except Exception:
                pass
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path.name}: {e}")
        if capacity_acquired:
            inference_limiter.release()

    # 4. Map to response schema
    matches_schema = [
        MatchItemSchema(
            raga_name=m.raga_name,
            similarity_score=m.similarity_score,
            confidence_level=m.confidence_level,
            resemblance_band=getattr(m, "resemblance_band", None),
            is_calibrated_probability=getattr(m, "is_calibrated_probability", False),
            calibrated_probability=getattr(m, "calibrated_probability", None),
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

