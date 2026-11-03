"""
Integration tests for FastAPI endpoints.
"""

from io import BytesIO
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from raga_intelligence.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def make_test_wav(duration: float = 2.0, sr: int = 24000) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    buf = BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    return buf.getvalue()


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "device" in data


def test_ready_endpoint(client: TestClient):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert "active_inference_backend" in data
    assert len(data["models_loaded"]) >= 2
    assert data["supported_ragas_count"] == 10


def test_models_endpoint(client: TestClient):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["supported_ragas"]) == 10
    assert "TonicRelativePCD" in data["available_encoders"]
    assert "AlignedChromaCQT" in data["available_encoders"]


def test_analyze_valid_audio(client: TestClient):
    wav_bytes = make_test_wav(duration=3.0)
    files = {"audio_file": ("test.wav", wav_bytes, "audio/wav")}
    response = client.post("/analyze", files=files, data={"tonic_hz": 220.0, "top_k": 3})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["ranked_matches"]) == 3
    assert data["duration_sec"] > 2.5
    assert "caveat_statement" in data


def test_analyze_invalid_format(client: TestClient):
    files = {"audio_file": ("test.txt", b"plain text", "text/plain")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_analyze_empty_file(client: TestClient):
    files = {"audio_file": ("test.wav", b"", "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_analyze_fake_extension_invalid_header(client: TestClient):
    """File has .wav extension but contains non-audio ASCII/binary garbage."""
    files = {"audio_file": ("exploit.wav", b"NOT_A_REAL_WAV_HEADER_JUST_GARBAGE_PAYLOAD", "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "File content does not match a supported audio format" in response.json()["detail"]


def test_analyze_oversized_file_rejected(client: TestClient):
    """File exceeding 50 MB limit must trigger HTTP 413 immediately."""
    # 51 MB dummy payload with valid WAV magic header at start
    oversized_size = 51 * 1024 * 1024
    header = b"RIFF" + (oversized_size - 8).to_bytes(4, "little") + b"WAVEfmt "
    oversized_data = header + b"\x00" * (oversized_size - len(header))

    files = {"audio_file": ("large.wav", oversized_data, "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 413
    assert "exceeds maximum allowed limit" in response.json()["detail"]


