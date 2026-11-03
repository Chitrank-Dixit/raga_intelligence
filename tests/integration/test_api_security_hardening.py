"""
Integration tests for API security hardening:
- Request correlation ID length and character boundaries
- Tonic frequency range enforcement
- Bounded file upload processing and 413 rejection
- Malformed audio header rejection
- Temporary file cleanup on failure paths
- Concurrent request execution
"""

from __future__ import annotations

import concurrent.futures
import io
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from raga_intelligence.api.app import app
from raga_intelligence.config import settings


@pytest.fixture
def client():
    return TestClient(app)


def make_test_wav(duration: float = 2.0, sr: int = 24000) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    return buf.getvalue()


def test_correlation_id_boundaries(client: TestClient):
    """Correlation IDs must be bounded to 1-64 alphanumeric/dash/dot/underscore."""
    # 1. Valid ID is preserved
    valid_id = "req_123-abc.v1"
    res1 = client.get("/health", headers={"X-Request-ID": valid_id})
    assert res1.status_code == 200
    assert res1.headers["X-Request-ID"] == valid_id

    # 2. Oversized ID (>64 chars) is replaced with UUID4
    oversized_id = "x" * 65
    res2 = client.get("/health", headers={"X-Request-ID": oversized_id})
    assert res2.status_code == 200
    returned_id2 = res2.headers["X-Request-ID"]
    assert returned_id2 != oversized_id
    assert len(returned_id2) == 36  # UUID4 format

    # 3. Malformed special characters (e.g. XSS payload) replaced with UUID4
    bad_id = "<script>alert('xss')</script>"
    res3 = client.get("/health", headers={"X-Request-ID": bad_id})
    assert res3.status_code == 200
    returned_id3 = res3.headers["X-Request-ID"]
    assert "<" not in returned_id3
    assert len(returned_id3) == 36

    # 4. Spaces and control characters replaced with UUID4
    spaced_id = "my request id"
    res4 = client.get("/health", headers={"X-Request-ID": spaced_id})
    assert res4.status_code == 200
    assert " " not in res4.headers["X-Request-ID"]


def test_tonic_frequency_range_validation(client: TestClient):
    """Tonic frequency must be finite and within [min_tonic_hz, max_tonic_hz]."""
    wav_bytes = make_test_wav(duration=2.0)
    files = {"audio_file": ("test.wav", wav_bytes, "audio/wav")}

    # 1. Valid tonic within range (50 - 1000 Hz)
    res_valid = client.post("/analyze", files=files, data={"tonic_hz": 140.0, "top_k": 3})
    assert res_valid.status_code == 200

    # 2. Below minimum safe range (< 50 Hz)
    files_low = {"audio_file": ("test.wav", wav_bytes, "audio/wav")}
    res_low = client.post("/analyze", files=files_low, data={"tonic_hz": 20.0, "top_k": 3})
    assert res_low.status_code == 400
    assert "outside the permitted safe range" in res_low.json()["detail"]

    # 3. Above maximum safe range (> 1000 Hz)
    files_high = {"audio_file": ("test.wav", wav_bytes, "audio/wav")}
    res_high = client.post("/analyze", files=files_high, data={"tonic_hz": 2500.0, "top_k": 3})
    assert res_high.status_code == 400
    assert "outside the permitted safe range" in res_high.json()["detail"]

    # 4. Negative tonic frequency
    files_neg = {"audio_file": ("test.wav", wav_bytes, "audio/wav")}
    res_neg = client.post("/analyze", files=files_neg, data={"tonic_hz": -100.0, "top_k": 3})
    assert res_neg.status_code == 400


def test_malformed_audio_rejected_400(client: TestClient):
    """Uploading plain text or random binary content without audio magic header must fail."""
    junk_data = b"This is plain text and definitely not a valid audio container file!"
    files = {"audio_file": ("corrupt.wav", junk_data, "audio/wav")}

    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "File content does not match a supported audio format" in response.json()["detail"]


def test_failure_path_temp_file_cleanup(client: TestClient):
    """Verify that aborted/failed requests do not leak temporary files on disk."""
    temp_dir = Path(tempfile.gettempdir())
    files_before = set(temp_dir.glob("tmp*.wav"))

    # Send invalid audio that fails header validation
    files = {"audio_file": ("fail_clean.wav", b"RIFF_fake_invalid_header_bytes", "audio/wav")}
    res = client.post("/analyze", files=files)
    assert res.status_code in (400, 422)

    files_after = set(temp_dir.glob("tmp*.wav"))
    new_files = files_after - files_before
    # No temporary files matching our upload pattern should remain
    assert len(new_files) == 0


def test_concurrent_requests(client: TestClient):
    """Verify thread-safety and concurrent request handling across endpoints."""
    wav_bytes = make_test_wav(duration=2.0)

    def send_health():
        return client.get("/health").status_code

    def send_analyze(idx: int):
        files = {"audio_file": (f"test_{idx}.wav", wav_bytes, "audio/wav")}
        return client.post("/analyze", files=files, data={"top_k": 2}).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        health_futures = [executor.submit(send_health) for _ in range(5)]
        analyze_futures = [executor.submit(send_analyze, i) for i in range(3)]

        for f in concurrent.futures.as_completed(health_futures):
            assert f.result() == 200

        for f in concurrent.futures.as_completed(analyze_futures):
            assert f.result() == 200

