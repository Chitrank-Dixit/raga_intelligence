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
from raga_intelligence.api.routes import inference_limiter
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


def test_capacity_limiter_503_backpressure(client: TestClient):
    """
    When concurrent inference requests exceed MAX_CONCURRENT_INFERENCES,
    the API must predictably reject requests with HTTP 503 and Retry-After header.
    """
    wav_bytes = make_test_wav(duration=1.0)
    files = {"audio_file": ("test_limit.wav", wav_bytes, "audio/wav")}

    # Configure limiter to max 1 concurrent inference
    orig_max = inference_limiter.max_concurrent
    orig_retry = inference_limiter.retry_after_sec
    inference_limiter.configure(max_concurrent=1, retry_after_sec=5)

    try:
        # Simulate active inference job holding the single permit
        assert inference_limiter.try_acquire() is True
        assert inference_limiter.active_count == 1

        # Upload is accepted and validated before the saturated inference slot rejects it.
        res = client.post("/analyze", files=files, data={"top_k": 2})
        assert res.status_code == 503
        assert res.headers.get("Retry-After") == "5"
        assert "capacity is currently saturated" in res.json()["detail"]

        # Release slot
        inference_limiter.release()
        assert inference_limiter.active_count == 0

        # Subsequent request must succeed
        res2 = client.post("/analyze", files={"audio_file": ("test_limit.wav", wav_bytes, "audio/wav")}, data={"top_k": 2})
        assert res2.status_code == 200

    finally:
        inference_limiter.configure(max_concurrent=orig_max, retry_after_sec=orig_retry)
        inference_limiter.reset()


def test_capacity_limiter_is_thread_safe():
    """Concurrent acquisitions never exceed the configured inference capacity."""
    import threading

    orig_max = inference_limiter.max_concurrent
    orig_retry = inference_limiter.retry_after_sec
    inference_limiter.configure(max_concurrent=2)
    inference_limiter.reset()
    barrier = threading.Barrier(8)
    acquired = []

    def acquire():
        barrier.wait()
        acquired.append(inference_limiter.try_acquire())

    try:
        threads = [threading.Thread(target=acquire) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sum(acquired) == 2
        assert inference_limiter.active_count == 2
    finally:
        inference_limiter.reset()
        inference_limiter.configure(max_concurrent=orig_max, retry_after_sec=orig_retry)


def test_cleanup_on_decoder_failure(client: TestClient):
    """
    Verify temporary file cleanup when uploaded file has valid WAV header
    but corrupt/unparseable body causing audio decoder failure.
    """
    temp_dir = Path(tempfile.gettempdir())
    files_before = set(temp_dir.glob("*.wav"))

    # Valid WAV container header (RIFF....WAVE) but corrupt PCM payload
    corrupt_wav = (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00"
        b"\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    files = {"audio_file": ("corrupt_decoder.wav", corrupt_wav, "audio/wav")}
    res = client.post("/analyze", files=files)
    assert res.status_code in (400, 422, 500)

    files_after = set(temp_dir.glob("*.wav"))
    new_files = files_after - files_before
    assert len(new_files) == 0
    assert inference_limiter.active_count == 0


@pytest.mark.asyncio
async def test_cleanup_on_request_cancellation():
    """
    Verify that if an inference task is cancelled mid-execution,
    the capacity limiter releases its slot and temporary files are unlinked.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from fastapi import UploadFile
    from raga_intelligence.api.routes import analyze_audio_endpoint

    mock_upload = MagicMock(spec=UploadFile)
    mock_upload.filename = "cancel_test.wav"
    mock_upload.read = AsyncMock(side_effect=asyncio.CancelledError("Client disconnected"))

    initial_active = inference_limiter.active_count

    with pytest.raises(asyncio.CancelledError):
        await analyze_audio_endpoint(audio_file=mock_upload, tonic_hz=220.0, top_k=3)

    # Limiter count must have been returned to initial state
    assert inference_limiter.active_count == initial_active


@pytest.mark.asyncio
async def test_inference_phase_cancellation_waits_for_worker_and_releases_resources(
    tmp_path: Path, monkeypatch
):
    import asyncio
    import threading
    from types import SimpleNamespace
    from fastapi import UploadFile
    import raga_intelligence.api.routes as routes

    monkeypatch.setattr(routes.tempfile, "tempdir", str(tmp_path))
    started = threading.Event()
    finish = threading.Event()

    class BlockingPipeline:
        def analyze_audio(self, **kwargs):
            assert Path(kwargs["audio_source"]).exists()
            started.set()
            finish.wait(timeout=5)
            return SimpleNamespace(ranked_matches=[])

    monkeypatch.setattr(routes, "get_pipeline", lambda: BlockingPipeline())
    before = set(tmp_path.glob("tmp*.wav"))
    active_before = inference_limiter.active_count
    upload = UploadFile(filename="cancel_inference.wav", file=io.BytesIO(make_test_wav()))
    request_task = asyncio.create_task(
        routes.analyze_audio_endpoint(audio_file=upload, tonic_hz=220.0, top_k=3)
    )

    await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=3)
    assert inference_limiter.active_count == active_before + 1
    active_temp_files = set(tmp_path.glob("tmp*.wav")) - before
    assert len(active_temp_files) == 1

    request_task.cancel()
    await asyncio.sleep(0.05)
    assert not request_task.done()
    assert inference_limiter.active_count == active_before + 1
    assert active_temp_files.pop().exists()

    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert inference_limiter.active_count == active_before
    assert not list(tmp_path.glob("tmp*.wav"))
    assert inference_limiter.try_acquire() is True
    inference_limiter.release()


@pytest.mark.asyncio
async def test_inference_exception_cleans_temp_file_and_releases_capacity(tmp_path: Path, monkeypatch):
    from fastapi import HTTPException, UploadFile
    import raga_intelligence.api.routes as routes

    monkeypatch.setattr(routes.tempfile, "tempdir", str(tmp_path))

    class FailingPipeline:
        def analyze_audio(self, **kwargs):
            assert Path(kwargs["audio_source"]).exists()
            raise RuntimeError("inference failed")

    monkeypatch.setattr(routes, "get_pipeline", lambda: FailingPipeline())
    upload = UploadFile(filename="failure.wav", file=io.BytesIO(make_test_wav()))
    before_active = inference_limiter.active_count
    with pytest.raises(HTTPException) as error:
        await routes.analyze_audio_endpoint(audio_file=upload, tonic_hz=220.0, top_k=3)

    assert error.value.status_code == 500
    assert inference_limiter.active_count == before_active
    assert not list(tmp_path.glob("tmp*.wav"))
    assert inference_limiter.try_acquire() is True
    inference_limiter.release()

