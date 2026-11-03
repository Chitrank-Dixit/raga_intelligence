"""
Unit tests for explicit hardware device verification and CPU fallback diagnostics (Prompt 18).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from raga_intelligence.config import verify_execution_device


def test_verify_execution_device_explicit_cpu() -> None:
    device, diag = verify_execution_device("cpu")
    assert device == torch.device("cpu")
    assert diag["selected_device"] == "cpu"
    assert diag["fallback_to_cpu"] is False
    assert "Explicit CPU" in diag["diagnostic_reason"]


def test_verify_execution_device_fallback_on_kernel_failure() -> None:
    # Simulate a CUDA environment where torch.cuda.is_available() is True,
    # but torch.matmul fails with a CUDA kernel incompatibility error
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 5060 Ti"), \
         patch("torch.cuda.get_device_capability", return_value=(12, 0)), \
         patch("torch.matmul", side_effect=RuntimeError("CUDA error: no kernel image is available for execution")):

        device, diag = verify_execution_device("cuda")

        assert device == torch.device("cpu")
        assert diag["selected_device"] == "cpu"
        assert diag["fallback_to_cpu"] is True
        assert diag["tensor_execution_verified"] is False
        assert "Explicitly falling back to CPU" in diag["diagnostic_reason"]


def test_verify_execution_device_no_cuda_available() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        device, diag = verify_execution_device("auto")
        assert device == torch.device("cpu")
        assert diag["cuda_is_available"] is False
        assert diag["selected_device"] == "cpu"

