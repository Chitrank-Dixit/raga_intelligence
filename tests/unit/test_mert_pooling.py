"""
Unit tests for MERT representation pooling with padding mask exclusion.
Verifies that padded frames do not corrupt embeddings across batches.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from raga_intelligence.models.mert_encoder import MERTFeatureExtractor


def test_pool_frames_excludes_padded_frames() -> None:
    extractor = MERTFeatureExtractor(pooling_type="mean")

    # Item 0 has 2 valid frames [1, 2], [3, 4] and 1 padded frame [1000, 1000]
    # Item 1 has 3 valid frames [2, 4], [4, 6], [6, 8]
    hidden = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [1000.0, 1000.0]],
        [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
    ], dtype=torch.float32)

    frame_mask = torch.tensor([
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=torch.int32)

    # 1. Mean pooling
    extractor.pooling_type = "mean"
    pooled_mean = extractor.pool_frames(hidden, frame_mask=frame_mask)
    # Item 0 mean should be (1+3)/2 = 2, (2+4)/2 = 3
    assert torch.allclose(pooled_mean[0], torch.tensor([2.0, 3.0]))
    # Item 1 mean should be (2+4+6)/3 = 4, (4+6+8)/3 = 6
    assert torch.allclose(pooled_mean[1], torch.tensor([4.0, 6.0]))

    # 2. Max pooling
    extractor.pooling_type = "max"
    pooled_max = extractor.pool_frames(hidden, frame_mask=frame_mask)
    # Item 0 max should ignore 1000 and return [3.0, 4.0]
    assert torch.allclose(pooled_max[0], torch.tensor([3.0, 4.0]))
    # Item 1 max should be [6.0, 8.0]
    assert torch.allclose(pooled_max[1], torch.tensor([6.0, 8.0]))

    # 3. Mean-Max Concat pooling
    extractor.pooling_type = "mean_max_concat"
    pooled_concat = extractor.pool_frames(hidden, frame_mask=frame_mask)
    assert torch.allclose(pooled_concat[0], torch.tensor([2.0, 3.0, 3.0, 4.0]))
    assert torch.allclose(pooled_concat[1], torch.tensor([4.0, 6.0, 6.0, 8.0]))


def test_pool_frames_without_mask_maintains_backward_compatibility() -> None:
    extractor = MERTFeatureExtractor(pooling_type="mean")
    hidden = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
    ], dtype=torch.float32)

    pooled = extractor.pool_frames(hidden)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 3.0]))


def test_batch_extraction_with_padding_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = MERTFeatureExtractor(pooling_type="mean")

    # Mock processor and model to test pipeline wiring without loading large model weights
    class MockModel(torch.nn.Module):
        def _get_feature_vector_attention_mask(self, length: int, mask: torch.Tensor) -> torch.Tensor:
            # Map length proportionally
            return (mask[:, :length] > 0).int()

        def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
            batch, length = input_values.shape
            # Hidden states with 4 frames
            hidden = torch.ones((batch, 4, 16), dtype=torch.float32)
            # Add large value to padded frame to detect if padding was included
            if attention_mask is not None:
                frame_mask = self._get_feature_vector_attention_mask(4, attention_mask)
                for b in range(batch):
                    for f in range(4):
                        if frame_mask[b, f] == 0:
                            hidden[b, f, :] = 100.0

            class MockOutput:
                last_hidden_state = hidden
            return MockOutput()

    class MockProcessor:
        def __call__(self, waveforms, sampling_rate, return_tensors, padding):
            # Batch of 2 items: item 0 length 2, item 1 length 4
            input_values = torch.zeros((len(waveforms), 4))
            attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.int32)
            return {"input_values": input_values, "attention_mask": attention_mask}

    extractor._is_loaded = True
    extractor.model = MockModel()
    extractor.processor = MockProcessor()

    w1 = np.ones(2)
    w2 = np.ones(4)
    embeddings = extractor.extract_batch([w1, w2], batch_size=2)

    assert embeddings.shape == (2, 16)
    # Item 0 has 2 valid frames of 1.0 and 2 padded frames of 100.0.
    # Masked mean should be 1.0, not (1+1+100+100)/4 = 50.5
    assert torch.allclose(embeddings[0], torch.ones(16))

