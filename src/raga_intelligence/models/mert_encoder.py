"""
MERT (Music Explanation and Representation Transformer) encoder integration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel

from raga_intelligence.audio.segmentation import AudioSegment
from raga_intelligence.config import settings

logger = logging.getLogger(__name__)


class MERTFeatureExtractor(nn.Module):
    """
    Extracts high-level acoustic and musical representations using pretrained MERT.
    
    Model: m-a-p/MERT-v1-95M (or configured variant)
    Input: 24 kHz mono waveforms
    Output: Frame-level or temporally-pooled embeddings
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        pooling_type: Literal["mean", "max", "mean_max_concat"] = "mean_max_concat",
        device: Optional[torch.device] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.pooling_type = pooling_type
        self.device = device or settings.get_torch_device()
        self.freeze = freeze

        self.processor: Optional[AutoFeatureExtractor] = None
        self.model: Optional[AutoModel] = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Loads MERT processor and weights from Hugging Face or local cache."""
        if self._is_loaded:
            return

        logger.info(f"Loading MERT model '{self.model_name}' on device '{self.device}'...")
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model.to(self.device)

            if self.freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()

            self._is_loaded = True
            logger.info("MERT model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load MERT model from Hugging Face: {e}")
            raise

    def pool_frames(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pools frame-level representations across time dimension (dim=1).
        
        Args:
            hidden_states: (Batch, Time, HiddenDim)
            
        Returns:
            Pooled tensor: (Batch, HiddenDim) or (Batch, 2 * HiddenDim)
        """
        if self.pooling_type == "mean":
            return torch.mean(hidden_states, dim=1)
        elif self.pooling_type == "max":
            return torch.max(hidden_states, dim=1)[0]
        elif self.pooling_type == "mean_max_concat":
            mean_pooled = torch.mean(hidden_states, dim=1)
            max_pooled = torch.max(hidden_states, dim=1)[0]
            return torch.cat([mean_pooled, max_pooled], dim=-1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def extract_segment_embedding(
        self,
        audio_segment: AudioSegment | np.ndarray,
        sample_rate: int = 24000,
    ) -> torch.Tensor:
        """Extracts pooled embedding vector for a single segment."""
        if not self._is_loaded:
            self.load_model()

        if isinstance(audio_segment, AudioSegment):
            waveform = audio_segment.samples
            sample_rate = audio_segment.sample_rate
        else:
            waveform = audio_segment

        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            last_hidden = outputs.last_hidden_state  # (1, T, 768)
            pooled = self.pool_frames(last_hidden)  # (1, output_dim)

        return pooled.squeeze(0).cpu()

    def extract_batch(
        self,
        segments: List[AudioSegment | np.ndarray],
        sample_rate: int = 24000,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Extracts embeddings for a batch of segments."""
        if not self._is_loaded:
            self.load_model()

        embeddings: List[torch.Tensor] = []
        for i in range(0, len(segments), batch_size):
            chunk = segments[i : i + batch_size]
            waveforms = [
                s.samples if isinstance(s, AudioSegment) else s for s in chunk
            ]
            inputs = self.processor(
                waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values)
                pooled = self.pool_frames(outputs.last_hidden_state)
                embeddings.append(pooled.cpu())

        return torch.cat(embeddings, dim=0)

