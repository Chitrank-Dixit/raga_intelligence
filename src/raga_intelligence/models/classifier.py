"""
Neural classification head on top of frozen audio embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from raga_intelligence.config import RAGA_LIST

logger = logging.getLogger(__name__)


class MERTClassifierHead(nn.Module):
    """
    Multi-layer classification head for pooled MERT music embeddings.
    """

    def __init__(
        self,
        input_dim: int = 1536,  # 768 * 2 for mean_max_concat
        hidden_dim: int = 256,
        num_classes: int = len(RAGA_LIST),
        dropout: float = 0.3,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_names = class_names or RAGA_LIST

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw classification logits.
        Args:
            x: Tensor of shape (Batch, input_dim)
        Returns:
            logits: Tensor of shape (Batch, num_classes)
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes normalized class probabilities via softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> List[str]:
        """Predicts class names."""
        probs = self.predict_proba(x)
        pred_indices = torch.argmax(probs, dim=-1).cpu().tolist()
        return [self.class_names[idx] for idx in pred_indices]

    def save_checkpoint(self, path: Path | str, extra_meta: Optional[Dict] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = extra_meta or {}
        checkpoint = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "metadata": meta,
        }
        torch.save(checkpoint, p)
        logger.info(f"Saved MERT classification head checkpoint to {p}")

    @classmethod
    def load_checkpoint(cls, path: Path | str, device: Optional[torch.device] = None) -> MERTClassifierHead:
        p = Path(path)
        checkpoint = torch.load(p, map_location=device or torch.device("cpu"), weights_only=False)
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=checkpoint["num_classes"],
            class_names=checkpoint["class_names"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        if device:
            model.to(device)
        return model

