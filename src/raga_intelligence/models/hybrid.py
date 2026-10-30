"""
Hybrid late-fusion model combining explicit pitch features and learned MERT embeddings.
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


class HybridRagaClassifier(nn.Module):
    """
    Late-fusion neural network that synthesizes domain-specific musical features
    (PCD, aligned chroma, spectral envelope) with deep audio representations (MERT).
    """

    def __init__(
        self,
        musical_dim: int = 35,
        mert_dim: int = 1536,
        musical_proj_dim: int = 64,
        mert_proj_dim: int = 128,
        num_classes: int = len(RAGA_LIST),
        dropout: float = 0.25,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.musical_dim = musical_dim
        self.mert_dim = mert_dim
        self.num_classes = num_classes
        self.class_names = class_names or RAGA_LIST

        # Musical feature branch
        self.musical_branch = nn.Sequential(
            nn.BatchNorm1d(musical_dim),
            nn.Linear(musical_dim, musical_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # MERT representation branch
        self.mert_branch = nn.Sequential(
            nn.LayerNorm(mert_dim),
            nn.Linear(mert_dim, mert_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion head
        fused_dim = musical_proj_dim + mert_proj_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, musical_feat: torch.Tensor, mert_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            musical_feat: (Batch, musical_dim)
            mert_feat: (Batch, mert_dim)
        Returns:
            logits: (Batch, num_classes)
        """
        m_proj = self.musical_branch(musical_feat)
        e_proj = self.mert_branch(mert_feat)
        fused = torch.cat([m_proj, e_proj], dim=-1)
        return self.fusion_head(fused)

    def predict_proba(self, musical_feat: torch.Tensor, mert_feat: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(musical_feat, mert_feat)
            return F.softmax(logits, dim=-1)

    def save_checkpoint(self, path: Path | str, meta: Optional[Dict] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "musical_dim": self.musical_dim,
            "mert_dim": self.mert_dim,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "metadata": meta or {},
        }
        torch.save(checkpoint, p)
        logger.info(f"Saved Hybrid classifier checkpoint to {p}")

    @classmethod
    def load_checkpoint(cls, path: Path | str, device: Optional[torch.device] = None) -> HybridRagaClassifier:
        p = Path(path)
        checkpoint = torch.load(p, map_location=device or torch.device("cpu"), weights_only=False)
        model = cls(
            musical_dim=checkpoint["musical_dim"],
            mert_dim=checkpoint["mert_dim"],
            num_classes=checkpoint["num_classes"],
            class_names=checkpoint["class_names"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        if device:
            model.to(device)
        return model

