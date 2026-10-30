"""
Raga reference prototypes derived from modal theory and empirical performance corpora.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from raga_intelligence.config import CANONICAL_RAGAS, RAGA_LIST
from raga_intelligence.features.swara import SWARA_NAMES, SWARA_SEMITONES


@dataclass
class RagaPrototype:
    """Canonical representation of a raga's melodic and interval structure."""
    raga_name: str
    thhat: str
    vadi: str
    samvadi: str
    allowed_swaras: List[str]
    swara_mask: np.ndarray       # 12-dim binary mask
    theoretical_pcd: np.ndarray  # 12-dim normalized theoretical probability distribution
    empirical_pcd: Optional[np.ndarray] = None  # Learned distribution from training corpus

    @property
    def effective_pcd(self) -> np.ndarray:
        """Returns empirical PCD if available, otherwise theoretical PCD."""
        if self.empirical_pcd is not None and np.sum(self.empirical_pcd) > 0:
            return self.empirical_pcd
        return self.theoretical_pcd


class RagaPrototypeRegistry:
    """Registry maintaining prototypes for all canonical ragas."""

    def __init__(self):
        self._prototypes: Dict[str, RagaPrototype] = {}
        self._initialize_canonical_prototypes()

    def _initialize_canonical_prototypes(self) -> None:
        for raga_name, info in CANONICAL_RAGAS.items():
            # Extract notes from arohana and avarohana
            notes = set()
            for token in info["arohana"].split() + info["avarohana"].split():
                clean = token.strip("'")
                if clean in SWARA_SEMITONES:
                    notes.add(clean)

            swara_mask = np.zeros(12, dtype=np.float32)
            for n in notes:
                swara_mask[SWARA_SEMITONES[n]] = 1.0

            # Compute theoretical pitch distribution:
            # Vadi gets highest weight (3.0), Samvadi second (2.0), Sa gets (2.0), others (1.0)
            weights = np.zeros(12, dtype=np.float32)
            for n in notes:
                idx = SWARA_SEMITONES[n]
                if n == info["vadi"]:
                    weights[idx] = 3.0
                elif n == info["samvadi"]:
                    weights[idx] = 2.0
                elif n == "S":
                    weights[idx] = 2.0
                else:
                    weights[idx] = 1.0

            total_weight = np.sum(weights)
            pcd = weights / total_weight if total_weight > 0 else swara_mask / 12.0

            self._prototypes[raga_name] = RagaPrototype(
                raga_name=raga_name,
                thhat=info["thhat"],
                vadi=info["vadi"],
                samvadi=info["samvadi"],
                allowed_swaras=sorted(list(notes), key=lambda x: SWARA_SEMITONES.get(x, 0)),
                swara_mask=swara_mask,
                theoretical_pcd=pcd,
            )

    def get(self, raga_name: str) -> Optional[RagaPrototype]:
        return self._prototypes.get(raga_name)

    def all_prototypes(self) -> List[RagaPrototype]:
        return list(self._prototypes.values())

    def update_empirical_prototypes(self, corpus_pcds: Dict[str, List[np.ndarray]]) -> None:
        """Updates empirical centroids using averaged PCDs from a corpus."""
        for raga_name, pcd_list in corpus_pcds.items():
            if raga_name in self._prototypes and len(pcd_list) > 0:
                mean_pcd = np.mean(np.vstack(pcd_list), axis=0).astype(np.float32)
                norm = np.sum(mean_pcd)
                if norm > 0:
                    self._prototypes[raga_name].empirical_pcd = mean_pcd / norm

