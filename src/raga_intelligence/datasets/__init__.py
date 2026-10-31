"""
Datasets, manifests, and adapters for Indian classical and cross-genre music.
"""

from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest
from raga_intelligence.datasets.adapters import BaseDatasetAdapter, SyntheticDatasetAdapter
from raga_intelligence.datasets.saraga import SaragaHindustaniAdapter
from raga_intelligence.datasets.hindustani_dataset import HindustaniDatasetAdapter

__all__ = [
    "DatasetItem",
    "DatasetManifest",
    "BaseDatasetAdapter",
    "SyntheticDatasetAdapter",
    "SaragaHindustaniAdapter",
    "HindustaniDatasetAdapter",
]

