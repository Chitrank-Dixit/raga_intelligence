"""
Datasets, manifests, adapters, and group-aware partitioning for Indian classical and cross-genre music.
"""

from raga_intelligence.datasets.adapters import BaseDatasetAdapter, SyntheticDatasetAdapter
from raga_intelligence.datasets.hindustani_dataset import HindustaniDatasetAdapter
from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest
from raga_intelligence.datasets.partitioning import GroupAwarePartitioner, GroupingDiagnostic
from raga_intelligence.datasets.saraga import SaragaHindustaniAdapter

__all__ = [
    "DatasetItem",
    "DatasetManifest",
    "BaseDatasetAdapter",
    "SyntheticDatasetAdapter",
    "SaragaHindustaniAdapter",
    "HindustaniDatasetAdapter",
    "GroupAwarePartitioner",
    "GroupingDiagnostic",
]
