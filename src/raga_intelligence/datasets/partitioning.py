"""
Group-aware data partitioning, performer/performance isolation, and multi-seed split generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest

logger = logging.getLogger(__name__)


@dataclass
class GroupingDiagnostic:
    """Diagnostic audit of group distribution across ragas and partition feasibility."""
    group_col: str
    total_items: int
    total_groups: int
    raga_group_counts: Dict[str, int]
    ragas_with_insufficient_groups: List[str]
    is_fully_partitionable: bool
    limitations: List[str] = field(default_factory=list)


class GroupAwarePartitioner:
    """
    Partitions dataset manifests ensuring zero data leakage across recording,
    performance, and performer boundaries.
    """

    @staticmethod
    def audit_grouping(
        manifest: DatasetManifest,
        group_col: Literal["performance_id", "performer"] = "performance_id",
        min_groups_required: int = 2,
    ) -> GroupingDiagnostic:
        """
        Audits whether every raga possesses sufficient distinct groups to support
        disjoint train/test isolation without leaking groups or dropping classes.
        """
        ragas = sorted(list({it.raga for it in manifest.items}))
        raga_group_counts: Dict[str, int] = {}
        all_groups = set()
        insufficient_ragas: List[str] = []
        limitations: List[str] = []

        for raga in ragas:
            raga_items = [it for it in manifest.items if it.raga == raga]
            groups = set()
            for it in raga_items:
                g_val = getattr(it, group_col, None) or it.recording_id
                groups.add(g_val)
            raga_group_counts[raga] = len(groups)
            all_groups.update(groups)

            if len(groups) < min_groups_required:
                insufficient_ragas.append(raga)
                limitations.append(
                    f"Raga '{raga}' contains only {len(groups)} distinct {group_col}(s). "
                    f"A strictly disjoint train/test partition requires >= {min_groups_required} groups. "
                    f"Performer/performance-independent evaluation cannot evaluate '{raga}' on unseen "
                    f"performers without dropping it from training or omitting it from testing."
                )

        is_partitionable = len(insufficient_ragas) == 0
        return GroupingDiagnostic(
            group_col=group_col,
            total_items=len(manifest.items),
            total_groups=len(all_groups),
            raga_group_counts=raga_group_counts,
            ragas_with_insufficient_groups=insufficient_ragas,
            is_fully_partitionable=is_partitionable,
            limitations=limitations,
        )

    @classmethod
    def create_group_aware_split(
        cls,
        manifest: DatasetManifest,
        group_col: Literal["performance_id", "performer"] = "performance_id",
        test_ratio: float = 0.20,
        val_ratio: float = 0.0,
        seed: int = 42,
    ) -> Tuple[DatasetManifest, GroupingDiagnostic]:
        """
        Creates a new DatasetManifest with deterministic group-aware train/val/test assignments.
        
        Guarantees that all audio clips from the same group (performer or performance) are assigned
        to exactly one split. If a raga has only 1 group, it is retained entirely in 'train' with an
        explicit logged limitation to prevent test leakage.
        """
        diagnostic = cls.audit_grouping(manifest, group_col=group_col, min_groups_required=2)
        rng = np.random.default_rng(seed)

        group_split_assignment: Dict[str, Literal["train", "val", "test"]] = {}
        ragas = sorted(list({it.raga for it in manifest.items}))

        for raga in ragas:
            raga_items = [it for it in manifest.items if it.raga == raga]
            # Map each unique group to its items
            raga_groups = sorted(list({getattr(it, group_col, None) or it.recording_id for it in raga_items}))
            n_groups = len(raga_groups)

            if n_groups < 2:
                # Cannot split safely into disjoint train and test without leaking
                for g in raga_groups:
                    group_split_assignment[g] = "train"
                continue

            # Deterministic permutation for this raga and seed
            shuffled_groups = [raga_groups[i] for i in rng.permutation(n_groups)]

            n_test = max(1, int(round(n_groups * test_ratio)))
            n_val = int(round(n_groups * val_ratio)) if val_ratio > 0 else 0

            # Ensure at least 1 group remains for training
            if n_test + n_val >= n_groups:
                if n_val > 0:
                    n_val = max(0, n_groups - n_test - 1)
                else:
                    n_test = max(1, n_groups - 1)

            test_groups = set(shuffled_groups[:n_test])
            val_groups = set(shuffled_groups[n_test : n_test + n_val])
            train_groups = set(shuffled_groups[n_test + n_val :])

            for g in test_groups:
                group_split_assignment[g] = "test"
            for g in val_groups:
                group_split_assignment[g] = "val"
            for g in train_groups:
                group_split_assignment[g] = "train"

        # Construct new manifest with updated split
        new_items: List[DatasetItem] = []
        for it in manifest.items:
            g_val = getattr(it, group_col, None) or it.recording_id
            assigned_split = group_split_assignment.get(g_val, "train")
            updated_item = it.model_copy(update={"split": assigned_split})
            new_items.append(updated_item)

        new_manifest = DatasetManifest(items=new_items)
        leakage_check = new_manifest.validate_splits()
        if not leakage_check["valid_no_recording_leakage"]:
            raise ValueError(f"Recording leakage detected: {leakage_check['leaked_recordings']}")
        if group_col == "performance_id" and not leakage_check["valid_no_performance_leakage"]:
            raise ValueError(f"Performance leakage detected: {leakage_check['leaked_performances']}")
        if group_col == "performer" and not leakage_check["valid_no_performer_leakage"]:
            raise ValueError(f"Performer leakage detected: {leakage_check['leaked_performers']}")

        return new_manifest, diagnostic

    @classmethod
    def generate_multi_seed_splits(
        cls,
        manifest: DatasetManifest,
        seeds: Sequence[int] = (42, 123, 999),
        group_col: Literal["performance_id", "performer"] = "performance_id",
        test_ratio: float = 0.20,
        val_ratio: float = 0.0,
    ) -> Dict[int, DatasetManifest]:
        """Generates consistent, group-isolated dataset splits across multiple random seeds."""
        splits_by_seed: Dict[int, DatasetManifest] = {}
        for s in seeds:
            split_m, _ = cls.create_group_aware_split(
                manifest=manifest,
                group_col=group_col,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                seed=s,
            )
            splits_by_seed[s] = split_m
        return splits_by_seed

