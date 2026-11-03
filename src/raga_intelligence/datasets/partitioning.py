"""
Globally consistent group-aware data partitioning, performer/performance isolation,
and multi-seed split generation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np

from raga_intelligence.datasets.manifest import DatasetItem, DatasetManifest

logger = logging.getLogger(__name__)


@dataclass
class GroupingDiagnostic:
    """Diagnostic audit of group distribution, multi-raga entanglement, and partition feasibility."""
    group_col: str
    total_items: int
    total_groups: int
    raga_group_counts: Dict[str, int]
    multi_raga_groups: Dict[str, List[str]]
    unknown_group_items_count: int
    untestable_ragas: List[str]
    is_fully_partitionable: bool
    is_isolation_verifiable: bool
    limitations: List[str] = field(default_factory=list)

    @property
    def ragas_with_insufficient_groups(self) -> List[str]:
        return self.untestable_ragas


class GroupAwarePartitioner:
    """
    Partitions dataset manifests ensuring zero data leakage across recording,
    performance, and performer boundaries.
    
    Guarantees:
    1. Every group ID is globally mapped to all its recordings.
    2. Each group is assigned to strictly ONE split (train, val, or test).
    3. No group crosses ragas and splits.
    4. Class representation is preserved where mathematically feasible.
    5. Untestable classes under strict isolation are explicitly identified and reported.
    """

    @classmethod
    def audit_grouping(
        cls,
        manifest: DatasetManifest,
        group_col: Literal["performance_id", "performer"] = "performance_id",
        min_groups_required: int = 2,
    ) -> GroupingDiagnostic:
        """
        Audits group distribution across ragas, detects multi-raga group entanglements,
        and determines whether strict disjoint partitioning is mathematically feasible.
        """
        group_to_items: Dict[str, List[DatasetItem]] = defaultdict(list)
        group_to_ragas: Dict[str, Set[str]] = defaultdict(set)
        raga_to_groups: Dict[str, Set[str]] = defaultdict(set)
        unknown_items_count = 0

        for it in manifest.items:
            raw_g = getattr(it, group_col, None)
            if raw_g in ("Unknown", "unknown", None, ""):
                unknown_items_count += 1
                g_key = f"__unknown_{it.recording_id}__"
            else:
                g_key = str(raw_g)

            group_to_items[g_key].append(it)
            group_to_ragas[g_key].add(it.raga)
            raga_to_groups[it.raga].add(g_key)

        all_ragas = sorted(list(raga_to_groups.keys()))
        raga_group_counts = {r: len(raga_to_groups[r]) for r in all_ragas}

        # Multi-raga groups
        multi_raga_groups: Dict[str, List[str]] = {}
        for g, ragas in group_to_ragas.items():
            if not g.startswith("__unknown_") and len(ragas) > 1:
                multi_raga_groups[g] = sorted(list(ragas))

        untestable_ragas: List[str] = []
        limitations: List[str] = []

        if unknown_items_count > 0:
            limitations.append(
                f"Dataset contains {unknown_items_count} recording(s) with missing or 'Unknown' {group_col} metadata. "
                f"Strict {group_col} isolation cannot be verified for these recordings."
            )

        for raga in all_ragas:
            valid_groups = [g for g in raga_to_groups[raga] if not g.startswith("__unknown_")]
            if len(valid_groups) < min_groups_required:
                untestable_ragas.append(raga)
                limitations.append(
                    f"Raga '{raga}' has only {len(valid_groups)} verifiable {group_col}(s). "
                    f"A strictly disjoint train/test partition requires >= {min_groups_required} distinct groups. "
                    f"'{raga}' cannot be evaluated on unseen {group_col}(s) without either leaking the group "
                    f"or omitting it from training."
                )

        if multi_raga_groups:
            limitations.append(
                f"{len(multi_raga_groups)} {group_col}(s) are associated with multiple ragas. "
                f"Assigning any multi-raga group to a split constrains all its associated ragas to that split."
            )

        is_partitionable = len(untestable_ragas) == 0
        is_verifiable = unknown_items_count == 0

        return GroupingDiagnostic(
            group_col=group_col,
            total_items=len(manifest.items),
            total_groups=len([g for g in group_to_items if not g.startswith("__unknown_")]),
            raga_group_counts=raga_group_counts,
            multi_raga_groups=multi_raga_groups,
            unknown_group_items_count=unknown_items_count,
            untestable_ragas=untestable_ragas,
            is_fully_partitionable=is_partitionable,
            is_isolation_verifiable=is_verifiable,
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
        Creates a new DatasetManifest with globally consistent group-aware split assignments.
        
        Guarantees:
        1. Global mapping from group ID to all recordings.
        2. Each group is assigned to exactly one split.
        3. No group crosses ragas or splits.
        4. Preserves representation in train and test where mathematically feasible.
        5. Classes that cannot be tested under strict isolation are logged in diagnostic.
        """
        diagnostic = cls.audit_grouping(manifest, group_col=group_col, min_groups_required=2)
        rng = np.random.default_rng(seed)

        # 1. Build global group-to-recordings and group-to-ragas mapping
        group_to_items: Dict[str, List[DatasetItem]] = defaultdict(list)
        group_to_ragas: Dict[str, Set[str]] = defaultdict(set)
        raga_to_groups: Dict[str, Set[str]] = defaultdict(set)
        unknown_groups: Set[str] = set()

        for it in manifest.items:
            raw_g = getattr(it, group_col, None)
            if raw_g in ("Unknown", "unknown", None, ""):
                g_key = f"__unknown_{it.recording_id}__"
                unknown_groups.add(g_key)
            else:
                g_key = str(raw_g)

            group_to_items[g_key].append(it)
            group_to_ragas[g_key].add(it.raga)
            raga_to_groups[it.raga].add(g_key)

        group_split_assignment: Dict[str, Literal["train", "val", "test"]] = {}

        # 2. Assign unverified/unknown groups to train (never leak into test)
        for ug in unknown_groups:
            group_split_assignment[ug] = "train"

        # 3. Identify and isolate untestable ragas (< 2 groups total)
        for raga in diagnostic.untestable_ragas:
            for g in raga_to_groups[raga]:
                if g not in group_split_assignment:
                    group_split_assignment[g] = "train"

        # 4. Partition remaining verifiable groups across ragas
        all_ragas = sorted(list(raga_to_groups.keys()))
        target_test_groups = max(1, int(round(len(group_to_items) * test_ratio)))

        # Phase A: Ensure every testable raga gets at least one test group
        for raga in all_ragas:
            if raga in diagnostic.untestable_ragas:
                continue

            raga_groups = sorted(list(raga_to_groups[raga]))
            # Check if this raga already has a group assigned to test via multi-raga group
            has_test = any(group_split_assignment.get(g) == "test" for g in raga_groups)
            if has_test:
                continue

            # Candidate groups must be unassigned
            unassigned = [g for g in raga_groups if g not in group_split_assignment]
            if not unassigned:
                continue

            # Permute deterministically for this raga and seed
            shuffled_candidates = [unassigned[i] for i in rng.permutation(len(unassigned))]
            chosen_test_group = None

            for cand in shuffled_candidates:
                # Check if assigning cand to test leaves other ragas of cand with at least 1 group for train
                cand_ragas = group_to_ragas[cand]
                safe = True
                for other_r in cand_ragas:
                    other_groups = raga_to_groups[other_r]
                    # Count how many groups of other_r could remain for train
                    remaining_for_train = [
                        og for og in other_groups
                        if og != cand and group_split_assignment.get(og) != "test"
                    ]
                    if len(remaining_for_train) == 0:
                        safe = False
                        break
                if safe:
                    chosen_test_group = cand
                    break

            # If no candidate was completely safe, fall back to first unassigned candidate
            if chosen_test_group is None:
                chosen_test_group = shuffled_candidates[0]

            group_split_assignment[chosen_test_group] = "test"

        # Phase B: Ensure every testable raga has at least one group in train
        for raga in all_ragas:
            if raga in diagnostic.untestable_ragas:
                continue
            raga_groups = sorted(list(raga_to_groups[raga]))
            has_train = any(group_split_assignment.get(g) == "train" for g in raga_groups)
            if not has_train:
                unassigned = [g for g in raga_groups if g not in group_split_assignment]
                if unassigned:
                    shuffled = [unassigned[i] for i in rng.permutation(len(unassigned))]
                    group_split_assignment[shuffled[0]] = "train"

        # Phase C: Allocate remaining unassigned groups according to ratio
        unassigned_all = sorted([g for g in group_to_items if g not in group_split_assignment])
        if unassigned_all:
            shuffled_all = [unassigned_all[i] for i in rng.permutation(len(unassigned_all))]
            current_test_count = sum(1 for s in group_split_assignment.values() if s == "test")
            needed_test = max(0, target_test_groups - current_test_count)

            for idx, g in enumerate(shuffled_all):
                if idx < needed_test:
                    group_split_assignment[g] = "test"
                elif val_ratio > 0 and idx < needed_test + int(round(len(unassigned_all) * val_ratio)):
                    group_split_assignment[g] = "val"
                else:
                    group_split_assignment[g] = "train"

        # 5. Build new manifest with strictly updated splits
        new_items: List[DatasetItem] = []
        for it in manifest.items:
            raw_g = getattr(it, group_col, None)
            g_key = f"__unknown_{it.recording_id}__" if raw_g in ("Unknown", "unknown", None, "") else str(raw_g)
            assigned_split = group_split_assignment.get(g_key, "train")
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

    @classmethod
    def partition_manifest(
        cls,
        manifest: DatasetManifest,
        group_col: Literal["performance_id", "performer"] = "performance_id",
        test_ratio: float = 0.20,
        val_ratio: float = 0.0,
        seed: int = 42,
    ) -> Tuple[DatasetManifest, DatasetManifest, DatasetManifest]:
        """Convenience method returning (train_manifest, val_manifest, test_manifest)."""
        full_m, _ = cls.create_group_aware_split(
            manifest=manifest,
            group_col=group_col,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_m = DatasetManifest(items=[it for it in full_m.items if it.split == "train"])
        val_m = DatasetManifest(items=[it for it in full_m.items if it.split == "val"])
        test_m = DatasetManifest(items=[it for it in full_m.items if it.split == "test"])
        return train_m, val_m, test_m

    partition = partition_manifest
