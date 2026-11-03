from __future__ import annotations

from dataclasses import replace

import pytest

from raga_intelligence.evaluation import tracker
from raga_intelligence.evaluation.tracker import ExperimentRecord, validate_evaluation_records


def make_record(model: str, seed: int) -> ExperimentRecord:
    return ExperimentRecord(
        run_id=f"{model}-{seed}",
        experiment_name=model,
        timestamp="2026-01-01T00:00:00+00:00",
        code_version="a" * 40,
        seed=seed,
        model_name=model,
        model_architecture="unit test model",
        hyperparameters={},
        feature_config={},
        dataset_name="test",
        dataset_split_info={},
        metrics={},
        system_environment={
            "os": "test", "python_version": "3.12", "torch_version": "2.x",
            "transformers_version": "4.x", "librosa_version": "0.x",
            "sklearn_version": "1.x", "device_name": "CPU",
        },
        dataset_fingerprint="b" * 64,
        evaluation_split_fingerprint="c" * 64,
        lockfile_hash="d" * 64,
        mert_model_revision="e" * 40,
        working_tree_dirty=False,
        working_tree_fingerprint="f" * 64,
        evaluation_id="eval-1",
    )


def test_clean_source_tree_accepts_commit(monkeypatch):
    monkeypatch.setattr(tracker, "get_working_tree_fingerprint", lambda: (False, "clean"))
    monkeypatch.setattr(tracker, "get_git_commit_hash", lambda: "a" * 40)
    assert tracker.assert_clean_source_tree() == ("a" * 40, True)


def test_dirty_source_tree_rejected_and_development_override_is_noncanonical(monkeypatch):
    monkeypatch.setattr(tracker, "get_working_tree_fingerprint", lambda: (True, "dirty"))
    monkeypatch.setattr(tracker, "get_git_commit_hash", lambda: "a" * 40)
    with pytest.raises(RuntimeError, match="requires a clean source working tree"):
        tracker.assert_clean_source_tree()
    assert tracker.assert_clean_source_tree(allow_dirty_development=True) == ("a" * 40, False)


def test_evaluation_records_reject_mixed_commit_sha_dataset_and_report_inputs():
    expected = {("model", 42)}
    clean = make_record("model", 42)
    assert validate_evaluation_records([clean], "eval-1", expected) == [clean]

    with pytest.raises(ValueError, match="inconsistent code_version"):
        validate_evaluation_records(
            [clean, replace(clean, seed=43, code_version="9" * 40)],
            "eval-1",
            {("model", 42), ("model", 43)},
        )

    with pytest.raises(ValueError, match="inconsistent dataset_fingerprint"):
        validate_evaluation_records(
            [clean, replace(clean, seed=43, dataset_fingerprint="8" * 64)],
            "eval-1",
            {("model", 42), ("model", 43)},
        )

    with pytest.raises(ValueError, match="incomplete or contain unexpected"):
        validate_evaluation_records(
            [clean, replace(clean, seed=43, evaluation_id="other-eval")],
            "eval-1",
            {("model", 42), ("model", 43)},
        )


def test_canonical_evaluation_rejects_dirty_experiment_records():
    dirty = replace(make_record("model", 42), working_tree_dirty=True)
    with pytest.raises(ValueError, match="dirty-working-tree"):
        validate_evaluation_records([dirty], "eval-1", {("model", 42)}, require_clean=True)
    accepted_as_development = validate_evaluation_records(
        [dirty], "eval-1", {("model", 42)}, require_clean=False
    )
    assert accepted_as_development[0].working_tree_dirty is True
