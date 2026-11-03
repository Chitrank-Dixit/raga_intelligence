"""
Lightweight JSON-based experiment tracking, environment introspection, and experiment logging.
Persists complete cryptographic and environment provenance for scientific reproducibility.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from raga_intelligence.config import settings
from raga_intelligence.features.schema import FeatureSchema

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Attempts to retrieve the current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    raise RuntimeError("Could not determine an exact Git commit SHA for this experiment")


def get_working_tree_fingerprint() -> tuple[bool, str]:
    """Fingerprint the uncommitted source tree alongside its base commit SHA."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3,
        check=True,
    ).stdout
    digest = hashlib.sha256()
    for line in sorted(status.splitlines()):
        path = line[3:]
        if not path.startswith(("scripts/", "src/", "tests/")) and path not in {
            "pyproject.toml", "uv.lock", "Containerfile", "compose.yaml"
        }:
            continue
        digest.update(line.encode("utf-8", errors="replace"))
        candidate = Path(path)
        if candidate.is_file():
            digest.update(candidate.read_bytes())
    return bool(status.strip()), digest.hexdigest()


def get_lockfile_hash() -> str:
    """Computes sha256 of uv.lock if present."""
    p = Path("uv.lock")
    if p.exists():
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            pass
    return "missing"


def get_environment_metadata() -> Dict[str, Any]:
    """Inspects and returns the hardware, OS, lockfile, and key ML library environment versions."""
    import librosa
    import numpy
    import scipy
    import sklearn
    import transformers

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None (CPU)"

    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchaudio_version": getattr(torch, "torchaudio", "N/A"),
        "transformers_version": transformers.__version__,
        "librosa_version": librosa.__version__,
        "sklearn_version": sklearn.__version__,
        "lockfile_hash": get_lockfile_hash(),
        "packages": {
            "torch": torch.__version__,
            "torchaudio": getattr(torch, "torchaudio", "N/A"),
            "transformers": transformers.__version__,
            "librosa": librosa.__version__,
            "sklearn": sklearn.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "cuda_available": cuda_available,
        "device_name": gpu_name,
    }


@dataclass
class ExperimentRecord:
    """Structured record of a single evaluated experiment run with complete provenance."""
    run_id: str
    experiment_name: str
    timestamp: str
    code_version: str
    seed: int
    model_name: str
    model_architecture: str
    hyperparameters: Dict[str, Any]
    feature_config: Dict[str, Any]
    dataset_name: str
    dataset_split_info: Dict[str, Any]
    metrics: Dict[str, Any]
    system_environment: Dict[str, Any]
    dataset_fingerprint: str = ""
    dataset_version: str = "1.0.0"
    recording_ids_per_split: Dict[str, List[str]] = field(default_factory=dict)
    evaluation_split_fingerprint: str = ""
    cache_fingerprints: Dict[str, Any] = field(default_factory=dict)
    feature_schema_version: str = "1.0.0"
    mert_model_revision: str = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
    lockfile_hash: str = ""
    notes: Optional[str] = None
    working_tree_dirty: bool = False
    working_tree_fingerprint: str = ""


class ExperimentTracker:
    """
    Manages filesystem-based JSON experiment logs in artifacts/experiments/.
    """

    def __init__(self, experiment_dir: Path | str = "./artifacts/experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        experiment_name: str,
        seed: int,
        model_name: str,
        model_architecture: str,
        hyperparameters: Dict[str, Any],
        feature_config: Dict[str, Any],
        metrics: Dict[str, Any],
        dataset_name: str = "HindustaniRaagSmall",
        dataset_split_info: Optional[Dict[str, Any]] = None,
        dataset_fingerprint: Optional[str] = None,
        dataset_version: str = "1.0.0",
        recording_ids_per_split: Optional[Dict[str, List[str]]] = None,
        evaluation_split_fingerprint: Optional[str] = None,
        cache_fingerprints: Optional[Dict[str, Any]] = None,
        feature_schema_version: Optional[str] = None,
        mert_model_revision: Optional[str] = None,
        notes: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> ExperimentRecord:
        """Logs a single experiment run to a JSON artifact on disk."""
        if not run_id:
            now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            short_uuid = str(uuid.uuid4())[:8]
            clean_name = experiment_name.lower().replace(" ", "_").replace("-", "_")
            run_id = f"{clean_name}_{now_str}_{short_uuid}"

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        code_version = get_git_commit_hash()
        working_tree_dirty, working_tree_fingerprint = get_working_tree_fingerprint()
        env_meta = get_environment_metadata()
        lock_hash = get_lockfile_hash()

        record = ExperimentRecord(
            run_id=run_id,
            experiment_name=experiment_name,
            timestamp=timestamp,
            code_version=code_version,
            working_tree_dirty=working_tree_dirty,
            working_tree_fingerprint=working_tree_fingerprint,
            seed=seed,
            model_name=model_name,
            model_architecture=model_architecture,
            hyperparameters=hyperparameters,
            feature_config=feature_config,
            dataset_name=dataset_name,
            dataset_split_info=dataset_split_info or {},
            metrics=metrics,
            system_environment=env_meta,
            dataset_fingerprint=dataset_fingerprint or "",
            dataset_version=dataset_version,
            recording_ids_per_split=recording_ids_per_split or {},
            evaluation_split_fingerprint=evaluation_split_fingerprint or "",
            cache_fingerprints=cache_fingerprints or {},
            feature_schema_version=feature_schema_version or FeatureSchema.SCHEMA_VERSION,
            mert_model_revision=mert_model_revision or settings.model.mert_model_revision,
            lockfile_hash=lock_hash,
            notes=notes,
        )

        file_path = self.experiment_dir / f"{run_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)

        logger.info(f"Logged experiment run '{run_id}' with full provenance to {file_path}")
        return record

    def get_run(self, run_id: str) -> Optional[ExperimentRecord]:
        """Loads an experiment record by ID."""
        file_path = self.experiment_dir / f"{run_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentRecord(**data)

    def list_runs(self) -> List[ExperimentRecord]:
        """Lists all experiment records sorted by timestamp."""
        runs: List[ExperimentRecord] = []
        for file in self.experiment_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "run_id" in data and "experiment_name" in data:
                    runs.append(ExperimentRecord(**data))
            except Exception as e:
                logger.warning(f"Failed to read experiment log {file}: {e}")
        runs.sort(key=lambda x: x.timestamp)
        return runs

    def to_markdown_summary(self) -> str:
        """Produces a consolidated Markdown summary of all logged runs."""
        runs = self.list_runs()
        if not runs:
            return "No experiments logged yet."

        lines = [
            "| Run ID | Model | Seed | Accuracy | Macro-F1 | Weighted-F1 | Log Loss | ECE |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in runs:
            m = r.metrics
            acc = m.get("accuracy", 0.0)
            mf1 = m.get("macro_f1", 0.0)
            wf1 = m.get("weighted_f1", 0.0)
            ll = m.get("multiclass_log_loss")
            ll_str = f"{ll:.4f}" if ll is not None else "N/A"
            ece = m.get("expected_calibration_error")
            ece_str = f"{ece:.4f}" if ece is not None else "N/A"
            lines.append(
                f"| `{r.run_id[:22]}` | **{r.model_name}** | {r.seed} | "
                f"`{acc:.4f}` | `{mf1:.4f}` | `{wf1:.4f}` | `{ll_str}` | `{ece_str}` |"
            )
        return "\n".join(lines)
