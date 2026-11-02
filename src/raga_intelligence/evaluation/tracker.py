"""
Lightweight JSON-based experiment tracking, environment introspection, and experiment logging.
"""

from __future__ import annotations

import datetime
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
    return "unknown_or_uncommitted"


def get_environment_metadata() -> Dict[str, Any]:
    """Inspects and returns the hardware, OS, and key ML library environment versions."""
    import librosa
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
        "cuda_available": cuda_available,
        "device_name": gpu_name,
    }


@dataclass
class ExperimentRecord:
    """Structured record of a single evaluated experiment run."""
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
    notes: Optional[str] = None


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
        env_meta = get_environment_metadata()

        record = ExperimentRecord(
            run_id=run_id,
            experiment_name=experiment_name,
            timestamp=timestamp,
            code_version=code_version,
            seed=seed,
            model_name=model_name,
            model_architecture=model_architecture,
            hyperparameters=hyperparameters,
            feature_config=feature_config,
            dataset_name=dataset_name,
            dataset_split_info=dataset_split_info or {},
            metrics=metrics,
            system_environment=env_meta,
            notes=notes,
        )

        file_path = self.experiment_dir / f"{run_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)

        logger.info(f"Logged experiment run '{run_id}' to {file_path}")
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
