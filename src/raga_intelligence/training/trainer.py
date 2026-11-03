"""
Model training, checkpointing, and evaluation orchestration.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from raga_intelligence.audio.loader import load_audio
from raga_intelligence.config import FeatureConfig, RAGA_LIST, settings
from raga_intelligence.datasets.manifest import DatasetManifest
from raga_intelligence.evaluation.metrics import EvaluationMetrics, compute_classification_metrics
from raga_intelligence.features.cache import FeatureCacheManager, compute_manifest_content_fingerprint
from raga_intelligence.features.extractor import BaselineFeatureExtractor
from raga_intelligence.features.schema import FeatureSchema
from raga_intelligence.models.baseline import BaselineRagaClassifier
from raga_intelligence.models.classifier import MERTClassifierHead
from raga_intelligence.models.hybrid import HybridRagaClassifier

logger = logging.getLogger(__name__)


def extract_features_from_manifest(
    manifest: DatasetManifest,
    cache_path: Optional[Path | str] = None,
    include_melodic_features: bool = False,
    force_refresh: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts explicit musical features (35-dim baseline, or 75-dim with melodic features)
    across train, val, and test splits with strict cryptographic cache fingerprinting.
    """
    manifest_hash = compute_manifest_content_fingerprint(manifest.items)
    config = FeatureConfig(include_melodic_features=include_melodic_features)

    expected_meta = {
        "dataset_fingerprint": manifest.compute_dataset_fingerprint(),
        "feature_schema_version": FeatureSchema.SCHEMA_VERSION,
        "extractor_version": "1.0.0",
        "include_melodic_features": bool(include_melodic_features),
        "pitch_tracker": str(config.pitch_tracker),
        "hop_length": int(config.hop_length),
        "fmin": float(config.fmin),
        "fmax": float(config.fmax),
        "sample_rate": 24000,
        "manifest_hash": manifest_hash,
        "items_count": len(manifest.items),
        "preprocessing_config": {
            "sample_rate": 24000,
            "hop_length": int(config.hop_length),
            "fmin": float(config.fmin),
            "fmax": float(config.fmax),
            "pitch_tracker": str(config.pitch_tracker),
            "include_melodic_features": bool(include_melodic_features),
        },
    }

    if cache_path:
        cached_data = FeatureCacheManager.load_validated_cache(
            cache_path=cache_path,
            expected_metadata=expected_meta,
            force_refresh=force_refresh,
        )
        if cached_data is not None:
            return (
                cached_data["X_train"],
                cached_data["y_train"],
                cached_data["X_val"],
                cached_data["y_val"],
                cached_data["X_test"],
                cached_data["y_test"],
            )

    extractor = BaselineFeatureExtractor(config=config)
    manifest_items = list(manifest.items)

    splits_data: Dict[str, Tuple[List[np.ndarray], List[str], List[str]]] = {
        "train": ([], [], []),
        "val": ([], [], []),
        "test": ([], [], []),
    }

    for item in manifest_items:
        if not Path(item.file_path).exists():
            continue
        try:
            signal = load_audio(item.file_path, target_sr=24000)
            feat = extractor.extract(signal, ground_truth_tonic_hz=item.tonic_hz)
            target_split = item.split if item.split in splits_data else "train"
            splits_data[target_split][0].append(feat.feature_vector)
            splits_data[target_split][1].append(item.raga)
            splits_data[target_split][2].append(item.recording_id)
        except Exception as e:
            logger.warning(f"Feature extraction failed for {item.file_path}: {e}")

    X_train = np.array(splits_data["train"][0], dtype=np.float32)
    y_train = np.array(splits_data["train"][1])

    X_val = np.array(splits_data["val"][0], dtype=np.float32)
    y_val = np.array(splits_data["val"][1])

    X_test = np.array(splits_data["test"][0], dtype=np.float32)
    y_test = np.array(splits_data["test"][1])

    if cache_path:
        arrays = {
            "X_train": X_train,
            "y_train": y_train,
            "recording_ids_train": np.array(splits_data["train"][2]),
            "ids_train": np.array(splits_data["train"][2]),
            "splits_train": np.array(["train"] * len(X_train)),
            "X_val": X_val,
            "y_val": y_val,
            "recording_ids_val": np.array(splits_data["val"][2]),
            "ids_val": np.array(splits_data["val"][2]),
            "splits_val": np.array(["val"] * len(X_val)),
            "X_test": X_test,
            "y_test": y_test,
            "recording_ids_test": np.array(splits_data["test"][2]),
            "ids_test": np.array(splits_data["test"][2]),
            "splits_test": np.array(["test"] * len(X_test)),
        }
        FeatureCacheManager.save_atomic_cache(
            cache_path=cache_path,
            arrays=arrays,
            metadata=expected_meta,
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def extract_mert_embeddings_from_manifest(
    manifest: DatasetManifest,
    cache_path: Optional[Path | str] = None,
    model_name: Optional[str] = None,
    model_revision: Optional[str] = None,
    pooling_type: Literal["mean", "max", "mean_max_concat"] = "mean_max_concat",
    sample_rate: int = 24000,
    max_duration_sec: float = 10.0,
    batch_size: int = 8,
    force_refresh: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts real MERT embeddings (pooled vectors) across train, val, and test splits.
    Caches embeddings to disk with strict cache fingerprinting.
    Fails explicitly if audio files are missing or embeddings cannot be generated.
    """
    effective_model_name = model_name or settings.model.mert_model_name
    effective_model_revision = model_revision or settings.model.mert_model_revision

    # 1. Compute stable manifest and configuration fingerprint
    manifest_hash = compute_manifest_content_fingerprint(manifest.items)

    expected_meta = {
        "dataset_fingerprint": manifest.compute_dataset_fingerprint(),
        "feature_schema_version": FeatureSchema.SCHEMA_VERSION,
        "model_name": str(effective_model_name),
        "model_revision": str(effective_model_revision),
        "pooling_type": str(pooling_type),
        "sample_rate": int(sample_rate),
        "max_duration_sec": float(max_duration_sec),
        "manifest_hash": manifest_hash,
        "items_count": len(manifest.items),
        "preprocessing_config": {
            "sample_rate": int(sample_rate),
            "max_duration_sec": float(max_duration_sec),
            "pooling_type": str(pooling_type),
            "batch_size": int(batch_size),
            "model_name": str(effective_model_name),
            "model_revision": str(effective_model_revision),
        },
    }

    # 2. Check cache validity
    if cache_path:
        cached_data = FeatureCacheManager.load_validated_cache(
            cache_path=cache_path,
            expected_metadata=expected_meta,
            force_refresh=force_refresh,
        )
        if cached_data is not None:
            return (
                cached_data["E_train"],
                cached_data["y_train"],
                cached_data["E_val"],
                cached_data["y_val"],
                cached_data["E_test"],
                cached_data["y_test"],
            )

    # 3. Fail explicitly if any audio file is missing
    for it in manifest.items:
        if not Path(it.file_path).exists():
            raise FileNotFoundError(
                f"Audio file missing for recording '{it.recording_id}': {it.file_path}. "
                "All manifest recordings must be accessible."
            )

    # 4. Load encoder
    from raga_intelligence.models.mert_encoder import MERTFeatureExtractor

    extractor = MERTFeatureExtractor(
        model_name=effective_model_name,
        model_revision=effective_model_revision,
        pooling_type=pooling_type,
        device=device,
    )
    extractor.load_model()

    splits_data: Dict[str, Tuple[List[np.ndarray], List[str], List[str]]] = {
        "train": ([], [], []),
        "val": ([], [], []),
        "test": ([], [], []),
    }

    max_samples = int(sample_rate * max_duration_sec)
    manifest_items = list(manifest.items)

    # 5. Extract in batches
    for i in range(0, len(manifest_items), batch_size):
        batch = manifest_items[i : i + batch_size]
        waveforms: List[np.ndarray] = []
        for it in batch:
            try:
                sig = load_audio(it.file_path, target_sr=sample_rate)
                samples = sig.samples[:max_samples] if len(sig.samples) > max_samples else sig.samples
                waveforms.append(samples)
            except Exception as e:
                raise RuntimeError(f"Audio decoding failed for recording '{it.recording_id}' at {it.file_path}: {e}") from e

        try:
            batch_embs = extractor.extract_batch(waveforms, sample_rate=sample_rate, batch_size=len(batch))
        except Exception as e:
            raise RuntimeError(f"MERT batch embedding extraction failed at batch offset {i}: {e}") from e

        for idx, it in enumerate(batch):
            split_key = it.split if it.split in splits_data else "train"
            splits_data[split_key][0].append(batch_embs[idx].numpy())
            splits_data[split_key][1].append(it.raga)
            splits_data[split_key][2].append(it.recording_id)

        if (i + len(batch)) % 50 == 0 or (i + len(batch)) == len(manifest_items):
            logger.info(f"Extracted MERT embeddings for {i + len(batch)}/{len(manifest_items)} clips...")

    E_train = np.array(splits_data["train"][0], dtype=np.float32) if splits_data["train"][0] else np.empty((0, 0), dtype=np.float32)
    y_train = np.array(splits_data["train"][1])

    E_val = np.array(splits_data["val"][0], dtype=np.float32) if splits_data["val"][0] else np.empty((0, 0), dtype=np.float32)
    y_val = np.array(splits_data["val"][1])

    E_test = np.array(splits_data["test"][0], dtype=np.float32) if splits_data["test"][0] else np.empty((0, 0), dtype=np.float32)
    y_test = np.array(splits_data["test"][1])

    # 6. Atomic cache write
    if cache_path:
        arrays = {
            "E_train": E_train,
            "y_train": y_train,
            "recording_ids_train": np.array(splits_data["train"][2]),
            "ids_train": np.array(splits_data["train"][2]),
            "splits_train": np.array(["train"] * len(E_train)),
            "E_val": E_val,
            "y_val": y_val,
            "recording_ids_val": np.array(splits_data["val"][2]),
            "ids_val": np.array(splits_data["val"][2]),
            "splits_val": np.array(["val"] * len(E_val)),
            "E_test": E_test,
            "y_test": y_test,
            "recording_ids_test": np.array(splits_data["test"][2]),
            "ids_test": np.array(splits_data["test"][2]),
            "splits_test": np.array(["test"] * len(E_test)),
        }
        FeatureCacheManager.save_atomic_cache(
            cache_path=cache_path,
            arrays=arrays,
            metadata=expected_meta,
        )

    return E_train, y_train, E_val, y_val, E_test, y_test


def _assert_no_split_leakage(manifest: DatasetManifest) -> None:
    val = manifest.validate_splits()
    if not val["valid_no_recording_leakage"]:
        raise ValueError(f"Fatal dataset leakage: recordings found in multiple splits: {val['leaked_recordings']}")
    if not val["valid_no_performance_leakage"]:
        raise ValueError(f"Fatal dataset leakage: performances found in multiple splits: {val['leaked_performances']}")
    if not val["valid_no_performer_leakage"]:
        raise ValueError(f"Fatal dataset leakage: performers found in multiple splits: {val['leaked_performers']}")


def train_baseline_pipeline(
    manifest: DatasetManifest,
    classifier_type: str = "svm",
    save_path: Optional[Path | str] = None,
    cache_path: Optional[Path | str] = None,
    include_melodic_features: bool = False,
    force_refresh: bool = False,
) -> Tuple[BaselineRagaClassifier, EvaluationMetrics]:
    """Trains and evaluates a conventional scikit-learn baseline classifier."""
    _assert_no_split_leakage(manifest)
    X_train, y_train, X_val, y_val, X_test, y_test = extract_features_from_manifest(
        manifest,
        cache_path=cache_path,
        include_melodic_features=include_melodic_features,
        force_refresh=force_refresh,
    )

    # Combine train + val if val is small, or fit on train
    X_fit = np.vstack([X_train, X_val]) if len(X_val) > 0 else X_train
    y_fit = np.concatenate([y_train, y_val]) if len(y_val) > 0 else y_train

    classifier = BaselineRagaClassifier(
        classifier_type=classifier_type,  # type: ignore
        class_names=RAGA_LIST,
    )
    classifier.fit(X_fit, y_fit)

    # Evaluate on held-out test split (or val/train fallback if test empty)
    X_eval = X_test if len(X_test) > 0 else (X_val if len(X_val) > 0 else X_train)
    y_eval = y_test if len(y_test) > 0 else (y_val if len(y_val) > 0 else y_train)

    y_pred = classifier.predict(X_eval)
    y_probs = classifier.predict_proba(X_eval)

    metrics = compute_classification_metrics(
        y_true=y_eval,
        y_pred=y_pred,
        class_names=RAGA_LIST,
        y_probs=y_probs,
    )

    if save_path:
        classifier.save(save_path)

    return classifier, metrics


def train_neural_head_pipeline(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    num_classes: int = len(RAGA_LIST),
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_path: Optional[Path | str] = None,
) -> Tuple[MERTClassifierHead, Dict[str, List[float]]]:
    """Trains a neural classification head using PyTorch on CPU/GPU."""
    device = settings.get_torch_device()
    model = MERTClassifierHead(input_dim=input_dim, num_classes=num_classes).to(device)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch_y).sum().item()

        val_loss = val_loss / len(val_ds) if len(val_ds) > 0 else 0.0
        val_acc = correct / len(val_ds) if len(val_ds) > 0 else 0.0

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            model.save_checkpoint(save_path, extra_meta={"epoch": epoch, "val_loss": val_loss})

    return model, history


def train_mert_pipeline(
    manifest: DatasetManifest,
    save_path: Optional[Path | str] = None,
    cache_path: Optional[Path | str] = None,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 42,
    model_name: str = "m-a-p/MERT-v1-95M",
    model_revision: str = "main",
    pooling_type: Literal["mean", "max", "mean_max_concat"] = "mean_max_concat",
    force_refresh: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[MERTClassifierHead, EvaluationMetrics]:
    """Trains neural classification head on real frozen MERT embeddings."""
    _assert_no_split_leakage(manifest)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or settings.get_torch_device()

    E_train, y_train, E_val, y_val, E_test, y_test = extract_mert_embeddings_from_manifest(
        manifest=manifest,
        cache_path=cache_path,
        model_name=model_name,
        model_revision=model_revision,
        pooling_type=pooling_type,
        force_refresh=force_refresh,
        device=device,
    )

    class_to_idx = {name: i for i, name in enumerate(RAGA_LIST)}
    for c in y_train:
        if c not in class_to_idx:
            raise ValueError(f"Unknown training raga label '{c}'. Expected one of: {RAGA_LIST}")
    for c in y_test:
        if c not in class_to_idx:
            raise ValueError(f"Unknown test raga label '{c}'. Expected one of: {RAGA_LIST}")
    y_train_idx = torch.tensor([class_to_idx[c] for c in y_train], dtype=torch.long)
    y_test_idx = torch.tensor([class_to_idx[c] for c in y_test], dtype=torch.long)

    E_train_t = torch.tensor(E_train, dtype=torch.float32)
    E_test_t = torch.tensor(E_test, dtype=torch.float32)

    model = MERTClassifierHead(input_dim=E_train.shape[1], num_classes=len(RAGA_LIST)).to(device)
    train_ds = TensorDataset(E_train_t, y_train_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(E_test_t.to(device))
        probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        pred_idx = np.argmax(probs, axis=1)
        y_pred = [RAGA_LIST[i] for i in pred_idx]

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        class_names=RAGA_LIST,
        y_probs=probs,
    )
    if save_path:
        meta = {
            "model_name": model_name,
            "model_revision": model_revision,
            "pooling_type": pooling_type,
            "seed": seed,
            "epochs": epochs,
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
        }
        model.save_checkpoint(save_path, extra_meta=meta)
    return model, metrics


def train_hybrid_pipeline(
    manifest: DatasetManifest,
    save_path: Optional[Path | str] = None,
    cache_path: Optional[Path | str] = None,
    mert_cache_path: Optional[Path | str] = None,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    seed: int = 42,
    model_name: str = "m-a-p/MERT-v1-95M",
    model_revision: str = "main",
    pooling_type: Literal["mean", "max", "mean_max_concat"] = "mean_max_concat",
    force_refresh: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[HybridRagaClassifier, EvaluationMetrics]:
    """Trains late-fusion hybrid model combining real pitch features and real MERT embeddings."""
    _assert_no_split_leakage(manifest)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or settings.get_torch_device()

    # 1. Extract/load explicit musical features
    X_train, y_train, X_val, y_val, X_test, y_test = extract_features_from_manifest(
        manifest, cache_path=cache_path
    )

    # 2. Extract/load real MERT embeddings (FAIL EXPLICITLY if unavailable, NO synthetic fallback)
    E_train, y_train_m, E_val, y_val_m, E_test, y_test_m = extract_mert_embeddings_from_manifest(
        manifest=manifest,
        cache_path=mert_cache_path,
        model_name=model_name,
        model_revision=model_revision,
        pooling_type=pooling_type,
        force_refresh=force_refresh,
        device=device,
    )

    # 3. Verify exact sample alignment across modalities
    if len(X_train) != len(E_train) or not np.array_equal(y_train, y_train_m):
        raise ValueError(
            f"Train split alignment mismatch: musical features count ({len(X_train)}) "
            f"!= MERT embeddings count ({len(E_train)})."
        )
    if len(X_test) != len(E_test) or not np.array_equal(y_test, y_test_m):
        raise ValueError(
            f"Test split alignment mismatch: musical features count ({len(X_test)}) "
            f"!= MERT embeddings count ({len(E_test)})."
        )

    class_to_idx = {name: i for i, name in enumerate(RAGA_LIST)}
    for c in y_train:
        if c not in class_to_idx:
            raise ValueError(f"Unknown training raga label '{c}'. Expected one of: {RAGA_LIST}")
    for c in y_test:
        if c not in class_to_idx:
            raise ValueError(f"Unknown test raga label '{c}'. Expected one of: {RAGA_LIST}")
    y_train_idx = torch.tensor([class_to_idx[c] for c in y_train], dtype=torch.long)
    y_test_idx = torch.tensor([class_to_idx[c] for c in y_test], dtype=torch.long)

    mert_dim = E_train.shape[1]
    musical_dim = X_train.shape[1]

    M_train = torch.tensor(X_train, dtype=torch.float32)
    M_test = torch.tensor(X_test, dtype=torch.float32)
    E_train_t = torch.tensor(E_train, dtype=torch.float32)
    E_test_t = torch.tensor(E_test, dtype=torch.float32)

    model = HybridRagaClassifier(
        musical_dim=musical_dim,
        mert_dim=mert_dim,
        num_classes=len(RAGA_LIST),
    ).to(device)

    train_ds = TensorDataset(M_train, E_train_t, y_train_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Fit strictly on train split
    model.train()
    for _ in range(epochs):
        for m_batch, e_batch, y_batch in train_loader:
            m_batch, e_batch, y_batch = m_batch.to(device), e_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(m_batch, e_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation strictly on held-out test split
    model.eval()
    with torch.no_grad():
        test_logits = model(M_test.to(device), E_test_t.to(device))
        probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        pred_idx = np.argmax(probs, axis=1)
        y_pred = [RAGA_LIST[i] for i in pred_idx]

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        class_names=RAGA_LIST,
        y_probs=probs,
    )

    if save_path:
        meta = {
            "model_name": model_name,
            "model_revision": model_revision,
            "pooling_type": pooling_type,
            "seed": seed,
            "epochs": epochs,
            "musical_dim": musical_dim,
            "mert_dim": mert_dim,
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
        }
        model.save_checkpoint(save_path, extra_meta=meta)

    return model, metrics
