"""
Model training, checkpointing, and evaluation orchestration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from raga_intelligence.audio.loader import load_audio
from raga_intelligence.config import RAGA_LIST, settings
from raga_intelligence.datasets.manifest import DatasetManifest
from raga_intelligence.evaluation.metrics import EvaluationMetrics, compute_classification_metrics
from raga_intelligence.features.extractor import BaselineFeatureExtractor
from raga_intelligence.models.baseline import BaselineRagaClassifier
from raga_intelligence.models.classifier import MERTClassifierHead
from raga_intelligence.models.hybrid import HybridRagaClassifier

logger = logging.getLogger(__name__)


def extract_features_from_manifest(
    manifest: DatasetManifest,
    cache_path: Optional[Path | str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts explicit musical features (35-dim) across train, val, and test splits.
    Caches features to disk if cache_path is specified.
    """
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["y_test"],
        )

    extractor = BaselineFeatureExtractor()

    splits_data: Dict[str, Tuple[List[np.ndarray], List[str]]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }

    for item in manifest.items:
        if not Path(item.file_path).exists():
            continue
        try:
            signal = load_audio(item.file_path, target_sr=24000)
            feat = extractor.extract(signal, ground_truth_tonic_hz=item.tonic_hz)
            target_split = item.split if item.split in splits_data else "train"
            splits_data[target_split][0].append(feat.feature_vector)
            splits_data[target_split][1].append(item.raga)
        except Exception as e:
            logger.warning(f"Feature extraction failed for {item.file_path}: {e}")

    X_train = np.array(splits_data["train"][0], dtype=np.float32)
    y_train = np.array(splits_data["train"][1])

    X_val = np.array(splits_data["val"][0], dtype=np.float32)
    y_val = np.array(splits_data["val"][1])

    X_test = np.array(splits_data["test"][0], dtype=np.float32)
    y_test = np.array(splits_data["test"][1])

    if cache_path:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        logger.info(f"Cached extracted features to {p}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def extract_mert_embeddings_from_manifest(
    manifest: DatasetManifest,
    cache_path: Optional[Path | str] = None,
    max_duration_sec: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts real MERT embeddings (1536-dim) across train, val, and test splits.
    Caches embeddings to disk if cache_path is specified.
    """
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached MERT embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["E_train"],
            data["y_train"],
            data["E_val"],
            data["y_val"],
            data["E_test"],
            data["y_test"],
        )

    from raga_intelligence.models.mert_encoder import MERTFeatureExtractor

    extractor = MERTFeatureExtractor()
    extractor.load_model()

    splits_data: Dict[str, Tuple[List[np.ndarray], List[str]]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }

    max_samples = int(24000 * max_duration_sec)
    for i, item in enumerate(manifest.items):
        if not Path(item.file_path).exists():
            continue
        try:
            signal = load_audio(item.file_path, target_sr=24000)
            samples = signal.samples[:max_samples] if len(signal.samples) > max_samples else signal.samples
            emb = extractor.extract_segment_embedding(samples, sample_rate=24000)
            target_split = item.split if item.split in splits_data else "train"
            splits_data[target_split][0].append(emb.numpy())
            splits_data[target_split][1].append(item.raga)
            if (i + 1) % 50 == 0 or (i + 1) == len(manifest.items):
                logger.info(f"Extracted MERT embeddings for {i + 1}/{len(manifest.items)} clips...")
        except Exception as e:
            logger.warning(f"MERT extraction failed for {item.file_path}: {e}")

    E_train = np.array(splits_data["train"][0], dtype=np.float32)
    y_train = np.array(splits_data["train"][1])

    E_val = np.array(splits_data["val"][0], dtype=np.float32)
    y_val = np.array(splits_data["val"][1])

    E_test = np.array(splits_data["test"][0], dtype=np.float32)
    y_test = np.array(splits_data["test"][1])

    if cache_path:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            E_train=E_train,
            y_train=y_train,
            E_val=E_val,
            y_val=y_val,
            E_test=E_test,
            y_test=y_test,
        )
        logger.info(f"Cached extracted MERT embeddings to {p}")

    return E_train, y_train, E_val, y_val, E_test, y_test


def train_baseline_pipeline(
    manifest: DatasetManifest,
    classifier_type: str = "svm",
    save_path: Optional[Path | str] = None,
    cache_path: Optional[Path | str] = None,
) -> Tuple[BaselineRagaClassifier, EvaluationMetrics]:
    """Trains and evaluates a conventional scikit-learn baseline classifier."""
    X_train, y_train, X_val, y_val, X_test, y_test = extract_features_from_manifest(
        manifest, cache_path=cache_path
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
) -> Tuple[MERTClassifierHead, EvaluationMetrics]:
    """Trains neural classification head on frozen MERT embeddings."""
    device = settings.get_torch_device()
    E_train, y_train, E_val, y_val, E_test, y_test = extract_mert_embeddings_from_manifest(
        manifest, cache_path=cache_path
    )
    class_to_idx = {name: i for i, name in enumerate(RAGA_LIST)}
    y_train_idx = torch.tensor([class_to_idx.get(c, 0) for c in y_train], dtype=torch.long)
    y_test_idx = torch.tensor([class_to_idx.get(c, 0) for c in y_test], dtype=torch.long)

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
        model.save_checkpoint(save_path)
    return model, metrics


def train_hybrid_pipeline(
    manifest: DatasetManifest,
    save_path: Optional[Path | str] = None,
    cache_path: Optional[Path | str] = None,
    mert_cache_path: Optional[Path | str] = None,
    epochs: int = 25,
) -> Tuple[HybridRagaClassifier, EvaluationMetrics]:
    """Trains late-fusion hybrid model combining pitch features and embeddings."""
    device = settings.get_torch_device()
    X_train, y_train, X_val, y_val, X_test, y_test = extract_features_from_manifest(
        manifest, cache_path=cache_path
    )

    class_to_idx = {name: i for i, name in enumerate(RAGA_LIST)}
    y_train_idx = torch.tensor([class_to_idx.get(c, 0) for c in y_train], dtype=torch.long)
    y_test_idx = torch.tensor([class_to_idx.get(c, 0) for c in y_test], dtype=torch.long)

    # Use real cached MERT embeddings if available, otherwise fallback to synthetic for fast testing
    if mert_cache_path and Path(mert_cache_path).exists():
        logger.info(f"Loading real cached MERT embeddings from {mert_cache_path} for hybrid training...")
        mert_data = np.load(mert_cache_path, allow_pickle=True)
        E_train = torch.tensor(mert_data["E_train"], dtype=torch.float32)
        E_test = torch.tensor(mert_data["E_test"], dtype=torch.float32)
        mert_dim = E_train.shape[1]
    else:
        mert_dim = 1536
        rng = np.random.default_rng(42)
        E_train = torch.tensor(rng.standard_normal((len(X_train), mert_dim)), dtype=torch.float32)
        E_test = torch.tensor(rng.standard_normal((len(X_test), mert_dim)), dtype=torch.float32)

    musical_dim = X_train.shape[1]
    M_train = torch.tensor(X_train, dtype=torch.float32)
    M_test = torch.tensor(X_test, dtype=torch.float32)

    model = HybridRagaClassifier(
        musical_dim=musical_dim,
        mert_dim=mert_dim,
        num_classes=len(RAGA_LIST),
    ).to(device)

    train_ds = TensorDataset(M_train, E_train, y_train_idx)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    model.train()
    for _ in range(epochs):
        for m_batch, e_batch, y_batch in train_loader:
            m_batch, e_batch, y_batch = m_batch.to(device), e_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(m_batch, e_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(M_test.to(device), E_test.to(device))
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
        model.save_checkpoint(save_path)

    return model, metrics
