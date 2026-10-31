"""
CLI training runner for Indian Raga Intelligence models.

Supports:
- Baseline classifier (PCD + Chroma + Spectral -> SVM / Logistic Regression / Random Forest)
- MERT transfer learning head
- Hybrid late-fusion model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from raga_intelligence.config import settings
from raga_intelligence.datasets.adapters import SyntheticDatasetAdapter
from raga_intelligence.datasets.manifest import DatasetManifest
from raga_intelligence.datasets.saraga import SaragaHindustaniAdapter
from raga_intelligence.evaluation.reporting import generate_markdown_report
from raga_intelligence.training.trainer import (
    train_baseline_pipeline,
    train_hybrid_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train raga recognition models.")
    parser.add_argument(
        "--model",
        choices=["baseline", "hybrid"],
        default="baseline",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "saraga", "hindustani"],
        default="synthetic",
        help="Dataset to use for training and evaluation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/hindustani_dataset",
        help="Path to Hindustani dataset root",
    )
    parser.add_argument(
        "--classifier-type",
        choices=["svm", "logistic_regression", "random_forest"],
        default="svm",
        help="Baseline classifier algorithm",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for neural training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="./reports",
        help="Directory to write evaluation markdown reports",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    rep_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or synthesize dataset manifest
    if args.dataset == "synthetic":
        logger.info("Using Synthetic Hindustani dataset fixtures...")
        adapter = SyntheticDatasetAdapter(
            output_dir="./data/synthetic",
            recordings_per_raga=4,
            duration_sec=5.0,
        )
        manifest = adapter.build_manifest()
    elif args.dataset == "hindustani":
        logger.info(f"Loading real Hindustani dataset from {args.data_dir}...")
        from raga_intelligence.datasets.hindustani_dataset import HindustaniDatasetAdapter
        adapter = HindustaniDatasetAdapter(dataset_dir=args.data_dir)
        manifest = adapter.build_manifest()
        if len(manifest.items) == 0:
            logger.error("No items found in dataset directory. Exiting.")
            return
    else:
        logger.info(f"Loading Saraga Hindustani dataset from {args.data_dir}...")
        adapter = SaragaHindustaniAdapter(dataset_dir=args.data_dir)
        manifest = adapter.build_manifest()
        if len(manifest.items) == 0:
            logger.error("No items found in Saraga directory. Exiting.")
            return

    # Validate split integrity
    validation = manifest.validate_splits()
    logger.info(
        f"Manifest validation: No recording leakage = {validation['valid_no_recording_leakage']} "
        f"(Train: {validation['train_recordings_count']}, Val: {validation['val_recordings_count']}, "
        f"Test: {validation['test_recordings_count']})"
    )

    # 2. Train selected model
    cache_path = Path("./artifacts/cache") / f"{args.dataset}_features.npz"
    if args.model == "baseline":
        save_path = out_dir / f"baseline_{args.classifier_type}.joblib"
        logger.info(f"Training baseline {args.classifier_type} classifier...")
        model, metrics = train_baseline_pipeline(
            manifest=manifest,
            classifier_type=args.classifier_type,
            save_path=save_path,
            cache_path=cache_path,
        )
        report_file = rep_dir / f"baseline_{args.classifier_type}_{args.dataset}_report.md"

    elif args.model == "hybrid":
        save_path = out_dir / f"hybrid_{args.dataset}_model.pt"
        logger.info(f"Training hybrid fusion model for {args.epochs} epochs...")
        model, metrics = train_hybrid_pipeline(
            manifest=manifest,
            save_path=save_path,
            cache_path=cache_path,
            epochs=args.epochs,
        )
        report_file = rep_dir / f"hybrid_fusion_{args.dataset}_report.md"

    # 3. Generate report
    report_md = generate_markdown_report(
        metrics=metrics,
        experiment_name=f"{args.model.upper()} ({args.dataset.capitalize()} Dataset)",
        dataset_name=args.dataset.capitalize(),
        notes=(
            f"Trained with {args.model} architecture. "
            f"Checkpoint persisted to `{save_path}`."
        ),
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Evaluation report written to {report_file}")
    logger.info(f"Macro-F1: {metrics.macro_f1:.4f} | Accuracy: {metrics.accuracy:.4f}")


if __name__ == "__main__":
    main()

