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
    train_mert_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train raga recognition models.")
    parser.add_argument(
        "--model",
        choices=["baseline", "mert", "hybrid", "mert_frozen"],
        default="baseline",
        help="Model architecture to train (baseline: handcrafted features; mert: frozen MERT encoder + classifier; hybrid: late fusion)",
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
        "--mert-model",
        type=str,
        default="m-a-p/MERT-v1-95M",
        help="Hugging Face model identifier for MERT",
    )
    parser.add_argument(
        "--mert-revision",
        type=str,
        default="main",
        help="Model revision (branch/tag/commit) for MERT",
    )
    parser.add_argument(
        "--pooling-type",
        choices=["mean", "max", "mean_max_concat"],
        default="mean_max_concat",
        help="Temporal pooling strategy for MERT frame-level hidden states",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs for neural training",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Target compute execution device (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and embedding extraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable training and initialization",
    )
    parser.add_argument(
        "--include-melodic",
        action="store_true",
        help="Include 40-dim sequential melodic and ornamentation features (extends baseline to 75 dims)",
    )
    parser.add_argument(
        "--force-refresh-cache",
        action="store_true",
        help="Force re-extraction of features and embeddings, bypassing cache",
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
    if validation["recording_isolation_status"] == "failed":
        raise ValueError(
            f"Fatal dataset leakage: recordings found in multiple splits: {validation['leaked_recordings']}"
        )
    if validation["performance_isolation_status"] == "failed":
        raise ValueError(
            f"Fatal dataset leakage: performances found in multiple splits: {validation['leaked_performances']}"
        )
    if validation["performer_isolation_status"] == "failed":
        raise ValueError(
            f"Fatal dataset leakage: performers found in multiple splits: {validation['leaked_performers']}"
        )

    summary = validation["split_summary"]
    logger.info(
        f"Manifest split validation PASSED: "
        f"Recordings: {summary['recordings_count']['train']} train / {summary['recordings_count']['val']} val / {summary['recordings_count']['test']} test. "
        f"Performer isolation: {validation['performer_isolation_status']}. "
        f"Known performers: {summary['performers_count']['train']} train / {summary['performers_count']['val']} val / {summary['performers_count']['test']} test."
    )
    if not validation.get("performer_metadata_complete", True):
        logger.warning(
            f"Performer metadata is incomplete ({validation.get('unknown_performer_count', 0)} items have no artist identity). "
            "Performer-level leakage cannot be fully guaranteed for those unannotated items."
        )

    # Normalize model selection and resolve device diagnostics
    model_choice = "mert" if args.model in ("mert", "mert_frozen") else args.model
    from raga_intelligence.config import verify_execution_device
    target_device, dev_diag = verify_execution_device(args.device)
    logger.info(f"Target compute execution device: {target_device} ({dev_diag['diagnostic_reason']})")

    # 2. Train selected model
    feature_suffix = "melodic_features" if args.include_melodic else "features"
    cache_path = Path("./artifacts/cache") / f"{args.dataset}_{feature_suffix}.npz"
    mert_cache_path = Path("./artifacts/cache") / f"{args.dataset}_mert_embeddings.npz"

    if model_choice == "baseline":
        model_name_tag = f"baseline_{args.classifier_type}_melodic" if args.include_melodic else f"baseline_{args.classifier_type}"
        save_path = out_dir / f"{model_name_tag}.joblib"
        logger.info(
            f"Training baseline {args.classifier_type} classifier "
            f"(include_melodic: {args.include_melodic}, dims: {75 if args.include_melodic else 35})..."
        )
        model, metrics = train_baseline_pipeline(
            manifest=manifest,
            classifier_type=args.classifier_type,
            save_path=save_path,
            cache_path=cache_path,
            include_melodic_features=args.include_melodic,
            force_refresh=args.force_refresh_cache,
        )
        report_file = rep_dir / f"{model_name_tag}_{args.dataset}_report.md"

    elif model_choice == "mert":
        save_path = out_dir / f"mert_{args.dataset}_model.pt"
        logger.info(
            f"Training frozen MERT classification head for {args.epochs} epochs "
            f"(model: {args.mert_model}, pooling: {args.pooling_type}, seed: {args.seed}, device: {target_device})..."
        )
        model, metrics = train_mert_pipeline(
            manifest=manifest,
            save_path=save_path,
            cache_path=mert_cache_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            model_name=args.mert_model,
            model_revision=args.mert_revision,
            pooling_type=args.pooling_type,
            force_refresh=args.force_refresh_cache,
            device=target_device,
        )
        report_file = rep_dir / f"mert_{args.dataset}_report.md"

    elif model_choice == "hybrid":
        save_path = out_dir / f"hybrid_{args.dataset}_model.pt"
        logger.info(
            f"Training hybrid fusion model for {args.epochs} epochs "
            f"(mert_model: {args.mert_model}, pooling: {args.pooling_type}, seed: {args.seed}, device: {target_device})..."
        )
        model, metrics = train_hybrid_pipeline(
            manifest=manifest,
            save_path=save_path,
            cache_path=cache_path,
            mert_cache_path=mert_cache_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            model_name=args.mert_model,
            model_revision=args.mert_revision,
            pooling_type=args.pooling_type,
            force_refresh=args.force_refresh_cache,
            device=target_device,
        )
        report_file = rep_dir / f"hybrid_fusion_{args.dataset}_report.md"

    # 3. Generate report
    report_md = generate_markdown_report(
        metrics=metrics,
        experiment_name=f"{model_choice.upper()} ({args.dataset.capitalize()} Dataset)",
        dataset_name=args.dataset.capitalize(),
        notes=(
            f"Trained with {model_choice} architecture. "
            f"Checkpoint persisted to `{save_path}`."
        ),
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Evaluation report written to {report_file}")
    logger.info(f"Macro-F1: {metrics.macro_f1:.4f} | Accuracy: {metrics.accuracy:.4f}")


if __name__ == "__main__":
    main()

