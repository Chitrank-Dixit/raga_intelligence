# Architectural Decision Records (ADR) - Indian Raga Intelligence

## ADR 001: Container-First Development with Podman and Hardware Passthrough
- **Date**: 2026-10-30
- **Status**: Accepted
- **Context**: The application requires local experimentation and deployment reproducibility on a Windows workstation equipped with WSL2 and an NVIDIA GeForce RTX 5060 Ti (16 GB VRAM). Docker is not guaranteed as a runtime; Podman 6.0.2 with `podman compose` is the standard.
- **Decision**: 
  - Standardize on multi-stage `Containerfile` based on `python:3.12-slim-bookworm` with ffmpeg and libsndfile.
  - Implement dual-profile Podman Compose configuration: default CPU service and a GPU-accelerated profile (`api-gpu`) utilizing `/dev/dxg` and `/usr/lib/wsl` driver passthrough.
  - Provide automatic CPU fallback when CUDA is unavailable.
- **Consequences**: Ensures clean local execution, prevents host environment pollution, and guarantees containerized deployment portability.

## ADR 002: Dataset Strategy and Research Data Governance
- **Date**: 2026-10-30
- **Status**: Accepted
- **Context**: Indian classical music requires authentic audio recordings with verified raga annotations, tonic ground truth, and performer metadata. Commercial scraping violates copyright and lacks verified labels.
- **Decision**:
  - Adopt the MTG CompMusic research corpus, specifically **Saraga Hindustani** (Zenodo Record `4301737`, licensed under CC BY-NC-SA 4.0) and IAM Raga Recognition Features (`7278505`).
  - Implement a modular `DatasetAdapter` interface supporting both local audio directories, Zenodo research archives, and synthetic test fixtures.
  - Enforce strict split integrity: train/validation/test splits are constructed strictly at the recording/performance and artist level, preventing temporal frame leakage.
- **Consequences**: Research compliance, legal redistribution safety, high label quality, and verifiable generalization benchmarks.

## ADR 003: Model Architecture - Hybrid Musical & Learned Representations
- **Date**: 2026-10-30
- **Status**: Accepted
- **Context**: Raga recognition is fundamentally modal and melodic (swara intervals, arohana/avarohana, vadi/samvadi, characteristic pakads/phrases) rather than purely timbral. Pretrained general music representations (MERT) capture rich harmonic and acoustic texture, but may overlook microtonal pitch movements (shrutis/gamaks) without explicit pitch grounding.
- **Decision**:
  - **Baseline**: Tonic-normalized pitch class distribution (PCD), pitch interval histograms, and chroma features classified by conventional estimators (Logistic Regression, Support Vector Classifier, Random Forest).
  - **Transfer Learning**: Pretrained MERT (`m-a-p/MERT-v1-95M`) at 24 kHz mono with frozen layer embeddings and temporal pooling (mean + max pooling).
  - **Hybrid Model**: Late-fusion concatenation of pitch-contour statistics and pooled MERT embeddings.
- **Consequences**: Empirical comparison allows determining whether deep audio representations alone suffice or if explicit musical knowledge enhances discriminative power.

## ADR 004: Cross-Genre Similarity Separation and Calibrated Abstention
- **Date**: 2026-10-30
- **Status**: Accepted
- **Context**: Western pop/rock and Indian film music are rarely composed within strict raga rules (chalan, vadi/samvadi, thhat, time cycle). Applying a classical softmax classifier to arbitrary music produces overconfident, musically invalid categorizations.
- **Decision**:
  - Decouple the system into two distinct pipelines:
    1. `RagaClassifier`: Closed-world classification for classical performances with validated labels.
    2. `MelodicSimilarityEngine`: Open-world similarity measurement computing prototype cosine distance, tonic-relative swara distribution correlation, and segment-level evidence accumulation.
  - Implement uncertainty estimation and an explicit *abstention mechanism* ("insufficient evidence" / out-of-distribution detection) when similarity scores fall below calibrated thresholds.
- **Consequences**: Honest predictions, avoidance of false claims, and musically grounded explanations for cross-genre users.

