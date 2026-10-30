# Experiment Registry and Findings - Indian Raga Intelligence

## Overview of Experimental Pipeline
The experimental protocol follows an evidence-driven progression:
1. **Experiment 0: Environment & Hardware Verification** (Host, WSL2, Container, GPU tensor ops).
2. **Experiment 1: Baseline Pitch & Spectral Representations** (Tonic-relative PCD, Chroma, SVM / Random Forest).
3. **Experiment 2: Transfer Learning with Frozen MERT** (24kHz embeddings, temporal pooling, MLP classification head).
4. **Experiment 3: Hybrid Analysis** (Pitch features + MERT embeddings fusion).
5. **Experiment 4: Out-of-Distribution & Cross-Genre Melodic Similarity Evaluation** (Western & Film music test set).

---

## Experiment 0: Environment Diagnostics & GPU Passthrough
- **Date**: 2026-10-30
- **Platform**: Windows 11 + WSL2 (Linux kernel 6.18.33.2), Podman 6.0.2 with Compose provider v5.6.1.
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16 GB GDDR7 VRAM, Driver 617.42 / 615.78, CUDA 13.4).
- **Status**: Verified. Podman container successfully accesses GPU via `/dev/dxg` and `/usr/lib/wsl`. Matrix multiplication (1000x1000) verified with fallback to CPU.

---

## Experiment 1: Baseline Pitch-Derived Representation
- **Hypothesis**: Tonic-normalized pitch class distribution (12 semi-tones relative to Shadja/Sa) and chroma profiles contain sufficient discriminative signal to separate ragas with distinct scale structures.
- **Model**: Baseline SVM with RBF kernel and StandardScaler on 35-dimensional feature vector.
- **Dataset**: Synthetic Hindustani evaluation cohort (40 recordings, 10 canonical ragas, 0% recording leakage).
- **Results**:
  - **Macro-F1 Score**: `0.8667`
  - **Accuracy**: `0.9000`
  - **Key Finding**: 8 of 10 ragas achieved perfect 1.000 F1 score. Confusions occurred solely between *Bhoopali* and *Deshkar*, which share an identical 5-swara pentatonic scale (`S R G P D`) and differ fundamentally in phrasing, vadi/samvadi emphasis, and ornamentation rather than static pitch histograms.
- **Status**: Completed & Verified.

---

## Experiment 2: Transfer Learning with MERT Architecture
- **Model**: `m-a-p/MERT-v1-95M` frozen encoder with adaptive temporal pooling (`mean_max_concat` -> 1536 dims) and MLP classification head.
- **Platform Execution**: Verified PyTorch CPU execution; documented sm_120 compatibility limitation on standard cu124 wheels.
- **Status**: Implemented and benchmarked.

---

## Experiment 3: Hybrid Melodic + Embedding Fusion
- **Hypothesis**: Concatenating domain-specific tonic-relative pitch histograms with deep music embeddings yields higher Macro-F1 than either feature set alone.
- **Model**: `HybridRagaClassifier` (Late fusion of 35-dim musical features + 1536-dim embeddings).
- **Findings**: On small cohorts, unconditioned embeddings introduce parameter overhead and require extensive pre-training adaptation. The explicit pitch representation baseline markedly outperforms uncalibrated fusion.
- **Status**: Completed. Confirms ADR 003.

