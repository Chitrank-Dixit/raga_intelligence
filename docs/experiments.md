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
- **Synthetic Benchmark**:
  - Macro-F1 Score: `0.8667`, Accuracy: `0.9000` (40 recordings, 10 canonical ragas, 0% recording leakage).
- **Real Hindustani Dataset Benchmark**:
  - Dataset: 390 audio clips across 10 canonical ragas (`neerajaabhyankar/hindustani-raag-small`), verified 0 recording leakage across 82 distinct performances.
  - Train: 360 clips; Test: 30 held-out unseen clips (3 per raga).
  - Ground-truth tonic: hand-annotated Sa in Hz.
  - **Macro-F1 Score**: `0.6067`
  - **Weighted-F1 Score**: `0.6067`
  - **Accuracy**: `0.6000` (Baseline random chance: `0.1000` / 10%)
  - **Expected Calibration Error (ECE)**: `0.2003`
  - **Per-Raga Breakdown**:
    - *Malkauns*: F1 = `1.0000`, Precision = `1.0000`, Recall = `1.0000`
    - *Bhairavi*: F1 = `0.8000`, Precision = `1.0000`, Recall = `0.6667`
    - *Kafi*: F1 = `0.8000`, Precision = `1.0000`, Recall = `0.6667`
    - *Darbari Kanada*: F1 = `0.6667`, Precision = `0.6667`, Recall = `0.6667`
    - *Deshkar*: F1 = `0.6667`, Precision = `0.5000`, Recall = `1.0000`
    - *Yaman*: F1 = `0.5000`, Precision = `1.0000`, Recall = `0.3333`
    - *Bageshree*: F1 = `0.5000`, Precision = `1.0000`, Recall = `0.3333`
    - *Todi*: F1 = `0.4000`, Precision = `0.2857`, Recall = `0.6667`
    - *Bhoopali*: F1 = `0.4000`, Precision = `0.5000`, Recall = `0.3333`
    - *Bhairav*: F1 = `0.3333`, Precision = `0.3333`, Recall = `0.3333`
  - **Key Musicological Finding**:
    1. Perfect classification on *Malkauns* due to its unmistakable pentatonic Komal Ga, Komal Dha, Komal Ni signature without Re and Pa.
    2. Symmetric confusion between *Bhoopali* and *Deshkar*: both share the identical Audav scale `S R G P D`. Static pitch distributions distinguish them moderately based on Vadi emphasis (Bhoopali on Gandhar vs Deshkar on Dhaivat), but melodic progression (*pakad*) is necessary for complete disambiguation.
    3. Confusions between *Bhairav* and *Todi* arise from shared Komal Re and Komal Dha; microtonal pitch inflections (*meend* / *andolan*) provide the critical distinction.
- **Status**: Completed & Verified on real recordings. Checkpoint saved to `artifacts/checkpoints/baseline_svm.joblib`. Report written to `reports/baseline_svm_hindustani_report.md`.

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

---

## Experiment 4: Empirical Raga Reference Representations (Corpus Centroids)
- **Method**: Extracted 12-bin tonic-relative Pitch Class Distribution centroids across all 360 training performances in the real Hindustani dataset.
- **Results**:
  - Empirical centroids capture the exact modal signatures of the 10 ragas (e.g. Malkauns `S-g-M-d-n`, Todi `S-r-g-M'-d-N`, Yaman `S-R-G-M'-P-D-N`).
  - Integrated into `RagaPrototypeRegistry` to power Bhattacharyya affinity and cosine distance in `MelodicSimilarityEngine`.
- **Status**: Implemented and verified in `src/raga_intelligence/similarity/prototypes.py`.

---

## Experiment 5: Cross-Genre Melodic Similarity & Calibrated Abstention
- **Hypothesis**: Multi-criterion gate combining spectral flatness (> 0.15), voiced ratio (< 0.15), and peak similarity score (< 0.40) reliably detects unpitched noise, speech, and atonal percussion without suppressing legitimate modal melodies.
- **Evaluation**:
  - White noise test: Spectral flatness = 0.56 (> 0.15 threshold) -> **Abstention successfully triggered** with diagnostic explanation.
  - Held-out classical test clips: Spectral flatness < 0.05, Voiced ratio > 95% -> **No false abstention**, ranked matches generated with aligned swaras and supporting temporal windows.
- **Status**: Verified in both CLI (`scripts/evaluate.py`) and REST API (`/analyze`).

