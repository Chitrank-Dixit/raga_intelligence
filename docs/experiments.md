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

## Experiment 2: Transfer Learning with Frozen MERT Architecture
- **Model**: `m-a-p/MERT-v1-95M` frozen encoder with adaptive temporal pooling (`mean_max_concat` -> 1536 dims) and `MERTClassifierHead` (LayerNorm + GELU + Dropout + Linear MLP).
- **Corpus**: 390 authentic recordings (360 train, 30 held-out test, 0 recording leakage).
- **Embedding Cache**: Persisted 1536-dim vectors to `artifacts/cache/hindustani_mert_embeddings.npz`.
- **Results**:
  - **Macro-F1 Score**: `0.3440`
  - **Accuracy**: `0.4000` (vs. random baseline `0.1000`)
  - **Expected Calibration Error (ECE)**: `0.1794`
- **Key Finding**: Pretrained audio embeddings alone outperform random chance by 4x, demonstrating that Western/global acoustic self-supervised representations capture meaningful tonal texture. However, without explicit tonic grounding and swara interval alignment, frozen embeddings significantly trail the explicit pitch baseline (`0.6067` F1).
- **Status**: Completed. Checkpoint saved to `artifacts/checkpoints/mert_frozen_hindustani_model.pt`. Report written to `reports/mert_frozen_hindustani_report.md`.

---

## Experiment 3: Hybrid Melodic + Embedding Late Fusion
- **Hypothesis**: Late-fusion combining domain-specific tonic-relative pitch class distributions (35 dims) with deep acoustic MERT representations (1536 dims) achieves superior discriminative performance over either modality alone.
- **Model**: `HybridRagaClassifier` (Dual linear projection heads with LayerNorm and late fusion).
- **Results**:
  - **Macro-F1 Score**: `0.7400`
  - **Accuracy**: `0.7667`
  - **Expected Calibration Error (ECE)**: `0.1808`
- **Side-by-Side Representation Comparison**:
  | Architecture | Input Representation | Macro-F1 | Accuracy | ECE |
  | :--- | :--- | :---: | :---: | :---: |
  | **Frozen MERT Head** | 1536-dim MERT Embeddings | 0.3440 | 0.4000 | 0.1794 |
  | **Baseline SVM** | 35-dim Pitch Class & Chroma | 0.6067 | 0.6000 | 0.2003 |
  | **Hybrid Late Fusion** | 35-dim Musical + 1536-dim MERT | **0.7400** | **0.7667** | **0.1808** |
- **Conclusion**: Confirms the central research hypothesis. Combining explicit tonic-relative pitch distributions with deep pretrained music representations delivers an absolute **+13.3% F1 gain over pitch baseline** and **+39.6% F1 gain over MERT alone**.
- **Status**: Completed. Checkpoint saved to `artifacts/checkpoints/hybrid_hindustani_model.pt`. Report written to `reports/hybrid_fusion_hindustani_report.md`.

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
- **Cross-Genre Evaluation Matrix**:
  | Test Track | Genre | Expected Raga Resemblance | Top Predicted Match | Similarity Score | Confidence | Abstention Status |
  | :--- | :--- | :--- | :--- | :---: | :--- | :---: |
  | **Bollywood (Bhairavi Melody)** | Indian Film / Bollywood | `Bhairavi` | **Bhairavi** | `0.863` | High | No |
  | **Bollywood (Yaman Ballad)** | Indian Film / Bollywood | `Yaman` | **Yaman** | `0.852` | High | No |
  | **Western Pop (Major Pentatonic)** | Western Pop / Folk | `Bhoopali` | **Bhoopali** | `0.918` | High | No |
  | **Western Blues (Minor Pentatonic)** | Western Blues / Rock | `Malkauns` | **Malkauns** | `0.953` | High | No |
  | **Atonal Noise Control** | Unpitched Control | `Abstain` | **None** | `0.000` | N/A | Yes (Correct) |
- **Key Musicological Insights**:
  1. **Bollywood / Film Music**: High similarity achieved on modal scale alignments with explicit swara evidence.
  2. **Western Pop**: Major pentatonic melodies naturally map to Raga *Bhoopali* (`S R G P D`), illustrating cross-cultural modal equivalence.
  3. **Western Blues**: Minor pentatonic contours map to Raga *Malkauns* (`S g M d n`), reflecting shared pitch intervals without false assertions of classical improvisation.
  4. **Abstention Gate**: Atonal noise triggers immediate rejection (`spectral flatness 0.56 > 0.15`), preventing spurious raga attribution.
- **Status**: Verified in CLI runner `scripts/evaluate_cross_genre.py` and documented in `reports/cross_genre_evaluation.md`.

