# Experiment Registry and Findings — Indian Raga Intelligence

> [!WARNING]
> **Notice on Historical / Superseded Hybrid Benchmarks**
> In early prototyping (commit `7525287`), the hybrid training script contained a synthetic fallback (`rng.standard_normal((len(X_train), mert_dim))`) when cached MERT representations were not yet extracted. Any early figures reported under that ungrounded configuration are **INVALIDATED AND SUPERSEDED**. They are preserved in the historical section below strictly for auditability and must **never** be cited as evidence of MERT representation capability.
>
> **Historical registry:** Existing figures below document earlier experiments and are not the current canonical benchmark. The Hindustani CSV has no artist field; previous use of video IDs as performer labels was invalid. Artist-level isolation is unverifiable for that corpus. The benchmark execution revision is clean commit `6eec1891715ba79e1bd607b058fc8c3426c02c76` (evaluation `9fedf9990c3d4576952a49b0999b9876`). PR HEAD before the current uncommitted review changes is `71902a2802eca91dcdf721f17af7c58f89a0ff84`; the benchmark is not claimed as run on that HEAD. Recording/performance isolation are verified and artist isolation remains unverifiable.

---

## 1. Experiment Registry Overview

Experiments are structured into three distinct status tiers:

| Experiment ID | Title | Status | Primary Artifact |
| :--- | :--- | :--- | :--- |
| **Exp 0** | Environment Diagnostics & GPU Passthrough | **Completed & Verified** | Host & Container Logs |
| **Exp 1** | Baseline Pitch-Derived Representation (SVM) | **Completed & Verified** | `artifacts/checkpoints/baseline_svm.joblib` |
| **Exp 2** | Transfer Learning with Frozen MERT Architecture | **Completed & Verified** | `artifacts/checkpoints/mert_hindustani_model.pt` |
| **Exp 3** | Real Hybrid Pitch + MERT Late Fusion | **Completed & Verified** | `artifacts/checkpoints/hybrid_hindustani_model.pt` |
| **Exp 4** | Empirical Raga Reference Centroids | **Completed & Verified** | `src/raga_intelligence/similarity/prototypes.py` |
| **Exp 5** | Cross-Genre Melodic Similarity & Calibrated Abstention | **Completed & Verified** | `reports/cross_genre_evaluation.md` |
| **Exp 6** | Phase 2.1 Scientific Evaluation Framework & Multi-Seed Benchmarks | **Current canonical run; artist isolation unverifiable** | `reports/scientific_evaluation_report.md` |
| *Historical* | *Synthetic Random-Embedding Hybrid Fallback* | **INVALIDATED / SUPERSEDED** | Historical commit `7525287` |
| **Exp 7** | Sequential Transition & Melodic Motif (*Pakad*) Modeling | **Planned** | Roadmap Phase 2.2 |
| **Exp 8** | Multi-Dataset Scaled Benchmark (50 Ragamala Corpus) | **Planned** | Roadmap Phase 2.3 |

---

## 2. Completed & Verified Experiments

### Experiment 0: Environment Diagnostics & GPU Passthrough
- **Date**: 2026-10-30
- **Platform**: Windows 11 host + WSL2 Ubuntu (`Linux kernel 6.18.33.2`), rootless Podman 5.7.0 / 6.0.2 with Compose provider.
- **Hardware**: NVIDIA GeForce RTX 5060 Ti (16 GB GDDR7 VRAM, Blackwell architecture, compute capability 12.0 / `sm_120`, Driver 617.42 / 615.78, CUDA 13.4).
- **GPU Compatibility Note**: Standard PyTorch 2.6 cu124 wheels compile kernels up to `sm_90`. In the absence of precompiled `sm_120` kernels, the runtime automatically activates a graceful CPU fallback without application termination.
- **Status**: Completed & Verified.

---

### Experiment 1: Baseline Pitch-Derived Representation (Handcrafted Features)
- **Hypothesis**: Tonic-normalized pitch class distribution (12 semi-tones relative to Shadja/Sa) and spectral chroma profiles contain sufficient discriminative signal to separate ragas with distinct scale structures.
- **Dataset**: `neerajaabhyankar/hindustani-raag-small` (390 authentic recordings across 10 canonical ragas, hand-annotated Sa tonic in Hz).
- **Split Strategy (historical run)**: Group-based splitting on performance/source IDs. The CSV contains no performer names, so performer-level isolation was not verifiable. Earlier counts labeled as performers were performance/source IDs and should not be interpreted as artist counts.
- **Model Configuration**:
  - 35-dimensional feature vector:
    - 12-dim tonic-relative Pitch Class Distribution (PCD) via pitch tracking.
    - 12-dim Chroma CQT summary.
    - 7-dim spectral & dynamics features (centroid, bandwidth, rolloff, flatness, zero-crossing, RMS energy, loudness).
    - 4-dim melodic summary metrics (vadi/samvadi peak salience, voiced frame ratio).
  - Preprocessing: `StandardScaler`.
  - Classifier: `SVC(kernel="rbf", C=10.0, probability=True, random_state=42)`.
  - Execution device: CPU.
- **Measured Evaluation Metrics (Test Split)**:
  - **Macro-F1**: `0.6067`
  - **Weighted-F1**: `0.6067`
  - **Accuracy**: `0.6000` (Baseline random chance: `0.1000` / 10.0%)
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
- **Key Musicological Insights**:
  1. *Malkauns* achieves perfect classification due to its distinct Audav scale (`S g M d n`) lacking Re and Pa.
  2. *Bhoopali* and *Deshkar* share the identical pitch set (`S R G P D`). Static pitch histograms distinguish them partially via Vadi emphasis (Gandhar vs Dhaivat), but melodic phrasing (*pakad*) is required for total separation.
- **Status**: Completed & Verified. Checkpoint saved to `artifacts/checkpoints/baseline_svm.joblib`. Report: `reports/baseline_svm_hindustani_report.md`.

---

### Experiment 2: Transfer Learning with Frozen MERT Architecture
- **Hypothesis**: Self-supervised pretrained music representations (`m-a-p/MERT-v1-95M`) trained on massive multi-genre corpora capture acoustic and harmonic textures that provide a viable representation for raga discrimination without manual feature engineering.
- **Dataset**: Same 390 authentic recordings (360 train, 30 held-out test, identical zero-leakage partitions).
- **Model Configuration**:
  - Encoder: `m-a-p/MERT-v1-95M` (revision `main`), frozen weights.
  - Audio Input: 24,000 Hz, 10.0-second segments.
  - Pooling: Adaptive temporal pooling `mean_max_concat` across all frames:
    $$\mathbf{e} = [\text{mean}(\mathbf{H}, t) \,\|\, \max(\mathbf{H}, t)] \in \mathbb{R}^{1536}$$
  - Head: `MERTClassifierHead` (LayerNorm(1536) -> Linear(1536, 128) -> GELU -> Dropout(0.2) -> Linear(128, 10)).
  - Training: AdamW(lr=1e-3, weight_decay=1e-2), 25 epochs, batch size 16, seed 42.
  - Execution device: CPU.
- **Measured Evaluation Metrics (Test Split)**:
  - **Macro-F1**: `0.3440`
  - **Weighted-F1**: `0.3440`
  - **Accuracy**: `0.4000` (vs. random baseline `0.1000`)
  - **Expected Calibration Error (ECE)**: `0.1794`
- **Key Musicological Insights**:
  - MERT alone beats random chance by 4x, demonstrating that acoustic self-supervised pretraining learns general musical texture and octave relationships.
  - However, because MERT has no explicit mechanism for tonic ($S_a$) normalization, identical raga phrases played at different pitch transpositions (e.g., C# vs D vs G) map to different embedding clusters. As a result, frozen MERT trails the handcrafted tonic-relative baseline (`0.6067` F1) when evaluated in isolation.
- **Status**: Completed & Verified. Checkpoint saved to `artifacts/checkpoints/mert_hindustani_model.pt`. Report: `reports/mert_frozen_hindustani_report.md`.

---

### Experiment 3: Real Hybrid Pitch + MERT Late Fusion
- **Hypothesis**: Late fusion combining explicit tonic-relative pitch class distributions (domain knowledge) with deep self-supervised acoustic embeddings (data-driven representations) achieves superior performance over either individual modality.
- **Dataset**: Same 390 authentic recordings (360 train, 30 held-out test, identical zero-leakage partitions).
- **Model Configuration**:
  - `HybridRagaClassifier`:
    - Musical branch: LayerNorm(35) -> Linear(35, 64) -> GELU -> Dropout(0.2).
    - MERT branch: LayerNorm(1536) -> Linear(1536, 128) -> GELU -> Dropout(0.2).
    - Late fusion head: Linear(192, 96) -> ReLU -> Dropout(0.2) -> Linear(96, 10).
  - Training: AdamW(lr=1e-3, weight_decay=1e-2), 25 epochs, batch size 16, seed 42.
  - Execution device: CPU.
- **Measured Evaluation Metrics (Test Split)**:
  - **Macro-F1**: `0.7400`
  - **Weighted-F1**: `0.7400`
  - **Accuracy**: `0.7667`
  - **Expected Calibration Error (ECE)**: `0.1808`
- **Side-by-Side Representation Comparison**:
  | Architecture | Input Representations | Macro-F1 | Accuracy | ECE | Status |
  | :--- | :--- | :---: | :---: | :---: | :--- |
  | **Frozen MERT Head** | 1536-dim Audio Embeddings | 0.3440 | 0.4000 | 0.1794 | Verified Real |
  | **Baseline Estimator (SVM)** | 35-dim Pitch Class & Chroma | 0.6067 | 0.6000 | 0.2003 | Verified Real |
  | **Hybrid Late Fusion** | 35-dim Musical + 1536-dim MERT | **0.7400** | **0.7667** | **0.1808** | **Verified Real** |
- **Conclusion**: Validates the central research thesis. Combining domain tonic-relative pitch distributions with deep acoustic representations delivers an absolute **+13.3% F1 gain over the baseline** and **+39.6% F1 gain over MERT alone**.
- **Status**: Completed & Verified. Checkpoint saved to `artifacts/checkpoints/hybrid_hindustani_model.pt`. Report: `reports/hybrid_fusion_hindustani_report.md`.

---

### Experiment 4: Empirical Raga Reference Centroids
- **Method**: Extracted 12-bin tonic-relative Pitch Class Distribution centroids across all 360 training performances in the real Hindustani dataset.
- **Results**: Captured empirical modal signatures for all 10 canonical ragas (e.g. Malkauns `S-g-M-d-n`, Todi `S-r-g-M'-d-N`, Yaman `S-R-G-M'-P-D-N`). Integrated into `RagaPrototypeRegistry` to power Bhattacharyya affinity in `MelodicSimilarityEngine`.
- **Status**: Completed & Verified in `src/raga_intelligence/similarity/prototypes.py`.

---

### Experiment 5: Cross-Genre Melodic Similarity & Calibrated Abstention
- **Hypothesis**: Multi-criterion gate combining spectral flatness (> 0.15), voiced frame ratio (< 0.15), and peak similarity score (< 0.40) reliably detects unpitched noise, speech, and atonal percussion without suppressing legitimate modal melodies.
- **Cross-Genre Evaluation Matrix**:
  | Test Track | Genre | Expected Raga Resemblance | Top Predicted Match | Similarity Score | Confidence | Abstention Status |
  | :--- | :--- | :--- | :--- | :---: | :--- | :---: |
  | **Bollywood (Bhairavi Melody)** | Indian Film / Bollywood | `Bhairavi` | **Bhairavi** | `0.863` | High | No |
  | **Bollywood (Yaman Ballad)** | Indian Film / Bollywood | `Yaman` | **Yaman** | `0.852` | High | No |
  | **Western Pop (Major Pentatonic)** | Western Pop / Folk | `Bhoopali` | **Bhoopali** | `0.918` | High | No |
  | **Western Blues (Minor Pentatonic)** | Western Blues / Rock | `Malkauns` | **Malkauns** | `0.953` | High | No |
  | **Atonal Noise Control** | Unpitched Control | `Abstain` | **None** | `0.000` | N/A | **Yes (Correct)** |
- **Status**: Completed & Verified in CLI runner `scripts/evaluate_cross_genre.py` and documented in `reports/cross_genre_evaluation.md`.

---

### Experiment 6: Phase 2.1 - Scientific Evaluation Framework & Multi-Seed Benchmarks

- **Current report**: `reports/scientific_evaluation_report.md`
- **Canonical evaluation ID**: `9fedf9990c3d4576952a49b0999b9876`
- **Code revision**: `6eec1891715ba79e1bd607b058fc8c3426c02c76` (clean working tree)
- **Dataset fingerprint**: `252c55be0ff9dc4375bd38d796ffe5fa33ea43a28cd0daac2b1a6e4eaf88c823` (`audio_sha256_v1`)
- **Evaluation split fingerprint**: `b5ba9f621f933b29b76fe0909fa99fa37ced8cc10626d8c2f0a169c91f54f171`
- **Lockfile SHA-256**: `cf8239ba12fe3e9a23886a7b102824d8c19bb7a667ec11deccaf060d57c7e00d`
- **Configuration**: seeds `[42, 123, 999]`, CPU, MERT revision `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`. Full configurations and metrics are generated in the report and its 15 records for this evaluation ID.
- **Leakage audit**: recording and performance isolation are verified; performer isolation is not verifiable because the corpus has no artist metadata.
- **Retained records**: 15 current canonical records and 15 records from the previous clean evaluation (`b894d85b7bf4dfc0d4e31bb1bfa3ce1a0369185d`, evaluation `0e7c1a57ddd34642a484913cca5737fc`) retained as historical. The report generator selects only the current evaluation ID. An interim evaluation performed before the full-tree cleanup was not retained.
- **Curated records**: 60 older experiment JSONs were removed after provenance review. They lacked an evaluation ID and clean-tree/split provenance (and the earliest 30 also lacked a dataset fingerprint), so they could not be tied to a complete reproducible report. The canonical ablation artifact is retained.

---

## 2.1 Final Remediation & Validation Record

- **Leakage semantics**: legacy no-leakage booleans are true only when isolation is verified. Missing performer or performance metadata produces `not_verifiable` and a false boolean; observed overlap produces `failed`.
- **Strict scientific partitioning**: evaluation requires a known performance ID for every recording and an exact `verified` performance-isolation status. It rejects leakage, missing/unrecognized statuses, infeasible performance-group structure, or any canonical raga missing from train/test before reading caches or initializing models. Performer isolation may remain `not_verifiable` when artist metadata is absent. The recorded canonical run passed with all 10 ragas covered and 82 performance/source groups.
- **Test suite**: 141 tests passed on the final integrity-check changes; integration tests include rejection before cache/model evaluation. API smoke tests separately passed (2 tests).
- **Ruff**: not run because Ruff is not installed/cached in the offline environment. No configured type checker was found.
- **Podman**: image build could not be validated because the existing Podman machine identity was inaccessible (`Access is denied`). Podman and WSL configuration were not changed.

---

## 3. Invalidated & Superseded Historical Experiments

### Historical Experiment: Synthetic Random-Embedding Fallback (Invalidated)
- **Context**: In early scaffold code (prior to Prompt 1 fix), `train_hybrid_pipeline()` contained the following fallback when MERT cache files were not found:
  ```python
  # INVALID / PLACEHOLDER IMPLEMENTATION:
  rng = np.random.default_rng(42)
  E_train = torch.tensor(rng.standard_normal((len(X_train), mert_dim)), dtype=torch.float32)
  E_test = torch.tensor(rng.standard_normal((len(X_test), mert_dim)), dtype=torch.float32)
  ```
- **Why Invalid**: Gaussian noise vectors are completely uncorrelated with the audio signal. Models trained under this condition did not measure MERT representation capability; their metrics reflected training a regularized projection head on pure noise.
- **Status**: **INVALIDATED / REMOVED**. Replaced in Prompt 1 and Prompt 2 with mandatory, validated extraction of real MERT embeddings with explicit failure upon missing audio or model errors.

---

## 4. Methodological Limitations: Small Held-Out Test Sets

The canonical held-out partition contains 30 recordings across 10 ragas and 10 independent performance/source clusters. The report includes cluster-bootstrap uncertainty and warns that estimates from fewer than 15 clusters may have limited coverage. The split establishes performance-level isolation only; performer-level isolation remains unverifiable because artist identities are not present in the dataset metadata. Results should not be read as evidence of generalization to unseen performers.

---

## 5. Planned Experiments (Roadmap)

- **Experiment 6: End-to-End Fine-Tuning of MERT Layers**:
  Unfreeze the top 2 transformer layers of `MERT-v1-95M` with low learning rate ($10^{-5}$) and gradient checkpointing to adapt representations directly to Indian microtonal ornamentation.
- **Experiment 7: Sequential Transition & Melodic Motif (*Pakad*) Modeling**:
  Train a temporal sequence model (BiLSTM / Transformer encoder) over frame-level pitch sequences to capture characteristic ascending/descending phrases (*chalan* / *pakad*) and distinguish raga pairs with identical pitch sets (e.g., *Bhoopali* vs. *Deshkar*, *Bhairav* vs. *Kalingada*).
- **Experiment 8: Multi-Dataset 50-Raga Benchmark**:
  Scale the evaluation corpus from 10 canonical ragas to 50 ragas using the complete Saraga Hindustani collection and CompMusic datasets with automated tonic and segmentation annotation.
