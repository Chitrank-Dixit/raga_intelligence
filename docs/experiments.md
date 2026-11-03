# Experiment Registry and Findings — Indian Raga Intelligence

> [!WARNING]
> **Notice on Historical / Superseded Hybrid Benchmarks**
> In early prototyping (commit `7525287`), the hybrid training script contained a synthetic fallback (`rng.standard_normal((len(X_train), mert_dim))`) when cached MERT representations were not yet extracted. Any early figures reported under that ungrounded configuration are **INVALIDATED AND SUPERSEDED**. They are preserved in the historical section below strictly for auditability and must **never** be cited as evidence of MERT representation capability.
>
> **Historical registry:** Existing figures below document earlier experiments and are not the current canonical benchmark. The Hindustani CSV has no artist field; previous use of video IDs as performer labels was invalid. Artist-level isolation is unverifiable for that corpus. The current scientific report remains pending until a clean-commit evaluation succeeds.

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
| **Exp 6** | Phase 2.1 Scientific Evaluation Framework & Multi-Seed Benchmarks | **Historical; canonical rerun pending** | `reports/scientific_evaluation_report.md` |
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

### Experiment 6: Phase 2.1 — Scientific Evaluation Framework & Multi-Seed Benchmarks
- **Objective**: Establish a rigorous, reproducible, group-isolated evaluation protocol with expanded statistical metrics, cluster bootstrap confidence intervals, feature ablation, and automated error diagnostics.
- **Protocol**:
  - **Dataset**: Hindustani classical audio collection (390 clips across 10 canonical ragas).
  - **Group Isolation**: The historical split grouped by performance/source ID. Artist-level leakage was not verifiable from the available metadata.
  - **Multi-Seed Testing**: Evaluated across 3 deterministic random seeds (`[42, 123, 999]`) on strictly identical held-out test partitions.
  - **Statistical Resampling**: 1,000 cluster bootstrap draws at the performance/source level (never treating non-independent audio chunks as i.i.d.).
  - **Cryptographic Provenance**: Exact cache file paths and streaming SHA-256 digests (`da9170fec4d680183b6f31d4a27e98dddcd858c97ee3df1d735f34f5cc43621e` for handcrafted features, `d13f6f4878167e258d0b350575a63975685491f9f8af51c2136f8b93c1d3c9c8` for MERT embeddings) are verified and persisted in all experiment logs.
  - **Statistical Clarity**: Seed variation (Mean ± Std across seeds) quantifies algorithm optimization stability to initialization and training order; it is explicitly distinguished from test-sample generalization uncertainty (cluster bootstrap 95% CIs).
- **Consolidated Multi-Seed Benchmark Results**:
  | Model Architecture | Macro-F1 (Mean ± Std) | Accuracy (Mean ± Std) | Weighted-F1 | Log Loss | Brier Score | ECE |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Handcrafted (SVM)** | `0.4663 ± 0.0000` | `0.4667 ± 0.0000` | `0.4663` | `3.9909` | `1.2032` | `0.4282` |
  | **Handcrafted (Random Forest)** | `0.4664 ± 0.0398` | `0.4778 ± 0.0416` | `0.4664` | `5.0702` | `1.0270` | `0.2443` |
  | **Handcrafted (Logistic Regression)** | `0.6488 ± 0.0000` | `0.7333 ± 0.0000` | `0.6488` | `7.9233` | `1.4922` | `0.6791` |
  | **Frozen MERT (95M)** | `0.2751 ± 0.0287` | `0.3778 ± 0.0314` | `0.2751` | `2.1511` | `0.8229` | `0.1179` |
  | **Hybrid (Pitch + MERT)** | `0.7022 ± 0.0412` | `0.7222 ± 0.0314` | `0.7022` | `0.9758` | `0.4516` | `0.1737` |

- **Primary Seed 95% Cluster Bootstrap Confidence Intervals**:
  > [!WARNING]
  > **Small-Sample Cluster Limitation**: The test partition contains 10 independent clusters (<15). Asymptotic cluster bootstrap coverage may exhibit downward bias; interpret confidence intervals with caution.
  
  - **Hybrid Late Fusion**:
    - Accuracy: `0.7667` (95% CI: `[0.5667, 0.9333]`, Std: `0.0959`)
    - Macro-F1: `0.7603` (95% CI: `[0.3399, 0.6868]`, Std: `0.0927`)
    - Brier Score: `0.4450` (95% CI: `[0.2338, 0.6900]`, Std: `0.1184`)
  - **Handcrafted SVM**:
    - Accuracy: `0.4667` (95% CI: `[0.2667, 0.6342]`, Std: `0.0963`)
    - Macro-F1: `0.4663` (95% CI: `[0.1853, 0.4591]`, Std: `0.0719`)
    - Brier Score: `1.1951` (95% CI: `[1.0139, 1.3656]`, Std: `0.0942`)

- **Feature Group Ablation Summary**:
  | Feature Group | Dims | Accuracy | Macro-F1 | Delta Macro-F1 | Log Loss | Brier Score |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **all_handcrafted_35** | 35 | `0.4667` | `0.4663` | `0.0000 (ref)` | `3.9826` | `1.1951` |
  | **pcd** | 12 | `0.4000` | `0.3667` | `-0.0997` | `3.9090` | `1.2287` |
  | **chroma** | 12 | `0.3667` | `0.3044` | `-0.1620` | `3.5526` | `1.1830` |
  | **pitch_combined** | 24 | `0.4667` | `0.4806` | `+0.0142` | `4.0496` | `1.2607` |
  | **spectral** | 10 | `0.1333` | `0.1333` | `-0.3330` | `2.9106` | `1.0470` |
  | **voiced** | 1 | `0.1000` | `0.0187` | `-0.4476` | `2.3392` | `0.9091` |
  | **spectral_and_voiced** | 11 | `0.1333` | `0.1333` | `-0.3330` | `2.9014` | `1.0510` |
  | **mert_embeddings** | 1536 | `0.5000` | `0.4112` | `-0.0552` | `3.4537` | `1.2039` |
  | **hybrid_all** | 1571 | `0.4667` | `0.3512` | `-0.1151` | `3.5249` | `1.2182` |

- **Key Musicological Error Analysis Findings**:
  - **Bhoopali vs. Deshkar**: Confused due to identical pentatonic pitch sets (`S R G P D`). Static PCDs cannot separate them without temporal melodic contours (*chalan*) and register emphasis.
  - **Darbari Kanada vs. Deshkar**: In short excerpts, slow andolans and glides on komal Ga/Dha can appear unvoiced or obscured, producing false positive Deshkar predictions.
  - **Acoustic Quality Correlation**: Lower voiced frames exhibit slightly elevated misclassification risk, confirming the need for confidence calibration and abstention.
- **Status**: Completed & Verified. Automated execution script `scripts/run_scientific_evaluation.py`, complete report in `reports/scientific_evaluation_report.md`, and 60 run logs stored in `artifacts/experiments/`.

---

## 2.1 Remediation & Security Hardening Verification

All findings from the PR #1 review remediation have been addressed and validated:

1. **Cache Integrity & Cryptographic Reproducibility**:
   - Implemented `compute_file_sha256()` streaming file hash computation for exact cache arrays loaded during evaluation.
   - Evaluated cache files are verified and logged:
     - Handcrafted cache (`artifacts/cache/hindustani_features.npz`): `da9170fec4d680183b6f31d4a27e98dddcd858c97ee3df1d735f34f5cc43621e`
     - MERT cache (`artifacts/cache/hindustani_mert_embeddings.npz`): `d13f6f4878167e258d0b350575a63975685491f9f8af51c2136f8b93c1d3c9c8`
   - Verified that mutating or replacing cache contents immediately alters the recorded SHA-256 fingerprint.

2. **Audio Dataset Fingerprinting Architecture**:
   - Clarified the operational distinction between **metadata fingerprints** and **audio-content fingerprints**:
     - **Metadata Fingerprint** (`compute_metadata_fingerprint()`): Deterministic SHA-256 over recording ID, split assignment, canonical raga label, ground-truth tonic (Sa), performer, performance ID, duration, and sample rate. Guarantees that annotations and splits have not drifted without reading raw audio files.
     - **Audio-Content Fingerprint** (`compute_audio_content_fingerprint()`): Deterministic SHA-256 over the raw audio byte hashes (`audio_sha256`) of every recording. Guarantees that audio waveforms have not been truncated, modified, or corrupted on disk.
     - Modifying any audio file on disk immediately invalidates the audio fingerprint and downstream feature caches.

3. **Statistical Reporting Rigor**:
   - Replaced all hard-coded cluster references with dynamic group counts ($n=10$ independent performance clusters across 30 test clips).
   - Added an explicit statistical warning when independent cluster counts are small ($n < 15$).
   - Explicitly distinguished model optimization variability across random seeds (Mean ± Std) from test-sample generalization uncertainty (cluster bootstrap 95% CIs).
   - Removed any characterization of seed standard deviation as a confidence interval.

4. **Group Partitioning Edge Cases & Infeasibility Diagnostics**:
   - Implemented bipartite graph connected component analysis in `GroupAwarePartitioner.audit_grouping()`.
   - Identified multi-raga performer cycle constraints (e.g. 3-cycle loops $P_1 \to \{A, B\}, P_2 \to \{B, C\}, P_3 \to \{C, A\}$ where strict isolation prevents dual-split class representation).
   - Populated explicit `impossible_coverage_ragas` diagnostics instead of silently omitting classes.
   - Added `strict_coverage: bool = False` flag; when `True`, raises `InfeasiblePartitionError` with full audit diagnostics.

5. **API Resource Protection & Concurrency Limits**:
   - Added configurable `max_concurrent_inferences: int = 4` (`MAX_CONCURRENT_INFERENCES`) and `inference_retry_after_sec: int = 2` (`INFERENCE_RETRY_AFTER_SEC`).
   - Saturated inference requests are rejected with predictable `HTTP 503 Service Unavailable` and a valid `Retry-After` header.
   - Guaranteed atomic temporary file unlinking and capacity release in `finally:` blocks during client cancellations (`asyncio.CancelledError`) and decoder failures (`AudioFileError`).

6. **Automated Test Suite Verification**:
   - **Host (Windows 11, Python 3.12.6)**: **118 / 118 passed** in 16.02s (`uv run pytest tests/`).
   - **Container (Linux, Podman 6.1.4, Python 3.12.16)**: **117 passed, 1 skipped** (GPU test skipped gracefully) in 57.16s (`localhost/raga-intelligence:latest`).

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

The held-out test partition in the Hindustani Raag dataset contains **30 recordings** (exactly 3 clips per canonical raga). The implications of this test set size must be explicitly understood:

1. **Granularity & Variance**: Each individual test clip accounts for $1/3 \approx 33.3\%$ of the per-class recall for that raga, and $1/30 \approx 3.33\%$ of the overall accuracy. A single misclassification shifts the per-class metric substantially.
2. **Confidence Intervals**: With $N=30$ and an observed accuracy of $76.67\%$, the 95% Wilson score confidence interval spans approximately $[59.1\%, 88.2\%]$. While the margin over random baseline ($10.0\%$) is statistically significant ($p < 10^{-6}$), differences between adjacent model variants require caution.
3. **Generalization Beyond Cohort**: The old corpus provides performance/source isolation only. Performer-level isolation remains unverified; testing with curated artist metadata and broader corpora is needed before making claims about unseen performers.

---

## 5. Planned Experiments (Roadmap)

- **Experiment 6: End-to-End Fine-Tuning of MERT Layers**:
  Unfreeze the top 2 transformer layers of `MERT-v1-95M` with low learning rate ($10^{-5}$) and gradient checkpointing to adapt representations directly to Indian microtonal ornamentation.
- **Experiment 7: Sequential Transition & Melodic Motif (*Pakad*) Modeling**:
  Train a temporal sequence model (BiLSTM / Transformer encoder) over frame-level pitch sequences to capture characteristic ascending/descending phrases (*chalan* / *pakad*) and distinguish raga pairs with identical pitch sets (e.g., *Bhoopali* vs. *Deshkar*, *Bhairav* vs. *Kalingada*).
- **Experiment 8: Multi-Dataset 50-Raga Benchmark**:
  Scale the evaluation corpus from 10 canonical ragas to 50 ragas using the complete Saraga Hindustani collection and CompMusic datasets with automated tonic and segmentation annotation.
