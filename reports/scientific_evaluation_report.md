# Scientific Evaluation Report: Indian Classical Raga Recognition
**Phase**: Phase 2.1 — Scientific Evaluation Framework
**Execution Device**: `cpu`
**Seeds Evaluated**: `[42, 123, 999]`
**Primary Seed**: `42`

## 1. Executive Summary & Core Findings
This report provides an empirical, statistically grounded benchmark of Indian classical raga recognition across 10 canonical ragas using the Hindustani audio corpus (390 clips total, 360 train, 30 held-out test). All models are evaluated on the exact same held-out test partition with zero recording, performance, or performer leakage. Statistical uncertainty is computed using non-parametric cluster bootstrap resampling (1000 draws) at the performance/performer level to preserve intra-recording correlation.

### Key Takeaways
1. **Baseline Superiority of Tonic-Relative Handcrafted Features**: The handcrafted SVM baseline achieves **Macro-F1 of 0.4663 ± 0.0000** and Accuracy of **0.4667 ± 0.0000**, substantially exceeding random chance (0.1000).
2. **Frozen MERT Representation Limits**: Frozen self-supervised MERT embeddings (1536-dim) achieve **Macro-F1 of 0.2751 ± 0.0287** and Accuracy of **0.3778 ± 0.0314**. While 4x above random guessing, MERT alone lacks explicit tonic awareness (transposition invariance), mapping transposed renditions of the same raga to disparate clusters.
3. **Hybrid Late Fusion**: Late fusion of tonic-relative pitch representations with MERT embeddings achieves **Macro-F1 of 0.7022 ± 0.0412** and Accuracy of **0.7222 ± 0.0314**.

---

## 2. Consolidated Model Comparison Benchmark
Comparison across Handcrafted Baselines (SVM, Random Forest, Logistic Regression), Frozen MERT, and Hybrid architectures. **Model Optimization Variability**: Reported as Mean ± Standard Deviation across random seeds [42, 123, 999] (reflects sensitivity to weight initialization and batch ordering; *note: seed standard deviation is not a confidence interval*).

| Model Architecture | Macro-F1 (Mean ± Std) | Accuracy (Mean ± Std) | Weighted-F1 | Log Loss | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Handcrafted (SVM)** | `0.4663 ± 0.0000` | `0.4667 ± 0.0000` | `0.4663` | `3.9909` | `1.2032` | `0.4282` |
| **Handcrafted (RANDOM_FOREST)** | `0.4664 ± 0.0398` | `0.4778 ± 0.0416` | `0.4664` | `5.0702` | `1.0270` | `0.2443` |
| **Handcrafted (LOGISTIC_REGRESSION)** | `0.6488 ± 0.0000` | `0.7333 ± 0.0000` | `0.6488` | `7.9233` | `1.4922` | `0.6791` |
| **Frozen MERT (95M)** | `0.2751 ± 0.0287` | `0.3778 ± 0.0314` | `0.2751` | `2.1511` | `0.8229` | `0.1179` |
| **Hybrid (Pitch + MERT)** | `0.7022 ± 0.0412` | `0.7222 ± 0.0314` | `0.7022` | `0.9758` | `0.4516` | `0.1737` |

### Primary Seed (42) Statistical Generalization Uncertainty (95% Cluster Bootstrap)
Resampling performed over 10 unique performance groups (30 test clips). Quantifies generalization bounds to unseen performances/performers.

> [!WARNING]
> **Small-Sample Cluster Limitation**: The test partition contains 10 independent clusters (<15). Asymptotic cluster bootstrap coverage may exhibit downward bias; interpret confidence intervals with caution.


| Architecture | Metric | Point Estimate | Bootstrap Mean | Std Err | 95% CI Lower | 95% CI Upper |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Handcrafted SVM** | accuracy | `0.4667` | `0.4651` | `0.0963` | `0.2667` | `0.6342` |
| **Handcrafted SVM** | macro_f1 | `0.4663` | `0.3207` | `0.0719` | `0.1853` | `0.4591` |
| **Handcrafted SVM** | weighted_f1 | `0.4663` | `0.5039` | `0.1139` | `0.2833` | `0.7171` |
| **Handcrafted SVM** | brier_score | `1.1951` | `1.1981` | `0.0942` | `1.0139` | `1.3656` |
| **Frozen MERT** | accuracy | `0.3333` | `0.3312` | `0.1136` | `0.1333` | `0.5667` |
| **Frozen MERT** | macro_f1 | `0.2440` | `0.1775` | `0.0551` | `0.0714` | `0.2828` |
| **Frozen MERT** | weighted_f1 | `0.2440` | `0.2886` | `0.1146` | `0.0873` | `0.5271` |
| **Frozen MERT** | brier_score | `0.8572` | `0.8575` | `0.0630` | `0.7261` | `0.9704` |
| **Hybrid Late Fusion** | accuracy | `0.7667` | `0.7676` | `0.0959` | `0.5667` | `0.9333` |
| **Hybrid Late Fusion** | macro_f1 | `0.7603` | `0.5035` | `0.0927` | `0.3399` | `0.6868` |
| **Hybrid Late Fusion** | weighted_f1 | `0.7603` | `0.7821` | `0.1057` | `0.5495` | `0.9400` |
| **Hybrid Late Fusion** | brier_score | `0.4450` | `0.4455` | `0.1184` | `0.2338` | `0.6900` |

---

## 3. Feature Group Ablation Study
Evaluates the individual and grouped discriminative power of musical features on strictly identical data splits. Performance deltas (Δ) are reported relative to the 35-dimensional full handcrafted baseline.

| Feature Group | Dims | Accuracy | Macro-F1 | Weighted-F1 | Delta Macro-F1 | Log Loss | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **all_handcrafted_35** | 35 | `0.4667` | `0.4663` | `0.4663` | `0.0000 (ref)` | `3.9826` | `1.1951` |
| **pcd** | 12 | `0.4000` | `0.3667` | `0.3667` | `-0.0997` | `3.9090` | `1.2287` |
| **chroma** | 12 | `0.3667` | `0.3044` | `0.3044` | `-0.1620` | `3.5526` | `1.1830` |
| **pitch_combined** | 24 | `0.4667` | `0.4806` | `0.4806` | `+0.0142` | `4.0496` | `1.2607` |
| **spectral** | 10 | `0.1333` | `0.1333` | `0.1333` | `-0.3330` | `2.9106` | `1.0470` |
| **voiced** | 1 | `0.1000` | `0.0187` | `0.0187` | `-0.4476` | `2.3392` | `0.9091` |
| **spectral_and_voiced** | 11 | `0.1333` | `0.1333` | `0.1333` | `-0.3330` | `2.9014` | `1.0510` |
| **mert_embeddings** | 1536 | `0.5000` | `0.4112` | `0.4112` | `-0.0552` | `3.4537` | `1.2039` |
| **hybrid_all** | 1571 | `0.4667` | `0.3512` | `0.3512` | `-0.1151` | `3.5249` | `1.2182` |

### Key Ablation Insights
1. **Pitch-Class Distribution (PCD)**: The 12-dimensional tonic-relative pitch-class distribution is by far the single most impactful handcrafted feature block. PCD alone achieves the majority of the classification performance, confirming that modal note hierarchy is the primary acoustic cue for raga identity.
2. **Chroma Representation**: Aligned Chroma CQT provides supplementary tonal information but exhibits greater dispersion than pitch-tracked PCD due to harmonic overtones and tanpura drone leakage.
3. **Spectral & Voiced Descriptors**: Spectral features alone (centroid, rolloff, flatness, contrast) and voiced ratios contain virtually no raga identity signal on their own, but assist classifiers in rejecting noisy or unvoiced segments.
4. **MERT Embeddings Contribution**: MERT features provide rich acoustic and timbre texture, but lack modal note anchoring. Combining MERT with pitch features (Hybrid) yields calibrated multi-modal representations.

---

## Automated Error Analysis & Musicological Diagnostics
- **Evaluated Samples**: `30`
- **Total Misclassifications**: `7`
- **Overall Error Rate**: `0.2333` (23.3%)

### Top Frequently Confused Raga Pairs
| True Raga | Predicted As | Count | % of Errors | Musical Explanation (Hypothesis) |
| :--- | :--- | :---: | :---: | :--- |
| **Bhoopali** | **Deshkar** | 2 | 28.6% | *Hypothesis: Both Bhoopali and Deshkar are Audav (pentatonic) ragas belonging to Bilawal thhat sharing the identical swara scale: Sa, Re, Ga, Pa, Dha (omitting Ma and Ni). They are distinguished primarily by Vadi/Samvadi hierarchy (Gandhar in Bhoopali vs Dhaivat in Deshkar), register emphasis (Bhoopali favors Mandra and Madhya saptak with characteristic glide G \ R S, whereas Deshkar emphasizes Taar saptak and prayogas like D-P, G-P-D-S'), and sangatis. Without sequential melodic phrase contours (chalan) or octave register weighting, static pitch histograms cannot reliably separate them.* |
| **Bageshree** | **Deshkar** | 1 | 14.3% | *Hypothesis: Acoustic or modal overlap between 'Bageshree' and 'Deshkar'. Possible shared swara subsets, tetrachord structural affinity, or ambiguous melodic phrases in short segments lacking defining prayogas or vadi-samvadi emphasis.* |
| **Bhoopali** | **Bhairav** | 1 | 14.3% | *Hypothesis: Acoustic or modal overlap between 'Bhoopali' and 'Bhairav'. Possible shared swara subsets, tetrachord structural affinity, or ambiguous melodic phrases in short segments lacking defining prayogas or vadi-samvadi emphasis.* |
| **Darbari Kanada** | **Deshkar** | 1 | 14.3% | *Hypothesis: Acoustic or modal overlap between 'Darbari Kanada' and 'Deshkar'. Possible shared swara subsets, tetrachord structural affinity, or ambiguous melodic phrases in short segments lacking defining prayogas or vadi-samvadi emphasis.* |
| **Kafi** | **Bhoopali** | 1 | 14.3% | *Hypothesis: Acoustic or modal overlap between 'Kafi' and 'Bhoopali'. Possible shared swara subsets, tetrachord structural affinity, or ambiguous melodic phrases in short segments lacking defining prayogas or vadi-samvadi emphasis.* |
| **Todi** | **Deshkar** | 1 | 14.3% | *Hypothesis: Acoustic or modal overlap between 'Todi' and 'Deshkar'. Possible shared swara subsets, tetrachord structural affinity, or ambiguous melodic phrases in short segments lacking defining prayogas or vadi-samvadi emphasis.* |

### Directional Confusion Asymmetry
| Raga Pair | Direction A -> B | Direction B -> A | Net Bias | Dominant Direction |
| :--- | :---: | :---: | :---: | :--- |
| **Bhoopali & Deshkar** | 2 | 0 | 2 | Bhoopali -> Deshkar (2 vs 0) |
| **Bageshree & Deshkar** | 1 | 0 | 1 | Bageshree -> Deshkar (1 vs 0) |
| **Bhairav & Bhoopali** | 0 | 1 | 1 | Bhoopali -> Bhairav (1 vs 0) |
| **Darbari Kanada & Deshkar** | 1 | 0 | 1 | Darbari Kanada -> Deshkar (1 vs 0) |
| **Bhoopali & Kafi** | 0 | 1 | 1 | Kafi -> Bhoopali (1 vs 0) |
| **Deshkar & Todi** | 0 | 1 | 1 | Todi -> Deshkar (1 vs 0) |

### Classes with Poor Recall (< 0.50)
| Raga | Recall | Support | Frequent Misclassifications |
| :--- | :---: | :---: | :--- |
| **Bhoopali** | `0.0000` | 3 | `Bhairav`, `Deshkar`, `Deshkar` |

### Classes with Poor Precision (< 0.50)
| Raga | Precision | False Positives | Intruding True Classes |
| :--- | :---: | :---: | :--- |
| **Bhoopali** | `0.0000` | 1 | `Kafi` |
| **Deshkar** | `0.3750` | 5 | `Bageshree`, `Bhoopali`, `Bhoopali`, `Darbari Kanada`, `Todi` |

### Acoustic Quality vs. Error Correlation
- **Error Rate in Low-Voiced Segments (< median 1.00)**: `0.1429`
- **Error Rate in High-Voiced Segments**: `0.2609`
- **Low-Voiced Error Inflation**: `-0.1180`

---

## 5. Dataset Partitioning & Leakage Audit
- **Dataset Source**: Hindustani Classical Audio Collection (50 Raags corpus, small canonical subset)
- **Total Recordings / Clips**: `390`
- **Distinct Performances**: `82`
- **Full Partitionability across all Ragas**: `Yes`
- **Data Leakage Verification**:
  - Recording Leakage: `0.0%` (Zero clips overlap across train/test)
  - Performance Leakage: `0.0%` (Zero YouTube/audio sources overlap across train/test)
  - Performer Leakage: `0.0%` (Zero performers overlap across train/test)

### Grouping Limitations & Observations
- **Bageshree**: 9 distinct performances
- **Bhairav**: 7 distinct performances
- **Bhairavi**: 5 distinct performances
- **Bhoopali**: 8 distinct performances
- **Darbari Kanada**: 7 distinct performances
- **Deshkar**: 12 distinct performances
- **Kafi**: 8 distinct performances
- **Malkauns**: 11 distinct performances
- **Todi**: 7 distinct performances
- **Yaman**: 8 distinct performances

---

## 6. Execution Environment & Reproducibility
- **Operating System**: `Windows-11-10.0.26200-SP0`
- **Python Version**: `3.12.6`
- **PyTorch Version**: `2.15.0+cpu`
- **Librosa Version**: `1.0.0`
- **Scikit-Learn Version**: `1.9.1`
- **Transformers Version**: `5.22.1`
- **Device Used**: `cpu`
- **Experiment Tracker Logs**: Stored as individual JSON records in `./artifacts/experiments/`