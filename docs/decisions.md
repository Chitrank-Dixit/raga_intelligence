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
- **Context**: Indian classical music evaluation requires authentic audio recordings with verified raga annotations and tonic ground truth. Performer isolation can only be claimed when genuine artist metadata is supplied; source/video IDs do not establish artist identity.
- **Decision**:
  - Adopt the MTG CompMusic research corpus, specifically **Saraga Hindustani** (Zenodo Record `4301737`, licensed under CC BY-NC-SA 4.0) and IAM Raga Recognition Features (`7278505`).
  - Implement a modular `DatasetAdapter` interface supporting both local audio directories, Zenodo research archives, and synthetic test fixtures.
  - Enforce recording and performance/source split integrity. Enforce artist-level isolation when artist metadata is available and otherwise report that level as unverifiable.
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

## ADR 005: Empirical Centroid Extraction over Theoretical Pitch Histograms
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**: Theoretical pitch distributions synthesized purely from textbook note sets and vadi/samvadi heuristics fail to capture natural human performance variance, intonation nuances, and habitual transitional notes (*sangatis*).
- **Decision**:
  - Compute 12-bin tonic-relative Pitch Class Distribution (PCD) centroids directly from the 360 training recordings of the authentic Hindustani corpus.
  - Populate empirical centroids into `RagaPrototypeRegistry` as the primary reference vector (`effective_pcd`), retaining theoretical distributions as fallback.
- **Consequences**: Melodic similarity accurately matches empirical performance practices and improves Bhattacharyya distance discrimination on real audio.

## ADR 006: Multi-Criterion Acoustic Gate for Out-of-Distribution Abstention
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**: Audio inputs submitted to a public API may consist of spoken language, field noise, unpitched percussion, or synthesizer noise without modal tonicity.
- **Decision**:
  - Enforce a 3-stage hierarchical gate prior to raga matching:
    1. **Spectral Flatness Gate**: Reject audio with Wiener entropy > 0.15 (indicative of white noise, unpitched percussion, or hiss).
    2. **Voiced Ratio Gate**: Reject audio with voiced frame proportion < 0.15 (indicative of silence, ambient noise, or spoken dialogue).
    3. **PCD Variance Gate**: Reject audio with pitch distribution standard deviation < 0.025 (indicative of uniform atonal noise).
- **Consequences**: Completely suppresses spurious raga predictions on noise and spoken audio, delivering transparent diagnostic reasons without false classifications.

