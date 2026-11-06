# Indian Raga Intelligence

Research-driven machine learning system for Indian classical raga recognition, tonic-normalized pitch-class distribution analysis, and cross-genre melodic similarity intelligence.

---

## 1. Project Mission & Overview

Indian classical music is fundamentally modal, organized around *ragas*—frameworks defined not merely by scale pitch sets, but by characteristic melodic motifs (*pakads*), specific ascending (*arohana*) and descending (*avarohana*) grammars, primary notes (*vadi* and *samvadi*), microtonal inflections (*shrutis*, *gamaks*), and temporal dynamics relative to a continuous drone tonic (*Shadja / Sa*).

This project develops an evidence-driven machine learning proof of concept (PoC) that:
1. **Performs modal analysis** on Hindustani classical vocal and instrumental performances.
2. **Measures melodic similarity** across cross-genre audio (including Bollywood and Western pop/rock).
3. **Calibrates confidence and supports abstention** ("Insufficient evidence / Non-raga melodic structure") to prevent false assertions.
4. **Compares explicit musical representations** (tonic-relative Pitch Class Distribution, Chroma CQT) with **deep learned audio representations** (MERT).

---

## 2. Core Principles & Design

- **Evidence-Driven Baseline First**: Establishes empirical baselines using tonic-normalized pitch-class distributions before transfer learning.
- **Split Integrity**: Splits are audited at recording and performance/source level. Artist isolation is reported as unverifiable unless genuine performer metadata is available.
- **Honest Predictions**: Explicit separation between *closed-world classical classification* and *open-world cross-genre melodic similarity*.
- **Container-First Workflow**: Local development and production containers configured via Podman and Compose.
- **Resource & Hardware Awareness**: Automatic detection and verified tensor execution for NVIDIA GPU passthrough, paired with a reliable CPU fallback.

---

## 3. Technology Stack

- **Runtime**: Python 3.12 (managed via `uv`).
- **Core ML & DSP**: PyTorch 2.6, Torchaudio, Transformers (MERT-v1-95M), Librosa, SoundFile, SciPy, Scikit-learn.
- **API & Application**: FastAPI, Pydantic v2, Uvicorn, Vanilla HTML5/JS responsive frontend.
- **Containers**: Podman 6.0.2 / 5.7.0, Podman Compose.

---

## 4. Hardware Verification & GPU Notes

- **Workstation GPU**: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM, Compute Capability 12.0 / `sm_120`).
- **Verified Status**: Standard PyTorch cu124 wheels compile kernels up to `sm_90`. When `sm_120` execution lacks compiled kernels, the system automatically falls back to CPU tensor execution without crashing.
- **Diagnostics**: Run `python scripts/diagnostics.py` to inspect hardware, decoders, and container tooling.

---

## 5. Quickstart & Local Execution

### Local Python Environment (via `uv`)
```powershell
# 1. Create and activate virtual environment
uv venv --python 3.12
.venv\Scripts\activate

# 2. Install dependencies and editable package
uv pip install -e ".[dev]"

# 3. Run environment diagnostics
python scripts/diagnostics.py

# 4. Run automated test suite
pytest -v

# 5. Start the API service
uvicorn raga_intelligence.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Containerized Execution (via Podman in WSL / Linux)
```bash
# Build the container image
podman-compose build

# Start the API service in background
podman-compose up -d

# View container logs
podman-compose logs -f

# Check health endpoint
curl http://localhost:8000/health

# Stop services
podman-compose down
```

---

## 6. Supported Canonical Hindustani Ragas

The initial research cohort covers 10 representative Hindustani ragas across diverse Thhats:

| Raga | Thhat | Arohana | Avarohana | Vadi / Samvadi |
| :--- | :--- | :--- | :--- | :--- |
| **Bhairav** | Bhairav | S r G M P d N S' | S' N d P M G r S | d / r |
| **Bhairavi** | Bhairavi | S r g M P d n S' | S' n d P M g r S | M / S |
| **Yaman** | Kalyan | 'N R G M' D N S' | S' N D P M' G R S | G / N |
| **Bhoopali** | Kalyan | S R G P D S' | S' D P G R S | G / D |
| **Deshkar** | Bilawal | S R G P D S' | S' D P G R S | D / G |
| **Bageshree** | Kafi | S g M D n S' | S' n D M g R S | M / S |
| **Malkauns** | Bhairavi | S g M d n S' | S' n d M g S | M / S |
| **Todi** | Todi | S r g M' P d N S' | S' N d P M' g r S | d / g |
| **Darbari Kanada** | Asavari | S R g M P d n S' | S' d n P M g M R S | R / P |
| **Kafi** | Kafi | S R g M P D n S' | S' n D P M g R S | P / S |

---

## 7. Research Datasets & Licensing

Dataset record fields and provenance semantics are defined in
[`docs/dataset_metadata_contract.md`](docs/dataset_metadata_contract.md). The
read-only dataset quality audit is documented in
[`docs/dataset_audit.md`](docs/dataset_audit.md).

- **Hindustani Raag Corpus**: 390 authentic recordings across 10 canonical ragas (`neerajaabhyankar/hindustani-raag-small`) with ground-truth hand-annotated Sa tonic (Hz). Partitioned with 0% recording leakage (360 train, 30 held-out test).
- **Saraga Hindustani**: Zenodo DOI `10.5281/zenodo.4301737` (CC BY-NC-SA 4.0).
- **CompMusic IAM Features**: Zenodo DOI `10.5281/zenodo.7278505` (CC BY-NC-ND 4.0).
- **Synthetic Test Fixtures**: Deterministically synthesized within `data/synthetic` under CC0-1.0 for reproducible CI testing without external network dependencies.

---

## 8. Empirical Benchmark Results

Benchmark execution revision: clean commit `6eec1891715ba79e1bd607b058fc8c3426c02c76` (evaluation `9fedf9990c3d4576952a49b0999b9876`). The PR branch HEAD before this final review change set is `71902a2802eca91dcdf721f17af7c58f89a0ff84`; the benchmark is not claimed as having run on that HEAD or on the current uncommitted changes. Dataset audio-inclusive fingerprint: `252c55be0ff9dc4375bd38d796ffe5fa33ea43a28cd0daac2b1a6e4eaf88c823` (`audio_sha256_v1`); split fingerprint: `b5ba9f621f933b29b76fe0909fa99fa37ced8cc10626d8c2f0a169c91f54f171`. The corpus has 390 clips (360 train / 30 held-out), 82 performance/source groups and 10 test performance clusters. Recording and performance isolation are verified; performer isolation is **not verifiable** because artist metadata is absent. These are not artist-independent results.

| Model | Macro-F1 (mean +/- seed SD) | Accuracy (mean +/- seed SD) | Evaluation configuration |
| :--- | :---: | :---: | :--- |
| Handcrafted SVM | 0.4086 +/- 0.0000 | 0.4333 +/- 0.0000 | 35 handcrafted features; RBF SVC, C=10 |
| Random Forest | 0.3540 +/- 0.0168 | 0.3556 +/- 0.0157 | 100 trees; max depth 12 |
| Logistic Regression | 0.5400 +/- 0.0000 | 0.5333 +/- 0.0000 | max_iter=1000 |
| Frozen MERT | 0.2671 +/- 0.0328 | 0.3667 +/- 0.0272 | MERT revision `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`; 1536-d embeddings; 25 epochs |
| Hybrid pitch + MERT | 0.4902 +/- 0.0469 | 0.4889 +/- 0.0567 | 35 handcrafted + 1536 MERT dimensions; 25 epochs |

Seed runs: `[42, 123, 999]`; CPU; model configurations are recorded in the report. Lockfile SHA-256: `cf8239ba12fe3e9a23886a7b102824d8c19bb7a667ec11deccaf060d57c7e00d`. See [the full report](reports/scientific_evaluation_report.md) and the 15 experiment records for per-run provenance and metrics. Historical registry entries are not directly comparable unless their dataset, split, code, and model configurations match.
---

## 9. Cross-Genre Melodic Similarity & Calibrated Abstention

The system decouples formal classical classification from open-world cross-genre melodic resemblance:

| Track Name | Genre | Expected Resemblance | Top Predicted Match | Similarity Score | Confidence | Abstention Status |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **Bollywood (Bhairavi Melody)** | Indian Film / Bollywood | `Bhairavi` | **Bhairavi** | 0.863 | High | No |
| **Bollywood (Yaman Ballad)** | Indian Film / Bollywood | `Yaman` | **Yaman** | 0.852 | High | No |
| **Western Pop (Major Pentatonic)** | Western Pop / Folk | `Bhoopali` | **Bhoopali** | 0.918 | High | No |
| **Western Blues (Minor Pentatonic)**| Western Blues / Rock | `Malkauns` | **Malkauns** | 0.953 | High | No |
| **Atonal Noise Control** | Unpitched Control | `Abstain` | **None** | 0.000 | N/A | **Yes (Correct)** |

---

## 10. CLI & Research Commands

```powershell
# Analyze any audio file with CLI inference runner
python scripts/evaluate.py "path/to/song.mp3" --top-k 5

# Run cross-genre evaluation benchmark
python scripts/evaluate_cross_genre.py

# Audit the configured Hindustani dataset without modifying it
python scripts/audit_dataset.py

# Train baseline SVM on Hindustani dataset
python scripts/train.py --dataset hindustani --model baseline --classifier-type svm

# Train frozen MERT classification head
python scripts/train.py --dataset hindustani --model mert_frozen --epochs 30

# Train Hybrid Late-Fusion model
python scripts/train.py --dataset hindustani --model hybrid --epochs 30
```

