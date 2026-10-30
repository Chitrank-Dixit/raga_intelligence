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
- **Strict Leakage Prevention**: Split generation strictly partitions audio at the **recording / performance and artist** level. Audio segments from the same performance never cross train and test boundaries.
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

- **Saraga Hindustani**: Zenodo DOI `10.5281/zenodo.4301737` (CC BY-NC-SA 4.0).
- **CompMusic IAM Features**: Zenodo DOI `10.5281/zenodo.7278505` (CC BY-NC-ND 4.0).
- **Synthetic Test Fixtures**: Deterministically synthesized within `data/synthetic` under CC0-1.0 for reproducible CI testing without external network dependencies.

