# Multi-stage Containerfile for Indian Raga Intelligence
# Base image: Python 3.12 on Debian Bookworm Slim
FROM docker.io/library/python:3.12-slim-bookworm AS base

# Install system audio libraries and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/artifacts/cache/huggingface \
    TORCH_HOME=/app/artifacts/cache/torch

WORKDIR /app

# Create non-root application user and directories
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app/data /app/artifacts /app/configs /app/reports && \
    chown -R appuser:appuser /app

# Copy dependency configuration
COPY --chown=appuser:appuser pyproject.toml README.md ./

# Install python dependencies as appuser
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install uv for fast, reliable package resolution inside container
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up virtual environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:/home/appuser/.local/bin:${PATH}"
RUN uv venv /app/.venv

# Install project dependencies with PyTorch
RUN uv pip install --no-cache \
    "torch>=2.4.0" \
    "torchaudio>=2.4.0" \
    "transformers>=4.42.0" \
    "librosa>=0.10.2" \
    "soundfile>=0.12.1" \
    "scikit-learn>=1.5.0" \
    "scipy>=1.13.0" \
    "numpy>=1.26.4,<2.2.0" \
    "pandas>=2.2.2" \
    "matplotlib>=3.9.0" \
    "pydantic>=2.7.0" \
    "pydantic-settings>=2.3.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.30.0" \
    "python-multipart>=0.0.9" \
    "pyyaml>=6.0.1" \
    "requests>=2.32.0" \
    "tqdm>=4.66.4" \
    "pytest>=8.2.0" \
    "httpx>=0.27.0"

# Copy source code and configuration
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser frontend/ ./frontend/

# Install the raga_intelligence package in editable mode
RUN uv pip install --no-cache -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "raga_intelligence.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

