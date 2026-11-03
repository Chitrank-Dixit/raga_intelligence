# Multi-stage Containerfile for Indian Raga Intelligence
# Base image: Python 3.12 on Debian Bookworm Slim
FROM docker.io/library/python:3.12-slim-bookworm AS base

# Install system audio libraries and minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/artifacts/cache/huggingface \
    TORCH_HOME=/app/artifacts/cache/torch

WORKDIR /app

# Create non-root application user and persistent volume directories
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app/data /app/artifacts /app/configs /app/reports && \
    chown -R appuser:appuser /app

USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure virtual environment path created by uv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:/home/appuser/.local/bin:${PATH}"

# Copy project metadata and the authoritative uv lockfile
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./

# Install all locked dependencies (including dev for testing) without installing the project yet (caching layer)
RUN uv sync --frozen --no-install-project --all-extras

# Copy source code, configs, scripts, frontend, and tests
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser frontend/ ./frontend/
COPY --chown=appuser:appuser tests/ ./tests/

# Install the project itself into the synced environment
RUN uv sync --frozen --all-extras

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint starts the lightweight inference API
ENTRYPOINT ["uvicorn", "raga_intelligence.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
