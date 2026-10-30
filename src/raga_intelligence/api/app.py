"""
FastAPI application factory and middleware configuration.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from raga_intelligence import __version__
from raga_intelligence.api.routes import router
from raga_intelligence.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(f"Starting Indian Raga Intelligence API v{__version__} ({settings.env})")
    device = settings.get_torch_device()
    logger.info(f"Target compute execution device: {device}")
    yield
    logger.info("Shutting down Indian Raga Intelligence API.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Indian Raga Intelligence API",
        description=(
            "Research-driven machine learning system for Indian classical raga recognition, "
            "pitch-class distribution analysis, and cross-genre melodic similarity intelligence."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # Static UI mount if frontend folder contains index.html
    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
    if frontend_dir.exists() and (frontend_dir / "index.html").exists():
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(frontend_dir / "index.html")

    return app


app = create_app()

