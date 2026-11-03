"""
FastAPI application factory and middleware configuration.
"""

from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from raga_intelligence import __version__
from raga_intelligence.api.routes import router, sanitize_error_detail
from raga_intelligence.config import settings

logger = logging.getLogger(__name__)

CORRELATION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """
    Ensures every incoming request has a unique correlation ID for end-to-end tracing.
    Validates client/proxy supplied X-Request-ID (1-64 alphanumeric, dash, dot, underscore);
    otherwise generates a new UUID4.
    """
    async def dispatch(self, request: Request, call_next):
        raw_id = request.headers.get("X-Request-ID")
        if raw_id and CORRELATION_ID_PATTERN.match(raw_id):
            request_id = raw_id
        else:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


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

    # 1. Request correlation tracing
    app.add_middleware(RequestCorrelationMiddleware)

    # 2. Configurable CORS security (replaces insecure wildcard)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allowed_methods,
        allow_headers=settings.cors_allowed_headers,
    )

    # 3. Global Exception Handling (Prevents traceback & path exposure)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        detail = sanitize_error_detail(str(exc.detail)) if isinstance(exc.detail, str) else exc.detail
        headers = {"X-Request-ID": request_id}
        if exc.headers:
            headers.update(exc.headers)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": detail,
                "request_id": request_id,
            },
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Internal server error [request_id={request_id}]: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred while processing the request.",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
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

