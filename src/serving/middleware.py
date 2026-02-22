"""
Middleware for the serving layer (SPEC §11).

Provides:
  - API-key authentication (via FastAPI dependency injection)
  - Rate limiting
  - Request/response logging
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from collections import defaultdict

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── Paths that never require an API key ─────────────────────
_PUBLIC_PATHS: set[str] = {
    "/health",
    "/ready",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
    "/admin/generate-key",
}

# ── FastAPI security scheme ─────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_api_keys() -> set[str]:
    """
    Load valid API keys from the ``API_KEYS`` env var.

    Keys are stored as a comma-separated list.  If the env var is
    empty or unset, authentication is **disabled** (open access).
    """
    raw = os.environ.get("API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def generate_api_key(prefix: str = "fina") -> str:
    """Return a new random API key like ``fina_2f8a…`` (40 hex chars)."""
    return f"{prefix}_{secrets.token_hex(20)}"


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> None:
    """
    FastAPI dependency that enforces API-key authentication.

    * If ``API_KEYS`` env var is empty/unset → open access (no auth).
    * Public paths (health, docs, etc.) are always open.
    * Otherwise, ``X-API-Key`` header must contain a valid key.
    """
    # Public paths — always open
    if request.url.path in _PUBLIC_PATHS:
        return

    valid_keys = _load_api_keys()

    # No keys configured — open access
    if not valid_keys:
        return

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header.",
        )

    if api_key not in valid_keys:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:12]
        logger.warning("Invalid API key attempt (hash=%s)", key_hash)
        raise HTTPException(status_code=401, detail="Invalid API key.")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter using a sliding window.

    Parameters
    ----------
    requests_per_minute : int
        Maximum requests per minute per client IP.
    burst : int
        Maximum burst size.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 600,
        burst: int = 50,
    ) -> None:
        super().__init__(app)
        self.rpm = requests_per_minute
        self.burst = burst
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < 60.0
        ]

        if len(self._requests[client_ip]) >= self.rpm:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
