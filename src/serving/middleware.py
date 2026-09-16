"""
Middleware for the serving layer (SPEC §11).

Provides:
  - Rate limiting
  - Request/response logging

SECURITY STEP 4: the legacy API-key machinery (``verify_api_key``,
``API_KEYS``, ``/admin/generate-key`` and its public-path set) was removed
entirely. Authentication is handled exclusively by the Supabase JWT
dependency ``require_auth`` (src/serving/auth.py).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)





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
            # NOTE: raising HTTPException here would surface as a 500, not a 429.
            # Starlette's ExceptionMiddleware is mounted *inside* this middleware,
            # so it never sees the exception and cannot convert it. Return the
            # response directly instead.
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
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
