"""
Middleware for the serving layer (SPEC §11).

Provides request/response logging.

SECURITY STEP 4: the legacy API-key machinery (``verify_api_key``,
``API_KEYS``, ``/admin/generate-key`` and its public-path set) was removed
entirely. Authentication is handled exclusively by the Supabase JWT
dependency ``require_auth`` (src/serving/auth.py).

SECURITY STEP 6: the legacy IP-keyed ``RateLimitMiddleware`` (hardcoded
600 rpm, unbounded per-IP history, applied to public paths, no Retry-After)
was removed and replaced by the identity-aware, bounded, environment-
configurable limiter in ``src/serving/rate_limit.py``.
"""

from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


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
