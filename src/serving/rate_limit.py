"""Identity-aware rate limiting for authenticated financial endpoints
(SECURITY STEP 6).

Design (derived from deployment inspection):

* Identity-aware: requests are keyed by the **verified JWT subject**
  (``principal.sub``) — never by client-supplied headers and never by raw
  tokens. Unauthenticated traffic can never reach the limiter because
  ``require_auth`` runs first and fails closed (generic 401).
* In-memory sliding-window counters. The serving deployment runs a single
  uvicorn worker (``--workers 1``), so an in-process counter is authoritative;
  no Redis counter dependency is introduced (the optional cache client may be
  unavailable and cross-worker counters would need shared state).
* Bounded memory: expired identities are removed before allocating buckets.
  At capacity, new identities are throttled rather than resetting active quotas.
* Thread-safe, process-local enforcement using a monotonic clock. Multiple
  workers/replicas require a shared atomic limiter before scaling out.
* Unexpected limiter failures return a generic 503 and log only the exception
  type. There is no unprotected fail-open path and no Redis dependency.

Logs record only that a limit was hit and the retry hint; JWTs, claims,
emails, financial payloads and transaction data are never logged.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time as _time
from collections import deque
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request

from src.serving.auth import AuthPrincipal, require_auth

logger = logging.getLogger(__name__)

#: Key namespace for authenticated-user buckets (never stores tokens/emails).
KEY_PREFIX = "rate_limit:user:"

#: Generic, safe client-facing message (no internals).
DETAIL_429 = "Rate limit exceeded. Please retry later."

#: Generic, safe message when abuse protection itself cannot run (fail closed).
DETAIL_503 = "Service temporarily unavailable. Please retry later."


@dataclass(frozen=True)
class RateLimitConfig:
    """Conservative private-beta defaults; override via environment."""

    enabled: bool = True
    requests: int = 60
    window_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not 1 <= self.requests <= 1000:
            raise ValueError("RATE_LIMIT_REQUESTS must be between 1 and 1000")
        if not math.isfinite(self.window_seconds) or not 1 <= self.window_seconds <= 3600:
            raise ValueError("RATE_LIMIT_WINDOW_SECONDS must be between 1 and 3600")

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        enabled = os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower()
        if enabled not in {"true", "false", "1", "0"}:
            raise ValueError("RATE_LIMIT_ENABLED must be true or false")
        return cls(
            enabled=enabled in {"true", "1"},
            requests=int(os.getenv("RATE_LIMIT_REQUESTS", "60")),
            window_seconds=float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
        )


class SlidingWindowLimiter:
    """Thread-safe, bounded in-memory sliding-window counter.

    See module docstring for the deployment and memory-safety rationale.
    """

    def __init__(self, config: RateLimitConfig, max_identities: int = 10_000) -> None:
        if not 1 <= max_identities <= 10_000:
            raise ValueError("max_identities must be between 1 and 10000")
        self.config = config
        self.max_identities = max_identities
        self._windows: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, max_identities: int = 10_000) -> "SlidingWindowLimiter":
        return cls(RateLimitConfig.from_env(), max_identities=max_identities)

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Record one hit for ``key`` and apply the sliding window.

        Returns ``(allowed, retry_after_seconds)``; ``retry_after_seconds``
        is ``0`` when the request is allowed.
        """
        now = _time.monotonic() if now is None else now
        with self._lock:
            window = self._windows.get(key)
            if window is None:
                if len(self._windows) >= self.max_identities:
                    cutoff = now - self.config.window_seconds
                    expired = [k for k, hits in self._windows.items() if hits[-1] <= cutoff]
                    for expired_key in expired:
                        self._windows.pop(expired_key)
                    if len(self._windows) >= self.max_identities:
                        retry = min(hits[-1] for hits in self._windows.values())
                        return False, max(1, math.ceil(retry + self.config.window_seconds - now))
                window = deque()
                self._windows[key] = window

            cutoff = now - self.config.window_seconds
            while window and window[0] <= cutoff:  # prune stale hits
                window.popleft()

            if len(window) >= self.config.requests:
                retry_after = max(1, math.ceil(window[0] + self.config.window_seconds - now))
                return False, retry_after

            window.append(now)
            return True, 0

    # -- Introspection helpers (tests / ops only; never expose contents) ---

    def snapshot_keys(self) -> list[str]:
        with self._lock:
            return sorted(self._windows)

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()


async def enforce_rate_limit(
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
) -> None:
    """FastAPI dependency: throttle authenticated financial requests.

    Must appear in a route's dependency list **after** ``require_auth`` so the
    authenticated subject is authoritative; FastAPI caches ``require_auth``'s
    result, so verification runs exactly once per request.
    """
    limiter: SlidingWindowLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None or not limiter.config.enabled:
        return

    key = f"{KEY_PREFIX}{principal.sub}"
    try:
        allowed, retry_after = limiter.check(key)
    except Exception as exc:  # noqa: BLE001 — fail closed, never silently unenforced
        # Log the exception TYPE only: no identities, tokens, or payloads.
        logger.error(
            "rate limiter failure (type: %s); failing closed", type(exc).__name__
        )
        raise HTTPException(status_code=503, detail=DETAIL_503) from None

    if not allowed:
        logger.warning("rate limit exceeded (scope=user, retry_after=%ss)", retry_after)
        raise HTTPException(
            status_code=429,
            detail=DETAIL_429,
            headers={"Retry-After": str(retry_after)},
        )

