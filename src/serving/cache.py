"""
Redis caching layer for the serving API (SPEC §11.3).

Cache strategy:
  - User feature vectors: TTL 6h, invalidate on new transaction
  - Forecasts: TTL 24h, invalidate on nightly batch run
  - Budget recommendations: TTL 7d, invalidate on preference change
  - SHAP explanations: TTL 30d, invalidate on model version change
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CacheClient:
    """
    Thin wrapper around Redis for inference caching.

    Falls back to an in-memory dict when Redis is not available.

    Parameters
    ----------
    redis_url : str
        Redis connection URL.
    default_ttl : int
        Default TTL in seconds.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
    ) -> None:
        self.default_ttl = default_ttl
        self._redis: Any = None
        self._local_cache: dict[str, Any] = {}

        try:
            import redis

            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            self._redis = client
            logger.info("Connected to Redis at %s", redis_url)
        except (ImportError, Exception) as exc:
            self._redis = None
            logger.warning(
                "Redis unavailable (%s) — using in-memory cache.", exc
            )

    @staticmethod
    def _make_key(namespace: str, *parts: str) -> str:
        """Build a cache key."""
        raw = ":".join([namespace] + list(parts))
        return raw

    def get(self, namespace: str, *parts: str) -> Any | None:
        """Retrieve a cached value."""
        key = self._make_key(namespace, *parts)

        if self._redis is not None:
            val = self._redis.get(key)
            if val is not None:
                return json.loads(val)
            return None

        return self._local_cache.get(key)

    def set(
        self,
        namespace: str,
        *parts: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Store a value in cache."""
        key = self._make_key(namespace, *parts)
        ttl = ttl or self.default_ttl

        if self._redis is not None:
            self._redis.setex(key, ttl, json.dumps(value, default=str))
        else:
            self._local_cache[key] = value

    def invalidate(self, namespace: str, *parts: str) -> None:
        """Remove a cached entry."""
        key = self._make_key(namespace, *parts)

        if self._redis is not None:
            self._redis.delete(key)
        else:
            self._local_cache.pop(key, None)

    def invalidate_namespace(self, namespace: str) -> int:
        """Remove all entries in a namespace."""
        if self._redis is not None:
            keys = self._redis.keys(f"{namespace}:*")
            if keys:
                return self._redis.delete(*keys)
            return 0

        count = 0
        to_remove = [k for k in self._local_cache if k.startswith(f"{namespace}:")]
        for k in to_remove:
            del self._local_cache[k]
            count += 1
        return count


# ── Pre-configured cache instances with SPEC TTLs ─────────────────────────────

# TTLs from SPEC §11.3
CACHE_TTLS = {
    "features": 6 * 3600,       # 6 hours
    "forecasts": 24 * 3600,     # 24 hours
    "budgets": 7 * 24 * 3600,   # 7 days
    "explanations": 30 * 24 * 3600,  # 30 days
}
