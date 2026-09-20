"""
Redis caching layer for the serving API (SPEC §11.3).

Cache strategy:
  - User feature vectors: TTL 6h, invalidate on new transaction
  - Forecasts: TTL 24h, invalidate on nightly batch run
  - Budget recommendations: TTL 7d, invalidate on preference change
  - SHAP explanations: TTL 30d, invalidate on model version change

Privacy/data-lifecycle (SECURITY STEP 8): both the Redis path and the
in-memory fallback honour TTLs — cached user-scoped data is never retained
indefinitely, and the fallback purges expired entries deterministically on
every write. Cache keys are identity-namespaced and never contain bearer
tokens, emails, or credentials.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
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
        # key -> (monotonic expiry, value). The in-memory fallback honours
        # TTLs exactly like the Redis path: cached user data must never be
        # retained indefinitely (privacy/data-lifecycle requirement).
        self._local_cache: dict[str, tuple[float, Any]] = {}

        try:
            import redis

            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            self._redis = client
            # SECURITY: never log the connection URL — it may embed credentials.
            logger.info("Connected to Redis cache")
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

        entry = self._local_cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            # Lazy expiry: a stale entry is dropped on first access.
            self._local_cache.pop(key, None)
            return None
        return value

    def set(
        self,
        namespace: str,
        *parts: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Store a value in cache."""
        key = self._make_key(namespace, *parts)
        ttl = self.default_ttl if ttl is None else int(ttl)

        if self._redis is not None:
            if ttl > 0:
                self._redis.setex(key, ttl, json.dumps(value, default=str))
            # ttl <= 0 means "do not retain" — nothing is stored.
        else:
            # Deterministic cleanup: expired entries are purged on every
            # write, so the fallback cache cannot accumulate stale user data.
            now = time.monotonic()
            expired = [k for k, (exp, _) in self._local_cache.items() if exp <= now]
            for k in expired:
                self._local_cache.pop(k, None)
            if ttl > 0:
                self._local_cache[key] = (now + ttl, value)
            # ttl <= 0 means "do not retain" — nothing is stored.

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


    # ── STEP 12J: user-scoped invalidation (never global) ──────────────────
    # User namespaces from PRIVACY_DATA_LIFECYCLE.md §5; keys contain only the
    # JWT subject. `budgets:<sub>` is an exact single key; the other
    # namespaces key as `<ns>:<sub>:<parts>` and are removed by prefix,
    # restricted to the caller's own subject.
    USER_CACHE_NAMESPACES = ("budgets", "forecasts", "features", "explanations")

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove every entry whose key starts with ``prefix``; returns count."""
        if self._redis is not None:
            keys = list(self._redis.keys(f"{prefix}*"))
            if keys:
                return int(self._redis.delete(*keys))
            return 0
        to_remove = [k for k in self._local_cache if k.startswith(prefix)]
        for k in to_remove:
            del self._local_cache[k]
        return len(to_remove)

    def purge_user(self, user_id: str) -> int:
        """Remove every cache entry belonging to ONE user (Step 12J).

        Scope is strictly per-user: the caller's exact ``budgets`` key plus
        their ``forecasts/features/explanations`` prefixed keys. Other users'
        entries and global namespaces are untouched. Keys contain only the
        JWT subject — never tokens, emails, or credentials.
        """
        removed = self.invalidate("budgets", user_id) or 0
        for namespace in self.USER_CACHE_NAMESPACES:
            if namespace == "budgets":
                continue  # already removed above as an exact key
            removed += self.invalidate_prefix(f"{namespace}:{user_id}:")
        return removed


# ── Pre-configured cache instances with SPEC TTLs ─────────────────────────────

# TTLs from SPEC §11.3
CACHE_TTLS = {
    "features": 6 * 3600,       # 6 hours
    "forecasts": 24 * 3600,     # 24 hours
    "budgets": 7 * 24 * 3600,   # 7 days
    "explanations": 30 * 24 * 3600,  # 30 days
}
