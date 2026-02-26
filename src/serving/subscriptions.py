"""
Subscription and usage-tracking models for Lemon Squeezy integration.

Stores plan mappings, per-key quotas, and subscription state.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Plan definitions ────────────────────────────────────────────


class PlanTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    BUSINESS = "business"


@dataclass
class PlanConfig:
    tier: PlanTier
    variant_id: int
    monthly_limit: int | None  # None = unlimited
    name: str


def _load_plan_configs() -> dict[int, PlanConfig]:
    """Build variant-ID → PlanConfig map from env vars."""
    free_id = int(os.getenv("PLAN_FREE_ID", "1346380"))
    pro_id = int(os.getenv("PLAN_PRO_ID", "1346293"))
    biz_id = int(os.getenv("PLAN_BUSINESS_ID", "1346355"))

    return {
        free_id: PlanConfig(
            tier=PlanTier.FREE,
            variant_id=free_id,
            monthly_limit=1_000,
            name="Free",
        ),
        pro_id: PlanConfig(
            tier=PlanTier.PRO,
            variant_id=pro_id,
            monthly_limit=50_000,
            name="Pro ($49.99/mo)",
        ),
        biz_id: PlanConfig(
            tier=PlanTier.BUSINESS,
            variant_id=biz_id,
            monthly_limit=None,
            name="Business ($199/mo)",
        ),
    }


PLAN_CONFIGS = _load_plan_configs()


def plan_for_variant(variant_id: int) -> PlanConfig:
    """Resolve a Lemon Squeezy variant ID to its PlanConfig."""
    cfg = PLAN_CONFIGS.get(variant_id)
    if cfg is None:
        raise ValueError(f"Unknown variant_id: {variant_id}")
    return cfg


# ── Subscription record ────────────────────────────────────────


@dataclass
class Subscription:
    """In-memory subscription record for an API key."""
    api_key: str
    customer_email: str
    plan: PlanConfig
    subscription_id: int
    status: str = "active"  # active | paused | cancelled | expired | past_due
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ── In-memory stores (swap for DB/Redis in production) ──────────


class SubscriptionStore:
    """
    Maps API keys → Subscription.

    In production replace with a database table; this in-memory store
    is sufficient for single-process / demo deployments.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, Subscription] = {}
        self._by_sub_id: dict[int, Subscription] = {}
        self._by_email: dict[str, Subscription] = {}

    # ── Mutators ────────────────────────────────────────────────

    def upsert(self, sub: Subscription) -> None:
        self._by_key[sub.api_key] = sub
        self._by_sub_id[sub.subscription_id] = sub
        self._by_email[sub.customer_email] = sub

    def remove_by_key(self, api_key: str) -> None:
        sub = self._by_key.pop(api_key, None)
        if sub:
            self._by_sub_id.pop(sub.subscription_id, None)
            self._by_email.pop(sub.customer_email, None)

    # ── Lookups ─────────────────────────────────────────────────

    def get_by_key(self, api_key: str) -> Optional[Subscription]:
        return self._by_key.get(api_key)

    def get_by_sub_id(self, sub_id: int) -> Optional[Subscription]:
        return self._by_sub_id.get(sub_id)

    def get_by_email(self, email: str) -> Optional[Subscription]:
        return self._by_email.get(email)

    def is_active(self, api_key: str) -> bool:
        sub = self.get_by_key(api_key)
        return sub is not None and sub.status in ("active",)


# Singleton
subscription_store = SubscriptionStore()


# ── Usage tracker ───────────────────────────────────────────────


class UsageTracker:
    """
    Tracks per-API-key monthly request counts.

    Resets automatically when the calendar month changes.
    """

    def __init__(self) -> None:
        # key → {month_key: count}
        self._counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    @staticmethod
    def _month_key() -> str:
        import datetime
        return datetime.datetime.utcnow().strftime("%Y-%m")

    def increment(self, api_key: str) -> int:
        """Increment and return the new count for the current month."""
        mk = self._month_key()
        self._counts[api_key][mk] += 1
        return self._counts[api_key][mk]

    def current_usage(self, api_key: str) -> int:
        mk = self._month_key()
        return self._counts[api_key].get(mk, 0)

    def check_quota(self, api_key: str, limit: int | None) -> bool:
        """Return True if the key is within its monthly quota."""
        if limit is None:  # unlimited
            return True
        return self.current_usage(api_key) < limit


# Singleton
usage_tracker = UsageTracker()
