"""
Consumer user store and usage tracking for the freemium model.

Provides:
  - User registration and lookup
  - Free-tier usage counting (5 free analyses)
  - Subscription status management
  - Analysis result history (read-only after limit)

Uses an in-memory store with JSON file persistence.
Swap for PostgreSQL / Redis in production.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────

FREE_TIER_LIMIT = int(os.getenv("FREE_TIER_LIMIT", "5"))
LEMON_SQUEEZY_CHECKOUT_URL = os.getenv(
    "LEMONSQUEEZY_CHECKOUT_URL",
    "https://fin-advisor.lemonsqueezy.com/buy",  # placeholder
)

_PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "consumer"


# ── Enums ───────────────────────────────────────────────────────


class SubStatus(str, Enum):
    FREE = "free"
    PAID = "paid"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"


# ── Data classes ────────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """A single saved analysis (classification / forecast / budget)."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    analysis_type: str = ""          # classify | forecast | budget
    request_summary: dict = field(default_factory=dict)
    response_data: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ConsumerUser:
    """A consumer-app user with freemium gating."""
    user_id: str = field(default_factory=lambda: f"cu-{uuid.uuid4().hex[:8]}")
    email: str = ""
    display_name: str = ""
    subscription_status: SubStatus = SubStatus.FREE
    free_tier_used: int = 0
    lemon_squeezy_customer_id: str | None = None
    lemon_squeezy_subscription_id: int | None = None
    analysis_history: list[AnalysisResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── Helpers ─────────────────────────────────────────────────

    @property
    def can_analyse(self) -> bool:
        """Can this user run a new analysis?"""
        if self.subscription_status == SubStatus.PAID:
            return True
        return self.free_tier_used < FREE_TIER_LIMIT

    @property
    def remaining_free(self) -> int:
        return max(0, FREE_TIER_LIMIT - self.free_tier_used)

    def record_analysis(self, result: AnalysisResult) -> None:
        self.free_tier_used += 1
        self.analysis_history.append(result)
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["subscription_status"] = self.subscription_status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ConsumerUser":
        d = dict(d)  # shallow copy
        d["subscription_status"] = SubStatus(d.get("subscription_status", "free"))
        history_raw = d.pop("analysis_history", [])
        user = cls(**{k: v for k, v in d.items() if k != "analysis_history"})
        user.analysis_history = [
            AnalysisResult(**h) if isinstance(h, dict) else h
            for h in history_raw
        ]
        return user


# ── In-memory store with JSON persistence ───────────────────────


class ConsumerUserStore:
    """
    Thread-safe-ish user store backed by a JSON file.

    For production: replace with PostgreSQL + SQLAlchemy or Redis.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._persist_path = persist_path or (_PERSIST_DIR / "users.json")
        self._users: dict[str, ConsumerUser] = {}
        self._by_email: dict[str, str] = {}   # email → user_id
        self._load()

    # ── Persistence ─────────────────────────────────────────────

    def _load(self) -> None:
        if self._persist_path.exists():
            try:
                with open(self._persist_path) as f:
                    raw = json.load(f)
                for uid, data in raw.items():
                    user = ConsumerUser.from_dict(data)
                    self._users[uid] = user
                    if user.email:
                        self._by_email[user.email] = uid
                logger.info("Loaded %d consumer users from %s", len(self._users), self._persist_path)
            except Exception as e:
                logger.warning("Failed to load consumer users: %s", e)

    def _save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w") as f:
            json.dump({uid: u.to_dict() for uid, u in self._users.items()}, f, indent=2)

    # ── CRUD ────────────────────────────────────────────────────

    def register(self, email: str, display_name: str = "") -> ConsumerUser:
        """Create a new user or return existing if email matches."""
        existing_uid = self._by_email.get(email)
        if existing_uid and existing_uid in self._users:
            return self._users[existing_uid]

        user = ConsumerUser(email=email, display_name=display_name or email.split("@")[0])
        self._users[user.user_id] = user
        if email:
            self._by_email[email] = user.user_id
        self._save()
        logger.info("Registered consumer user %s (%s)", user.user_id, email)
        return user

    def get(self, user_id: str) -> Optional[ConsumerUser]:
        return self._users.get(user_id)

    def get_by_email(self, email: str) -> Optional[ConsumerUser]:
        uid = self._by_email.get(email)
        return self._users.get(uid) if uid else None

    def update(self, user: ConsumerUser) -> None:
        user.updated_at = time.time()
        self._users[user.user_id] = user
        if user.email:
            self._by_email[user.email] = user.user_id
        self._save()

    def delete(self, user_id: str) -> None:
        user = self._users.pop(user_id, None)
        if user and user.email:
            self._by_email.pop(user.email, None)
        self._save()


# ── Singleton ───────────────────────────────────────────────────

consumer_store = ConsumerUserStore()
