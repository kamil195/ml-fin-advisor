"""STEP 12M — persistence tests: financial profile endpoints (12G).

Covers: upsert keys the profile to ``principal.sub`` (never a body field);
one user can never fetch/update another's profile; profile update purges the
caller's user-scoped derived caches (Step 12J); the schema matches the
product's current inputs exactly (no speculative fields).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.persistence import DETAIL_503, PersistenceUnavailable

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

USER_A = "user-AAA"
USER_B = "user-BBB"


class FakeStore:
    """Minimal mirror of PostgresStore's profile contract."""

    def __init__(self) -> None:
        self.profiles: dict[str, dict[str, Any]] = {}
        self.fail_methods: set[str] = set()

    def _maybe_fail(self, method: str) -> None:
        if method in self.fail_methods:
            raise PersistenceUnavailable("InjectedFailure")

    def get_profile(self, owner_sub):
        self._maybe_fail("get_profile")
        return self.profiles.get(owner_sub)

    def upsert_profile(self, owner_sub, data):
        self._maybe_fail("upsert_profile")
        row = dict(data)
        row.setdefault("created_at", "2026-01-01T00:00:00+00:00")
        row["updated_at"] = "2026-06-01T00:00:00+00:00"
        self.profiles[owner_sub] = row
        return row

    def delete_profile(self, owner_sub):
        self._maybe_fail("delete_profile")
        return self.profiles.pop(owner_sub, None) is not None

    def list_transactions(self, owner_sub, limit=200, offset=0):
        return []

    def delete_user_data(self, owner_sub):
        profile = self.profiles.pop(owner_sub, None) is not None
        return {
            "profile_deleted": profile,
            "transactions_deleted": 0,
            "batches_deleted": 0,
        }


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def client(auth_env, make_auth_headers, store):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        c.headers.update(make_auth_headers(sub=USER_A))
        yield c


@pytest.fixture
def client_b(auth_env, make_auth_headers, store):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        c.headers.update(make_auth_headers(sub=USER_B))
        yield c


_PROFILE = {
    "income": 5200.0,
    "savings_target": 500.0,
    "liquid_buffer": 1000.0,
    "total_debt": 300.0,
    "monthly_debt_payments": 120.0,
}



# ── profile upsert / fetch / cross-user isolation ────────────────────────────


def test_upsert_uses_principal_sub(client, store):
    """(10) profile row is keyed to principal.sub; no body field can set it."""
    r = client.put("/consumer/profile", json=_PROFILE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["profile"]["income"] == 5200.0
    assert list(store.profiles) == [USER_A]


def test_profile_body_cannot_spoof_owner(client, store):
    """No ``owner_sub``/``user_id`` field in the body can change ownership."""
    body = {**_PROFILE, "owner_sub": USER_B, "user_id": USER_B}
    r = client.put("/consumer/profile", json=body)
    assert r.status_code == 200
    assert list(store.profiles) == [USER_A]  # extra fields simply ignored


def test_get_profile_returns_own_only(client, store):
    client.put("/consumer/profile", json=_PROFILE)
    r = client.get("/consumer/profile")
    assert r.status_code == 200
    assert r.json()["profile"]["income"] == 5200.0


def test_get_profile_404_when_absent(client):
    r = client.get("/consumer/profile")
    assert r.status_code == 404


def test_user_b_cannot_fetch_user_a_profile(client, client_b):
    """(11) profiles are strictly owner-scoped — no cross-user read."""
    client.put("/consumer/profile", json=_PROFILE)
    r = client_b.get("/consumer/profile")
    assert r.status_code == 404  # USER_B has no profile of their own
    # …and once B saves one, it is B's own — never A's.
    client_b.put("/consumer/profile", json={"income": 1.0})
    r = client_b.get("/consumer/profile")
    assert r.status_code == 200
    assert r.json()["profile"]["income"] == 1.0


def test_user_b_cannot_overwrite_user_a_profile(client, client_b, store):
    """(12) B's upsert writes B's row only; A's row is untouched."""
    client.put("/consumer/profile", json=_PROFILE)
    client_b.put("/consumer/profile", json={"income": 9.0})
    assert store.profiles[USER_A]["income"] == 5200.0
    assert store.profiles[USER_B]["income"] == 9.0


def test_profile_validation_rejects_negative_income(client):
    r = client.put("/consumer/profile", json={"income": -5})
    assert r.status_code == 422


def test_profile_fields_match_current_product_schema(client):
    """(14) only the product's current fields are accepted/returned.

    STEP 14 added the two optional Safe-to-Spend inputs — ``next_payday``
    (horizon end) and ``safety_buffer`` (reserve); see SPEC §31 and
    ``migrations/002_safe_to_spend.sql``. Both round-trip here, and neither
    is ever invented when omitted.
    """
    r = client.put(
        "/consumer/profile",
        json={**_PROFILE, "next_payday": "2026-10-15", "safety_buffer": 150.0},
    )
    assert r.status_code == 200, r.text
    profile = r.json()["profile"]
    assert set(profile) == {
        "income", "savings_target", "liquid_buffer", "total_debt",
        "monthly_debt_payments", "next_payday", "safety_buffer",
        "created_at", "updated_at",
    }
    assert profile["next_payday"] == "2026-10-15"
    assert profile["safety_buffer"] == 150.0


def test_profile_failure_is_generic_503(client, store):
    store.fail_methods.add("upsert_profile")
    r = client.put("/consumer/profile", json=_PROFILE)
    assert r.status_code == 503
    assert DETAIL_503 in r.text
    assert "INSERT" not in r.text
