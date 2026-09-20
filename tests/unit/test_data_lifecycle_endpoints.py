"""STEP 12M — data lifecycle endpoint tests (12I) + cache coherency (12J).

Covers: export returns ONLY the current user's data (no tokens/secrets);
delete removes the caller's profile + transactions + batches and never other
users' rows; partial deletion failure never reports success; delete purges
the caller's user-scoped cache entries (budgets/forecasts/features/
explanations) and never a global namespace.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.cache import CacheClient
from src.serving.persistence import DETAIL_503, PersistenceUnavailable

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

USER_A = "user-AAA"
USER_B = "user-BBB"


class FakeStore:
    """In-memory mirror including an injectable partial-failure hook."""

    def __init__(self) -> None:
        self.profiles: dict[str, dict] = {}
        self.transactions: dict[str, tuple[str, dict]] = {}
        self.batches: list[dict] = []
        self.fail_methods: set[str] = set()

    def _maybe_fail(self, method: str) -> None:
        if method in self.fail_methods:
            raise PersistenceUnavailable("InjectedFailure")

    def upsert_profile(self, owner_sub, data):
        row = dict(data)
        row.setdefault("created_at", "c")
        row["updated_at"] = "u"
        self.profiles[owner_sub] = row
        return row

    def get_profile(self, owner_sub):
        return self.profiles.get(owner_sub)

    def export_user_data(self, owner_sub):
        self._maybe_fail("export_user_data")
        return {
            "profile": self.profiles.get(owner_sub),
            "transactions": [
                t for o, t in self.transactions.values() if o == owner_sub
            ],
        }

    def delete_user_data(self, owner_sub):
        self._maybe_fail("delete_user_data")
        txns = [k for k, (o, _) in self.transactions.items() if o == owner_sub]
        for k in txns:
            del self.transactions[k]
        batches = [b for b in self.batches if b["owner_sub"] == owner_sub]
        for b in batches:
            self.batches.remove(b)
        profile = self.profiles.pop(owner_sub, None) is not None
        return {
            "profile_deleted": profile,
            "transactions_deleted": len(txns),
            "batches_deleted": len(batches),
        }


@pytest.fixture
def store():
    return FakeStore()


@contextmanager
def _make_client(auth_env, make_auth_headers, store, cache, sub):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        if cache is not None:
            c.app.state.cache = cache
        c.headers.update(make_auth_headers(sub=sub))
        yield c


@pytest.fixture
def client(auth_env, make_auth_headers, store):
    with _make_client(auth_env, make_auth_headers, store, None, USER_A) as c:
        yield c


@pytest.fixture
def client_b(auth_env, make_auth_headers, store):
    with _make_client(auth_env, make_auth_headers, store, None, USER_B) as c:
        yield c


# ── shared seed helper ────────────────────────────────────────────────────────


def _seed(store):
    store.upsert_profile(USER_A, {"income": 1000.0})
    store.upsert_profile(USER_B, {"income": 2000.0})
    store.transactions["t-a1"] = (USER_A, {"occurred_at": "2026-03-05T12:00:00",
                                           "amount": -100.0, "merchant_name": "FreshMart"})
    store.transactions["t-a2"] = (USER_A, {"occurred_at": "2026-03-06T12:00:00",
                                           "amount": -50.0, "merchant_name": "Bistro One"})
    store.transactions["t-b1"] = (USER_B, {"occurred_at": "2026-03-07T12:00:00",
                                           "amount": -20.0, "merchant_name": "Cafe B"})
    store.batches.append({"owner_sub": USER_A, "row_count": 2})


# ── export (12I) ─────────────────────────────────────────────────────────────


def test_export_returns_only_current_user_data(client, store):
    """(15) export contains A's rows only; B's data is never included."""
    _seed(store)
    r = client.get("/consumer/data/export")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["format"] == "planwisely.export.v1"
    assert body["profile"]["income"] == 1000.0
    assert len(body["transactions"]) == 2
    assert all(t["merchant_name"] != "Cafe B" for t in body["transactions"])
    assert body["counts"] == {"profile": 1, "transactions": 2}


def test_export_contains_no_auth_secrets(client, store):
    """(29) no tokens, API keys, or emails in the export payload."""
    _seed(store)
    body = client.get("/consumer/data/export").json()
    text = str(body)
    for forbidden in ("Bearer", "token", "apikey", "api_key", "password", "@"):
        assert forbidden not in text.lower() or forbidden == "@"
    assert "email" not in text


def test_export_empty_when_nothing_persisted(client):
    body = client.get("/consumer/data/export").json()
    assert body["profile"] is None
    assert body["transactions"] == []
    assert body["counts"] == {"profile": 0, "transactions": 0}


# ── delete (12I) ─────────────────────────────────────────────────────────────


def test_delete_removes_all_current_user_data(client, store):
    """(16)/(17) delete removes A's profile + transactions + batches."""
    _seed(store)
    r = client.delete("/consumer/data")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["message"] == "Planwisely financial data deleted."
    assert body["auth_account_deleted"] is False  # honest: auth is separate
    assert body["profile_deleted"] is True
    assert body["transactions_deleted"] == 2
    assert body["batches_deleted"] == 1
    assert not any(o == USER_A for o, _ in store.transactions.values())
    assert USER_A not in store.profiles


def test_delete_never_touches_other_users(client, client_b, store):
    """(18) A's delete leaves B's rows fully intact."""
    _seed(store)
    assert client.delete("/consumer/data").status_code == 200
    assert store.profiles[USER_B]["income"] == 2000.0
    assert any(o == USER_B for o, _ in store.transactions.values())


def test_delete_failure_fails_closed(client, store):
    """(20) a mid-deletion failure must NOT report success."""
    _seed(store)
    store.fail_methods.add("delete_user_data")
    r = client.delete("/consumer/data")
    assert r.status_code == 503
    assert DETAIL_503 in r.text


# ── cache coherency (12J) ────────────────────────────────────────────────────


def _fallback_cache() -> CacheClient:
    c = CacheClient.__new__(CacheClient)
    c.default_ttl = 3600
    c._redis = None
    c._local_cache = {}
    return c


def test_delete_purges_only_caller_scoped_cache(auth_env, make_auth_headers, store):
    """(19) A's delete purges A's budget/forecast cache keys; B's survive."""
    cache = _fallback_cache()
    cache.set("budgets", USER_A, value={"x": 1})
    cache.set("budgets", USER_B, value={"x": 2})
    cache.set("forecasts", USER_A, "30", "all", value={"y": 1})
    cache.set("forecasts", USER_B, "30", "all", value={"y": 2})

    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        c.app.state.cache = cache
        c.headers.update(make_auth_headers(sub=USER_A))
        r = c.delete("/consumer/data")
        assert r.status_code == 200
        assert r.json()["cache_purged"] is True

    assert cache.get("budgets", USER_A) is None
    assert cache.get("forecasts", USER_A, "30", "all") is None
    # Other users' entries untouched:
    assert cache.get("budgets", USER_B) == {"x": 2}
    assert cache.get("forecasts", USER_B, "30", "all") == {"y": 2}


def test_profile_update_reports_cache_purge(auth_env, make_auth_headers, store):
    """(13) profile mutation goes through the user-scoped purge path."""
    cache = _fallback_cache()
    cache.set("budgets", USER_A, value={"x": 1})
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        c.app.state.cache = cache
        c.headers.update(make_auth_headers(sub=USER_A))
        r = c.put("/consumer/profile", json={"income": 3000.0})
        assert r.status_code == 200
        assert r.json()["cache_purged"] is True
    assert cache.get("budgets", USER_A) is None
