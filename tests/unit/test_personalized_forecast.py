"""STEP 13M — user-specific forecasting endpoint tests.

Covers: owner isolation (a user's forecast uses ONLY their own persisted rows,
and another user's rows cannot change it), statuses (personalized /
limited_history / insufficient_history), the honest labelling of the legacy
global artifact, user-specific fallback, cache scoping + invalidation after
ingest/profile/delete, explicit horizon, generic DB/500 errors, and
determinism. No network and no real Postgres: a FakeStore implements the
owner-scoped repository contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.models.forecaster import personal_forecast as pf
from src.serving.app import create_app
from src.serving.cache import CacheClient
from src.serving.persistence import (
    DETAIL_503,
    DETAIL_NOT_CONFIGURED,
    PersistenceUnavailable,
)
from src.serving.routes import ingest as ingest_module

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

USER_A = "user-AAA"
USER_B = "user-BBB"
DAY = timedelta(days=1)

CSV_HEADER = "user_id,timestamp,amount,currency,merchant_name,merchant_mcc,account_type,channel"


def _history_start(days: int) -> datetime:
    """Anchor history to *yesterday* so it always falls inside the endpoint's
    120-day lookback window (relative dates, not a fixed past calendar date)."""
    end = datetime.now(timezone.utc).replace(
        hour=12, minute=0, second=0, microsecond=0
    ) - DAY
    return end - (days - 1) * DAY


def _rows(days: int, amount: float = -100.0, *, offset_hours: int = 0) -> list[dict]:
    """One deterministic spend per day for ``days`` days, ending yesterday."""
    start = _history_start(days)
    return [
        {
            "occurred_at": start + i * DAY + timedelta(hours=offset_hours),
            "amount": amount,
            "category_l2": "Groceries",
            "is_pending": False,
        }
        for i in range(days)
    ]


def _mem_cache() -> CacheClient:
    """Guaranteed in-memory cache (no Redis, no network)."""
    c = CacheClient.__new__(CacheClient)
    c.default_ttl = 60
    c._redis = None
    c._local_cache = {}
    return c


class FakeStore:
    """Owner-scoped repository fake (mirrors PostgresStore's public contract)."""

    def __init__(self) -> None:
        self.history: dict[str, list[dict]] = {}
        self.fail_methods: set[str] = set()
        self.profiles: dict[str, dict] = {}
        self.transactions: dict[str, tuple[str, dict]] = {}
        self.batches: list[dict] = []
        self.inserted: list[dict] = []
        self.fetch_calls: list[str] = []

    def _maybe_fail(self, method: str) -> None:
        if method in self.fail_methods:
            raise PersistenceUnavailable("InjectedFailure")

    # ── STEP 13B: the only history source ───────────────────────────────────
    def fetch_spend_history(self, owner_sub: str, since: datetime) -> list[dict]:
        self.fetch_calls.append(owner_sub)
        self._maybe_fail("fetch_spend_history")
        return [
            row for row in self.history.get(owner_sub, []) if row["occurred_at"] >= since
        ]

    # ── used by the ingest / lifecycle routes ──────────────────────────────
    def create_ingest_batch(self, owner_sub: str, source: str, row_count: int) -> str:
        self.batches.append(
            {"owner_sub": owner_sub, "source": source, "row_count": row_count}
        )
        return "batch-0001"

    def insert_transactions(
        self, owner_sub: str, rows: list[dict], ingest_batch_id: str
    ) -> tuple[int, int]:
        self.inserted.extend(rows)
        return len(rows), 0

    def upsert_profile(self, owner_sub: str, data: dict) -> dict:
        row = dict(data)
        row["updated_at"] = "u"
        self.profiles[owner_sub] = row
        return row

    def get_profile(self, owner_sub: str) -> dict | None:
        return self.profiles.get(owner_sub)

    def delete_user_data(self, owner_sub: str) -> dict:
        self._maybe_fail("delete_user_data")
        self.history.pop(owner_sub, None)
        return {"profile_deleted": False, "transactions_deleted": 0, "batches_deleted": 0}

    def export_user_data(self, owner_sub: str) -> dict:
        return {"profile": self.profiles.get(owner_sub), "transactions": []}


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@contextmanager
def _client(auth_env, make_auth_headers, store, sub, cache=None):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        if cache is not None:
            c.app.state.cache = cache
        c.headers.update(make_auth_headers(sub=sub))
        yield c


@pytest.fixture
def client(auth_env, make_auth_headers, store):
    with _client(auth_env, make_auth_headers, store, USER_A) as c:
        yield c


def _personalized_store(store: FakeStore) -> FakeStore:
    """A's history is Tier A; B's is deliberately tiny and different."""
    store.history[USER_A] = _rows(40, amount=-100.0)
    store.history[USER_B] = _rows(30, amount=-1.0)
    return store


# ── ownership / isolation ────────────────────────────────────────────────────


def test_uses_only_callers_own_transactions(client, store):
    """(1) the forecast is built from the caller's own persisted rows."""
    _personalized_store(store)
    r = client.get("/consumer/forecast?horizon_days=30")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["personalization_status"] == pf.STATUS_PERSONALIZED
    assert body["transaction_count_used"] == 40
    assert body["history_days_used"] == 40
    assert body["user_id"] == USER_A
    # Only A's history was ever queried — never a global or B query.
    assert store.fetch_calls and set(store.fetch_calls) == {USER_A}
    assert 2000.0 < body["expected_spend"] < 3200.0


def test_other_users_data_cannot_change_output(auth_env, make_auth_headers, store):
    """(2) B's rows (even extreme ones) never influence A's forecast."""
    _personalized_store(store)
    cache = _mem_cache()
    with _client(auth_env, make_auth_headers, store, USER_A, cache) as c:
        first = c.get("/consumer/forecast?horizon_days=30").json()
        cache.purge_user(USER_A)
        store.history[USER_B] = _rows(60, amount=-99999.0)  # B suddenly extreme
        second = c.get("/consumer/forecast?horizon_days=30").json()
    assert first["expected_spend"] == second["expected_spend"]
    assert first["daily"] == second["daily"]


def test_missing_jwt_is_401(auth_env, store):
    """(4) unauthenticated access is refused."""
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        assert c.get("/consumer/forecast").status_code == 401


def test_invalid_jwt_is_401(auth_env, store):
    """(5) a bogus bearer token is refused."""
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        r = c.get("/consumer/forecast", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_query_user_id_cannot_override_principal(client, store):
    """(6) no request field can select whose history is forecast."""
    _personalized_store(store)
    r = client.get(f"/consumer/forecast?user_id={USER_B}&horizon_days=30")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_id"] == USER_A
    assert body["transaction_count_used"] == 40  # A's rows, not B's 30
    assert set(store.fetch_calls) == {USER_A}


def test_cross_user_legacy_path_remains_forbidden(client):
    """(3) ownership on the legacy path is unchanged."""
    assert client.get(f"/v1/forecast/{USER_B}").status_code == 403


# ── statuses / honesty ───────────────────────────────────────────────────────


def test_sufficient_history_is_personalized(client, store):
    """(7) Tier A reports 'personalized' and names the method used."""
    _personalized_store(store)
    body = client.get("/consumer/forecast").json()
    assert body["personalization_status"] == pf.STATUS_PERSONALIZED
    assert body["method"] == pf.PRIMARY_METHOD
    assert body["engine_version"] == pf.FORECAST_ENGINE_VERSION
    assert body["fallback_status"] == pf.FALLBACK_NONE


def test_limited_history_is_flagged(client, store):
    """(8) Tier B still forecasts, but flags reduced confidence."""
    store.history[USER_A] = _rows(20, amount=-50.0)
    body = client.get("/consumer/forecast").json()
    assert body["personalization_status"] == pf.STATUS_LIMITED
    assert body["expected_spend"] is not None
    assert "low-confidence" in body["message"]


def test_insufficient_history_returns_no_number(client, store):
    """(8) Tier C produces NO number and states what is missing."""
    store.history[USER_A] = _rows(3, amount=-50.0)
    body = client.get("/consumer/forecast").json()
    assert body["personalization_status"] == pf.STATUS_INSUFFICIENT
    assert body["expected_spend"] is None
    assert body["daily"] == []
    assert body["total"] == {}
    assert body["requirements"]["observed"]["history_days"] == 3
    assert "no global forecast" in body["message"].lower()


def test_no_global_artifact_is_labelled_personalized(client, store):
    """(9)(22)(23) the legacy endpoint states exactly what it is."""
    _personalized_store(store)
    r = client.get(f"/v1/forecast/{USER_A}")
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    body = r.json()
    assert body["personalization_status"] == "not_personalized"
    assert body["fallback_status"] == "global_reference"
    assert body["method"] == "global_static_artifact"
    assert body["history_days_used"] == 0
    assert body["transaction_count_used"] == 0

    personal = client.get("/consumer/forecast").json()
    assert personal["method"] != "global_static_artifact"
    assert personal["personalization_status"] == pf.STATUS_PERSONALIZED


def test_fallback_is_user_specific_only():
    """(10) when the preferred method cannot be applied, the fallback uses the
    SAME user's own baseline — never a global or another user's series."""
    short = pf.build_daily_series(
        [(row["occurred_at"], row["amount"]) for row in _rows(10, amount=-80.0)]
    )
    payload = pf.predict(short, 7)
    assert payload["method"] == pf.FALLBACK_METHOD
    assert payload["fallback_status"] == pf.FALLBACK_USER_BASELINE
    # The number is exactly this user's own recent level, spread over the horizon.
    assert payload["expected_spend"] == round(pf.recent_mean_level(short) * 7, 2)
    # A different user's series produces a different forecast from the same code.
    other = pf.build_daily_series(
        [(row["occurred_at"], row["amount"]) for row in _rows(10, amount=-500.0)]
    )
    assert pf.predict(other, 7)["expected_spend"] != payload["expected_spend"]
    # And nothing global can ever appear in a personalized fallback.
    assert pf.FALLBACK_GLOBAL_REFERENCE != payload["fallback_status"]


def test_prophet_request_is_substituted_explicitly():
    """(13K) requesting a method the server does not use is reported, not hidden."""
    series = pf.build_daily_series(
        [(row["occurred_at"], row["amount"]) for row in _rows(40, amount=-100.0)]
    )
    payload = pf.predict(series, 30, method=pf.METHOD_PROPHET)
    assert payload["method"] == pf.PRIMARY_METHOD
    assert payload["fallback_status"] == pf.FALLBACK_PRIMARY_SUBSTITUTED
    assert payload["personalization_status"] == pf.STATUS_PERSONALIZED


def test_horizon_is_explicit(client, store):
    """(16) the requested horizon is reported and reflected in the points."""
    _personalized_store(store)
    body = client.get("/consumer/forecast?horizon_days=14").json()
    assert body["horizon_days"] == 14
    assert len(body["daily"]) == 14
    start = datetime.fromisoformat(body["forecast_start"]).date()
    end = datetime.fromisoformat(body["forecast_end"]).date()
    assert (end - start).days == 13


def test_response_is_deterministic(client, store, auth_env, make_auth_headers):
    """(24) same user + same data => identical output."""
    _personalized_store(store)
    first = client.get("/consumer/forecast?horizon_days=30").json()
    with _client(auth_env, make_auth_headers, store, USER_A, _mem_cache()) as c2:
        second = c2.get("/consumer/forecast?horizon_days=30").json()
    assert first["expected_spend"] == second["expected_spend"]
    assert first["daily"] == second["daily"]
    assert first["residual_sigma"] == second["residual_sigma"]


# ── failures ─────────────────────────────────────────────────────────────────


def test_db_failure_returns_generic_503(client, store):
    """(20) a persistence failure is generic and leaks nothing."""
    _personalized_store(store)
    store.fail_methods.add("fetch_spend_history")
    r = client.get("/consumer/forecast")
    assert r.status_code == 503
    assert r.json()["detail"] == DETAIL_503
    for leaked in ("SELECT", "owner_sub", "transactions", "psycopg", "postgres"):
        assert leaked not in r.text


def test_persistence_not_configured_returns_503(auth_env, make_auth_headers):
    """Deployments without DATABASE_URL report this explicitly."""
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = None
        c.headers.update(make_auth_headers(sub=USER_A))
        r = c.get("/consumer/forecast")
    assert r.status_code == 503
    assert r.json()["detail"] == DETAIL_NOT_CONFIGURED


def test_internal_failure_is_sanitized(client, store, monkeypatch):
    """(21) an unexpected error never leaks internals to the client."""
    _personalized_store(store)

    def _boom(*_a, **_k):
        raise RuntimeError("LEAK_ATTEMPT /models/secret.joblib")

    monkeypatch.setattr(pf, "predict", _boom)
    r = client.get("/consumer/forecast")
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error."
    assert "LEAK_ATTEMPT" not in r.text


# ── cache: scoping + invalidation (13I) ──────────────────────────────────────


def _fc_keys(cache, sub: str) -> list[str]:
    return [k for k in cache._local_cache if k.startswith(f"forecasts:{sub}:")]


def test_cache_key_is_user_scoped_and_token_free(client, store):
    """(11)(15) the key carries the subject, never a token, and is horizon+version aware."""
    _personalized_store(store)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/forecast?horizon_days=30")

    keys = _fc_keys(cache, USER_A)
    assert keys, "personalized forecast should be cached"
    key = keys[0]
    assert key.startswith(f"forecasts:{USER_A}:personal:30:")
    assert pf.FORECAST_ENGINE_VERSION in key
    assert USER_B not in key
    assert "eyJ" not in key and "Bearer" not in key
    # Horizon is part of the key, so a different horizon cannot serve a stale value.
    client.get("/consumer/forecast?horizon_days=14")
    assert len(_fc_keys(cache, USER_A)) == 2


def test_cache_invalidates_after_ingest(client, store, monkeypatch):
    """(12) ingesting transactions drops this user's cached forecast."""
    _personalized_store(store)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/forecast?horizon_days=30")
    assert _fc_keys(cache, USER_A)

    async def _stub_classify(_request, _req, history=None):
        return SimpleNamespace(
            category_l1="Essentials", category_l2="Groceries", confidence=0.9
        )

    monkeypatch.setattr(
        ingest_module, "classify_transaction_with_history", _stub_classify
    )
    body = f"{CSV_HEADER}\n{USER_A},2026-04-05T12:00:00,-100.0,USD,FreshMart,5411,CHECKING,POS\n"
    r = client.post(
        "/consumer/transactions/ingest-csv",
        content=body.encode(),
        headers={"Content-Type": "text/csv"},
    )
    assert r.status_code == 200, r.text
    assert _fc_keys(cache, USER_A) == [], "ingest must invalidate the user's forecast cache"


def test_cache_invalidates_after_profile_update_and_keeps_other_users(client, store):
    """(13) profile mutation purges the caller's entries only — never another user's."""
    _personalized_store(store)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/forecast?horizon_days=30")
    cache.set(
        "forecasts", USER_B, "personal", "30", pf.FORECAST_ENGINE_VERSION, value={"x": 1}
    )
    assert _fc_keys(cache, USER_A) and _fc_keys(cache, USER_B)

    r = client.put("/consumer/profile", json={"income": 5000.0})
    assert r.status_code == 200, r.text
    assert _fc_keys(cache, USER_A) == []
    assert _fc_keys(cache, USER_B), "another user's cache must never be purged"


def test_cache_invalidates_after_delete(client, store):
    """(14) deleting the user's financial data purges their cached forecast."""
    _personalized_store(store)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/forecast?horizon_days=30")
    assert _fc_keys(cache, USER_A)

    r = client.delete("/consumer/data")
    assert r.status_code == 200, r.text
    assert _fc_keys(cache, USER_A) == []


def test_insufficient_result_is_not_cached(client, store):
    """Honesty: a Tier C answer is not cached, so new data takes effect at once."""
    store.history[USER_A] = _rows(2, amount=-10.0)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/forecast")
    assert _fc_keys(cache, USER_A) == []