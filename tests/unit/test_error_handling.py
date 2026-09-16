"""Step 5 regression tests: unexpected internal exceptions must never leak
implementation details to clients.

For every representative route a forced ``RuntimeError("SECRET_MARKER")`` is
injected at a real seam (model object, state cache, engine instance). The
assertions prove:

* the client receives HTTP 500 with the single generic detail
  ``"Internal server error."``
* the injected marker, exception class names, and tracebacks are absent from
  the response body
* deliberate, useful client errors (401 / 422 / 413 / 404 / 503) remain intact
* server-side diagnostics still exist (caplog) without touching secrets

Also covers the Step 5 cache logging fix: a Redis URL containing credentials
must never be written to logs verbatim.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

GENERIC = "Internal server error."
MARKER = "SECRET_MARKER_NEVER_FOR_CLIENT"


def _boom(_=None, **__):
    raise RuntimeError(f"{MARKER} /models/serving/classifier_lgb.joblib")


class _ExplodingState:
    """Replaces a cached-state object whose ``.get()`` explodes unexpectedly."""

    def get(self, *_args, **_kwargs):
        _boom()


def _assert_sanitized(resp) -> None:
    assert resp.status_code == 500, resp.text
    body = resp.text
    assert GENERIC in body
    assert MARKER not in body
    assert "RuntimeError" not in body
    assert "Traceback" not in body
    assert ".joblib" not in body
    assert body.count(GENERIC) == 1


# ── fixtures / payloads ───────────────────────────────────────────────────────


@pytest.fixture
def client(auth_env, make_auth_headers):
    # raise_server_exceptions=False -> the ASGI stack converts unexpected
    # exceptions into real 500 responses instead of re-raising inside the
    # test, exactly what production uvicorn serves to clients.
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.headers.update(make_auth_headers())
        yield c


def _txn(**over) -> dict:
    base = {
        "user_id": "user-123",
        "timestamp": "2026-03-05T12:00:00",
        "amount": -100.0,
        "merchant_name": "Merchant",
        "merchant_mcc": 5411,
        "account_type": "CHECKING",
        "channel": "POS",
    }
    base.update(over)
    return base


# ── forced unexpected exceptions → sanitized 500 ─────────────────────────────


def test_v1_classify_internal_error_sanitized(client, monkeypatch):
    monkeypatch.setattr(
        client.app.state.classifier, "predict_proba", _boom, raising=True
    )
    r = client.post("/v1/classify", json={"transaction": _txn()})
    _assert_sanitized(r)


def test_v1_forecast_internal_error_sanitized(client, monkeypatch):
    monkeypatch.setattr(
        client.app.state, "forecast_data", _ExplodingState(), raising=False
    )
    r = client.get("/v1/forecast/user-123")
    _assert_sanitized(r)


def test_v1_budget_internal_error_sanitized(client, monkeypatch):
    monkeypatch.setattr(
        client.app.state, "budget_data", _ExplodingState(), raising=False
    )
    r = client.get("/v1/budget/user-123")
    _assert_sanitized(r)


def test_consumer_forecast_live_internal_error_sanitized(client, monkeypatch):
    from src.models.forecaster import prophet_model

    monkeypatch.setattr(prophet_model, "ProphetModel", _boom, raising=True)
    r = client.post(
        "/consumer/forecast/live",
        json={
            "transactions": [
                {
                    "date": "2026-03-05",
                    "merchant": "FreshMart",
                    "amount": -300.0,
                    "category": "Groceries",
                }
            ],
            "horizon_days": 30,
        },
    )
    _assert_sanitized(r)


def test_consumer_budget_live_internal_error_sanitized(client, monkeypatch):
    from src.serving.routes import live as live_module

    class _BoomOptimiser:
        def __init__(self, *a, **k):
            pass

        def optimise(self, *a, **k):
            _boom()

    monkeypatch.setattr(
        live_module, "BudgetOptimiser", _BoomOptimiser, raising=True
    )
    r = client.post(
        "/consumer/budget/live",
        json={
            "transactions": [
                {
                    "date": "2026-03-05",
                    "merchant": "FreshMart",
                    "amount": -300.0,
                    "category": "Groceries",
                }
            ],
            "income": 5000.0,
            "savings_target": 800.0,
        },
    )
    _assert_sanitized(r)


# ── deliberate client errors must stay useful and unchanged ──────────────────


def test_401_detail_unchanged():
    with TestClient(create_app()) as c:
        r = c.post("/v1/classify", json={"transaction": _txn()})
    assert r.status_code == 401
    assert r.json() == {"detail": "Not authenticated"}


def test_422_detail_unchanged(client):
    r = client.post("/consumer/classify/live", json={"merchant": "FreshMart"})
    assert r.status_code == 422
    assert "Field required" in r.text
    assert MARKER not in r.text


def test_404_detail_unchanged(client):
    r = client.get("/v1/definitely-not-a-route")
    assert r.status_code == 404
    assert "Not Found" in r.text


def test_413_detail_unchanged(client, monkeypatch):
    from src.serving.routes import ingest as ingest_module

    # Ingest takes the CSV as a RAW body with Content-Type: text/csv (no
    # multipart); the rows-limit 413 fires right after CSV parsing.
    monkeypatch.setattr(ingest_module, "MAX_ROWS", 1)
    body = (
        "user_id,timestamp,amount,currency,merchant_name,merchant_mcc,account_type,channel\n"
        "user-123,2026-03-05T12:00:00,-100.0,USD,M1,5411,CHECKING,POS\n"
        "user-123,2026-03-06T12:00:00,-100.0,USD,M2,5411,CHECKING,POS\n"
    )
    r = client.post(
        "/consumer/transactions/ingest-csv",
        content=body.encode(),
        headers={"Content-Type": "text/csv; charset=utf-8"},
    )
    assert r.status_code == 413, r.text
    assert "Too many rows" in r.text
    assert MARKER not in r.text


def test_classify_503_when_artifacts_missing(client, monkeypatch):
    monkeypatch.setattr(client.app.state, "classifier", None, raising=False)
    r = client.post("/v1/classify", json={"transaction": _txn()})
    assert r.status_code == 503


# ── server-side diagnostics remain, secrets stay out ─────────────────────────


def test_classify_internal_error_still_logged_server_side(
    client, monkeypatch, caplog
):
    import logging

    from src.serving.routes import classify as classify_module

    monkeypatch.setattr(
        client.app.state.classifier, "predict_proba", _boom, raising=True
    )
    with caplog.at_level(logging.ERROR, logger=classify_module.logger.name):
        client.post("/v1/classify", json={"transaction": _txn()})
    assert any(
        "Classification failed unexpectedly" in rec.getMessage()
        for rec in caplog.records
    )


# ── cache logging redaction (B8) ─────────────────────────────────────────────


def test_redis_url_credentials_not_logged(caplog, monkeypatch):
    import logging
    import sys
    import types

    import src.serving.cache as cache_module

    class _FakeRedis:
        def ping(self):
            return True

    fake_redis_mod = types.ModuleType("redis")
    fake_redis_mod.from_url = lambda url, decode_responses=False: _FakeRedis()
    monkeypatch.setitem(sys.modules, "redis", fake_redis_mod)

    with caplog.at_level(logging.INFO, logger=cache_module.logger.name):
        cache_module.CacheClient(
            redis_url="redis://user:SUPER_SECRET@example:6379/0"
        )
    assert "SUPER_SECRET" not in caplog.text
    assert "example:6379" not in caplog.text
    assert any("Redis" in rec.getMessage() for rec in caplog.records)

