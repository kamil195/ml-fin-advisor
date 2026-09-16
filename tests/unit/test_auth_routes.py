"""
AUTH STEP 2 — tests for the authentication boundary.

Two layers:

1. Pure authentication behaviour on a minimal probe app (no model artifacts):
   ``require_auth`` accept/reject paths and the legacy ``verify_api_key``
   fail-closed behaviour.

2. Production wiring: the real ``create_app()`` route table is exercised so a
   valid Supabase JWT with ``API_KEYS`` empty demonstrably reaches the
   financial routes (artifact-absent handlers answer 503/200 — never 401),
   while missing/invalid/malformed credentials are rejected with 401, and the
   operational routes (/health, /ready, /admin/generate-key) stay public.

No test in this module performs a real Supabase/JWKS network call: JWKS
resolution is routed to a locally generated RSA key (see conftest.py), and the
missing-configuration test fails before any client is constructed.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.auth import AuthPrincipal, require_auth

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# ── shared payloads ──────────────────────────────────────────────────────────

TXN = {
    "user_id": "user-123",  # must equal the authenticated JWT sub (AUTH STEP 3)
    "timestamp": "2026-03-05T12:00:00",
    "amount": -100.0,
    "currency": "USD",
    "merchant_name": "FreshMart",
    "merchant_mcc": 5411,
    "account_type": "CHECKING",
    "channel": "POS",
}


def _advise_txn(**over) -> dict:
    base = {
        "user_id": "user-123",  # authenticated JWT sub (AUTH STEP 3)
        "timestamp": "2026-03-05T12:00:00",
        "amount": -100.0,
        "merchant_name": "Merchant",
        "merchant_mcc": 5411,
        "account_type": "CHECKING",
        "channel": "POS",
        "category_l1": "FOOD & DINING",
        "category_l2": "Groceries",
        "confidence": 0.95,
    }
    base.update(over)
    return base


def _advise_body() -> dict:
    """Minimal valid /consumer/advise payload (no artifacts required)."""
    return {
        "transactions": [
            _advise_txn(
                amount=5000.0,
                merchant_name="Payroll - Acme Corp",
                merchant_mcc=0,
                channel="TRANSFER",
                category_l1="FINANCIAL",
                category_l2="Income",
            ),
            _advise_txn(amount=-1500.0, merchant_name="LandlordCo", merchant_mcc=6513,
                        channel="TRANSFER", category_l1="HOUSING",
                        category_l2="Rent/Mortgage"),
            _advise_txn(amount=-300.0, merchant_name="FreshMart"),
            _advise_txn(amount=-300.0, merchant_name="Bistro One", merchant_mcc=5812,
                        category_l2="Restaurants"),
        ]
    }


CSV_BODY = (
    "user_id,timestamp,amount,currency,merchant_name,merchant_mcc,account_type,channel\n"
    "user-123,2026-03-05T12:00:00,-100.0,USD,FreshMart,5411,CHECKING,POS\n"
)

# (method, path, json-body, raw-body) for every financial route under protection
PROTECTED_ROUTES = [
    ("POST", "/v1/classify", {"transaction": TXN}, None),
    ("GET", "/v1/forecast/user-123", None, None),
    ("GET", "/v1/budget/user-123", None, None),
    ("POST", "/consumer/advise", _advise_body(), None),
    ("POST", "/consumer/transactions/ingest-csv", None, CSV_BODY),
    ("POST", "/consumer/classify/live", {"transaction": TXN}, None),
    ("POST", "/consumer/forecast/live", {}, None),
    ("POST", "/consumer/budget/live", {}, None),
]

# ── probe apps (pure auth behaviour, no artifacts) ───────────────────────────


def _protected_probe_app() -> FastAPI:
    """Minimal app exposing one require_auth-protected route (bare_app pattern)."""
    app = FastAPI()

    @app.get("/probe")
    async def probe(principal: AuthPrincipal = Depends(require_auth)):
        return {"sub": principal.sub}

    return app


@pytest.fixture
def probe_client(auth_env):
    return TestClient(_protected_probe_app())


@pytest.fixture
def bare_client(auth_env):
    """Production app, no Authorization header (anonymous caller)."""
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def app_client(auth_env, make_auth_headers):
    """Production app with a valid mocked JWT on every request."""
    with TestClient(create_app()) as c:
        c.headers.update(make_auth_headers())
        yield c

# ══ 1. require_auth accept/reject paths (probe app) ══════════════════════════


def test_probe_without_token_rejected(probe_client):
    assert probe_client.get("/probe").status_code == 401


def test_probe_with_malformed_authorization_header_rejected(probe_client):
    for header in ("not-a-bearer-token", "Bearer", "Bearer  ", "Basic dXNlcjpwYXNz"):
        r = probe_client.get("/probe", headers={"Authorization": header})
        assert r.status_code == 401, header


def test_probe_with_invalid_token_rejected(probe_client):
    r = probe_client.get("/probe", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_probe_with_expired_token_rejected(probe_client, make_auth_headers):
    r = probe_client.get("/probe", headers=make_auth_headers(exp_offset=-10))
    assert r.status_code == 401


def test_probe_with_wrong_audience_rejected(probe_client, make_auth_headers):
    r = probe_client.get("/probe", headers=make_auth_headers(aud="other-app"))
    assert r.status_code == 401


def test_probe_with_valid_token_accepted(probe_client, make_auth_headers):
    r = probe_client.get("/probe", headers=make_auth_headers(sub="user-123"))
    assert r.status_code == 200
    assert r.json()["sub"] == "user-123"


# ══ 2. fail-closed configuration ═════════════════════════════════════════════


def test_missing_supabase_url_fails_closed(monkeypatch, make_auth_headers):
    """No SUPABASE_URL → 401 on protected route; no crash, no silent open access."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    with TestClient(_protected_probe_app()) as c:
        r = c.get("/probe", headers=make_auth_headers())
    assert r.status_code == 401


def test_production_app_fails_closed_without_supabase_url(monkeypatch, make_auth_headers):
    """Same fail-closed guarantee through the real create_app() wiring."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    with TestClient(create_app()) as c:
        r = c.get("/v1/forecast/some-user", headers=make_auth_headers())
    assert r.status_code == 401




# ══ 3. production route wiring ═══════════════════════════════════════════════


def test_all_financial_routes_reject_anonymous(bare_client):
    """Every protected financial route: no Authorization header → 401."""
    for method, path, json_body, raw in PROTECTED_ROUTES:
        headers = {"Content-Type": "text/csv"} if raw else {}
        r = bare_client.request(method, path, json=json_body, content=raw, headers=headers)
        assert r.status_code == 401, f"{method} {path} → {r.status_code}"


def test_all_financial_routes_reject_invalid_token(bare_client):
    for method, path, json_body, raw in PROTECTED_ROUTES:
        headers = {"Authorization": "Bearer invalid.token.value"}
        if raw:
            headers["Content-Type"] = "text/csv"
        r = bare_client.request(method, path, json=json_body, content=raw, headers=headers)
        assert r.status_code == 401, f"{method} {path} → {r.status_code}"


def test_valid_jwt_reaches_financial_routes_with_empty_api_keys(app_client):
    """THE Step 2 regression: valid JWT + empty API_KEYS must NOT be blocked
    by the legacy layer (previously the global verify_api_key 401'd first).

    Artifact-absent handlers answer 503 (data not loaded), 422 (body
    validation on live probes) or 200 (advise needs no artifacts) — anything
    but 401 proves the request passed the JWT layer and reached the handler.
    """
    for method, path, json_body, raw in PROTECTED_ROUTES:
        headers = {"Content-Type": "text/csv"} if raw else None
        r = app_client.request(method, path, json=json_body, content=raw, headers=headers)
        assert r.status_code != 401, f"{method} {path} blocked: {r.status_code}"
        # 404 = subject not present in the per-user budget artifact (no fallback).
        assert r.status_code in (200, 404, 422, 503), f"{method} {path} → {r.status_code}"


def test_v1_classify_passes_jwt_layer_with_empty_api_keys(app_client):
    r = app_client.post("/v1/classify", json={"transaction": TXN})
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    assert r.status_code == 200, r.text


def test_v1_forecast_ownership_param_reaches_handler(app_client):
    """Valid JWT (sub=user-123) reaches GET /v1/forecast/{user_id} for its OWN
    path — the caller-controlled identity is no longer accepted.

    200 = serving artifacts present; 503 = artifacts absent (fresh checkout).
    Both prove the request passed the JWT layer and the ownership gate.
    """
    r = app_client.get("/v1/forecast/user-123")
    assert r.status_code in (200, 503)
    if r.status_code == 503:
        assert "Forecast data not available" in r.json()["detail"]

    # Cross-user path is refused regardless of artifacts (AUTH STEP 3).
    assert app_client.get("/v1/forecast/some-user").status_code == 403


def test_v1_budget_ownership_param_reaches_handler(app_client):
    """Valid JWT (sub=user-123) reaches GET /v1/budget/{user_id} for its OWN
    path — the caller-controlled identity is no longer accepted.

    200 = exact-match entry for the subject in the serving artifact;
    404 = subject is not in the per-user artifact (no fallback);
    503 = artifacts absent (fresh checkout);
    all prove the request passed the JWT layer and the ownership gate.
    """
    r = app_client.get("/v1/budget/user-123")
    assert r.status_code in (200, 404, 503)
    if r.status_code == 503:
        assert "Budget data not available" in r.json()["detail"]
    if r.status_code == 404:
        assert "No budget data for user" in r.json()["detail"]

    # Cross-user path is refused regardless of artifacts (AUTH STEP 3).
    assert app_client.get("/v1/budget/some-user").status_code == 403


def test_consumer_advise_full_success_with_valid_jwt(app_client):
    """/consumer/advise needs no model artifacts → full 200 through JWT auth."""
    r = app_client.post("/consumer/advise", json=_advise_body())
    assert r.status_code == 200, r.text
    body = r.json()
    assert "decision_id" in body and "recommendation" in body


# ══ 4. public routes remain public ═══════════════════════════════════════════


def test_health_remains_public(bare_client):
    r = bare_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_ready_remains_public(bare_client):
    r = bare_client.get("/ready")
    assert r.status_code == 200
    assert "components" in r.json()


def test_admin_generate_key_removed(bare_client):
    """STEP 4: the legacy unauthenticated admin key-generation route is gone."""
    r = bare_client.get("/admin/generate-key")
    assert r.status_code in (404, 405)


def test_x_api_key_cannot_authenticate_financial_route(bare_client):
    """STEP 4: legacy API-key authentication is removed entirely. A JWT-free
    financial request carrying X-API-Key must still be rejected with 401 —
    Supabase JWT remains the sole authentication authority."""
    r = bare_client.get("/v1/budget/user-123", headers={"X-API-Key": "any-key-value"})
    assert r.status_code == 401


def test_docs_and_openapi_remain_public(bare_client):
    assert bare_client.get("/docs").status_code == 200
    assert bare_client.get("/openapi.json").status_code == 200
