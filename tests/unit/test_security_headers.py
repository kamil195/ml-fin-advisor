"""Step 7 regression tests: HTTP / browser security boundary hardening.

Covers, against the real ``create_app()`` ASGI stack:

* security headers on success, auth-error, and generic-500 responses
* HSTS absent by default and present with the expected value when enabled
* TrustedHostMiddleware: allowed host accepted, unknown host rejected (400)
* CORS: explicit-origin allowlist, no ``allow-credentials``, controlled
  methods/headers; unapproved origins receive no permissive allow-origin
* wildcard ``*`` host/origin misconfiguration is rejected at config time
  (so wildcard + credentials can never occur together)
* /docs, /redoc, /health, /ready remain functional
* Step 5 generic-500 sanitization still holds (and now carries the headers)
* JWT-protected financial endpoints remain protected; ``X-API-Key`` still
  cannot authenticate (legacy API-key surface stays absent)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.auth import require_auth
from src.serving.security import SecurityConfig

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

GENERIC_500 = "Internal server error."
SECURITY_HEADERS = (
    "x-content-type-options",
    "referrer-policy",
    "x-frame-options",
    "permissions-policy",
)


@pytest.fixture
def client():
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        yield c


# ── 1-4. Security headers present ────────────────────────────────────────────


def test_security_headers_on_docs(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["x-frame-options"] == "DENY"
    assert "camera=()" in r.headers["permissions-policy"]
    assert "microphone=()" in r.headers["permissions-policy"]
    assert "geolocation=()" in r.headers["permissions-policy"]


def test_security_headers_on_auth_error(client):
    r = client.post("/v1/classify", json={})
    assert r.status_code == 401
    for h in SECURITY_HEADERS:
        assert h in r.headers


def test_security_headers_on_generic_500(client, make_auth_headers):
    """Step 5 sanitization survives, and the 500 carries the headers too."""
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.dependency_overrides[require_auth] = lambda: (
            (_ for _ in ()).throw(RuntimeError("SECRET_MARKER_NEVER_FOR_CLIENT"))
        )
        r = c.post("/v1/classify", json={}, headers=make_auth_headers())
    assert r.status_code == 500
    assert GENERIC_500 in r.text
    assert "SECRET_MARKER" not in r.text
    assert "RuntimeError" not in r.text
    for h in SECURITY_HEADERS:
        assert h in r.headers


# ── 5-6. HSTS configuration ──────────────────────────────────────────────────


def test_hsts_absent_when_disabled(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "strict-transport-security" not in r.headers


def test_hsts_present_when_enabled(monkeypatch):
    monkeypatch.setenv("SECURITY_HSTS_ENABLED", "true")
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert (
        r.headers["strict-transport-security"]
        == "max-age=31536000; includeSubDomains"
    )


# ── 7-8. Trusted hosts ───────────────────────────────────────────────────────


def test_allowed_host_accepted(client):
    assert client.get("/health").status_code == 200  # default TestClient host


def test_disallowed_host_rejected(client):
    r = client.get("/health", headers={"Host": "unapproved.invalid"})
    assert r.status_code == 400


# ── 9-11. CORS behaviour ─────────────────────────────────────────────────────


def test_allowed_cors_origin_preflight(client):
    r = client.options(
        "/v1/classify",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-credentials" not in r.headers
    assert "POST" in r.headers["access-control-allow-methods"]


def test_unapproved_origin_gets_no_allow_origin(client):
    r = client.get("/docs", headers={"Origin": "https://unapproved.invalid"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_wildcard_with_credentials_configuration_rejected(monkeypatch):
    """Config-level guarantee: '*' can never reach CORSMiddleware."""
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*,http://localhost:5173")
    with pytest.raises(ValueError):
        SecurityConfig.from_env()
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://ok.example.com")
    monkeypatch.setenv("ALLOWED_HOSTS", "*")
    with pytest.raises(ValueError):
        SecurityConfig.from_env()


# ── 12-13. Docs / health behavior preserved ──────────────────────────────────


def test_docs_and_redoc_still_function(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_health_and_ready_still_function(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


# ── 14-16. Prior security semantics intact ───────────────────────────────────


def test_generic_500_behavior_preserved(client, make_auth_headers):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.dependency_overrides[require_auth] = lambda: (
            (_ for _ in ()).throw(RuntimeError("boom"))
        )
        r = c.post("/v1/classify", json={}, headers=make_auth_headers())
    assert r.status_code == 500
    assert r.json()["detail"] == GENERIC_500


def test_financial_endpoints_remain_jwt_protected(client):
    for method, path, json_body in (
        ("post", "/v1/classify", {}),
        ("post", "/consumer/classify/live", {}),
        ("post", "/consumer/advise", {}),
        ("post", "/consumer/transactions/ingest-csv", {}),
    ):
        assert getattr(client, method)(path, json=json_body).status_code == 401
    assert client.get("/v1/forecast/user-123").status_code == 401
    assert client.get("/v1/budget/user-123").status_code == 401


def test_x_api_key_still_cannot_authenticate(client):
    r = client.post(
        "/v1/classify", json={}, headers={"X-API-Key": "legacy-key-not-auth"}
    )
    assert r.status_code == 401
    assert "access-control-allow-credentials" not in r.headers
