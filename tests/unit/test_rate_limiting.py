"""SECURITY STEP 6 — identity-aware rate limiting tests.

Covers: under-limit success, 429 + Retry-After, per-subject bucket isolation,
generic 429 body (no identity/internal leakage), public-route immunity,
ownership and JWT-auth primacy (the limiter can never bypass authentication),
X-API-Key irrelevance, limiter key hygiene (no tokens in keys), limiter
internal-failure availability, the disable switch, and bounded memory
(stale-entry pruning + identity-registry cap).
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.rate_limit import (
    DETAIL_429,
    KEY_PREFIX,
    RateLimitConfig,
    SlidingWindowLimiter,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

USER_A = "user-AAA"
USER_B = "user-BBB"


@pytest.fixture
def limited_client(auth_env, make_auth_headers, monkeypatch):
    """App with a small deterministic limit (3 req / 60 s) for USER_A."""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    with TestClient(create_app()) as c:
        c.headers.update(make_auth_headers(sub=USER_A))
        yield c


def _forecast(client: TestClient, path_user: str = USER_A):
    return client.get(f"/v1/forecast/{path_user}")


def _skip_if_no_artifacts(resp) -> None:
    if resp.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")


# ── core limiting behavior ────────────────────────────────────────────────────


def test_under_limit_requests_succeed(limited_client):
    r = _forecast(limited_client)
    _skip_if_no_artifacts(r)
    assert r.status_code == 200, r.text
    assert _forecast(limited_client).status_code == 200
    assert _forecast(limited_client).status_code == 200


def test_exceeding_limit_returns_429_with_retry_after(limited_client):
    _skip_if_no_artifacts(_forecast(limited_client))
    assert _forecast(limited_client).status_code == 200
    assert _forecast(limited_client).status_code == 200
    r4 = _forecast(limited_client)
    assert r4.status_code == 429, r4.text
    assert r4.json() == {"detail": DETAIL_429}
    retry_after = r4.headers.get("Retry-After")
    assert retry_after is not None
    assert int(retry_after) >= 1


def test_429_body_is_generic_and_leaks_nothing(limited_client):
    for _ in range(5):
        _forecast(limited_client)
    r = _forecast(limited_client)
    assert r.status_code == 429
    text = r.text
    assert USER_A not in text
    assert "Bearer" not in text
    assert "eyJ" not in text  # JWT fragment
    assert "rate_limit:user" not in text  # internal key namespace
    assert "127.0.0.1" not in text
    assert "testclient" not in text.lower()


def test_public_health_ready_are_not_limited(limited_client):
    for _ in range(5):
        _forecast(limited_client)  # exhaust USER_A's financial bucket
    for _ in range(6):
        assert limited_client.get("/health").status_code == 200
        assert limited_client.get("/ready").status_code == 200


# ── identity, auth primacy, and bypass resistance ─────────────────────────────


def test_different_subjects_have_separate_buckets(auth_env, make_auth_headers, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    with TestClient(create_app()) as c:
        c.headers.update(make_auth_headers(sub=USER_A))
        _skip_if_no_artifacts(_forecast(c))
        assert _forecast(c).status_code == 200
        assert _forecast(c).status_code == 429  # USER_A exhausted
        c.headers.update(make_auth_headers(sub=USER_B))  # different subject
        r = _forecast(c, USER_B)
        _skip_if_no_artifacts(r)
        assert r.status_code == 200  # USER_B unaffected


def test_ownership_check_unaffected_by_limiter(limited_client):
    """Cross-user path stays 403 (limiter present, ownership authoritative)."""
    r = _forecast(limited_client, USER_B)
    assert r.status_code == 403


def test_missing_jwt_gets_401_not_429(auth_env, monkeypatch):
    """The limiter runs after auth — missing JWT is an auth failure, always."""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1")
    with TestClient(create_app()) as c:
        r = c.get("/v1/forecast/anonymous")
    assert r.status_code == 401
    assert r.json() == {"detail": "Not authenticated"}


def test_x_api_key_cannot_authenticate(auth_env, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1")
    with TestClient(create_app()) as c:
        r = c.get("/v1/forecast/anonymous", headers={"X-API-Key": "legacy-key"})
    assert r.status_code == 401


def test_x_api_key_header_does_not_bypass_exhausted_bucket(limited_client):
    for _ in range(5):
        _forecast(limited_client)
    r = limited_client.get(
        f"/v1/forecast/{USER_A}", headers={"X-API-Key": "legacy-key"}
    )
    assert r.status_code == 429


def test_limiter_keys_never_contain_tokens(limited_client):
    _skip_if_no_artifacts(_forecast(limited_client))
    keys = limited_client.app.state.rate_limiter.snapshot_keys()
    assert keys
    for k in keys:
        assert k.startswith(KEY_PREFIX)
        assert "Bearer" not in k
        assert "eyJ" not in k


# ── availability & configuration ──────────────────────────────────────────────


def test_limiter_internal_failure_returns_safe_503(
    limited_client, monkeypatch, caplog
):
    limiter = limited_client.app.state.rate_limiter

    def _boom(_key, _now=None):
        raise RuntimeError("limiter exploded secret-marker")

    monkeypatch.setattr(limiter, "check", _boom)
    with caplog.at_level(logging.ERROR, logger="src.serving.rate_limit"):
        r = _forecast(limited_client)
    assert r.status_code == 503
    assert r.json() == {"detail": "Service temporarily unavailable. Please retry later."}
    assert "secret-marker" not in r.text
    assert "secret-marker" not in caplog.text
    assert USER_A not in caplog.text
    assert any("rate limiter failure" in rec.getMessage() for rec in caplog.records)


def test_disabled_limiter_never_returns_429(auth_env, make_auth_headers, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1")
    with TestClient(create_app()) as c:
        c.headers.update(make_auth_headers(sub=USER_A))
        _skip_if_no_artifacts(_forecast(c))
        for _ in range(6):
            assert _forecast(c).status_code != 429


# ── unit: bounded memory (stale pruning + identity cap) ───────────────────────


def test_sliding_window_prunes_stale_entries():
    limiter = SlidingWindowLimiter(RateLimitConfig(True, 2, 10.0))
    assert limiter.check("k", now=0.0) == (True, 0)
    assert limiter.check("k", now=1.0) == (True, 0)
    allowed, retry = limiter.check("k", now=2.0)
    assert not allowed
    assert retry >= 1
    # Window slides: long-expired hits are pruned and the request is allowed.
    allowed, _ = limiter.check("k", now=100.0)
    assert allowed
    assert len(limiter._windows["k"]) == 1


def test_identity_registry_is_bounded():
    limiter = SlidingWindowLimiter(RateLimitConfig(True, 5, 60.0), max_identities=5)
    for i in range(10):
        limiter.check(f"{KEY_PREFIX}u{i}", now=1.0)
    assert len(limiter.snapshot_keys()) == 5



def test_capacity_preserves_active_quota_until_expiry():
    limiter = SlidingWindowLimiter(RateLimitConfig(True, 1, 60), max_identities=1)
    assert limiter.check("active", now=0) == (True, 0)
    assert limiter.check("new", now=1) == (False, 59)
    assert limiter.snapshot_keys() == ["active"]
    assert limiter.check("active", now=2) == (False, 58)
    assert limiter.check("new", now=60) == (True, 0)
    assert limiter.snapshot_keys() == ["new"]

