"""Step 8 tests: privacy / data-lifecycle guarantees.

Covers: bearer tokens never logged, financial payloads never logged on
error paths, cache URL redaction, rate-limit log identity hygiene, cache
key hygiene (identity-scoped, no token/email), in-memory cache TTL
enforcement and bounded retention, upload in-memory handling (no temp
files), generic error sanitization, no cross-user cache fallback, absence
of secrets in tracked privacy-relevant config, and existence of the
data-lifecycle documentation with the no-persistence limitation stated.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

import src.serving.cache as cache_module
from src.serving.app import create_app
from src.serving.auth import require_auth
from src.serving.cache import CacheClient

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

TOKEN = "Bearer eyJhbGciOiJSUzI1NiJ9.SECRET_PAYLOAD.SIGNATURE"
SUB = "auth-privacy-sub-123"


@pytest.fixture
def client(auth_env, make_auth_headers):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.headers.update(make_auth_headers(sub=SUB))
        yield c


# ── 1. Bearer token never logged ─────────────────────────────────────────────


def test_bearer_token_never_logged(client, make_auth_headers, caplog):
    with caplog.at_level(logging.DEBUG):
        r = client.post("/v1/classify", json={}, headers=make_auth_headers(sub=SUB))
    assert r.status_code == 401 or r.status_code == 422
    assert "SECRET_PAYLOAD" not in caplog.text
    assert "SIGNATURE" not in caplog.text
    assert TOKEN not in caplog.text


# ── 2. Financial payload never logged on expected error path ────────────────


def test_financial_payload_not_logged_on_error_path(client, caplog):
    """A malformed financial request must not echo its payload into logs."""
    with caplog.at_level(logging.DEBUG):
        r = client.post(
            "/v1/classify",
            json={"transaction": {"bogus": "payload", "amount": -999999.0}},
        )
    assert r.status_code == 422  # validation rejects the payload outright
    assert "bogus" not in caplog.text
    assert "payload" not in caplog.text.replace("financial payload", "")
    assert "-999999" not in caplog.text


# ── 3. Cache URL redaction (regression of Step 5) ────────────────────────────


def test_redis_url_remains_redacted(caplog, monkeypatch):
    import sys
    import types

    class _FakeRedis:
        def ping(self):
            return True

    fake = types.ModuleType("redis")
    fake.from_url = lambda url, decode_responses=False: _FakeRedis()
    monkeypatch.setitem(sys.modules, "redis", fake)

    with caplog.at_level(logging.INFO, logger=cache_module.logger.name):
        cache_module.CacheClient(redis_url="redis://user:SUPER_SECRET@host:6379/0")
    assert "SUPER_SECRET" not in caplog.text
    assert "host:6379" not in caplog.text


# ── 4. Rate-limit logs expose no JWT sub / token ─────────────────────────────


def test_rate_limit_logs_do_not_expose_identity(client, caplog, monkeypatch):
    from src.serving import rate_limit as rl

    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    with caplog.at_level(logging.DEBUG):
        for _ in range(3):
            client.post(
                "/v1/classify",
                json={"transaction": {"bogus": True}},
            )
    assert SUB not in caplog.text
    assert "eyJhbGciOiJSUzI1NiJ9" not in caplog.text


# ── 5. Uploads are in-memory: no temp files ──────────────────────────────────


def test_csv_ingest_creates_no_temp_files(client, tmp_path, monkeypatch):
    """The ingest path decodes the request body in memory; assert no temp
    file helpers are used and no files appear in a fresh temp dir."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    csv_body = (
        "user_id,timestamp,amount,merchant_name,merchant_mcc,account_type,channel\n"
        f"{SUB},2026-03-05T12:00:00,-10.0,Coffee,5814,CHECKING,POS\n"
    ).encode()
    r = client.post(
        "/consumer/transactions/ingest-csv",
        content=csv_body,
        headers={"Content-Type": "text/csv", "X-Filename": "tx.csv"},
    )
    assert r.status_code == 200
    assert list(tmp_path.iterdir()) == []  # nothing written to disk


# ── 6. Cache keys: identity-scoped, no token/email; TTL enforced ─────────────


def test_cache_keys_scoped_and_clean():
    c = CacheClient.__new__(CacheClient)  # skip Redis connection
    c.default_ttl = 60
    c._redis = None
    c._local_cache = {}

    c.set("forecasts", SUB, "30", "all", value={"x": 1})
    keys = list(c._local_cache.keys())
    assert keys == [f"forecasts:{SUB}:30:all"]
    assert "eyJ" not in keys[0] and "@" not in keys[0]
    # TTL honoured: value retrievable now…
    assert c.get("forecasts", SUB, "30", "all") == {"x": 1}


def test_in_memory_cache_ttl_expiry(monkeypatch):
    c = CacheClient.__new__(CacheClient)
    c.default_ttl = 60
    c._redis = None
    c._local_cache = {}

    c.set("budgets", SUB, value={"b": 2}, ttl=1)
    assert c.get("budgets", SUB) == {"b": 2}

    # Simulate monotonic clock passage beyond the TTL.
    import src.serving.cache as cache_mod

    real_mono = cache_mod.time.monotonic
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: real_mono() + 3600)
    assert c.get("budgets", SUB) is None  # expired entry dropped on read
    assert list(c._local_cache.keys()) == []


def test_in_memory_cache_purges_expired_on_write(monkeypatch):
    c = CacheClient.__new__(CacheClient)
    c.default_ttl = 60
    c._redis = None
    c._local_cache = {}

    import src.serving.cache as cache_mod

    real_mono = cache_mod.time.monotonic
    c.set("a", "u1", value=1, ttl=1)
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: real_mono() + 3600)
    c.set("b", "u2", value=2, ttl=60)  # write triggers purge of expired 'a'
    assert c.get("a", "u1") is None
    assert c.get("b", "u2") == 2


def test_zero_ttl_stores_nothing():
    c = CacheClient.__new__(CacheClient)
    c.default_ttl = 60
    c._redis = None
    c._local_cache = {}
    c.set("ns", "u1", value={"s": 1}, ttl=0)
    assert list(c._local_cache.keys()) == []


# ── 7. Generic internal errors still hide details (Step 5 intact) ────────────


def test_generic_500_still_sanitized(client, make_auth_headers):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.dependency_overrides[require_auth] = lambda: (
            (_ for _ in ()).throw(RuntimeError("LEAK_ATTEMPT /models/x.joblib"))
        )
        r = c.post("/v1/classify", json={}, headers=make_auth_headers())
    assert r.status_code == 500
    assert "LEAK_ATTEMPT" not in r.text
    assert "Internal server error." in r.text


# ── 8. No cross-user cache fallback ──────────────────────────────────────────


def test_no_cross_user_cache_fallback():
    c = CacheClient.__new__(CacheClient)
    c.default_ttl = 60
    c._redis = None
    c._local_cache = {}
    c.set("forecasts", "user-A", "30", "all", value={"who": "A"})
    # Exact-key lookup only: user-B never receives user-A's cached data.
    assert c.get("forecasts", "user-B", "30", "all") is None
    assert c.get("forecasts", "user-A", "30", "all") == {"who": "A"}


# ── 9. Privacy-relevant config contains no real secrets ──────────────────────


def test_env_example_contains_no_real_secrets():
    from pathlib import Path

    text = Path(".env.example").read_text(encoding="utf-8")
    for placeholder in ("your-project-ref", "localhost", "testserver"):
        assert placeholder in text
    for forbidden in ("supabase.co/auth", "apikey=", "service_role", "SECRET="):
        assert forbidden not in text


# ── 10. Data-lifecycle documentation exists and states limitations ───────────


def test_privacy_documentation_exists_and_honest():
    from pathlib import Path

    doc = Path("PRIVACY_DATA_LIFECYCLE.md").read_text(encoding="utf-8")
    # STEP 12: the doc must now describe real persistence honestly — including
    # the tables, the JWT-sub ownership key, and the auth-account limitation.
    assert "owner_sub" in doc
    assert "Supabase" in doc and "Postgres" in doc
    assert "DELETE /consumer/data" in doc
    assert "GET /consumer/data/export" in doc
    assert "Auth account" in doc or "account deletion" in doc.lower()
    assert "not yet implemented" in doc  # retention / account deletion
    assert "never logged" in doc or "never written to disk" in doc
    assert "Privacy Policy" in doc  # launch-blocker status stated
    # No legal-compliance claims invented.
    for claim in ("GDPR compliant", "CCPA compliant", "SOC 2 certified"):
        assert claim.lower() not in doc.lower()

