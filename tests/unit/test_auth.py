"""Unit tests for ``src/serving/auth.py`` — deterministic, no network access.

Generates a throwaway RSA keypair and a matching JWKS structure, then exercises
the verifier through a local key resolver (PyJWT ``PyJWK`` built from the JWKS
dict). Real crypto, real signature/exp/aud/iss validation — no external calls.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import jwt as pyjwt  # noqa: N812
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving import auth as auth_module  # noqa: E402
from src.serving.auth import (  # noqa: E402
    AuthError,
    AuthPrincipal,
    parse_authorization_header,
    require_auth,
    verify_token,
)

KID = "test-key-1"
ISSUER = "https://test-project.supabase.co/auth/v1"
AUDIENCE = "authenticated"


class _StubJWKClient:
    """Minimal ``get_signing_key_from_jwt``-compatible key resolver.

    Returns a real :class:`jwt.PyJWK` built from our generated JWKS so the
    verification path uses genuine key-parsing/signature checks. Mirrors the
    PyJWKClient contract by refusing tokens whose header ``kid`` is unknown
    (the production client raises a PyJWT error for unknown keys).
    """

    def __init__(self, jwk, kid: str) -> None:
        self._jwk = jwk
        self._kid = kid

    def get_signing_key_from_jwt(self, token: str) -> pyjwt.PyJWK:
        header = pyjwt.get_unverified_header(token)
        if header.get("kid") != self._kid:
            raise ValueError("no signing key found for kid")
        return self._jwk


def _b64url_int(value: int) -> str:
    length = (value.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(value.to_bytes(length, "big")).rstrip(b"=").decode(
        "ascii"
    )


@pytest.fixture(scope="module")
def keys() -> dict:
    """A module-scoped RSA keypair + matching JWKS dict."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    numbers = private_key.public_key().public_numbers()
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": KID,
                "n": _b64url_int(numbers.n),
                "e": _b64url_int(numbers.e),
            }
        ]
    }
    return {"private_pem": private_pem.decode("ascii"), "jwks": jwks}


@pytest.fixture
def jwks_client(keys) -> _StubJWKClient:
    jwk = pyjwt.PyJWK(keys["jwks"]["keys"][0])
    return _StubJWKClient(jwk, KID)


@pytest.fixture
def make_token(keys):
    def _make(
        *,
        sub: str = "user-123",
        aud: str = AUDIENCE,
        iss: str = ISSUER,
        exp_offset: int = 3600,
        email: str | None = None,
        kid: str = KID,
        extra: dict | None = None,
    ) -> str:
        now = int(time.time())
        payload = {
            "sub": sub,
            "aud": aud,
            "iss": iss,
            "iat": now - 60,
            "exp": now + exp_offset,
        }
        if email is not None:
            payload["email"] = email
        if extra:
            payload.update(extra)
        return pyjwt.encode(
            payload, keys["private_pem"], algorithm="RS256", headers={"kid": kid}
        )

    return _make


@pytest.fixture
def app(jwks_client, monkeypatch):
    """A minimal FastAPI app with ``require_auth`` as a route dependency."""
    monkeypatch.setenv("SUPABASE_URL", "https://test-project.supabase.co")
    monkeypatch.setattr(auth_module, "_get_jwks_client", lambda cfg: jwks_client)

    test_app = FastAPI()

    @test_app.get("/me")
    async def me(principal: AuthPrincipal = Depends(require_auth)) -> dict:
        return {"sub": principal.sub, "email": principal.email}

    return test_app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c
# ── Authorization header parsing ─────────────────────────────────────────────


def test_parse_missing_header():
    with pytest.raises(AuthError):
        parse_authorization_header(None)
    with pytest.raises(AuthError):
        parse_authorization_header("")
    with pytest.raises(AuthError):
        parse_authorization_header("   ")


def test_parse_malformed_header():
    with pytest.raises(AuthError):
        parse_authorization_header("Basic dXNlcjpwYXNz")
    with pytest.raises(AuthError):
        parse_authorization_header("Bearer")
    with pytest.raises(AuthError):
        parse_authorization_header("Bearer   ")
    with pytest.raises(AuthError):
        parse_authorization_header("Bearer abc def")


def test_parse_valid_header():
    assert parse_authorization_header("Bearer abc.def.ghi") == "abc.def.ghi"
    assert parse_authorization_header("bearer tok-en") == "tok-en"  # case-insensitive


# ── Direct verify_token (unit) ───────────────────────────────────────────────


def test_verify_token_valid(jwks_client, make_token, monkeypatch):
    # Fail-closed config guard (AUTH STEP 2): verification requires SUPABASE_URL.
    monkeypatch.setenv("SUPABASE_URL", "https://test-project.supabase.co")
    principal = verify_token(
        make_token(sub="user-42"),
        jwks_client=jwks_client,
        audience=AUDIENCE,
        issuer=ISSUER,
    )
    assert principal.sub == "user-42"
    assert principal.email is None


def test_verify_token_carries_email_but_sub_is_authoritative(
    jwks_client, make_token, monkeypatch
):
    # Fail-closed config guard (AUTH STEP 2): verification requires SUPABASE_URL.
    monkeypatch.setenv("SUPABASE_URL", "https://test-project.supabase.co")
    principal = verify_token(
        make_token(sub="user-42", email="someone@example.com"),
        jwks_client=jwks_client,
        audience=AUDIENCE,
        issuer=ISSUER,
    )
    # Identity MUST come from sub, never from the email claim.
    assert principal.sub == "user-42"
    assert principal.email == "someone@example.com"


def test_verify_token_requires_sub(jwks_client, make_token):
    token = make_token(extra={"sub": None})
    with pytest.raises(AuthError):
        verify_token(token, jwks_client=jwks_client, audience=AUDIENCE, issuer=ISSUER)


def test_verify_token_expired(jwks_client, make_token):
    token = make_token(exp_offset=-3600)
    with pytest.raises(AuthError):
        verify_token(token, jwks_client=jwks_client, audience=AUDIENCE, issuer=ISSUER)


def test_verify_token_wrong_audience(jwks_client, make_token):
    token = make_token(aud="other-receiver")
    with pytest.raises(AuthError):
        verify_token(token, jwks_client=jwks_client, audience=AUDIENCE, issuer=ISSUER)


def test_verify_token_wrong_issuer(jwks_client, make_token):
    token = make_token(iss="https://evil.supabase.co/auth/v1")
    with pytest.raises(AuthError):
        verify_token(token, jwks_client=jwks_client, audience=AUDIENCE, issuer=ISSUER)
def test_verify_token_tampered(jwks_client, make_token):
    good = make_token()
    header, payload, signature = good.split(".")
    # Flip one character inside the payload (keeps base64 valid) -> bad signature.
    tampered = payload[:5] + ("A" if payload[5] != "A" else "B") + payload[6:]
    with pytest.raises(AuthError):
        verify_token(
            f"{header}.{tampered}.{signature}",
            jwks_client=jwks_client,
            audience=AUDIENCE,
            issuer=ISSUER,
        )


def test_verify_token_random_junk(jwks_client):
    with pytest.raises(AuthError):
        verify_token(
            "not-a-jwt",
            jwks_client=jwks_client,
            audience=AUDIENCE,
            issuer=ISSUER,
        )


def test_verify_token_unknown_kid(jwks_client, make_token):
    token = make_token(kid="unknown-kid-999")
    with pytest.raises(AuthError):
        verify_token(token, jwks_client=jwks_client, audience=AUDIENCE, issuer=ISSUER)


# ── FastAPI dependency (end-to-end) ──────────────────────────────────────────


def test_dependency_requires_token(client):
    assert client.get("/me").status_code == 401


def test_dependency_rejects_malformed_header(client):
    assert client.get("/me", headers={"Authorization": "garbage"}).status_code == 401
    assert client.get("/me", headers={"Authorization": "Basic abc"}).status_code == 401
    assert client.get("/me", headers={"Authorization": "Bearer"}).status_code == 401


def test_dependency_rejects_invalid_token(client):
    r = client.get("/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401
    # Generic message: no crypto internals leaked.
    assert "Not authenticated" in r.text


def test_dependency_rejects_expired_token(client, make_token):
    token = make_token(exp_offset=-3600)
    r = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_dependency_rejects_tampered_token(client, make_token):
    header, payload, signature = make_token().split(".")
    tampered = payload[:8] + ("A" if payload[8] != "A" else "B") + payload[9:]
    r = client.get(
        "/me", headers={"Authorization": f"Bearer {header}.{tampered}.{signature}"}
    )
    assert r.status_code == 401


def test_dependency_valid_token_returns_principal(client, make_token):
    token = make_token(sub="user-42", email="bob@example.com")
    r = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["sub"] == "user-42"
    assert body["email"] == "bob@example.com"


def test_dependency_email_never_becomes_identity(client, make_token):
    token = make_token(sub="user-42", email="attacker@example.com")
    r = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["sub"] == "user-42"


def test_dependency_missing_config_fails_closed(monkeypatch):
    """With no SUPABASE_URL and no injected resolver, auth must fail closed 401."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)

    bare_app = FastAPI()

    @bare_app.get("/me")
    async def me(principal: AuthPrincipal = Depends(require_auth)) -> dict:
        return {"sub": principal.sub}

    with TestClient(bare_app) as c:
        r = c.get("/me", headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 401
    assert "Not authenticated" in r.text