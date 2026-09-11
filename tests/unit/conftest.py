"""Shared pytest fixtures for the serving/auth test suite.

Provides a deterministic, network-free Supabase-compatible JWT fixture so
endpoint tests can exercise `require_auth`-protected routes. Reuses the exact
RSA-keypair + JWKS pattern already proven in ``tests/unit/test_auth.py``; no
real Supabase network call is ever made (verification runs against a local
``PyJWK`` resolver).
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving import auth as auth_module  # noqa: E402
from src.serving.auth import AuthPrincipal, _AuthConfig  # noqa: E402

KID = "test-key-1"
ISSUER = "https://test-project.supabase.co/auth/v1"
AUDIENCE = "authenticated"


class _StubJWKClient:
    """Minimal ``get_signing_key_from_jwt``-compatible key resolver.

    Mirrors the PyJWKClient contract by refusing tokens whose header ``kid``
    is unknown. Returns a real ``jwt.PyJWK`` built from our generated JWKS so
    the verification path uses genuine signature checks (real crypto, no
    network).
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
    return base64.urlsafe_b64encode(value.to_bytes(length, "big")).rstrip(b"=").decode("ascii")


@pytest.fixture(scope="session")
def supabase_keys():
    """Session-scoped RSA keypair + matching JWKS dict (deterministic content)."""
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
    jwk = pyjwt.PyJWK(jwks["keys"][0])
    return {
        "private_pem": private_pem.decode("ascii"),
        "jwks": jwks,
        "jwk": jwk,
    }


@pytest.fixture
def jwks_client(supabase_keys):
    return _StubJWKClient(supabase_keys["jwk"], KID)


@pytest.fixture
def make_token(supabase_keys):
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
            "exp": now + exp_offset,
            "iat": now,
        }
        if email is not None:
            payload["email"] = email
        if extra:
            payload.update(extra)
        headers = {"kid": kid, "alg": "RS256", "typ": "JWT"}
        return pyjwt.encode(payload, supabase_keys["private_pem"], algorithm="RS256", headers=headers)

    return _make


@pytest.fixture
def _patch_jwks(monkeypatch, jwks_client):
    """Route all JWKS resolution to the local resolver (no network)."""
    monkeypatch.setattr(auth_module, "_get_jwks_client", lambda cfg: jwks_client)


@pytest.fixture
def auth_env(monkeypatch, _patch_jwks):
    """Set a valid SUPABASE_URL so JWT verification can be configured."""
    monkeypatch.setenv("SUPABASE_URL", "https://test-project.supabase.co")
    monkeypatch.delenv("API_KEYS", raising=False)
    return jwks_client


@pytest.fixture
def make_auth_headers(make_token):
    def _headers(*, sub: str = "user-123", **over) -> dict[str, str]:
        return {"Authorization": f"Bearer {make_token(sub=sub, **over)}"}

    return _headers
