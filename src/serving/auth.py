"""Supabase JWT verification for the serving layer (authentication foundation).

Strategy
--------
Supabase signs access tokens with asymmetric algorithms and publishes the
current signing keys at ``{SUPABASE_URL}/auth/v1/.well-known/jwks.json``. The
project's active signing key is ECC P-256 (``ES256``); ``RS256`` also remains
supported for verification (legacy/rotated RSA keys). This module verifies
tokens against that JWKS using the PyJWT ``PyJWKClient``. The key set is cached
by the client, so verification normally performs **no** network I/O per request;
a new key set is fetched only when the JWT references an unknown ``kid``
(Supabase key rotation).

Only an explicit asymmetric algorithm allowlist is accepted: ``ES256`` and
``RS256``. The token header ``alg`` is never trusted dynamically -- ``none``,
``HS256`` (the legacy Supabase shared-secret scheme) and every other algorithm
are rejected. This backend deliberately has **no** shared-secret/JWT-secret
verification path, so legacy HS256 tokens are not accepted here.

Only verification lives here -- deliberately:

* Supabase owns signup/login/password handling and token issuance. This project
  never handles user credentials.
* Identity is taken **exclusively** from the verified JWT ``sub`` claim. Request
  bodies, URL path segments and ``email`` claims are never used as identity.
* Tokens must be presented as ``Authorization: Bearer <JWT>``.

Configuration (environment variables)
-------------------------------------
``SUPABASE_URL``
    Base URL of the Supabase project, e.g. ``https://<project-ref>.supabase.co``.
    The JWKS URL and the expected issuer are derived from this value, so no
    redundant URL variable is required. REQUIRED in production; missing
    configuration is a hard failure (fail closed).

``SUPABASE_JWT_AUDIENCE``
    Expected JWT ``aud`` claim. Supabase access tokens carry ``"authenticated"``
    by default; override only for a non-default receiver setup.

Behaviour
---------
Fail closed: missing/partial configuration, a missing/malformed Authorization
header, an unsupported algorithm (anything outside the explicit
``ES256``/``RS256`` allowlist, including ``HS256`` and ``none``), or an
invalid/expired/tampered/wrong-audience/wrong-issuer token is rejected. The
FastAPI dependency maps every failure to a generic HTTP 401. JWT contents,
Authorization headers and signing secrets are never logged.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

# Explicit asymmetric allowlist, never derived from the token header. Supabase's
# current signing key is ECC P-256 (ES256); RS256 remains for legacy/rotated RSA
# keys. HS256/shared-secret verification is intentionally NOT offered here
# (algorithm-confusion and shared-secret leak surface) -- fail closed instead.
_ALGORITHMS = ("ES256", "RS256")
_DEFAULT_AUDIENCE = "authenticated"
_GENERIC_DETAIL = "Not authenticated"


class AuthError(Exception):
    """Raised when authentication cannot be established (maps to HTTP 401)."""


@dataclass(frozen=True)
class AuthPrincipal:
    """Minimal verified identity extracted from a JWT.

    ``sub`` is authoritative and the only identity used for authorization.
    ``email`` is carried for display convenience and is NEVER used as identity.
    """

    sub: str
    email: str | None = None


@dataclass(frozen=True)
class _AuthConfig:
    supabase_url: str
    jwks_url: str | None
    issuer: str | None
    audience: str

    @classmethod
    def from_env(cls) -> "_AuthConfig":
        url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
        jwks_url = f"{url}/auth/v1/.well-known/jwks.json" if url else None
        issuer = f"{url}/auth/v1" if url else None
        audience = (
            os.environ.get("SUPABASE_JWT_AUDIENCE", "").strip() or _DEFAULT_AUDIENCE
        )
        return cls(url, jwks_url, issuer, audience)


_jwks_client: jwt.PyJWKClient | None = None
_jwks_client_lock = threading.Lock()


def _get_jwks_client(cfg: _AuthConfig) -> jwt.PyJWKClient:
    """Return a cached :class:`jwt.PyJWKClient` for the configured JWKS URL.

    The client caches fetched keys, so a new key set is fetched only once per
    process (or when an unknown ``kid`` is seen after key rotation), not per
    request.
    """
    if cfg.jwks_url is None:
        raise AuthError(
            "Supabase authentication is not configured: SUPABASE_URL is not set"
        )
    global _jwks_client
    with _jwks_client_lock:
        if _jwks_client is None:
            _jwks_client = jwt.PyJWKClient(cfg.jwks_url, cache_keys=True)
        return _jwks_client


def parse_authorization_header(authorization: str | None) -> str:
    """Extract and return the Bearer token from an Authorization header.

    Raises :class:`AuthError` for missing/malformed headers. The header value
    is never logged.
    """
    if not authorization:
        raise AuthError("missing Authorization header")
    scheme, _, rest = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise AuthError("Authorization scheme must be Bearer")
    token = rest.strip()
    if not token or " " in token:
        raise AuthError("malformed Authorization header")
    return token


def verify_token(
    token: str,
    *,
    jwks_client: jwt.PyJWKClient | None = None,
    audience: str | None = None,
    issuer: str | None = None,
) -> AuthPrincipal:
    """Verify a JWT and return the authenticated principal.

    ``jwks_client``, ``audience`` and ``issuer`` may be injected (tests, callers
    with explicit configuration); defaults come from environment
    configuration. Raises :class:`AuthError` on any failure and never lets
    underlying crypto-library exceptions escape (nothing sensitive is exposed).
    """
    cfg = _AuthConfig.from_env()
    if not cfg.supabase_url:
        # Fail-closed: JWT authentication cannot be configured without the
        # Supabase project URL. Return a 401 (not a startup crash).
        raise AuthError(
            "Supabase authentication is not configured: SUPABASE_URL is not set"
        )
    client = jwks_client if jwks_client is not None else _get_jwks_client(cfg)
    aud = audience if audience is not None else cfg.audience
    iss = issuer if issuer is not None else cfg.issuer

    try:
        signing_key = client.get_signing_key_from_jwt(token)
        payload: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=_ALGORITHMS,
            audience=aud,
            issuer=iss,
            options={"require": ["sub", "exp"], "verify_iss": iss is not None},
        )
    except AuthError:
        raise
    except Exception as exc:  # defensive: never leak library internals to callers
        logger.debug("JWT verification failed: %s", type(exc).__name__)
        raise AuthError("invalid or expired token") from exc

    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise AuthError("token is missing a subject claim")
    email = payload.get("email")
    return AuthPrincipal(sub=subject, email=email if isinstance(email, str) else None)


async def require_auth(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> AuthPrincipal:
    """FastAPI dependency: require a valid Supabase Bearer JWT.

    Returns the verified :class:`AuthPrincipal` or raises a generic HTTP 401.
    Applied at the router level to the financial routes in ``create_app()``.
    """
    try:
        token = parse_authorization_header(authorization)
        return verify_token(token)
    except AuthError:
        raise HTTPException(status_code=401, detail=_GENERIC_DETAIL)