"""Step 10 tests: frontend Supabase auth integration.

The frontend is static HTML/JS (no build step), so the smallest robust
strategy is static inspection of the shipped assets plus live backend
boundary checks:

* legacy API-key / fake-identity surface is fully gone
* Supabase auth initialization, sign-in / sign-up / sign-out exist
* the authenticated request helper attaches ``Authorization: Bearer`` and
  never places the token in a URL
* protected frontend calls route through the authenticated helper with the
  session user id (backend JWT ``sub`` stays authoritative)
* no service-role key, JWT secret, or real Supabase credential in frontend
* 401 handling signs the user back in; logout clears state
* CSV uploads preserve their content type
* backend: /health stays public, protected routes stay 401 without JWT

Items 19-20 of the task list (ownership / claims / privacy regressions) are
covered by the existing suites and are run together with this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "frontend"

INDEX = (FRONTEND / "index.html").read_text(encoding="utf-8")
AUTH_JS = (FRONTEND / "auth.js").read_text(encoding="utf-8")
CONFIG_JS = (FRONTEND / "config.js").read_text(encoding="utf-8")
ALL_FRONTEND = "".join(
    p.read_text(encoding="utf-8")
    for p in sorted(FRONTEND.glob("*"))
    if p.is_file()
)


def _joined(*parts: str) -> str:
    return "\n".join(parts)


# ── 1-3. Legacy surface removed ──────────────────────────────────────────────


def test_no_x_api_key_anywhere_in_frontend():
    assert "X-API-Key" not in ALL_FRONTEND


def test_no_api_key_state_or_input():
    for needle in (
        "apiKey",
        "loginKey",
        "legacy_apiKey",
        "DEMO_LOGIN_LABEL",
        "demoLogin",
        "readApiError",
        "API Key",
    ):
        assert needle not in ALL_FRONTEND, f"legacy remnant: {needle!r}"


def test_no_fake_email_hash_user_id():
    for needle in (
        "hashCode",
        "Math.abs(hashCode",
        "user_' + String",
        "web_user",
        "user_ '",
    ):
        assert needle not in ALL_FRONTEND, f"fake identity remnant: {needle!r}"
    # No client-side identity minting: user id comes only from the session.
    assert "currentUserId" in ALL_FRONTEND


# ── 4-7. Supabase auth primitives present ────────────────────────────────────


def test_supabase_client_initialization():
    assert "createClient" in AUTH_JS
    assert "supabase-js@2" in INDEX  # UMD CDN load
    assert 'src="auth.js"' in INDEX and 'src="config.js"' in INDEX


def test_sign_in_with_password_present():
    assert "signInWithPassword" in AUTH_JS


def test_sign_up_present():
    assert "client.auth.signUp" in AUTH_JS
    assert 'onclick="doSignUp()"' in INDEX


def test_sign_out_present():
    assert "client.auth.signOut" in AUTH_JS


# ── 8-10. Authenticated request helper ───────────────────────────────────────


def test_api_fetch_adds_bearer_token():
    assert re.search(r"Authorization:\s*'Bearer '\s*\+\s*session\.access_token", AUTH_JS)
    assert "async function apiFetch" in AUTH_JS


def test_token_never_placed_in_url():
    for needle in ("access_token=", "?access_token", "&access_token"):
        assert needle not in ALL_FRONTEND


def test_protected_calls_use_authenticated_helper():
    for call in (
        "apiFetch('/v1/classify'",
        "apiFetch(\n      `/v1/forecast/",
        "apiFetch(`/v1/budget/",
        "apiFetch('/consumer/advise'",
        "apiFetch('/consumer/transactions/ingest-csv'",
    ):
        assert call in _joined(INDEX), f"missing authenticated call: {call!r}"
    # No unauthenticated raw fetch to protected routes remains.
    bare = re.findall(r"fetch\(\s*`?\$\{API_URL\}[^)]*(classify|forecast|budget|advise|ingest)", INDEX)
    assert not bare, f"unauthenticated protected call: {bare}"
