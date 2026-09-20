"""Step 11 tests: frontend security hardening (static asset inspection).

The frontend is static HTML/JS with no build step, so these tests inspect the
shipped assets. Every assertion either proves a risky pattern is absent or
proves the safe replacement is present. Claims/privacy guarantees are enforced
by the existing suites (test_claims_integrity.py, test_privacy_data_handling.py)
which run alongside this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "frontend"

INDEX = (FRONTEND / "index.html").read_text(encoding="utf-8")
AUTH_JS = (FRONTEND / "auth.js").read_text(encoding="utf-8")
CONFIG_JS = (FRONTEND / "config.js").read_text(encoding="utf-8")
ALL_FRONTEND = "".join(
    p.read_text(encoding="utf-8", errors="ignore")
    for p in sorted(FRONTEND.glob("*"))
    if p.is_file()
)

HTML_SINKS = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write")
DANGEROUS_EVAL = ("eval(", "new Function")


# ── 1-2. DOM injection safety ────────────────────────────────────────────────


@pytest.mark.parametrize("sink", HTML_SINKS)
def test_no_html_injection_sinks(sink: str) -> None:
    """No raw HTML sink anywhere: auth messages, backend errors, model output,
    transaction descriptions, category labels and CSV-derived values are all
    rendered through DOM construction + textContent."""
    assert sink not in ALL_FRONTEND, f"unsafe HTML sink present: {sink}"


def test_auth_messages_use_textcontent() -> None:
    assert "m.textContent = message" in AUTH_JS
    assert "classList.toggle('error'" in AUTH_JS  # a class, never HTML


def test_rendering_uses_safe_dom_apis() -> None:
    # The forecast/budget/advice renderers build nodes explicitly.
    assert "document.createElement" in INDEX
    assert INDEX.count("createElement") >= 20


# ── 3-4. no token logging ────────────────────────────────────────────────────


def test_no_console_logging_at_all() -> None:
    """No logging in the frontend means no token/state/payload leakage."""
    assert "console." not in ALL_FRONTEND


def test_tokens_never_reach_logs_dom_urls_or_storage() -> None:
    bad = (
        "console.",
        "innerHTML",
        "textContent",
        "localStorage",
        "sessionStorage",
        "document.cookie",
        "href",
        "src=",
    )
    lines = [ln for ln in ALL_FRONTEND.splitlines() if "access_token" in ln]
    assert lines, "access_token should only appear on the Authorization path"
    for line in lines:
        assert not any(t in line for t in bad), f"token leak path: {line.strip()}"
    assert "refresh_token" not in ALL_FRONTEND


# ── 5-9. no legacy auth surface / no client-authoritative identity ───────────


@pytest.mark.parametrize(
    "needle",
    ("service_role", "JWT_SECRET", "X-API-Key", "apiKey", "legacy_apiKey", "API Key"),
)
def test_no_privileged_credential_or_api_key_surface(needle: str) -> None:
    assert needle not in ALL_FRONTEND, f"forbidden credential surface: {needle!r}"


def test_public_config_holds_placeholders_only() -> None:
    """config.js carries public placeholders — never a real credential."""
    assert "YOUR_SUPABASE_URL" in CONFIG_JS
    assert "YOUR_SUPABASE_ANON_KEY" in CONFIG_JS


# ── 10-11. authenticated request path ───────────────────────────────────────


def test_protected_requests_use_authenticated_helper() -> None:
    for call in (
        "apiFetch('/v1/classify'",
        "apiFetch(`/v1/budget/",
        "apiFetch('/consumer/advise'",
        "apiFetch('/consumer/transactions/ingest-csv'",
    ):
        assert call in INDEX, f"protected call not authenticated: {call!r}"


def test_bearer_token_attached_and_refresh_is_bounded() -> None:
    assert re.search(
        r"Authorization:\s*'Bearer '\s*\+\s*session\.access_token", AUTH_JS
    )
    assert "refreshSession" in AUTH_JS
    assert AUTH_JS.count("response.status === 401") == 1  # one retry, no loop


# ── 12-14. links, dynamic code, supply chain ────────────────────────────────


def test_blank_target_links_are_protected() -> None:
    for tag in re.findall(r"<a\b[^>]*>", INDEX, flags=re.IGNORECASE):
        if 'target="_blank"' in tag:
            assert 'rel="noopener' in tag, f"unprotected _blank link: {tag}"


def test_no_dynamic_code_execution() -> None:
    for pattern in DANGEROUS_EVAL:
        assert pattern not in ALL_FRONTEND, f"dynamic execution: {pattern}"


def test_supabase_cdn_is_version_pinned() -> None:
    match = re.search(r"supabase-js@(\d+\.\d+\.\d+)/dist/umd/supabase\.js", INDEX)
    assert match, "Supabase JS must be loaded from a fully pinned CDN version"
    assert "@2/dist" not in INDEX and "@latest" not in INDEX


# ── 15. storage review ──────────────────────────────────────────────────────


def test_no_manual_token_or_data_persistence() -> None:
    for api in (
        "localStorage.setItem",
        "sessionStorage.setItem",
        "window.localStorage",
        "window.sessionStorage",
        "localStorage[",
        "document.cookie",
        "indexedDB",
    ):
        assert api not in ALL_FRONTEND, f"manual persistence by our code: {api}"
    # localStorage appears ONLY in the docstring documenting the official
    # Supabase client's own session handling.
    for line in AUTH_JS.splitlines():
        if "localStorage" in line:
            assert line.lstrip().startswith("*"), line


# ── 16. backend error text safety ───────────────────────────────────────────


def test_backend_error_text_is_rendered_safely() -> None:
    assert len([ln for ln in INDEX.splitlines() if "apiErrorMessage(" in ln]) >= 4
    assert "throw new Error(await apiErrorMessage(res))" in INDEX
    assert "msg.textContent = 'Upload failed: ' + d;" in INDEX
    # Only allow-listed statuses may surface a backend `detail`, never as HTML.
    assert "var SAFE_DETAIL_STATUS = { 400: 1, 404: 1, 409: 1, 413: 1, 415: 1, 422: 1 };" in AUTH_JS
    assert "STATUS_MESSAGES[response.status]" in AUTH_JS


# ── 17. file upload safety ──────────────────────────────────────────────────


def test_upload_is_size_limited_and_filename_neutralised() -> None:
    assert "var MAX_CSV_BYTES = 2000000" in INDEX
    assert "if(file.size > MAX_CSV_BYTES)" in INDEX
    assert ".replace(/[^A-Za-z0-9._-]/g, '_')" in INDEX
    assert "'X-Filename':safeName" in INDEX
    assert "'X-Filename':(file.name" not in INDEX
    assert "console." not in INDEX  # file contents/names never logged


# ── 18-19. auth state rendering ─────────────────────────────────────────────


def test_app_shell_hidden_until_session_verified() -> None:
    assert 'id="appShell" style="display:none"' in INDEX
    assert 'id="loginOverlay"' in INDEX
    assert "if (shell) shell.style.display = 'flex';" in AUTH_JS  # showApp
    assert "if (shell) shell.style.display = 'none';" in AUTH_JS  # showLogin
    assert "client.auth\n      .getSession()" in AUTH_JS


def test_current_user_comes_from_supabase_session() -> None:
    assert "_currentUser = _session ? _session.user : null;" in AUTH_JS
    assert "if (!_currentUser || !_currentUser.id)" in AUTH_JS
    assert "onAuthStateChange" in AUTH_JS


def test_logout_clears_state_and_returns_to_login() -> None:
    assert "await client.auth.signOut()" in AUTH_JS
    assert "_session = null;" in AUTH_JS and "_currentUser = null;" in AUTH_JS

    assert "PLACEHOLDER" in AUTH_JS  # runtime placeholder detection


def test_no_client_generated_user_identity() -> None:
    assert "Math.random" not in ALL_FRONTEND
    assert not re.search(r"['\"]user[-_]'?\s*\+", ALL_FRONTEND)
    assert "currentUserId" in ALL_FRONTEND
