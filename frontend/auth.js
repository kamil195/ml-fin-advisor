/**
 * Planwisely frontend authentication (Step 10).
 *
 * Supabase Auth is the browser authentication provider; the FastAPI backend
 * remains the source of truth for identity (verified JWT `sub`). This module:
 *
 *   - creates the Supabase client from public config only (config.js)
 *   - restores the session on load and follows auth state changes
 *   - implements email/password sign-in, sign-up and sign-out
 *   - exposes apiFetch(), the single authenticated request helper that
 *     attaches `Authorization: Bearer <access_token>` to protected calls
 *
 * Session persistence: handled entirely by the official Supabase JS client
 * (localStorage under `sb-<project-ref>-auth-token`, with automatic token
 * refresh). This module never persists, logs, or displays tokens itself and
 * never decodes JWT claims for authorization decisions.
 *
 * Privacy: access tokens are never logged and never placed in URLs.
 */
(function () {
  'use strict';

  var cfg = (typeof window !== 'undefined' && window.PLANWISELY_CONFIG) || {};
  var SUPABASE_URL = cfg.SUPABASE_URL || '';
  var SUPABASE_ANON_KEY = cfg.SUPABASE_ANON_KEY || '';
  var PLACEHOLDER = /^YOUR_SUPABASE_/;

  var _client = null;
  var _session = null;
  var _currentUser = null;

  function AuthRequiredError(message) {
    var e = new Error(message || 'Authentication required.');
    e.name = 'AuthRequiredError';
    return e;
  }
  window.AuthRequiredError = AuthRequiredError;

  function authConfigured() {
    return (
      Boolean(SUPABASE_URL) &&
      Boolean(SUPABASE_ANON_KEY) &&
      !PLACEHOLDER.test(SUPABASE_URL) &&
      !PLACEHOLDER.test(SUPABASE_ANON_KEY)
    );
  }
  window.authConfigured = authConfigured;

  function ensureClient() {
    if (!authConfigured()) {
      throw new Error(
        'Authentication is not configured. SUPABASE_URL and SUPABASE_ANON_KEY must be set in frontend/config.js at deploy time.'
      );
    }
    if (!_client) {
      _client = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: {
          persistSession: true, // Supabase JS persists the session itself
          autoRefreshToken: true,
          detectSessionInUrl: true,
        },
      });
    }
    return _client;
  }

  function el(id) {
    return document.getElementById(id);
  }

  function setAuthMessage(message, isError) {
    var m = el('authMessage');
    if (!m) return;
    m.textContent = message || '';
    m.classList.toggle('error', Boolean(isError));
    m.style.display = message ? 'block' : 'none';
  }

  function setAuthLoading(loading) {
    var b = el('authSubmit');
    if (b) {
      b.disabled = loading;
      b.textContent = loading ? 'Signing in…' : 'Sign in';
    }
    var s = el('authSignupBtn');
    if (s) s.disabled = loading;
  }

  function emailFor(session) {
    return (session && session.user && session.user.email) || '';
  }

  function showApp(session) {
    _session = session || null;
    _currentUser = _session ? _session.user : null;
    var name = (emailFor(_session).split('@')[0] || 'user');
    var nameEl = el('userName');
    var emailEl = el('userEmail');
    var avatarEl = el('userAvatar');
    if (nameEl) nameEl.textContent = name;
    if (emailEl) emailEl.textContent = emailFor(_session);
    if (avatarEl) avatarEl.textContent = (name[0] || 'U').toUpperCase();
    var overlay = el('loginOverlay');
    var shell = el('appShell');
    if (overlay) overlay.style.display = 'none';
    if (shell) shell.style.display = 'flex';
  }

  function showLogin() {
    _session = null;
    _currentUser = null;
    var overlay = el('loginOverlay');
    var shell = el('appShell');
    if (overlay) overlay.style.display = 'flex';
    if (shell) shell.style.display = 'none';
  }

  function currentUserId() {
    if (!_currentUser || !_currentUser.id) {
      throw AuthRequiredError('Your session has expired. Please sign in again.');
    }
    return _currentUser.id;
  }
  window.currentUserId = currentUserId;

  function friendlyAuthError(message) {
    var m = String(message || '');
    if (/invalid login credentials/i.test(m)) return 'Incorrect email or password.';
    if (/email not confirmed/i.test(m)) {
      return 'Please confirm your email address, then sign in.';
    }
    if (/rate limit/i.test(m)) {
      return 'Too many attempts. Please wait a moment and try again.';
    }
    if (/password should be at least|at least \d+ characters/i.test(m)) {
      return 'Password is too short. Please choose a stronger password.';
    }
    if (/user already registered/i.test(m)) {
      return 'An account with this email already exists. Please sign in.';
    }
    return m ? 'Authentication failed: ' + m : 'Authentication failed. Please try again.';
  }

  async function doSignIn() {
    var emailEl = el('authEmail');
    var passwordEl = el('authPassword');
    var email = emailEl ? emailEl.value.trim() : '';
    var password = passwordEl ? passwordEl.value : '';
    if (!email || !password) {
      setAuthMessage('Enter your email and password.', true);
      return;
    }
    setAuthLoading(true);
    setAuthMessage('');
    try {
      var client = ensureClient();
      var res = await client.auth.signInWithPassword({ email: email, password: password });
      if (res.error) {
        setAuthMessage(friendlyAuthError(res.error.message), true);
        return;
      }
      applySession(res.data && res.data.session);
    } catch (e) {
      setAuthMessage('Could not sign in. Please try again.', true);
    } finally {
      setAuthLoading(false);
    }
  }

  async function doSignUp() {
    var emailEl = el('authEmail');
    var passwordEl = el('authPassword');
    var email = emailEl ? emailEl.value.trim() : '';
    var password = passwordEl ? passwordEl.value : '';
    if (!email || !password) {
      setAuthMessage('Enter your email and a password to create an account.', true);
      return;
    }
    setAuthLoading(true);
    setAuthMessage('');
    try {
      var client = ensureClient();
      var res = await client.auth.signUp({ email: email, password: password });
      if (res.error) {
        setAuthMessage(friendlyAuthError(res.error.message), true);
        return;
      }
      if (!res.data || !res.data.session) {
        // Email confirmation required — the user is NOT signed in yet.
        setAuthMessage(
          'Account created. Check your email to confirm your address, then sign in.'
        );
        return;
      }
      applySession(res.data.session);
    } catch (e) {
      setAuthMessage('Could not create the account. Please try again.', true);
    } finally {
      setAuthLoading(false);
    }
  }

  async function doLogout() {
    _session = null;
    _currentUser = null;
    try {
      var client = ensureClient();
      await client.auth.signOut(); // official client clears its own session
    } catch (e) {
      /* sign-out is best-effort; local state is cleared regardless */
    }
    showLogin();
    setAuthMessage('Signed out.');
  }

  function applySession(session) {
    if (session && session.user) {
      showApp(session);
    } else {
      showLogin();
    }
  }

  function initAuth() {
    if (!el('loginOverlay')) return; // page without the auth UI
    if (!authConfigured()) {
      showLogin();
      setAuthMessage(
        'Authentication is not configured for this deployment. SUPABASE_URL and SUPABASE_ANON_KEY must be set in frontend/config.js at deploy time.',
        true
      );
      var b = el('authSubmit');
      var s = el('authSignupBtn');
      if (b) b.disabled = true;
      if (s) s.disabled = true;
      return;
    }
    var client = ensureClient();
    client.auth
      .getSession()
      .then(function (res) {
        applySession(res && res.data && res.data.session);
      })
      .catch(function () {
        showLogin();
      });
    client.auth.onAuthStateChange(function (_event, session) {
      applySession(session);
    });
  }
  window.initAuth = initAuth;
  window.doSignIn = doSignIn;
  window.doSignUp = doSignUp;
  window.doLogout = doLogout;

  /* ── Authenticated request helper ──────────────────────────────────── */

  function apiBase() {
    return (window.PLANWISELY_CONFIG && window.PLANWISELY_CONFIG.API_BASE) || '';
  }

  async function apiFetch(path, options) {
    options = options || {};
    var url = /^https?:\/\//i.test(path)
      ? path
      : apiBase() + (path.charAt(0) === '/' ? path : '/' + path);
    var client = ensureClient();
    var res = await client.auth.getSession();
    var session = res && res.data && res.data.session;
    if (!session || !session.access_token) {
      throw AuthRequiredError('Your session has expired. Please sign in again.');
    }
    var headers = Object.assign({}, options.headers || {}, {
      Authorization: 'Bearer ' + session.access_token,
    });
    var response = await fetch(url, Object.assign({}, options, { headers: headers }));
    if (response.status === 401) {
      // One transparent session refresh + single retry — never a loop.
      var refreshed = await client.auth.refreshSession();
      var fresh = refreshed && refreshed.data && refreshed.data.session;
      if (fresh && fresh.access_token) {
        headers = Object.assign({}, options.headers || {}, {
          Authorization: 'Bearer ' + fresh.access_token,
        });
        response = await fetch(url, Object.assign({}, options, { headers: headers }));
      }
    }
    return response;
  }
  window.apiFetch = apiFetch;

  var SAFE_DETAIL_STATUS = { 400: 1, 404: 1, 409: 1, 413: 1, 415: 1, 422: 1 };
  var STATUS_MESSAGES = {
    401: 'Your session has expired. Please sign in again.',
    403: 'You do not have access to this resource.',
    429: 'Too many requests. Please wait a moment and try again.',
    500: 'Something went wrong on our side. Please try again later.',
    503: 'The service is temporarily unavailable. Please try again.',
  };

  async function apiErrorMessage(response) {
    if (response.status === 429) {
      var retryAfter = response.headers.get('Retry-After');
      if (retryAfter && Number(retryAfter) > 0) {
        return 'Too many requests. Please retry in ' + retryAfter + ' seconds.';
      }
    }
    var fallback =
      STATUS_MESSAGES[response.status] ||
      (response.status >= 500
        ? 'Something went wrong on our side. Please try again later.'
        : 'Please check the submitted data.');
    if (SAFE_DETAIL_STATUS[response.status]) {
      try {
        var payload = await response.clone().json();
        var detail = payload && payload.detail;
        if (typeof detail === 'string') return detail;
        if (detail && typeof detail.message === 'string') return detail.message;
        if (detail && typeof detail.detail === 'string') return detail.detail;
      } catch (e) {
        /* fall through to the generic message */
      }
    }
    return fallback;
  }
  window.apiErrorMessage = apiErrorMessage;
})();
