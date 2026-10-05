# Changelog

Verified repository history for Planwisely. Only changes supported by repository history or the currently verified working tree are recorded. Format inspired by Keep a Changelog.

---

## [Unreleased]

### Known Issues (carried forward)

- `/admin/generate-key` publicly accessible — admin lockdown planned (Step 4)
- Plaintext API-key logging concerns
- Error detail leakage via `detail=str(exc)` in some endpoints
- No rate limiting / per-identity abuse protections
- No security headers / Trusted Host hardening
- No privacy policy, terms, support contact, or data-deletion flow
- No production observability / monitoring
- Global/static forecast artifact

---

## 2026-09-30 — AUTH: ES256 + RS256 JWKS verification (Working tree — not yet committed)

**Supabase JWT algorithm compatibility** — `src/serving/auth.py`.

- **Problem:** authenticated requests returned 401 because verification only allowlisted RS256 while the Supabase project's current signing key is ECC P-256 — current access tokens are ES256.
- **Change:** explicit asymmetric allowlist `("ES256", "RS256")` in `_ALGORITHMS`. ES256 verifies the current key; RS256 remains for legacy/rotated RSA keys. The header `alg` is never trusted dynamically; `none`, `HS256` (the legacy Supabase shared-secret scheme) and every other algorithm are rejected fail-closed. No shared-secret verification path was added, and no JWT shared-secret env var is read anywhere in this backend.
- **Unchanged:** JWKS endpoint and issuer derived from `SUPABASE_URL`, audience from `SUPABASE_JWT_AUDIENCE`, `exp`/`sub` requirements, Bearer parsing, identity exclusively from the verified `sub`, generic `401 {"detail":"Not authenticated"}`, and no logging of tokens, headers, keys or claims.
- **Tests:** `tests/unit/test_auth.py` 20 → **28** (added: ES256 valid via JWKS; ES256 wrong-audience; ES256 expired; HS256-with-known-`kid` rejected; PS256 rejected; `alg: none` rejected; ES256 end-to-end through `require_auth`; HS256 → generic 401 through `require_auth`). Targeted run: `test_auth.py` + `test_auth_routes.py` **48 passed**.
- **Docs:** `src/serving/auth.py` module docstring, SPEC §9/§9.1, ARCHITECTURE §6 flow, README Authentication section — all now state ES256 + RS256 verified via JWKS, with HS256/shared-secret tokens intentionally not accepted on this path.
- **Verification (2026-09-30):** targeted auth (`test_auth.py` + `test_auth_routes.py`) **48 passed**; claims/branding/forecast/frontend-security/security-headers batch **111 passed**; Step 14 batch (`test_safe_to_spend.py` + `test_persistence_profiles.py`) **80 passed**; full unit suite **503 passed, 1 skipped, 0 failed** (1356.77 s); `compileall src pipelines run_pipeline.py tests` exit 0; CI lint gate `flake8 --select=F821` exit 0; `git diff --check` exit 0; serving-app import sanity OK.

---

## 2026-09-30 — PRODUCT STEP 14 (Working tree — not yet committed)

**Safe-to-Spend engine and endpoints** — `src/services/safe_to_spend.py`, `src/serving/routes/safe_to_spend.py`.

- **Formula:** `safe_to_spend = liquid_buffer − protected obligations before payday − expected spending before payday − safety buffer − scenario adjustment`, computed by a pure, deterministic engine (`safe-to-spend-v1`) from the caller's own persisted data only — no wall-clock reads, no randomness, no global artifact.
- **New endpoints:** `GET /consumer/safe-to-spend` and `POST /consumer/safe-to-spend/scenario` (JWT-protected; identity solely from `principal.sub`; no request field selects whose data is used; the scenario affects that one response and is never persisted).
- **Honest statuses:** `missing_profile_data` / `missing_balance` / `missing_payday` for missing inputs; `insufficient_history` returns requirements and no amount; `limited_history` is labelled low-confidence; negative results are returned as-is (`is_negative`), never floored; an unconfigured buffer is `0.00` with `buffer_configured: false`.
- **Payday:** configured `next_payday` (1–45 days) → cadence measured from the caller's own income deposits → `missing_payday`. Protected obligations are the six `HARD_PROTECTED_CATEGORIES`, inferred from observed cadence and counted only when due after today and on/before payday (already-paid items are not counted twice; `Uncategorized` is never protected).
- **Profile inputs:** optional `next_payday` and `safety_buffer` added to the profile write/read shape (`migrations/002_safe_to_spend.sql`); both are optional and never invented.
- **Caching:** `safe_to_spend:<sub>:<payday>:<scenario_hash>:<engine>`, 1 h TTL; amount-less statuses are not cached; `as_of` must be today to reuse an entry; ingest / profile mutation / data deletion purge the caller's prefix via `purge_user`.
- **Tests:** `tests/unit/test_safe_to_spend.py` (71 tests: engine arithmetic and honesty, payday resolution, obligation inference, forecast horizon, endpoints, owner isolation, cache scoping/freshness/invalidation, read-only scenario, error mapping). `tests/unit/test_persistence_profiles.py` updated for the two new profile fields.
- **Docs:** SPEC §3, §8.1, §15, §25 corrected and §31 rewritten (the route previously cited a non-existent §21 — now §31); ARCHITECTURE §16c added; ROADMAP Phase 8 updated; README capability/endpoint/limitations tables updated.
- **Verification (2026-09-30):** full unit suite **495 passed, 1 skipped** (~1272 s); focused batch (Safe-to-Spend + profile persistence + claims/branding/forecast doc scans) **149 passed**; `compileall src tests` exit 0.

---

## 2026-09-13 — AUTH STEP 3 (Committed locally; NOT pushed)

**Commit:** `9b34da0`
**Message:** `feat(auth): enforce user ownership and cross-user isolation`

User ownership and cross-user isolation for all protected financial endpoints. The verified Supabase JWT subject (`principal.sub`) is now the authoritative owner identity.

- **Path ownership:** `GET /v1/forecast/{user_id}` and `GET /v1/budget/{user_id}` enforce `user_id == principal.sub`; mismatch → 403
- **Body ownership:** `/v1/classify` transaction `user_id`; `/consumer/advise` per-transaction `user_id`s and top-level `user_id` must match `principal.sub` → 403 on mismatch
- **CSV ownership:** every parsed row in `POST /consumer/transactions/ingest-csv` must carry `user_id == principal.sub` → 403 before processing
- **Internal processing** uses `principal.sub`; response `user_id` fields echo `principal.sub`
- **Budget lookup** is exact per-user; prefix matching removed; first-user fallback removed
- **Cache isolation:** keys namespaced by authenticated identity — `("forecasts", sub, horizon, categories)`, `("budgets", sub)`
- **Live identity propagation:** `/consumer/classify/live` and `/consumer/forecast/live` receive the authenticated principal; `/consumer/budget/live` remains stateless
- **New tests:** `tests/unit/test_ownership.py` (22 tests) — path/body/CSV ownership, cross-user matrix, cache isolation, public-route regression

Test evidence (verified): `test_ownership.py` 22 passed; `test_auth.py` 20; `test_auth_routes.py` 21; `test_classify_endpoint.py` 7; `test_csv_ingest_endpoint.py` 10; `test_advise_endpoint.py` 12. Full unit suite: **224 passed, 1 skipped** (~413 s). `compileall` exit 0.

Known remaining limitations (unchanged by Step 3):

- Forecast artifact remains **global/demo/static** — authorization isolation only, NOT model personalization
- Cache TTL behavior unchanged (forecasts 24h, budgets 7d)
- `/admin/generate-key` remains public (Step 4 scope)

---

## 2026-09-13 — AUTH STEP 2 (Committed locally; NOT pushed)

**Commit:** `024465c`
**Message:** `feat(auth): protect financial routes with require_auth (fail-closed)`

- Eight financial endpoints protected with Supabase JWT `require_auth` (router-level and handler-level):
  `POST /v1/classify`, `GET /v1/forecast/{user_id}`, `GET /v1/budget/{user_id}`, `POST /consumer/advise`, `POST /consumer/transactions/ingest-csv`, `POST /consumer/classify/live`, `POST /consumer/forecast/live`, `POST /consumer/budget/live`
- Global `verify_api_key` middleware removed from `create_app()`; empty `API_KEYS` remains fail-closed (no consumer bypass)
- `/consumer/` bypass path deleted; missing `SUPABASE_URL` → 401 fail-closed without startup crash
- Intentionally public routes preserved: `/health`, `/ready`, `/docs`, `/openapi.json`, `/redoc`, `/favicon.ico`, `/admin/generate-key` (known issue)
- Tests at commit time: `test_auth_routes.py` 21 passed; full unit suite green (203 passed, 1 skipped pre-Step-3)
- 9 files changed, 565 insertions(+), 46 deletions(-)

---

## AUTH STEP 1 (Committed locally; NOT pushed)

**Commit:** `1a5eb8a13d6235d1f8e11d7dfd73767e67894702`
**Message:** `feat(auth): add Supabase JWT verification foundation`

- `AuthPrincipal`, `require_auth`, `verify_token`
- Supabase JWKS retrieval and RS256 signature/claims verification (as implemented at that commit; ES256 support added later — see the 2026-09-30 AUTH ES256/RS256 entry)
- Fail-closed semantics: 401 on missing/invalid token or missing configuration
- Unit tests for token verification (`tests/unit/test_auth.py`, 20 tests)

---

## Earlier Verified Work (repository history; summarized)

Exact hashes/dates are not re-verified here; `git log` is the authoritative history. Nothing below is claimed as pushed.

- **FeatureService:** 157/157 train/serving feature parity verified; max absolute difference 0.0; temporal-leakage-safe, prior-history-only transforms
- **Classifier:** LightGBM integration with confidence thresholding; temporal-split evaluation on synthetic labels — accuracy ~89.10%, macro-F1 ~85.56%, weighted-F1 ~89.11%, top-3 ~98.07%, ECE ~7.81%, majority baseline 4.56% — **synthetic evaluation, not production metrics**
- **Forecast:** Prophet-based artifact; evaluation MAPE ~6.10% — synthetic/demo, not production accuracy; artifact is global/static
- **Budget:** constraint-based `BudgetOptimiser` with `HARD_PROTECTED_CATEGORIES` (Rent/Mortgage, Utilities, Home Insurance, Insurance Premiums, Loan Payments, Taxes); Uncategorized excluded from cuts; acceptance ~78.86% on synthetic evaluation
- **Financial semantics:** income only from INCOME category; transfers/refunds/savings movements informational; deterministic Scenario and Decision engines
- **Ingestion:** CSV validation with row caps and file-size limits
- **Live budget hard protection:** `5a044d7` on origin/main — `fix: enforce hard protection in live budget`
- **Claim corrections:** earlier unsupported figures (99.96% classifier accuracy, 92% forecast accuracy, 80.9% budget acceptance) superseded by the verified synthetic-context figures above; corrected metrics remain synthetic, never production user metrics
