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
- Supabase JWKS retrieval and RS256 signature/claims verification
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
