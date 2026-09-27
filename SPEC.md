# Planwisely Technical Specification

**Version:** 2.0
**Date:** 2026-09-13
**Status:** Active Development — Not Launch Ready

---

## 1. Purpose

Planwisely is a Pakistan-first B2C financial intelligence platform that helps salaried professionals (ages 22–40) answer:

> "How much can I safely spend before payday?"

This document is the **technical implementation source of truth**. It is written for developers, maintainers, AI coding agents, security reviewers, and future contributors.

---

## 2. Product Definition

Planwisely combines **specialized machine learning** with **deterministic financial logic** to produce authoritative financial intelligence.

| Layer | Role |
|---|---|
| **Specialized ML** | Transaction classification, expense forecasting, behavioral feature extraction |
| **Deterministic Financial Logic** | Budget optimization, scenario analysis, decision recommendations, financial profiling |
| **Optional LLM (Future)** | Interface, explanation, orchestration — never the source of financial truth |

The LLM is **not required** for the system to be "AI." Planwisely already uses specialized AI/ML. No external LLM provider is currently integrated.

---

## 3. Core Product Promise

**"Know what you can safely spend before payday."**

Safe-to-Spend is the central product direction. It is **not yet implemented** as a production calculation. It is a design target that requires backend support.

---

## 4. Technical Principles

| Principle | Description |
|---|---|
| **Specialized ML + Deterministic Logic** | Financial truth comes from specialized models and deterministic systems, not from LLMs |
| **Authorization Isolation** | Authenticated JWT `sub` is the authoritative user identity for all protected operations |
| **Fail-Closed** | Missing or invalid credentials deny access by default |
| **Determinism** | No randomness or wall-clock reads in financial calculations; identical inputs produce identical outputs |
| **Honesty** | No fabricated income; missing information is represented as `None`, never guessed |
| **Pakistan-First** | Initial market is Pakistan; architecture designed for eventual multi-currency/locale support |
| **No Automated Financial Actions** | Recommendations only — the system never auto-executes transactions or account changes |


---

## 5. Current Implementation Status

| Component | Status |
|---|---|
| Transaction Classification | Implemented — LightGBM; synthetic evaluation ~89.10% accuracy |
| Feature Pipeline | Implemented — numerical, temporal, behavioral, text features |
| Expense Forecasting | Implemented — global/demo/static artifact; ~6.10% MAPE |
| Budget Optimization | Implemented — constraint-based with hard-protected categories |
| Scenario / Decision Engines | Implemented — deterministic |
| Financial Profile | Implemented — deterministic aggregation |
| CSV Ingestion | Implemented — validation, row caps, file limits |
| Authentication (Step 1) | COMMITTED LOCALLY (`1a5eb8a`) — NOT pushed |
| Protected Routes (Step 2) | COMMITTED LOCALLY (`024465c`) — NOT pushed |
| Ownership / Isolation (Step 3) | COMMITTED LOCALLY (`9b34da0`) — NOT pushed |
| Safe-to-Spend | Planned — not implemented |
| AI Copilot | Future — not implemented |
| Financial Intelligence API | Future — not implemented |
| Production Database | Not implemented — artifacts + cache only |

---

## 6. Repository / Deployment Overview

```
src/
├── data/           Schemas, ingestion, synthetic-data generation
├── evaluation/     Metrics, comparisons, fairness auditing
├── features/       Numerical, temporal, behavioral, text features
├── models/         ML models (classifier, forecaster, recommender)
├── serving/        FastAPI app, middleware, routes, auth, cache
├── services/       Financial logic (profile, scenario, decision)
└── utils/          Constants, logging, privacy

tests/unit/         Unit tests
models/serving/     Demonstration model artifacts (read-only)
frontend/           Browser-based interface
infrastructure/     Docker, deployment configuration
```

### 6.1 Code-Referenced Section Numbering

Module docstrings cite the legacy SPEC numbering (from the original February 2026 specification). Those citations remain canonical for navigating the code:

| Code citation | Meaning | This document |
|---|---|---|
| `SPEC §5.1` | Raw transaction schema | §16 |
| `SPEC §5.2` / `§5.2.2` | Feature specification | §17–18 |
| `SPEC §5.3` | Category taxonomy (6 L1 / 30 L2) | §16 |
| `SPEC §8.2` | Budget optimisation constraints | §20 |
| `SPEC §11.2.1`–`§11.2.3` | Serving endpoints (classify / forecast / budget) | §8 |
| `SPEC §11.3` | Cache strategy and TTLs | §15, §25 |
| `SPEC Appendix B` | MCC → category mappings | `src/utils/constants.py` |


---

## 7. Serving Architecture

The serving layer is a **FastAPI** application:

- **Application entry point:** `src/serving/app.py` — `create_app()` registers routers, middleware, and dependencies
- **Route handlers:** `src/serving/routes/` (classify, forecast, budget, advise, ingest, live, health)
- **Authentication:** `src/serving/auth.py`
- **Middleware:** `src/serving/middleware.py`
- **Caching:** `src/serving/cache.py`

---

## 8. API Surface

### 8.1 Protected Endpoints (require Supabase JWT)

| Method | Path | Description |
|---|---|---|
| POST | `/v1/classify` | Classify a transaction |
| GET | `/v1/forecast/{user_id}` | Expense forecast for an owner |
| GET | `/v1/budget/{user_id}` | Budget recommendations for an owner |
| POST | `/consumer/advise` | Deterministic financial advice |
| POST | `/consumer/transactions/ingest-csv` | CSV ingestion with ownership checks; persists user-owned transactions (when `DATABASE_URL` is configured) |
| POST | `/consumer/classify/live` | Live classification |
| POST | `/consumer/forecast/live` | Live forecast |
| POST | `/consumer/budget/live` | Live budget |
| GET | `/consumer/profile` | Caller's saved financial profile (owner = JWT `sub`) |
| PUT | `/consumer/profile` | Create/update the caller's financial profile |
| GET | `/consumer/transactions` | List the caller's persisted transactions |
| DELETE | `/consumer/transactions/{id}` | Delete one caller-owned transaction |
| GET | `/consumer/data/export` | Export all caller-owned persisted data (JSON) |
| DELETE | `/consumer/data` | Delete all caller-owned persisted data (fail-closed, transactional) |

### 8.2 Public Endpoints (intentionally public)

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/ready` | Model readiness |
| GET | `/docs`, `/redoc`, `/openapi.json` | API documentation |
| GET | `/favicon.ico` | Favicon |

---

## 9. Authentication Architecture

Planwisely uses **Supabase JWT** authentication. Tokens are verified against the Supabase JWKS endpoint using RS256.

| Component | File | Description |
|---|---|---|
| `AuthPrincipal` | `src/serving/auth.py` | Verified JWT principal containing `sub` |
| `require_auth` | `src/serving/auth.py` | FastAPI dependency that verifies the JWT |
| `verify_token` | `src/serving/auth.py` | Token verification logic |
| Auth middleware | `src/serving/middleware.py` | Middleware layer for auth |

### 9.1 Authentication Flow

```
Client
  → Supabase Auth (login/signup)
  → JWT (RS256 signed)
  → FastAPI require_auth dependency
  → verify_token (JWKS verification)
  → AuthPrincipal(sub=...)
  → Route handler
```

### 9.2 Fail-Closed Behavior

- Missing token → 401
- Invalid or expired token → 401
- `SUPABASE_URL` not configured → 401 (no startup crash)
- Empty `API_KEYS` → 401 (no consumer bypass)

---

## 10. Authorization / Ownership Model

### 10.1 Fundamental Invariant

```
authenticated AuthPrincipal.sub == financial resource owner
```

A client-provided `user_id` must **NEVER** establish ownership.

### 10.2 Authoritative Identity

The verified JWT subject (`principal.sub`) is the **only** authoritative user identity for protected financial operations. Ownership is **never** derived from:

- Request body `user_id`
- Email
- Browser/localStorage identity
- Arbitrary caller-provided identifiers
- Cache keys controlled by the caller

---

## 11. Identity Propagation

`principal.sub` is propagated through:

1. **Router-level dependency:** `dependencies=[Depends(require_auth)]`
2. **Handler parameter:** `principal: AuthPrincipal = Depends(require_auth)`
3. **Internal processing:** all financial operations use `principal.sub`
4. **Cache keys:** user-scoped cache keys include `principal.sub`
5. **Response fields:** response `user_id` fields use `principal.sub`

---

## 12. Path `user_id` Policy

For protected routes containing `{user_id}`:

```
requested user_id == authenticated principal.sub
```

| Endpoint | Policy |
|---|---|
| `GET /v1/forecast/{user_id}` | must equal `principal.sub` → 403 on mismatch |
| `GET /v1/budget/{user_id}` | must equal `principal.sub` → 403 on mismatch |

---

## 13. Body `user_id` Policy

| Endpoint | Field | Policy |
|---|---|---|
| `POST /v1/classify` | `transaction.user_id` | must match `principal.sub` |
| `POST /consumer/advise` | `transactions[*].user_id` | each must match `principal.sub` |
| `POST /consumer/advise` | top-level `user_id` (if supplied) | must match `principal.sub` |

Matching subject → accepted. Mismatched subject → 403 Forbidden. Internal processing always uses `principal.sub`.

---

## 14. CSV Ownership Policy

For `POST /consumer/transactions/ingest-csv`:

- Every parsed transaction `user_id` must match `principal.sub`
- Any row claiming a different owner → 403 before any processing
- All existing CSV validation, row caps, file limits, and classification behavior preserved

---

## 15. Cache Isolation

| Resource | Cache Key Pattern |
|---|---|
| Forecasts | `("forecasts", principal.sub, horizon, categories)` |
| Budgets | `("budgets", principal.sub)` |
| Features | `("features", principal.sub, ...)` |
| Explanations | `("explanations", ...)` |

**TTLs:** Features 6h, Forecasts 24h, Budgets 7d, Explanations 30d.

**Isolation guarantee:** the ownership gate (403) runs **before** any cache read, so User A cannot retrieve User B's cache entry by manipulating a body/path `user_id`.

---

## 16. Transaction Semantics

Canonical model (`src/data/models.py` — `Transaction`): `transaction_id`, `user_id`, `timestamp` (UTC), `amount` (**signed** — negative debit, positive credit), `currency` (ISO 4217), `merchant_name`, `merchant_mcc` (ISO 18245), `account_type`, `channel` (POS/ONLINE/ATM/TRANSFER/RECURRING), location, `raw_description`, `is_pending`, optional `category_l1` / `category_l2`.

Category taxonomy: 30 L2 categories across 6 L1 groups (`src/utils/constants.py`).

---

## 17. Classification Architecture

- **Model:** LightGBM gradient-boosted trees; 30-class L2 category prediction
- **Features:** 157 (numerical + temporal + behavioral + text)

**Synthetic evaluation — NOT production customer metrics:**

| Metric | Value |
|---|---|
| Accuracy | 89.10% |
| Macro-F1 | 85.56% |
| Weighted-F1 | 89.11% |
| Top-3 Accuracy | 98.07% |
| ECE | 7.81% |
| Majority Baseline | 4.56% |

Evaluation uses synthetic labels; do not present these as production accuracy.

**Fairness auditing:** equal-opportunity difference and demographic-parity checks across income quintile, account age, and geographic region (`src/evaluation/fairness_audit.py`, `configs/fairness_config.yaml`).


---

## 18. FeatureService

- **Feature parity:** 157/157 verified between training and serving
- **Max absolute difference:** 0.0
- Temporal-leakage-safe transforms (prior-history-only z-scores, rolling windows)
- Implementation basis: `src/features/numerical_features.py`
- **Temporal features:** `hour_of_day`, `day_of_week`, `day_of_month`, `days_since_payday`, `is_weekend`, `is_holiday`, `month_phase` (cyclical encodings)
- **Behavioral features:** `spending_regime`, `impulse_score`, `habit_strength`, `income_cycle_phase`, `lifestyle_drift_30d`
- **Parity evidence:** `tests/unit/test_train_serve_parity.py` routes serving through `FeatureService` to prove train/serve parity


---

## 19. Forecast Architecture

Two distinct paths; every response states which one produced it.

### 19.1 Personalized — `GET /consumer/forecast` (STEP 13)

```
principal.sub ─► transactions.owner_sub (Step 12 persistence)
             ─► zero-filled daily spend series (spend only; credits not netted)
             ─► quality tier: personalized | limited_history | insufficient_history
             ─► user-specific method (weekday profile; recent-mean fallback)
             ─► daily p10/p50/p90 + horizon totals (interval from measured residuals)
```

- **Target:** total spend pressure over an explicit horizon (7–90 days, default 30).
- **Minimum history:** Tier A ≥28 days / ≥20 spend transactions / ≥12 spend days; Tier B ≥14 days / ≥8 transactions; below that → `insufficient_history` + `requirements`, and **no numeric forecast is returned**.
- **Method selection is measured, not assumed.** `pipelines/forecast_benchmark.py` runs expanding-window walk-forward validation (no random split anywhere) over 30 synthetic users (180-day history, 30-day horizon): weekday profile **37.66% WAPE** (median per-user 9.47%), Prophet 41.91% (balanced 5-user sample), recent-mean 53.55%, EWMA 53.64%, non-personalized global mean 56.00%. **Offline/synthetic — not customer or production accuracy.**
- **Fallback policy:** the fallback is the *same user's* recent-mean baseline (`fallback_status="user_recent_baseline"`; `"primary_substituted"` when a requested method is replaced). A global artifact is **never** substituted, and no other user's rows can influence an output — every query is scoped by `owner_sub`.
- **Caching:** `forecasts:<sub>:personal:<horizon>:<engine_version>`, TTL 24 h, purged on ingest / profile change / data deletion. Insufficient-history responses are not cached.
- **Engine version:** `user-v1`.

### 19.2 Legacy reference — `GET /v1/forecast/{user_id}`

Replays the global/static training artifact and now says so explicitly:
`personalization_status="not_personalized"`, `fallback_status="global_reference"`,
`method="global_static_artifact"`, `history_days_used=0`. It is retained as a
benchmark/demo reference. Artifact MAPE ≈ 6.10% belongs to a **different dataset**
(per-category weekly totals over a demo period) and is **not comparable** to the
personalized benchmark in §19.1.

**Historical limitation (still true for this endpoint):** the artifact is
**global/demo/static** and **NOT genuinely personalized per user** — per-request
authorization isolation only, not model personalization. Not customer production accuracy.

---

## 20. Budget Architecture

- **Solver:** constraint-based `BudgetOptimiser` (`scipy.optimize.linprog`, HiGHS) with heuristic fallback
- **Objective:** minimize weighted deviation from baselines subject to income minus savings target

**Hard-protected categories (never reduced):** Rent/Mortgage, Utilities, Home Insurance, Insurance Premiums, Loan Payments, Taxes (`HARD_PROTECTED_CATEGORIES`). `Uncategorized` is excluded from optimization.

**Budget acceptance:** ~78.86% on synthetic evaluation. Do not market any previously circulated, unverified acceptance figure as verified.

**Explanations:** each recommendation includes SHAP feature attributions, an anchor rule, and a counterfactual via the ExplanationEngine (`src/models/recommender/explanations.py`).


---

## 21. Scenario Engine

`src/services/scenario_engine.py` — deterministic. `FinancialProfile + ScenarioParams → ScenarioResult` with statuses `feasible` / `partial` / `infeasible`. Delegates required cuts to `BudgetOptimiser`; screens cuts through the behavioral feasibility checker when history is supplied. No randomness; no wall-clock reads; sorted dict iteration where order affects arithmetic.

---

## 22. Decision Engine

`src/services/decision_engine.py` — deterministic orchestration:

```
Transactions → FinancialProfile → ScenarioResult → DecisionResult
```

Performs **no financial arithmetic of its own** — every number in `DecisionResult` is read from `FinancialProfile` / `ScenarioResult`. Safety invariants (transfer/refund semantics, confidence gating, Uncategorized protection, income never invented) are inherited from the underlying services.

---

## 23. Financial Profile

`src/services/financial_profile.py` — deterministic aggregation of classified transactions. Honesty rules: income ONLY from `CategoryL2.INCOME`; refunds/reversals/transfers/savings movements never inflate income or expenses (reported informationally); income is never derived from spending; missing information is `None`. Monthly normalisation divides by distinct calendar months covered.

---

## 24. CSV Ingestion

`src/serving/routes/ingest.py` — validates file size, row count, schema, and encoding; row caps enforced. Step 3 adds per-row ownership: each parsed `txn.user_id` must equal `principal.sub` → 403 before any classification or storage.

---

## 25. Caching

`src/serving/cache.py` — `CacheClient`: Redis primary, in-memory dict fallback. Identity-namespaced per §15. TTLs per SPEC §11.3 (features 6h, forecasts 24h, budgets 7d, explanations 30d).

---

## 26. Error Handling

HTTP 400/401/403/404/500. **Known risk:** error detail leakage via `detail=str(exc)` in some endpoints — not yet sanitized (planned hardening work).

---

## 27. Rate Limiting

**Not implemented.** Per-identity rate limiting and abuse protections are planned hardening work.

---

## 28. Admin / Security Surface

`/admin/generate-key` is publicly accessible — **KNOWN SECURITY ISSUE**. Admin lockdown is Step 4 scope; intentionally left unchanged in Step 3.

---

## 29. Frontend Integration Requirements

- Obtain JWT from Supabase Auth; send `Authorization: Bearer <token>`
- Client-side `user_id` is **untrusted** — the backend derives identity from the JWT
- The frontend is **never** an authorization boundary
- Render API data safely; do not render raw backend errors
- Clear sensitive client caches on logout/account change

---

## 30. Supabase Integration

- **Auth:** Supabase Auth for user management and JWT issuance
- **Verification:** RS256 via Supabase JWKS
- **Configuration:** `SUPABASE_URL` environment variable (missing → fail-closed 401, no startup crash)

---

## 31. Future Safe-to-Spend Architecture

Safe-to-Spend answers: "How much can I safely spend before payday?" Conceptual inputs: current financial position, upcoming obligations, budget constraints, forecast, savings goals, transaction behavior.

**Status: PLANNED — not implemented.** The frontend must not calculate it independently; backend logic is required.

---

## 32. Future AI Copilot

- Calls **authorized Planwisely tools**; never invents financial answers
- Provider-agnostic (OpenAI, Anthropic, Gemini, or future providers)
- Fine-tuning is NOT an MVP requirement; only when proprietary data and benchmarks justify it
- Training a general-purpose LLM from scratch is NOT part of the roadmap
- **Status: FUTURE — not implemented.** No LLM is currently integrated.

---

## 33. Provider-Agnostic AI Architecture

No external LLM provider is currently integrated. Future integration must remain provider-agnostic, with financial calculations remaining in specialized models / deterministic systems and the LLM confined to interface/explanation/orchestration.

---

## 34. Future Financial Intelligence API

Planwisely may become a specialized **Financial Intelligence Layer/API** that other applications and AI agents call. **Status: FUTURE — not implemented.**

---

## 35. Agent Interoperability

External AI agents may eventually call Planwisely's Financial Intelligence API using scoped authentication/authorization. **Status: FUTURE — not implemented.**

---

## 36. Privacy / Data Handling

Privacy policy, terms of service, support/contact, and account/data deletion flows are **not yet implemented** — documented as launch-blocking gaps.

---

## 37. Testing

| Suite | Result |
|---|---|
| `tests/unit/test_ownership.py` | 22 passed |
| `tests/unit/test_auth.py` | 20 passed |
| `tests/unit/test_auth_routes.py` | 21 passed |
| `tests/unit/test_classify_endpoint.py` | 7 passed |
| `tests/unit/test_csv_ingest_endpoint.py` | 10 passed |
| `tests/unit/test_advise_endpoint.py` | 12 passed |
| **Full unit suite** | **224 passed, 1 skipped** (2 warnings, ~413 s) |
| `compileall src/serving tests/unit` | exit 0 |

Results reflect the verified Step 3 implementation (commit `9b34da0`).

---

## 38. Current Auth Implementation Status

| Step | State | Commit |
|---|---|---|
| Step 1 — Supabase JWT foundation | COMMITTED LOCALLY, NOT pushed | `1a5eb8a13d6235d1f8e11d7dfd73767e67894702` |
| Step 2 — protected financial routes | COMMITTED LOCALLY, NOT pushed | `024465c` |
| Step 3 — ownership / isolation | COMMITTED LOCALLY, NOT pushed | `9b34da0` |

All three auth steps are committed locally; none are pushed.

---

## 39. Open Launch Blockers

| Issue | Status |
|---|---|
| `/admin/generate-key` public | Unfixed — Step 4 |
| Plaintext API-key logging | Known concern |
| Error leakage via `detail=str(exc)` | Known risk |
| Rate limiting / abuse controls | Not implemented |
| Privacy policy / terms / support | Not implemented |
| Account & data deletion | Not implemented |
| Security headers / Trusted Host | Not implemented |
| Frontend security (XSS, storage) | Known risks |
| Cache/Redis operational hardening | Known risks |
| Observability / monitoring | Not implemented |
| Model artifact loading (pickle can execute code on load) | Load only from trusted sources |


---

## 40. Protected Areas / Invariants

Do not casually modify: FeatureService, BudgetOptimiser, `HARD_PROTECTED_CATEGORIES`, DecisionEngine, ScenarioEngine, financial_profile logic, classification semantics, transaction semantics, auth code, ownership enforcement, cache implementation.

---

## 41. Step 3 Acceptance State

Complete. **Committed locally — not pushed.** Acceptance criteria (path/body/CSV ownership, exact budget lookup, cache namespacing, live identity propagation, ownership tests) are verified by the test evidence in §37.

---

## 42. Next Security Steps

1. **Step 4:** admin endpoint lockdown
2. Error sanitization
3. Rate limiting / abuse controls
4. Cache/Redis hardening
5. Security headers / deployment hardening
6. Observability & security logging

---

## 43. Launch Gate

Planwisely is **NOT LAUNCH READY.** Current status: **ACTIVE DEVELOPMENT.**
