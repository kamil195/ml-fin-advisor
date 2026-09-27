# Planwisely Architecture

**Version:** 1.1
**Date:** 2026-09-13

---

## 1. Architectural Goals

- **Trustworthy financial intelligence** through specialized ML + deterministic logic
- **Strong authorization isolation** via JWT subject
- **Fail-closed** security posture
- **Deterministic** financial calculations
- **Pakistan-first** with future multi-currency support

---

## 2. Product Intelligence Pipeline

```text
Financial Data
  → Transaction Intelligence (classification)
  → Financial Profile (aggregation)
  → Behavior / Forecast / Budget
  → Scenario Engine (what-if analysis)
  → Decision Layer (recommendations)
  → Optional AI Copilot (future)
  → Recommended Actions
  → Outcome Tracking (future)
```

---

## 3. Current System Architecture

```text
+--------------------------------------------------------------------------------------------------+
|                                      PLANWISELY SYSTEM                                          |
+--------------------------------------------------------------------------------------------------+

  User Browser / Mobile App
       |
       | HTTPS + JWT (Authorization: Bearer <token>)
       v
  +----+------------------------+
  |       FastAPI Application       |
  |  +---------------------------+ |
  |  |   require_auth dependency  | |
  |  |   (JWT verification)      | |
  |  +---------------------------+ |
  |              |                  |
  |              v                  |
  |  +---------------------------+ |
  |  |   AuthPrincipal.sub       | |
  |  |   (authoritative identity)| |
  |  +---------------------------+ |
  |              |                  |
  |              v                  |
  |  +---------------------------+ |
  |  |   Ownership Enforcement   | |
  |  |   (path/body/CSV checks)  | |
  |  +---------------------------+ |
  +--------------+------------------+
                 |
                 v
  +-------------------------------+
  |      Financial Services       |
  |  +-------------------------+  |
  |  | Classification          |  |
  |  | Forecast                |  |
  |  | BudgetOptimiser         |  |
  |  | Financial Profile       |  |
  |  | Scenario Engine         |  |
  |  | Decision Engine         |  |
  |  +-------------------------+  |
  +-------------------------------+
                 |
                 v
  +-------------------------------+
  |      Cache / Storage          |
  |  Redis (primary)              |
  |  In-memory (fallback)         |
  |  Model artifacts (read-only)  |
  +-------------------------------+
```

---

## 4. Deployment / Runtime Model

- **Runtime:** Python FastAPI application
- **Server:** Uvicorn/ASGI
- **Containerization:** Docker (infrastructure/)
- **Primary market:** Pakistan
- **No production database** — uses artifacts and in-memory storage

---

## 5. Trust Boundaries

```text
UNTRUSTED                          TRUSTED
+------------------------+         +------------------------+
| User Browser           |         | FastAPI Application    |
| Frontend JavaScript    |         | require_auth           |
| Client-provided user_id| ------> | AuthPrincipal.sub      |
| localStorage           |         | Financial Services     |
| CSV file contents      |         | Cache (namespaced)     |
+------------------------+         +------------------------+
```

Every untrusted input crosses the boundary only through server-side `require_auth` verification plus ownership enforcement. **CRITICAL:** the frontend is **NOT** an authorization boundary. Client-provided `user_id` is **untrusted**. All ownership decisions are made server-side using `AuthPrincipal.sub`.

---

## 6. Authentication Flow

```text
1. User logs in via Supabase Auth
2. Supabase issues JWT (RS256 signed)
3. Client includes JWT in Authorization header
4. FastAPI require_auth dependency intercepts request
5. verify_token validates JWT against Supabase JWKS
6. AuthPrincipal(sub=...) extracted from verified claims
7. AuthPrincipal injected into route handler
```

---

## 7. Authorization / User Isolation

```text
principal.sub = authoritative user identity

For every protected operation:
  - Path user_id must equal principal.sub
  - Body user_id must equal principal.sub
  - CSV row user_id must equal principal.sub
  - Cache keys include principal.sub
  - Budget lookup is exact (no prefix/fallback)
```

---

## 8. Request Identity Lifecycle

```text
Request → require_auth → verify_token → AuthPrincipal
        → Ownership Check → Handler → Response
```

---

## 9. API Routing

| Router | Prefix | Auth |
|---|---|---|
| classify | /v1 | Required |
| forecast | /v1 | Required |
| budget | /v1 | Required |
| advise | /consumer | Required |
| ingest | /consumer | Required |
| live | /consumer | Required (classify/forecast) |
| health | / | Public |
| admin | /admin | **Public (known issue)** |

---

## 10. Serving Modules

| Module | File | Responsibility |
|---|---|---|
| app.py | src/serving/app.py | FastAPI app creation, router registration |
| auth.py | src/serving/auth.py | JWT verification, AuthPrincipal, require_auth |
| middleware.py | src/serving/middleware.py | Auth middleware |
| cache.py | src/serving/cache.py | Redis/in-memory cache |
| classify.py | src/serving/routes/classify.py | Transaction classification |
| forecast.py | src/serving/routes/forecast.py | Expense forecasting |
| budget.py | src/serving/routes/budget.py | Budget optimization |
| advise.py | src/serving/routes/advise.py | Financial advice |
| ingest.py | src/serving/routes/ingest.py | CSV ingestion |
| personal_forecast.py | src/serving/routes/personal_forecast.py | User-specific forecasting (STEP 13) |
| live.py | src/serving/routes/live.py | Live endpoints |

---

## 11. Transaction Ingestion

```text
CSV Upload → File Validation → Row Parsing → Ownership Check (each row)
          → Classification → Storage
```

Validation: file size, row count, schema, encoding.
Ownership: each row user_id must match principal.sub.

---

## 12. Classification Pipeline

```text
Transaction → Feature Extraction (157 features) → LightGBM → Category Prediction
```

Features: numerical, temporal, behavioral, text.

---

## 13. FeatureService

- 157/157 feature parity verified
- Temporal-leakage-safe transforms
- User-scoped rolling windows

---

## 14. Forecasting

```text
Historical Data → Prophet Model → Monthly Forecast by Category
```

**LIMITATION:** Current artifact is global/demo/static. NOT genuinely personalized.

---

## 15. Budget Optimisation

```text
Financial Profile → BudgetOptimiser (scipy linprog) → Category Allocations
```

Hard-protected categories: Rent/Mortgage, Utilities, Home Insurance, Insurance Premiums, Loan Payments, Taxes.

---

## 16. Financial Profile

```text
Classified Transactions → build_financial_profile → FinancialProfile
```

---

## 16a. Persistence (STEP 12 — user-owned, Supabase Postgres)

```text
JWT sub (owner) ──► src/serving/persistence.py (psycopg 3, server-side DSN)
                      ├─ user_profiles   (profile inputs, PK owner_sub)
                      ├─ transactions    (classified CSV rows, owner-scoped)
                      └─ ingest_batches  (upload audit trail)
```

* Identity/ownership: every row carries `owner_sub` = verified JWT subject;
  every query is explicitly scoped by it (no unscoped fallback, no RLS
  reliance — the backend's privileged role bypasses RLS).
* Schema: `migrations/001_user_persistence.sql` (plain versioned SQL, no ORM;
  SQLAlchemy is only a transitive MLflow dependency).
* Lifecycle: `GET /consumer/data/export` and `DELETE /consumer/data` are real
  and transactional (fail-closed); mutations purge the caller's user-scoped
  cache namespaces via `CacheClient.purge_user`.
* NOT persisted: tokens/emails, derived forecast/budget/advise outputs,
  habit strengths / compliance history. Deployment without `DATABASE_URL`
  disables only the user-data routes (generic 503); everything else runs.
* Deliberately unchanged in this step: classifier, forecast math, budget
  math, ScenarioEngine/DecisionEngine; Safe-to-Spend remains unimplemented.
  (Forecasting became user-specific in STEP 13 — see §16b.)

Honesty rules: Income ONLY from CategoryL2.INCOME. No fabricated income.

---

## 16b. Personalized forecasting (STEP 13 — user-specific, no global substitution)

```text
principal.sub ─► fetch_spend_history(owner_sub, since)   (120-day lookback)
             ─► zero-filled daily spend series (spend rows only)
             ─► quality tier → personalized | limited_history | insufficient_history
             ─► weekday profile (primary) · recent mean (user-specific fallback)
             ─► daily p10/p50/p90 + horizon totals, cached per user+horizon
```

* Identity: `GET /consumer/forecast` derives the owner solely from
  `principal.sub` — a caller cannot name another user, and every query in
  `src/serving/persistence.py` is owner-scoped, so no other user's rows can
  influence a result (isolation is enforced by the query, not by hope).
* Honest statuses: ≥28 days / ≥20 spend txns / ≥12 spend days →
  `personalized`; ≥14 days / ≥8 txns → `limited_history`; otherwise
  `insufficient_history` with a `requirements` block and **no number** —
  never fabricated precision.
* Fallback policy: a global artifact is never substituted for missing history.
  The fallback is the same user's own recent mean
  (`fallback_status="user_recent_baseline"`), and a requested-but-unused
  method is reported explicitly (`"primary_substituted"`).
* Method selection is measured, not assumed: `pipelines/forecast_benchmark.py`
  (expanding-window walk-forward, no random split) — weekday profile
  **37.66% WAPE** vs non-personalized global mean **56.00%** over 30 synthetic
  users (180-day history, 30-day horizon). Synthetic/offline only; never a
  production or customer accuracy claim.
* Legacy `GET /v1/forecast/{user_id}` keeps replaying the static artifact and
  now labels itself `personalization_status="not_personalized"` /
  `fallback_status="global_reference"`; its MAPE ≈ 6.10% comes from a
  different dataset and is not comparable to the numbers above.
* Cache: `forecasts:<sub>:personal:<horizon>:user-v1` (24h TTL), purged by
  ingest / profile change / data deletion; insufficient responses not cached.

---

## 17. Scenario Engine

```text
FinancialProfile + ScenarioParams → run_scenario → ScenarioResult
```

Deterministic. Status: feasible/partial/infeasible.

---

## 18. Decision Engine

```text
ScenarioResult → to_decision → DecisionResult
```

No financial arithmetic of its own. All numbers from underlying services.

---

## 19. Cache Architecture

| Layer | Technology | Fallback |
|---|---|---|
| Primary | Redis | In-memory dict |
| Namespace | principal.sub included | Same |
| TTLs | Features 6h, Forecasts 24h, Budgets 7d, Explanations 30d | Same |

---

## 20. Data Persistence Status

**No production database.** Current state:
- Model artifacts (read-only): models/serving/
- Cache: Redis or in-memory
- No persistent user data storage

---

## 21. Frontend / Backend Boundary

- Frontend: Browser-based interface (frontend/)
- Backend: FastAPI (src/serving/)
- Communication: HTTPS + JWT
- **Frontend is NOT an authorization boundary**

---

## 22. Error Handling

- HTTP 400 (bad request), 401 (unauthorized), 403 (forbidden), 404 (not found), 500 (server error)
- **Known risk:** Error detail leakage via detail=str(exc)

---

## 23. Observability

**Not implemented.** Future requirement: structured logs, monitoring, error tracking.

---

## 24. Security Evolution

| Step | Status |
|---|---|
| Step 1: JWT foundation | Committed locally |
| Step 2: Protected routes | Committed locally |
| Step 3: Ownership/isolation | Committed locally |
| Step 4: Admin lockdown | Planned |
| Rate limiting | Planned |
| Security headers | Planned |

---

## 25. Future Architecture

- **Safe-to-Spend:** Central calculation (planned)
- **AI Copilot:** Provider-agnostic LLM interface (future)
- **Financial Intelligence API:** External API productization (future)
- **Agent Interoperability:** Scoped auth for external agents (future)
- **Multi-tenancy:** Per-user data isolation (future)
- **Privacy/Data Lifecycle:** Deletion, export (future)

---

## 26. Current Architecture Risks

- No production database
- Global/static forecast artifact
- /admin/generate-key public
- No rate limiting
- Error leakage
- No security headers
- No observability
- Frontend security risks
- Browser-side identity/storage risks
