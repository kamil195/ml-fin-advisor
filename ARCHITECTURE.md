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

Honesty rules: Income ONLY from CategoryL2.INCOME. No fabricated income.

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
