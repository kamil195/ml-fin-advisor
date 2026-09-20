# Planwisely

**Know what you can safely spend before payday.**

Pakistan-first B2C financial intelligence for salaried professionals (ages 22–40).

---

## What is Planwisely?

Planwisely combines **specialized machine learning** with **deterministic financial logic** to produce authoritative financial intelligence.

- **Specialized ML:** Transaction classification, expense forecasting, behavioral features
- **Deterministic Logic:** Budget optimization, scenario analysis, decision recommendations
- **Optional LLM (Future):** Interface/explanation layer — never the source of financial truth

The LLM is **not required** for the system to be "AI." Planwisely already uses specialized AI/ML.

---

## Core Intelligence Pipeline

```text
Financial Data
  → Transaction Intelligence
  → Financial Profile
  → Behavior / Forecast / Budget
  → Scenario Engine
  → Decision Layer
  → Recommended Actions
```

---

## Current Capabilities

| Capability | Status |
|---|---|
| Transaction Classification | LightGBM, 30 categories, 157 features |
| Expense Forecasting | Prophet-based (global/demo/static artifact) |
| Budget Optimization | Constraint-based with hard-protected categories |
| Scenario Analysis | Deterministic |
| Decision Recommendations | Deterministic |
| CSV Ingestion | With validation and ownership enforcement |
| Authentication | Supabase JWT (Steps 1–3 committed) |
| Safe-to-Spend | Planned |
| AI Copilot | Future |

---

## Architecture Overview

```text
User → Frontend → Supabase Auth → JWT → FastAPI → require_auth
    → AuthPrincipal.sub → Ownership Enforcement → Financial Services
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for trust boundaries and data flows.

---

## Repository / Runtime Overview

```text
src/
+-- data/           Schemas, ingestion, synthetic-data
+-- evaluation/     Metrics, comparisons, auditing
+-- features/       Numerical, temporal, behavioral, text features
+-- models/         ML models (classifier, forecaster, recommender)
+-- serving/        FastAPI app, middleware, routes, auth, cache
+-- services/       Financial logic (profile, scenario, decision)
+-- utils/          Constants, logging, privacy
tests/              Unit and integration tests
models/serving/     Demonstration model artifacts
frontend/           Browser-based interface
infrastructure/     Docker, deployment
```

---

## Getting Started

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Replace placeholder values with your own local credentials. Never commit the real `.env` file.

### 3. Generate demonstration data (optional)

```bash
python -m src.data.mock_generator
```

### 4. Run tests

```bash
pytest tests/unit/ -v
```

### 5. Start the API

```bash
uvicorn src.serving.app:app --reload
```

Interactive API documentation is available at `http://localhost:8000/docs`.

### Docker

```bash
docker compose up --build
```

This starts the API and Redis services; the API is exposed on port `8000`.

---

## API Overview

### Protected Endpoints (require Supabase JWT)

| Method | Path | Description |
|---|---|---|
| POST | `/v1/classify` | Classify a transaction |
| GET | `/v1/forecast/{user_id}` | Expense forecast |
| GET | `/v1/budget/{user_id}` | Budget recommendations |
| POST | `/consumer/advise` | Financial advice |
| POST | `/consumer/transactions/ingest-csv` | CSV ingestion |
| POST | `/consumer/classify/live` | Live classification |
| POST | `/consumer/forecast/live` | Live forecast |
| POST | `/consumer/budget/live` | Live budget |

### Public Endpoints

| Method | Path |
|---|---|
| GET | `/health`, `/ready`, `/docs`, `/openapi.json`, `/redoc`, `/favicon.ico` |
| POST | `/admin/generate-key` (**KNOWN SECURITY ISSUE**) |

---

## Authentication

Supabase JWT (RS256) verified via JWKS. Fail-closed: missing/invalid token → 401.

| Component | File |
|---|---|
| AuthPrincipal | `src/serving/auth.py` |
| require_auth | `src/serving/auth.py` |
| Auth middleware | `src/serving/middleware.py` |

---

## Authorization / User Isolation

**Fundamental invariant:** `AuthPrincipal.sub` = financial resource owner

- Path `{user_id}` must equal `principal.sub` → 403 on mismatch
- Body `user_id` must match `principal.sub` → 403 on mismatch
- CSV rows must match `principal.sub` → 403 on mismatch
- Cache keys are namespaced by `principal.sub`
- Budget lookup is exact (no prefix/fallback)

---

## Classification

LightGBM, 30 L2 categories, 157 features.

**Synthetic evaluation (NOT production metrics):**

| Metric | Value |
|---|---|
| Accuracy | 89.10% |
| Macro-F1 | 85.56% |
| Weighted-F1 | 89.11% |
| Top-3 | 98.07% |
| ECE | 7.81% |

---

## Forecasting

Prophet-based. **LIMITATION:** Current artifact is global/demo/static — NOT genuinely personalized. Authorization isolation only.

MAPE: ~6.10% (synthetic/demo data).

---

## Budget Optimisation

Constraint-based (scipy.optimize.linprog).

**Hard-protected categories:** Rent/Mortgage, Utilities, Home Insurance, Insurance Premiums, Loan Payments, Taxes.

Acceptance: ~78.86% (synthetic/demo).

---

## Scenario / Decision Systems

Deterministic engines. No randomness, no LLM calls. All numbers from underlying services.

---

## Current Security Status

| Item | Status |
|---|---|
| Auth Step 1 (JWT foundation) | Committed locally, not pushed |
| Auth Step 2 (protected routes) | Committed locally, not pushed |
| Auth Step 3 (ownership/isolation) | Committed locally, not pushed |
| /admin/generate-key | Public — known issue |
| Rate limiting | Not implemented |
| Privacy policy / terms | Not implemented |
| Security headers | Not implemented |

---

## Testing

| Suite | Result |
|---|---|
| Auth / ownership / error-handling / rate-limiting / security-headers batch | 103 passed |
| Privacy / claims / frontend-auth / branding / frontend-security / CSV-ingest batch | 107 passed |
| Advise + classify endpoint suites | 15 passed |
| Persistence: transactions / profiles / data-lifecycle (Step 12) | 27 passed |
| **Full unit suite** | **389 passed, 1 skipped** |
| compileall | exit 0 |

---

## Current Limitations

- Forecast artifact is global/demo/static (not personalized); no personalized forecasting yet
- Safe-to-Spend not implemented/validated yet
- Persistence (Step 12) is an MVP foundation: profile + transactions only; retention/backup policy unresolved; Supabase Auth account deletion is separate and not implemented
- No privacy policy/terms
- No observability/monitoring
- Outputs are educational/portfolio demonstrations on synthetic/demo data — not professional financial advice; not connected to real bank accounts

---

## Documentation

| Document | Purpose |
|---|---|
| SPEC.md | Technical implementation source of truth |
| README.md | Repository/developer entry point (this file) |
| ARCHITECTURE.md | Architecture, trust boundaries, data flows |
| ROADMAP.md | Dependency-driven execution roadmap |
| FRONTEND_DESIGN_SPEC.md | Frontend/UI/UX design source of truth |
| CHANGELOG.md | Verified implementation history |

---

## Future Direction

- Safe-to-Spend calculation
- AI Copilot (provider-agnostic)
- Financial Intelligence API
- Agent interoperability
- International expansion (Pakistan-first)

---

## Development Principles

1. Specialized ML + deterministic logic = authoritative financial intelligence
2. LLM is optional, never the source of financial truth
3. Authorization isolation: JWT sub is authoritative
4. Fail-closed by default
5. Deterministic financial calculations
6. Honest representation of missing information

---

## Author

**Muhammad Kamil Shah**
BS Data Science

## License

See the repository license file for usage terms.

---

**Status: Active Development — Not Launch Ready**
