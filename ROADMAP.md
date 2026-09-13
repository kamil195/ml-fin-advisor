# Planwisely Roadmap

**Version:** 1.0
**Date:** 2026-09-13

---

## Guiding Principle

```
Security → Data Isolation → Financial Correctness → Trust → UX → Intelligence → AI Interface → Platform Expansion
```

---

## Phase 0 — Preserve Verified Foundations

**Status: COMPLETE**

- FeatureService parity (157/157)
- Classifier integration (LightGBM)
- BudgetOptimiser protections (hard-protected categories)
- Deterministic decision/scenario logic
- Canonical financial semantics
- Existing tested ingestion protections

**Do not casually rewrite these systems.**

---

## Phase 1 — Security / Authorization

**Status: IN PROGRESS**

### Step 1: Supabase JWT Foundation
- **Status:** COMPLETE / COMMITTED LOCALLY / NOT PUSHED
- Commit: `1a5eb8a13d6235d1f8e11d7dfd73767e67894702`
- AuthPrincipal, require_auth, verify_token, JWKS verification, fail-closed

### Step 2: Protected Financial Routes
- **Status:** COMPLETE / COMMITTED LOCALLY / NOT PUSHED
- Commit: `024465c`
- Eight protected endpoints, global API-key middleware removed

### Step 3: User Ownership / Cross-User Isolation
- **Status:** COMPLETE / COMMITTED LOCALLY / NOT PUSHED
- **Commit:** `9b34da0` — `feat(auth): enforce user ownership and cross-user isolation`
- Path/body/CSV ownership enforcement
- Exact budget lookup (no prefix/fallback)
- Per-user cache namespace
- Live identity propagation
- 22 ownership tests, full suite 224 passed

### Step 4: Admin Endpoint Lockdown
- **Status:** PLANNED
- Lock down `/admin/generate-key`

### After Step 4:
- Error sanitization
- Rate limiting / abuse controls
- Cache/Redis hardening
- Deployment/browser security headers
- Observability/security logging

---

## Phase 2 — Frontend Auth

**Status:** PLANNED

- Supabase frontend auth integration
- Remove fake/local auth patterns
- Authenticated route handling
- Session lifecycle
- Logout/cache clearing
- Backend-issued/verified ownership alignment

---

## Phase 3 — Browser / API Hardening

**Status:** PLANNED

- XSS remediation
- Safe DOM rendering
- Security headers
- Trusted hosts
- Safe error UX
- Sensitive browser storage reduction
- User-scoped cache behavior
- Dependency hygiene

---

## Phase 4 — Privacy / Legal / Trust

**Status:** PLANNED

- Privacy Policy
- Terms of Service
- Contact/support
- Data deletion / account deletion flow
- Data-use explanation
- Truthful product claims
- Demo data labeling

---

## Phase 5 — Claim / Trust Correction

**Status:** PLANNED

Remove unsupported claims:
- ~~99.96% classifier accuracy~~ → Verified: ~89.10% (synthetic)
- ~~92% forecast accuracy~~ → Verified: ~6.10% MAPE (synthetic)
- ~~80.9% budget acceptance~~ → Verified: ~78.86% (synthetic)

Use verified contextualized metrics only. Never market synthetic benchmarks as customer metrics.

---

## Phase 6 — Observability / MLOps

**Status:** PLANNED

- Structured logs
- Monitoring
- Error tracking
- Model evaluation
- Drift monitoring
- Dataset/version tracking
- Outcome measurement

---

## Phase 7 — Real User B2C Validation

**Status:** PLANNED

- Real users, real financial workflows
- Behavioral feedback
- Safe-to-Spend usefulness
- Budget/forecast/decision usefulness
- Retention, trust

---

## Phase 8 — Personalization

**Status:** PLANNED

Replace global/static forecast with properly scoped per-user inference.
Do not fake personalization.

---

## Phase 9 — AI Copilot

**Status:** FUTURE

Only after core financial intelligence is reliable.
- Copilot calls structured Planwisely tools
- Provider-agnostic
- No free-form LLM output as authoritative financial logic

---

## Phase 10 — Outcome Tracking

**Status:** FUTURE

Track: recommendation → user action → financial result.
Improves decision quality over time.

---

## Phase 11 — Financial Intelligence API

**Status:** FUTURE

Productize specialized financial capabilities as authenticated APIs/tools.

---

## Phase 12 — Agent Interoperability

**Status:** FUTURE

External AI agents call Planwisely financial intelligence through scoped authorization.

---

## Phase 13 — B2B / Platform

**Status:** FUTURE

- Fintech integrations
- Employers / financial wellness
- Financial advisors
- Embedded financial intelligence

---

## Phase 14 — International Expansion

**Status:** FUTURE

Pakistan-first. Design for eventual:
- Multiple currencies
- Locales
- Financial behaviors
- Regional data providers

---

## Launch Gate

Planwisely is **NOT LAUNCH READY.**

Minimum for launch:
1. All security steps (1–4) committed and pushed
2. Frontend auth integrated
3. Browser/API hardening complete
4. Privacy policy and terms published
5. Unsupported claims removed
6. Observability in place
7. Real user validation begun
