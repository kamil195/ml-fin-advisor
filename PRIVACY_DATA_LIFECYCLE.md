# Privacy & Data Lifecycle (Technical / Internal)

> **Status:** internal technical documentation. This is **not** a
> lawyer-reviewed Privacy Policy and makes **no legal or security-compliance
> claims of any kind**; it does not replace a user-facing privacy policy,
> which remains a launch blocker.

## 1. Current State

- **User-owned persistence now exists (STEP 12):** a Supabase **Postgres**
  database, reached server-side only via a direct `DATABASE_URL` (psycopg 3).
  Three owner-scoped tables are live: `user_profiles` (the financial-profile
  inputs the product uses today), `transactions` (CSV-ingested, classified
  transaction rows) and `ingest_batches` (upload audit trail). Schema lives in
  `migrations/001_user_persistence.sql`.
- **Ownership key is the JWT subject.** Every row carries `owner_sub` =
  verified `AuthPrincipal.sub`; every backend query is explicitly scoped by
  it. No request field (CSV column, body, query, path) can set or override the
  owner. The backend connects with a privileged role that **bypasses RLS**,
  so application-level scoping is the active enforcement; RLS policies for a
  future browser-safe access path are intentionally not yet enabled.
- **User-facing lifecycle endpoints are real:** `GET /consumer/data/export`
  returns the caller's profile + transactions as JSON; `DELETE /consumer/data`
  removes the caller's profile, transactions, and ingest batches **in one
  transaction** (fail-closed: a failure returns a generic 503 and success is
  never falsely reported), then purges the caller's user-scoped cache
  namespaces. **This deletes Planwisely financial data only** — the Supabase
  Auth account itself is NOT deleted; account deletion remains a separate
  operation and is not yet implemented.
- **Deliberately NOT persisted:** JWTs/tokens/API keys (never), email
  addresses (identity is `sub` only), classifier/forecast/budget/advise
  outputs (derived on demand and held only in TTL-bounded user-scoped cache),
  habit strengths / compliance history (per-request inputs).
- Deployments without `DATABASE_URL` run persistence-free: user-data routes
  return a generic 503 and all other behavior is unchanged.
- **Still unresolved:** explicit retention periods, backup/restore behaviour,
  and access/deletion auditability remain deployment-level concerns (Supabase
  backups), not yet enforced in code. No compliance claim is made anywhere.

## 2. Data Classification (what exists today)

| Class | Examples | Today's reality |
|---|---|---|
| **IDENTITY** | JWT `sub`, optional `email` claim | `sub` is the ownership key of every persisted row (`owner_sub`). `email` is never used as identity and **never persisted**. JWTs/tokens are never stored or logged (see §4). |
| **FINANCIAL** | transaction descriptions, amounts, dates, categories, CSV uploads, financial profile | **Persisted on request**: classified transactions (`transactions` table) and profile inputs (`user_profiles`) — always owned by `owner_sub`, deletable via `DELETE /consumer/data`, exportable via `GET /consumer/data/export`. CSV uploads are still decoded in memory with no temp files. Derived outputs (forecasts/budgets/advise) remain request-scoped + TTL-bounded cache only. |
| **TECHNICAL** | request paths, status codes, timings, rate-limit counters, cache state | May be logged (§4) or cached (§5). |
| **MODEL ARTIFACTS** | `models/serving/*.joblib`, `*_results.json` | Static, training-time files in the repository. Not derived from any real user; contain no personal data. |

## 3. Data Minimization

Only the fields required for the current financial computation are accepted:
classification needs description/amount/timestamp (+MCC/channel/account
type); forecasting/budget need the same user-scoped inputs. No analytics,
no tracking, no cookies, no third-party scripts in the served frontend, and
no collection of identifiers beyond the verified JWT subject.

## 4. Logging Policy

**Allowed (current behaviour):** HTTP method/path + status + duration; log
messages that name the event type only; exception **class names** and
tracebacks **server-side only** (never in client responses — Step 5).

**Prohibited and verified by tests (`tests/unit/test_privacy_data_handling.py`):**
- bearer tokens / full JWT claims — never logged
- transaction descriptions, CSV row contents, raw financial payloads — never logged
- user email addresses, raw user IDs (e.g. JWT `sub`) — never logged;
  cache-hit messages are identity-free (`Cache HIT for … (user-scoped)`)
- Redis URLs / credentials — never logged (redaction test exists since Step 5)

## 5. Cache (Redis / in-memory fallback)

## 6. Persistence Implementation Status (STEP 12 contract)

The Step-8 contract is now implemented (except where noted):

1. **Ownership by JWT subject** — ✅ every row is keyed to `owner_sub` =
   verified `AuthPrincipal.sub`; every repository method requires it and every
   statement is explicitly scoped by it (no unscoped fallback).
2. **Deletion** — ✅ `DELETE /consumer/data` removes all user-owned rows
   (profile, transactions, ingest batches) in ONE transaction, fail-closed on
   partial failure.
3. **Export** — ✅ `GET /consumer/data/export` returns all user-owned data
   (profile + transactions, `planwisely.export.v1` JSON).
4. **Cache/state deletion** — ✅ mutations purge the caller's user-scoped
   namespaces (`budgets:<sub>`, `forecasts/features/explanations:<sub>:*`)
   via `CacheClient.purge_user` — never a global invalidation.
5. **Audit result** — ✅ delete/export responses report exactly what was
   removed/returned, per store.
6. **Fail closed on partial deletion** — ✅ any statement failure rolls back
   the whole operation and surfaces a generic 503; success is never falsely
   reported.
7. **Retention** — ⏳ **not yet implemented**: explicit retention periods,
   backup/restore behaviour, and auditability of access/deletion remain
   deployment concerns (Supabase backups) and are unresolved.

Additional Step-12 facts:

* **Deduplication:** transactions carry a stable dedupe key
  (`owner_sub, occurred_at, amount, merchant_name, merchant_mcc`); re-uploading
  an identical CSV skips duplicates instead of duplicating rows. Known
  limitation: two genuinely distinct purchases with identical merchant/MCC at
  the same timestamp would collapse into one row (accepted for the MVP).
* **Classification output is persisted with the transaction** (category +
  confidence at ingest time); forecasts/budgets/advise outputs are NOT
  persisted — they are recomputed or replayed from static artifacts and cached
  with TTLs only.
* **Failure model:** database unavailability or statement failure produces a
  generic 503 (`Persistence is temporarily unavailable. Please retry later.`).
  SQL text, values, table names, and the DSN are never logged or returned.
  Until Step 12, "no user data is retained anywhere" was true; today retained
  data is exactly the three tables above, owned by `sub`, deletable via
  `DELETE /consumer/data`.

## 7. User-Facing Policy Status (launch blockers)

- A user-facing **Privacy Policy** and **Terms** do not exist yet and are
  **required before any public launch**.
- No compliance certification is claimed anywhere in this repository.


- Keys are identity-**namespaced** (`forecasts:<sub>:…`, `budgets:<sub>`) —
  no cross-user fallback exists (exact-key lookup only; ownership tests cover this).
- Keys contain **only the JWT subject** — never tokens, emails, or credentials.
- **Both** the Redis path and the in-memory fallback honour TTLs
  (forecasts 24 h, budgets 7 d, features 6 h, explanations 30 d). The
  fallback stores `(expiry, value)` with a monotonic clock, lazily expires
  on read and purges expired entries on every write, so stale user-scoped
  data cannot persist (Step 8 fix; previously the fallback ignored TTLs).
