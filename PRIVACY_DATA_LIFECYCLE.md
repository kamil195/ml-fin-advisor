# Privacy & Data Lifecycle (Technical / Internal)

> **Status:** internal technical documentation. This is **not** a
> lawyer-reviewed Privacy Policy and makes **no legal or security-compliance
> claims of any kind**; it does not replace a user-facing privacy policy,
> which remains a launch blocker.

## 1. Current State

- **No production database / persistence layer exists.** No user records,
  no transaction store, no profile store. Nothing about a user is retained
  beyond the lifetime of a single request or a TTL-bounded cache entry.
- **No deletion/export endpoints exist** — and none are pretended. Deleting
  data is impossible to promise while nothing is persisted; see §6 for the
  contract the future persistence step must satisfy.

## 2. Data Classification (what exists today)

| Class | Examples | Today's reality |
|---|---|---|
| **IDENTITY** | JWT `sub`, optional `email` claim | Held only for the request lifetime (server memory). Never persisted. Never logged (see §4). `email` is never used as identity. |
| **FINANCIAL** | transaction descriptions, amounts, dates, categories, CSV uploads, financial profile, forecasts, budgets | Processed **in memory per request** and returned to the caller. **Never written to disk, never persisted, never logged.** CSV uploads (`POST /consumer/transactions/ingest-csv`) are decoded from the raw request body in memory — no temp files are created and nothing remains after the response. |
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

## 6. Future Persistence Requirements (contract for Step 12)

Any future persistence layer **must** implement, before launch:

1. **Ownership by JWT subject** — every record is keyed to `AuthPrincipal.sub`.
2. **Deletion** — `delete_user_data(user_id)` removes all user-owned records
   across all stores (DB rows, cache entries, derived artifacts).
3. **Export** — `export_user_data(user_id)` returns all user-owned data.
4. **Cache/state deletion** — deleting a user also purges their cache
   namespaces (`forecasts:<sub>:*`, `budgets:<sub>`, `features:<sub>:*`,
   `explanations:<sub>:*`).
5. **Audit result** — the delete/export operation reports exactly what was
   removed/returned, per store.
6. **Fail closed on partial deletion** — a failed partial delete must not
   report success; it must surface an error state.
7. **Retention** — explicit retention periods enforced in code (not fake env
   placeholders), plus defined backup and restore behaviour and auditability
   of access/deletion.

Until that layer exists: **no user data is retained anywhere**, so there is
nothing to delete or export — and no code pretends otherwise.

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
