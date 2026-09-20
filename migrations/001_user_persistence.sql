-- ─────────────────────────────────────────────────────────────────────────────
-- Planwisely — STEP 12: user-owned persistence (Supabase / Postgres)
-- 001_user_persistence.sql
--
-- Scope (deliberate MVP foundation only — nothing speculative):
--   * user_profiles   — the financial-profile inputs the product uses today
--                       (budget/live + advise inputs: income, savings_target,
--                       liquid_buffer, total_debt, monthly_debt_payments)
--   * ingest_batches  — audit trail for CSV ingest runs (owner, source, count)
--   * transactions    — user-owned transaction records persisted by
--                       POST /consumer/transactions/ingest-csv
--
-- Deliberately NOT stored:
--   * JWTs / bearer tokens / API keys (never)
--   * email addresses (identity is the JWT `sub` only)
--   * classifier/forecast/budget/advise outputs (derived on demand or replayed
--     from static artifacts — recomputed, cached with TTLs, not persisted)
--   * habit strengths / compliance history (per-request inputs today)
--
-- Ownership model:
--   Every row carries `owner_sub` = the verified JWT subject
--   (``AuthPrincipal.sub``). The backend scopes EVERY query by `owner_sub`
--   explicitly; no repository method has an unscoped fallback. The backend
--   connects with a privileged Postgres role, which BYPASSES RLS — so
--   application-level scoping is mandatory and is the only enforcement in
--   this step. Row-Level Security policies for browser-safe direct access are
--   intentionally NOT enabled here (no such access path exists yet; an
--   untested policy would be a false claim). See PRIVACY_DATA_LIFECYCLE.md §6.
--
-- Apply with:  psql "$DATABASE_URL" -f migrations/001_user_persistence.sql
-- (Idempotent: CREATE TABLE/INDEX IF NOT EXISTS.)
-- ─────────────────────────────────────────────────────────────────────────────

BEGIN;

-- ── 1. Financial profile (user-supplied inputs only) ────────────────────────
-- One row per user; `owner_sub` is the primary key, so a user can never have
-- two profiles and no unscoped row can exist.
CREATE TABLE IF NOT EXISTS user_profiles (
    owner_sub             TEXT PRIMARY KEY,
    income                NUMERIC(14, 2) NOT NULL DEFAULT 0,
    savings_target        NUMERIC(14, 2),
    liquid_buffer         NUMERIC(14, 2),
    total_debt            NUMERIC(14, 2),
    monthly_debt_payments NUMERIC(14, 2),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── 2. Ingest batches (audit trail) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingest_batches (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_sub   TEXT NOT NULL,
    source      TEXT NOT NULL,
    row_count   INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ingest_batches_owner
    ON ingest_batches (owner_sub, created_at DESC);

-- ── 3. Transactions (user-owned) ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_sub        TEXT NOT NULL,
    occurred_at      TIMESTAMPTZ NOT NULL,
    amount           DOUBLE PRECISION NOT NULL,   -- signed; negative = debit
    currency         TEXT NOT NULL DEFAULT 'USD',
    merchant_name    TEXT NOT NULL,
    merchant_mcc     INTEGER NOT NULL,
    account_type     TEXT NOT NULL,
    channel          TEXT NOT NULL,
    location_city    TEXT,
    location_country TEXT,
    raw_description  TEXT,
    is_pending       BOOLEAN NOT NULL DEFAULT FALSE,
    category_l1      TEXT,
    category_l2      TEXT,
    confidence       DOUBLE PRECISION,
    source           TEXT NOT NULL DEFAULT 'csv_ingest',
    ingest_batch_id  UUID REFERENCES ingest_batches (id) ON DELETE SET NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_transactions_owner_time
    ON transactions (owner_sub, occurred_at DESC);

-- Stable deduplication key: re-uploading the same CSV must not create
-- duplicate rows. (owner, timestamp, amount, merchant, MCC) is a natural
-- stable identity for CSV-sourced transactions; ON CONFLICT DO NOTHING skips
-- repeats. Known limitation (documented in PRIVACY_DATA_LIFECYCLE.md §6): two
-- genuinely distinct purchases with identical merchant/MCC at the same
-- second would collapse into one row — accepted for the MVP.
CREATE UNIQUE INDEX IF NOT EXISTS uq_transactions_owner_dedupe
    ON transactions (owner_sub, occurred_at, amount, merchant_name, merchant_mcc);

COMMIT;
