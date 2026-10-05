-- ─────────────────────────────────────────────────────────────────────────────
-- Planwisely — STEP 14: Safe-to-Spend inputs
-- 002_safe_to_spend.sql
--
-- Smallest safe addition to `user_profiles` that the Safe-to-Spend endpoint
-- needs (GET /consumer/safe-to-spend). BOTH columns are optional and nullable:
--
--   * next_payday    — the user's next payday (the horizon end for
--                      Safe-to-Spend). When it is NULL the backend falls back to
--                      a cadence measured from the user's OWN income deposits;
--                      if that is not possible the endpoint returns
--                      `missing_payday` and no amount. A payday is never
--                      silently assumed.
--   * safety_buffer  — the amount the user wants to keep untouched before
--                      payday. NULL means "not configured": the engine then uses
--                      0.00 and says so explicitly (the reported amount is an
--                      upper bound). No default buffer is invented.
--
-- Available funds need NO new column: `liquid_buffer` already exists and is
-- documented in `src/data/models.py` (FinancialProfile) as "Cash/liquid assets
-- available today" — the product's existing cash-position input.
--
-- Additive and backward compatible: existing rows keep NULL, existing inserts
-- and updates are unaffected, no backfill and no data rewrite. Idempotent, so it
-- is safe to re-run.
--
-- Apply with:  psql "$DATABASE_URL" -f migrations/002_safe_to_spend.sql
-- ─────────────────────────────────────────────────────────────────────────────

BEGIN;

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS next_payday DATE,
    ADD COLUMN IF NOT EXISTS safety_buffer NUMERIC(14, 2);

-- A negative buffer is meaningless; NULL ("not configured") stays allowed.
ALTER TABLE user_profiles
    DROP CONSTRAINT IF EXISTS ck_user_profiles_safety_buffer_non_negative;
ALTER TABLE user_profiles
    ADD CONSTRAINT ck_user_profiles_safety_buffer_non_negative
    CHECK (safety_buffer IS NULL OR safety_buffer >= 0);

COMMIT;
