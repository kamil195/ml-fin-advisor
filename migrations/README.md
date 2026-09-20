# Migrations

Plain, versioned SQL files applied in order. No migration framework is used
(the project has no ORM layer; SQLAlchemy appears in the virtualenv only as a
transitive MLflow dependency and is deliberately not a runtime dependency).

## Applying

```bash
psql "$DATABASE_URL" -f migrations/001_user_persistence.sql
```

Files are idempotent (`CREATE TABLE/INDEX IF NOT EXISTS`, single transaction),
so re-running is safe.

## Files

| File | Purpose |
|---|---|
| `001_user_persistence.sql` | `user_profiles`, `ingest_batches`, `transactions` + indexes + stable dedupe unique index |

## Security stance

* Every table carries `owner_sub` (the verified JWT subject) and every
  backend query is scoped by it explicitly — see `src/serving/persistence.py`.
* The backend connects with a privileged role that **bypasses RLS**;
  application-level scoping is therefore the only enforcement and no RLS
  policy is enabled in this step (no browser-direct table access exists).
* No tokens, emails, or derived model outputs are stored.
