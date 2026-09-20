"""
Server-side persistence for user-owned data (STEP 12).

Runtime database: Supabase Postgres via a server-side ``DATABASE_URL`` and a
direct psycopg 3 connection. One coherent client; no ORM; no second SDK. The
connection is created lazily and serialized behind a lock (single-worker
deployment; sync endpoints run in FastAPI's thread pool).

Ownership model (JWT contract preserved):

* Identity is the verified JWT subject (``AuthPrincipal.sub``).
* Every row is owned by ``owner_sub``; every method takes ``owner_sub`` first
  and every statement is explicitly scoped by it. No unscoped fallback exists.

Failure model:

* Connection/statement failures raise PersistenceUnavailable (carrying only
  the exception type — never SQL text, values, table names, or the DSN).
* Multi-statement operations are transactional: a failure rolls back, so a
  partial delete can never be reported as success (fail closed).

Privacy: the DSN is never logged; the schema stores no tokens/emails
(see migrations/001_user_persistence.sql).
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DSN_ENV_VAR = "DATABASE_URL"
TIMEOUT_ENV_VAR = "DATABASE_CONNECT_TIMEOUT"
DEFAULT_CONNECT_TIMEOUT_S = 5.0

DETAIL_503 = "Persistence is temporarily unavailable. Please retry later."
DETAIL_NOT_CONFIGURED = "Persistence is not configured on this deployment."


class PersistenceUnavailable(RuntimeError):
    """The database is unreachable or a statement failed (generic; no SQL/DSN)."""


class PersistenceConfig:
    """Server-side connection settings resolved from the environment."""

    __slots__ = ("dsn", "connect_timeout")

    def __init__(self, dsn: str, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S) -> None:
        self.dsn = dsn
        self.connect_timeout = connect_timeout

    @classmethod
    def env_var_name(cls) -> str:
        """Name of the environment variable holding the DSN (for safe logs)."""
        return DSN_ENV_VAR

    @classmethod
    def from_env(cls) -> "PersistenceConfig | None":
        """Config from ``DATABASE_URL``, or ``None`` when unset/blank.

        A blank/unset DSN means persistence is *disabled* on the deployment —
        a supported state (non-persistence endpoints are unaffected), matching
        the pre-Step-12 baseline.
        """
        raw = os.getenv(DSN_ENV_VAR, "").strip()
        if not raw:
            return None
        raw_timeout = os.getenv(TIMEOUT_ENV_VAR, "").strip()
        try:
            timeout = float(raw_timeout) if raw_timeout else DEFAULT_CONNECT_TIMEOUT_S
        except ValueError:
            timeout = DEFAULT_CONNECT_TIMEOUT_S
        return cls(dsn=raw, connect_timeout=max(1.0, min(timeout, 30.0)))


class PostgresStore:
    """Owner-scoped persistence operations on Supabase Postgres.

    Every method requires ``owner_sub`` (verified JWT subject) and every
    statement is scoped by it. Operations run on one lazily-created,
    lock-serialized connection; multi-statement operations are transactional
    (all-or-nothing).
    """

    _TXN_COLS = (
        "id, occurred_at, amount, currency, merchant_name, merchant_mcc, "
        "account_type, channel, location_city, location_country, "
        "raw_description, is_pending, category_l1, category_l2, confidence, "
        "source, ingest_batch_id, created_at"
    )
    _PROFILE_COLS = (
        "owner_sub, income, savings_target, liquid_buffer, "
        "total_debt, monthly_debt_payments, created_at, updated_at"
    )

    def __init__(self, config: PersistenceConfig) -> None:
        self._config = config
        self._conn: Any = None
        self._lock = threading.Lock()

    # ── connection handling ─────────────────────────────────────────────

    def _connection(self) -> Any:
        """Return the live connection, (re)connecting lazily when needed."""
        if self._conn is None or self._conn.closed:
            import psycopg
            from psycopg.rows import dict_row

            self._conn = psycopg.connect(
                self._config.dsn,
                connect_timeout=int(self._config.connect_timeout),
                row_factory=dict_row,
            )
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        """Run statements in one serialized, commit-on-success transaction."""
        with self._lock:
            try:
                yield self._connection()
                self._conn.commit()
            except Exception as exc:  # noqa: BLE001 — fail closed, generic surface
                self._rollback_quiet()
                # Only the exception TYPE is carried/logged: no SQL text,
                # bound values, table names, or DSN.
                logger.error("persistence operation failed (type: %s)", type(exc).__name__)
                raise PersistenceUnavailable(type(exc).__name__) from None

    def _rollback_quiet(self) -> None:
        try:
            if self._conn is not None and not self._conn.closed:
                self._conn.rollback()
        except Exception:  # noqa: BLE001 — unusable connection; drop it
            self._conn = None

    def close(self) -> None:
        """Close the connection (shutdown path; best-effort)."""
        with self._lock:
            if self._conn is not None and not self._conn.closed:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._conn = None

    # ── domain methods ───────────────────────────────────────────────────

    def insert_transactions(
        self,
        owner_sub: str,
        rows: list[dict[str, Any]],
        ingest_batch_id: str | None = None,
    ) -> tuple[int, int]:
        """Insert transactions owned by ``owner_sub``; returns (persisted, skipped).

        Stable deduplication: ``UNIQUE(owner_sub, occurred_at, amount,
        merchant_name, merchant_mcc, account_type, channel)`` — an identical
        re-upload is skipped per row instead of silently duplicated (Step 12F).
        The whole batch is transactional: either every new row is stored or
        none is.
        """
        if not rows:
            return 0, 0
        sql = (
            "INSERT INTO transactions ("
            "owner_sub, occurred_at, amount, currency, merchant_name, merchant_mcc,"
            " account_type, channel, location_city, location_country,"
            " raw_description, is_pending, category_l1, category_l2, confidence,"
            " source, ingest_batch_id) VALUES ("
            "%(owner_sub)s, %(occurred_at)s, %(amount)s, %(currency)s,"
            " %(merchant_name)s, %(merchant_mcc)s, %(account_type)s, %(channel)s,"
            " %(location_city)s, %(location_country)s, %(raw_description)s,"
            " %(is_pending)s, %(category_l1)s, %(category_l2)s, %(confidence)s,"
            " %(source)s, %(ingest_batch_id)s)"
            " ON CONFLICT (owner_sub, occurred_at, amount, merchant_name,"
            " merchant_mcc) DO NOTHING"
        )
        persisted = skipped = 0
        with self._transaction() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    payload = dict(row)
                    payload["owner_sub"] = owner_sub
                    payload["ingest_batch_id"] = ingest_batch_id
                    payload.setdefault("currency", "USD")
                    payload.setdefault("source", "csv_ingest")
                    cur.execute(sql, payload)
                    persisted += cur.rowcount if cur.rowcount > 0 else 0
                    skipped += 1 if cur.rowcount == 0 else 0
        return persisted, skipped

    def list_transactions(
        self, owner_sub: str, limit: int = 200, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List ``owner_sub``'s transactions, newest first (explicitly scoped)."""
        sql = (
            f"SELECT {_TXN_COLS} FROM transactions WHERE owner_sub = %s"
            " ORDER BY occurred_at DESC, created_at DESC LIMIT %s OFFSET %s"
        )
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (owner_sub, limit, offset))
                return list(cur.fetchall())

    def delete_transaction(self, owner_sub: str, transaction_id: str) -> bool:
        """Delete one transaction iff it is owned by ``owner_sub`` (scoped)."""
        sql = "DELETE FROM transactions WHERE id = %s AND owner_sub = %s"
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (transaction_id, owner_sub))
                return cur.rowcount > 0

    def create_ingest_batch(
        self, owner_sub: str, source: str, row_count: int
    ) -> str:
        """Create an ingest-batch audit row for ``owner_sub``; returns its id."""
        sql = (
            "INSERT INTO ingest_batches (owner_sub, source, row_count)"
            " VALUES (%s, %s, %s) RETURNING id"
        )
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (owner_sub, source, row_count))
                return str(cur.fetchone()["id"])

    def get_profile(self, owner_sub: str) -> dict[str, Any] | None:
        """Return ``owner_sub``'s financial profile row, or ``None``."""
        sql = f"SELECT {_PROFILE_COLS} FROM user_profiles WHERE owner_sub = %s"
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (owner_sub,))
                return cur.fetchone()

    def upsert_profile(self, owner_sub: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create/update ``owner_sub``'s profile; returns the stored row.

        Only the product's current fields are written; no speculative columns.
        """
        sql = (
            "INSERT INTO user_profiles ("
            "owner_sub, income, savings_target, liquid_buffer, total_debt,"
            " monthly_debt_payments) VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (owner_sub) DO UPDATE SET"
            " income = EXCLUDED.income,"
            " savings_target = EXCLUDED.savings_target,"
            " liquid_buffer = EXCLUDED.liquid_buffer,"
            " total_debt = EXCLUDED.total_debt,"
            " monthly_debt_payments = EXCLUDED.monthly_debt_payments,"
            " updated_at = now()"
            f" RETURNING {_PROFILE_COLS}"
        )
        params = (
            owner_sub,
            data.get("income"),
            data.get("savings_target"),
            data.get("liquid_buffer"),
            data.get("total_debt"),
            data.get("monthly_debt_payments"),
        )
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()

    def delete_profile(self, owner_sub: str) -> bool:
        """Delete ``owner_sub``'s profile row (scoped)."""
        sql = "DELETE FROM user_profiles WHERE owner_sub = %s"
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (owner_sub,))
                return cur.rowcount > 0

    def export_user_data(
        self, owner_sub: str
    ) -> dict[str, Any]:
        """Return all user-owned rows for ``owner_sub`` (profile + transactions)."""
        profile_sql = (
            f"SELECT {_PROFILE_COLS} FROM user_profiles WHERE owner_sub = %s"
        )
        txn_sql = (
            f"SELECT {_TXN_COLS} FROM transactions WHERE owner_sub = %s"
            " ORDER BY occurred_at DESC, created_at DESC"
        )
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(profile_sql, (owner_sub,))
                profile = cur.fetchone()
                cur.execute(txn_sql, (owner_sub,))
                transactions = list(cur.fetchall())
        return {"profile": profile, "transactions": transactions}

    def delete_user_data(self, owner_sub: str) -> dict[str, Any]:
        """Delete ALL user-owned rows for ``owner_sub`` in ONE transaction.

        Fails closed: any statement failure rolls the whole operation back, so
        partial deletion is never reported as success (Step 12I / privacy §6).
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM transactions WHERE owner_sub = %s", (owner_sub,)
                )
                txns = cur.rowcount
                cur.execute(
                    "DELETE FROM ingest_batches WHERE owner_sub = %s", (owner_sub,)
                )
                batches = cur.rowcount
                cur.execute(
                    "DELETE FROM user_profiles WHERE owner_sub = %s", (owner_sub,)
                )
                profiles = cur.rowcount
        return {
            "profile_deleted": profiles > 0,
            "transactions_deleted": txns,
            "batches_deleted": batches,
        }
