"""STEP 12M — persistence tests: transaction ingest + read/delete endpoints.

Strategy (Step 12L): the full unit suite must stay network-free, so these
tests exercise the endpoint ⇄ repository contract through an in-memory
:class:`FakeStore` mirroring the exact ``PostgresStore`` method signatures
(mandatory ``owner_sub`` first parameter, ``PersistenceUnavailable`` failure
mode, dedupe tuple). No production Supabase is ever contacted; SQL text is
validated in the Phase-5 security sweep.

Ownership rules under test:

* ``owner_sub`` always comes from the verified JWT subject — never a
  body/query field (the CSV's own ``user_id`` column is never authority).
* Cross-user reads return only the caller's rows; foreign/unknown resource
  ids → 404; persistence failures → generic 503 with no SQL/DSN leakage.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.persistence import (
    DETAIL_503,
    DETAIL_NOT_CONFIGURED,
    PersistenceUnavailable,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

USER_A = "user-AAA"
USER_B = "user-BBB"

CSV_HEADER = (
    "user_id,timestamp,amount,currency,merchant_name,merchant_mcc,"
    "account_type,channel"
)


def _row(**over) -> str:
    base = {
        "user_id": USER_A,
        "timestamp": "2026-03-05T12:00:00",
        "amount": "-100.0",
        "currency": "USD",
        "merchant_name": "FreshMart",
        "merchant_mcc": "5411",
        "account_type": "CHECKING",
        "channel": "POS",
    }
    base.update(over)
    return ",".join(base.values())


def _csv(*rows: str) -> str:
    return "\n".join([CSV_HEADER, *rows]) + "\n"


_CSV_HEADERS = {"Content-Type": "text/csv; charset=utf-8"}


class FakeStore:
    """In-memory stand-in mirroring PostgresStore's owner-scoped contract."""

    def __init__(self) -> None:
        self.profiles: dict[str, dict[str, Any]] = {}
        self.transactions: dict[str, tuple[str, dict[str, Any]]] = {}
        self._keys: set[tuple] = set()
        self.batches: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.fail_methods: set[str] = set()

    def _maybe_fail(self, method: str) -> None:
        self.calls.append(method)
        if method in self.fail_methods:
            raise PersistenceUnavailable("InjectedFailure")

    def create_ingest_batch(self, owner_sub, source, row_count):
        self._maybe_fail("create_ingest_batch")
        batch_id = f"batch-{uuid.uuid4()}"
        self.batches.append(
            {"id": batch_id, "owner_sub": owner_sub, "row_count": row_count}
        )
        return batch_id

    def insert_transactions(self, owner_sub, rows, ingest_batch_id=None):
        self._maybe_fail("insert_transactions")
        persisted = skipped = 0
        for row in rows:
            key = (
                owner_sub, row["occurred_at"], row["amount"],
                row["merchant_name"], row["merchant_mcc"],
            )
            if key in self._keys:
                skipped += 1
                continue
            self._keys.add(key)
            txn_id = str(uuid.uuid4())
            self.transactions[txn_id] = (owner_sub, dict(row))
            persisted += 1
        return persisted, skipped

    def list_transactions(self, owner_sub, limit=200, offset=0):
        self._maybe_fail("list_transactions")
        rows = [t for owner, t in self.transactions.values() if owner == owner_sub]
        rows.sort(key=lambda r: r["occurred_at"], reverse=True)
        return rows[offset:offset + limit]

    def delete_transaction(self, owner_sub, transaction_id):
        self._maybe_fail("delete_transaction")
        entry = self.transactions.get(transaction_id)
        if entry is None or entry[0] != owner_sub:
            return False
        del self.transactions[transaction_id]
        return True

    def get_profile(self, owner_sub):
        self._maybe_fail("get_profile")
        return self.profiles.get(owner_sub)

    def upsert_profile(self, owner_sub, data):
        self._maybe_fail("upsert_profile")
        row = dict(data)
        row.setdefault("created_at", "2026-01-01T00:00:00+00:00")
        row["updated_at"] = "2026-06-01T00:00:00+00:00"
        self.profiles[owner_sub] = row
        return row

    def delete_profile(self, owner_sub):
        self._maybe_fail("delete_profile")
        return self.profiles.pop(owner_sub, None) is not None

    def export_user_data(self, owner_sub):
        self._maybe_fail("export_user_data")
        return {
            "profile": self.profiles.get(owner_sub),
            "transactions": [
                t for owner, t in self.transactions.values() if owner == owner_sub
            ],
        }

    def delete_user_data(self, owner_sub):
        self._maybe_fail("delete_user_data")
        txns = [k for k, (o, _) in self.transactions.items() if o == owner_sub]
        for k in txns:
            del self.transactions[k]
        batches = [b for b in self.batches if b["owner_sub"] == owner_sub]
        for b in batches:
            self.batches.remove(b)
        profile = self.profiles.pop(owner_sub, None) is not None
        return {
            "profile_deleted": profile,
            "transactions_deleted": len(txns),
            "batches_deleted": len(batches),
        }



# ── fixtures ─────────────────────────────────────────────────────────────────


def _client(auth_env, make_auth_headers, store, sub=USER_A):
    """Authenticated client with ``store`` installed after startup.

    ``DATABASE_URL`` is absent in tests so the lifespan sets
    ``app.state.store = None``; the fake (or ``None``) is installed to control
    the persistence behaviour under test.
    """
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        c.headers.update(make_auth_headers(sub=sub))
        yield c


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def client(auth_env, make_auth_headers, store):
    yield from _client(auth_env, make_auth_headers, store)


@pytest.fixture
def client_no_db(auth_env, make_auth_headers):
    yield from _client(auth_env, make_auth_headers, None)


@pytest.fixture
def client_b(auth_env, make_auth_headers, store):
    yield from _client(auth_env, make_auth_headers, store, sub=USER_B)


def _ingest_two(client):
    body = _csv(
        _row(),
        _row(amount="-250.0", merchant_name="Bistro One", merchant_mcc="5812",
             timestamp="2026-03-06T19:30:00"),
    )
    r = client.post(
        "/consumer/transactions/ingest-csv", content=body.encode(),
        headers=_CSV_HEADERS,
    )
    assert r.status_code == 200, r.text
    return r.json()


# ── STEP 12F: ingest persists user-owned rows ────────────────────────────────


def test_ingest_persists_rows_owned_by_jwt_subject(client, store):
    """(1) persisted rows are keyed to principal.sub, not CSV user_id."""
    out = _ingest_two(client)
    assert out["persistence"]["enabled"] is True
    assert out["persistence"]["persisted_rows"] == 2
    assert out["persistence"]["duplicate_rows_skipped"] == 0
    assert len(store.batches) == 1
    assert store.batches[0]["owner_sub"] == USER_A
    assert all(owner == USER_A for owner, _ in store.transactions.values())


def test_csv_user_id_cannot_override_owner(client, store):
    """(2) rows claiming a different user_id are 403'd and never persisted."""
    body = _csv(_row(user_id=USER_B))
    r = client.post(
        "/consumer/transactions/ingest-csv", content=body.encode(),
        headers=_CSV_HEADERS,
    )
    assert r.status_code == 403
    assert store.transactions == {} and store.batches == []


def test_reingest_same_csv_deduplicates(client, store):
    """Accidental repeated upload must not silently duplicate rows (12F)."""
    _ingest_two(client)
    out = _ingest_two(client)
    assert out["persistence"]["persisted_rows"] == 0
    assert out["persistence"]["duplicate_rows_skipped"] == 2
    assert len(store.transactions) == 2


def test_list_returns_only_own_transactions(client, client_b, store):
    """(3)/(5) user A's list never contains user B's rows and vice versa."""
    _ingest_two(client)
    rows_b = client_b.get("/consumer/transactions").json()["transactions"]
    rows_a = client.get("/consumer/transactions").json()["transactions"]
    assert rows_b == []
    assert len(rows_a) == 2
    assert all(t["merchant_name"] in {"FreshMart", "Bistro One"} for t in rows_a)


def test_delete_transaction_scoped_to_owner(client, client_b, store):
    """(4) user B cannot delete user A's transaction — 404, row survives."""
    _ingest_two(client)
    txn_id = next(iter(store.transactions))
    r = client_b.delete(f"/consumer/transactions/{txn_id}")
    assert r.status_code == 404
    assert len(store.transactions) == 2

    r = client.delete(f"/consumer/transactions/{txn_id}")
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert len(store.transactions) == 1


def test_unknown_transaction_id_is_404(client):
    r = client.delete("/consumer/transactions/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


# ── failure modes (fail closed, no leakage) ──────────────────────────────────


def test_db_failure_on_ingest_returns_generic_503(client, store):
    """(8) PersistenceUnavailable → generic 503; no partial success reported."""
    store.fail_methods.add("create_ingest_batch")
    body = _csv(_row())
    r = client.post(
        "/consumer/transactions/ingest-csv", content=body.encode(),
        headers=_CSV_HEADERS,
    )
    assert r.status_code == 503
    assert DETAIL_503 in r.text
    assert store.transactions == {}  # nothing half-persisted
    assert "INSERT" not in r.text and "postgres" not in r.text.lower()


def test_db_failure_on_list_returns_generic_503(client, store):
    store.fail_methods.add("list_transactions")
    r = client.get("/consumer/transactions")
    assert r.status_code == 503
    assert "SELECT" not in r.text


def test_persistence_not_configured_returns_503(client_no_db):
    """(9) no DATABASE_URL → user-data routes fail closed with generic 503."""
    r = client_no_db.get("/consumer/transactions")
    assert r.status_code == 503
    assert r.json()["detail"] == DETAIL_NOT_CONFIGURED


def test_ingest_response_contract_additive(client_no_db):
    """Without persistence the ingest contract is unchanged except the
    additive ``persistence`` block (enabled=False)."""
    out = _ingest_two(client_no_db)
    assert out["persistence"] == {"enabled": False}
    assert out["classified_count"] == 2
