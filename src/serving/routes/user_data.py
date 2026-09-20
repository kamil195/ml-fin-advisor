"""
User-owned persistence endpoints (STEP 12).

    GET    /consumer/profile
    PUT    /consumer/profile
    GET    /consumer/transactions
    DELETE /consumer/transactions/{transaction_id}
    GET    /consumer/data/export
    DELETE /consumer/data

Identity & ownership:

* Every operation is scoped to ``principal.sub`` — the verified JWT subject.
  No request field (body, query, path) can specify or override the owner.
* Foreign *identity* in a request body → 403 (existing project convention).
* A foreign or unknown *resource id* → 404 (existence is not disclosed to
  callers who guess ids); this is the resource-level counterpart of the
  project's 403 identity rule, applied consistently on every route here.

Failure model:

* Persistence not configured (no ``DATABASE_URL``) → generic 503 (deployment
  state); database unreachable / statement failure → generic 503 (fail
  closed). No SQL text, values, table names, or DSN reach the client.

Cache coherency (STEP 12J): every mutation purges the *calling user's* derived
cache entries (budgets / forecasts / features / explanations namespaces) via
``CacheClient.purge_user`` — never a global invalidation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.serving.auth import AuthPrincipal, require_auth
from src.serving.persistence import (
    DETAIL_503,
    DETAIL_NOT_CONFIGURED,
    PersistenceUnavailable,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consumer", tags=["User Data"])

#: Maximum page size for transaction listing.
MAX_TRANSACTIONS_LIMIT = 500

_TX_404 = "Transaction not found."


class FinancialProfileIn(BaseModel):
    """The financial-profile inputs the product uses today (nothing speculative).

    Mirrors the fields currently supplied by callers: ``income`` and
    ``savings_target`` (budget optimiser) and ``liquid_buffer``,
    ``total_debt``, ``monthly_debt_payments`` (advise / financial profile).
    """

    income: float = Field(..., ge=0, description="Monthly gross income (>= 0).")
    savings_target: float | None = Field(default=None, ge=0)
    liquid_buffer: float | None = Field(default=None, ge=0)
    total_debt: float | None = Field(default=None, ge=0)
    monthly_debt_payments: float | None = Field(default=None, ge=0)


def _require_store(request: Request) -> Any:
    """Return the app's persistence store or raise a generic 503."""
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail=DETAIL_NOT_CONFIGURED)
    return store


def _call_store(store: Any, method: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run a store operation; translate failures to a generic, safe 503."""
    try:
        return getattr(store, method)(*args, **kwargs)
    except PersistenceUnavailable:
        raise HTTPException(status_code=503, detail=DETAIL_503) from None


def _purge_user_caches(request: Request, sub: str) -> bool:
    """Purge the calling user's derived cache entries (best-effort, scoped).

    Returns ``True`` when the purge ran cleanly. Failure is logged by *type
    only* (no identity) and reported honestly in responses; leftover entries
    are user-scoped and TTL-bounded.
    """
    cache = getattr(request.app.state, "cache", None)
    if cache is None:
        return False
    try:
        cache.purge_user(sub)
    except Exception as exc:  # noqa: BLE001 — cache is derived state; never fatal
        logger.warning("user cache purge failed (type: %s)", type(exc).__name__)
        return False
    return True


def _num(value: Any) -> float | None:
    """JSON-safe money value (Postgres NUMERIC arrives as Decimal)."""
    return None if value is None else round(float(value), 2)


def _profile_out(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Public profile shape: product fields + timestamps, no owner_sub echo."""
    if row is None:
        return None
    return {
        "income": _num(row.get("income")),
        "savings_target": _num(row.get("savings_target")),
        "liquid_buffer": _num(row.get("liquid_buffer")),
        "total_debt": _num(row.get("total_debt")),
        "monthly_debt_payments": _num(row.get("monthly_debt_payments")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _transaction_out(row: dict[str, Any]) -> dict[str, Any]:
    """Public transaction shape. ``owner_sub`` is intentionally not echoed."""
    return {
        "id": str(row.get("id")),
        "occurred_at": row.get("occurred_at"),
        "amount": round(float(row.get("amount", 0.0)), 2),
        "currency": row.get("currency"),
        "merchant_name": row.get("merchant_name"),
        "merchant_mcc": row.get("merchant_mcc"),
        "account_type": row.get("account_type"),
        "channel": row.get("channel"),
        "location_city": row.get("location_city"),
        "location_country": row.get("location_country"),
        "raw_description": row.get("raw_description"),
        "is_pending": row.get("is_pending"),
        "category_l1": row.get("category_l1"),
        "category_l2": row.get("category_l2"),
        "confidence": row.get("confidence"),
        "source": row.get("source"),
        "created_at": row.get("created_at"),
    }


# ── financial profile ────────────────────────────────────────────────────────


@router.get("/profile")
def get_profile(
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
) -> dict[str, Any]:
    """Return the caller's saved financial profile (404 when none saved)."""
    store = _require_store(request)
    row = _call_store(store, "get_profile", principal.sub)
    if row is None:
        raise HTTPException(status_code=404, detail="No financial profile saved yet.")
    return {"profile": _profile_out(row)}


@router.put("/profile")
def upsert_profile(
    body: FinancialProfileIn,
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
) -> dict[str, Any]:
    """Create or update the caller's financial profile (owner = principal.sub)."""
    store = _require_store(request)
    row = _call_store(store, "upsert_profile", principal.sub, body.model_dump())
    cache_purged = _purge_user_caches(request, principal.sub)
    return {"profile": _profile_out(row), "cache_purged": cache_purged}


@router.get("/transactions")
def list_transactions(
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
    limit: int = Query(default=200, ge=1, le=MAX_TRANSACTIONS_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List the caller's persisted transactions, newest first (owner-scoped)."""
    store = _require_store(request)
    rows = _call_store(store, "list_transactions", principal.sub, limit, offset)
    return {"transactions": [_transaction_out(r) for r in rows], "count": len(rows)}


@router.delete("/transactions/{transaction_id}")
def delete_transaction(
    transaction_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
) -> dict[str, Any]:
    """Delete one of the caller's transactions (foreign/unknown id → 404)."""
    store = _require_store(request)
    deleted = _call_store(store, "delete_transaction", principal.sub, transaction_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=_TX_404)
    cache_purged = _purge_user_caches(request, principal.sub)
    return {"deleted": True, "cache_purged": cache_purged}


@router.get("/data/export")
def export_user_data(
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
) -> dict[str, Any]:
    """Export ALL of the caller's persisted Planwisely data as JSON.

    Contains only user-owned profile + transaction rows — no tokens, no auth
    secrets, no cache state, no other user's data.
    """
    store = _require_store(request)
    result = _call_store(store, "export_user_data", principal.sub)
    profile_row = result["profile"]
    return {
        "format": "planwisely.export.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": principal.sub,
        "profile": _profile_out(profile_row),
        "transactions": [_transaction_out(r) for r in result["transactions"]],
        "counts": {
            "profile": 1 if profile_row is not None else 0,
            "transactions": len(result["transactions"]),
        },
    }


@router.delete("/data")
def delete_user_data(
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
) -> dict[str, Any]:
    """Delete ALL of the caller's persisted Planwisely financial data.

    Fails closed: if the deletion cannot be completed transactionally, a
    generic 503 is returned and success is never falsely reported. This
    deletes the caller's *financial data* — it does NOT delete the Supabase
    Auth account itself (that is a separate Supabase operation).
    """
    store = _require_store(request)
    result = _call_store(store, "delete_user_data", principal.sub)
    cache_purged = _purge_user_caches(request, principal.sub)
    return {
        "message": "Planwisely financial data deleted.",
        "auth_account_deleted": False,
        "profile_deleted": result["profile_deleted"],
        "transactions_deleted": result["transactions_deleted"],
        "batches_deleted": result["batches_deleted"],
        "cache_purged": cache_purged,
    }
