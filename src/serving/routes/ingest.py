"""CSV transaction ingestion — thin HTTP adapter.

Reuses the EXISTING pipeline pieces; implements no new validation or
classification logic:

* ``validate_and_parse_rows`` (src/data/ingestion.py) — row validation and
  Transaction construction (accepts in-memory dicts, so the upload is decoded
  in memory; the training-side parquet pipeline is NOT used).
* ``classify_transaction`` (src/serving/routes/classify.py) — the exact
  single-transaction serving path (same artifacts, same features, same
  confidence semantics).

Batch behaviour: valid rows are classified in chronological order so the
serving path sees a deterministic sequence. Per-row classification uses the
same per-transaction feature semantics as /v1/classify (single-txn), which is
the current serving contract — this adapter does not change it.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.data.ingestion import validate_and_parse_rows
from src.data.models import Transaction, ValidationReport
from src.serving.auth import AuthPrincipal, require_auth
from src.serving.persistence import DETAIL_503, PersistenceUnavailable
from src.serving.routes.classify import (
    ClassifyRequest,
    ClassifyResponse,
    classify_transaction_with_history,
)

router = APIRouter(prefix="/consumer", tags=["CSV Ingestion"])

MAX_UPLOAD_BYTES = 2_000_000  # 2 MB
MAX_ROWS = 1_000
_ALLOWED_CONTENT_TYPES = {"text/csv", "application/csv", "application/vnd.ms-excel"}


class IngestCsvResponse(BaseModel):
    """Minimal response contract for CSV ingestion."""

    total_rows: int
    valid_rows: int
    invalid_rows: int
    classified_count: int
    rejected: ValidationReport
    classified: list[dict[str, Any]]
    # STEP 12F: persistence outcome for this upload (additive field).
    # enabled=False when the deployment has no DATABASE_URL configured.
    persistence: dict[str, Any] = {}


def _looks_like_csv(filename: str | None, content_type: str | None) -> bool:
    if filename and filename.lower().endswith(".csv"):
        return True
    # Strip parameters, e.g. "text/csv; charset=utf-8".
    base = (content_type or "").split(";")[0].strip().lower()
    return base in _ALLOWED_CONTENT_TYPES


def _classify_item(t: Transaction, res: ClassifyResponse) -> dict[str, Any]:
    """Merge the transaction with its classification (confidence preserved)."""
    item = t.model_dump(mode="json")
    item.update(
        category_l1=res.category_l1,
        category_l2=res.category_l2,
        confidence=res.confidence,
    )
    return item


@router.post("/transactions/ingest-csv", response_model=IngestCsvResponse)
async def ingest_transactions_csv(
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
) -> IngestCsvResponse:
    """Raw-body CSV upload (no multipart dependency required).

    Send the CSV text as the request body with ``Content-Type: text/csv``;
    optionally set ``X-Filename`` (used for extension validation, e.g. by
    browser uploads).
    """
    filename = request.headers.get("x-filename")
    content_type = request.headers.get("content-type")
    if not _looks_like_csv(filename, content_type):
        raise HTTPException(status_code=415, detail="Only CSV files are accepted")

    raw = await request.body()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV too large ({len(raw)} bytes); limit is {MAX_UPLOAD_BYTES} bytes",
        )

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8 text") from exc

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise HTTPException(status_code=422, detail="CSV contains no data rows")
    if len(rows) > MAX_ROWS:
        raise HTTPException(
            status_code=413, detail=f"Too many rows ({len(rows)}); limit is {MAX_ROWS}"
        )

    transactions, report = validate_and_parse_rows(rows)
    if not transactions:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All rows failed validation",
                "report": report.model_dump(mode="json"),
            },
        )

    # AUTH STEP 3: every transaction belongs to the authenticated subject.
    # A caller-provided CSV user identity is never trusted; any row claiming a
    # different owner is rejected outright (403) before further processing.
    for txn in transactions:
        if txn.user_id != principal.sub:
            raise HTTPException(
                status_code=403,
                detail="Forbidden: CSV contains transactions for another user.",
            )

    # Chronological order keeps batch processing deterministic. Per-user history
    # is accumulated as we iterate so each transaction is classified with the
    # strictly-prior transactions for that user — matching training semantics
    # (z-score, rolling spend, 24h count, etc. computed over prior rows only).
    ordered = sorted(transactions, key=lambda t: t.timestamp)
    classified: list[dict[str, Any]] = []
    persist_rows: list[dict[str, Any]] = []
    user_history: dict[str, list[Transaction]] = {}
    for t in ordered:
        uid = t.user_id
        prior = user_history.get(uid, [])
        res = await classify_transaction_with_history(
            ClassifyRequest(transaction=t), request, history=prior
        )
        classified.append(_classify_item(t, res))
        # STEP 12F: persistable row for the same classified transaction, built
        # from the validated Transaction (identity = principal.sub, enforced
        # above) — the CSV's user_id column is never used as authority.
        persist_rows.append({
            "occurred_at": t.timestamp,
            "amount": t.amount,
            "currency": t.currency,
            "merchant_name": t.merchant_name,
            "merchant_mcc": t.merchant_mcc,
            "account_type": getattr(t.account_type, "value", t.account_type),
            "channel": getattr(t.channel, "value", t.channel),
            "location_city": t.location_city,
            "location_country": getattr(t.location_country, "value", t.location_country),
            "raw_description": t.raw_description or "",
            "is_pending": bool(t.is_pending),
            "category_l1": getattr(res.category_l1, "value", res.category_l1),
            "category_l2": getattr(res.category_l2, "value", res.category_l2),
            "confidence": res.confidence,
            "source": "csv_ingest",
        })
        # Current transaction becomes part of history for later rows of this user.
        user_history.setdefault(uid, []).append(t)

    # ── STEP 12F: persist user-owned transactions (when configured) ────────
    store = getattr(request.app.state, "store", None)
    persistence_block: dict[str, Any] = {"enabled": store is not None}
    if store is not None:
        try:
            batch_id = store.create_ingest_batch(
                principal.sub, "csv_ingest", len(transactions)
            )
            persisted, skipped = store.insert_transactions(
                principal.sub, persist_rows, batch_id
            )
        except PersistenceUnavailable:
            # Fail closed: a persistence failure must not masquerade as a
            # successful ingest. Generic, safe 503 — no SQL/DSN details.
            raise HTTPException(status_code=503, detail=DETAIL_503) from None
        persistence_block.update(
            {
                "ingest_batch_id": batch_id,
                "persisted_rows": persisted,
                "duplicate_rows_skipped": skipped,
            }
        )

    return IngestCsvResponse(
        total_rows=len(rows),
        valid_rows=len(transactions),
        invalid_rows=report.invalid_rows,
        classified_count=len(classified),
        rejected=report,
        classified=classified,
        persistence=persistence_block,
    )
