"""
Endpoint-level tests for POST /consumer/transactions/ingest-csv (CSV adapter).

The endpoint takes the CSV text as the RAW request body (Content-Type:
text/csv) — no multipart dependency. It reuses validate_and_parse_rows()
for validation and the existing /v1/classify inference path per row, so
these tests exercise the real artifacts and the real classifier.

The final test proves the full MVP handoff:
CSV → ingest → classified transactions → /consumer/advise → DecisionResult.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.routes import ingest as ingest_module

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

CSV_HEADER = "user_id,timestamp,amount,currency,merchant_name,merchant_mcc,account_type,channel"


def _row(**over) -> str:
    base = {
        "user_id": "u-csv",
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


_HEADERS = {"Content-Type": "text/csv; charset=utf-8"}  # params must be stripped


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


# ── happy path ───────────────────────────────────────────────────────────────


def test_valid_csv_with_multiple_transactions(client):
    body = _csv(
        _row(),
        _row(amount="-250.0", merchant_name="Bistro One", merchant_mcc="5812",
             timestamp="2026-03-06T19:30:00"),
    )
    r = client.post("/consumer/transactions/ingest-csv", content=body.encode(), headers=_HEADERS)
    assert r.status_code == 200
    out = r.json()
    assert out["total_rows"] == 2
    assert out["valid_rows"] == 2
    assert out["invalid_rows"] == 0
    assert out["classified_count"] == 2
    assert out["rejected"]["issues"] == []
    for item in out["classified"]:
        # classified output contains the required transaction + prediction fields
        for key in ("user_id", "timestamp", "amount", "currency", "merchant_name",
                    "merchant_mcc", "account_type", "channel",
                    "category_l1", "category_l2", "confidence"):
            assert key in item
        # confidence preserved from the classifier's own prediction
        assert isinstance(item["confidence"], float)
        assert 0.0 <= item["confidence"] <= 1.0
        assert item["category_l2"]  # classified to some category


def test_transactions_processed_chronologically(client):
    body = _csv(
        _row(timestamp="2026-03-08T10:00:00"),   # later row first in the file
        _row(amount="-50.0", timestamp="2026-03-05T09:00:00"),
    )
    r = client.post("/consumer/transactions/ingest-csv", content=body.encode(), headers=_HEADERS)
    assert r.status_code == 200
    ts = [item["timestamp"] for item in r.json()["classified"]]
    assert ts == sorted(ts)  # 2026-03-05 before 2026-03-08 regardless of file order


def test_x_filename_header_allows_any_content_type(client):
    r = client.post(
        "/consumer/transactions/ingest-csv",
        content=_csv(_row()).encode(),
        headers={"Content-Type": "application/octet-stream", "X-Filename": "export.csv"},
    )
    assert r.status_code == 200


# ── mixed / invalid validation ───────────────────────────────────────────────


def test_mixed_valid_and_invalid_rows(client):
    body = _csv(
        _row(),  # row 0: valid
        _row(amount="abc", merchant_name="Broken Row", merchant_mcc="5411"),  # row 1: invalid
    )
    r = client.post("/consumer/transactions/ingest-csv", content=body.encode(), headers=_HEADERS)
    assert r.status_code == 200
    out = r.json()
    assert out["total_rows"] == 2
    assert out["valid_rows"] == 1
    assert out["invalid_rows"] == 1
    assert out["classified_count"] == 1
    issues = out["rejected"]["issues"]
    assert len(issues) >= 1
    # validation errors carry row information
    assert issues[0]["row_index"] == 1
    assert issues[0]["field"]
    assert issues[0]["issue"]


def test_all_invalid_csv_returns_422_with_report(client):
    body = _csv(
        _row(amount="abc"),
        _row(amount="xyz", merchant_name="Also Broken"),
    )
    r = client.post("/consumer/transactions/ingest-csv", content=body.encode(), headers=_HEADERS)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "message" in detail and "report" in detail
    assert detail["report"]["invalid_rows"] == 2
    assert detail["report"]["issues"]


def test_malformed_csv_with_no_data_rows(client):
    r = client.post(
        "/consumer/transactions/ingest-csv",
        content=b"this is not a csv at all",
        headers=_HEADERS,
    )
    assert r.status_code == 422
    assert "no data rows" in r.json()["detail"]


def test_non_csv_upload_rejected(client):
    r = client.post(
        "/consumer/transactions/ingest-csv",
        content=b'{"not": "a csv"}',
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 415


def test_oversized_upload_rejected(client, monkeypatch):
    monkeypatch.setattr(ingest_module, "MAX_UPLOAD_BYTES", 10)
    r = client.post(
        "/consumer/transactions/ingest-csv",
        content=(_csv(_row()) * 5).encode(),  # > 10 bytes
        headers=_HEADERS,
    )
    assert r.status_code == 413
    assert "limit" in r.json()["detail"]


def test_excessive_row_count_rejected(client, monkeypatch):
    monkeypatch.setattr(ingest_module, "MAX_ROWS", 1)
    body = _csv(_row(), _row(timestamp="2026-03-06T12:00:00"))
    r = client.post("/consumer/transactions/ingest-csv", content=body.encode(), headers=_HEADERS)
    assert r.status_code == 413
    assert "Too many rows" in r.json()["detail"]


# ── full MVP integration: CSV → ingest → advise ─────────────────────────────


def test_csv_ingest_feeds_advise_endpoint(client):
    """CSV → ingestion → classification → /consumer/advise → DecisionResult."""
    body = _csv(
        _row(amount="5000.0", merchant_name="Payroll - Acme Corp", merchant_mcc="0",
             channel="TRANSFER", timestamp="2026-03-01T09:00:00"),
        _row(amount="-300.0", merchant_name="FreshMart", merchant_mcc="5411",
             timestamp="2026-03-05T12:00:00"),
        _row(amount="-250.0", merchant_name="Bistro One", merchant_mcc="5812",
             timestamp="2026-03-06T19:30:00"),
    )
    ing = client.post("/consumer/transactions/ingest-csv", content=body.encode(), headers=_HEADERS)
    assert ing.status_code == 200
    classified = ing.json()["classified"]
    assert len(classified) == 3

    adv = client.post("/consumer/advise", json={"transactions": classified})
    assert adv.status_code == 200
    decision = adv.json()
    # existing DecisionResult contract, produced from CSV-originated data
    for key in ("decision_id", "decision_type", "recommendation", "reasoning",
                "supporting_metrics", "warnings"):
        assert key in decision

