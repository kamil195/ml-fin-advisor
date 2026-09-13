"""
Endpoint-level tests for POST /consumer/advise (DecisionEngine adapter).

Covers the thin-adapter contract: classified payload → confidence gate →
DecisionEngine.advise() → existing DecisionResult response.

Hand-computed baseline used throughout (all amounts monthly):
    income +5000 | rent -1500 (protected) | groceries -300 | restaurants -300
    → baseline: income 5000, expenses 2100, savings 2900.
    Cuttable LP capacity = 30% of groceries+restaurants = 180.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.routes.advise import CONFIDENCE_GATE_THRESHOLD
from src.serving.routes.advise import ClassifiedTransactionIn, _to_gated_transaction

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _txn(**over) -> dict:
    """A high-confidence (0.95) debit/credit with overridable fields."""
    base = {
        "user_id": "user-123",  # authenticated JWT sub (AUTH STEP 3)
        "timestamp": "2026-03-05T12:00:00",
        "amount": -100.0,
        "merchant_name": "Merchant",
        "merchant_mcc": 5411,
        "account_type": "CHECKING",
        "channel": "POS",
    }
    base.update(over)
    return base


def _history() -> list[dict]:
    return [
        _txn(
            amount=5000.0,
            merchant_name="Payroll - Acme Corp",
            merchant_mcc=0,
            channel="TRANSFER",
            category_l1="FINANCIAL",
            category_l2="Income",
            confidence=0.95,
        ),
        _txn(
            amount=-1500.0,
            merchant_name="LandlordCo",
            merchant_mcc=6513,
            channel="TRANSFER",
            category_l1="HOUSING",
            category_l2="Rent/Mortgage",
            confidence=0.95,
        ),
        _txn(
            amount=-300.0,
            merchant_name="FreshMart",
            category_l1="FOOD & DINING",
            category_l2="Groceries",
            confidence=0.95,
        ),
        _txn(
            amount=-300.0,
            merchant_name="Bistro One",
            merchant_mcc=5812,
            category_l1="FOOD & DINING",
            category_l2="Restaurants",
            confidence=0.95,
        ),
    ]


def _body(**over) -> dict:
    body = {"transactions": _history()}
    body.update(over)
    return body


@pytest.fixture
def client(auth_env, make_auth_headers):
    """Authenticated test client (AUTH STEP 2).

    ``/consumer/advise`` is protected by ``require_auth``; the shared
    ``auth_env`` fixture (conftest.py) sets ``SUPABASE_URL`` and routes JWKS
    resolution to the local test key, so no real Supabase call is made.
    """
    with TestClient(create_app()) as c:
        c.headers.update(make_auth_headers())
        yield c


# ── happy path / contract ────────────────────────────────────────────────────


def test_valid_classified_transactions_reach_decision_engine(client):
    r = client.post("/consumer/advise", json=_body())
    assert r.status_code == 200
    out = r.json()
    for key in (
        "decision_id",
        "decision_type",
        "recommendation",
        "reasoning",
        "supporting_metrics",
        "warnings",
    ):
        assert key in out
    assert out["confidence"] is None  # deterministic rule-based decision
    m = out["supporting_metrics"]
    assert m["baseline.monthly_income"] == pytest.approx(5000.0)
    assert m["baseline.total_expenses"] == pytest.approx(2100.0)
    assert m["baseline.monthly_savings"] == pytest.approx(2900.0)


def test_decision_result_contract_is_existing_schema(client):
    r = client.post("/consumer/advise", json=_body())
    assert r.status_code == 200
    out = r.json()
    assert set(out) == {
        "decision_id",
        "scenario_id",
        "decision_type",
        "recommendation",
        "reasoning",
        "supporting_metrics",
        "confidence",
        "alternatives",
        "warnings",
        "generated_at",
    }


# ── confidence gating through the API ────────────────────────────────────────


def test_high_confidence_categories_are_cuttable(client):
    """Gap 180 == LP capacity of the two flexible categories → fully achieved."""
    body = _body(params={"scenario_id": "s-flex", "savings_target": 3080.0})
    r = client.post("/consumer/advise", json=body)
    assert r.status_code == 200
    assert r.json()["supporting_metrics"]["scenario.monthly_savings"] == pytest.approx(
        3080.0
    )


# ── confidence gating: low-confidence ordinary transactions ──────────────────


def test_low_confidence_ordinary_becomes_uncategorized(client):
    """Ordinary low-confidence debit: category cleared, amount preserved, counted
    in expenses, but never a scenario cut target."""
    p = ClassifiedTransactionIn(
        **_txn(category_l1="FOOD & DINING", category_l2="Restaurants", confidence=0.50)
    )
    t = _to_gated_transaction(p)
    assert t.category_l1 is None and t.category_l2 is None
    assert t.amount == pytest.approx(-100.0)
    assert t.channel.value == "POS"
    assert t.merchant_name == "Merchant"

    # API-level: counted in total expenses (2100 + 200 = 2300) ...
    low_conf = _txn(
        amount=-200.0,
        merchant_name="Bistro Two",
        category_l1="FOOD & DINING",
        category_l2="Restaurants",
        confidence=0.50,
    )
    counted = _body()
    counted["transactions"].append(low_conf)
    r = client.post("/consumer/advise", json=counted)
    assert r.status_code == 200
    m = r.json()["supporting_metrics"]
    assert m["baseline.total_expenses"] == pytest.approx(2300.0)

    # ... but NOT cuttable: flexible pool stays 600 → max squeeze 300
    # (gap 800 from baseline savings 2700; if the 200 were wrongly flexible,
    # the pool would be 800 → squeeze 400 → savings 3100).
    gated = _body(params={"scenario_id": "s-unc", "savings_target": 3500.0})
    gated["transactions"].append(low_conf)
    r2 = client.post("/consumer/advise", json=gated)
    assert r2.status_code == 200
    assert r2.json()["supporting_metrics"]["scenario.monthly_savings"] == pytest.approx(
        3000.0
    )


def test_low_confidence_authoritative_rules_preserved(client):
    """Input-side signals (transfer / savings / income credits) survive the gate."""
    body = _body()
    body["transactions"][0]["confidence"] = 0.55  # income credit, low confidence
    body["transactions"].append(
        _txn(amount=-400.0, merchant_name="Internal Move", channel="TRANSFER", confidence=0.30)
    )
    body["transactions"].append(
        _txn(
            amount=-500.0,
            merchant_name="Broker Move",
            category_l1="FINANCIAL",
            category_l2="Savings & Investments",
            confidence=0.40,
        )
    )
    r = client.post("/consumer/advise", json=body)
    assert r.status_code == 200
    m = r.json()["supporting_metrics"]
    assert m["baseline.monthly_income"] == pytest.approx(5000.0)
    assert m["baseline.total_expenses"] == pytest.approx(2100.0)


# ── validation ────────────────────────────────────────────────────────────────


def test_missing_history_rejected(client):
    r = client.post("/consumer/advise", json={"transactions": []})
    assert r.status_code == 422


def test_malformed_payloads_rejected(client):
    # confidence out of range
    r1 = client.post(
        "/consumer/advise", json=_body(transactions=[_txn(confidence=1.5)])
    )
    assert r1.status_code == 422
    # category-hierarchy mismatch must 422 (mirrored validator), not 500
    r2 = client.post(
        "/consumer/advise",
        json=_body(
            transactions=[_txn(category_l1="HOUSING", category_l2="Groceries")]
        ),
    )
    assert r2.status_code == 422
    # missing required field
    bad = _txn()
    del bad["merchant_name"]
    r3 = client.post("/consumer/advise", json=_body(transactions=[bad]))
    assert r3.status_code == 422


# ── protection & coexistence ─────────────────────────────────────────────────


def test_protected_categories_protected_through_api(client):
    """Target 3500 (gap 600) exceeds flexible capacity 300 → partial, rent intact."""
    body = _body(params={"scenario_id": "s-prot", "savings_target": 3500.0})
    r = client.post("/consumer/advise", json=body)
    assert r.status_code == 200
    out = r.json()
    m = out["supporting_metrics"]
    assert m["scenario.monthly_savings"] == pytest.approx(3200.0)
    assert m["scenario.total_expenses"] == pytest.approx(1800.0)  # 2100 - 300 flexible
    assert any("essential" in w for w in out["warnings"])


def test_existing_routes_still_work(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200

