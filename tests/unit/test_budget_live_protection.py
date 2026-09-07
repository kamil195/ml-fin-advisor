"""
Regression tests: /consumer/budget/live must honour the canonical
hard-protection policy regardless of the caller's category vocabulary.

The live route accepts Planwisely's 10-class bucket names while the canonical
cut-protection policy (``HARD_PROTECTED_CATEGORIES``, enforced by
``BudgetOptimiser`` floors) is expressed in the 30-class taxonomy. Before the
fix, a Planwisely-labelled protected obligation (e.g. "Housing" — Rent/Mortgage
money) had no optimiser floor and could be reduced by the "insufficient cuts"
headroom fallback whenever the savings target exceeded discretionary capacity.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.serving.routes.live import router

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

INCOME = 3000.0
TARGET = 2500.0  # envelope = 500 → every scenario below forces real cuts


@pytest.fixture
def client():
    # Minimal app: budget/live builds its own BudgetOptimiser and needs no
    # model artifacts, so the full serving app (lifespan model loading) is
    # deliberately not used here.
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _post(client, spend, income=INCOME, savings_target=TARGET):
    """spend: list of (category | None, negative amount) tuples."""
    transactions = [
        {"date": "2026-03-05", "merchant": "M", "amount": amt, "category": cat}
        for cat, amt in spend
    ]
    return client.post(
        "/consumer/budget/live",
        json={
            "transactions": transactions,
            "income": income,
            "savings_target": savings_target,
        },
    )


def _alloc(body, category):
    for a in body["allocations"]:
        if a["category"] == category:
            return a
    return None


def _assert_fully_protected(body, category):
    a = _alloc(body, category)
    assert a is not None, f"{category} missing from allocations: {body['allocations']}"
    assert a["cut_amount"] == pytest.approx(0.0), a
    assert a["cut_pct"] == pytest.approx(0.0), a
    assert a["recommended_budget"] == pytest.approx(a["current_spend"]), a
    assert a["is_discretionary"] is False, a


# ── 1–6. The six hard-protected categories can never be cut ──────────────────


def test_rent_mortgage_cannot_be_cut(client):
    resp = _post(client, [("Rent/Mortgage", -2000.0), ("Food & Dining", -500.0)])
    assert resp.status_code == 200, resp.text
    _assert_fully_protected(resp.json(), "Rent/Mortgage")


def test_utilities_cannot_be_cut(client):
    resp = _post(client, [("Utilities", -800.0), ("Shopping", -600.0)])
    assert resp.status_code == 200, resp.text
    _assert_fully_protected(resp.json(), "Utilities")


def test_home_insurance_cannot_be_cut(client):
    resp = _post(client, [("Home Insurance", -300.0), ("Entertainment", -400.0)])
    assert resp.status_code == 200, resp.text
    _assert_fully_protected(resp.json(), "Home Insurance")


def test_insurance_premiums_cannot_be_cut(client):
    resp = _post(client, [("Insurance Premiums", -250.0), ("Food & Dining", -500.0)])
    assert resp.status_code == 200, resp.text
    _assert_fully_protected(resp.json(), "Insurance Premiums")


def test_loan_payments_cannot_be_cut(client):
    resp = _post(client, [("Loan Payments", -400.0), ("Shopping", -500.0)])
    assert resp.status_code == 200, resp.text
    _assert_fully_protected(resp.json(), "Loan Payments")


def test_taxes_cannot_be_cut(client):
    resp = _post(client, [("Taxes", -350.0), ("Food & Dining", -500.0)])
    assert resp.status_code == 200, resp.text
    _assert_fully_protected(resp.json(), "Taxes")


# ── Planwisely-vocabulary bypass regressions (the actual fix) ─────────────────


def test_planwisely_housing_bucket_cannot_be_cut(client):
    """'Housing' carries Rent/Mortgage + Home Insurance money in the 10-class
    vocabulary. Before the fix this bucket had no optimiser floor and lost up
    to ~50% of its baseline once the target exceeded discretionary capacity.
    """
    resp = _post(client, [("Housing", -2000.0), ("Food & Dining", -500.0)])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_fully_protected(body, "Housing")
    # The discretionary bucket absorbs the cuts instead.
    dining = _alloc(body, "Food & Dining")
    assert dining is not None and dining["cut_amount"] > 0.0, dining


def test_planwisely_health_and_other_buckets_cannot_be_cut(client):
    resp = _post(
        client,
        [("Health", -300.0), ("Other", -200.0), ("Shopping", -500.0)],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_fully_protected(body, "Health")
    _assert_fully_protected(body, "Other")
    shopping = _alloc(body, "Shopping")
    assert shopping is not None and shopping["cut_amount"] > 0.0, shopping


# ── 7. Normal discretionary categories can still be cut ───────────────────────


def test_discretionary_categories_still_cut(client):
    resp = _post(client, [("Shopping", -600.0), ("Food & Dining", -400.0)])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    shopping = _alloc(body, "Shopping")
    dining = _alloc(body, "Food & Dining")
    assert shopping is not None and shopping["cut_amount"] > 0.0, shopping
    assert dining is not None and dining["cut_amount"] > 0.0, dining


def test_discretionary_cut_while_protected_intact(client):
    resp = _post(client, [("Rent/Mortgage", -2000.0), ("Food & Dining", -500.0)])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    dining = _alloc(body, "Food & Dining")
    assert dining is not None and dining["cut_amount"] > 0.0, dining
    assert dining["recommended_budget"] < dining["current_spend"], dining
    _assert_fully_protected(body, "Rent/Mortgage")


# ── 8. Uncategorized is neither protected nor a cut target ────────────────────


def test_uncategorized_excluded_from_optimisation(client):
    resp = _post(
        client,
        [(None, -300.0), ("Uncategorized", -200.0), ("Food & Dining", -500.0)],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    categories = {a["category"] for a in body["allocations"]}
    assert categories == {"Food & Dining"}, categories


def test_all_uncategorized_returns_422(client):
    resp = _post(client, [(None, -100.0), ("Uncategorized", -50.0)])
    assert resp.status_code == 422, resp.text

