"""
Unit tests for the FinancialProfile aggregation service
(``src.services.financial_profile``).

All examples are small, hand-computed, and deterministic. Windows are
chosen within a single calendar month so monthly values equal raw sums,
unless a test specifically exercises the month-count normalisation.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.models import Transaction  # noqa: E402
from src.services.financial_profile import (  # noqa: E402
    build_financial_profile,
    expense_ratios,
)
from src.utils.constants import AccountType, CategoryL2, Channel  # noqa: E402

UTC = timezone.utc


def _txn(
    amount: float,
    category: CategoryL2 | None,
    channel: Channel = Channel.POS,
    ts: datetime | None = None,
    currency: str = "USD",
    user_id: str = "u-1",
) -> Transaction:
    return Transaction(
        user_id=user_id,
        timestamp=ts or datetime(2026, 3, 5, tzinfo=UTC),
        amount=amount,
        merchant_name="Test Merchant",
        merchant_mcc=5411,
        account_type=AccountType.CHECKING,
        channel=channel,
        category_l2=category,
        currency=currency,
    )


def _standard_txns() -> list[Transaction]:
    """Income 5000; rent 1500 (fixed); groceries 450; discretionary 420."""
    return [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER,
             datetime(2026, 3, 1, tzinfo=UTC)),
        _txn(-1_500.0, CategoryL2.RENT_MORTGAGE, Channel.TRANSFER,
             datetime(2026, 3, 2, tzinfo=UTC)),
        _txn(-450.0, CategoryL2.GROCERIES, Channel.POS,
             datetime(2026, 3, 5, tzinfo=UTC)),
        _txn(-120.0, CategoryL2.RESTAURANTS, Channel.POS,
             datetime(2026, 3, 8, tzinfo=UTC)),
        _txn(-300.0, CategoryL2.ELECTRONICS, Channel.ONLINE,
             datetime(2026, 3, 12, tzinfo=UTC)),
    ]


# ── Core bucketing ──────────────────────────────────────────────────────────────


def test_income_and_expense_buckets():
    p = build_financial_profile(_standard_txns(), period="2026-03")

    assert p.monthly_income == 5_000.0
    assert p.fixed_expenses == 1_500.0
    # variable = groceries 450 + discretionary 420 (discretionary ⊆ variable)
    assert p.variable_expenses == 870.0
    assert p.discretionary_expenses == 420.0
    assert p.total_expenses == 2_370.0
    assert p.monthly_savings == 2_630.0
    assert p.savings_rate == pytest.approx(2_630.0 / 5_000.0)
    assert p.monthly_net_cash_flow == 2_630.0


def test_expense_ratios():
    ratios = expense_ratios(build_financial_profile(_standard_txns()))
    assert ratios["fixed"] == pytest.approx(1_500.0 / 2_370.0)
    assert ratios["variable"] == pytest.approx(870.0 / 2_370.0)
    assert ratios["discretionary"] == pytest.approx(420.0 / 2_370.0)


def test_recurring_channel_wins_over_discretionary_taxonomy():
    """A RECURRING subscription is a fixed obligation, not discretionary."""
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER),
        _txn(-15.0, CategoryL2.SUBSCRIPTIONS_STREAMING, Channel.RECURRING),
    ]
    p = build_financial_profile(txns)
    assert p.fixed_expenses == 15.0
    assert p.recurring_expenses == 15.0
    assert p.discretionary_expenses == 0.0
    assert p.variable_expenses == 0.0


def test_fixed_category_without_recurring_channel():
    """Rent paid by bank TRANSFER is still fixed via the category signal."""
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER),
        _txn(-1_500.0, CategoryL2.RENT_MORTGAGE, Channel.TRANSFER),
    ]
    p = build_financial_profile(txns)
    assert p.fixed_expenses == 1_500.0
    assert p.recurring_expenses == 0.0  # channel was TRANSFER, not RECURRING


def test_uncategorized_transactions_lump_into_variable():
    txns = [
        _txn(2_000.0, CategoryL2.INCOME, Channel.TRANSFER),
        _txn(-100.0, None, Channel.ATM),
    ]
    p = build_financial_profile(txns)
    assert p.variable_expenses == 100.0
    assert p.category_spending == {"Uncategorized": 100.0}


# ── Honesty rules: never invent income ─────────────────────────────────────────


def test_no_income_is_never_fabricated():
    """Debits only → income 0.0 (not spend × 1.3), savings_rate None."""
    txns = [_txn(-100.0, CategoryL2.GROCERIES)]
    p = build_financial_profile(txns)
    assert p.monthly_income == 0.0
    assert p.savings_rate is None
    assert p.monthly_savings == -100.0
    assert p.monthly_net_cash_flow == -100.0


def test_balance_sheet_fields_never_invented():
    p = build_financial_profile(_standard_txns())
    assert p.liquid_buffer is None
    assert p.months_of_buffer is None
    assert p.total_debt is None
    assert p.monthly_debt_payments is None

    p2 = build_financial_profile(
        _standard_txns(),
        liquid_buffer=9_000.0,
        total_debt=12_000.0,
        monthly_debt_payments=250.0,
    )
    assert p2.liquid_buffer == 9_000.0
    assert p2.total_debt == 12_000.0
    assert p2.monthly_debt_payments == 250.0
    assert p2.months_of_buffer == pytest.approx(9_000.0 / 2_370.0)


# ── Windowing & normalisation ──────────────────────────────────────────────────


def test_distinct_calendar_months_normalise_totals():
    """Same salary and rent in March AND April → 2 months → halves monthly."""
    txns = [
        _txn(3_000.0, CategoryL2.INCOME, Channel.TRANSFER,
             datetime(2026, 3, 1, tzinfo=UTC)),
        _txn(-1_000.0, CategoryL2.RENT_MORTGAGE, Channel.TRANSFER,
             datetime(2026, 3, 2, tzinfo=UTC)),
        _txn(3_000.0, CategoryL2.INCOME, Channel.TRANSFER,
             datetime(2026, 4, 1, tzinfo=UTC)),
        _txn(-1_000.0, CategoryL2.RENT_MORTGAGE, Channel.TRANSFER,
             datetime(2026, 4, 2, tzinfo=UTC)),
    ]
    p = build_financial_profile(txns)
    assert p.monthly_income == pytest.approx(3_000.0)
    assert p.fixed_expenses == pytest.approx(1_000.0)
    assert p.savings_rate == pytest.approx(2_000.0 / 3_000.0)


def test_observation_days_trailing_filter():
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER,
             datetime(2026, 3, 1, tzinfo=UTC)),
        _txn(-450.0, CategoryL2.GROCERIES, Channel.POS,
             datetime(2026, 3, 5, tzinfo=UTC)),
        _txn(-300.0, CategoryL2.ELECTRONICS, Channel.ONLINE,
             datetime(2026, 3, 28, tzinfo=UTC)),
    ]
    p = build_financial_profile(txns, observation_days=7)
    # Only the Mar-28 transaction survives the trailing 7-day window.
    assert p.total_expenses == 300.0
    assert p.monthly_income == 0.0  # the Mar-1 salary fell outside the window
    assert p.savings_rate is None


# ── Metadata, currency, determinism ────────────────────────────────────────────


def test_empty_transaction_list_represents_missing_data():
    p = build_financial_profile([], user_id="u-42", period="2026-03")
    assert p.user_id == "u-42"
    assert p.monthly_income == 0.0
    assert p.total_expenses == 0.0
    assert p.savings_rate is None
    assert p.category_spending == {}
    with pytest.raises(ValueError, match="user_id is required"):
        build_financial_profile([])


def test_user_id_defaults_from_transactions_and_currency_detection():
    txns = [
        _txn(1_000.0, CategoryL2.INCOME, Channel.TRANSFER, currency="usd"),
        _txn(-100.0, CategoryL2.GROCERIES, currency="USD"),
        _txn(-50.0, CategoryL2.GROCERIES, currency="EUR"),
    ]
    p = build_financial_profile(txns)
    assert p.user_id == "u-1"  # inherited from transactions
    assert p.currency == "USD"  # most frequent; case-normalised


def test_deterministic_and_generated_at_passthrough():
    a = build_financial_profile(_standard_txns())
    b = build_financial_profile(_standard_txns())
    assert a == b  # no wall-clock, no randomness → byte-identical profiles

    stamp = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    c = build_financial_profile(_standard_txns(), generated_at=stamp)
    assert c.generated_at == stamp


# ── Money-movement semantics (transfers / savings / refunds) ──────────────────


def test_salary_via_transfer_counts_as_income():
    """Rule precedence: a labelled INCOME credit stays income on the TRANSFER rail."""
    p = build_financial_profile(
        [_txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER)]
    )
    assert p.monthly_income == 5_000.0
    assert p.transfers_total == 0.0


def test_internal_transfer_debit_not_an_expense():
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER),
        _txn(-800.0, None, Channel.TRANSFER),  # unlabelled move between accounts
    ]
    p = build_financial_profile(txns)
    assert p.total_expenses == 0.0
    assert p.transfers_total == 800.0
    assert p.monthly_savings == 5_000.0
    assert p.category_spending == {}


def test_internal_transfer_credit_not_income():
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER),
        _txn(-1_200.0, CategoryL2.RENT_MORTGAGE, Channel.RECURRING),
        _txn(300.0, None, Channel.TRANSFER),  # money moved back
    ]
    p = build_financial_profile(txns)
    assert p.monthly_income == 5_000.0
    assert p.total_expenses == 1_200.0
    assert p.transfers_total == 300.0


def test_savings_transfer_debit_not_an_expense():
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER),
        _txn(-1_000.0, CategoryL2.SAVINGS_INVESTMENTS, Channel.TRANSFER),
        _txn(-450.0, CategoryL2.GROCERIES),
    ]
    p = build_financial_profile(txns)
    assert p.total_expenses == 450.0
    assert p.savings_transfers_total == 1_000.0
    assert p.monthly_savings == 4_550.0
    assert "Savings & Investments" not in p.category_spending
    assert p.category_spending["Groceries"] == 450.0


def test_savings_credit_also_not_income():
    """Withdrawal from savings into checking is a movement, not income."""
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER),
        _txn(-1_000.0, CategoryL2.SAVINGS_INVESTMENTS, Channel.TRANSFER),
        _txn(400.0, CategoryL2.SAVINGS_INVESTMENTS, Channel.TRANSFER),
    ]
    p = build_financial_profile(txns)
    assert p.monthly_income == 5_000.0
    assert p.savings_transfers_total == 1_400.0  # gross, both directions
    assert p.total_expenses == 0.0


def test_refund_credit_not_income_and_not_netted():
    """A labelled-expense credit is a refund: not income, tracked separately."""
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER),
        _txn(-120.0, CategoryL2.RESTAURANTS),
        _txn(60.0, CategoryL2.RESTAURANTS),  # partial refund
    ]
    p = build_financial_profile(txns)
    assert p.monthly_income == 5_000.0
    assert p.refunds_total == 60.0
    assert p.total_expenses == 120.0  # gross spend; refund not netted
    assert p.monthly_savings == 4_880.0
    assert p.category_spending["Restaurants"] == 120.0


def test_unlabelled_pos_credit_treated_as_refund_not_income():
    p = build_financial_profile([_txn(75.0, None, Channel.POS)])
    assert p.monthly_income == 0.0
    assert p.refunds_total == 75.0
    assert p.savings_rate is None


def test_mixed_transactions_stay_mathematically_consistent():
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER,
             datetime(2026, 3, 1, tzinfo=UTC)),
        _txn(-1_500.0, CategoryL2.RENT_MORTGAGE, Channel.TRANSFER,
             datetime(2026, 3, 2, tzinfo=UTC)),
        _txn(-700.0, CategoryL2.GROCERIES,
             ts=datetime(2026, 3, 6, tzinfo=UTC)),
        _txn(-1_000.0, CategoryL2.SAVINGS_INVESTMENTS, Channel.TRANSFER,
             datetime(2026, 3, 8, tzinfo=UTC)),
        _txn(-200.0, None, Channel.TRANSFER,
             datetime(2026, 3, 9, tzinfo=UTC)),
        _txn(60.0, CategoryL2.RESTAURANTS,
             ts=datetime(2026, 3, 10, tzinfo=UTC)),
        _txn(-120.0, CategoryL2.RESTAURANTS,
             ts=datetime(2026, 3, 11, tzinfo=UTC)),
    ]
    p = build_financial_profile(txns)
    # Invariants: buckets add up; categories sum to total expenses.
    assert p.total_expenses == pytest.approx(p.fixed_expenses + p.variable_expenses)
    assert sum(p.category_spending.values()) == pytest.approx(p.total_expenses)
    assert p.monthly_savings == pytest.approx(p.monthly_income - p.total_expenses)
    assert p.monthly_net_cash_flow == pytest.approx(p.monthly_savings)
    # Totals: expenses 1500 + 700 + 120; transfers/savings/refunds excluded.
    assert p.total_expenses == pytest.approx(2_320.0)
    assert p.monthly_income == pytest.approx(5_000.0)
    assert p.transfers_total == pytest.approx(200.0)
    assert p.savings_transfers_total == pytest.approx(1_000.0)
    assert p.refunds_total == pytest.approx(60.0)