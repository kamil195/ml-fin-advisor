"""
Focused validation tests for the canonical planning contracts added to
``src/data/models.py`` (FinancialProfile, BehavioralProfile, ScenarioParams,
ScenarioResult, DecisionResult).

These are contracts for the Financial Intelligence layer
(Transactions → Classification → FinancialProfile → Behaviour/Forecast/Budget
→ Scenario → Decision → AI Copilot → Outcome Tracking). No route wiring yet —
the tests pin down the validation behaviour future layers will rely on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.models import (  # noqa: E402
    BehavioralProfile,
    DecisionResult,
    FinancialProfile,
    MetricChange,
    ScenarioParams,
    ScenarioResult,
)


# ── FinancialProfile ────────────────────────────────────────────────────────────


def test_financial_profile_minimal_valid():
    p = FinancialProfile(user_id="u-123", monthly_income=5_000.0)
    assert p.monthly_income == 5_000.0
    assert p.fixed_expenses == 0.0 and p.variable_expenses == 0.0
    assert p.total_expenses is None and p.savings_rate is None


def test_financial_profile_full_population():
    p = FinancialProfile(
        user_id="u-123",
        monthly_income=5_000.0,
        fixed_expenses=1_800.0,
        variable_expenses=1_200.0,
        discretionary_expenses=400.0,
        recurring_expenses=900.0,
        total_expenses=3_000.0,
        monthly_savings=500.0,
        savings_rate=0.10,
        total_debt=12_000.0,
        monthly_debt_payments=250.0,
        monthly_net_cash_flow=2_000.0,
        liquid_buffer=9_000.0,
        months_of_buffer=3.0,
        category_spending={"Groceries": 450.0, "Rent/Mortgage": 1_500.0},
    )
    assert p.savings_rate == 0.10
    assert p.category_spending["Groceries"] == 450.0


def test_financial_profile_rejects_negative_components():
    with pytest.raises(ValidationError):
        FinancialProfile(user_id="u", monthly_income=1.0, fixed_expenses=-1.0)


def test_financial_profile_rejects_savings_rate_above_one():
    with pytest.raises(ValidationError):
        FinancialProfile(user_id="u", monthly_income=100.0, savings_rate=1.5)


def test_financial_profile_currency_normalised():
    p = FinancialProfile(user_id="u", monthly_income=1.0, currency="eur")
    assert p.currency == "EUR"


# ── BehavioralProfile ───────────────────────────────────────────────────────────


def test_behavioral_profile_valid():
    b = BehavioralProfile(
        user_id="u-123",
        observation_days=90,
        spending_regime="elevated",
        discretionary_ratio=0.32,
        impulse_score=0.4,
        spending_volatility=0.18,
        habit_strength={"Groceries": 0.9, "Restaurants": 0.55},
        top_patterns=["Groceries every Sunday"],
    )
    assert b.spending_regime == "elevated"
    assert b.habit_strength["Restaurants"] == 0.55


def test_behavioral_profile_rejects_unknown_regime():
    with pytest.raises(ValidationError):
        BehavioralProfile(user_id="u", spending_regime="wild")


def test_behavioral_profile_rejects_out_of_range_scores():
    with pytest.raises(ValidationError):
        BehavioralProfile(user_id="u", impulse_score=1.5)
    with pytest.raises(ValidationError):
        BehavioralProfile(user_id="u", habit_strength={"Food": 2.0})


# ── ScenarioParams ──────────────────────────────────────────────────────────────


def test_scenario_params_defaults_are_identity():
    s = ScenarioParams()
    assert s.income_change_pct == 0.0
    assert s.expense_change_pct == 0.0
    assert s.horizon_months == 1
    assert s.category_changes == {}
    assert s.savings_target is None


def test_scenario_params_valid_changes():
    s = ScenarioParams(
        horizon_months=6,
        income_change_pct=-50.0,
        expense_change_pct=-10.0,
        savings_target=600.0,
        category_changes={"Shopping & Entertainment": -25.0, "Groceries": 5.0},
    )
    assert s.horizon_months == 6
    assert s.category_changes["Groceries"] == 5.0


def test_scenario_params_rejects_income_change_below_minus_100():
    with pytest.raises(ValidationError):
        ScenarioParams(income_change_pct=-120.0)


def test_scenario_params_rejects_category_reduction_below_minus_100():
    with pytest.raises(ValidationError, match="cannot reduce below"):
        ScenarioParams(category_changes={"Groceries": -150.0})


def test_scenario_params_rejects_bad_horizon_and_negative_target():
    with pytest.raises(ValidationError):
        ScenarioParams(horizon_months=0)
    with pytest.raises(ValidationError):
        ScenarioParams(horizon_months=37)
    with pytest.raises(ValidationError):
        ScenarioParams(savings_target=-1.0)


# ── ScenarioResult & DecisionResult ─────────────────────────────────────────────


def _baseline_profile() -> FinancialProfile:
    return FinancialProfile(user_id="u-123", monthly_income=5_000.0, fixed_expenses=1_800.0)


def test_scenario_result_valid_with_changes():
    r = ScenarioResult(
        scenario_id="s-1",
        params=ScenarioParams(income_change_pct=10.0),
        status="feasible",
        resulting_profile=_baseline_profile(),
        key_changes=[
            MetricChange(metric="monthly_income", before=5_000.0, after=5_500.0, delta=500.0)
        ],
        warnings=[],
    )
    assert r.key_changes[0].delta == 500.0
    assert r.resulting_profile.monthly_income == 5_000.0


def test_scenario_result_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ScenarioResult(
            scenario_id="s-1",
            status="maybe",
            resulting_profile=_baseline_profile(),
        )


def test_decision_result_valid():
    d = DecisionResult(
        decision_id="d-1",
        scenario_id="s-1",
        decision_type="budget_adjustment",
        recommendation="Cap Restaurants at $220/month for 3 months.",
        reasoning="Restaurants run 34% above the peer median.",
        supporting_metrics={"restaurants_monthly": 330.0, "peer_median": 220.0},
        confidence=0.82,
        alternatives=["Cancel one streaming subscription"],
    )
    assert d.confidence == 0.82
    assert "Restaurants" in d.recommendation


def test_decision_result_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        DecisionResult(
            decision_type="budget_adjustment",
            recommendation="Save more.",
            confidence=1.2,
        )


# ── Cross-contract serialisation (AI-copilot payload stability) ────────────────


def test_contracts_round_trip_through_json():
    r = ScenarioResult(
        scenario_id="s-1",
        params=ScenarioParams(expense_change_pct=-5.0, savings_target=500.0),
        status="partial",
        resulting_profile=_baseline_profile(),
        key_changes=[
            MetricChange(metric="total_expenses", before=3000.0, after=2850.0, delta=-150.0)
        ],
        warnings=["Fixed expenses could not be reduced."],
    )
    d = DecisionResult(
        decision_type="savings_goal",
        recommendation="Automate $500/month transfer.",
        confidence=0.7,
    )
    r2 = ScenarioResult.model_validate_json(r.model_dump_json())
    d2 = DecisionResult.model_validate_json(d.model_dump_json())
    assert r2 == r and d2 == d