"""
Unit tests for the deterministic Scenario Engine
(``src.services.scenario_engine``).

Baseline numbers are hand-computed:

    income 5000 | fixed 1500 (Rent) | variable 1300
    (Groceries 700 + Restaurants 300 + Electronics 300,
     of which discretionary 600) | total 2800 | savings 2200 (rate 0.44)

The budget optimiser's default caps (30% cut per category, 50% floors)
and the hard protection of essential categories (Rent/Mortgage is in
``HARD_PROTECTED_CATEGORIES``) mean the maximum cut across the FLEXIBLE
baseline categories only (Groceries 700 + Restaurants 300 + Electronics
300 = 1300) is 390 via the LP, and up to 650 (0.50 x 1300) via the
heuristic squeeze. Tests are designed around those bounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.models import FinancialProfile, ScenarioParams  # noqa: E402
from src.services.scenario_engine import run_scenario  # noqa: E402


def _baseline() -> FinancialProfile:
    return FinancialProfile(
        user_id="u-1",
        monthly_income=5_000.0,
        fixed_expenses=1_500.0,
        variable_expenses=1_300.0,
        discretionary_expenses=600.0,
        recurring_expenses=1_200.0,
        total_expenses=2_800.0,
        category_spending={
            "Rent/Mortgage": 1_500.0,
            "Groceries": 700.0,
            "Restaurants": 300.0,
            "Electronics": 300.0,
        },
        monthly_savings=2_200.0,
        savings_rate=0.44,
        monthly_net_cash_flow=2_200.0,
    )


# ── No target scenarios ────────────────────────────────────────────────────────


def test_identity_scenario_is_feasible_and_unchanged():
    r = run_scenario(_baseline(), ScenarioParams(scenario_id="s0"))
    assert r.status == "feasible"
    assert r.resulting_profile == _baseline()  # nothing changed
    assert all(c.delta == 0 for c in r.key_changes if not c.metric.startswith("category:"))
    assert r.warnings == []


def test_income_decrease_creates_deficit_warning():
    r = run_scenario(_baseline(), ScenarioParams(income_change_pct=-50.0))
    assert r.resulting_profile.monthly_income == 2_500.0
    assert r.resulting_profile.monthly_savings == -300.0
    assert r.resulting_profile.savings_rate is None  # not representable
    assert r.status == "feasible"  # no target → scenario applies, but is flagged
    assert any("exceeds income" in w for w in r.warnings)
    inc = next(c for c in r.key_changes if c.metric == "monthly_income")
    assert inc.delta == -2_500.0


def test_expense_reduction_scales_all_buckets():
    r = run_scenario(_baseline(), ScenarioParams(expense_change_pct=-20.0))
    p = r.resulting_profile
    assert p.fixed_expenses == pytest.approx(1_200.0)
    assert p.variable_expenses == pytest.approx(1_040.0)
    assert p.discretionary_expenses == pytest.approx(480.0)
    assert p.total_expenses == pytest.approx(2_240.0)
    assert p.monthly_savings == pytest.approx(2_760.0)
    assert p.savings_rate == pytest.approx(0.552)
    total_change = next(c for c in r.key_changes if c.metric == "total_expenses")
    assert total_change.delta == pytest.approx(-560.0)


def test_category_increase_hits_correct_buckets():
    r = run_scenario(_baseline(), ScenarioParams(category_changes={"Restaurants": 50.0}))
    p = r.resulting_profile
    assert p.category_spending["Restaurants"] == pytest.approx(450.0)
    assert p.discretionary_expenses == pytest.approx(750.0)
    assert p.variable_expenses == pytest.approx(1_450.0)
    assert p.total_expenses == pytest.approx(2_950.0)
    cat_change = next(
        c for c in r.key_changes if c.metric == "category:Restaurants"
    )
    assert cat_change.delta == pytest.approx(150.0)


# ── Savings-target scenarios ───────────────────────────────────────────────────


def test_target_already_met_requires_no_cuts():
    r = run_scenario(_baseline(), ScenarioParams(savings_target=2_000.0))
    assert r.status == "feasible"
    assert r.resulting_profile.monthly_savings == 2_200.0  # unchanged
    assert not [c for c in r.key_changes if c.metric.startswith("category:")]


def test_savings_target_met_via_flexible_cuts():
    """Gap of 300 is within the flexible-only 390 LP capacity → feasible."""
    r = run_scenario(_baseline(), ScenarioParams(scenario_id="s-save",
                                                 savings_target=2_500.0))
    assert r.status == "feasible"
    assert r.resulting_profile.monthly_savings == pytest.approx(2_500.0, abs=0.05)
    assert r.resulting_profile.total_expenses == pytest.approx(2_500.0, abs=0.05)
    # Rent/Mortgage is hard-protected: never reduced.
    assert r.resulting_profile.category_spending["Rent/Mortgage"] == 1_500.0
    cat_changes = [c for c in r.key_changes if c.metric.startswith("category:")]
    assert cat_changes and all(c.delta <= 0 for c in cat_changes)
    # Only flexible categories were cut (which ones is an LP-vertex detail;
    # the policy contract is: protected untouched, gap met via flexible only).
    cut_metrics = {c.metric for c in cat_changes if c.delta < 0}
    assert cut_metrics
    assert cut_metrics <= {"category:Groceries", "category:Restaurants",
                           "category:Electronics"}


def test_savings_target_requiring_protected_cut_is_partial():
    """Gap of 800 exceeds flexible-only capacity (650) but rent is protected
    → status partial, rent preserved, warning about protected obligations."""
    r = run_scenario(_baseline(), ScenarioParams(scenario_id="s-save",
                                                 savings_target=3_000.0))
    assert r.status == "partial"
    assert r.resulting_profile.monthly_savings < 3_000.0
    assert r.resulting_profile.category_spending["Rent/Mortgage"] == 1_500.0
    assert r.resulting_profile.fixed_expenses == 1_500.0
    # No cut on the protected rent category in key_changes.
    assert not [c for c in r.key_changes if c.metric == "category:Rent/Mortgage"]
    assert any("hard-protected essential obligations" in w for w in r.warnings)
    assert any("not fully met" in w for w in r.warnings)


def test_infeasible_when_no_income():
    broke = FinancialProfile(
        user_id="u-broke",
        monthly_income=0.0,
        fixed_expenses=1_500.0,
        variable_expenses=1_300.0,
        discretionary_expenses=600.0,
        total_expenses=2_800.0,
        monthly_savings=-2_800.0,
        monthly_net_cash_flow=-2_800.0,
    )
    r = run_scenario(broke, ScenarioParams(savings_target=500.0))
    assert r.status == "infeasible"
    assert any("No income" in w for w in r.warnings)


# ── Behavioral feasibility integration (reuses FeasibilityChecker) ─────────────


def test_behaviorally_infeasible_cuts_downgrade_to_partial():
    """
    With a strong Groceries habit (0.9) and flat history, the checker allows
    only ~5% reduction — but closing the 800 gap requires deep flexible cuts
    (~50% of discretionary) → status becomes partial.
    """
    df = pd.DataFrame(
        {
            "user_id": ["u-1"] * 3,
            "amount": [-100.0, -100.0, -100.0],
            "category_l2": ["Groceries"] * 3,
        }
    )
    r = run_scenario(
        _baseline(),
        ScenarioParams(savings_target=3_000.0),
        transactions_df=df,
        habit_strengths={"Groceries": 0.9},
    )
    assert any("not behaviorally feasible" in w for w in r.warnings)
    assert r.status == "partial"


def test_no_history_df_skips_behavioral_screen():
    r = run_scenario(_baseline(), ScenarioParams(savings_target=3_000.0))
    assert not [w for w in r.warnings if "behaviorally" in w]


# ── Determinism ────────────────────────────────────────────────────────────────


def test_same_inputs_produce_identical_results():
    params = ScenarioParams(scenario_id="s-det", savings_target=3_000.0)
    a = run_scenario(_baseline(), params)
    b = run_scenario(_baseline(), params)
    assert a == b