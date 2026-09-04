"""
Unit tests for hard-essential-category protection in BudgetOptimiser.

Hand-computed setup (income 5000, baselines total 3000):

    Rent/Mortgage 1500 | Insurance Premiums 200   (protected, floors = baseline)
    Groceries 700 | Restaurants 300 | Electronics 300 (flexible, 50% floors)

    Flexible max cut = 30% caps  ->  210 + 90 + 90 = 390   (LP-feasible region)
    Flexible max cut = 50% floors ->  350 + 150 + 150 = 650 (heuristic squeeze)

A savings target of 2390 needs exactly 390 of cuts (LP-optimal, unique).
A target of 3000 needs 1000 of cuts — beyond even the 650 squeeze, so the
result is best-effort with protected categories untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.recommender.budget_optimizer import BudgetOptimiser  # noqa: E402
from src.utils.constants import HARD_PROTECTED_CATEGORIES  # noqa: E402

BASELINES = {
    "Rent/Mortgage": 1500.0,
    "Insurance Premiums": 200.0,
    "Groceries": 700.0,
    "Restaurants": 300.0,
    "Electronics": 300.0,
}


def _cuts(result) -> dict[str, float]:
    return {a.category: a.cut_amount for a in result.allocations}


def test_protected_categories_receive_zero_cut_under_large_gap():
    """Gap (1000) far exceeds flexible capacity (650): protected stay at baseline."""
    result = BudgetOptimiser().optimise(
        income=5000.0, savings_target=3000.0, category_baselines=dict(BASELINES)
    )
    cuts = _cuts(result)
    assert cuts["Rent/Mortgage"] == pytest.approx(0.0, abs=1e-6)
    assert cuts["Insurance Premiums"] == pytest.approx(0.0, abs=1e-6)
    # Flexible categories absorb everything they can (heuristic squeeze to 50% floors)
    assert cuts["Groceries"] == pytest.approx(350.0, abs=1e-6)
    assert cuts["Restaurants"] == pytest.approx(150.0, abs=1e-6)
    assert cuts["Electronics"] == pytest.approx(150.0, abs=1e-6)
    # Best effort, not the requested target
    assert result.savings_achieved == pytest.approx(2650.0, abs=0.01)
    assert result.savings_achieved < 3000.0


def test_protection_applies_by_default_without_explicit_argument():
    """Omitting protected_categories still shields every HARD_PROTECTED category."""
    baselines = {c: 500.0 for c in HARD_PROTECTED_CATEGORIES} | {"Dining Out": 400.0}
    result = BudgetOptimiser().optimise(
        income=1000.0, savings_target=800.0, category_baselines=baselines
    )
    cuts = _cuts(result)
    for cat in HARD_PROTECTED_CATEGORIES:
        assert cuts[cat] == pytest.approx(0.0, abs=1e-6), cat
    assert cuts["Dining Out"] > 0.0


def test_flexible_target_achievable_without_touching_protected():
    """Target 2390 needs exactly the 390 flexible LP capacity → optimal, protected untouched."""
    result = BudgetOptimiser().optimise(
        income=5000.0, savings_target=2390.0, category_baselines=dict(BASELINES)
    )
    cuts = _cuts(result)
    assert result.solver_status == "optimal"
    assert cuts["Rent/Mortgage"] == pytest.approx(0.0, abs=1e-6)
    assert cuts["Insurance Premiums"] == pytest.approx(0.0, abs=1e-6)
    assert cuts["Groceries"] == pytest.approx(210.0, abs=1e-6)
    assert cuts["Restaurants"] == pytest.approx(90.0, abs=1e-6)
    assert cuts["Electronics"] == pytest.approx(90.0, abs=1e-6)
    assert result.savings_achieved == pytest.approx(2390.0, abs=0.01)


def test_empty_protected_set_disables_protection():
    """Documented escape hatch: protected_categories=set() allows cutting rent."""
    result = BudgetOptimiser().optimise(
        income=5000.0,
        savings_target=3000.0,
        category_baselines=dict(BASELINES),
        protected_categories=set(),
    )
    assert _cuts(result)["Rent/Mortgage"] > 0.0


def test_is_discretionary_forced_false_for_protected():
    """Protected categories must not inherit discretionary weighting."""
    result = BudgetOptimiser().optimise(
        income=5000.0,
        savings_target=2390.0,
        category_baselines=dict(BASELINES),
        is_discretionary={c: True for c in BASELINES},
    )
    flags = {a.category: a.is_discretionary for a in result.allocations}
    assert flags["Rent/Mortgage"] is False
    assert flags["Insurance Premiums"] is False
    assert flags["Groceries"] is True
