"""
Unit tests for the deterministic decision orchestration layer
(``src.services.decision_engine``).

Baseline transaction set (single calendar month → monthly == raw sums):

    income 5000 | rent 1500 (fixed) | groceries 700 | restaurants 300
    electronics 300 (discretionary 600) | total 2800 | savings 2200 (rate 0.44)

Matches the Scenario Engine's documented test baseline so the optimiser
behaviour (30% caps / 50% floors) is identical to ``test_scenario_engine.py``.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.models import ScenarioParams, Transaction  # noqa: E402
from src.services.decision_engine import advise  # noqa: E402
from src.services.financial_profile import build_financial_profile  # noqa: E402
from src.services.scenario_engine import run_scenario  # noqa: E402
from src.utils.constants import AccountType, CategoryL2, Channel  # noqa: E402

UTC = timezone.utc


def _txn(
    amount: float,
    category: CategoryL2 | None,
    channel: Channel = Channel.POS,
    ts: datetime | None = None,
) -> Transaction:
    return Transaction(
        user_id="u-1",
        timestamp=ts or datetime(2026, 3, 5, tzinfo=UTC),
        amount=amount,
        merchant_name="Test Merchant",
        merchant_mcc=5411,
        account_type=AccountType.CHECKING,
        channel=channel,
        category_l2=category,
        currency="USD",
    )


def _standard_txns() -> list[Transaction]:
    return [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER,
             datetime(2026, 3, 1, tzinfo=UTC)),
        _txn(-1_500.0, CategoryL2.RENT_MORTGAGE, Channel.TRANSFER,
             datetime(2026, 3, 2, tzinfo=UTC)),
        _txn(-700.0, CategoryL2.GROCERIES, Channel.POS,
             datetime(2026, 3, 5, tzinfo=UTC)),
        _txn(-300.0, CategoryL2.RESTAURANTS, Channel.POS,
             datetime(2026, 3, 8, tzinfo=UTC)),
        _txn(-300.0, CategoryL2.ELECTRONICS, Channel.ONLINE,
             datetime(2026, 3, 12, tzinfo=UTC)),
    ]


# ── Feasible / partial / infeasible ────────────────────────────────────────────


def test_feasible_scenario_identity() -> None:
    d = advise(
        _standard_txns(),
        ScenarioParams(scenario_id="s0"),
        period="2026-03",
        user_id="u-1",
    )
    assert d.decision_type == "plan_review"
    assert "feasible" in d.recommendation
    assert d.scenario_id == "s0"
    assert d.decision_id == "decision-s0"
    assert d.supporting_metrics["scenario.monthly_income"] == 5_000.0
    assert d.supporting_metrics["scenario.monthly_savings"] == 2_200.0
    assert d.warnings == []


def test_partial_scenario_savings_target_high() -> None:
    d = advise(
        _standard_txns(),
        ScenarioParams(scenario_id="s-high", savings_target=5_000.0),
        period="2026-03",
        user_id="u-1",
    )
    assert d.decision_type == "budget_adjustment"
    assert "cannot be fully met" in d.recommendation
    achieved = d.supporting_metrics["scenario.monthly_savings"]
    assert achieved < 5_000.0
    assert achieved > 2_200.0  # best-effort cuts still helped
    assert any("not fully met" in w for w in d.warnings)
    assert any(a.startswith("Retarget monthly savings") for a in d.alternatives)


def test_infeasible_scenario_no_income() -> None:
    txns = [_txn(-100.0, CategoryL2.GROCERIES, Channel.POS,
                 datetime(2026, 3, 5, tzinfo=UTC))]
    d = advise(
        txns,
        ScenarioParams(scenario_id="s-inf", savings_target=200.0),
        period="2026-03",
        user_id="u-1",
    )
    assert d.decision_type == "budget_adjustment"
    assert "cannot be achieved" in d.recommendation
    assert any("No income" in w for w in d.warnings)
    assert any(a.startswith("Consider a lower savings target") for a in d.alternatives)


# ── Goal / income scenarios ────────────────────────────────────────────────────


def test_savings_target_scenario_met_with_flexible_cuts() -> None:
    d = advise(
        _standard_txns(),
        ScenarioParams(scenario_id="s-save", savings_target=2_500.0),
        period="2026-03",
        user_id="u-1",
    )
    assert d.decision_type == "savings_goal"
    assert "feasible" in d.recommendation
    assert d.supporting_metrics["scenario.monthly_savings"] == pytest.approx(
        2_500.0, abs=0.05
    )
    # Rent/Mortgage preserved; only flexible categories cut.
    assert d.supporting_metrics["scenario.total_expenses"] == pytest.approx(2_500.0)
    assert "monthly_savings" in d.reasoning
    assert not any("hard-protected essential" in w for w in d.warnings)


def test_savings_target_requiring_protected_cut_is_partial() -> None:
    """Gap of 800 exceeds flexible-only capacity but rent is protected →
    decision reflects partial + the hard-protected warning."""
    d = advise(
        _standard_txns(),
        ScenarioParams(scenario_id="s-3000", savings_target=3_000.0),
        period="2026-03",
        user_id="u-1",
    )
    assert d.decision_type == "budget_adjustment"
    assert "cannot be fully met" in d.recommendation
    assert d.supporting_metrics["scenario.monthly_savings"] < 3_000.0
    assert any("hard-protected essential obligations" in w for w in d.warnings)
    assert any(a.startswith("Retarget monthly savings") for a in d.alternatives)


def test_income_decrease_scenario() -> None:
    d = advise(
        _standard_txns(),
        ScenarioParams(scenario_id="s-ld", income_change_pct=-50.0),
        period="2026-03",
        user_id="u-1",
    )
    assert d.decision_type == "income_change"
    assert d.supporting_metrics["scenario.monthly_income"] == 2_500.0
    assert any("exceeds income" in w for w in d.warnings)
# ── Safety invariants ──────────────────────────────────────────────────────────


def test_uncategorized_counted_but_never_a_cut_target() -> None:
    """Uncategorized amounts remain counted but cannot be optimised away."""
    txns = [
        _txn(2_000.0, CategoryL2.INCOME, Channel.TRANSFER,
             datetime(2026, 3, 1, tzinfo=UTC)),
        _txn(-1_000.0, None, Channel.POS, datetime(2026, 3, 5, tzinfo=UTC)),
        _txn(-200.0, CategoryL2.GROCERIES, Channel.POS,
             datetime(2026, 3, 7, tzinfo=UTC)),
    ]

    # Direct engine-level guard: no cut applied to "Uncategorized".
    profile = build_financial_profile(txns, user_id="u-1", period="2026-03")
    scen = run_scenario(
        profile, ScenarioParams(scenario_id="s-u", savings_target=1_800.0)
    )
    assert scen.resulting_profile.category_spending["Uncategorized"] == 1_000.0
    assert not [
        c for c in scen.key_changes if c.metric == "category:Uncategorized"
    ]

    # Orchestrator-level: amount fully counted → savings reflect the 1000
    # uncategorized outflow (income 2000 − expenses 1200 = 800).
    d = advise(txns, ScenarioParams(savings_target=1_800.0), period="2026-03",
               user_id="u-1")
    assert d.supporting_metrics["baseline.total_expenses"] == 1_200.0
    assert d.supporting_metrics["scenario.total_expenses"] == 1_200.0
    assert d.supporting_metrics["scenario.monthly_savings"] == 800.0
    assert any("not fully met" in w for w in d.warnings)


def test_transfer_and_savings_movements_stay_excluded() -> None:
    """Refunds, internal transfers and savings movements never touch income
    or expenses, so they never distort the DecisionResult metrics."""
    txns = [
        _txn(5_000.0, CategoryL2.INCOME, Channel.TRANSFER,
             datetime(2026, 3, 1, tzinfo=UTC)),
        _txn(-1_500.0, CategoryL2.RENT_MORTGAGE, Channel.TRANSFER,
             datetime(2026, 3, 2, tzinfo=UTC)),
        _txn(-700.0, CategoryL2.GROCERIES, Channel.POS,
             datetime(2026, 3, 5, tzinfo=UTC)),
        _txn(-500.0, None, Channel.TRANSFER, datetime(2026, 3, 6, tzinfo=UTC)),
        _txn(300.0, CategoryL2.SAVINGS_INVESTMENTS, Channel.TRANSFER,
             datetime(2026, 3, 7, tzinfo=UTC)),
        _txn(100.0, CategoryL2.GROCERIES, Channel.POS,
             datetime(2026, 3, 8, tzinfo=UTC)),
    ]
    d = advise(txns, ScenarioParams(scenario_id="s-move"), period="2026-03",
               user_id="u-1")
    # Income not inflated by the savings credit (300) or the refund (100).
    assert d.supporting_metrics["scenario.monthly_income"] == 5_000.0
    # Expenses not inflated by the internal transfer debit (500).
    assert d.supporting_metrics["scenario.total_expenses"] == 2_200.0
    assert d.supporting_metrics["scenario.monthly_savings"] == 2_800.0


# ── Status reflection & determinism ────────────────────────────────────────────


def test_decision_reflects_scenario_status() -> None:
    cases = {
        "plan_review": ScenarioParams(scenario_id="s-ok"),
        "budget_adjustment": ScenarioParams(
            scenario_id="s-adj", savings_target=5_000.0
        ),
    }
    for decision_type, params in cases.items():
        d = advise(_standard_txns(), params, period="2026-03", user_id="u-1")
        assert d.decision_type == decision_type
        assert (d.warnings == []) == (decision_type == "plan_review")


def test_identical_inputs_produce_identical_output() -> None:
    kwargs = dict(period="2026-03", user_id="u-1")
    params = ScenarioParams(scenario_id="s-save", savings_target=3_000.0)
    d1 = advise(_standard_txns(), params, **kwargs)
    d2 = advise(_standard_txns(), params, **kwargs)
    assert d1 == d2
    assert d1.model_dump() == d2.model_dump()
    assert d1.generated_at is None  # no wall-clock stamp
    assert d1.confidence is None  # rule-based, no invented certainty