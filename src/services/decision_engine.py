"""Deterministic decision orchestration service.

    Transactions → FinancialProfile → ScenarioResult → DecisionResult

This is the smallest internal wiring layer connecting the existing
financial-intelligence components. It performs NO financial arithmetic of its
own: every number in the :class:`DecisionResult` is read directly from the
:class:`FinancialProfile` and :class:`ScenarioResult` produced by the existing
``build_financial_profile`` and ``run_scenario`` services.

Safety invariants (all inherited from the underlying services, not re-implemented
here):

* transfer / refund / savings-movement semantics (``build_financial_profile``);
* confidence gating (the caller supplies already-classified transactions);
* ``Uncategorized`` is never a scenario cut target (``run_scenario``);
* fixed/essential category protections (``run_scenario`` + ``BudgetOptimiser``);
* income is never invented.

Determinism: no randomness, no wall-clock reads, no external/LLM calls.
Identical inputs produce an identical ``DecisionResult`` (``generated_at`` is
not set unless the caller passes it).
"""

from __future__ import annotations

from collections.abc import Sequence

from src.data.models import (
    DecisionResult,
    FinancialProfile,
    ScenarioParams,
    ScenarioResult,
    Transaction,
)
from src.services.financial_profile import build_financial_profile
from src.services.scenario_engine import run_scenario

__all__ = ["advise", "to_decision"]


def advise(
    transactions: Sequence[Transaction],
    params: ScenarioParams,
    *,
    user_id: str | None = None,
    period: str | None = None,
    observation_days: int | None = None,
    liquid_buffer: float | None = None,
    total_debt: float | None = None,
    monthly_debt_payments: float | None = None,
    transactions_df=None,
    habit_strengths: dict[str, float] | None = None,
    compliance_history: dict[str, float] | None = None,
    decision_id: str | None = None,
) -> DecisionResult:
    """
    Build the baseline profile, run the scenario, and convert the outcome into
    a canonical :class:`DecisionResult`.

    Parameters
    ----------
    transactions:
        Already-classified transactions for the aggregation window.
    params:
        Scenario assumptions (the identity ``ScenarioParams()`` is a no-change
        scenario).
    user_id, period, observation_days, liquid_buffer, total_debt,
    monthly_debt_payments:
        Passed through to ``build_financial_profile`` (see its docstring).
    transactions_df, habit_strengths, compliance_history:
        Optional inputs forwarded to ``run_scenario`` for the behavioral
        feasibility screen.
    decision_id:
        Optional explicit decision identifier; defaults to
        ``decision-<scenario_id>`` when the scenario carries an id.
    """
    profile = build_financial_profile(
        transactions,
        user_id=user_id,
        period=period,
        observation_days=observation_days,
        liquid_buffer=liquid_buffer,
        total_debt=total_debt,
        monthly_debt_payments=monthly_debt_payments,
    )
    scenario = run_scenario(
        profile,
        params,
        transactions_df=transactions_df,
        habit_strengths=habit_strengths,
        compliance_history=compliance_history,
    )
    return to_decision(profile, scenario, decision_id=decision_id)


def to_decision(
    baseline: FinancialProfile,
    scenario: ScenarioResult,
    *,
    decision_id: str | None = None,
) -> DecisionResult:
    """
    Convert a :class:`ScenarioResult` into a :class:`DecisionResult`.

    Deterministic and arithmetic-free: every metric, delta and wording is read
    from the existing profile/scenario outputs.
    """
    params = scenario.params if scenario.params is not None else ScenarioParams()
    final = scenario.resulting_profile
    status = scenario.status

    # ── decision_type (coarse, deterministic) ─────────────────────────────
    if status != "feasible" and params.savings_target is not None:
        decision_type = "budget_adjustment"
    elif params.savings_target is not None:
        decision_type = "savings_goal"
    elif params.income_change_pct != 0:
        decision_type = "income_change"
    elif params.expense_change_pct != 0:
        decision_type = "expense_change"
    elif params.category_changes:
        decision_type = "category_adjustment"
    else:
        decision_type = "plan_review"

    # ── recommendation (status wording only) ─────────────────────────────
    if status == "feasible":
        recommendation = (
            "The scenario is feasible as requested — no adjustment is required."
        )
    elif status == "partial":
        achieved = final.monthly_savings
        target = params.savings_target
        if target is not None:
            recommendation = (
                f"The savings target cannot be fully met within safe limits; "
                f"the resulting plan achieves {achieved:,.2f} per month "
                f"(target {target:,.2f})."
            )
        else:
            recommendation = (
                "The scenario is only partially achievable within the current "
                "financial state; see the warnings for required adjustments."
            )
    else:  # infeasible
        recommendation = (
            "The scenario cannot be achieved under the current financial state "
            "(no feasible budget adjustment exists)."
        )

    # ── reasoning: headline deltas as computed by the scenario engine ─────
    changes = {c.metric: c for c in scenario.key_changes}
    parts: list[str] = []
    for metric in ("monthly_income", "total_expenses", "monthly_savings"):
        c = changes.get(metric)
        if c is not None and c.before is not None and c.after is not None:
            parts.append(f"{metric}: {c.before:,.2f} → {c.after:,.2f}")
    reasoning = "; ".join(parts) if parts else "No material change to headline metrics."

    # ── supporting metrics: pulled from existing outputs only ─────────────
    metrics: dict[str, float] = {}
    for name, value in (
        ("baseline.monthly_income", baseline.monthly_income),
        ("baseline.total_expenses", baseline.total_expenses),
        ("baseline.monthly_savings", baseline.monthly_savings),
        ("scenario.monthly_income", final.monthly_income),
        ("scenario.total_expenses", final.total_expenses),
        ("scenario.monthly_savings", final.monthly_savings),
    ):
        if value is not None:
            metrics[name] = float(value)
    if final.savings_rate is not None:
        metrics["scenario.savings_rate"] = float(final.savings_rate)

    # ── alternatives: deterministic, derived from existing outputs ────────
    alternatives: list[str] = []
    if status == "partial" and params.savings_target is not None:
        achieved = final.monthly_savings
        if achieved is not None:
            alternatives.append(
                "Retarget monthly savings to "
                f"{achieved:,.2f} to make the plan fully feasible."
            )
        if final.savings_rate is not None:
            alternatives.append(
                f"Express the goal as a savings rate of {final.savings_rate:.1%}."
            )
    if status == "infeasible":
        alternatives.append(
            "Consider a lower savings target, an income-side change, or a longer "
            "horizon."
        )
    if (
        params.savings_target is None
        and baseline.savings_rate is not None
        and baseline.savings_rate < 0.10
    ):
        alternatives.append(
            "Consider setting a modest monthly savings target to build a financial "
            "buffer."
        )

    return DecisionResult(
        decision_id=decision_id
        or (f"decision-{scenario.scenario_id}" if scenario.scenario_id else None),
        scenario_id=scenario.scenario_id,
        decision_type=decision_type,
        recommendation=recommendation,
        reasoning=reasoning,
        supporting_metrics=metrics,
        confidence=None,
        alternatives=alternatives,
        warnings=list(scenario.warnings),
    )