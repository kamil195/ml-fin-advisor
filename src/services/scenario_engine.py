"""
Deterministic Scenario Engine.

    FinancialProfile → ScenarioParams → run_scenario → ScenarioResult

Applies ``ScenarioParams`` to a baseline ``FinancialProfile`` and, when the
scenario carries a savings target the projected plan cannot already meet,
delegates the required cuts to the existing ``BudgetOptimiser`` (reused, not
modified). When a transaction history is supplied, proposed cuts are also
screened through the existing ``FeasibilityChecker``.

Determinism: no wall-clock reads, no randomness; dict iteration is sorted
where it affects arithmetic. The same (baseline, params) pair always yields
an identical ScenarioResult.

Status semantics (matches ``ScenarioResult.status``):

* ``feasible``  — constraints met without adjustments (or optimiser found an
  exact solution and every cut is behaviorally feasible).
* ``partial``   — applied with adjustments: best-effort cuts could not fully
  fund the target, and/or some cuts exceed behaviorally feasible limits.
* ``infeasible`` — the target can not be pursued at all (e.g. no income).

Honesty rules: nothing here invents money. Income stays what the profile
says it is; cuts respect the optimiser's floors/caps; unknown balance-sheet
items stay ``None``.
"""

from __future__ import annotations

from src.data.models import (
    FinancialProfile,
    MetricChange,
    ScenarioParams,
    ScenarioResult,
)
from src.models.recommender.budget_optimizer import BudgetOptimiser
from src.models.recommender.feasibility import FeasibilityChecker
from src.utils.constants import (
    DISCRETIONARY_CATEGORIES,
    HARD_PROTECTED_CATEGORIES,
)

from .financial_profile import FIXED_CATEGORIES, UNCATEGORIZED_KEY

_EPS = 0.01  # cent-level tolerance for float comparisons


def _is_fixed_category(cat: str) -> bool:
    return cat in {c.value for c in FIXED_CATEGORIES}


def _is_discretionary_category(cat: str) -> bool:
    return any(c.value == cat for c in DISCRETIONARY_CATEGORIES)


def _project_profile(
    baseline: FinancialProfile,
    params: ScenarioParams,
) -> tuple[FinancialProfile, list[str]]:
    """Apply ScenarioParams to the baseline; returns (projected, warnings)."""
    warnings: list[str] = []

    income = max(0.0, baseline.monthly_income * (1 + params.income_change_pct / 100.0))
    exp_factor = max(0.0, 1 + params.expense_change_pct / 100.0)

    fixed = baseline.fixed_expenses * exp_factor
    variable = baseline.variable_expenses * exp_factor
    discretionary = baseline.discretionary_expenses * exp_factor
    recurring = baseline.recurring_expenses * exp_factor
    cats = dict(baseline.category_spending)

    # Category-level adjustments (sorted → deterministic order).
    for cat in sorted(params.category_changes):
        pct = params.category_changes[cat]
        base_val = cats.get(cat)
        if base_val is None:
            if pct > 0:
                # Not observed before: a positive change is interpreted as an
                # absolute monthly spend amount (there is no base to scale).
                cats[cat] = float(pct)
                if _is_fixed_category(cat):
                    fixed += float(pct)
                else:
                    variable += float(pct)
                    if _is_discretionary_category(cat):
                        discretionary += float(pct)
                warnings.append(
                    f"Category '{cat}' is not in the baseline; the +{pct:g} "
                    "change was interpreted as absolute monthly spend."
                )
            else:
                warnings.append(
                    f"Category '{cat}' is not in the baseline; the {pct:g}% "
                    "change was ignored."
                )
            continue

        new_val = max(0.0, base_val * (1 + pct / 100.0))
        applied = new_val - base_val
        cats[cat] = new_val
        if _is_fixed_category(cat):
            fixed = max(0.0, fixed + applied)
        else:
            variable = max(0.0, variable + applied)
            if _is_discretionary_category(cat):
                discretionary = max(0.0, discretionary + applied)

    total = fixed + variable
    savings = income - total
    savings_rate = (
        savings / income if income > 0 and savings >= 0 else None
    )
    if income > 0 and savings < 0:
        warnings.append(
            "Projected outflow exceeds income; the savings rate is not "
            "representable and is reported as None."
        )
    if income > 0 and income < fixed:
        warnings.append("Income does not cover fixed obligations.")

    months_of_buffer = (
        baseline.liquid_buffer / total
        if baseline.liquid_buffer is not None and total > 0
        else None
    )

    projected = FinancialProfile(
        user_id=baseline.user_id,
        currency=baseline.currency,
        period=baseline.period,
        monthly_income=income,
        fixed_expenses=fixed,
        variable_expenses=variable,
        discretionary_expenses=discretionary,
        recurring_expenses=recurring,
        total_expenses=total,
        category_spending=cats,
        monthly_savings=savings,
        savings_rate=savings_rate,
        total_debt=baseline.total_debt,
        monthly_debt_payments=baseline.monthly_debt_payments,
        monthly_net_cash_flow=savings,
        liquid_buffer=baseline.liquid_buffer,
        months_of_buffer=months_of_buffer,
        generated_at=baseline.generated_at,
    )
    return projected, warnings


def _apply_cuts(
    profile: FinancialProfile,
    allocations: list,
) -> FinancialProfile:
    """Apply BudgetAllocation cuts to a profile's buckets and categories."""
    cats = dict(profile.category_spending)
    fixed = profile.fixed_expenses
    variable = profile.variable_expenses
    discretionary = profile.discretionary_expenses

    for alloc in sorted(allocations, key=lambda a: a.category):
        cut = float(alloc.cut_amount)
        if cut <= 0:
            continue
        if alloc.category in HARD_PROTECTED_CATEGORIES:
            # Defense-in-depth: hard-protected categories are never reduced,
            # even if an allocation were somehow produced for them.
            continue
        if alloc.category in cats:
            cats[alloc.category] = max(0.0, cats[alloc.category] - cut)
        if _is_fixed_category(alloc.category):
            fixed = max(0.0, fixed - cut)
        else:
            variable = max(0.0, variable - cut)
            if _is_discretionary_category(alloc.category):
                discretionary = max(0.0, discretionary - cut)

    total = fixed + variable
    savings = profile.monthly_income - total
    savings_rate = savings / profile.monthly_income if (
        profile.monthly_income > 0 and savings >= 0
    ) else None
    months_of_buffer = (
        profile.liquid_buffer / total if profile.liquid_buffer is not None and total > 0 else None
    )

    return FinancialProfile(
        user_id=profile.user_id,
        currency=profile.currency,
        period=profile.period,
        monthly_income=profile.monthly_income,
        fixed_expenses=fixed,
        variable_expenses=variable,
        discretionary_expenses=discretionary,
        recurring_expenses=profile.recurring_expenses,
        total_expenses=total,
        category_spending=cats,
        monthly_savings=savings,
        savings_rate=savings_rate,
        total_debt=profile.total_debt,
        monthly_debt_payments=profile.monthly_debt_payments,
        monthly_net_cash_flow=savings,
        liquid_buffer=profile.liquid_buffer,
        months_of_buffer=months_of_buffer,
        generated_at=profile.generated_at,
    )


def _behavioral_warnings(
    allocations: list,
    *,
    user_id: str,
    transactions_df,
    habit_strengths: dict[str, float] | None,
    compliance_history: dict[str, float] | None,
) -> list[str]:
    """Screen proposed cuts through the existing FeasibilityChecker (if df given)."""
    if transactions_df is None:
        return []
    proposed = {
        a.category: a.cut_pct / 100.0 for a in allocations if a.cut_amount > 0
    }
    if not proposed:
        return []
    checker = FeasibilityChecker()
    results = checker.check_all(
        transactions_df,
        user_id,
        categories=list(proposed),
        habit_strengths=habit_strengths or {},
        proposed_reductions=proposed,
        compliance_history=compliance_history,
    )
    return [
        f"Cuts of {r.max_reduction_pct:.0%}-cap exceeded: reducing "
        f"'{r.category}' by {proposed[r.category]:.0%} is not behaviorally "
        "feasible (habit/variance/compliance)."
        for r in results
        if not r.feasible and r.category in proposed
    ]


def _metric_changes(
    baseline: FinancialProfile,
    final: FinancialProfile,
) -> list[MetricChange]:
    """Before/after deltas for the headline metrics plus changed categories."""
    changes: list[MetricChange] = []
    for field in (
        "monthly_income",
        "fixed_expenses",
        "variable_expenses",
        "discretionary_expenses",
        "total_expenses",
        "monthly_savings",
    ):
        before = float(getattr(baseline, field))
        after = float(getattr(final, field))
        changes.append(
            MetricChange(
                metric=field,
                before=round(before, 4),
                after=round(after, 4),
                delta=round(after - before, 4),
            )
        )

    rate_b = baseline.savings_rate
    rate_a = final.savings_rate
    if rate_b is not None or rate_a is not None:
        changes.append(
            MetricChange(
                metric="savings_rate",
                before=rate_b,
                after=rate_a,
                delta=round(rate_a - rate_b, 4) if rate_a is not None and rate_b is not None else None,
            )
        )

    for cat in sorted(set(baseline.category_spending) | set(final.category_spending)):
        before = baseline.category_spending.get(cat, 0.0)
        after = final.category_spending.get(cat, 0.0)
        if abs(after - before) > _EPS:
            changes.append(
                MetricChange(
                    metric=f"category:{cat}",
                    before=round(before, 4),
                    after=round(after, 4),
                    delta=round(after - before, 4),
                )
            )
    return changes


def run_scenario(
    baseline: FinancialProfile,
    params: ScenarioParams,
    *,
    transactions_df=None,
    habit_strengths: dict[str, float] | None = None,
    compliance_history: dict[str, float] | None = None,
) -> ScenarioResult:
    """
    Apply ``params`` to ``baseline`` and produce a canonical ScenarioResult.

    When the scenario sets a ``savings_target`` the projected plan cannot
    already meet, the gap is delegated to the existing ``BudgetOptimiser``;
    if a transaction history (``transactions_df``) is provided, the proposed
    cuts are additionally screened through the existing ``FeasibilityChecker``
    and behaviorally infeasible cuts downgrade the status to ``partial``.
    """
    warnings: list[str] = []

    # ── 1. Baseline → projected (apply the scenario's own assumptions) ───
    projected, proj_warnings = _project_profile(baseline, params)
    warnings.extend(proj_warnings)
    final = projected
    status = "feasible"

    # ── 2. Savings target → close the gap with the budget optimiser ──────
    target = params.savings_target
    if target is not None and target > 0:
        gap = target - projected.monthly_savings
        if projected.monthly_income <= 0:
            status = "infeasible"
            warnings.append(
                "No income observed in the scenario; a savings target cannot be met."
            )
        elif gap > _EPS:
            optimiser = BudgetOptimiser()
            protected = set(HARD_PROTECTED_CATEGORIES)
            cuttable = {
                c: v
                for c, v in sorted(projected.category_spending.items())
                if v > 0 and c != UNCATEGORIZED_KEY
            }
            if not cuttable:
                status = "infeasible"
                warnings.append(
                    "No category spending available to cut toward the "
                    "savings target."
                )
            else:
                opt = optimiser.optimise(
                    income=projected.monthly_income,
                    savings_target=target,
                    category_baselines=cuttable,
                    is_discretionary={
                        c: not _is_fixed_category(c) for c in cuttable
                    },
                    protected_categories=protected,
                )
                if opt.savings_achieved <= _EPS:
                    status = "infeasible"
                    warnings.append(
                        "No reductions could be applied toward the savings target."
                    )
                else:
                    final = _apply_cuts(projected, opt.allocations)
                    achieved = final.monthly_savings
                    if opt.solver_status != "optimal":
                        warnings.append(
                            "Budget optimiser fell back to heuristic cuts "
                            f"(status={opt.solver_status})."
                        )
                    if achieved < target - _EPS:
                        status = "partial"
                        remaining_gap = target - achieved
                        if any(c in protected and v > 0 for c, v in projected.category_spending.items()):
                            warnings.append(
                                "The savings target cannot be fully met without "
                                "cutting hard-protected essential obligations "
                                "(e.g., Rent/Mortgage, Insurance Premiums); "
                                "these were preserved."
                            )
                        warnings.append(
                            f"Savings target not fully met: achieved "
                            f"{achieved:,.2f} of {target:,.2f} (shortfall "
                            f"{remaining_gap:,.2f}) after best-effort cuts."
                        )

                    # ── 3. Behavioral feasibility screen (optional input) ─
                    behav = _behavioral_warnings(
                        opt.allocations,
                        user_id=baseline.user_id,
                        transactions_df=transactions_df,
                        habit_strengths=habit_strengths,
                        compliance_history=compliance_history,
                    )
                    if behav:
                        warnings.extend(behav)
                        if status == "feasible":
                            status = "partial"

    elif target is not None and target <= 0:
        warnings.append("Savings target is zero or negative; treated as no target.")

    if projected.monthly_savings < 0 and target is None:
        warnings.append(
            "Resulting plan spends more than income (negative monthly savings)."
        )

    return ScenarioResult(
        scenario_id=params.scenario_id,
        params=params,
        status=status,
        resulting_profile=final,
        key_changes=_metric_changes(baseline, final),
        warnings=warnings,
    )