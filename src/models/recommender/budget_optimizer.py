"""
Constraint-based budget optimisation (SPEC §8.2, Steps 1–4).

Solves:
  minimize  Σ_c w_c · |budget[c] - user_preference[c]|
  s.t.  Σ_c budget[c] ≤ income - savings_target
        budget[c] ≥ floor[c]            (essential minimums)
        budget[c] ≤ ceiling[c]          (behavioral max reduction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BudgetAllocation:
    """Optimised budget for a single category."""

    category: str
    budget: float
    baseline: float
    cut_amount: float
    cut_pct: float
    is_discretionary: bool


@dataclass
class OptimisationResult:
    """Result of the budget optimisation."""

    allocations: list[BudgetAllocation] = field(default_factory=list)
    total_budget: float = 0.0
    savings_achieved: float = 0.0
    solver_status: str = "not_run"


class BudgetOptimiser:
    """
    Constraint-based budget optimiser using scipy.optimize.linprog.

    Parameters
    ----------
    savings_floor_pct : float
        Minimum percentage of income allocated to savings.
    max_reduction_cap : float
        Maximum percentage reduction for any single category.
    """

    def __init__(
        self,
        savings_floor_pct: float = 0.10,
        max_reduction_cap: float = 0.30,
    ) -> None:
        self.savings_floor_pct = savings_floor_pct
        self.max_reduction_cap = max_reduction_cap

    def optimise(
        self,
        income: float,
        savings_target: float,
        category_baselines: dict[str, float],
        category_floors: dict[str, float] | None = None,
        elasticity_scores: dict[str, float] | None = None,
        is_discretionary: dict[str, bool] | None = None,
    ) -> OptimisationResult:
        """
        Run the budget optimisation.

        Parameters
        ----------
        income : float
            Estimated monthly income.
        savings_target : float
            Desired monthly savings.
        category_baselines : dict
            Forecast p50 spend per category (baseline budgets).
        category_floors : dict | None
            Minimum spend per category (essential minimums).
        elasticity_scores : dict | None
            Per-category elasticity (higher = easier to reduce).
        is_discretionary : dict | None
            Whether each category is discretionary.
        """
        categories = list(category_baselines.keys())
        n = len(categories)

        if n == 0:
            return OptimisationResult(solver_status="no_categories")

        baselines = np.array([category_baselines[c] for c in categories])

        # Defaults
        if category_floors is None:
            category_floors = {c: baselines[i] * 0.5 for i, c in enumerate(categories)}
        floors = np.array([category_floors.get(c, 0.0) for c in categories])

        if elasticity_scores is None:
            elasticity_scores = {c: 0.5 for c in categories}
        elasticities = np.array([elasticity_scores.get(c, 0.5) for c in categories])

        if is_discretionary is None:
            is_discretionary = {c: True for c in categories}

        # Compute budget envelope
        budget_envelope = income - savings_target
        total_baseline = baselines.sum()

        if total_baseline <= budget_envelope:
            # No cuts needed — baselines fit within envelope
            allocations = [
                BudgetAllocation(
                    category=c,
                    budget=round(baselines[i], 2),
                    baseline=round(baselines[i], 2),
                    cut_amount=0.0,
                    cut_pct=0.0,
                    is_discretionary=is_discretionary.get(c, True),
                )
                for i, c in enumerate(categories)
            ]
            return OptimisationResult(
                allocations=allocations,
                total_budget=round(total_baseline, 2),
                savings_achieved=round(income - total_baseline, 2),
                solver_status="no_cuts_needed",
            )

        # Need to cut |gap| from discretionary categories
        gap = total_baseline - budget_envelope

        # Distribute cuts proportional to elasticity × baseline
        # (higher elasticity → larger share of cuts)
        disc_mask = np.array([is_discretionary.get(c, True) for c in categories])
        cut_weights = elasticities * baselines * disc_mask
        total_weight = cut_weights.sum()

        if total_weight <= 0:
            # Can't cut anything — force proportional cuts
            cut_weights = baselines * disc_mask
            total_weight = max(cut_weights.sum(), 1e-6)

        raw_cuts = gap * (cut_weights / total_weight)

        # Apply constraints: floors and max reduction cap
        ceilings = baselines * self.max_reduction_cap
        actual_cuts = np.minimum(raw_cuts, ceilings)
        actual_cuts = np.minimum(actual_cuts, baselines - floors)
        actual_cuts = np.maximum(actual_cuts, 0)

        # If cuts are insufficient, distribute remainder
        remaining = gap - actual_cuts.sum()
        if remaining > 0:
            # Try to squeeze more from uncapped categories
            headroom = np.minimum(
                baselines - floors - actual_cuts,
                baselines * 0.5,  # hard max 50% cut
            ).clip(min=0)
            headroom_total = headroom.sum()
            if headroom_total > 0:
                extra = remaining * (headroom / headroom_total)
                actual_cuts += np.minimum(extra, headroom)

        budgets = baselines - actual_cuts

        try:
            # Attempt scipy refinement
            from scipy.optimize import linprog

            # Objective: minimise weighted deviation from baseline
            c_obj = elasticities  # cut more where elastic

            # x = cuts per category (≥ 0)
            A_ub = None
            b_ub = None

            # Equality: total cuts = gap
            A_eq = np.ones((1, n))
            b_eq = np.array([gap])

            # Bounds: 0 ≤ x ≤ min(ceiling, baseline - floor)
            bounds = [
                (0, min(ceilings[i], max(baselines[i] - floors[i], 0)))
                for i in range(n)
            ]

            result = linprog(
                -c_obj,  # maximise weighted cuts (linprog minimises)
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

            if result.success:
                actual_cuts = result.x
                budgets = baselines - actual_cuts
                solver_status = "optimal"
            else:
                solver_status = "scipy_fallback"

        except (ImportError, Exception) as exc:
            logger.debug("scipy solver not available or failed: %s", exc)
            solver_status = "heuristic"

        allocations = []
        for i, c in enumerate(categories):
            cut_pct = (actual_cuts[i] / baselines[i] * 100) if baselines[i] > 0 else 0
            allocations.append(
                BudgetAllocation(
                    category=c,
                    budget=round(float(budgets[i]), 2),
                    baseline=round(float(baselines[i]), 2),
                    cut_amount=round(float(actual_cuts[i]), 2),
                    cut_pct=round(float(cut_pct), 1),
                    is_discretionary=is_discretionary.get(c, True),
                )
            )

        return OptimisationResult(
            allocations=allocations,
            total_budget=round(float(budgets.sum()), 2),
            savings_achieved=round(float(income - budgets.sum()), 2),
            solver_status=solver_status,
        )
