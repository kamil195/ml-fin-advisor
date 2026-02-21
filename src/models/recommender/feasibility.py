"""
Behavioral feasibility check for budget recommendations (SPEC §8.2, Step 3).

Ensures that budget cuts are achievable given the user's habit strength,
historical variance, and compliance history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeasibilityResult:
    """Feasibility assessment for a single category."""

    category: str
    max_reduction_pct: float
    habit_strength: float
    variance_score: float
    compliance_score: float
    feasible: bool


class FeasibilityChecker:
    """
    Assess whether proposed budget cuts are behaviorally feasible.

    Parameters
    ----------
    habit_weight : float
        Weight of habit strength in feasibility score.
    variance_weight : float
        Weight of spending variance in feasibility score.
    compliance_weight : float
        Weight of historical compliance in feasibility score.
    max_reduction_cap : float
        Hard cap on maximum percentage reduction.
    """

    def __init__(
        self,
        habit_weight: float = 0.4,
        variance_weight: float = 0.35,
        compliance_weight: float = 0.25,
        max_reduction_cap: float = 0.30,
    ) -> None:
        self.habit_weight = habit_weight
        self.variance_weight = variance_weight
        self.compliance_weight = compliance_weight
        self.max_reduction_cap = max_reduction_cap

    def _compute_variance_score(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str,
    ) -> float:
        """
        Compute spending variance score (higher = more variable = more
        reducible).
        """
        mask = (df["user_id"] == user_id) & (df["amount"] < 0)
        if "category_l2" in df.columns:
            mask = mask & (df["category_l2"] == category)

        amounts = df.loc[mask, "amount"].abs()
        if len(amounts) < 3:
            return 0.5  # default

        cv = float(amounts.std() / amounts.mean()) if amounts.mean() > 0 else 0
        # Higher CV → more variable → potentially more reducible
        return min(cv, 1.0)

    def _compute_compliance_score(
        self,
        compliance_history: dict[str, float] | None,
        category: str,
    ) -> float:
        """
        Historical compliance (% of months user stayed within budget).

        Higher compliance → more feasible to reduce further.
        """
        if compliance_history is None:
            return 0.5  # no history

        return compliance_history.get(category, 0.5)

    def check(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str,
        habit_strength: float,
        proposed_reduction_pct: float,
        compliance_history: dict[str, float] | None = None,
    ) -> FeasibilityResult:
        """
        Check whether a proposed budget reduction is feasible.

        Parameters
        ----------
        df : pd.DataFrame
            Historical transaction data.
        user_id : str
            Target user.
        category : str
            Target category.
        habit_strength : float
            Habit strength index (0–1) from behavior model.
        proposed_reduction_pct : float
            Proposed percentage reduction (0–1).
        compliance_history : dict | None
            Historical budget compliance per category.
        """
        variance = self._compute_variance_score(df, user_id, category)
        compliance = self._compute_compliance_score(compliance_history, category)

        # Feasibility score: higher habit = harder to reduce
        # Higher variance = more room to reduce
        # Higher compliance = user can stick to budgets
        difficulty = (
            self.habit_weight * habit_strength
            + self.variance_weight * (1 - variance)
            + self.compliance_weight * (1 - compliance)
        )

        # Max reduction = inverse of difficulty, capped
        max_reduction = (1 - difficulty) * self.max_reduction_cap
        max_reduction = max(max_reduction, 0.05)  # always allow at least 5%

        feasible = proposed_reduction_pct <= max_reduction

        return FeasibilityResult(
            category=category,
            max_reduction_pct=round(max_reduction, 3),
            habit_strength=round(habit_strength, 3),
            variance_score=round(variance, 3),
            compliance_score=round(compliance, 3),
            feasible=feasible,
        )

    def check_all(
        self,
        df: pd.DataFrame,
        user_id: str,
        categories: list[str],
        habit_strengths: dict[str, float],
        proposed_reductions: dict[str, float],
        compliance_history: dict[str, float] | None = None,
    ) -> list[FeasibilityResult]:
        """Check feasibility for all categories."""
        results = []
        for cat in categories:
            result = self.check(
                df,
                user_id,
                cat,
                habit_strengths.get(cat, 0.5),
                proposed_reductions.get(cat, 0.0),
                compliance_history,
            )
            results.append(result)
        return results
