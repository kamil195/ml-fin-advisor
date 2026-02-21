"""
Habit Strength Index (SPEC §9.2.3).

Measures how ingrained a spending pattern is for a user × category pair.

  H(c, u) = α · Recurrence + β · Consistency + γ · Duration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HabitResult:
    """Habit strength for a single user × category."""

    user_id: str
    category: str
    habit_strength: float
    recurrence: float
    consistency: float
    duration_months: float


class HabitIndex:
    """
    Compute how habitual a user's spending is per category.

    Parameters
    ----------
    alpha : float
        Weight for recurrence component.
    beta : float
        Weight for consistency component.
    gamma : float
        Weight for duration component.
    duration_cap_months : int
        Maximum duration (in months) before the component saturates at 1.0.
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.35,
        gamma: float = 0.25,
        duration_cap_months: int = 12,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.duration_cap_months = duration_cap_months

    def compute(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str,
    ) -> HabitResult:
        """
        Compute habit strength for a single user × category.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``user_id``, ``category_l2``, ``timestamp``.
        """
        mask = (df["user_id"] == user_id) & (df["category_l2"] == category)
        subset = df.loc[mask].sort_values("timestamp")

        if len(subset) < 2:
            return HabitResult(
                user_id=user_id,
                category=category,
                habit_strength=0.0,
                recurrence=0.0,
                consistency=0.0,
                duration_months=0.0,
            )

        ts = pd.to_datetime(subset["timestamp"])
        span_days = (ts.max() - ts.min()).days or 1

        # Recurrence: frequency relative to expected (1 txn/week baseline)
        expected_weekly = span_days / 7.0
        actual = len(subset)
        recurrence = min(actual / max(expected_weekly, 1.0), 1.0)

        # Consistency: 1 − coefficient of variation of inter-purchase intervals
        gaps = ts.diff().dt.total_seconds().dropna()
        if len(gaps) >= 2 and gaps.mean() > 0:
            cv = float(gaps.std() / gaps.mean())
            consistency = max(1.0 - cv, 0.0)
        else:
            consistency = 0.0

        # Duration: months since first purchase, capped
        duration_months = min(
            span_days / 30.0, self.duration_cap_months
        )
        duration_norm = duration_months / self.duration_cap_months

        # Weighted sum
        h = (
            self.alpha * recurrence
            + self.beta * consistency
            + self.gamma * duration_norm
        )
        habit_strength = round(min(h, 1.0), 4)

        return HabitResult(
            user_id=user_id,
            category=category,
            habit_strength=habit_strength,
            recurrence=round(recurrence, 4),
            consistency=round(consistency, 4),
            duration_months=round(duration_months, 2),
        )

    def compute_all(self, df: pd.DataFrame) -> list[HabitResult]:
        """Compute habit strength for all user × category pairs."""
        results: list[HabitResult] = []

        if "category_l2" not in df.columns:
            logger.warning("category_l2 not in DataFrame — skipping habit index.")
            return results

        for (uid, cat) in df.groupby(["user_id", "category_l2"]).groups:
            result = self.compute(df, uid, cat)
            results.append(result)

        logger.info("Computed habit index for %d user×category pairs.", len(results))
        return results

    def compute_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute per-row habit strength aligned to the input index.

        This is the main entry point for the feature pipeline.
        """
        habit = pd.Series(0.0, index=df.index)

        if "category_l2" not in df.columns:
            return habit

        for (uid, cat), grp in df.groupby(["user_id", "category_l2"]):
            result = self.compute(df, uid, cat)
            habit.loc[grp.index] = result.habit_strength

        return habit
