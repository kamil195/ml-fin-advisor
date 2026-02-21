"""
Recommendation evaluation metrics (SPEC §8.6).

Metrics:
  - Acceptance Rate (target ≥ 78%)
  - Budget Adherence (target ≥ 65%)
  - Explanation Helpfulness (target ≥ 4.0 / 5.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecommendationMetrics:
    """Complete recommendation evaluation result."""

    acceptance_rate: float = 0.0
    adherence_rate: float = 0.0
    avg_helpfulness: float = 0.0
    savings_goal_hit_rate: float = 0.0
    n_users: int = 0
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def meets_targets(self) -> dict[str, bool]:
        """Check whether SPEC targets are met."""
        return {
            "acceptance_ge_78%": self.acceptance_rate >= 0.78,
            "adherence_ge_65%": self.adherence_rate >= 0.65,
            "helpfulness_ge_4.0": self.avg_helpfulness >= 4.0,
            "savings_goal_ge_40%": self.savings_goal_hit_rate >= 0.40,
        }


def compute_acceptance_rate(
    recommendations: list[dict],
    user_actions: list[dict],
) -> float:
    """
    Fraction of recommended budgets kept without user modification.

    Parameters
    ----------
    recommendations : list[dict]
        Each with keys: user_id, category, recommended_budget.
    user_actions : list[dict]
        Each with keys: user_id, category, action ('accepted'|'modified'|'ignored').
    """
    if not recommendations:
        return 0.0

    action_lookup = {
        (a["user_id"], a["category"]): a["action"]
        for a in user_actions
    }

    accepted = sum(
        1
        for r in recommendations
        if action_lookup.get((r["user_id"], r["category"])) == "accepted"
    )

    return accepted / len(recommendations)


def compute_adherence_rate(
    budgets: list[dict],
    actual_spend: list[dict],
) -> float:
    """
    Fraction of months where user stayed within recommended budget.

    Parameters
    ----------
    budgets : list[dict]
        Each with keys: user_id, category, period, budget.
    actual_spend : list[dict]
        Each with keys: user_id, category, period, spend.
    """
    if not budgets:
        return 0.0

    spend_lookup = {
        (s["user_id"], s["category"], s["period"]): s["spend"]
        for s in actual_spend
    }

    within = sum(
        1
        for b in budgets
        if spend_lookup.get(
            (b["user_id"], b["category"], b["period"]), float("inf")
        )
        <= b["budget"]
    )

    return within / len(budgets)


def evaluate_recommendations(
    recommendations: list[dict],
    user_actions: list[dict] | None = None,
    budgets: list[dict] | None = None,
    actual_spend: list[dict] | None = None,
    helpfulness_scores: list[float] | None = None,
    savings_outcomes: list[bool] | None = None,
) -> RecommendationMetrics:
    """
    Evaluate recommendation quality.

    All parameters except ``recommendations`` are optional; metrics that
    can't be computed are left at 0.0.
    """
    acceptance = (
        compute_acceptance_rate(recommendations, user_actions)
        if user_actions
        else 0.0
    )
    adherence = (
        compute_adherence_rate(budgets, actual_spend)
        if budgets and actual_spend
        else 0.0
    )
    helpfulness = (
        float(np.mean(helpfulness_scores))
        if helpfulness_scores
        else 0.0
    )
    savings_rate = (
        float(np.mean(savings_outcomes))
        if savings_outcomes
        else 0.0
    )

    n_users = len({r["user_id"] for r in recommendations}) if recommendations else 0

    metrics = RecommendationMetrics(
        acceptance_rate=round(acceptance, 4),
        adherence_rate=round(adherence, 4),
        avg_helpfulness=round(helpfulness, 2),
        savings_goal_hit_rate=round(savings_rate, 4),
        n_users=n_users,
    )

    logger.info(
        "Recommendation eval: acceptance=%.1f%%, adherence=%.1f%%",
        metrics.acceptance_rate * 100, metrics.adherence_rate * 100,
    )
    return metrics
