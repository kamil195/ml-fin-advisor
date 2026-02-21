"""
Budget Recommender (SPEC §8).

Constraint-optimised, interpretable budget recommendations with
SHAP attributions, anchor rules, counterfactuals, and peer benchmarks.
"""

from src.models.recommender.budget_optimizer import BudgetOptimiser
from src.models.recommender.explanations import ExplanationEngine
from src.models.recommender.feasibility import FeasibilityChecker
from src.models.recommender.templates import render_recommendation, render_template

__all__ = [
    "BudgetOptimiser",
    "ExplanationEngine",
    "FeasibilityChecker",
    "render_recommendation",
    "render_template",
]