"""
Evaluation module (SPEC §6.4, §7.6, §8.6, §10.3, §12.2).

Provides metrics for classification, forecasting, recommendations,
fairness auditing, and champion/challenger model comparison.
"""

from src.evaluation.classification_metrics import (
    ClassificationMetrics,
    evaluate_classifier,
)
from src.evaluation.fairness_audit import FairnessMetrics, run_fairness_audit
from src.evaluation.forecast_metrics import ForecastMetrics, evaluate_forecasts
from src.evaluation.model_comparison import ComparisonResult, compare_models
from src.evaluation.recommendation_metrics import (
    RecommendationMetrics,
    evaluate_recommendations,
)

__all__ = [
    "ClassificationMetrics",
    "ComparisonResult",
    "FairnessMetrics",
    "ForecastMetrics",
    "RecommendationMetrics",
    "compare_models",
    "evaluate_classifier",
    "evaluate_forecasts",
    "evaluate_recommendations",
    "run_fairness_audit",
]