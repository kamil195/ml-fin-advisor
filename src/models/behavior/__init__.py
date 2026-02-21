"""
Behavior Modeling Layer (SPEC §9).

Cross-cutting layer that captures latent user spending behaviors and
provides behavioral features consumed by all three ML modules.

Components
----------
regime_detector
    Bayesian Online Changepoint Detection for spending regime shifts.
impulse_scorer
    Logistic-regression model for impulse-purchase detection.
habit_index
    Habit strength computation (recurrence × consistency × duration).
income_cycle
    Pay-cycle detection and alignment.
"""

from src.models.behavior.habit_index import HabitIndex
from src.models.behavior.impulse_scorer import ImpulseScorer
from src.models.behavior.income_cycle import IncomeCycleDetector
from src.models.behavior.regime_detector import BayesianChangepointDetector

__all__ = [
    "BayesianChangepointDetector",
    "HabitIndex",
    "ImpulseScorer",
    "IncomeCycleDetector",
]