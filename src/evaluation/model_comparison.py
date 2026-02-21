"""
Champion / Challenger model comparison (SPEC §10.3).

Protocol:
  1. Evaluate on test set using primary metrics.
  2. Paired bootstrap test (n=10,000, α=0.05).
  3. Promote if significantly better on primary metric and not
     significantly worse on secondary metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of champion vs. challenger comparison."""

    metric_name: str
    champion_value: float
    challenger_value: float
    difference: float
    p_value: float
    significant: bool
    challenger_wins: bool


def paired_bootstrap_test(
    metric_fn: callable,
    y_true: np.ndarray,
    y_pred_champion: np.ndarray,
    y_pred_challenger: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> ComparisonResult:
    """
    Paired bootstrap significance test comparing two models.

    Parameters
    ----------
    metric_fn : callable
        Function(y_true, y_pred) → float.
    y_true : (N,) true labels.
    y_pred_champion : (N,) champion predictions.
    y_pred_challenger : (N,) challenger predictions.
    n_bootstrap : int
        Number of bootstrap iterations.
    alpha : float
        Significance level.
    higher_is_better : bool
        If True, challenger wins when metric is higher.

    Returns
    -------
    ComparisonResult
    """
    n = len(y_true)
    rng = np.random.default_rng(seed=42)

    champion_score = metric_fn(y_true, y_pred_champion)
    challenger_score = metric_fn(y_true, y_pred_challenger)
    observed_diff = challenger_score - champion_score

    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_champion = metric_fn(y_true[idx], y_pred_champion[idx])
        boot_challenger = metric_fn(y_true[idx], y_pred_challenger[idx])
        bootstrap_diffs[i] = boot_challenger - boot_champion

    if higher_is_better:
        p_value = float((bootstrap_diffs <= 0).mean())
        challenger_wins = p_value < alpha and observed_diff > 0
    else:
        p_value = float((bootstrap_diffs >= 0).mean())
        challenger_wins = p_value < alpha and observed_diff < 0

    return ComparisonResult(
        metric_name=metric_fn.__name__ if hasattr(metric_fn, "__name__") else "metric",
        champion_value=round(float(champion_score), 4),
        challenger_value=round(float(challenger_score), 4),
        difference=round(float(observed_diff), 4),
        p_value=round(float(p_value), 4),
        significant=p_value < alpha,
        challenger_wins=challenger_wins,
    )


def compare_models(
    y_true: np.ndarray,
    y_pred_champion: np.ndarray,
    y_pred_challenger: np.ndarray,
    y_prob_champion: np.ndarray | None = None,
    y_prob_challenger: np.ndarray | None = None,
    n_bootstrap: int = 10_000,
) -> list[ComparisonResult]:
    """
    Run full champion/challenger comparison with multiple metrics.

    Returns a list of ComparisonResult — one per metric.
    """
    from sklearn.metrics import accuracy_score, f1_score

    def macro_f1(y_t: np.ndarray, y_p: np.ndarray) -> float:
        return float(f1_score(y_t, y_p, average="macro", zero_division=0))

    def accuracy(y_t: np.ndarray, y_p: np.ndarray) -> float:
        return float(accuracy_score(y_t, y_p))

    results = [
        paired_bootstrap_test(
            macro_f1, y_true, y_pred_champion, y_pred_challenger,
            n_bootstrap=n_bootstrap, higher_is_better=True,
        ),
        paired_bootstrap_test(
            accuracy, y_true, y_pred_champion, y_pred_challenger,
            n_bootstrap=n_bootstrap, higher_is_better=True,
        ),
    ]

    for r in results:
        logger.info(
            "Model comparison [%s]: champion=%.4f, challenger=%.4f, "
            "p=%.4f, significant=%s, challenger_wins=%s",
            r.metric_name, r.champion_value, r.challenger_value,
            r.p_value, r.significant, r.challenger_wins,
        )

    return results
