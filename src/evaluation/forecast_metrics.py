"""
Forecast evaluation metrics (SPEC §7.6).

Metrics:
  - MAPE (target ≤ 12% for 30-day)
  - CRPS (minimise)
  - Coverage of 90% prediction interval (target 85–95%)
  - WAPE (target ≤ 10%)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ForecastMetrics:
    """Complete forecast evaluation result."""

    mape: float = 0.0
    median_ape: float = 0.0
    wape: float = 0.0
    crps: float = 0.0
    coverage_90: float = 0.0
    n_series: int = 0
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def meets_targets(self, horizon_days: int = 30) -> dict[str, bool]:
        """Check whether SPEC targets are met."""
        mape_target = {30: 0.12, 60: 0.18, 90: 0.25}.get(horizon_days, 0.12)
        return {
            f"mape_le_{mape_target:.0%}": self.mape <= mape_target,
            "wape_le_10%": self.wape <= 0.10,
            "coverage_85_95%": 0.85 <= self.coverage_90 <= 0.95,
        }


def compute_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


def compute_wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Weighted Absolute Percentage Error (sum of |error| / sum of |actual|)."""
    total_actual = np.abs(y_true).sum()
    if total_actual == 0:
        return 0.0
    return float(np.abs(y_true - y_pred).sum() / total_actual)


def compute_crps(
    y_true: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> float:
    """
    Approximate CRPS using quantile forecasts.

    CRPS ≈ Σ_q 2 · (I(y ≤ q) - τ) · (q - y)
    where τ is the quantile level.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    crps_sum = 0.0
    for y, q10, q50, q90 in zip(y_true, p10, p50, p90):
        for q_val, tau in [(q10, 0.1), (q50, 0.5), (q90, 0.9)]:
            indicator = 1.0 if y <= q_val else 0.0
            crps_sum += 2 * abs(indicator - tau) * abs(q_val - y)

    return float(crps_sum / (n * 3))


def compute_coverage(
    y_true: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
) -> float:
    """Fraction of actuals within the 10th–90th percentile interval."""
    in_range = (y_true >= p10) & (y_true <= p90)
    return float(in_range.mean()) if len(y_true) > 0 else 0.0


def evaluate_forecasts(
    actuals: dict[str, float],
    predictions: dict[str, dict[str, float]],
) -> ForecastMetrics:
    """
    Evaluate forecast quality across categories.

    Parameters
    ----------
    actuals : dict[str, float]
        Actual spend per category.
    predictions : dict[str, dict[str, float]]
        Forecast per category with keys: p10, p50, p90.
    """
    categories = list(actuals.keys())
    y_true = np.array([actuals[c] for c in categories])
    y_pred = np.array([predictions[c]["p50"] for c in categories])
    p10 = np.array([predictions[c]["p10"] for c in categories])
    p90 = np.array([predictions[c]["p90"] for c in categories])

    mape = compute_mape(y_true, y_pred)
    wape = compute_wape(y_true, y_pred)
    crps = compute_crps(y_true, p10, y_pred, p90)
    coverage = compute_coverage(y_true, p10, p90)

    # Per-category breakdown
    per_cat = {}
    for i, c in enumerate(categories):
        if y_true[i] > 0:
            cat_ape = abs(y_true[i] - y_pred[i]) / y_true[i]
        else:
            cat_ape = 0.0
        per_cat[c] = {
            "actual": float(y_true[i]),
            "predicted": float(y_pred[i]),
            "ape": round(float(cat_ape), 4),
        }

    metrics = ForecastMetrics(
        mape=round(mape, 4),
        median_ape=round(float(np.median(np.abs(y_true - y_pred) / np.maximum(y_true, 1))), 4),
        wape=round(wape, 4),
        crps=round(crps, 4),
        coverage_90=round(coverage, 4),
        n_series=len(categories),
        per_category=per_cat,
    )

    logger.info(
        "Forecast eval: MAPE=%.2f%%, WAPE=%.2f%%, Coverage=%.1f%%",
        metrics.mape * 100, metrics.wape * 100, metrics.coverage_90 * 100,
    )
    return metrics
