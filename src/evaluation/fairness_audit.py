"""
Fairness audit module (SPEC §12.1–12.2).

Stratified evaluation across income quintiles, account age, and
geographic region. Computes equal-opportunity difference, demographic
parity, and recommendation aggressiveness parity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FairnessMetrics:
    """Fairness audit results."""

    equal_opportunity_diff: float = 0.0
    demographic_parity_diff: float = 0.0
    recommendation_aggressiveness_parity: float = 0.0
    stratified_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def meets_targets(self) -> dict[str, bool]:
        return {
            "equal_opportunity_le_0.05": self.equal_opportunity_diff <= 0.05,
            "demographic_parity_le_0.08": self.demographic_parity_diff <= 0.08,
            "aggressiveness_within_10%": self.recommendation_aggressiveness_parity <= 0.10,
        }


def compute_equal_opportunity_diff(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Compute maximum recall difference across groups.

    Equal opportunity requires positive class recall to be similar
    across all groups.
    """
    unique_groups = np.unique(groups)
    recalls = []

    for g in unique_groups:
        mask = groups == g
        g_true = y_true[mask]
        g_pred = y_pred[mask]

        if g_true.sum() == 0:
            continues = True
            recalls.append(0.0)
            continue

        recall = float((g_true & g_pred).sum() / g_true.sum())
        recalls.append(recall)

    if len(recalls) < 2:
        return 0.0

    return float(max(recalls) - min(recalls))


def compute_demographic_parity_diff(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Compute maximum positive-prediction-rate difference across groups.
    """
    unique_groups = np.unique(groups)
    rates = []

    for g in unique_groups:
        mask = groups == g
        rate = float(y_pred[mask].mean())
        rates.append(rate)

    if len(rates) < 2:
        return 0.0

    return float(max(rates) - min(rates))


def compute_aggressiveness_parity(
    recommendations: pd.DataFrame,
    group_col: str,
) -> float:
    """
    Compute max difference in mean % cut suggested across groups.

    ``recommendations`` must have columns: ``group_col``, ``cut_pct``.
    """
    if recommendations.empty or group_col not in recommendations.columns:
        return 0.0

    group_means = recommendations.groupby(group_col)["cut_pct"].mean()

    if len(group_means) < 2:
        return 0.0

    mean_cut = group_means.mean()
    if mean_cut == 0:
        return 0.0

    return float((group_means.max() - group_means.min()) / mean_cut)


def run_fairness_audit(
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    impulse_labels: np.ndarray | None = None,
    impulse_groups: np.ndarray | None = None,
    recommendations: pd.DataFrame | None = None,
    group_col: str = "income_quintile",
) -> FairnessMetrics:
    """
    Run a comprehensive fairness audit.

    All parameters are optional; metrics that can't be computed are
    left at 0.0.
    """
    eo_diff = 0.0
    dp_diff = 0.0
    agg_parity = 0.0

    if y_true is not None and y_pred is not None and groups is not None:
        eo_diff = compute_equal_opportunity_diff(y_true, y_pred, groups)

    if impulse_labels is not None and impulse_groups is not None:
        dp_diff = compute_demographic_parity_diff(impulse_labels, impulse_groups)

    if recommendations is not None:
        agg_parity = compute_aggressiveness_parity(recommendations, group_col)

    metrics = FairnessMetrics(
        equal_opportunity_diff=round(eo_diff, 4),
        demographic_parity_diff=round(dp_diff, 4),
        recommendation_aggressiveness_parity=round(agg_parity, 4),
    )

    logger.info(
        "Fairness audit: EO=%.4f, DP=%.4f, AggParity=%.4f",
        metrics.equal_opportunity_diff,
        metrics.demographic_parity_diff,
        metrics.recommendation_aggressiveness_parity,
    )
    return metrics
