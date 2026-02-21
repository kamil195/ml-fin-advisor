"""
Forecast training loop (SPEC §7.5, §10).

Orchestrates model tournament, evaluation, and artefact persistence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.forecaster.model_selector import ModelSelector

logger = logging.getLogger(__name__)


@dataclass
class ForecastMetrics:
    """Aggregate forecast evaluation metrics."""

    user_id: str
    category: str
    model: str
    mape: float
    coverage_90: float  # % of actuals within p10–p90


class ForecastTrainer:
    """
    End-to-end training pipeline for expense forecasting.

    Parameters
    ----------
    output_dir : str | Path
        Directory for model artefacts.
    val_horizon_weeks : int
        Number of weeks to hold out for validation.
    """

    def __init__(
        self,
        output_dir: str | Path = "models/forecaster",
        val_horizon_weeks: int = 4,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_horizon_weeks = val_horizon_weeks
        self.selector = ModelSelector()

    def _temporal_split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and validation by time."""
        ts = pd.to_datetime(df["timestamp"])
        cutoff = ts.max() - pd.Timedelta(weeks=self.val_horizon_weeks)
        train = df[ts <= cutoff]
        val = df[ts > cutoff]
        return train, val

    def _compute_mape(
        self,
        actual: float,
        predicted: float,
    ) -> float:
        """Compute MAPE for a single value."""
        if actual == 0:
            return 0.0 if predicted == 0 else 1.0
        return abs(actual - predicted) / abs(actual)

    def train(
        self,
        df: pd.DataFrame,
        categories: list[str] | None = None,
    ) -> list[ForecastMetrics]:
        """
        Train and evaluate forecast models for all users and categories.

        Uses an expanding-window strategy with temporal validation.
        """
        train_df, val_df = self._temporal_split(df)

        logger.info(
            "Forecast training: %d train rows, %d val rows.",
            len(train_df), len(val_df),
        )

        if categories is None:
            if "category_l2" in df.columns:
                categories = df["category_l2"].dropna().unique().tolist()[:10]
            else:
                categories = ["all"]

        # Run model selection and training
        selections = self.selector.select_all(train_df, categories)

        # Evaluate
        metrics: list[ForecastMetrics] = []
        for sel in selections:
            forecast = self.selector.predict(
                sel.user_id,
                sel.category,
                self.val_horizon_weeks,
            )

            # Compute actual spend in validation window
            mask = (val_df["user_id"] == sel.user_id) & (val_df["amount"] < 0)
            if sel.category != "all" and "category_l2" in val_df.columns:
                mask = mask & (val_df["category_l2"] == sel.category)

            actual_spend = float(val_df.loc[mask, "amount"].abs().sum())

            if forecast["p50"]:
                predicted = sum(forecast["p50"])
                mape = self._compute_mape(actual_spend, predicted)

                # Coverage: is actual within p10–p90 range?
                p10_total = sum(forecast["p10"])
                p90_total = sum(forecast["p90"])
                in_range = 1.0 if p10_total <= actual_spend <= p90_total else 0.0
            else:
                mape = 1.0
                in_range = 0.0

            fm = ForecastMetrics(
                user_id=sel.user_id,
                category=sel.category,
                model=sel.selected_model,
                mape=round(mape, 4),
                coverage_90=in_range,
            )
            metrics.append(fm)

        # Summary
        if metrics:
            avg_mape = np.mean([m.mape for m in metrics])
            avg_coverage = np.mean([m.coverage_90 for m in metrics])
            logger.info(
                "Forecast evaluation: avg MAPE=%.2f%%, avg coverage=%.1f%%",
                avg_mape * 100, avg_coverage * 100,
            )

        return metrics
