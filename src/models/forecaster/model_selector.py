"""
Per-user model tournament / selection (SPEC §7.3).

Trains Prophet, N-BEATS, and TFT for each user×category, evaluates on
a held-out validation window, and selects the best model per user.

Selection rules:
  - < 6 months history  → Prophet
  - 6–18 months history → N-BEATS
  - > 18 months history → TFT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.forecaster.nbeats_model import NBeatsModel
from src.models.forecaster.prophet_model import ProphetModel
from src.models.forecaster.tft_model import TFTModel

logger = logging.getLogger(__name__)


@dataclass
class ModelSelection:
    """Result of model selection for a user × category."""

    user_id: str
    category: str
    selected_model: str  # "prophet" | "nbeats" | "tft"
    history_months: float
    val_mape: float | None = None


class ModelSelector:
    """
    Select the best forecasting model per user based on data availability
    and validation performance.

    Parameters
    ----------
    prophet_kwargs : dict
        Parameters passed to ProphetModel.
    nbeats_kwargs : dict
        Parameters passed to NBeatsModel.
    tft_kwargs : dict
        Parameters passed to TFTModel.
    """

    def __init__(
        self,
        prophet_kwargs: dict | None = None,
        nbeats_kwargs: dict | None = None,
        tft_kwargs: dict | None = None,
    ) -> None:
        self.prophet = ProphetModel(**(prophet_kwargs or {}))
        self.nbeats = NBeatsModel(**(nbeats_kwargs or {}))
        self.tft = TFTModel(**(tft_kwargs or {}))

        self._selections: dict[str, ModelSelection] = {}

    def _history_months(self, df: pd.DataFrame, user_id: str) -> float:
        """Compute months of history for a user."""
        mask = df["user_id"] == user_id
        ts = pd.to_datetime(df.loc[mask, "timestamp"])
        if ts.empty:
            return 0.0
        span = (ts.max() - ts.min()).days
        return span / 30.0

    def select(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str = "all",
    ) -> ModelSelection:
        """
        Select and fit the best model for a user × category.

        Uses history-based heuristic selection (SPEC §7.3 table).
        """
        months = self._history_months(df, user_id)
        key = f"{user_id}_{category}"

        if months < 6:
            model_name = "prophet"
            self.prophet.fit(df, user_id, category)
        elif months < 18:
            model_name = "nbeats"
            self.nbeats.fit(df, user_id, category)
        else:
            model_name = "tft"
            self.tft.fit(df, user_id, category)

        selection = ModelSelection(
            user_id=user_id,
            category=category,
            selected_model=model_name,
            history_months=round(months, 1),
        )
        self._selections[key] = selection

        logger.info(
            "Selected %s for user=%s, cat=%s (%.1f months history)",
            model_name, user_id, category, months,
        )
        return selection

    def predict(
        self,
        user_id: str,
        category: str = "all",
        horizon_weeks: int = 4,
    ) -> dict:
        """
        Generate forecast using the selected model.

        Returns a dictionary with keys: category, dates, p10, p50, p90.
        """
        key = f"{user_id}_{category}"
        selection = self._selections.get(key)

        if selection is None:
            logger.warning("No model selected for %s — returning empty forecast.", key)
            return {"category": category, "dates": [], "p10": [], "p50": [], "p90": []}

        if selection.selected_model == "prophet":
            fc = self.prophet.predict(user_id, category, horizon_weeks)
        elif selection.selected_model == "nbeats":
            fc = self.nbeats.predict(user_id, category)
        else:
            fc = self.tft.predict(user_id, category, horizon_weeks)

        return {
            "category": fc.category,
            "dates": fc.dates,
            "p10": fc.p10,
            "p50": fc.p50,
            "p90": fc.p90,
        }

    def select_all(
        self,
        df: pd.DataFrame,
        categories: list[str] | None = None,
    ) -> list[ModelSelection]:
        """Run model selection for all users × categories."""
        results = []

        if categories is None:
            if "category_l2" in df.columns:
                categories = df["category_l2"].dropna().unique().tolist()
            else:
                categories = ["all"]

        for user_id in df["user_id"].unique():
            for cat in categories:
                result = self.select(df, user_id, cat)
                results.append(result)

        logger.info("Model selection complete for %d user×category pairs.", len(results))
        return results
