"""
Expense Forecaster (SPEC §7).

Multi-horizon probabilistic forecasting with model tournament:
  Prophet (stable users) → N-BEATS (moderate) → TFT (complex).
"""

from src.models.forecaster.model_selector import ModelSelector
from src.models.forecaster.nbeats_model import NBeatsModel
from src.models.forecaster.prophet_model import ProphetModel
from src.models.forecaster.tft_model import TFTModel
from src.models.forecaster.train import ForecastTrainer

__all__ = [
    "ForecastTrainer",
    "ModelSelector",
    "NBeatsModel",
    "ProphetModel",
    "TFTModel",
]