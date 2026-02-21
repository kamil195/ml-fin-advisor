"""
Feature engineering pipeline for the ML Fin-Advisor.

Modules
-------
text_features
    Merchant embeddings, TF-IDF trigrams, MCC embeddings.
numerical_features
    Amount transforms, rolling aggregates, z-scores.
temporal_features
    Cyclical encodings, holiday flags, month phase.
behavioral_features
    Regime, impulse, habit, income-cycle, and drift features.
feature_store
    Feast integration for online/offline materialisation.
"""

from src.features.behavioral_features import extract_behavioral_features
from src.features.numerical_features import extract_numerical_features
from src.features.temporal_features import extract_temporal_features
from src.features.text_features import extract_text_features

__all__ = [
    "extract_behavioral_features",
    "extract_numerical_features",
    "extract_temporal_features",
    "extract_text_features",
]