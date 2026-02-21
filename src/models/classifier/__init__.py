"""
Transaction Classifier (SPEC §6).

Multi-class supervised learning pipeline with stacked architecture:
  text_tower → MLP fusion → LightGBM meta-learner → softmax (30 classes).
"""

from src.models.classifier.meta_learner import MetaLearner
from src.models.classifier.mlp import ClassifierMLP
from src.models.classifier.text_tower import TextTower
from src.models.classifier.train import ClassifierTrainer, TrainingConfig

__all__ = [
    "ClassifierMLP",
    "ClassifierTrainer",
    "MetaLearner",
    "TextTower",
    "TrainingConfig",
]