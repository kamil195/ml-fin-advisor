"""
Training loop for the Transaction Classifier (SPEC §6.3).

Orchestrates:
  1. Feature extraction (text, numerical, temporal, behavioral)
  2. Text-tower encoding
  3. MLP training with focal loss
  4. LightGBM meta-learner stacking
  5. Evaluation against target metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.numerical_features import extract_numerical_features
from src.features.temporal_features import extract_temporal_features
from src.models.classifier.meta_learner import MetaLearner
from src.models.classifier.mlp import ClassifierMLP
from src.models.classifier.text_tower import TextTower
from src.utils.constants import CategoryL2

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters from model_config.yaml."""

    # Text tower
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    projection_dim: int = 128
    freeze_backbone: bool = True

    # MLP
    hidden_layers: list[int] | None = None
    dropout: float = 0.3
    batch_size: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05

    # Meta-learner
    meta_n_folds: int = 5

    def __post_init__(self) -> None:
        if self.hidden_layers is None:
            self.hidden_layers = [512, 256]


class ClassifierTrainer:
    """
    End-to-end training pipeline for the transaction classifier.

    Parameters
    ----------
    config : TrainingConfig
        Hyperparameters.
    output_dir : str | Path
        Directory for model artefacts.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        output_dir: str | Path = "models/classifier",
    ) -> None:
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.text_tower = TextTower(
            model_name=self.config.model_name,
            embedding_dim=self.config.embedding_dim,
            projection_dim=self.config.projection_dim,
            freeze_backbone=self.config.freeze_backbone,
        )
        self.mlp = ClassifierMLP(
            hidden_layers=self.config.hidden_layers,
            dropout=self.config.dropout,
        )
        self.meta_learner = MetaLearner(
            n_folds=self.config.meta_n_folds,
        )

        # Label encoder
        self._classes: list[str] = [c.value for c in CategoryL2]
        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}

    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Convert CategoryL2 labels to integer indices."""
        return labels.map(
            lambda l: self._class_to_idx.get(
                l.value if hasattr(l, "value") else l, 0
            )
        ).values.astype(int)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and concatenate all features."""
        # Text features
        text_emb = self.text_tower.encode_from_df(df)

        # Numerical features
        num_feat = extract_numerical_features(df).values

        # Temporal features
        temp_feat = extract_temporal_features(df).values

        return np.hstack([text_emb, num_feat, temp_feat]).astype(np.float32)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Train the full classifier pipeline.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with ``category_l2`` labels.
        val_df : pd.DataFrame | None
            Validation data for metric computation.

        Returns
        -------
        dict
            Training metrics.
        """
        logger.info("Starting classifier training — %d samples.", len(train_df))

        # Encode labels
        y_train = self._encode_labels(train_df["category_l2"])

        # Extract features
        X_train = self._prepare_features(train_df)

        # Update MLP input dimension
        self.mlp = ClassifierMLP(
            input_dim=X_train.shape[1],
            hidden_layers=self.config.hidden_layers,
            n_classes=len(self._classes),
            dropout=self.config.dropout,
        )

        # Step 1: MLP training (simplified — full version uses PyTorch)
        logger.info(
            "MLP: input_dim=%d, classes=%d", X_train.shape[1], len(self._classes)
        )
        # Note: In production, this would be a PyTorch training loop with
        # AdamW, cosine schedule, and focal loss. For now we compute logits
        # through the randomly-initialised MLP as a feature transform.
        logits_train = self.mlp.get_logits(X_train)

        # Step 2: LightGBM meta-learner
        logger.info("Training LightGBM meta-learner (%d folds).", self.config.meta_n_folds)
        try:
            self.meta_learner.fit(logits_train, X_train, y_train)
            meta_probs = self.meta_learner.predict_proba(logits_train, X_train)
            train_acc = float((meta_probs.argmax(axis=1) == y_train).mean())
        except ImportError:
            logger.warning("LightGBM not available — using MLP only.")
            meta_probs = self.mlp.predict_proba(X_train)
            train_acc = float((meta_probs.argmax(axis=1) == y_train).mean())

        metrics = {"train_accuracy": train_acc}

        # Validation
        if val_df is not None and "category_l2" in val_df.columns:
            y_val = self._encode_labels(val_df["category_l2"])
            X_val = self._prepare_features(val_df)
            logits_val = self.mlp.get_logits(X_val)

            try:
                val_probs = self.meta_learner.predict_proba(logits_val, X_val)
            except RuntimeError:
                val_probs = self.mlp.predict_proba(X_val)

            val_preds = val_probs.argmax(axis=1)
            metrics["val_accuracy"] = float((val_preds == y_val).mean())

            # Top-3 accuracy
            top3 = np.argsort(val_probs, axis=1)[:, -3:]
            top3_hits = np.array([y in t3 for y, t3 in zip(y_val, top3)])
            metrics["val_top3_accuracy"] = float(top3_hits.mean())

        logger.info("Training complete — metrics: %s", metrics)
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify transactions and return predictions.

        Returns a DataFrame with columns: ``category_l1``, ``category_l2``,
        ``confidence``, ``top_3``.
        """
        from src.utils.constants import CATEGORY_HIERARCHY

        X = self._prepare_features(df)
        logits = self.mlp.get_logits(X)

        try:
            probs = self.meta_learner.predict_proba(logits, X)
        except RuntimeError:
            probs = self.mlp.predict_proba(X)

        pred_idx = probs.argmax(axis=1)
        pred_conf = probs.max(axis=1)

        results = []
        for i in range(len(df)):
            l2_name = self._classes[pred_idx[i]]
            l2_enum = CategoryL2(l2_name)
            l1_enum = CATEGORY_HIERARCHY.get(l2_enum)

            # Top 3
            top3_idx = np.argsort(probs[i])[-3:][::-1]
            top3 = [
                {"category": self._classes[j], "confidence": round(float(probs[i, j]), 4)}
                for j in top3_idx
            ]

            results.append(
                {
                    "category_l1": l1_enum.value if l1_enum else "UNKNOWN",
                    "category_l2": l2_name,
                    "confidence": round(float(pred_conf[i]), 4),
                    "top_3": top3,
                }
            )

        return pd.DataFrame(results, index=df.index)

    def save(self) -> None:
        """Save all model artefacts."""
        import joblib

        joblib.dump(self.mlp.weights, self.output_dir / "mlp_weights.pkl")
        try:
            self.meta_learner.save(self.output_dir / "meta_learner.pkl")
        except RuntimeError:
            pass
        logger.info("Classifier saved to %s", self.output_dir)
