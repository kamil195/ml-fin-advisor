"""
Multi-modal fusion MLP for the Transaction Classifier (SPEC §6.2).

Concatenates text-tower output, numerical features, and temporal/behavioral
features, then passes through a 2-layer MLP (512 → 256 → 30 classes).

Uses focal loss (γ=2.0) with optional label smoothing for class imbalance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def focal_loss(
    probs: np.ndarray,
    targets: np.ndarray,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> float:
    """
    Focal loss: FL(p_t) = -(1 - p_t)^γ · log(p_t).

    Parameters
    ----------
    probs : (N, C) predicted probabilities
    targets : (N,) integer class labels
    gamma : focusing parameter
    label_smoothing : uniform smoothing factor
    """
    n, c = probs.shape
    eps = 1e-7

    if label_smoothing > 0:
        one_hot = np.eye(c)[targets]
        smoothed = one_hot * (1 - label_smoothing) + label_smoothing / c
    else:
        smoothed = np.eye(c)[targets]

    pt = (probs * smoothed).sum(axis=-1).clip(eps, 1 - eps)
    loss = -((1 - pt) ** gamma) * np.log(pt)
    return float(loss.mean())


class ClassifierMLP:
    """
    2-layer MLP for multi-class transaction classification.

    Architecture: input → Dense(512) → GELU → Dropout → Dense(256)
    → GELU → Dropout → Dense(n_classes)

    This is a NumPy reference implementation for inference. Training
    should use the PyTorch version in ``train.py``.

    Parameters
    ----------
    input_dim : int
        Total concatenated feature dimension.
    hidden_layers : list[int]
        Hidden layer sizes (default: [512, 256]).
    n_classes : int
        Number of output classes (default: 30).
    dropout : float
        Dropout rate (applied during training only).
    """

    def __init__(
        self,
        input_dim: int = 464,
        hidden_layers: list[int] | None = None,
        n_classes: int = 30,
        dropout: float = 0.3,
    ) -> None:
        if hidden_layers is None:
            hidden_layers = [512, 256]

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.dropout = dropout
        self.weights: list[dict[str, np.ndarray]] = []
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialisation for all layers."""
        rng = np.random.default_rng(seed=42)
        dims = [self.input_dim] + self.hidden_layers + [self.n_classes]

        self.weights = []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append({"W": W, "b": b})

    def forward(
        self,
        X: np.ndarray,
        training: bool = False,
    ) -> np.ndarray:
        """
        Forward pass through the MLP.

        Returns logits of shape ``(N, n_classes)``.
        """
        h = X.astype(np.float32)

        for i, layer in enumerate(self.weights):
            h = h @ layer["W"] + layer["b"]
            if i < len(self.weights) - 1:
                h = gelu(h)
                if training and self.dropout > 0:
                    rng = np.random.default_rng()
                    mask = rng.binomial(1, 1 - self.dropout, h.shape)
                    h = h * mask / (1 - self.dropout)

        return h

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities."""
        logits = self.forward(X, training=False)
        return softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return self.predict_proba(X).argmax(axis=-1)

    def load_weights(self, weights: list[dict[str, np.ndarray]]) -> None:
        """Load pre-trained weights."""
        self.weights = weights

    def get_logits(self, X: np.ndarray) -> np.ndarray:
        """Return raw logits (used by the meta-learner as input features)."""
        return self.forward(X, training=False)
