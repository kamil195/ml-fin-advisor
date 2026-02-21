"""
Text Tower for the Transaction Classifier (SPEC §6.2).

Frozen Sentence-Transformer backbone with a learned linear projection
layer that maps 384-d embeddings to a lower-dimensional space for fusion
with numerical/temporal features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TextTower:
    """
    Sentence-Transformer backbone + projection head.

    Parameters
    ----------
    model_name : str
        HuggingFace Sentence-Transformer identifier.
    embedding_dim : int
        Native embedding dimensionality of the backbone.
    projection_dim : int
        Output dimensionality after projection.
    freeze_backbone : bool
        If True, backbone weights are frozen during training.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        projection_dim: int = 128,
        freeze_backbone: bool = True,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.freeze_backbone = freeze_backbone
        self._encoder: Any = None
        self._projection_weight: np.ndarray | None = None
        self._projection_bias: np.ndarray | None = None

    def _load_encoder(self) -> None:
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
            logger.info("TextTower loaded: %s", self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — "
                "using random embeddings for development."
            )

    def _init_projection(self) -> None:
        """Xavier-style initialisation for projection layer."""
        if self._projection_weight is not None:
            return
        rng = np.random.default_rng(seed=42)
        fan_in, fan_out = self.embedding_dim, self.projection_dim
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self._projection_weight = rng.uniform(
            -limit, limit, (self.embedding_dim, self.projection_dim)
        ).astype(np.float32)
        self._projection_bias = np.zeros(self.projection_dim, dtype=np.float32)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Encode texts through backbone + projection.

        Returns shape ``(N, projection_dim)``.
        """
        self._load_encoder()
        self._init_projection()

        if self._encoder is not None:
            raw = self._encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        else:
            rng = np.random.default_rng(seed=hash(str(texts[:3])) % 2**31)
            raw = rng.standard_normal((len(texts), self.embedding_dim)).astype(
                np.float32
            )

        # Linear projection
        projected = raw @ self._projection_weight + self._projection_bias
        return projected

    def encode_from_df(self, df: "pd.DataFrame") -> np.ndarray:
        """Build input strings from DataFrame columns and encode."""
        import pandas as pd

        texts = (
            df["merchant_name"].fillna("")
            + " "
            + df["raw_description"].fillna("")
        ).str.strip().tolist()
        return self.encode(texts)
