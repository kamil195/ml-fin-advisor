"""
Text feature extraction for transaction classification.

Implements SPEC §5.2.1:
  - Merchant embedding via Sentence-Transformer
  - Character-trigram TF-IDF (SVD-reduced)
  - Learned MCC code embedding
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ── Merchant Embedding (Sentence-Transformer) ─────────────────────────────────


class MerchantEmbedder:
    """
    Encodes ``merchant_name + raw_description`` via a frozen
    Sentence-Transformer and an optional learned linear projection.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: ``all-MiniLM-L6-v2``).
    projection_dim : int | None
        If set, project the embedding to this dimensionality.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        projection_dim: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.projection_dim = projection_dim
        self._model: Any = None
        self._projection: np.ndarray | None = None

    # Lazy-load to keep import time fast
    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded Sentence-Transformer: %s", self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — "
                "falling back to random embeddings for development."
            )
            self._model = None

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Return (N, D) embedding matrix for a list of texts."""
        self._load_model()

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        else:
            # Deterministic fallback for offline dev / CI
            rng = np.random.default_rng(seed=42)
            embeddings = rng.standard_normal((len(texts), 384)).astype(np.float32)

        if self.projection_dim is not None:
            embeddings = self._apply_projection(embeddings)

        return embeddings

    def _apply_projection(self, embeddings: np.ndarray) -> np.ndarray:
        """PCA / random-projection to ``projection_dim``."""
        if self._projection is None:
            rng = np.random.default_rng(seed=0)
            raw_dim = embeddings.shape[1]
            self._projection = rng.standard_normal(
                (raw_dim, self.projection_dim)
            ).astype(np.float32)
            # Orthogonalise via QR
            q, _ = np.linalg.qr(self._projection)
            self._projection = q[:, : self.projection_dim]
        return embeddings @ self._projection

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build text inputs from a DataFrame with ``merchant_name`` and
        ``raw_description`` columns, then encode.
        """
        texts = (
            df["merchant_name"].fillna("")
            + " "
            + df["raw_description"].fillna("")
        ).str.strip().tolist()
        return self.encode(texts)


# ── Character-Trigram TF-IDF ───────────────────────────────────────────────────


class MerchantTrigramEncoder:
    """
    Character-trigram TF-IDF vectoriser with SVD dimensionality reduction.

    Parameters
    ----------
    max_features : int
        Maximum number of TF-IDF features before SVD.
    svd_components : int
        Number of SVD components to keep.
    """

    def __init__(
        self,
        max_features: int = 5_000,
        svd_components: int = 64,
    ) -> None:
        self.max_features = max_features
        self.svd_components = svd_components
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 3),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=svd_components, random_state=42)
        self._fitted = False

    def fit(self, texts: list[str]) -> MerchantTrigramEncoder:
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self._fitted = True
        logger.info(
            "Trigram encoder fitted — explained variance: %.2f%%",
            self.svd.explained_variance_ratio_.sum() * 100,
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        tfidf = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf).astype(np.float32)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        tfidf = self.vectorizer.fit_transform(texts)
        result = self.svd.fit_transform(tfidf).astype(np.float32)
        self._fitted = True
        return result

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        texts = df["merchant_name"].fillna("").tolist()
        return self.transform(texts)


# ── MCC Embedding ──────────────────────────────────────────────────────────────


class MCCEmbedder:
    """
    Learned embedding for Merchant Category Codes (MCC).

    During training, embeddings are jointly optimised with the classifier.
    For feature extraction, we expose a lookup table initialised via
    category-label supervision (or random if not yet trained).

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the MCC embedding vector.
    """

    def __init__(self, embedding_dim: int = 16) -> None:
        self.embedding_dim = embedding_dim
        self._label_encoder = LabelEncoder()
        self._embeddings: np.ndarray | None = None

    def fit(self, mcc_codes: np.ndarray | list[int]) -> MCCEmbedder:
        """Fit the label encoder and initialise random embeddings."""
        self._label_encoder.fit(mcc_codes)
        n_unique = len(self._label_encoder.classes_)
        rng = np.random.default_rng(seed=42)
        self._embeddings = rng.standard_normal(
            (n_unique, self.embedding_dim)
        ).astype(np.float32)
        # L2-normalise rows
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._embeddings /= np.maximum(norms, 1e-8)
        return self

    def transform(self, mcc_codes: np.ndarray | list[int]) -> np.ndarray:
        if self._embeddings is None:
            raise RuntimeError("Call fit() before transform().")
        indices = self._label_encoder.transform(mcc_codes)
        return self._embeddings[indices]

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        return self.transform(df["merchant_mcc"].values)


# ── Convenience: extract all text features ─────────────────────────────────────


def extract_text_features(
    df: pd.DataFrame,
    *,
    merchant_embedder: MerchantEmbedder | None = None,
    trigram_encoder: MerchantTrigramEncoder | None = None,
    mcc_embedder: MCCEmbedder | None = None,
    fit: bool = False,
) -> np.ndarray:
    """
    Extract and concatenate all text-derived features for a transaction
    DataFrame.

    Returns an ``(N, D)`` matrix where *D* ≈ 384 + 64 + 16 = 464 by default.
    """
    if merchant_embedder is None:
        merchant_embedder = MerchantEmbedder()
    if trigram_encoder is None:
        trigram_encoder = MerchantTrigramEncoder()
    if mcc_embedder is None:
        mcc_embedder = MCCEmbedder()

    # Sentence-Transformer embeddings
    emb = merchant_embedder.transform_df(df)

    # Trigram TF-IDF
    texts = df["merchant_name"].fillna("").tolist()
    if fit:
        trigram = trigram_encoder.fit_transform(texts)
    else:
        trigram = trigram_encoder.transform(texts)

    # MCC embeddings
    if fit:
        mcc_embedder.fit(df["merchant_mcc"].values)
    mcc = mcc_embedder.transform_df(df)

    return np.hstack([emb, trigram, mcc])
