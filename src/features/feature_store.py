"""
Feature Store integration (SPEC §9.3).

Abstracts Feast offline/online stores for training and real-time inference.
Provides a local-file fallback when Feast is not configured.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStoreClient:
    """
    Lightweight wrapper around Feast (or a local Parquet fallback).

    Parameters
    ----------
    offline_path : str | Path
        Directory for Parquet-based offline store.
    use_feast : bool
        Attempt to use Feast. Falls back to local storage on import failure.
    """

    def __init__(
        self,
        offline_path: str | Path = "data/features",
        use_feast: bool = False,
    ) -> None:
        self.offline_path = Path(offline_path)
        self.offline_path.mkdir(parents=True, exist_ok=True)
        self._feast_store: Any = None

        if use_feast:
            try:
                from feast import FeatureStore

                self._feast_store = FeatureStore(repo_path=".")
                logger.info("Connected to Feast feature store.")
            except (ImportError, Exception) as exc:
                logger.warning("Feast unavailable (%s) — using local Parquet.", exc)

    # ── Offline (training) ─────────────────────────────────────────────────

    def save_offline(
        self,
        features: pd.DataFrame,
        feature_group: str,
    ) -> Path:
        """Persist a feature DataFrame to the offline store."""
        path = self.offline_path / f"{feature_group}.parquet"
        features.to_parquet(path, index=True)
        logger.info("Saved %d rows to %s", len(features), path)
        return path

    def load_offline(self, feature_group: str) -> pd.DataFrame:
        """Load a feature group from the offline store."""
        path = self.offline_path / f"{feature_group}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature group not found: {path}")
        return pd.read_parquet(path)

    # ── Online (inference) ─────────────────────────────────────────────────

    def get_online_features(
        self,
        entity_rows: list[dict[str, Any]],
        feature_refs: list[str],
    ) -> pd.DataFrame:
        """
        Retrieve features for given entities from the online store.

        Falls back to the offline store when Feast is not available.
        """
        if self._feast_store is not None:
            response = self._feast_store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            )
            return response.to_df()

        # Fallback: scan offline Parquet files
        logger.debug("Online store unavailable — scanning offline store.")
        frames: list[pd.DataFrame] = []
        for ref in feature_refs:
            group = ref.split(":")[0] if ":" in ref else ref
            try:
                df = self.load_offline(group)
                frames.append(df)
            except FileNotFoundError:
                logger.warning("Feature group '%s' not found offline.", group)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1)

    # ── Materialisation ────────────────────────────────────────────────────

    def materialise(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> None:
        """Trigger Feast materialisation (online store refresh)."""
        if self._feast_store is not None:
            self._feast_store.materialize(
                start_date=start_date or datetime(2020, 1, 1),
                end_date=end_date or datetime.utcnow(),
            )
            logger.info("Feast materialisation complete.")
        else:
            logger.info("Feast not available — materialisation skipped.")

    def list_feature_groups(self) -> list[str]:
        """List available offline feature groups."""
        return [
            p.stem
            for p in self.offline_path.glob("*.parquet")
        ]
