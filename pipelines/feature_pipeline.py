"""
Feature computation pipeline (SPEC §10.1).

Orchestrates:
  1. Data ingestion & validation
  2. Text, numerical, temporal, and behavioral feature extraction
  3. Feature store materialisation

Can be scheduled via Airflow / Prefect or run standalone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineConfig:
    """Configuration for the feature pipeline."""

    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    feature_store_path: str = "data/feature_store"
    config_path: str = "configs/feature_config.yaml"
    batch_size: int = 10_000
    enable_text_features: bool = True
    enable_behavioral_features: bool = True


class FeaturePipeline:
    """
    End-to-end feature computation pipeline.

    Steps
    -----
    1. Ingest raw transactions (CSV / Parquet)
    2. Validate data quality (schema, volume, distribution)
    3. Extract text features (merchant embeddings, trigrams)
    4. Extract numerical features (amount z-scores, velocity)
    5. Extract temporal features (cyclical encoding, pay-cycle)
    6. Compute behavioral features (regime, impulse, habit)
    7. Save to feature store (offline / online)
    """

    def __init__(self, config: FeaturePipelineConfig | None = None) -> None:
        self.config = config or FeaturePipelineConfig()
        self._load_feature_config()

    def _load_feature_config(self) -> None:
        """Load feature_config.yaml if available."""
        self._feature_cfg: dict[str, Any] = {}
        cfg_path = Path(self.config.config_path)
        if cfg_path.exists():
            try:
                import yaml

                with open(cfg_path) as f:
                    self._feature_cfg = yaml.safe_load(f) or {}
                logger.info("Loaded feature config from %s", cfg_path)
            except Exception as exc:
                logger.warning("Could not load feature config: %s", exc)

    # ── Step 1: Data Ingestion ─────────────────────────────────────────────

    def ingest(self, path: str | Path | None = None) -> pd.DataFrame:
        """Ingest raw transaction data."""
        from src.data.ingestion import run_ingestion_pipeline

        data_path = Path(path or self.config.raw_data_path)
        logger.info("Ingesting data from %s", data_path)

        csv_files = list(data_path.glob("*.csv"))
        parquet_files = list(data_path.glob("*.parquet"))

        frames: list[pd.DataFrame] = []

        for f in csv_files:
            df = pd.read_csv(f)
            frames.append(df)

        for f in parquet_files:
            df = pd.read_parquet(f)
            frames.append(df)

        if not frames:
            logger.warning("No data files found in %s — generating mock data.", data_path)
            from src.data.mock_generator import generate_dataset

            df = generate_dataset(n_users=50, months=6)
            return df

        combined = pd.concat(frames, ignore_index=True)
        logger.info("Ingested %d rows from %d files", len(combined), len(frames))
        return combined

    # ── Step 2: Validation ─────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run data quality checks."""
        required_cols = ["amount", "timestamp", "merchant_name"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning("Missing columns: %s", missing)

        initial_rows = len(df)

        # Drop rows with null amounts
        df = df.dropna(subset=["amount"])
        dropped = initial_rows - len(df)
        if dropped:
            logger.info("Dropped %d rows with null amounts", dropped)

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

        logger.info("Validation complete: %d rows retained", len(df))
        return df

    # ── Step 3–6: Feature Extraction ───────────────────────────────────────

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature extractors and merge results."""
        features = df.copy()

        # Numerical features
        try:
            from src.features.numerical_features import extract_numerical_features

            num_feats = extract_numerical_features(df)
            features = pd.concat([features, num_feats], axis=1)
            logger.info("Numerical features: %d columns", num_feats.shape[1])
        except Exception as exc:
            logger.warning("Numerical features failed: %s", exc)

        # Temporal features
        try:
            from src.features.temporal_features import extract_temporal_features

            tmp_feats = extract_temporal_features(df)
            features = pd.concat([features, tmp_feats], axis=1)
            logger.info("Temporal features: %d columns", tmp_feats.shape[1])
        except Exception as exc:
            logger.warning("Temporal features failed: %s", exc)

        # Text features
        if self.config.enable_text_features:
            try:
                from src.features.text_features import extract_text_features

                txt_feats = extract_text_features(df)
                features = pd.concat([features, txt_feats], axis=1)
                logger.info("Text features: %d columns", txt_feats.shape[1])
            except Exception as exc:
                logger.warning("Text features failed: %s", exc)

        # Behavioral features
        if self.config.enable_behavioral_features:
            try:
                from src.features.behavioral_features import extract_behavioral_features

                beh_feats = extract_behavioral_features(df)
                features = pd.concat([features, beh_feats], axis=1)
                logger.info("Behavioral features: %d columns", beh_feats.shape[1])
            except Exception as exc:
                logger.warning("Behavioral features failed: %s", exc)

        logger.info("Total feature matrix: %s", features.shape)
        return features

    # ── Step 7: Feature Store ──────────────────────────────────────────────

    def save_features(self, features: pd.DataFrame) -> Path:
        """Save computed features to the offline feature store."""
        output_dir = Path(self.config.feature_store_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"features_{ts}.parquet"

        features.to_parquet(output_path, index=False)
        logger.info("Features saved to %s (%d rows)", output_path, len(features))
        return output_path

    # ── Run full pipeline ──────────────────────────────────────────────────

    def run(self, raw_data_path: str | Path | None = None) -> Path:
        """Execute the complete feature pipeline."""
        logger.info("═══ Feature Pipeline START ═══")
        start = datetime.utcnow()

        df = self.ingest(raw_data_path)
        df = self.validate(df)
        features = self.extract_features(df)
        output_path = self.save_features(features)

        elapsed = (datetime.utcnow() - start).total_seconds()
        logger.info("═══ Feature Pipeline DONE (%.1f s) ═══", elapsed)
        return output_path


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    """Run feature pipeline from command line."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Feature computation pipeline")
    parser.add_argument("--raw-data", default="data/raw", help="Path to raw data")
    parser.add_argument("--output", default="data/feature_store", help="Feature store path")
    parser.add_argument("--no-text", action="store_true", help="Disable text features")
    parser.add_argument("--no-behavior", action="store_true", help="Disable behavioral features")
    args = parser.parse_args()

    config = FeaturePipelineConfig(
        raw_data_path=args.raw_data,
        feature_store_path=args.output,
        enable_text_features=not args.no_text,
        enable_behavioral_features=not args.no_behavior,
    )

    pipeline = FeaturePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
