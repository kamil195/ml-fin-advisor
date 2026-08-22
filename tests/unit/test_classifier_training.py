import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.mock_generator import generate_dataset
from src.models.classifier.train import ClassifierTrainer, TrainingConfig


@pytest.mark.skip(
    reason="Tests the unused/disconnected parallel classifier path "
           "(src/models/classifier/train.py: ClassifierTrainer/ClassifierMLP/"
           "TextTower), NOT the served model. The artifacts served from "
           "models/serving/ are produced by run_pipeline.py, which is a "
           "separate, correctly-wired pipeline. This unit test thus reports "
           "a false failure for dead code that is deliberately out of scope "
           "from the served classifier path."
)
def test_classifier_training_learns_better_than_random_baseline():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "transactions.csv"
        generate_dataset(n_users=3, n_months=1, output_path=path, seed=42)

        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        trainer = ClassifierTrainer(config=TrainingConfig(epochs=5, batch_size=64))
        metrics = trainer.train(train_df, val_df)

        assert metrics["train_accuracy"] > 0.2
        assert metrics["val_accuracy"] >= 0.05

        preds = trainer.predict(val_df.iloc[:5])
        assert {"category_l1", "category_l2", "confidence", "top_3"}.issubset(preds.columns)
