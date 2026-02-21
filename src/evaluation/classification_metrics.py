"""
Classification evaluation metrics (SPEC §6.4).

Metrics:
  - Macro-F1 (target ≥ 0.92)
  - Top-3 accuracy (target ≥ 0.98)
  - Expected Calibration Error (ECE, target ≤ 0.03)
  - Per-class Recall (min ≥ 0.80)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Complete classification evaluation result."""

    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    top3_accuracy: float = 0.0
    ece: float = 0.0
    per_class_recall: dict[str, float] = field(default_factory=dict)
    min_class_recall: float = 0.0
    accuracy: float = 0.0
    n_samples: int = 0

    def meets_targets(self) -> dict[str, bool]:
        """Check whether SPEC targets are met."""
        return {
            "macro_f1_ge_0.92": self.macro_f1 >= 0.92,
            "top3_accuracy_ge_0.98": self.top3_accuracy >= 0.98,
            "ece_le_0.03": self.ece <= 0.03,
            "min_recall_ge_0.80": self.min_class_recall >= 0.80,
        }


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int = 3,
) -> float:
    """Compute top-k accuracy."""
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    hits = np.array([y in tk for y, tk in zip(y_true, top_k)])
    return float(hits.mean())


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Partitions predictions into ``n_bins`` by confidence level and
    measures the gap between confidence and accuracy in each bin.
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        bin_size = in_bin.sum()
        if bin_size == 0:
            continue

        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = accuracies[in_bin].mean()
        ece += (bin_size / len(y_true)) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_per_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute recall per class."""
    _, recalls, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    if class_names is None:
        class_names = [str(i) for i in range(len(recalls))]

    return {name: round(float(r), 4) for name, r in zip(class_names, recalls)}


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> ClassificationMetrics:
    """
    Run full classification evaluation.

    Parameters
    ----------
    y_true : (N,) integer labels
    y_pred : (N,) predicted integer labels
    y_prob : (N, C) predicted probabilities (optional)
    class_names : list[str] | None
        Human-readable class names
    """
    macro_f1 = compute_macro_f1(y_true, y_pred)
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    accuracy = float((y_true == y_pred).mean())
    per_class_recall = compute_per_class_recall(y_true, y_pred, class_names)
    min_recall = min(per_class_recall.values()) if per_class_recall else 0.0

    top3_acc = 0.0
    ece = 0.0
    if y_prob is not None:
        top3_acc = compute_top_k_accuracy(y_true, y_prob, k=3)
        ece = compute_ece(y_true, y_prob)

    metrics = ClassificationMetrics(
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        top3_accuracy=round(top3_acc, 4),
        ece=round(ece, 4),
        per_class_recall=per_class_recall,
        min_class_recall=round(min_recall, 4),
        accuracy=round(accuracy, 4),
        n_samples=len(y_true),
    )

    logger.info(
        "Classification eval: F1=%.4f, Top3=%.4f, ECE=%.4f, MinRecall=%.4f",
        metrics.macro_f1, metrics.top3_accuracy, metrics.ece, metrics.min_class_recall,
    )
    return metrics
