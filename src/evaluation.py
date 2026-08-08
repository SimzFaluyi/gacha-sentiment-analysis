"""
Evaluation utilities for sentiment-classification experiments.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, float]:
    """
    Calculate macro-averaged precision, recall and F1.
    """

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "macro_f1": float(f1),
    }


def create_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create a labelled confusion matrix."""

    if labels is None:
        labels = [
            "Negative",
            "Neutral",
            "Positive",
        ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )


def create_class_distribution(
    labels: Sequence[str],
) -> pd.Series:
    """Calculate class distribution."""

    return pd.Series(labels).value_counts()


def generate_classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> str:
    """Generate a text classification report."""

    return classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
