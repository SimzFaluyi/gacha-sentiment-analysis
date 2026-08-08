"""
Visualisation utilities for sentiment-analysis experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_class_distribution(
    labels: pd.Series,
    title: str = "Sentiment Class Distribution",
) -> None:
    """Plot sentiment-class distribution."""

    counts = labels.value_counts()

    plt.figure(figsize=(7, 5))

    sns.barplot(
        x=counts.index,
        y=counts.values,
    )

    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Records")
    plt.tight_layout()

    plt.show()


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    title: str = "Confusion Matrix",
    output_path: str | None = None,
) -> None:
    """Plot a confusion matrix."""

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
