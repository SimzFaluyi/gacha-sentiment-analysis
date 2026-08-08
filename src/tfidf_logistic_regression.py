"""
TF-IDF + Logistic Regression sentiment classification.

This module implements the traditional supervised machine-learning
component of the proposed research framework.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class TfidfLogisticRegressionModel:
    """Container for the TF-IDF + Logistic Regression pipeline."""

    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    pipeline: Pipeline


def create_model(
    max_features: int = 5000,
    random_state: int = 42,
) -> TfidfLogisticRegressionModel:
    """
    Create the TF-IDF + Logistic Regression pipeline.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=(1, 2),
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )

    return TfidfLogisticRegressionModel(
        vectorizer=vectorizer,
        classifier=classifier,
        pipeline=pipeline,
    )


def train_model(
    texts: list[str],
    labels: list[str],
    max_features: int = 5000,
) -> TfidfLogisticRegressionModel:
    """Train the TF-IDF + Logistic Regression model."""

    model = create_model(
        max_features=max_features
    )

    model.pipeline.fit(
        texts,
        labels,
    )

    return model


def predict(
    model: TfidfLogisticRegressionModel,
    texts: list[str],
) -> list[str]:
    """Generate sentiment predictions."""

    return model.pipeline.predict(texts).tolist()
