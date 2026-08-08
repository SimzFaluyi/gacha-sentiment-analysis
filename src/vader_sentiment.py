"""
VADER sentiment-analysis implementation.

VADER is used as the rule-based baseline for the proposed research.
"""

from __future__ import annotations

from typing import Dict

from nltk.sentiment import SentimentIntensityAnalyzer


def create_vader_analyser() -> SentimentIntensityAnalyzer:
    """Create and return a VADER sentiment analyser."""

    return SentimentIntensityAnalyzer()


def classify_vader_score(compound_score: float) -> str:
    """
    Convert a VADER compound score into a three-class sentiment label.

    Thresholds follow the conventional VADER interpretation:

    compound >= 0.05  -> Positive
    compound <= -0.05 -> Negative
    otherwise          -> Neutral
    """

    if compound_score >= 0.05:
        return "Positive"

    if compound_score <= -0.05:
        return "Negative"

    return "Neutral"


def analyse_text(
    text: str,
    analyser: SentimentIntensityAnalyzer | None = None,
) -> Dict[str, float | str]:
    """Analyse a single text string using VADER."""

    if analyser is None:
        analyser = create_vader_analyser()

    scores = analyser.polarity_scores(text)

    sentiment = classify_vader_score(
        scores["compound"]
    )

    return {
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "positive": scores["pos"],
        "compound": scores["compound"],
        "sentiment": sentiment,
    }


def analyse_texts(texts: list[str]) -> list[Dict[str, float | str]]:
    """Analyse multiple text strings using VADER."""

    analyser = create_vader_analyser()

    return [
        analyse_text(
            text,
            analyser,
        )
        for text in texts
    ]
