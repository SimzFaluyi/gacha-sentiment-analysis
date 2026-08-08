"""
DistilBERT sentiment-analysis component.

This module provides the transformer-based inference framework.

Final empirical fine-tuning/evaluation requires an appropriately
authorised and ethically approved research dataset.
"""

from __future__ import annotations

from typing import Any


DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def create_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
) -> Any:
    """
    Create a Hugging Face sentiment-analysis pipeline.

    This uses a pretrained DistilBERT sentiment model for technical
    demonstration. It must not be presented as an empirically
    fine-tuned model for the gacha research task.
    """

    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model=model_name,
    )


def predict(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict]:
    """Generate transformer-based sentiment predictions."""

    classifier = create_pipeline(model_name)

    return classifier(texts)
