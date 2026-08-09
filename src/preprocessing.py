"""
Text preprocessing utilities for the gacha sentiment analysis project.

The functions in this module are designed to provide conservative
preprocessing for informal social-media text.

No Reddit data are included in this repository.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


DEFAULT_TEXT_COLUMN = "text"


def validate_input_dataframe(
    dataframe: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> None:
    """
    Validate that the input dataframe contains the required text column.

    Parameters
    ----------
    dataframe:
        Input pandas DataFrame.
    text_column:
        Name of the column containing text.

    Raises
    ------
    TypeError
        If dataframe is not a pandas DataFrame.
    ValueError
        If the required text column is missing.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if text_column not in dataframe.columns:
        raise ValueError(
            f"Required text column '{text_column}' was not found."
        )


def remove_missing_text(
    dataframe: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> pd.DataFrame:
    """Remove rows where the text field is missing or empty."""

    validate_input_dataframe(dataframe, text_column)

    result = dataframe.copy()

    result[text_column] = result[text_column].fillna("").astype(str)

    result = result[
        result[text_column].str.strip().ne("")
    ].copy()

    return result.reset_index(drop=True)


def remove_duplicates(
    dataframe: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> pd.DataFrame:
    """Remove duplicate text entries."""

    validate_input_dataframe(dataframe, text_column)

    result = dataframe.copy()

    result = result.drop_duplicates(
        subset=[text_column],
        keep="first",
    )

    return result.reset_index(drop=True)


def remove_short_comments(
    dataframe: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
    minimum_characters: int = 10,
) -> pd.DataFrame:
    """
    Remove comments below the configured character threshold.

    The threshold is configurable because the final research threshold
    should be established before approved empirical data collection.
    """

    validate_input_dataframe(dataframe, text_column)

    result = dataframe.copy()

    result = result[
        result[text_column].astype(str).str.len()
        >= minimum_characters
    ].copy()

    return result.reset_index(drop=True)


def normalise_whitespace(text: str) -> str:
    """Normalise repeated whitespace characters."""

    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Apply conservative text cleaning.

    The method intentionally avoids aggressive removal of punctuation
    or social-media language because these may contain sentiment cues.
    """

    text = normalise_whitespace(text)

    return text


def preprocess_dataframe(
    dataframe: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
    minimum_characters: int = 10,
) -> pd.DataFrame:
    """
    Run the main preprocessing pipeline.

    Processing order:

    1. Validate input
    2. Remove missing/empty text
    3. Remove duplicate text
    4. Remove very short comments
    5. Apply conservative text normalisation
    """

    validate_input_dataframe(dataframe, text_column)

    result = remove_missing_text(
        dataframe,
        text_column,
    )

    result = remove_duplicates(
        result,
        text_column,
    )

    result = remove_short_comments(
        result,
        text_column,
        minimum_characters,
    )

    result[text_column] = result[text_column].apply(clean_text)

    return result.reset_index(drop=True)


def filter_by_keywords(
    dataframe: pd.DataFrame,
    keywords: Iterable[str],
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> pd.DataFrame:
    """
    Filter records containing at least one specified keyword.

    Matching is case-insensitive.

    This function is intended to demonstrate the proposed sampling
    strategy. The final research keyword list must be documented
    before any approved empirical data collection.
    """

    validate_input_dataframe(dataframe, text_column)

    keyword_list = [
        keyword.strip()
        for keyword in keywords
        if keyword and keyword.strip()
    ]

    if not keyword_list:
        return dataframe.copy()

    pattern = "|".join(
        re.escape(keyword)
        for keyword in keyword_list
    )

    mask = dataframe[text_column].astype(str).str.contains(
        pattern,
        case=False,
        regex=True,
        na=False,
    )

    return dataframe.loc[mask].reset_index(drop=True)