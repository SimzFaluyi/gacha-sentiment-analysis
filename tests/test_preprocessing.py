import pandas as pd

from src.preprocessing import (
    clean_text,
    preprocess_dataframe,
    remove_duplicates,
    remove_missing_text,
    remove_short_comments,
)


def test_remove_missing_text():
    dataframe = pd.DataFrame(
        {
            "text": [
                "This is valid text",
                None,
                "",
            ]
        }
    )

    result = remove_missing_text(dataframe)

    assert len(result) == 1
    assert result.iloc[0]["text"] == "This is valid text"


def test_remove_duplicates():
    dataframe = pd.DataFrame(
        {
            "text": [
                "The pity system is good",
                "The pity system is good",
                "The banner is expensive",
            ]
        }
    )

    result = remove_duplicates(dataframe)

    assert len(result) == 2


def test_remove_short_comments():
    dataframe = pd.DataFrame(
        {
            "text": [
                "ok",
                "The pity system is expensive",
            ]
        }
    )

    result = remove_short_comments(
        dataframe,
        minimum_characters=10,
    )

    assert len(result) == 1


def test_clean_text():
    text = "This    banner    is expensive"

    result = clean_text(text)

    assert result == "This banner is expensive"


def test_preprocess_dataframe():
    dataframe = pd.DataFrame(
        {
            "text": [
                "The pity system is very useful",
                None,
                "The pity system is very useful",
                "ok",
            ]
        }
    )

    result = preprocess_dataframe(
        dataframe,
        minimum_characters=10,
    )

    assert len(result) == 1
