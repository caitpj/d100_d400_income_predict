import polars as pl
import pytest

from income_predict_d100_d400.cleaning import (
    binarize_marital_status,
    encode_education,
    replace_question_marks_with_nan,
)


@pytest.mark.parametrize(
    "education, expected", [("Bachelors", 13), ("HS-grad", 9), ("Masters", 14)]
)
def test_encode_education(education, expected):
    df = pl.DataFrame({"education": [education]})
    result = encode_education(df)
    assert result["education"].item(0) == expected


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("Private", "Private"),
        ("?", None),
        (" ?", None),
    ],
)
def test_replace_question_marks(raw_value, expected):
    df = pl.DataFrame({"workclass": [raw_value]})
    # Pre-clean whitespace as replace_question_marks_with_nan expects trimmed strings
    # or relies on the full pipeline. The function itself calls strip_chars().
    result = replace_question_marks_with_nan(df)

    actual = result["workclass"].item(0)

    if expected is None:
        assert actual is None
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "status, expected",
    [
        ("Married-civ-spouse", True),
        ("Married-AF-spouse", True),
        ("Divorced", False),
        ("Never-married", False),
    ],
)
def test_binarize_marital_status(status, expected):
    """Test is_married_healthy creation."""
    df = pl.DataFrame({"marital_status": [status]})
    result = binarize_marital_status(df)

    assert "is_married_healthy" in result.columns
    assert result["is_married_healthy"].item(0) == expected
