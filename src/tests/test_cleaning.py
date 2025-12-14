import numpy as np
import pandas as pd
import pytest

from income_predict_d100_d400.cleaning import (
    encode_education,
    replace_question_marks_with_nan,
)


@pytest.mark.parametrize(
    "education, expected", [("Bachelors", 13), ("HS-grad", 9), ("Masters", 14)]
)
def test_encode_education(education, expected):
    df = pd.DataFrame({"education": [education]})
    result = encode_education(df)
    assert result["education"].iloc[0] == expected


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("Private", "Private"),
        ("?", np.nan),
        (" ?", np.nan),
    ],
)
def test_replace_question_marks(raw_value, expected):
    df = pd.DataFrame({"workclass": [raw_value]})
    result = replace_question_marks_with_nan(df)

    actual = result["workclass"].iloc[0]

    if pd.isna(expected):
        assert pd.isna(actual)
    else:
        assert actual == expected
