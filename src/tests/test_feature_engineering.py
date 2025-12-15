import numpy as np
import pytest

from income_predict_d100_d400.feature_engineering import SignedLogTransformer


def test_signed_log_transformer_integration():
    """Tests the transformer against known inputs and outputs."""
    data = np.array([[0, 10, -10]])
    transformer = SignedLogTransformer().fit(data)
    expected_output = np.array([[0.0, 2.397895, -2.397895]])

    transformed = transformer.transform(data)
    np.testing.assert_allclose(transformed, expected_output, rtol=1e-5)


def test_signed_log_transformer_shape():
    """Verify it handles 2D arrays correctly and preserves shape."""
    data = np.array([[1, 2], [3, 4]])
    transformer = SignedLogTransformer().fit(data)
    transformed = transformer.transform(data)
    assert transformed.shape == data.shape


@pytest.mark.parametrize(
    "input_data,expected_output",
    [
        # Case 1: Standard positive, negative, and zero
        (
            np.array([[0, 1, -1]]),
            np.array([[0.0, 0.693147, -0.693147]]),
        ),
        # Case 2: Large numbers (outliers)
        (
            np.array([[100, -100]]),
            np.array([[4.61512, -4.61512]]),
        ),
        # Case 3: Small fractions
        (
            np.array([[0.5, -0.5]]),
            np.array([[0.405465, -0.405465]]),
        ),
    ],
    ids=["standard_case", "large_outliers", "small_fractions"],
)
def test_signed_log_transformer_parametrised(input_data, expected_output):
    """Parametrised tests for various input scenarios."""
    transformer = SignedLogTransformer().fit(input_data)
    transformed = transformer.transform(input_data)
    np.testing.assert_allclose(transformed, expected_output, rtol=1e-5)
