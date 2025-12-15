import numpy as np
import pytest

from income_predict_d100_d400.feature_engineering import SimpleStandardScaler


def test_simple_standard_scaler_integration():
    """Tests the scaler against known inputs and outputs."""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = SimpleStandardScaler().fit(data)

    expected_mean = np.array([3.0, 4.0])
    expected_std = np.array([1.63299, 1.63299])

    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5)

    transformed = scaler.transform(data)
    expected_transformed = (data - expected_mean) / expected_std
    np.testing.assert_allclose(transformed, expected_transformed, rtol=1e-5)


def test_constant_column():
    """Tests that constant columns (zero variance) do not cause division by zero."""
    data_const = np.array([[1], [1], [1]])
    scaler = SimpleStandardScaler().fit(data_const)

    assert scaler.scale_ is not None
    assert scaler.scale_[0] == 1.0
    transformed = scaler.transform(data_const)
    np.testing.assert_array_equal(transformed, np.zeros((3, 1)))


@pytest.mark.parametrize(
    "input_data,expected_mean,expected_scale",
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([3.0, 4.0]),
            np.array([1.633, 1.633]),
        ),
        (np.array([[0, 0], [0, 0]]), np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        (
            np.array([[-10, 100], [10, -100]]),
            np.array([0.0, 0.0]),
            np.array([10.0, 100.0]),
        ),
        (np.array([[1.5]]), np.array([1.5]), np.array([1.0])),
    ],
    ids=["standard_case", "zero_variance", "negative_values", "single_observation"],
)
def test_simple_standard_scaler_parametrised(input_data, expected_mean, expected_scale):
    scaler = SimpleStandardScaler().fit(input_data)
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-3)
    np.testing.assert_allclose(scaler.scale_, expected_scale, rtol=1e-2)
